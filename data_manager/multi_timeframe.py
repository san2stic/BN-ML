from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from data_manager.data_cleaner import DataCleaner
from data_manager.features_engineer import FeatureEngineer
from data_manager.fetch_data import BinanceDataManager


_TF_PATTERN = re.compile(r"^\s*(\d+)\s*([mhdwMHDW])\s*$")
_DEFAULT_MTF_FEATURES = (
    "ema_9",
    "ema_21",
    "ema_50",
    "macd_hist",
    "rsi_14",
    "atr_ratio",
    "bb_percent_b",
    "volume_ratio_20",
    "roc_1h",
    "roc_4h",
    "dist_high_24h",
    "dist_low_24h",
)


def timeframe_to_minutes(timeframe: str) -> int:
    match = _TF_PATTERN.match(str(timeframe))
    if match is None:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    amount = int(match.group(1))
    unit = match.group(2).lower()
    scale = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    return amount * scale[unit]


class MultiTimeframeFeatureBuilder:
    """Builds base + higher-timeframe aligned feature frames for one symbol."""

    def __init__(
        self,
        config: dict,
        data_manager: BinanceDataManager,
        cleaner: DataCleaner | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        self.config = config
        self.data_manager = data_manager
        self.cleaner = cleaner or DataCleaner()
        self.feature_engineer = feature_engineer or FeatureEngineer()

        model_cfg = config.get("model", {})
        mtf_cfg = model_cfg.get("multi_timeframe", {})

        self.enabled = bool(mtf_cfg.get("enabled", True))
        self.base_timeframe = str(mtf_cfg.get("base_timeframe", "15m")).strip().lower()
        self.prefix = str(mtf_cfg.get("prefix", "mtf")).strip().lower() or "mtf"

        requested = mtf_cfg.get("timeframes", ["1h", "4h", "1d"])
        if isinstance(requested, str):
            requested = [requested]

        self.timeframes = self._sanitize_timeframes(requested)

        raw_cols = mtf_cfg.get("feature_columns", list(_DEFAULT_MTF_FEATURES))
        if not isinstance(raw_cols, list):
            raw_cols = list(_DEFAULT_MTF_FEATURES)
        self.htf_feature_columns = tuple(
            [str(col).strip() for col in raw_cols if isinstance(col, str) and str(col).strip()]
        ) or _DEFAULT_MTF_FEATURES

        self.extra_candles_buffer = max(0, int(mtf_cfg.get("extra_candles_buffer", 48)))
        self.min_candles_per_timeframe = max(40, int(mtf_cfg.get("min_candles_per_timeframe", 120)))
        self.max_candles_per_timeframe = max(
            self.min_candles_per_timeframe,
            int(mtf_cfg.get("max_candles_per_timeframe", 360)),
        )

    def describe(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "base_timeframe": self.base_timeframe,
            "timeframes": list(self.timeframes),
            "prefix": self.prefix,
            "feature_columns": list(self.htf_feature_columns),
        }

    def build(self, symbol: str, limit: int) -> pd.DataFrame:
        base_limit = max(120, int(limit))
        base_ohlcv = self.data_manager.fetch_ohlcv(symbol=symbol, timeframe=self.base_timeframe, limit=base_limit)
        base_ohlcv = self.cleaner.clean_ohlcv(base_ohlcv)
        frame = self.feature_engineer.build(base_ohlcv).sort_values("timestamp").reset_index(drop=True)

        if not self.enabled or not self.timeframes:
            return frame

        out = frame.copy()
        for timeframe in self.timeframes:
            tf_limit = self._derive_limit(base_limit=base_limit, timeframe=timeframe)
            htf_ohlcv = self.data_manager.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=tf_limit)
            htf_ohlcv = self.cleaner.clean_ohlcv(htf_ohlcv)
            htf_frame = self.feature_engineer.build(htf_ohlcv).sort_values("timestamp").reset_index(drop=True)

            feature_cols = [c for c in self.htf_feature_columns if c in htf_frame.columns]
            if not feature_cols:
                continue

            tf_key = self._tf_key(timeframe)
            renamed = {c: f"{self.prefix}_{tf_key}_{c}" for c in feature_cols}
            to_merge = htf_frame[["timestamp", *feature_cols]].rename(columns=renamed).sort_values("timestamp")

            out = pd.merge_asof(
                out.sort_values("timestamp"),
                to_merge,
                on="timestamp",
                direction="backward",
            )

        out = self._add_confluence_features(out)
        all_nan_cols = [c for c in out.columns if c != "timestamp" and out[c].isna().all()]
        if all_nan_cols:
            out = out.drop(columns=all_nan_cols)
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return out.reset_index(drop=True)

    def _sanitize_timeframes(self, requested: list) -> tuple[str, ...]:
        base_minutes = timeframe_to_minutes(self.base_timeframe)
        cleaned: list[str] = []
        for raw in requested:
            tf = str(raw).strip().lower()
            if not tf:
                continue
            try:
                tf_minutes = timeframe_to_minutes(tf)
            except ValueError:
                continue
            if tf_minutes <= base_minutes:
                continue
            if tf in cleaned:
                continue
            cleaned.append(tf)
        return tuple(cleaned)

    def _derive_limit(self, base_limit: int, timeframe: str) -> int:
        base_minutes = timeframe_to_minutes(self.base_timeframe)
        tf_minutes = timeframe_to_minutes(timeframe)
        scaled = int(math.ceil(max(1, base_limit) * base_minutes / max(1, tf_minutes)))
        candles = scaled + self.extra_candles_buffer
        candles = max(candles, self.min_candles_per_timeframe)
        candles = min(candles, self.max_candles_per_timeframe)
        return int(candles)

    @staticmethod
    def _tf_key(timeframe: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", str(timeframe).strip().lower())

    def _add_confluence_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        trend_strength_series: list[pd.Series] = []
        trend_vote_series: list[pd.Series] = []
        macd_vote_series: list[pd.Series] = []
        rsi_bias_series: list[pd.Series] = []

        def _collect(prefix: str) -> None:
            ema_fast = frame.get(f"{prefix}ema_9")
            ema_slow = frame.get(f"{prefix}ema_21")
            macd_hist = frame.get(f"{prefix}macd_hist")
            rsi = frame.get(f"{prefix}rsi_14")

            if ema_fast is not None and ema_slow is not None:
                rel = ((ema_fast / ema_slow.replace(0.0, np.nan)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                trend_strength_series.append((rel * 100.0).rename(f"{prefix}trend_strength"))
                trend_vote_series.append(pd.Series(np.sign(rel), index=frame.index, name=f"{prefix}trend_vote"))

            if macd_hist is not None:
                macd_vote_series.append(
                    pd.Series(np.sign(macd_hist.fillna(0.0)), index=frame.index, name=f"{prefix}macd_vote")
                )

            if rsi is not None:
                centered = ((rsi.fillna(50.0) - 50.0) / 50.0).clip(-1.0, 1.0)
                rsi_bias_series.append(centered.rename(f"{prefix}rsi_bias"))

        _collect("")
        for timeframe in self.timeframes:
            tf_key = self._tf_key(timeframe)
            _collect(f"{self.prefix}_{tf_key}_")

        if trend_strength_series:
            trend_df = pd.concat(trend_strength_series, axis=1)
            vote_df = pd.concat(trend_vote_series, axis=1)
            frame[f"{self.prefix}_trend_strength_mean"] = trend_df.mean(axis=1)
            frame[f"{self.prefix}_trend_strength_std"] = trend_df.std(axis=1).fillna(0.0)
            frame[f"{self.prefix}_trend_consensus"] = vote_df.mean(axis=1)
            frame[f"{self.prefix}_trend_conflict"] = 1.0 - frame[f"{self.prefix}_trend_consensus"].abs()

        if macd_vote_series:
            macd_df = pd.concat(macd_vote_series, axis=1)
            frame[f"{self.prefix}_macd_consensus"] = macd_df.mean(axis=1)

        if rsi_bias_series:
            rsi_df = pd.concat(rsi_bias_series, axis=1)
            frame[f"{self.prefix}_rsi_bias_mean"] = rsi_df.mean(axis=1)
            frame[f"{self.prefix}_rsi_bias_std"] = rsi_df.std(axis=1).fillna(0.0)

        signal_cols = [
            col
            for col in (
                f"{self.prefix}_trend_consensus",
                f"{self.prefix}_macd_consensus",
                f"{self.prefix}_rsi_bias_mean",
            )
            if col in frame.columns
        ]
        if signal_cols:
            frame[f"{self.prefix}_confluence_score"] = frame[signal_cols].mean(axis=1)

        return frame
