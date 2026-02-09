from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = ohlcv.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        for period in [9, 21, 50, 100, 200]:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().replace(0, np.nan)
        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        tr = pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / close.replace(0, np.nan)

        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_mid"] = ma20
        df["bb_upper"] = ma20 + 2 * std20
        df["bb_lower"] = ma20 - 2 * std20
        width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_percent_b"] = (close - df["bb_lower"]) / width

        signed_volume = np.where(close >= close.shift(1), volume, -volume)
        df["obv"] = pd.Series(signed_volume, index=df.index).cumsum()
        df["volume_ratio_20"] = volume / volume.rolling(20).mean().replace(0, np.nan)

        for period, label in [(4, "1h"), (16, "4h"), (96, "1d")]:
            df[f"roc_{label}"] = (close / close.shift(period) - 1) * 100

        df["dist_high_24h"] = close / high.rolling(96).max() - 1
        df["dist_low_24h"] = close / low.rolling(96).min() - 1

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return df

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        excluded = {"timestamp", "open", "high", "low", "close", "volume", "label", "label_edge", "symbol"}
        return [c for c in df.columns if c not in excluded]
