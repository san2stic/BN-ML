from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data_manager.data_cleaner import DataCleaner
from data_manager.features_engineer import FeatureEngineer
from data_manager.multi_timeframe import MultiTimeframeFeatureBuilder, timeframe_to_minutes


class _DummyDataManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
        self.calls.append((symbol, timeframe, int(limit)))
        return _synthetic_ohlcv(timeframe=timeframe, limit=limit)


def _synthetic_ohlcv(timeframe: str, limit: int) -> pd.DataFrame:
    tf_minutes = timeframe_to_minutes(timeframe)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = [start + pd.Timedelta(minutes=tf_minutes * i) for i in range(limit)]

    slope = {
        "15m": 0.0004,
        "1h": 0.0009,
        "4h": 0.0015,
        "1d": 0.0022,
    }.get(timeframe, 0.0005)
    base = 100.0 + (tf_minutes / 10.0)

    close = base * np.cumprod(1 + np.full(limit, slope, dtype=float))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.maximum(close * 0.0012, 0.01)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.linspace(10_000, 20_000, num=limit)

    return pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_mtf_builder_adds_prefixed_features_and_confluence() -> None:
    config = {
        "model": {
            "multi_timeframe": {
                "enabled": True,
                "base_timeframe": "15m",
                "timeframes": ["1h", "4h", "1d"],
                "min_candles_per_timeframe": 80,
                "max_candles_per_timeframe": 180,
                "extra_candles_buffer": 24,
            }
        }
    }
    manager = _DummyDataManager()
    builder = MultiTimeframeFeatureBuilder(
        config=config,
        data_manager=manager,  # type: ignore[arg-type]
        cleaner=DataCleaner(),
        feature_engineer=FeatureEngineer(),
    )

    frame = builder.build(symbol="BTC/USDC", limit=240)

    assert len(frame) == 240
    assert "mtf_1h_macd_hist" in frame.columns
    assert "mtf_4h_ema_21" in frame.columns
    assert "mtf_1d_ema_21" in frame.columns
    assert "mtf_confluence_score" in frame.columns
    assert not pd.isna(frame.iloc[-1]["mtf_confluence_score"])
    assert ("BTC/USDC", "15m", 240) in manager.calls


def test_mtf_builder_sanitizes_requested_timeframes() -> None:
    config = {
        "model": {
            "multi_timeframe": {
                "enabled": True,
                "base_timeframe": "15m",
                "timeframes": ["5m", "abc", "1h", "1h", "4h"],
            }
        }
    }
    builder = MultiTimeframeFeatureBuilder(
        config=config,
        data_manager=_DummyDataManager(),  # type: ignore[arg-type]
        cleaner=DataCleaner(),
        feature_engineer=FeatureEngineer(),
    )

    summary = builder.describe()
    assert summary["timeframes"] == ["1h", "4h"]
