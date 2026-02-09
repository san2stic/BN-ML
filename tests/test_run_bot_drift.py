from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.drift_monitor import MarketDriftMonitor
from scripts.run_bot import TradingRuntime


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)


class _DriftDataManager:
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        baseline = rng.normal(0.0, 0.0018, size=260)
        recent = rng.normal(0.005, 0.0035, size=100)
        rets = np.concatenate([baseline, recent])
        close = 100.0 * np.cumprod(1.0 + rets)
        close = close[-limit:]
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=len(close), freq="15min", tz="UTC"),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(len(close)),
            }
        )


def test_update_market_drift_state_sets_runtime_flags() -> None:
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.config = {"model": {"drift": {"enabled": True, "baseline_window": 220, "recent_window": 80}}}
    runtime.base_timeframe = "15m"
    runtime.data_manager = _DriftDataManager()
    runtime.account_state = {}
    runtime.logger = _DummyLogger()
    runtime._quote_asset = lambda: "USDT"  # type: ignore[assignment]
    runtime.drift_monitor = MarketDriftMonitor(runtime.config)

    runtime._update_market_drift_state()

    assert "market_drift_detected" in runtime.account_state
    assert "market_regime" in runtime.account_state
    assert runtime.account_state["market_drift_symbol"] == "BTC/USDT"
