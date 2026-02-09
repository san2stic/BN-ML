from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from scripts.run_bot import TradingRuntime


@dataclass
class _DummyLogger:
    warnings: list[str] = field(default_factory=list)

    def warning(self, msg: str, *args) -> None:
        if args:
            self.warnings.append(msg % args)
        else:
            self.warnings.append(msg)


class _PassCleaner:
    def clean_ohlcv(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame


class _PassFeatures:
    def build(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame


class _StaticAtrDataManager:
    def __init__(self, atr_values: list[float]) -> None:
        self.atr_values = atr_values

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
        values = self.atr_values[-limit:]
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="15min", tz="UTC"),
                "open": [100.0] * len(values),
                "high": [101.0] * len(values),
                "low": [99.0] * len(values),
                "close": [100.0] * len(values),
                "volume": [1000.0] * len(values),
                "atr_ratio": values,
            }
        )


def _build_runtime_for_volatility(data_manager) -> TradingRuntime:
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.config = {"risk": {"volatility_baseline_window": 30}}
    runtime.atr_ohlcv_limit = 100
    runtime.base_timeframe = "15m"
    runtime.data_manager = data_manager
    runtime.cleaner = _PassCleaner()
    runtime.features = _PassFeatures()
    runtime.account_state = {"market_volatility_ratio": 1.0}
    runtime.logger = _DummyLogger()
    runtime._quote_asset = lambda: "USDT"  # type: ignore[assignment]
    return runtime


def test_updates_market_volatility_ratio_from_runtime_data() -> None:
    atr_values = [0.01] * 60 + [0.02]
    runtime = _build_runtime_for_volatility(_StaticAtrDataManager(atr_values))

    runtime._update_market_volatility_ratio()

    assert runtime.account_state["market_volatility_symbol"] == "BTC/USDT"
    assert abs(float(runtime.account_state["market_volatility_ratio"]) - 2.0) < 0.05
    assert float(runtime.account_state["market_volatility_current_atr_ratio"]) == 0.02
    assert float(runtime.account_state["market_volatility_baseline_atr_ratio"]) > 0


def test_volatility_refresh_failure_keeps_previous_ratio() -> None:
    class _FailingDataManager:
        def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
            raise RuntimeError("market data unavailable")

    runtime = _build_runtime_for_volatility(_FailingDataManager())
    runtime.account_state["market_volatility_ratio"] = 1.33

    runtime._update_market_volatility_ratio()

    assert runtime.account_state["market_volatility_ratio"] == 1.33
    assert runtime.logger.warnings
