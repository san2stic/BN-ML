from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml_engine.santrade_intelligence import SanTradeIntelligenceSnapshot
from scripts.run_bot import TradingRuntime


@dataclass
class _DummyStore:
    state: dict

    def set_state(self, key: str, value) -> None:  # noqa: ANN001
        self.state[key] = value


class _DummyIntelligence:
    def update(self, pairs, opportunities, quote_asset: str):  # noqa: ANN001
        del pairs, opportunities, quote_asset
        return SanTradeIntelligenceSnapshot(
            generated_at="2026-02-10T12:00:00+00:00",
            enabled=True,
            signal="SELL",
            confidence=84.0,
            market_score=-0.44,
            market_score_pct=28.0,
            market_regime="risk_off",
            predicted_move_pct=-1.6,
            symbols_scanned=30,
            opportunities_scanned=18,
            buy_ratio=0.10,
            sell_ratio=0.75,
            hold_ratio=0.15,
            avg_ml_score=42.0,
            avg_technical_score=40.0,
            avg_momentum_score=36.0,
            avg_global_score=41.0,
            avg_atr_ratio=0.020,
            avg_spread_pct=0.23,
            avg_correlation_btc=0.62,
            score_dispersion=8.5,
            model_samples=52,
            model_ready=True,
            benchmark_symbol="BTC/USDT",
            benchmark_price=50000.0,
        )


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)


def test_update_market_intelligence_updates_account_state_and_persists_snapshot(tmp_path: Path) -> None:
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.market_intelligence = _DummyIntelligence()
    runtime.account_state = {}
    runtime.store = _DummyStore(state={})
    runtime.logger = _DummyLogger()
    runtime.config = {"monitoring": {"metrics_dir": str(tmp_path)}}
    runtime._quote_asset = lambda: "USDT"  # type: ignore[assignment]

    runtime._update_market_intelligence(pairs=["BTC/USDT", "ETH/USDT"], opportunities_all=[])

    assert runtime.account_state["market_intelligence_signal"] == "SELL"
    assert runtime.account_state["market_intelligence_regime"] == "risk_off"
    assert runtime.account_state["market_intelligence_model_samples"] == 52
    assert "santrade_intelligence" in runtime.store.state
    assert runtime.store.state["santrade_intelligence"]["signal"] == "SELL"
