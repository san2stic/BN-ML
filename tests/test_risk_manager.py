from __future__ import annotations

from bn_ml.types import Opportunity, Signal
from trader.risk_manager import RiskManager


def _config() -> dict:
    return {
        "scanner": {
            "min_ml_confidence": 65,
            "spread_max_pct": 0.15,
            "orderbook_depth_min_usdt": 50000,
        },
        "risk": {
            "max_positions": 5,
            "min_position_pct_active": 0.01,
            "max_position_pct_active": 0.20,
            "max_portfolio_exposure_pct": 0.70,
            "max_position_drawdown_pct": 0.03,
            "max_daily_risk_pct": 0.02,
            "max_weekly_risk_pct": 0.06,
            "min_net_profit_pct": 0.30,
            "max_correlation": 0.70,
            "circuit_breakers": {
                "daily_drawdown_stop_pct": -5.0,
                "max_consecutive_losses": 3,
                "volatility_spike_ratio": 1.5,
                "drift_block_enabled": True,
            },
        },
    }


def _opportunity(action: str = "BUY", confidence: float = 70.0, correlation: float = 0.5) -> Opportunity:
    return Opportunity(
        symbol="BTC/USDT",
        ml_score=70,
        technical_score=65,
        momentum_score=60,
        global_score=68,
        signal=Signal(symbol="BTC/USDT", action=action, confidence=confidence, strength=70),
        spread_pct=0.05,
        orderbook_depth_usdt=120000,
        atr_ratio=0.01,
        expected_net_profit_pct=0.45,
        correlation_with_btc=correlation,
    )


def _state() -> dict:
    return {
        "total_capital": 10000,
        "active_capital": 6000,
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "daily_realized_usdt": 0.0,
        "weekly_realized_usdt": 0.0,
        "consecutive_losses": 0,
        "market_volatility_ratio": 1.0,
        "market_drift_detected": False,
        "win_rate": 0.56,
        "avg_win": 1.8,
        "avg_loss": 1.0,
    }


def test_allows_valid_buy_signal() -> None:
    rm = RiskManager(_config())
    allowed, reasons, size = rm.can_open_position(_opportunity(), [], _state())

    assert allowed is True
    assert reasons == []
    assert size > 0


def test_blocks_non_buy_signal() -> None:
    rm = RiskManager(_config())
    allowed, reasons, _ = rm.can_open_position(_opportunity(action="HOLD"), [], _state())

    assert allowed is False
    assert any("not BUY" in reason for reason in reasons)


def test_blocks_high_correlation() -> None:
    rm = RiskManager(_config())
    allowed, reasons, _ = rm.can_open_position(_opportunity(correlation=0.9), [], _state())

    assert allowed is False
    assert any("Correlation" in reason for reason in reasons)


def test_blocks_when_daily_risk_budget_exhausted() -> None:
    rm = RiskManager(_config())
    state = _state()
    state["daily_realized_usdt"] = -250.0  # 2.5% loss on 10_000 total capital
    allowed, reasons, _ = rm.can_open_position(_opportunity(), [], state)

    assert allowed is False
    assert any("Daily risk budget exhausted" in reason for reason in reasons)


def test_blocks_when_projected_weekly_risk_would_exceed_budget() -> None:
    rm = RiskManager(_config())
    state = _state()
    state["weekly_realized_usdt"] = -590.0  # 5.9% already consumed, projected trade should exceed 6%
    allowed, reasons, _ = rm.can_open_position(_opportunity(), [], state)

    assert allowed is False
    assert any("Weekly risk budget would be exceeded" in reason for reason in reasons)


def test_blocks_when_market_drift_breaker_active() -> None:
    rm = RiskManager(_config())
    state = _state()
    state["market_drift_detected"] = True

    allowed, reasons, _ = rm.can_open_position(_opportunity(), [], state)
    assert allowed is False
    assert any("drift circuit breaker active" in reason.lower() for reason in reasons)
