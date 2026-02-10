from __future__ import annotations

from bn_ml.domain_types import Opportunity, Signal
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


def _opportunity(
    action: str = "BUY",
    confidence: float = 70.0,
    correlation: float = 0.5,
    symbol: str = "ETH/USDT",
    spread_pct: float = 0.05,
    depth_usdt: float = 120000,
) -> Opportunity:
    return Opportunity(
        symbol=symbol,
        ml_score=70,
        technical_score=65,
        momentum_score=60,
        global_score=68,
        signal=Signal(symbol=symbol, action=action, confidence=confidence, strength=70),
        spread_pct=spread_pct,
        orderbook_depth_usdt=depth_usdt,
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
        "market_intelligence_signal": "HOLD",
        "market_intelligence_confidence": 0.0,
        "market_intelligence_regime": "neutral",
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


def test_dynamic_pair_filter_relaxes_major_pair_limits() -> None:
    cfg = _config()
    cfg["scanner"]["dynamic_pair_filters"] = {
        "enabled": True,
        "major_bases": ["ETH"],
        "spread_factor_major": 1.20,
        "depth_factor_major": 0.80,
        "correlation_bonus_major": 0.15,
    }
    rm = RiskManager(cfg)
    opp = _opportunity(symbol="ETH/USDT", spread_pct=0.17, depth_usdt=42_000, correlation=0.84)

    allowed, reasons, _ = rm.can_open_position(opp, [], _state())

    assert allowed is True
    assert reasons == []


def test_dynamic_pair_filter_tightens_alt_pair_limits() -> None:
    cfg = _config()
    cfg["scanner"]["dynamic_pair_filters"] = {
        "enabled": True,
        "major_bases": ["ETH"],
        "spread_factor_alt": 0.80,
        "depth_factor_alt": 1.20,
        "correlation_penalty_alt": -0.10,
    }
    rm = RiskManager(cfg)
    opp = _opportunity(symbol="TRX/USDT", spread_pct=0.13, depth_usdt=55_000, correlation=0.65)

    allowed, reasons, _ = rm.can_open_position(opp, [], _state())

    assert allowed is False
    assert any("Spread too high" in reason for reason in reasons)
    assert any("Orderbook depth too low" in reason for reason in reasons)
    assert any("Correlation threshold exceeded" in reason for reason in reasons)


def test_dynamic_pair_filter_supports_symbol_override() -> None:
    cfg = _config()
    cfg["scanner"]["dynamic_pair_filters"] = {
        "enabled": True,
        "by_symbol": {
            "TRX/USDT": {
                "spread_max_pct": 0.20,
                "orderbook_depth_min_usdt": 40_000,
                "max_correlation": 0.90,
            }
        },
    }
    rm = RiskManager(cfg)
    opp = _opportunity(symbol="TRX/USDT", spread_pct=0.18, depth_usdt=45_000, correlation=0.85)

    allowed, reasons, _ = rm.can_open_position(opp, [], _state())

    assert allowed is True
    assert reasons == []


def test_dynamic_pair_filter_benchmark_allows_btc_correlation() -> None:
    cfg = _config()
    cfg["scanner"]["dynamic_pair_filters"] = {
        "enabled": True,
        "major_bases": ["BTC"],
        "correlation_limit_benchmark": 1.0,
    }
    rm = RiskManager(cfg)
    opp = _opportunity(symbol="BTC/USDT", correlation=0.98)

    allowed, reasons, _ = rm.can_open_position(opp, [], _state())

    assert allowed is True
    assert reasons == []


def test_blocks_when_santrade_intelligence_bearish_breaker_active() -> None:
    cfg = _config()
    cfg["risk"]["circuit_breakers"]["market_intelligence_block_enabled"] = True
    cfg["risk"]["circuit_breakers"]["market_intelligence_min_confidence"] = 72
    rm = RiskManager(cfg)
    state = _state()
    state["market_intelligence_signal"] = "SELL"
    state["market_intelligence_confidence"] = 81.0

    allowed, reasons, _ = rm.can_open_position(_opportunity(), [], state)

    assert allowed is False
    assert any("SanTradeIntelligence bearish circuit breaker active." in reason for reason in reasons)


def test_blocks_when_santrade_intelligence_risk_off_breaker_active() -> None:
    cfg = _config()
    cfg["risk"]["circuit_breakers"]["market_intelligence_block_enabled"] = True
    cfg["risk"]["circuit_breakers"]["market_intelligence_block_on_risk_off"] = True
    rm = RiskManager(cfg)
    state = _state()
    state["market_intelligence_regime"] = "risk_off"

    allowed, reasons, _ = rm.can_open_position(_opportunity(), [], state)

    assert allowed is False
    assert any("SanTradeIntelligence risk-off circuit breaker active." in reason for reason in reasons)
