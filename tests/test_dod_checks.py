from __future__ import annotations

from bn_ml.dod_checks import evaluate_dod_daily
from bn_ml.state_store import StateStore


def _base_config() -> dict:
    return {
        "risk": {
            "max_positions": 5,
            "max_daily_risk_pct": 0.02,
            "max_weekly_risk_pct": 0.06,
            "circuit_breakers": {
                "daily_drawdown_stop_pct": -5.0,
                "volatility_spike_ratio": 1.5,
                "drift_block_enabled": True,
            },
        }
    }


def test_dod_daily_passes_when_no_violation(tmp_path) -> None:
    db = str(tmp_path / "state.db")
    store = StateStore(db_path=db)
    store.insert_cycle(
        opportunities=3,
        opened_positions=0,
        data={"open_positions": 2, "daily_pnl_pct": -1.2, "weekly_pnl_pct": -1.8},
    )
    store.insert_trade(
        symbol="BTC/USDC",
        side="BUY",
        size_usdt=100,
        price=100,
        mode="paper",
        extra={
            "daily_risk_used_pct": 0.8,
            "weekly_risk_used_pct": 1.4,
            "market_volatility_ratio": 1.1,
            "market_drift_detected": False,
        },
    )

    result = evaluate_dod_daily(config=_base_config(), db_path=db)
    assert result["status"] == "PASS"
    assert result["violations_count"] == 0


def test_dod_daily_detects_risk_violations(tmp_path) -> None:
    db = str(tmp_path / "state.db")
    store = StateStore(db_path=db)
    store.insert_cycle(
        opportunities=5,
        opened_positions=1,
        data={"open_positions": 7, "daily_pnl_pct": -6.0, "weekly_pnl_pct": -7.0},
    )
    store.insert_trade(
        symbol="ETH/USDC",
        side="BUY",
        size_usdt=120,
        price=100,
        mode="paper",
        extra={
            "daily_risk_used_pct": 2.2,
            "weekly_risk_used_pct": 6.2,
            "market_volatility_ratio": 1.8,
            "market_drift_detected": True,
        },
    )

    result = evaluate_dod_daily(config=_base_config(), db_path=db)
    ids = {item["id"] for item in result["violations"]}
    assert result["status"] == "FAIL"
    assert "max_positions_exceeded" in ids
    assert "buy_after_daily_risk_budget_exhausted" in ids
    assert "buy_after_weekly_risk_budget_exhausted" in ids
    assert "buy_while_volatility_breaker_active" in ids
    assert "buy_while_drift_breaker_active" in ids
