from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bn_ml.domain_types import Position
from trader.exit_manager import ExitManager


def _cfg() -> dict:
    return {
        "risk": {
            "trailing_activation_pct": 0.02,
            "trailing_atr_mult": 1.0,
            "time_stop_hours": 48,
            "max_position_drawdown_pct": 0.03,
        }
    }


def _position() -> Position:
    return Position(
        symbol="BTC/USDT",
        side="LONG",
        size_usdt=1000,
        entry_price=100,
        stop_loss=95,
        take_profit_1=102,
        take_profit_2=104,
        opened_at=datetime.now(timezone.utc) - timedelta(hours=2),
        extra={"initial_size_usdt": 1000, "tp1_hit": False, "tp2_hit": False, "trailing_active": False},
    )


def test_tp_hits_create_partial_actions() -> None:
    em = ExitManager(_cfg())
    pos = _position()

    d = em.evaluate_long(position=pos, price=105, atr_value=1.0, now=datetime.now(timezone.utc))

    assert d.close_all is False
    assert 0.5 in d.partial_fracs
    assert 0.3 in d.partial_fracs


def test_stop_loss_closes_position() -> None:
    em = ExitManager(_cfg())
    pos = _position()

    d = em.evaluate_long(position=pos, price=94, atr_value=1.0, now=datetime.now(timezone.utc))

    assert d.close_all is True
    assert d.close_reason == "stop_loss"


def test_time_stop_closes_flat_position() -> None:
    em = ExitManager(_cfg())
    pos = _position()
    pos.opened_at = datetime.now(timezone.utc) - timedelta(hours=60)

    d = em.evaluate_long(position=pos, price=100.1, atr_value=0.5, now=datetime.now(timezone.utc))

    assert d.close_all is True
    assert d.close_reason == "time_stop"
