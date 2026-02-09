from __future__ import annotations

import pytest

from trader.order_manager import OrderManager


def test_validate_order_constraints_accepts_valid_order() -> None:
    limits = {"min_qty": 0.001, "max_qty": 1000.0, "min_notional": 5.0}
    OrderManager._validate_order_constraints(symbol="BTC/USDT", amount=0.01, price=1000.0, limits=limits)


def test_validate_order_constraints_rejects_min_qty() -> None:
    limits = {"min_qty": 0.01, "max_qty": None, "min_notional": None}
    with pytest.raises(ValueError, match="minQty"):
        OrderManager._validate_order_constraints(symbol="BTC/USDT", amount=0.001, price=1000.0, limits=limits)


def test_validate_order_constraints_rejects_min_notional() -> None:
    limits = {"min_qty": 0.001, "max_qty": None, "min_notional": 10.0}
    with pytest.raises(ValueError, match="minNotional"):
        OrderManager._validate_order_constraints(symbol="BTC/USDT", amount=0.001, price=1000.0, limits=limits)


def test_paper_sell_uses_base_qty() -> None:
    manager = OrderManager(config={}, paper=True)
    res = manager.place_market_sell(symbol="BTC/USDT", price=100.0, base_qty=0.25)

    assert res["base_qty"] == pytest.approx(0.25)
    assert res["size_usdt"] == pytest.approx(25.0)
