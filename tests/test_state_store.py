from __future__ import annotations

from datetime import datetime, timezone

from bn_ml.state_store import StateStore
from bn_ml.domain_types import Position


def test_store_roundtrip_positions(tmp_path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))

    position = Position(
        symbol="BTC/USDT",
        side="LONG",
        size_usdt=1200,
        entry_price=42000,
        stop_loss=41000,
        take_profit_1=42800,
        take_profit_2=43600,
        opened_at=datetime.now(timezone.utc),
    )

    store.upsert_position(position)
    loaded = store.load_open_positions()

    assert len(loaded) == 1
    assert loaded[0].symbol == "BTC/USDT"
    assert loaded[0].status == "OPEN"


def test_store_account_state(tmp_path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))
    default = {"total_capital": 10000, "win_rate": 0.56}

    loaded_default = store.load_account_state(default)
    assert loaded_default["total_capital"] == 10000

    updated = {"total_capital": 12000, "win_rate": 0.60}
    store.save_account_state(updated)

    loaded = store.load_account_state(default)
    assert loaded["total_capital"] == 12000
    assert loaded["win_rate"] == 0.60
