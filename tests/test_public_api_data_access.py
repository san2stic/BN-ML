from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bn_ml.domain_types import Position
from bn_ml.state_store import StateStore
from public_api.data_access import (
    load_account_state,
    load_recent_trades,
    load_runtime_summary,
    load_santrade_intelligence,
    read_prediction_snapshot,
)


def _write_scan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "symbol,signal,confidence,global_score,spread_pct,depth_usdt\n"
        "BTC/USDT,BUY,87.5,78.2,0.02,120000\n"
        "ETH/USDT,HOLD,51.0,55.3,0.03,98000\n"
        "SOL/USDT,SELL,72.3,64.1,0.05,76000\n",
        encoding="utf-8",
    )


def test_read_prediction_snapshot_filters_and_limits(tmp_path: Path) -> None:
    scan_path = tmp_path / "metrics" / "latest_scan.csv"
    _write_scan(scan_path)

    snapshot = read_prediction_snapshot(scan_path, limit=2)
    assert snapshot["total_rows"] == 3
    assert snapshot["returned_rows"] == 2
    assert snapshot["rows"][0]["symbol"] == "BTC/USDT"

    buy_only = read_prediction_snapshot(scan_path, limit=10, signal="buy")
    assert buy_only["total_rows"] == 1
    assert buy_only["rows"][0]["signal"] == "BUY"


def test_read_prediction_snapshot_missing_file(tmp_path: Path) -> None:
    snapshot = read_prediction_snapshot(tmp_path / "missing.csv")
    assert snapshot["total_rows"] == 0
    assert snapshot["rows"] == []


def test_runtime_sqlite_access(tmp_path: Path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))
    store.save_account_state({"total_capital": 2500.0, "active_capital": 1500.0, "daily_pnl_pct": 1.2})
    store.set_state(
        "santrade_intelligence",
        {
            "signal": "BUY",
            "confidence": 77.0,
            "market_regime": "bull_acceleration",
        },
    )

    pos = Position(
        symbol="BTC/USDT",
        side="LONG",
        size_usdt=400.0,
        entry_price=45000.0,
        stop_loss=43000.0,
        take_profit_1=46000.0,
        take_profit_2=47000.0,
        opened_at=datetime.now(timezone.utc),
    )
    store.upsert_position(pos)
    store.insert_trade("BTC/USDT", "BUY", 400.0, 45000.0, "paper", extra={"confidence": 88})
    store.insert_cycle(opportunities=3, opened_positions=1, data={"paper": True})

    db_path = tmp_path / "state.db"
    account = load_account_state(db_path)
    assert account["total_capital"] == 2500.0

    trades = load_recent_trades(db_path, limit=5)
    assert len(trades) == 1
    assert trades[0]["extra"]["confidence"] == 88

    summary = load_runtime_summary(db_path)
    assert summary["open_positions"] == 1
    assert summary["total_trades"] == 1
    assert summary["total_cycles"] == 1

    intelligence = load_santrade_intelligence(db_path)
    assert intelligence["signal"] == "BUY"
