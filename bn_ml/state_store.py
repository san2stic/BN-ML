from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bn_ml.types import Position


class StateStore:
    def __init__(self, db_path: str = "artifacts/state/bn_ml.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    size_usdt REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size_usdt REAL NOT NULL,
                    price REAL NOT NULL,
                    mode TEXT NOT NULL,
                    extra_json TEXT
                );

                CREATE TABLE IF NOT EXISTS cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    opportunities INTEGER NOT NULL,
                    opened_positions INTEGER NOT NULL,
                    data_json TEXT
                );

                CREATE TABLE IF NOT EXISTS kv_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                );
                """
            )
            self._ensure_positions_extra_column(conn)

    @staticmethod
    def _ensure_positions_extra_column(conn: sqlite3.Connection) -> None:
        cols = conn.execute("PRAGMA table_info(positions)").fetchall()
        col_names = {row["name"] for row in cols}
        if "extra_json" not in col_names:
            try:
                conn.execute("ALTER TABLE positions ADD COLUMN extra_json TEXT DEFAULT '{}'")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise

    def upsert_position(self, position: Position) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO positions (
                    symbol, side, size_usdt, entry_price, stop_loss, take_profit_1, take_profit_2, opened_at, status, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    side=excluded.side,
                    size_usdt=excluded.size_usdt,
                    entry_price=excluded.entry_price,
                    stop_loss=excluded.stop_loss,
                    take_profit_1=excluded.take_profit_1,
                    take_profit_2=excluded.take_profit_2,
                    opened_at=excluded.opened_at,
                    status=excluded.status,
                    extra_json=excluded.extra_json
                """,
                (
                    position.symbol,
                    position.side,
                    position.size_usdt,
                    position.entry_price,
                    position.stop_loss,
                    position.take_profit_1,
                    position.take_profit_2,
                    position.opened_at.isoformat(),
                    position.status,
                    json.dumps(position.extra or {}),
                ),
            )

    def delete_position(self, symbol: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    def load_open_positions(self) -> list[Position]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall()

        positions: list[Position] = []
        for row in rows:
            positions.append(
                Position(
                    symbol=row["symbol"],
                    side=row["side"],
                    size_usdt=float(row["size_usdt"]),
                    entry_price=float(row["entry_price"]),
                    stop_loss=float(row["stop_loss"]),
                    take_profit_1=float(row["take_profit_1"]),
                    take_profit_2=float(row["take_profit_2"]),
                    opened_at=datetime.fromisoformat(row["opened_at"]),
                    status=row["status"],
                    extra=json.loads(row["extra_json"]) if row["extra_json"] else {},
                )
            )
        return positions

    def insert_trade(
        self,
        symbol: str,
        side: str,
        size_usdt: float,
        price: float,
        mode: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (ts, symbol, side, size_usdt, price, mode, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    side,
                    float(size_usdt),
                    float(price),
                    mode,
                    json.dumps(extra or {}),
                ),
            )

    def insert_cycle(self, opportunities: int, opened_positions: int, data: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cycles (ts, opportunities, opened_positions, data_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    int(opportunities),
                    int(opened_positions),
                    json.dumps(data or {}),
                ),
            )

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kv_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, payload, ts),
            )

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value_json FROM kv_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value_json"])
        except Exception:
            return default

    def insert_model_metrics(self, metrics: dict[str, float]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO model_metrics (ts, metrics_json) VALUES (?, ?)",
                (datetime.now(timezone.utc).isoformat(), json.dumps(metrics)),
            )

    def recent_sell_stats(self, hours: int = 24) -> dict[str, float]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, extra_json FROM trades WHERE side='SELL' AND ts >= ?",
                (cutoff.isoformat(),),
            ).fetchall()

        pnl_values = []
        for row in rows:
            try:
                extra = json.loads(row["extra_json"] or "{}")
            except Exception:
                extra = {}
            pnl = extra.get("pnl_pct")
            if pnl is not None:
                pnl_values.append(float(pnl))

        if not pnl_values:
            return {"count": 0.0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0}

        wins = [x for x in pnl_values if x > 0]
        losses = [x for x in pnl_values if x <= 0]
        return {
            "count": float(len(pnl_values)),
            "win_rate": float(len(wins) / len(pnl_values)),
            "avg_win": float(sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss": float(abs(sum(losses) / len(losses))) if losses else 0.0,
        }

    def load_account_state(self, default: dict[str, Any]) -> dict[str, Any]:
        stored = self.get_state("account_state", default)
        merged = default.copy()
        if isinstance(stored, dict):
            merged.update(stored)
        return merged

    def save_account_state(self, state: dict[str, Any]) -> None:
        self.set_state("account_state", state)

    def export_positions_snapshot(self, out_path: str = "artifacts/state/open_positions.json") -> None:
        positions = [asdict(pos) for pos in self.load_open_positions()]
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for item in positions:
            obj = dict(item)
            obj["opened_at"] = obj["opened_at"].isoformat()
            serializable.append(obj)
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
