from __future__ import annotations

import csv
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIGNALS = {"BUY", "SELL", "HOLD"}
SCAN_NUMERIC_FIELDS = (
    "confidence",
    "ml_score",
    "technical_score",
    "momentum_score",
    "global_score",
    "spread_pct",
    "depth_usdt",
    "correlation_btc",
)


def _normalize_signal(value: str | None) -> str | None:
    if not value:
        return None
    candidate = str(value).strip().upper()
    return candidate if candidate in SIGNALS else None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def read_prediction_snapshot(scan_path: Path, limit: int | None = 100, signal: str | None = None) -> dict[str, Any]:
    if not scan_path.exists():
        return {"generated_at": None, "age_sec": None, "total_rows": 0, "returned_rows": 0, "rows": []}

    requested_signal = _normalize_signal(signal)
    rows: list[dict[str, Any]] = []

    with scan_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = dict(raw)
            row["signal"] = _normalize_signal(row.get("signal")) or "HOLD"
            for field in SCAN_NUMERIC_FIELDS:
                row[field] = _to_float(row.get(field), default=0.0)
            if requested_signal and row["signal"] != requested_signal:
                continue
            rows.append(row)

    rows.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    total_rows = len(rows)
    if limit is not None:
        bounded = max(1, min(int(limit), 2000))
        rows = rows[:bounded]

    mtime = scan_path.stat().st_mtime
    generated_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    age_sec = max(0.0, time.time() - mtime)

    return {
        "generated_at": generated_at,
        "age_sec": age_sec,
        "total_rows": total_rows,
        "returned_rows": len(rows),
        "rows": rows,
    }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def load_account_state(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "kv_state"):
            return {}
        row = conn.execute("SELECT value_json FROM kv_state WHERE key='account_state'").fetchone()
        if row is None:
            return {}
        return _safe_json(row[0])
    finally:
        conn.close()


def load_recent_trades(db_path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []

    bounded = max(1, min(int(limit), 1000))
    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "trades"):
            return []
        rows = conn.execute(
            """
            SELECT id, ts, symbol, side, size_usdt, price, mode, extra_json
            FROM trades
            ORDER BY id DESC
            LIMIT ?
            """,
            (bounded,),
        ).fetchall()
    finally:
        conn.close()

    payload: list[dict[str, Any]] = []
    for row in rows:
        extra = _safe_json(row[7])
        payload.append(
            {
                "id": int(row[0]),
                "ts": row[1],
                "symbol": row[2],
                "side": row[3],
                "size_usdt": _to_float(row[4]),
                "price": _to_float(row[5]),
                "mode": row[6],
                "extra": extra,
            }
        )
    return payload


def _query_count(conn: sqlite3.Connection, table: str, where: str = "") -> int:
    if not _table_exists(conn, table):
        return 0
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql = f"{sql} WHERE {where}"
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0


def load_runtime_summary(db_path: Path) -> dict[str, Any]:
    summary = {"open_positions": 0, "total_trades": 0, "total_cycles": 0}
    if not db_path.exists():
        return summary

    conn = sqlite3.connect(db_path)
    try:
        summary["open_positions"] = _query_count(conn, "positions", "status='OPEN'")
        summary["total_trades"] = _query_count(conn, "trades")
        summary["total_cycles"] = _query_count(conn, "cycles")
        return summary
    finally:
        conn.close()
