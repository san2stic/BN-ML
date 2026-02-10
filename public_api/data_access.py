from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
import zipfile
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def load_training_status(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "kv_state"):
            return {}
        row = conn.execute("SELECT value_json FROM kv_state WHERE key='training_status'").fetchone()
        if row is None:
            return {}
        payload = _safe_json(row[0])
        return payload if isinstance(payload, dict) else {}
    finally:
        conn.close()


def load_santrade_intelligence(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "kv_state"):
            return {}
        row = conn.execute("SELECT value_json FROM kv_state WHERE key='santrade_intelligence'").fetchone()
        if row is None:
            return {}
        payload = _safe_json(row[0])
        return payload if isinstance(payload, dict) else {}
    finally:
        conn.close()


def market_index_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    market_score = _to_float(payload.get("market_score"), default=0.0)
    raw_index_value = payload.get("market_score_pct")
    index_value = _to_float(raw_index_value, default=((market_score + 1.0) * 50.0))
    signal = _normalize_signal(str(payload.get("signal", "HOLD"))) or "HOLD"
    return {
        "generated_at": payload.get("generated_at"),
        "index_value": _clamp(index_value, 0.0, 100.0),
        "market_score": _clamp(market_score, -1.0, 1.0),
        "signal": signal,
        "confidence": _clamp(_to_float(payload.get("confidence"), default=0.0), 0.0, 100.0),
        "market_regime": str(payload.get("market_regime", "unknown")),
        "profile": str(payload.get("profile", "neutral")),
        "benchmark_symbol": payload.get("benchmark_symbol"),
        "benchmark_price": max(0.0, _to_float(payload.get("benchmark_price"), default=0.0)),
        "predicted_move_pct": _to_float(payload.get("predicted_move_pct"), default=0.0),
        "model_samples": max(0, int(_to_float(payload.get("model_samples"), default=0.0))),
        "enabled": bool(payload.get("enabled", False)),
    }


def load_market_index(db_path: Path) -> dict[str, Any]:
    payload = load_santrade_intelligence(db_path)
    if not payload:
        return {
            "generated_at": None,
            "index_value": 50.0,
            "market_score": 0.0,
            "signal": "HOLD",
            "confidence": 0.0,
            "market_regime": "unknown",
            "profile": "neutral",
            "benchmark_symbol": None,
            "benchmark_price": 0.0,
            "predicted_move_pct": 0.0,
            "model_samples": 0,
            "enabled": False,
        }
    return market_index_from_payload(payload)


def load_market_index_history(db_path: Path, limit: int = 240) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []

    bounded = max(1, min(int(limit), 2000))
    conn = sqlite3.connect(db_path)
    try:
        if not _table_exists(conn, "cycles"):
            return []
        rows = conn.execute(
            """
            SELECT ts, data_json
            FROM cycles
            ORDER BY id DESC
            LIMIT ?
            """,
            (bounded,),
        ).fetchall()
    finally:
        conn.close()

    history: list[dict[str, Any]] = []
    for ts, data_json in reversed(rows):
        payload = _safe_json(data_json)
        if not payload:
            continue
        score_raw = payload.get("market_intelligence_score")
        if score_raw is None:
            continue
        score = _clamp(_to_float(score_raw, default=0.0), -1.0, 1.0)
        signal = _normalize_signal(payload.get("market_intelligence_signal")) or "HOLD"
        history.append(
            {
                "ts": ts,
                "index_value": _clamp((score + 1.0) * 50.0, 0.0, 100.0),
                "market_score": score,
                "confidence": _clamp(_to_float(payload.get("market_intelligence_confidence"), default=0.0), 0.0, 100.0),
                "signal": signal,
                "market_regime": str(payload.get("market_intelligence_regime", "unknown")),
                "profile": str(payload.get("market_intelligence_profile", "neutral")),
            }
        )
    return history


def list_model_bundles(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []

    bundles: list[dict[str, Any]] = []
    for bundle_dir in sorted(models_dir.iterdir()):
        if not bundle_dir.is_dir():
            continue
        files = [p for p in bundle_dir.rglob("*") if p.is_file()]
        if not files:
            continue
        metadata: dict[str, Any] = {}
        metadata_path = bundle_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = _safe_json(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        size_bytes = 0
        for file_path in files:
            try:
                size_bytes += int(file_path.stat().st_size)
            except OSError:
                continue

        bundles.append(
            {
                "model_key": bundle_dir.name,
                "symbol": str(metadata.get("symbol", bundle_dir.name)),
                "trained_at": metadata.get("trained_at"),
                "file_count": len(files),
                "size_bytes": size_bytes,
                "size_mb": float(size_bytes / (1024 * 1024)),
            }
        )
    return bundles


def build_models_archive(models_dir: Path, model_key: str | None = None) -> tuple[str, bytes, int, int]:
    if model_key:
        safe_key = str(model_key).strip()
        if not safe_key or "/" in safe_key or "\\" in safe_key or ".." in safe_key:
            raise ValueError("Invalid model_key")
        target_dir = (models_dir / safe_key).resolve()
        if not target_dir.exists() or not target_dir.is_dir() or target_dir.parent.resolve() != models_dir.resolve():
            raise FileNotFoundError("Model bundle not found")
        files = [p for p in target_dir.rglob("*") if p.is_file()]
        if not files:
            raise FileNotFoundError("Model bundle is empty")
        archive_prefix = safe_key
        archive_name = f"bnml_models_{safe_key}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
        root_dir = target_dir
    else:
        if not models_dir.exists() or not models_dir.is_dir():
            raise FileNotFoundError("Models directory not found")
        files = [p for p in models_dir.rglob("*") if p.is_file()]
        if not files:
            raise FileNotFoundError("No model files found")
        archive_prefix = "models"
        archive_name = f"bnml_models_all_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip"
        root_dir = models_dir

    file_count = 0
    total_bytes = 0
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(files):
            try:
                rel_path = file_path.relative_to(root_dir)
            except ValueError:
                continue
            arcname = Path(archive_prefix) / rel_path
            archive.write(file_path, arcname=str(arcname))
            file_count += 1
            try:
                total_bytes += int(file_path.stat().st_size)
            except OSError:
                continue

    if file_count <= 0:
        raise FileNotFoundError("No model files found")
    return archive_name, buffer.getvalue(), file_count, total_bytes
