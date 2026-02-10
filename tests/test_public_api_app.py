from __future__ import annotations

import io
import json
import sqlite3
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from public_api.app import create_app


def _write_test_config(path: Path, metrics_dir: Path, db_path: Path, models_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "monitoring:",
                f"  metrics_dir: {metrics_dir}",
                "storage:",
                f"  sqlite_path: {db_path}",
                "public_api:",
                "  default_limit: 50",
                f"  models_dir: {models_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_scan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "symbol,signal,confidence,global_score,spread_pct,depth_usdt\n"
        "BTC/USDT,BUY,88.0,72.0,0.02,120000\n",
        encoding="utf-8",
    )


def _write_state_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                opportunities INTEGER NOT NULL,
                opened_positions INTEGER NOT NULL,
                data_json TEXT
            );
            """
        )
        payload = {
            "status": "running",
            "phase": "training",
            "trigger": "periodic",
            "current_symbol": "BTC/USDT",
            "progress_pct": 42.5,
            "symbols_queued": 8,
            "symbols_completed": 3,
            "symbols_trained": 3,
            "symbols_errors": 0,
        }
        intelligence = {
            "enabled": True,
            "signal": "SELL",
            "confidence": 79.0,
            "market_score": -0.33,
            "market_score_pct": 33.5,
            "market_regime": "risk_off",
            "model_samples": 42,
            "generated_at": "2026-02-10T10:00:00+00:00",
        }
        conn.execute(
            "INSERT OR REPLACE INTO kv_state (key, value_json, updated_at) VALUES (?, ?, datetime('now'))",
            ("training_status", json.dumps(payload)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO kv_state (key, value_json, updated_at) VALUES (?, ?, datetime('now'))",
            ("santrade_intelligence", json.dumps(intelligence)),
        )
        conn.execute(
            "INSERT INTO cycles (ts, opportunities, opened_positions, data_json) VALUES (?, ?, ?, ?)",
            (
                "2026-02-10T09:55:00+00:00",
                12,
                1,
                json.dumps(
                    {
                        "market_intelligence_signal": "SELL",
                        "market_intelligence_confidence": 75.0,
                        "market_intelligence_score": -0.30,
                        "market_intelligence_regime": "risk_off",
                        "market_intelligence_profile": "defensive",
                    }
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _write_models(models_dir: Path) -> None:
    bundle = models_dir / "BTC_USDT"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "rf.joblib").write_text("dummy-model", encoding="utf-8")
    (bundle / "metadata.json").write_text(
        json.dumps({"symbol": "BTC/USDT", "trained_at": "2026-02-09T20:00:00+00:00"}),
        encoding="utf-8",
    )


def test_site_index_and_favicon(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "configs" / "bot.yaml"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    db_path = tmp_path / "artifacts" / "state" / "bn_ml.db"
    models_dir = tmp_path / "models"
    _write_test_config(config_path, metrics_dir, db_path, models_dir)
    _write_scan(metrics_dir / "latest_scan.csv")

    monkeypatch.setenv("BNML_CONFIG_PATH", str(config_path))

    client = TestClient(create_app())
    index = client.get("/")
    assert index.status_code == 200
    assert "BN-ML Live Predictions" in index.text

    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert "image/svg+xml" in str(favicon.headers.get("content-type", ""))


def test_ws_predictions_stream(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "configs" / "bot.yaml"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    db_path = tmp_path / "artifacts" / "state" / "bn_ml.db"
    models_dir = tmp_path / "models"
    _write_test_config(config_path, metrics_dir, db_path, models_dir)
    _write_scan(metrics_dir / "latest_scan.csv")

    monkeypatch.setenv("BNML_CONFIG_PATH", str(config_path))
    client = TestClient(create_app())

    with client.websocket_connect("/ws/predictions?limit=1") as ws:
        payload = ws.receive_json()
        assert payload["type"] == "predictions"
        assert payload["payload"]["returned_rows"] == 1

    with client.websocket_connect("/ws/market/intelligence?limit=10") as ws:
        payload = ws.receive_json()
        assert payload["type"] == "market_intelligence"
        assert "latest" in payload["payload"]


def test_training_and_model_download_endpoints(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "configs" / "bot.yaml"
    metrics_dir = tmp_path / "artifacts" / "metrics"
    db_path = tmp_path / "artifacts" / "state" / "bn_ml.db"
    models_dir = tmp_path / "models"
    _write_test_config(config_path, metrics_dir, db_path, models_dir)
    _write_scan(metrics_dir / "latest_scan.csv")
    _write_state_db(db_path)
    _write_models(models_dir)

    monkeypatch.setenv("BNML_CONFIG_PATH", str(config_path))
    client = TestClient(create_app())

    training = client.get("/api/v1/training/status")
    assert training.status_code == 200
    assert training.json()["status"] == "running"
    assert training.json()["current_symbol"] == "BTC/USDT"

    market_intel = client.get("/api/v1/market/intelligence")
    assert market_intel.status_code == 200
    assert market_intel.json()["signal"] == "SELL"
    assert market_intel.json()["market_regime"] == "risk_off"

    market_index = client.get("/api/v1/market/index")
    assert market_index.status_code == 200
    assert market_index.json()["signal"] == "SELL"
    assert market_index.json()["index_value"] == 33.5

    market_index_history = client.get("/api/v1/market/index/history?limit=5")
    assert market_index_history.status_code == 200
    assert market_index_history.json()["returned_rows"] >= 1
    assert market_index_history.json()["latest"]["signal"] == "SELL"

    models = client.get("/api/v1/models")
    assert models.status_code == 200
    payload = models.json()
    assert payload["total_bundles"] == 1
    assert payload["rows"][0]["model_key"] == "BTC_USDT"

    archive_all = client.get("/api/v1/models/download")
    assert archive_all.status_code == 200
    assert "application/zip" in str(archive_all.headers.get("content-type", ""))
    with zipfile.ZipFile(io.BytesIO(archive_all.content)) as bundle:
        names = set(bundle.namelist())
    assert "models/BTC_USDT/rf.joblib" in names
    assert "models/BTC_USDT/metadata.json" in names

    archive_one = client.get("/api/v1/models/download?model_key=BTC_USDT")
    assert archive_one.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive_one.content)) as bundle:
        names = set(bundle.namelist())
    assert "BTC_USDT/rf.joblib" in names
    assert "BTC_USDT/metadata.json" in names

    missing = client.get("/api/v1/models/download?model_key=UNKNOWN")
    assert missing.status_code == 404
