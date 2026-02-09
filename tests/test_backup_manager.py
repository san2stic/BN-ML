from __future__ import annotations

import json
import time

from bn_ml.backup import RuntimeBackupManager


def _write(path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_runtime_backup_copies_runtime_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    _write(tmp_path / "artifacts/state/bn_ml.db", "db")
    _write(tmp_path / "artifacts/metrics/latest_scan.csv", "scan")
    _write(tmp_path / "models/BTC_USDT/rf.joblib", "model")

    cfg = {
        "storage": {
            "sqlite_path": "artifacts/state/bn_ml.db",
            "backup": {
                "enabled": True,
                "base_dir": "artifacts/backups",
                "interval_minutes": 60,
                "keep_last": 5,
                "include_state_db": True,
                "include_models": True,
                "include_metrics": True,
                "include_logs": False,
            },
        },
        "monitoring": {"metrics_dir": "artifacts/metrics", "logs_dir": "artifacts/logs"},
    }

    manager = RuntimeBackupManager(config=cfg)
    out_dir = manager.maybe_backup(force=True)
    assert out_dir is not None
    assert (out_dir / "state/bn_ml.db").exists()
    assert (out_dir / "models/BTC_USDT/rf.joblib").exists()
    assert (out_dir / "metrics/latest_scan.csv").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("copied")


def test_runtime_backup_prunes_old_backups(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "artifacts/state/bn_ml.db", "db")

    cfg = {
        "storage": {
            "sqlite_path": "artifacts/state/bn_ml.db",
            "backup": {
                "enabled": True,
                "base_dir": "artifacts/backups",
                "interval_minutes": 60,
                "keep_last": 1,
                "include_state_db": True,
                "include_models": False,
                "include_metrics": False,
                "include_logs": False,
            },
        },
        "monitoring": {"metrics_dir": "artifacts/metrics", "logs_dir": "artifacts/logs"},
    }

    manager = RuntimeBackupManager(config=cfg)
    first = manager.maybe_backup(force=True)
    time.sleep(0.01)
    second = manager.maybe_backup(force=True)

    assert first is not None and second is not None
    backups = [p for p in (tmp_path / "artifacts/backups").iterdir() if p.is_dir()]
    assert len(backups) == 1
