from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RuntimeBackupManager:
    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("bn_ml.backup")

        storage_cfg = config.get("storage", {})
        backup_cfg = storage_cfg.get("backup", {}) if isinstance(storage_cfg.get("backup", {}), dict) else {}

        self.enabled = bool(backup_cfg.get("enabled", False))
        self.base_dir = Path(str(backup_cfg.get("base_dir", "artifacts/backups")))
        self.interval_sec = max(60, int(backup_cfg.get("interval_minutes", 60)) * 60)
        self.keep_last = max(1, int(backup_cfg.get("keep_last", 24)))

        self.include_models = bool(backup_cfg.get("include_models", True))
        self.include_state_db = bool(backup_cfg.get("include_state_db", True))
        self.include_metrics = bool(backup_cfg.get("include_metrics", True))
        self.include_logs = bool(backup_cfg.get("include_logs", False))

        self.sqlite_path = Path(str(storage_cfg.get("sqlite_path", "artifacts/state/bn_ml.db")))
        monitoring_cfg = config.get("monitoring", {})
        self.metrics_dir = Path(str(monitoring_cfg.get("metrics_dir", "artifacts/metrics")))
        self.logs_dir = Path(str(monitoring_cfg.get("logs_dir", "artifacts/logs")))
        self.models_dir = Path("models")

        self._last_backup_ts: float = 0.0

    def _backup_label(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    def _copy_path(self, src: Path, dst: Path) -> dict[str, Any]:
        if not src.exists():
            return {"path": str(src), "status": "missing"}
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return {"path": str(src), "status": "copied_dir"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {"path": str(src), "status": "copied_file"}

    def _prune(self) -> None:
        backups = [p for p in self.base_dir.iterdir() if p.is_dir()]
        backups.sort(key=lambda p: p.name)
        while len(backups) > self.keep_last:
            old = backups.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            self.logger.info("Pruned old backup: %s", old)

    def maybe_backup(self, force: bool = False) -> Path | None:
        if not self.enabled:
            return None

        now_ts = time.time()
        if not force and self._last_backup_ts and (now_ts - self._last_backup_ts) < self.interval_sec:
            return None

        self.base_dir.mkdir(parents=True, exist_ok=True)
        out_dir = self.base_dir / self._backup_label()
        out_dir.mkdir(parents=True, exist_ok=True)

        copied: list[dict[str, Any]] = []
        if self.include_state_db:
            copied.append(self._copy_path(self.sqlite_path, out_dir / "state" / self.sqlite_path.name))
        if self.include_models:
            copied.append(self._copy_path(self.models_dir, out_dir / "models"))
        if self.include_metrics:
            copied.append(self._copy_path(self.metrics_dir, out_dir / "metrics"))
        if self.include_logs:
            copied.append(self._copy_path(self.logs_dir, out_dir / "logs"))

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "copied": copied,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        self._last_backup_ts = now_ts
        self._prune()
        self.logger.info("Runtime backup created: %s", out_dir)
        return out_dir
