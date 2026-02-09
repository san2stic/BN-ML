from __future__ import annotations

import logging
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _ensure_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    try:
        with tempfile.NamedTemporaryFile(prefix=".bnml-write-test-", dir=str(path), delete=True):
            pass
    except OSError:
        return False
    return True


def resolve_writable_logs_dir(config: dict) -> tuple[Path, bool]:
    monitoring_cfg = config.get("monitoring", {})
    requested_dir = Path(str(monitoring_cfg.get("logs_dir", "artifacts/logs"))).expanduser()
    if _ensure_writable_dir(requested_dir):
        return requested_dir, False

    fallback_dir = Path(os.environ.get("BNML_FALLBACK_LOGS_DIR", "/tmp/bn_ml/logs")).expanduser()
    if _ensure_writable_dir(fallback_dir):
        print(
            f"[bn_ml] logs_dir '{requested_dir}' is not writable; using fallback '{fallback_dir}'.",
            file=sys.stderr,
        )
        return fallback_dir, True

    raise PermissionError(
        f"Configured logs_dir '{requested_dir}' is not writable and fallback '{fallback_dir}' is unavailable."
    )


def setup_logger(config: dict) -> logging.Logger:
    level_name = config.get("monitoring", {}).get("log_level", "INFO")
    log_level = getattr(logging, level_name.upper(), logging.INFO)

    logs_dir, using_fallback = resolve_writable_logs_dir(config)

    logger = logging.getLogger("bn_ml")
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)

    logger.addHandler(stream)
    try:
        file_handler = RotatingFileHandler(logs_dir / "bot.log", maxBytes=2_000_000, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("File logging disabled: unable to open %s (%s)", logs_dir / "bot.log", exc)

    if using_fallback:
        logger.warning("Configured logs_dir is not writable; using fallback directory %s", logs_dir)

    return logger
