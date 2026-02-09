from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(config: dict) -> logging.Logger:
    level_name = config.get("monitoring", {}).get("log_level", "INFO")
    log_level = getattr(logging, level_name.upper(), logging.INFO)

    logs_dir = Path(config.get("monitoring", {}).get("logs_dir", "artifacts/logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bn_ml")
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)

    file_handler = RotatingFileHandler(logs_dir / "bot.log", maxBytes=2_000_000, backupCount=5)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)

    return logger
