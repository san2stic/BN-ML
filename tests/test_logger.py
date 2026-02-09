from __future__ import annotations

import logging
from pathlib import Path

import monitoring.logger as logger_module


def _reset_bnml_logger() -> None:
    logger = logging.getLogger("bn_ml")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _flush_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def test_setup_logger_writes_to_configured_logs_dir(tmp_path) -> None:
    _reset_bnml_logger()
    logs_dir = tmp_path / "logs"

    logger = logger_module.setup_logger({"monitoring": {"logs_dir": str(logs_dir)}})
    logger.info("hello configured logs")
    _flush_handlers(logger)

    assert (logs_dir / "bot.log").exists()
    _reset_bnml_logger()


def test_setup_logger_falls_back_when_configured_logs_dir_unwritable(tmp_path, monkeypatch) -> None:
    _reset_bnml_logger()
    requested = tmp_path / "blocked"
    fallback = tmp_path / "fallback"
    fallback.mkdir(parents=True, exist_ok=True)
    original_ensure_writable_dir = logger_module._ensure_writable_dir

    def _fake_ensure_writable_dir(path: Path) -> bool:
        if path == requested:
            return False
        if path == fallback:
            return True
        return original_ensure_writable_dir(path)

    monkeypatch.setattr(logger_module, "_ensure_writable_dir", _fake_ensure_writable_dir)
    monkeypatch.setenv("BNML_FALLBACK_LOGS_DIR", str(fallback))

    logger = logger_module.setup_logger({"monitoring": {"logs_dir": str(requested)}})
    logger.info("hello fallback logs")
    _flush_handlers(logger)

    assert (fallback / "bot.log").exists()
    _reset_bnml_logger()
