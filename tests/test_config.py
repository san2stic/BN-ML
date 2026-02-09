from __future__ import annotations

from pathlib import Path

import pytest

from bn_ml.config import load_config


def test_load_config_reads_explicit_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "custom.yaml"
    cfg_path.write_text("environment: paper\nbase_quote: USDT\n", encoding="utf-8")

    loaded = load_config(cfg_path)
    assert loaded["environment"] == "paper"
    assert loaded["base_quote"] == "USDT"


def test_load_config_missing_custom_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_uses_packaged_default_when_default_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    loaded = load_config()
    assert loaded["exchange"]["name"] == "binance"
    assert "risk" in loaded
