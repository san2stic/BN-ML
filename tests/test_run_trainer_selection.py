from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_trainer import model_needs_training, select_symbols_to_train


def _write_bundle(models_dir: Path, symbol_key: str, trained_at: datetime) -> None:
    out = models_dir / symbol_key
    out.mkdir(parents=True, exist_ok=True)
    (out / "rf.joblib").write_text("dummy", encoding="utf-8")
    (out / "metadata.json").write_text(
        json.dumps({"trained_at": trained_at.isoformat()}),
        encoding="utf-8",
    )


def test_model_needs_training_missing_and_fresh(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    needs_missing, reason_missing = model_needs_training(
        symbol="BTC/USDC",
        models_dir=str(models_dir),
        max_model_age_hours=24,
    )
    assert needs_missing is True
    assert reason_missing == "missing_rf_model"

    now = datetime.now(timezone.utc)
    _write_bundle(models_dir, "ETH_USDC", now)

    needs_fresh, reason_fresh = model_needs_training(
        symbol="ETH/USDC",
        models_dir=str(models_dir),
        max_model_age_hours=24,
    )
    assert needs_fresh is False
    assert reason_fresh == "up_to_date"


def test_model_needs_training_stale(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    stale_dt = datetime.now(timezone.utc) - timedelta(hours=50)
    _write_bundle(models_dir, "SOL_USDC", stale_dt)

    needs, reason = model_needs_training(
        symbol="SOL/USDC",
        models_dir=str(models_dir),
        max_model_age_hours=24,
    )
    assert needs is True
    assert reason.startswith("stale_model_")


def test_select_symbols_to_train_missing_only(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    _write_bundle(models_dir, "ETH_USDC", now)

    selected, skipped = select_symbols_to_train(
        symbols=["BTC/USDC", "ETH/USDC"],
        train_missing_only=True,
        models_dir=str(models_dir),
        max_model_age_hours=24,
    )
    assert selected == ["BTC/USDC"]
    assert skipped == {"ETH/USDC": "up_to_date"}

    selected_all, skipped_all = select_symbols_to_train(
        symbols=["BTC/USDC", "ETH/USDC"],
        train_missing_only=False,
        models_dir=str(models_dir),
        max_model_age_hours=24,
    )
    assert selected_all == ["BTC/USDC", "ETH/USDC"]
    assert skipped_all == {}

