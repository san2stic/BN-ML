from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import scripts.runpod_train_only as runpod_train_only


def test_build_models_archive_file(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    bundle = models_dir / "BTC_USDC"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "rf.joblib").write_text("rf", encoding="utf-8")
    (bundle / "metadata.json").write_text("{}", encoding="utf-8")

    archive_path = tmp_path / "exports" / "models_latest.zip"
    file_count, size_bytes = runpod_train_only.build_models_archive_file(
        models_dir=models_dir,
        archive_path=archive_path,
    )
    assert file_count == 2
    assert size_bytes > 0
    assert archive_path.exists()

    with zipfile.ZipFile(archive_path, mode="r") as archive:
        names = set(archive.namelist())
    assert "models/BTC_USDC/rf.joblib" in names
    assert "models/BTC_USDC/metadata.json" in names


def test_run_train_only_uses_trainer_and_exports_archive(tmp_path: Path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    archive_path = tmp_path / "artifacts" / "exports" / "models_latest.zip"

    def fake_load_env_file() -> None:
        return None

    def fake_load_config(path: str):  # type: ignore[no-untyped-def]
        return {"environment": "paper"}

    def fake_train_once(**kwargs):  # type: ignore[no-untyped-def]
        out = Path(kwargs["models_dir"]) / "ETH_USDC"
        out.mkdir(parents=True, exist_ok=True)
        (out / "rf.joblib").write_text("rf", encoding="utf-8")
        (out / "metadata.json").write_text("{}", encoding="utf-8")
        return {
            "aggregate": {
                "symbols_requested": 5,
                "symbols_queued_for_training": 3,
                "symbols_trained": 3,
                "symbols_skipped_up_to_date": 2,
                "symbols_skipped_errors": 0,
            }
        }

    monkeypatch.setattr(runpod_train_only, "load_env_file", fake_load_env_file)
    monkeypatch.setattr(runpod_train_only, "load_config", fake_load_config)
    monkeypatch.setattr(runpod_train_only, "train_once", fake_train_once)

    args = argparse.Namespace(
        config="configs/bot.yaml",
        models_dir=str(models_dir),
        archive_path=str(archive_path),
        paper=True,
        symbol=[],
        train_missing_only=True,
        max_model_age_hours=None,
    )
    result = runpod_train_only.run_train_only(args)
    assert result["status"] == "ok"
    assert result["mode"] == "train_only"
    assert result["paper"] is True
    assert result["archive_files"] == 2
    assert result["symbols_trained"] == 3
    assert archive_path.exists()
