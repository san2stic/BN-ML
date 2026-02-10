from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

from bn_ml.config import load_config
from bn_ml.env import load_env_file
from scripts.run_trainer import train_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BN-ML training only and export a models archive (RunPod-safe).")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--archive-path", default="artifacts/exports/models_latest.zip")
    parser.add_argument("--paper", action="store_true", help="Force paper mode for data/training")
    parser.add_argument("--symbol", action="append", default=[], help="Optional symbol(s) to train, can repeat")
    parser.set_defaults(train_missing_only=None)
    parser.add_argument("--train-missing-only", dest="train_missing_only", action="store_true")
    parser.add_argument("--train-all", dest="train_missing_only", action="store_false")
    parser.add_argument("--max-model-age-hours", type=float, default=None)
    return parser.parse_args()


def build_models_archive_file(*, models_dir: Path, archive_path: Path) -> tuple[int, int]:
    src = models_dir.expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Models directory not found: {src}")

    files = sorted([p for p in src.rglob("*") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No model files found in: {src}")

    dst = archive_path.expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")

    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in files:
            rel = file_path.relative_to(src)
            bundle.write(file_path, arcname=str(Path("models") / rel))

    tmp_path.replace(dst)
    return len(files), int(dst.stat().st_size)


def run_train_only(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file()
    config = load_config(args.config)
    symbols = args.symbol if args.symbol else None
    paper = bool(args.paper or config.get("environment", "paper") == "paper")

    result = train_once(
        config=config,
        paper=paper,
        symbols=symbols,
        train_missing_only=args.train_missing_only,
        max_model_age_hours=args.max_model_age_hours,
        models_dir=args.models_dir,
        progress_trigger="runpod_train_only",
    )
    aggregate = result.get("aggregate", {})

    file_count, archive_bytes = build_models_archive_file(
        models_dir=Path(args.models_dir),
        archive_path=Path(args.archive_path),
    )
    payload = {
        "status": "ok",
        "mode": "train_only",
        "paper": paper,
        "models_dir": str(Path(args.models_dir).expanduser().resolve()),
        "archive_path": str(Path(args.archive_path).expanduser().resolve()),
        "archive_files": int(file_count),
        "archive_bytes": int(archive_bytes),
        "symbols_requested": int(aggregate.get("symbols_requested", 0)),
        "symbols_queued_for_training": int(aggregate.get("symbols_queued_for_training", 0)),
        "symbols_trained": int(aggregate.get("symbols_trained", 0)),
        "symbols_skipped_up_to_date": int(aggregate.get("symbols_skipped_up_to_date", 0)),
        "symbols_skipped_errors": int(aggregate.get("symbols_skipped_errors", 0)),
    }
    return payload


def main() -> None:
    args = parse_args()
    payload = run_train_only(args)
    print("RunPod train-only result:")
    for key in (
        "status",
        "mode",
        "paper",
        "archive_path",
        "archive_files",
        "archive_bytes",
        "symbols_requested",
        "symbols_queued_for_training",
        "symbols_trained",
        "symbols_skipped_up_to_date",
        "symbols_skipped_errors",
    ):
        print(f"- {key}: {payload[key]}")


if __name__ == "__main__":
    main()
