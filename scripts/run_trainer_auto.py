from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from typing import Any

from bn_ml.config import load_config
from scripts.run_trainer import train_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BN-ML trainer in a periodic loop")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--symbol", action="append", default=[], help="Optional symbol(s) to train, can repeat")
    parser.set_defaults(train_missing_only=True)
    parser.add_argument(
        "--train-missing-only",
        dest="train_missing_only",
        action="store_true",
        help="Train only symbols with missing or stale models (default for auto trainer)",
    )
    parser.add_argument(
        "--train-all",
        dest="train_missing_only",
        action="store_false",
        help="Train all candidate symbols on each cycle",
    )
    parser.add_argument(
        "--max-model-age-hours",
        type=float,
        default=None,
        help="When using --train-missing-only, retrain models older than this age (default from config, fallback 24h)",
    )
    parser.add_argument("--models-dir", default="models", help="Model directory root")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=None,
        help="Loop interval in seconds (default: model.retrain_interval_hours * 3600)",
    )
    parser.add_argument("--startup-delay-seconds", type=float, default=0.0, help="Delay before first training cycle")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Max number of cycles to run (0 = infinite, useful for tests/ops dry run)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit on first training error instead of retrying at next cycle",
    )
    return parser.parse_args()


def resolve_interval_seconds(config: dict[str, Any], override_seconds: float | None) -> float:
    if override_seconds is not None:
        try:
            return max(30.0, float(override_seconds))
        except (TypeError, ValueError):
            return 30.0

    model_cfg = config.get("model", {})
    try:
        retrain_interval_hours = float(model_cfg.get("retrain_interval_hours", 6))
    except (TypeError, ValueError):
        retrain_interval_hours = 6.0
    return max(30.0, retrain_interval_hours * 3600.0)


def _sleep_with_interrupt(total_seconds: float) -> None:
    remaining = max(0.0, float(total_seconds))
    while remaining > 0:
        chunk = min(1.0, remaining)
        time.sleep(chunk)
        remaining -= chunk


def run_loop(
    *,
    config: dict[str, Any],
    paper: bool,
    symbols: list[str] | None,
    train_missing_only: bool,
    max_model_age_hours: float | None,
    models_dir: str,
    interval_seconds: float,
    startup_delay_seconds: float,
    max_cycles: int,
    fail_fast: bool,
) -> int:
    if startup_delay_seconds > 0:
        print(f"[trainer-auto] Startup delay: {startup_delay_seconds:.1f}s")
        _sleep_with_interrupt(startup_delay_seconds)

    cycle = 0
    while True:
        cycle += 1
        cycle_started_at = datetime.now(timezone.utc).isoformat()
        print(f"[trainer-auto] Cycle {cycle} started at {cycle_started_at}")
        try:
            result = train_once(
                config=config,
                paper=paper,
                symbols=symbols,
                train_missing_only=train_missing_only,
                max_model_age_hours=max_model_age_hours,
                models_dir=models_dir,
            )
            aggregate = result.get("aggregate", {})
            print(
                "[trainer-auto] Cycle %s done: trained=%s queued=%s errors=%s up_to_date=%s"
                % (
                    cycle,
                    aggregate.get("symbols_trained", 0),
                    aggregate.get("symbols_queued_for_training", 0),
                    aggregate.get("symbols_skipped_errors", 0),
                    aggregate.get("symbols_skipped_up_to_date", 0),
                )
            )
        except KeyboardInterrupt:
            print("[trainer-auto] Interrupted by user.")
            return 130
        except Exception as exc:
            print(f"[trainer-auto] Cycle {cycle} failed: {exc}")
            if fail_fast:
                return 1

        if max_cycles > 0 and cycle >= max_cycles:
            print(f"[trainer-auto] Reached max cycles: {max_cycles}")
            return 0

        print(f"[trainer-auto] Sleeping {interval_seconds:.1f}s before next cycle")
        _sleep_with_interrupt(interval_seconds)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paper = args.paper or config.get("environment", "paper") == "paper"
    symbols = args.symbol if args.symbol else None
    interval_seconds = resolve_interval_seconds(config=config, override_seconds=args.interval_seconds)

    print(
        "[trainer-auto] Starting loop: paper=%s train_missing_only=%s interval_seconds=%.1f max_cycles=%s"
        % (
            paper,
            bool(args.train_missing_only),
            interval_seconds,
            args.max_cycles,
        )
    )

    exit_code = run_loop(
        config=config,
        paper=paper,
        symbols=symbols,
        train_missing_only=bool(args.train_missing_only),
        max_model_age_hours=args.max_model_age_hours,
        models_dir=args.models_dir,
        interval_seconds=interval_seconds,
        startup_delay_seconds=float(args.startup_delay_seconds),
        max_cycles=int(args.max_cycles),
        fail_fast=bool(args.fail_fast),
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
