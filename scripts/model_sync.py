from __future__ import annotations

import argparse
import copy
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from bn_ml.config import load_config
from bn_ml.env import load_env_file
from bn_ml.model_sync import (
    GitSyncSettings,
    ModelSyncError,
    next_daily_run,
    parse_daily_time,
    publish_models_to_git,
    pull_models_from_git,
)
from scripts.run_trainer import train_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronize BN-ML models with a dedicated GitHub repository.")
    sub = parser.add_subparsers(dest="command", required=True)

    publish = sub.add_parser("publish", help="Train (optional) then push models to GitHub.")
    _add_common_sync_args(publish)
    _add_publish_train_args(publish)

    pull = sub.add_parser("pull", help="Pull model repo then sync local models directory.")
    _add_common_sync_args(pull)

    daemon = sub.add_parser("daemon", help="Run daily scheduled sync as publisher or client.")
    daemon.add_argument("--role", choices=["publisher", "client"], required=True)
    daemon.add_argument("--at", default=None, help="Daily local schedule (HH:MM). Defaults: publisher=00:00, client=06:00")
    daemon.add_argument("--run-on-start", action="store_true", help="Execute one run immediately at startup.")
    daemon.add_argument("--poll-interval-sec", type=int, default=30)
    _add_common_sync_args(daemon)
    _add_publish_train_args(daemon)
    return parser.parse_args()


def _add_common_sync_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--repo-dir", default=None, help="Git repo path used to publish/distribute models")
    parser.add_argument("--models-dir", default="models", help="Local model directory to publish/populate")
    parser.add_argument("--repo-models-subdir", default=None, help="Model folder inside repo (default: model_sync.repo_models_subdir or models)")
    parser.add_argument("--remote", default=None, help="Git remote name (default: model_sync.remote or origin)")
    parser.add_argument("--branch", default=None, help="Git branch (default: model_sync.branch or main)")
    parser.add_argument("--allow-dirty-worktree", action="store_true", help="Allow sync even if repo worktree is dirty")


def _add_publish_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--skip-training", action="store_true", help="Skip trainer before publishing")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--symbol", action="append", default=[], help="Optional symbol(s) to train, can repeat")
    parser.set_defaults(train_missing_only=None)
    parser.add_argument(
        "--train-missing-only",
        dest="train_missing_only",
        action="store_true",
        help="Train only symbols with missing/stale models",
    )
    parser.add_argument(
        "--train-all",
        dest="train_missing_only",
        action="store_false",
        help="Train all symbols",
    )
    parser.add_argument("--max-model-age-hours", type=float, default=None)
    parser.add_argument("--commit-message", default=None, help="Git commit message for publish")


def _sync_cfg(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("model_sync", {})
    return raw if isinstance(raw, dict) else {}


def _resolve_schedule(args: argparse.Namespace, config: dict[str, Any], role: str) -> tuple[int, int]:
    if args.at:
        return parse_daily_time(args.at)
    sync_cfg = _sync_cfg(config)
    role_cfg = sync_cfg.get(role, {}) if isinstance(sync_cfg.get(role, {}), dict) else {}
    default_value = "00:00" if role == "publisher" else "06:00"
    return parse_daily_time(str(role_cfg.get("schedule", default_value)))


def _resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> GitSyncSettings:
    sync_cfg = _sync_cfg(config)
    repo_dir_raw = args.repo_dir or sync_cfg.get("repo_dir")
    if not repo_dir_raw:
        raise ModelSyncError("Missing repo path. Set --repo-dir or model_sync.repo_dir in config.")

    repo_models_subdir = str(args.repo_models_subdir or sync_cfg.get("repo_models_subdir", "models"))
    remote = str(args.remote or sync_cfg.get("remote", "origin"))
    branch = str(args.branch or sync_cfg.get("branch", "main"))
    return GitSyncSettings(
        repo_dir=Path(repo_dir_raw).expanduser().resolve(),
        models_dir=Path(args.models_dir).expanduser().resolve(),
        repo_models_subdir=repo_models_subdir,
        remote=remote,
        branch=branch,
        allow_dirty_worktree=bool(args.allow_dirty_worktree),
    )


def _run_publish(args: argparse.Namespace) -> None:
    load_env_file()
    config = load_config(args.config)
    settings = _resolve_settings(args, config)
    sync_cfg = _sync_cfg(config)
    publisher_cfg = sync_cfg.get("publisher", {}) if isinstance(sync_cfg.get("publisher", {}), dict) else {}

    train_before_push = bool(publisher_cfg.get("train_before_push", True))
    if args.skip_training:
        train_before_push = False

    if train_before_push:
        paper = bool(args.paper or config.get("environment", "paper") == "paper")
        symbols = args.symbol if args.symbol else None
        train_config = copy.deepcopy(config)
        train_result = train_once(
            config=train_config,
            paper=paper,
            symbols=symbols,
            train_missing_only=args.train_missing_only,
            max_model_age_hours=args.max_model_age_hours,
            models_dir=str(settings.models_dir),
        )
        aggregate = train_result.get("aggregate", {})
        print("Training summary:")
        for key in (
            "symbols_requested",
            "symbols_queued_for_training",
            "symbols_trained",
            "symbols_skipped_up_to_date",
            "symbols_skipped_errors",
        ):
            print(f"- {key}: {aggregate.get(key, 0)}")

    result = publish_models_to_git(settings=settings, commit_message=args.commit_message)
    print("Publish result:")
    print(f"- committed: {result['committed']}")
    print(f"- pushed: {result['pushed']}")
    print(f"- branch: {result['branch']}")
    print(f"- remote: {result['remote']}")
    print(f"- changed_paths: {len(result['changed_paths'])}")


def _run_pull(args: argparse.Namespace) -> None:
    load_env_file()
    config = load_config(args.config)
    settings = _resolve_settings(args, config)
    result = pull_models_from_git(settings=settings)
    print("Pull result:")
    print(f"- pulled: {result['pulled']}")
    print(f"- model_files: {result['model_files']}")
    print(f"- models_dir: {result['models_dir']}")
    print(f"- branch: {result['branch']}")
    print(f"- remote: {result['remote']}")


def _run_daemon(args: argparse.Namespace) -> None:
    load_env_file()
    config = load_config(args.config)
    role = str(args.role)
    hour, minute = _resolve_schedule(args=args, config=config, role=role)
    now = datetime.now().astimezone()
    next_run = now if bool(args.run_on_start) else next_daily_run(now, hour, minute)
    poll_interval = max(5, int(args.poll_interval_sec))

    print(f"Starting {role} daemon on local schedule {hour:02d}:{minute:02d}")
    while True:
        now = datetime.now().astimezone()
        wait_sec = (next_run - now).total_seconds()
        if wait_sec > 0:
            time.sleep(min(wait_sec, poll_interval))
            continue

        started = datetime.now().astimezone().isoformat()
        print(f"[{started}] Running {role} sync job")
        try:
            if role == "publisher":
                _run_publish(args)
            else:
                _run_pull(args)
            finished = datetime.now().astimezone().isoformat()
            print(f"[{finished}] {role} sync job complete")
        except Exception as exc:
            failed = datetime.now().astimezone().isoformat()
            print(f"[{failed}] {role} sync job failed: {exc}")

        next_run = next_daily_run(datetime.now().astimezone(), hour, minute)


def main() -> None:
    args = parse_args()
    try:
        if args.command == "publish":
            _run_publish(args)
        elif args.command == "pull":
            _run_pull(args)
        elif args.command == "daemon":
            _run_daemon(args)
        else:
            raise SystemExit(f"Unknown command: {args.command}")
    except ModelSyncError as exc:
        print(f"Model sync error: {exc}")
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
