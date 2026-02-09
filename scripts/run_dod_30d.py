from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bn_ml.config import load_config
from bn_ml.dod_checks import evaluate_dod_daily, generate_dod_summary, render_dod_summary_markdown, write_dod_daily_report
from bn_ml.env import load_env_file
from scripts.run_bot import TradingRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 30-day DoD paper campaign with automated daily checks.")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--disable-retrain", action="store_true")
    parser.add_argument("--daily-out-dir", default="artifacts/reports/dod/daily")
    parser.add_argument("--summary-out", default="artifacts/reports/dod/dod_v1_summary.md")
    parser.add_argument("--summary-json", default="artifacts/reports/dod/dod_v1_summary.json")
    parser.add_argument("--strict", action="store_true", help="Stop campaign if a daily violation is detected.")
    return parser.parse_args()


def _force_paper_real_market_data(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["environment"] = "paper"
    cfg.setdefault("data", {})
    cfg["data"]["paper_market_data_mode"] = "live"
    return cfg


def _utc_midnight_next(now: datetime) -> datetime:
    midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return midnight + timedelta(days=1)


def main() -> None:
    load_env_file()
    args = parse_args()
    config = _force_paper_real_market_data(load_config(args.config))
    runtime = TradingRuntime(config=config, paper=True, disable_retrain=args.disable_retrain)

    start = datetime.now(timezone.utc)
    end = start + timedelta(days=max(1, int(args.days)))
    next_check_at = _utc_midnight_next(start)
    scan_interval = int(config.get("scanner", {}).get("scan_interval_sec", 300))

    print(f"DoD campaign started (UTC): {start.isoformat()}")
    print(f"Scheduled end (UTC): {end.isoformat()}")
    print(f"Daily check directory: {args.daily_out_dir}")

    try:
        while datetime.now(timezone.utc) < end:
            loop_start = time.time()
            runtime.run_cycle()
            now = datetime.now(timezone.utc)

            while now >= next_check_at:
                day_to_check = (next_check_at - timedelta(days=1)).date()
                result = evaluate_dod_daily(
                    config=config,
                    db_path=str(config.get("storage", {}).get("sqlite_path", "artifacts/state/bn_ml.db")),
                    day_value=day_to_check,
                )
                json_path, _ = write_dod_daily_report(result=result, out_dir=args.daily_out_dir)
                print(f"[DAILY CHECK] {day_to_check.isoformat()} status={result['status']} -> {json_path}")
                if args.strict and int(result.get("violations_count", 0)) > 0:
                    print("Strict mode enabled and violations detected. Stopping campaign.")
                    return
                next_check_at = next_check_at + timedelta(days=1)

            elapsed = time.time() - loop_start
            sleep_sec = max(0.0, scan_interval - elapsed)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    except KeyboardInterrupt:
        print("DoD campaign interrupted by user.")
    finally:
        runtime.shutdown()

        summary = generate_dod_summary(
            config=config,
            db_path=str(config.get("storage", {}).get("sqlite_path", "artifacts/state/bn_ml.db")),
            days=max(1, int(args.days)),
            daily_dir=args.daily_out_dir,
        )
        md = render_dod_summary_markdown(summary)

        out_md = Path(args.summary_out)
        out_json = Path(args.summary_json)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"DoD summary markdown: {out_md}")
        print(f"DoD summary json: {out_json}")


if __name__ == "__main__":
    main()
