from __future__ import annotations

import argparse
from datetime import date, datetime

from bn_ml.config import load_config
from bn_ml.dod_checks import evaluate_dod_daily, write_dod_daily_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily DoD risk/compliance checks on runtime artifacts.")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--db-path", default=None, help="Override sqlite path")
    parser.add_argument("--date", default=None, help="UTC day to evaluate in YYYY-MM-DD (default: today)")
    parser.add_argument("--out-dir", default="artifacts/reports/dod/daily")
    parser.add_argument("--fail-on-violation", action="store_true")
    return parser.parse_args()


def _parse_day(raw: str | None) -> date | None:
    if not raw:
        return None
    return datetime.strptime(raw, "%Y-%m-%d").date()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    db_path = str(args.db_path or config.get("storage", {}).get("sqlite_path", "artifacts/state/bn_ml.db"))
    target_day = _parse_day(args.date)

    result = evaluate_dod_daily(config=config, db_path=db_path, day_value=target_day)
    json_path, md_path = write_dod_daily_report(result=result, out_dir=args.out_dir)

    print(f"DoD daily check written: {json_path}")
    print(f"DoD daily markdown: {md_path}")
    print(f"Status: {result['status']} | violations={result['violations_count']}")
    if result.get("violations"):
        for item in result["violations"]:
            print(f"- {item.get('id')}: {item.get('detail')}")

    if args.fail_on_violation and int(result.get("violations_count", 0)) > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
