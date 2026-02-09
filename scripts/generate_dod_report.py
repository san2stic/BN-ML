from __future__ import annotations

import argparse
import json
from pathlib import Path

from bn_ml.config import load_config
from bn_ml.dod_checks import generate_dod_summary, render_dod_summary_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DoD v1 synthesis report from runtime artifacts.")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--db-path", default=None, help="Override sqlite path")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--daily-dir", default="artifacts/reports/dod/daily")
    parser.add_argument("--out-md", default="artifacts/reports/dod/dod_v1_summary.md")
    parser.add_argument("--out-json", default="artifacts/reports/dod/dod_v1_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    db_path = str(args.db_path or config.get("storage", {}).get("sqlite_path", "artifacts/state/bn_ml.db"))

    summary = generate_dod_summary(
        config=config,
        db_path=db_path,
        days=args.days,
        daily_dir=args.daily_dir,
    )
    markdown = render_dod_summary_markdown(summary)

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"DoD summary markdown: {out_md}")
    print(f"DoD summary json: {out_json}")
    print(markdown)


if __name__ == "__main__":
    main()
