from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _to_fraction(value: Any, default: float) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = default
    if raw > 1.0:
        raw = raw / 100.0
    return max(0.0, raw)


def _day_bounds_utc(day_value: date | None = None) -> tuple[datetime, datetime]:
    day_obj = day_value or datetime.now(timezone.utc).date()
    start = datetime.combine(day_obj, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _read_window_tables(db_path: str, start: datetime, end: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()

    with sqlite3.connect(path) as conn:
        trades = pd.read_sql_query(
            """
            SELECT ts, symbol, side, size_usdt, price, extra_json
            FROM trades
            WHERE ts >= ? AND ts < ?
            ORDER BY ts ASC
            """,
            conn,
            params=(start.isoformat(), end.isoformat()),
        )
        cycles = pd.read_sql_query(
            """
            SELECT ts, opportunities, opened_positions, data_json
            FROM cycles
            WHERE ts >= ? AND ts < ?
            ORDER BY ts ASC
            """,
            conn,
            params=(start.isoformat(), end.isoformat()),
        )

    for frame in (trades, cycles):
        if not frame.empty and "ts" in frame.columns:
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    return trades, cycles


def _parse_json_column(frame: pd.DataFrame, source_col: str, out_col: str = "extra") -> pd.DataFrame:
    out = frame.copy()

    def _safe_parse(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    out[out_col] = out[source_col].apply(_safe_parse) if (not out.empty and source_col in out.columns) else [{}] * len(out)
    return out


def evaluate_dod_daily(config: dict[str, Any], db_path: str, day_value: date | None = None) -> dict[str, Any]:
    start, end = _day_bounds_utc(day_value)
    trades, cycles = _read_window_tables(db_path=db_path, start=start, end=end)

    trades = _parse_json_column(trades, "extra_json", out_col="extra") if not trades.empty else trades
    cycles = _parse_json_column(cycles, "data_json", out_col="data") if not cycles.empty else cycles

    risk_cfg = config.get("risk", {})
    breaker_cfg = risk_cfg.get("circuit_breakers", {})
    max_positions = int(risk_cfg.get("max_positions", 5))
    drawdown_stop_pct = float(breaker_cfg.get("daily_drawdown_stop_pct", -5.0))
    vol_spike_ratio = float(breaker_cfg.get("volatility_spike_ratio", 1.5))
    drift_block_enabled = bool(breaker_cfg.get("drift_block_enabled", False))

    max_daily_risk_pct = _to_fraction(risk_cfg.get("max_daily_risk_pct", 0.02), 0.02) * 100.0
    max_weekly_risk_pct = _to_fraction(risk_cfg.get("max_weekly_risk_pct", 0.06), 0.06) * 100.0

    buys = trades[trades["side"] == "BUY"].copy() if not trades.empty else pd.DataFrame()
    sells = trades[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()

    if not sells.empty:
        sells["pnl_usdt"] = pd.to_numeric(sells["extra"].apply(lambda x: x.get("pnl_usdt")), errors="coerce").fillna(0.0)
    if not cycles.empty:
        cycles["open_positions"] = pd.to_numeric(cycles["data"].apply(lambda x: x.get("open_positions")), errors="coerce").fillna(0.0)
        cycles["daily_pnl_pct"] = pd.to_numeric(cycles["data"].apply(lambda x: x.get("daily_pnl_pct")), errors="coerce").fillna(0.0)
        cycles["opened_positions"] = pd.to_numeric(cycles["opened_positions"], errors="coerce").fillna(0.0)
    if not buys.empty:
        buys["daily_risk_used_pct"] = pd.to_numeric(buys["extra"].apply(lambda x: x.get("daily_risk_used_pct")), errors="coerce")
        buys["weekly_risk_used_pct"] = pd.to_numeric(buys["extra"].apply(lambda x: x.get("weekly_risk_used_pct")), errors="coerce")
        buys["market_volatility_ratio"] = pd.to_numeric(
            buys["extra"].apply(lambda x: x.get("market_volatility_ratio")), errors="coerce"
        )
        buys["market_drift_detected"] = buys["extra"].apply(lambda x: bool(x.get("market_drift_detected", False)))

    violations: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    max_open_positions_seen = int(cycles["open_positions"].max()) if not cycles.empty else 0
    if max_open_positions_seen > max_positions:
        violations.append(
            {
                "id": "max_positions_exceeded",
                "detail": f"Observed open_positions={max_open_positions_seen} > max_positions={max_positions}",
            }
        )
    checks["max_open_positions_seen"] = max_open_positions_seen

    if not cycles.empty:
        breaker_rows = cycles[(cycles["daily_pnl_pct"] <= drawdown_stop_pct) & (cycles["opened_positions"] > 0)]
        if not breaker_rows.empty:
            violations.append(
                {
                    "id": "opened_while_daily_drawdown_breaker_active",
                    "detail": f"{len(breaker_rows)} cycle(s) opened positions while daily_pnl_pct <= {drawdown_stop_pct}",
                }
            )

    if not buys.empty:
        over_daily = buys[buys["daily_risk_used_pct"] >= max_daily_risk_pct] if "daily_risk_used_pct" in buys.columns else pd.DataFrame()
        if not over_daily.empty:
            violations.append(
                {
                    "id": "buy_after_daily_risk_budget_exhausted",
                    "detail": f"{len(over_daily)} BUY trade(s) with daily_risk_used_pct >= {max_daily_risk_pct:.4f}",
                }
            )

        over_weekly = buys[buys["weekly_risk_used_pct"] >= max_weekly_risk_pct] if "weekly_risk_used_pct" in buys.columns else pd.DataFrame()
        if not over_weekly.empty:
            violations.append(
                {
                    "id": "buy_after_weekly_risk_budget_exhausted",
                    "detail": f"{len(over_weekly)} BUY trade(s) with weekly_risk_used_pct >= {max_weekly_risk_pct:.4f}",
                }
            )

        over_vol = buys[buys["market_volatility_ratio"] > vol_spike_ratio] if "market_volatility_ratio" in buys.columns else pd.DataFrame()
        if not over_vol.empty:
            violations.append(
                {
                    "id": "buy_while_volatility_breaker_active",
                    "detail": f"{len(over_vol)} BUY trade(s) with market_volatility_ratio > {vol_spike_ratio:.4f}",
                }
            )

        if drift_block_enabled:
            drift_rows = buys[buys["market_drift_detected"]]
            if not drift_rows.empty:
                violations.append(
                    {
                        "id": "buy_while_drift_breaker_active",
                        "detail": f"{len(drift_rows)} BUY trade(s) executed while market_drift_detected=true",
                    }
                )

    realized_pnl_usdt = float(sells["pnl_usdt"].sum()) if not sells.empty else 0.0
    win_rate = float((sells["pnl_usdt"] > 0).mean()) if (not sells.empty and "pnl_usdt" in sells.columns) else 0.0

    result = {
        "date_utc": start.date().isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "status": "PASS" if not violations else "FAIL",
        "violations_count": len(violations),
        "violations": violations,
        "metrics": {
            "cycles": int(len(cycles)),
            "trades": int(len(trades)),
            "buys": int(len(buys)),
            "sells": int(len(sells)),
            "realized_pnl_usdt": realized_pnl_usdt,
            "sell_win_rate": win_rate,
            "max_open_positions_seen": max_open_positions_seen,
        },
        "checks": checks,
    }
    return result


def write_dod_daily_report(result: dict[str, Any], out_dir: str = "artifacts/reports/dod/daily") -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    day_key = str(result.get("date_utc", datetime.now(timezone.utc).date().isoformat()))
    json_path = target / f"{day_key}.json"
    md_path = target / f"{day_key}.md"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# DoD Daily Check - {day_key}",
        "",
        f"- Status: **{result.get('status', 'UNKNOWN')}**",
        f"- Violations: {int(result.get('violations_count', 0))}",
        f"- Cycles: {int(result.get('metrics', {}).get('cycles', 0))}",
        f"- Trades: {int(result.get('metrics', {}).get('trades', 0))}",
        f"- BUY: {int(result.get('metrics', {}).get('buys', 0))}",
        f"- SELL: {int(result.get('metrics', {}).get('sells', 0))}",
        f"- Realized PnL (USDT): {float(result.get('metrics', {}).get('realized_pnl_usdt', 0.0)):.4f}",
        "",
    ]
    violations = result.get("violations", [])
    if violations:
        lines.append("## Violations")
        for item in violations:
            lines.append(f"- `{item.get('id', 'unknown')}`: {item.get('detail', '')}")
    else:
        lines.append("No violations detected.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def generate_dod_summary(
    config: dict[str, Any],
    db_path: str,
    days: int = 30,
    daily_dir: str = "artifacts/reports/dod/daily",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=max(1, int(days)))
    trades, cycles = _read_window_tables(db_path=db_path, start=start, end=now)
    trades = _parse_json_column(trades, "extra_json", out_col="extra") if not trades.empty else trades
    cycles = _parse_json_column(cycles, "data_json", out_col="data") if not cycles.empty else cycles

    if not trades.empty:
        sells = trades[trades["side"] == "SELL"].copy()
        sells["pnl_usdt"] = pd.to_numeric(sells["extra"].apply(lambda x: x.get("pnl_usdt")), errors="coerce").fillna(0.0)
        realized_pnl_usdt = float(sells["pnl_usdt"].sum())
        sell_win_rate = float((sells["pnl_usdt"] > 0).mean()) if not sells.empty else 0.0
    else:
        sells = pd.DataFrame()
        realized_pnl_usdt = 0.0
        sell_win_rate = 0.0

    run_days_observed = 0.0
    if not cycles.empty and cycles["ts"].notna().any():
        ts_min = cycles["ts"].min()
        ts_max = cycles["ts"].max()
        run_days_observed = max(0.0, (ts_max - ts_min).total_seconds() / 86400.0)

    daily_reports_dir = Path(daily_dir)
    violations_count = 0
    reports_count = 0
    if daily_reports_dir.exists():
        for path in sorted(daily_reports_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            reports_count += 1
            violations_count += int(payload.get("violations_count", 0))

    backtest_path = Path("artifacts/metrics/backtest_summary.csv")
    backtest_available = backtest_path.exists()

    mon_cfg = config.get("monitoring", {})
    alert_cfg = mon_cfg.get("alerting", {})
    alerting_ready = bool(mon_cfg.get("alerts_enabled", True)) and bool(
        str(alert_cfg.get("webhook_url", "")).strip()
        or (str(alert_cfg.get("telegram_bot_token", "")).strip() and str(alert_cfg.get("telegram_chat_id", "")).strip())
        or str((alert_cfg.get("email", {}) or {}).get("smtp_host", "")).strip()
    )

    docs_ready = Path("docs/runbook_incident.md").exists() and Path("docs/deployment_docker.md").exists()

    checklist = {
        "paper_run_duration_met": run_days_observed >= float(days),
        "zero_risk_violations": violations_count == 0,
        "backtest_report_available": backtest_available,
        "dashboard_and_alerting_operational": alerting_ready,
        "docker_and_incident_docs_present": docs_ready,
    }

    return {
        "generated_at": now.isoformat(),
        "window_start": start.isoformat(),
        "window_end": now.isoformat(),
        "days_target": int(days),
        "run_days_observed": run_days_observed,
        "reports_count": reports_count,
        "violations_count": violations_count,
        "realized_pnl_usdt": realized_pnl_usdt,
        "sell_win_rate": sell_win_rate,
        "checklist": checklist,
    }


def render_dod_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# DoD v1 Operational Summary",
        "",
        f"- Generated at (UTC): {summary.get('generated_at')}",
        f"- Window: {summary.get('window_start')} -> {summary.get('window_end')}",
        f"- Target days: {int(summary.get('days_target', 30))}",
        f"- Observed run days: {float(summary.get('run_days_observed', 0.0)):.2f}",
        f"- Daily reports: {int(summary.get('reports_count', 0))}",
        f"- Violations: {int(summary.get('violations_count', 0))}",
        f"- Realized PnL (USDT): {float(summary.get('realized_pnl_usdt', 0.0)):.4f}",
        f"- SELL win rate: {float(summary.get('sell_win_rate', 0.0)):.4f}",
        "",
        "## DoD Checklist",
    ]
    checklist = summary.get("checklist", {})
    for key, value in checklist.items():
        status = "PASS" if bool(value) else "FAIL"
        lines.append(f"- `{key}`: **{status}**")
    return "\n".join(lines) + "\n"
