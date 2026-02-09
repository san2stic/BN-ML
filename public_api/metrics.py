from __future__ import annotations

from pathlib import Path
from typing import Any

from prometheus_client.core import GaugeMetricFamily

from public_api.data_access import load_account_state, load_runtime_summary, read_prediction_snapshot


class BNMLRuntimeCollector:
    def __init__(self, db_path: Path, scan_path: Path) -> None:
        self.db_path = db_path
        self.scan_path = scan_path

    @staticmethod
    def _count_by_signal(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for row in rows:
            signal = str(row.get("signal", "HOLD")).upper()
            if signal not in counts:
                signal = "HOLD"
            counts[signal] += 1
        return counts

    def collect(self):  # pragma: no cover - validated by integration scrape
        scan = read_prediction_snapshot(self.scan_path, limit=None)
        rows = scan.get("rows", [])
        counts = self._count_by_signal(rows if isinstance(rows, list) else [])

        scan_rows = GaugeMetricFamily("bnml_scan_rows", "Total rows in latest scan snapshot")
        scan_rows.add_metric([], float(scan.get("total_rows", 0)))
        yield scan_rows

        scan_age = GaugeMetricFamily("bnml_scan_age_seconds", "Age of latest scan snapshot in seconds")
        scan_age.add_metric([], float(scan.get("age_sec") or 0.0))
        yield scan_age

        signal_count = GaugeMetricFamily("bnml_scan_signal_count", "Signal counts in latest scan", labels=["signal"])
        for signal, count in counts.items():
            signal_count.add_metric([signal], float(count))
        yield signal_count

        top_conf = max((float(row.get("confidence", 0.0)) for row in rows), default=0.0)
        confidence = GaugeMetricFamily("bnml_scan_top_confidence", "Top confidence score in latest scan")
        confidence.add_metric([], top_conf)
        yield confidence

        account = load_account_state(self.db_path)
        account_total = GaugeMetricFamily("bnml_account_total_capital", "Account total capital")
        account_total.add_metric([], float(account.get("total_capital", 0.0)))
        yield account_total

        account_active = GaugeMetricFamily("bnml_account_active_capital", "Account active capital")
        account_active.add_metric([], float(account.get("active_capital", 0.0)))
        yield account_active

        daily_pnl = GaugeMetricFamily("bnml_account_daily_pnl_pct", "Account daily pnl percent")
        daily_pnl.add_metric([], float(account.get("daily_pnl_pct", 0.0)))
        yield daily_pnl

        weekly_pnl = GaugeMetricFamily("bnml_account_weekly_pnl_pct", "Account weekly pnl percent")
        weekly_pnl.add_metric([], float(account.get("weekly_pnl_pct", 0.0)))
        yield weekly_pnl

        losses = GaugeMetricFamily("bnml_account_consecutive_losses", "Account consecutive losses")
        losses.add_metric([], float(account.get("consecutive_losses", 0)))
        yield losses

        drift = GaugeMetricFamily("bnml_market_drift_detected", "1 when market drift is detected, otherwise 0")
        drift.add_metric([], 1.0 if account.get("market_drift_detected") else 0.0)
        yield drift

        summary = load_runtime_summary(self.db_path)
        open_positions = GaugeMetricFamily("bnml_open_positions", "Count of open positions")
        open_positions.add_metric([], float(summary.get("open_positions", 0)))
        yield open_positions

        total_trades = GaugeMetricFamily("bnml_total_trades", "Total number of recorded trades")
        total_trades.add_metric([], float(summary.get("total_trades", 0)))
        yield total_trades

        total_cycles = GaugeMetricFamily("bnml_total_cycles", "Total number of recorded runtime cycles")
        total_cycles.add_metric([], float(summary.get("total_cycles", 0)))
        yield total_cycles

