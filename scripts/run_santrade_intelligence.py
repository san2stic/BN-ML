from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bn_ml.config import load_config
from bn_ml.env import load_env_file
from bn_ml.state_store import StateStore
from bn_ml.symbols import normalize_symbols
from data_manager.fetch_data import BinanceDataManager
from ml_engine.predictor import MLEnsemblePredictor
from ml_engine.santrade_intelligence import SanTradeIntelligence
from monitoring.logger import setup_logger
from scanner.opportunity_scanner import MultiPairScanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone SanTradeIntelligence market engine")
    parser.add_argument("--config", default="configs/bot.yaml", help="Path to YAML config")
    parser.add_argument("--paper", action="store_true", help="Force paper data mode")
    parser.add_argument("--live", action="store_true", help="Force live data mode")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval-seconds", type=float, default=None, help="Loop interval seconds override")
    return parser.parse_args()


def resolve_interval_seconds(config: dict[str, Any], override_seconds: float | None) -> float:
    if override_seconds is not None:
        try:
            return max(15.0, float(override_seconds))
        except (TypeError, ValueError):
            return 15.0

    model_cfg = config.get("model", {})
    sti_cfg = model_cfg.get("santrade_intelligence", {}) if isinstance(model_cfg.get("santrade_intelligence", {}), dict) else {}
    scanner_cfg = config.get("scanner", {})
    try:
        runtime_interval = float(sti_cfg.get("runtime_interval_sec", scanner_cfg.get("scan_interval_sec", 300)))
    except (TypeError, ValueError):
        runtime_interval = 300.0
    return max(15.0, runtime_interval)


def _sleep_with_interrupt(total_seconds: float) -> None:
    remaining = max(0.0, float(total_seconds))
    while remaining > 0:
        chunk = min(1.0, remaining)
        time.sleep(chunk)
        remaining -= chunk


class SanTradeIntelligenceRuntime:
    def __init__(self, config: dict[str, Any], paper: bool) -> None:
        self.config = config
        self.paper = paper
        self.logger = setup_logger(config)
        self.data_manager = BinanceDataManager(config=config, paper=paper)
        self.predictor = MLEnsemblePredictor(model_dir="models", missing_model_callback=None)
        self.scanner = MultiPairScanner(config=config, data_manager=self.data_manager, predictor=self.predictor)
        self.market_intelligence = SanTradeIntelligence(config=config, data_manager=self.data_manager, logger=self.logger)
        db_path = str(config.get("storage", {}).get("sqlite_path", "artifacts/state/bn_ml.db"))
        self.store = StateStore(db_path=db_path)
        self._universe_cache_pairs: list[str] = []
        self._universe_cache_ts: float = 0.0

    def _quote_asset(self) -> str:
        configured = str(self.config.get("base_quote", "")).strip().upper()
        if configured:
            return configured
        universe_cfg = self.config.get("universe", {})
        user_selected = normalize_symbols(universe_cfg.get("user_selected_pairs", []))
        pairs = user_selected or normalize_symbols(universe_cfg.get("pairs", []))
        if pairs:
            try:
                return str(pairs[0]).split("/")[-1].upper()
            except Exception:
                pass
        return "USDT"

    def _resolve_pairs_for_scan(self, force_refresh: bool = False) -> list[str]:
        universe_cfg = self.config.get("universe", {})
        configured_pairs = normalize_symbols(universe_cfg.get("pairs", []))
        user_selected_pairs = normalize_symbols(universe_cfg.get("user_selected_pairs", []))
        user_selected_only = bool(universe_cfg.get("user_selected_only", False))

        if user_selected_only:
            selected = user_selected_pairs or configured_pairs
            if not selected:
                self.logger.warning(
                    "universe.user_selected_only is enabled but no pairs are configured "
                    "(expected universe.user_selected_pairs or universe.pairs)."
                )
            return selected

        dynamic_enabled = bool(universe_cfg.get("dynamic_base_quote_pairs", False))
        if not dynamic_enabled:
            return configured_pairs

        refresh_sec = int(universe_cfg.get("dynamic_refresh_sec", 900))
        now_ts = time.time()
        if not force_refresh and self._universe_cache_pairs and (now_ts - self._universe_cache_ts) < max(30, refresh_sec):
            return self._universe_cache_pairs

        quote = self._quote_asset()
        min_volume = float(universe_cfg.get("min_24h_volume_usdt", 1_000_000))
        max_pairs = int(universe_cfg.get("max_pairs_scanned", 150))

        discovered = self.data_manager.discover_pairs_by_quote(
            quote=quote,
            min_quote_volume_usdt=min_volume,
            max_pairs=max_pairs,
        )
        discovered = normalize_symbols(discovered)
        if discovered:
            self._universe_cache_pairs = discovered
            self._universe_cache_ts = now_ts
            return discovered
        return configured_pairs

    def _export_market_intelligence(self, snapshot: dict[str, Any]) -> None:
        metrics_dir = Path(str(self.config.get("monitoring", {}).get("metrics_dir", "artifacts/metrics")))
        metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(snapshot)
        pd.DataFrame([payload]).to_csv(metrics_dir / "latest_market_intelligence.csv", index=False)
        (metrics_dir / "latest_market_intelligence.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _merge_snapshot_into_account_state(account_state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        updated = dict(account_state)
        updated["market_intelligence_signal"] = payload.get("signal", "HOLD")
        updated["market_intelligence_confidence"] = float(payload.get("confidence", 0.0))
        updated["market_intelligence_score"] = float(payload.get("market_score", 0.0))
        updated["market_intelligence_raw_score"] = float(payload.get("raw_market_score", 0.0))
        updated["market_intelligence_smoothed_score"] = float(payload.get("smoothed_market_score", 0.0))
        updated["market_intelligence_regime"] = str(payload.get("market_regime", "unknown"))
        updated["market_intelligence_predicted_move_pct"] = float(payload.get("predicted_move_pct", 0.0))
        updated["market_intelligence_symbols"] = int(payload.get("symbols_scanned", 0))
        updated["market_intelligence_model_samples"] = int(payload.get("model_samples", 0))
        updated["market_intelligence_profile"] = str(payload.get("profile", "neutral"))
        updated["market_intelligence_coverage_ratio"] = float(payload.get("data_coverage_ratio", 0.0))
        updated["market_intelligence_directional_streak"] = int(payload.get("directional_streak", 0))
        updated["market_intelligence_updated_at"] = payload.get("generated_at")
        return updated

    def run_cycle(self) -> dict[str, Any]:
        pairs = self._resolve_pairs_for_scan(force_refresh=False)
        opportunities, opportunities_all = self.scanner.scan_details(pairs)

        snapshot = self.market_intelligence.update(
            pairs=pairs,
            opportunities=opportunities_all,
            quote_asset=self._quote_asset(),
        )
        payload = snapshot.to_dict()
        self.store.set_state("santrade_intelligence", payload)
        current_account_state = self.store.load_account_state(default={})
        self.store.save_account_state(self._merge_snapshot_into_account_state(current_account_state, payload))
        self._export_market_intelligence(payload)

        self.logger.info(
            "SanTradeIntelligence cycle done: signal=%s conf=%.1f regime=%s scanned=%s selected=%s model_samples=%s",
            payload.get("signal", "HOLD"),
            float(payload.get("confidence", 0.0)),
            payload.get("market_regime", "unknown"),
            len(opportunities_all),
            len(opportunities),
            int(payload.get("model_samples", 0)),
        )
        return payload


def run_loop(
    *,
    runtime: SanTradeIntelligenceRuntime,
    interval_seconds: float,
    max_cycles: int = 0,
) -> int:
    cycle = 0
    while True:
        cycle += 1
        try:
            started = datetime.now(timezone.utc).isoformat()
            runtime.logger.info("SanTradeIntelligence cycle %s started at %s", cycle, started)
            runtime.run_cycle()
        except KeyboardInterrupt:
            runtime.logger.info("SanTradeIntelligence interrupted by user.")
            return 130
        except Exception as exc:
            runtime.logger.exception("SanTradeIntelligence cycle %s failed: %s", cycle, exc)

        if max_cycles > 0 and cycle >= max_cycles:
            return 0

        _sleep_with_interrupt(interval_seconds)


def main() -> None:
    load_env_file()
    args = parse_args()
    config = load_config(args.config)
    if args.paper and args.live:
        raise ValueError("Cannot use --paper and --live together.")
    if args.live:
        paper = False
    elif args.paper:
        paper = True
    else:
        paper = config.get("environment", "paper") == "paper"

    runtime = SanTradeIntelligenceRuntime(config=config, paper=paper)

    if args.once:
        runtime.run_cycle()
        return

    interval_seconds = resolve_interval_seconds(config=config, override_seconds=args.interval_seconds)
    exit_code = run_loop(
        runtime=runtime,
        interval_seconds=interval_seconds,
        max_cycles=0,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
