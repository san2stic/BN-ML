from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bn_ml.config import load_config
from bn_ml.env import load_env_file
from bn_ml.exchange import call_with_retry
from bn_ml.backup import RuntimeBackupManager
from bn_ml.state_store import StateStore
from bn_ml.symbols import normalize_symbols
from data_manager.data_cleaner import DataCleaner
from data_manager.features_engineer import FeatureEngineer
from data_manager.fetch_data import BinanceDataManager
from ml_engine.adaptive_trainer import AdaptiveRetrainer, BackgroundRetrainWorker
from ml_engine.drift_monitor import MarketDriftMonitor
from ml_engine.predictor import MLEnsemblePredictor
from ml_engine.santrade_intelligence import SanTradeIntelligence
from monitoring.alerter import Alerter
from monitoring.logger import resolve_writable_logs_dir, setup_logger
from monitoring.realtime_prices import BinanceRealtimePriceMonitor
from scanner.opportunity_scanner import MultiPairScanner
from scripts.run_trainer import train_once
from trader.exit_manager import ExitManager
from trader.order_manager import OrderConstraintError, OrderManager
from trader.position_manager import PositionManager
from trader.risk_manager import RiskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BN-ML trading bot")
    parser.add_argument("--config", default="configs/bot.yaml", help="Path to YAML config")
    parser.add_argument("--paper", action="store_true", help="Force paper mode")
    parser.add_argument("--live", action="store_true", help="Force live mode")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit")
    parser.add_argument("--disable-retrain", action="store_true", help="Disable adaptive retraining")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not auto-launch Streamlit dashboard")
    parser.add_argument("--dashboard-port", type=int, default=None, help="Dashboard port override")
    parser.add_argument("--dashboard-address", default=None, help="Dashboard bind address override")
    return parser.parse_args()


def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def maybe_start_dashboard(config: dict, args: argparse.Namespace, logger) -> tuple[subprocess.Popen | None, Any | None]:
    dashboard_cfg = config.get("monitoring", {}).get("dashboard", {})
    auto_launch = bool(dashboard_cfg.get("auto_launch_with_bot", True))

    if args.no_dashboard or not auto_launch:
        return None, None

    address = str(args.dashboard_address or dashboard_cfg.get("address", "127.0.0.1"))
    port = int(args.dashboard_port or dashboard_cfg.get("port", 8501))

    if _is_port_open(address, port):
        logger.info("Dashboard already reachable on http://%s:%s. Skipping launch.", address, port)
        return None, None

    script = str(dashboard_cfg.get("script", "monitoring/dashboard.py"))
    workspace = Path(__file__).resolve().parents[1]
    logs_dir, _ = resolve_writable_logs_dir(config)
    dashboard_log_path = logs_dir / "dashboard.log"
    try:
        log_fh = dashboard_log_path.open("a", encoding="utf-8")
        stdout_target = log_fh
        stderr_target = log_fh
    except OSError as exc:
        logger.warning("Dashboard log file unavailable at %s (%s). Redirecting to /dev/null.", dashboard_log_path, exc)
        log_fh = None
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script,
        "--server.port",
        str(port),
        "--server.address",
        address,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    streamlit_env = os.environ.copy()
    streamlit_env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    streamlit_env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        env=streamlit_env,
        stdin=subprocess.DEVNULL,
        stdout=stdout_target,
        stderr=stderr_target,
    )
    logger.warning("Dashboard auto-launched on http://%s:%s (pid=%s)", address, port, proc.pid)
    return proc, log_fh


def stop_dashboard(proc: subprocess.Popen | None, log_fh: Any | None, logger) -> None:
    try:
        if proc is not None and proc.poll() is None:
            logger.info("Stopping dashboard process pid=%s", proc.pid)
            proc.terminate()
            proc.wait(timeout=6)
    except Exception:
        if proc is not None and proc.poll() is None:
            proc.kill()
    finally:
        try:
            if log_fh is not None:
                log_fh.close()
        except Exception:
            pass


class TradingRuntime:
    def __init__(self, config: dict, paper: bool, disable_retrain: bool = False) -> None:
        self.config = config
        self.paper = paper
        self.logger = setup_logger(config)
        self.alerter = Alerter(config=config, enabled=bool(config.get("monitoring", {}).get("alerts_enabled", True)))
        self.disable_retrain = disable_retrain
        self.backup_manager = RuntimeBackupManager(config=config, logger=self.logger)

        db_path = str(config.get("storage", {}).get("sqlite_path", "artifacts/state/bn_ml.db"))
        self.store = StateStore(db_path=db_path)
        self._init_training_status()

        self.account_state = self.store.load_account_state(default=self._default_account_state())
        self._model_components_lock = threading.Lock()
        self.retrainer = AdaptiveRetrainer(config=config, store=self.store)
        self.retrain_worker: BackgroundRetrainWorker | None = None

        self.data_manager = BinanceDataManager(config=config, paper=paper)
        self.predictor = MLEnsemblePredictor(
            model_dir="models",
            missing_model_callback=self._on_missing_model if self._missing_model_auto_train_enabled() else None,
        )
        self.scanner = MultiPairScanner(config=config, data_manager=self.data_manager, predictor=self.predictor)
        self.cleaner = DataCleaner()
        self.features = FeatureEngineer()
        mtf_cfg = config.get("model", {}).get("multi_timeframe", {})
        self.base_timeframe = str(mtf_cfg.get("base_timeframe", "15m")).strip().lower() or "15m"
        self.atr_ohlcv_limit = int(config.get("risk", {}).get("atr_ohlcv_limit", 150))

        self.risk_manager = RiskManager(config)
        self.drift_monitor = MarketDriftMonitor(config)
        self.market_intelligence = SanTradeIntelligence(config=config, data_manager=self.data_manager, logger=self.logger)
        self.exit_manager = ExitManager(config)
        self.order_manager = OrderManager(config=config, paper=paper)
        self.position_manager = PositionManager(store=self.store)
        rt_cfg = config.get("monitoring", {}).get("realtime_prices", {})
        self.realtime_prices = BinanceRealtimePriceMonitor(
            enabled=bool(rt_cfg.get("enabled", True)),
            max_symbols=int(rt_cfg.get("max_symbols", 30)),
            reconnect_delay_sec=float(rt_cfg.get("reconnect_delay_sec", 3.0)),
            logger=self.logger,
        )

        self._universe_cache_pairs: list[str] = []
        self._universe_cache_ts: float = 0.0
        self._preflight_runtime()
        self._sync_realtime_price_stream(force=True)
        self.store.save_account_state(self.account_state)
        self._start_retrain_worker_if_enabled()

    def _init_training_status(self) -> None:
        existing = self.store.get_state("training_status", None)
        if isinstance(existing, dict) and existing:
            return
        self.store.set_state(
            "training_status",
            {
                "status": "idle",
                "phase": "waiting",
                "trigger": "startup",
                "started_at": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_symbol": None,
                "current_index": 0,
                "symbols_requested": 0,
                "symbols_queued": 0,
                "symbols_completed": 0,
                "symbols_trained": 0,
                "symbols_errors": 0,
                "symbols_skipped_up_to_date": 0,
                "progress_pct": 0.0,
            },
        )

    def _make_training_progress_callback(
        self,
        *,
        trigger: str,
        extra: dict[str, Any] | None = None,
    ):
        extra_payload = dict(extra or {})

        def _callback(update: dict[str, Any]) -> None:
            payload = dict(update)
            payload["trigger"] = trigger
            if extra_payload:
                payload.update(extra_payload)
            self.store.set_state("training_status", payload)

        return _callback

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

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _resolve_configured_capital_total(raw_value: Any, *, paper: bool, fallback: float = 10_000.0) -> float:
        if isinstance(raw_value, str):
            mode = raw_value.strip().lower()
            if mode in {"auto", "binance", "exchange"}:
                # In live mode this is refreshed from the authenticated Binance wallet.
                return fallback if paper else 0.0

        try:
            parsed = float(raw_value)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
        return fallback

    @staticmethod
    def _is_stable(asset: str) -> bool:
        return asset.upper() in {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USDP", "DAI"}

    def _ticker_last(self, exchange: Any, symbol: str) -> float:
        ticker = call_with_retry(lambda: exchange.fetch_ticker(symbol), retries=2, backoff_sec=0.3)
        return float(ticker.get("last") or ticker.get("close") or 0.0)

    def _asset_price_in_quote(self, exchange: Any, asset: str, quote: str) -> float:
        asset_u = asset.upper()
        quote_u = quote.upper()
        if asset_u == quote_u:
            return 1.0

        if self._is_stable(asset_u) and self._is_stable(quote_u):
            return 1.0

        direct = f"{asset_u}/{quote_u}"
        inverse = f"{quote_u}/{asset_u}"

        if direct in exchange.markets:
            px = self._ticker_last(exchange, direct)
            if px > 0:
                return px

        if inverse in exchange.markets:
            px_inv = self._ticker_last(exchange, inverse)
            if px_inv > 0:
                return 1.0 / px_inv

        # Cross via USDT for non-stable assets when direct quote route is unavailable.
        bridge = "USDT"
        if quote_u != bridge:
            bridge_direct = f"{asset_u}/{bridge}"
            bridge_inverse = f"{bridge}/{asset_u}"
            bridge_to_quote = 1.0 if self._is_stable(quote_u) else 0.0

            if not self._is_stable(quote_u):
                bridge_quote = f"{bridge}/{quote_u}"
                quote_bridge = f"{quote_u}/{bridge}"
                if bridge_quote in exchange.markets:
                    bridge_to_quote = self._ticker_last(exchange, bridge_quote)
                elif quote_bridge in exchange.markets:
                    px = self._ticker_last(exchange, quote_bridge)
                    bridge_to_quote = (1.0 / px) if px > 0 else 0.0

            if bridge_to_quote > 0:
                if bridge_direct in exchange.markets:
                    px = self._ticker_last(exchange, bridge_direct)
                    if px > 0:
                        return px * bridge_to_quote
                if bridge_inverse in exchange.markets:
                    px_inv = self._ticker_last(exchange, bridge_inverse)
                    if px_inv > 0:
                        return (1.0 / px_inv) * bridge_to_quote

        return 0.0

    def _sync_live_capital(self) -> None:
        if self.paper:
            return
        try:
            exchange = self.order_manager._build_exchange()
            balance = call_with_retry(lambda: exchange.fetch_balance(), retries=3, backoff_sec=0.5)

            quote = self._quote_asset()
            totals = balance.get("total") or {}
            free_map = balance.get("free") or {}
            used_map = balance.get("used") or {}

            total_quote = self._to_float(totals.get(quote))
            converted_assets = 0

            for asset, raw_qty in totals.items():
                asset_u = str(asset).upper()
                qty = self._to_float(raw_qty)
                if qty <= 0 or asset_u == quote:
                    continue
                px = self._asset_price_in_quote(exchange=exchange, asset=asset_u, quote=quote)
                if px <= 0:
                    continue
                total_quote += qty * px
                converted_assets += 1

            active_ratio = float(self.config.get("risk", {}).get("capital_active_ratio", 0.60))
            self.account_state["total_capital"] = float(total_quote)
            self.account_state["active_capital"] = float(total_quote) * active_ratio
            self.account_state["exchange_quote_asset"] = quote
            self.account_state["exchange_quote_free"] = self._to_float(free_map.get(quote))
            self.account_state["exchange_quote_used"] = self._to_float(used_map.get(quote))
            self.account_state["exchange_assets_valued"] = int(converted_assets)
            self.account_state["exchange_synced_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            self.logger.warning("Live capital sync failed: %s", exc)

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
            self.logger.info("Dynamic universe resolved: %s pairs for */%s", len(discovered), quote)
            return discovered
        return configured_pairs

    def _resolve_pairs_for_training(self) -> list[str]:
        universe_cfg = self.config.get("universe", {})
        configured_pairs = normalize_symbols(universe_cfg.get("pairs", []))
        user_selected_pairs = normalize_symbols(universe_cfg.get("user_selected_pairs", []))
        user_selected_only = bool(universe_cfg.get("user_selected_only", False))

        if user_selected_only:
            return user_selected_pairs or configured_pairs

        dynamic_enabled = bool(universe_cfg.get("dynamic_base_quote_pairs", False))
        train_dynamic = bool(universe_cfg.get("train_dynamic_pairs", dynamic_enabled))

        if not train_dynamic:
            return configured_pairs

        pairs = self._resolve_pairs_for_scan(force_refresh=False)
        max_pairs = int(universe_cfg.get("train_max_pairs", universe_cfg.get("max_pairs_scanned", 150)))
        if max_pairs > 0:
            return pairs[:max_pairs]
        return pairs

    def _retrain_enabled(self) -> bool:
        return (not self.disable_retrain) and bool(self.config.get("model", {}).get("auto_retrain_enabled", True))

    def _missing_model_auto_train_enabled(self) -> bool:
        return self._retrain_enabled() and bool(self.config.get("model", {}).get("auto_train_missing_models", True))

    def _train_all_pairs_once(self) -> dict[str, Any]:
        return train_once(
            config=self.config,
            paper=self.paper,
            symbols=self._resolve_pairs_for_training(),
            train_missing_only=bool(self.config.get("universe", {}).get("train_missing_only", False)),
            max_model_age_hours=float(self.config.get("universe", {}).get("model_max_age_hours", 24)),
            progress_callback=self._make_training_progress_callback(trigger="periodic"),
        )

    def _train_missing_symbols_once(self, symbols: list[str]) -> dict[str, Any]:
        train_config = self.config
        disable_hpo_for_missing = bool(self.config.get("model", {}).get("auto_train_missing_disable_hpo", True))
        if disable_hpo_for_missing:
            train_config = copy.deepcopy(self.config)
            model_cfg = train_config.setdefault("model", {})
            hpo_cfg = model_cfg.setdefault("hpo", {})
            hpo_cfg["enabled"] = False

        self.logger.info(
            "Background missing-model training batch started (symbols=%s, hpo_enabled=%s)",
            len(symbols),
            not disable_hpo_for_missing,
        )
        return train_once(
            config=train_config,
            paper=self.paper,
            symbols=symbols,
            train_missing_only=True,
            max_model_age_hours=None,
            progress_callback=self._make_training_progress_callback(
                trigger="missing_models",
                extra={"hpo_enabled": bool(not disable_hpo_for_missing)},
            ),
        )

    def _on_models_updated(self, reason: str, result: dict[str, Any] | None) -> None:
        with self._model_components_lock:
            self.predictor = MLEnsemblePredictor(
                model_dir="models",
                missing_model_callback=self._on_missing_model if self._missing_model_auto_train_enabled() else None,
            )
            self.scanner = MultiPairScanner(
                config=self.config,
                data_manager=self.data_manager,
                predictor=self.predictor,
            )

        trained = 0
        if isinstance(result, dict):
            aggregate = result.get("aggregate")
            if isinstance(aggregate, dict):
                try:
                    trained = int(aggregate.get("symbols_trained", 0))
                except (TypeError, ValueError):
                    trained = 0
        training_status = self.store.get_state("training_status", {})
        if isinstance(training_status, dict):
            training_status["models_reloaded_at"] = datetime.now(timezone.utc).isoformat()
            training_status["reload_reason"] = reason
            self.store.set_state("training_status", training_status)
        self.logger.info("Model components reloaded after background training (%s, symbols_trained=%s)", reason, trained)

    def _on_missing_model(self, symbol: str) -> None:
        if not self._missing_model_auto_train_enabled():
            return
        worker = self.retrain_worker
        if worker is None:
            return
        worker.queue_missing_symbol(symbol)

    def _start_retrain_worker_if_enabled(self) -> None:
        if not self._retrain_enabled():
            return

        self.retrain_worker = BackgroundRetrainWorker(
            config=self.config,
            retrainer=self.retrainer,
            get_account_state=lambda: dict(self.account_state),
            periodic_train_func=self._train_all_pairs_once,
            missing_train_func=self._train_missing_symbols_once,
            on_models_updated=self._on_models_updated,
            logger=self.logger,
        )
        self.retrain_worker.start()

        if self._missing_model_auto_train_enabled():
            startup_symbols = self._resolve_pairs_for_training()
            if startup_symbols:
                self.logger.info("Queueing startup missing-model check for %s symbols", len(startup_symbols))
                self.retrain_worker.queue_missing_symbols(startup_symbols)

    def shutdown(self) -> None:
        if self.retrain_worker is not None:
            self.retrain_worker.stop(timeout_sec=30.0)
            self.retrain_worker = None
        try:
            self.market_intelligence.flush_state()
        except Exception as exc:
            self.logger.warning("SanTradeIntelligence shutdown flush failed: %s", exc)
        self.realtime_prices.stop()

    def _sync_realtime_price_stream(self, force: bool = False) -> None:
        rt_cfg = self.config.get("monitoring", {}).get("realtime_prices", {})
        if not bool(rt_cfg.get("enabled", True)):
            return

        max_symbols = max(1, int(rt_cfg.get("max_symbols", 30)))
        include_scan_pairs = bool(rt_cfg.get("include_scan_pairs", True))
        symbols: list[str] = []

        for position in self.position_manager.get_open_positions():
            if position.symbol not in symbols:
                symbols.append(position.symbol)

        if include_scan_pairs and len(symbols) < max_symbols:
            pairs = self._resolve_pairs_for_scan(force_refresh=False)
            for pair in pairs:
                if pair not in symbols:
                    symbols.append(pair)
                if len(symbols) >= max_symbols:
                    break

        if not symbols:
            return
        self.realtime_prices.update_symbols(symbols[:max_symbols])

    def _preflight_runtime(self) -> None:
        if self.paper:
            self.logger.info("Runtime mode: PAPER")
            return

        exchange_cfg = self.config.get("exchange", {})
        if bool(exchange_cfg.get("testnet", True)):
            raise ValueError("Live execution requested but exchange.testnet is true. Set testnet: false for real Binance.")

        if not os.getenv("BINANCE_API_KEY", "").strip() or not os.getenv("BINANCE_API_SECRET", "").strip():
            raise ValueError("Live execution requires BINANCE_API_KEY and BINANCE_API_SECRET environment variables.")

        pairs = self._resolve_pairs_for_scan(force_refresh=True)
        if not pairs:
            raise ValueError("No tradable pairs found for configured universe.")

        sample_symbol = pairs[0]
        ticker = self.data_manager.fetch_ticker(sample_symbol)
        if float(ticker.get("last") or 0.0) <= 0:
            raise ValueError(f"Unable to fetch live market data for {sample_symbol}.")

        exchange = self.order_manager._build_exchange()
        balance = call_with_retry(lambda: exchange.fetch_balance(), retries=3, backoff_sec=0.5)
        quote = sample_symbol.split("/")[-1]
        total_quote = float((balance.get("total") or {}).get(quote, 0.0))

        self.logger.warning(
            "Runtime mode: LIVE (Binance Spot). Sample %s last=%.8f, total %s balance=%.8f",
            sample_symbol,
            float(ticker.get("last")),
            quote,
            total_quote,
        )
        self._sync_live_capital()

    def _default_account_state(self) -> dict[str, Any]:
        risk = self.config.get("risk", {})
        fallback_total = self._to_float(risk.get("capital_total_fallback", 10_000.0))
        if fallback_total <= 0:
            fallback_total = 10_000.0
        total = self._resolve_configured_capital_total(
            risk.get("capital_total", fallback_total),
            paper=self.paper,
            fallback=fallback_total,
        )
        active = total * float(risk.get("capital_active_ratio", 0.60))

        now = datetime.now(timezone.utc)
        iso_year, iso_week, _ = now.isocalendar()
        return {
            "total_capital": total,
            "active_capital": active,
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "daily_realized_usdt": 0.0,
            "weekly_realized_usdt": 0.0,
            "consecutive_losses": 0,
            "market_volatility_ratio": 1.0,
            "market_volatility_current_atr_ratio": 0.0,
            "market_volatility_baseline_atr_ratio": 0.0,
            "market_volatility_symbol": "",
            "market_volatility_updated_at": None,
            "market_drift_detected": False,
            "market_drift_ks_stat": 0.0,
            "market_drift_p_value": 1.0,
            "market_drift_vol_ratio": 1.0,
            "market_regime": "unknown",
            "market_drift_symbol": "",
            "market_drift_updated_at": None,
            "market_intelligence_signal": "HOLD",
            "market_intelligence_confidence": 0.0,
            "market_intelligence_score": 0.0,
            "market_intelligence_raw_score": 0.0,
            "market_intelligence_smoothed_score": 0.0,
            "market_intelligence_regime": "unknown",
            "market_intelligence_predicted_move_pct": 0.0,
            "market_intelligence_symbols": 0,
            "market_intelligence_model_samples": 0,
            "market_intelligence_profile": "defensive",
            "market_intelligence_coverage_ratio": 0.0,
            "market_intelligence_directional_streak": 0,
            "market_intelligence_updated_at": None,
            "win_rate": 0.56,
            "avg_win": 1.8,
            "avg_loss": 1.0,
            "baseline_win_rate": 0.56,
            "current_win_rate_24h": 0.56,
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "sum_win_pct": 0.0,
            "sum_loss_pct": 0.0,
            "last_day": now.date().isoformat(),
            "last_week": f"{iso_year}-W{iso_week:02d}",
        }

    def _roll_period_counters(self, now: datetime) -> None:
        day_key = now.date().isoformat()
        iso_year, iso_week, _ = now.isocalendar()
        week_key = f"{iso_year}-W{iso_week:02d}"

        if self.account_state.get("last_day") != day_key:
            self.account_state["daily_realized_usdt"] = 0.0
            self.account_state["daily_pnl_pct"] = 0.0
            self.account_state["last_day"] = day_key

        if self.account_state.get("last_week") != week_key:
            self.account_state["weekly_realized_usdt"] = 0.0
            self.account_state["weekly_pnl_pct"] = 0.0
            self.account_state["last_week"] = week_key

    def _export_scan(self, opportunities, opportunities_all) -> None:
        metrics_dir = Path(str(self.config.get("monitoring", {}).get("metrics_dir", "artifacts/metrics")))
        metrics_dir.mkdir(parents=True, exist_ok=True)

        columns = [
            "symbol",
            "signal",
            "confidence",
            "ml_score",
            "technical_score",
            "momentum_score",
            "global_score",
            "spread_pct",
            "depth_usdt",
            "correlation_btc",
        ]
        def _rows(data):
            rows = []
            for opp in data:
                rows.append(
                    {
                        "symbol": opp.symbol,
                        "signal": opp.signal.action,
                        "confidence": opp.signal.confidence,
                        "ml_score": opp.ml_score,
                        "technical_score": opp.technical_score,
                        "momentum_score": opp.momentum_score,
                        "global_score": opp.global_score,
                        "spread_pct": opp.spread_pct,
                        "depth_usdt": opp.orderbook_depth_usdt,
                        "correlation_btc": opp.correlation_with_btc,
                    }
                )
            return rows

        rows_all = _rows(opportunities_all)
        rows_selected = _rows(opportunities)

        pd.DataFrame(rows_all, columns=columns).to_csv(metrics_dir / "latest_scan.csv", index=False)
        pd.DataFrame(rows_selected, columns=columns).to_csv(metrics_dir / "latest_opportunities.csv", index=False)

    def _export_market_intelligence(self, snapshot: dict[str, Any]) -> None:
        metrics_dir = Path(str(self.config.get("monitoring", {}).get("metrics_dir", "artifacts/metrics")))
        metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(snapshot)
        pd.DataFrame([payload]).to_csv(metrics_dir / "latest_market_intelligence.csv", index=False)
        (metrics_dir / "latest_market_intelligence.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _refresh_recent_performance(self) -> None:
        stats = self.store.recent_sell_stats(hours=24)
        if stats.get("count", 0.0) > 0:
            self.account_state["current_win_rate_24h"] = float(stats.get("win_rate", 0.0))

    def _update_market_volatility_ratio(self) -> None:
        risk_cfg = self.config.get("risk", {})
        baseline_window = max(20, int(risk_cfg.get("volatility_baseline_window", 96)))
        limit = max(self.atr_ohlcv_limit, baseline_window + 5)

        configured_symbol = str(risk_cfg.get("volatility_benchmark_symbol", "")).strip().upper()
        quote = self._quote_asset()
        symbols_to_try: list[str] = []
        for symbol in [configured_symbol, f"BTC/{quote}", "BTC/USDT"]:
            if symbol and symbol not in symbols_to_try:
                symbols_to_try.append(symbol)

        for symbol in symbols_to_try:
            try:
                frame = self.data_manager.fetch_ohlcv(symbol=symbol, timeframe=self.base_timeframe, limit=limit)
                frame = self.cleaner.clean_ohlcv(frame)
                frame = self.features.build(frame)
                atr = pd.to_numeric(frame.get("atr_ratio"), errors="coerce").replace(
                    [float("inf"), float("-inf")],
                    pd.NA,
                )
                atr = atr.dropna()
                if atr.empty or len(atr) < 3:
                    continue

                current = float(atr.iloc[-1])
                history = atr.iloc[:-1].tail(baseline_window)
                baseline = float(history.median()) if not history.empty else 0.0
                if baseline <= 0:
                    baseline = float(self.account_state.get("market_volatility_baseline_atr_ratio", current))
                baseline = max(baseline, 1e-9)

                ratio = max(0.10, min(5.0, current / baseline))
                self.account_state["market_volatility_ratio"] = float(ratio)
                self.account_state["market_volatility_current_atr_ratio"] = current
                self.account_state["market_volatility_baseline_atr_ratio"] = baseline
                self.account_state["market_volatility_symbol"] = symbol
                self.account_state["market_volatility_updated_at"] = datetime.now(timezone.utc).isoformat()
                return
            except Exception:
                continue

        self.logger.warning("Unable to refresh market_volatility_ratio this cycle.")

    def _update_market_drift_state(self) -> None:
        if not self.drift_monitor.enabled:
            self.account_state["market_drift_detected"] = False
            self.account_state["market_regime"] = "disabled"
            return

        risk_cfg = self.config.get("risk", {})
        configured_symbol = str(risk_cfg.get("drift_benchmark_symbol", "")).strip().upper()
        quote = self._quote_asset()
        symbols_to_try: list[str] = []
        for symbol in [configured_symbol, f"BTC/{quote}", "BTC/USDT"]:
            if symbol and symbol not in symbols_to_try:
                symbols_to_try.append(symbol)

        limit = self.drift_monitor.baseline_window + self.drift_monitor.recent_window + 8
        for symbol in symbols_to_try:
            try:
                frame = self.data_manager.fetch_ohlcv(symbol=symbol, timeframe=self.base_timeframe, limit=limit)
                if frame.empty or "close" not in frame.columns:
                    continue
                drift = self.drift_monitor.evaluate_from_close(frame["close"])
                self.account_state["market_drift_detected"] = bool(drift.drift_detected)
                self.account_state["market_drift_ks_stat"] = float(drift.ks_stat)
                self.account_state["market_drift_p_value"] = float(drift.p_value)
                self.account_state["market_drift_vol_ratio"] = float(drift.vol_ratio)
                self.account_state["market_regime"] = str(drift.regime)
                self.account_state["market_drift_symbol"] = symbol
                self.account_state["market_drift_updated_at"] = datetime.now(timezone.utc).isoformat()
                return
            except Exception:
                continue

        self.logger.warning("Unable to refresh market drift state this cycle.")

    def _update_market_intelligence(self, pairs: list[str], opportunities_all) -> None:
        intelligence = getattr(self, "market_intelligence", None)
        if intelligence is None:
            return
        try:
            snapshot = intelligence.update(
                pairs=pairs,
                opportunities=opportunities_all,
                quote_asset=self._quote_asset(),
            )
            payload = snapshot.to_dict()

            self.account_state["market_intelligence_signal"] = payload.get("signal", "HOLD")
            self.account_state["market_intelligence_confidence"] = float(payload.get("confidence", 0.0))
            self.account_state["market_intelligence_score"] = float(payload.get("market_score", 0.0))
            self.account_state["market_intelligence_raw_score"] = float(payload.get("raw_market_score", 0.0))
            self.account_state["market_intelligence_smoothed_score"] = float(payload.get("smoothed_market_score", 0.0))
            self.account_state["market_intelligence_regime"] = str(payload.get("market_regime", "unknown"))
            self.account_state["market_intelligence_predicted_move_pct"] = float(payload.get("predicted_move_pct", 0.0))
            self.account_state["market_intelligence_symbols"] = int(payload.get("symbols_scanned", 0))
            self.account_state["market_intelligence_model_samples"] = int(payload.get("model_samples", 0))
            self.account_state["market_intelligence_profile"] = str(payload.get("profile", "neutral"))
            self.account_state["market_intelligence_coverage_ratio"] = float(payload.get("data_coverage_ratio", 0.0))
            self.account_state["market_intelligence_directional_streak"] = int(payload.get("directional_streak", 0))
            self.account_state["market_intelligence_updated_at"] = payload.get("generated_at")

            self.store.set_state("santrade_intelligence", payload)
            self._export_market_intelligence(payload)
        except Exception as exc:
            self.logger.warning("SanTradeIntelligence update failed: %s", exc)

    def _estimate_atr_value(self, symbol: str, price: float) -> float:
        frame = self.data_manager.fetch_ohlcv(symbol=symbol, timeframe=self.base_timeframe, limit=self.atr_ohlcv_limit)
        frame = self.cleaner.clean_ohlcv(frame)
        frame = self.features.build(frame)
        atr_ratio = float(frame.iloc[-1].get("atr_ratio", 0.01))
        return max(0.001, atr_ratio * max(price, 1e-9))

    def _register_realized_trade(self, pnl_usdt: float, pnl_pct: float) -> None:
        self.account_state["closed_trades"] = int(self.account_state.get("closed_trades", 0)) + 1

        if pnl_usdt > 0:
            self.account_state["wins"] = int(self.account_state.get("wins", 0)) + 1
            self.account_state["sum_win_pct"] = float(self.account_state.get("sum_win_pct", 0.0)) + pnl_pct
            self.account_state["consecutive_losses"] = 0
        else:
            self.account_state["losses"] = int(self.account_state.get("losses", 0)) + 1
            self.account_state["sum_loss_pct"] = float(self.account_state.get("sum_loss_pct", 0.0)) + abs(pnl_pct)
            self.account_state["consecutive_losses"] = int(self.account_state.get("consecutive_losses", 0)) + 1

        closed = max(int(self.account_state.get("closed_trades", 1)), 1)
        wins = int(self.account_state.get("wins", 0))
        losses = int(self.account_state.get("losses", 0))

        self.account_state["win_rate"] = float(wins / closed)
        self.account_state["avg_win"] = float(self.account_state.get("sum_win_pct", 0.0) / wins) if wins else 1.8
        self.account_state["avg_loss"] = (
            float(self.account_state.get("sum_loss_pct", 0.0) / losses) if losses else 1.0
        )

        total_capital = float(self.account_state.get("total_capital", 0.0)) + pnl_usdt
        self.account_state["total_capital"] = total_capital

        active_ratio = float(self.config.get("risk", {}).get("capital_active_ratio", 0.60))
        self.account_state["active_capital"] = total_capital * active_ratio

        daily_realized = float(self.account_state.get("daily_realized_usdt", 0.0)) + pnl_usdt
        weekly_realized = float(self.account_state.get("weekly_realized_usdt", 0.0)) + pnl_usdt
        self.account_state["daily_realized_usdt"] = daily_realized
        self.account_state["weekly_realized_usdt"] = weekly_realized

        denom = max(total_capital, 1e-9)
        self.account_state["daily_pnl_pct"] = (daily_realized / denom) * 100.0
        self.account_state["weekly_pnl_pct"] = (weekly_realized / denom) * 100.0

    def _execute_sell(
        self,
        symbol: str,
        entry_price: float,
        fallback_price: float,
        reason: str,
        requested_base_qty: float | None = None,
        requested_size_usdt: float | None = None,
    ) -> tuple[float, float, float, float]:
        result = self.order_manager.place_market_sell(
            symbol=symbol,
            price=fallback_price,
            size_usdt=requested_size_usdt,
            base_qty=requested_base_qty,
        )
        exit_price = float(result.get("price", fallback_price))
        executed_base_qty = float(result.get("base_qty", 0.0))
        executed_quote_size = float(result.get("size_usdt", 0.0))

        cost_basis_closed = executed_base_qty * max(entry_price, 1e-9)
        pnl_usdt = executed_quote_size - cost_basis_closed
        pnl_pct = ((exit_price / max(entry_price, 1e-9)) - 1.0) * 100.0

        self.store.insert_trade(
            symbol=symbol,
            side="SELL",
            size_usdt=executed_quote_size,
            price=exit_price,
            mode="paper" if self.paper else "live",
            extra={
                "reason": reason,
                "entry_price": entry_price,
                "base_qty": executed_base_qty,
                "cost_basis_closed": cost_basis_closed,
                "pnl_usdt": pnl_usdt,
                "pnl_pct": pnl_pct,
            },
        )
        self._register_realized_trade(pnl_usdt=pnl_usdt, pnl_pct=pnl_pct)
        return executed_base_qty, cost_basis_closed, exit_price, pnl_usdt

    def _manage_open_positions(self) -> tuple[int, int]:
        now = datetime.now(timezone.utc)
        self._roll_period_counters(now)

        full_closes = 0
        partial_closes = 0

        for position in list(self.position_manager.get_open_positions()):
            try:
                price = self.realtime_prices.get_price(position.symbol)
                if price is None or price <= 0:
                    price = self.data_manager.fetch_last_price(position.symbol)
                atr_value = self._estimate_atr_value(symbol=position.symbol, price=price)

                decision = self.exit_manager.evaluate_long(position=position, price=price, atr_value=atr_value, now=now)
                position.stop_loss = decision.new_stop_loss
                position.extra = decision.updated_extra

                initial_base_qty = float(
                    position.extra.get("initial_base_qty", position.size_usdt / max(position.entry_price, 1e-9))
                )
                remaining_base_qty = float(
                    position.extra.get("remaining_base_qty", position.size_usdt / max(position.entry_price, 1e-9))
                )
                position.extra["initial_base_qty"] = initial_base_qty
                position.extra["remaining_base_qty"] = remaining_base_qty

                for frac in decision.partial_fracs:
                    if remaining_base_qty <= 1e-12:
                        break
                    requested_base_qty = min(remaining_base_qty, initial_base_qty * frac)
                    if requested_base_qty <= 1e-12:
                        continue
                    try:
                        executed_base_qty, cost_basis_closed, exit_price, pnl_usdt = self._execute_sell(
                            symbol=position.symbol,
                            entry_price=position.entry_price,
                            fallback_price=price,
                            reason="tp1" if frac >= 0.49 else "tp2",
                            requested_base_qty=requested_base_qty,
                        )
                    except ValueError as exc:
                        self.logger.info(
                            "PARTIAL CLOSE skipped %s frac=%.2f reason=%s",
                            position.symbol,
                            frac,
                            exc,
                        )
                        continue

                    remaining_base_qty = max(0.0, remaining_base_qty - executed_base_qty)
                    position.size_usdt = max(0.0, position.size_usdt - cost_basis_closed)
                    position.extra["remaining_base_qty"] = remaining_base_qty
                    partial_closes += 1
                    self.logger.info(
                        "PARTIAL CLOSE %s frac=%.2f base=%.8f exit=%.4f pnl=%.2f",
                        position.symbol,
                        frac,
                        executed_base_qty,
                        exit_price,
                        pnl_usdt,
                    )

                min_size = float(self.account_state.get("active_capital", 0.0)) * float(
                    self.config.get("risk", {}).get("min_position_pct_active", 0.01)
                )

                should_close_all = decision.close_all or (0 < position.size_usdt < min_size) or (0 < remaining_base_qty <= 1e-12)
                close_reason = decision.close_reason or ("below_min_size" if should_close_all else None)

                if should_close_all and remaining_base_qty > 1e-12:
                    try:
                        executed_base_qty, cost_basis_closed, exit_price, pnl_usdt = self._execute_sell(
                            symbol=position.symbol,
                            entry_price=position.entry_price,
                            fallback_price=price,
                            reason=close_reason or "close_all",
                            requested_base_qty=remaining_base_qty,
                        )
                        remaining_base_qty = max(0.0, remaining_base_qty - executed_base_qty)
                        position.size_usdt = max(0.0, position.size_usdt - cost_basis_closed)
                        position.extra["remaining_base_qty"] = remaining_base_qty
                        self.logger.info(
                            "CLOSE %s reason=%s base=%.8f exit=%.4f pnl=%.2f",
                            position.symbol,
                            close_reason,
                            executed_base_qty,
                            exit_price,
                            pnl_usdt,
                        )
                    except ValueError as exc:
                        position.extra["close_blocked_reason"] = str(exc)
                        self.logger.info("CLOSE blocked %s reason=%s", position.symbol, exc)

                if position.size_usdt <= 1e-6 or remaining_base_qty <= 1e-12:
                    self.position_manager.mark_closed(position.symbol)
                    full_closes += 1
                else:
                    self.position_manager.update(position)
            except Exception as exc:
                self.logger.exception("Error while managing position %s: %s", position.symbol, exc)

        return full_closes, partial_closes

    def run_cycle(self) -> None:
        self._sync_realtime_price_stream(force=False)
        self._sync_live_capital()
        self.store.save_account_state(self.account_state)
        full_closes, partial_closes = self._manage_open_positions()
        self._refresh_recent_performance()
        self._update_market_volatility_ratio()
        self._update_market_drift_state()

        pairs = self._resolve_pairs_for_scan(force_refresh=False)
        with self._model_components_lock:
            scanner = self.scanner
        opportunities, opportunities_all = scanner.scan_details(pairs)
        self._export_scan(opportunities, opportunities_all)
        self._update_market_intelligence(pairs=pairs, opportunities_all=opportunities_all)

        self.logger.info("Scan completed: %s selected opportunities out of %s scanned", len(opportunities), len(opportunities_all))

        opened_count = 0
        for opp in opportunities:
            try:
                if self.position_manager.has_open(opp.symbol):
                    continue

                allowed, reasons, size_usdt = self.risk_manager.can_open_position(
                    opportunity=opp,
                    open_positions=self.position_manager.list_open(),
                    account_state=self.account_state,
                )

                if not allowed:
                    self.logger.info("Skip %s: %s", opp.symbol, "; ".join(reasons))
                    if any(
                        ("circuit breaker" in reason.lower()) or ("risk budget exhausted" in reason.lower())
                        for reason in reasons
                    ):
                        self.alerter.send(f"Risk gate blocked {opp.symbol}: {'; '.join(reasons)}")
                    continue

                price = self.data_manager.fetch_last_price(opp.symbol)
                atr_value = max(0.001, opp.atr_ratio * price)
                pos = self.order_manager.place_market_buy(opp.symbol, size_usdt=size_usdt, price=price, atr=atr_value)
                self.position_manager.add(pos)
                opened_count += 1
                risk_snapshot = self.risk_manager.risk_budget_snapshot(account_state=self.account_state, size_usdt=size_usdt)

                self.store.insert_trade(
                    symbol=pos.symbol,
                    side="BUY",
                    size_usdt=pos.size_usdt,
                    price=pos.entry_price,
                    mode="paper" if self.paper else "live",
                    extra={
                        "confidence": opp.signal.confidence,
                        "global_score": opp.global_score,
                        "daily_risk_used_pct": risk_snapshot.get("daily_used_pct", 0.0),
                        "weekly_risk_used_pct": risk_snapshot.get("weekly_used_pct", 0.0),
                        "daily_risk_with_trade_pct": risk_snapshot.get("daily_used_with_trade_pct", 0.0),
                        "weekly_risk_with_trade_pct": risk_snapshot.get("weekly_used_with_trade_pct", 0.0),
                        "market_volatility_ratio": float(self.account_state.get("market_volatility_ratio", 1.0)),
                        "market_drift_detected": bool(self.account_state.get("market_drift_detected", False)),
                        "market_regime": str(self.account_state.get("market_regime", "unknown")),
                        "santrade_intelligence_signal": str(self.account_state.get("market_intelligence_signal", "HOLD")),
                        "santrade_intelligence_confidence": float(
                            self.account_state.get("market_intelligence_confidence", 0.0)
                        ),
                        "santrade_intelligence_profile": str(
                            self.account_state.get("market_intelligence_profile", "neutral")
                        ),
                        "santrade_intelligence_score": float(
                            self.account_state.get("market_intelligence_score", 0.0)
                        ),
                        "santrade_intelligence_regime": str(
                            self.account_state.get("market_intelligence_regime", "unknown")
                        ),
                    },
                )
                self.logger.info("OPEN %s size=%.2f entry=%.4f", pos.symbol, pos.size_usdt, pos.entry_price)
            except OrderConstraintError as exc:
                self.logger.info("Skip %s: %s", opp.symbol, exc)
            except Exception as exc:
                self.logger.exception("Error while processing %s: %s", opp.symbol, exc)

        self.store.insert_cycle(
            opportunities=len(opportunities),
            opened_positions=opened_count,
            data={
                "paper": self.paper,
                "open_positions": len(self.position_manager.list_open()),
                "full_closes": full_closes,
                "partial_closes": partial_closes,
                "daily_pnl_pct": self.account_state.get("daily_pnl_pct", 0.0),
                "weekly_pnl_pct": self.account_state.get("weekly_pnl_pct", 0.0),
                "market_intelligence_signal": self.account_state.get("market_intelligence_signal", "HOLD"),
                "market_intelligence_confidence": self.account_state.get("market_intelligence_confidence", 0.0),
                "market_intelligence_score": self.account_state.get("market_intelligence_score", 0.0),
                "market_intelligence_regime": self.account_state.get("market_intelligence_regime", "unknown"),
                "market_intelligence_profile": self.account_state.get("market_intelligence_profile", "neutral"),
            },
        )
        self._sync_live_capital()
        self.store.save_account_state(self.account_state)
        self.store.export_positions_snapshot()
        self.backup_manager.maybe_backup(force=False)

        if opened_count == 0 and full_closes == 0 and partial_closes == 0:
            self.alerter.send("No position changes in this cycle.")


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

    runtime = TradingRuntime(config=config, paper=paper, disable_retrain=args.disable_retrain)
    dashboard_proc: subprocess.Popen | None = None
    dashboard_log_fh = None

    scan_interval = int(config.get("scanner", {}).get("scan_interval_sec", 300))

    try:
        if args.once:
            runtime.run_cycle()
            return

        dashboard_proc, dashboard_log_fh = maybe_start_dashboard(config=config, args=args, logger=runtime.logger)

        while True:
            runtime.run_cycle()
            time.sleep(scan_interval)
    except KeyboardInterrupt:
        runtime.logger.info("Shutdown requested by user (KeyboardInterrupt)")
    finally:
        runtime.shutdown()
        stop_dashboard(proc=dashboard_proc, log_fh=dashboard_log_fh, logger=runtime.logger)


if __name__ == "__main__":
    main()
