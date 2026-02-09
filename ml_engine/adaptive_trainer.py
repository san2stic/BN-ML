from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

from bn_ml.state_store import StateStore


class AdaptiveRetrainer:
    def __init__(self, config: dict, store: StateStore) -> None:
        self.config = config
        self.store = store

    def should_retrain(self, account_state: dict[str, Any], now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc)

        interval_h = float(self.config.get("model", {}).get("retrain_interval_hours", 6))
        last_retrain_iso = self.store.get_state("last_retrain_at", None)

        if not last_retrain_iso:
            return True, "no previous training found"

        try:
            last_retrain_at = datetime.fromisoformat(last_retrain_iso)
        except Exception:
            return True, "invalid last_retrain_at format"

        if now - last_retrain_at >= timedelta(hours=interval_h):
            return True, f"interval {interval_h}h elapsed"

        baseline = float(account_state.get("baseline_win_rate", 0.56))
        current = float(account_state.get("current_win_rate_24h", baseline))

        if baseline <= 0:
            return False, "baseline win rate unavailable"

        drop_pct = ((baseline - current) / baseline) * 100
        trigger = float(self.config.get("model", {}).get("retrain_degradation_trigger_pct", 15))
        if drop_pct >= trigger:
            return True, f"performance degradation {drop_pct:.2f}% >= {trigger:.2f}%"

        return False, "conditions not met"

    def maybe_retrain(
        self,
        account_state: dict[str, Any],
        trainer_func: Callable[[], dict[str, Any]],
    ) -> tuple[bool, str, dict[str, Any] | None]:
        should, reason = self.should_retrain(account_state)
        if not should:
            return False, reason, None

        result = trainer_func()
        self.store.set_state("last_retrain_at", datetime.now(timezone.utc).isoformat())
        metrics = result.get("metrics") if isinstance(result, dict) else None
        if isinstance(metrics, dict):
            self.store.insert_model_metrics(metrics)
        return True, reason, result


class BackgroundRetrainWorker:
    def __init__(
        self,
        *,
        config: dict,
        retrainer: AdaptiveRetrainer,
        get_account_state: Callable[[], dict[str, Any]],
        periodic_train_func: Callable[[], dict[str, Any]],
        missing_train_func: Callable[[list[str]], dict[str, Any]],
        on_models_updated: Callable[[str, dict[str, Any] | None], None] | None = None,
        logger: logging.Logger | None = None,
        check_interval_sec: float | None = None,
    ) -> None:
        self.config = config
        self.retrainer = retrainer
        self.get_account_state = get_account_state
        self.periodic_train_func = periodic_train_func
        self.missing_train_func = missing_train_func
        self.on_models_updated = on_models_updated

        model_cfg = self.config.get("model", {})
        raw_interval = check_interval_sec if check_interval_sec is not None else model_cfg.get("retrain_check_interval_sec", 30)
        self.check_interval_sec = max(0.5, float(raw_interval))
        try:
            parsed_batch_size = int(model_cfg.get("auto_train_missing_batch_size", 5))
        except (TypeError, ValueError):
            parsed_batch_size = 5
        self.missing_batch_size = max(1, parsed_batch_size)
        self.logger = logger or logging.getLogger("bn_ml.retrain_worker")

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lock = threading.Lock()
        self._pending_missing_symbols: set[str] = set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.set()
        self._thread = threading.Thread(target=self._run, name="bn-ml-retrain-worker", daemon=True)
        self._thread.start()
        self.logger.info(
            "Background retrain worker started (check_interval_sec=%.2f, missing_batch_size=%s)",
            self.check_interval_sec,
            self.missing_batch_size,
        )

    def stop(self, timeout_sec: float = 10.0) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._wake_event.set()
        self._thread.join(timeout=max(0.1, float(timeout_sec)))
        if self._thread.is_alive():
            self.logger.warning("Background retrain worker did not stop within %.1fs", timeout_sec)
        else:
            self.logger.info("Background retrain worker stopped")
        self._thread = None

    def queue_missing_symbol(self, symbol: str) -> None:
        self.queue_missing_symbols([symbol])

    def queue_missing_symbols(self, symbols: Iterable[str]) -> None:
        normalized: set[str] = set()
        for symbol in symbols:
            text = str(symbol).strip().upper()
            if text:
                normalized.add(text)
        if not normalized:
            return
        with self._lock:
            self._pending_missing_symbols.update(normalized)
        self._wake_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=self.check_interval_sec)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break

            did_missing_train = self._train_missing_models()
            if did_missing_train:
                continue
            self._maybe_periodic_retrain()

    def _train_missing_models(self) -> bool:
        with self._lock:
            symbols = sorted(self._pending_missing_symbols)
            self._pending_missing_symbols.clear()

        if not symbols:
            return False

        batch = symbols[: self.missing_batch_size]
        remaining = symbols[self.missing_batch_size :]
        if remaining:
            with self._lock:
                self._pending_missing_symbols.update(remaining)
            self._wake_event.set()

        self.logger.info(
            "Background auto-train missing models start (batch=%s queued=%s remaining=%s)",
            len(batch),
            len(symbols),
            len(remaining),
        )

        try:
            result = self.missing_train_func(batch)
        except Exception as exc:
            self.logger.exception("Background auto-train for missing models failed: %s", exc)
            return True

        trained_count = self._trained_count(result)
        self.logger.info(
            "Background auto-train missing models done (batch=%s trained=%s remaining=%s)",
            len(batch),
            trained_count,
            len(remaining),
        )
        if trained_count > 0:
            self._notify_models_updated(reason="missing_models", result=result)
        return True

    def _maybe_periodic_retrain(self) -> None:
        try:
            retrained, reason, result = self.retrainer.maybe_retrain(
                account_state=self.get_account_state(),
                trainer_func=self.periodic_train_func,
            )
        except Exception as exc:
            self.logger.exception("Background retraining failed: %s", exc)
            return

        if not retrained:
            return

        self.logger.info("Background retraining triggered: %s", reason)
        if self._trained_count(result) > 0:
            self._notify_models_updated(reason=f"periodic:{reason}", result=result)

    def _notify_models_updated(self, reason: str, result: dict[str, Any] | None) -> None:
        if self.on_models_updated is None:
            return
        try:
            self.on_models_updated(reason, result)
        except Exception as exc:
            self.logger.exception("on_models_updated callback failed: %s", exc)

    @staticmethod
    def _trained_count(result: dict[str, Any] | None) -> int:
        if not isinstance(result, dict):
            return 0
        aggregate = result.get("aggregate")
        if isinstance(aggregate, dict):
            try:
                return int(aggregate.get("symbols_trained", 0))
            except (TypeError, ValueError):
                return 0
        saved_models = result.get("saved_models")
        if isinstance(saved_models, list):
            return len(saved_models)
        return 0
