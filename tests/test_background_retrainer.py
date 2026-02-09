from __future__ import annotations

from datetime import datetime, timezone
import threading

from bn_ml.state_store import StateStore
from ml_engine.adaptive_trainer import AdaptiveRetrainer, BackgroundRetrainWorker


def test_background_worker_runs_periodic_retrain_and_callback(tmp_path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))
    retrainer = AdaptiveRetrainer(config={"model": {"retrain_interval_hours": 6}}, store=store)

    lock = threading.Lock()
    periodic_runs = 0
    callback_reasons: list[str] = []
    periodic_event = threading.Event()
    callback_event = threading.Event()

    def periodic_train_func() -> dict:
        nonlocal periodic_runs
        with lock:
            periodic_runs += 1
        periodic_event.set()
        return {"aggregate": {"symbols_trained": 1}}

    def missing_train_func(symbols: list[str]) -> dict:
        return {"aggregate": {"symbols_trained": 0}}

    def on_models_updated(reason: str, result: dict | None) -> None:
        with lock:
            callback_reasons.append(reason)
        callback_event.set()

    worker = BackgroundRetrainWorker(
        config={"model": {"retrain_check_interval_sec": 0.05, "retrain_interval_hours": 6}},
        retrainer=retrainer,
        get_account_state=lambda: {"baseline_win_rate": 0.56, "current_win_rate_24h": 0.56},
        periodic_train_func=periodic_train_func,
        missing_train_func=missing_train_func,
        on_models_updated=on_models_updated,
    )

    worker.start()
    try:
        assert periodic_event.wait(2.0)
        assert callback_event.wait(2.0)
    finally:
        worker.stop(timeout_sec=2.0)

    with lock:
        assert periodic_runs >= 1
        assert any(reason.startswith("periodic:") for reason in callback_reasons)


def test_background_worker_queues_missing_models_once(tmp_path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))
    store.set_state("last_retrain_at", datetime.now(timezone.utc).isoformat())

    retrainer = AdaptiveRetrainer(
        config={"model": {"retrain_interval_hours": 9999, "retrain_degradation_trigger_pct": 50}},
        store=store,
    )

    lock = threading.Lock()
    missing_batches: list[list[str]] = []
    callback_reasons: list[str] = []
    missing_event = threading.Event()
    periodic_called = threading.Event()

    def periodic_train_func() -> dict:
        periodic_called.set()
        return {"aggregate": {"symbols_trained": 0}}

    def missing_train_func(symbols: list[str]) -> dict:
        with lock:
            missing_batches.append(symbols)
        missing_event.set()
        return {"aggregate": {"symbols_trained": len(symbols)}}

    def on_models_updated(reason: str, result: dict | None) -> None:
        with lock:
            callback_reasons.append(reason)

    worker = BackgroundRetrainWorker(
        config={"model": {"retrain_check_interval_sec": 0.05, "retrain_interval_hours": 9999}},
        retrainer=retrainer,
        get_account_state=lambda: {"baseline_win_rate": 0.56, "current_win_rate_24h": 0.56},
        periodic_train_func=periodic_train_func,
        missing_train_func=missing_train_func,
        on_models_updated=on_models_updated,
    )

    worker.start()
    try:
        worker.queue_missing_symbols(["btc/usdc", "BTC/USDC", "ETH/USDC"])
        assert missing_event.wait(2.0)
    finally:
        worker.stop(timeout_sec=2.0)

    with lock:
        assert missing_batches == [["BTC/USDC", "ETH/USDC"]]
        assert callback_reasons == ["missing_models"]

    assert periodic_called.is_set() is False


def test_background_worker_splits_missing_batches(tmp_path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))
    store.set_state("last_retrain_at", datetime.now(timezone.utc).isoformat())
    retrainer = AdaptiveRetrainer(config={"model": {"retrain_interval_hours": 9999}}, store=store)

    lock = threading.Lock()
    batches: list[list[str]] = []
    done = threading.Event()

    def periodic_train_func() -> dict:
        return {"aggregate": {"symbols_trained": 0}}

    def missing_train_func(symbols: list[str]) -> dict:
        with lock:
            batches.append(symbols)
            if len(batches) >= 2:
                done.set()
        return {"aggregate": {"symbols_trained": len(symbols)}}

    worker = BackgroundRetrainWorker(
        config={
            "model": {
                "retrain_check_interval_sec": 0.05,
                "retrain_interval_hours": 9999,
                "auto_train_missing_batch_size": 2,
            }
        },
        retrainer=retrainer,
        get_account_state=lambda: {"baseline_win_rate": 0.56, "current_win_rate_24h": 0.56},
        periodic_train_func=periodic_train_func,
        missing_train_func=missing_train_func,
    )

    worker.start()
    try:
        worker.queue_missing_symbols(["B/USDC", "A/USDC", "C/USDC"])
        assert done.wait(2.0)
    finally:
        worker.stop(timeout_sec=2.0)

    with lock:
        assert batches[0] == ["A/USDC", "B/USDC"]
        assert batches[1] == ["C/USDC"]
