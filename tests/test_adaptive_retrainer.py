from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bn_ml.state_store import StateStore
from ml_engine.adaptive_trainer import AdaptiveRetrainer


def test_retrain_without_history(tmp_path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))
    retrainer = AdaptiveRetrainer(config={"model": {"retrain_interval_hours": 6}}, store=store)

    should, reason = retrainer.should_retrain(account_state={"baseline_win_rate": 0.56, "current_win_rate_24h": 0.56})

    assert should is True
    assert "no previous training" in reason


def test_retrain_on_degradation(tmp_path) -> None:
    store = StateStore(db_path=str(tmp_path / "state.db"))
    now = datetime.now(timezone.utc)
    store.set_state("last_retrain_at", (now - timedelta(hours=1)).isoformat())

    retrainer = AdaptiveRetrainer(
        config={"model": {"retrain_interval_hours": 6, "retrain_degradation_trigger_pct": 15}},
        store=store,
    )

    should, reason = retrainer.should_retrain(account_state={"baseline_win_rate": 0.60, "current_win_rate_24h": 0.48})

    assert should is True
    assert "degradation" in reason
