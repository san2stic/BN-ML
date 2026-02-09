from __future__ import annotations

import scripts.run_trainer_auto as run_trainer_auto


def test_resolve_interval_seconds_uses_config_hours() -> None:
    interval = run_trainer_auto.resolve_interval_seconds(
        config={"model": {"retrain_interval_hours": 2}},
        override_seconds=None,
    )
    assert interval == 7200.0


def test_resolve_interval_seconds_override_and_floor() -> None:
    interval = run_trainer_auto.resolve_interval_seconds(
        config={"model": {"retrain_interval_hours": 6}},
        override_seconds=5,
    )
    assert interval == 30.0


def test_run_loop_respects_max_cycles(monkeypatch) -> None:
    calls: list[dict] = []
    sleeps: list[float] = []

    def _fake_train_once(**kwargs):
        calls.append(kwargs)
        return {
            "aggregate": {
                "symbols_trained": 1,
                "symbols_queued_for_training": 1,
                "symbols_skipped_errors": 0,
                "symbols_skipped_up_to_date": 0,
            }
        }

    monkeypatch.setattr(run_trainer_auto, "train_once", _fake_train_once)
    monkeypatch.setattr(run_trainer_auto, "_sleep_with_interrupt", lambda sec: sleeps.append(sec))

    code = run_trainer_auto.run_loop(
        config={"model": {"retrain_interval_hours": 6}},
        paper=True,
        symbols=["BTC/USDC"],
        train_missing_only=True,
        max_model_age_hours=24.0,
        models_dir="models",
        interval_seconds=42.0,
        startup_delay_seconds=0.0,
        max_cycles=2,
        fail_fast=False,
    )

    assert code == 0
    assert len(calls) == 2
    assert sleeps == [42.0]
