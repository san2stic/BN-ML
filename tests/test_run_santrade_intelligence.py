from __future__ import annotations

import scripts.run_santrade_intelligence as run_santrade_intelligence


def test_resolve_interval_seconds_uses_config_default() -> None:
    interval = run_santrade_intelligence.resolve_interval_seconds(
        config={"model": {"santrade_intelligence": {"runtime_interval_sec": 120}}},
        override_seconds=None,
    )
    assert interval == 120.0


def test_resolve_interval_seconds_override_has_floor() -> None:
    interval = run_santrade_intelligence.resolve_interval_seconds(
        config={},
        override_seconds=4.0,
    )
    assert interval == 15.0


def test_merge_snapshot_into_account_state_sets_runtime_fields() -> None:
    payload = {
        "signal": "SELL",
        "confidence": 81.0,
        "market_score": -0.40,
        "raw_market_score": -0.48,
        "smoothed_market_score": -0.40,
        "market_regime": "risk_off",
        "predicted_move_pct": -1.2,
        "symbols_scanned": 28,
        "model_samples": 42,
        "profile": "defensive",
        "data_coverage_ratio": 0.74,
        "directional_streak": 3,
        "generated_at": "2026-02-10T12:00:00+00:00",
    }

    merged = run_santrade_intelligence.SanTradeIntelligenceRuntime._merge_snapshot_into_account_state(
        {"total_capital": 1000.0},
        payload,
    )

    assert merged["market_intelligence_signal"] == "SELL"
    assert merged["market_intelligence_regime"] == "risk_off"
    assert merged["market_intelligence_profile"] == "defensive"
    assert merged["market_intelligence_directional_streak"] == 3


def test_run_loop_respects_max_cycles(monkeypatch) -> None:
    sleeps: list[float] = []

    class _Runtime:
        class _Logger:
            def info(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                return None

            def exception(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                return None

        def __init__(self) -> None:
            self.logger = self._Logger()
            self.cycles = 0

        def run_cycle(self) -> dict:
            self.cycles += 1
            return {"signal": "HOLD"}

    runtime = _Runtime()
    monkeypatch.setattr(run_santrade_intelligence, "_sleep_with_interrupt", lambda sec: sleeps.append(sec))

    code = run_santrade_intelligence.run_loop(
        runtime=runtime,  # type: ignore[arg-type]
        interval_seconds=22.0,
        max_cycles=2,
    )

    assert code == 0
    assert runtime.cycles == 2
    assert sleeps == [22.0]


def test_runtime_shutdown_flushes_state() -> None:
    class _DummyMarketIntelligence:
        def __init__(self) -> None:
            self.flush_calls = 0

        def flush_state(self) -> None:
            self.flush_calls += 1

    class _DummyLogger:
        def warning(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            return None

    runtime = object.__new__(run_santrade_intelligence.SanTradeIntelligenceRuntime)
    runtime.market_intelligence = _DummyMarketIntelligence()
    runtime.logger = _DummyLogger()

    runtime.shutdown()

    assert runtime.market_intelligence.flush_calls == 1


def test_runtime_shutdown_handles_flush_errors() -> None:
    class _FailingMarketIntelligence:
        def flush_state(self) -> None:
            raise RuntimeError("flush failed")

    class _DummyLogger:
        def __init__(self) -> None:
            self.warning_messages: list[str] = []

        def warning(self, msg: str, *args) -> None:  # noqa: ANN002
            if args:
                self.warning_messages.append(msg % args)
            else:
                self.warning_messages.append(msg)

    runtime = object.__new__(run_santrade_intelligence.SanTradeIntelligenceRuntime)
    runtime.market_intelligence = _FailingMarketIntelligence()
    runtime.logger = _DummyLogger()

    runtime.shutdown()

    assert runtime.logger.warning_messages
