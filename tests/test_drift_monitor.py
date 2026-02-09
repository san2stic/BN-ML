from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.drift_monitor import MarketDriftMonitor


def test_drift_monitor_detects_distribution_shift() -> None:
    rng = np.random.default_rng(42)
    baseline = rng.normal(0.0, 0.002, size=300)
    recent = rng.normal(0.006, 0.004, size=120)
    rets = np.concatenate([baseline, recent])
    close = 100.0 * np.cumprod(1.0 + rets)

    monitor = MarketDriftMonitor(
        config={
            "model": {
                "drift": {
                    "enabled": True,
                    "baseline_window": 240,
                    "recent_window": 80,
                    "ks_threshold": 0.15,
                    "p_value_threshold": 0.20,
                    "vol_ratio_threshold": 1.2,
                }
            }
        }
    )
    metrics = monitor.evaluate_from_close(pd.Series(close))

    assert metrics.drift_detected is True
    assert metrics.ks_stat > 0
    assert metrics.sample_recent == 80


def test_drift_monitor_returns_stable_when_similar_windows() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0, 0.0015, size=500)
    close = 100.0 * np.cumprod(1.0 + rets)
    monitor = MarketDriftMonitor(config={"model": {"drift": {"enabled": True, "baseline_window": 220, "recent_window": 60}}})

    metrics = monitor.evaluate_from_close(pd.Series(close))
    assert metrics.drift_detected is False
    assert metrics.regime in {"stable", "insufficient_data"}


def test_drift_monitor_handles_insufficient_data() -> None:
    close = pd.Series([100.0, 100.1, 100.2, 100.3])
    monitor = MarketDriftMonitor(config={"model": {"drift": {"enabled": True, "baseline_window": 50, "recent_window": 20}}})
    metrics = monitor.evaluate_from_close(close)
    assert metrics.drift_detected is False
    assert metrics.regime == "insufficient_data"
