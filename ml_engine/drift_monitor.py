from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DriftMetrics:
    drift_detected: bool
    ks_stat: float
    p_value: float
    baseline_vol: float
    recent_vol: float
    vol_ratio: float
    sample_baseline: int
    sample_recent: int
    regime: str


class MarketDriftMonitor:
    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg = config.get("model", {})
        drift_cfg = model_cfg.get("drift", {}) if isinstance(model_cfg.get("drift", {}), dict) else {}

        self.enabled = bool(drift_cfg.get("enabled", True))
        self.baseline_window = max(40, int(drift_cfg.get("baseline_window", 240)))
        self.recent_window = max(20, int(drift_cfg.get("recent_window", 64)))
        self.ks_threshold = float(drift_cfg.get("ks_threshold", 0.24))
        self.p_value_threshold = float(drift_cfg.get("p_value_threshold", 0.05))
        self.vol_ratio_threshold = float(drift_cfg.get("vol_ratio_threshold", 1.8))

    @staticmethod
    def _two_sample_ks(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        if x.size == 0 or y.size == 0:
            return 0.0, 1.0

        x = np.sort(np.asarray(x, dtype=float))
        y = np.sort(np.asarray(y, dtype=float))
        grid = np.sort(np.unique(np.concatenate([x, y])))
        if grid.size == 0:
            return 0.0, 1.0

        cdf_x = np.searchsorted(x, grid, side="right") / max(len(x), 1)
        cdf_y = np.searchsorted(y, grid, side="right") / max(len(y), 1)
        d = float(np.max(np.abs(cdf_x - cdf_y)))

        n = len(x)
        m = len(y)
        en = (n * m) / max(n + m, 1)
        # Asymptotic p-value approximation (good enough for runtime guardrail).
        p = float(min(1.0, max(0.0, 2.0 * np.exp(-2.0 * en * d * d))))
        return d, p

    def evaluate_from_close(self, close: pd.Series) -> DriftMetrics:
        if not self.enabled:
            return DriftMetrics(False, 0.0, 1.0, 0.0, 0.0, 1.0, 0, 0, "disabled")

        returns = pd.to_numeric(close, errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        needed = self.baseline_window + self.recent_window
        if len(returns) < needed:
            return DriftMetrics(False, 0.0, 1.0, 0.0, 0.0, 1.0, 0, 0, "insufficient_data")

        recent = returns.iloc[-self.recent_window :].to_numpy(dtype=float)
        baseline = returns.iloc[-(self.baseline_window + self.recent_window) : -self.recent_window].to_numpy(dtype=float)

        ks_stat, p_value = self._two_sample_ks(baseline, recent)
        baseline_vol = float(np.std(baseline)) if baseline.size else 0.0
        recent_vol = float(np.std(recent)) if recent.size else 0.0
        vol_ratio = float(recent_vol / max(baseline_vol, 1e-12))

        ks_trigger = ks_stat >= self.ks_threshold and p_value <= self.p_value_threshold
        vol_trigger = vol_ratio >= self.vol_ratio_threshold
        drift = bool(ks_trigger or vol_trigger)

        if vol_ratio >= self.vol_ratio_threshold:
            regime = "high_volatility_shift"
        elif ks_trigger:
            regime = "distribution_shift"
        else:
            regime = "stable"

        return DriftMetrics(
            drift_detected=drift,
            ks_stat=ks_stat,
            p_value=p_value,
            baseline_vol=baseline_vol,
            recent_vol=recent_vol,
            vol_ratio=vol_ratio,
            sample_baseline=int(len(baseline)),
            sample_recent=int(len(recent)),
            regime=regime,
        )
