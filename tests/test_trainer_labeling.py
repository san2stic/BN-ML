from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_trainer import _label_edge_profile


def test_label_edge_profile_has_cost_floor() -> None:
    close = 100.0 * np.cumprod(1.0 + np.linspace(-0.001, 0.001, 120))
    frame = pd.DataFrame({"close": close, "atr_ratio": np.full_like(close, 0.01)})

    config = {
        "model": {
            "labeling": {
                "horizon_candles": 4,
                "volatility_window": 20,
                "volatility_multiplier": 0.6,
                "atr_multiplier": 0.35,
                "min_move_pct": 0.001,
                "max_move_pct": 0.02,
                "round_trip_cost_multiplier": 2.0,
                "cost_safety_multiplier": 1.5,
            }
        },
        "risk": {"fees_pct": 0.10, "slippage_pct": 0.05},
    }

    edge, horizon, info = _label_edge_profile(config=config, frame=frame)

    expected_min_edge = ((0.10 + 0.05) / 100.0) * 2.0 * 1.5
    assert horizon == 4
    assert info["min_edge_pct"] == pytest.approx(expected_min_edge)
    assert float(edge.min()) >= expected_min_edge - 1e-12
