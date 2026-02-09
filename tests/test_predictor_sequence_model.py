from __future__ import annotations

import numpy as np
import pandas as pd

from bn_ml.symbols import symbol_to_model_key
from ml_engine.predictor import MLEnsemblePredictor


class _SeqDummyModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([1] * max(len(x) - 1, 0) + [2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(len(x)):
            if i == len(x) - 1:
                rows.append([0.05, 0.10, 0.85])
            else:
                rows.append([0.30, 0.55, 0.15])
        return np.asarray(rows, dtype=float)

    @property
    def classes_(self):
        return np.array([0, 1, 2], dtype=int)


def test_predictor_uses_sequence_model_last_window_output() -> None:
    predictor = MLEnsemblePredictor(model_dir="unused")
    key = symbol_to_model_key("BTC/USDC")
    predictor._model_cache[key] = {
        "models": {"lstm": _SeqDummyModel()},
        "metadata": {
            "feature_columns": ["f1", "f2"],
            "ensemble_weights": {"lstm": 1.0},
            "decision_params": {"min_buy_proba": 0.4, "min_sell_proba": 0.4, "min_margin": 0.02, "hold_bias": 0.01},
        },
    }

    frame = pd.DataFrame([{"f1": 0.1, "f2": 0.2}, {"f1": 0.2, "f2": 0.1}, {"f1": 0.8, "f2": 0.4}])
    signal = predictor.predict(symbol="BTC/USDC", frame=frame, feature_columns=["f1", "f2"])
    assert signal.action == "BUY"
    assert signal.metadata["proba_buy"] > 0.8
