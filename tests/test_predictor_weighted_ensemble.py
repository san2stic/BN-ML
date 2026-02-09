from __future__ import annotations

import numpy as np
import pandas as pd

from bn_ml.symbols import symbol_to_model_key
from ml_engine.predictor import MLEnsemblePredictor


class _DummyModel:
    def __init__(self, pred: int, proba: list[float]) -> None:
        self.pred = int(pred)
        self._proba = np.asarray(proba, dtype=float)
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def predict(self, row: np.ndarray) -> np.ndarray:
        return np.array([self.pred], dtype=int)

    def predict_proba(self, row: np.ndarray) -> np.ndarray:
        return np.asarray([self._proba], dtype=float)


class _DummyXGBCudaBooster:
    def predict(self, matrix) -> np.ndarray:  # type: ignore[no-untyped-def]
        return np.asarray([[0.10, 0.20, 0.70]], dtype=float)


class _DummyXGBCudaModel:
    def __init__(self) -> None:
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def get_xgb_params(self) -> dict[str, str]:
        return {"device": "cuda"}

    def get_booster(self) -> _DummyXGBCudaBooster:
        return _DummyXGBCudaBooster()

    def predict(self, row: np.ndarray) -> np.ndarray:
        raise AssertionError("predict() should not be called for CUDA XGB path")

    def predict_proba(self, row: np.ndarray) -> np.ndarray:
        raise AssertionError("predict_proba() should not be called for CUDA XGB path")


def test_weighted_ensemble_prefers_stronger_model_weight() -> None:
    predictor = MLEnsemblePredictor(model_dir="unused")
    key = symbol_to_model_key("BTC/USDC")
    predictor._model_cache[key] = {
        "models": {
            "rf": _DummyModel(pred=2, proba=[0.05, 0.15, 0.80]),
            "xgb": _DummyModel(pred=0, proba=[0.80, 0.15, 0.05]),
        },
        "metadata": {
            "feature_columns": ["f1"],
            "ensemble_weights": {"rf": 0.85, "xgb": 0.15},
            "decision_params": {
                "min_buy_proba": 0.40,
                "min_sell_proba": 0.40,
                "min_margin": 0.05,
                "hold_bias": 0.01,
            },
        },
    }

    signal = predictor.predict(symbol="BTC/USDC", frame=pd.DataFrame([{"f1": 1.0}]), feature_columns=["f1"])

    assert signal.action == "BUY"
    assert signal.metadata["proba_buy"] > signal.metadata["proba_sell"]


def test_weighted_ensemble_holds_when_margin_too_low() -> None:
    predictor = MLEnsemblePredictor(model_dir="unused")
    key = symbol_to_model_key("ETH/USDC")
    predictor._model_cache[key] = {
        "models": {
            "rf": _DummyModel(pred=2, proba=[0.30, 0.15, 0.55]),
            "xgb": _DummyModel(pred=0, proba=[0.52, 0.18, 0.30]),
        },
        "metadata": {
            "feature_columns": ["f1", "f_missing"],
            "ensemble_weights": {"rf": 0.55, "xgb": 0.45},
            "decision_params": {
                "min_buy_proba": 0.35,
                "min_sell_proba": 0.35,
                "min_margin": 0.30,
                "hold_bias": 0.01,
            },
        },
    }

    signal = predictor.predict(symbol="ETH/USDC", frame=pd.DataFrame([{"f1": 2.0}]), feature_columns=["f1"])

    assert signal.action == "HOLD"
    assert signal.metadata["margin"] < 0.30


def test_xgb_cuda_path_uses_booster_predict_without_predict_proba() -> None:
    predictor = MLEnsemblePredictor(model_dir="unused")
    key = symbol_to_model_key("SOL/USDC")
    predictor._model_cache[key] = {
        "models": {"xgb": _DummyXGBCudaModel()},
        "metadata": {
            "feature_columns": ["f1"],
            "ensemble_weights": {"xgb": 1.0},
            "decision_params": {
                "min_buy_proba": 0.40,
                "min_sell_proba": 0.40,
                "min_margin": 0.05,
                "hold_bias": 0.01,
            },
        },
    }

    signal = predictor.predict(symbol="SOL/USDC", frame=pd.DataFrame([{"f1": 1.0}]), feature_columns=["f1"])
    assert signal.action == "BUY"
    assert signal.metadata["proba_buy"] > signal.metadata["proba_sell"]
