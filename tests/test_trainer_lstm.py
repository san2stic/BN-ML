from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.trainer import EnsembleTrainer


def _dataset(n: int = 280) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(123)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.003, size=n))
    df = pd.DataFrame(
        {
            "close": close,
            "f1": pd.Series(close).pct_change().fillna(0).rolling(3).mean().fillna(0),
            "f2": pd.Series(close).pct_change().fillna(0).rolling(6).std().fillna(0),
            "f3": rng.normal(0, 1, size=n),
        }
    )
    fwd = df["close"].shift(-3) / df["close"] - 1
    df["label"] = np.where(fwd > 0.0013, 2, np.where(fwd < -0.0013, 0, 1))
    df = df.dropna().reset_index(drop=True)
    return df, ["f1", "f2", "f3"]


def test_trainer_can_include_lstm_sequence_model() -> None:
    df, features = _dataset()
    config = {
        "model": {
            "random_state": 42,
            "feature_limit": 3,
            "hpo": {"enabled": False},
            "rf": {"n_estimators": 40, "max_depth": 5, "min_samples_leaf": 5},
            "xgb": {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.05},
            "lgb": {"enabled": False},
            "lstm": {"enabled": True, "sequence_length": 24, "hidden_size": 32, "hidden_size_2": 16, "epochs": 8},
        }
    }

    trainer = EnsembleTrainer(config)
    result = trainer.train(df=df, features=features, target_col="label")

    assert "lstm" in result.models
    assert "lstm_cv_accuracy" in result.metrics
