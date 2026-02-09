from __future__ import annotations

import numpy as np
import pandas as pd

from ml_engine.trainer import EnsembleTrainer


def _dataset(n: int = 320) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.003, size=n)
    close = 100 * np.cumprod(1 + rets)

    df = pd.DataFrame(
        {
            "close": close,
            "f1": pd.Series(close).pct_change().fillna(0).rolling(3).mean().fillna(0),
            "f2": pd.Series(close).pct_change().fillna(0).rolling(8).std().fillna(0),
            "f3": rng.normal(0, 1, size=n),
        }
    )
    fwd = df["close"].shift(-3) / df["close"] - 1
    df["label"] = np.where(fwd > 0.0015, 2, np.where(fwd < -0.0015, 0, 1))
    df = df.dropna().reset_index(drop=True)
    return df, ["f1", "f2", "f3"]


def test_risk_metrics_positive_for_positive_returns() -> None:
    returns = np.array([0.001, 0.002, 0.0015, -0.0002, 0.0012], dtype=float)
    sharpe, sortino = EnsembleTrainer._risk_metrics(returns)

    assert sharpe > 0
    assert sortino > 0


def test_train_with_hpo_returns_best_params() -> None:
    df, features = _dataset()
    config = {
        "model": {
            "random_state": 42,
            "feature_limit": 3,
            "hpo": {
                "enabled": True,
                "max_samples": 300,
                "max_splits": 3,
                "rf_trials": 3,
                "xgb_trials": 1,
                "objective_weight_sharpe": 0.45,
                "objective_weight_sortino": 0.45,
                "objective_weight_accuracy": 0.10,
            },
            "rf": {
                "n_estimators": 50,
                "max_depth": 6,
                "min_samples_leaf": 5,
            },
            "xgb": {
                "n_estimators": 30,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.01,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 5,
            },
        }
    }

    trainer = EnsembleTrainer(config)
    result = trainer.train(df=df, features=features, target_col="label")

    assert "rf" in result.models
    assert "rf" in result.best_params
    assert "rf_hpo_objective" in result.metrics
    assert "rf" in result.validation_metrics
    assert result.ensemble_weights
    assert abs(sum(result.ensemble_weights.values()) - 1.0) < 1e-6
    assert len(result.selected_features) == 3


def test_strategy_returns_after_costs_penalize_turnover() -> None:
    trainer = EnsembleTrainer(
        {
            "model": {"hpo": {"enabled": False}},
            "risk": {"fees_pct": 0.10, "slippage_pct": 0.05},
        }
    )

    close = np.array([100.0, 100.5, 100.2, 100.9, 100.4], dtype=float)
    preds = np.array([2, 0, 2, 0, 2], dtype=int)

    gross = EnsembleTrainer._strategy_returns(close, preds)
    net, turnover = trainer._strategy_returns_after_costs(close, preds)

    assert turnover > 0.0
    assert float(np.sum(net)) < float(np.sum(gross))
