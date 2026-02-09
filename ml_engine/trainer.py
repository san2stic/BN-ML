from __future__ import annotations

from dataclasses import dataclass
import itertools
import logging
import os
from pathlib import Path
from typing import Any
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score

from bn_ml.hardware import resolve_xgb_device

logger = logging.getLogger("bn_ml.trainer")


# ---------------------------------------------------------------------------
# Purged time-series cross-validation
# ---------------------------------------------------------------------------

class PurgedTimeSeriesSplit:
    """TimeSeriesSplit with a gap (purge) between train and validation sets
    to prevent label leakage from forward-looking targets."""

    def __init__(self, n_splits: int = 3, purge_gap: int = 4) -> None:
        self.n_splits = max(2, n_splits)
        self.purge_gap = max(0, purge_gap)

    def split(self, X, y=None, groups=None):
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.purge_gap
            val_end = min(val_start + fold_size, n_samples)

            if val_start >= n_samples or val_start >= val_end:
                continue

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    models: dict
    metrics: dict[str, float]
    selected_features: list[str]
    best_params: dict[str, dict[str, Any]]
    ensemble_weights: dict[str, float]
    validation_metrics: dict[str, dict[str, float]]
    decision_params: dict[str, float]


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class EnsembleTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_cfg = config.get("model", {})
        self.hpo_cfg = self.model_cfg.get("hpo", {})
        self.random_state = int(self.model_cfg.get("random_state", 42))
        self.accel_cfg = self.model_cfg.get("acceleration", {})
        self.cpu_n_jobs = int(self.accel_cfg.get("cpu_n_jobs", -1))
        self.allow_cuda_fallback = bool(self.accel_cfg.get("allow_cuda_fallback", True))
        self.xgb_device, self.xgb_device_reason = resolve_xgb_device(str(self.accel_cfg.get("mode", "auto")))
        self.ensemble_cfg = self.model_cfg.get("ensemble", {})
        self._purge_gap = max(0, int(self.model_cfg.get("labeling", {}).get("horizon_candles", 4)))

        risk_cfg = config.get("risk", {})
        fees_pct = max(0.0, float(risk_cfg.get("fees_pct", 0.10)))
        slippage_pct = max(0.0, float(risk_cfg.get("slippage_pct", 0.05)))
        execution_cfg = self.model_cfg.get("execution", {})
        cost_multiplier = max(0.0, float(execution_cfg.get("cost_multiplier", 1.0)))
        self.trade_cost_per_turnover = ((fees_pct + slippage_pct) / 100.0) * cost_multiplier

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(self, df: pd.DataFrame, features: list[str], target_col: str = "label") -> TrainingResult:
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")
        if len(df) < 200:
            raise ValueError("Not enough samples for robust time-series training (<200 rows)")
        if df[target_col].nunique() < 2:
            raise ValueError("Need at least 2 classes in target for training")

        selected_features = self._select_features(df=df, feature_cols=features, target_col=target_col)
        x = df[selected_features].values
        y = df[target_col].values
        close = df["close"].values

        best_params: dict[str, dict[str, Any]] = {}
        metrics: dict[str, float] = {}
        metrics["xgb_device_cuda"] = 1.0 if self.xgb_device == "cuda" else 0.0

        # --- Train RF ---
        rf_params = self._rf_base_params()
        if self._hpo_enabled():
            best_rf_params, rf_hpo_metrics = self._optimize_model(model_kind="rf", x=x, y=y, close=close)
            if best_rf_params:
                rf_params.update(best_rf_params)
                best_params["rf"] = rf_params
            metrics.update({f"rf_hpo_{k}": v for k, v in rf_hpo_metrics.items()})

        rf = self._build_rf(rf_params)
        rf_cv_acc = self._cross_val_accuracy(rf, x, y)
        rf.fit(x, y)

        models: dict[str, Any] = {"rf": rf}
        metrics["rf_cv_accuracy"] = rf_cv_acc

        # --- Train XGBoost ---
        xgb_model, xgb_params, xgb_extra_metrics = self._train_boosting_with_optional_hpo(
            model_kind="xgb", x=x, y=y, close=close,
        )
        if xgb_model is not None:
            models["xgb"] = xgb_model
            if xgb_params:
                best_params["xgb"] = xgb_params
            self._collect_train_metrics(metrics, "xgb", xgb_model, x, y, close)
            metrics.update(xgb_extra_metrics)

        # --- Train LightGBM ---
        lgb_model, lgb_params, lgb_extra_metrics = self._train_boosting_with_optional_hpo(
            model_kind="lgb", x=x, y=y, close=close,
        )
        if lgb_model is not None:
            models["lgb"] = lgb_model
            if lgb_params:
                best_params["lgb"] = lgb_params
            self._collect_train_metrics(metrics, "lgb", lgb_model, x, y, close)
            metrics.update(lgb_extra_metrics)

        # --- RF train metrics ---
        self._collect_train_metrics(metrics, "rf", rf, x, y, close)

        # --- Validation & ensemble ---
        split_idx = self._validation_split_idx(len(x))
        x_valid, y_valid = x[split_idx:], y[split_idx:]

        # Calibrate models if configured
        calibration = str(self.ensemble_cfg.get("calibration", "none")).lower()
        if calibration in ("isotonic", "sigmoid") and len(x_valid) >= 30:
            models = self._calibrate_models(models, x_valid, y_valid, method=calibration)
            metrics["calibration_applied"] = 1.0
        else:
            metrics["calibration_applied"] = 0.0

        validation_metrics = self._model_validation_metrics(models=models, x=x, y=y, close=close)
        ensemble_weights = self._build_ensemble_weights(validation_metrics)
        for model_name, weight in ensemble_weights.items():
            metrics[f"ensemble_weight_{model_name}"] = float(weight)
        for model_name, stats in validation_metrics.items():
            for k, v in stats.items():
                metrics[f"{model_name}_valid_{k}"] = float(v)

        return TrainingResult(
            models=models,
            metrics=metrics,
            selected_features=selected_features,
            best_params=best_params,
            ensemble_weights=ensemble_weights,
            validation_metrics=validation_metrics,
            decision_params=self._decision_params(),
        )

    def save_models(self, models: dict, out_dir: str = "models") -> list[Path]:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        for name, model in models.items():
            model_path = path / f"{name}.joblib"
            tmp_path = path / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            joblib.dump(model, tmp_path)
            tmp_path.replace(model_path)
            saved_paths.append(model_path)
        return saved_paths

    # -----------------------------------------------------------------------
    # Collect train metrics for any model
    # -----------------------------------------------------------------------

    def _collect_train_metrics(self, metrics: dict, prefix: str, model, x, y, close):
        preds = model.predict(x)
        net_returns, turnover = self._strategy_returns_after_costs(close, preds)
        sharpe, sortino = self._risk_metrics(net_returns)
        metrics[f"{prefix}_train_accuracy"] = float(accuracy_score(y, preds))
        metrics[f"{prefix}_train_f1_macro"] = float(f1_score(y, preds, average="macro"))
        metrics[f"{prefix}_train_net_return"] = self._total_return(net_returns)
        metrics[f"{prefix}_train_max_drawdown"] = self._max_drawdown(net_returns)
        metrics[f"{prefix}_train_turnover"] = turnover
        metrics[f"{prefix}_train_sharpe_net"] = sharpe
        metrics[f"{prefix}_train_sortino_net"] = sortino

    # -----------------------------------------------------------------------
    # HPO
    # -----------------------------------------------------------------------

    def _hpo_enabled(self) -> bool:
        return bool(self.hpo_cfg.get("enabled", True))

    def _hpo_backend(self) -> str:
        return str(self.hpo_cfg.get("backend", "grid")).lower()

    def _hpo_view(self, x: np.ndarray, y: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_samples = int(self.hpo_cfg.get("max_samples", 1200))
        if len(x) <= max_samples:
            return x, y, close
        return x[-max_samples:], y[-max_samples:], close[-max_samples:]

    def _optimize_model(
        self,
        model_kind: str,
        x: np.ndarray,
        y: np.ndarray,
        close: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        backend = self._hpo_backend()
        if backend == "optuna":
            try:
                return self._optimize_model_optuna(model_kind=model_kind, x=x, y=y, close=close)
            except ImportError:
                logger.warning("Optuna not installed, falling back to grid HPO")
        return self._optimize_model_grid(model_kind=model_kind, x=x, y=y, close=close)

    def _optimize_model_optuna(
        self,
        model_kind: str,
        x: np.ndarray,
        y: np.ndarray,
        close: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        x_opt, y_opt, close_opt = self._hpo_view(x, y, close)
        trial_key = f"{model_kind}_trials"
        default_trials = {"rf": 20, "xgb": 15, "lgb": 15}.get(model_kind, 10)
        n_trials = int(self.hpo_cfg.get(trial_key, default_trials))

        def objective(trial):
            params = self._suggest_params(trial, model_kind)
            eval_metrics = self._evaluate_candidate(
                model_kind=model_kind, params=params, x=x_opt, y=y_opt, close=close_opt,
            )
            return float(eval_metrics.get("objective", -1e18))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_metrics = self._evaluate_candidate(
            model_kind=model_kind, params=best_params, x=x_opt, y=y_opt, close=close_opt,
        )
        return best_params, best_metrics

    def _suggest_params(self, trial, model_kind: str) -> dict[str, Any]:
        if model_kind == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100, step=10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.6, 0.8]),
            }
        if model_kind == "xgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "early_stopping_rounds": 40,
            }
        if model_kind == "lgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60, step=5),
            }
        raise ValueError(f"Unknown model kind: {model_kind}")

    def _optimize_model_grid(
        self,
        model_kind: str,
        x: np.ndarray,
        y: np.ndarray,
        close: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        x_opt, y_opt, close_opt = self._hpo_view(x, y, close)

        if model_kind == "rf":
            candidates = self._rf_candidates()
            trial_budget = int(self.hpo_cfg.get("rf_trials", 8))
        elif model_kind == "xgb":
            candidates = self._xgb_candidates()
            trial_budget = int(self.hpo_cfg.get("xgb_trials", 6))
        elif model_kind == "lgb":
            candidates = self._lgb_candidates()
            trial_budget = int(self.hpo_cfg.get("lgb_trials", 6))
        else:
            raise ValueError(f"Unknown model kind: {model_kind}")

        if not candidates:
            return {}, {}

        candidates = self._sample_candidates(candidates, trial_budget)

        best_params: dict[str, Any] = {}
        best_metrics: dict[str, float] = {}
        best_objective = -1e18

        for params in candidates:
            eval_metrics = self._evaluate_candidate(
                model_kind=model_kind, params=params, x=x_opt, y=y_opt, close=close_opt,
            )
            objective = float(eval_metrics.get("objective", -1e18))
            if objective > best_objective:
                best_objective = objective
                best_params = params
                best_metrics = eval_metrics

        return best_params, best_metrics

    # -----------------------------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------------------------

    def _make_cv(self, n_samples: int, max_splits: int | None = None) -> PurgedTimeSeriesSplit:
        n_splits = self._time_splits(n_samples)
        if max_splits is not None:
            n_splits = min(n_splits, max_splits)
        return PurgedTimeSeriesSplit(n_splits=max(2, n_splits), purge_gap=self._purge_gap)

    def _cross_val_accuracy(self, model, x: np.ndarray, y: np.ndarray) -> float:
        tscv = self._make_cv(len(x))
        scores: list[float] = []
        for train_idx, valid_idx in tscv.split(x):
            model.fit(x[train_idx], y[train_idx])
            preds = model.predict(x[valid_idx])
            scores.append(accuracy_score(y[valid_idx], preds))
        return float(np.mean(scores)) if scores else 0.0

    def _evaluate_candidate(
        self,
        model_kind: str,
        params: dict[str, Any],
        x: np.ndarray,
        y: np.ndarray,
        close: np.ndarray,
    ) -> dict[str, float]:
        max_splits = int(self.hpo_cfg.get("max_splits", 5))
        tscv = self._make_cv(len(x), max_splits=max_splits)

        accuracies: list[float] = []
        all_returns: list[np.ndarray] = []
        fold_turnovers: list[float] = []
        fold_net_returns: list[float] = []
        fold_drawdowns: list[float] = []

        for train_idx, valid_idx in tscv.split(x):
            x_train, x_valid = x[train_idx], x[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            close_valid = close[valid_idx]

            if model_kind == "xgb":
                model, _ = self._fit_xgb_with_fallback(
                    params=params, x_train=x_train, y_train=y_train, use_early_stopping=False,
                )
            elif model_kind == "lgb":
                model = self._build_lgb(params)
                if model is None:
                    continue
                model.fit(x_train, y_train)
            else:
                model = self._build_model(model_kind=model_kind, params=params)
                model.fit(x_train, y_train)
            preds = model.predict(x_valid)

            accuracies.append(float(accuracy_score(y_valid, preds)))
            net_returns, turnover = self._strategy_returns_after_costs(close_valid, preds)
            all_returns.append(net_returns)
            fold_turnovers.append(turnover)
            fold_net_returns.append(self._total_return(net_returns))
            fold_drawdowns.append(self._max_drawdown(net_returns))

        mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
        returns = np.concatenate(all_returns) if all_returns else np.array([])
        sharpe, sortino = self._risk_metrics(returns)
        mean_turnover = float(np.mean(fold_turnovers)) if fold_turnovers else 0.0
        mean_net_return = float(np.mean(fold_net_returns)) if fold_net_returns else 0.0
        mean_drawdown = float(np.mean(fold_drawdowns)) if fold_drawdowns else 0.0

        w_sharpe = float(self.hpo_cfg.get("objective_weight_sharpe", 0.45))
        w_sortino = float(self.hpo_cfg.get("objective_weight_sortino", 0.45))
        w_acc = float(self.hpo_cfg.get("objective_weight_accuracy", 0.10))
        w_return = float(self.hpo_cfg.get("objective_weight_return", 0.20))
        w_drawdown = float(self.hpo_cfg.get("objective_weight_drawdown", 0.10))
        w_turnover = float(self.hpo_cfg.get("objective_weight_turnover", 0.05))

        return_score = float(np.clip(mean_net_return * 40.0, -5.0, 5.0))
        drawdown_score = float(np.clip(mean_drawdown * 40.0, 0.0, 5.0))
        turnover_score = float(np.clip(mean_turnover * 2.0, 0.0, 5.0))

        objective = (
            w_sharpe * sharpe
            + w_sortino * sortino
            + w_acc * mean_acc
            + w_return * return_score
            - w_drawdown * drawdown_score
            - w_turnover * turnover_score
        )

        return {
            "objective": float(objective),
            "walkforward_accuracy": mean_acc,
            "walkforward_sharpe": float(sharpe),
            "walkforward_sortino": float(sortino),
            "walkforward_net_return": mean_net_return,
            "walkforward_max_drawdown": mean_drawdown,
            "walkforward_turnover": mean_turnover,
            "walkforward_return_score": return_score,
            "walkforward_drawdown_score": drawdown_score,
            "walkforward_turnover_score": turnover_score,
        }

    # -----------------------------------------------------------------------
    # Model builders
    # -----------------------------------------------------------------------

    def _build_model(self, model_kind: str, params: dict[str, Any]):
        if model_kind == "rf":
            return self._build_rf(params)
        if model_kind == "xgb":
            return self._build_xgb(params, use_early_stopping=False)
        if model_kind == "lgb":
            m = self._build_lgb(params)
            if m is None:
                raise ValueError("LightGBM is not available in this environment")
            return m
        raise ValueError(f"Unknown model kind: {model_kind}")

    def _train_boosting_with_optional_hpo(
        self,
        model_kind: str,
        x: np.ndarray,
        y: np.ndarray,
        close: np.ndarray,
    ) -> tuple[Any | None, dict[str, Any] | None, dict[str, float]]:
        # Check availability
        if model_kind == "xgb" and self._build_xgb({}, use_early_stopping=True) is None:
            return None, None, {}
        if model_kind == "lgb":
            lgb_cfg = self.model_cfg.get("lgb", {})
            if not bool(lgb_cfg.get("enabled", True)):
                return None, None, {}
            if self._build_lgb({}) is None:
                return None, None, {}

        base_params_fn = {"xgb": self._xgb_base_params, "lgb": self._lgb_base_params}
        params = base_params_fn.get(model_kind, self._xgb_base_params)()
        metrics: dict[str, float] = {}

        if self._hpo_enabled():
            best_params, hpo_metrics = self._optimize_model(model_kind=model_kind, x=x, y=y, close=close)
            if best_params:
                params.update(best_params)
            metrics.update({f"{model_kind}_hpo_{k}": v for k, v in hpo_metrics.items()})

        split_idx = self._validation_split_idx(len(x))
        use_early_stopping = model_kind == "xgb" and split_idx > 80 and (len(x) - split_idx) > 80

        if model_kind == "xgb":
            if use_early_stopping:
                x_train, x_valid = x[:split_idx], x[split_idx:]
                y_train, y_valid = y[:split_idx], y[split_idx:]
                model, used_device = self._fit_xgb_with_fallback(
                    params=params, x_train=x_train, y_train=y_train,
                    eval_set=[(x_valid, y_valid)], use_early_stopping=True,
                )
                best_iter = getattr(model, "best_iteration", None)
                if best_iter is not None:
                    metrics["xgb_best_iteration"] = float(best_iter)
            else:
                model, used_device = self._fit_xgb_with_fallback(
                    params=params, x_train=x, y_train=y, use_early_stopping=False,
                )
            metrics["xgb_device_used_cuda"] = 1.0 if used_device == "cuda" else 0.0
        elif model_kind == "lgb":
            model = self._build_lgb(params)
            if model is None:
                return None, None, {}
            model.fit(x, y)
        else:
            return None, None, {}

        return model, params, metrics

    # -----------------------------------------------------------------------
    # Feature selection (permutation importance + correlation pruning)
    # -----------------------------------------------------------------------

    def _select_features(self, df: pd.DataFrame, feature_cols: list[str], target_col: str) -> list[str]:
        feature_limit = int(self.model_cfg.get("feature_limit", 50))
        if len(feature_cols) <= feature_limit:
            return feature_cols

        x_probe = df[feature_cols].values
        y_probe = df[target_col].values

        rf_probe = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=self.random_state,
            n_jobs=self.cpu_n_jobs,
            class_weight="balanced_subsample",
        )
        rf_probe.fit(x_probe, y_probe)

        # Permutation importance (more reliable than Gini)
        perm_result = permutation_importance(
            rf_probe, x_probe, y_probe,
            n_repeats=5, random_state=self.random_state, n_jobs=self.cpu_n_jobs,
        )
        importances = list(zip(feature_cols, perm_result.importances_mean))
        importances.sort(key=lambda item: item[1], reverse=True)

        # Take top candidates (slightly more than limit to allow correlation pruning)
        top_candidates = [name for name, _ in importances[: feature_limit + 10]]

        # Remove highly correlated features (>0.95)
        selected = self._prune_correlated(df, top_candidates, threshold=0.95)
        return selected[:feature_limit]

    @staticmethod
    def _prune_correlated(df: pd.DataFrame, features: list[str], threshold: float = 0.95) -> list[str]:
        if len(features) <= 1:
            return features
        corr = df[features].corr().abs()
        to_drop: set[str] = set()
        for i in range(len(features)):
            if features[i] in to_drop:
                continue
            for j in range(i + 1, len(features)):
                if features[j] in to_drop:
                    continue
                if corr.iloc[i, j] > threshold:
                    to_drop.add(features[j])
        return [f for f in features if f not in to_drop]

    # -----------------------------------------------------------------------
    # Probability calibration
    # -----------------------------------------------------------------------

    @staticmethod
    def _calibrate_models(models: dict, x_valid: np.ndarray, y_valid: np.ndarray, method: str = "isotonic") -> dict:
        from sklearn.calibration import CalibratedClassifierCV

        calibrated = {}
        for name, model in models.items():
            try:
                cal = CalibratedClassifierCV(model, method=method, cv="prefit")
                cal.fit(x_valid, y_valid)
                calibrated[name] = cal
            except Exception:
                calibrated[name] = model
        return calibrated

    # -----------------------------------------------------------------------
    # Model base params
    # -----------------------------------------------------------------------

    def _rf_base_params(self) -> dict[str, Any]:
        rf_cfg = self.model_cfg.get("rf", {})
        return {
            "n_estimators": int(rf_cfg.get("n_estimators", 300)),
            "max_depth": int(rf_cfg.get("max_depth", 8)),
            "min_samples_leaf": int(rf_cfg.get("min_samples_leaf", 50)),
            "max_features": rf_cfg.get("max_features", "sqrt"),
        }

    def _xgb_base_params(self) -> dict[str, Any]:
        xgb_cfg = self.model_cfg.get("xgb", {})
        return {
            "n_estimators": int(xgb_cfg.get("n_estimators", 500)),
            "max_depth": int(xgb_cfg.get("max_depth", 6)),
            "learning_rate": float(xgb_cfg.get("learning_rate", 0.03)),
            "subsample": float(xgb_cfg.get("subsample", 0.9)),
            "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.8)),
            "reg_alpha": float(xgb_cfg.get("reg_alpha", 0.01)),
            "reg_lambda": float(xgb_cfg.get("reg_lambda", 1.0)),
            "early_stopping_rounds": int(xgb_cfg.get("early_stopping_rounds", 40)),
        }

    def _lgb_base_params(self) -> dict[str, Any]:
        lgb_cfg = self.model_cfg.get("lgb", {})
        return {
            "n_estimators": int(lgb_cfg.get("n_estimators", 400)),
            "max_depth": int(lgb_cfg.get("max_depth", -1)),
            "num_leaves": int(lgb_cfg.get("num_leaves", 31)),
            "learning_rate": float(lgb_cfg.get("learning_rate", 0.03)),
            "subsample": float(lgb_cfg.get("subsample", 0.8)),
            "colsample_bytree": float(lgb_cfg.get("colsample_bytree", 0.8)),
            "reg_alpha": float(lgb_cfg.get("reg_alpha", 0.01)),
            "reg_lambda": float(lgb_cfg.get("reg_lambda", 1.0)),
            "min_child_samples": int(lgb_cfg.get("min_child_samples", 30)),
        }

    # -----------------------------------------------------------------------
    # Model constructors
    # -----------------------------------------------------------------------

    def _build_rf(self, params: dict[str, Any]):
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 8)),
            min_samples_leaf=int(params.get("min_samples_leaf", 50)),
            max_features=params.get("max_features", "sqrt"),
            random_state=self.random_state,
            n_jobs=self.cpu_n_jobs,
            class_weight="balanced_subsample",
        )

    def _build_xgb(self, params: dict[str, Any], use_early_stopping: bool, device_override: str | None = None):
        try:
            from xgboost import XGBClassifier
        except Exception:
            return None

        device = device_override or self.xgb_device
        kwargs: dict[str, Any] = {
            "n_estimators": int(params.get("n_estimators", 500)),
            "max_depth": int(params.get("max_depth", 6)),
            "learning_rate": float(params.get("learning_rate", 0.03)),
            "subsample": float(params.get("subsample", 0.9)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
            "reg_alpha": float(params.get("reg_alpha", 0.01)),
            "reg_lambda": float(params.get("reg_lambda", 1.0)),
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": device,
            "random_state": self.random_state,
        }
        if use_early_stopping:
            kwargs["early_stopping_rounds"] = int(params.get("early_stopping_rounds", 40))
        return XGBClassifier(**kwargs)

    def _build_lgb(self, params: dict[str, Any]):
        try:
            from lightgbm import LGBMClassifier
        except Exception:
            return None

        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth", -1)),
            num_leaves=int(params.get("num_leaves", 31)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.01)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            min_child_samples=int(params.get("min_child_samples", 30)),
            objective="multiclass",
            num_class=3,
            random_state=self.random_state,
            n_jobs=self.cpu_n_jobs,
            verbose=-1,
        )

    def _fit_xgb_with_fallback(
        self,
        params: dict[str, Any],
        x_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        use_early_stopping: bool = False,
    ) -> tuple[Any, str]:
        desired_device = self.xgb_device
        model = self._build_xgb(params, use_early_stopping=use_early_stopping, device_override=desired_device)
        if model is None:
            raise ValueError("XGBoost is not available in this environment")

        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None and use_early_stopping:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False

        try:
            model.fit(x_train, y_train, **fit_kwargs)
            return model, desired_device
        except Exception:
            if desired_device == "cuda" and self.allow_cuda_fallback:
                cpu_model = self._build_xgb(params, use_early_stopping=use_early_stopping, device_override="cpu")
                if cpu_model is None:
                    raise
                cpu_model.fit(x_train, y_train, **fit_kwargs)
                return cpu_model, "cpu"
            raise

    # -----------------------------------------------------------------------
    # Grid HPO candidates (legacy / fallback)
    # -----------------------------------------------------------------------

    def _rf_candidates(self) -> list[dict[str, Any]]:
        base = self._rf_base_params()
        n_estimators = sorted({max(100, int(base["n_estimators"] * 0.7)), int(base["n_estimators"]), int(base["n_estimators"] * 1.3)})
        max_depth = sorted({max(3, int(base["max_depth"]) - 2), int(base["max_depth"]), int(base["max_depth"]) + 2})
        min_leaf = sorted({max(5, int(base["min_samples_leaf"]) // 2), int(base["min_samples_leaf"]), int(base["min_samples_leaf"]) * 2})
        max_features = [base.get("max_features", "sqrt"), 0.8]

        candidates = []
        for ne, md, ml, mf in itertools.product(n_estimators, max_depth, min_leaf, max_features):
            candidates.append({
                "n_estimators": int(ne), "max_depth": int(md),
                "min_samples_leaf": int(ml), "max_features": mf,
            })
        return candidates

    def _xgb_candidates(self) -> list[dict[str, Any]]:
        base = self._xgb_base_params()
        n_estimators = sorted({max(120, int(base["n_estimators"] * 0.35)), max(180, int(base["n_estimators"] * 0.60)), max(250, int(base["n_estimators"] * 0.85))})
        max_depth = sorted({max(3, int(base["max_depth"]) - 1), int(base["max_depth"])})
        learning_rate = sorted({max(0.01, base["learning_rate"] * 0.7), base["learning_rate"], min(0.2, base["learning_rate"] * 1.3)})
        reg_alpha = sorted({0.0, base["reg_alpha"], base["reg_alpha"] * 5})

        candidates = []
        for ne, md, lr, ra in itertools.product(n_estimators, max_depth, learning_rate, reg_alpha):
            candidates.append({
                "n_estimators": int(ne), "max_depth": int(md), "learning_rate": float(lr),
                "subsample": float(base["subsample"]), "colsample_bytree": float(base["colsample_bytree"]),
                "reg_alpha": float(ra), "reg_lambda": float(base["reg_lambda"]),
                "early_stopping_rounds": int(base["early_stopping_rounds"]),
            })
        return candidates

    def _lgb_candidates(self) -> list[dict[str, Any]]:
        base = self._lgb_base_params()
        n_estimators = sorted({max(150, int(base["n_estimators"] * 0.6)), int(base["n_estimators"]), int(base["n_estimators"] * 1.3)})
        num_leaves = sorted({max(15, base["num_leaves"] // 2), base["num_leaves"], min(63, base["num_leaves"] * 2)})
        learning_rate = sorted({max(0.01, base["learning_rate"] * 0.7), base["learning_rate"], min(0.2, base["learning_rate"] * 1.3)})

        candidates = []
        for ne, nl, lr in itertools.product(n_estimators, num_leaves, learning_rate):
            p = dict(base)
            p.update({"n_estimators": int(ne), "num_leaves": int(nl), "learning_rate": float(lr)})
            candidates.append(p)
        return candidates

    def _sample_candidates(self, candidates: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        if budget <= 0 or budget >= len(candidates):
            return candidates
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(candidates), size=budget, replace=False)
        return [candidates[int(i)] for i in idx]

    # -----------------------------------------------------------------------
    # Strategy metrics
    # -----------------------------------------------------------------------

    @staticmethod
    def _strategy_returns(close: np.ndarray, preds: np.ndarray) -> np.ndarray:
        if len(close) == 0:
            return np.array([], dtype=float)
        signal = np.where(preds == 2, 1, np.where(preds == 0, -1, 0)).astype(float)
        price_ret = pd.Series(close).pct_change().fillna(0.0).to_numpy(dtype=float)
        shifted_signal = np.roll(signal, 1)
        shifted_signal[0] = 0.0
        return shifted_signal * price_ret

    def _strategy_returns_after_costs(self, close: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, float]:
        if len(close) == 0:
            return np.array([], dtype=float), 0.0
        signal = np.where(preds == 2, 1, np.where(preds == 0, -1, 0)).astype(float)
        price_ret = pd.Series(close).pct_change().fillna(0.0).to_numpy(dtype=float)
        shifted_signal = np.roll(signal, 1)
        shifted_signal[0] = 0.0

        gross_returns = shifted_signal * price_ret
        turnover = np.abs(signal - shifted_signal)
        net_returns = gross_returns - turnover * self.trade_cost_per_turnover

        mean_turnover = float(np.mean(turnover)) if turnover.size else 0.0
        return net_returns, mean_turnover

    @staticmethod
    def _total_return(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        equity = np.cumprod(1.0 + returns)
        return float(equity[-1] - 1.0)

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        equity = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(equity)
        running_max = np.where(running_max <= 0, 1.0, running_max)
        drawdowns = 1.0 - (equity / running_max)
        return float(np.max(drawdowns)) if drawdowns.size else 0.0

    # -----------------------------------------------------------------------
    # Validation & ensemble
    # -----------------------------------------------------------------------

    def _validation_split_idx(self, n: int) -> int:
        split_idx = int(n * 0.8)
        if split_idx <= 80 or (n - split_idx) <= 40:
            split_idx = int(n * 0.7)
        if split_idx <= 60 or (n - split_idx) <= 30:
            split_idx = max(1, n - max(30, n // 5))
        return split_idx

    def _model_validation_metrics(
        self,
        models: dict[str, Any],
        x: np.ndarray,
        y: np.ndarray,
        close: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        if not models:
            return {}

        split_idx = self._validation_split_idx(len(x))
        x_valid = x[split_idx:]
        y_valid = y[split_idx:]
        close_valid = close[split_idx:]
        if len(x_valid) == 0:
            x_valid, y_valid, close_valid = x, y, close

        metrics: dict[str, dict[str, float]] = {}
        for model_name, model in models.items():
            preds = model.predict(x_valid)
            net_returns, turnover = self._strategy_returns_after_costs(close_valid, preds)
            sharpe, sortino = self._risk_metrics(net_returns)
            acc = float(accuracy_score(y_valid, preds))
            f1 = float(f1_score(y_valid, preds, average="macro"))
            net_return = self._total_return(net_returns)
            max_drawdown = self._max_drawdown(net_returns)
            score = self._ensemble_score(
                accuracy=acc, f1_macro=f1, sharpe=sharpe, sortino=sortino,
                net_return=net_return, max_drawdown=max_drawdown, turnover=turnover,
            )
            metrics[model_name] = {
                "accuracy": acc, "f1_macro": f1,
                "sharpe_net": float(sharpe), "sortino_net": float(sortino),
                "net_return": float(net_return), "max_drawdown": float(max_drawdown),
                "turnover": float(turnover), "ensemble_score": float(score),
            }
        return metrics

    @staticmethod
    def _ensemble_score(
        *, accuracy: float, f1_macro: float, sharpe: float, sortino: float,
        net_return: float, max_drawdown: float, turnover: float,
    ) -> float:
        sharpe_s = float(np.clip((sharpe + 2.0) / 6.0, 0.0, 1.0))
        sortino_s = float(np.clip((sortino + 2.0) / 6.0, 0.0, 1.0))
        ret_s = float(np.clip(net_return / 0.05, -1.0, 1.0))
        dd_s = float(np.clip(max_drawdown / 0.10, 0.0, 1.0))
        turn_s = float(np.clip(turnover / 1.2, 0.0, 1.0))

        score = (
            0.30 * accuracy + 0.20 * f1_macro
            + 0.15 * sharpe_s + 0.10 * sortino_s
            + 0.20 * max(ret_s, 0.0)
            - 0.10 * dd_s - 0.05 * turn_s
        )
        return float(max(score, 0.02))

    @staticmethod
    def _build_ensemble_weights(validation_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
        if not validation_metrics:
            return {}
        raw = {name: max(1e-6, float(stats.get("ensemble_score", 0.0))) for name, stats in validation_metrics.items()}
        total = float(sum(raw.values()))
        if total <= 1e-12:
            equal = 1.0 / max(len(raw), 1)
            return {name: equal for name in raw}
        return {name: float(value / total) for name, value in raw.items()}

    def _decision_params(self) -> dict[str, float]:
        return {
            "min_buy_proba": float(self.ensemble_cfg.get("min_buy_proba", 0.42)),
            "min_sell_proba": float(self.ensemble_cfg.get("min_sell_proba", 0.42)),
            "min_margin": float(self.ensemble_cfg.get("min_margin", 0.07)),
            "hold_bias": float(self.ensemble_cfg.get("hold_bias", 0.02)),
        }

    @staticmethod
    def _risk_metrics(returns: np.ndarray) -> tuple[float, float]:
        if returns.size == 0:
            return 0.0, 0.0
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))
        scale = float(np.sqrt(max(returns.size, 1)))

        sharpe = 0.0 if std_ret <= 1e-12 else (mean_ret / std_ret) * scale

        downside = returns[returns < 0]
        if downside.size == 0:
            sortino = sharpe
        else:
            downside_std = float(np.std(downside))
            sortino = sharpe if downside_std <= 1e-12 else (mean_ret / downside_std) * scale

        sharpe = float(np.clip(sharpe, -10.0, 10.0))
        sortino = float(np.clip(sortino, -10.0, 10.0))
        return sharpe, sortino

    @staticmethod
    def _time_splits(n_samples: int) -> int:
        if n_samples < 300:
            return 2
        if n_samples < 600:
            return 3
        if n_samples < 1000:
            return 4
        return 5
