from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from bn_ml.symbols import symbol_to_model_key
from bn_ml.domain_types import Signal


class MLEnsemblePredictor:
    LABEL_TO_ACTION = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def __init__(self, model_dir: str = "models", missing_model_callback: Callable[[str], None] | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.logger = logging.getLogger("bn_ml.predictor")
        self._model_cache: dict[str, dict] = {}
        self._missing_model_callback = missing_model_callback
        self._missing_reported: set[str] = set()

    def _load_model_bundle(self, directory: Path) -> dict:
        models = {}
        for name in ["rf", "xgb", "lgb", "lstm"]:
            path = directory / f"{name}.joblib"
            if path.exists():
                try:
                    models[name] = joblib.load(path)
                except Exception as exc:
                    self.logger.warning("Skipping model %s due to load error: %s", path, exc)

        metadata = {}
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.warning("Skipping metadata %s due to parse error: %s", meta_path, exc)

        return {"models": models, "metadata": metadata}

    def _bundle_for_symbol(self, symbol: str) -> dict:
        key = symbol_to_model_key(symbol)
        if key in self._model_cache:
            return self._model_cache[key]

        symbol_dir = self.model_dir / key
        if symbol_dir.exists():
            bundle = self._load_model_bundle(symbol_dir)
            if bundle["models"]:
                self._missing_reported.discard(symbol)
                self._model_cache[key] = bundle
                return bundle

        self.logger.warning("No trained model bundle found for %s (%s). Falling back to rule-based signal.", symbol, key)
        self._model_cache[key] = {"models": {}, "metadata": {}}
        if symbol not in self._missing_reported and self._missing_model_callback is not None:
            self._missing_reported.add(symbol)
            try:
                self._missing_model_callback(symbol)
            except Exception as exc:
                self.logger.warning("missing_model_callback failed for %s: %s", symbol, exc)
        return self._model_cache[key]

    def predict(self, symbol: str, frame: pd.DataFrame, feature_columns: list[str]) -> Signal:
        bundle = self._bundle_for_symbol(symbol)
        models = bundle.get("models", {})
        metadata = bundle.get("metadata", {})

        if frame.empty:
            return self._fallback_rule_based(symbol, frame=pd.DataFrame([{}]))

        trained_features = metadata.get("feature_columns", [])
        active_features = trained_features if trained_features else feature_columns
        row = self._build_row(frame=frame, features=active_features)

        if not models or row is None:
            return self._fallback_rule_based(symbol, frame)

        ensemble_weights = self._resolve_ensemble_weights(models=models, metadata=metadata)
        aggregated_proba = np.zeros(3, dtype=float)
        model_votes: dict[str, int] = {}

        for name, model in models.items():
            model_input = self._model_input(name=name, frame=frame, features=active_features, row=row)
            if model_input is None:
                continue
            pred = int(self._predict_latest(name=name, model=model, model_input=model_input))
            model_votes[name] = pred
            model_proba = self._model_proba(name=name, model=model, model_input=model_input, pred=pred)
            aggregated_proba += ensemble_weights.get(name, 0.0) * model_proba

        if float(np.sum(aggregated_proba)) <= 1e-12:
            return self._fallback_rule_based(symbol, frame)
        aggregated_proba = aggregated_proba / float(np.sum(aggregated_proba))

        buy_p = float(aggregated_proba[2])
        sell_p = float(aggregated_proba[0])
        hold_p = float(aggregated_proba[1])

        decision_cfg = metadata.get("decision_params", {})
        min_buy_proba = float(decision_cfg.get("min_buy_proba", 0.42))
        min_sell_proba = float(decision_cfg.get("min_sell_proba", 0.42))
        min_margin = float(decision_cfg.get("min_margin", 0.07))
        hold_bias = float(decision_cfg.get("hold_bias", 0.02))

        sorted_probs = np.sort(aggregated_proba)
        margin = float(sorted_probs[-1] - sorted_probs[-2]) if sorted_probs.size >= 2 else 0.0

        if buy_p >= min_buy_proba and buy_p >= sell_p + hold_bias and margin >= min_margin:
            action_idx = 2
        elif sell_p >= min_sell_proba and sell_p >= buy_p + hold_bias and margin >= min_margin:
            action_idx = 0
        else:
            action_idx = 1

        confidence = float(np.max(aggregated_proba) * 100.0)
        strength = float(min(max(confidence, 0), 100))

        return Signal(
            symbol=symbol,
            action=self.LABEL_TO_ACTION[action_idx],
            confidence=confidence,
            strength=strength,
            metadata={
                "proba_sell": sell_p,
                "proba_hold": hold_p,
                "proba_buy": buy_p,
                "margin": margin,
                "ensemble_weights": ensemble_weights,
                "model_votes": model_votes,
            },
        )

    @staticmethod
    def _build_row(frame: pd.DataFrame, features: list[str]) -> np.ndarray | None:
        if not features:
            return None
        latest = frame.iloc[-1]
        values = []
        for name in features:
            value = latest.get(name, 0.0)
            if pd.isna(value):
                value = 0.0
            values.append(float(value))
        arr = np.asarray(values, dtype=float).reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[1] == 0:
            return None
        return arr

    @staticmethod
    def _build_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray | None:
        if frame.empty or not features:
            return None
        cols = []
        for name in features:
            if name in frame.columns:
                cols.append(pd.to_numeric(frame[name], errors="coerce").fillna(0.0))
            else:
                cols.append(pd.Series([0.0] * len(frame), index=frame.index))
        if not cols:
            return None
        mat = np.column_stack([c.to_numpy(dtype=float) for c in cols])
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _model_input(cls, name: str, frame: pd.DataFrame, features: list[str], row: np.ndarray | None) -> np.ndarray | None:
        if name == "lstm":
            return cls._build_matrix(frame=frame, features=features)
        return row

    @staticmethod
    def _predict_latest(name: str, model, model_input: np.ndarray) -> int:
        if name == "xgb":
            xgb_proba = MLEnsemblePredictor._xgb_cuda_predict_proba(model=model, model_input=model_input)
            if xgb_proba is not None and xgb_proba.size > 0:
                return int(np.argmax(xgb_proba))
        preds = np.asarray(model.predict(model_input))
        if preds.size == 0:
            return 1
        return int(preds[-1])

    @staticmethod
    def _resolve_ensemble_weights(models: dict[str, object], metadata: dict) -> dict[str, float]:
        raw = metadata.get("ensemble_weights", {})
        weights: dict[str, float] = {}
        for name in models:
            if isinstance(raw, dict):
                try:
                    weights[name] = max(0.0, float(raw.get(name, 1.0)))
                except (TypeError, ValueError):
                    weights[name] = 1.0
            else:
                weights[name] = 1.0
        total = float(sum(weights.values()))
        if total <= 1e-12:
            equal = 1.0 / max(len(weights), 1)
            return {name: equal for name in weights}
        return {name: float(weight / total) for name, weight in weights.items()}

    @staticmethod
    def _model_proba(name: str, model, model_input: np.ndarray, pred: int) -> np.ndarray:
        one_hot = np.zeros(3, dtype=float)
        one_hot[int(np.clip(pred, 0, 2))] = 1.0

        if name == "xgb":
            xgb_proba = MLEnsemblePredictor._xgb_cuda_predict_proba(model=model, model_input=model_input)
            if xgb_proba is not None and xgb_proba.size > 0:
                return xgb_proba

        if not hasattr(model, "predict_proba"):
            return one_hot

        try:
            all_rows = np.asarray(model.predict_proba(model_input), dtype=float)
            raw = np.asarray(all_rows[-1], dtype=float) if all_rows.ndim == 2 else np.asarray(all_rows, dtype=float)
        except Exception:
            return one_hot
        if raw.size == 0:
            return one_hot

        aligned = MLEnsemblePredictor._align_class_proba(raw=raw, classes=getattr(model, "classes_", None))

        total = float(np.sum(aligned))
        if total <= 1e-12:
            return one_hot
        return aligned / total

    @staticmethod
    def _align_class_proba(raw: np.ndarray, classes) -> np.ndarray:
        aligned = np.zeros(3, dtype=float)
        if classes is not None and len(classes) == len(raw):
            for idx, cls in enumerate(classes):
                try:
                    cls_idx = int(cls)
                except (TypeError, ValueError):
                    continue
                if 0 <= cls_idx <= 2:
                    aligned[cls_idx] = float(raw[idx])
            return aligned

        take = min(3, raw.size)
        aligned[:take] = raw[:take]
        return aligned

    @staticmethod
    def _xgb_cuda_predict_proba(model, model_input: np.ndarray) -> np.ndarray | None:
        # Avoid XGBoost CPU<->GPU mismatch warnings by using explicit DMatrix path
        # when the booster runs on CUDA and features come from numpy on CPU.
        if not hasattr(model, "get_booster") or not hasattr(model, "get_xgb_params"):
            return None
        try:
            params = model.get_xgb_params() or {}
            device = str(params.get("device", "")).lower()
        except Exception:
            return None
        if not device.startswith("cuda"):
            return None
        try:
            from xgboost import DMatrix
            booster = model.get_booster()
            raw_all = np.asarray(booster.predict(DMatrix(model_input)), dtype=float)
            raw = np.asarray(raw_all[-1], dtype=float) if raw_all.ndim == 2 else np.asarray(raw_all, dtype=float)
            if raw.size == 0:
                return None
            aligned = MLEnsemblePredictor._align_class_proba(raw=raw, classes=getattr(model, "classes_", None))
            total = float(np.sum(aligned))
            if total <= 1e-12:
                return None
            return aligned / total
        except Exception:
            return None

    @staticmethod
    def _fallback_rule_based(symbol: str, frame: pd.DataFrame) -> Signal:
        rsi = float(frame.iloc[-1].get("rsi_14", 50))
        macd_hist = float(frame.iloc[-1].get("macd_hist", 0))

        if rsi < 35 and macd_hist > 0:
            action = "BUY"
            confidence = 68.0
        elif rsi > 70 and macd_hist < 0:
            action = "SELL"
            confidence = 66.0
        else:
            action = "HOLD"
            confidence = 55.0

        strength = min(max(confidence, 0), 100)
        return Signal(symbol=symbol, action=action, confidence=confidence, strength=strength)
