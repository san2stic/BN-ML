from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bn_ml.config import load_config
from bn_ml.symbols import symbol_to_model_key
from data_manager.data_cleaner import DataCleaner
from data_manager.features_engineer import FeatureEngineer
from data_manager.fetch_data import BinanceDataManager
from data_manager.multi_timeframe import MultiTimeframeFeatureBuilder
from ml_engine.trainer import EnsembleTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BN-ML per-symbol ensemble models")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--symbol", action="append", default=[], help="Optional symbol(s) to train, can repeat")
    parser.set_defaults(train_missing_only=None)
    parser.add_argument(
        "--train-missing-only",
        dest="train_missing_only",
        action="store_true",
        help="Train only symbols with missing or stale models",
    )
    parser.add_argument(
        "--train-all",
        dest="train_missing_only",
        action="store_false",
        help="Train all candidate symbols (ignore missing-only filter)",
    )
    parser.add_argument(
        "--max-model-age-hours",
        type=float,
        default=None,
        help="When using --train-missing-only, retrain models older than this age (default from config, fallback 24h)",
    )
    parser.add_argument("--models-dir", default="models", help="Model directory root")
    return parser.parse_args()


def _label_edge_profile(config: dict, frame: pd.DataFrame) -> tuple[pd.Series, int, dict[str, float]]:
    model_cfg = config.get("model", {})
    label_cfg = model_cfg.get("labeling", {})
    risk_cfg = config.get("risk", {})

    horizon = max(1, int(label_cfg.get("horizon_candles", 4)))
    vol_window = max(8, int(label_cfg.get("volatility_window", 96)))
    vol_multiplier = max(0.0, float(label_cfg.get("volatility_multiplier", 0.6)))
    atr_multiplier = max(0.0, float(label_cfg.get("atr_multiplier", 0.35)))
    min_move_pct = max(0.0, float(label_cfg.get("min_move_pct", 0.0015)))
    max_move_pct = max(min_move_pct, float(label_cfg.get("max_move_pct", 0.02)))

    one_way_cost_pct = max(0.0, float(risk_cfg.get("fees_pct", 0.10)) + float(risk_cfg.get("slippage_pct", 0.05))) / 100.0
    round_trip_cost_multiplier = max(1.0, float(label_cfg.get("round_trip_cost_multiplier", 2.0)))
    cost_safety_multiplier = max(0.0, float(label_cfg.get("cost_safety_multiplier", 1.35)))
    min_edge_pct = max(min_move_pct, one_way_cost_pct * round_trip_cost_multiplier * cost_safety_multiplier)

    min_periods = max(5, vol_window // 4)
    realized_vol = frame["close"].pct_change().rolling(vol_window, min_periods=min_periods).std().fillna(0.0).abs()
    atr_ratio = frame.get("atr_ratio", pd.Series(0.0, index=frame.index)).fillna(0.0).abs()

    edge = (min_edge_pct + vol_multiplier * realized_vol + atr_multiplier * atr_ratio).clip(lower=min_edge_pct, upper=max_move_pct)
    info = {
        "horizon_candles": float(horizon),
        "volatility_window": float(vol_window),
        "volatility_multiplier": float(vol_multiplier),
        "atr_multiplier": float(atr_multiplier),
        "min_move_pct": float(min_move_pct),
        "max_move_pct": float(max_move_pct),
        "min_edge_pct": float(min_edge_pct),
        "one_way_cost_pct": float(one_way_cost_pct),
        "round_trip_cost_multiplier": float(round_trip_cost_multiplier),
        "cost_safety_multiplier": float(cost_safety_multiplier),
    }
    return edge, horizon, info


def build_symbol_dataset(config: dict, paper: bool, symbol: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    cleaner = DataCleaner()
    feat = FeatureEngineer()
    data = BinanceDataManager(config=config, paper=paper)
    mtf_builder = MultiTimeframeFeatureBuilder(
        config=config,
        data_manager=data,
        cleaner=cleaner,
        feature_engineer=feat,
    )

    train_limit = int(config.get("model", {}).get("train_ohlcv_limit", 1800))
    frame = mtf_builder.build(symbol=symbol, limit=train_limit)
    frame["symbol"] = symbol

    edge, horizon, label_info = _label_edge_profile(config=config, frame=frame)
    # Multi-class labels: BUY (2), HOLD (1), SELL (0)
    fwd = frame["close"].shift(-horizon) / frame["close"] - 1
    frame["label_edge"] = edge
    frame["label"] = np.where(fwd > edge, 2, np.where(fwd < -edge, 0, 1))

    ds = frame.dropna().reset_index(drop=True)
    feature_cols = feat.feature_columns(ds)
    label_info["effective_edge_mean_pct"] = float(ds["label_edge"].mean()) if not ds.empty else 0.0
    label_info["effective_edge_median_pct"] = float(ds["label_edge"].median()) if not ds.empty else 0.0
    label_info["multi_timeframe"] = mtf_builder.describe()
    return ds, feature_cols, label_info


def resolve_training_symbols(config: dict, paper: bool, symbols: list[str] | None = None) -> list[str]:
    if symbols:
        return symbols

    universe_cfg = config.get("universe", {})
    configured = list(universe_cfg.get("pairs", []))

    dynamic_enabled = bool(universe_cfg.get("dynamic_base_quote_pairs", False))
    train_dynamic = bool(universe_cfg.get("train_dynamic_pairs", dynamic_enabled))
    if not train_dynamic:
        return configured

    quote = str(config.get("base_quote", "USDT")).upper()
    min_volume = float(universe_cfg.get("min_24h_volume_usdt", 1_000_000))
    max_pairs = int(universe_cfg.get("train_max_pairs", universe_cfg.get("max_pairs_scanned", 150)))

    data = BinanceDataManager(config=config, paper=paper)
    discovered = data.discover_pairs_by_quote(
        quote=quote,
        min_quote_volume_usdt=min_volume,
        max_pairs=max_pairs,
    )
    return discovered or configured


def _parse_trained_at(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def model_needs_training(
    *,
    symbol: str,
    models_dir: str,
    max_model_age_hours: float | None,
) -> tuple[bool, str]:
    out_dir = Path(models_dir) / symbol_to_model_key(symbol)
    rf_path = out_dir / "rf.joblib"
    metadata_path = out_dir / "metadata.json"

    if not rf_path.exists():
        return True, "missing_rf_model"
    if not metadata_path.exists():
        return True, "missing_metadata"

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return True, "invalid_metadata"

    trained_at = _parse_trained_at(payload.get("trained_at"))
    if trained_at is None:
        return True, "missing_trained_at"

    if max_model_age_hours is not None and max_model_age_hours > 0:
        age_h = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600.0
        if age_h >= float(max_model_age_hours):
            return True, f"stale_model_{age_h:.1f}h"

    return False, "up_to_date"


def select_symbols_to_train(
    *,
    symbols: list[str],
    train_missing_only: bool,
    models_dir: str,
    max_model_age_hours: float | None,
) -> tuple[list[str], dict[str, str]]:
    if not train_missing_only:
        return symbols, {}

    selected: list[str] = []
    skipped_reasons: dict[str, str] = {}
    for symbol in symbols:
        needs, reason = model_needs_training(
            symbol=symbol,
            models_dir=models_dir,
            max_model_age_hours=max_model_age_hours,
        )
        if needs:
            selected.append(symbol)
        else:
            skipped_reasons[symbol] = reason
    return selected, skipped_reasons


def train_once(
    config: dict,
    paper: bool,
    symbols: list[str] | None = None,
    train_missing_only: bool | None = None,
    max_model_age_hours: float | None = None,
    models_dir: str = "models",
) -> dict:
    trainer = EnsembleTrainer(config)
    universe_cfg = config.get("universe", {})
    if train_missing_only is None:
        train_missing_only = bool(universe_cfg.get("train_missing_only", False))
    if max_model_age_hours is None:
        max_model_age_hours = float(universe_cfg.get("model_max_age_hours", 24))

    target_symbols_all = resolve_training_symbols(config=config, paper=paper, symbols=symbols)
    target_symbols, skipped_up_to_date = select_symbols_to_train(
        symbols=target_symbols_all,
        train_missing_only=bool(train_missing_only),
        models_dir=models_dir,
        max_model_age_hours=max_model_age_hours,
    )

    saved_paths: list[str] = []
    metrics_by_symbol: dict[str, dict] = {}
    best_params_by_symbol: dict[str, dict] = {}
    skipped_errors: dict[str, str] = {}

    for symbol in target_symbols:
        try:
            dataset, features, label_info = build_symbol_dataset(config=config, paper=paper, symbol=symbol)
            result = trainer.train(dataset, features=features, target_col="label")

            model_key = symbol_to_model_key(symbol)
            out_dir = Path(models_dir) / model_key
            paths = trainer.save_models(result.models, out_dir=str(out_dir))
            saved_paths.extend([str(p) for p in paths])

            metadata = {
                "symbol": symbol,
                "model_key": model_key,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_columns": result.selected_features,
                "best_params": result.best_params,
                "dataset_rows": int(len(dataset)),
                "label_counts": {str(k): int(v) for k, v in dataset["label"].value_counts().to_dict().items()},
                "ensemble_weights": result.ensemble_weights,
                "validation_metrics": result.validation_metrics,
                "decision_params": result.decision_params,
                "labeling": label_info,
                "metrics": result.metrics,
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            metrics_by_symbol[symbol] = {
                "dataset_rows": int(len(dataset)),
                "feature_count": int(len(result.selected_features)),
                "label_edge_mean_pct": float(label_info.get("effective_edge_mean_pct", 0.0)),
                **result.metrics,
            }
            best_params_by_symbol[symbol] = result.best_params
        except Exception as exc:
            skipped_errors[symbol] = str(exc)

    aggregate = {
        "symbols_requested": len(target_symbols_all),
        "symbols_queued_for_training": len(target_symbols),
        "symbols_trained": len(metrics_by_symbol),
        "symbols_skipped_up_to_date": len(skipped_up_to_date),
        "symbols_skipped_errors": len(skipped_errors),
    }

    return {
        "saved_models": saved_paths,
        "metrics": metrics_by_symbol,
        "best_params": best_params_by_symbol,
        "aggregate": aggregate,
        "skipped_up_to_date": skipped_up_to_date,
        "skipped_errors": skipped_errors,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paper = args.paper or config.get("environment", "paper") == "paper"

    symbols = args.symbol if args.symbol else None
    try:
        train_result = train_once(
            config=config,
            paper=paper,
            symbols=symbols,
            train_missing_only=args.train_missing_only,
            max_model_age_hours=args.max_model_age_hours,
            models_dir=args.models_dir,
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return

    print("Training complete")
    print("Summary:")
    for key, value in train_result["aggregate"].items():
        print(f"- {key}: {value}")

    print("Saved models:")
    for p in train_result["saved_models"]:
        print(f"- {p}")

    print("Per-symbol metrics:")
    for symbol, symbol_metrics in train_result["metrics"].items():
        print(f"- {symbol}")
        for key, value in symbol_metrics.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")

    print("Per-symbol best params:")
    for symbol, params in train_result["best_params"].items():
        print(f"- {symbol}: {params}")

    if train_result["skipped_up_to_date"]:
        print("Skipped symbols (up-to-date):")
        for symbol, reason in train_result["skipped_up_to_date"].items():
            print(f"- {symbol}: {reason}")

    if train_result["skipped_errors"]:
        print("Skipped symbols (errors):")
        for symbol, reason in train_result["skipped_errors"].items():
            print(f"- {symbol}: {reason}")


if __name__ == "__main__":
    main()
