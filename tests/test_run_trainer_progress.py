from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from bn_ml.state_store import StateStore
import scripts.run_trainer as run_trainer


class _DummyTrainer:
    def __init__(self, config: dict) -> None:
        self.config = config

    def train(self, dataset: pd.DataFrame, features: list[str], target_col: str = "label") -> SimpleNamespace:
        return SimpleNamespace(
            models={"rf": object()},
            selected_features=features,
            best_params={"rf": {}},
            ensemble_weights={"rf": 1.0},
            validation_metrics={"rf": {"accuracy": 0.5}},
            decision_params={"buy_threshold": 0.5, "sell_threshold": 0.5},
            metrics={"rf_train_accuracy": 0.6, "rf_train_f1_macro": 0.5},
        )

    def save_models(self, models: dict, out_dir: str = "models") -> list[Path]:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / "rf.joblib"
        model_path.write_text("dummy", encoding="utf-8")
        return [model_path]


def _dataset_for_symbol(symbol: str) -> tuple[pd.DataFrame, list[str], dict[str, float | dict]]:
    frame = pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2, 1.3],
            "label": [0, 1, 2, 1],
            "f1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    return frame, ["f1"], {"effective_edge_mean_pct": 0.003, "multi_timeframe": {"enabled": False}, "symbol": symbol}


def test_train_once_emits_progress_and_completes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_trainer, "EnsembleTrainer", _DummyTrainer)
    monkeypatch.setattr(run_trainer, "build_symbol_dataset", lambda config, paper, symbol: _dataset_for_symbol(symbol))

    updates: list[dict] = []
    result = run_trainer.train_once(
        config={"model": {}, "universe": {"train_missing_only": False}},
        paper=True,
        symbols=["BTC/USDC", "ETH/USDC"],
        models_dir=str(tmp_path / "models"),
        progress_callback=lambda payload: updates.append(payload),
    )

    assert result["aggregate"]["symbols_trained"] == 2
    assert updates
    assert updates[0]["status"] == "running"
    assert updates[0]["phase"] == "queued"
    assert any(update.get("phase") == "training" and update.get("current_symbol") == "BTC/USDC" for update in updates)
    assert updates[-1]["status"] == "completed"
    assert updates[-1]["progress_pct"] == 100.0
    assert updates[-1]["symbols_trained"] == 2


def test_train_once_progress_tracks_symbol_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_trainer, "EnsembleTrainer", _DummyTrainer)

    def _build(config: dict, paper: bool, symbol: str):
        if symbol == "ETH/USDC":
            raise ValueError("boom")
        return _dataset_for_symbol(symbol)

    monkeypatch.setattr(run_trainer, "build_symbol_dataset", _build)

    updates: list[dict] = []
    result = run_trainer.train_once(
        config={"model": {}, "universe": {"train_missing_only": False}},
        paper=True,
        symbols=["BTC/USDC", "ETH/USDC"],
        models_dir=str(tmp_path / "models"),
        progress_callback=lambda payload: updates.append(payload),
    )

    assert result["aggregate"]["symbols_trained"] == 1
    assert result["aggregate"]["symbols_skipped_errors"] == 1
    assert any("last_error" in update for update in updates)
    assert updates[-1]["status"] == "completed"
    assert updates[-1]["symbols_errors"] == 1


def test_train_once_persists_training_status_without_callback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_trainer, "EnsembleTrainer", _DummyTrainer)
    monkeypatch.setattr(run_trainer, "build_symbol_dataset", lambda config, paper, symbol: _dataset_for_symbol(symbol))

    db_path = tmp_path / "state.db"
    result = run_trainer.train_once(
        config={
            "model": {},
            "universe": {"train_missing_only": False},
            "storage": {"sqlite_path": str(db_path)},
        },
        paper=True,
        symbols=["BTC/USDC"],
        models_dir=str(tmp_path / "models"),
    )

    assert result["aggregate"]["symbols_trained"] == 1

    store = StateStore(db_path=str(db_path))
    status = store.get_state("training_status", {})
    assert status["status"] == "completed"
    assert status["phase"] == "done"
    assert status["trigger"] == "manual"
    assert status["symbols_queued"] == 1
    assert status["symbols_completed"] == 1
    assert status["symbols_trained"] == 1
    assert status["progress_pct"] == 100.0
