from __future__ import annotations

import pandas as pd

from ml_engine.predictor import MLEnsemblePredictor


def test_missing_model_callback_is_emitted_once_per_symbol(tmp_path) -> None:
    calls: list[str] = []
    predictor = MLEnsemblePredictor(
        model_dir=str(tmp_path / "models"),
        missing_model_callback=lambda symbol: calls.append(symbol),
    )

    frame = pd.DataFrame([{"rsi_14": 50.0, "macd_hist": 0.0}])
    predictor.predict(symbol="BTC/USDC", frame=frame, feature_columns=["rsi_14", "macd_hist"])
    predictor.predict(symbol="BTC/USDC", frame=frame, feature_columns=["rsi_14", "macd_hist"])

    assert calls == ["BTC/USDC"]
