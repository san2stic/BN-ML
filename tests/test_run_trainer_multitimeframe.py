from __future__ import annotations

from scripts.run_trainer import build_symbol_dataset


def test_build_symbol_dataset_includes_mtf_features_when_enabled() -> None:
    config = {
        "environment": "paper",
        "base_quote": "USDC",
        "model": {
            "train_ohlcv_limit": 420,
            "multi_timeframe": {
                "enabled": True,
                "base_timeframe": "15m",
                "timeframes": ["1h", "4h"],
                "min_candles_per_timeframe": 80,
                "max_candles_per_timeframe": 180,
                "extra_candles_buffer": 24,
            },
            "labeling": {
                "horizon_candles": 4,
                "volatility_window": 64,
                "volatility_multiplier": 0.6,
                "atr_multiplier": 0.35,
                "min_move_pct": 0.0015,
                "max_move_pct": 0.02,
            },
        },
        "risk": {"fees_pct": 0.1, "slippage_pct": 0.05},
    }

    ds, features, info = build_symbol_dataset(config=config, paper=True, symbol="BTC/USDC")

    assert len(ds) > 250
    assert "mtf_confluence_score" in ds.columns
    assert any(col.startswith("mtf_1h_") for col in ds.columns)
    assert any(col.startswith("mtf_4h_") for col in ds.columns)
    assert any(col.startswith("mtf_") for col in features)
    assert info["multi_timeframe"]["enabled"] is True
    assert info["multi_timeframe"]["base_timeframe"] == "15m"
