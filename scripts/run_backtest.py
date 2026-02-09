from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bn_ml.config import load_config
from data_manager.data_cleaner import DataCleaner
from data_manager.features_engineer import FeatureEngineer
from data_manager.fetch_data import BinanceDataManager
from data_manager.multi_timeframe import MultiTimeframeFeatureBuilder
from ml_engine.validator import BacktestValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BN-ML baseline backtest")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--paper", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paper = args.paper or config.get("environment", "paper") == "paper"

    symbol = config.get("universe", {}).get("pairs", ["BTC/USDT"])[0]
    manager = BinanceDataManager(config=config, paper=paper)
    cleaner = DataCleaner()
    feat = FeatureEngineer()
    mtf_builder = MultiTimeframeFeatureBuilder(
        config=config,
        data_manager=manager,
        cleaner=cleaner,
        feature_engineer=feat,
    )

    backtest_limit = int(config.get("model", {}).get("backtest_ohlcv_limit", 2000))
    df = mtf_builder.build(symbol=symbol, limit=backtest_limit)

    df["signal"] = 0
    df.loc[(df["rsi_14"] < 35) & (df["macd_hist"] > 0), "signal"] = 1
    df.loc[(df["rsi_14"] > 70) & (df["macd_hist"] < 0), "signal"] = -1

    result = BacktestValidator.run(df)

    out_dir = Path("artifacts/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(out_dir / "backtest_summary.csv", index=False)

    print("Backtest complete")
    for key, value in result.items():
        print(f"- {key}: {value:.6f}")


if __name__ == "__main__":
    main()
