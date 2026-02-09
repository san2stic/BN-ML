from __future__ import annotations

import pandas as pd


class DataCleaner:
    """Basic OHLCV cleaning suitable for fast iterative development."""

    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
        out[numeric_cols] = out[numeric_cols].ffill().bfill()
        out = out[out["volume"] >= 0]
        return out.reset_index(drop=True)
