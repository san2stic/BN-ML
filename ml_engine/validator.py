from __future__ import annotations

import pandas as pd


class BacktestValidator:
    """Lightweight vectorized backtest for baseline sanity checks."""

    @staticmethod
    def run(frame: pd.DataFrame) -> dict[str, float]:
        df = frame.copy()
        if "signal" not in df.columns:
            raise ValueError("Backtest requires a 'signal' column with values {-1, 0, 1}.")

        df["ret"] = df["close"].pct_change().fillna(0)
        df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["ret"]

        total_return = float((1 + df["strategy_ret"]).prod() - 1)
        max_dd = float((df["strategy_ret"].cumsum().cummax() - df["strategy_ret"].cumsum()).max())
        win_rate = float((df["strategy_ret"] > 0).mean())

        return {
            "total_return": total_return,
            "max_drawdown_proxy": max_dd,
            "win_rate": win_rate,
        }
