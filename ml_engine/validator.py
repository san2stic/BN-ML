from __future__ import annotations

import numpy as np
import pandas as pd


class BacktestValidator:
    """Vectorized backtest with transaction costs and comprehensive metrics."""

    @staticmethod
    def run(frame: pd.DataFrame, trade_cost: float = 0.0) -> dict[str, float]:
        df = frame.copy()
        if "signal" not in df.columns:
            raise ValueError("Backtest requires a 'signal' column with values {-1, 0, 1}.")

        df["ret"] = df["close"].pct_change().fillna(0)
        df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["ret"]

        # Apply transaction costs
        if trade_cost > 0:
            turnover = df["signal"].diff().abs().fillna(0)
            df["strategy_ret"] -= turnover * trade_cost

        equity = (1 + df["strategy_ret"]).cumprod()
        total_return = float(equity.iloc[-1] - 1)

        # Max drawdown (multiplicative, correct)
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max.replace(0, np.nan)
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        # Win rate on actual trades
        trades = df.loc[df["strategy_ret"] != 0, "strategy_ret"]
        win_rate = float((trades > 0).mean()) if len(trades) > 0 else 0.0

        # Sharpe ratio (annualized for 15m bars: 96 bars/day * 365)
        ann_factor = float(np.sqrt(96 * 365))
        mean_ret = float(df["strategy_ret"].mean())
        std_ret = float(df["strategy_ret"].std())
        sharpe = float((mean_ret / std_ret) * ann_factor) if std_ret > 1e-12 else 0.0

        # Sortino ratio
        downside = df.loc[df["strategy_ret"] < 0, "strategy_ret"]
        downside_std = float(downside.std()) if len(downside) > 0 else 0.0
        sortino = float((mean_ret / downside_std) * ann_factor) if downside_std > 1e-12 else 0.0

        # Profit factor
        gross_profit = float(trades[trades > 0].sum()) if (trades > 0).any() else 0.0
        gross_loss_abs = float(trades[trades < 0].sum().abs()) if (trades < 0).any() else 0.0
        profit_factor = gross_profit / gross_loss_abs if gross_loss_abs > 1e-12 else 0.0

        # Calmar ratio
        calmar = float(total_return / abs(max_drawdown)) if abs(max_drawdown) > 1e-12 else 0.0

        # Trade count
        trade_count = int((df["signal"].diff().abs() > 0).sum())

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar,
            "trade_count": trade_count,
        }
