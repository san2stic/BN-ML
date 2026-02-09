from __future__ import annotations

from typing import Any

from bn_ml.types import Opportunity


class RiskManager:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.risk_cfg = config.get("risk", {})
        self.scanner_cfg = config.get("scanner", {})

    def can_open_position(
        self,
        opportunity: Opportunity,
        open_positions: list[dict[str, Any]],
        account_state: dict[str, float],
    ) -> tuple[bool, list[str], float]:
        reasons: list[str] = []

        if opportunity.signal.action != "BUY":
            reasons.append("Signal is not BUY.")

        if opportunity.signal.confidence < float(self.scanner_cfg.get("min_ml_confidence", 65)):
            reasons.append("ML confidence below threshold.")

        if opportunity.spread_pct > float(self.scanner_cfg.get("spread_max_pct", 0.15)):
            reasons.append("Spread too high.")

        if opportunity.orderbook_depth_usdt < float(self.scanner_cfg.get("orderbook_depth_min_usdt", 50_000)):
            reasons.append("Orderbook depth too low.")

        if opportunity.expected_net_profit_pct < float(self.risk_cfg.get("min_net_profit_pct", 0.30)):
            reasons.append("Expected net profit below minimum.")

        if len(open_positions) >= int(self.risk_cfg.get("max_positions", 5)):
            reasons.append("Max number of positions reached.")

        if account_state.get("daily_pnl_pct", 0.0) <= float(
            self.risk_cfg.get("circuit_breakers", {}).get("daily_drawdown_stop_pct", -5.0)
        ):
            reasons.append("Daily drawdown circuit breaker active.")

        if account_state.get("consecutive_losses", 0) >= int(
            self.risk_cfg.get("circuit_breakers", {}).get("max_consecutive_losses", 3)
        ):
            reasons.append("Consecutive losses circuit breaker active.")

        if account_state.get("market_volatility_ratio", 1.0) > float(
            self.risk_cfg.get("circuit_breakers", {}).get("volatility_spike_ratio", 1.5)
        ):
            reasons.append("Market volatility circuit breaker active.")

        correlation_limit = float(self.risk_cfg.get("max_correlation", 0.70))
        if opportunity.correlation_with_btc > correlation_limit:
            reasons.append("Correlation threshold exceeded.")

        size_usdt = self.position_size(account_state=account_state, atr_ratio=opportunity.atr_ratio)

        active_capital = float(account_state.get("active_capital", 0.0))
        current_exposure = float(sum(p.get("size_usdt", 0.0) for p in open_positions))
        max_exposure = active_capital * float(self.risk_cfg.get("max_portfolio_exposure_pct", 0.70))
        if current_exposure + size_usdt > max_exposure:
            reasons.append("Portfolio exposure limit exceeded.")

        return len(reasons) == 0, reasons, size_usdt

    def position_size(self, account_state: dict[str, float], atr_ratio: float) -> float:
        active_capital = float(account_state.get("active_capital", 0.0))
        max_positions = int(self.risk_cfg.get("max_positions", 5))

        win_rate = float(account_state.get("win_rate", 0.55))
        avg_win = float(account_state.get("avg_win", 1.8))
        avg_loss = max(float(account_state.get("avg_loss", 1.0)), 1e-6)

        kelly = max(0.0, (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss)
        base_size = (active_capital * kelly) / max(max_positions, 1)

        min_size = active_capital * float(self.risk_cfg.get("min_position_pct_active", 0.01))
        max_size = active_capital * float(self.risk_cfg.get("max_position_pct_active", 0.20))

        vol_adjust = max(float(atr_ratio), 0.005)
        adjusted_size = base_size / vol_adjust

        return max(min_size, min(max_size, adjusted_size))
