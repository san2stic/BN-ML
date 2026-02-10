from __future__ import annotations

from typing import Any

from bn_ml.domain_types import Opportunity


class RiskManager:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.risk_cfg = config.get("risk", {})
        self.scanner_cfg = config.get("scanner", {})
        dynamic_cfg = self.scanner_cfg.get("dynamic_pair_filters", {})
        self.dynamic_pair_filters_cfg = dynamic_cfg if isinstance(dynamic_cfg, dict) else {}

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_fraction(value: Any, default: float) -> float:
        raw = RiskManager._to_float(value, default)
        if raw > 1.0:
            raw = raw / 100.0
        return max(0.0, raw)

    @staticmethod
    def _risk_budget_used_pct(realized_usdt: float, total_capital: float, fallback_pnl_pct: float) -> float:
        if total_capital > 0:
            return max(0.0, (-realized_usdt / total_capital) * 100.0)
        return max(0.0, -fallback_pnl_pct)

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        parts = str(symbol).upper().split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return parts[0], ""

    def _pair_filter_thresholds(self, symbol: str) -> tuple[float, float, float]:
        spread_limit = float(self.scanner_cfg.get("spread_max_pct", 0.15))
        depth_limit = float(self.scanner_cfg.get("orderbook_depth_min_usdt", 50_000))
        correlation_limit = float(self.risk_cfg.get("max_correlation", 0.70))

        cfg = self.dynamic_pair_filters_cfg
        if not bool(cfg.get("enabled", False)):
            return spread_limit, depth_limit, correlation_limit

        base_asset, _ = self._split_symbol(symbol)
        major_bases = {
            str(x).strip().upper()
            for x in cfg.get("major_bases", ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE"])
            if str(x).strip()
        }
        is_major = base_asset in major_bases

        if is_major:
            spread_limit *= self._to_float(cfg.get("spread_factor_major"), 1.10)
            depth_limit *= self._to_float(cfg.get("depth_factor_major"), 0.90)
            correlation_limit += self._to_float(cfg.get("correlation_bonus_major"), 0.05)
        else:
            spread_limit *= self._to_float(cfg.get("spread_factor_alt"), 0.95)
            depth_limit *= self._to_float(cfg.get("depth_factor_alt"), 1.10)
            correlation_limit += self._to_float(cfg.get("correlation_penalty_alt"), -0.05)

        if base_asset == "BTC":
            correlation_limit = max(correlation_limit, self._to_float(cfg.get("correlation_limit_benchmark"), 1.0))

        by_symbol = cfg.get("by_symbol", {})
        symbol_key = str(symbol).upper()
        if isinstance(by_symbol, dict):
            override = None
            for key, value in by_symbol.items():
                if str(key).strip().upper() == symbol_key and isinstance(value, dict):
                    override = value
                    break
            if override:
                if "spread_max_pct" in override:
                    spread_limit = self._to_float(override.get("spread_max_pct"), spread_limit)
                if "orderbook_depth_min_usdt" in override:
                    depth_limit = self._to_float(override.get("orderbook_depth_min_usdt"), depth_limit)
                if "max_correlation" in override:
                    correlation_limit = self._to_float(override.get("max_correlation"), correlation_limit)

        spread_limit = max(0.0, spread_limit)
        depth_limit = max(0.0, depth_limit)
        correlation_limit = min(1.0, max(0.0, correlation_limit))
        return spread_limit, depth_limit, correlation_limit

    def risk_budget_snapshot(self, account_state: dict[str, float], size_usdt: float = 0.0) -> dict[str, float]:
        total_capital = max(
            self._to_float(account_state.get("total_capital"), 0.0),
            self._to_float(account_state.get("active_capital"), 0.0),
            1e-9,
        )
        max_daily_risk_frac = self._to_fraction(self.risk_cfg.get("max_daily_risk_pct", 0.02), default=0.02)
        max_weekly_risk_frac = self._to_fraction(self.risk_cfg.get("max_weekly_risk_pct", 0.06), default=0.06)
        per_trade_risk_frac = self._to_fraction(self.risk_cfg.get("max_position_drawdown_pct", 0.03), default=0.03)

        daily_realized_usdt = self._to_float(account_state.get("daily_realized_usdt"), 0.0)
        weekly_realized_usdt = self._to_float(account_state.get("weekly_realized_usdt"), 0.0)
        daily_used_pct = self._risk_budget_used_pct(
            realized_usdt=daily_realized_usdt,
            total_capital=total_capital,
            fallback_pnl_pct=self._to_float(account_state.get("daily_pnl_pct"), 0.0),
        )
        weekly_used_pct = self._risk_budget_used_pct(
            realized_usdt=weekly_realized_usdt,
            total_capital=total_capital,
            fallback_pnl_pct=self._to_float(account_state.get("weekly_pnl_pct"), 0.0),
        )
        projected_trade_risk_pct = (self._to_float(size_usdt, 0.0) * per_trade_risk_frac / total_capital) * 100.0

        return {
            "total_capital": total_capital,
            "daily_used_pct": daily_used_pct,
            "weekly_used_pct": weekly_used_pct,
            "daily_budget_pct": max_daily_risk_frac * 100.0,
            "weekly_budget_pct": max_weekly_risk_frac * 100.0,
            "projected_trade_risk_pct": projected_trade_risk_pct,
            "daily_used_with_trade_pct": daily_used_pct + projected_trade_risk_pct,
            "weekly_used_with_trade_pct": weekly_used_pct + projected_trade_risk_pct,
        }

    def can_open_position(
        self,
        opportunity: Opportunity,
        open_positions: list[dict[str, Any]],
        account_state: dict[str, float],
    ) -> tuple[bool, list[str], float]:
        reasons: list[str] = []
        spread_limit, depth_limit, correlation_limit = self._pair_filter_thresholds(opportunity.symbol)

        if opportunity.signal.action != "BUY":
            reasons.append("Signal is not BUY.")

        if opportunity.signal.confidence < float(self.scanner_cfg.get("min_ml_confidence", 65)):
            reasons.append("ML confidence below threshold.")

        if opportunity.spread_pct > spread_limit:
            reasons.append("Spread too high.")

        if opportunity.orderbook_depth_usdt < depth_limit:
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
        if bool(self.risk_cfg.get("circuit_breakers", {}).get("drift_block_enabled", False)) and bool(
            account_state.get("market_drift_detected", False)
        ):
            reasons.append("Market drift circuit breaker active.")

        cb_cfg = self.risk_cfg.get("circuit_breakers", {})
        if bool(cb_cfg.get("market_intelligence_block_enabled", False)):
            intelligence_signal = str(account_state.get("market_intelligence_signal", "HOLD")).strip().upper()
            intelligence_confidence = self._to_float(account_state.get("market_intelligence_confidence"), 0.0)
            confidence_min = self._to_float(cb_cfg.get("market_intelligence_min_confidence"), 70.0)
            if intelligence_signal == "SELL" and intelligence_confidence >= confidence_min:
                reasons.append("SanTradeIntelligence bearish circuit breaker active.")

            risk_off_enabled = bool(cb_cfg.get("market_intelligence_block_on_risk_off", True))
            intelligence_regime = str(account_state.get("market_intelligence_regime", "")).strip().lower()
            if risk_off_enabled and intelligence_regime == "risk_off":
                reasons.append("SanTradeIntelligence risk-off circuit breaker active.")

        if opportunity.correlation_with_btc > correlation_limit:
            reasons.append("Correlation threshold exceeded.")

        size_usdt = self.position_size(account_state=account_state, atr_ratio=opportunity.atr_ratio)

        # Strict pre-trade risk budget enforcement (daily + weekly), including projected risk if order is opened.
        snapshot = self.risk_budget_snapshot(account_state=account_state, size_usdt=size_usdt)
        daily_used_pct = snapshot["daily_used_pct"]
        weekly_used_pct = snapshot["weekly_used_pct"]
        daily_budget_pct = snapshot["daily_budget_pct"]
        weekly_budget_pct = snapshot["weekly_budget_pct"]

        if daily_used_pct >= daily_budget_pct:
            reasons.append("Daily risk budget exhausted.")
        elif snapshot["daily_used_with_trade_pct"] > daily_budget_pct:
            reasons.append("Daily risk budget would be exceeded by this trade.")

        if weekly_used_pct >= weekly_budget_pct:
            reasons.append("Weekly risk budget exhausted.")
        elif snapshot["weekly_used_with_trade_pct"] > weekly_budget_pct:
            reasons.append("Weekly risk budget would be exceeded by this trade.")

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
