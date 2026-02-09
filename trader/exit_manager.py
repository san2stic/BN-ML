from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from bn_ml.domain_types import Position


@dataclass
class ExitDecision:
    close_all: bool
    close_reason: str | None
    partial_fracs: list[float]
    new_stop_loss: float
    updated_extra: dict


class ExitManager:
    def __init__(self, config: dict) -> None:
        self.risk = config.get("risk", {})

    def evaluate_long(self, position: Position, price: float, atr_value: float, now: datetime) -> ExitDecision:
        extra = dict(position.extra or {})
        initial_size = float(extra.get("initial_size_usdt", position.size_usdt))
        extra["initial_size_usdt"] = initial_size

        tp1_hit = bool(extra.get("tp1_hit", False))
        tp2_hit = bool(extra.get("tp2_hit", False))
        trailing_active = bool(extra.get("trailing_active", False))
        max_price_seen = max(float(extra.get("max_price_seen", position.entry_price)), price)
        extra["max_price_seen"] = max_price_seen

        pnl_pct = ((price / max(position.entry_price, 1e-9)) - 1.0) * 100.0
        hold_hours = (now - position.opened_at).total_seconds() / 3600.0

        stop_loss = float(position.stop_loss)
        partial_fracs: list[float] = []

        trailing_activation = float(self.risk.get("trailing_activation_pct", 0.02)) * 100.0
        trail_atr_mult = float(self.risk.get("trailing_atr_mult", 1.0))

        if pnl_pct >= trailing_activation:
            trailing_active = True
            new_stop = price - trail_atr_mult * max(atr_value, 1e-9)
            stop_loss = max(stop_loss, new_stop)

        extra["trailing_active"] = trailing_active

        if not tp1_hit and price >= position.take_profit_1:
            partial_fracs.append(0.50)
            tp1_hit = True

        if not tp2_hit and price >= position.take_profit_2:
            partial_fracs.append(0.30)
            tp2_hit = True

        extra["tp1_hit"] = tp1_hit
        extra["tp2_hit"] = tp2_hit

        max_position_drawdown = float(self.risk.get("max_position_drawdown_pct", 0.03)) * 100.0
        time_stop_hours = float(self.risk.get("time_stop_hours", 48))

        if price <= stop_loss:
            return ExitDecision(
                close_all=True,
                close_reason="stop_loss",
                partial_fracs=partial_fracs,
                new_stop_loss=stop_loss,
                updated_extra=extra,
            )

        if pnl_pct <= -max_position_drawdown:
            return ExitDecision(
                close_all=True,
                close_reason="hard_drawdown",
                partial_fracs=partial_fracs,
                new_stop_loss=stop_loss,
                updated_extra=extra,
            )

        if hold_hours >= time_stop_hours and abs(pnl_pct) < 0.5:
            return ExitDecision(
                close_all=True,
                close_reason="time_stop",
                partial_fracs=partial_fracs,
                new_stop_loss=stop_loss,
                updated_extra=extra,
            )

        return ExitDecision(
            close_all=False,
            close_reason=None,
            partial_fracs=partial_fracs,
            new_stop_loss=stop_loss,
            updated_extra=extra,
        )
