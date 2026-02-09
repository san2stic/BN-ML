from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    strength: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Opportunity:
    symbol: str
    ml_score: float
    technical_score: float
    momentum_score: float
    global_score: float
    signal: Signal
    spread_pct: float
    orderbook_depth_usdt: float
    atr_ratio: float
    expected_net_profit_pct: float
    correlation_with_btc: float


@dataclass
class Position:
    symbol: str
    side: str
    size_usdt: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    opened_at: datetime
    status: str = "OPEN"
    extra: dict = field(default_factory=dict)
