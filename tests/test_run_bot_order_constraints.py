from __future__ import annotations

import threading

from bn_ml.domain_types import Opportunity, Signal
from scripts.run_bot import TradingRuntime
from trader.order_manager import OrderConstraintError


class _DummyLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.exceptions: list[str] = []

    def info(self, msg: str, *args) -> None:
        self.infos.append(msg % args if args else msg)

    def exception(self, msg: str, *args) -> None:
        self.exceptions.append(msg % args if args else msg)


class _DummyStore:
    def __init__(self) -> None:
        self.inserted_trades = 0
        self.inserted_cycles = 0

    def save_account_state(self, _state) -> None:
        return None

    def insert_trade(self, **_kwargs) -> None:
        self.inserted_trades += 1

    def insert_cycle(self, **_kwargs) -> None:
        self.inserted_cycles += 1

    def export_positions_snapshot(self) -> None:
        return None


class _DummyPositionManager:
    def has_open(self, _symbol: str) -> bool:
        return False

    def list_open(self) -> list[dict]:
        return []

    def add(self, _position) -> None:
        return None


class _DummyScanner:
    def __init__(self, opportunity: Opportunity) -> None:
        self._opportunity = opportunity

    def scan_details(self, _pairs: list[str]):
        return [self._opportunity], [self._opportunity]


class _DummyRiskManager:
    def can_open_position(self, opportunity, open_positions, account_state):
        del opportunity, open_positions, account_state
        return True, [], 3.37

    def risk_budget_snapshot(self, account_state, size_usdt: float = 0.0):
        del account_state, size_usdt
        return {}


class _DummyDataManager:
    def fetch_last_price(self, _symbol: str) -> float:
        return 1.0


class _DummyOrderManager:
    def place_market_buy(self, symbol: str, size_usdt: float, price: float, atr: float):
        del size_usdt, price, atr
        raise OrderConstraintError(f"Order notional below minNotional for {symbol}: 3.37 < 5.0")


class _DummyAlerter:
    def send(self, _message: str) -> None:
        return None


class _DummyBackupManager:
    def maybe_backup(self, force: bool = False) -> None:
        del force
        return None


def test_run_cycle_skips_order_constraint_error_without_exception_log() -> None:
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime._model_components_lock = threading.Lock()
    runtime.logger = _DummyLogger()
    runtime.store = _DummyStore()
    runtime.account_state = {"active_capital": 1000.0}
    runtime.position_manager = _DummyPositionManager()
    runtime.risk_manager = _DummyRiskManager()
    runtime.data_manager = _DummyDataManager()
    runtime.order_manager = _DummyOrderManager()
    runtime.alerter = _DummyAlerter()
    runtime.backup_manager = _DummyBackupManager()
    runtime.config = {}
    runtime.paper = False

    opportunity = Opportunity(
        symbol="TRX/USDC",
        ml_score=80.0,
        technical_score=75.0,
        momentum_score=70.0,
        global_score=78.0,
        signal=Signal(symbol="TRX/USDC", action="BUY", confidence=82.0, strength=80.0),
        spread_pct=0.04,
        orderbook_depth_usdt=100_000.0,
        atr_ratio=0.01,
        expected_net_profit_pct=0.8,
        correlation_with_btc=0.2,
    )
    runtime.scanner = _DummyScanner(opportunity)

    runtime._sync_realtime_price_stream = lambda force=False: None  # type: ignore[assignment]
    runtime._sync_live_capital = lambda: None  # type: ignore[assignment]
    runtime._manage_open_positions = lambda: (0, 0)  # type: ignore[assignment]
    runtime._refresh_recent_performance = lambda: None  # type: ignore[assignment]
    runtime._update_market_volatility_ratio = lambda: None  # type: ignore[assignment]
    runtime._update_market_drift_state = lambda: None  # type: ignore[assignment]
    runtime._resolve_pairs_for_scan = lambda force_refresh=False: ["TRX/USDC"]  # type: ignore[assignment]
    runtime._export_scan = lambda opportunities, opportunities_all: None  # type: ignore[assignment]

    runtime.run_cycle()

    assert any("Skip TRX/USDC: Order notional below minNotional" in msg for msg in runtime.logger.infos)
    assert runtime.logger.exceptions == []
    assert runtime.store.inserted_trades == 0
    assert runtime.store.inserted_cycles == 1
