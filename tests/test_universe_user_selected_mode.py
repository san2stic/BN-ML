from __future__ import annotations

from scripts.run_bot import TradingRuntime
from scripts.run_trainer import resolve_training_symbols


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)


class _NoDiscoverDataManager:
    def __init__(self) -> None:
        self.called = False

    def discover_pairs_by_quote(self, quote: str, min_quote_volume_usdt: float, max_pairs: int) -> list[str]:
        self.called = True
        return ["SHOULD/NOT_BE_USED"]


def _build_runtime(config: dict, data_manager: _NoDiscoverDataManager) -> TradingRuntime:
    runtime = TradingRuntime.__new__(TradingRuntime)
    runtime.config = config
    runtime.data_manager = data_manager
    runtime.logger = _DummyLogger()
    runtime._universe_cache_pairs = []
    runtime._universe_cache_ts = 0.0
    runtime._quote_asset = lambda: "USDC"  # type: ignore[assignment]
    return runtime


def test_run_bot_scan_pairs_user_selected_only_bypasses_dynamic_discovery() -> None:
    data_manager = _NoDiscoverDataManager()
    runtime = _build_runtime(
        config={
            "base_quote": "USDC",
            "universe": {
                "user_selected_only": True,
                "user_selected_pairs": ["btc/usdc", "ETH/USDC", "BTC/USDC"],
                "pairs": ["SOL/USDC"],
                "dynamic_base_quote_pairs": True,
            },
        },
        data_manager=data_manager,
    )

    pairs = runtime._resolve_pairs_for_scan(force_refresh=False)

    assert pairs == ["BTC/USDC", "ETH/USDC"]
    assert data_manager.called is False


def test_run_bot_training_pairs_user_selected_only_fallbacks_to_pairs() -> None:
    runtime = _build_runtime(
        config={
            "base_quote": "USDC",
            "universe": {
                "user_selected_only": True,
                "user_selected_pairs": [],
                "pairs": ["sol/usdc"],
                "dynamic_base_quote_pairs": True,
                "train_dynamic_pairs": True,
            },
        },
        data_manager=_NoDiscoverDataManager(),
    )

    pairs = runtime._resolve_pairs_for_training()

    assert pairs == ["SOL/USDC"]


def test_run_trainer_resolve_symbols_user_selected_only_skips_dynamic(monkeypatch) -> None:
    class _ExplodingDataManager:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("dynamic discovery should be skipped in user_selected_only mode")

    monkeypatch.setattr("scripts.run_trainer.BinanceDataManager", _ExplodingDataManager)
    symbols = resolve_training_symbols(
        config={
            "base_quote": "USDC",
            "universe": {
                "user_selected_only": True,
                "user_selected_pairs": ["eth/usdc", "BTC/USDC", "ETH/USDC"],
                "pairs": ["SOL/USDC"],
                "dynamic_base_quote_pairs": True,
                "train_dynamic_pairs": True,
            },
        },
        paper=True,
    )

    assert symbols == ["ETH/USDC", "BTC/USDC"]
