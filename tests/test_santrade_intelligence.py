from __future__ import annotations

from pathlib import Path

from bn_ml.domain_types import Opportunity, Signal
from ml_engine.santrade_intelligence import SanTradeIntelligence


class _DummyDataManager:
    def __init__(self, prices: list[float]) -> None:
        self._prices = prices
        self._index = 0

    def fetch_last_price(self, symbol: str) -> float:
        del symbol
        if not self._prices:
            return 0.0
        value = self._prices[min(self._index, len(self._prices) - 1)]
        self._index += 1
        return float(value)


def _config(state_path: Path, profile: str = "neutral", **overrides) -> dict:
    sti_cfg = {
        "enabled": True,
        "profile": profile,
        "min_pairs": 4,
        "min_samples_for_model": 6,
        "persist_every_updates": 1,
        "online_target_hold_band_pct": 0.04,
        "min_directional_confidence": 55,
        "min_directional_streak": 1,
        "state_path": str(state_path),
    }
    sti_cfg.update(overrides)
    return {
        "model": {
            "random_state": 7,
            "santrade_intelligence": sti_cfg,
        }
    }


def _opportunity(symbol: str, action: str, *, momentum: float, ml_score: float, atr: float, spread: float) -> Opportunity:
    return Opportunity(
        symbol=symbol,
        ml_score=ml_score,
        technical_score=68.0,
        momentum_score=momentum,
        global_score=(ml_score * 0.55) + (momentum * 0.45),
        signal=Signal(symbol=symbol, action=action, confidence=82.0, strength=80.0),
        spread_pct=spread,
        orderbook_depth_usdt=150_000.0,
        atr_ratio=atr,
        expected_net_profit_pct=0.75,
        correlation_with_btc=0.45,
    )


def _bullish_opportunities() -> list[Opportunity]:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT"]
    return [
        _opportunity(symbol, "BUY", momentum=74.0, ml_score=78.0, atr=0.010, spread=0.04)
        for symbol in symbols
    ]


def _risk_off_opportunities() -> list[Opportunity]:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT"]
    return [
        _opportunity(symbol, "SELL", momentum=28.0, ml_score=30.0, atr=0.030, spread=0.32)
        for symbol in symbols
    ]


def _balanced_opportunities() -> list[Opportunity]:
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT", "XRP/USDT"]
    opps: list[Opportunity] = []
    for idx, symbol in enumerate(symbols):
        action = "BUY" if idx < 3 else "HOLD"
        opps.append(_opportunity(symbol, action, momentum=56.0, ml_score=55.0, atr=0.009, spread=0.04))
    return opps


def test_emits_bullish_market_signal_when_breadth_is_positive(tmp_path: Path) -> None:
    engine = SanTradeIntelligence(
        config=_config(tmp_path / "sti_state.joblib"),
        data_manager=_DummyDataManager([100.0, 100.2]),
    )

    snapshot = engine.update(
        pairs=[opp.symbol for opp in _bullish_opportunities()],
        opportunities=_bullish_opportunities(),
        quote_asset="USDT",
    )

    assert snapshot.signal == "BUY"
    assert snapshot.market_regime == "bull_acceleration"
    assert snapshot.confidence > 60.0


def test_flags_risk_off_regime_on_stress_conditions(tmp_path: Path) -> None:
    engine = SanTradeIntelligence(
        config=_config(tmp_path / "sti_state.joblib"),
        data_manager=_DummyDataManager([100.0, 99.5]),
    )

    snapshot = engine.update(
        pairs=[opp.symbol for opp in _risk_off_opportunities()],
        opportunities=_risk_off_opportunities(),
        quote_asset="USDT",
    )

    assert snapshot.signal == "SELL"
    assert snapshot.market_regime == "risk_off"
    assert snapshot.predicted_move_pct < 0


def test_online_training_accumulates_samples_and_sets_model_ready(tmp_path: Path) -> None:
    state_path = tmp_path / "sti_state.joblib"
    prices = [100 + (i * 0.07) for i in range(40)]
    engine = SanTradeIntelligence(
        config=_config(state_path),
        data_manager=_DummyDataManager(prices),
    )

    snapshot = None
    for idx in range(20):
        opportunities = _bullish_opportunities() if idx % 2 == 0 else _risk_off_opportunities()
        snapshot = engine.update(
            pairs=[opp.symbol for opp in opportunities],
            opportunities=opportunities,
            quote_asset="USDT",
        )

    assert snapshot is not None
    assert snapshot.model_samples >= 10
    assert snapshot.model_ready is True
    assert state_path.exists()


def test_flush_state_persists_and_recovers_when_interval_not_reached(tmp_path: Path) -> None:
    state_path = tmp_path / "sti_flush.joblib"
    config = _config(state_path, persist_every_updates=99)
    engine = SanTradeIntelligence(
        config=config,
        data_manager=_DummyDataManager([100.0, 100.2, 99.9]),
    )

    engine.update(
        pairs=[opp.symbol for opp in _bullish_opportunities()],
        opportunities=_bullish_opportunities(),
        quote_asset="USDT",
    )
    engine.update(
        pairs=[opp.symbol for opp in _risk_off_opportunities()],
        opportunities=_risk_off_opportunities(),
        quote_asset="USDT",
    )

    assert not state_path.exists()
    assert engine._model_samples > 0

    engine.flush_state()
    assert state_path.exists()

    restored = SanTradeIntelligence(
        config=config,
        data_manager=_DummyDataManager([100.0]),
    )
    assert restored._model_samples == engine._model_samples
    assert restored._model_fitted == engine._model_fitted


def test_aggressive_profile_is_more_directional_than_defensive_profile(tmp_path: Path) -> None:
    opportunities = _balanced_opportunities()
    prices = [100.0, 100.05, 100.1]

    aggressive = SanTradeIntelligence(
        config=_config(
            tmp_path / "sti_aggressive.joblib",
            profile="aggressive",
            min_directional_confidence=0,
            min_directional_streak=1,
            min_pairs=4,
        ),
        data_manager=_DummyDataManager(prices),
    )
    defensive = SanTradeIntelligence(
        config=_config(
            tmp_path / "sti_defensive.joblib",
            profile="defensive",
            min_directional_confidence=0,
            min_directional_streak=1,
            min_pairs=4,
        ),
        data_manager=_DummyDataManager(prices),
    )

    aggressive_snapshot = aggressive.update(
        pairs=[opp.symbol for opp in opportunities],
        opportunities=opportunities,
        quote_asset="USDT",
    )
    defensive_snapshot = defensive.update(
        pairs=[opp.symbol for opp in opportunities],
        opportunities=opportunities,
        quote_asset="USDT",
    )

    assert aggressive_snapshot.signal == "BUY"
    assert defensive_snapshot.signal == "HOLD"


def test_requires_consecutive_cycles_before_directional_signal_when_streak_enabled(tmp_path: Path) -> None:
    engine = SanTradeIntelligence(
        config=_config(
            tmp_path / "sti_streak.joblib",
            profile="neutral",
            min_directional_streak=2,
            min_directional_confidence=0,
        ),
        data_manager=_DummyDataManager([100.0, 100.2, 100.4]),
    )

    first = engine.update(
        pairs=[opp.symbol for opp in _bullish_opportunities()],
        opportunities=_bullish_opportunities(),
        quote_asset="USDT",
    )
    second = engine.update(
        pairs=[opp.symbol for opp in _bullish_opportunities()],
        opportunities=_bullish_opportunities(),
        quote_asset="USDT",
    )

    assert first.signal == "HOLD"
    assert second.signal == "BUY"
