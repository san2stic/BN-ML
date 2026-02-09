from __future__ import annotations

import pandas as pd

from data_manager.fetch_data import BinanceDataManager


class _FakeExchange:
    def __init__(self) -> None:
        self.markets = {"BTC/USDT": {"spot": True, "active": True, "base": "BTC", "quote": "USDT"}}

    def load_markets(self) -> dict:
        return self.markets

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500):
        return [
            [1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 1200.0],
            [1_700_000_900_000, 100.5, 102.0, 100.0, 101.5, 1800.0],
        ]

    def fetch_ticker(self, symbol: str) -> dict:
        return {"symbol": symbol, "last": 101.5, "bid": 101.4, "ask": 101.6, "quoteVolume": 12_000_000}

    def fetch_order_book(self, symbol: str, limit: int = 100) -> dict:
        return {"bids": [[101.4, 10]], "asks": [[101.6, 10]]}

    def fetch_tickers(self) -> dict:
        return {"BTC/USDT": {"last": 101.5, "quoteVolume": 12_000_000}}


def test_paper_mode_can_use_live_exchange_market_data(monkeypatch) -> None:
    exchange = _FakeExchange()
    monkeypatch.setattr("data_manager.fetch_data.build_binance_spot_exchange", lambda config, require_auth=False: exchange)
    monkeypatch.setattr("data_manager.fetch_data.call_with_retry", lambda fn, retries=3, backoff_sec=0.5: fn())

    manager = BinanceDataManager(config={"data": {"paper_market_data_mode": "live"}}, paper=True)
    frame = manager.fetch_ohlcv("BTC/USDT", timeframe="15m", limit=2)
    ticker = manager.fetch_ticker("BTC/USDT")

    assert isinstance(frame, pd.DataFrame)
    assert float(frame.iloc[-1]["close"]) == 101.5
    assert float(ticker["last"]) == 101.5
    assert manager.paper_market_data_mode == "live"


def test_paper_live_mode_falls_back_to_synthetic_on_exchange_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_manager.fetch_data.build_binance_spot_exchange",
        lambda config, require_auth=False: (_ for _ in ()).throw(RuntimeError("no exchange")),
    )
    monkeypatch.setattr("data_manager.fetch_data.call_with_retry", lambda fn, retries=3, backoff_sec=0.5: fn())

    manager = BinanceDataManager(config={"data": {"paper_market_data_mode": "live"}}, paper=True)
    frame = manager.fetch_ohlcv("BTC/USDT", timeframe="15m", limit=5)
    ticker = manager.fetch_ticker("BTC/USDT")

    assert len(frame) == 5
    assert float(ticker.get("last", 0.0)) > 0.0
