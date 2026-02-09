from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from bn_ml.exchange import build_binance_spot_exchange, call_with_retry


class BinanceDataManager:
    def __init__(self, config: dict, paper: bool = True) -> None:
        self.config = config
        self.paper = paper
        self._exchange = None

    def _build_exchange(self):
        if self.paper:
            return None
        if self._exchange is not None:
            return self._exchange

        self._exchange = build_binance_spot_exchange(config=self.config, require_auth=False)
        call_with_retry(lambda: self._exchange.load_markets(), retries=3)
        return self._exchange

    @staticmethod
    def _is_excluded_base_asset(base: str) -> bool:
        base_u = base.upper()
        if base_u.endswith(("UP", "DOWN", "BULL", "BEAR")):
            return True
        if base_u.startswith(("LD", "BEAR", "BULL")):
            return True
        # Stable-vs-stable "pairs" are usually not useful for this strategy universe.
        if base_u in {"USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USDP", "DAI", "USD1", "USDE"}:
            return True
        # Exclude fiat bases in quote universe mode.
        if base_u in {"EUR", "GBP", "TRY", "BRL", "AUD", "RUB", "UAH", "NGN", "PLN", "RON", "ZAR"}:
            return True
        return False

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def discover_pairs_by_quote(
        self,
        quote: str,
        min_quote_volume_usdt: float = 1_000_000.0,
        max_pairs: int = 150,
    ) -> list[str]:
        quote_u = str(quote).upper()
        configured_pairs = list(self.config.get("universe", {}).get("pairs", []))

        if self.paper:
            pairs = [p for p in configured_pairs if str(p).upper().endswith(f"/{quote_u}")]
            return pairs or configured_pairs

        exchange = self._build_exchange()
        tickers = call_with_retry(lambda: exchange.fetch_tickers(), retries=3)
        markets = getattr(exchange, "markets", {}) or {}

        ranked: list[tuple[str, float]] = []
        for symbol, ticker in tickers.items():
            if "/" not in symbol:
                continue

            market = markets.get(symbol, {})
            if market:
                if market.get("spot") is False:
                    continue
                if market.get("active") is False:
                    continue
                quote_sym = str(market.get("quote") or "").upper()
                base_sym = str(market.get("base") or "").upper()
            else:
                base_sym, quote_sym = symbol.split("/", 1)
                base_sym = str(base_sym).upper()
                quote_sym = str(quote_sym).upper()

            if quote_sym != quote_u:
                continue
            if self._is_excluded_base_asset(base_sym):
                continue

            quote_vol = self._safe_float(ticker.get("quoteVolume"))
            if quote_vol <= 0:
                last = self._safe_float(ticker.get("last") or ticker.get("close"))
                base_vol = self._safe_float(ticker.get("baseVolume"))
                quote_vol = last * base_vol

            if quote_vol < float(min_quote_volume_usdt):
                continue

            ranked.append((symbol, quote_vol))

        ranked.sort(key=lambda item: item[1], reverse=True)
        pairs = [sym for sym, _ in ranked]

        if max_pairs > 0:
            pairs = pairs[: int(max_pairs)]

        # Fallback to configured pairs if exchange discovery returned nothing.
        if not pairs:
            pairs = [p for p in configured_pairs if str(p).upper().endswith(f"/{quote_u}")]
            pairs = pairs or configured_pairs
        return pairs

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> pd.DataFrame:
        if self.paper:
            return self._generate_synthetic_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

        exchange = self._build_exchange()
        raw = call_with_retry(lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit), retries=3)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        if self.paper:
            frame = self._generate_synthetic_ohlcv(symbol=symbol, timeframe="1m", limit=2)
            last = float(frame.iloc[-1]["close"])
            quote_vol = float(frame["close"].mul(frame["volume"]).sum())
            return {
                "symbol": symbol,
                "last": last,
                "bid": last * 0.9998,
                "ask": last * 1.0002,
                "quoteVolume": quote_vol,
            }

        exchange = self._build_exchange()
        return call_with_retry(lambda: exchange.fetch_ticker(symbol), retries=3)

    def fetch_last_price(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        return float(ticker.get("last") or ticker.get("close") or 0.0)

    def fetch_spread_pct(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        mid = (bid + ask) / 2 if bid and ask else float(ticker.get("last") or 0.0)

        if mid <= 0:
            return 999.0

        if bid <= 0 or ask <= 0:
            return 0.15

        return ((ask - bid) / mid) * 100

    def fetch_orderbook_depth_usdt(self, symbol: str, depth_pct: float = 0.5) -> float:
        if self.paper:
            return 200_000.0

        exchange = self._build_exchange()
        orderbook = call_with_retry(lambda: exchange.fetch_order_book(symbol, limit=100), retries=3)

        ticker = self.fetch_ticker(symbol)
        mid = float(ticker.get("last") or 0.0)
        if mid <= 0:
            return 0.0

        lower = mid * (1 - depth_pct / 100)
        upper = mid * (1 + depth_pct / 100)

        depth = 0.0
        for price, amount in orderbook.get("bids", []):
            if float(price) >= lower:
                depth += float(price) * float(amount)
        for price, amount in orderbook.get("asks", []):
            if float(price) <= upper:
                depth += float(price) * float(amount)
        return depth

    def fetch_quote_volume_24h(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        quote_vol = ticker.get("quoteVolume")
        if quote_vol is not None:
            return float(quote_vol)

        last = float(ticker.get("last") or 0.0)
        base_vol = float(ticker.get("baseVolume") or 0.0)
        return last * base_vol

    def estimate_correlation_with_btc(self, symbol: str, timeframe: str = "15m", limit: int = 300) -> float:
        if symbol == "BTC/USDT":
            return 1.0

        if self.paper:
            return 0.5

        target = self.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        btc = self.fetch_ohlcv(symbol="BTC/USDT", timeframe=timeframe, limit=limit)

        target_ret = target["close"].pct_change().dropna()
        btc_ret = btc["close"].pct_change().dropna()

        m = min(len(target_ret), len(btc_ret))
        if m < 30:
            return 0.0

        corr = float(np.corrcoef(target_ret.iloc[-m:], btc_ret.iloc[-m:])[0, 1])
        if np.isnan(corr):
            return 0.0
        return corr

    @staticmethod
    def _generate_synthetic_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        seed = abs(hash((symbol, timeframe))) % (2**32)
        rng = np.random.default_rng(seed)

        step_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        minutes = step_map.get(timeframe, 15)

        end_ts = datetime.now(timezone.utc)
        timestamps = [end_ts - timedelta(minutes=minutes * i) for i in range(limit)][::-1]

        base_price = 100 + (seed % 900)
        returns = rng.normal(0, 0.002, size=limit)
        close = base_price * np.cumprod(1 + returns)
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        spread = np.maximum(close * 0.002, 0.01)
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.uniform(5_000, 100_000, size=limit)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
