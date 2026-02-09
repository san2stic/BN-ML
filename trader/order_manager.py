from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bn_ml.exchange import build_binance_spot_exchange, call_with_retry
from bn_ml.types import Position


class OrderManager:
    def __init__(self, config: dict, paper: bool = True) -> None:
        self.config = config
        self.paper = paper
        self._exchange = None

    def _build_exchange(self):
        if self.paper:
            return None
        if self._exchange is not None:
            return self._exchange

        self._exchange = build_binance_spot_exchange(config=self.config, require_auth=True)
        call_with_retry(lambda: self._exchange.load_markets(), retries=3)
        return self._exchange

    def place_market_buy(self, symbol: str, size_usdt: float, price: float, atr: float) -> Position:
        stop_loss_mult = float(self.config.get("risk", {}).get("stop_loss_atr_mult", 1.5))
        stop_loss = price - stop_loss_mult * atr
        tp1 = price * 1.02
        tp2 = price * 1.04

        if self.paper:
            base_qty = float(size_usdt / max(price, 1e-9))
            return Position(
                symbol=symbol,
                side="LONG",
                size_usdt=float(size_usdt),
                entry_price=float(price),
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                opened_at=datetime.now(timezone.utc),
                extra={
                    "initial_size_usdt": float(size_usdt),
                    "initial_base_qty": base_qty,
                    "remaining_base_qty": base_qty,
                    "tp1_hit": False,
                    "tp2_hit": False,
                    "trailing_active": False,
                },
            )

        exchange = self._build_exchange()
        amount = self._amount_from_notional(exchange=exchange, symbol=symbol, size_usdt=size_usdt, price=price)
        order = call_with_retry(
            lambda: exchange.create_order(symbol=symbol, type="market", side="buy", amount=amount),
            retries=3,
        )

        avg_price = float(order.get("average") or order.get("price") or price)
        filled_base = float(order.get("filled") or amount)
        quote_cost = float(order.get("cost") or (avg_price * filled_base))

        return Position(
            symbol=symbol,
            side="LONG",
            size_usdt=quote_cost,
            entry_price=avg_price,
            stop_loss=avg_price - stop_loss_mult * atr,
            take_profit_1=avg_price * 1.02,
            take_profit_2=avg_price * 1.04,
            opened_at=datetime.now(timezone.utc),
            extra={
                "initial_size_usdt": quote_cost,
                "initial_base_qty": filled_base,
                "remaining_base_qty": filled_base,
                "tp1_hit": False,
                "tp2_hit": False,
                "trailing_active": False,
            },
        )

    def place_market_sell(
        self,
        symbol: str,
        price: float,
        size_usdt: float | None = None,
        base_qty: float | None = None,
    ) -> dict[str, Any]:
        if base_qty is None and size_usdt is None:
            raise ValueError("Either size_usdt or base_qty is required for SELL")

        if self.paper:
            if base_qty is None:
                base_qty = float(size_usdt or 0.0) / max(price, 1e-9)
            quote_size = float(base_qty) * float(price)
            return {
                "symbol": symbol,
                "side": "SELL",
                "status": "closed-paper",
                "size_usdt": quote_size,
                "base_qty": float(base_qty),
                "price": float(price),
            }

        exchange = self._build_exchange()
        amount = self._resolve_sell_amount(
            exchange=exchange,
            symbol=symbol,
            price=price,
            size_usdt=size_usdt,
            base_qty=base_qty,
        )

        order = call_with_retry(
            lambda: exchange.create_order(symbol=symbol, type="market", side="sell", amount=amount),
            retries=3,
        )

        avg_price = float(order.get("average") or order.get("price") or price)
        filled_base = float(order.get("filled") or amount)
        quote_size = float(order.get("cost") or (avg_price * filled_base))
        return {
            "symbol": symbol,
            "side": "SELL",
            "status": str(order.get("status") or "closed"),
            "size_usdt": quote_size,
            "base_qty": filled_base,
            "price": avg_price,
            "order_id": order.get("id"),
        }

    def _amount_from_notional(self, exchange, symbol: str, size_usdt: float, price: float) -> float:
        amount = max(float(size_usdt) / max(float(price), 1e-9), 0.0)
        return self._normalize_amount(exchange=exchange, symbol=symbol, amount=amount, price=float(price))

    def _resolve_sell_amount(
        self,
        exchange,
        symbol: str,
        price: float,
        size_usdt: float | None,
        base_qty: float | None,
    ) -> float:
        candidate = float(base_qty) if base_qty is not None else float(size_usdt or 0.0) / max(float(price), 1e-9)
        return self._normalize_amount(exchange=exchange, symbol=symbol, amount=candidate, price=float(price))

    def _normalize_amount(self, exchange, symbol: str, amount: float, price: float) -> float:
        precise = exchange.amount_to_precision(symbol, max(amount, 0.0))
        parsed = float(precise)
        limits = self._market_limits(exchange=exchange, symbol=symbol)
        self._validate_order_constraints(symbol=symbol, amount=parsed, price=price, limits=limits)
        return parsed

    @staticmethod
    def _validate_order_constraints(symbol: str, amount: float, price: float, limits: dict[str, float | None]) -> None:
        if amount <= 0:
            raise ValueError(f"Order size too small for market precision: {symbol}")

        min_qty = limits.get("min_qty")
        max_qty = limits.get("max_qty")
        min_notional = limits.get("min_notional")

        eps = 1e-12
        if min_qty is not None and amount + eps < min_qty:
            raise ValueError(f"Order amount below minQty for {symbol}: {amount} < {min_qty}")
        if max_qty is not None and amount - eps > max_qty:
            raise ValueError(f"Order amount above maxQty for {symbol}: {amount} > {max_qty}")

        notional = amount * max(price, 0.0)
        if min_notional is not None and notional + eps < min_notional:
            raise ValueError(f"Order notional below minNotional for {symbol}: {notional} < {min_notional}")

    @staticmethod
    def _market_limits(exchange, symbol: str) -> dict[str, float | None]:
        market = exchange.market(symbol)
        min_qty = OrderManager._to_float(market.get("limits", {}).get("amount", {}).get("min"))
        max_qty = OrderManager._to_float(market.get("limits", {}).get("amount", {}).get("max"))
        min_notional = OrderManager._to_float(market.get("limits", {}).get("cost", {}).get("min"))

        info = market.get("info", {})
        filters = info.get("filters", []) if isinstance(info, dict) else []
        for f in filters:
            ftype = str(f.get("filterType", ""))
            if ftype == "LOT_SIZE":
                min_qty = min_qty or OrderManager._to_float(f.get("minQty"))
                max_qty = max_qty or OrderManager._to_float(f.get("maxQty"))
            elif ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
                min_notional = min_notional or OrderManager._to_float(f.get("minNotional") or f.get("notional"))

        return {
            "min_qty": min_qty,
            "max_qty": max_qty,
            "min_notional": min_notional,
        }

    @staticmethod
    def _to_float(value) -> float | None:
        if value in (None, "", "0", 0):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
