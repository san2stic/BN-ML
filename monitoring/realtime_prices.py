from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any


class BinanceRealtimePriceMonitor:
    def __init__(
        self,
        enabled: bool = True,
        max_symbols: int = 30,
        reconnect_delay_sec: float = 3.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_symbols = max(1, int(max_symbols))
        self.reconnect_delay_sec = max(0.5, float(reconnect_delay_sec))
        self.logger = logger or logging.getLogger("bn_ml.realtime")

        self._lock = threading.RLock()
        self._prices: dict[str, float] = {}
        self._stream_symbol_map: dict[str, str] = {}
        self._symbols: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws_app = None
        self._ws_unavailable_logged = False

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol).strip().upper()

    def _stream_for_symbol(self, symbol: str) -> str:
        return self._normalize_symbol(symbol).replace("/", "").lower() + "@miniTicker"

    def _build_url(self, symbols: list[str]) -> tuple[str, dict[str, str]]:
        streams: list[str] = []
        stream_symbol_map: dict[str, str] = {}
        for symbol in symbols[: self.max_symbols]:
            stream = self._stream_for_symbol(symbol)
            streams.append(stream)
            stream_symbol_map[stream.lower()] = self._normalize_symbol(symbol)
        return f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}", stream_symbol_map

    def _import_ws_module(self):
        try:
            import websocket  # type: ignore

            return websocket
        except Exception:
            return None

    def _on_message(self, _ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return
        stream = str(payload.get("stream", "")).lower()
        symbol = self._stream_symbol_map.get(stream)
        if not symbol:
            return
        raw_price = data.get("c") or data.get("C") or data.get("last")
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            return
        if price <= 0:
            return
        with self._lock:
            self._prices[symbol] = price

    def _on_error(self, _ws, error: Any) -> None:
        self.logger.warning("Realtime monitor error: %s", error)

    def _on_close(self, _ws, code: Any, msg: Any) -> None:
        if self._stop_event.is_set():
            return
        self.logger.info("Realtime monitor websocket closed code=%s msg=%s", code, msg)

    def _run(self, url: str) -> None:
        ws_mod = self._import_ws_module()
        if ws_mod is None:
            if not self._ws_unavailable_logged:
                self._ws_unavailable_logged = True
                self.logger.warning("websocket-client not installed. Realtime monitor disabled.")
            return

        while not self._stop_event.is_set():
            try:
                self._ws_app = ws_mod.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self.logger.warning("Realtime websocket loop failed: %s", exc)

            if self._stop_event.is_set():
                return
            time.sleep(self.reconnect_delay_sec)

    def _restart(self) -> bool:
        if not self.enabled:
            return False
        if not self._symbols:
            return False
        url, stream_symbol_map = self._build_url(self._symbols)
        self._stream_symbol_map = stream_symbol_map

        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(url,), daemon=True, name="bn-ml-realtime-ws")
        self._thread.start()
        return True

    def update_symbols(self, symbols: list[str]) -> bool:
        if not self.enabled:
            return False
        normalized: list[str] = []
        for symbol in symbols:
            sym = self._normalize_symbol(symbol)
            if sym and sym not in normalized:
                normalized.append(sym)
        normalized = normalized[: self.max_symbols]
        if not normalized:
            return False

        with self._lock:
            current = list(self._symbols)
            if current == normalized and self._thread is not None and self._thread.is_alive():
                return True
            self._symbols = normalized
        return self._restart()

    def get_price(self, symbol: str) -> float | None:
        with self._lock:
            return self._prices.get(self._normalize_symbol(symbol))

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._ws_app is not None:
                self._ws_app.close()
        except Exception:
            pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
