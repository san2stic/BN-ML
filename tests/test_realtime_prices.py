from __future__ import annotations

import json
import time

from monitoring.realtime_prices import BinanceRealtimePriceMonitor


def test_realtime_monitor_builds_stream_url() -> None:
    monitor = BinanceRealtimePriceMonitor(enabled=True, max_symbols=5)
    url, mapping = monitor._build_url(["BTC/USDT", "ETH/USDT"])

    assert "btcusdt@miniTicker" in url
    assert "ethusdt@miniTicker" in url
    assert mapping["btcusdt@miniticker"] == "BTC/USDT"


def test_realtime_monitor_updates_price_from_message() -> None:
    monitor = BinanceRealtimePriceMonitor(enabled=True)
    monitor._stream_symbol_map = {"btcusdt@miniticker": "BTC/USDT"}
    monitor._on_message(None, json.dumps({"stream": "btcusdt@miniTicker", "data": {"c": "101.23"}}))

    assert monitor.get_price("BTC/USDT") == 101.23


def test_realtime_monitor_start_and_stop_with_fake_ws(monkeypatch) -> None:
    class _FakeWSModule:
        class WebSocketApp:
            def __init__(self, url, on_message=None, on_error=None, on_close=None):
                self.url = url
                self.on_message = on_message
                self.on_error = on_error
                self.on_close = on_close

            def run_forever(self, ping_interval=20, ping_timeout=10):
                stream = self.url.split("streams=")[-1].split("/")[0]
                if self.on_message:
                    self.on_message(self, json.dumps({"stream": stream, "data": {"c": "111.11"}}))

            def close(self):
                return

    monitor = BinanceRealtimePriceMonitor(enabled=True, reconnect_delay_sec=0.5)
    monkeypatch.setattr(monitor, "_import_ws_module", lambda: _FakeWSModule)

    started = monitor.update_symbols(["BTC/USDT"])
    time.sleep(0.05)
    price = monitor.get_price("BTC/USDT")
    monitor.stop()

    assert started is True
    assert price == 111.11
