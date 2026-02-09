from __future__ import annotations

from scripts.run_bot import TradingRuntime


def test_resolve_configured_capital_total_numeric() -> None:
    value = TradingRuntime._resolve_configured_capital_total(250.5, paper=False, fallback=10_000.0)
    assert value == 250.5


def test_resolve_configured_capital_total_auto_live() -> None:
    value = TradingRuntime._resolve_configured_capital_total("auto", paper=False, fallback=10_000.0)
    assert value == 0.0


def test_resolve_configured_capital_total_auto_paper_fallback() -> None:
    value = TradingRuntime._resolve_configured_capital_total("binance", paper=True, fallback=321.0)
    assert value == 321.0


def test_resolve_configured_capital_total_invalid_fallback() -> None:
    value = TradingRuntime._resolve_configured_capital_total("not-a-number", paper=False, fallback=123.0)
    assert value == 123.0
