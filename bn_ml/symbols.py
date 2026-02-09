from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def symbol_to_model_key(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").upper()


def normalize_symbol(symbol: Any) -> str:
    return str(symbol).strip().upper()


def normalize_symbols(symbols: Iterable[Any] | str | None) -> list[str]:
    if symbols is None:
        return []

    if isinstance(symbols, str):
        raw_items = symbols.split(",")
    else:
        raw_items = list(symbols)

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        symbol = normalize_symbol(raw)
        if not symbol or symbol in seen:
            continue
        normalized.append(symbol)
        seen.add(symbol)
    return normalized
