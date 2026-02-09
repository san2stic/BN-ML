from __future__ import annotations


def symbol_to_model_key(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").upper()
