from __future__ import annotations

import os
import time
from typing import Any, Callable


class ExchangeError(RuntimeError):
    pass


def _must_get_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ExchangeError(f"Missing required environment variable: {name}")
    return value


def _env_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def build_binance_spot_exchange(config: dict, require_auth: bool) -> Any:
    import ccxt

    exchange_cfg = config.get("exchange", {})
    params: dict[str, Any] = {
        "enableRateLimit": True,
        "rateLimit": int(exchange_cfg.get("rate_limit_ms", 200)),
        "timeout": int(exchange_cfg.get("timeout_ms", 10_000)),
        "options": {"defaultType": "spot"},
    }

    if require_auth:
        params["apiKey"] = _must_get_env("BINANCE_API_KEY")
        params["secret"] = _must_get_env("BINANCE_API_SECRET")

    exchange = ccxt.binance(params)

    env_testnet = _env_bool("BINANCE_TESTNET")
    use_testnet = bool(exchange_cfg.get("testnet", True)) if env_testnet is None else env_testnet
    if use_testnet:
        exchange.set_sandbox_mode(True)

    return exchange


def call_with_retry(func: Callable[[], Any], retries: int = 3, backoff_sec: float = 0.5) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - relies on external API behavior
            last_error = exc
            if attempt >= retries - 1:
                break
            sleep_s = backoff_sec * (2**attempt)
            time.sleep(sleep_s)
    raise ExchangeError(str(last_error)) from last_error
