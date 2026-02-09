from __future__ import annotations

import argparse

from bn_ml.config import load_config
from bn_ml.env import load_env_file
from bn_ml.state_store import StateStore
from data_manager.fetch_data import BinanceDataManager
from monitoring.logger import setup_logger
from trader.order_manager import OrderManager
from trader.position_manager import PositionManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergency kill switch: close all open positions")
    parser.add_argument("--config", default="configs/bot.yaml")
    parser.add_argument("--paper", action="store_true", help="Force paper mode")
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    config = load_config(args.config)
    paper = args.paper or config.get("environment", "paper") == "paper"

    logger = setup_logger(config)
    store = StateStore(db_path=str(config.get("storage", {}).get("sqlite_path", "artifacts/state/bn_ml.db")))
    data_manager = BinanceDataManager(config=config, paper=paper)
    order_manager = OrderManager(config=config, paper=paper)
    position_manager = PositionManager(store=store)

    open_positions = position_manager.get_open_positions()
    if not open_positions:
        logger.warning("Kill switch requested but no open positions were found.")
        return

    closed = 0
    for position in open_positions:
        symbol = position.symbol
        size_usdt = float(position.size_usdt)
        try:
            price = data_manager.fetch_last_price(symbol)
            remaining_base_qty = float(
                position.extra.get("remaining_base_qty", position.size_usdt / max(position.entry_price, 1e-9))
            )
            result = order_manager.place_market_sell(symbol=symbol, price=price, base_qty=remaining_base_qty)
            position_manager.mark_closed(symbol)
            store.insert_trade(
                symbol=symbol,
                side="SELL",
                size_usdt=float(result.get("size_usdt", size_usdt)),
                price=float(result.get("price", price)),
                mode="paper" if paper else "live",
                extra={
                    "reason": "kill-switch",
                    "status": result.get("status"),
                    "base_qty": float(result.get("base_qty", remaining_base_qty)),
                },
            )
            closed += 1
            logger.warning(
                "KILL-SWITCH CLOSE %s base=%.8f price=%.4f",
                symbol,
                float(result.get("base_qty", remaining_base_qty)),
                float(result.get("price", price)),
            )
        except Exception as exc:
            logger.exception("Kill-switch failed for %s: %s", symbol, exc)

    logger.warning("Kill switch completed: %s/%s positions closed", closed, len(open_positions))


if __name__ == "__main__":
    main()
