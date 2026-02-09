from __future__ import annotations

from dataclasses import asdict

from bn_ml.state_store import StateStore
from bn_ml.domain_types import Position


class PositionManager:
    def __init__(self, store: StateStore | None = None) -> None:
        self.store = store
        self.positions: dict[str, Position] = {}
        self._hydrate_from_store()

    def _hydrate_from_store(self) -> None:
        if self.store is None:
            return
        for pos in self.store.load_open_positions():
            self.positions[pos.symbol] = pos

    def add(self, position: Position) -> None:
        self.positions[position.symbol] = position
        if self.store is not None:
            self.store.upsert_position(position)

    def update(self, position: Position) -> None:
        self.add(position)

    def remove(self, symbol: str) -> None:
        self.positions.pop(symbol, None)
        if self.store is not None:
            self.store.delete_position(symbol)

    def mark_closed(self, symbol: str) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return
        pos.status = "CLOSED"
        if self.store is not None:
            self.store.upsert_position(pos)
        self.positions.pop(symbol, None)

    def list_open(self) -> list[dict]:
        return [asdict(pos) for pos in self.positions.values() if pos.status == "OPEN"]

    def get_open_positions(self) -> list[Position]:
        return [pos for pos in self.positions.values() if pos.status == "OPEN"]

    def get(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def has_open(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return bool(pos and pos.status == "OPEN")
