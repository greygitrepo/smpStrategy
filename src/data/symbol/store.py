from __future__ import annotations

"""Container for managing discovered trading symbols."""

import threading
from typing import Iterable, Sequence

from .constants import DEFAULT_SYMBOL_LIMIT


class DiscoveredSymbolStore:
    """In-memory container for candidate trading symbols."""

    def __init__(
        self,
        *,
        max_symbols: int = DEFAULT_SYMBOL_LIMIT,
        exclusions: Iterable[str] | None = None,
    ) -> None:
        self._max_symbols = max_symbols
        self._exclusions = {sym.upper() for sym in (exclusions or [])}
        self._symbols: list[str] = []
        self._lock = threading.Lock()

    @property
    def max_symbols(self) -> int:
        return self._max_symbols

    @max_symbols.setter
    def max_symbols(self, value: int) -> None:
        if value <= 0:
            raise ValueError("max_symbols must be positive")
        with self._lock:
            self._max_symbols = value
            if len(self._symbols) > value:
                self._symbols = self._symbols[:value]

    def exclusions(self) -> set[str]:
        with self._lock:
            return set(self._exclusions)

    def add_exclusions(self, symbols: Iterable[str]) -> None:
        with self._lock:
            self._exclusions.update(sym.upper() for sym in symbols)
            self._symbols = [s for s in self._symbols if s not in self._exclusions]

    def remove_exclusions(self, symbols: Iterable[str]) -> None:
        with self._lock:
            for sym in symbols:
                self._exclusions.discard(sym.upper())

    def clear(self) -> None:
        with self._lock:
            self._symbols.clear()

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._symbols)

    def update(self, symbols: Sequence[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for sym in symbols:
            sym = sym.upper()
            if sym in seen or sym in self._exclusions:
                continue
            seen.add(sym)
            unique.append(sym)
            if len(unique) >= self._max_symbols:
                break
        with self._lock:
            self._symbols = unique
            return list(self._symbols)
