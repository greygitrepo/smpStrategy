from __future__ import annotations

"""Strategy for discovering tradable symbols via the Bybit REST API."""

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.symbol_strategy_config import SymbolStrategyConfig
    from src.exchange.bybit_v5 import BybitV5Client

from .client import build_client
from .constants import (
    DEFAULT_CATEGORY,
    DEFAULT_DISCOVERY_INTERVAL,
    DEFAULT_SETTLE_COIN,
    DEFAULT_SYMBOL_LIMIT,
)
from .store import DiscoveredSymbolStore
from .strategies import SymbolFinder, create_symbol_finder

logger = logging.getLogger("smpStrategy.symbol.discovery")


def _maybe_load_symbol_strategy_config() -> Optional["SymbolStrategyConfig"]:
    from src.config.symbol_strategy_config import maybe_load_symbol_strategy_config

    return maybe_load_symbol_strategy_config()

class SymbolDiscoveryStrategy:
    """Fetch and maintain candidate symbols suitable for USDT perpetual trading."""

    def __init__(
        self,
        *,
        client: Optional["BybitV5Client"] = None,
        store: Optional[DiscoveredSymbolStore] = None,
        category: Optional[str] = None,
        settle_coin: Optional[str] = None,
        refresh_interval: Optional[float] = None,
        strategy_name: Optional[str] = None,
        limit: Optional[int] = None,
        strategy_config: Optional["SymbolStrategyConfig"] = None,
        volume_skip: Optional[int] = None,
    ) -> None:
        config = strategy_config or _maybe_load_symbol_strategy_config()

        self._category = (
            category
            or (config.category if config else None)
            or DEFAULT_CATEGORY
        )
        self._settle_coin = (
            settle_coin
            or (config.settle_coin if config else None)
            or DEFAULT_SETTLE_COIN
        )
        self.refresh_interval = (
            refresh_interval
            if refresh_interval is not None
            else (config.refresh_interval if config else None)
        ) or DEFAULT_DISCOVERY_INTERVAL
        self._strategy_name = (
            strategy_name
            or (config.strategy if config else None)
            or "top_volume"
        )
        self._limit = (
            limit if limit is not None else (config.limit if config else None)
        ) or DEFAULT_SYMBOL_LIMIT
        self._volume_skip = (
            volume_skip
            if volume_skip is not None
            else ((config.volume_skip if config else None) or 0)
        )

        self._client = client or build_client()
        self._store = store or DiscoveredSymbolStore(max_symbols=self._limit)
        if store is not None and hasattr(store, "max_symbols"):
            try:
                store.max_symbols = self._limit
            except (AttributeError, ValueError):
                logger.debug("Failed to set store max_symbols to %s", self._limit)
        self._finder: SymbolFinder = create_symbol_finder(
            self._strategy_name,
            self._client,
            category=self._category,
            settle_coin=self._settle_coin,
            limit=self._limit,
            volume_skip=self._volume_skip,
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def store(self) -> DiscoveredSymbolStore:
        return self._store

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    @property
    def limit(self) -> int:
        return self._limit

    def fetch_once(self) -> list[str]:
        candidates = self._finder.fetch()
        updated = self._store.update(candidates)
        logger.debug("Discovered symbols updated: %s", updated)
        return updated

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="SymbolDiscoveryStrategy",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.refresh_interval * 2)
        self._thread = None

    def _run(self) -> None:
        logger.info(
            "Starting symbol discovery thread: interval=%ss category=%s settleCoin=%s strategy=%s limit=%s",
            self.refresh_interval,
            self._category,
            self._settle_coin,
            self._strategy_name,
            self._limit,
        )
        while not self._stop_event.is_set():
            start = time.perf_counter()
            try:
                self.fetch_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Symbol discovery failed: %s", exc)
            elapsed = time.perf_counter() - start
            wait_time = max(0.0, self.refresh_interval - elapsed)
            if self._stop_event.wait(wait_time):
                break
        logger.info("Symbol discovery thread stopped")
