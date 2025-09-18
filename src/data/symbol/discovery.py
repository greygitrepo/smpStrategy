from __future__ import annotations

"""Strategy for discovering tradable symbols via the Bybit REST API."""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from .client import build_client
from .constants import (
    DEFAULT_CATEGORY,
    DEFAULT_DISCOVERY_INTERVAL,
    DEFAULT_SETTLE_COIN,
)
from .store import DiscoveredSymbolStore

logger = logging.getLogger("smpStrategy.symbol.discovery")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.exchange.bybit_v5 import BybitV5Client


class SymbolDiscoveryStrategy:
    """Fetch and maintain candidate symbols suitable for USDT perpetual trading."""

    def __init__(
        self,
        *,
        client: Optional["BybitV5Client"] = None,
        store: Optional[DiscoveredSymbolStore] = None,
        category: str = DEFAULT_CATEGORY,
        settle_coin: str = DEFAULT_SETTLE_COIN,
        refresh_interval: float = DEFAULT_DISCOVERY_INTERVAL,
    ) -> None:
        self._client = client or build_client()
        self._store = store or DiscoveredSymbolStore()
        self._category = category
        self._settle_coin = settle_coin
        self.refresh_interval = refresh_interval

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def store(self) -> DiscoveredSymbolStore:
        return self._store

    def fetch_once(self) -> list[str]:
        response = self._client.get_symbols(category=self._category)
        candidates = self._filter_symbols(response)
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
            "Starting symbol discovery thread: interval=%ss category=%s settleCoin=%s",
            self.refresh_interval,
            self._category,
            self._settle_coin,
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

    def _filter_symbols(self, payload: dict[str, Any]) -> list[str]:
        result = payload.get("result") or {}
        instruments = result.get("list") or []
        selected: list[str] = []
        for item in instruments:
            if item.get("status") != "Trading":
                continue
            if item.get("settleCoin") != self._settle_coin:
                continue
            symbol = item.get("symbol")
            if not symbol:
                continue
            selected.append(symbol.upper())
        return selected
