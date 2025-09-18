from __future__ import annotations

"""Concrete symbol-finding strategies for the discovery manager."""

import math
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Iterable

from .constants import DEFAULT_SETTLE_COIN

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.exchange.bybit_v5 import BybitV5Client


class SymbolFinder(ABC):
    """Abstract base class for symbol discovery strategies."""

    def __init__(
        self,
        client: "BybitV5Client",
        *,
        category: str,
        settle_coin: str = DEFAULT_SETTLE_COIN,
        limit: int,
    ) -> None:
        self._client = client
        self._category = category
        self._settle_coin = settle_coin
        self._limit = max(1, limit)

    @abstractmethod
    def fetch(self) -> list[str]:
        """Return a list of candidate symbols."""

    def _enforce_limit(self, symbols: Iterable[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            sym = symbol.upper()
            if sym in seen:
                continue
            seen.add(sym)
            unique.append(sym)
            if len(unique) >= self._limit:
                break
        return unique

    def _is_settle_match(self, symbol: str) -> bool:
        if not symbol:
            return False
        if self._settle_coin and symbol.upper().endswith(self._settle_coin.upper()):
            return True
        return not self._settle_coin


class TopVolumeSymbolFinder(SymbolFinder):
    """Pick symbols with the highest 24h turnover."""

    def fetch(self) -> list[str]:  # noqa: D401 - abstract impl
        response = self._client.get_tickers(category=self._category)
        tickers = response.get("result", {}).get("list") or []
        def _volume_key(item: dict[str, Any]) -> float:
            for key in ("turnover24h", "volume24h", "usdIndexPrice"):
                value = item.get(key)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return 0.0
        sorted_tickers = sorted(
            tickers,
            key=_volume_key,
            reverse=True,
        )
        symbols: list[str] = []
        for item in sorted_tickers:
            symbol = item.get("symbol")
            if not symbol:
                continue
            if self._settle_coin and not self._is_settle_match(symbol):
                continue
            symbols.append(symbol.upper())
        return self._enforce_limit(symbols)


class NewListingSymbolFinder(SymbolFinder):
    """Pick newest tradable symbols, excluding any containing digits."""

    _digit_pattern = re.compile(r"\d")

    def fetch(self) -> list[str]:  # noqa: D401 - abstract impl
        response = self._client.get_symbols(category=self._category)
        instruments = response.get("result", {}).get("list") or []
        def _launch_time(item: dict[str, Any]) -> float:
            for key in ("launchTime", "createdTime", "listTime"):
                value = item.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    try:
                        return float(str(value))
                    except (TypeError, ValueError):
                        continue
            return -math.inf
        filtered = [
            item
            for item in instruments
            if item.get("status") == "Trading"
            and (not self._settle_coin or item.get("settleCoin") == self._settle_coin)
            and item.get("symbol")
            and not self._digit_pattern.search(str(item.get("symbol")))
        ]
        sorted_instruments = sorted(filtered, key=_launch_time, reverse=True)
        symbols = [item["symbol"].upper() for item in sorted_instruments]
        return self._enforce_limit(symbols)


_STRATEGY_FACTORIES: dict[str, Callable[["BybitV5Client", str, str, int], SymbolFinder]] = {
    "top_volume": lambda client, category, settle_coin, limit: TopVolumeSymbolFinder(
        client,
        category=category,
        settle_coin=settle_coin,
        limit=limit,
    ),
    "new_listing": lambda client, category, settle_coin, limit: NewListingSymbolFinder(
        client,
        category=category,
        settle_coin=settle_coin,
        limit=limit,
    ),
}


def create_symbol_finder(
    name: str,
    client: "BybitV5Client",
    *,
    category: str,
    settle_coin: str,
    limit: int,
) -> SymbolFinder:
    factory = _STRATEGY_FACTORIES.get(name.lower())
    if factory is None:
        valid = ", ".join(sorted(_STRATEGY_FACTORIES))
        raise ValueError(f"Unknown symbol discovery strategy '{name}'. Valid options: {valid}")
    return factory(client, category, settle_coin, limit)


__all__ = [
    "SymbolFinder",
    "TopVolumeSymbolFinder",
    "NewListingSymbolFinder",
    "create_symbol_finder",
]
