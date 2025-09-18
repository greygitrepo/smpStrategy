from __future__ import annotations

"""Utilities for tracking active derivative positions."""

import logging
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

from .client import build_client
from .constants import DEFAULT_CATEGORY, DEFAULT_SETTLE_COIN
from .snapshots import PositionSnapshot

logger = logging.getLogger("smpStrategy.symbol.positions")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.exchange.bybit_v5 import BybitV5Client


class ActivePositionTracker:
    """Track currently open USDT-settled positions and keep snapshots of their state."""

    def __init__(
        self,
        *,
        client: "BybitV5Client" | None = None,
        category: str = DEFAULT_CATEGORY,
        settle_coin: str = DEFAULT_SETTLE_COIN,
    ) -> None:
        self._client = client or build_client()
        self._category = category
        self._settle_coin = settle_coin
        self._lock = threading.Lock()
        self._positions: Dict[str, PositionSnapshot] = {}
        # Initial load as requested
        self.refresh()

    def positions(self) -> Dict[str, PositionSnapshot]:
        with self._lock:
            return dict(self._positions)

    def refresh(self) -> Dict[str, PositionSnapshot]:
        response = self._client.get_positions(
            category=self._category,
            settleCoin=self._settle_coin,
        )
        positions = self._parse_positions(response)
        with self._lock:
            self._positions = positions
            return dict(self._positions)

    def handle_position_event(self, event: dict[str, Any]) -> PositionSnapshot | None:
        symbol = (event.get("symbol") or "").upper()
        if not symbol:
            return None
        size = float(event.get("size", 0) or 0)
        if size == 0:
            with self._lock:
                removed = self._positions.pop(symbol, None)
            if removed:
                logger.debug("Position closed for %s", symbol)
            return None
        snapshot = self._build_snapshot(event)
        with self._lock:
            self._positions[symbol] = snapshot
        logger.debug("Position updated via event: %s", snapshot)
        return snapshot

    def _parse_positions(self, payload: dict[str, Any]) -> Dict[str, PositionSnapshot]:
        result = payload.get("result") or {}
        entries = result.get("list") or []
        positions: Dict[str, PositionSnapshot] = {}
        for entry in entries:
            size = float(entry.get("size", 0) or 0)
            if size == 0:
                continue
            symbol = (entry.get("symbol") or "").upper()
            if not symbol:
                continue
            snapshot = self._build_snapshot(entry)
            positions[symbol] = snapshot
        return positions

    def _build_snapshot(self, data: dict[str, Any]) -> PositionSnapshot:
        def _to_float(val: Any) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        return PositionSnapshot(
            symbol=(data.get("symbol") or "").upper(),
            size=_to_float(data.get("size")),
            side=str(data.get("side", "")).capitalize(),
            entry_price=_to_float(data.get("avgPrice")),
            leverage=_to_float(data.get("leverage")),
            unrealized_pnl=_to_float(data.get("unrealisedPnl")),
            updated_at=datetime.now(timezone.utc),
            raw=data,
        )
