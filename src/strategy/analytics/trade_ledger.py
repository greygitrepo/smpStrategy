from __future__ import annotations

"""Utilities for persisting trade outcomes and detecting closed positions."""

import json
import logging
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.data.symbol.snapshots import PositionSnapshot
from src.exchange.bybit_v5 import BybitAPIError, BybitV5Client

from .models import LedgerEntry, TradeRecord

logger = logging.getLogger("smpStrategy.analytics.trade_ledger")


def _parse_timestamp(value: Optional[str | int | float]) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, (int, float)):
            # bybit returns milliseconds
            if value > 1_000_000_000_000:
                value = value / 1000.0
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            return datetime.fromisoformat(value)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to parse timestamp value: %s", value)
    return datetime.now(timezone.utc)


class TradeLedger:
    """Track open entries and persist realized trades when positions close."""

    def __init__(
        self,
        *,
        client: BybitV5Client,
        category: str,
        settle_coin: str,
        history_file: Path,
        max_history_cache: int = 500,
    ) -> None:
        self._client = client
        self._category = category
        self._settle_coin = settle_coin
        self._history_file = history_file
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._max_history_cache = max_history_cache
        self._lock = threading.Lock()
        self._open: Dict[str, LedgerEntry] = {}
        self._seen_close_ids: set[str] = set()
        self._load_existing_history()

    def _load_existing_history(self) -> None:
        if not self._history_file.exists():
            return
        try:
            with self._history_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed trade history line: %s", line)
                        continue
                    close_id = payload.get("close_id")
                    if close_id:
                        self._seen_close_ids.add(str(close_id))
        except OSError as exc:
            logger.warning("Failed to load trade history from %s: %s", self._history_file, exc)

    def bootstrap(self, positions: Dict[str, PositionSnapshot]) -> None:
        """Seed ledger with currently open positions so future closes are tracked."""
        with self._lock:
            for symbol, snapshot in positions.items():
                if abs(snapshot.size) <= 0:
                    continue
                self._open[symbol] = LedgerEntry(
                    symbol=symbol,
                    side=snapshot.side,
                    size=abs(snapshot.size),
                    entry_price=snapshot.entry_price,
                    notional=abs(snapshot.entry_price * snapshot.size),
                    margin_required=0.0,
                    order_id=None,
                    opened_at=snapshot.updated_at,
                    context={"bootstrap": True},
                )

    def register_entry(
        self,
        snapshot: PositionSnapshot,
        *,
        order_id: Optional[str],
        qty: float,
        notional: float,
        margin_required: float,
        context: Optional[dict[str, object]] = None,
    ) -> None:
        with self._lock:
            self._open[snapshot.symbol] = LedgerEntry(
                symbol=snapshot.symbol,
                side=snapshot.side,
                size=abs(snapshot.size) if snapshot.size else abs(qty),
                entry_price=snapshot.entry_price,
                notional=abs(notional),
                margin_required=margin_required,
                order_id=order_id,
                opened_at=snapshot.updated_at,
                context=context or {},
            )

    def open_entries(self) -> Dict[str, LedgerEntry]:
        with self._lock:
            return deepcopy(self._open)

    def sync(self, positions: Dict[str, PositionSnapshot]) -> List[TradeRecord]:
        """Detect closed symbols by diffing active positions and persist realized trades."""
        closed: List[TradeRecord] = []
        with self._lock:
            active_symbols = {sym.upper(): snap for sym, snap in positions.items() if abs(snap.size) > 0}
            # Detect closures
            for symbol, entry in list(self._open.items()):
                snapshot = active_symbols.get(symbol)
                if snapshot is not None:
                    # refresh entry size/price for ongoing trades
                    entry.size = abs(snapshot.size)
                    entry.entry_price = snapshot.entry_price
                    continue
                record = self._build_trade_record(entry)
                if record is not None:
                    closed.append(record)
                self._open.pop(symbol, None)
            # Add new positions that appeared without register_entry (manual opens)
            for symbol, snapshot in active_symbols.items():
                if symbol in self._open:
                    continue
                self._open[symbol] = LedgerEntry(
                    symbol=symbol,
                    side=snapshot.side,
                    size=abs(snapshot.size),
                    entry_price=snapshot.entry_price,
                    notional=abs(snapshot.entry_price * snapshot.size),
                    margin_required=0.0,
                    order_id=None,
                    opened_at=snapshot.updated_at,
                    context={"autodiscovered": True},
                )
        if closed:
            self._append_records(closed)
        return closed

    def _append_records(self, records: Iterable[TradeRecord]) -> None:
        try:
            with self._history_file.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=True))
                    handle.write("\n")
        except OSError as exc:
            logger.error("Failed to persist trade history to %s: %s", self._history_file, exc)

    def _build_trade_record(self, entry: LedgerEntry) -> Optional[TradeRecord]:
        try:
            response = self._client.get_closed_pnl(
                category=self._category,
                symbol=entry.symbol,
                settleCoin=self._settle_coin,
                limit=20,
            )
        except BybitAPIError as api_exc:
            logger.error(
                "Bybit API rejected closed PnL fetch for %s: %s",
                entry.symbol,
                api_exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error while fetching closed PnL for %s: %s",
                entry.symbol,
                exc,
            )
            return None
        records = response.get("result", {}).get("list") or []
        def _safe_timestamp(value: Any) -> datetime:
            parsed = _parse_timestamp(value)
            if parsed is None:
                return datetime.now(timezone.utc)
            return parsed

        for item in sorted(records, key=lambda it: _safe_timestamp(it.get("updatedTime")).timestamp(), reverse=True):
            close_id = str(item.get("execId") or item.get("orderId") or item.get("id") or item.get("tradeId") or "")
            if close_id and close_id in self._seen_close_ids:
                continue
            closed_at = _safe_timestamp(
                item.get("updatedTime")
                or item.get("updateTime")
                or item.get("closedTime")
                or item.get("closeTime")
                or item.get("timestamp")
            )
            raw_entry_price = item.get("avgEntryPrice") or item.get("avgPrice") or item.get("entryPrice") or item.get("price")
            try:
                entry_price = float(raw_entry_price)
            except (TypeError, ValueError):
                entry_price = entry.entry_price
            exit_price = float(item.get("avgExitPrice") or item.get("avgPrice") or item.get("lastPrice") or item.get("price") or 0.0)
            pnl = float(item.get("closedPnl") or item.get("realisedPnl") or item.get("pnl") or 0.0)
            qty = float(item.get("qty") or item.get("closedSize") or entry.size)
            metadata = {
                "context": entry.context,
                "raw": item,
            }
            created_time = (
                item.get("createdTime")
                or item.get("createTime")
                or item.get("openTime")
                or item.get("startTime")
            )
            opened_at = _safe_timestamp(created_time) if created_time else entry.opened_at
            if opened_at > closed_at and isinstance(entry.context, dict):
                for key in ("entry_timestamp", "decision_timestamp"):
                    ctx_value = entry.context.get(key)
                    if not ctx_value:
                        continue
                    ctx_parsed = _safe_timestamp(ctx_value)
                    if ctx_parsed <= closed_at:
                        opened_at = ctx_parsed
                        break
                else:
                    opened_at = entry.opened_at
            holding_seconds = max(0.0, (closed_at - opened_at).total_seconds())
            notional = entry.notional if entry.notional else abs(entry_price * qty)
            return_pct = pnl / notional if notional else None
            record = TradeRecord(
                symbol=entry.symbol,
                side=entry.side,
                size=qty,
                entry_price=entry_price,
                exit_price=exit_price,
                realized_pnl=pnl,
                notional=notional,
                opened_at=opened_at,
                closed_at=closed_at,
                holding_seconds=holding_seconds,
                order_id=entry.order_id,
                close_id=close_id or None,
                return_pct=return_pct,
                metadata=metadata,
            )
            if close_id:
                self._seen_close_ids.add(close_id)
            return record
        logger.debug("No new closed PnL entries found for %s", entry.symbol)
        return None


__all__ = ["TradeLedger"]
