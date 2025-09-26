from __future__ import annotations

"""Dataclasses shared by strategy analytics utilities."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass(slots=True)
class LedgerEntry:
    """Cached view of an open position enhanced with strategy context."""

    symbol: str
    side: str
    size: float
    entry_price: float
    notional: float
    margin_required: float
    order_id: Optional[str]
    opened_at: datetime
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TradeRecord:
    """Realized trade outcome enriched with contextual metadata."""

    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    realized_pnl: float
    notional: float
    opened_at: datetime
    closed_at: datetime
    holding_seconds: float
    order_id: Optional[str] = None
    close_id: Optional[str] = None
    return_pct: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "realized_pnl": self.realized_pnl,
            "notional": self.notional,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "holding_seconds": self.holding_seconds,
            "order_id": self.order_id,
            "close_id": self.close_id,
            "return_pct": self.return_pct,
            "metadata": self.metadata,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TradeRecord":
        opened_at = cls._parse_datetime(payload.get("opened_at"))
        closed_at = cls._parse_datetime(payload.get("closed_at"))
        return cls(
            symbol=str(payload.get("symbol", "")).upper(),
            side=str(payload.get("side", "")),
            size=float(payload.get("size", 0.0) or 0.0),
            entry_price=float(payload.get("entry_price", 0.0) or 0.0),
            exit_price=float(payload.get("exit_price", 0.0) or 0.0),
            realized_pnl=float(payload.get("realized_pnl", 0.0) or 0.0),
            notional=float(payload.get("notional", 0.0) or 0.0),
            opened_at=opened_at,
            closed_at=closed_at,
            holding_seconds=float(payload.get("holding_seconds", 0.0) or 0.0),
            order_id=payload.get("order_id"),
            close_id=payload.get("close_id"),
            return_pct=float(payload.get("return_pct")) if payload.get("return_pct") is not None else None,
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            # assume unix timestamp in seconds
            return datetime.fromtimestamp(float(value), tz=datetime.now().astimezone().tzinfo)
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        # Fallback to naive UTC now to avoid crashing downstream metrics
        return datetime.utcnow()


__all__ = ["LedgerEntry", "TradeRecord"]
