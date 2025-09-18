from __future__ import annotations

"""Data structures for representing position state."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class PositionSnapshot:
    """Lightweight view of an open derivatives position."""

    symbol: str
    size: float
    side: str
    entry_price: float
    leverage: float
    unrealized_pnl: float
    updated_at: datetime
    raw: dict[str, Any] = field(repr=False)
