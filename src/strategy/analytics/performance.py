from __future__ import annotations

"""Performance aggregation utilities for adaptive strategy tuning."""

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional

from .models import TradeRecord

logger = logging.getLogger("smpStrategy.analytics.performance")


@dataclass(slots=True)
class PerformanceSummary:
    trades: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    avg_win: float
    avg_loss: float
    net_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_return_pct: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "net_pnl": self.net_pnl,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "avg_return_pct": self.avg_return_pct,
        }


class PerformanceTracker:
    """Maintain rolling performance metrics backed by persisted trade history."""

    def __init__(
        self,
        *,
        history_file: Path,
        output_file: Path,
        window_size: int = 50,
    ) -> None:
        self._history_file = history_file
        self._output_file = output_file
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        self._window_size = max(1, window_size)
        self._total_records: list[TradeRecord] = []
        self._recent: Deque[TradeRecord] = deque(maxlen=self._window_size)
        self._load_history()

    def _load_history(self) -> None:
        if not self._history_file.exists():
            return
        try:
            with self._history_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = TradeRecord.from_dict(json.loads(line))
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed trade history line during load")
                        continue
                    self._total_records.append(record)
                    self._recent.append(record)
        except OSError as exc:
            logger.warning("Failed to preload trade history from %s: %s", self._history_file, exc)

    def record(self, records: Iterable[TradeRecord]) -> Optional[dict[str, dict[str, float | int]]]:
        added = False
        for record in records:
            self._total_records.append(record)
            self._recent.append(record)
            added = True
        if not added:
            return None
        total_summary = self._compute_summary(self._total_records)
        window_summary = self._compute_summary(list(self._recent))
        snapshot = {
            "total": total_summary.to_dict(),
            "window": window_summary.to_dict(),
        }
        self._write_snapshot(snapshot)
        return snapshot

    def _compute_summary(self, records: list[TradeRecord]) -> PerformanceSummary:
        trades = len(records)
        if trades == 0:
            return PerformanceSummary(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wins = 0
        losses = 0
        ties = 0
        sum_win = 0.0
        sum_loss = 0.0
        returns = []
        for record in records:
            pnl = record.realized_pnl
            if pnl > 0:
                wins += 1
                sum_win += pnl
            elif pnl < 0:
                losses += 1
                sum_loss += abs(pnl)
            else:
                ties += 1
            if record.return_pct is not None:
                returns.append(record.return_pct)
            elif record.notional:
                returns.append(pnl / record.notional)
        win_rate = wins / trades if trades else 0.0
        avg_win = (sum_win / wins) if wins else 0.0
        avg_loss = (sum_loss / losses) if losses else 0.0
        net_pnl = sum_win - sum_loss
        gross_profit = sum_win
        gross_loss = sum_loss
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf") if gross_profit else 0.0
        avg_return_pct = sum(returns) / len(returns) if returns else 0.0
        return PerformanceSummary(
            trades=trades,
            wins=wins,
            losses=losses,
            ties=ties,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            net_pnl=net_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            avg_return_pct=avg_return_pct,
        )

    def _write_snapshot(self, payload: dict[str, dict[str, float | int]]) -> None:
        try:
            with self._output_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except OSError as exc:
            logger.error("Failed to persist performance snapshot to %s: %s", self._output_file, exc)


__all__ = ["PerformanceTracker", "PerformanceSummary"]
