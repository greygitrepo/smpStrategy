from __future__ import annotations

"""Adaptive tuning helpers for new-listing strategy configuration."""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.new_listing_strategy_config import NewListingStrategyConfig

logger = logging.getLogger("smpStrategy.analytics.tuning")


class AdaptiveParameterManager:
    """Apply heuristic parameter tweaks based on rolling performance metrics."""

    def __init__(
        self,
        *,
        config: NewListingStrategyConfig,
        log_file: Path,
        min_total_trades: int = 30,
        min_gap_trades: int = 15,
    ) -> None:
        self._config = config
        self._log_file = log_file
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._min_total_trades = min_total_trades
        self._min_gap_trades = min_gap_trades
        self._last_update_trade_count = 0
        self._revision = 0

    def maybe_update(self, snapshot: dict[str, dict[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
        total = snapshot.get("total") or {}
        window = snapshot.get("window") or {}
        total_trades = int(total.get("trades", 0))
        if total_trades < self._min_total_trades:
            return None
        if total_trades - self._last_update_trade_count < self._min_gap_trades:
            return None
        win_rate = float(window.get("win_rate", 0.0))
        profit_factor = float(window.get("profit_factor", 0.0))
        avg_win = float(window.get("avg_win", 0.0))
        avg_loss = float(window.get("avg_loss", 0.0))
        net_pnl = float(window.get("net_pnl", 0.0))
        changes: Dict[str, Dict[str, float]] = {}
        if win_rate < 0.45 or profit_factor < 0.9 or net_pnl < 0:
            changes.update(self._adjust_for_drawdown(avg_win=avg_win, avg_loss=avg_loss))
        elif win_rate > 0.6 and profit_factor > 1.2 and net_pnl > 0:
            changes.update(self._adjust_for_strength(avg_win=avg_win, avg_loss=avg_loss))
        if not changes:
            return None
        self._last_update_trade_count = total_trades
        self._revision += 1
        self._log_revision(changes, snapshot)
        return changes

    def _adjust_for_drawdown(self, *, avg_win: float, avg_loss: float) -> Dict[str, Dict[str, float]]:
        changes: Dict[str, Dict[str, float]] = {}
        self._apply_change(
            "allocation_pct",
            max(self._config.allocation_pct * 0.9, 0.01),
            changes,
        )
        self._apply_change(
            "sl_pct",
            max(self._config.sl_pct * 0.9, 0.004),
            changes,
        )
        # tighten fallback trigger (require stronger momentum before fallback kicks in)
        self._apply_change(
            "fallback_threshold_pct",
            min(self._config.fallback_threshold_pct * 1.1, 0.12),
            changes,
        )
        # become more selective on ATR by lowering tolerance
        self._apply_change(
            "atr_skip_pct",
            max(self._config.atr_skip_pct * 0.9, 0.01),
            changes,
        )
        # widen TP slightly if winners are small relative to losers
        if avg_win and avg_loss and avg_win < avg_loss:
            self._apply_change(
                "tp_pct",
                max(self._config.tp_pct * 0.95, 0.006),
                changes,
            )
        return changes

    def _adjust_for_strength(self, *, avg_win: float, avg_loss: float) -> Dict[str, Dict[str, float]]:
        changes: Dict[str, Dict[str, float]] = {}
        self._apply_change(
            "allocation_pct",
            min(self._config.allocation_pct * 1.1, 0.18),
            changes,
        )
        self._apply_change(
            "tp_pct",
            min(self._config.tp_pct * 1.1, 0.06),
            changes,
        )
        # allow positions to breathe slightly more if losses are tightly cut
        if avg_loss and avg_loss < avg_win:
            self._apply_change(
                "sl_pct",
                min(self._config.sl_pct * 1.05, 0.05),
                changes,
            )
        # relax ATR filter marginally to capture more opportunities
        self._apply_change(
            "atr_skip_pct",
            min(self._config.atr_skip_pct * 1.05, 0.2),
            changes,
        )
        # fallback entries can be looser when momentum regime is strong
        self._apply_change(
            "fallback_threshold_pct",
            max(self._config.fallback_threshold_pct * 0.95, 0.01),
            changes,
        )
        return changes

    def _apply_change(
        self,
        field: str,
        new_value: float,
        changes: Dict[str, Dict[str, float]],
    ) -> None:
        old_value = getattr(self._config, field)
        if math.isclose(old_value, new_value, rel_tol=1e-6, abs_tol=1e-6):
            return
        setattr(self._config, field, new_value)
        changes[field] = {"old": old_value, "new": new_value}

    def _log_revision(
        self,
        changes: Dict[str, Dict[str, float]],
        snapshot: dict[str, dict[str, Any]],
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "revision": self._revision,
            "changes": changes,
            "metrics": snapshot,
        }
        try:
            with self._log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")
        except OSError as exc:
            logger.error("Failed to persist parameter tuning log to %s: %s", self._log_file, exc)


__all__ = ["AdaptiveParameterManager"]
