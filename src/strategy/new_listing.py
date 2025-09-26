from __future__ import annotations

"""New listing trading strategy implementation."""

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from src.config.new_listing_strategy_config import (
    NewListingStrategyConfig,
    TimeframeRequirement,
    default_new_listing_strategy_config,
    maybe_load_new_listing_strategy_config,
)
from src.data.symbol.positions import ActivePositionTracker
from src.data.symbol.snapshots import PositionSnapshot
from src.data.symbol.discovery import SymbolDiscoveryStrategy
from src.data.wallet_manager import WalletDataManager
from src.exchange.bybit_v5 import BybitAPIError, BybitV5Client
from src.strategy.analytics import (
    AdaptiveParameterManager,
    PerformanceTracker,
    TradeLedger,
)

logger = logging.getLogger("smpStrategy.strategy.new_listing")

_EXCLUSION_FILE_NAME = "newListingStrategy.exclusions.json"


@dataclass(slots=True)
class Candle:
    """Normalized candle payload used for EMA and fallback calculations."""

    start: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class EntryDecision:
    """Result of an entry attempt including sizing and dynamic TP/SL."""

    notional: float
    margin_required: float
    tp_pct: float
    sl_pct: float
    qty: float
    side: str
    order_id: Optional[str] = None


@dataclass(slots=True)
class TimeframeDataResult:
    """Container for collected timeframe data and chosen requirement block."""

    requirement: TimeframeRequirement
    frames: dict[str, list[Candle]]


class NewListingTradingStrategy:
    """Execute the new-listing momentum strategy once per invocation."""

    def __init__(
        self,
        *,
        client: BybitV5Client,
        discovery: SymbolDiscoveryStrategy,
        wallet_manager: WalletDataManager,
        position_tracker: ActivePositionTracker,
        config: Optional[NewListingStrategyConfig] = None,
        category: Optional[str] = None,
    ) -> None:
        resolved_config = config
        if resolved_config is None:
            resolved_config = maybe_load_new_listing_strategy_config()
        if resolved_config is None:
            resolved_config = default_new_listing_strategy_config()
            logger.warning("Falling back to in-memory defaults for new-listing strategy config")
        if not resolved_config.requirements:
            raise ValueError("New listing strategy requires at least one timeframe requirement block")
        self._client = client
        self._discovery = discovery
        self._wallet_manager = wallet_manager
        self._position_tracker = position_tracker
        self._config = resolved_config
        self._category = category or getattr(client, "default_category", None) or "linear"
        self._leverage_cache: set[str] = set()
        self._static_exclusions: set[str] = {sym.upper() for sym in self._config.exclude_symbols}
        self._dynamic_exclusions: set[str] = set()
        self._excluded_symbols: set[str] = set(self._static_exclusions)
        self._exclusion_file = self._resolve_exclusion_file()
        self._load_persisted_exclusions()
        if self._static_exclusions:
            logger.info("Static exclusions configured: %s", ", ".join(sorted(self._static_exclusions)))
        analytics_root = Path(__file__).resolve().parents[2] / "analytics"
        analytics_root.mkdir(parents=True, exist_ok=True)
        history_file = analytics_root / "trade_history.jsonl"
        performance_file = analytics_root / "performance_snapshot.json"
        tuning_log_file = analytics_root / "parameter_tuning.jsonl"
        optimized_config_path = analytics_root / "newListingStrategy_optimized.ini"
        optimizer_root = analytics_root / "optimizer"
        optimizer_root.mkdir(parents=True, exist_ok=True)
        settle_coin = getattr(position_tracker, "settle_coin", "USDT")
        self._trade_ledger = TradeLedger(
            client=self._client,
            category=self._category,
            settle_coin=settle_coin,
            history_file=history_file,
        )
        self._performance_tracker = PerformanceTracker(
            history_file=history_file,
            output_file=performance_file,
        )
        self._parameter_manager = AdaptiveParameterManager(
            config=self._config,
            log_file=tuning_log_file,
            model_dir=optimizer_root,
            optimized_config_path=optimized_config_path,
            evaluation_trades=20,
            evaluation_duration=timedelta(hours=2),
            candidate_trade_interval=20,
            candidate_time_interval=timedelta(hours=2),
        )
        self._trade_ledger.bootstrap(position_tracker.positions())

    def run_once(self, *, initial_candidates: Optional[Iterable[str]] = None) -> None:
        if not self._config.enabled:
            logger.info("New listing strategy disabled in config; skipping execution")
            return
        logger.debug("Starting new listing strategy run: category=%s", self._category)
        positions = self._safe_refresh_positions()
        self._update_analytics_after_refresh(positions)
        wallet_snapshot = self._wallet_manager.fetch_once()
        available = wallet_snapshot.available_balance
        if available <= 0:
            logger.info(
                "Available balance is non-positive; wallet_available=%s", available
            )
            return
        if initial_candidates is not None:
            symbols = [sym.upper() for sym in initial_candidates]
        else:
            symbols = self._fetch_candidate_symbols()
        if not symbols:
            logger.info("No candidate symbols retrieved from discovery store")
            return
        filtered_symbols: list[str] = []
        for symbol in symbols:
            upper_symbol = symbol.upper()
            if upper_symbol in self._excluded_symbols:
                logger.debug("Skipping %s because it is excluded", upper_symbol)
                continue
            filtered_symbols.append(upper_symbol)
        if not filtered_symbols:
            logger.info(
                "All candidate symbols are excluded; skipping run (exclusions=%s)",
                ", ".join(sorted(self._excluded_symbols)) or "<empty>",
            )
            return
        symbols = filtered_symbols
        logger.info(
            "Evaluating %s candidates (available=%s)",
            len(symbols),
            available,
        )
        active_symbols = {sym.upper() for sym in positions}
        remaining_margin = available
        max_new_positions = max(0, self._config.max_new_positions)
        open_slots = max(0, max_new_positions - len(active_symbols))
        if open_slots == 0:
            logger.info(
                "No capacity for new positions (open=%s max=%s); skipping run",
                len(active_symbols),
                max_new_positions,
            )
            return
        max_considered_slots = min(open_slots, len(symbols))
        if max_considered_slots <= 0:
            logger.debug("No eligible symbols after filtering; skipping entry evaluation")
            return
        allocation_fraction = max(0.0, self._config.allocation_pct)
        new_positions_opened = 0
        for idx, symbol in enumerate(symbols):
            symbol = symbol.upper()
            logger.debug("[%s/%s] Processing symbol %s", idx + 1, len(symbols), symbol)
            if symbol in active_symbols:
                logger.debug("Skipping %s because a position is already open", symbol)
                continue
            if new_positions_opened >= open_slots:
                logger.info(
                    "Reached available slot limit (%s remaining slots); stopping evaluation loop",
                    open_slots,
                )
                break
            slots_remaining = max_considered_slots - new_positions_opened
            if slots_remaining <= 0:
                logger.debug("No slots remaining after updates; breaking out")
                break
            per_slot_margin = remaining_margin * allocation_fraction
            per_slot_margin = min(per_slot_margin, remaining_margin / slots_remaining)
            per_slot_margin = max(0.0, per_slot_margin)
            logger.debug(
                "Slot margin target for %s: %.6f (remaining=%.6f slots_left=%s)",
                symbol,
                per_slot_margin,
                remaining_margin,
                slots_remaining,
            )
            logger.debug(
                "Attempting entry for %s with remaining_margin=%.6f",
                symbol,
                remaining_margin,
            )
            try:
                entry = self._evaluate_symbol(symbol, remaining_margin, per_slot_margin)
            except Exception as exc:  # noqa: BLE001 - log and continue per symbol
                logger.exception("Failed to evaluate symbol %s: %s", symbol, exc)
                continue
            if entry is None:
                # None indicates that evaluation aborted without a decision
                continue
            notional_used = entry.notional
            margin_used = entry.margin_required
            if notional_used <= 0 or margin_used <= 0:
                logger.debug("No order placed for %s", symbol)
                continue
            remaining_margin = max(0.0, remaining_margin - margin_used)
            active_symbols.add(symbol)
            new_positions_opened += 1
            logger.info(
                "Order placed for %s using notional=%.6f (margin_used=%.6f leverage=%s); remaining_margin=%.6f open_slots_remaining=%s",
                symbol,
                notional_used,
                margin_used,
                self._config.leverage,
                remaining_margin,
                open_slots - new_positions_opened,
            )
            if remaining_margin <= 0:
                logger.info("Reached capital allocation margin limit; stopping evaluation loop")
                break

    # ----- Position management helpers -----
    def _safe_refresh_positions(self) -> dict[str, PositionSnapshot]:
        try:
            refreshed = self._position_tracker.refresh()
            logger.debug("Refreshed positions: %s", list(refreshed))
            return refreshed
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to refresh positions; using last known snapshot: %s", exc)
            return self._position_tracker.positions()

    def _wait_for_fill(
        self,
        *,
        symbol: str,
        side: str,
        expected_qty: float,
        order_id: Optional[str],
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.5,
    ) -> Optional[PositionSnapshot]:
        if expected_qty <= 0:
            return None
        normalized_symbol = symbol.upper()
        normalized_side = side.lower()
        deadline = time.monotonic() + max(timeout_s, 1.0)
        poll_interval = max(poll_interval_s, 0.1)
        while time.monotonic() < deadline:
            try:
                positions = self._position_tracker.refresh()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Failed to refresh positions while waiting for fill on %s: %s",
                    normalized_symbol,
                    exc,
                )
                positions = self._position_tracker.positions()
            snapshot = positions.get(normalized_symbol)
            if snapshot and snapshot.side.lower() == normalized_side and abs(snapshot.size) > 0:
                logger.debug(
                    "Fill confirmed for %s order_id=%s size=%.8f entry=%.6f",
                    normalized_symbol,
                    order_id,
                    snapshot.size,
                    snapshot.entry_price,
                )
                return snapshot
            time.sleep(poll_interval)
        logger.warning(
            "Timed out waiting for %s order %s to fill; TP/SL not submitted",
            normalized_symbol,
            order_id or "<unknown>",
        )
        return None

    def _apply_trading_stop(
        self,
        snapshot: PositionSnapshot,
        *,
        instrument: dict[str, float],
        tp_pct: Optional[float] = None,
        sl_pct: Optional[float] = None,
    ) -> None:
        entry_price = snapshot.entry_price
        if entry_price <= 0:
            logger.debug(
                "Skipping TP/SL for %s because entry price is non-positive",
                snapshot.symbol,
            )
            return
        tp_pct = self._config.tp_pct if tp_pct is None else max(0.0, tp_pct)
        sl_pct = self._config.sl_pct if sl_pct is None else max(0.0, sl_pct)
        if tp_pct == 0.0 and sl_pct == 0.0:
            logger.debug("TP/SL percentages are zero; nothing to configure for %s", snapshot.symbol)
            return
        tick_size = instrument.get("tick_size") or 0.0
        side = snapshot.side.lower()
        tp_price: Optional[float]
        sl_price: Optional[float]
        if side == "buy":
            tp_price = entry_price * (1 + tp_pct) if tp_pct > 0 else None
            sl_price = entry_price * (1 - sl_pct) if sl_pct > 0 else None
        elif side == "sell":
            tp_price = entry_price * (1 - tp_pct) if tp_pct > 0 else None
            sl_price = entry_price * (1 + sl_pct) if sl_pct > 0 else None
        else:
            logger.debug("Unknown position side for %s: %s", snapshot.symbol, snapshot.side)
            return
        if tp_price is not None:
            tp_price = self._quantize_price(tp_price, tick_size)
        if sl_price is not None:
            sl_price = self._quantize_price(sl_price, tick_size)
        try:
            self._client.set_trading_stop(
                symbol=snapshot.symbol,
                takeProfit=tp_price,
                stopLoss=sl_price,
                tpTriggerBy="LastPrice" if tp_price is not None else None,
                slTriggerBy="LastPrice" if sl_price is not None else None,
                category=self._category,
            )
            logger.info(
                "Configured TP/SL for %s: tp=%s sl=%s",
                snapshot.symbol,
                f"{tp_price:.6f}" if tp_price is not None else "<disabled>",
                f"{sl_price:.6f}" if sl_price is not None else "<disabled>",
            )
        except BybitAPIError as api_exc:
            logger.error(
                "Bybit API rejected TP/SL for %s: %s",
                snapshot.symbol,
                api_exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error while configuring TP/SL for %s: %s",
                snapshot.symbol,
                exc,
            )

    @staticmethod
    def _quantize_price(price: float, tick_size: float) -> float:
        if tick_size and tick_size > 0:
            steps = round(price / tick_size)
            aligned = steps * tick_size
        else:
            aligned = price
        return round(aligned, 10)

    def _exclude_symbol(self, symbol: str, reason: str) -> None:
        if symbol in self._excluded_symbols:
            logger.debug("Symbol %s already excluded: %s", symbol, reason)
            return
        if symbol in self._static_exclusions:
            logger.debug("Symbol %s is statically excluded; reason=%s", symbol, reason)
            return
        self._dynamic_exclusions.add(symbol)
        self._excluded_symbols.add(symbol)
        logger.warning("Excluding %s from discovery: %s", symbol, reason)
        self._persist_exclusions()

    def _resolve_exclusion_file(self) -> Path:
        return Path(__file__).resolve().parents[2] / "config" / _EXCLUSION_FILE_NAME

    def _load_persisted_exclusions(self) -> None:
        path = self._exclusion_file
        try:
            if not path.exists():
                return
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load persisted exclusions from %s: %s", path, exc)
            return
        if isinstance(payload, list):
            dynamic = {str(value).upper() for value in payload if str(value).strip()}
            self._dynamic_exclusions.update(dynamic)
            self._excluded_symbols.update(dynamic)
            if dynamic:
                logger.info(
                    "Loaded %s persisted exclusions: %s",
                    len(dynamic),
                    ", ".join(sorted(dynamic)),
                )
        else:
            logger.warning(
                "Unexpected exclusion payload in %s (expected list, got %s)",
                path,
                type(payload),
            )

    def _persist_exclusions(self) -> None:
        path = self._exclusion_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    sorted(self._dynamic_exclusions),
                    handle,
                    ensure_ascii=True,
                    indent=2,
                )
        except OSError as exc:
            logger.error("Failed to persist exclusions to %s: %s", path, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while persisting exclusions to %s: %s", path, exc)

    def _log_decision_snapshot(self, payload: dict[str, Any]) -> None:
        try:
            message = json.dumps(payload, default=self._json_default, ensure_ascii=True)
        except TypeError:
            logger.debug("Failed to serialize decision snapshot payload; emitting fallback repr")
            message = repr(payload)
        logger.info("DecisionSnapshot %s", message)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, set):
            return sorted(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

    def _register_entry(
        self,
        snapshot: PositionSnapshot,
        *,
        order_id: Optional[str],
        qty: float,
        notional: float,
        margin_required: float,
        decision_context: dict[str, Any],
        tp_pct: float,
        sl_pct: float,
    ) -> None:
        if not hasattr(self, "_trade_ledger"):
            return
        context = dict(decision_context)
        context.update(
            {
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "notional": notional,
                "margin_required": margin_required,
                "order_id": order_id,
                "allocation_pct": decision_context.get("allocation_pct", self._config.allocation_pct),
                "atr_skip_pct": decision_context.get("atr_skip_pct", self._config.atr_skip_pct),
                "ema_period": decision_context.get("ema_period", self._config.ema_period),
                "weight_5m": decision_context.get("weight_5m", self._config.weight_5m),
                "weight_15m": decision_context.get("weight_15m", self._config.weight_15m),
                "weight_30m": decision_context.get("weight_30m", self._config.weight_30m),
                "fallback_threshold_pct": decision_context.get(
                    "fallback_threshold_pct", self._config.fallback_threshold_pct
                ),
            }
        )
        self._trade_ledger.register_entry(
            snapshot,
            order_id=order_id,
            qty=qty,
            notional=notional,
            margin_required=margin_required,
            context=context,
        )

    def _update_analytics_after_refresh(
        self,
        positions: dict[str, PositionSnapshot],
    ) -> None:
        if not hasattr(self, "_trade_ledger"):
            return
        try:
            closed_records = self._trade_ledger.sync(positions)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Trade ledger sync failed: %s", exc)
            closed_records = []
        snapshot = None
        if closed_records:
            try:
                snapshot = self._performance_tracker.record(closed_records)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Performance tracker update failed: %s", exc)
        try:
            self._parameter_manager.process_trades(closed_records, snapshot)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Parameter manager processing failed: %s", exc)

    # ----- Symbol evaluation -----
    def _fetch_candidate_symbols(self) -> list[str]:
        try:
            return self._discovery.fetch_once()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Symbol discovery fetch failed: %s", exc)
            return []

    def _evaluate_symbol(
        self,
        symbol: str,
        available: float,
        margin_cap: float,
    ) -> Optional[EntryDecision]:
        decision_details: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "available_margin": round(available, 8),
            "margin_cap": round(margin_cap, 8),
        }

        def _skip(reason: str, extra: Optional[dict[str, Any]] = None) -> Optional[EntryDecision]:
            payload = dict(decision_details)
            payload["status"] = "skipped"
            payload["reason"] = reason
            if extra:
                payload.update(extra)
            self._log_decision_snapshot(payload)
            return None

        if available <= 0:
            logger.debug("Skipping %s because available balance is exhausted", symbol)
            return _skip("available_balance_exhausted")
        if margin_cap <= 0:
            logger.debug("Skipping %s because margin cap is non-positive", symbol)
            return _skip("margin_cap_non_positive")
        timeframe_result = self._collect_timeframe_data(symbol)
        if timeframe_result is None:
            return _skip("timeframe_data_unavailable")
        decision_details["requirement"] = timeframe_result.requirement.name
        frames = timeframe_result.frames
        if not frames:
            return _skip("timeframe_data_incomplete")
        decision_details["candles"] = {tf: len(frames.get(tf, [])) for tf in ("5m", "15m", "30m")}
        candles_5m = frames.get("5m", [])
        atr_ratio = self._compute_atr_ratio(candles_5m, self._config.atr_period)
        if atr_ratio is None:
            logger.debug("ATR unavailable for %s; insufficient candles", symbol)
            return _skip("atr_unavailable")
        decision_details["atr"] = {
            "ratio": round(atr_ratio, 6),
            "period": self._config.atr_period,
            "skip_pct": self._config.atr_skip_pct,
        }
        if self._config.atr_skip_pct > 0 and atr_ratio > self._config.atr_skip_pct:
            logger.info(
                "Skipping %s due to high volatility (ATR=%.2f%% threshold=%.2f%%)",
                symbol,
                atr_ratio * 100.0,
                self._config.atr_skip_pct * 100.0,
            )
            return _skip("volatility_exceeds_threshold")
        trend, trend_meta = self._determine_trend(symbol, frames)
        if trend is None:
            logger.info("Trend indeterminate for %s; skipping entry (EMA score missing)", symbol)
            return _skip("trend_indeterminate", {"trend": trend_meta})
        logger.debug("Trend for %s resolved to %s", symbol, trend)
        trend_info = dict(trend_meta)
        trend_info["direction"] = trend
        decision_details["trend"] = trend_info
        decision_context = {
            "symbol": symbol,
            "trend": trend_info,
            "atr_ratio": atr_ratio,
            "requirement": timeframe_result.requirement.name,
            "candles": decision_details.get("candles", {}),
            "allocation_pct": self._config.allocation_pct,
            "atr_skip_pct": self._config.atr_skip_pct,
            "ema_period": self._config.ema_period,
            "weight_5m": self._config.weight_5m,
            "weight_15m": self._config.weight_15m,
            "weight_30m": self._config.weight_30m,
            "fallback_threshold_pct": self._config.fallback_threshold_pct,
        }
        entry = self._attempt_entry(
            symbol,
            trend,
            available,
            margin_cap,
            frames,
            atr_ratio,
            decision_context,
        )
        if entry is None:
            logger.debug("Entry attempt for %s returned no trade", symbol)
            return _skip("entry_not_placed", {"trend": trend_info})
        payload = dict(decision_details)
        payload["status"] = "entered"
        payload["entry"] = {
            "notional": round(entry.notional, 6),
            "margin_required": round(entry.margin_required, 6),
            "tp_pct": round(entry.tp_pct, 6),
            "sl_pct": round(entry.sl_pct, 6),
            "qty": round(entry.qty, 8),
            "side": entry.side,
            "order_id": entry.order_id,
        }
        self._log_decision_snapshot(payload)
        return entry

    def _collect_timeframe_data(
        self,
        symbol: str,
    ) -> Optional[TimeframeDataResult]:
        requirements = self._config.requirements
        max_required_30 = max(req.min_30m for req in requirements)
        base_limit_30 = max(max_required_30, self._config.ema_period + 2)
        candles_30 = self._fetch_candles(symbol, "30", base_limit_30)
        if not candles_30:
            logger.debug("No 30m candles retrieved for %s", symbol)
            return None
        available_30 = len(candles_30)
        requirement = self._select_requirement(available_30, requirements)
        if requirement is None:
            logger.debug(
                "No matching requirement block for %s (available_30=%s)",
                symbol,
                available_30,
            )
            return None
        logger.debug(
            "Requirement '%s' selected for %s (available_30=%s)",
            requirement.name,
            symbol,
            available_30,
        )
        candles_30 = candles_30[-max(requirement.min_30m, self._config.ema_period + 1) :]
        candles_15 = self._fetch_candles(
            symbol,
            "15",
            max(requirement.min_15m, self._config.ema_period + 1),
        )
        min_5m_required = max(requirement.min_5m, self._config.min_5m_bars)
        candles_5 = self._fetch_candles(
            symbol,
            "5",
            max(min_5m_required, self._config.ema_period + 1),
        )
        if len(candles_30) < requirement.min_30m or len(candles_15) < requirement.min_15m:
            logger.debug(
                "Insufficient 15m/30m data for %s (15m=%s 30m=%s)",
                symbol,
                len(candles_15),
                len(candles_30),
            )
            return TimeframeDataResult(requirement=requirement, frames={})
        if len(candles_5) < min_5m_required:
            logger.debug(
                "Insufficient 5m data for %s (required=%s got=%s)",
                symbol,
                min_5m_required,
                len(candles_5),
            )
            return TimeframeDataResult(requirement=requirement, frames={})
        frames = {
            "5m": candles_5,
            "15m": candles_15,
            "30m": candles_30,
        }
        return TimeframeDataResult(requirement=requirement, frames=frames)

    def _compute_atr_ratio(
        self,
        candles: list[Candle],
        period: int,
    ) -> Optional[float]:
        if period <= 0:
            return None
        if len(candles) < period + 1:
            return None
        trs: list[float] = []
        start_idx = len(candles) - period
        for idx in range(start_idx, len(candles)):
            current = candles[idx]
            prev = candles[idx - 1]
            high_low = current.high - current.low
            high_close = abs(current.high - prev.close)
            low_close = abs(current.low - prev.close)
            tr = max(high_low, high_close, low_close)
            trs.append(tr)
        if not trs:
            return None
        atr = sum(trs) / len(trs)
        price = candles[-1].close
        if price <= 0:
            return None
        return atr / price

    def _select_requirement(
        self,
        available_30: int,
        requirements: Iterable[TimeframeRequirement],
    ) -> Optional[TimeframeRequirement]:
        for req in requirements:
            if available_30 < req.min_available_30m:
                continue
            if req.max_available_30m is not None and available_30 > req.max_available_30m:
                continue
            return req
        return None

    def _fetch_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        try:
            response = self._client.get_kline(
                category=self._category,
                symbol=symbol,
                interval=interval,
                limit=min(max(limit, 1), 200),
            )
        except BybitAPIError as api_exc:
            logger.error(
                "Bybit API error while fetching %s kline for %s: %s",
                interval,
                symbol,
                api_exc,
            )
            return []
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unexpected error while fetching %s kline for %s: %s",
                interval,
                symbol,
                exc,
            )
            return []
        candles: list[Candle] = []
        payload = response.get("result", {}).get("list") or []
        for item in payload:
            try:
                start = int(item[0])
                open_price = float(item[1])
                high = float(item[2])
                low = float(item[3])
                close = float(item[4])
                volume = float(item[5]) if len(item) > 5 else 0.0
            except (TypeError, ValueError, IndexError):
                logger.debug("Skipping malformed candle entry for %s interval=%s: %s", symbol, interval, item)
                continue
            candles.append(Candle(start=start, open=open_price, high=high, low=low, close=close, volume=volume))
        candles.sort(key=lambda c: c.start)
        return candles

    def _determine_trend(
        self,
        symbol: str,
        timeframe_data: dict[str, list[Candle]],
    ) -> tuple[Optional[str], dict[str, Any]]:
        meta: dict[str, Any] = {}
        ema_scores: Optional[dict[str, float]]
        try:
            ema_scores = self._compute_ema_scores(timeframe_data)
        except ValueError as exc:
            logger.debug("EMA score computation failed for %s: %s", symbol, exc)
            meta["ema_error"] = str(exc)
            ema_scores = None
        if ema_scores:
            aggregate = sum(ema_scores.values())
            logger.debug("EMA trend scores for %s: %s (aggregate=%.6f)", symbol, ema_scores, aggregate)
            meta.update(
                {
                    "method": "ema",
                    "ema_scores": {key: round(value, 6) for key, value in ema_scores.items()},
                    "ema_aggregate": round(aggregate, 6),
                }
            )
            if aggregate > 0:
                return "Long", meta
            if aggregate < 0:
                return "Short", meta
        logger.debug("EMA analysis inconclusive for %s; attempting fallback", symbol)
        fallback_trend, fallback_meta = self._fallback_trend(symbol, timeframe_data.get("5m", []))
        meta.update(fallback_meta)
        return fallback_trend, meta

    def _compute_ema_scores(self, timeframe_data: dict[str, list[Candle]]) -> dict[str, float]:
        period = self._config.ema_period
        if period < 2:
            raise ValueError("EMA period must be >= 2")
        scores: dict[str, float] = {}
        for tf in ("5m", "15m", "30m"):
            candles = timeframe_data.get(tf)
            if not candles or len(candles) < period + 1:
                raise ValueError(f"Not enough candles for {tf}")
            closes = [c.close for c in candles]
            ema_values = self._ema(closes, period)
            if len(ema_values) < 2:
                raise ValueError(f"EMA length too short for {tf}")
            prev, latest = ema_values[-2], ema_values[-1]
            if prev == 0:
                raise ValueError(f"Previous EMA is zero for {tf}")
            slope_pct = ((latest - prev) / prev) * 100.0
            weight = self._config.weights[tf]
            scores[tf] = slope_pct * weight
        return scores

    def _ema(self, values: list[float], period: int) -> list[float]:
        multiplier = 2.0 / (period + 1)
        ema_values: list[float] = []
        ema = None
        for price in values:
            if ema is None:
                ema = price
            else:
                ema = (price - ema) * multiplier + ema
            ema_values.append(ema)
        return ema_values

    def _fallback_trend(
        self,
        symbol: str,
        candles_5m: list[Candle],
    ) -> tuple[Optional[str], dict[str, Any]]:
        meta: dict[str, Any] = {"method": "fallback"}
        if not candles_5m:
            logger.debug("Fallback trend skipped for %s: missing 5m data", symbol)
            meta["reason"] = "missing_5m_data"
            return None, meta
        last = candles_5m[-1]
        if last.open <= 0:
            logger.debug("Invalid open price in fallback for %s: %s", symbol, last)
            meta["reason"] = "invalid_open"
            return None, meta
        change_pct = ((last.close - last.open) / last.open) * 100.0
        threshold = self._config.fallback_threshold_pct
        meta.update(
            {
                "change_pct": round(change_pct, 6),
                "threshold_pct": threshold,
            }
        )
        logger.debug(
            "Fallback change for %s: %.4f%% (threshold=%.2f%%)",
            symbol,
            change_pct,
            threshold,
        )
        if change_pct >= threshold:
            meta["fallback_triggered"] = True
            meta["direction"] = "Long"
            return "Long", meta
        if change_pct <= -threshold:
            meta["fallback_triggered"] = True
            meta["direction"] = "Short"
            return "Short", meta
        meta["fallback_triggered"] = False
        return None, meta

    def _attempt_entry(
        self,
        symbol: str,
        trend: str,
        available: float,
        margin_cap: float,
        timeframe_data: dict[str, list[Candle]],
        atr_ratio: float,
        decision_context: dict[str, Any],
    ) -> Optional[EntryDecision]:
        direction = trend.lower()
        if direction not in {"long", "short"}:
            logger.debug("Unknown trend signal for %s: %s", symbol, trend)
            return None
        if available <= 0:
            logger.debug("Available margin exhausted for %s; skipping entry", symbol)
            return None
        last_close = timeframe_data.get("5m")[-1].close if timeframe_data.get("5m") else None
        if last_close is None or last_close <= 0:
            logger.debug("Missing last close for %s; aborting order", symbol)
            return None
        instrument = self._load_instrument_details(symbol)
        if instrument is None:
            logger.debug("Instrument metadata missing for %s", symbol)
            return None
        min_qty = instrument["min_qty"]
        qty_step = instrument["qty_step"]
        max_qty = instrument.get("max_qty", 0.0)
        leverage = max(self._config.leverage, 1.0)
        min_order_value = instrument.get("min_order_value", 0.0) or 5.0
        min_notional = max(min_qty * last_close, min_order_value)
        if min_notional <= 0:
            logger.debug("Invalid min notional for %s (min_qty=%s last_close=%s)", symbol, min_qty, last_close)
            return None
        available_margin = max(0.0, available)
        max_affordable_notional = available_margin * leverage
        min_margin_required = min_notional / leverage if leverage > 0 else min_notional
        if available_margin < min_margin_required:
            logger.info(
                "Skipping %s because available margin %.6f is below minimum required %.6f",
                symbol,
                available_margin,
                min_margin_required,
            )
            return None
        budget_margin = max(margin_cap, min_margin_required)
        if budget_margin > available_margin:
            budget_margin = available_margin
        if budget_margin > margin_cap + 1e-8 and margin_cap > 0:
            logger.debug(
                "Adjusted margin target for %s from %.6f to %.6f to satisfy minimum order",
                symbol,
                margin_cap,
                budget_margin,
            )
        planned_notional = budget_margin * leverage
        target_notional = max(min_notional, min(planned_notional, max_affordable_notional))
        logger.debug(
            "%s sizing: price=%.6f leverage=%.2f min_notional=%.6f planned_notional=%.6f max_affordable=%.6f target_notional=%.6f",
            symbol,
            last_close,
            leverage,
            min_notional,
            planned_notional,
            max_affordable_notional,
            target_notional,
        )
        if target_notional < min_notional:
            logger.info(
                "Skipping %s because computed notional %.6f is below exchange minimum %.6f",
                symbol,
                target_notional,
                min_notional,
            )
            return None
        desired_qty = target_notional / last_close
        qty = self._quantize_quantity(
            desired_qty,
            min_qty,
            qty_step,
            max_qty,
            max_affordable_notional,
            last_close,
        )
        if qty is None or qty <= 0:
            logger.debug("Quantized quantity insufficient for %s (qty=%s)", symbol, qty)
            return None
        notional = qty * last_close
        if notional < min_notional:
            logger.debug(
                "Quantized notional %.6f is below minimum %.6f for %s",
                notional,
                min_notional,
                symbol,
            )
            return None
        margin_required = notional / leverage
        if margin_required > available_margin + 1e-8:
            logger.debug(
                "Margin required %.6f exceeds available margin %.6f for %s",
                margin_required,
                available_margin,
                symbol,
            )
            return None
        if margin_required > budget_margin + 1e-8:
            logger.debug(
                "Margin usage for %s expanded above slot target %.6f -> %.6f",
                symbol,
                budget_margin,
                margin_required,
            )
        base_sl_pct = self._config.sl_pct
        dynamic_sl_pct = max(base_sl_pct, atr_ratio * self._config.atr_sl_multiplier)
        sl_cap = self._config.atr_sl_cap
        if sl_cap is not None and sl_cap > 0:
            dynamic_sl_pct = min(dynamic_sl_pct, sl_cap)
        if dynamic_sl_pct <= 0:
            dynamic_sl_pct = base_sl_pct if base_sl_pct > 0 else atr_ratio
        base_tp_pct = self._config.tp_pct
        rr_ratio = base_tp_pct / base_sl_pct if base_sl_pct > 0 else 1.5
        dynamic_tp_pct = max(base_tp_pct, dynamic_sl_pct * rr_ratio)
        if self._config.atr_tp_bonus > 0:
            dynamic_tp_pct += atr_ratio * self._config.atr_tp_bonus
        logger.debug(
            "%s final sizing: qty=%.8f notional=%.6f margin_required=%.6f",
            symbol,
            qty,
            notional,
            margin_required,
        )
        side = "Buy" if direction == "long" else "Sell"
        if not self._ensure_leverage(symbol):
            logger.error("Aborting entry for %s because leverage configuration failed", symbol)
            return None
        try:
            response = self._client.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                orderType="Market",
                timeInForce="IOC",
                category=self._category,
            )
            logger.debug("Order response for %s: %s", symbol, response)
            order_result = response.get("result") if isinstance(response, dict) else None
            order_id = None
            if isinstance(order_result, dict):
                order_id = order_result.get("orderId")
            snapshot = self._wait_for_fill(
                symbol=symbol,
                side=side,
                expected_qty=qty,
                order_id=order_id,
            )
            if snapshot is not None:
                try:
                    self._register_entry(
                        snapshot,
                        order_id=order_id,
                        qty=qty,
                        notional=notional,
                        margin_required=margin_required,
                        decision_context=decision_context,
                        tp_pct=dynamic_tp_pct,
                        sl_pct=dynamic_sl_pct,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to register entry analytics for %s: %s",
                        symbol,
                        exc,
                    )
                self._apply_trading_stop(
                    snapshot,
                    instrument=instrument,
                    tp_pct=dynamic_tp_pct,
                    sl_pct=dynamic_sl_pct,
                )
            return EntryDecision(
                notional=notional,
                margin_required=margin_required,
                tp_pct=dynamic_tp_pct,
                sl_pct=dynamic_sl_pct,
                qty=qty,
                side=side,
                order_id=order_id,
            )
        except BybitAPIError as api_exc:
            logger.error("Bybit API rejected entry for %s: %s", symbol, api_exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during order submission for %s: %s", symbol, exc)
        return None

    def _quantize_quantity(
        self,
        desired_qty: float,
        min_qty: float,
        qty_step: float,
        max_qty: float,
        max_notional: float,
        last_close: float,
    ) -> Optional[float]:
        qty = max(desired_qty, min_qty)
        if qty_step > 0:
            min_steps = math.ceil(min_qty / qty_step)
            steps = max(math.ceil(qty / qty_step), min_steps)
            qty = steps * qty_step
            min_qty_aligned = max(min_qty, min_steps * qty_step)
        else:
            min_qty_aligned = min_qty
        if qty < min_qty_aligned:
            qty = min_qty_aligned
        if max_qty and max_qty > 0:
            qty = min(qty, max_qty)
        notional = qty * last_close
        if notional > max_notional:
            if qty_step > 0:
                max_steps = math.floor((max_notional / last_close) / qty_step)
                qty = max_steps * qty_step
            else:
                qty = max_notional / last_close
            logger.debug(
                "Adjusted quantity to fit max_notional: qty=%.8f max_notional=%.6f",
                qty,
                max_notional,
            )
            notional = qty * last_close
        if qty <= 0:
            return None
        if qty < min_qty:
            return None
        if notional <= 0:
            return None
        return qty

    def _load_instrument_details(self, symbol: str) -> Optional[dict[str, float]]:
        try:
            response = self._client.get_symbols(category=self._category, symbol=symbol)
        except BybitAPIError as api_exc:
            logger.error("Bybit API error while loading instrument info for %s: %s", symbol, api_exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while loading instrument info for %s: %s", symbol, exc)
            return None
        instruments = response.get("result", {}).get("list") or []
        normalized_symbol = symbol.upper()
        for item in instruments:
            if str(item.get("symbol", "")).upper() != normalized_symbol:
                continue
            lot = item.get("lotSizeFilter") or {}
            price = item.get("priceFilter") or {}
            try:
                min_qty = float(lot.get("minOrderQty", 0) or 0)
                qty_step = float(lot.get("qtyStep", 0) or 0)
                max_qty = float(lot.get("maxOrderQty", 0) or 0)
                tick_size = float(price.get("tickSize", 0) or 0)
                min_order_value = float(lot.get("minOrderValue", 0) or 0)
            except (TypeError, ValueError):
                logger.debug("Failed to parse lot/price filter for %s: %s", symbol, item)
                return None
            if qty_step == 0 and min_qty > 0:
                qty_step = min_qty
            return {
                "min_qty": min_qty,
                "qty_step": qty_step,
                "max_qty": max_qty,
                "tick_size": tick_size,
                "min_order_value": min_order_value,
            }
        logger.debug("Instrument info not found for %s", symbol)
        return None

    def _ensure_leverage(self, symbol: str) -> bool:
        leverage = self._config.leverage
        if leverage <= 0:
            logger.debug(
                "Skipping leverage setup for %s because configured leverage=%.2f",
                symbol,
                leverage,
            )
            return True
        key = symbol.upper()
        if key in self._leverage_cache:
            return True
        try:
            self._client.set_leverage(
                symbol=key,
                buyLeverage=leverage,
                sellLeverage=leverage,
                category=self._category,
            )
        except BybitAPIError as api_exc:
            if getattr(api_exc, "ret_code", None) == 110043:
                logger.info("Leverage already set to %.2f for %s", leverage, key)
                self._leverage_cache.add(key)
                return True
            logger.error("Failed to set leverage %.2f for %s: %s", leverage, key, api_exc)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while setting leverage for %s: %s", key, exc)
            return False
        self._leverage_cache.add(key)
        logger.info("Leverage set to %.2f for %s", leverage, key)
        return True
