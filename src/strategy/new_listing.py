from __future__ import annotations

"""New listing trading strategy implementation."""

import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

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

logger = logging.getLogger("smpStrategy.strategy.new_listing")

_STOP_LOSS_WINDOW = timedelta(hours=12)
_STOP_LOSS_LIMIT = 3
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
        self._stop_loss_events: dict[str, deque[datetime]] = {}
        self._exclusion_file = self._resolve_exclusion_file()
        self._load_persisted_exclusions()
        if self._static_exclusions:
            logger.info("Static exclusions configured: %s", ", ".join(sorted(self._static_exclusions)))

    def run_once(self, *, initial_candidates: Optional[Iterable[str]] = None) -> None:
        if not self._config.enabled:
            logger.info("New listing strategy disabled in config; skipping execution")
            return
        logger.debug("Starting new listing strategy run: category=%s", self._category)
        positions = self._safe_refresh_positions()
        self._manage_open_positions(positions)
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
        remaining_available = available
        new_positions_opened = 0
        max_new_positions = self._config.max_new_positions
        for idx, symbol in enumerate(symbols):
            symbol = symbol.upper()
            logger.debug("[%s/%s] Processing symbol %s", idx + 1, len(symbols), symbol)
            if symbol in active_symbols:
                logger.debug("Skipping %s because a position is already open", symbol)
                continue
            if new_positions_opened >= max_new_positions:
                logger.info(
                    "Reached max new positions limit (%s); stopping evaluation loop",
                    max_new_positions,
                )
                break
            try:
                notional_used = self._evaluate_symbol(symbol, remaining_available)
            except Exception as exc:  # noqa: BLE001 - log and continue per symbol
                logger.exception("Failed to evaluate symbol %s: %s", symbol, exc)
                continue
            if notional_used is None:
                # None indicates that evaluation aborted without a decision
                continue
            if notional_used <= 0:
                logger.debug("No order placed for %s", symbol)
                continue
            remaining_available = max(0.0, remaining_available - notional_used)
            active_symbols.add(symbol)
            new_positions_opened += 1
            logger.info(
                "Order placed for %s using approx notional=%s; remaining_available=%s",
                symbol,
                notional_used,
                remaining_available,
            )
            if remaining_available <= 0:
                logger.info("Reached capital allocation limit; stopping evaluation loop")
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

    def _manage_open_positions(self, positions: dict[str, PositionSnapshot]) -> None:
        if not positions:
            return
        for symbol, snapshot in positions.items():
            leverage = snapshot.leverage if snapshot.leverage > 0 else 1.0
            tp_threshold = self._config.tp_pct * leverage * 100.0
            sl_threshold = self._config.sl_pct * leverage * 100.0
            logger.debug(
                "Evaluating %s with leverage=%.2f (TP=%.2f%% SL=%.2f%%)",
                symbol,
                leverage,
                tp_threshold,
                sl_threshold,
            )
            pnl_rate = snapshot.pnl_rate
            if pnl_rate >= tp_threshold:
                logger.info("Taking profit on %s with pnl_rate=%.2f%%", symbol, pnl_rate)
                self._close_position(snapshot)
            elif pnl_rate <= -sl_threshold:
                logger.info("Stopping loss on %s with pnl_rate=%.2f%%", symbol, pnl_rate)
                self._handle_stop_loss(snapshot)

    def _close_position(self, snapshot: PositionSnapshot) -> None:
        side = snapshot.side.lower()
        if side not in {"buy", "sell"}:
            logger.warning("Unknown position side for %s: %s", snapshot.symbol, snapshot.side)
            return
        opposite = "sell" if side == "buy" else "buy"
        qty = abs(snapshot.size)
        if qty <= 0:
            logger.debug("Zero size encountered for %s; skipping close", snapshot.symbol)
            return
        try:
            self._client.place_order(
                symbol=snapshot.symbol,
                side=opposite,
                qty=qty,
                orderType="Market",
                timeInForce="IOC",
                reduceOnly=True,
                category=self._category,
            )
            logger.info("Submitted reduce-only close for %s size=%s", snapshot.symbol, qty)
        except BybitAPIError as api_exc:
            logger.error("Failed to close %s due to API error: %s", snapshot.symbol, api_exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while closing %s: %s", snapshot.symbol, exc)

    def _handle_stop_loss(self, snapshot: PositionSnapshot) -> None:
        symbol = snapshot.symbol.upper()
        self._record_stop_loss(symbol)
        self._close_position(snapshot)

    def _record_stop_loss(self, symbol: str) -> None:
        now = datetime.now(timezone.utc)
        events = self._stop_loss_events.setdefault(symbol, deque())
        events.append(now)
        while events and now - events[0] > _STOP_LOSS_WINDOW:
            events.popleft()
        if len(events) >= _STOP_LOSS_LIMIT:
            hours = int(_STOP_LOSS_WINDOW.total_seconds() // 3600)
            reason = f"{len(events)} stop losses within the last {hours}h"
            self._exclude_symbol(symbol, reason)
            events.clear()

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
    ) -> Optional[float]:
        if available <= 0:
            logger.debug("Skipping %s because available balance is exhausted", symbol)
            return 0.0
        timeframe_data = self._collect_timeframe_data(symbol)
        if timeframe_data is None:
            return None
        if not timeframe_data:
            logger.debug("Timeframe data incomplete for %s", symbol)
            return None
        trend = self._determine_trend(symbol, timeframe_data)
        if trend is None:
            logger.info("Trend indeterminate for %s; no position taken", symbol)
            return 0.0
        order_notional = self._attempt_entry(symbol, trend, available, timeframe_data)
        return order_notional

    def _collect_timeframe_data(
        self,
        symbol: str,
    ) -> Optional[dict[str, list[Candle]]]:
        requirements = self._config.requirements
        max_required_60 = max(req.min_60m for req in requirements)
        base_limit_60 = max(max_required_60, self._config.ema_period + 2)
        candles_60 = self._fetch_candles(symbol, "60", base_limit_60)
        if not candles_60:
            logger.debug("No 60m candles retrieved for %s", symbol)
            return None
        available_60 = len(candles_60)
        requirement = self._select_requirement(available_60, requirements)
        if requirement is None:
            logger.debug(
                "No matching requirement block for %s (available_60=%s)",
                symbol,
                available_60,
            )
            return None
        candles_60 = candles_60[-max(requirement.min_60m, self._config.ema_period + 1) :]
        candles_30 = self._fetch_candles(
            symbol,
            "30",
            max(requirement.min_30m, self._config.ema_period + 1),
        )
        candles_15 = self._fetch_candles(
            symbol,
            "15",
            max(requirement.min_15m, self._config.ema_period + 1),
        )
        candles_5 = self._fetch_candles(
            symbol,
            "5",
            max(self._config.min_5m_bars, self._config.ema_period + 1),
        )
        if len(candles_30) < requirement.min_30m or len(candles_15) < requirement.min_15m:
            logger.debug(
                "Insufficient 15m/30m data for %s (15m=%s 30m=%s)",
                symbol,
                len(candles_15),
                len(candles_30),
            )
            return {}
        if len(candles_60) < requirement.min_60m:
            logger.debug(
                "Insufficient 60m data for %s (required=%s got=%s)",
                symbol,
                requirement.min_60m,
                len(candles_60),
            )
            return {}
        if len(candles_5) < self._config.min_5m_bars:
            logger.debug(
                "Insufficient 5m data for %s (required=%s got=%s)",
                symbol,
                self._config.min_5m_bars,
                len(candles_5),
            )
            return {}
        return {
            "15m": candles_15,
            "30m": candles_30,
            "60m": candles_60,
            "5m": candles_5,
        }

    def _select_requirement(
        self,
        available_60: int,
        requirements: Iterable[TimeframeRequirement],
    ) -> Optional[TimeframeRequirement]:
        for req in requirements:
            if available_60 < req.min_available_60m:
                continue
            if req.max_available_60m is not None and available_60 > req.max_available_60m:
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
    ) -> Optional[str]:
        try:
            ema_scores = self._compute_ema_scores(timeframe_data)
        except ValueError as exc:
            logger.debug("EMA score computation failed for %s: %s", symbol, exc)
            ema_scores = None
        if not ema_scores:
            logger.debug("EMA analysis skipped for %s; attempting fallback", symbol)
            return self._fallback_trend(symbol, timeframe_data.get("5m", []))
        aggregate = sum(ema_scores.values())
        logger.debug("EMA trend scores for %s: %s (aggregate=%.6f)", symbol, ema_scores, aggregate)
        if aggregate > 0:
            return "Long"
        if aggregate < 0:
            return "Short"
        return self._fallback_trend(symbol, timeframe_data.get("5m", []))

    def _compute_ema_scores(self, timeframe_data: dict[str, list[Candle]]) -> dict[str, float]:
        period = self._config.ema_period
        if period < 2:
            raise ValueError("EMA period must be >= 2")
        scores: dict[str, float] = {}
        for tf in ("15m", "30m", "60m"):
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

    def _fallback_trend(self, symbol: str, candles_5m: list[Candle]) -> Optional[str]:
        if not candles_5m:
            logger.debug("Fallback trend skipped for %s: missing 5m data", symbol)
            return None
        last = candles_5m[-1]
        if last.open <= 0:
            logger.debug("Invalid open price in fallback for %s: %s", symbol, last)
            return None
        change_pct = ((last.close - last.open) / last.open) * 100.0
        threshold = self._config.fallback_threshold_pct
        logger.debug(
            "Fallback change for %s: %.4f%% (threshold=%.2f%%)",
            symbol,
            change_pct,
            threshold,
        )
        if change_pct >= threshold:
            return "Long"
        if change_pct <= -threshold:
            return "Short"
        return None

    def _attempt_entry(
        self,
        symbol: str,
        trend: str,
        available: float,
        timeframe_data: dict[str, list[Candle]],
    ) -> float:
        direction = trend.lower()
        if direction not in {"long", "short"}:
            logger.debug("Unknown trend signal for %s: %s", symbol, trend)
            return 0.0
        last_close = timeframe_data.get("5m")[-1].close if timeframe_data.get("5m") else None
        if last_close is None or last_close <= 0:
            logger.debug("Missing last close for %s; aborting order", symbol)
            return 0.0
        instrument = self._load_instrument_details(symbol)
        if instrument is None:
            logger.debug("Instrument metadata missing for %s", symbol)
            return 0.0
        min_qty = instrument["min_qty"]
        qty_step = instrument["qty_step"]
        max_qty = instrument.get("max_qty", 0.0)
        min_notional = min_qty * last_close
        if min_notional <= 0:
            logger.debug("Invalid min notional for %s (min_qty=%s last_close=%s)", symbol, min_qty, last_close)
            return 0.0
        if available < min_notional:
            logger.info(
                "Skipping %s because available=%s < min_notional=%s",
                symbol,
                available,
                min_notional,
            )
            return 0.0
        target_notional = max(min_notional, available * self._config.allocation_pct)
        target_notional = min(target_notional, available)
        desired_qty = target_notional / last_close
        qty = self._quantize_quantity(desired_qty, min_qty, qty_step, max_qty, available, last_close)
        if qty is None or qty <= 0 or qty * last_close < min_notional:
            logger.debug("Quantized quantity insufficient for %s (qty=%s)", symbol, qty)
            return 0.0
        side = "Buy" if direction == "long" else "Sell"
        if not self._ensure_leverage(symbol):
            logger.error("Aborting entry for %s because leverage configuration failed", symbol)
            return 0.0
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
            return qty * last_close
        except BybitAPIError as api_exc:
            logger.error("Bybit API rejected entry for %s: %s", symbol, api_exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during order submission for %s: %s", symbol, exc)
        return 0.0

    def _quantize_quantity(
        self,
        desired_qty: float,
        min_qty: float,
        qty_step: float,
        max_qty: float,
        available: float,
        last_close: float,
    ) -> Optional[float]:
        qty = max(desired_qty, min_qty)
        if qty_step > 0:
            min_steps = math.ceil(min_qty / qty_step)
            steps = max(math.floor(qty / qty_step), min_steps)
            qty = steps * qty_step
            min_qty_aligned = max(min_qty, min_steps * qty_step)
        else:
            min_qty_aligned = min_qty
        if qty < min_qty_aligned:
            qty = min_qty_aligned
        if max_qty and max_qty > 0:
            qty = min(qty, max_qty)
        notional = qty * last_close
        if notional > available:
            if qty_step > 0:
                max_steps = math.floor((available / last_close) / qty_step)
                qty = max_steps * qty_step
            else:
                qty = available / last_close
        if qty <= 0:
            return None
        if qty < min_qty:
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
            logger.error("Failed to set leverage %.2f for %s: %s", leverage, key, api_exc)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while setting leverage for %s: %s", key, exc)
            return False
        self._leverage_cache.add(key)
        logger.info("Leverage set to %.2f for %s", leverage, key)
        return True
