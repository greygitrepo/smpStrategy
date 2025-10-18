from __future__ import annotations

"""Online optimisation utilities for new-listing strategy parameters."""

import argparse
import copy
import json
import logging
import random
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.config.new_listing_strategy_config import (
    NewListingStrategyConfig,
    TUNABLE_PARAMETER_SPECS,
    load_new_listing_strategy_config,
    write_new_listing_strategy_config,
)
from .models import TradeRecord

logger = logging.getLogger("smpStrategy.analytics.optimizer")


class OnlineLinearRegressor:
    """Simple online linear regressor trained with SGD."""

    def __init__(self, learning_rate: float = 0.0015) -> None:
        self.learning_rate = learning_rate
        self.weights: Dict[str, float] = {}
        self.bias: float = 0.0

    def predict(self, features: Dict[str, float]) -> float:
        score = self.bias
        for name, value in features.items():
            score += self.weights.get(name, 0.0) * value
        return score

    def partial_fit(self, features: Dict[str, float], target: float) -> None:
        prediction = self.predict(features)
        error = target - prediction
        lr = self.learning_rate
        for name, value in features.items():
            self.weights[name] = self.weights.get(name, 0.0) + lr * error * value
        self.bias += lr * error

    def to_dict(self) -> Dict[str, Any]:
        return {"learning_rate": self.learning_rate, "weights": self.weights, "bias": self.bias}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OnlineLinearRegressor":
        instance = cls(learning_rate=float(payload.get("learning_rate", 0.001)))
        instance.weights = {str(k): float(v) for k, v in payload.get("weights", {}).items()}
        instance.bias = float(payload.get("bias", 0.0))
        return instance


@dataclass
class ExperimentState:
    candidate: Dict[str, float]
    baseline_values: Dict[str, float]
    baseline_avg_pnl: float
    baseline_metric_name: str
    baseline_snapshot: Dict[str, Any]
    start_time: datetime
    start_trade_index: int
    trades: list[TradeRecord] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    pnl: float = 0.0

    def should_finish(self, now: datetime, max_trades: int, max_duration: timedelta) -> bool:
        trade_done = len(self.trades) >= max_trades
        duration_done = now - self.start_time >= max_duration
        return trade_done or duration_done


class AdaptiveParameterManager:
    """Online-learning driven parameter tuner with experiment/rollback flow."""

    _FEATURE_ALPHA = 0.05
    _MIN_SUCCESS_WIN_RATE = 0.60
    _RECENT_FEATURE_MAXLEN = 64
    _RECENT_FEATURE_WEIGHT = 0.45
    _CANDIDATE_SAMPLE_SIZE = 48
    _MIN_EXPERIMENT_TRADES = 10
    _CONFIDENCE_Z = 1.96
    _POLY_SQUARED_FEATURES: tuple[str, ...] = (
        "allocation_pct",
        "tp_pct",
        "sl_pct",
        "atr_skip_pct",
        "candidate_score",
        "score_threshold",
        "trend_range_pct",
    )
    _POLY_INTERACTION_FEATURES: tuple[tuple[str, str], ...] = (
        ("allocation_pct", "tp_pct"),
        ("allocation_pct", "sl_pct"),
        ("tp_pct", "sl_pct"),
        ("weight_5m", "weight_15m"),
        ("weight_15m", "weight_30m"),
        ("score_threshold", "candidate_score"),
        ("trend_filter_min_atr", "trend_filter_min_range_pct"),
        ("trend_range_pct", "trend_reversals"),
    )
    _PARAM_UPPER_BOUNDS: Dict[str, float] = {
        "trend_filter_min_ema": 0.009,          # 0.9%
        "trend_filter_min_atr": 0.02,          # 2%
        "trend_filter_min_range_pct": 0.012,   # 1.2%
        "fallback_threshold_pct": 0.05,        # 5%
        "trend_reverse_candle_threshold_pct": 0.03,  # 3%
    }

    def __init__(
        self,
        *,
        config: NewListingStrategyConfig,
        log_file: Path,
        model_dir: Path,
        optimized_config_path: Path,
        evaluation_trades: int,
        evaluation_duration: timedelta,
        candidate_trade_interval: int,
        candidate_time_interval: timedelta,
    ) -> None:
        self._config = config
        self._log_file = log_file
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._optimized_config_path = optimized_config_path
        self._evaluation_trades = max(1, evaluation_trades)
        self._evaluation_duration = evaluation_duration
        self._candidate_trade_interval = max(1, candidate_trade_interval)
        self._candidate_time_interval = candidate_time_interval
        self._history_file = self._model_dir / "history.jsonl"
        self._data_file = self._model_dir / "training_data.jsonl"
        self._model_state_file = self._model_dir / "model_state.json"
        self._model = OnlineLinearRegressor()
        self._feature_means: Dict[str, float] = {}
        self._recent_features: deque[Dict[str, float]] = deque(maxlen=self._RECENT_FEATURE_MAXLEN)
        self._total_trades = 0
        self._trades_since_update = 0
        self._last_candidate_time = datetime.now(timezone.utc)
        self._experiment: Optional[ExperimentState] = None
        self._param_specs = self._build_param_specs()
        self._load_model_state()
        self._enforce_config_limits()
        self._persist_config()

    def process_trades(
        self,
        trades: Iterable[TradeRecord],
        performance_snapshot: Optional[dict[str, dict[str, Any]]],
    ) -> None:
        trades = list(trades)
        if trades:
            for trade in trades:
                features = self._build_features(trade)
                self._update_feature_means(features)
                model_features = self._augment_model_features(features)
                target = self._compute_trade_return(trade)
                self._model.partial_fit(model_features, target)
                self._append_training_example(model_features, target, trade)
                self._total_trades += 1
                if self._experiment is not None:
                    self._experiment.trades.append(trade)
                    self._experiment.returns.append(target)
                    self._experiment.pnl += trade.realized_pnl
                self._trades_since_update += 1
            self._save_model_state()
        self._evaluate(performance_snapshot)

    # ------------------------------------------------------------------
    # Internal helpers
    def _build_features(self, trade: TradeRecord) -> Dict[str, float]:
        context = dict(trade.metadata.get("context", {}))
        trend = context.get("trend", {})
        features: Dict[str, float] = {}
        features["allocation_pct"] = float(context.get("allocation_pct", self._config.allocation_pct))
        features["tp_pct"] = float(context.get("tp_pct", self._config.tp_pct))
        features["sl_pct"] = float(context.get("sl_pct", self._config.sl_pct))
        features["atr_skip_pct"] = float(context.get("atr_skip_pct", self._config.atr_skip_pct))
        features["ema_period"] = float(context.get("ema_period", self._config.ema_period))
        features["weight_5m"] = float(context.get("weight_5m", self._config.weight_5m))
        features["weight_15m"] = float(context.get("weight_15m", self._config.weight_15m))
        features["weight_30m"] = float(context.get("weight_30m", self._config.weight_30m))
        features["fallback_threshold_pct"] = float(
            context.get("fallback_threshold_pct", self._config.fallback_threshold_pct)
        )
        features["min_notional_buffer_pct"] = float(
            context.get("min_notional_buffer_pct", getattr(self._config, "min_notional_buffer_pct", 0.0))
        )
        features["trend_filter_min_ema"] = float(
            context.get("trend_filter_min_ema", self._config.trend_filter_min_ema)
        )
        features["trend_filter_min_atr"] = float(
            context.get("trend_filter_min_atr", self._config.trend_filter_min_atr)
        )
        features["trend_filter_min_range_pct"] = float(
            context.get("trend_filter_min_range_pct", self._config.trend_filter_min_range_pct)
        )
        features["trend_filter_max_reversals"] = float(
            context.get("trend_filter_max_reversals", self._config.trend_filter_max_reversals)
        )
        features["trend_filter_range_lookback"] = float(
            context.get("trend_filter_range_lookback", self._config.trend_filter_range_lookback)
        )
        features["trend_slope_window"] = float(
            context.get("trend_slope_window", self._config.trend_slope_window)
        )
        features["trend_consistency_min_signals"] = float(
            context.get(
                "trend_consistency_min_signals",
                self._config.trend_consistency_min_signals,
            )
        )
        features["trend_reverse_candle_threshold_pct"] = float(
            context.get(
                "trend_reverse_candle_threshold_pct",
                self._config.trend_reverse_candle_threshold_pct,
            )
        )
        features["fallback_window"] = float(
            context.get("fallback_window", self._config.fallback_window)
        )
        features["fallback_min_consecutive"] = float(
            context.get(
                "fallback_min_consecutive",
                self._config.fallback_min_consecutive,
            )
        )
        features["fallback_min_atr"] = float(
            context.get("fallback_min_atr", self._config.fallback_min_atr)
        )
        features["fallback_max_ema"] = float(
            context.get("fallback_max_ema", self._config.fallback_max_ema)
        )
        features["score_threshold"] = float(
            context.get("score_threshold", self._config.score_threshold)
        )
        features["score_ema_weight"] = float(
            context.get("score_ema_weight", self._config.score_ema_weight)
        )
        features["score_range_weight"] = float(
            context.get("score_range_weight", self._config.score_range_weight)
        )
        features["score_atr_weight"] = float(
            context.get("score_atr_weight", self._config.score_atr_weight)
        )
        features["score_reversal_penalty"] = float(
            context.get(
                "score_reversal_penalty",
                self._config.score_reversal_penalty,
            )
        )
        features["score_atr_ceiling"] = float(
            context.get("score_atr_ceiling", self._config.score_atr_ceiling)
        )
        features["candidate_score"] = float(context.get("score", 0.0))
        features["atr_ratio"] = float(context.get("atr_ratio", self._config.trend_filter_min_atr))
        features["ema_aggregate"] = float(trend.get("ema_aggregate", 0.0))
        direction = str(trend.get("direction", ""))
        features["is_long"] = 1.0 if direction.lower() == "long" else 0.0
        features["is_short"] = 1.0 if direction.lower() == "short" else 0.0
        trend_filter_metrics = context.get("trend_filter", {})
        features["trend_range_pct"] = float(trend_filter_metrics.get("range_pct", 0.0))
        features["trend_reversals"] = float(trend_filter_metrics.get("reversals", 0.0))
        features["holding_minutes"] = trade.holding_seconds / 60.0
        return features

    def _update_feature_means(self, features: Dict[str, float]) -> None:
        alpha = self._FEATURE_ALPHA
        for name, value in features.items():
            previous = self._feature_means.get(name)
            if previous is None:
                self._feature_means[name] = value
            else:
                self._feature_means[name] = previous + alpha * (value - previous)
        self._recent_features.append(dict(features))

    def _augment_model_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """Generate polynomial and interaction features for richer model capacity."""

        enriched = dict(base_features)
        for name in self._POLY_SQUARED_FEATURES:
            value = enriched.get(name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            enriched[f"{name}__sq"] = numeric * numeric
        for left, right in self._POLY_INTERACTION_FEATURES:
            if left not in enriched or right not in enriched:
                continue
            try:
                left_val = float(enriched[left])
                right_val = float(enriched[right])
            except (TypeError, ValueError):
                continue
            enriched[self._interaction_feature_name(left, right)] = left_val * right_val
        return enriched

    @staticmethod
    def _interaction_feature_name(left: str, right: str) -> str:
        if left <= right:
            first, second = left, right
        else:
            first, second = right, left
        return f"{first}__x__{second}"

    def _recent_feature_average(self) -> Dict[str, float]:
        if not self._recent_features:
            return {}
        aggregate: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for snapshot in self._recent_features:
            for name, value in snapshot.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                aggregate[name] = aggregate.get(name, 0.0) + numeric
                counts[name] = counts.get(name, 0) + 1
        return {name: aggregate[name] / counts[name] for name in aggregate if counts.get(name)}

    def _compute_trade_return(self, trade: TradeRecord) -> float:
        """Derive a position-level return target for model training."""

        if trade.return_pct is not None:
            return float(trade.return_pct)
        if trade.notional:
            try:
                return float(trade.realized_pnl / trade.notional)
            except ZeroDivisionError:
                return 0.0
        context_notional = (
            trade.metadata.get("context", {}).get("notional") if trade.metadata else None
        )
        try:
            notional = float(context_notional) if context_notional is not None else None
        except (TypeError, ValueError):
            notional = None
        if notional:
            try:
                return float(trade.realized_pnl / notional)
            except ZeroDivisionError:
                return 0.0
        return 0.0

    def _append_training_example(
        self,
        features: Dict[str, float],
        target: float,
        trade: TradeRecord,
    ) -> None:
        record = {
            "timestamp": trade.closed_at.isoformat(),
            "symbol": trade.symbol,
            "features": features,
            "target": target,
            "metadata": {
                "return_pct": trade.return_pct,
                "realized_pnl": trade.realized_pnl,
                "notional": trade.notional,
                "size": trade.size,
            },
        }
        try:
            with self._data_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")
        except OSError as exc:
            logger.error("Failed to append training data to %s: %s", self._data_file, exc)

    def _evaluate(self, snapshot: Optional[dict[str, dict[str, Any]]]) -> None:
        now = datetime.now(timezone.utc)
        if self._experiment is not None:
            if self._experiment.should_finish(now, self._evaluation_trades, self._evaluation_duration):
                self._complete_experiment(snapshot, now)
            return
        if self._total_trades == 0:
            return
        if self._trades_since_update < self._candidate_trade_interval:
            return
        if now - self._last_candidate_time < self._candidate_time_interval:
            return
        candidate = self._select_candidate()
        if candidate is None:
            return
        self._start_experiment(candidate, snapshot, now)

    def _select_candidate(self) -> Optional[Dict[str, float]]:
        if not self._feature_means:
            return None
        baseline_features = self._project_features({})
        baseline_score = self._model.predict(baseline_features)
        best_candidate: Optional[Dict[str, float]] = None
        best_score = baseline_score
        for _ in range(self._CANDIDATE_SAMPLE_SIZE):
            candidate = self._sample_candidate()
            candidate = self._clamp_parameters(candidate)
            if not self._is_candidate_within_bounds(candidate):
                continue
            features = self._project_features(candidate)
            score = self._model.predict(features)
            if score > best_score + 1e-9:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _project_features(self, candidate: Dict[str, float]) -> Dict[str, float]:
        projected = dict(self._feature_means)
        recent_average = self._recent_feature_average()
        if recent_average:
            weight = self._RECENT_FEATURE_WEIGHT
            complement = 1.0 - weight
            for name, value in recent_average.items():
                base = projected.get(name, value)
                try:
                    base_val = float(base)
                except (TypeError, ValueError):
                    base_val = value
                projected[name] = complement * base_val + weight * value
        projected.update(candidate)
        return self._augment_model_features(projected)

    def _sample_candidate(
        self,
        *,
        rng: Optional[random.Random] = None,
        vary: Optional[set[str]] = None,
    ) -> Dict[str, float]:
        generator = rng or random
        candidate: Dict[str, float] = {}
        for name, spec in self._param_specs.items():
            base = getattr(self._config, name)
            if vary is not None and name not in vary:
                candidate[name] = base
                continue
            if spec["type"] == "int":
                step = spec["variation"]
                value = int(round(base + generator.randint(-step, step)))
                value = max(spec["min"], min(spec["max"], value))
            else:
                variation = spec["variation"]
                min_value = spec["min"]
                max_value = spec["max"]
                lower = max(min_value, base - variation)
                upper = min(max_value, base + variation)
                if lower > upper:
                    lower, upper = min_value, max_value
                value = round(generator.uniform(lower, upper), 6)
            candidate[name] = value
        # maintain intuitive ordering between EMA weights
        candidate["weight_5m"] = max(candidate["weight_5m"], candidate["weight_15m"], candidate["weight_30m"])
        candidate["weight_15m"] = min(candidate["weight_5m"], candidate["weight_15m"])
        candidate["weight_30m"] = min(candidate["weight_15m"], candidate["weight_30m"])
        return candidate

    def _start_experiment(
        self,
        candidate: Dict[str, float],
        snapshot: Optional[dict[str, dict[str, Any]]],
        now: datetime,
    ) -> None:
        candidate = self._clamp_parameters(candidate)
        if not self._is_candidate_within_bounds(candidate):
            logger.warning("Rejected candidate outside allowed bounds: %s", candidate)
            return
        baseline_values = {name: getattr(self._config, name) for name in candidate}
        window_metrics = (snapshot or {}).get("window", {})
        baseline_trades = int(window_metrics.get("trades", 0)) or 1
        baseline_avg = float(window_metrics.get("net_pnl", 0.0)) / baseline_trades
        baseline_metric_name = "avg_pnl"
        avg_return = window_metrics.get("avg_return_pct")
        if avg_return is not None:
            try:
                baseline_avg = float(avg_return)
                baseline_metric_name = "avg_return_pct"
            except (TypeError, ValueError):
                baseline_avg = float(window_metrics.get("net_pnl", 0.0)) / baseline_trades
                baseline_metric_name = "avg_pnl"
        if not math.isfinite(baseline_avg):
            baseline_avg = 0.0
        self._experiment = ExperimentState(
            candidate=candidate,
            baseline_values=baseline_values,
            baseline_avg_pnl=baseline_avg,
            baseline_metric_name=baseline_metric_name,
            baseline_snapshot=window_metrics,
            start_time=now,
            start_trade_index=self._total_trades,
        )
        for name, value in candidate.items():
            setattr(self._config, name, value)
        self._persist_config()
        self._log_event(
            "experiment_start",
            {
                "candidate": candidate,
                "baseline": baseline_values,
                "baseline_avg_pnl": baseline_avg,
                "baseline_metric_name": baseline_metric_name,
                "baseline_avg_metric": baseline_avg,
                "min_trades_required": self._MIN_EXPERIMENT_TRADES,
                "start_time": now.isoformat(),
            },
        )
        self._trades_since_update = 0
        self._last_candidate_time = now

    def _complete_experiment(self, snapshot: Optional[dict[str, dict[str, Any]]], now: datetime) -> None:
        if self._experiment is None:
            return
        trades = self._experiment.trades
        trade_count = len(trades)
        pnl = self._experiment.pnl
        raw_returns = self._experiment.returns
        returns = [value for value in raw_returns if math.isfinite(value)]
        if len(returns) != len(raw_returns):
            self._experiment.returns = returns
        avg_return = sum(returns) / len(returns) if returns else 0.0
        baseline_avg = self._experiment.baseline_avg_pnl
        if len(returns) > 1:
            std_dev = statistics.stdev(returns)
            stderr = std_dev / math.sqrt(len(returns))
            ci_lower = avg_return - self._CONFIDENCE_Z * stderr
        else:
            std_dev = 0.0
            stderr = None
            ci_lower = avg_return
        avg_pnl = pnl / max(1, trade_count)
        success = (
            trade_count >= self._MIN_EXPERIMENT_TRADES
            and len(returns) >= self._MIN_EXPERIMENT_TRADES
            and stderr is not None
            and ci_lower >= baseline_avg
        )
        if trade_count == 0 or not returns:
            success = False
        window_snapshot = snapshot.get("window", {}) if snapshot else {}
        win_rate = float(window_snapshot.get("win_rate", 0.0) or 0.0)
        window_trades = int(window_snapshot.get("trades", 0) or 0)
        if window_trades > 0 and win_rate < self._MIN_SUCCESS_WIN_RATE:
            success = False
        result = "accepted" if success else "rolled_back"
        if not success:
            for name, value in self._experiment.baseline_values.items():
                setattr(self._config, name, value)
        self._persist_config()
        self._log_event(
            "experiment_end",
            {
                "candidate": self._experiment.candidate,
                "result": result,
                "avg_pnl": avg_pnl,
                "avg_return_pct": avg_return,
                "return_ci_lower": ci_lower,
                "return_std": std_dev,
                "return_stderr": stderr,
                "baseline_metric_name": self._experiment.baseline_metric_name,
                "baseline_avg_metric": baseline_avg,
                "baseline_avg_pnl": baseline_avg,
                "trades": len(trades),
                "min_trades_required": self._MIN_EXPERIMENT_TRADES,
                "pnl": pnl,
                "win_rate": win_rate,
                "ended_at": now.isoformat(),
            },
        )
        self._experiment = None
        self._trades_since_update = 0
        self._save_model_state()

    def _persist_config(self) -> None:
        try:
            write_new_listing_strategy_config(self._config, self._optimized_config_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to persist optimized config snapshot: %s", exc)

    def suggest_best_parameters(
        self,
        *,
        samples: int = 256,
        seed: Optional[int] = None,
        vary: Optional[Iterable[str]] = None,
    ) -> Optional[Dict[str, float]]:
        if samples <= 0:
            return None
        if not self._feature_means:
            logger.debug("Cannot suggest parameters: feature means unavailable (no trade data yet)")
            return None
        restrict = set(vary) if vary is not None else None
        rng = random.Random(seed) if seed is not None else random
        baseline_score = self._model.predict(self._project_features({}))
        best_score = baseline_score
        best_candidate: Optional[Dict[str, float]] = None
        for _ in range(samples):
            candidate = self._sample_candidate(rng=rng, vary=restrict)
            score = self._model.predict(self._project_features(candidate))
            if score > best_score + 1e-9:
                best_score = score
                best_candidate = candidate
        if best_candidate is None:
            logger.debug("Suggested parameters fall back to baseline; no improvement detected")
        else:
            logger.debug("Suggested parameters with predicted score %.6f (baseline %.6f)", best_score, baseline_score)
        return best_candidate

    def force_new_experiment(
        self,
        snapshot: Optional[dict[str, dict[str, Any]]] = None,
    ) -> bool:
        """Start a new candidate experiment immediately, bypassing idle thresholds."""

        now = datetime.now(timezone.utc)
        if self._total_trades == 0:
            logger.debug("Cannot force tuner experiment: no trade history yet")
            return False
        if self._experiment is not None:
            logger.info("Forcing parameter tuner to abort current experiment and restart")
            for name, value in self._experiment.baseline_values.items():
                setattr(self._config, name, value)
            self._persist_config()
            self._log_event(
                "experiment_force_abort",
                {
                    "candidate": self._experiment.candidate,
                    "baseline": self._experiment.baseline_values,
                    "start_time": self._experiment.start_time.isoformat(),
                    "forced_at": now.isoformat(),
                },
            )
            self._experiment = None
            self._trades_since_update = 0
        candidate = self._select_candidate()
        if candidate is None:
            logger.debug("Force tuner experiment requested but no improved candidate available")
            return False
        self._start_experiment(candidate, snapshot, now)
        logger.info("Parameter tuner force-started experiment after inactivity")
        return True

    def _clamp_parameters(self, candidate: Dict[str, float]) -> Dict[str, float]:
        for name, upper in self._PARAM_UPPER_BOUNDS.items():
            if name in candidate:
                if candidate[name] > upper:
                    candidate[name] = upper
        return candidate

    def _is_candidate_within_bounds(self, candidate: Dict[str, float]) -> bool:
        for name, value in candidate.items():
            spec = self._param_specs.get(name)
            if spec is None:
                continue
            lower = spec["min"]
            upper = spec["max"]
            if value < lower or value > upper:
                return False
        return True

    def _enforce_config_limits(self) -> None:
        updated = False
        for name, upper in self._PARAM_UPPER_BOUNDS.items():
            current = getattr(self._config, name, None)
            if current is None:
                continue
            if current > upper:
                setattr(self._config, name, upper)
                updated = True
        if updated:
            logger.debug("Clamped configuration values to safety bounds")

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        try:
            with self._history_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")
        except OSError as exc:
            logger.error("Failed to append optimizer history to %s: %s", self._history_file, exc)
        try:
            with self._log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")
        except OSError as exc:
            logger.error("Failed to append tuning log to %s: %s", self._log_file, exc)

    def _save_model_state(self) -> None:
        state = {
            "model": self._model.to_dict(),
            "feature_means": self._feature_means,
            "total_trades": self._total_trades,
            "trades_since_update": self._trades_since_update,
            "last_candidate_time": self._last_candidate_time.isoformat(),
        }
        if self._experiment is not None:
            state["experiment"] = {
                "candidate": self._experiment.candidate,
                "baseline_values": self._experiment.baseline_values,
                "baseline_avg_pnl": self._experiment.baseline_avg_pnl,
                "baseline_metric_name": self._experiment.baseline_metric_name,
                "baseline_snapshot": self._experiment.baseline_snapshot,
                "start_time": self._experiment.start_time.isoformat(),
                "start_trade_index": self._experiment.start_trade_index,
                "pnl": self._experiment.pnl,
                "trades": [trade.to_dict() for trade in self._experiment.trades],
                "returns": self._experiment.returns,
            }
        try:
            with self._model_state_file.open("w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=True, indent=2)
        except OSError as exc:
            logger.error("Failed to persist optimizer state to %s: %s", self._model_state_file, exc)

    def _load_model_state(self) -> None:
        if not self._model_state_file.exists():
            return
        try:
            with self._model_state_file.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load optimizer state from %s: %s", self._model_state_file, exc)
            return
        model_payload = state.get("model")
        if isinstance(model_payload, dict):
            self._model = OnlineLinearRegressor.from_dict(model_payload)
        self._feature_means = {
            str(k): float(v) for k, v in (state.get("feature_means") or {}).items()
        }
        self._total_trades = int(state.get("total_trades", 0) or 0)
        self._trades_since_update = int(state.get("trades_since_update", 0) or 0)
        last_time = state.get("last_candidate_time")
        if last_time:
            try:
                self._last_candidate_time = datetime.fromisoformat(last_time)
            except ValueError:
                self._last_candidate_time = datetime.now(timezone.utc)
        experiment_payload = state.get("experiment")
        if isinstance(experiment_payload, dict):
            try:
                baseline_metric_name = str(
                    experiment_payload.get("baseline_metric_name", "avg_pnl")
                )
                returns_payload = experiment_payload.get("returns", [])
                returns = [float(value) for value in returns_payload] if returns_payload else []
                self._experiment = ExperimentState(
                    candidate={str(k): float(v) for k, v in experiment_payload.get("candidate", {}).items()},
                    baseline_values={
                        str(k): float(v)
                        for k, v in experiment_payload.get("baseline_values", {}).items()
                    },
                    baseline_avg_pnl=float(experiment_payload.get("baseline_avg_pnl", 0.0)),
                    baseline_metric_name=baseline_metric_name,
                    baseline_snapshot=dict(experiment_payload.get("baseline_snapshot", {})),
                    start_time=datetime.fromisoformat(experiment_payload.get("start_time")),
                    start_trade_index=int(experiment_payload.get("start_trade_index", 0)),
                    trades=[TradeRecord.from_dict(item) for item in experiment_payload.get("trades", [])],
                    returns=returns,
                    pnl=float(experiment_payload.get("pnl", 0.0)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to restore experiment state: %s", exc)
                self._experiment = None

    def _build_param_specs(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(TUNABLE_PARAMETER_SPECS)


__all__ = ["AdaptiveParameterManager"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest optimized parameters from the online tuner state.")
    parser.add_argument(
        "--config",
        default="analytics/newListingStrategy_optimized.ini",
        help="Path to the base strategy config to load (defaults to optimized snapshot).",
    )
    parser.add_argument(
        "--analytics-dir",
        default="analytics",
        help="Directory containing optimizer artifacts (history, model state).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of random candidates to evaluate when suggesting parameters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible candidate sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the suggested parameters without writing them to disk.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the suggested config (defaults to --config path).",
    )
    parser.add_argument(
        "--all-params",
        action="store_true",
        default=True,
        help="Allow optimisation to adjust every tunable parameter instead of only the newly added ones.",
    )
    return parser.parse_args()


def _initialize_manager(config_path: Path, analytics_dir: Path) -> AdaptiveParameterManager:
    base_config = load_new_listing_strategy_config(config_path)
    log_file = analytics_dir / "parameter_tuning.jsonl"
    model_dir = analytics_dir / "optimizer"
    optimized_path = config_path
    manager = AdaptiveParameterManager(
        config=copy.deepcopy(base_config),
        log_file=log_file,
        model_dir=model_dir,
        optimized_config_path=optimized_path,
        evaluation_trades=15,
        evaluation_duration=timedelta(hours=1),
        candidate_trade_interval=15,
        candidate_time_interval=timedelta(hours=1),
    )
    return manager


def _main() -> None:
    new_param_names = {
        "trend_slope_window",
        "trend_consistency_min_signals",
        "fallback_window",
        "fallback_min_consecutive",
        "fallback_min_atr",
        "fallback_max_ema",
        "score_threshold",
        "score_ema_weight",
        "score_range_weight",
        "score_atr_weight",
        "score_reversal_penalty",
        "score_atr_ceiling",
    }
    args = _parse_args()
    config_path = Path(args.config).expanduser()
    analytics_dir = Path(args.analytics_dir).expanduser()
    analytics_dir.mkdir(parents=True, exist_ok=True)
    manager = _initialize_manager(config_path, analytics_dir)
    vary = None if args.all_params else new_param_names
    candidate = manager.suggest_best_parameters(samples=args.samples, seed=args.seed, vary=vary)
    if not candidate:
        print("Unable to suggest improved parameters; ensure optimizer has collected trade data." )
        return
    target_path = Path(args.output).expanduser() if args.output else config_path
    try:
        updated_config = load_new_listing_strategy_config(target_path)
    except (FileNotFoundError, ValueError):
        updated_config = copy.deepcopy(manager._config)
    apply_keys = set(candidate.keys()) if args.all_params else new_param_names
    suggested_values = {k: v for k, v in candidate.items() if k in apply_keys}
    if not suggested_values:
        print("No parameter suggestions produced; try allowing all parameters with --all-params.")
        return
    for name, value in suggested_values.items():
        setattr(updated_config, name, value)
    if args.dry_run:
        print("Suggested parameters (dry-run, not saved):")
        for key in sorted(suggested_values):
            print(f"  {key} = {suggested_values[key]}")
        return
    write_new_listing_strategy_config(updated_config, target_path)
    print(f"Suggested parameters written to {target_path}")


if __name__ == "__main__":
    _main()
