from __future__ import annotations

"""Online optimisation utilities for new-listing strategy parameters."""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.config.new_listing_strategy_config import (
    NewListingStrategyConfig,
    write_new_listing_strategy_config,
)
from .models import TradeRecord

logger = logging.getLogger("smpStrategy.analytics.optimizer")


class OnlineLinearRegressor:
    """Simple online linear regressor trained with SGD."""

    def __init__(self, learning_rate: float = 0.001) -> None:
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
    baseline_snapshot: Dict[str, Any]
    start_time: datetime
    start_trade_index: int
    trades: list[TradeRecord] = field(default_factory=list)
    pnl: float = 0.0

    def should_finish(self, now: datetime, max_trades: int, max_duration: timedelta) -> bool:
        trade_done = len(self.trades) >= max_trades
        duration_done = now - self.start_time >= max_duration
        return trade_done or duration_done


class AdaptiveParameterManager:
    """Online-learning driven parameter tuner with experiment/rollback flow."""

    _FEATURE_ALPHA = 0.05

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
        self._total_trades = 0
        self._trades_since_update = 0
        self._last_candidate_time = datetime.now(timezone.utc)
        self._experiment: Optional[ExperimentState] = None
        self._param_specs = self._build_param_specs()
        self._load_model_state()
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
                self._model.partial_fit(features, trade.realized_pnl)
                self._append_training_example(features, trade.realized_pnl, trade)
                self._total_trades += 1
                if self._experiment is not None:
                    self._experiment.trades.append(trade)
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
        features["atr_ratio"] = float(context.get("atr_ratio", 0.0))
        features["ema_aggregate"] = float(trend.get("ema_aggregate", 0.0))
        direction = str(trend.get("direction", ""))
        features["is_long"] = 1.0 if direction.lower() == "long" else 0.0
        features["is_short"] = 1.0 if direction.lower() == "short" else 0.0
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
        for _ in range(16):
            candidate = self._sample_candidate()
            features = self._project_features(candidate)
            score = self._model.predict(features)
            if score > best_score + 1e-9:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _project_features(self, candidate: Dict[str, float]) -> Dict[str, float]:
        projected = dict(self._feature_means)
        projected.update(candidate)
        return projected

    def _sample_candidate(self) -> Dict[str, float]:
        candidate: Dict[str, float] = {}
        for name, spec in self._param_specs.items():
            base = getattr(self._config, name)
            if spec["type"] == "int":
                step = spec["variation"]
                value = int(round(base + random.randint(-step, step)))
                value = max(spec["min"], min(spec["max"], value))
            else:
                variation = spec["variation"]
                value = base + random.uniform(-variation, variation)
                value = max(spec["min"], min(spec["max"], value))
                value = round(value, 6)
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
        baseline_values = {name: getattr(self._config, name) for name in candidate}
        window_metrics = (snapshot or {}).get("window", {})
        baseline_trades = int(window_metrics.get("trades", 0)) or 1
        baseline_avg = float(window_metrics.get("net_pnl", 0.0)) / baseline_trades
        self._experiment = ExperimentState(
            candidate=candidate,
            baseline_values=baseline_values,
            baseline_avg_pnl=baseline_avg,
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
                "start_time": now.isoformat(),
            },
        )
        self._trades_since_update = 0
        self._last_candidate_time = now

    def _complete_experiment(self, snapshot: Optional[dict[str, dict[str, Any]]], now: datetime) -> None:
        if self._experiment is None:
            return
        trades = self._experiment.trades
        pnl = self._experiment.pnl
        avg = pnl / max(1, len(trades))
        baseline_avg = self._experiment.baseline_avg_pnl
        success = avg >= baseline_avg
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
                "avg_pnl": avg,
                "baseline_avg_pnl": baseline_avg,
                "trades": len(trades),
                "pnl": pnl,
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
                "baseline_snapshot": self._experiment.baseline_snapshot,
                "start_time": self._experiment.start_time.isoformat(),
                "start_trade_index": self._experiment.start_trade_index,
                "pnl": self._experiment.pnl,
                "trades": [trade.to_dict() for trade in self._experiment.trades],
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
                self._experiment = ExperimentState(
                    candidate={str(k): float(v) for k, v in experiment_payload.get("candidate", {}).items()},
                    baseline_values={
                        str(k): float(v)
                        for k, v in experiment_payload.get("baseline_values", {}).items()
                    },
                    baseline_avg_pnl=float(experiment_payload.get("baseline_avg_pnl", 0.0)),
                    baseline_snapshot=dict(experiment_payload.get("baseline_snapshot", {})),
                    start_time=datetime.fromisoformat(experiment_payload.get("start_time")),
                    start_trade_index=int(experiment_payload.get("start_trade_index", 0)),
                    trades=[TradeRecord.from_dict(item) for item in experiment_payload.get("trades", [])],
                    pnl=float(experiment_payload.get("pnl", 0.0)),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to restore experiment state: %s", exc)
                self._experiment = None

    def _build_param_specs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "allocation_pct": {"type": "float", "min": 0.01, "max": 0.2, "variation": 0.02},
            "tp_pct": {"type": "float", "min": 0.005, "max": 0.06, "variation": 0.005},
            "sl_pct": {"type": "float", "min": 0.004, "max": 0.08, "variation": 0.006},
            "atr_skip_pct": {"type": "float", "min": 0.005, "max": 0.2, "variation": 0.01},
            "fallback_threshold_pct": {"type": "float", "min": 0.005, "max": 0.12, "variation": 0.01},
            "ema_period": {"type": "int", "min": 3, "max": 30, "variation": 2},
            "weight_5m": {"type": "float", "min": 1.0, "max": 5.0, "variation": 0.3},
            "weight_15m": {"type": "float", "min": 0.5, "max": 4.0, "variation": 0.3},
            "weight_30m": {"type": "float", "min": 0.1, "max": 3.0, "variation": 0.3},
        }


__all__ = ["AdaptiveParameterManager"]
