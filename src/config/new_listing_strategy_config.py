from __future__ import annotations

"""Configuration loader for the new-listing trading strategy."""

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_CONFIG_ENV_VAR = "NEW_LISTING_STRATEGY_CONFIG"
_DEFAULT_FILE_NAME = "newListingStrategy.ini"
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(slots=True)
class TimeframeRequirement:
    """Thresholds for minimum kline counts per timeframe."""

    name: str
    min_available_30m: int
    max_available_30m: Optional[int]
    min_5m: int
    min_15m: int
    min_30m: int


@dataclass(slots=True)
class NewListingStrategyConfig:
    """Top-level configuration values for the new-listing strategy."""

    enabled: bool = True
    enable_parameter_tuner: bool = True
    ema_period: int = 9
    leverage: float = 3.0
    weight_5m: float = 3.0
    weight_15m: float = 2.0
    weight_30m: float = 1.0
    fallback_threshold_pct: float = 3.0
    allocation_pct: float = 0.05
    tp_pct: float = 0.014
    sl_pct: float = 0.014
    min_5m_bars: int = 20
    max_new_positions: int = 3
    requirements: tuple[TimeframeRequirement, ...] = ()
    exclude_symbols: tuple[str, ...] = ()
    atr_period: int = 14
    atr_skip_pct: float = 0.04
    atr_sl_multiplier: float = 1.2
    atr_sl_cap: Optional[float] = None
    atr_tp_bonus: float = 0.0
    tuning_min_total_trades: int = 30
    tuning_min_gap_trades: int = 15
    trend_filter_min_ema: float = 0.005
    trend_filter_min_atr: float = 0.015
    trend_filter_min_range_pct: float = 0.01
    trend_filter_max_reversals: int = 3
    trend_filter_range_lookback: int = 12
    trend_slope_window: int = 3
    trend_consistency_min_signals: int = 2
    fallback_window: int = 3
    fallback_min_consecutive: int = 2
    fallback_min_atr: float = 0.008
    fallback_max_ema: float = 0.004
    score_threshold: float = 1.5
    score_ema_weight: float = 1.0
    score_range_weight: float = 0.8
    score_atr_weight: float = 0.6
    score_reversal_penalty: float = 0.7
    score_atr_ceiling: float = 5.0

    @property
    def weights(self) -> dict[str, float]:
        return {
            "5m": self.weight_5m,
            "15m": self.weight_15m,
            "30m": self.weight_30m,
        }


_DEFAULT_REQUIREMENTS: tuple[TimeframeRequirement, ...] = (
    TimeframeRequirement(
        name="high",
        min_available_30m=10,
        max_available_30m=None,
        min_5m=20,
        min_15m=40,
        min_30m=20,
    ),
    TimeframeRequirement(
        name="medium",
        min_available_30m=5,
        max_available_30m=9,
        min_5m=15,
        min_15m=20,
        min_30m=10,
    ),
    TimeframeRequirement(
        name="low",
        min_available_30m=1,
        max_available_30m=2,
        min_5m=10,
        min_15m=4,
        min_30m=2,
    ),
)


def _normalize_percent(value: float, *, assume_percent: bool = False) -> float:
    if assume_percent and value >= 1:
        return value / 100.0
    return value


def _parse_symbol_list(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    symbols = [sym.strip().upper() for sym in raw.split(",")]
    return tuple(sorted({sym for sym in symbols if sym}))


def _parse_requirement(section_name: str, section: configparser.SectionProxy) -> TimeframeRequirement:
    min_available = max(0, section.getint("min_available_30m", fallback=0))
    max_available_raw = section.get("max_available_30m", fallback="").strip()
    max_available = int(max_available_raw) if max_available_raw else None
    if max_available is not None and max_available <= 0:
        max_available = None
    return TimeframeRequirement(
        name=section.get("name", fallback=section_name.split(".")[-1]),
        min_available_30m=min_available,
        max_available_30m=max_available,
        min_5m=max(1, section.getint("min_5m", fallback=10)),
        min_15m=max(1, section.getint("min_15m", fallback=10)),
        min_30m=max(1, section.getint("min_30m", fallback=min_available or 1)),
    )


def resolve_new_listing_strategy_config_path(
    path: str | os.PathLike[str] | None = None,
) -> Path:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.environ.get(_CONFIG_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(_CONFIG_DIR / _DEFAULT_FILE_NAME)
    candidates.append(Path(_DEFAULT_FILE_NAME))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_new_listing_strategy_config(
    path: str | os.PathLike[str] | None = None,
) -> NewListingStrategyConfig:
    config_path = resolve_new_listing_strategy_config_path(path)
    parser = configparser.ConfigParser()
    read_files = parser.read(config_path)
    if not read_files:
        raise FileNotFoundError(
            f"new listing strategy config not found at {config_path}"
        )
    base_section_name = "new_listing_strategy"
    if base_section_name not in parser:
        raise ValueError(
            f"config missing '[{base_section_name}]' section in {config_path}"
        )
    base = parser[base_section_name]
    enabled = base.getboolean("enabled", fallback=True)
    enable_parameter_tuner = base.getboolean("enable_parameter_tuner", fallback=True)
    ema_period = max(2, base.getint("ema_period", fallback=9))
    leverage = max(
        1.0,
        base.getfloat("leverage", fallback=3.0),
    )
    allocation_pct = max(
        0.0,
        _normalize_percent(
            base.getfloat("allocation_pct", fallback=0.05),
            assume_percent=True,
        ),
    )
    fallback_threshold_pct = max(
        0.0,
        _normalize_percent(
            base.getfloat("fallback_threshold_pct", fallback=3.0),
            assume_percent=True,
        ),
    )
    atr_period = max(1, base.getint("atr_period", fallback=12))
    atr_skip_pct = max(
        0.0,
        _normalize_percent(
            base.getfloat("atr_skip_pct", fallback=7.5),
            assume_percent=True,
        ),
    )
    atr_sl_multiplier = max(0.0, base.getfloat("atr_sl_multiplier", fallback=1.0))
    atr_sl_cap_raw = base.get("atr_sl_cap", fallback="").strip()
    atr_sl_cap: Optional[float]
    if atr_sl_cap_raw:
        try:
            atr_sl_cap = max(
                0.0,
                _normalize_percent(float(atr_sl_cap_raw), assume_percent=True),
            )
        except ValueError:
            atr_sl_cap = None
    else:
        atr_sl_cap = None
    atr_tp_bonus = max(
        0.0,
        _normalize_percent(
            base.getfloat("atr_tp_bonus", fallback=0.5),
            assume_percent=True,
        ),
    )
    tp_pct = max(
        0.0,
        _normalize_percent(
            base.getfloat("tp_pct", fallback=1.4),
            assume_percent=True,
        ),
    )
    sl_pct = max(
        0.0,
        _normalize_percent(
            base.getfloat("sl_pct", fallback=1.4),
            assume_percent=True,
        ),
    )
    min_5m_bars = max(1, base.getint("min_5m_bars", fallback=20))
    max_new_positions = max(1, base.getint("max_new_positions", fallback=3))
    tuning_min_total_trades = max(1, base.getint("tuning_min_total_trades", fallback=30))
    tuning_min_gap_trades = max(1, base.getint("tuning_min_gap_trades", fallback=15))
    trend_filter_min_ema = max(
        0.0,
        _normalize_percent(
            base.getfloat("trend_filter_min_ema", fallback=0.6),
            assume_percent=True,
        ),
    )
    trend_filter_min_atr = max(
        0.0,
        _normalize_percent(
            base.getfloat("trend_filter_min_atr", fallback=1.5),
            assume_percent=True,
        ),
    )
    trend_filter_min_range_pct = max(
        0.0,
        _normalize_percent(
            base.getfloat("trend_filter_min_range_pct", fallback=0.8),
            assume_percent=True,
        ),
    )
    trend_filter_max_reversals = max(0, base.getint("trend_filter_max_reversals", fallback=3))
    trend_filter_range_lookback = max(3, base.getint("trend_filter_range_lookback", fallback=12))
    trend_slope_window = max(1, base.getint("trend_slope_window", fallback=3))
    trend_consistency_min_signals = max(
        1, base.getint("trend_consistency_min_signals", fallback=2)
    )
    fallback_window = max(1, base.getint("fallback_window", fallback=3))
    fallback_min_consecutive = max(
        1, base.getint("fallback_min_consecutive", fallback=2)
    )
    fallback_min_atr = max(
        0.0,
        _normalize_percent(
            base.getfloat("fallback_min_atr", fallback=0.8),
            assume_percent=True,
        ),
    )
    fallback_max_ema = max(
        0.0,
        _normalize_percent(
            base.getfloat("fallback_max_ema", fallback=0.4),
            assume_percent=True,
        ),
    )
    score_threshold = max(0.0, base.getfloat("score_threshold", fallback=1.5))
    score_ema_weight = max(0.0, base.getfloat("score_ema_weight", fallback=1.0))
    score_range_weight = max(0.0, base.getfloat("score_range_weight", fallback=0.8))
    score_atr_weight = max(0.0, base.getfloat("score_atr_weight", fallback=0.6))
    score_reversal_penalty = max(0.0, base.getfloat("score_reversal_penalty", fallback=0.7))
    score_atr_ceiling = max(0.0, base.getfloat("score_atr_ceiling", fallback=5.0))
    requirement_sections: list[TimeframeRequirement] = []
    prefix = f"{base_section_name}."
    for section_name in parser.sections():
        if not section_name.startswith(prefix):
            continue
        requirement_sections.append(
            _parse_requirement(section_name, parser[section_name])
        )
    requirements = tuple(
        sorted(
            requirement_sections or _DEFAULT_REQUIREMENTS,
            key=lambda item: item.min_available_30m,
            reverse=True,
        )
    )
    exclude_symbols = _parse_symbol_list(base.get("exclude_symbols", fallback=""))
    config = NewListingStrategyConfig(
        enabled=enabled,
        ema_period=ema_period,
        enable_parameter_tuner=enable_parameter_tuner,
        weight_5m=base.getfloat("weight_5m", fallback=3.0),
        weight_15m=base.getfloat("weight_15m", fallback=2.0),
        weight_30m=base.getfloat("weight_30m", fallback=1.0),
        fallback_threshold_pct=fallback_threshold_pct,
        leverage=leverage,
        allocation_pct=allocation_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        min_5m_bars=min_5m_bars,
        max_new_positions=max_new_positions,
        requirements=requirements,
        exclude_symbols=exclude_symbols,
        atr_period=atr_period,
        atr_skip_pct=atr_skip_pct,
        atr_sl_multiplier=atr_sl_multiplier,
        atr_sl_cap=atr_sl_cap,
        atr_tp_bonus=atr_tp_bonus,
        tuning_min_total_trades=tuning_min_total_trades,
        tuning_min_gap_trades=tuning_min_gap_trades,
        trend_filter_min_ema=trend_filter_min_ema,
        trend_filter_min_atr=trend_filter_min_atr,
        trend_filter_min_range_pct=trend_filter_min_range_pct,
        trend_filter_max_reversals=trend_filter_max_reversals,
        trend_filter_range_lookback=trend_filter_range_lookback,
        trend_slope_window=trend_slope_window,
        trend_consistency_min_signals=trend_consistency_min_signals,
        fallback_window=fallback_window,
        fallback_min_consecutive=fallback_min_consecutive,
        fallback_min_atr=fallback_min_atr,
        fallback_max_ema=fallback_max_ema,
        score_threshold=score_threshold,
        score_ema_weight=score_ema_weight,
        score_range_weight=score_range_weight,
        score_atr_weight=score_atr_weight,
        score_reversal_penalty=score_reversal_penalty,
        score_atr_ceiling=score_atr_ceiling,
    )
    return config


def maybe_load_new_listing_strategy_config(
    path: str | os.PathLike[str] | None = None,
    *,
    strict: bool = False,
) -> Optional[NewListingStrategyConfig]:
    try:
        return load_new_listing_strategy_config(path)
    except (FileNotFoundError, ValueError):
        if strict:
            raise
        return None


def _format_float(value: float) -> str:
    formatted = f"{value:.6f}"
    while formatted.endswith("0") and "." in formatted:
        formatted = formatted[:-1]
    if formatted.endswith("."):
        formatted = formatted[:-1]
    return formatted or "0"


def _format_percent(value: float) -> str:
    return _format_float(value * 100.0)


def _sanitize_requirement_name(name: str, index: int) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    slug = "_".join(filter(None, slug.split("_")))
    if not slug:
        slug = f"req_{index + 1}"
    return slug


def write_new_listing_strategy_config(
    config: NewListingStrategyConfig,
    path: str | os.PathLike[str],
) -> Path:
    parser = configparser.ConfigParser()
    section_name = "new_listing_strategy"
    base_section = {
        "enabled": "true" if config.enabled else "false",
        "enable_parameter_tuner": "true" if config.enable_parameter_tuner else "false",
        "ema_period": str(config.ema_period),
        "leverage": _format_float(config.leverage),
        "weight_5m": _format_float(config.weight_5m),
        "weight_15m": _format_float(config.weight_15m),
        "weight_30m": _format_float(config.weight_30m),
        "fallback_threshold_pct": _format_percent(config.fallback_threshold_pct),
        "allocation_pct": _format_percent(config.allocation_pct),
        "tp_pct": _format_percent(config.tp_pct),
        "sl_pct": _format_percent(config.sl_pct),
        "min_5m_bars": str(config.min_5m_bars),
        "max_new_positions": str(config.max_new_positions),
        "atr_period": str(config.atr_period),
        "atr_skip_pct": _format_percent(config.atr_skip_pct),
        "atr_sl_multiplier": _format_float(config.atr_sl_multiplier),
        "atr_tp_bonus": _format_float(config.atr_tp_bonus),
        "tuning_min_total_trades": str(config.tuning_min_total_trades),
        "tuning_min_gap_trades": str(config.tuning_min_gap_trades),
        "exclude_symbols": ", ".join(config.exclude_symbols),
        "trend_filter_min_ema": _format_float(config.trend_filter_min_ema),
        "trend_filter_min_atr": _format_float(config.trend_filter_min_atr),
        "trend_filter_min_range_pct": _format_float(config.trend_filter_min_range_pct),
        "trend_filter_max_reversals": str(config.trend_filter_max_reversals),
        "trend_filter_range_lookback": str(config.trend_filter_range_lookback),
        "trend_slope_window": str(config.trend_slope_window),
        "trend_consistency_min_signals": str(config.trend_consistency_min_signals),
        "fallback_window": str(config.fallback_window),
        "fallback_min_consecutive": str(config.fallback_min_consecutive),
        "fallback_min_atr": _format_float(config.fallback_min_atr),
        "fallback_max_ema": _format_float(config.fallback_max_ema),
        "score_threshold": _format_float(config.score_threshold),
        "score_ema_weight": _format_float(config.score_ema_weight),
        "score_range_weight": _format_float(config.score_range_weight),
        "score_atr_weight": _format_float(config.score_atr_weight),
        "score_reversal_penalty": _format_float(config.score_reversal_penalty),
        "score_atr_ceiling": _format_float(config.score_atr_ceiling),
    }
    if config.atr_sl_cap is not None:
        base_section["atr_sl_cap"] = _format_percent(config.atr_sl_cap)
    else:
        base_section["atr_sl_cap"] = ""
    if config.exclude_symbols:
        base_section["exclude_symbols"] = ", ".join(config.exclude_symbols)
    parser[section_name] = base_section

    for idx, requirement in enumerate(config.requirements):
        slug = _sanitize_requirement_name(requirement.name, idx)
        req_section_name = f"{section_name}.requirement_{slug}"
        parser[req_section_name] = {
            "name": requirement.name,
            "min_available_30m": str(requirement.min_available_30m),
            "max_available_30m": "" if requirement.max_available_30m is None else str(requirement.max_available_30m),
            "min_5m": str(requirement.min_5m),
            "min_15m": str(requirement.min_15m),
            "min_30m": str(requirement.min_30m),
        }

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return output_path


def default_new_listing_strategy_config() -> NewListingStrategyConfig:
    """Return an in-memory config with the documented default requirements."""

    return NewListingStrategyConfig(
        leverage=3.0,
        enable_parameter_tuner=True,
        requirements=tuple(
            TimeframeRequirement(
                name=req.name,
                min_available_30m=req.min_available_30m,
                max_available_30m=req.max_available_30m,
                min_5m=req.min_5m,
                min_15m=req.min_15m,
                min_30m=req.min_30m,
            )
            for req in _DEFAULT_REQUIREMENTS
        ),
        max_new_positions=3,
        exclude_symbols=(),
        atr_period=12,
        atr_skip_pct=0.075,
        atr_sl_multiplier=1.8,
        atr_sl_cap=0.05,
        atr_tp_bonus=0.0025,
        tuning_min_total_trades=30,
        tuning_min_gap_trades=15,
        trend_filter_min_ema=0.006,
        trend_filter_min_atr=0.015,
        trend_filter_min_range_pct=0.008,
        trend_filter_max_reversals=3,
        trend_filter_range_lookback=12,
        trend_slope_window=3,
        trend_consistency_min_signals=2,
        fallback_window=3,
        fallback_min_consecutive=2,
        fallback_min_atr=0.008,
        fallback_max_ema=0.004,
        score_threshold=1.5,
        score_ema_weight=1.0,
        score_range_weight=0.8,
        score_atr_weight=0.6,
        score_reversal_penalty=0.7,
        score_atr_ceiling=5.0,
    )
