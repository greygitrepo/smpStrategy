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
    min_available_60m: int
    max_available_60m: Optional[int]
    min_5m: int
    min_15m: int
    min_30m: int
    min_60m: int


@dataclass(slots=True)
class NewListingStrategyConfig:
    """Top-level configuration values for the new-listing strategy."""

    enabled: bool = True
    ema_period: int = 9
    leverage: float = 3.0
    weight_15m: float = 3.0
    weight_30m: float = 2.0
    weight_60m: float = 1.0
    fallback_threshold_pct: float = 3.0
    allocation_pct: float = 0.05
    tp_pct: float = 0.014
    sl_pct: float = 0.014
    min_5m_bars: int = 20
    requirements: tuple[TimeframeRequirement, ...] = ()

    @property
    def weights(self) -> dict[str, float]:
        return {
            "15m": self.weight_15m,
            "30m": self.weight_30m,
            "60m": self.weight_60m,
        }


_DEFAULT_REQUIREMENTS: tuple[TimeframeRequirement, ...] = (
    TimeframeRequirement(
        name="high",
        min_available_60m=10,
        max_available_60m=None,
        min_5m=20,
        min_15m=40,
        min_30m=20,
        min_60m=10,
    ),
    TimeframeRequirement(
        name="medium",
        min_available_60m=5,
        max_available_60m=9,
        min_5m=15,
        min_15m=20,
        min_30m=10,
        min_60m=5,
    ),
    TimeframeRequirement(
        name="low",
        min_available_60m=1,
        max_available_60m=2,
        min_5m=10,
        min_15m=4,
        min_30m=2,
        min_60m=1,
    ),
)


def _normalize_percent(value: float, *, assume_percent: bool = False) -> float:
    if assume_percent and value > 1:
        return value / 100.0
    return value


def _parse_requirement(section_name: str, section: configparser.SectionProxy) -> TimeframeRequirement:
    min_available = max(0, section.getint("min_available_60m", fallback=0))
    max_available_raw = section.get("max_available_60m", fallback="").strip()
    max_available = int(max_available_raw) if max_available_raw else None
    if max_available is not None and max_available <= 0:
        max_available = None
    return TimeframeRequirement(
        name=section.get("name", fallback=section_name.split(".")[-1]),
        min_available_60m=min_available,
        max_available_60m=max_available,
        min_5m=max(1, section.getint("min_5m", fallback=10)),
        min_15m=max(1, section.getint("min_15m", fallback=10)),
        min_30m=max(1, section.getint("min_30m", fallback=5)),
        min_60m=max(1, section.getint("min_60m", fallback=min_available or 1)),
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
            key=lambda item: item.min_available_60m,
            reverse=True,
        )
    )
    config = NewListingStrategyConfig(
        enabled=enabled,
        ema_period=ema_period,
        weight_15m=base.getfloat("weight_15m", fallback=3.0),
        weight_30m=base.getfloat("weight_30m", fallback=2.0),
        weight_60m=base.getfloat("weight_60m", fallback=1.0),
        fallback_threshold_pct=fallback_threshold_pct,
        leverage=leverage,
        allocation_pct=allocation_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        min_5m_bars=min_5m_bars,
        requirements=requirements,
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


def default_new_listing_strategy_config() -> NewListingStrategyConfig:
    """Return an in-memory config with the documented default requirements."""

    return NewListingStrategyConfig(
        leverage=3.0,
        requirements=tuple(
            TimeframeRequirement(
                name=req.name,
                min_available_60m=req.min_available_60m,
                max_available_60m=req.max_available_60m,
                min_5m=req.min_5m,
                min_15m=req.min_15m,
                min_30m=req.min_30m,
                min_60m=req.min_60m,
            )
            for req in _DEFAULT_REQUIREMENTS
        )
    )
