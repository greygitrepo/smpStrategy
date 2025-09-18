from __future__ import annotations

"""Configuration loader for symbol discovery strategies."""

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data.symbol.constants import (
    DEFAULT_CATEGORY,
    DEFAULT_DISCOVERY_INTERVAL,
    DEFAULT_SETTLE_COIN,
    DEFAULT_SYMBOL_LIMIT,
)

_CONFIG_ENV_VAR = "SYMBOL_STRATEGY_CONFIG"
_DEFAULT_FILE_NAME = "symbolStrategy.ini"
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(slots=True)
class SymbolStrategyConfig:
    strategy: str
    limit: int = DEFAULT_SYMBOL_LIMIT
    refresh_interval: float = DEFAULT_DISCOVERY_INTERVAL
    category: str = DEFAULT_CATEGORY
    settle_coin: str = DEFAULT_SETTLE_COIN


def _parse_config(parser: configparser.ConfigParser, path: Path) -> SymbolStrategyConfig:
    if "strategy" not in parser:
        raise ValueError(
            f"symbol strategy config missing [strategy] section in {path}"
        )
    section = parser["strategy"]
    strategy = section.get("name", fallback="top_volume").strip()
    limit = section.getint("limit", fallback=DEFAULT_SYMBOL_LIMIT)
    refresh_interval = section.getfloat(
        "refresh_interval", fallback=DEFAULT_DISCOVERY_INTERVAL
    )
    category = section.get("category", fallback=DEFAULT_CATEGORY).strip() or DEFAULT_CATEGORY
    settle_coin = section.get("settle_coin", fallback=DEFAULT_SETTLE_COIN).strip() or DEFAULT_SETTLE_COIN
    return SymbolStrategyConfig(
        strategy=strategy,
        limit=max(1, limit),
        refresh_interval=max(1.0, refresh_interval),
        category=category,
        settle_coin=settle_coin,
    )


def resolve_symbol_strategy_config_path(
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


def load_symbol_strategy_config(
    path: str | os.PathLike[str] | None = None,
) -> SymbolStrategyConfig:
    config_path = resolve_symbol_strategy_config_path(path)
    parser = configparser.ConfigParser()
    read_files = parser.read(config_path)
    if not read_files:
        raise FileNotFoundError(f"symbol strategy config not found at {config_path}")
    return _parse_config(parser, config_path)


def maybe_load_symbol_strategy_config(
    path: str | os.PathLike[str] | None = None,
    *,
    strict: bool = False,
) -> Optional[SymbolStrategyConfig]:
    try:
        return load_symbol_strategy_config(path)
    except (FileNotFoundError, ValueError):
        if strict:
            raise
        return None
