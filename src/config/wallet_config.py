from __future__ import annotations

"""Configuration loader for wallet polling settings."""

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data.wallet_manager.constants import DEFAULT_REFRESH_INTERVAL

_CONFIG_ENV_VAR = "WALLET_CONFIG"
_DEFAULT_FILE_NAME = "wallet.ini"
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass(slots=True)
class WalletConfig:
    refresh_interval: float = DEFAULT_REFRESH_INTERVAL


def resolve_wallet_config_path(path: str | os.PathLike[str] | None = None) -> Path:
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


def load_wallet_config(path: str | os.PathLike[str] | None = None) -> WalletConfig:
    config_path = resolve_wallet_config_path(path)
    parser = configparser.ConfigParser()
    read_files = parser.read(config_path)
    if not read_files:
        raise FileNotFoundError(f"wallet config not found at {config_path}")
    section = parser["wallet"] if "wallet" in parser else parser[parser.sections()[0]]
    refresh_interval = section.getfloat("refresh_interval", fallback=DEFAULT_REFRESH_INTERVAL)
    return WalletConfig(refresh_interval=max(1.0, refresh_interval))


def maybe_load_wallet_config(
    path: str | os.PathLike[str] | None = None,
    *,
    strict: bool = False,
) -> Optional[WalletConfig]:
    try:
        return load_wallet_config(path)
    except (FileNotFoundError, ValueError, KeyError):
        if strict:
            raise
        return None
