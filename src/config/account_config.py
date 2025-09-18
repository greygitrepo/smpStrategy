from __future__ import annotations

"""Utilities for managing Bybit account configuration stored under config/accountConfig.ini."""

import configparser
import getpass
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_CONFIG_ENV_VAR = "BYBIT_ACCOUNT_CONFIG"
_DEFAULT_FILE_NAME = "accountConfig.ini"
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass
class BybitAccountConfig:
    """Parsed configuration values for a single Bybit account."""

    api_key: str
    api_secret: str
    testnet: bool = True
    category: str = "linear"
    account_name: Optional[str] = None

    def as_env(self) -> dict[str, str]:
        """Return env-var friendly mapping for quick integration with existing workflows."""

        env = {
            "BYBIT_API_KEY": self.api_key,
            "BYBIT_API_SECRET": self.api_secret,
            "TESTNET": "true" if self.testnet else "false",
            "BYBIT_CATEGORY": self.category,
        }
        if self.account_name:
            env["BYBIT_ACCOUNT_NAME"] = self.account_name
        return env


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve configuration path using explicit value, env override, or project default."""

    candidates = []
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
    # Fall back to first candidate even if it does not exist for creation workflows
    return candidates[0]


def load_account_config(
    path: str | os.PathLike[str] | None = None,
) -> BybitAccountConfig:
    """Load the Bybit account configuration from disk."""

    config_path = resolve_config_path(path)
    parser = configparser.ConfigParser()
    read_files = parser.read(config_path)
    if not read_files:
        raise FileNotFoundError(f"account config not found at {config_path}")
    if "bybit" not in parser:
        raise ValueError(
            "section [bybit] missing in accountConfig.ini; run interactive setup first"
        )
    section = parser["bybit"]
    api_key = section.get("api_key", fallback="").strip()
    api_secret = section.get("api_secret", fallback="").strip()
    if not api_key or not api_secret:
        raise ValueError("api_key/api_secret missing in accountConfig.ini")
    account_name = section.get("account_name", fallback=None)
    testnet = section.getboolean("testnet", fallback=True)
    category = section.get("category", fallback="linear").strip() or "linear"
    return BybitAccountConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        category=category,
        account_name=account_name.strip() if account_name else None,
    )


def maybe_load_account_config(
    path: str | os.PathLike[str] | None = None,
    *,
    strict: bool = False,
) -> Optional[BybitAccountConfig]:
    """Best-effort loader that returns None when the file is absent or incomplete."""

    try:
        return load_account_config(path)
    except (FileNotFoundError, ValueError):
        if strict:
            raise
        return None


def interactive_setup(
    path: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
) -> Path:
    """Prompt the user for account credentials and write config/accountConfig.ini.

    Parameters
    ----------
    path:
        Optional location for the config file. Defaults to config/accountConfig.ini in the
        project root, or the `BYBIT_ACCOUNT_CONFIG` override if it resolves to an existing file.
    force:
        Overwrite an existing file instead of prompting the user to confirm.
    """

    config_path = resolve_config_path(path)
    if config_path.exists() and not force:
        answer = input(
            f"Config file {config_path} already exists. Overwrite? [y/N]: "
        ).strip()
        if answer.lower() not in {"y", "yes"}:
            return config_path
    print("Enter your Bybit account credentials. Values are kept locally in INI format.")
    account_name = input("Account nickname (optional): ").strip() or None
    api_key = input("API key: ").strip()
    api_secret = getpass.getpass("API secret: ").strip()
    testnet_answer = input("Use testnet? [Y/n]: ").strip().lower()
    testnet = testnet_answer not in {"n", "no", "false", "0"}
    category = input("Default category (linear/inverse/spot) [linear]: ").strip() or "linear"

    parser = configparser.ConfigParser()
    parser["bybit"] = {
        "account_name": account_name or "",
        "api_key": api_key,
        "api_secret": api_secret,
        "testnet": "true" if testnet else "false",
        "category": category,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as fh:
        parser.write(fh)
    print(f"Saved Bybit account configuration to {config_path}")
    return config_path


def fetch_wallet_balance_from_config(
    path: str | os.PathLike[str] | None = None,
    *,
    account_type: str = "UNIFIED",
    coin: Optional[str] = None,
) -> dict:
    """Convenience helper to fetch wallet balance using credentials stored in config."""

    config = load_account_config(path)
    from ..exchange.bybit_v5 import BybitV5Client

    client = BybitV5Client(
        api_key=config.api_key,
        api_secret=config.api_secret,
        testnet=config.testnet,
        category=config.category,
    )
    return client.get_wallet_balance(accountType=account_type, coin=coin)


if __name__ == "__main__":
    interactive_setup()
