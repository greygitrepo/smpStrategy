from __future__ import annotations

"""Helpers for constructing Bybit API clients used by symbol utilities."""

from src.config.account_config import maybe_load_account_config, resolve_config_path
from src.exchange.bybit_v5 import BybitV5Client


def build_client() -> BybitV5Client:
    config = maybe_load_account_config()
    if config is None:
        default_path = resolve_config_path()
        raise FileNotFoundError(
            "accountConfig.ini not found. Run `python -m src.config.account_config` or copy "
            f"{default_path.parent / 'sampleConfig.ini'} to create one."
        )
    return BybitV5Client(
        api_key=config.api_key,
        api_secret=config.api_secret,
        testnet=config.testnet,
        category=config.category,
    )
