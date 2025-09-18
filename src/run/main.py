from __future__ import annotations

"""Main entry point for launching the project directly from an IDE (e.g. F5)."""

import logging
import sys
from pathlib import Path

# allow running as a script (e.g. F5 in IDE) without manual PYTHONPATH tweaks
if __package__ is None or __package__ == "":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.config.account_config import (  # noqa: E402  pylint: disable=wrong-import-position
    maybe_load_account_config,
    resolve_config_path,
)
from src.data import WalletDataManager  # noqa: E402  pylint: disable=wrong-import-position
from src.exchange.bybit_v5 import BybitV5Client  # noqa: E402  pylint: disable=wrong-import-position

logger = logging.getLogger("smpStrategy")


def _require_account_config() -> BybitV5Client:
    config = maybe_load_account_config()
    if config is None:
        default_path = resolve_config_path()
        sample_path = Path(default_path.parent, "sampleConfig.ini")
        raise FileNotFoundError(
            "accountConfig.ini not found. Run `python -m src.config.account_config` or copy "
            f"{sample_path} to create one."
        )
    logger.info(
        "Loaded account config: name=%s, testnet=%s, category=%s",
        config.account_name or "default",
        config.testnet,
        config.category,
    )
    return BybitV5Client(
        api_key=config.api_key,
        api_secret=config.api_secret,
        testnet=config.testnet,
        category=config.category,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    client = _require_account_config()
    logger.info("Bybit REST client ready: base_url=%s", client.base_url)
    wallet_manager = WalletDataManager(client=client)
    snapshot = wallet_manager.fetch_once()
    logger.info(
        "USDT wallet: equity=%s available=%s wallet_balance=%s",
        snapshot.total_equity,
        snapshot.available_balance,
        snapshot.wallet_balance,
    )
    # TODO: Extend with the actual trading strategy logic.


if __name__ == "__main__":
    main()
