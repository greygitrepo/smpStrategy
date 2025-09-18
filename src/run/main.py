from __future__ import annotations

"""Main entry point for launching the project directly from an IDE (e.g. F5)."""

import logging
import sys
import time
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
from src.config import (  # noqa: E402  pylint: disable=wrong-import-position
    default_new_listing_strategy_config,
    maybe_load_new_listing_strategy_config,
    maybe_load_symbol_strategy_config,
)
from src.data import WalletDataManager  # noqa: E402  pylint: disable=wrong-import-position
from src.data.symbol import (  # noqa: E402  pylint: disable=wrong-import-position
    ActivePositionTracker,
    DEFAULT_CATEGORY,
    DEFAULT_SETTLE_COIN,
    SymbolDiscoveryStrategy,
)
from src.exchange.bybit_v5 import BybitV5Client  # noqa: E402  pylint: disable=wrong-import-position
from src.strategy import NewListingTradingStrategy  # noqa: E402  pylint: disable=wrong-import-position

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
    strategy_config = maybe_load_symbol_strategy_config()
    category = (
        strategy_config.category
        if strategy_config and strategy_config.category
        else getattr(client, "default_category", DEFAULT_CATEGORY)
    )
    settle_coin = (
        strategy_config.settle_coin
        if strategy_config and strategy_config.settle_coin
        else DEFAULT_SETTLE_COIN
    )
    position_tracker = ActivePositionTracker(
        client=client,
        category=category,
        settle_coin=settle_coin,
    )
    positions = position_tracker.positions()
    if positions:
        for symbol, pos in positions.items():
            logger.info(
                "Open position %s: side=%s size=%s entry_price=%s leverage=%s unrealized_pnl=%s pnl_rate=%s%%",
                symbol,
                pos.side,
                pos.size,
                pos.entry_price,
                pos.leverage,
                pos.unrealized_pnl,
                f"{pos.pnl_rate:.2f}",
            )
    else:
        logger.info(
            "No open positions detected: category=%s settle_coin=%s",
            category,
            settle_coin,
        )
    discovery = SymbolDiscoveryStrategy(
        client=client,
        strategy_config=strategy_config,
    )
    symbols: list[str] = []
    try:
        symbols = discovery.fetch_once()
        if symbols:
            logger.info(
                "Discovered symbols (%s, limit=%s): %s",
                discovery.strategy_name,
                discovery.limit,
                ", ".join(symbols),
            )
        else:
            logger.warning(
                "Symbol discovery (%s) returned no symbols.",
                discovery.strategy_name,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Symbol discovery failed: %s", exc)
    trading_config = (
        maybe_load_new_listing_strategy_config()
        or default_new_listing_strategy_config()
    )
    trading_strategy = NewListingTradingStrategy(
        client=client,
        discovery=discovery,
        wallet_manager=wallet_manager,
        position_tracker=position_tracker,
        config=trading_config,
        category=category,
    )
    first_symbols: list[str] | None = symbols
    logger.info("Starting continuous new-listing strategy loop with 5s delay")
    try:
        while True:
            try:
                trading_strategy.run_once(initial_candidates=first_symbols)
            except Exception as exc:  # noqa: BLE001
                logger.exception("New listing trading strategy failed: %s", exc)
            first_symbols = None
            time.sleep(5.0)
    except KeyboardInterrupt:
        logger.info("Received interrupt, stopping trading loop")


if __name__ == "__main__":
    main()
