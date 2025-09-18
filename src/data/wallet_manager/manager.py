from __future__ import annotations

"""Utilities for polling Bybit wallet balances on a fixed interval."""

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.config.account_config import maybe_load_account_config, resolve_config_path
from src.config.wallet_config import maybe_load_wallet_config
from src.data.wallet_manager.constants import DEFAULT_REFRESH_INTERVAL
from src.exchange.bybit_v5 import BybitV5Client

logger = logging.getLogger("smpStrategy.wallet")


@dataclass(slots=True)
class WalletSnapshot:
    account_type: str
    coin: str
    total_equity: float
    available_balance: float
    wallet_balance: float
    fetched_at: datetime
    raw: dict[str, Any]


class WalletDataManager:
    """Periodically load wallet state for USDT-based derivatives trading."""

    def __init__(
        self,
        *,
        client: Optional[BybitV5Client] = None,
        account_type: str = "UNIFIED",
        coin: str = "USDT",
        refresh_interval: Optional[float] = None,
        wallet_config_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if client is None:
            config = maybe_load_account_config()
            if config is None:
                default_path = resolve_config_path()
                raise FileNotFoundError(
                    "accountConfig.ini not found. Run `python -m src.config.account_config` "
                    f"or create one based on {default_path.parent / 'sampleConfig.ini'}."
                )
            client = BybitV5Client(
                api_key=config.api_key,
                api_secret=config.api_secret,
                testnet=config.testnet,
                category=config.category,
            )
        self._client = client
        self._account_type = account_type
        self._coin = coin

        if refresh_interval is None:
            wallet_config = maybe_load_wallet_config(wallet_config_path)
            if wallet_config is not None:
                refresh_interval = wallet_config.refresh_interval
        self.refresh_interval = refresh_interval or DEFAULT_REFRESH_INTERVAL

        self._snapshot_lock = threading.Lock()
        self._snapshot: Optional[WalletSnapshot] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def snapshot(self) -> Optional[WalletSnapshot]:
        with self._snapshot_lock:
            return self._snapshot

    def fetch_once(self) -> WalletSnapshot:
        response = self._client.get_wallet_balance(
            accountType=self._account_type,
            coin=self._coin,
        )
        snapshot = self._parse_response(response)
        with self._snapshot_lock:
            self._snapshot = snapshot
        return snapshot

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="WalletDataManager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.refresh_interval * 2)
        self._thread = None

    def _run(self) -> None:
        logger.info(
            "Starting wallet refresh thread: interval=%ss account_type=%s coin=%s",
            self.refresh_interval,
            self._account_type,
            self._coin,
        )
        while not self._stop_event.is_set():
            start = time.perf_counter()
            try:
                self.fetch_once()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to refresh wallet data: %s", exc)
            elapsed = time.perf_counter() - start
            wait_time = max(0.0, self.refresh_interval - elapsed)
            if self._stop_event.wait(wait_time):
                break
        logger.info("Wallet refresh thread stopped")

    def _parse_response(self, response: dict[str, Any]) -> WalletSnapshot:
        result = response.get("result") or {}
        entries = result.get("list") or []
        if not entries:
            raise ValueError("Unexpected wallet response: missing list")
        entry = entries[0]
        coin_entries = entry.get("coin") or []
        if not coin_entries:
            raise ValueError("Unexpected wallet response: missing coin data")
        desired_coin = next(
            (c for c in coin_entries if c.get("coin") == self._coin),
            coin_entries[0],
        )
        def _get_float(source: dict[str, Any], key: str) -> Optional[float]:
            value = source.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.debug("Failed to cast %s=%s to float", key, value)
                return None

        def _as_float(key: str) -> float:
            value = _get_float(desired_coin, key)
            return value if value is not None else 0.0

        def _available_balance() -> float:
            for key in ("totalAvailableBalance",):
                value = _get_float(entry, key)
                if value is not None:
                    return value
            coin_keys = ("availableBalance", "availableToTrade", "availableToWithdraw")
            for key in coin_keys:
                value = _get_float(desired_coin, key)
                if value is not None:
                    return value
            wallet_balance = _get_float(desired_coin, "walletBalance")
            total_position_im = _get_float(desired_coin, "totalPositionIM") or 0.0
            total_order_im = _get_float(desired_coin, "totalOrderIM") or 0.0
            if wallet_balance is not None:
                available = wallet_balance - total_position_im - total_order_im
                return max(0.0, available)
            equity = _get_float(desired_coin, "equity")
            if equity is not None:
                available = equity - total_position_im - total_order_im
                return max(0.0, available)
            return 0.0
        snapshot = WalletSnapshot(
            account_type=entry.get("accountType", self._account_type),
            coin=desired_coin.get("coin", self._coin),
            total_equity=_as_float("equity"),
            available_balance=_available_balance(),
            wallet_balance=_as_float("walletBalance"),
            fetched_at=datetime.now(timezone.utc),
            raw=response,
        )
        logger.debug("Wallet snapshot updated: %s", snapshot)
        return snapshot

    def __enter__(self) -> "WalletDataManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, D401
        self.stop()
