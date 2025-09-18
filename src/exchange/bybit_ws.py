"""Optional Bybit v5 private WebSocket client (testnet/mainnet).

Requires `websocket-client` package. If not installed, this module can be imported
but the client will raise at runtime with a clear message.

Features:
- Auth via v5 login payload (ts, apiKey, recvWindow, sign)
- Subscriptions: order, execution, position
- Ping/pong and basic reconnect backoff skeleton

Usage is optional and controlled via env `ENABLE_PRIVATE_WS=true` in the runner.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from typing import Any, Callable, Optional


try:
    import websocket  # type: ignore
except Exception:  # noqa: BLE001
    websocket = None  # type: ignore


DEFAULT_WS_MAINNET = "wss://stream.bybit.com/v5/private"
DEFAULT_WS_TESTNET = "wss://stream-testnet.bybit.com/v5/private"


from ..config.account_config import (
    BybitAccountConfig,
    maybe_load_account_config,
    resolve_config_path,
)


class BybitPrivateWS:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        testnet: bool | None = None,
        recv_window_ms: int = 5000,
        on_message: Optional[Callable[[dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        config_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if websocket is None:
            raise RuntimeError(
                "websocket-client is required for private WS; please install it"
            )
        should_try_config = False
        resolved_path: str | os.PathLike[str] | None = None
        if config_path is not None:
            should_try_config = True
            resolved_path = config_path
        else:
            env_path = os.environ.get("BYBIT_ACCOUNT_CONFIG")
            if env_path:
                should_try_config = True
                resolved_path = env_path
            else:
                default_candidate = resolve_config_path()
                if default_candidate.exists():
                    should_try_config = True
                    resolved_path = default_candidate

        account_config: BybitAccountConfig | None = None
        if should_try_config:
            account_config = maybe_load_account_config(
                resolved_path, strict=config_path is not None
            )

        self._account_config = account_config
        self.api_key = api_key or (account_config.api_key if account_config else None)
        if not self.api_key:
            self.api_key = os.environ.get("BYBIT_API_KEY", "")
        self.api_secret = api_secret or (
            account_config.api_secret if account_config else None
        )
        if not self.api_secret:
            self.api_secret = os.environ.get("BYBIT_API_SECRET", "")
        if testnet is None:
            if account_config is not None:
                testnet = account_config.testnet
            else:
                testnet = os.environ.get("TESTNET", "true").lower() == "true"
        self.url = DEFAULT_WS_TESTNET if testnet else DEFAULT_WS_MAINNET
        self.recv_window_ms = recv_window_ms
        self.on_message = on_message
        self.on_error = on_error
        self.ws: Optional["websocket.WebSocketApp"] = None
        self._thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()

    @property
    def account_config(self) -> BybitAccountConfig | None:
        return self._account_config

    def _sign(self, ts_ms: int) -> str:
        prehash = f"{ts_ms}{self.api_key}{self.recv_window_ms}"
        return hmac.new(
            self.api_secret.encode(), prehash.encode(), hashlib.sha256
        ).hexdigest()

    def _on_open(self, ws: "websocket.WebSocketApp") -> None:  # noqa: ARG002
        ts = int(time.time() * 1000)
        payload = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.api_key,
                    "timestamp": ts,
                    "recvWindow": self.recv_window_ms,
                    "sign": self._sign(ts),
                }
            ],
        }
        ws.send(json.dumps(payload))
        # Subscribe to default channels
        sub = {"op": "subscribe", "args": ["order", "execution", "position"]}
        ws.send(json.dumps(sub))

    def _on_message(self, ws: "websocket.WebSocketApp", message: str) -> None:  # noqa: ARG002
        try:
            data = json.loads(message)
            if self.on_message:
                self.on_message(data)
        except Exception as e:  # noqa: BLE001
            if self.on_error:
                self.on_error(e)

    def _on_error(self, ws: "websocket.WebSocketApp", error: Exception) -> None:  # noqa: ARG002
        if self.on_error:
            self.on_error(error)

    def _on_close(
        self, ws: "websocket.WebSocketApp", status_code: int, msg: str
    ) -> None:  # noqa: ARG002
        # Will reconnect in run loop if not stopped
        pass

    def start(self) -> None:
        if websocket is None:
            raise RuntimeError(
                "websocket-client is required for private WS; please install it"
            )
        self._should_stop.clear()
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        def _run() -> None:
            backoff = 1
            while not self._should_stop.is_set():
                try:
                    self.ws.run_forever(ping_interval=20, ping_timeout=10)  # type: ignore[arg-type]
                    backoff = 1
                except Exception as e:  # noqa: BLE001
                    if self.on_error:
                        self.on_error(e)
                    time.sleep(backoff)
                    backoff = min(60, backoff * 2)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._should_stop.set()
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
