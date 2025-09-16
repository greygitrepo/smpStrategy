"""Bybit v5 REST client (testnet/mainnet) with HMAC auth and retries.

This client implements a minimal subset required for live testnet smoke:
- get_symbols
- get_orderbook
- place_order
- cancel_order
- get_open_orders
- get_positions
- get_wallet_balance

Env vars used (with reasonable defaults for testnet):
- BYBIT_API_KEY, BYBIT_API_SECRET
- TESTNET=true|false (default true)
- BYBIT_CATEGORY=linear (linear|inverse|spot)

Notes:
- Signing follows v5 spec: sign = HMAC_SHA256(secret, ts + apiKey + recvWindow + payload)
- payload is querystring for GET/DELETE, or minified JSON string for POST
- Headers: X-BAPI-API-KEY, X-BAPI-SIGN, X-BAPI-TIMESTAMP, X-BAPI-RECV-WINDOW, Content-Type
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import httpx


DEFAULT_BASE_MAINNET = "https://api.bybit.com"
DEFAULT_BASE_TESTNET = "https://api-testnet.bybit.com"


class BybitAPIError(Exception):
    def __init__(self, ret_code: int, ret_msg: str, data: Any | None = None):
        super().__init__(f"Bybit API error {ret_code}: {ret_msg}")
        self.ret_code = ret_code
        self.ret_msg = ret_msg
        self.data = data


class EdgeProtectionError(BybitAPIError):
    """Raised when Bybit public edge returns HTML/non-JSON (403/5xx) after retries."""

    pass


def _canon_side(side: str) -> str:
    """Normalize user-provided side into Bybit v5 canonical casing.

    Accepts common variants: buy/BUY/long -> Buy, sell/SELL/short -> Sell.
    """
    s = str(side).strip().lower()
    if s in {"buy", "long"}:
        return "Buy"
    if s in {"sell", "short"}:
        return "Sell"
    # Already canonical? guard for unexpected casing
    if s in {"buy", "sell"}:
        return s.capitalize()
    raise ValueError(f"invalid side: {side}")


class BybitV5Client:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        testnet: bool | None = None,
        base_url: Optional[str] = None,
        recv_window_ms: int = 5000,
        timeout: float = 10.0,
        category: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("BYBIT_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            # Allow unauthenticated market endpoints, but warn on private usage
            pass
        if testnet is None:
            testnet = os.environ.get("TESTNET", "true").lower() == "true"
        self.base_url = base_url or (
            DEFAULT_BASE_TESTNET if testnet else DEFAULT_BASE_MAINNET
        )
        self.recv_window_ms = recv_window_ms
        # Configure granular timeouts to avoid long hangs on CI/public edges
        try:
            timeout_obj = httpx.Timeout(connect=5.0, read=7.0, write=5.0, pool=5.0)
        except Exception:
            timeout_obj = timeout
        self._client = httpx.Client(timeout=timeout_obj)
        self.default_category = category or os.environ.get("BYBIT_CATEGORY", "linear")

    # -------- Signing helpers --------
    @staticmethod
    def _canonical_query(params: Dict[str, Any] | None) -> str:
        if not params:
            return ""
        # Exclude None values; preserve insertion order to match actual sent query
        # Some Bybit edges verify signature against the exact key order sent.
        items = [(k, v) for k, v in params.items() if v is not None]
        return "&".join(
            f"{k}={kv if isinstance((kv := v), str) else json.dumps(v, separators=(',', ':'))}"
            for k, v in items
        )

    @staticmethod
    def _minified_json(data: Dict[str, Any] | None) -> str:
        if not data:
            return ""
        # Remove None to avoid signing nulls inadvertently
        clean = {k: v for k, v in data.items() if v is not None}
        return json.dumps(clean, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _build_prehash(ts: str, api_key: str, recv_window: str, payload: str) -> str:
        return f"{ts}{api_key}{recv_window}{payload}"

    def _sign(self, ts_ms: int, payload: str) -> str:
        recv = str(self.recv_window_ms)
        prehash = self._build_prehash(str(ts_ms), self.api_key, recv, payload)
        sig = hmac.new(
            self.api_secret.encode(), prehash.encode(), hashlib.sha256
        ).hexdigest()
        return sig

    # -------- Core request --------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
        data: Dict[str, Any] | None = None,
        auth: bool = False,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Some Bybit edges return non-JSON without UA; set a stable UA
            "User-Agent": "cdx-trading-bot/1.0 (+httpx)",
        }
        ts_ms = int(time.time() * 1000)
        start_ts = time.time()
        total_timeout = float(os.environ.get("BYBIT_HTTP_TOTAL_TIMEOUT", "20"))
        attempt = 0
        while True:
            # Total elapsed guard
            if (time.time() - start_ts) > total_timeout:
                raise BybitAPIError(
                    408, f"total timeout exceeded {total_timeout}s for {path}"
                )
            # Ensure we send the exact same payload that we sign
            clean_params: Dict[str, Any] | None = None
            clean_data: Dict[str, Any] | None = None
            payload_for_sign = ""
            if auth:
                if method.upper() in {"GET", "DELETE"}:
                    # Exclude None values from query params for both signing and sending
                    if params:
                        clean_params = {
                            k: v for k, v in params.items() if v is not None
                        }
                    else:
                        clean_params = None
                    payload_for_sign = self._canonical_query(clean_params)
                else:
                    # Exclude None values from body for both signing and sending
                    if data:
                        clean_data = {k: v for k, v in data.items() if v is not None}
                    else:
                        clean_data = None
                    payload_for_sign = self._minified_json(clean_data)
                headers.update(
                    {
                        "X-BAPI-API-KEY": self.api_key,
                        "X-BAPI-TIMESTAMP": str(ts_ms),
                        "X-BAPI-RECV-WINDOW": str(self.recv_window_ms),
                        "X-BAPI-SIGN": self._sign(ts_ms, payload_for_sign),
                        # v5 signing type: 2 = HMAC-SHA256
                        "X-BAPI-SIGN-TYPE": "2",
                    }
                )

            try:
                # Use cleaned payloads when auth is required to avoid sign mismatches
                resp = self._client.request(
                    method,
                    url,
                    params=(clean_params if auth else params),
                    json=(clean_data if auth else data),
                    headers=headers,
                )
            except httpx.RequestError:
                if attempt >= max_retries:
                    raise
                time.sleep(2**attempt)
                attempt += 1
                continue

            if resp.status_code == 429 and attempt < max_retries:
                time.sleep(2**attempt)
                attempt += 1
                continue

            # Attempt to parse JSON; if not JSON, handle potential HTML edge protections
            try:
                j = resp.json()
            except Exception:
                body = None
                try:
                    body = resp.text
                except Exception:
                    body = "<unreadable>"
                snippet = (body or "")[:200]
                status = resp.status_code
                hdrs = dict(resp.headers)
                # Retry on common edge-protection statuses (HTML/non-JSON)
                if status in {403, 502, 520, 521} and attempt < max_retries:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                if status in {403, 502, 520, 521}:
                    raise EdgeProtectionError(
                        status,
                        f"HTTP {status} non-JSON at {path}: {snippet}",
                        {"status": status, "headers": hdrs, "body": snippet},
                    )
                raise BybitAPIError(
                    status,
                    f"HTTP {status} non-JSON response: {snippet}",
                    {"status": status, "headers": hdrs, "body": body},
                )
            ret_code = j.get("retCode", 0)
            if ret_code != 0:
                # Retry on transient codes, else raise
                if attempt < max_retries and ret_code in {10006, 10016, 10018, 110001}:
                    time.sleep(2**attempt)
                    attempt += 1
                    continue
                raise BybitAPIError(ret_code, j.get("retMsg", "unknown"), j)
            return j

    # -------- Public market endpoints --------
    def get_symbols(self, category: Optional[str] = None) -> Dict[str, Any]:
        params = {"category": category or self.default_category}
        return self._request(
            "GET", "/v5/market/instruments-info", params=params, auth=False
        )

    # Backward/explicit alias per spec
    def get_instruments(self, category: Optional[str] = None) -> Dict[str, Any]:
        return self.get_symbols(category=category)

    def get_orderbook(
        self, symbol: str, depth: int = 1, category: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {
            "category": category or self.default_category,
            "symbol": symbol,
            "limit": depth,
        }
        return self._request("GET", "/v5/market/orderbook", params=params, auth=False)

    def get_tickers(
        self, category: Optional[str] = None, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {"category": category or self.default_category, "symbol": symbol}
        return self._request("GET", "/v5/market/tickers", params=params, auth=False)

    def get_fee_rate(
        self, category: Optional[str] = None, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch account-specific maker/taker fee rates for a symbol/category (requires auth)."""
        params = {"category": category or self.default_category, "symbol": symbol}
        return self._request("GET", "/v5/account/fee-rate", params=params, auth=True)

    def get_executions(
        self,
        *,
        symbol: Optional[str] = None,
        category: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: Optional[int] = None,
        orderId: Optional[str] = None,
        orderLinkId: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List recent executions (fills). Supports filtering by symbol/time/order.

        Note: Bybit returns paginated results; for simplicity we expose a single-call fetch.
        """
        params: Dict[str, Any] = {
            "category": category or self.default_category,
            "symbol": symbol,
            "start": start,
            "end": end,
            "limit": limit,
            "orderId": orderId,
            "orderLinkId": orderLinkId,
        }
        return self._request("GET", "/v5/execution/list", params=params, auth=True)

    # -------- Private trading/account endpoints --------
    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: str | float,
        price: str | float | None = None,
        orderType: str = "Market",
        timeInForce: str = "GTC",
        orderLinkId: Optional[str] = None,
        category: Optional[str] = None,
        reduceOnly: Optional[bool] = None,
        takeProfit: Optional[str | float] = None,
        stopLoss: Optional[str | float] = None,
        tpTriggerBy: Optional[str] = None,
        slTriggerBy: Optional[str] = None,
        positionIdx: Optional[int] = None,
    ) -> Dict[str, Any]:
        canon_side = _canon_side(side)
        cat = category or self.default_category or "linear"
        payload = {
            "category": cat,
            "symbol": symbol,
            "side": canon_side,
            "orderType": orderType,
            "qty": str(qty),
            "timeInForce": timeInForce,
            "orderLinkId": orderLinkId,
            "price": str(price) if price is not None else None,
            "reduceOnly": reduceOnly,
            "takeProfit": str(takeProfit) if takeProfit is not None else None,
            "stopLoss": str(stopLoss) if stopLoss is not None else None,
            "tpTriggerBy": tpTriggerBy,
            "slTriggerBy": slTriggerBy,
            "positionIdx": positionIdx,
        }
        return self._request("POST", "/v5/order/create", data=payload, auth=True)

    def cancel_order(
        self,
        *,
        symbol: str,
        orderId: Optional[str] = None,
        orderLinkId: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "category": category or self.default_category,
            "symbol": symbol,
            "orderId": orderId,
            "orderLinkId": orderLinkId,
        }
        return self._request("POST", "/v5/order/cancel", data=payload, auth=True)

    def get_open_orders(
        self, symbol: Optional[str] = None, *, category: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {
            "category": category or self.default_category,
            "symbol": symbol,
            "openOnly": 1,
        }
        return self._request("GET", "/v5/order/realtime", params=params, auth=True)

    def get_positions(
        self,
        *,
        category: Optional[str] = None,
        symbol: Optional[str] = None,
        settleCoin: Optional[str] = None,
    ) -> Dict[str, Any]:
        if symbol is None and settleCoin is None:
            raise ValueError(
                "Bybit v5 requires either symbol or settleCoin for positions list"
            )
        params = {
            "category": category or self.default_category,
            "symbol": symbol,
            "settleCoin": settleCoin,
        }
        return self._request("GET", "/v5/position/list", params=params, auth=True)

    def get_wallet_balance(
        self, *, accountType: str = "UNIFIED", coin: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {"accountType": accountType, "coin": coin}
        return self._request(
            "GET", "/v5/account/wallet-balance", params=params, auth=True
        )

    def set_leverage(
        self,
        *,
        symbol: str,
        buyLeverage: int | float,
        sellLeverage: int | float,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "category": category or self.default_category,
            "symbol": symbol,
            "buyLeverage": str(buyLeverage),
            "sellLeverage": str(sellLeverage),
        }
        return self._request(
            "POST", "/v5/position/set-leverage", data=payload, auth=True
        )

    def set_trading_stop(
        self,
        *,
        symbol: str,
        takeProfit: Optional[str | float] = None,
        stopLoss: Optional[str | float] = None,
        trailingStop: Optional[str | float] = None,
        tpTriggerBy: Optional[str] = None,
        slTriggerBy: Optional[str] = None,
        category: Optional[str] = None,
        positionIdx: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "category": category or self.default_category,
            "symbol": symbol,
            "takeProfit": str(takeProfit) if takeProfit is not None else None,
            "stopLoss": str(stopLoss) if stopLoss is not None else None,
            "trailingStop": str(trailingStop) if trailingStop is not None else None,
            "tpTriggerBy": tpTriggerBy,
            "slTriggerBy": slTriggerBy,
            "positionIdx": positionIdx,
        }
        return self._request(
            "POST", "/v5/position/trading-stop", data=payload, auth=True
        )

    def close_position_market(
        self,
        *,
        symbol: str,
        side: str,
        qty: str | float,
        category: Optional[str] = None,
        positionIdx: Optional[int] = None,
        orderLinkId: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Close position with a market reduce-only order."""
        canon_side = _canon_side(side)
        payload = {
            "category": category or self.default_category,
            "symbol": symbol,
            "side": canon_side,
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": True,
            "timeInForce": "IOC",
            "positionIdx": positionIdx,
            "orderLinkId": orderLinkId,
        }
        return self._request("POST", "/v5/order/create", data=payload, auth=True)

    # -------- Instrument utilities --------
    @staticmethod
    def extract_symbol_filters(resp: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Extract tickSize/qtyStep/minQty style filters from instruments-info response.

        Returns keys: tickSize, qtyStep, minOrderQty (if available), lotSizeFilter, priceFilter
        """
        try:
            items = resp.get("result", {}).get("list", [])
            for it in items:
                if it.get("symbol") == symbol:
                    pf = it.get("priceFilter", {})
                    lf = it.get("lotSizeFilter", {})
                    return {
                        "tickSize": float(pf.get("tickSize"))
                        if pf.get("tickSize") is not None
                        else None,
                        "qtyStep": float(lf.get("qtyStep"))
                        if lf.get("qtyStep") is not None
                        else None,
                        "minOrderQty": float(lf.get("minOrderQty"))
                        if lf.get("minOrderQty") is not None
                        else None,
                        "priceFilter": pf,
                        "lotSizeFilter": lf,
                    }
        except Exception:
            pass
        return {"tickSize": None, "qtyStep": None, "minOrderQty": None}

    # -------- Utilities --------
    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
