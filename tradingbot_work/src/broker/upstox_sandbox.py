from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

from src.broker.models import (
    BrokerOrder,
    OrderSide,
    OrderStatus,
    OrderType,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = os.getenv(
    "UPSTOX_SANDBOX_BASE_URL",
    "https://api-sandbox.upstox.com/v3",
).rstrip("/")

V2_BASE_URL = os.getenv(
    "UPSTOX_SANDBOX_V2_BASE_URL",
    "https://api-sandbox.upstox.com/v2",
).rstrip("/")

DEFAULT_TIMEOUT = float(
    os.getenv("UPSTOX_HTTP_TIMEOUT", "15")
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UpstoxSandboxError(RuntimeError):
    """Base exception for Upstox Sandbox broker errors."""


class UpstoxSandboxHTTPError(UpstoxSandboxError):
    """Raised when Upstox returns a non-2xx response."""

    def __init__(
        self,
        status_code: int,
        payload: Any,
    ) -> None:
        self.status_code = status_code
        self.payload = payload

        super().__init__(
            f"Upstox Sandbox HTTP {status_code}: {payload}"
        )


# ---------------------------------------------------------------------------
# Upstox order status mapping
# ---------------------------------------------------------------------------

IST = ZoneInfo("Asia/Kolkata")

UPSTOX_STATUS_MAP: dict[str, OrderStatus] = {
    "put order req received": OrderStatus.SUBMITTED,
    "validation pending": OrderStatus.SUBMITTED,
    "open pending": OrderStatus.SUBMITTED,
    "after market order req received": OrderStatus.SUBMITTED,
    "open": OrderStatus.OPEN,
    "trigger pending": OrderStatus.OPEN,
    "modify pending": OrderStatus.OPEN,
    "modify validation pending": OrderStatus.OPEN,
    "cancel pending": OrderStatus.OPEN,
    "not cancelled": OrderStatus.OPEN,
    "not modified": OrderStatus.OPEN,
    "modified": OrderStatus.MODIFIED,
    "partially filled": OrderStatus.PARTIALLY_FILLED,
    "complete": OrderStatus.FILLED,
    "filled": OrderStatus.FILLED,
    "cancelled": OrderStatus.CANCELLED,
    "cancelled after market order": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------

class UpstoxSandboxBroker:
    """
    Upstox Sandbox broker adapter.

    IMPORTANT:
    - This adapter is intended for Sandbox/PAPER execution.
    - It does NOT enable live trading.
    - The access token is read from the environment.
    """

    BASE_URL = BASE_URL
    V2_BASE_URL = V2_BASE_URL

    def __init__(
        self,
        access_token: Optional[str] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = BASE_URL,
        v2_base_url: str = V2_BASE_URL,
    ) -> None:

        self.access_token = (
            access_token
            or os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN")
            or os.getenv("UPSTOX_ACCESS_TOKEN")
        )

        self.timeout = timeout

        self.base_url = base_url.rstrip("/")
        self.v2_base_url = v2_base_url.rstrip("/")

        for url in (self.base_url, self.v2_base_url):
            if "api-sandbox.upstox.com" not in url:
                raise UpstoxSandboxError(
                    f"Refusing non-sandbox Upstox host: {url}"
                )

        self._connected = False

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    # -----------------------------------------------------------------------
    # Connection
    # -----------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Validate broker configuration.

        We intentionally do not call a live account/charges endpoint here.
        Sandbox connectivity is established when an actual Sandbox API
        request is made.
        """

        if not self.access_token:
            raise UpstoxSandboxError(
                "UPSTOX_SANDBOX_ACCESS_TOKEN is not configured."
            )

        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -----------------------------------------------------------------------
    # Headers
    # -----------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if not self.access_token:
            raise UpstoxSandboxError(
                "Upstox Sandbox access token is not configured."
            )

        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    # -----------------------------------------------------------------------
    # Generic HTTP request
    # -----------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        use_v2: bool = False,
    ) -> dict[str, Any]:

        if not self.access_token:
            raise UpstoxSandboxError(
                "Upstox Sandbox access token is not configured."
            )

        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            root = (
                self.v2_base_url
                if use_v2
                else self.base_url
            )

            url = f"{root}/{path.lstrip('/')}"

        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=self._headers(),
                json=json,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise UpstoxSandboxError(
                f"Upstox Sandbox request failed: {exc}"
            ) from exc

        # ---------------------------------------------------------------
        # Parse response safely
        # ---------------------------------------------------------------

        try:
            payload = response.json()
        except ValueError:
            payload = response.text

        # ---------------------------------------------------------------
        # HTTP status validation
        #
        # Do not use response.ok here because some tests/mocks may only
        # provide status_code.
        # ---------------------------------------------------------------

        if response.status_code < 200 or response.status_code >= 300:
            raise UpstoxSandboxHTTPError(
                response.status_code,
                payload,
            )

        if isinstance(payload, dict):
            return payload

        return {
            "status": "success",
            "data": payload,
        }

    # -----------------------------------------------------------------------
    # Place Order
    # -----------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        *,
        product: str = "D",
        validity: str = "DAY",
        trigger_price: float = 0.0,
        disclosed_quantity: int = 0,
        is_amo: bool = False,
        slice: bool = False,
        market_protection: int = -1,
        tag: Optional[str] = None,
    ) -> str:
        """
        Place an order through Upstox Sandbox V3.

        MARKET:
            price is always sent as 0.

        LIMIT:
            price must be > 0.

        SL:
            price and trigger_price are required.

        SL-M:
            trigger_price is required and price remains 0.
        """

        if not self._connected:
            self.connect()

        if not symbol:
            raise ValueError(
                "instrument_token is required."
            )

        if quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero."
            )

        normalized_side = str(side).upper()

        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError(
                f"Invalid transaction side: {side}"
            )

        normalized_order_type = str(
            order_type
        ).upper()

        if normalized_order_type not in {
            "MARKET",
            "LIMIT",
            "SL",
            "SL-M",
        }:
            raise ValueError(
                f"Invalid order type: {order_type}"
            )

        # ---------------------------------------------------------------
        # Upstox V3 payload
        #
        # Current Upstox API expects price to be 0 for MARKET orders.
        # ---------------------------------------------------------------

        payload: dict[str, Any] = {
            "quantity": int(quantity),
            "product": product,
            "validity": validity,
            "price": 0,
            "instrument_token": symbol,
            "order_type": normalized_order_type,
            "transaction_type": normalized_side,
            "disclosed_quantity": int(
                disclosed_quantity
            ),
            "trigger_price": float(
                trigger_price
            ),
            "is_amo": bool(is_amo),
            "slice": bool(slice),
            "market_protection": int(
                market_protection
            ),
        }

        if tag:
            payload["tag"] = str(tag)[:40]

        # ---------------------------------------------------------------
        # LIMIT
        # ---------------------------------------------------------------

        if normalized_order_type == "LIMIT":

            if price is None or float(price) <= 0:
                raise ValueError(
                    "LIMIT order requires price > 0."
                )

            payload["price"] = float(price)

            payload["trigger_price"] = 0.0

        # ---------------------------------------------------------------
        # MARKET
        # ---------------------------------------------------------------

        elif normalized_order_type == "MARKET":

            # IMPORTANT:
            # Upstox Sandbox rejects the strategy entry price here.
            # MARKET order must use price 0.
            payload["price"] = 0.0
            payload["trigger_price"] = 0.0

        # ---------------------------------------------------------------
        # SL
        # ---------------------------------------------------------------

        elif normalized_order_type == "SL":

            if price is None or float(price) <= 0:
                raise ValueError(
                    "SL order requires price > 0."
                )

            if trigger_price is None or float(
                trigger_price
            ) <= 0:
                raise ValueError(
                    "SL order requires trigger_price > 0."
                )

            payload["price"] = float(price)
            payload["trigger_price"] = float(
                trigger_price
            )

        # ---------------------------------------------------------------
        # SL-M
        # ---------------------------------------------------------------

        elif normalized_order_type == "SL-M":

            if trigger_price is None or float(
                trigger_price
            ) <= 0:
                raise ValueError(
                    "SL-M order requires trigger_price > 0."
                )

            payload["price"] = 0.0
            payload["trigger_price"] = float(
                trigger_price
            )

        # ---------------------------------------------------------------
        # Send request
        # ---------------------------------------------------------------

        response = self._request(
            "POST",
            "/order/place",
            json=payload,
        )

        data = response.get("data") or {}

        # Current V3 response:
        #
        # {
        #   "status": "success",
        #   "data": {
        #       "order_ids": ["..."]
        #   }
        # }
        #

        order_ids = data.get("order_ids")

        order_id: Optional[str] = None

        if isinstance(order_ids, list) and order_ids:
            order_id = str(order_ids[0])

        # Backward-compatible fallback.
        if not order_id:
            legacy_order_id = data.get(
                "order_id"
            )

            if legacy_order_id:
                order_id = str(
                    legacy_order_id
                )

        if not order_id:
            raise UpstoxSandboxError(
                "Upstox Sandbox place-order response "
                "did not contain order_id/order_ids. "
                f"Response={response}"
            )

        return order_id

    # -----------------------------------------------------------------------
    # Get Order Details
    # -----------------------------------------------------------------------

    def get_order(
        self,
        order_id: str,
    ) -> BrokerOrder:

        if not order_id:
            raise ValueError(
                "order_id is required."
            )

        # Upstox V2 order details endpoint.
        response = self._request(
            "GET",
            "/order/details",
            params={
                "order_id": str(order_id),
            },
            use_v2=True,
        )

        data = response.get("data") or {}

        # Some responses may return a list.
        if isinstance(data, list):

            if not data:
                raise UpstoxSandboxError(
                    f"No order found for {order_id}"
                )

            raw_order = data[0]

        else:
            raw_order = data

        if not isinstance(raw_order, dict):
            raise UpstoxSandboxError(
                f"Unexpected order details response: "
                f"{response}"
            )

        return self._to_broker_order(
            raw_order,
            fallback_order_id=str(order_id),
        )

    # -----------------------------------------------------------------------
    # Get Order History
    # -----------------------------------------------------------------------

    def get_order_history(
        self,
        order_id: str,
    ) -> list[BrokerOrder]:

        if not order_id:
            raise ValueError(
                "order_id is required."
            )

        response = self._request(
            "GET",
            "/order/history",
            params={
                "order_id": str(order_id),
            },
            use_v2=True,
        )

        data = response.get("data") or []

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return []

        result: list[BrokerOrder] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            result.append(
                self._to_broker_order(
                    item,
                    fallback_order_id=str(
                        order_id
                    ),
                )
            )

        return result

    # -----------------------------------------------------------------------
    # Get all orders / Order Book
    # -----------------------------------------------------------------------

    def get_orders(self) -> list[dict[str, Any]]:

        response = self._request(
            "GET",
            "/order/retrieve-all",
            use_v2=True,
        )

        data = response.get("data") or []

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return []

        return [
            self._to_broker_order(item).model_dump()
            for item in data
            if isinstance(item, dict)
        ]

    # -----------------------------------------------------------------------
    # Cancel Order
    # -----------------------------------------------------------------------

    def cancel_order(
        self,
        order_id: str,
    ) -> bool:

        if not order_id:
            raise ValueError(
                "order_id is required."
            )

        response = self._request(
            "DELETE",
            "/order/cancel",
            params={
                "order_id": str(order_id),
            },
        )

        status = str(
            response.get("status", "")
        ).lower()

        if status == "success":
            return True

        # Some APIs may return successful data without
        # explicitly returning status.
        data = response.get("data")

        if data is not None:
            return True

        raise UpstoxSandboxError(
            f"Unexpected cancel response: {response}"
        )

    # -----------------------------------------------------------------------
    # Modify Order
    # -----------------------------------------------------------------------

    def modify_order(
        self,
        order_id: str,
        *,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        order_type: Optional[str] = None,
        trigger_price: Optional[float] = None,
        validity: str = "DAY",
        disclosed_quantity: int = 0,
        slice: bool = False,
    ) -> bool:

        if not order_id:
            raise ValueError(
                "order_id is required."
            )

        payload: dict[str, Any] = {
            "validity": validity,
            "disclosed_quantity": int(
                disclosed_quantity
            ),
            "slice": bool(slice),
        }

        if quantity is not None:
            if quantity <= 0:
                raise ValueError(
                    "quantity must be greater than zero."
                )

            payload["quantity"] = int(
                quantity
            )

        if order_type is not None:
            payload["order_type"] = str(
                order_type
            ).upper()

        if price is not None:
            payload["price"] = float(
                price
            )

        if trigger_price is not None:
            payload["trigger_price"] = float(
                trigger_price
            )

        self._request(
            "PUT",
            "/order/modify",
            params={
                "order_id": str(order_id),
            },
            json=payload,
        )

        return True

    # -----------------------------------------------------------------------
    # Position / Portfolio helpers
    # -----------------------------------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:

        response = self._request(
            "GET",
            "/portfolio/short-term-positions",
        )

        data = response.get("data") or []

        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return []

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    # -----------------------------------------------------------------------
    # Internal order conversion
    # -----------------------------------------------------------------------

    def _to_broker_order(
        self,
        raw: dict[str, Any],
        *,
        fallback_order_id: Optional[str] = None,
    ) -> BrokerOrder:

        order_id = (
            raw.get("order_id")
            or raw.get("orderId")
            or fallback_order_id
            or ""
        )

        symbol = (
            raw.get("instrument_token")
            or raw.get("instrumentToken")
            or raw.get("symbol")
            or ""
        )

        side = (
            raw.get("transaction_type")
            or raw.get("transactionType")
            or raw.get("side")
            or ""
        )

        quantity = self._safe_int(
            raw.get("quantity"),
            default=0,
        )

        filled_quantity = self._safe_int(
            raw.get("filled_quantity")
            if raw.get("filled_quantity")
            is not None
            else raw.get("filledQuantity"),
            default=0,
        )

        price = self._safe_float(
            raw.get("price"),
            default=0.0,
        )

        average_price = self._safe_float(
            raw.get("average_price")
            if raw.get("average_price")
            is not None
            else raw.get("averagePrice"),
            default=0.0,
        )

        order_type = (
            raw.get("order_type")
            or raw.get("orderType")
            or ""
        )

        raw_status = (
            raw.get("status")
            or raw.get("order_status")
            or raw.get("orderStatus")
            or ""
        )

        status = self._map_status(
            raw_status,
            filled_quantity=filled_quantity,
            quantity=quantity,
        )

        return BrokerOrder(
            order_id=str(order_id),
            symbol=str(symbol),
            side=OrderSide(str(side).upper()),
            quantity=quantity,
            order_type=OrderType(str(order_type).upper()),
            price=price or None,
            status=status,
            created_at=self._parse_timestamp(
                raw.get("order_timestamp")
                or raw.get("orderTimestamp")
            ),
            filled_quantity=filled_quantity,
            average_fill_price=average_price or None,
            rejection_reason=(
                raw.get("status_message")
                or raw.get("statusMessage")
                if status == OrderStatus.REJECTED
                else None
            ),
            tag=raw.get("tag") or None,
        )

    @staticmethod
    def _map_status(
        raw_status: Any,
        *,
        filled_quantity: int,
        quantity: int,
    ) -> OrderStatus:

        key = (
            str(raw_status)
            .strip()
            .lower()
            .replace("_", " ")
        )

        status = UPSTOX_STATUS_MAP.get(
            key,
            OrderStatus.SUBMITTED,
        )

        if (
            status == OrderStatus.OPEN
            and 0 < filled_quantity < quantity
        ):
            return OrderStatus.PARTIALLY_FILLED

        return status

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> datetime:

        if value:
            try:
                parsed = datetime.fromisoformat(
                    str(value)
                )
            except ValueError:
                parsed = None

            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=IST)
                return parsed

        return datetime.now(timezone.utc)

    # -----------------------------------------------------------------------
    # Safe conversions
    # -----------------------------------------------------------------------

    @staticmethod
    def _safe_int(
        value: Any,
        *,
        default: int = 0,
    ) -> int:

        if value is None:
            return default

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(
        value: Any,
        *,
        default: float = 0.0,
    ) -> float:

        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

UpstoxSandbox = UpstoxSandboxBroker
UpstoxBroker = UpstoxSandboxBroker