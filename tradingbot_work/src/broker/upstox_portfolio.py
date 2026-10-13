from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import requests

from src.market_data.transport import WebSocketClientTransport

AUTHORIZE_URL = "https://api.upstox.com/v2/feed/portfolio-stream-feed/authorize"


class UpstoxPortfolioStreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrderUpdate:
    order_id: str
    status: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    pending_quantity: int
    average_price: float
    exchange_order_id: str | None = None
    status_message: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class PositionUpdate:
    instrument_key: str
    symbol: str
    quantity: int
    average_price: float
    buy_value: float = 0.0
    day_buy_value: float = 0.0
    day_sell_value: float = 0.0
    unrealized_pnl: float | None = None


def authorize_portfolio_stream(access_token: str | None = None, *, update_types: tuple[str, ...] = ("order", "position"), timeout: float = 10.0) -> str:
    token = (access_token or os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN") or os.getenv("ACCESS_TOKEN") or "").strip()
    if not token:
        raise UpstoxPortfolioStreamError("Upstox access token is required")
    params = {"update_types": ",".join(update_types)}
    response = requests.get(AUTHORIZE_URL, params=params, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=timeout)
    if response.status_code >= 400:
        raise UpstoxPortfolioStreamError(f"portfolio authorize HTTP {response.status_code}: {response.text[:500]}")
    body = response.json()
    uri = body.get("data", {}).get("authorized_redirect_uri")
    if not uri or not uri.startswith("wss://"):
        raise UpstoxPortfolioStreamError("Invalid portfolio authorized_redirect_uri")
    return uri


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_portfolio_message(message: str | bytes) -> OrderUpdate | PositionUpdate | None:
    try:
        payload = json.loads(message.decode() if isinstance(message, bytes) else message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpstoxPortfolioStreamError("Invalid portfolio stream JSON payload") from exc

    update_type = payload.get("update_type")
    if update_type == "order":
        return OrderUpdate(
            order_id=str(payload.get("order_id") or ""),
            status=str(payload.get("status") or "").strip(),
            symbol=str(payload.get("instrument_key") or payload.get("instrument_token") or payload.get("trading_symbol") or ""),
            side=str(payload.get("transaction_type") or "").upper(),
            quantity=_as_int(payload.get("quantity")),
            filled_quantity=_as_int(payload.get("filled_quantity")),
            pending_quantity=_as_int(payload.get("pending_quantity")),
            average_price=_as_float(payload.get("average_price")),
            exchange_order_id=payload.get("exchange_order_id") or None,
            status_message=payload.get("status_message") or None,
            timestamp=payload.get("exchange_timestamp") or payload.get("order_timestamp") or None,
        )
    if update_type == "position":
        return PositionUpdate(
            instrument_key=str(payload.get("instrument_key") or payload.get("instrument_token") or ""),
            symbol=str(payload.get("trading_symbol") or payload.get("instrument_key") or ""),
            quantity=_as_int(payload.get("quantity")),
            average_price=_as_float(payload.get("average_price")),
            buy_value=_as_float(payload.get("buy_value")),
            day_buy_value=_as_float(payload.get("day_buy_value")),
            day_sell_value=_as_float(payload.get("day_sell_value")),
            unrealized_pnl=_as_float(payload.get("unrealized_pnl")) if payload.get("unrealized_pnl") is not None else None,
        )
    return None


class UpstoxPortfolioStreamer:
    """Real-time Upstox order/position stream with injectable transport for tests."""

    def __init__(self, access_token: str | None = None, *, transport_factory: Callable[[str], Any] | None = None, timeout: float = 10.0):
        self.access_token = access_token
        self.timeout = timeout
        self.transport_factory = transport_factory or (lambda url: WebSocketClientTransport(url, timeout=timeout))
        self.transport: Any | None = None
        self.connected = False

    def connect(self) -> None:
        uri = authorize_portfolio_stream(self.access_token, timeout=self.timeout)
        self.transport = self.transport_factory(uri)
        if hasattr(self.transport, "connect"):
            self.transport.connect()
        self.connected = True

    def recv(self) -> OrderUpdate | PositionUpdate | None:
        if not self.transport or not self.connected:
            raise UpstoxPortfolioStreamError("portfolio streamer is not connected")
        return parse_portfolio_message(self.transport.recv())

    def close(self) -> None:
        if self.transport:
            self.transport.close()
        self.transport = None
        self.connected = False
