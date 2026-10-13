from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from src.market_data.models import Tick
from src.market_data.transport import WebSocketClientTransport
from src.market_data.websocket import WebSocketMarketDataFeed, json_tick_parser


@dataclass(frozen=True)
class MarketDataEndpoint:
    url: str
    headers: tuple[str, ...] = ()
    timeout_seconds: float = 10.0


class ConfiguredBrokerMarketDataFeed(WebSocketMarketDataFeed):
    """Broker-neutral production feed.

    A broker adapter supplies its endpoint, authentication headers, subscription
    payload builder and parser. No broker protocol is guessed here.
    """

    def __init__(
        self,
        endpoint: MarketDataEndpoint,
        *,
        subscribe_message: Callable[[list[str]], str],
        parser: Callable[[str | bytes], list[Tick]] = json_tick_parser,
        **kwargs,
    ):
        super().__init__(
            endpoint.url,
            transport_factory=lambda url: self._transport(endpoint, url),
            subscribe_message=subscribe_message,
            parser=parser,
            **kwargs,
        )

    @staticmethod
    def _transport(endpoint: MarketDataEndpoint, url: str) -> WebSocketClientTransport:
        transport = WebSocketClientTransport(url, headers=list(endpoint.headers), timeout=endpoint.timeout_seconds)
        transport.connect()
        return transport


def json_subscription(symbols: list[str]) -> str:
    return json.dumps({"action": "subscribe", "symbols": symbols}, separators=(",", ":"))


def endpoint_from_environment() -> MarketDataEndpoint:
    url = os.getenv("BROKER_WS_URL", "").strip()
    if not url:
        raise ValueError("BROKER_WS_URL is required for a real market-data feed")
    token = os.getenv("ACCESS_TOKEN", "").strip()
    headers = (f"Authorization: Bearer {token}",) if token else ()
    timeout = float(os.getenv("BROKER_WS_TIMEOUT_SECONDS", "10"))
    return MarketDataEndpoint(url=url, headers=headers, timeout_seconds=timeout)
