"""Upstox V3 live market-data feed factory (read-only WebSocket)."""
from __future__ import annotations

from src.market_data.transport import WebSocketClientTransport
from src.market_data.upstox_v3 import (
    UpstoxV3MarketDataError,
    authorize_v3,
    decode_ticks,
    resolve_market_data_access_token,
    subscription_payload,
)
from src.market_data.websocket import WebSocketMarketDataFeed


def upstox_market_data_token() -> tuple[str, str]:
    return resolve_market_data_access_token()


def upstox_access_token() -> str:
    return upstox_market_data_token()[0]


def create_upstox_v3_feed(
    *,
    access_token: str | None = None,
    stale_after_seconds: int = 15,
) -> WebSocketMarketDataFeed:
    token, _ = resolve_market_data_access_token(access_token)
    try:
        url = authorize_v3(token)
    except UpstoxV3MarketDataError as exc:
        if "401" in str(exc):
            raise UpstoxV3MarketDataError(
                f"{exc} — market data authorize needs UPSTOX_MARKET_DATA_ACCESS_TOKEN "
                f"or ACCESS_TOKEN (not UPSTOX_SANDBOX_ACCESS_TOKEN)."
            ) from exc
        raise

    def transport_factory(ws_url: str):
        transport = WebSocketClientTransport(ws_url, timeout=10.0)
        transport.connect()
        return transport

    def subscribe(keys: list[str]) -> bytes:
        return subscription_payload(keys, mode="ltpc")

    return WebSocketMarketDataFeed(
        url,
        transport_factory=transport_factory,
        subscribe_message=subscribe,
        parser=decode_ticks,
        stale_after_seconds=stale_after_seconds,
        reconnect_enabled=True,
    )
