import os

import pytest

from src.market_data.provider import MarketDataEndpoint, ConfiguredBrokerMarketDataFeed, endpoint_from_environment, json_subscription
from src.market_data.transport import WebSocketClientTransport


def test_subscription_payload():
    assert json_subscription(["NIFTY", "RELIANCE"]) == '{"action":"subscribe","symbols":["NIFTY","RELIANCE"]}'


def test_endpoint_requires_url(monkeypatch):
    monkeypatch.delenv("BROKER_WS_URL", raising=False)
    with pytest.raises(ValueError, match="BROKER_WS_URL"):
        endpoint_from_environment()


def test_transport_validates_timeout():
    with pytest.raises(ValueError):
        WebSocketClientTransport("wss://example.invalid", timeout=0)


def test_configured_feed_constructs_without_network_call():
    endpoint = MarketDataEndpoint("wss://example.invalid")
    feed = ConfiguredBrokerMarketDataFeed(endpoint, subscribe_message=json_subscription)
    assert feed.url == endpoint.url
    assert feed.health().subscribed_symbols == []
