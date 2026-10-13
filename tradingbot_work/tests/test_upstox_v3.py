from datetime import datetime, timezone
import json
import pytest

from src.market_data.upstox_v3_proto import FeedResponse
from src.market_data.upstox_v3 import (
    UpstoxV3MarketDataError,
    decode_ticks,
    resolve_market_data_access_token,
    subscription_payload,
)


def make_feed():
    response = FeedResponse()
    response.currentTs = 1760000000000
    response.type = 1
    feed = response.feeds["NSE_INDEX|Nifty 50"]
    feed.ltpc.ltp = 25100.25
    feed.ltpc.ltt = 1760000000123
    feed.ltpc.ltq = 50
    return response.SerializeToString()


def test_upstox_v3_binary_decode():
    ticks = decode_ticks(make_feed())
    assert len(ticks) == 1
    assert ticks[0].symbol == "NSE_INDEX|Nifty 50"
    assert ticks[0].ltp == 25100.25
    assert ticks[0].last_quantity == 50
    assert ticks[0].timestamp.tzinfo == timezone.utc


def test_subscription_is_binary_and_v3_shape():
    payload = json.loads(subscription_payload(["NSE_INDEX|Nifty 50"], "ltpc").decode())
    assert payload["method"] == "sub"
    assert payload["data"]["mode"] == "ltpc"
    assert payload["data"]["instrumentKeys"] == ["NSE_INDEX|Nifty 50"]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        subscription_payload(["NSE_INDEX|Nifty 50"], "bad")


def test_invalid_protobuf_rejected():
    with pytest.raises(UpstoxV3MarketDataError):
        decode_ticks(b"not-protobuf")


def test_market_data_token_prefers_market_data_env(monkeypatch):
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "sandbox-only")
    monkeypatch.setenv("ACCESS_TOKEN", "fallback-token")
    monkeypatch.setenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", "market-data-token")
    token, source = resolve_market_data_access_token()
    assert token == "market-data-token"
    assert source == "UPSTOX_MARKET_DATA_ACCESS_TOKEN"


def test_market_data_token_falls_back_to_access_token(monkeypatch):
    monkeypatch.delenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "sandbox-only")
    monkeypatch.setenv("ACCESS_TOKEN", "fallback-token")
    token, source = resolve_market_data_access_token()
    assert token == "fallback-token"
    assert source == "ACCESS_TOKEN"


def test_market_data_token_never_uses_sandbox(monkeypatch):
    monkeypatch.delenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "sandbox-only")
    with pytest.raises(UpstoxV3MarketDataError, match="cannot be used for market data"):
        resolve_market_data_access_token()


def test_market_data_token_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", "from-env")
    token, source = resolve_market_data_access_token("explicit")
    assert token == "explicit"
    assert source == "access_token_argument"
