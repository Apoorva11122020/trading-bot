"""Upstox V3 WebSocket subscription must use binary frames (market-data path)."""
from __future__ import annotations

import json
import queue
import time

import websocket

from src.market_data.transport import WebSocketClientTransport
from src.market_data.upstox_v3 import subscription_payload
from src.market_data.websocket import WebSocketMarketDataFeed, json_tick_parser


def test_upstox_v3_subscription_payload_is_binary():
    payload = subscription_payload(["NSE_INDEX|Nifty 50"], mode="ltpc")
    assert isinstance(payload, bytes)
    body = json.loads(payload.decode())
    assert body["method"] == "sub"
    assert body["data"]["instrumentKeys"] == ["NSE_INDEX|Nifty 50"]


def test_transport_sends_binary_opcode_for_bytes():
    sent: list[tuple[object, int | None]] = []

    class FakeSocket:
        def send(self, data, opcode=None):
            sent.append((data, opcode))

    transport = WebSocketClientTransport("wss://example.invalid/feed")
    transport._socket = FakeSocket()
    payload = subscription_payload(["NSE_INDEX|Nifty 50"])
    transport.send(payload)
    assert sent[0][0] == payload
    assert sent[0][1] == websocket.ABNF.OPCODE_BINARY


class ScriptedTransport:
    def __init__(self, messages):
        self.messages = queue.Queue()
        for message in messages:
            self.messages.put(message)
        self.sent: list[object] = []

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return self.messages.get(timeout=0.1)

    def close(self):
        pass


def test_feed_sends_binary_subscription_on_connect():
    transport = ScriptedTransport([json.dumps({"symbol": "NSE_INDEX|Nifty 50", "ltp": 1.0, "timestamp": "2026-10-01T04:30:00Z"})])
    feed = WebSocketMarketDataFeed(
        "wss://example.invalid/feed",
        transport_factory=lambda _url: transport,
        subscribe_message=lambda symbols: subscription_payload(symbols),
        parser=json_tick_parser,
    )
    feed.subscribe(["NSE_INDEX|Nifty 50"])
    feed.connect()
    deadline = time.time() + 2
    while feed.subscription_messages_sent < 1 and time.time() < deadline:
        time.sleep(0.02)
    feed.disconnect()
    assert feed.subscription_messages_sent == 1
    assert transport.sent
    assert isinstance(transport.sent[0], bytes)
