from datetime import datetime, timezone

from src.market_data.models import MarketDataStatus
from src.market_data.websocket import WebSocketMarketDataFeed, json_tick_parser


class FakeTransport:
    def __init__(self, messages=None, fail_after=False):
        self.messages = list(messages or [])
        self.closed = False
        self.sent = []
        self.fail_after = fail_after

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        if self.messages:
            return self.messages.pop(0)
        if self.fail_after:
            raise RuntimeError("socket disconnected")
        raise RuntimeError("socket idle")

    def close(self):
        self.closed = True


def test_json_parser_supports_single_tick():
    ticks = json_tick_parser('{"symbol":"NIFTY","ltp":101.5,"timestamp":"2026-10-03T09:30:00+00:00"}')
    assert len(ticks) == 1
    assert ticks[0].symbol == "NIFTY"
    assert ticks[0].ltp == 101.5
    assert ticks[0].timestamp.tzinfo is not None


def test_json_parser_supports_batch():
    ticks = json_tick_parser('{"ticks":[{"symbol":"A","ltp":10},{"symbol":"B","ltp":20}]}')
    assert [t.symbol for t in ticks] == ["A", "B"]


def test_websocket_feed_connect_subscribe_and_publish():
    transport = FakeTransport([
        '{"symbol":"NIFTY","ltp":100,"timestamp":"2026-10-03T09:30:00+00:00"}'
    ])
    feed = WebSocketMarketDataFeed(
        "wss://example.invalid",
        transport_factory=lambda _: transport,
        subscribe_message=lambda symbols: str(symbols),
    )
    received = []
    feed.subscribe(["NIFTY"])
    feed.set_tick_handler(received.append)
    feed.connect()
    feed._thread.join(timeout=1)
    assert received[0].symbol == "NIFTY"
    assert feed.health().status == MarketDataStatus.DISCONNECTED
    assert transport.sent[0] == "['NIFTY']"


def test_websocket_feed_reconnects_after_failure():
    first = FakeTransport(fail_after=True)
    second = FakeTransport([
        '{"symbol":"RELIANCE","ltp":2500,"timestamp":"2026-10-03T09:31:00+00:00"}'
    ])
    transports = iter([first, second])
    received = []
    feed = WebSocketMarketDataFeed(
        "wss://example.invalid",
        transport_factory=lambda _: next(transports),
        subscribe_message=lambda symbols: "subscribe:" + ",".join(symbols),
        initial_delay_seconds=0,
        max_delay_seconds=0,
        max_attempts=2,
    )
    feed.subscribe(["RELIANCE"])
    feed.set_tick_handler(received.append)
    feed.connect()
    feed._thread.join(timeout=1)
    assert received and received[0].ltp == 2500
    assert feed.health().reconnect_attempts >= 1


def test_websocket_feed_subscribe_merge_keeps_existing_symbols():
    transport = FakeTransport([])
    feed = WebSocketMarketDataFeed(
        "wss://example.invalid",
        transport_factory=lambda _: transport,
        subscribe_message=lambda symbols: ",".join(symbols),
    )
    feed.subscribe(["NIFTY"])
    feed.subscribe_merge(["NSE_FO|44620"])
    feed.connect()
    feed._thread.join(timeout=1)
    assert feed.health().subscribed_symbols == ["NIFTY", "NSE_FO|44620"]


def test_websocket_feed_ignores_unsubscribed_symbol():
    transport = FakeTransport([
        '{"symbol":"BANKNIFTY","ltp":50000,"timestamp":"2026-10-03T09:30:00+00:00"}'
    ])
    feed = WebSocketMarketDataFeed(
        "wss://example.invalid",
        transport_factory=lambda _: transport,
        subscribe_message=lambda symbols: "ok",
    )
    received = []
    feed.subscribe(["NIFTY"])
    feed.set_tick_handler(received.append)
    feed.connect()
    feed._thread.join(timeout=1)
    assert received == []
