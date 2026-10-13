import json
import queue
import time

from src.market_data.websocket import WebSocketMarketDataFeed, json_tick_parser


def test_parser_skips_malformed_and_invalid_rows_but_keeps_valid_ones():
    message = json.dumps({"ticks": [
        {"symbol": "A", "ltp": 100.0, "timestamp": "2026-10-01T04:30:00Z"},
        {"symbol": "B", "ltp": -1, "timestamp": "2026-10-01T04:30:00Z"},
        {"symbol": "C", "ltp": "NaN", "timestamp": "2026-10-01T04:30:00Z"},
        {"symbol": "D", "ltp": 0, "timestamp": "2026-10-01T04:30:00Z"},
        {"symbol": "E", "ltp": 10.0, "timestamp": "not-a-time"},
        {"ltp": 10.0},
        {"symbol": "F"},
        "garbage",
        {"symbol": "G", "ltp": "abc"},
        {"symbol": "H", "ltp": 101.5, "timestamp": 1790829000000},
    ]})
    ticks = json_tick_parser(message)
    assert [t.symbol for t in ticks] == ["A", "H"]


def test_missing_timestamp_uses_receive_time():
    ticks = json_tick_parser(json.dumps({"symbol": "A", "ltp": 10.0}))
    assert ticks[0].timestamp.tzinfo is not None


class ScriptedTransport:
    def __init__(self, messages):
        self.messages = queue.Queue()
        for m in messages:
            self.messages.put(m)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        try:
            return self.messages.get(timeout=0.05)
        except queue.Empty:
            time.sleep(0.01)
            return json.dumps({"ticks": []})

    def close(self):
        self.closed = True


def test_feed_survives_malformed_message_and_drops_duplicate_ticks():
    good = {"symbol": "A", "ltp": 100.0, "timestamp": "2026-10-01T04:30:00Z"}
    transport = ScriptedTransport([
        "{not json",
        json.dumps(good),
        json.dumps(good),
        json.dumps({**good, "ltp": 100.5, "timestamp": "2026-10-01T04:30:01Z"}),
    ])
    factory_calls = []

    def factory(url):
        factory_calls.append(url)
        return transport

    feed = WebSocketMarketDataFeed("wss://example.invalid/feed", transport_factory=factory,
                                   subscribe_message=lambda s: json.dumps({"sub": s}))
    received = []
    feed.set_tick_handler(received.append)
    feed.subscribe(["A"])
    feed.connect()
    deadline = time.time() + 3
    while len(received) < 2 and time.time() < deadline:
        time.sleep(0.01)
    feed.disconnect()

    assert [t.ltp for t in received] == [100.0, 100.5]
    assert feed.malformed_messages == 1
    assert feed.duplicate_ticks == 1
    assert len(factory_calls) == 1  # malformed payload did not force a reconnect


def test_stale_feed_blocks_trading():
    from datetime import datetime, timedelta, timezone
    from src.market_data.health import MarketDataHealthMonitor
    from src.market_data.models import Tick

    monitor = MarketDataHealthMonitor(stale_after_seconds=15)
    monitor.connected(["A"])
    now = datetime(2026, 10, 1, 4, 30, tzinfo=timezone.utc)
    monitor.record_tick(Tick(symbol="A", timestamp=now, ltp=100.0))
    assert monitor.can_trade(now + timedelta(seconds=5)) is True
    assert monitor.can_trade(now + timedelta(seconds=16)) is False
