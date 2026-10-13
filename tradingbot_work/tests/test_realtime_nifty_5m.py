"""Real-time NIFTY 5m candle + S2 validation (no broker orders)."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.market_data.candle_builder import CandleBuilder
from src.market_data.models import Tick
from src.market_data.paper import PaperMarketDataFeed
from src.paper_trading.config import load_paper_trading_settings
from src.market_data.websocket import WebSocketMarketDataFeed, json_tick_parser
from src.validation.realtime_nifty_5m import RealtimeNifty5mValidator, tick_to_ist

IST = ZoneInfo("Asia/Kolkata")
UNDERLYING = "NSE_INDEX|Nifty 50"


def _ist(h, m, day=1):
    return datetime(2026, 10, day, h, m, tzinfo=IST)


def _tick(h, m, price, day=1, volume=0.0):
    return Tick(symbol=UNDERLYING, timestamp=_ist(h, m, day), ltp=price, volume=volume)


def test_tick_aggregation_into_five_minute_candles():
    builder = CandleBuilder(5)
    assert builder.update(tick_to_ist(_tick(9, 15, 100.0))) is None
    assert builder.update(tick_to_ist(_tick(9, 17, 102.0, volume=500.0))) is None
    completed = builder.update(tick_to_ist(_tick(9, 20, 101.0)))
    assert completed is not None
    assert completed.open == 100.0
    assert completed.high == 102.0
    assert completed.low == 100.0
    assert completed.close == 102.0
    assert completed.volume == 500.0


def test_candle_boundary_uses_ist_bucket():
    builder = CandleBuilder(5)
    # UTC 03:50 == IST 09:20 — closes 09:15–09:19 IST bucket when next IST bucket tick arrives
    utc_tick = Tick(
        symbol=UNDERLYING,
        timestamp=datetime(2026, 10, 1, 3, 45, tzinfo=timezone.utc),
        ltp=100.0,
    )
    builder.update(tick_to_ist(utc_tick))
    close_utc = Tick(
        symbol=UNDERLYING,
        timestamp=datetime(2026, 10, 1, 3, 50, tzinfo=timezone.utc),
        ltp=105.0,
    )
    completed = builder.update(tick_to_ist(close_utc))
    assert completed is not None
    assert completed.timestamp == _ist(9, 15)
    assert completed.close == 100.0


def test_timezone_normalization_to_ist():
    utc = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc)
    tick = Tick(symbol=UNDERLYING, timestamp=utc, ltp=1.0)
    ist = tick_to_ist(tick)
    assert ist.timestamp.tzinfo == IST
    assert ist.timestamp.hour == 9
    assert ist.timestamp.minute == 30


def test_out_of_order_and_duplicate_ticks():
    builder = CandleBuilder(5)
    builder.update(tick_to_ist(_tick(9, 15, 100.0)))
    builder.update(tick_to_ist(_tick(9, 16, 101.0)))
    assert builder.update(tick_to_ist(_tick(9, 14, 50.0))) is None
    builder.update(tick_to_ist(_tick(9, 17, 102.0)))
    completed = builder.update(tick_to_ist(_tick(9, 20, 103.0)))
    assert completed is not None
    assert completed.high == 102.0
    assert completed.close == 102.0


def test_stale_market_data_skips_candle_signal_evaluation(tmp_path, monkeypatch):
    feed = PaperMarketDataFeed()
    settings = load_paper_trading_settings()
    validator = RealtimeNifty5mValidator(
        feed=feed,
        settings=settings,
        reports_dir=tmp_path,
    )
    feed.subscribe([UNDERLYING])
    feed.connect()
    feed.publish(_tick(9, 15, 22900.0))
    feed.publish(_tick(9, 20, 22910.0))
    assert validator.stats.completed_candles == 1

    real_can_trade = feed.can_trade

    def blocked_after_first(now=None):
        if validator.stats.completed_candles >= 1:
            return False
        return real_can_trade(now)

    monkeypatch.setattr(feed, "can_trade", blocked_after_first)
    feed.publish(_tick(9, 25, 22930.0))
    feed.publish(_tick(9, 30, 22940.0))
    assert validator.stats.stale_events >= 1
    assert validator.stats.completed_candles == 1
    assert validator.stats.orders_placed == 0


def test_disconnect_health_is_tracked(tmp_path):
    feed = PaperMarketDataFeed()
    validator = RealtimeNifty5mValidator(feed=feed, reports_dir=tmp_path)
    feed.subscribe([UNDERLYING])
    feed.connect()
    feed.publish(_tick(9, 15, 100.0))
    feed.disconnect()
    validator._track_health()
    assert validator.stats.disconnect_events >= 1


def _publish_five_min_bar(feed, h: int, m: int, high: float, low: float, close: float) -> None:
    feed.publish(_tick(h, m, low))
    feed.publish(_tick(h, m + 2, high))
    feed.publish(_tick(h, m + 4, close))


def test_s2_signal_on_completed_candles_only(tmp_path):
    feed = PaperMarketDataFeed()
    settings = load_paper_trading_settings()
    validator = RealtimeNifty5mValidator(feed=feed, settings=settings, reports_dir=tmp_path)
    feed.subscribe([UNDERLYING])
    feed.connect()

    bars = [
        (9, 15, 22950.0, 22900.0, 22920.0),
        (9, 20, 22950.0, 22900.0, 22925.0),
        (9, 25, 22950.0, 22900.0, 22930.0),
        (9, 30, 22950.0, 22900.0, 22935.0),
        (9, 35, 23010.0, 22980.0, 23000.0),
    ]
    for h, m, hi, lo, close in bars:
        _publish_five_min_bar(feed, h, m, hi, lo, close)
        feed.publish(_tick(h, m + 5, close + 1.0))
    assert validator.stats.completed_candles == 5
    assert validator.stats.buy_signals >= 1
    assert validator.stats.orders_placed == 0
    candle_logs = [row for row in validator._logs if row.get("type") == "CANDLE_CLOSED"]
    assert candle_logs
    assert all("final_signal" in row for row in candle_logs)
    assert all(row["orders_placed"] == 0 for row in candle_logs)


def test_diagnostics_show_ticks_reaching_validator(tmp_path):
    feed = PaperMarketDataFeed()
    validator = RealtimeNifty5mValidator(feed=feed, reports_dir=tmp_path)
    feed.subscribe([UNDERLYING])
    feed.connect()
    feed.publish(_tick(9, 15, 100.0))
    d = validator.diagnostics()
    assert d.ticks_received == 1
    assert d.last_tick_ltp == 100.0
    assert d.open_candle_start is not None


def test_wrong_symbol_increments_ignored_counter(tmp_path):
    feed = PaperMarketDataFeed()
    validator = RealtimeNifty5mValidator(feed=feed, reports_dir=tmp_path)
    validator.on_tick(Tick(symbol="OTHER", timestamp=_ist(9, 15), ltp=100.0))
    assert validator.diagnostics().ticks_ignored_symbol == 1
    assert validator.diagnostics().ticks_received == 0


def test_websocket_feed_delivered_count_reaches_validator(tmp_path):
    import queue
    import json
    import time as time_mod

    class T:
        def __init__(self):
            self.q = queue.Queue()
            self.sent = []

        def send(self, message):
            self.sent.append(message)

        def recv(self):
            return self.q.get(timeout=0.05)

        def close(self):
            pass

    transport = T()
    transport.q.put(json.dumps({"symbol": UNDERLYING, "ltp": 200.0, "timestamp": "2026-10-01T04:00:00+00:00"}))

    feed = WebSocketMarketDataFeed(
        "wss://example.invalid",
        transport_factory=lambda _u: transport,
        subscribe_message=lambda s: json.dumps({"sub": s}),
        parser=json_tick_parser,
    )
    validator = RealtimeNifty5mValidator(feed=feed, reports_dir=tmp_path)
    feed.subscribe([UNDERLYING])
    feed.connect()
    deadline = time_mod.time() + 2
    while validator.diagnostics().ticks_received < 1 and time_mod.time() < deadline:
        time_mod.sleep(0.02)
    feed.disconnect()
    d = validator.diagnostics()
    assert d.feed_ticks_delivered >= 1
    assert d.ticks_received == 1


def test_validator_never_places_orders(tmp_path, monkeypatch):
    def _fail_place_order(*_args, **_kwargs):
        raise AssertionError("place_order must not be called during validation")

    monkeypatch.setattr("src.broker.paper.PaperBroker.place_order", _fail_place_order)
    feed = PaperMarketDataFeed()
    validator = RealtimeNifty5mValidator(feed=feed, reports_dir=tmp_path)
    feed.subscribe([UNDERLYING])
    feed.connect()
    for m in (15, 20, 25, 30):
        feed.publish(_tick(9, m, 100.0 + m))
    assert validator.stats.orders_placed == 0
