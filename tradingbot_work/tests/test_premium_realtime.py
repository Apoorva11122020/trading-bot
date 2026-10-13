"""Realtime option-premium WebSocket tracker for PAPER signal validation."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.database import Database
from src.market_data.models import MarketDataStatus, Tick
from src.paper_trading.signal_tracker import MARK_PREFIX, LiveSignalTracker, SignalRecord
from src.signal_accuracy.premium_realtime import LivePremiumRealtimeRunner
from src.signal_accuracy.realtime import TrackerSettings

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 7, 11, 0, tzinfo=IST)
KEY = "NSE_FO|44620"


class FakeFeed:
    def __init__(self):
        self.symbols: list[str] = []
        self.connected = False
        self._handler = None

    def set_tick_handler(self, handler):
        self._handler = handler

    def subscribe(self, symbols):
        self.symbols = list(symbols)

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def health(self):
        return SimpleNamespace(status=MarketDataStatus.CONNECTED)


@pytest.fixture()
def db():
    return Database(":memory:")


@pytest.fixture()
def tracker(db, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    return LiveSignalTracker(db, square_off="15:15")


def _pending(tracker):
    tracker.register(SignalRecord.create(
        timestamp=NOW, underlying="NIFTY", direction="BUY", active_from=NOW, strategy="S2",
        instrument_key=KEY, trading_symbol="NIFTY 22700 CE 13 OCT 26", option_type="CE",
        strike=22700.0, expiry="2026-10-13", signal_price=22711.0, entry_price=130.0,
        stop_loss=110.0, target=170.0, quantity=65, source="TEST",
    ))


def test_subscribes_pending_option_keys(db, tracker, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    _pending(tracker)
    feed = FakeFeed()
    runner = LivePremiumRealtimeRunner(feed, db, tracker, settings=TrackerSettings.load(), out_dir=tmp_path)
    runner._sync_subscriptions()
    assert feed.symbols == [KEY]
    assert runner.stats.subscription_status == "CONNECTED"


def test_option_tick_updates_mark_without_resolving(db, tracker, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    _pending(tracker)
    sid = next(iter(tracker._pending))
    feed = FakeFeed()
    runner = LivePremiumRealtimeRunner(feed, db, tracker, out_dir=tmp_path)
    runner._sync_subscriptions()
    runner.on_tick(Tick(symbol=KEY, ltp=132.5, volume=0, timestamp=NOW))
    mark = db.get_state(f"{MARK_PREFIX}{sid}")
    assert mark["last_tick_price"] == 132.5
    assert mark["price"] == 132.5
    assert db.get_live_signals("2026-10-07")[0]["status"] == "PENDING"


def test_completed_candle_can_resolve_but_stale_tick_does_not(db, tracker, tmp_path, monkeypatch):
    from src.paper_trading import signal_tracker as st

    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    _pending(tracker)
    feed = FakeFeed()
    runner = LivePremiumRealtimeRunner(feed, db, tracker, out_dir=tmp_path)
    runner._sync_subscriptions()
    monkeypatch.setattr(st, "evaluate_candle", lambda *a, **k: {
        "status": "WIN", "exit_price": 200.0, "exit_time": NOW.isoformat(), "pnl": 100.0,
        "r_multiple": 1.0, "resolved_at": NOW.isoformat(), "reason": "TARGET",
    })
    runner.on_tick(Tick(symbol=KEY, ltp=200.0, volume=0, timestamp=NOW))
    assert db.get_live_signals("2026-10-07")[0]["status"] == "PENDING"
    candle = SimpleNamespace(timestamp=NOW, open=200.0, high=200.0, low=200.0, close=200.0)
    tracker.on_candle(KEY, candle)
    assert db.get_live_signals("2026-10-07")[0]["status"] == "WIN"


def test_start_connects_before_subscription(db, tracker, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    _pending(tracker)
    feed = FakeFeed()
    runner = LivePremiumRealtimeRunner(feed, db, tracker, out_dir=tmp_path)
    runner.start("2026-10-07")
    assert feed.connected is True
    assert feed.symbols == [KEY]
    runner.stop("2026-10-07")


def test_option_feed_ok_requires_fresh_ticks_when_running(db, tracker, tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    _pending(tracker)
    feed = FakeFeed()
    runner = LivePremiumRealtimeRunner(feed, db, tracker, out_dir=tmp_path, clock=lambda: NOW)
    runner._running = True
    runner._pending_keys = {KEY}
    runner.stats.subscription_status = "CONNECTED"
    assert runner.option_feed_ok() is False
    runner.stats.last_option_tick_time = NOW.isoformat()
    assert runner.option_feed_ok() is True


def test_status_payload_reports_subscription_and_ticks(db, tracker, tmp_path):
    feed = FakeFeed()
    runner = LivePremiumRealtimeRunner(feed, db, tracker, out_dir=tmp_path)
    runner.stats.subscription_status = "CONNECTED"
    runner.stats.option_ticks_received = 3
    payload = runner.status_payload(running=True, session_date="2026-10-07")
    assert payload["subscription_status"] == "CONNECTED"
    assert payload["orders_placed"] == 0
    assert payload["option_ticks_received"] == 3
