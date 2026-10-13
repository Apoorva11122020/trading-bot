"""Realtime option-premium WebSocket tracker for PAPER signal validation (no orders).

Subscribes to every pending option ``instrument_key`` in ``live_signal_outcomes``,
updates display marks from live ticks, and resolves outcomes only from completed
5-minute premium candles (REST backfill + tick-built candles).

Must not import ``src.paper_trading`` engine/broker modules beyond ``signal_tracker``.
"""
from __future__ import annotations

import logging
import threading
import time as time_mod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from src.database import Database
from src.market_data.candle_builder import CandleBuilder
from src.market_data.interface import MarketDataFeed
from src.market_data.models import MarketDataStatus, Tick
from src.paper_trading.signal_tracker import LiveSignalTracker
from src.safety.live_gate import assert_paper_trading_only, live_trading_enabled, trading_mode
from src.signal_accuracy.realtime import IST, TrackerSettings, create_feed, write_status
from src.signal_accuracy.tracker import to_ist

log = logging.getLogger(__name__)

PREMIUM_STATUS_FILE = "premium_status.json"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "reports" / "live_signal_accuracy"
OPTION_TICK_DISCONNECT_SECONDS = 60
OPTION_SUBSCRIBE_GRACE_SECONDS = 45
OPTION_RECONNECT_COOLDOWN_SECONDS = 5


@dataclass
class PremiumFeedStats:
    option_ticks_received: int = 0
    option_ticks_parsed: int = 0
    subscribed_instruments: list[str] = field(default_factory=list)
    last_option_tick_time: str | None = None
    last_option_tick_price: float | None = None
    last_option_tick_symbol: str | None = None
    subscription_status: str = "PENDING"
    last_subscription_sent_at: str | None = None
    subscription_messages_sent: int = 0
    reconnect_events: int = 0
    orders_placed: int = 0


class LivePremiumRealtimeRunner:
    """WebSocket option ticks + optional REST poll loop for signal registration."""

    def __init__(self, feed: MarketDataFeed, database: Database, tracker: LiveSignalTracker, *,
                 settings: TrackerSettings | None = None, out_dir: str | Path = DEFAULT_OUT,
                 rest_poll: Callable[[], None] | None = None, poll_seconds: int = 30,
                 clock: Callable[[], datetime] | None = None):
        assert_paper_trading_only()
        self.feed = feed
        self.database = database
        self.tracker = tracker
        self.settings = settings or TrackerSettings.load()
        self.out_dir = Path(out_dir)
        self.rest_poll = rest_poll
        self.poll_seconds = max(0, int(poll_seconds))
        self.clock = clock or (lambda: datetime.now(IST))
        self.stats = PremiumFeedStats()
        self._builders: dict[str, CandleBuilder] = {}
        self._lock = threading.RLock()
        self._running = False
        self._stopping = False
        self._pending_keys: set[str] = set()
        self._last_reconnect_mono = 0.0
        feed.set_tick_handler(self.on_tick)

    def _builder(self, symbol: str) -> CandleBuilder:
        if symbol not in self._builders:
            self._builders[symbol] = CandleBuilder(self.settings.interval_minutes)
        return self._builders[symbol]

    def pending_option_keys(self) -> list[str]:
        keys = sorted(k for k in self.tracker.pending_keys() if k)
        self._pending_keys = set(keys)
        return keys

    def _feed_connected(self) -> bool:
        try:
            return self.feed.health().status == MarketDataStatus.CONNECTED
        except Exception:
            return False

    def _send_subscription(self, symbols: list[str]) -> None:
        if not symbols:
            return
        if hasattr(self.feed, "subscribe_merge"):
            self.feed.subscribe_merge(symbols)
        else:
            self.feed.subscribe(symbols)
        self.stats.last_subscription_sent_at = to_ist(self.clock()).isoformat()
        self.stats.subscription_messages_sent += 1

    def _sync_subscriptions(self) -> None:
        keys = self.pending_option_keys()
        symbols = list(dict.fromkeys(keys))
        self.stats.subscribed_instruments = symbols
        if not symbols:
            self.stats.subscription_status = "IDLE"
            return
        try:
            if self._feed_connected():
                self._send_subscription(symbols)
                self.stats.subscription_status = "CONNECTED"
            else:
                self._send_subscription(symbols)
                self.stats.subscription_status = "PENDING"
        except Exception as exc:
            log.warning("option subscription failed: %s", exc)
            self.stats.subscription_status = "FAILED"

    def option_feed_ok(self) -> bool:
        """True when pending options are subscribed and ticks are fresh enough to register new signals."""
        with self._lock:
            if not self._running:
                return True
            if not self._pending_keys and not self.pending_option_keys():
                return True
            if self.stats.subscription_status not in {"CONNECTED", "IDLE"}:
                return False
            now = to_ist(self.clock())
            if self.stats.last_option_tick_time:
                age = (now - to_ist(self.stats.last_option_tick_time)).total_seconds()
                return age <= OPTION_TICK_DISCONNECT_SECONDS
            if self.stats.last_subscription_sent_at:
                grace = (now - to_ist(self.stats.last_subscription_sent_at)).total_seconds()
                return grace <= OPTION_SUBSCRIBE_GRACE_SECONDS
            return False

    def on_tick(self, tick: Tick) -> None:
        with self._lock:
            if self._stopping:
                return
            symbol = tick.symbol.strip()
            if symbol not in self._pending_keys:
                return
            ts = to_ist(tick.timestamp)
            self.stats.option_ticks_received += 1
            self.stats.option_ticks_parsed += 1
            self.stats.last_option_tick_time = ts.isoformat()
            self.stats.last_option_tick_price = float(tick.ltp)
            self.stats.last_option_tick_symbol = symbol
            if self.stats.subscription_status != "CONNECTED":
                self.stats.subscription_status = "CONNECTED"
            self.tracker.on_tick(symbol, float(tick.ltp), ts)
            completed = self._builder(symbol).update(tick.model_copy(update={"timestamp": ts}))
            if completed is not None:
                self.tracker.on_candle(symbol, completed)

    def _maybe_reconnect(self) -> None:
        if not self._pending_keys or self._stopping:
            return
        now_mono = time_mod.monotonic()
        if now_mono - self._last_reconnect_mono < OPTION_RECONNECT_COOLDOWN_SECONDS:
            return
        now = to_ist(self.clock())
        stale = False
        if self.stats.last_option_tick_time:
            age = (now - to_ist(self.stats.last_option_tick_time)).total_seconds()
            stale = age > OPTION_TICK_DISCONNECT_SECONDS
        elif self.stats.last_subscription_sent_at:
            age = (now - to_ist(self.stats.last_subscription_sent_at)).total_seconds()
            stale = age > OPTION_SUBSCRIBE_GRACE_SECONDS
        if not stale and self._feed_connected() and self.stats.subscription_status == "CONNECTED":
            return
        self._last_reconnect_mono = now_mono
        self.stats.subscription_status = "RECONNECTING"
        self.stats.reconnect_events += 1
        log.info("option premium feed stale; reconnecting (pending=%s)", len(self._pending_keys))
        try:
            self.feed.disconnect()
        except Exception:
            pass
        try:
            self.feed.connect()
            self._sync_subscriptions()
        except Exception as exc:
            log.warning("option premium reconnect failed: %s", exc)
            self.stats.subscription_status = "FAILED"

    def status_payload(self, *, running: bool, session_date: str) -> dict[str, Any]:
        with self._lock:
            health = self.feed.health().status.value
            return {
                "type": "PREMIUM_SIGNAL_TRACKER_STATUS",
                "updated_at": to_ist(self.clock()).isoformat(),
                "running": running,
                "session_date": session_date,
                "mode": "REALTIME_WEBSOCKET",
                "subscription_status": self.stats.subscription_status,
                "subscribed_instruments": list(self.stats.subscribed_instruments),
                "option_ticks_received": self.stats.option_ticks_received,
                "option_ticks_parsed": self.stats.option_ticks_parsed,
                "last_option_tick_time": self.stats.last_option_tick_time,
                "last_option_tick_price": self.stats.last_option_tick_price,
                "last_option_tick_symbol": self.stats.last_option_tick_symbol,
                "last_subscription_sent_at": self.stats.last_subscription_sent_at,
                "subscription_messages_sent": self.stats.subscription_messages_sent,
                "reconnect_events": self.stats.reconnect_events,
                "market_data_status": health,
                "orders_placed": self.stats.orders_placed,
                "trading_mode": trading_mode(),
                "live_trading_enabled": live_trading_enabled(),
            }

    def heartbeat(self, *, running: bool, session_date: str) -> None:
        write_status(self.out_dir / PREMIUM_STATUS_FILE,
                     self.status_payload(running=running, session_date=session_date))

    def start(self, session_date: str) -> None:
        with self._lock:
            self._running = True
            self._stopping = False
        self.feed.connect()
        self._sync_subscriptions()
        self.heartbeat(running=True, session_date=session_date)

    def stop(self, session_date: str) -> None:
        with self._lock:
            self._stopping = True
            self._running = False
        try:
            self.feed.disconnect()
        except Exception:
            pass
        self.heartbeat(running=False, session_date=session_date)

    def poll_rest(self) -> None:
        if self.rest_poll is not None:
            self.rest_poll()
        self._sync_subscriptions()

    def run(self, session_date: str, *, until: datetime | None = None) -> None:
        """Block until ``until`` or interrupted; REST poll + heartbeat run in the loop."""
        self.start(session_date)
        if self.rest_poll is not None:
            self.poll_rest()
        last_rest = last_hb = last_watch = time_mod.monotonic()
        try:
            while until is None or to_ist(self.clock()) < until:
                time_mod.sleep(0.5)
                now_mono = time_mod.monotonic()
                if self.poll_seconds and now_mono - last_rest >= self.poll_seconds:
                    last_rest = now_mono
                    self.poll_rest()
                if now_mono - last_watch >= 5.0:
                    last_watch = now_mono
                    self._maybe_reconnect()
                if now_mono - last_hb >= 1.0:
                    last_hb = now_mono
                    self.heartbeat(running=True, session_date=session_date)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop(session_date)


def start_premium_session(*, day: date, database: Database, tracker: LiveSignalTracker,
                          rest_poll: Callable[[], None], out_dir: str | Path = DEFAULT_OUT,
                          poll_seconds: int = 30, until: datetime | None = None,
                          settings: TrackerSettings | None = None) -> LivePremiumRealtimeRunner:
    """Start the realtime option-premium WebSocket loop (blocking)."""
    settings = settings or TrackerSettings.load()
    feed = create_feed(stale_after_seconds=settings.stale_data_seconds)
    runner = LivePremiumRealtimeRunner(feed, database, tracker, settings=settings, out_dir=out_dir,
                                       rest_poll=rest_poll, poll_seconds=poll_seconds)
    runner.run(day.isoformat(), until=until)
    return runner
