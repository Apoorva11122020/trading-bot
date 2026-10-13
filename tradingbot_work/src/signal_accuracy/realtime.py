"""Realtime NIFTY 5m S2 signal-accuracy runner on the existing market-data feed (PAPER, no orders).

This module must not import ``src.paper_trading``: that package imports the
broker, TradingEngine and OrderManager as a side effect of import.
"""
from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import yaml

from src.backtest.research_strategies import enrich_s2, generate_signals
from src.config.research_loader import load_research_config
from src.market_data.candle_builder import CandleBuilder
from src.market_data.interface import MarketDataFeed
from src.market_data.models import MarketDataStatus, Tick
from src.safety.live_gate import assert_paper_trading_only, live_trading_enabled, trading_mode
from src.session.session_manager import SessionManager
from src.signal_accuracy.tracker import (
    IST,
    TRACKING,
    LevelConvention,
    OutcomeTracker,
    SessionStore,
    SignalEvent,
    iso,
    new_signal,
    summarize,
    to_ist,
    write_reports,
)

ROOT = Path(__file__).resolve().parents[2]
PAPER_CONFIG = ROOT / "config" / "paper_trading.yaml"
RESEARCH_CONFIG = ROOT / "config" / "strategy_research.yaml"
DEFAULT_OUT = ROOT / "reports" / "realtime_signal_accuracy"
STATUS_FILE = "live_status.json"


def write_status(path: Path, payload: dict[str, Any]) -> bool:
    """Atomically replace the heartbeat file; a failed write is skipped (the next poll retries)."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


@dataclass(frozen=True)
class TrackerSettings:
    strategy_id: str
    underlying: str
    timeframe: str
    interval_minutes: int
    instrument_key: str
    stale_data_seconds: int
    history_candles: int
    market_open: str
    allow_entries_from: str
    stop_new_entries: str
    square_off: str
    market_close: str

    @classmethod
    def load(cls, path: str | Path = PAPER_CONFIG) -> "TrackerSettings":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["paper_trading"]
        session = raw["session"]
        return cls(
            strategy_id=str(raw.get("strategy_id", "S2")).upper(),
            underlying=str(raw.get("underlying", "NIFTY")).upper(),
            timeframe=str(raw.get("timeframe", "5m")),
            interval_minutes=int(raw["interval_minutes"]),
            instrument_key=str(raw["instrument_key"]),
            stale_data_seconds=int(raw["stale_data_seconds"]),
            history_candles=int(raw["history_candles"]),
            market_open=str(session.get("market_open", "09:15")),
            allow_entries_from=str(session.get("allow_entries_from", "09:15")),
            stop_new_entries=str(session.get("stop_new_entries", "14:30")),
            square_off=str(session.get("square_off", "15:15")),
            market_close=str(session.get("market_close", "15:30")),
        )


@dataclass
class DataQuality:
    websocket_connected: bool = False
    feed_messages: int = 0
    parsed_ticks: int = 0
    delivered_ticks: int = 0
    ticks_received: int = 0
    ignored_symbols: int = 0
    outside_session_ticks: int = 0
    out_of_order_ticks: int = 0
    stale_events: int = 0
    disconnect_events: int = 0
    data_gaps: int = 0
    completed_candles: int = 0
    incomplete_candles: int = 0
    evaluated_candles: int = 0
    opening_range_incomplete: int = 0
    signals_generated: int = 0
    duplicate_signals_skipped: int = 0
    signals_outside_entry_window: int = 0
    restored_signals: int = 0
    restored_candles: int = 0
    orders_placed: int = 0


def create_feed(*, stale_after_seconds: int):
    """Read-only Upstox V3 WebSocket feed (same wiring as the existing realtime validator)."""
    from src.market_data.transport import WebSocketClientTransport
    from src.market_data.upstox_v3 import (
        authorize_v3,
        decode_ticks,
        resolve_market_data_access_token,
        subscription_payload,
    )
    from src.market_data.websocket import WebSocketMarketDataFeed

    token, _ = resolve_market_data_access_token(None)
    url = authorize_v3(token)

    def transport_factory(ws_url: str):
        transport = WebSocketClientTransport(ws_url, timeout=10.0)
        transport.connect()
        return transport

    return WebSocketMarketDataFeed(
        url,
        transport_factory=transport_factory,
        subscribe_message=lambda keys: subscription_payload(keys, mode="ltpc"),
        parser=decode_ticks,
        stale_after_seconds=stale_after_seconds,
        reconnect_enabled=True,
    )


def _f(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _hms(value: str | None) -> str:
    return to_ist(value).strftime("%H:%M:%S") if value else "-"


class LiveSignalAccuracyRunner:
    """Completed real 5m candles -> existing S2 generator -> signal events -> tick-level outcomes."""

    def __init__(self, feed: MarketDataFeed, *, settings: TrackerSettings | None = None, research=None,
                 out_dir: str | Path = DEFAULT_OUT, session_id: str | None = None,
                 clock: Callable[[], datetime] | None = None, echo: Callable[[str], None] = print):
        assert_paper_trading_only()
        self.feed = feed
        self.settings = settings or TrackerSettings.load()
        self.research = research or load_research_config(RESEARCH_CONFIG)
        self.clock = clock or (lambda: datetime.now(IST))
        self.echo = echo
        s = self.settings
        if s.strategy_id not in self.research.strategies:
            raise ValueError(f"Strategy {s.strategy_id} is not defined in the research config")
        self.s2_cfg = dict(self.research.strategies[s.strategy_id])
        self.range_start = time.fromisoformat(self.s2_cfg["range_start"])
        self.range_end = time.fromisoformat(self.s2_cfg["range_end"])
        self.interval = timedelta(minutes=s.interval_minutes)
        self.convention = LevelConvention.from_research(self.research, square_off=s.square_off,
                                                        interval_minutes=s.interval_minutes)
        self.session_manager = SessionManager(SimpleNamespace(
            market_open=s.market_open, allow_entries_from=s.allow_entries_from,
            stop_new_entries=s.stop_new_entries, market_close=s.market_close, force_exit=s.square_off,
        ))
        self.session_id = session_id or to_ist(self.clock()).strftime("%Y%m%d")
        self.store = SessionStore(Path(out_dir), self.session_id)
        self.dq = DataQuality()
        self.tracker = OutcomeTracker(self.convention, on_event=self._on_signal_event)
        self.builder = CandleBuilder(s.interval_minutes)
        self.history: deque[dict[str, Any]] = deque(maxlen=s.history_candles)
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self._lock = threading.RLock()
        self._listening_since: datetime | None = None
        self._last_tick_ts: datetime | None = None
        self._last_tick_wall: datetime | None = None
        self._last_ltp: float | None = None
        self._tainted: set[datetime] = set()
        self._in_gap = False
        self._last_health: MarketDataStatus | None = None
        self._connected_count: int | None = None
        self._disconnect_counted = False
        self._stopping = False
        self._restore()
        feed.set_tick_handler(self.on_tick)

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        assert_paper_trading_only()
        with self._lock:
            self.started_at = to_ist(self.clock())
            self._listening_since = self.started_at
        self.store.append("SESSION_START", {
            "started_at": self.started_at.isoformat(), "instrument_key": self.settings.instrument_key,
            "strategy": self.settings.strategy_id, "timeframe": self.settings.timeframe,
            "convention": asdict(self.convention), "safety": self.safety(),
        })
        self.feed.subscribe([self.settings.instrument_key])
        self.feed.connect()
        self.heartbeat()

    def finish(self, reason: str = "RUN_ENDED_BEFORE_OUTCOME") -> tuple[dict[str, Any], dict[str, Path]]:
        with self._lock:
            self._stopping = True
        try:
            self.feed.disconnect()
        except Exception as exc:  # noqa: BLE001 - shutdown must still write the report
            self.echo(f"[DATA] feed disconnect error: {type(exc).__name__}")
        with self._lock:
            self.ended_at = to_ist(self.clock())
            self.tracker.finalize(self.ended_at, reason)
            summary = self.summary()
            dq = self.diagnostics()
            self.store.append("SESSION_END", {"ended_at": self.ended_at.isoformat(), "summary": summary,
                                              "data_quality": dq})
            paths = write_reports(
                self.store, session=summary, signals=list(self.tracker.signals.values()), data_quality=dq,
                safety=self.safety(),
                meta={"started_at": iso(self.started_at) if self.started_at else None,
                      "ended_at": self.ended_at.isoformat(), "instrument_key": self.settings.instrument_key,
                      "convention": {k: str(v) for k, v in asdict(self.convention).items()}},
            )
        self.heartbeat()
        return summary, paths

    # -- feed callbacks --------------------------------------------------------------

    def on_tick(self, tick: Tick) -> None:
        with self._lock:
            if self._stopping:
                return
            if tick.symbol.strip() != self.settings.instrument_key.strip():
                self.dq.ignored_symbols += 1
                return
            ts = to_ist(tick.timestamp)
            self.dq.ticks_received += 1
            self._track_health(ts)
            if not self.session_manager.status(ts).market_open:
                self.dq.outside_session_ticks += 1
                return
            prev = self._last_tick_ts
            if prev is not None and ts < prev:
                self.dq.out_of_order_ticks += 1
                return
            if prev is not None and ts.date() == prev.date() and \
                    (ts - prev).total_seconds() > self.settings.stale_data_seconds:
                self._gap(prev, ts, "TICK_GAP", count=not self._in_gap)
            elif self._in_gap:
                self._taint(ts, ts)
            self._in_gap = False
            self._last_tick_ts = ts
            self._last_tick_wall = to_ist(self.clock())
            self._last_ltp = float(tick.ltp)
            completed = self.builder.update(tick.model_copy(update={"timestamp": ts}))
            if completed is not None:
                self._on_candle(completed)
            self.tracker.on_tick(ts, float(tick.ltp))

    def poll(self, now: datetime | None = None) -> None:
        """Periodic health/staleness check from the main loop; refreshes the dashboard heartbeat."""
        try:
            self._poll(now)
        finally:
            self.heartbeat()

    def _poll(self, now: datetime | None) -> None:
        with self._lock:
            if self._stopping:
                return
            now = to_ist(now or self.clock())
            self._track_health(now)
            if self._in_gap or self._last_tick_wall is None:
                return
            if not self.session_manager.status(now).market_open:
                return
            if (now - self._last_tick_wall).total_seconds() > self.settings.stale_data_seconds:
                self.dq.stale_events += 1
                self._gap(self._last_tick_ts, now, "STALE_FEED")
                self._in_gap = True

    # -- candles and S2 ------------------------------------------------------------------

    def _on_candle(self, candle) -> None:
        ts = to_ist(candle.timestamp)
        row = {"timestamp": ts, "open": float(candle.open), "high": float(candle.high),
               "low": float(candle.low), "close": float(candle.close), "volume": float(candle.volume)}
        record = {**row, "timestamp": ts.isoformat()}
        if self._listening_since is None or ts < self._listening_since or ts in self._tainted:
            reason = "DATA_GAP" if ts in self._tainted else "STARTED_MID_CANDLE"
            self.dq.incomplete_candles += 1
            self.store.append("CANDLE", {"candle": record, "complete": False, "decision": f"SKIPPED_{reason}"})
            self.echo(f"[CANDLE] {ts:%H:%M} incomplete ({reason}) | S2 not evaluated")
            return
        if any(h["timestamp"] == ts for h in self.history):
            return
        self.dq.completed_candles += 1
        self.history.append(row)
        decision = self._evaluate(row)
        self.store.append("CANDLE", {"candle": record, "complete": True, "decision": decision})
        self.echo(f"[CANDLE] {ts:%H:%M} O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} "
                  f"C={row['close']:.2f} | S2={decision}")

    def _evaluate(self, row: dict[str, Any]) -> str:
        ts = row["timestamp"]
        self.dq.evaluated_candles += 1
        if ts.time() > self.range_end and not self._opening_range_complete(ts):
            self.dq.opening_range_incomplete += 1
            return "SKIPPED_OPENING_RANGE_INCOMPLETE"
        frame = pd.DataFrame(sorted(self.history, key=lambda r: r["timestamp"]))
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST)
        signals = generate_signals(self.settings.strategy_id, frame, self.settings.underlying, self.research)
        latest = [sig for sig in signals if to_ist(sig.timestamp) == ts]
        if not latest:
            return "NO_SIGNAL"
        direction = latest[-1].direction
        if not self.session_manager.is_entry_allowed(ts):
            self.dq.signals_outside_entry_window += 1
            return f"{direction}_OUTSIDE_ENTRY_WINDOW"
        atr = enrich_s2(frame, self.s2_cfg)["atr"].iloc[-1]
        event = new_signal(
            session_id=self.session_id, candle=row, direction=direction,
            atr=None if pd.isna(atr) else float(atr), convention=self.convention,
            instrument=self.settings.underlying, timeframe=self.settings.timeframe,
            strategy=self.settings.strategy_id,
        )
        if not self.tracker.register(event):
            self.dq.duplicate_signals_skipped += 1
            return f"{direction}_DUPLICATE"
        self.dq.signals_generated += 1
        return direction

    def _opening_range_complete(self, ts: datetime) -> bool:
        present = {h["timestamp"] for h in self.history}
        slot = datetime.combine(ts.date(), self.range_start, IST)
        end = datetime.combine(ts.date(), self.range_end, IST)
        while slot < end:
            if slot not in present:
                return False
            slot += self.interval
        return True

    # -- data quality ----------------------------------------------------------------------

    def _bucket(self, ts: datetime) -> datetime:
        n = self.settings.interval_minutes
        return ts.replace(minute=(ts.minute // n) * n, second=0, microsecond=0)

    def _taint(self, start: datetime, end: datetime) -> None:
        start, end = min(start, end), max(start, end)
        bucket = self._bucket(start)
        while bucket <= end:
            self._tainted.add(bucket)
            bucket += self.interval

    def _gap(self, start: datetime | None, end: datetime, reason: str, count: bool = True) -> None:
        start = start or end
        if count:
            self.dq.data_gaps += 1
        self._taint(start, end)
        closed = self.tracker.mark_gap(end, reason)
        if count or closed:
            self.store.append("DATA_QUALITY", {"event": reason, "gap_from": iso(start), "gap_to": iso(end),
                                               "signals_unresolved": [e.signal_id for e in closed]})
            self.echo(f"[DATA] {reason} {start:%H:%M:%S} -> {end:%H:%M:%S} | candles in gap not evaluated | "
                      f"open signals marked UNRESOLVED: {len(closed)}")

    def _track_health(self, now: datetime) -> None:
        if self._stopping:
            return
        status = self.feed.health().status
        if status in (MarketDataStatus.CONNECTED, MarketDataStatus.STALE):
            self.dq.websocket_connected = True
        count = getattr(self.feed, "connected_count", None)
        reconnected = isinstance(count, int) and self._connected_count is not None \
            and count > self._connected_count >= 1
        if isinstance(count, int):
            self._connected_count = count
        previous, self._last_health = self._last_health, status
        dropped = status in (MarketDataStatus.DISCONNECTED, MarketDataStatus.CONNECTING) and \
            previous in (MarketDataStatus.CONNECTED, MarketDataStatus.STALE)
        if dropped or (reconnected and not self._disconnect_counted):
            self.dq.disconnect_events += 1
            self._gap(self._last_tick_ts, now, "DISCONNECTED")
            self._in_gap = True
        if dropped:
            self._disconnect_counted = True
        elif reconnected:
            self._disconnect_counted = False

    def _restore(self) -> None:
        signals, candles = self.store.load()
        for event in signals.values():
            self.tracker.restore(event)
        rows = {}
        for raw in candles:
            ts = to_ist(raw["timestamp"])
            rows[ts] = {"timestamp": ts, **{k: float(raw[k]) for k in ("open", "high", "low", "close", "volume")}}
        for ts in sorted(rows):
            self.history.append(rows[ts])
        self.dq.restored_signals = len(signals)
        self.dq.restored_candles = len(self.history)
        if not signals and not candles:
            return
        closed = self.tracker.mark_gap(to_ist(self.clock()), "PROCESS_RESTART")
        self.store.append("DATA_QUALITY", {"event": "RESTART", "restored_signals": len(signals),
                                           "restored_candles": len(self.history),
                                           "corrupt_lines_skipped": self.store.corrupt_lines,
                                           "signals_unresolved": [e.signal_id for e in closed]})
        self.echo(f"[RESTART] session {self.session_id}: restored {len(signals)} signal(s), "
                  f"{len(self.history)} candle(s); open signals marked UNRESOLVED: {len(closed)}")

    # -- reporting ---------------------------------------------------------------------------

    def _on_signal_event(self, kind: str, ev: SignalEvent) -> None:
        self.store.append(kind, {"signal": ev.as_dict()})
        name = self.settings.underlying
        if kind == "SIGNAL":
            self.echo(f"[SIGNAL] {_hms(ev.timestamp_ist)} | {name} | {ev.strategy_name} | {ev.direction} | "
                      f"signal_close={_f(ev.signal_price)} | entry=pending ({ev.entry_timing}) | "
                      f"ATR={_f(ev.atr)} | stop_distance={_f(ev.stop_distance)} | signal_id={ev.signal_id}")
        elif kind == "TRACKING":
            self.echo(f"[ENTRY] {_hms(ev.entry_time)} | signal_id={ev.signal_id} | {ev.direction} | "
                      f"entry={_f(ev.entry_price)} | SL={_f(ev.stop_price)} | TARGET={_f(ev.target_price)}")
        elif kind == "OUTCOME":
            self.echo(f"[OUTCOME] {_hms(ev.outcome_timestamp)} | signal_id={ev.signal_id} | {ev.direction} | "
                      f"{ev.status} | entry={_f(ev.entry_price)} | exit={_f(ev.exit_price)} | "
                      f"pnl_points={_f(ev.pnl_points)} | MFE={_f(ev.mfe_points)} | MAE={_f(ev.mae_points)} | "
                      f"reason={ev.reason}")

    def print_tracking(self) -> None:
        with self._lock:
            for ev in self.tracker.active():
                if ev.status == TRACKING:
                    self.echo(f"[TRACKING] signal_id={ev.signal_id} | {ev.direction} | current={_f(ev.last_price)} "
                              f"| MFE={_f(ev.mfe_points)} | MAE={_f(ev.mae_points)}")
                else:
                    self.echo(f"[TRACKING] signal_id={ev.signal_id} | {ev.direction} | awaiting entry price")

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            self.dq.feed_messages = int(getattr(self.feed, "messages_received", 0) or 0)
            self.dq.parsed_ticks = int(getattr(self.feed, "ticks_parsed", 0) or 0)
            self.dq.delivered_ticks = int(getattr(self.feed, "ticks_delivered", 0) or 0)
            return {**asdict(self.dq), "market_data_status": self.feed.health().status.value}

    def diagnostics_line(self) -> str:
        d = self.diagnostics()
        return (f"ws_connected={d['websocket_connected']} status={d['market_data_status']} "
                f"feed_msgs={d['feed_messages']} parsed={d['parsed_ticks']} delivered={d['delivered_ticks']} "
                f"ticks={d['ticks_received']} ignored={d['ignored_symbols']} candles={d['completed_candles']} "
                f"incomplete={d['incomplete_candles']} signals={d['signals_generated']} "
                f"stale={d['stale_events']} disconnects={d['disconnect_events']} last_tick="
                f"{self._last_tick_ts.isoformat() if self._last_tick_ts else None}")

    def summary(self) -> dict[str, Any]:
        with self._lock:
            day = (self.started_at or to_ist(self.clock())).date().isoformat()
            return summarize(self.tracker.signals.values(), session_date=day, strategy=self.settings.strategy_id,
                             instrument=self.settings.underlying, timeframe=self.settings.timeframe)

    def safety(self) -> dict[str, Any]:
        return {"trading_mode": trading_mode(), "live_trading_enabled": live_trading_enabled(),
                "paper_only": True, "orders_placed": self.dq.orders_placed,
                "order_path": "none - no broker, engine or order manager is imported"}

    def status_payload(self) -> dict[str, Any]:
        """Point-in-time engine / feed health for the read-only dashboard (observed values only)."""
        with self._lock:
            s = self.settings
            last = self.history[-1] if self.history else None
            return {
                "type": "REALTIME_SIGNAL_TRACKER_STATUS",
                "session_id": self.session_id,
                "updated_at": to_ist(self.clock()).isoformat(),
                "running": self.started_at is not None and self.ended_at is None and not self._stopping,
                "started_at": iso(self.started_at) if self.started_at else None,
                "ended_at": iso(self.ended_at) if self.ended_at else None,
                "instrument_key": s.instrument_key, "underlying": s.underlying, "strategy": s.strategy_id,
                "timeframe": s.timeframe, "interval_minutes": s.interval_minutes,
                "stale_data_seconds": s.stale_data_seconds,
                "market_data_status": self.feed.health().status.value,
                "in_data_gap": self._in_gap,
                "last_tick_time": iso(self._last_tick_ts) if self._last_tick_ts else None,
                "last_tick_price": self._last_ltp,
                "last_completed_candle": {**last, "timestamp": iso(last["timestamp"])} if last else None,
                "completed_candles": self.dq.completed_candles,
                "signals_generated": self.dq.signals_generated,
                "orders_placed": self.dq.orders_placed,
                "trading_mode": trading_mode(), "live_trading_enabled": live_trading_enabled(),
            }

    def heartbeat(self) -> None:
        write_status(self.store.out_dir / STATUS_FILE, self.status_payload())
