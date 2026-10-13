"""Real-time NIFTY 5m candle + S2 signal validation (market data only, no orders)."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.research_strategies import enrich_s2
from src.market_data.candle_builder import CandleBuilder
from src.market_data.interface import MarketDataFeed
from src.market_data.models import MarketDataStatus, Tick
from src.models.market import Candle
from src.paper_trading.config import PaperTradingSettings, load_paper_trading_settings, merged_strategy_config
from src.paper_trading.s2_entry import S2SignalDetector
from src.safety.live_gate import assert_paper_trading_only

IST = ZoneInfo("Asia/Kolkata")


def tick_to_ist(tick: Tick, zone: ZoneInfo = IST) -> Tick:
    ts = tick.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return tick.model_copy(update={"timestamp": ts.astimezone(zone)})


def symbols_match(expected: str, actual: str) -> bool:
    return expected.strip() == actual.strip()


def ist_bucket_bounds(ts: datetime, interval_minutes: int) -> tuple[datetime, datetime]:
    minute = (ts.minute // interval_minutes) * interval_minutes
    start = ts.replace(minute=minute, second=0, microsecond=0)
    return start, start + timedelta(minutes=interval_minutes)


@dataclass
class ValidationDiagnostics:
    ticks_received: int = 0
    ticks_ignored_symbol: int = 0
    last_tick_timestamp: str | None = None
    last_tick_ltp: float | None = None
    last_tick_symbol: str | None = None
    open_candle_start: str | None = None
    open_candle_end: str | None = None
    completed_candles: int = 0
    websocket_connected: bool = False
    subscription_messages_sent: int = 0
    feed_ticks_delivered: int = 0
    feed_ticks_parsed: int = 0
    feed_messages_received: int = 0


@dataclass
class ValidationRunStats:
    completed_candles: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    no_signal_candles: int = 0
    stale_events: int = 0
    disconnect_events: int = 0
    orders_placed: int = 0

    @property
    def signals_generated(self) -> int:
        return self.buy_signals + self.sell_signals


@dataclass
class RealtimeNifty5mValidator:
    """Aggregate live ticks into 5m IST candles and evaluate S2 on bar close only."""

    feed: MarketDataFeed
    settings: PaperTradingSettings | None = None
    reports_dir: str | Path = "reports/realtime_validation"
    history_limit: int | None = None
    stats: ValidationRunStats = field(default_factory=ValidationRunStats)
    _history: deque[dict[str, Any]] = field(default_factory=deque, init=False, repr=False)
    _logs: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _detector: S2SignalDetector | None = field(default=None, init=False, repr=False)
    _s2_cfg: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _candle_builder: CandleBuilder | None = field(default=None, init=False, repr=False)
    _last_health: str | None = field(default=None, init=False, repr=False)
    _diagnostics: ValidationDiagnostics = field(default_factory=ValidationDiagnostics, init=False, repr=False)
    _open_bucket_start: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        assert_paper_trading_only()
        self.settings = self.settings or load_paper_trading_settings()
        _, research, _ = merged_strategy_config(self.settings)
        self._s2_cfg = dict(research.strategies["S2"])
        self._detector = S2SignalDetector(
            self.settings.strategy_id,
            self.settings.underlying,
            research,
        )
        limit = self.history_limit or self.settings.history_candles
        self._history = deque(maxlen=limit)
        self._candle_builder = CandleBuilder(self.settings.interval_minutes)
        self.feed.set_tick_handler(self.on_tick)

    @property
    def instrument_key(self) -> str:
        return self.settings.instrument_key

    def start(self) -> None:
        assert_paper_trading_only()
        self.feed.subscribe([self.instrument_key])
        self.feed.connect()

    def stop(self) -> None:
        self.feed.disconnect()

    def diagnostics(self) -> ValidationDiagnostics:
        health = self.feed.health()
        ws_ok = health.status in (MarketDataStatus.CONNECTED, MarketDataStatus.STALE)
        feed = self.feed
        return replace(
            self._diagnostics,
            completed_candles=self.stats.completed_candles,
            websocket_connected=ws_ok,
            subscription_messages_sent=getattr(feed, "subscription_messages_sent", 0),
            feed_ticks_delivered=getattr(feed, "ticks_delivered", 0),
            feed_ticks_parsed=getattr(feed, "ticks_parsed", 0),
            feed_messages_received=getattr(feed, "messages_received", 0),
        )

    def on_tick(self, tick: Tick) -> None:
        if tick.ltp <= 0:
            return
        if not symbols_match(self.instrument_key, tick.symbol):
            self._diagnostics.ticks_ignored_symbol += 1
            return
        self._diagnostics.ticks_received += 1
        self._diagnostics.last_tick_timestamp = tick.timestamp.isoformat()
        self._diagnostics.last_tick_ltp = tick.ltp
        self._diagnostics.last_tick_symbol = tick.symbol
        self._track_health()
        ist_tick = tick_to_ist(tick)
        start, end = ist_bucket_bounds(ist_tick.timestamp, self._candle_builder.interval_minutes)
        self._open_bucket_start = start
        self._diagnostics.open_candle_start = start.isoformat()
        self._diagnostics.open_candle_end = end.isoformat()
        completed = self._candle_builder.update(ist_tick)
        if completed is None:
            return
        self._on_completed_candle(completed)

    def format_diagnostics_line(self) -> str:
        d = self.diagnostics()
        return (
            f"ws_connected={d.websocket_connected} subs_sent={d.subscription_messages_sent} "
            f"feed_msgs={d.feed_messages_received} feed_parsed={d.feed_ticks_parsed} "
            f"feed_delivered={d.feed_ticks_delivered} validator_ticks={d.ticks_received} "
            f"ignored_symbol={d.ticks_ignored_symbol} "
            f"last_tick={d.last_tick_timestamp} ltp={d.last_tick_ltp} "
            f"open_candle=[{d.open_candle_start} .. {d.open_candle_end}) "
            f"completed_5m={d.completed_candles}"
        )

    def _track_health(self) -> None:
        health = self.feed.health()
        status = health.status.value
        if status == self._last_health:
            return
        if status == MarketDataStatus.STALE.value:
            self.stats.stale_events += 1
            self._log_event("MARKET_DATA_STALE", market_data_status=status)
        elif status == MarketDataStatus.DISCONNECTED.value:
            self.stats.disconnect_events += 1
            self._log_event("MARKET_DATA_DISCONNECTED", market_data_status=status)
        self._last_health = status

    def _on_completed_candle(self, candle: Candle) -> None:
        health = self.feed.health()
        if not self.feed.can_trade(candle.timestamp):
            self.stats.stale_events += 1
            self._log_event(
                "CANDLE_SKIPPED_STALE",
                candle_timestamp=candle.timestamp.isoformat(),
                market_data_status=health.status.value,
            )
            return

        self.stats.completed_candles += 1
        row = candle.model_dump()
        row["timestamp"] = candle.timestamp
        self._history.append(row)
        frame = pd.DataFrame(list(self._history))
        conditions = self._strategy_conditions(frame)
        research_signal = self._detector.on_candles(frame) if self._detector else None

        if research_signal is None:
            final_signal = "NO SIGNAL"
            self.stats.no_signal_candles += 1
        else:
            final_signal = research_signal.direction
            if final_signal == "BUY":
                self.stats.buy_signals += 1
            elif final_signal == "SELL":
                self.stats.sell_signals += 1
            conditions = {**conditions, **research_signal.conditions}

        entry = {
            "type": "CANDLE_CLOSED",
            "candle_timestamp": candle.timestamp.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "strategy_conditions": conditions,
            "final_signal": final_signal,
            "market_data_status": health.status.value,
            "orders_placed": 0,
        }
        self._logs.append(entry)
        print(
            f"[CANDLE_CLOSED] ts={entry['candle_timestamp']} "
            f"O={candle.open:.2f} H={candle.high:.2f} L={candle.low:.2f} C={candle.close:.2f} "
            f"signal={final_signal}",
            flush=True,
        )

    def _strategy_conditions(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {}
        enriched = enrich_s2(frame, self._s2_cfg)
        row = enriched.iloc[-1]
        ts = pd.Timestamp(row["timestamp"])
        range_end = self._s2_cfg.get("range_end", "09:30")
        hour, minute = map(int, range_end.split(":"))
        after_range = ts.time() > datetime(2000, 1, 1, hour, minute).time()
        orb_high = row.get("orb_high")
        orb_low = row.get("orb_low")
        return {
            "after_opening_range": bool(after_range),
            "orb_high": float(orb_high) if pd.notna(orb_high) else None,
            "orb_low": float(orb_low) if pd.notna(orb_low) else None,
            "close": float(row["close"]),
        }

    def _log_event(self, event_type: str, **payload: Any) -> None:
        self._logs.append({"type": event_type, **payload})

    def write_report(self, session_id: str | None = None) -> Path:
        out_dir = Path(self.reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sid = session_id or datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"session_{sid}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in self._logs:
                handle.write(json.dumps(row, default=str) + "\n")
            handle.write(
                json.dumps(
                    {
                        "type": "SUMMARY",
                        "completed_candles": self.stats.completed_candles,
                        "buy_signals": self.stats.buy_signals,
                        "sell_signals": self.stats.sell_signals,
                        "no_signal_candles": self.stats.no_signal_candles,
                        "signals_generated": self.stats.signals_generated,
                        "stale_events": self.stats.stale_events,
                        "disconnect_events": self.stats.disconnect_events,
                        "orders_placed": self.stats.orders_placed,
                    }
                )
                + "\n"
            )
        return path

    def summary(self) -> dict[str, Any]:
        return {
            "completed_candles": self.stats.completed_candles,
            "signals_generated": self.stats.signals_generated,
            "buy_signals": self.stats.buy_signals,
            "sell_signals": self.stats.sell_signals,
            "no_signal_candles": self.stats.no_signal_candles,
            "stale_events": self.stats.stale_events,
            "disconnect_events": self.stats.disconnect_events,
            "orders_placed": self.stats.orders_placed,
        }
