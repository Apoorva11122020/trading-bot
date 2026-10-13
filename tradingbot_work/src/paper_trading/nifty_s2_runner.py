"""Real-time NIFTY S2 paper trading runner."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.broker.factory import create_broker
from src.database import Database
from src.engine import TradingEngine
from src.indicators.technical import atr as calc_atr
from src.market_data.candle_builder import CandleBuilder
from src.market_data.interface import MarketDataFeed
from src.market_data.models import Tick
from src.models.market import Candle
from src.options.instrument_resolver import build_resolver
from src.paper_trading.config import load_paper_trading_settings, merged_strategy_config
from src.paper_trading.journal import PAPER_STATE_KEY, PaperTradingJournal
from src.paper_trading.s2_entry import S2OptionEntryBuilder, S2SignalDetector
from src.paper_trading.signal_tracker import LiveSignalTracker, SignalRecord
from src.position.monitor import PositionMonitor
from src.options.strategy import option_exit_config
from src.risk.risk_manager import RiskManager
from src.safety.live_gate import assert_paper_trading_only, live_trading_enabled, trading_mode
from src.session.session_manager import SessionManager


def print_startup_banner(settings, research, strategy_config) -> None:
    print("=== NIFTY S2 Paper Trading ===")
    print(f"TRADING_MODE={trading_mode()}")
    print(f"LIVE_TRADING_ENABLED={live_trading_enabled()}")
    print(f"strategy={settings.strategy_id}")
    print(f"underlying={settings.underlying}")
    print(f"timeframe={settings.timeframe}")
    print(f"daily_loss_limit=₹{research.risk.max_daily_loss_inr:,.0f}")
    print(
        f"trading_window={settings.session.allow_entries_from}"
        f"-{settings.session.stop_new_entries} IST"
    )
    print("PAPER TRADING ONLY | LIVE TRADING DISABLED")
    print("==============================\n")


class NiftyS2PaperRunner:
    """Orchestrates real-time S2 signals, risk, and paper execution."""

    def __init__(
        self,
        *,
        database: Database,
        feed: MarketDataFeed,
        settings=None,
        broker=None,
    ):
        assert_paper_trading_only()
        self.settings = settings or load_paper_trading_settings()
        self.strategy_config, self.research_config, _ = merged_strategy_config(self.settings)
        self.database = database
        self.feed = feed
        self.broker = broker or create_broker()
        self.session_id = uuid.uuid4().hex[:12]
        self.journal = PaperTradingJournal(database, self.settings.reports_dir, self.session_id)
        self.session = SessionManager(self.strategy_config.session)
        self.risk_manager = RiskManager(self.strategy_config.risk)
        self.resolver = build_resolver(self.strategy_config.options)
        self.detector = S2SignalDetector(
            self.settings.strategy_id,
            self.settings.underlying,
            self.research_config,
        )
        self.entry_builder = S2OptionEntryBuilder(
            self.strategy_config,
            self.research_config,
            self.risk_manager,
            self.resolver,
        )
        self.candle_builder = CandleBuilder(self.settings.interval_minutes)
        self.option_candle_builder = CandleBuilder(self.settings.interval_minutes)
        self._underlying_history: deque[dict] = deque(maxlen=self.settings.history_candles)
        self._option_history: deque[dict] = deque(maxlen=self.settings.history_candles)
        self._active_option_key: str | None = None
        self._resume_after: datetime | None = None
        self._lock = threading.RLock()
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False

        self.engine = TradingEngine(
            self.strategy_config,
            self.broker,
            database,
            equity=float(self.settings.initial_equity),
            market_health=lambda: self.feed.health(),
            start_poller=True,
        )
        self.engine.monitor = PositionMonitor(
            option_exit_config(self.strategy_config),
            self.session,
        )
        self.engine.lifecycle.monitor = self.engine.monitor
        self.signal_tracker = LiveSignalTracker(database, square_off=self.settings.session.square_off)
        self.feed.set_tick_handler(self.on_tick)

    @property
    def underlying_key(self) -> str:
        return self.settings.instrument_key

    def _candles_df(self, history: deque[dict]) -> pd.DataFrame:
        if not history:
            return pd.DataFrame()
        return pd.DataFrame(list(history))

    def _option_df(self) -> pd.DataFrame:
        return self._candles_df(self._option_history)

    def _restore_open_option(self) -> None:
        symbols = self.engine.lifecycle.open_symbols()
        if symbols:
            self._active_option_key = symbols[0]
            self.feed.subscribe([self.underlying_key, self._active_option_key])
            self.journal.contract({"instrument_key": self._active_option_key, "restored": True})

    def start(self) -> None:
        assert_paper_trading_only()
        print_startup_banner(self.settings, self.research_config, self.strategy_config)
        self.engine.start()
        self._restore_open_option()
        self.feed.subscribe([self.underlying_key] if not self._active_option_key else [self.underlying_key, self._active_option_key])
        self.feed.connect()
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="paper-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        self.journal.event("PAPER_SESSION_START", "INFO", "NIFTY S2 paper session started", session_id=self.session_id)
        self._publish_status()

    def stop(self) -> None:
        self._running = False
        self.feed.disconnect()
        self.engine.stop()
        self.journal.event("PAPER_SESSION_STOP", "INFO", "NIFTY S2 paper session stopped")
        self._publish_status(running=False)

    def _heartbeat_loop(self) -> None:
        while self._running:
            self._publish_status()
            self.engine.heartbeat()
            time.sleep(5)

    def _publish_status(self, running: bool | None = None) -> None:
        health = self.feed.health()
        account = self.engine.risk.account_state()
        remaining = self.risk_manager.remaining_daily_loss_budget(account)
        open_symbols = self.engine.lifecycle.open_symbols()
        position = None
        unrealized = 0.0
        entry_premium = None
        current_premium = None
        if open_symbols:
            sym = open_symbols[0]
            entry_premium = self.broker.prices.get(sym)
            current_premium = self.broker.prices.get(sym)
            qty = self.engine.order_manager.expected_positions.get(sym, 0)
            avg = self.engine.order_manager.position_avg.get(sym, 0.0)
            if current_premium and qty:
                unrealized = (current_premium - avg) * qty
            position = {"symbol": sym, "quantity": qty, "average_entry": avg}

        status = {
            "session_id": self.session_id,
            "running": self._running if running is None else running,
            "safety_banner": "PAPER TRADING ONLY | LIVE TRADING DISABLED",
            "mode": trading_mode(),
            "live_trading_enabled": live_trading_enabled(),
            "strategy": self.settings.strategy_id,
            "underlying": self.settings.underlying,
            "timeframe": self.settings.timeframe,
            "current_option": self.journal._status.get("current_option"),
            "latest_signal": self.journal._status.get("latest_signal"),
            "paper_position": position,
            "entry_premium": entry_premium,
            "current_premium": current_premium,
            "unrealized_pnl": unrealized,
            "realized_daily_pnl": account.daily_pnl,
            "remaining_daily_risk_budget": remaining,
            "daily_loss_limit_inr": self.risk_manager.daily_loss_limit(account),
            "trades_today": account.trades_today,
            "last_market_data_at": health.last_tick_at.isoformat() if health.last_tick_at else None,
            "market_data_status": health.status.value,
            "market_data_error": health.last_error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.journal.update_status(status)

    def on_tick(self, tick: Tick) -> None:
        with self._lock:
            if tick.ltp <= 0:
                return
            self.broker.set_ltp(tick.symbol, tick.ltp)

            if tick.symbol == self.underlying_key:
                completed = self.candle_builder.update(tick)
                if completed is not None:
                    self._on_underlying_candle(completed)
            elif tick.symbol == self._active_option_key:
                completed = self.option_candle_builder.update(tick)
                if completed is not None:
                    self._option_history.append(completed.model_dump())
                    self._track_candle(tick.symbol, completed)

            self._manage_open_positions(tick)

    def _track_candle(self, instrument_key: str, candle: Candle) -> None:
        try:
            self.signal_tracker.on_candle(instrument_key, candle)
        except Exception as exc:  # observation layer must never disturb paper trading
            self.journal.event("SIGNAL_TRACKER_ERROR", "WARNING", str(exc))

    def _track_signal(self, research_signal, result) -> None:
        try:
            signal, contract = result.signal, result.contract or {}
            self.signal_tracker.register(SignalRecord.create(
                timestamp=research_signal.timestamp,
                underlying=self.settings.underlying,
                direction=research_signal.direction,
                strategy=self.settings.strategy_id,
                active_from=pd.Timestamp(research_signal.timestamp)
                + pd.Timedelta(minutes=self.settings.interval_minutes),
                option_type=contract.get("option_type"),
                strike=contract.get("strike"),
                expiry=str(contract["expiry"]) if contract.get("expiry") else None,
                instrument_key=contract.get("instrument_key") or self._active_option_key,
                trading_symbol=contract.get("trading_symbol"),
                signal_price=research_signal.spot,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                target=signal.target,
                quantity=signal.quantity,
                source="LIVE",
            ))
        except Exception as exc:
            self.journal.event("SIGNAL_TRACKER_ERROR", "WARNING", str(exc))

    def _manage_open_positions(self, tick: Tick) -> None:
        for symbol in list(self.engine.lifecycle.open_symbols()):
            price = self.broker.prices.get(symbol)
            if price is None or price <= 0:
                continue
            frame = self._option_df() if symbol == self._active_option_key else self._candles_df(self._underlying_history)
            atr_value = 0.0
            if not frame.empty and "high" in frame and "low" in frame:
                series = calc_atr(frame, self.strategy_config.indicators.atr_period)
                if not series.empty and pd.notna(series.iloc[-1]):
                    atr_value = float(series.iloc[-1])
            if atr_value <= 0:
                continue
            records = self.engine.on_price(symbol, price, atr_value, tick.timestamp)
            for record in records:
                self.journal.order_stage(
                    "EXIT",
                    order_id=record.order_id,
                    symbol=record.symbol,
                    status=record.status.value,
                    pnl=record.realized_pnl,
                )
            if symbol in self.engine.lifecycle.open_symbols():
                continue
            if symbol == self._active_option_key:
                self._active_option_key = None
                self._resume_after = tick.timestamp
                self.feed.subscribe([self.underlying_key])

    def _on_underlying_candle(self, candle: Candle) -> None:
        if not self.feed.can_trade(candle.timestamp):
            self.journal.market_data(self.feed.health().status.value, reason="STALE_OR_DISCONNECTED")
            return

        self._underlying_history.append(candle.model_dump())

        if self._resume_after is not None and candle.timestamp <= self._resume_after:
            return

        if self.engine.lifecycle.open_symbols():
            return

        block_reasons = self.engine.risk.entry_block_reasons()
        if block_reasons:
            if "DAILY_LOSS_LOCKED" in block_reasons:
                self.journal.risk_lock(reasons=block_reasons)
            self.journal.skipped("ENTRY_BLOCKED", reasons=block_reasons)
            return

        if not self.session.is_entry_allowed(candle.timestamp):
            self.journal.skipped("OUTSIDE_ENTRY_SESSION", time=candle.timestamp.isoformat())
            return

        frame = self._candles_df(self._underlying_history)
        if frame.empty:
            return

        research_signal = self.detector.on_candles(frame)
        if research_signal is None:
            return

        self.journal.signal(research_signal.as_dict())

        if self._active_option_key is None:
            # Need option premium candles before entry — subscribe first tick window.
            try:
                contract = self.entry_builder._resolve_contract(research_signal)
            except Exception as exc:
                self.journal.skipped("CONTRACT_RESOLUTION_FAILED", error=str(exc))
                return
            self._active_option_key = contract.instrument_key
            self.feed.subscribe([self.underlying_key, self._active_option_key])
            self.journal.contract(contract.as_dict())

        option_frame = self._option_df()
        if option_frame.empty:
            self.journal.skipped("OPTION_PREMIUM_DATA_UNAVAILABLE", instrument_key=self._active_option_key)
            return

        account = self.engine.risk.account_state()
        result = self.entry_builder.build(
            research_signal,
            option_frame,
            account,
            open_option_positions=len(self.engine.lifecycle.open_symbols()),
            session_allowed=self.session.is_entry_allowed(research_signal.timestamp),
        )
        self.journal.risk_decision(
            result.signal is not None,
            (result.risk or {}).get("reasons", []) if result.risk else ([result.skip_reason] if result.skip_reason else []),
            sizing=result.sizing,
        )
        if result.contract:
            self.journal.contract(result.contract)

        if result.signal is None:
            self.journal.skipped(result.skip_reason or "RISK_REJECTED", sizing=result.sizing)
            return

        self._track_signal(research_signal, result)
        self.journal.order_stage("SIGNAL", signal_id=result.signal.signal_id)
        instrument = result.contract or result.signal.confidence_metadata.get("instrument")
        try:
            record = self.engine.submit_entry(result.signal, instrument=instrument)
        except RuntimeError as exc:
            self.journal.skipped(str(exc), stage="SUBMIT_ENTRY")
            return

        if instrument and hasattr(self.engine.order_manager, "submit_signal"):
            pass

        self.journal.order_stage(
            "ORDER",
            order_id=record.order_id,
            status=record.status.value,
            quantity=record.filled_quantity,
            price=record.average_fill_price,
        )
        if record.status.value == "FILLED":
            self.journal.order_stage("FILLED", order_id=record.order_id)
            self._resume_after = None

    def status(self) -> dict[str, Any]:
        return self.database.get_state(PAPER_STATE_KEY, {}) or {}
