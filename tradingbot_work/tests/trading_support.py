"""Shared deterministic builders for lifecycle tests (PaperBroker + SQLite)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.broker.paper import PaperBroker
from src.config.loader import load_strategy_config
from src.database import Database
from src.execution.order_manager import OrderManager
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.position.lifecycle import PositionLifecycle
from src.position.monitor import PositionMonitor
from src.recovery import RecoveryManager
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager

ROOT = Path(__file__).resolve().parents[1]
IST = ZoneInfo("Asia/Kolkata")
CFG = load_strategy_config(ROOT / "config" / "strategy.yaml")
SYMBOL = "DEMO"
EQUITY = 100_000.0


def ts(hour: int, minute: int, day: int = 1) -> datetime:
    # 2026-10-01 is a Thursday (NSE trading day).
    return datetime(2026, 10, day, hour, minute, tzinfo=IST)


def entry(signal_id: str, side: str = "BUY", qty: int = 100, price: float = 100.0,
          at: datetime | None = None, symbol: str = SYMBOL) -> Signal:
    """Entry whose risk-sized quantity is exactly `qty` for 100k equity:
    0.5% risk = 500, stop distance 5 -> 100 shares."""
    long = side == "BUY"
    return Signal(
        signal_id=signal_id, timestamp=at or ts(10, 0), symbol=symbol,
        side=SignalSide.BUY if long else SignalSide.SELL, signal_type=SignalType.ENTRY,
        entry_price=price, quantity=qty,
        stop_loss=price - 5 if long else price + 5,
        target=price + 10 if long else price - 10,
        reason="TEST_ENTRY", risk_status="ALLOWED",
    )


def exit_signal(signal_id: str, side: str, qty: int, price: float = 100.0,
                reason: str = "MANUAL", symbol: str = SYMBOL) -> Signal:
    return Signal(
        signal_id=signal_id, timestamp=ts(11, 0), symbol=symbol,
        side=SignalSide(side), signal_type=SignalType.EXIT, entry_price=price,
        quantity=qty, reason=reason, risk_status="ALLOWED",
    )


def account() -> AccountState:
    return AccountState(equity=EQUITY)


class Stack:
    """PaperBroker + Database + OrderManager + PositionMonitor + Lifecycle."""

    def __init__(self, tmp_path, broker: PaperBroker | None = None, db: Database | None = None):
        self.tmp_path = tmp_path
        if broker is None:
            broker = PaperBroker()
            broker.connect()
            broker.set_ltp(SYMBOL, 100.0)
        self.broker = broker
        self.db = db or Database(Path(tmp_path) / "stack.db")
        self.session = SessionManager(CFG.session)
        self.om = OrderManager(self.broker, RiskManager(CFG.risk), self.session, database=self.db,
                               max_slippage_percent=CFG.execution.max_slippage_percent)
        self.monitor = PositionMonitor(CFG, self.session)
        self.lifecycle = PositionLifecycle(self.om, self.monitor, account)
        self.report = None

    def restart(self) -> "Stack":
        """Simulate a process restart: fresh in-memory objects, same DB and broker."""
        new = Stack(self.tmp_path, broker=self.broker, db=self.db)
        new.report = RecoveryManager(new.broker, new.db, new.om, new.lifecycle).recover()
        return new

    def managed(self, symbol: str = SYMBOL):
        return self.lifecycle.managed[symbol]

    def price(self, price: float, at: datetime, atr: float = 1.0, symbol: str = SYMBOL):
        self.broker.set_ltp(symbol, price)
        return self.lifecycle.on_price(symbol, price, atr, at)

    def events(self, event_type: str):
        return self.db.get_events(limit=1000, event_type=event_type)
