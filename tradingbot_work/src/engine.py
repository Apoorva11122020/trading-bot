from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from src.database import Database
from src.execution.order_manager import ExecutionRecord, OrderManager
from src.execution.poller import OrderPoller
from src.market_data.models import MarketDataHealth
from src.models.signals import Signal
from src.position.lifecycle import PositionLifecycle
from src.position.monitor import PositionMonitor
from src.recovery import RecoveryManager, RecoveryReport
from src.risk.controller import RiskController
from src.risk.risk_manager import RiskManager
from src.safety.live_gate import assert_startup_safe, startup_safety_snapshot
from src.session.session_manager import SessionManager


RUNTIME_STATUS_KEY = "runtime_status"


class TradingEngine:
    """Wires the runtime together in PAPER / sandbox mode.

    Startup: live-trading gate -> broker connect -> reconciliation
    (orders, fills, positions, managed lifecycles) -> order poller.
    Every order still goes through OrderManager; new entries pass the
    RiskController gate, exits never do.
    """

    def __init__(
        self,
        config,
        broker,
        database: Database,
        equity: float,
        market_health: Callable[[], MarketDataHealth | None] | None = None,
        clock: Callable[[], datetime] | None = None,
        start_poller: bool = True,
    ):
        self.config = config
        self.broker = broker
        self.database = database
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.market_health = market_health
        self.start_poller = start_poller

        self.session = SessionManager(config.session)
        self.risk_manager = RiskManager(config.risk)
        self.order_manager = OrderManager(
            broker,
            self.risk_manager,
            self.session,
            max_slippage_percent=config.execution.max_slippage_percent,
            database=database,
        )
        self.risk = RiskController(
            config,
            database,
            self.session,
            equity=equity,
            order_manager=self.order_manager,
            broker=broker,
            market_health=market_health,
            clock=self.clock,
        )
        self.order_manager.entry_gate = self.risk
        self.monitor = PositionMonitor(config, self.session)
        self.lifecycle = PositionLifecycle(
            self.order_manager,
            self.monitor,
            self.risk.account_state,
            database=database,
        )
        self.poller = OrderPoller(
            self.order_manager,
            interval_seconds=config.execution.order_poll_interval_seconds,
            max_backoff_seconds=config.execution.order_poll_max_backoff_seconds,
            status_timeout_seconds=config.execution.order_status_timeout_seconds,
        )
        self.recovery = RecoveryManager(broker, database, self.order_manager, self.lifecycle)
        self.last_recovery: RecoveryReport | None = None
        self.started = False

    def start(self) -> RecoveryReport:
        assert_startup_safe()
        if not self.broker.is_connected():
            self.broker.connect()
        self.last_recovery = self.recovery.recover()
        if self.start_poller:
            self.poller.start()
        self.started = True
        self.heartbeat()
        return self.last_recovery

    def stop(self) -> None:
        self.poller.stop()
        self.started = False
        self.heartbeat()

    def submit_entry(self, signal: Signal, instrument: dict | None = None) -> ExecutionRecord:
        return self.lifecycle.submit_entry(signal, instrument=instrument)

    def on_price(self, symbol: str, price: float, atr_value: float, timestamp: datetime) -> list[ExecutionRecord]:
        submitted: list[ExecutionRecord] = []
        record = self.lifecycle.on_price(symbol, price, atr_value, timestamp)
        if record is not None:
            submitted.append(record)
        submitted.extend(self.lifecycle.eod_sweep(timestamp))
        return submitted

    def heartbeat(self) -> dict[str, Any]:
        health = self.market_health() if self.market_health else None
        status = {
            "running": self.started,
            "broker_connected": bool(self.broker.is_connected()),
            "market_data_status": health.status.value if health else "UNKNOWN",
            "last_tick_at": health.last_tick_at.isoformat() if health and health.last_tick_at else None,
            "open_orders": len(self.order_manager.open_order_ids()),
            "managed_positions": self.lifecycle.open_symbols(),
            "poller_failures": self.poller.consecutive_failures,
            "safety": startup_safety_snapshot(),
            "updated_at": self.clock().isoformat(),
        }
        self.database.set_state(RUNTIME_STATUS_KEY, status)
        return status
