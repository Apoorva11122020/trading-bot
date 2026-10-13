from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.market_data.models import MarketDataHealth, MarketDataStatus
from src.models.risk import AccountState
from src.models.signals import Signal
from src.safety.kill_switch import KillSwitch, SafetyState


IST = ZoneInfo("Asia/Kolkata")


KILL_SWITCH_KEY = "kill_switch"


def kill_switch_state(database) -> dict[str, Any]:
    return database.get_state(KILL_SWITCH_KEY, {}) or {}


def set_kill_switch(database, active: bool, reason: str) -> bool:
    """Persist the kill switch. Returns False when already in that state.
    The database is the single source of truth, so the dashboard process
    and the trading engine always agree."""
    if bool(kill_switch_state(database).get("active")) == active:
        return False
    database.set_state(KILL_SWITCH_KEY, {
        "active": active,
        "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(),
    })
    database.add_event(
        "KILL_SWITCH_ACTIVATED" if active else "KILL_SWITCH_RESUMED",
        "CRITICAL" if active else "WARNING",
        "Kill switch activated" if active else "Kill switch resumed",
        {"reason": reason},
    )
    return True


class PersistentKillSwitch(KillSwitch):
    """Kill switch whose latched state lives in the database, so it
    survives restarts and can be operated from the dashboard process."""

    STATE_KEY = KILL_SWITCH_KEY

    def __init__(self, database):
        super().__init__()
        self.database = database

    @property
    def is_halted(self) -> bool:
        return bool(kill_switch_state(self.database).get("active"))

    @property
    def state(self) -> SafetyState:
        return SafetyState.HALTED if self.is_halted else SafetyState.RUNNING

    def activate(self, reason: str, metadata: dict | None = None) -> bool:
        return set_kill_switch(self.database, True, reason)

    def resume(self, reason: str = "EXPLICIT_RESUME") -> bool:
        return set_kill_switch(self.database, False, reason)

    @property
    def reason(self) -> str | None:
        return kill_switch_state(self.database).get("reason")


class RiskController:
    """Persistent risk state and the single gate for NEW entries.

    Daily loss and trade counts are derived from the persisted trade journal
    (IST trading day), so they survive restarts. Once the daily loss limit is
    hit the lock is latched for the rest of the IST day. Exits are never
    gated here: protective exits must always be possible.
    """

    DAILY_LOCK_KEY = "daily_loss_lock"

    def __init__(
        self,
        config,
        database,
        session_manager,
        equity: float,
        order_manager=None,
        broker=None,
        market_health: Callable[[], MarketDataHealth | None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        if equity <= 0:
            raise ValueError("equity must be > 0")
        self.config = config
        self.database = database
        self.session_manager = session_manager
        self.equity = equity
        self.order_manager = order_manager
        self.broker = broker
        self.market_health = market_health
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.kill_switch = PersistentKillSwitch(database)

    # --------------------------------------------------
    # Day boundaries (IST)
    # --------------------------------------------------

    def now(self) -> datetime:
        value = self.clock()
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    def trading_day(self) -> str:
        return self.now().astimezone(IST).date().isoformat()

    def day_start_utc(self) -> datetime:
        day = self.now().astimezone(IST).date()
        return datetime.combine(day, time(0, 0), tzinfo=IST).astimezone(timezone.utc)

    # --------------------------------------------------
    # Account state
    # --------------------------------------------------

    def daily_realized_pnl(self) -> float:
        return self.database.realized_pnl_since(self.day_start_utc())

    def trades_today(self) -> int:
        return self.database.trades_opened_since(self.day_start_utc())

    def _positions(self) -> dict[str, tuple[int, float]]:
        if self.order_manager is not None:
            return {
                symbol: (qty, self.order_manager.position_avg.get(symbol, 0.0))
                for symbol, qty in self.order_manager.expected_positions.items()
            }
        result = {}
        for row in self.database.get_positions():
            qty = int(row["quantity"])
            if qty:
                result[row["symbol"]] = (qty, float(row["average_entry"]))
        return result

    def account_state(self) -> AccountState:
        positions = self._positions()
        return AccountState(
            equity=self.equity,
            daily_pnl=self.daily_realized_pnl(),
            open_positions=len(positions),
            trades_today=self.trades_today(),
            current_exposure=sum(abs(q) * avg for q, avg in positions.values()),
        )

    # --------------------------------------------------
    # Locks
    # --------------------------------------------------

    def daily_loss_limit(self) -> float:
        inr = getattr(self.config.risk, "max_daily_loss_inr", None)
        if inr is not None and float(inr) > 0:
            return float(inr)
        return self.equity * self.config.risk.max_daily_loss_percent / 100

    def daily_loss_locked(self) -> bool:
        today = self.trading_day()
        saved = self.database.get_state(self.DAILY_LOCK_KEY, {}) or {}
        if saved.get("date") == today:
            return True
        pnl = self.daily_realized_pnl()
        if pnl <= -self.daily_loss_limit():
            self.database.set_state(self.DAILY_LOCK_KEY, {
                "date": today,
                "pnl": pnl,
                "limit": self.daily_loss_limit(),
            })
            self.database.add_event(
                "DAILY_LOSS_LOCK", "CRITICAL",
                "Daily loss limit reached; new entries locked for the rest of the trading day",
                {"date": today, "pnl": pnl, "limit": self.daily_loss_limit()},
            )
            return True
        return False

    def market_data_stale(self) -> bool:
        if self.market_health is None:
            return False
        health = self.market_health()
        if health is None:
            return True
        if health.status != MarketDataStatus.CONNECTED or health.last_tick_at is None:
            return True
        age = (self.now() - health.last_tick_at).total_seconds()
        return age > self.config.execution.stale_data_seconds

    def entry_block_reasons(self, signal: Signal | None = None) -> list[str]:
        reasons: list[str] = []
        if self.kill_switch.is_halted:
            reasons.append("KILL_SWITCH_ACTIVE")
        if self.daily_loss_locked():
            reasons.append("DAILY_LOSS_LOCKED")
        if self.trades_today() >= self.config.risk.max_trades_per_day:
            reasons.append("MAX_TRADES_PER_DAY_REACHED")
        if self.broker is not None and not self.broker.is_connected():
            reasons.append("BROKER_DISCONNECTED")
        if self.market_data_stale():
            reasons.append("STALE_MARKET_DATA")
        if not self.session_manager.is_entry_allowed(self.now()):
            reasons.append("OUTSIDE_ENTRY_SESSION")
        if reasons:
            self.database.add_event(
                "ENTRY_BLOCKED", "WARNING", "New entry blocked by risk controls",
                {"reasons": reasons, "symbol": signal.symbol if signal else None},
            )
        return reasons

    __call__ = entry_block_reasons

    def status(self) -> dict[str, Any]:
        account = self.account_state()
        return {
            "trading_day": self.trading_day(),
            "equity": self.equity,
            "daily_realized_pnl": account.daily_pnl,
            "daily_loss_limit": self.daily_loss_limit(),
            "daily_loss_locked": self.daily_loss_locked(),
            "trades_today": account.trades_today,
            "max_trades_per_day": self.config.risk.max_trades_per_day,
            "open_positions": account.open_positions,
            "max_open_positions": self.config.risk.max_open_positions,
            "current_exposure": account.current_exposure,
            "max_exposure": self.equity * self.config.risk.max_portfolio_exposure_percent / 100,
            "kill_switch_active": self.kill_switch.is_halted,
            "kill_switch_reason": self.kill_switch.reason if self.kill_switch.is_halted else None,
            "market_data_stale": self.market_data_stale(),
            "broker_connected": self.broker.is_connected() if self.broker is not None else None,
            "entry_session_open": self.session_manager.is_entry_allowed(self.now()),
        }
