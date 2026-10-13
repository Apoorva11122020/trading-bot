from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class SessionConfig:
    market_open: time | str = time(9, 15)
    entry_start: time | str = time(9, 30)
    entry_end: time | str = time(14, 45)
    market_close: time | str = time(15, 30)
    force_exit: time | str = time(15, 15)

    def __post_init__(self):
        for field_name in (
            "market_open", "entry_start", "entry_end",
            "market_close", "force_exit",
        ):
            value = getattr(self, field_name)
            if isinstance(value, str):
                parts = value.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Invalid time for {field_name}: {value}")
                object.__setattr__(
                    self,
                    field_name,
                    time(int(parts[0]), int(parts[1])),
                )


@dataclass(frozen=True)
class SessionStatus:
    timestamp: datetime
    trading_day: bool
    market_open: bool
    entry_allowed: bool
    force_exit: bool
    reason: str


class SessionManager:
    """Indian cash-market session guard.

    Times are interpreted in Asia/Kolkata regardless of the machine's
    local timezone. Holiday dates can be injected from an exchange calendar
    source later without changing this interface.
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        holidays: set[date] | None = None,
    ):
        if config is None:
            self.config = SessionConfig()
        elif isinstance(config, SessionConfig):
            self.config = config
        else:
            # Accept the existing Pydantic config model used by Phases 1-7.
            # config/strategy.yaml names the entry window
            # allow_entries_from / stop_new_entries.
            self.config = SessionConfig(
                market_open=getattr(config, "market_open", "09:15"),
                entry_start=getattr(config, "entry_start", None)
                or getattr(config, "allow_entries_from", "09:30"),
                entry_end=getattr(config, "entry_end", None)
                or getattr(config, "stop_new_entries", "14:45"),
                market_close=getattr(config, "market_close", "15:30"),
                force_exit=getattr(config, "force_exit", "15:15"),
            )
        self.holidays = holidays or set()

    @staticmethod
    def to_ist(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=IST)
        return timestamp.astimezone(IST)

    def is_trading_day(self, timestamp: datetime) -> bool:
        ts = self.to_ist(timestamp)
        return ts.weekday() < 5 and ts.date() not in self.holidays

    def status(self, timestamp: datetime) -> SessionStatus:
        ts = self.to_ist(timestamp)
        day_ok = self.is_trading_day(ts)
        current = ts.time().replace(tzinfo=None)

        market_open = (
            day_ok
            and self.config.market_open <= current < self.config.market_close
        )
        entry_allowed = (
            day_ok
            and self.config.entry_start <= current < self.config.entry_end
        )
        force_exit = (
            day_ok
            and current >= self.config.force_exit
            and current < self.config.market_close
        )

        if not day_ok:
            reason = "NON_TRADING_DAY"
        elif current < self.config.market_open:
            reason = "PRE_MARKET"
        elif current < self.config.entry_start:
            reason = "ENTRY_WINDOW_NOT_OPEN"
        elif current < self.config.entry_end:
            reason = "ENTRY_WINDOW_OPEN"
        elif current < self.config.force_exit:
            reason = "ENTRY_WINDOW_CLOSED"
        elif current < self.config.market_close:
            reason = "FORCE_EXIT_WINDOW"
        else:
            reason = "MARKET_CLOSED"

        return SessionStatus(
            timestamp=ts,
            trading_day=day_ok,
            market_open=market_open,
            entry_allowed=entry_allowed,
            force_exit=force_exit,
            reason=reason,
        )

    def can_enter(self, timestamp: datetime) -> bool:
        return self.status(timestamp).entry_allowed

    # Backward-compatible aliases used by earlier phases.
    def is_entry_allowed(self, timestamp: datetime) -> bool:
        return self.can_enter(timestamp)

    def is_force_exit_window(self, timestamp: datetime) -> bool:
        return self.should_force_exit(timestamp)

    def should_force_exit(self, timestamp: datetime) -> bool:
        return self.status(timestamp).force_exit

    def can_manage_position(self, timestamp: datetime) -> bool:
        # Existing positions may be managed during the regular session and
        # during the force-exit window.
        return self.status(timestamp).market_open
