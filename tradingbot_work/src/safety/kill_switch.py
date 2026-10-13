from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class SafetyState(str, Enum):
    RUNNING = "RUNNING"
    HALTED = "HALTED"


@dataclass
class SafetyEvent:
    event_type: str
    timestamp: datetime
    reason: str
    metadata: dict = field(default_factory=dict)


class KillSwitch:
    """Latched emergency trading stop.

    Once halted, new entries remain blocked until an explicit resume().
    The switch is independent of strategy signals.
    """

    def __init__(self):
        self._state = SafetyState.RUNNING
        self._events: list[SafetyEvent] = []

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def is_halted(self) -> bool:
        return self._state == SafetyState.HALTED

    @property
    def events(self) -> list[SafetyEvent]:
        return list(self._events)

    def activate(self, reason: str, metadata: dict | None = None) -> bool:
        """Latch HALTED state. Returns True only on the first activation."""
        if self.is_halted:
            self._events.append(
                SafetyEvent(
                    event_type="KILL_SWITCH_ALREADY_ACTIVE",
                    timestamp=datetime.now(timezone.utc),
                    reason=reason,
                    metadata=metadata or {},
                )
            )
            return False

        self._state = SafetyState.HALTED
        self._events.append(
            SafetyEvent(
                event_type="KILL_SWITCH_ACTIVATED",
                timestamp=datetime.now(timezone.utc),
                reason=reason,
                metadata=metadata or {},
            )
        )
        return True

    def resume(self, reason: str = "EXPLICIT_RESUME") -> bool:
        """Resume only through an explicit call."""
        if not self.is_halted:
            return False

        self._state = SafetyState.RUNNING
        self._events.append(
            SafetyEvent(
                event_type="KILL_SWITCH_RESUMED",
                timestamp=datetime.now(timezone.utc),
                reason=reason,
            )
        )
        return True

    def allow_entry(self) -> bool:
        return not self.is_halted

    def assert_entry_allowed(self) -> None:
        if self.is_halted:
            raise RuntimeError("TRADING_HALTED_BY_KILL_SWITCH")

    def snapshot(self) -> dict:
        return {
            "state": self.state.value,
            "halted": self.is_halted,
            "event_count": len(self._events),
        }
