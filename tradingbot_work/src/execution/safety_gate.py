from src.models.signals import SignalType
from src.safety.kill_switch import KillSwitch


class SafetyGate:
    """Final safety gate immediately before an order reaches OrderManager."""

    def __init__(self, kill_switch: KillSwitch):
        self.kill_switch = kill_switch

    def validate(self, signal) -> None:
        # EXIT signals are intentionally allowed while halted so that an
        # emergency close can still be executed.
        if signal.signal_type == SignalType.EXIT:
            return

        self.kill_switch.assert_entry_allowed()

    def can_enter(self) -> bool:
        return self.kill_switch.allow_entry()
