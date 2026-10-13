from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.execution.safety_gate import SafetyGate
from src.models.signals import Signal, SignalSide, SignalType
from src.safety.emergency_controller import EmergencyController
from src.safety.kill_switch import KillSwitch, SafetyState


def entry_signal():
    return Signal(
        signal_id="kill-test",
        timestamp=datetime(2026, 10, 3, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        symbol="RELIANCE",
        side=SignalSide.BUY,
        signal_type=SignalType.ENTRY,
        entry_price=100,
        quantity=10,
        stop_loss=98,
        target=104,
        reason="TEST",
        risk_status="ALLOWED",
    )


def exit_signal():
    return entry_signal().model_copy(
        update={
            "signal_id": "exit-test",
            "signal_type": SignalType.EXIT,
            "side": SignalSide.SELL,
        }
    )


def test_activate_latches_halted_state():
    ks = KillSwitch()
    assert ks.state == SafetyState.RUNNING
    assert ks.activate("manual emergency") is True
    assert ks.is_halted
    assert ks.allow_entry() is False
    assert ks.snapshot()["state"] == "HALTED"


def test_second_activation_does_not_toggle_or_resume():
    ks = KillSwitch()
    assert ks.activate("first") is True
    assert ks.activate("second") is False
    assert ks.is_halted
    assert ks.events[-1].event_type == "KILL_SWITCH_ALREADY_ACTIVE"


def test_entry_blocked_but_exit_allowed_while_halted():
    ks = KillSwitch()
    gate = SafetyGate(ks)
    ks.activate("emergency")

    with pytest.raises(RuntimeError, match="TRADING_HALTED_BY_KILL_SWITCH"):
        gate.validate(entry_signal())

    # Existing positions must still be closable during a halt.
    gate.validate(exit_signal())


def test_resume_requires_explicit_action():
    ks = KillSwitch()
    ks.activate("emergency")
    assert ks.allow_entry() is False
    assert ks.resume("operator confirmed") is True
    assert ks.allow_entry() is True
    assert ks.state == SafetyState.RUNNING


def test_resume_when_running_does_nothing():
    ks = KillSwitch()
    assert ks.resume() is False
    assert ks.state == SafetyState.RUNNING


def test_emergency_cancels_pending_orders():
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("RELIANCE", 100)
    broker.set_next_execution(OrderStatus.OPEN)
    oid = broker.place_order("RELIANCE", "BUY", 10, "MARKET")

    ks = KillSwitch()
    controller = EmergencyController(ks, broker)
    result = controller.activate("manual", cancel_pending=True)

    assert result.activated
    assert oid in result.cancelled_order_ids
    assert broker.get_orders()[0]["status"] == OrderStatus.CANCELLED
    assert ks.is_halted


def test_emergency_can_close_positions():
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("RELIANCE", 100)
    broker.place_order("RELIANCE", "BUY", 10, "MARKET")

    ks = KillSwitch()
    controller = EmergencyController(ks, broker)
    result = controller.activate(
        "emergency close",
        cancel_pending=True,
        close_positions=True,
    )

    assert "RELIANCE" in result.closed_symbols
    assert broker.get_positions() == []
    assert result.errors == []


def test_halted_state_survives_object_use_until_resume():
    ks = KillSwitch()
    gate = SafetyGate(ks)
    ks.activate("disconnect")

    for _ in range(3):
        assert gate.can_enter() is False
        with pytest.raises(RuntimeError):
            gate.validate(entry_signal())

    assert ks.resume() is True
    assert gate.can_enter() is True


def test_events_are_audit_trail():
    ks = KillSwitch()
    ks.activate("test", {"source": "unit"})
    ks.resume("operator")
    assert [e.event_type for e in ks.events] == [
        "KILL_SWITCH_ACTIVATED",
        "KILL_SWITCH_RESUMED",
    ]
    assert ks.events[0].metadata["source"] == "unit"
