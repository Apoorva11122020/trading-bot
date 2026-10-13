from datetime import datetime
from zoneinfo import ZoneInfo

from src.config.loader import load_strategy_config
from src.models.positions import Position
from src.position.monitor import PositionMonitor
from src.session.session_manager import SessionManager


def setup():
    cfg = load_strategy_config("config/strategy.yaml")
    return cfg, PositionMonitor(cfg, SessionManager(cfg.session))


def position(side="LONG"):
    return Position(
        symbol="RELIANCE",
        side=side,
        quantity=100,
        average_entry=100.0,
        current_price=100.0,
        stop_loss=97.0 if side == "LONG" else 103.0,
        target=106.0 if side == "LONG" else 94.0,
        entry_time=datetime(2026, 10, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )


def ts(hour, minute):
    return datetime(2026, 10, 1, hour, minute, tzinfo=ZoneInfo("Asia/Kolkata"))


def test_long_stop_loss():
    _, monitor = setup()
    p = position("LONG")
    signal = monitor.evaluate(p, 96.5, 1.0, ts(11, 0))
    assert signal is not None
    assert signal.reason == "STOP_LOSS"
    assert signal.side.value == "SELL"


def test_long_target():
    _, monitor = setup()
    p = position("LONG")
    signal = monitor.evaluate(p, 106.0, 1.0, ts(11, 0))
    assert signal is not None
    assert signal.reason == "TARGET"


def test_short_stop_and_target():
    _, monitor = setup()
    p = position("SHORT")
    assert monitor.evaluate(p, 103.5, 1.0, ts(11, 0)).reason == "STOP_LOSS"

    _, monitor = setup()
    assert monitor.evaluate(p, 94.0, 1.0, ts(11, 0)).reason == "TARGET"


def test_breakeven_activates_after_one_r():
    _, monitor = setup()
    p = position("LONG")

    assert monitor.evaluate(p, 102.5, 0.5, ts(11, 0)) is None
    state = monitor.state("RELIANCE")
    assert state.breakeven_activated is False

    assert monitor.evaluate(p, 103.0, 0.5, ts(11, 1)) is None
    state = monitor.state("RELIANCE")
    assert state.breakeven_activated is True
    assert state.active_stop >= 100.0

    signal = monitor.evaluate(p, 99.9, 0.5, ts(11, 2))
    assert signal is not None
    assert signal.reason == "TRAILING_STOP"


def test_trailing_stop_moves_only_in_favorable_direction():
    _, monitor = setup()
    p = position("LONG")

    assert monitor.evaluate(p, 103.0, 1.0, ts(11, 0)) is None
    first = monitor.state("RELIANCE").active_stop

    assert monitor.evaluate(p, 104.0, 1.0, ts(11, 1)) is None
    second = monitor.state("RELIANCE").active_stop
    assert second >= first

    assert monitor.evaluate(p, 103.5, 1.0, ts(11, 2)) is None
    exit_signal = monitor.evaluate(p, 102.9, 1.0, ts(11, 3))
    assert exit_signal is not None
    assert exit_signal.reason == "TRAILING_STOP"
    third = monitor.state("RELIANCE").active_stop
    assert third == second


def test_eod_force_exit():
    _, monitor = setup()
    p = position("LONG")
    signal = monitor.evaluate(p, 100.5, 1.0, ts(15, 15))
    assert signal is not None
    assert signal.reason == "FORCE_EXIT_EOD"


def test_duplicate_exit_is_blocked_until_state_is_completed():
    _, monitor = setup()
    p = position("LONG")

    first = monitor.evaluate(p, 96.5, 1.0, ts(11, 0))
    second = monitor.evaluate(p, 96.0, 1.0, ts(11, 1))
    assert first is not None
    assert second is None

    monitor.mark_exit_completed("RELIANCE")
    third = monitor.evaluate(p, 96.0, 1.0, ts(11, 2))
    assert third is not None


def test_gap_through_stop_still_exits():
    _, monitor = setup()
    p = position("LONG")
    signal = monitor.evaluate(p, 90.0, 1.0, ts(11, 0))
    assert signal is not None
    assert signal.reason == "STOP_LOSS"


def test_invalid_atr_is_rejected():
    _, monitor = setup()
    p = position("LONG")
    try:
        monitor.evaluate(p, 100.5, 0.0, ts(11, 0))
        assert False, "Expected ValueError"
    except ValueError:
        pass
