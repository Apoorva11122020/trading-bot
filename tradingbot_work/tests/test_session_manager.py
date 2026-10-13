from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from src.session.session_manager import SessionConfig, SessionManager


IST = ZoneInfo("Asia/Kolkata")


def dt(hour, minute, second=0):
    return datetime(2026, 10, 5, hour, minute, second, tzinfo=IST)


def test_entry_window_starts_at_0930():
    sm = SessionManager()
    assert sm.can_enter(dt(9, 29, 59)) is False
    assert sm.can_enter(dt(9, 30)) is True
    assert sm.can_enter(dt(14, 44, 59)) is True
    assert sm.can_enter(dt(14, 45)) is False


def test_market_open_window():
    sm = SessionManager()
    assert sm.status(dt(9, 14, 59)).market_open is False
    assert sm.status(dt(9, 15)).market_open is True
    assert sm.status(dt(15, 29, 59)).market_open is True
    assert sm.status(dt(15, 30)).market_open is False


def test_force_exit_window():
    sm = SessionManager()
    assert sm.should_force_exit(dt(15, 14, 59)) is False
    assert sm.should_force_exit(dt(15, 15)) is True
    assert sm.should_force_exit(dt(15, 29, 59)) is True
    assert sm.should_force_exit(dt(15, 30)) is False


def test_weekend_is_not_trading_day():
    sm = SessionManager()
    saturday = datetime(2026, 10, 3, 10, 0, tzinfo=IST)
    sunday = datetime(2026, 10, 4, 10, 0, tzinfo=IST)
    assert sm.is_trading_day(saturday) is False
    assert sm.is_trading_day(sunday) is False
    assert sm.can_enter(saturday) is False


def test_configured_holiday_is_blocked():
    holiday = date(2026, 10, 5)
    sm = SessionManager(holidays={holiday})
    assert sm.is_trading_day(dt(10, 0)) is False
    assert sm.can_enter(dt(10, 0)) is False
    assert sm.status(dt(10, 0)).reason == "NON_TRADING_DAY"


def test_utc_timestamp_is_converted_to_ist():
    sm = SessionManager()
    # 04:00 UTC = 09:30 IST
    ts = datetime(2026, 10, 5, 4, 0, tzinfo=timezone.utc)
    assert sm.can_enter(ts) is True
    assert sm.status(ts).timestamp.tzinfo == IST


def test_naive_timestamp_is_interpreted_as_ist():
    sm = SessionManager()
    assert sm.can_enter(datetime(2026, 10, 5, 9, 30)) is True


def test_reason_changes_across_session():
    sm = SessionManager()
    assert sm.status(dt(8, 30)).reason == "PRE_MARKET"
    assert sm.status(dt(9, 20)).reason == "ENTRY_WINDOW_NOT_OPEN"
    assert sm.status(dt(10, 0)).reason == "ENTRY_WINDOW_OPEN"
    assert sm.status(dt(15, 0)).reason == "ENTRY_WINDOW_CLOSED"
    assert sm.status(dt(15, 20)).reason == "FORCE_EXIT_WINDOW"
    assert sm.status(dt(16, 0)).reason == "MARKET_CLOSED"
