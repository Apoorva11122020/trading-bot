from datetime import datetime, timezone

import pytest

from src.broker.models import OrderStatus
from src.position.lifecycle import EOD_FORCE_EXIT

from trading_support import SYMBOL, Stack, account, entry, ts


@pytest.fixture
def s(tmp_path):
    stack = Stack(tmp_path)
    yield stack
    stack.db.close()


def test_no_eod_action_before_force_exit_time(s):
    s.lifecycle.submit_entry(entry("E1"))
    assert s.lifecycle.eod_sweep(ts(15, 14)) == []
    assert s.managed().closed is False


def test_eod_closes_long_with_sell_through_order_manager(s):
    s.lifecycle.submit_entry(entry("E1"))
    s.broker.set_ltp(SYMBOL, 101.0)
    submitted = s.lifecycle.eod_sweep(ts(15, 15))
    assert len(submitted) == 1
    assert submitted[0].side == "SELL"
    assert submitted[0].reason == EOD_FORCE_EXIT
    assert s.managed().closed is True
    assert s.managed().close_reason == EOD_FORCE_EXIT
    trade = s.db.get_trades()[0]
    assert trade["exit_reason"] == EOD_FORCE_EXIT
    assert trade["realized_pnl"] == pytest.approx(100.0)
    assert s.db.get_order(submitted[0].order_id)["reason"] == EOD_FORCE_EXIT


def test_eod_closes_short_with_buy(s):
    s.lifecycle.submit_entry(entry("S1", side="SELL"))
    submitted = s.lifecycle.eod_sweep(ts(15, 20))
    assert submitted[0].side == "BUY"
    assert s.managed().closed is True
    assert s.om.expected_positions == {}


def test_eod_is_not_duplicated_and_retries_after_reject(s):
    s.lifecycle.submit_entry(entry("E1"))
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    first = s.lifecycle.eod_sweep(ts(15, 15))
    assert len(first) == 1
    orders_after_first = len(s.broker.get_orders())

    # Pending EOD exit: repeated sweeps and price ticks do not duplicate it.
    assert s.lifecycle.eod_sweep(ts(15, 16)) == []
    assert s.price(100.0, ts(15, 16)) is None
    assert len(s.broker.get_orders()) == orders_after_first

    s.broker.reject_order(first[0].order_id, "RMS")
    s.om.refresh_open_orders()
    assert s.managed().closed is False
    assert s.managed().exit_order_id is None

    retry = s.lifecycle.eod_sweep(ts(15, 17))
    assert len(retry) == 1 and retry[0].order_id != first[0].order_id
    assert s.managed().closed is True
    assert len(s.db.get_trades()) == 1


def test_eod_uses_asia_kolkata_time(s):
    s.lifecycle.submit_entry(entry("E1"))
    # 09:45 UTC == 15:15 IST
    submitted = s.lifecycle.eod_sweep(datetime(2026, 10, 1, 9, 45, tzinfo=timezone.utc))
    assert len(submitted) == 1


def test_monitor_force_exit_is_reported_as_eod_force_exit(s):
    s.lifecycle.submit_entry(entry("E1"))
    record = s.price(100.5, ts(15, 15))
    assert record.reason == EOD_FORCE_EXIT
    assert s.managed().close_reason == EOD_FORCE_EXIT


def test_no_new_entries_after_cutoff(s):
    with pytest.raises(RuntimeError, match="ENTRY_OUTSIDE_TRADING_SESSION"):
        s.om.submit_signal(entry("LATE", at=ts(14, 50)), account())
    assert s.broker.get_orders() == []


def test_pending_entry_is_cancelled_at_eod(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    record = s.lifecycle.submit_entry(entry("E1"))
    assert s.lifecycle.eod_sweep(ts(15, 15)) == []
    assert record.status == OrderStatus.CANCELLED
    assert s.managed().closed is True
    assert s.managed().close_reason == "ENTRY_NOT_FILLED"
