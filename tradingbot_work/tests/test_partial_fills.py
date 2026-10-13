import pytest

from src.broker.models import OrderStatus
from src.broker.upstox_portfolio import OrderUpdate

from trading_support import SYMBOL, Stack, account, entry, exit_signal, ts


@pytest.fixture
def s(tmp_path):
    stack = Stack(tmp_path)
    yield stack
    stack.db.close()


def _open_entry(s, signal_id="E1"):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    return s.om.submit_signal(entry(signal_id), account())


def test_partial_entry_then_full_entry_updates_position_and_average(s):
    rec = _open_entry(s)
    assert rec.status == OrderStatus.OPEN
    assert s.om.expected_positions == {}

    s.broker.fill_order(rec.order_id, 40, 100.0)
    s.om.refresh_order(rec.order_id)
    assert rec.status == OrderStatus.PARTIALLY_FILLED
    assert rec.filled_quantity == 40
    assert s.om.expected_positions == {SYMBOL: 40}
    trade = s.db.get_trade(rec.order_id)
    assert trade["status"] == "OPEN" and trade["quantity"] == 40 and trade["entry_price"] == 100.0

    s.broker.fill_order(rec.order_id, 60, 101.0)
    s.om.refresh_order(rec.order_id)
    assert rec.status == OrderStatus.FILLED
    assert rec.filled_quantity == 100
    assert rec.average_fill_price == pytest.approx(100.6)
    assert s.om.expected_positions == {SYMBOL: 100}
    assert s.om.position_avg[SYMBOL] == pytest.approx(100.6)

    trade = s.db.get_trade(rec.order_id)
    assert trade["quantity"] == 100
    assert trade["entry_price"] == pytest.approx(100.6)
    executions = s.db.get_executions(rec.order_id)
    assert [(e["quantity"], e["price"]) for e in executions] == [(40, 100.0), (60, pytest.approx(101.0))]
    position = s.db.get_positions()[0]
    assert position["side"] == "LONG" and position["quantity"] == 100
    assert position["average_entry"] == pytest.approx(100.6)
    assert s.om.reconcile()["ok"] is True


def test_repeated_broker_reports_are_idempotent(s):
    rec = _open_entry(s)
    s.broker.fill_order(rec.order_id, 40, 100.0)
    for _ in range(3):
        s.om.refresh_order(rec.order_id)
        s.om.refresh_open_orders()
    assert s.om.expected_positions == {SYMBOL: 40}
    assert len(s.db.get_executions(rec.order_id)) == 1
    assert s.db.get_trade(rec.order_id)["quantity"] == 40
    statuses = [row["status"] for row in s.db.get_order_history(rec.order_id)]
    assert statuses == ["SUBMITTED", "OPEN", "PARTIALLY_FILLED"]


def test_partial_exit_keeps_trade_open_until_flat(s):
    rec = _open_entry(s)
    s.broker.fill_order(rec.order_id, 40, 100.0)
    s.broker.fill_order(rec.order_id, 60, 101.0)
    s.om.refresh_order(rec.order_id)

    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    x = s.om.submit_signal(exit_signal("X1", "SELL", 100), account())
    assert s.om.expected_positions == {SYMBOL: 100}

    s.broker.fill_order(x.order_id, 30, 102.0)
    s.om.refresh_order(x.order_id)
    assert s.om.expected_positions == {SYMBOL: 70}
    trade = s.db.get_trade(rec.order_id)
    assert trade["status"] == "OPEN"
    assert trade["exit_quantity"] == 30
    assert trade["realized_pnl"] == pytest.approx((102.0 - 100.6) * 30)

    with pytest.raises(RuntimeError, match="EXIT_ALREADY_PENDING"):
        s.om.submit_signal(exit_signal("X-DUP", "SELL", 70), account())

    s.broker.fill_order(x.order_id, 70, 103.0)
    s.om.refresh_order(x.order_id)
    assert s.om.expected_positions == {}
    trade = s.db.get_trade(rec.order_id)
    assert trade["status"] == "CLOSED"
    assert trade["exit_quantity"] == 100
    assert trade["exit_price"] == pytest.approx(102.7)
    assert trade["realized_pnl"] == pytest.approx((102.7 - 100.6) * 100)
    assert x.realized_pnl == pytest.approx((102.7 - 100.6) * 100)
    assert s.db.get_positions() == []
    assert s.om.reconcile()["ok"] is True


def test_exit_can_never_flip_position(s):
    rec = _open_entry(s)
    s.broker.fill_order(rec.order_id, 40, 100.0)
    s.om.refresh_order(rec.order_id)
    with pytest.raises(RuntimeError, match="EXIT_QUANTITY_EXCEEDS_POSITION"):
        s.om.submit_signal(exit_signal("X-BIG", "SELL", 41), account())
    with pytest.raises(RuntimeError, match="NO_OPEN_POSITION_TO_EXIT"):
        s.om.submit_signal(exit_signal("X-WRONG", "BUY", 10), account())


def test_opposite_entry_is_blocked_while_position_open(s):
    s.om.submit_signal(entry("E-LONG"), account())
    with pytest.raises(RuntimeError, match="ENTRY_CONFLICTS_WITH_OPEN_POSITION"):
        s.om.submit_signal(entry("E-SHORT", side="SELL"), account())


def test_stream_fill_price_is_derived_per_fill_from_cumulative_average(s):
    rec = _open_entry(s)
    s.om.apply_order_update(OrderUpdate(rec.order_id, "open", SYMBOL, "BUY", 100, 40, 60, 100.0))
    s.om.apply_order_update(OrderUpdate(rec.order_id, "open", SYMBOL, "BUY", 100, 40, 60, 100.0))
    assert rec.status == OrderStatus.PARTIALLY_FILLED
    s.om.apply_order_update(OrderUpdate(rec.order_id, "complete", SYMBOL, "BUY", 100, 100, 0, 100.6))
    prices = [e["price"] for e in s.db.get_executions(rec.order_id)]
    assert prices == [pytest.approx(100.0), pytest.approx(101.0)]
    assert s.om.expected_positions == {SYMBOL: 100}


def test_lifecycle_tracks_partial_entry_and_partial_exit(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    rec = s.lifecycle.submit_entry(entry("E1"))
    assert s.managed().position is None

    s.broker.fill_order(rec.order_id, 40, 100.0)
    s.om.refresh_open_orders()
    assert s.managed().position.quantity == 40

    s.broker.fill_order(rec.order_id, 60, 101.0)
    s.om.refresh_open_orders()
    position = s.managed().position
    assert position.quantity == 100
    assert position.average_entry == pytest.approx(100.6)

    # Target exit fills 30 then the remainder is cancelled by the broker.
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    first_exit = s.price(110.0, ts(11, 0))
    assert first_exit.requested_quantity == 100
    s.broker.fill_order(first_exit.order_id, 30, 110.0)
    s.broker.cancel_order(first_exit.order_id)
    s.om.refresh_open_orders()
    assert s.managed().closed is False
    assert s.managed().position.quantity == 70
    assert s.managed().exit_order_id is None

    second_exit = s.price(110.0, ts(11, 5))
    assert second_exit.requested_quantity == 70
    assert s.managed().closed is True
    expected = (110.0 - 100.6) * 100
    assert s.managed().realized_pnl == pytest.approx(expected)
    assert s.db.get_trades()[0]["status"] == "CLOSED"
    assert s.db.get_trades()[0]["realized_pnl"] == pytest.approx(expected)


def test_exit_cancels_unfilled_entry_remainder_first(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    rec = s.lifecycle.submit_entry(entry("E1"))
    s.broker.fill_order(rec.order_id, 40, 100.0)
    s.om.refresh_open_orders()

    exit_record = s.price(94.0, ts(11, 0))
    assert rec.status == OrderStatus.CANCELLED
    assert rec.filled_quantity == 40
    assert exit_record.requested_quantity == 40
    assert s.managed().closed is True
    assert s.managed().realized_pnl == pytest.approx((94.0 - 100.0) * 40)
    assert any(e["event_type"] == "ENTRY_REMAINDER_CANCELLED" for e in s.db.get_events(limit=200))
