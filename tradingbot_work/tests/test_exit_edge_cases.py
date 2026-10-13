import pytest

from src.broker.models import OrderStatus

from trading_support import SYMBOL, Stack, account, entry, ts


@pytest.fixture
def s(tmp_path):
    stack = Stack(tmp_path)
    yield stack
    stack.db.close()


def test_no_exit_is_evaluated_before_entry_fill(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    s.lifecycle.submit_entry(entry("E1"))
    assert s.price(90.0, ts(11, 0)) is None
    assert s.price(120.0, ts(11, 1)) is None
    assert s.monitor.state(SYMBOL) is None
    assert [o for o in s.broker.get_orders() if o["side"] == "SELL"] == []


def test_breakeven_stop_reason_when_trailing_is_below_entry(s):
    s.lifecycle.submit_entry(entry("E1"))
    assert s.price(106.0, ts(11, 0), atr=10.0) is None
    state = s.monitor.state(SYMBOL)
    assert state.breakeven_activated is True
    assert state.active_stop == pytest.approx(100.0)
    s.price(99.9, ts(11, 1), atr=10.0)
    assert s.managed().exit_reason == "BREAKEVEN_STOP"


def test_long_stop_only_moves_up(s):
    s.lifecycle.submit_entry(entry("E1"))
    stops = []
    for minute, price in enumerate([103.0, 106.0, 104.5, 108.0, 107.0]):
        s.monitor.evaluate(s.managed().position, price, 3.0, ts(11, minute))
        stops.append(s.monitor.state(SYMBOL).active_stop)
    assert stops == sorted(stops)


def test_exit_reason_persisted_on_trade_and_order(s):
    s.lifecycle.submit_entry(entry("E1"))
    record = s.price(94.0, ts(11, 0))
    assert s.db.get_trades()[0]["exit_reason"] == "STOP_LOSS"
    assert s.db.get_order(record.order_id)["reason"] == "STOP_LOSS"
    assert s.db.get_order(record.order_id)["signal_type"] == "EXIT"


def test_slippage_is_measured_at_fill_and_never_rejects_a_confirmed_fill(s):
    s.broker.set_ltp(SYMBOL, 100.5)
    record = s.om.submit_signal(entry("E1"), account())
    assert record.status == OrderStatus.FILLED
    assert record.slippage_abs == pytest.approx(0.5)
    assert record.slippage_pct == pytest.approx(0.5)
    assert s.om.expected_positions == {SYMBOL: 100}

    row = s.db.get_order(record.order_id)
    assert row["slippage_pct"] == pytest.approx(0.5)
    execution = s.db.get_executions(record.order_id)[0]
    assert execution["reference_price"] == 100.0
    assert execution["slippage_pct"] == pytest.approx(0.5)
    assert s.db.get_trade(record.order_id)["entry_slippage_pct"] == pytest.approx(0.5)

    events = s.events("SLIPPAGE_LIMIT_EXCEEDED")
    assert len(events) == 1
    assert '"limit_pct": 0.1' in events[0]["metadata_json"]


def test_slippage_within_limit_raises_no_alert(s):
    s.broker.set_ltp(SYMBOL, 100.05)
    record = s.om.submit_signal(entry("E1"), account())
    assert record.slippage_pct == pytest.approx(0.05)
    assert s.events("SLIPPAGE_LIMIT_EXCEEDED") == []


def test_unfilled_order_has_no_slippage(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    record = s.om.submit_signal(entry("E1"), account())
    assert record.slippage_pct is None
    assert s.events("SLIPPAGE_LIMIT_EXCEEDED") == []
