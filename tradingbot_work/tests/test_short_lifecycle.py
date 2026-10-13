import pytest

from src.broker.models import OrderStatus

from trading_support import SYMBOL, Stack, account, entry, exit_signal, ts


@pytest.fixture
def s(tmp_path):
    stack = Stack(tmp_path)
    yield stack
    stack.db.close()


def _open_short(s):
    record = s.lifecycle.submit_entry(entry("S1", side="SELL"))
    position = s.managed().position
    assert record.status == OrderStatus.FILLED
    assert position.side == "SHORT" and position.quantity == 100
    assert s.om.expected_positions == {SYMBOL: -100}
    assert s.db.get_positions()[0]["side"] == "SHORT"
    return record


def test_short_target_exit_buys_back_with_short_pnl(s):
    _open_short(s)
    exit_record = s.price(90.0, ts(11, 0))
    assert exit_record.side == "BUY"
    assert s.managed().exit_reason == "TARGET"
    assert s.managed().closed is True
    assert s.managed().realized_pnl == pytest.approx((100.0 - 90.0) * 100)
    trade = s.db.get_trades()[0]
    assert trade["side"] == "SELL" and trade["status"] == "CLOSED"
    assert trade["realized_pnl"] == pytest.approx(1000.0)
    assert trade["exit_reason"] == "TARGET"
    assert s.om.reconcile()["ok"] is True


def test_short_stop_loss_exit(s):
    _open_short(s)
    s.price(105.5, ts(11, 0))
    assert s.managed().exit_reason == "STOP_LOSS"
    assert s.managed().realized_pnl == pytest.approx((100.0 - 105.5) * 100)


def test_short_trailing_stop_only_moves_down(s):
    _open_short(s)
    assert s.price(93.0, ts(11, 0)) is None
    state = s.monitor.state(SYMBOL)
    assert state.breakeven_activated is True
    assert state.active_stop == pytest.approx(94.0)

    exit_record = s.price(94.5, ts(11, 1))
    assert exit_record is not None and exit_record.side == "BUY"
    assert s.monitor.state(SYMBOL) is None
    assert s.managed().exit_reason == "TRAILING_STOP"
    assert s.managed().realized_pnl == pytest.approx((100.0 - 94.5) * 100)


def test_short_stop_never_loosens(s):
    _open_short(s)
    stops = []
    for minute, price in enumerate([97.0, 95.0, 96.5, 94.0, 95.5]):
        s.broker.set_ltp(SYMBOL, price)
        s.monitor.evaluate(s.managed().position, price, 3.0, ts(11, minute))
        stops.append(s.monitor.state(SYMBOL).active_stop)
    assert stops == sorted(stops, reverse=True)


def test_short_partial_exit_and_duplicate_prevention(s):
    _open_short(s)
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    exit_record = s.price(90.0, ts(11, 0))
    s.broker.fill_order(exit_record.order_id, 40, 95.0)
    s.om.refresh_open_orders()
    assert s.om.expected_positions == {SYMBOL: -60}
    assert s.managed().position.quantity == 60
    assert s.managed().closed is False

    assert s.price(89.0, ts(11, 1)) is None
    with pytest.raises(RuntimeError, match="EXIT_ALREADY_PENDING"):
        s.om.submit_signal(exit_signal("DUP", "BUY", 60), account())

    s.broker.fill_order(exit_record.order_id, 60, 91.0)
    s.om.refresh_open_orders()
    assert s.managed().closed is True
    expected = (100.0 - 95.0) * 40 + (100.0 - 91.0) * 60
    assert s.managed().realized_pnl == pytest.approx(expected)
    assert s.db.get_trades()[0]["realized_pnl"] == pytest.approx(expected)
    assert s.broker.get_positions() == []


@pytest.mark.parametrize("exit_price, expected", [(97.0, 300.0), (103.0, -300.0)])
def test_short_pnl_formula_is_entry_minus_exit(tmp_path, exit_price, expected):
    s = Stack(tmp_path)
    s.om.submit_signal(entry("S1", side="SELL"), account())
    s.broker.set_ltp(SYMBOL, exit_price)
    record = s.om.submit_signal(exit_signal("X1", "BUY", 100, price=exit_price), account())
    assert record.realized_pnl == pytest.approx(expected)
    assert s.db.get_trades()[0]["realized_pnl"] == pytest.approx(expected)
    assert s.broker.realized_pnl[SYMBOL] == pytest.approx(expected)
    s.db.close()


def test_long_pnl_formula_is_exit_minus_entry(tmp_path):
    s = Stack(tmp_path)
    s.om.submit_signal(entry("L1"), account())
    s.broker.set_ltp(SYMBOL, 97.0)
    record = s.om.submit_signal(exit_signal("X1", "SELL", 100, price=97.0), account())
    assert record.realized_pnl == pytest.approx(-300.0)
    s.db.close()
