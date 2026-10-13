import pytest

from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker


@pytest.fixture
def broker():
    b = PaperBroker()
    b.connect()
    b.set_ltp("X", 100.0)
    return b


def _order(broker, oid):
    return broker.orders[oid]


def test_market_order_fills_at_ltp(broker):
    oid = broker.place_order("X", "BUY", 10, "MARKET")
    order = _order(broker, oid)
    assert order.status == OrderStatus.FILLED
    assert order.average_fill_price == 100.0


def test_non_marketable_limit_rests_until_price_reaches_it(broker):
    oid = broker.place_order("X", "BUY", 10, "LIMIT", price=99.0)
    assert _order(broker, oid).status == OrderStatus.OPEN
    assert broker.get_positions() == []
    broker.set_ltp("X", 99.5)
    assert _order(broker, oid).status == OrderStatus.OPEN
    broker.set_ltp("X", 98.8)
    order = _order(broker, oid)
    assert order.status == OrderStatus.FILLED
    assert order.average_fill_price == 99.0
    assert broker.get_positions()[0]["quantity"] == 10


def test_marketable_sell_limit_fills_immediately(broker):
    oid = broker.place_order("X", "SELL", 5, "LIMIT", price=99.0)
    assert _order(broker, oid).status == OrderStatus.FILLED
    assert broker.get_positions()[0]["quantity"] == -5


def test_controlled_partial_fills_weighted_average(broker):
    broker.set_next_execution(OrderStatus.OPEN)
    oid = broker.place_order("X", "BUY", 100, "MARKET", tag="T1")
    broker.fill_order(oid, 40, 100.0)
    order = _order(broker, oid)
    assert order.status == OrderStatus.PARTIALLY_FILLED and order.tag == "T1"
    broker.fill_order(oid, 60, 101.0)
    order = _order(broker, oid)
    assert order.status == OrderStatus.FILLED
    assert order.average_fill_price == pytest.approx(100.6)
    assert broker.get_positions()[0]["average_price"] == pytest.approx(100.6)
    with pytest.raises(RuntimeError):
        broker.fill_order(oid, 1)


def test_overfill_is_rejected(broker):
    broker.set_next_execution(OrderStatus.OPEN)
    oid = broker.place_order("X", "BUY", 10, "MARKET")
    with pytest.raises(ValueError):
        broker.fill_order(oid, 11)


def test_reject_and_cancel(broker):
    broker.set_next_execution(OrderStatus.OPEN)
    oid = broker.place_order("X", "BUY", 10, "MARKET")
    assert broker.reject_order(oid, "RMS").status == OrderStatus.REJECTED
    broker.set_next_execution(OrderStatus.OPEN)
    oid2 = broker.place_order("X", "BUY", 10, "MARKET")
    broker.fill_order(oid2, 4)
    broker.cancel_order(oid2)
    order = _order(broker, oid2)
    assert order.status == OrderStatus.CANCELLED and order.filled_quantity == 4
    with pytest.raises(RuntimeError):
        broker.reject_order(oid2, "late")


def test_realized_pnl_long_and_short(broker):
    broker.place_order("X", "BUY", 10, "MARKET")
    broker.set_ltp("X", 105.0)
    broker.place_order("X", "SELL", 10, "MARKET")
    assert broker.realized_pnl["X"] == pytest.approx(50.0)
    assert broker.get_positions() == []

    broker.place_order("X", "SELL", 10, "MARKET")
    broker.set_ltp("X", 102.0)
    broker.place_order("X", "BUY", 4, "MARKET")
    assert broker.realized_pnl["X"] == pytest.approx(50.0 + 12.0)
    assert broker.get_positions()[0]["quantity"] == -6
    assert broker.get_account()["realized_pnl"] == pytest.approx(62.0)


def test_rejected_order_does_not_move_position(broker):
    broker.set_next_execution(OrderStatus.REJECTED, rejection_reason="MARGIN")
    oid = broker.place_order("X", "BUY", 10, "MARKET")
    assert _order(broker, oid).filled_quantity == 0
    assert broker.get_positions() == []


def test_invalid_orders(broker):
    with pytest.raises(ValueError):
        broker.place_order("X", "BUY", 0, "MARKET")
    with pytest.raises(ValueError):
        broker.place_order("X", "BUY", 1, "LIMIT")
    with pytest.raises(KeyError):
        broker.place_order("UNKNOWN", "BUY", 1, "MARKET")
