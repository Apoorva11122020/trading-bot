import time

import pytest

from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.execution.poller import OrderPoller

from trading_support import SYMBOL, Stack, account, entry


class CountingBroker(PaperBroker):
    def __init__(self):
        super().__init__()
        self.get_orders_calls = 0
        self.place_calls = 0
        self.failures_remaining = 0

    def get_orders(self):
        self.get_orders_calls += 1
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ConnectionError("broker temporarily unavailable")
        return super().get_orders()

    def place_order(self, *args, **kwargs):
        self.place_calls += 1
        return super().place_order(*args, **kwargs)


@pytest.fixture
def s(tmp_path):
    broker = CountingBroker()
    broker.connect()
    broker.set_ltp(SYMBOL, 100.0)
    stack = Stack(tmp_path, broker=broker)
    yield stack
    stack.db.close()


def _open_order(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    return s.lifecycle.submit_entry(entry("E1"))


def test_poll_applies_fills_through_order_manager(s):
    rec = _open_order(s)
    poller = OrderPoller(s.om, interval_seconds=1)
    s.broker.fill_order(rec.order_id, 100, 100.0)
    result = poller.poll_once()
    assert result["status"] == "ok"
    assert rec.order_id in result["refreshed"]
    assert rec.status == OrderStatus.FILLED
    assert s.managed().position.quantity == 100


def test_poll_stops_querying_once_orders_are_terminal(s):
    rec = _open_order(s)
    s.broker.fill_order(rec.order_id, 100, 100.0)
    poller = OrderPoller(s.om, interval_seconds=1)
    poller.poll_once()
    calls = s.broker.get_orders_calls
    for _ in range(5):
        assert poller.poll_once()["status"] == "idle"
    assert s.broker.get_orders_calls == calls


def test_poll_failure_backs_off_with_bound_and_never_resubmits(s):
    _open_order(s)
    placed = s.broker.place_calls
    poller = OrderPoller(s.om, interval_seconds=1, max_backoff_seconds=8)
    s.broker.failures_remaining = 10
    delays = []
    for _ in range(5):
        assert poller.poll_once()["status"] == "error"
        delays.append(poller.next_delay())
    assert delays == [2, 4, 8, 8, 8]
    assert s.broker.place_calls == placed
    assert len(s.events("ORDER_POLL_FAILED")) == 5

    s.broker.failures_remaining = 0
    assert poller.poll_once()["status"] == "ok"
    assert poller.next_delay() == 1


def test_order_status_timeout_is_reported_once(s):
    _open_order(s)
    now = [0.0]
    poller = OrderPoller(s.om, interval_seconds=1, status_timeout_seconds=60, clock=lambda: now[0])
    poller.poll_once()
    now[0] = 61.0
    assert poller.poll_once()["timed_out"]
    now[0] = 120.0
    assert poller.poll_once()["timed_out"] == []
    assert len(s.events("ORDER_STATUS_TIMEOUT")) == 1


def test_background_thread_starts_and_stops_cleanly(s):
    rec = _open_order(s)
    poller = OrderPoller(s.om, interval_seconds=0.01)
    poller.start()
    try:
        s.broker.fill_order(rec.order_id, 100, 100.0)
        deadline = time.time() + 5
        while rec.status != OrderStatus.FILLED and time.time() < deadline:
            time.sleep(0.01)
        assert rec.status == OrderStatus.FILLED
    finally:
        poller.stop()
    assert poller.running is False


def test_invalid_intervals_are_rejected(s):
    with pytest.raises(ValueError):
        OrderPoller(s.om, interval_seconds=0)
