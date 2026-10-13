import sqlite3

import pytest
import requests

from src.broker.models import OrderStatus
from src.broker.upstox_sandbox import UpstoxSandboxBroker, UpstoxSandboxHTTPError
from src.database import Database
from src.execution.order_manager import OrderInDoubtError, OrderManager, client_tag
from src.execution.poller import OrderPoller
from src.models.risk import AccountState
from src.position.lifecycle import PositionLifecycle
from src.position.monitor import PositionMonitor
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager

from trading_support import CFG, entry, exit_signal, ts

SANDBOX = "https://api-sandbox.upstox.com"
INSTRUMENT = "NSE_EQ|INE669E01016"


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload


class FlakySandbox:
    """Controlled stand-in for the Upstox Sandbox HTTP API."""

    def __init__(self):
        self.orders = {}
        self.posts = 0
        self.place_mode = "ok"
        self.book_down = False
        self._seq = 0

    def add_order(self, payload):
        self._seq += 1
        oid = f"SBX-{self._seq}"
        self.orders[oid] = {
            "order_id": oid,
            "instrument_token": payload["instrument_token"],
            "transaction_type": payload["transaction_type"],
            "order_type": payload["order_type"],
            "quantity": payload["quantity"],
            "price": payload["price"],
            "filled_quantity": 0,
            "average_price": 0.0,
            "status": "open",
            "tag": payload.get("tag"),
            "order_timestamp": "2026-10-01 10:00:00",
        }
        return oid

    def __call__(self, method, url, headers=None, json=None, params=None, timeout=None):
        assert url.startswith(SANDBOX + "/"), url
        path = url[len(SANDBOX):]
        if method == "POST" and path == "/v3/order/place":
            self.posts += 1
            mode = self.place_mode
            if mode == "timeout_before_place":
                raise requests.exceptions.Timeout("read timed out")
            if mode == "http503":
                return _Resp({"status": "error"}, 503)
            if mode == "http400":
                return _Resp({"status": "error", "errors": [{"message": "invalid"}]}, 400)
            oid = self.add_order(json)
            if mode == "timeout_after_place":
                raise requests.exceptions.Timeout("read timed out")
            return _Resp({"status": "success", "data": {"order_ids": [oid]}})
        if method == "GET" and path == "/v2/order/retrieve-all":
            if self.book_down:
                raise requests.exceptions.ConnectionError("order book unavailable")
            return _Resp({"status": "success", "data": list(self.orders.values())})
        if method == "GET" and path == "/v3/portfolio/short-term-positions":
            return _Resp({"status": "success", "data": []})
        raise AssertionError(f"unexpected call {method} {path}")

    def fill(self, oid, price):
        self.orders[oid].update(status="complete", filled_quantity=self.orders[oid]["quantity"], average_price=price)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    sandbox = FlakySandbox()
    monkeypatch.setattr("requests.request", sandbox)
    broker = UpstoxSandboxBroker("test-token")
    broker.connect()
    db = Database(tmp_path / "failures.db")
    session = SessionManager(CFG.session)
    om = OrderManager(broker, RiskManager(CFG.risk), session, database=db)
    yield sandbox, broker, db, om
    db.close()


def _entry(signal_id="E1"):
    return entry(signal_id, symbol=INSTRUMENT)


def acct():
    return AccountState(equity=100_000)


def test_timeout_after_broker_accepted_is_adopted_by_tag_not_resubmitted(env):
    sandbox, _, db, om = env
    sandbox.place_mode = "timeout_after_place"
    record = om.submit_signal(_entry(), acct())
    assert record.status == OrderStatus.OPEN
    assert record.tag == client_tag("E1")
    assert sandbox.posts == 1 and len(sandbox.orders) == 1
    assert db.get_events(event_type="ORDER_IN_DOUBT_RESOLVED")


def test_timeout_without_order_is_in_doubt_and_blocks_resubmission(env):
    sandbox, _, db, om = env
    sandbox.place_mode = "timeout_before_place"
    with pytest.raises(OrderInDoubtError, match="ORDER_IN_DOUBT"):
        om.submit_signal(_entry(), acct())
    assert "E1" in om.in_doubt
    assert db.get_events(event_type="ORDER_IN_DOUBT")

    sandbox.place_mode = "ok"
    with pytest.raises(OrderInDoubtError, match="UNRESOLVED"):
        om.submit_signal(_entry(), acct())
    with pytest.raises(RuntimeError, match="ENTRY_IN_DOUBT"):
        om.submit_signal(_entry("E2"), acct())
    assert sandbox.posts == 1

    # The order appears at the broker late (e.g. after exchange delay):
    # polling adopts it by client tag; nothing is resubmitted.
    late = sandbox.add_order({"instrument_token": INSTRUMENT, "transaction_type": "BUY",
                              "order_type": "MARKET", "quantity": 100, "price": 0.0,
                              "tag": client_tag("E1")})
    OrderPoller(om).poll_once()
    assert late in om.records
    assert om.signal_to_order["E1"] == late
    assert om.in_doubt == {}
    assert sandbox.posts == 1


def test_operator_release_requires_broker_to_have_no_tagged_order(env):
    sandbox, _, _, om = env
    sandbox.place_mode = "timeout_before_place"
    with pytest.raises(OrderInDoubtError):
        om.submit_signal(_entry(), acct())
    assert om.release_in_doubt("E1") is True
    sandbox.place_mode = "ok"
    record = om.submit_signal(_entry(), acct())
    assert record.status == OrderStatus.OPEN
    assert sandbox.posts == 2 and len(sandbox.orders) == 1


def test_http_5xx_is_treated_as_in_doubt(env):
    sandbox, _, _, om = env
    sandbox.place_mode = "http503"
    with pytest.raises(OrderInDoubtError):
        om.submit_signal(_entry(), acct())
    assert "E1" in om.in_doubt


def test_http_4xx_is_a_definite_rejection(env):
    sandbox, _, db, om = env
    sandbox.place_mode = "http400"
    with pytest.raises(UpstoxSandboxHTTPError):
        om.submit_signal(_entry(), acct())
    assert om.in_doubt == {}
    assert db.get_events(event_type="ORDER_SUBMIT_FAILED")


def test_status_unavailable_after_placement_is_reconciled_by_polling(env):
    sandbox, _, db, om = env
    sandbox.book_down = True
    record = om.submit_signal(_entry(), acct())
    assert record.status == OrderStatus.SUBMITTED
    assert db.get_events(event_type="ORDER_STATUS_UNAVAILABLE")

    poller = OrderPoller(om, interval_seconds=1)
    assert poller.poll_once()["status"] == "error"
    sandbox.book_down = False
    sandbox.fill(record.order_id, 100.2)
    assert poller.poll_once()["status"] == "ok"
    assert record.status == OrderStatus.FILLED
    assert om.expected_positions == {INSTRUMENT: 100}
    assert sandbox.posts == 1


def test_in_doubt_exit_blocks_duplicate_exit(env, monkeypatch):
    sandbox, _, _, om = env
    record = om.submit_signal(_entry(), acct())
    sandbox.fill(record.order_id, 100.0)
    om.refresh_order(record.order_id)

    sandbox.place_mode = "timeout_before_place"
    with pytest.raises(OrderInDoubtError):
        om.submit_signal(exit_signal("X1", "SELL", 100, symbol=INSTRUMENT), acct())
    sandbox.place_mode = "ok"
    with pytest.raises(RuntimeError, match="EXIT_IN_DOUBT"):
        om.submit_signal(exit_signal("X2", "SELL", 100, symbol=INSTRUMENT), acct())
    sells = [o for o in sandbox.orders.values() if o["transaction_type"] == "SELL"]
    assert sells == []


def test_lifecycle_exit_retry_is_blocked_while_exit_in_doubt(env):
    sandbox, _, _, om = env
    monitor = PositionMonitor(CFG, om.session_manager)
    lifecycle = PositionLifecycle(om, monitor, acct)
    record = lifecycle.submit_entry(_entry())
    sandbox.fill(record.order_id, 100.0)
    lifecycle.refresh(INSTRUMENT)

    sandbox.place_mode = "timeout_before_place"
    with pytest.raises(OrderInDoubtError):
        lifecycle.on_price(INSTRUMENT, 94.0, 1.0, ts(11, 0))
    sandbox.place_mode = "ok"
    with pytest.raises(RuntimeError, match="EXIT_IN_DOUBT"):
        lifecycle.on_price(INSTRUMENT, 93.0, 1.0, ts(11, 1))
    assert [o for o in sandbox.orders.values() if o["transaction_type"] == "SELL"] == []


def test_database_failure_rolls_back_memory_so_fill_is_not_lost(env, monkeypatch):
    sandbox, _, db, om = env
    record = om.submit_signal(_entry(), acct())
    sandbox.fill(record.order_id, 100.0)

    original = db.save_execution

    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "save_execution", broken)
    poller = OrderPoller(om, interval_seconds=1)
    assert poller.poll_once()["status"] == "error"
    assert record.status == OrderStatus.OPEN
    assert record.filled_quantity == 0
    assert om.expected_positions == {}

    monkeypatch.setattr(db, "save_execution", original)
    assert poller.poll_once()["status"] == "ok"
    assert record.status == OrderStatus.FILLED
    assert om.expected_positions == {INSTRUMENT: 100}
    assert db.get_trade(record.order_id)["quantity"] == 100


def test_disconnected_broker_blocks_new_orders(env):
    _, broker, _, om = env
    broker.disconnect()
    with pytest.raises(RuntimeError, match="BROKER_DISCONNECTED"):
        om.submit_signal(_entry(), acct())
