from datetime import datetime, timezone

import pytest

from src.broker.models import OrderSide, OrderStatus, OrderType, BrokerOrder
from src.execution.order_manager import ExecutionRecord, OrderManager
from src.integration.upstox_e2e import UpstoxSandboxE2EHarness
from src.broker.upstox_portfolio import OrderUpdate, PositionUpdate


class Dummy:
    pass


def make_manager():
    om = OrderManager(Dummy(), Dummy(), Dummy())
    om.records["OID1"] = ExecutionRecord(
        signal_id="SIG1", order_id="OID1", symbol="NSE_EQ|ABC", signal_type="ENTRY",
        status=OrderStatus.SUBMITTED, requested_quantity=10, filled_quantity=0,
        average_fill_price=None, created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    om.signal_to_order["SIG1"] = "OID1"
    return om


def test_order_stream_lifecycle_updates_without_resubmission():
    om = make_manager()
    r = om.apply_order_update(OrderUpdate("OID1", "open", "NSE_EQ|ABC", "BUY", 10, 0, 10, 0))
    assert r.status == OrderStatus.OPEN
    r = om.apply_order_update(OrderUpdate("OID1", "partially filled", "NSE_EQ|ABC", "BUY", 10, 4, 6, 100))
    assert r.status == OrderStatus.PARTIALLY_FILLED and r.filled_quantity == 4
    r = om.apply_order_update(OrderUpdate("OID1", "complete", "NSE_EQ|ABC", "BUY", 10, 10, 0, 101))
    assert r.status == OrderStatus.FILLED and r.filled_quantity == 10


def test_out_of_order_fill_is_ignored():
    om = make_manager()
    om.apply_order_update(OrderUpdate("OID1", "complete", "NSE_EQ|ABC", "BUY", 10, 10, 0, 101))
    r = om.apply_order_update(OrderUpdate("OID1", "partial", "NSE_EQ|ABC", "BUY", 10, 4, 6, 100))
    assert r.status == OrderStatus.FILLED and r.filled_quantity == 10


def test_filled_is_not_overwritten_by_late_cancel():
    om = make_manager()
    om.apply_order_update(OrderUpdate("OID1", "complete", "NSE_EQ|ABC", "BUY", 10, 10, 0, 101))
    r = om.apply_order_update(OrderUpdate("OID1", "cancelled", "NSE_EQ|ABC", "BUY", 10, 10, 0, 101))
    assert r.status == OrderStatus.FILLED


def test_unknown_order_fails_closed():
    om = make_manager()
    with pytest.raises(KeyError, match="UNKNOWN_BROKER_ORDER"):
        om.apply_order_update(OrderUpdate("UNKNOWN", "complete", "NSE_EQ|ABC", "BUY", 1, 1, 0, 100))


def test_position_stream_reconciliation():
    om = make_manager()
    om.expected_positions["NSE_EQ|ABC"] = 10
    result = om.apply_position_update(PositionUpdate("NSE_EQ|ABC", "NSE_EQ|ABC", 10, 101))
    assert result["mismatch"] is False
    result = om.apply_position_update(PositionUpdate("NSE_EQ|ABC", "NSE_EQ|ABC", 9, 101))
    assert result["mismatch"] is True


def test_e2e_harness_normalizes_events():
    harness = UpstoxSandboxE2EHarness()
    event = OrderUpdate("OID1", "complete", "ABC", "BUY", 1, 1, 0, 100)
    assert harness.normalize_event(event)["status"] == "complete"
