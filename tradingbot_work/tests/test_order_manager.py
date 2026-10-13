from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.config.loader import load_strategy_config
from src.execution.order_manager import OrderManager
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager


def manager():
    cfg = load_strategy_config("config/strategy.yaml")
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("RELIANCE", 100.0)
    om = OrderManager(
        broker,
        RiskManager(cfg.risk),
        SessionManager(cfg.session),
        max_slippage_percent=0.10,
    )
    return broker, om


def signal(signal_id="sig-1", side=SignalSide.BUY, signal_type=SignalType.ENTRY):
    return Signal(
        signal_id=signal_id,
        timestamp=datetime(2026, 10, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        symbol="RELIANCE",
        side=side,
        signal_type=signal_type,
        entry_price=100.0,
        quantity=500,
        stop_loss=99.0 if side == SignalSide.BUY else 101.0,
        target=102.0 if side == SignalSide.BUY else 98.0,
        reason="TEST",
        risk_status="ALLOWED",
    )


def account():
    return AccountState(equity=100_000)


def test_filled_order_is_tracked_and_position_reconciles():
    broker, om = manager()
    record = om.submit_signal(signal(), account())

    assert record.status == OrderStatus.FILLED
    assert record.filled_quantity == 500
    reconciliation = om.reconcile()
    assert reconciliation["ok"] is True
    assert reconciliation["broker_positions"]["RELIANCE"] == 500


def test_duplicate_signal_is_blocked():
    _, om = manager()
    om.submit_signal(signal(), account())
    with pytest.raises(RuntimeError, match="DUPLICATE_SIGNAL"):
        om.submit_signal(signal(), account())


def test_blocked_signal_never_reaches_broker():
    broker, om = manager()
    s = signal()
    s.blocked = True
    s.block_reason = "DAILY_LOSS_LIMIT_REACHED"
    with pytest.raises(RuntimeError, match="SIGNAL_BLOCKED"):
        om.submit_signal(s, account())
    assert broker.get_orders() == []


def test_rejected_order_is_tracked():
    broker, om = manager()
    broker.set_next_execution(
        OrderStatus.REJECTED,
        rejection_reason="INSUFFICIENT_MARGIN",
    )
    record = om.submit_signal(signal(), account())
    assert record.status == OrderStatus.REJECTED
    assert record.rejection_reason == "INSUFFICIENT_MARGIN"
    assert om.reconcile()["ok"] is True


def test_partial_fill_is_tracked():
    broker, om = manager()
    broker.set_next_execution(OrderStatus.PARTIALLY_FILLED, filled_quantity=200)
    record = om.submit_signal(signal(), account())

    assert record.status == OrderStatus.PARTIALLY_FILLED
    assert record.filled_quantity == 200
    assert om.reconcile()["broker_positions"]["RELIANCE"] == 200


def test_cancel_and_modify_are_explicit():
    broker, om = manager()
    # Paper broker normally fills immediately, so use a direct pending order
    # to exercise the lifecycle methods.
    broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    record = om.submit_signal(signal(), account())
    modified = om.modify(record.order_id, price=99.5)
    assert modified.status == OrderStatus.MODIFIED

    # The paper broker permits cancelling a non-filled order.
    cancelled = om.cancel(record.order_id)
    assert cancelled.status == OrderStatus.CANCELLED


def test_disconnected_broker_blocks_submission():
    broker, om = manager()
    broker.disconnect()
    with pytest.raises(RuntimeError, match="BROKER_DISCONNECTED"):
        om.submit_signal(signal(), account())


def test_fresh_risk_check_blocks_when_daily_loss_limit_is_hit():
    _, om = manager()
    bad_account = AccountState(equity=100_000, daily_pnl=-3_001)
    with pytest.raises(RuntimeError, match="FRESH_RISK_CHECK"):
        om.submit_signal(signal(), bad_account)


def test_reconciliation_detects_position_mismatch():
    broker, om = manager()
    om.submit_signal(signal(), account())
    broker.positions["RELIANCE"]["quantity"] = 400
    report = om.reconcile()
    assert report["ok"] is False
    assert report["mismatches"]["RELIANCE"] == {
        "expected": 500,
        "actual": 400,
    }
