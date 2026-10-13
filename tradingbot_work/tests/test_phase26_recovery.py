from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.database import Database
from src.execution.order_manager import OrderManager
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.recovery import RecoveryManager
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager
from src.config.loader import load_strategy_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_strategy_config(ROOT / "config" / "strategy.yaml")


def make_entry(ts, symbol="DEMO", qty=500, price=100.0):
    return Signal(
        signal_id=f"REC-{symbol}-{ts.timestamp()}", timestamp=ts,
        symbol=symbol, side=SignalSide.BUY, signal_type=SignalType.ENTRY,
        entry_price=price, quantity=qty, stop_loss=99.0, target=102.0,
        reason="RECOVERY_TEST", risk_status="ALLOWED",
    )


def test_restart_reconstructs_positions_from_broker(tmp_path):
    db_path = tmp_path / "restart.db"
    db = Database(db_path)
    broker = PaperBroker(); broker.connect(); broker.set_ltp("DEMO", 100)
    manager1 = OrderManager(broker, RiskManager(CONFIG.risk), SessionManager(CONFIG.session), database=db)
    ts = datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    manager1.submit_signal(make_entry(ts), AccountState(equity=100000))

    # Simulate a process restart: a new manager has no in-memory state.
    manager2 = OrderManager(broker, RiskManager(CONFIG.risk), SessionManager(CONFIG.session), database=db)
    report = RecoveryManager(broker, db, manager2).recover()
    assert report.ok
    assert report.reconstructed_positions == {"DEMO": 500}
    assert manager2.reconcile()["ok"]
    db.close()


def test_broker_disconnect_blocks_recovery(tmp_path):
    db = Database(tmp_path / "disconnect.db")
    broker = PaperBroker(); manager = OrderManager(broker, RiskManager(CONFIG.risk), SessionManager(CONFIG.session), database=db)
    with pytest.raises(RuntimeError, match="BROKER_DISCONNECTED_DURING_RECOVERY"):
        RecoveryManager(broker, db, manager).recover()
    db.close()


def test_unknown_open_order_is_not_retried(tmp_path):
    db = Database(tmp_path / "unknown.db")
    broker = PaperBroker(); broker.connect(); broker.set_ltp("DEMO", 100)
    broker.set_next_execution(OrderStatus.OPEN)
    order_id = broker.place_order("DEMO", "BUY", 5, "MARKET")
    manager = OrderManager(broker, RiskManager(CONFIG.risk), SessionManager(CONFIG.session), database=db)
    report = RecoveryManager(broker, db, manager).recover()
    assert order_id in report.order_mismatches
    assert len(broker.get_orders()) == 1
    assert any(row["event_type"] == "RECOVERY_UNKNOWN_ORDERS" for row in db.get_events())
    db.close()


def test_persisted_position_mismatch_is_reported(tmp_path):
    db = Database(tmp_path / "mismatch.db")
    db.upsert_position({"symbol":"DEMO", "side":"BUY", "quantity":10, "average_entry":100.0})
    broker = PaperBroker(); broker.connect(); broker.set_ltp("DEMO", 100)
    manager = OrderManager(broker, RiskManager(CONFIG.risk), SessionManager(CONFIG.session), database=db)
    report = RecoveryManager(broker, db, manager).recover()
    assert report.position_mismatches["DEMO"]["persisted"] == 10
    assert report.position_mismatches["DEMO"]["broker"] == 0
    assert not report.ok
    db.close()
