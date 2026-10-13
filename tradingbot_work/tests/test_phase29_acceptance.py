from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from src.api.app import app
from src.broker.paper import PaperBroker
from src.config.loader import load_strategy_config
from src.database import Database
from src.execution.order_manager import OrderManager
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.risk.risk_manager import RiskManager
from src.safety.kill_switch import KillSwitch
from src.safety.live_gate import assert_startup_safe, startup_safety_snapshot
from src.session.session_manager import SessionManager


def _signal(signal_id, kind, side, price, qty=100):
    return Signal(
        signal_id=signal_id,
        timestamp=datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        symbol="DEMO",
        side=side,
        signal_type=kind,
        entry_price=price,
        quantity=qty,
        stop_loss=price - 1 if kind == SignalType.ENTRY and side == SignalSide.BUY else None,
        target=price + 2 if kind == SignalType.ENTRY and side == SignalSide.BUY else None,
        reason="PHASE29_ACCEPTANCE",
        risk_status="ALLOWED",
    )


def test_phase29_pre_live_acceptance(tmp_path):
    cfg = load_strategy_config("config/strategy.yaml")
    assert cfg.system.trading_mode == "PAPER"
    assert_startup_safe()
    snap = startup_safety_snapshot()
    assert snap["live_order_submission_allowed"] is False

    db = Database(tmp_path / "acceptance.db")
    broker = PaperBroker(initial_cash=100_000)
    broker.connect()
    broker.set_ltp("DEMO", 100.0)
    manager = OrderManager(
        broker,
        RiskManager(cfg.risk),
        SessionManager(cfg.session),
        database=db,
    )

    entry = _signal("P29-ENTRY", SignalType.ENTRY, SignalSide.BUY, 100.0, qty=500)
    result = manager.submit_signal(entry, AccountState(equity=100_000))
    assert result.status.value == "FILLED"
    assert broker.get_positions()[0]["quantity"] == 500
    assert db.counts()["trades"] == 1

    broker.set_ltp("DEMO", 102.0)
    exit_signal = _signal("P29-EXIT", SignalType.EXIT, SignalSide.SELL, 102.0, qty=500)
    exit_result = manager.submit_signal(exit_signal, AccountState(equity=100_000, open_positions=1))
    assert exit_result.status.value == "FILLED"
    assert broker.get_positions() == []
    trade = db.get_trades(10)[0]
    assert trade["status"] == "CLOSED"
    assert float(trade["realized_pnl"]) == 1000.0

    ks = KillSwitch()
    ks.activate("PHASE29_ACCEPTANCE")
    assert ks.is_halted is True
    assert ks.allow_entry() is False
    db.close()


def test_phase29_dashboard_exposes_paper_state():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/orders").status_code == 200
    assert client.get("/api/trades").status_code == 200
    assert client.get("/api/positions").status_code == 200
    assert client.get("/api/events").status_code == 200
