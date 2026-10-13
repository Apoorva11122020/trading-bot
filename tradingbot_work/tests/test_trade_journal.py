from datetime import datetime
from zoneinfo import ZoneInfo

from src.broker.paper import PaperBroker
from src.database import Database
from src.execution.order_manager import OrderManager
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager
from src.config.loader import load_strategy_config


def _manager(tmp_path):
    cfg = load_strategy_config("config/strategy.yaml")
    broker = PaperBroker()
    broker.connect()
    db = Database(tmp_path / "journal.db")
    return broker, db, OrderManager(broker, RiskManager(cfg.risk), SessionManager(cfg.session), database=db)


def test_filled_entry_and_exit_create_closed_trade(tmp_path):
    broker, db, om = _manager(tmp_path)
    broker.set_ltp("RELIANCE", 100.0)
    ts = datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))

    entry = Signal(signal_id="J-ENTRY", timestamp=ts, symbol="RELIANCE", side=SignalSide.BUY,
                   signal_type=SignalType.ENTRY, entry_price=100, quantity=500,
                   stop_loss=99, target=102, reason="TEST", risk_status="ALLOWED")
    om.submit_signal(entry, AccountState(equity=100000))

    broker.set_ltp("RELIANCE", 102.0)
    exit_signal = Signal(signal_id="J-EXIT", timestamp=ts, symbol="RELIANCE", side=SignalSide.SELL,
                         signal_type=SignalType.EXIT, entry_price=102, quantity=500,
                         reason="TARGET", risk_status="ALLOWED")
    om.submit_signal(exit_signal, AccountState(equity=100000, open_positions=1))

    trades = db.get_trades()
    assert len(trades) == 1
    assert trades[0]["status"] == "CLOSED"
    assert trades[0]["entry_price"] == 100.0
    assert trades[0]["exit_price"] == 102.0
    assert trades[0]["realized_pnl"] == 1000.0
    db.close()
