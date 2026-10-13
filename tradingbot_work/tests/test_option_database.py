"""Option order persistence and OrderManager instrument metadata."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.database import Database
from src.execution.order_manager import OrderManager
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.risk.risk_manager import RiskManager

IST = ZoneInfo("Asia/Kolkata")
KEY = "NSE_FO|50917"
META = {"market_type": "INDEX_OPTION", "underlying": "NIFTY", "option_type": "CE", "strike": 23000.0,
        "expiry": "2026-10-06", "lot_size": 65, "instrument_key": KEY}
OPTION_COLUMNS = ("market_type", "underlying", "option_type", "strike", "expiry", "lot_size", "instrument_key")


class AlwaysOpenSession:
    def is_entry_allowed(self, ts):
        return True


def risk_manager():
    return RiskManager(SimpleNamespace(risk_per_trade_percent=0.5, max_daily_loss_inr=3000.0,
               max_daily_loss_percent=2.0, max_open_positions=3,
                                       max_trades_per_day=6, max_portfolio_exposure_percent=50.0))


def option_signal(quantity=195, symbol=KEY, signal_id="opt-1"):
    return Signal(signal_id=signal_id, timestamp=datetime(2026, 10, 1, 11, tzinfo=IST), symbol=symbol,
                  side=SignalSide.BUY, signal_type=SignalType.ENTRY, entry_price=120.0, quantity=quantity,
                  stop_loss=100.0, target=160.0, reason="ALL_ENTRY_CONDITIONS_MET", risk_status="ALLOWED")


def paper_broker():
    broker = PaperBroker()
    broker.set_ltp(KEY, 120.0)
    broker.set_next_execution(OrderStatus.OPEN)
    broker.connect()
    return broker


def test_migration_adds_option_columns_to_existing_database(tmp_path):
    path = tmp_path / "old.db"
    Database(path).close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO orders (order_id, signal_id, symbol, side, order_type, quantity, status, "
                 "filled_quantity, created_at, updated_at) VALUES ('EQ1','s','RELIANCE','BUY','MARKET',5,"
                 "'FILLED',5,'2026-10-01T00:00:00','2026-10-01T00:00:00')")
    conn.commit()
    conn.close()

    db = Database(path)
    columns = {row["name"] for row in db.connection.execute("PRAGMA table_info(orders)")}
    assert set(OPTION_COLUMNS) <= columns
    legacy = db.get_order("EQ1")
    assert legacy["symbol"] == "RELIANCE" and legacy["instrument_key"] is None
    db.close()
    Database(path).close()  # re-opening is idempotent


def test_equity_orders_default_to_equity_market_type(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_order({"order_id": "E1", "symbol": "RELIANCE", "side": "BUY", "order_type": "MARKET",
                   "quantity": 5, "status": "SUBMITTED"})
    row = db.get_order("E1")
    assert row["market_type"] == "EQUITY" and row["option_type"] is None and row["lot_size"] is None


def test_option_metadata_is_persisted_and_kept_on_updates(tmp_path):
    db = Database(tmp_path / "t.db")
    db.save_order({"order_id": "O1", "symbol": KEY, "side": "BUY", "order_type": "MARKET", "quantity": 195,
                   "status": "SUBMITTED", **META})
    db.save_order({"order_id": "O1", "symbol": KEY, "side": "BUY", "order_type": "MARKET", "quantity": 195,
                   "status": "CANCELLED"})
    row = db.get_order("O1")
    assert {c: row[c] for c in OPTION_COLUMNS} == META
    assert row["status"] == "CANCELLED"
    history = [h["status"] for h in db.get_order_history("O1")]
    assert history == ["SUBMITTED", "CANCELLED"]


def test_order_manager_persists_option_metadata(tmp_path):
    db = Database(tmp_path / "t.db")
    broker = paper_broker()
    om = OrderManager(broker, risk_manager(), AlwaysOpenSession(), database=db)
    record = om.submit_signal(option_signal(), AccountState(equity=1_000_000), instrument=META)
    assert record.instrument == META
    om.cancel(record.order_id)
    row = db.get_order(record.order_id)
    assert row["symbol"] == KEY and row["quantity"] == 195 and row["status"] == "CANCELLED"
    assert {c: row[c] for c in OPTION_COLUMNS} == META

    restored = OrderManager(paper_broker(), risk_manager(), AlwaysOpenSession(), database=db)
    restored.restore_from_database()
    assert restored.records[record.order_id].instrument == META


def test_order_manager_rejects_non_lot_multiple_and_key_mismatch(tmp_path):
    om = OrderManager(paper_broker(), risk_manager(), AlwaysOpenSession(), database=Database(tmp_path / "t.db"))
    with pytest.raises(RuntimeError, match="QUANTITY_NOT_LOT_MULTIPLE"):
        om.submit_signal(option_signal(quantity=100), AccountState(equity=1_000_000), instrument=META)
    with pytest.raises(RuntimeError, match="INSTRUMENT_KEY_MISMATCH"):
        om.submit_signal(option_signal(symbol="NSE_FO|1", signal_id="opt-2"), AccountState(equity=1_000_000),
                         instrument=META)
    assert om.records == {}


def test_duplicate_option_signal_is_refused(tmp_path):
    om = OrderManager(paper_broker(), risk_manager(), AlwaysOpenSession(), database=Database(tmp_path / "t.db"))
    om.submit_signal(option_signal(), AccountState(equity=1_000_000), instrument=META)
    with pytest.raises(RuntimeError, match="DUPLICATE_SIGNAL"):
        om.submit_signal(option_signal(), AccountState(equity=1_000_000), instrument=META)


def test_order_params_are_sent_only_to_brokers_that_accept_them():
    class ProductBroker(PaperBroker):
        def __init__(self):
            super().__init__()
            self.calls = []

        def place_order(self, symbol, side, quantity, order_type, price=None, tag=None, *, product="I"):
            self.calls.append({"product": product, "order_type": order_type})
            return super().place_order(symbol, side, quantity, order_type, price, tag)

        def get_order(self, order_id):
            return self.orders[order_id]

    broker = ProductBroker()
    broker.set_ltp(KEY, 120.0)
    broker.set_next_execution(OrderStatus.OPEN)
    broker.connect()
    om = OrderManager(broker, risk_manager(), AlwaysOpenSession(), order_type="LIMIT",
                      order_params={"product": "D", "unsupported": 1})
    om.submit_signal(option_signal(), AccountState(equity=1_000_000), instrument=META)
    assert broker.calls == [{"product": "D", "order_type": "LIMIT"}]

    plain = paper_broker()
    OrderManager(plain, risk_manager(), AlwaysOpenSession(), order_params={"product": "D"}).submit_signal(
        option_signal(signal_id="opt-3"), AccountState(equity=1_000_000))
    assert len(plain.orders) == 1


def test_order_manager_rejects_unknown_order_type():
    with pytest.raises(ValueError):
        OrderManager(paper_broker(), risk_manager(), AlwaysOpenSession(), order_type="SL")


def test_equity_order_manager_defaults_are_unchanged(tmp_path):
    db = Database(tmp_path / "t.db")
    broker = PaperBroker()
    broker.set_ltp("RELIANCE", 100.0)
    broker.connect()
    om = OrderManager(broker, risk_manager(), AlwaysOpenSession(), database=db)
    signal = Signal(signal_id="eq-1", timestamp=datetime(2026, 10, 1, 11, tzinfo=IST), symbol="RELIANCE",
                    side=SignalSide.BUY, signal_type=SignalType.ENTRY, entry_price=100.0, quantity=5,
                    stop_loss=98.0, target=104.0, reason="x", risk_status="ALLOWED")
    record = om.submit_signal(signal, AccountState(equity=100_000))
    assert record.order_type == "MARKET" and record.instrument is None
    assert db.get_order(record.order_id)["market_type"] == "EQUITY"
