import sqlite3

import pytest

from src.database import Database


def _order(**kw):
    base = {"order_id": "O1", "symbol": "DEMO", "side": "BUY", "order_type": "MARKET",
            "quantity": 10, "status": "OPEN"}
    return base | kw


def test_transaction_rolls_back_all_writes(tmp_path):
    db = Database(tmp_path / "tx.db")
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.save_order(_order())
            db.add_event("X", "INFO", "inside")
            raise RuntimeError("boom")
    assert db.get_order("O1") is None
    assert db.get_events(event_type="X") == []
    db.close()


def test_nested_transaction_commits_once(tmp_path):
    db = Database(tmp_path / "tx.db")
    with db.transaction():
        db.save_order(_order())
        with db.transaction():
            db.save_order(_order(order_id="O2"))
    reopened = Database(tmp_path / "tx.db")
    assert reopened.get_order("O1") and reopened.get_order("O2")
    db.close()
    reopened.close()


def test_execution_ids_make_fills_idempotent(tmp_path):
    db = Database(tmp_path / "exec.db")
    fill = {"order_id": "O1", "symbol": "DEMO", "side": "BUY", "quantity": 4, "price": 100.0,
            "cumulative_filled": 4}
    assert db.save_execution(fill) is True
    assert db.save_execution(fill) is False
    assert len(db.get_executions("O1")) == 1
    db.close()


def test_order_status_history_only_on_change(tmp_path):
    db = Database(tmp_path / "hist.db")
    db.save_order(_order())
    db.save_order(_order())
    db.save_order(_order(status="PARTIALLY_FILLED", filled_quantity=4))
    db.save_order(_order(status="FILLED", filled_quantity=10))
    assert [r["status"] for r in db.get_order_history("O1")] == ["OPEN", "PARTIALLY_FILLED", "FILLED"]
    assert [r["order_id"] for r in db.get_open_orders()] == []
    db.close()


def test_trade_upsert_never_duplicates(tmp_path):
    db = Database(tmp_path / "trade.db")
    trade = {"trade_id": "T1", "order_id": "O1", "symbol": "DEMO", "side": "BUY",
             "quantity": 10, "entry_price": 100.0, "opened_at": "2026-10-01T04:30:00+00:00",
             "status": "OPEN"}
    db.save_trade(trade)
    db.save_trade(trade | {"quantity": 20})
    rows = db.get_trades()
    assert len(rows) == 1 and rows[0]["quantity"] == 20
    db.close()


def test_old_database_files_are_migrated(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE orders (order_id TEXT PRIMARY KEY, signal_id TEXT, symbol TEXT NOT NULL,
            side TEXT NOT NULL, order_type TEXT NOT NULL, quantity INTEGER NOT NULL,
            status TEXT NOT NULL, filled_quantity INTEGER NOT NULL DEFAULT 0,
            average_fill_price REAL, requested_price REAL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, reason TEXT);
        CREATE TABLE trades (trade_id TEXT PRIMARY KEY, order_id TEXT NOT NULL, signal_id TEXT,
            symbol TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL, entry_price REAL,
            exit_price REAL, realized_pnl REAL, opened_at TEXT, closed_at TEXT, status TEXT NOT NULL);
        INSERT INTO orders VALUES ('OLD','S','DEMO','BUY','MARKET',1,'FILLED',1,100,100,
            '2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00',NULL);
    """)
    conn.commit()
    conn.close()

    db = Database(path)
    columns = {r["name"] for r in db.connection.execute("PRAGMA table_info(orders)")}
    assert {"tag", "signal_type", "slippage_pct", "realized_pnl"} <= columns
    assert db.get_order("OLD")["status"] == "FILLED"
    db.close()


def test_timestamps_are_stored_in_utc(tmp_path):
    db = Database(tmp_path / "tz.db")
    db.save_order(_order())
    assert db.get_order("O1")["updated_at"].endswith("+00:00")
    db.close()


def test_system_state_round_trip(tmp_path):
    db = Database(tmp_path / "state.db")
    db.set_state("k", {"a": 1})
    assert db.get_state("k") == {"a": 1}
    assert db.get_state("missing", "default") == "default"
    db.close()
