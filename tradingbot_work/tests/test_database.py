from pathlib import Path

from src.database import Database


def test_database_creates_schema(tmp_path):
    db = Database(tmp_path / "trading.db")
    assert db.counts() == {
        "orders": 0,
        "trades": 0,
        "positions": 0,
        "system_events": 0,
    }
    db.close()


def test_order_is_inserted_and_updated(tmp_path):
    db = Database(tmp_path / "trading.db")

    db.save_order({
        "order_id": "O1",
        "signal_id": "S1",
        "symbol": "RELIANCE",
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": 10,
        "status": "OPEN",
        "filled_quantity": 0,
        "average_fill_price": None,
        "requested_price": 100,
    })

    assert db.get_order("O1")["status"] == "OPEN"

    db.save_order({
        "order_id": "O1",
        "signal_id": "S1",
        "symbol": "RELIANCE",
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": 10,
        "status": "FILLED",
        "filled_quantity": 10,
        "average_fill_price": 100.2,
        "requested_price": 100,
    })

    row = db.get_order("O1")
    assert row["status"] == "FILLED"
    assert row["filled_quantity"] == 10
    assert row["average_fill_price"] == 100.2
    assert db.counts()["orders"] == 1
    db.close()


def test_trade_is_persisted(tmp_path):
    db = Database(tmp_path / "trading.db")
    db.save_trade({
        "trade_id": "T1",
        "order_id": "O1",
        "signal_id": "S1",
        "symbol": "RELIANCE",
        "side": "LONG",
        "quantity": 10,
        "entry_price": 100,
        "status": "OPEN",
        "opened_at": "2026-10-05T04:00:00+00:00",
    })

    rows = db.get_trades()
    assert len(rows) == 1
    assert rows[0]["trade_id"] == "T1"
    assert rows[0]["status"] == "OPEN"
    db.close()


def test_position_upsert_and_delete(tmp_path):
    db = Database(tmp_path / "trading.db")
    record = {
        "symbol": "RELIANCE",
        "side": "LONG",
        "quantity": 10,
        "average_entry": 100,
        "current_price": 102,
        "unrealized_pnl": 20,
    }
    db.upsert_position(record)
    assert db.get_positions()[0]["quantity"] == 10

    record["quantity"] = 5
    record["current_price"] = 101
    db.upsert_position(record)
    assert db.get_positions()[0]["quantity"] == 5
    assert db.get_positions()[0]["current_price"] == 101

    db.delete_position("RELIANCE")
    assert db.get_positions() == []
    db.close()


def test_system_event_contains_metadata(tmp_path):
    db = Database(tmp_path / "trading.db")
    event_id = db.add_event(
        "KILL_SWITCH_ACTIVATED",
        "CRITICAL",
        "Emergency stop activated",
        {"source": "operator", "reason": "manual"},
    )
    row = db.get_events()[0]
    assert row["event_id"] == event_id
    assert row["event_type"] == "KILL_SWITCH_ACTIVATED"
    assert '"source": "operator"' in row["metadata_json"]
    db.close()


def test_data_survives_database_reopen(tmp_path):
    path = tmp_path / "trading.db"
    db = Database(path)
    db.add_event("STARTUP", "INFO", "System started")
    db.close()

    reopened = Database(path)
    assert reopened.counts()["system_events"] == 1
    assert reopened.get_events()[0]["message"] == "System started"
    reopened.close()


def test_multiple_records_and_limit(tmp_path):
    db = Database(tmp_path / "trading.db")
    for i in range(5):
        db.add_event("TEST", "INFO", f"event-{i}")

    assert len(db.get_events(limit=3)) == 3
    assert db.counts()["system_events"] == 5
    db.close()
