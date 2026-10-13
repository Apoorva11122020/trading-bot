"""Regression: orders submitted through OrderManager must reach the dashboard database."""
from __future__ import annotations

import importlib.util

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.database import Database
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager
from tests.trading_support import CFG, ROOT, SYMBOL, account, entry


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    db = Database(tmp_path / "trading.db")
    monkeypatch.setattr(api, "database", db)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DASHBOARD_API_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    yield TestClient(api.app), db
    db.close()


class _SandboxLikeBroker(PaperBroker):
    """PaperBroker plus the single-order lookup the Sandbox demo uses."""

    def get_order(self, order_id: str):
        return self.orders[order_id]


def _working_broker(symbol: str) -> PaperBroker:
    broker = _SandboxLikeBroker()
    broker.connect()
    broker.set_ltp(symbol, 100.0)
    broker.set_next_execution(OrderStatus.OPEN)
    return broker


def test_submitted_and_cancelled_order_is_visible_on_dashboard(api_client):
    client, db = api_client
    broker = _working_broker(SYMBOL)
    om = OrderManager(broker, RiskManager(CFG.risk), SessionManager(CFG.session),
                      max_slippage_percent=0.10, database=db)

    record = om.submit_signal(entry("DASH-1"), account())

    rows = {r["order_id"]: dict(r) for r in db.get_orders(limit=100)}
    assert record.order_id in rows
    assert rows[record.order_id]["status"] == record.status.value
    assert rows[record.order_id]["status"] not in ("CANCELLED", "FILLED")

    om.cancel(record.order_id)

    api_rows = client.get("/api/orders").json()
    assert [r["order_id"] for r in api_rows] == [record.order_id]
    assert api_rows[0]["status"] == "CANCELLED"
    assert api_rows[0]["symbol"] == SYMBOL
    assert api_rows[0]["quantity"] == record.requested_quantity

    history = [h["status"] for h in client.get(f"/api/orders/{record.order_id}/history").json()]
    assert history[-1] == "CANCELLED"
    assert len(history) >= 2


def _load_demo_script():
    spec = importlib.util.spec_from_file_location(
        "demo_e2e_sandbox_under_test", ROOT / "scripts" / "demo_e2e_sandbox.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_script_persists_sandbox_order_to_database_path(tmp_path, monkeypatch, api_client):
    client, _ = api_client
    db_path = tmp_path / "demo_trading.db"
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("DATABASE_PATH", str(db_path))

    demo = _load_demo_script()
    broker = _working_broker(demo.UPSTOX_INSTRUMENT)
    monkeypatch.setattr(demo, "UpstoxSandboxBroker", lambda: broker)
    monkeypatch.setattr(demo, "RUN_SANDBOX_ORDER", True)

    signal = entry("DEMO-DB-1")
    record = demo.execute_with_order_manager(
        signal=signal,
        config=CFG,
        risk_manager=RiskManager(CFG.risk),
        session_manager=SessionManager(CFG.session),
    )

    assert record is not None
    assert broker.orders[record.order_id].status == OrderStatus.CANCELLED

    persisted = Database(db_path)
    try:
        rows = {r["order_id"]: dict(r) for r in persisted.get_orders(limit=100)}
        assert rows[record.order_id]["status"] == "CANCELLED"
        assert rows[record.order_id]["symbol"] == demo.UPSTOX_INSTRUMENT
        statuses = [h["status"] for h in persisted.get_order_history(record.order_id)]
        assert statuses[-1] == "CANCELLED" and len(statuses) >= 2

        monkeypatch.setattr(api, "database", persisted)
        api_rows = client.get("/api/orders").json()
        assert any(r["order_id"] == record.order_id and r["status"] == "CANCELLED"
                   for r in api_rows)
    finally:
        persisted.close()
