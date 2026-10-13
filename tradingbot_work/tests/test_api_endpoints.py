import json

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.database import Database


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = Database(tmp_path / "api.db")
    monkeypatch.setattr(api, "database", db)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DASHBOARD_API_AUTH_REQUIRED", raising=False)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    yield TestClient(api.app), db
    db.close()


def test_status_reports_mode_risk_and_runtime(client):
    c, db = client
    db.set_state("runtime_status", {"running": True, "broker_connected": True,
                                    "market_data_status": "CONNECTED", "updated_at": "x"})
    data = c.get("/api/status").json()
    assert data["trading_mode"] == "PAPER"
    assert data["live_trading_enabled"] is False
    assert data["broker_status"] == "CONNECTED"
    assert data["market_data_status"] == "CONNECTED"
    assert data["bot_status"] == "RUNNING"
    assert data["kill_switch_active"] is False
    assert data["daily_pnl"] == 0.0


def test_health_reports_database_state(client, monkeypatch):
    c, db = client
    assert c.get("/health").json()["database"] == "ok"

    def down():
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "ping", down)
    body = c.get("/health").json()
    assert body["status"] == "degraded" and body["database"] == "unavailable"


def test_risk_config_equity_endpoints(client):
    c, db = client
    db.save_trade({"trade_id": "T1", "order_id": "T1", "symbol": "DEMO", "side": "BUY", "quantity": 1,
                   "entry_price": 100, "exit_price": 110, "realized_pnl": 10.0,
                   "opened_at": "2026-10-01T04:30:00+00:00", "closed_at": "2026-10-01T05:00:00+00:00",
                   "status": "CLOSED"})
    risk = c.get("/api/risk").json()
    assert "limits" in risk and risk["kill_switch_active"] is False
    config = c.get("/api/config").json()
    assert config["live_trading_enabled"] is False
    assert config["strategy"]["indicators"]["ema_fast"] == 20
    assert c.get("/api/equity").json()[-1]["cumulative_pnl"] == 10.0


def test_config_endpoint_never_returns_secrets(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "super-secret-value-123")
    monkeypatch.setenv("DASHBOARD_API_KEY", "dash-key-xyz")
    for path in ("/api/config", "/api/status", "/api/risk", "/health"):
        text = c.get(path, headers={"X-API-Key": "dash-key-xyz"}).text
        assert "super-secret-value-123" not in text
        assert "dash-key-xyz" not in text


def test_kill_switch_control_always_requires_api_key(client, monkeypatch):
    c, db = client
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    assert c.post("/api/kill-switch/activate").status_code == 503

    monkeypatch.setenv("DASHBOARD_API_KEY", "control-key")
    assert c.post("/api/kill-switch/activate").status_code == 401
    assert c.post("/api/kill-switch/activate", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = c.post("/api/kill-switch/activate", json={"reason": "TEST"}, headers={"X-API-Key": "control-key"})
    assert ok.status_code == 200 and ok.json()["kill_switch_active"] is True
    assert db.get_state("kill_switch")["active"] is True
    assert c.get("/api/status").json()["kill_switch_active"] is True

    resumed = c.post("/api/kill-switch/resume", headers={"X-API-Key": "control-key"})
    assert resumed.json()["kill_switch_active"] is False
    assert db.get_state("kill_switch")["active"] is False


def test_order_history_endpoint(client):
    c, db = client
    db.save_order({"order_id": "O1", "symbol": "DEMO", "side": "BUY", "order_type": "MARKET",
                   "quantity": 1, "status": "OPEN"})
    db.save_order({"order_id": "O1", "symbol": "DEMO", "side": "BUY", "order_type": "MARKET",
                   "quantity": 1, "status": "FILLED", "filled_quantity": 1, "average_fill_price": 100})
    history = c.get("/api/orders/O1/history").json()
    assert [h["status"] for h in history] == ["OPEN", "FILLED"]


def test_production_requires_key_for_new_endpoints(client, monkeypatch):
    c, _ = client
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_KEY", "prod-key")
    for path in ("/api/risk", "/api/config", "/api/equity", "/api/orders/x/history"):
        assert c.get(path).status_code == 401
        assert c.get(path, headers={"X-API-Key": "prod-key"}).status_code == 200
