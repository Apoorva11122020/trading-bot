"""Historical data and research API endpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("DASHBOARD_API_KEY", "test-dashboard-key")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    from src.api import app as api_module
    api_module.database = __import__("src.database", fromlist=["Database"]).Database(str(tmp_path / "api.db"))
    api_module.historical_service = __import__(
        "src.historical_data.service", fromlist=["HistoricalDataService"]
    ).HistoricalDataService(database=api_module.database, store_root=tmp_path / "historical")
    return TestClient(api_module.app)


def test_historical_status_requires_auth_in_production(monkeypatch, client):
    monkeypatch.setenv("APP_ENV", "production")
    r = client.get("/api/historical/options/status")
    assert r.status_code == 401


def test_historical_status_ok(client):
    r = client.get("/api/historical/options/status", headers={"X-API-Key": "test-dashboard-key"})
    assert r.status_code == 200
    body = r.text
    assert "UPSTOX_SANDBOX_ACCESS_TOKEN" not in body
    token = os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN")
    if token:
        assert token not in body


def test_download_refuses_live(monkeypatch, client):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    r = client.post(
        "/api/historical/options/download",
        headers={"X-API-Key": "test-dashboard-key"},
        json={"underlyings": ["NIFTY"], "from_date": "2026-09-01", "to_date": "2026-09-07"},
    )
    assert r.status_code == 403


def test_strategies_list(client):
    r = client.get("/api/backtest/options/strategies", headers={"X-API-Key": "test-dashboard-key"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["strategies"]) == 4


def test_research_dashboard_assets():
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "researchView" in html
    assert "DOWNLOAD HISTORICAL DATA" in html
    assert "research.js" in html
