import os
from pathlib import Path
from fastapi.testclient import TestClient


def test_production_dashboard_requires_api_key(monkeypatch):
    from src.api.app import app
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DASHBOARD_API_KEY", "phase35-secret")
    client = TestClient(app)
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/status", headers={"X-API-Key": "phase35-secret"}).status_code == 200


def test_production_dashboard_missing_key_fails_closed(monkeypatch):
    from src.api.app import app
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("DASHBOARD_API_KEY", raising=False)
    client = TestClient(app)
    assert client.get("/api/status").status_code == 503


def test_security_runtime_files_are_ignored():
    text = Path(".gitignore").read_text()
    assert "data/" in text
    assert "logs/*.jsonl" in text
    assert "*.db" in text


def test_docker_runs_non_root_and_has_healthcheck():
    text = Path("Dockerfile").read_text()
    assert "USER appuser" in text
    assert "HEALTHCHECK" in text


def test_production_auth_is_documented_in_env_example():
    text = Path(".env.example").read_text()
    assert "DASHBOARD_API_KEY" in text
    assert "APP_ENV=development" in text
