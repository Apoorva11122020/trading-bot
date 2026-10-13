import json
import logging
import re
from pathlib import Path

from src.logging.logger import JsonFormatter, _redact

ROOT = Path(__file__).resolve().parents[1]


def test_redaction_covers_prefixed_secret_keys_and_bearer_strings():
    data = _redact({
        "UPSTOX_SANDBOX_ACCESS_TOKEN": "abc",
        "client_secret": "s",
        "x-api-key": "k",
        "nested": {"refresh_token": "r", "safe": 1},
        "error": "401 for Authorization: Bearer abc.def-123",
    })
    assert data["UPSTOX_SANDBOX_ACCESS_TOKEN"] == "***REDACTED***"
    assert data["client_secret"] == "***REDACTED***"
    assert data["x-api-key"] == "***REDACTED***"
    assert data["nested"]["refresh_token"] == "***REDACTED***"
    assert data["nested"]["safe"] == 1
    assert "abc.def-123" not in data["error"]


def test_formatter_never_emits_token_values():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", None, None)
    record.context = {"access_token": "tok-123", "detail": "Bearer tok-123"}
    text = JsonFormatter().format(record)
    assert "tok-123" not in text
    json.loads(text)


def test_dockerignore_excludes_secrets_and_runtime_state():
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    for entry in (".env", ".env.*", "data/", "logs/", ".venv/", "*.db"):
        assert entry in lines
    assert "!.env.example" in lines


def test_gitignore_excludes_env_files():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines and ".env.*" in lines and "!.env.example" in lines


def test_env_example_has_only_placeholders():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "LIVE_TRADING_ENABLED=false" in text
    assert not re.search(r"eyJ[A-Za-z0-9_-]{20,}", text)


def test_dockerfile_runs_as_non_root_and_requirements_cover_imports():
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in docker
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("python-dotenv", "fastapi", "uvicorn", "requests", "pandas", "pydantic", "pyyaml"):
        assert package in reqs


def test_compose_binds_localhost_and_defaults_to_paper():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8000:8000"' in compose
    assert "TRADING_MODE: ${TRADING_MODE:-PAPER}" in compose
    assert "LIVE_TRADING_ENABLED: ${LIVE_TRADING_ENABLED:-false}" in compose


def test_no_tokens_in_tracked_source():
    pattern = re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}")
    for folder in ("src", "scripts", "tests", "config", "dashboard", "docs"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".js", ".html", ".md", ".json"}:
                assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path


def test_strategy_config_is_paper_and_live_disabled():
    from src.config.loader import load_strategy_config
    cfg = load_strategy_config(ROOT / "config" / "strategy.yaml")
    assert cfg.system.trading_mode == "PAPER"
    assert cfg.system.live_trading_enabled is False


def test_no_cors_wildcard_or_debug_in_api():
    source = (ROOT / "src" / "api" / "app.py").read_text(encoding="utf-8")
    assert not re.search(r"allow_origins\s*=\s*\[\s*[\"']\*[\"']", source)
    assert "allow_origin_regex" not in source
    assert "debug=True" not in source

    from fastapi.testclient import TestClient

    import src.api.app as api

    client = TestClient(api.app)
    preflight = {"Access-Control-Request-Method": "GET"}
    evil = client.options("/api/status", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in evil.headers
    local = client.options("/api/status", headers={"Origin": "http://localhost:3000", **preflight})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_strategy_engine_has_no_hardcoded_indicator_periods():
    source = (ROOT / "src" / "strategy" / "strategy_engine.py").read_text(encoding="utf-8")
    assert not re.search(r"rolling\(\s*\d+\s*\)", source)
    assert not re.search(r"span=\s*\d+", source)
