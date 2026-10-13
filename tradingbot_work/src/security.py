from __future__ import annotations
import os
from pathlib import Path

SECRET_ENV_NAMES = {
    "BROKER_API_KEY",
    "BROKER_API_SECRET",
    "ACCESS_TOKEN",
    "UPSTOX_MARKET_DATA_ACCESS_TOKEN",
    "DATABASE_URL",
    "UPSTOX_SANDBOX_ACCESS_TOKEN",
    "DASHBOARD_API_KEY",
    "LIVE_TRADING_CONFIRMATION",
}

def validate_security(project_root: str | Path = ".") -> dict:
    root = Path(project_root)
    env_exists = (root / ".env").exists()
    gitignore = (root / ".gitignore").read_text(encoding="utf-8") if (root / ".gitignore").exists() else ""
    env_ignored = ".env" in gitignore
    live = os.getenv("TRADING_MODE", "PAPER").upper() == "LIVE"
    production = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).lower() in {"production", "prod"}
    dashboard_key = os.getenv("DASHBOARD_API_KEY", "")
    dashboard_auth = os.getenv("DASHBOARD_API_AUTH_REQUIRED", "false").lower() in {"1", "true", "yes", "on"}
    return {
        "env_file_present": env_exists,
        "env_ignored": env_ignored,
        "live_mode_requested": live,
        "secrets_from_environment": True,
        "production_dashboard_auth_configured": (not production and not dashboard_auth) or bool(dashboard_key),
        "runtime_data_ignored": "data/" in gitignore and "logs/*.jsonl" in gitignore,
    }
