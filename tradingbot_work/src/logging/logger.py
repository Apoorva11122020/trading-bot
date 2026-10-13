"""Structured JSON logging with secret redaction and rotating files."""
from __future__ import annotations
import json, logging, os, re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

SECRET_KEYS = {"api_key", "api_secret", "access_token", "token", "password", "secret", "authorization"}
# Any key containing one of these fragments is treated as secret
# (e.g. UPSTOX_SANDBOX_ACCESS_TOKEN, client_secret, x-api-key).
SECRET_KEY_FRAGMENTS = ("token", "secret", "password", "api_key", "apikey", "api-key", "authorization", "credential")
REDACTED = "***REDACTED***"
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]+")


def _is_secret_key(key: Any) -> bool:
    name = str(key).lower()
    return name in SECRET_KEYS or any(fragment in name for fragment in SECRET_KEY_FRAGMENTS)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: (REDACTED if _is_secret_key(k) else _redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return _BEARER.sub("Bearer " + REDACTED, value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context:
            payload["context"] = _redact(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(_redact(payload), default=str, separators=(",", ":"))


def get_logger(name: str = "trading_bot", log_dir: str | Path = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.handlers:
        return logger
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(Path(log_dir) / "trading-bot.jsonl", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    logger.addHandler(console)
    return logger


def log_event(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    logger.log(level, message, extra={"context": _redact(context)})
