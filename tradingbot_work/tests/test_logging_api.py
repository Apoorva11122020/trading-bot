from pathlib import Path
from fastapi.testclient import TestClient
from src.logging.logger import get_logger
from src.api.app import app


def test_structured_logger_redacts_secrets(tmp_path, caplog):
    logger = get_logger("test_structured", tmp_path)
    from src.logging.logger import log_event
    log_event(logger, 20, "test", api_key="SECRET", nested={"access_token": "TOKEN", "safe": 1})
    for handler in logger.handlers:
        handler.close()
    text = (Path(tmp_path) / "trading-bot.jsonl").read_text()
    assert "SECRET" not in text
    assert "TOKEN" not in text
    assert "REDACTED" in text


def test_api_health_and_routes():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/status").json()["trading_mode"] == "PAPER"
    assert client.get("/api/orders").status_code == 200
    assert client.get("/api/positions").status_code == 200
