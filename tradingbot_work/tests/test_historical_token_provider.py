from __future__ import annotations

import logging

import pytest

from src.historical_data import upstox_client as uc
from src.historical_data.extension import safe_token, safe_token_source
from src.historical_data.models import HistoricalDataError

ALL_NAMES = uc.TOKEN_ENV_NAMES + uc.FORBIDDEN_TOKEN_ENV_NAMES
HIST, MARKET, ACCESS, SANDBOX = "hist-secret-111", "market-secret-222", "access-secret-333", "sandbox-secret-444"


@pytest.fixture
def clean_env(monkeypatch):
    for name in ALL_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class Response:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code, self._body, self.headers, self.text = status, body, {}, ""

    def json(self):
        return self._body


class Session:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.headers_seen: list[dict] = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.headers_seen.append(headers)
        return self.response


def test_precedence_excludes_sandbox():
    assert uc.TOKEN_ENV_NAMES == ("UPSTOX_HISTORICAL_ACCESS_TOKEN", "UPSTOX_MARKET_DATA_ACCESS_TOKEN", "ACCESS_TOKEN")
    assert "UPSTOX_SANDBOX_ACCESS_TOKEN" not in uc.TOKEN_ENV_NAMES


def test_historical_token_detected_first(clean_env):
    for name, value in zip(ALL_NAMES, (HIST, MARKET, ACCESS, SANDBOX)):
        clean_env.setenv(name, value)
    assert uc.token_source() == "UPSTOX_HISTORICAL_ACCESS_TOKEN"
    assert uc.configured_token() == HIST


def test_market_data_token_fallback(clean_env):
    clean_env.setenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", MARKET)
    clean_env.setenv("ACCESS_TOKEN", ACCESS)
    clean_env.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", SANDBOX)
    assert uc.token_source() == "UPSTOX_MARKET_DATA_ACCESS_TOKEN"
    assert uc.configured_token() == MARKET


def test_access_token_fallback(clean_env):
    clean_env.setenv("ACCESS_TOKEN", ACCESS)
    clean_env.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", SANDBOX)
    assert uc.token_source() == "ACCESS_TOKEN"
    assert uc.configured_token() == ACCESS


def test_sandbox_token_never_selected(clean_env, tmp_path):
    clean_env.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", SANDBOX)
    env_file = tmp_path / ".env"
    env_file.write_text(f"UPSTOX_SANDBOX_ACCESS_TOKEN={SANDBOX}\nUPSTOX_HISTORICAL_ACCESS_TOKEN=\n", encoding="utf-8")
    clean_env.setattr(uc, "ENV_FILE", env_file)
    assert uc.token_source() is None and uc.configured_token() is None
    client = uc.UpstoxHistoricalClient(session=Session(Response(200, {})))
    with pytest.raises(HistoricalDataError) as exc:
        client.candles("NSE_INDEX|Nifty 50", "2026-10-05", "2026-10-05")
    assert exc.value.code == "CREDENTIALS_MISSING"
    assert "SANDBOX" not in exc.value.message and SANDBOX not in exc.value.message


def test_default_client_reads_project_env_file_without_mutating(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    content = f"UPSTOX_SANDBOX_ACCESS_TOKEN={SANDBOX}\nUPSTOX_MARKET_DATA_ACCESS_TOKEN={MARKET}\nACCESS_TOKEN={ACCESS}\n"
    env_file.write_text(content, encoding="utf-8")
    clean_env.setattr(uc, "ENV_FILE", env_file)
    session = Session(Response(200, {"status": "success", "data": {"candles": []}}))
    assert uc.UpstoxHistoricalClient(session=session).candles("NSE_INDEX|Nifty 50", "2026-10-05", "2026-10-05") == []
    assert session.headers_seen[0]["Authorization"] == f"Bearer {MARKET}"
    assert env_file.read_text(encoding="utf-8") == content
    import os
    assert all(name not in os.environ for name in ALL_NAMES)


def test_process_env_overrides_env_file_for_same_name(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"ACCESS_TOKEN={ACCESS}\n", encoding="utf-8")
    clean_env.setattr(uc, "ENV_FILE", env_file)
    clean_env.setenv("ACCESS_TOKEN", "process-secret-555")
    assert uc.configured_token() == "process-secret-555"


def test_extension_uses_same_provider(clean_env):
    clean_env.setenv("ACCESS_TOKEN", ACCESS)
    clean_env.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", SANDBOX)
    assert safe_token_source() == uc.token_source() == "ACCESS_TOKEN"
    assert safe_token() == uc.configured_token() == ACCESS


@pytest.mark.parametrize("status,body", [
    (200, {"status": "success", "data": {"candles": []}}),
    (401, {"status": "error", "errors": [{"errorCode": "UDAPI100050", "message": "Invalid token used to access API"}]}),
])
def test_token_value_never_logged(clean_env, caplog, capsys, status, body):
    clean_env.setenv("UPSTOX_HISTORICAL_ACCESS_TOKEN", HIST)
    client = uc.UpstoxHistoricalClient(session=Session(Response(status, body)), max_retries=0)
    caplog.set_level(logging.DEBUG)
    try:
        client.candles("NSE_INDEX|Nifty 50", "2026-10-05", "2026-10-05")
    except HistoricalDataError as exc:
        assert HIST not in str(exc) and HIST not in repr(exc.as_dict())
    out = capsys.readouterr()
    for text in (caplog.text, out.out, out.err, repr(client), uc.token_source()):
        assert HIST not in text
