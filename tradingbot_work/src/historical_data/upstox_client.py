"""Read-only Upstox historical-data client.

Endpoints (official docs):
  V3 candles (instruments that are currently listed):
    GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}
  Expired instruments (Upstox Plus plan, live-account token):
    GET /v2/expired-instruments/expiries?instrument_key=...
    GET /v2/expired-instruments/option/contract?instrument_key=...&expiry_date=...
    GET /v2/expired-instruments/future/contract?instrument_key=...&expiry_date=...
    GET /v2/expired-instruments/historical-candle/{expired_key}/{interval}/{to_date}/{from_date}

Only GET requests are made. This module never places, modifies or cancels
orders. The access token is read from the environment, sent only in the
Authorization header and never logged or returned.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from dotenv import dotenv_values

from src.logging.logger import log_event

from .models import HistoricalCandle, HistoricalDataError, parse_candle_rows

API_BASE = "https://api.upstox.com"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
# Checked in order. The Sandbox order token is only valid on api-sandbox and is
# never used for historical market data.
TOKEN_ENV_NAMES = ("UPSTOX_HISTORICAL_ACCESS_TOKEN", "UPSTOX_MARKET_DATA_ACCESS_TOKEN", "ACCESS_TOKEN")
FORBIDDEN_TOKEN_ENV_NAMES = ("UPSTOX_SANDBOX_ACCESS_TOKEN",)
EXPIRED_INTERVALS = {1: "1minute", 3: "3minute", 5: "5minute", 15: "15minute", 30: "30minute"}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

logger = logging.getLogger("historical_data.upstox")


def _token_settings() -> dict[str, str]:
    """Allowed token variables: process environment over the project .env (read-only, nothing is mutated)."""
    values: dict[str, str] = {}
    if ENV_FILE.is_file():
        values.update({k: v for k, v in dotenv_values(ENV_FILE).items() if k in TOKEN_ENV_NAMES and v})
    values.update({k: os.environ[k] for k in TOKEN_ENV_NAMES if os.environ.get(k)})
    return values


def _resolve_token() -> tuple[str | None, str | None]:
    settings = _token_settings()
    for name in TOKEN_ENV_NAMES:
        value = (settings.get(name) or "").strip()
        if value:
            return name, value
    return None, None


def configured_token() -> str | None:
    return _resolve_token()[1]


def token_source() -> str | None:
    """Name (never the value) of the environment variable that supplies the token."""
    return _resolve_token()[0]


def _default_session():
    import requests

    return requests.Session()


def _iso(value: date | str) -> str:
    return value.isoformat() if isinstance(value, date) else date.fromisoformat(str(value)[:10]).isoformat()


class UpstoxHistoricalClient:
    def __init__(
        self,
        token_provider: Callable[[], str | None] = configured_token,
        *,
        session=None,
        base_url: str = API_BASE,
        timeout: float = 20.0,
        max_retries: int = 4,
        backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        min_interval_seconds: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._token_provider = token_provider
        self._session = session
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.min_interval_seconds = float(min_interval_seconds)
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last_request = 0.0
        self.requests_made = 0

    def __repr__(self) -> str:
        return f"UpstoxHistoricalClient(base_url={self.base_url!r})"

    @property
    def session(self):
        if self._session is None:
            self._session = _default_session()
        return self._session

    def _headers(self) -> dict[str, str]:
        token = self._token_provider()
        if not token:
            raise HistoricalDataError(
                "CREDENTIALS_MISSING",
                "No Upstox access token configured (set one of: " + ", ".join(TOKEN_ENV_NAMES) + ")")
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval_seconds - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = self._clock()

    @staticmethod
    def _error_details(response) -> tuple[str | None, str]:
        try:
            body = response.json()
        except ValueError:
            return None, (response.text or "")[:200]
        errors = body.get("errors") if isinstance(body, dict) else None
        if errors and isinstance(errors, list) and isinstance(errors[0], dict):
            first = errors[0]
            return first.get("errorCode") or first.get("error_code"), str(first.get("message") or "")[:300]
        return None, str(body)[:200]

    def _classify(self, response) -> HistoricalDataError:
        status = response.status_code
        code, message = self._error_details(response)
        if code == "UDAPI1149":
            return HistoricalDataError(
                "PLAN_NOT_SUPPORTED", f"Expired-instrument data requires the Upstox Plus plan ({code}: {message})",
                status=status)
        if status == 401:
            return HistoricalDataError("AUTH_FAILED", f"Upstox rejected the access token ({code}: {message})", status=status)
        if status == 403:
            return HistoricalDataError("PERMISSION_DENIED", f"Access denied ({code}: {message})", status=status)
        if status == 404:
            return HistoricalDataError("NOT_FOUND", f"Resource not found ({code}: {message})", status=status)
        if status == 429:
            return HistoricalDataError("RATE_LIMITED", f"Upstox rate limit hit ({code}: {message})",
                                       status=status, retryable=True)
        if status >= 500:
            return HistoricalDataError("UPSTOX_SERVER_ERROR", f"Upstox server error {status}", status=status,
                                       retryable=True)
        return HistoricalDataError("BAD_REQUEST", f"Upstox request rejected ({code}: {message})", status=status)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        headers = self._headers()
        url = self.base_url + path
        attempt = 0
        while True:
            attempt += 1
            self._throttle()
            started = time.perf_counter()
            try:
                self.requests_made += 1
                response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            except Exception as exc:
                error = HistoricalDataError("NETWORK_ERROR", f"{type(exc).__name__} calling Upstox", retryable=True)
                retry_after = None
            else:
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                if response.status_code == 200:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise HistoricalDataError("INVALID_RESPONSE", "Upstox returned non-JSON content",
                                                  status=200) from exc
                    if not isinstance(body, dict) or body.get("status") != "success":
                        raise HistoricalDataError("INVALID_RESPONSE", "Upstox response status is not 'success'",
                                                  status=200)
                    log_event(logger, logging.DEBUG, "upstox_historical_ok", path=path.split("?")[0],
                              elapsed_ms=elapsed_ms, attempt=attempt)
                    return body.get("data")
                error = self._classify(response)
                retry_after = response.headers.get("Retry-After") if hasattr(response, "headers") else None
            final = not error.retryable or attempt > self.max_retries
            log_event(logger, logging.WARNING, "upstox_historical_error" if final else "upstox_historical_retry",
                      path=path.split("?")[0], code=error.code, status=error.status, attempt=attempt,
                      retryable=error.retryable)
            if final:
                raise error
            delay = min(self.max_backoff_seconds, self.backoff_seconds * (2 ** (attempt - 1)))
            try:
                if retry_after is not None:
                    delay = max(delay, min(self.max_backoff_seconds, float(retry_after)))
            except (TypeError, ValueError):
                pass
            self._sleep(delay)

    @staticmethod
    def _candles(data: Any, instrument_key: str) -> list[HistoricalCandle]:
        if not isinstance(data, dict) or not isinstance(data.get("candles"), list):
            raise HistoricalDataError("INVALID_RESPONSE", f"{instrument_key}: response has no candles list", status=200)
        return parse_candle_rows(data["candles"], instrument_key)

    # ------------------------------------------------------------------
    # Public, read-only API
    # ------------------------------------------------------------------

    def candles(self, instrument_key: str, from_date: date | str, to_date: date | str,
                *, unit: str = "minutes", interval: int = 5) -> list[HistoricalCandle]:
        """V3 candles for an instrument that is listed today."""
        path = (f"/v3/historical-candle/{quote(instrument_key, safe='')}/{unit}/{int(interval)}/"
                f"{_iso(to_date)}/{_iso(from_date)}")
        return self._candles(self._get(path), instrument_key)

    def intraday_candles(self, instrument_key: str, *, unit: str = "minutes",
                         interval: int = 5) -> list[HistoricalCandle]:
        """V3 intraday candles for the current trading session."""
        path = f"/v3/historical-candle/intraday/{quote(instrument_key, safe='')}/{unit}/{int(interval)}"
        return self._candles(self._get(path), instrument_key)

    def expired_expiries(self, underlying_key: str) -> list[date]:
        data = self._get("/v2/expired-instruments/expiries", {"instrument_key": underlying_key})
        if not isinstance(data, list):
            raise HistoricalDataError("INVALID_RESPONSE", "Expiries response is not a list", status=200)
        return sorted(date.fromisoformat(str(item)[:10]) for item in data)

    def expired_option_contracts(self, underlying_key: str, expiry: date | str) -> list[dict]:
        data = self._get("/v2/expired-instruments/option/contract",
                         {"instrument_key": underlying_key, "expiry_date": _iso(expiry)})
        if not isinstance(data, list):
            raise HistoricalDataError("INVALID_RESPONSE", "Expired option contracts response is not a list", status=200)
        return data

    def expired_future_contracts(self, underlying_key: str, expiry: date | str) -> list[dict]:
        data = self._get("/v2/expired-instruments/future/contract",
                         {"instrument_key": underlying_key, "expiry_date": _iso(expiry)})
        if not isinstance(data, list):
            raise HistoricalDataError("INVALID_RESPONSE", "Expired future contracts response is not a list", status=200)
        return data

    def expired_candles(self, expired_key: str, from_date: date | str, to_date: date | str,
                        *, interval: int = 5) -> list[HistoricalCandle]:
        if int(interval) not in EXPIRED_INTERVALS:
            raise HistoricalDataError("INVALID_INTERVAL", f"Expired candles support {sorted(EXPIRED_INTERVALS)} minutes")
        path = (f"/v2/expired-instruments/historical-candle/{quote(expired_key, safe='')}/"
                f"{EXPIRED_INTERVALS[int(interval)]}/{_iso(to_date)}/{_iso(from_date)}")
        return self._candles(self._get(path), expired_key)
