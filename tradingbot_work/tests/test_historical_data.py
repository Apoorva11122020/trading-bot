"""Historical data models, client, validation and storage."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.historical_data.downloader import month_chunks
from src.historical_data.models import HistoricalCandle, HistoricalDataError, candles_to_frame, parse_candle_rows
from src.historical_data.storage import ParquetCandleStore
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.historical_data.validation import validate_candles

IST = ZoneInfo("Asia/Kolkata")


def test_parse_candle_rows():
    rows = [["2026-10-01T09:15:00+05:30", 100, 101, 99, 100.5, 10, 5]]
    candles = parse_candle_rows(rows, "NSE_FO|1")
    assert len(candles) == 1
    assert candles[0].close == 100.5
    assert candles[0].open_interest == 5.0


def test_validate_candles_rejects_bad_ohlc():
    frame = pd.DataFrame([{
        "timestamp": datetime(2026, 10, 1, 9, 15, tzinfo=IST),
        "instrument_key": "NSE_FO|1", "open": 10, "high": 9, "low": 8, "close": 9.5, "volume": 1, "open_interest": 0,
    }])
    clean, report = validate_candles(frame)
    assert report.rejected_rows == 1
    assert clean.empty


def test_validate_drops_outside_session_without_counting_rejected():
    frame = pd.DataFrame([
        {"timestamp": datetime(2026, 10, 1, 9, 15, tzinfo=IST), "instrument_key": "k", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 1, "open_interest": 0},
        {"timestamp": datetime(2026, 10, 1, 15, 30, tzinfo=IST), "instrument_key": "k", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 1, "open_interest": 0},
    ])
    clean, report = validate_candles(frame)
    assert len(clean) == 1
    assert report.outside_session_rows == 1
    assert report.rejected_rows == 0


def test_month_chunks():
    chunks = month_chunks(date(2026, 1, 15), date(2026, 3, 2))
    assert chunks[0] == (date(2026, 1, 15), date(2026, 1, 31))
    assert chunks[-1][1] == date(2026, 3, 2)


def test_parquet_roundtrip(tmp_path):
    store = ParquetCandleStore(tmp_path)
    frame = candles_to_frame([
        HistoricalCandle(datetime(2026, 10, 1, 9, 15, tzinfo=IST), "NSE_FO|1", 1, 2, 0.5, 1.5, 10, 1),
    ], underlying="NIFTY", expiry="2026-10-06", strike=23000, option_type="CE")
    store.write("options", "NIFTY", "NSE_FO|1", frame, expiry=date(2026, 10, 6))
    back = store.read("options", "NIFTY", "NSE_FO|1", expiry=date(2026, 10, 6))
    assert len(back) == 1
    assert float(back.iloc[0]["close"]) == 1.5


def test_client_requires_token():
    client = UpstoxHistoricalClient(token_provider=lambda: None)
    with pytest.raises(HistoricalDataError) as exc:
        client.candles("NSE_INDEX|Nifty 50", "2026-10-01", "2026-10-01")
    assert exc.value.code == "CREDENTIALS_MISSING"


def test_client_retries_retryable(monkeypatch):
    calls = {"n": 0}

    class Resp:
        status_code = 429
        headers = {}
        def json(self):
            return {"status": "error", "errors": [{"errorCode": "RATE", "message": "slow down"}]}

    class Session:
        def get(self, *a, **k):
            calls["n"] += 1
            return Resp()

    client = UpstoxHistoricalClient(token_provider=lambda: "token", session=Session(), max_retries=2, sleep=lambda _: None)
    with pytest.raises(HistoricalDataError) as exc:
        client.candles("NSE_INDEX|Nifty 50", "2026-10-01", "2026-10-01")
    assert exc.value.code == "RATE_LIMITED"
    assert calls["n"] == 3



def test_out_of_session_bad_prices_do_not_count_as_rejected():
    frame = pd.DataFrame([
        {
            "timestamp": datetime(2026, 10, 1, 9, 15, tzinfo=IST),
            "instrument_key": "k", "open": 10, "high": 11, "low": 9, "close": 10.5,
            "volume": 1, "open_interest": 0,
        },
        {
            "timestamp": datetime(2026, 10, 1, 15, 30, tzinfo=IST),
            "instrument_key": "k", "open": 10, "high": 8, "low": 9, "close": 10.5,
            "volume": 1, "open_interest": 0,
        },
    ])
    clean, report = validate_candles(frame)
    assert len(clean) == 1
    assert report.outside_session_rows == 1
    assert report.rejected_rows == 0
    assert report.issues == {}


def test_missing_volume_is_rejected_and_reported():
    frame = pd.DataFrame([{
        "timestamp": datetime(2026, 10, 1, 9, 15, tzinfo=IST),
        "instrument_key": "k", "open": 10, "high": 11, "low": 9, "close": 10.5,
        "volume": float("nan"), "open_interest": 0,
    }])
    clean, report = validate_candles(frame)
    assert clean.empty
    assert report.rejected_rows == 1
    assert report.issues["MISSING_VOLUME"] == 1
