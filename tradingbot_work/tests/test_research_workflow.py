"""Strategy research download + backtest workflow."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.historical_data.models import HistoricalCandle, HistoricalDataError
from src.historical_data.service import UNDERLYING_MISSING_HINT, HistoricalDataService

IST = ZoneInfo("Asia/Kolkata")


class FakeClient:
    def candles(self, instrument_key, from_date, to_date, *, unit="minutes", interval=5):
        day = date.fromisoformat(str(from_date)[:10])
        end = date.fromisoformat(str(to_date)[:10])
        rows = []
        price = 22400.0
        while day <= end:
            if day.weekday() < 5:
                for hour, minute in ((9, 15), (9, 20), (9, 25)):
                    rows.append(HistoricalCandle(
                        datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST),
                        instrument_key, price, price + 5, price - 5, price + 1, 0, 0,
                    ))
                    price += 2
            day = date.fromordinal(day.toordinal() + 1)
        return rows


def test_download_underlying_sync_stores_parquet(tmp_path):
    service = HistoricalDataService(store_root=tmp_path / "historical")
    service.client = FakeClient()
    verification = service.download_underlying_sync("NIFTY", date(2026, 9, 28), date(2026, 10, 1))
    assert verification["candle_count"] > 0
    assert verification["files"]
    assert service.has_underlying_data("NIFTY", date(2026, 9, 28), date(2026, 10, 1))


def test_backtest_missing_underlying_message(tmp_path):
    service = HistoricalDataService(store_root=tmp_path / "historical")
    with pytest.raises(HistoricalDataError) as exc:
        service.run_backtest_sync(["NIFTY"], date(2026, 9, 28), date(2026, 10, 1))
    assert exc.value.code == "UNDERLYING_DATA_MISSING"
    assert UNDERLYING_MISSING_HINT in exc.value.message


def test_run_strategy_research_script_has_hint():
    text = (__import__("pathlib").Path(__file__).resolve().parents[1]
            / "scripts" / "run_strategy_research.py").read_text(encoding="utf-8")
    assert "UNDERLYING_MISSING_HINT" in text
    assert "download the required underlying data" in UNDERLYING_MISSING_HINT
