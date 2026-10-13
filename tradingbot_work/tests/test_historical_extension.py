from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config.loader import load_strategy_config
from src.historical_data import extension as ext
from src.historical_data.models import HistoricalCandle, HistoricalDataError
from src.historical_data.option_coverage import RequiredOptionContract
from src.historical_data.storage import ParquetCandleStore
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.historical_data.validation import expected_slots

ROOT = Path(__file__).resolve().parents[1]
KEY = "NSE_INDEX|Nifty 50"
OPT_KEY = "NSE_FO|99999"
IST = ext.IST


def session_candles(day: date, key: str = KEY, base: float = 25000.0, *, utc: bool = False,
                    skip: int = 0) -> list[HistoricalCandle]:
    rows = []
    for i, slot in enumerate(expected_slots(day)):
        if i < skip:
            continue
        ts = slot.tz_convert("UTC") if utc else slot
        price = base + i
        rows.append(HistoricalCandle(ts.to_pydatetime(), key, price, price + 2, price - 2, price + 1, 100.0, 0.0))
    return rows


class FakeClient:
    def __init__(self, data: dict[str, list[HistoricalCandle]] | None = None,
                 errors: dict[str, HistoricalDataError] | None = None) -> None:
        self.data = data or {}
        self.errors = errors or {}
        self.calls: list[tuple] = []

    def candles(self, key, start, end, *, unit="minutes", interval=5):
        self.calls.append(("candles", key, start, end))
        if key in self.errors:
            raise self.errors[key]
        return [c for c in self.data.get(key, []) if start <= c.timestamp.astimezone(IST).date() <= end]

    def expired_option_contracts(self, underlying_key, expiry):
        self.calls.append(("expired_option_contracts", underlying_key, expiry))
        raise HistoricalDataError("PLAN_NOT_SUPPORTED", "Expired instruments need Upstox Plus")

    def expired_candles(self, key, start, end, *, interval=5):
        self.calls.append(("expired_candles", key, start, end))
        return []


def run_underlying(client, store, start=date(2026, 10, 6), end=date(2026, 10, 7)):
    limitations: list = []
    result = ext.download_underlying(client, store, underlying="NIFTY", instrument_key=KEY, start=start,
                                     end=end, requested_end=end, interval=5, limitations=limitations)
    return result, limitations


# ------------------------------------------------------------------ date range
def test_latest_completed_day_excludes_open_session_and_weekend():
    assert ext.latest_completed_day(datetime(2026, 10, 5, 11, 0, tzinfo=IST)) == date(2026, 10, 2)
    assert ext.latest_completed_day(datetime(2026, 10, 5, 15, 30, tzinfo=IST)) == date(2026, 10, 5)
    assert ext.latest_completed_day(datetime(2026, 10, 4, 18, 0, tzinfo=IST)) == date(2026, 10, 2)


def test_resolve_range_clamps_future_end_and_rejects_inverted():
    start, end, notes = ext.resolve_range(date(2026, 10, 2), date(2026, 10, 9),
                                          datetime(2026, 10, 5, 11, 0, tzinfo=IST))
    assert (start, end) == (date(2026, 10, 2), date(2026, 10, 2))
    assert notes and "clamped" in notes[0]
    with pytest.raises(HistoricalDataError):
        ext.resolve_range(date(2026, 10, 5), date(2026, 10, 2), datetime(2026, 10, 6, tzinfo=IST))


def test_download_filters_candles_outside_requested_range(tmp_path):
    data = session_candles(date(2026, 10, 1)) + session_candles(date(2026, 10, 6))
    result, _ = run_underlying(FakeClient({KEY: data}), ParquetCandleStore(tmp_path),
                               start=date(2026, 10, 6), end=date(2026, 10, 6))
    assert result.new_trading_days == ["2026-10-06"]
    assert result.candles_downloaded == 75


# --------------------------------------------------------------- IST handling
def test_timestamps_stored_in_ist_even_if_api_returns_utc(tmp_path):
    store = ParquetCandleStore(tmp_path)
    result, _ = run_underlying(FakeClient({KEY: session_candles(date(2026, 10, 6), utc=True)}), store,
                               end=date(2026, 10, 6))
    stored = store.read("underlying", "NIFTY", KEY)
    assert result.non_ist_timestamps == 0
    assert str(stored["timestamp"].dt.tz) == "Asia/Kolkata"
    assert stored["timestamp"].min().strftime("%H:%M") == "09:15"


def test_non_ist_detection():
    utc = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-06 03:45"]).tz_localize("UTC")})
    naive = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-06 09:15"])})
    assert ext.non_ist_timestamps(utc) == 1
    assert ext.non_ist_timestamps(naive) == 1


# ---------------------------------------------------- duplicates / missing slots
def test_duplicate_candles_detected_and_removed(tmp_path):
    rows = session_candles(date(2026, 10, 6))
    store = ParquetCandleStore(tmp_path)
    result, _ = run_underlying(FakeClient({KEY: rows + rows[:5]}), store, end=date(2026, 10, 6))
    assert result.duplicates_removed == 5
    assert len(store.read("underlying", "NIFTY", KEY)) == 75


def test_missing_candles_detected_not_filled(tmp_path):
    store = ParquetCandleStore(tmp_path)
    result, _ = run_underlying(FakeClient({KEY: session_candles(date(2026, 10, 6), skip=3)}), store,
                               end=date(2026, 10, 6))
    assert result.missing_candles == 3
    assert result.missing_by_day == {"2026-10-06": 3}
    assert len(store.read("underlying", "NIFTY", KEY)) == 72


def test_invalid_ohlc_reported_as_validation_error(tmp_path):
    rows = session_candles(date(2026, 10, 6))
    bad = rows[10]
    rows[10] = HistoricalCandle(bad.timestamp, KEY, bad.open, bad.low - 10, bad.low, bad.close, 1.0, 0.0)
    result, _ = run_underlying(FakeClient({KEY: rows}), ParquetCandleStore(tmp_path), end=date(2026, 10, 6))
    assert result.validation_errors.get("INVALID_OHLC") == 1
    assert result.rejected_rows == 1


# ---------------------------------------------------------- Parquet persistence
def test_parquet_persistence_uses_existing_partition_layout(tmp_path):
    store = ParquetCandleStore(tmp_path)
    result, _ = run_underlying(FakeClient({KEY: session_candles(date(2026, 10, 6))
                                           + session_candles(date(2026, 10, 7))}), store)
    paths = store.files("underlying", "NIFTY", KEY)
    assert [p.relative_to(tmp_path).as_posix() for p in paths] == [
        "underlying/NIFTY/year=2026/month=10/NSE_INDEX_Nifty_50.parquet"]
    assert result.candles_written == 150
    assert result.latest_stored_candle.startswith("2026-10-07T15:25")


def test_existing_data_is_never_overwritten(tmp_path):
    store = ParquetCandleStore(tmp_path)
    original = session_candles(date(2026, 10, 1), base=24000.0)
    from src.historical_data.models import candles_to_frame
    store.write("underlying", "NIFTY", KEY, candles_to_frame(original))
    overlap = session_candles(date(2026, 10, 1), base=99999.0) + session_candles(date(2026, 10, 6))
    result, _ = run_underlying(FakeClient({KEY: overlap}), store, start=date(2026, 10, 1), end=date(2026, 10, 6))
    stored = store.read("underlying", "NIFTY", KEY)
    old_day = stored[stored["timestamp"].dt.date == date(2026, 10, 1)]
    assert result.candles_skipped_existing == 75
    assert result.candles_written == 75
    assert result.existing_rows_before == 75 and result.existing_rows_after == 150
    assert old_day["open"].min() == 24000.0 and old_day["open"].max() < 99999.0


def test_no_data_returned_records_weekdays_without_data(tmp_path):
    result, _ = run_underlying(FakeClient({KEY: session_candles(date(2026, 10, 1))}),
                               ParquetCandleStore(tmp_path), start=date(2026, 10, 2), end=date(2026, 10, 2))
    assert result.new_trading_days == []
    assert result.weekdays_without_data == ["2026-10-02"]
    assert result.candles_written == 0


def test_permission_error_recorded_as_limitation_and_auth_failure_stops(tmp_path):
    client = FakeClient(errors={KEY: HistoricalDataError("PERMISSION_DENIED", "denied", status=403)})
    result, limitations = run_underlying(client, ParquetCandleStore(tmp_path), end=date(2026, 10, 6))
    assert limitations[0]["code"] == "PERMISSION_DENIED" and result.candles_written == 0
    client = FakeClient(errors={KEY: HistoricalDataError("AUTH_FAILED", "401", status=401)})
    with pytest.raises(HistoricalDataError):
        run_underlying(client, ParquetCandleStore(tmp_path), end=date(2026, 10, 6))


# --------------------------------------------------------------- option coverage
def _contract(key: str, expiry: date, signal_ts: str) -> RequiredOptionContract:
    return RequiredOptionContract(instrument_key=key, underlying="NIFTY", expiry=expiry, strike=25000.0,
                                  option_type="CE", trading_symbol=f"NIFTY 25000 CE {expiry}", lot_size=65,
                                  freeze_quantity=None, tick_size=0.05, from_date=date(2026, 10, 6),
                                  to_date=date(2026, 10, 6), signal_count=1, signal_times=(signal_ts,))


def _option_setup(tmp_path, monkeypatch, contracts, signals):
    store = ParquetCandleStore(tmp_path)
    from src.historical_data.models import candles_to_frame
    store.write("underlying", "NIFTY", KEY, candles_to_frame(session_candles(date(2026, 10, 6))))
    monkeypatch.setattr(ext, "research_signals", lambda *a, **k: signals)
    monkeypatch.setattr(ext, "extract_required_contracts", lambda *a, **k: (contracts, []))
    return store


def _run_options(client, store, today=date(2026, 10, 6)):
    limitations: list = []
    result = ext.extend_options(client, store, underlying="NIFTY", underlying_key=KEY, strategy_id="S2",
                                research_config=SimpleNamespace(option_mapping={"entry_timing": "next_open"}),
                                strategy_config=load_strategy_config(ROOT / "config" / "strategy.yaml"),
                                resolver_factory=lambda d: None, start=date(2026, 10, 6),
                                end=date(2026, 10, 6), interval=5, today=today, limitations=limitations)
    return result, limitations


def test_required_option_contract_downloaded_and_covered(tmp_path, monkeypatch):
    signal_ts = pd.Timestamp("2026-10-06 10:00", tz=IST)
    contract = _contract(OPT_KEY, date(2026, 10, 13), signal_ts.isoformat())
    store = _option_setup(tmp_path, monkeypatch, [contract], [SimpleNamespace(timestamp=signal_ts.to_pydatetime())])
    client = FakeClient({OPT_KEY: session_candles(date(2026, 10, 6), key=OPT_KEY, base=120.0)})
    result, limitations = _run_options(client, store)
    assert result.contracts_required == 1 and result.contracts_downloaded == 1
    assert result.coverage_percent == 100.0 and not limitations
    stored = store.read("options", "NIFTY", OPT_KEY, expiry=date(2026, 10, 13))
    assert len(stored) == 75 and set(stored["option_type"]) == {"CE"}
    assert {c[1] for c in client.calls if c[0] == "candles"} == {OPT_KEY}

    again, _ = _run_options(client, store)
    assert again.contracts_already_stored == 1 and again.contracts_downloaded == 0


def test_expired_contract_without_plan_recorded_as_unavailable(tmp_path, monkeypatch):
    signal_ts = pd.Timestamp("2026-10-06 10:00", tz=IST)
    contract = _contract(OPT_KEY, date(2026, 10, 6), signal_ts.isoformat())
    store = _option_setup(tmp_path, monkeypatch, [contract], [SimpleNamespace(timestamp=signal_ts.to_pydatetime())])
    result, limitations = _run_options(FakeClient(), store, today=date(2026, 10, 8))
    assert result.contracts_unavailable == 1 and result.coverage_percent == 0.0
    assert limitations[0]["code"] == "PLAN_NOT_SUPPORTED"
    assert store.read("options", "NIFTY", OPT_KEY, expiry=date(2026, 10, 6)).empty


def test_no_new_underlying_means_no_option_requests(tmp_path):
    client = FakeClient()
    result, _ = _run_options(client, ParquetCandleStore(tmp_path))
    assert result.contracts_required == 0 and result.coverage_percent is None and client.calls == []


# ------------------------------------------------------ safety: orders / tokens
FORBIDDEN_CALLS = ("place_order", "OrderManager", "TradingEngine", "api-sandbox", "/order")


@pytest.mark.parametrize("path", ["src/historical_data/extension.py", "scripts/extend_nifty_historical_data.py"])
def test_extension_code_has_no_order_execution(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    for name in FORBIDDEN_CALLS:
        assert name not in source


def test_sandbox_token_never_used(monkeypatch):
    for name in ext.SAFE_TOKEN_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(ext.FORBIDDEN_TOKEN_ENV, "sandbox-secret-value")
    assert ext.safe_token_source() is None and ext.safe_token() is None
    monkeypatch.setenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", "market-secret-value")
    assert ext.safe_token_source() == "UPSTOX_MARKET_DATA_ACCESS_TOKEN"
    client = UpstoxHistoricalClient(token_provider=ext.safe_token, session=object())
    assert client._token_provider() == "market-secret-value"
    assert "market-secret-value" not in repr(client)


def _load_script():
    spec = importlib.util.spec_from_file_location("extend_script", ROOT / "scripts" / "extend_nifty_historical_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_end_to_end_report_and_token_never_logged(tmp_path, monkeypatch, capsys):
    secret = "tok-SECRET-123456789"
    for name in ext.SAFE_TOKEN_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", secret)
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "sandbox-" + secret)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    script = _load_script()
    monkeypatch.setattr(script, "extend_options", lambda *a, **k: ext.OptionResult())

    from src.historical_data.service import HistoricalDataService

    class Service(HistoricalDataService):
        def underlying_key(self, underlying):
            return KEY

    service = Service(store_root=tmp_path / "store")
    client = FakeClient({KEY: session_candles(date(2026, 10, 6))})
    report = script.run_extension(start=date(2026, 10, 2), end=date(2026, 10, 9), service=service,
                                  client=client, out_dir=tmp_path / "out",
                                  now=datetime(2026, 10, 7, 9, 0, tzinfo=IST))
    out = capsys.readouterr().out
    assert "DATA EXTENSION COMPLETE" in out
    assert report["token_source"] == "UPSTOX_MARKET_DATA_ACCESS_TOKEN"
    assert report["effective_range"]["end"] == "2026-10-06"
    assert report["summary"]["new_trading_days"] == 1 and report["summary"]["candles_written"] == 75
    assert report["underlying"]["weekdays_without_data"] == ["2026-10-02", "2026-10-05"]
    files = [tmp_path / "out" / f"NIFTY_historical_extension.{e}" for e in ("json", "csv", "md")]
    assert all(f.exists() for f in files)
    json.loads(files[0].read_text(encoding="utf-8"))
    for text in [out] + [f.read_text(encoding="utf-8") for f in files]:
        assert secret not in text
    assert all(call[0] == "candles" for call in client.calls)


def test_end_to_end_without_new_data_stops(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    script = _load_script()
    from src.historical_data.service import HistoricalDataService

    class Service(HistoricalDataService):
        def underlying_key(self, underlying):
            return KEY

    report = script.run_extension(start=date(2026, 10, 2), end=date(2026, 10, 5),
                                  service=Service(store_root=tmp_path / "store"), client=FakeClient(),
                                  out_dir=tmp_path / "out", now=datetime(2026, 10, 5, 11, 0, tzinfo=IST))
    out = capsys.readouterr().out
    assert "DATA EXTENSION STOPPED" in out
    assert report["data_available"] is False
    assert any(l["code"] == "NO_NEW_DATA" for l in report["limitations"])
