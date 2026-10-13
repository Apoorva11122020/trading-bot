from __future__ import annotations

import inspect
import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import ACTIVE, EXPIRED, HistoricalContract
from src.historical_data.models import HistoricalCandle, HistoricalDataError
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.options.selection import select_strike
from src.safety import live_gate

IST = u6.IST
NIFTY_KEY = "NSE_INDEX|Nifty 50"
TOKEN = "tok_SECRET_value_1234567890"
START, END = date(2026, 4, 27), date(2026, 5, 6)
HOLIDAY = date(2026, 5, 1)
DAYS = [d for d in (START + timedelta(days=i) for i in range((END - START).days + 1))
        if d.weekday() < 5 and d != HOLIDAY]
EXPIRED_EXP = [date(2026, 4, 21), date(2026, 4, 28), date(2026, 5, 5)]
ACTIVE_EXP = [date(2026, 5, 12), date(2026, 5, 19)]
TODAY = date(2026, 5, 8)
STRIKES = [23500.0 + 50 * i for i in range(21)]


def _ts(d: date, i: int) -> datetime:
    return (pd.Timestamp(datetime.combine(d, time(9, 15)), tz=IST) + pd.Timedelta(minutes=5 * i)).to_pydatetime()


def _spot_candles(days=DAYS, seed=1):
    rng = np.random.default_rng(seed)
    out, price = [], 24000.0
    for d in days:
        for i in range(75):
            o = price
            c = o + rng.normal(0, 8)
            out.append(HistoricalCandle(_ts(d, i), NIFTY_KEY, o, max(o, c) + 1, min(o, c) - 1, c, 0.0, None))
            price = c
    return out


def _contract(expiry, strike, kind, source):
    tag = "EXP" if source == EXPIRED else "ACT"
    return HistoricalContract(instrument_key=f"NSE_FO|{tag}|{expiry}|{strike:g}|{kind}", underlying="NIFTY",
                              expiry=expiry, instrument_type=kind, lot_size=65, source=source, strike=strike,
                              trading_symbol=f"NIFTY {strike:g} {kind} {expiry}")


class FakeCatalog:
    def __init__(self, *, expiries_error=None, chain_errors=()):
        self.expiries_error = expiries_error
        self.chain_errors = set(chain_errors)
        self.calls = []

    def today(self):
        return TODAY

    def underlying_key(self, name):
        return NIFTY_KEY

    def active_contracts(self):
        return [_contract(e, s, k, ACTIVE) for e in ACTIVE_EXP for s in STRIKES for k in ("CE", "PE")]

    def expired_expiries(self, name):
        self.calls.append(("expiries", name))
        if self.expiries_error:
            raise self.expiries_error
        return list(EXPIRED_EXP)

    def expired_option_contracts(self, name, expiry):
        self.calls.append(("chain", expiry))
        if expiry in self.chain_errors:
            raise HistoricalDataError("PLAN_NOT_SUPPORTED", "Plus plan required", status=403)
        return [_contract(expiry, s, k, EXPIRED) for s in STRIKES for k in ("CE", "PE")]


class FakeClient:
    def __init__(self, *, spot=None, errors=None, empty=(), dup=False, invalid=False, skip_slots=()):
        self.spot = spot if spot is not None else _spot_candles()
        self.errors = errors or {}
        self.empty = set(empty)
        self.dup, self.invalid, self.skip_slots = dup, invalid, set(skip_slots)
        self.calls = []

    def _options(self, key, start, end):
        if key in self.empty:
            return []
        out = []
        for d in DAYS:
            if start <= d <= end:
                for i in range(75):
                    if i in self.skip_slots:
                        continue
                    p = 100.0 + i * 0.1
                    out.append(HistoricalCandle(_ts(d, i), key, p, p + 1, p - 1, p + 0.5, 10.0, 1000.0))
        return out[::-1]

    def _maybe_fail(self, key):
        if key in self.errors:
            raise self.errors[key]

    def candles(self, key, start, end, *, interval=5, unit="minutes"):
        self.calls.append(("v3", key, start, end))
        self._maybe_fail(key)
        if key == NIFTY_KEY:
            rows = [c for c in self.spot if start <= c.timestamp.date() <= end][::-1]
            if self.dup and rows:
                rows = rows + rows[:5]
            if self.invalid and rows:
                bad = rows[0]
                rows[0] = HistoricalCandle(bad.timestamp, bad.instrument_key, bad.open, bad.low - 5, bad.low,
                                           bad.close, 0.0, None)
            return rows
        return self._options(key, start, end)

    def expired_candles(self, key, start, end, *, interval=5):
        self.calls.append(("expired", key, start, end))
        self._maybe_fail(key)
        return self._options(key, start, end)


def _run(tmp_path, client=None, catalog=None, **kw):
    layout = u6.Layout(tmp_path / "upstox_6m")
    s = u6.run_download(client=client or FakeClient(), catalog=catalog or FakeCatalog(), layout=layout, start=START,
                        end=END, report_dir=tmp_path / "reports", **kw)
    return layout, s


def _item(tmp_path, key="NSE_FO|EXP|x", month="2026-04", source=EXPIRED, kind="CE", start=date(2026, 4, 27),
          end=date(2026, 4, 28), expiry=date(2026, 4, 28)):
    return u6.DownloadItem(instrument_key=key, instrument_type=kind, source=source, month=month, start=start, end=end,
                           path=tmp_path / "options" / month / "f.parquet", underlying="NIFTY", expiry=expiry,
                           strike=24000.0, trading_symbol="NIFTY 24000 CE")


# ------------------------------------------------------------------ chunking / date boundaries
def test_monthly_chunks_cover_target_period_exactly():
    w = u6.month_windows(date(2026, 4, 1), date(2026, 9, 30))
    assert [(m, a.isoformat(), b.isoformat()) for m, a, b in w] == [
        ("2026-04", "2026-04-01", "2026-04-30"), ("2026-05", "2026-05-01", "2026-05-31"),
        ("2026-06", "2026-06-01", "2026-06-30"), ("2026-07", "2026-07-01", "2026-07-31"),
        ("2026-08", "2026-08-01", "2026-08-31"), ("2026-09", "2026-09-01", "2026-09-30")]


def test_partial_month_boundaries_and_no_overlap():
    w = u6.month_windows(date(2026, 4, 15), date(2026, 6, 10))
    assert [(a, b) for _, a, b in w] == [(date(2026, 4, 15), date(2026, 4, 30)), (date(2026, 5, 1), date(2026, 5, 31)),
                                         (date(2026, 6, 1), date(2026, 6, 10))]
    for (_, _, b), (_, a2, _) in zip(w, w[1:]):
        assert a2 == b + timedelta(days=1)
    assert u6._month_end("2026-02") == date(2026, 2, 28)


def test_underlying_requests_are_monthly_and_calendar_days_not_assumed(tmp_path):
    client = FakeClient()
    layout, s = _run(tmp_path, client)
    v3 = [c for c in client.calls if c[1] == NIFTY_KEY]
    assert [(c[2], c[3]) for c in v3] == [(START, date(2026, 4, 30)), (date(2026, 5, 1), END)]
    assert s["underlying"]["trading_days"] == len(DAYS)
    assert HOLIDAY.isoformat() in s["underlying"]["weekdays_without_candles"]
    assert sorted(p.name for p in layout.underlying_files()) == ["2026-04.parquet", "2026-05.parquet"]


# ------------------------------------------------------------------ API parsing / retry
class _Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        return self._body


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, headers, params))
        return self.responses.pop(0)


def _ok(rows):
    return _Resp(200, {"status": "success", "data": {"candles": rows}})


def test_api_response_parsing_and_v3_path():
    rows = [["2026-04-01T09:20:00+05:30", 10, 12, 9, 11, 100, 5], ["2026-04-01T09:15:00+05:30", 9, 11, 8, 10, 50, 4]]
    sess = _Session([_ok(rows)])
    c = UpstoxHistoricalClient(lambda: TOKEN, session=sess, min_interval_seconds=0, sleep=lambda s: None)
    got = c.candles(NIFTY_KEY, date(2026, 4, 1), date(2026, 4, 30), interval=5)
    assert sess.calls[0][0].endswith("/v3/historical-candle/NSE_INDEX%7CNifty%2050/minutes/5/2026-04-30/2026-04-01")
    assert [g.close for g in got] == [11.0, 10.0] and got[0].open_interest == 5.0
    assert str(got[0].timestamp.utcoffset()) == "5:30:00"
    clean, raw = u6.prepare_candles(got, date(2026, 4, 1), date(2026, 4, 30))
    assert list(clean["close"]) == [10.0, 11.0] and raw["raw_rows"] == 2


def test_expired_candle_endpoint_path():
    sess = _Session([_ok([["2026-04-28T09:15:00+05:30", 1, 2, 0.5, 1.5, 10, 100]])])
    c = UpstoxHistoricalClient(lambda: TOKEN, session=sess, min_interval_seconds=0, sleep=lambda s: None)
    c.expired_candles("NSE_FO|12345|28-04-2026", date(2026, 4, 27), date(2026, 4, 28), interval=5)
    assert "/v2/expired-instruments/historical-candle/NSE_FO%7C12345%7C28-04-2026/5minute/2026-04-28/2026-04-27" \
        in sess.calls[0][0]


def test_retry_is_exponential_and_bounded():
    sleeps = []
    sess = _Session([_Resp(429, {}), _Resp(503, {}), _ok([])])
    c = UpstoxHistoricalClient(lambda: TOKEN, session=sess, min_interval_seconds=0, sleep=sleeps.append,
                               backoff_seconds=1.0)
    assert c.candles(NIFTY_KEY, date(2026, 4, 1), date(2026, 4, 2)) == []
    assert sleeps == [1.0, 2.0]
    sess2 = _Session([_Resp(500, {}) for _ in range(10)])
    c2 = UpstoxHistoricalClient(lambda: TOKEN, session=sess2, min_interval_seconds=0, sleep=lambda s: None,
                                max_retries=3)
    with pytest.raises(HistoricalDataError) as e:
        c2.candles(NIFTY_KEY, date(2026, 4, 1), date(2026, 4, 2))
    assert e.value.code == "UPSTOX_SERVER_ERROR" and len(sess2.calls) == 4


# ------------------------------------------------------------------ parquet / resume
def test_parquet_round_trip_keeps_ist_timestamps(tmp_path):
    clean, _ = u6.prepare_candles(_spot_candles(DAYS[:1]), DAYS[0], DAYS[0])
    p = tmp_path / "x" / "2026-04.parquet"
    u6.write_parquet(p, clean)
    back = u6.read_parquet(p)
    assert len(back) == 75 and str(back["timestamp"].dt.tz) == "Asia/Kolkata"
    assert (back["timestamp"] == clean["timestamp"]).all() and not p.with_suffix(".parquet.tmp").exists()


def test_resume_skips_valid_files_and_redownloads_corrupt_ones(tmp_path):
    client = FakeClient()
    layout, _ = _run(tmp_path, client)
    first = len(client.calls)
    client.calls.clear()
    _run(tmp_path, client)
    assert not [c for c in client.calls if c[0] in ("v3", "expired")], "valid files must not be re-downloaded"
    bad = layout.underlying_files()[0]
    bad.write_bytes(b"not parquet")
    client.calls.clear()
    _run(tmp_path, client, resume=True)
    assert [(c[1], c[2]) for c in client.calls] == [(NIFTY_KEY, START)]
    assert first > 2


def test_resume_skips_terminal_statuses_but_default_retries(tmp_path):
    m = u6.Manifest(tmp_path / "manifest.csv")
    it = _item(tmp_path)
    client = FakeClient(empty={it.instrument_key})
    row, action = u6.fetch_item(client, it, m)
    assert action == u6.EMPTY and row["status"] == u6.EMPTY and not it.path.exists()
    calls = len(client.calls)
    _, action = u6.fetch_item(client, it, u6.Manifest(tmp_path / "manifest.csv"), resume=True)
    assert action == "SKIPPED_TERMINAL" and len(client.calls) == calls
    client.empty.clear()
    row, action = u6.fetch_item(client, it, u6.Manifest(tmp_path / "manifest.csv"))
    assert action == u6.SUCCESS and row["candle_count"] == 150


def test_manifest_has_required_columns_and_one_row_per_window(tmp_path):
    layout, s = _run(tmp_path)
    rows = u6._read_csv(layout.manifest)
    required = ["instrument_key", "instrument_type", "expiry", "strike", "option_type", "month", "from_date",
                "to_date", "candle_count", "status", "http_status", "download_timestamp", "file_path"]
    assert set(required) <= set(rows[0])
    assert len({(r["instrument_key"], r["month"]) for r in rows}) == len(rows)
    assert {r["status"] for r in rows} <= set(u6.STATUSES)


# ------------------------------------------------------------------ duplicates / OHLC
def test_duplicate_candles_removed_before_storage(tmp_path):
    layout, s = _run(tmp_path, FakeClient(dup=True))
    f = u6.read_parquet(layout.underlying_files()[0])
    assert not f["timestamp"].duplicated().any()
    assert s["quality"]["duplicates_removed"] == 10  # 5 per monthly chunk
    assert s["quality"]["aggregate"]["duplicate_timestamps"] == 0 and s["underlying"]["candles"] == 75 * len(DAYS)


def test_invalid_ohlc_rows_are_rejected_not_repaired(tmp_path):
    layout, s = _run(tmp_path, FakeClient(invalid=True))
    assert s["quality"]["raw_rejected_rows"] == 2  # one per monthly chunk
    assert s["quality"]["aggregate"]["invalid_ohlc"] == 0
    assert s["underlying"]["missing_intervals"] == 2 and s["underlying"]["status"] == "INCOMPLETE"
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-04-27 09:20", "2026-04-27 09:15", "2026-04-27 09:15"]),
                          "open": [10, 10, 10], "high": [9, 11, 11], "low": [8, 9, 9], "close": [10, 10, 10],
                          "volume": [-1, 5, 5], "open_interest": [1, -2, -2]})
    q = u6.quality_check(frame)
    assert not q["tz_ist"] and not q["sorted"] and q["duplicate_timestamps"] == 1 and q["duplicate_rows"] == 1
    assert q["invalid_ohlc"] == 1 and q["negative_volume"] == 1 and q["negative_oi"] == 1 and not q["ok"]


# ------------------------------------------------------------------ contract plan (point-in-time)
def _plan(catalog=None, start=START):
    clean, _ = u6.prepare_candles(_spot_candles(), START, END)
    disc = u6.discover_contracts(catalog or FakeCatalog(), "NIFTY", start, END)
    reqs, unresolved = u6.plan_requirements(clean, disc)
    return clean, disc, reqs, unresolved


def test_nearest_listed_expiry_per_session_and_dte():
    _, disc, reqs, unresolved = _plan()
    assert not unresolved
    exp = {r.trading_date: r.contract.expiry for r in reqs}
    assert exp[date(2026, 4, 27)] == date(2026, 4, 28) and exp[date(2026, 4, 28)] == date(2026, 4, 28)
    assert exp[date(2026, 4, 29)] == date(2026, 5, 5) and exp[date(2026, 5, 5)] == date(2026, 5, 5)
    assert exp[date(2026, 5, 6)] == date(2026, 5, 12)
    for r in reqs:
        assert r.dte == (r.contract.expiry - r.trading_date).days >= 0
        assert r.contract.source == (EXPIRED if r.contract.expiry < TODAY else ACTIVE)
    assert date(2026, 4, 21) not in {r.contract.expiry for r in reqs}


def test_strikes_match_s2_selection_rules_atm_itm_otm():
    clean, disc, reqs, _ = _plan()
    for d in DAYS:
        closes = clean[clean["timestamp"].dt.date == d]["close"]
        expiry = next(r.contract.expiry for r in reqs if r.trading_date == d)
        for kind in ("CE", "PE"):
            want = set()
            for c in closes:
                for mode, off in (("ATM", 0), ("ITM", 1), ("OTM", 1)):
                    want.add(select_strike(STRIKES, option_type=kind, mode=mode, spot=float(c), offset=off))
            got = {r.contract.strike for r in reqs if r.trading_date == d and r.contract.instrument_type == kind}
            assert got == want, (d, kind)
            assert all(r.contract.expiry == expiry for r in reqs if r.trading_date == d)
    roles = {(r.contract.instrument_type, r.role) for r in reqs}
    assert roles == {("CE", "ATM"), ("CE", "ITM"), ("CE", "OTM"), ("PE", "ATM"), ("PE", "ITM"), ("PE", "OTM")}
    for r in reqs:
        lo_atm = min(x.contract.strike for x in reqs if x.trading_date == r.trading_date and x.role == "ATM"
                     and x.contract.instrument_type == r.contract.instrument_type)
        if r.role != "ATM":
            below = r.contract.strike < lo_atm
            assert r.role == (("ITM" if below else "OTM") if r.contract.instrument_type == "CE"
                              else ("OTM" if below else "ITM"))


def test_contract_metadata_written(tmp_path):
    layout, _ = _run(tmp_path)
    rows = u6._read_csv(layout.contracts)
    assert set(rows[0]) == set(u6.CONTRACT_COLUMNS)
    for r in rows:
        assert r["instrument_key"] and r["option_type"] in ("CE", "PE") and r["underlying"] == "NIFTY"
        assert int(r["dte"]) == (date.fromisoformat(r["expiry"]) - date.fromisoformat(r["trading_date"])).days
    back = u6.requirements_from_rows(rows)
    assert len(back) == len(rows) and back[0].contract.expiry == date.fromisoformat(rows[0]["expiry"])


def test_without_expired_calendar_past_sessions_are_unresolved_not_mapped_to_live_contracts():
    err = HistoricalDataError("PLAN_NOT_SUPPORTED", "Plus plan required", status=403)
    _, disc, reqs, unresolved = _plan(FakeCatalog(expiries_error=err))
    assert disc.calendar_from == TODAY
    assert not reqs and {u["reason"] for u in unresolved} == {"EXPIRY_CALENDAR_INCOMPLETE"}
    assert len(unresolved) == len(DAYS)


def test_failed_chain_is_unresolved_never_replaced_by_next_expiry():
    _, _, reqs, unresolved = _plan(FakeCatalog(chain_errors={date(2026, 5, 5)}))
    bad_days = {d for d in DAYS if date(2026, 4, 29) <= d <= date(2026, 5, 5)}
    assert {u["trading_date"] for u in unresolved} == bad_days
    assert {u["reason"] for u in unresolved} == {"CONTRACT_LIST_UNAVAILABLE"}
    assert not {r.trading_date for r in reqs} & bad_days


# ------------------------------------------------------------------ expired / missing handling
def test_expired_contracts_use_expired_endpoint_and_live_use_v3(tmp_path):
    client = FakeClient()
    _run(tmp_path, client)
    exp = [c for c in client.calls if c[0] == "expired"]
    v3 = [c for c in client.calls if c[0] == "v3" and c[1] != NIFTY_KEY]
    assert exp and all("|EXP|" in c[1] for c in exp)
    assert v3 and all("|ACT|" in c[1] for c in v3)


def test_unavailable_contract_recorded_and_not_fabricated(tmp_path):
    _, _, reqs, _ = _plan()
    key = next(r.contract.instrument_key for r in reqs if r.role == "ATM" and r.contract.source == EXPIRED)
    err = HistoricalDataError("PLAN_NOT_SUPPORTED", "Plus plan required", status=403)
    layout, s = _run(tmp_path, FakeClient(errors={key: err}))
    rows = [r for r in u6._read_csv(layout.manifest) if r["instrument_key"] == key]
    assert rows and all(r["status"] == u6.UNAVAILABLE and r["http_status"] == "403" for r in rows)
    assert not any(Path(r["file_path"]).exists() for r in rows)
    missing = [r for r in u6._read_csv(layout.missing) if r["instrument_key"] == key]
    assert missing and all(r["reason"].startswith("UNAVAILABLE") for r in missing)
    assert key in s["options"]["missing_contracts"]


def test_missing_intervals_reported_never_filled(tmp_path):
    layout, s = _run(tmp_path, FakeClient(skip_slots={10, 11}))
    f = u6.read_parquet(layout.option_files()[0])
    assert (f.groupby(f["timestamp"].dt.date).size() == 73).all()
    gaps = [r for r in u6._read_csv(layout.missing) if r["reason"] == "MISSING_INTERVALS" and r["scope"] != "UNDERLYING"]
    assert gaps and all(r["count"] == "2" for r in gaps)
    assert s["options"]["atm_interval_completeness"] == pytest.approx(73 / 75)


def test_fatal_auth_error_stops_and_is_recorded(tmp_path):
    err = HistoricalDataError("AUTH_FAILED", "Upstox rejected the access token", status=401)
    with pytest.raises(HistoricalDataError):
        _run(tmp_path, FakeClient(errors={NIFTY_KEY: err}))
    rows = u6._read_csv(tmp_path / "upstox_6m" / "metadata" / "download_manifest.csv")
    assert rows[0]["status"] == u6.FAILED and rows[0]["error_code"] == "AUTH_FAILED"


# ------------------------------------------------------------------ coverage verdicts / reports
def test_full_fake_coverage_is_sufficient_and_reports_written(tmp_path):
    layout, s = _run(tmp_path)
    assert s["underlying"]["status"] == "OK" and s["options"]["status"] == "OK"
    assert s["option_coverage"] == "SUFFICIENT" and s["quality"]["pass"]
    assert s["options"]["ce"] > 0 and s["options"]["pe"] > 0 and s["options"]["candles"] > 0
    for k in ("quality", "coverage", "contracts", "missing", "manifest", "summary"):
        assert Path(s["paths"][k]).is_file()
    cov = Path(s["paths"]["coverage"]).read_text(encoding="utf-8")
    assert "## Month by month" in cov and "2026-04" in cov and "2026-05" in cov


def test_missing_option_premiums_make_coverage_insufficient(tmp_path):
    _, _, reqs, _ = _plan()
    atm = {r.contract.instrument_key for r in reqs if r.role == "ATM"}
    layout, s = _run(tmp_path, FakeClient(empty=atm))
    assert s["underlying"]["status"] == "OK"
    assert s["options"]["status"] == "INCOMPLETE" and s["option_coverage"] == "INSUFFICIENT"
    assert u6.INSUFFICIENT_TEXT in Path(s["paths"]["coverage"]).read_text(encoding="utf-8")


def test_validate_mode_makes_no_api_calls(tmp_path):
    _, s1 = _run(tmp_path)

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"API used in --validate: {name}")

    _, s2 = _run(tmp_path, client=Boom(), catalog=Boom(), validate_only=True)
    assert s2["options"]["contracts"] == s1["options"]["contracts"]
    assert s2["option_coverage"] == s1["option_coverage"]


# ------------------------------------------------------------------ token safety
def test_token_never_logged_or_in_errors(caplog):
    caplog.set_level(logging.DEBUG)
    sess = _Session([_Resp(401, {"status": "error", "errors": [{"errorCode": "UDAPI100050", "message": "Invalid token"}]})])
    c = UpstoxHistoricalClient(lambda: TOKEN, session=sess, min_interval_seconds=0, sleep=lambda s: None)
    with pytest.raises(HistoricalDataError) as e:
        c.candles(NIFTY_KEY, date(2026, 4, 1), date(2026, 4, 2))
    assert sess.calls[0][1]["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in str(e.value) and TOKEN not in repr(c) and TOKEN not in caplog.text


def test_no_token_in_any_output_and_leak_check_detects_planted_token(tmp_path, capsys):
    layout, s = _run(tmp_path, progress=print)
    ok, leaks = u6.token_leak_check([layout.root, tmp_path / "reports"], [TOKEN])
    assert ok and not leaks
    assert TOKEN not in capsys.readouterr().out
    (layout.metadata / "oops.txt").write_text(f"x {TOKEN} y", encoding="utf-8")
    ok, leaks = u6.token_leak_check([layout.root], [TOKEN])
    assert not ok and leaks == [(layout.metadata / "oops.txt").as_posix()]
    assert u6.token_leak_check([layout.root], ["x "]) == (True, [])  # values shorter than 8 chars are ignored


# ------------------------------------------------------------------ safety
def test_paper_only_and_no_order_code():
    assert live_gate.trading_mode() == "PAPER" and live_gate.live_trading_enabled() is False
    src = inspect.getsource(u6)
    for bad in ("place_order", "OrderManager", "requests.post", ".post("):
        assert bad not in src
    script = (Path(__file__).resolve().parents[1] / "scripts" / "download_upstox_6m.py").read_text(encoding="utf-8")
    assert "assert_paper_trading_only" in script and "order_placement_blocked" in script
    assert "place_order" not in script and "print(f\"Token source variable: {source}" in script
