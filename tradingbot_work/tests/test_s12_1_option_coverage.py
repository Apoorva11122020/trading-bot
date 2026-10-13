from __future__ import annotations

import inspect
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s12_1_option_coverage as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import HistoricalContract
from src.options.selection import atm_strike
from src.research import s12_1_option_coverage as s121
from src.research import s12_data_quality as s12
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures, bars

TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S12-1"
SYNTH_PERIOD = (date(2026, 4, 1), date(2026, 9, 30))
SYNTH_WINDOW = (date(2026, 4, 1), date(2026, 5, 31))
GAP_DAY = date(2026, 5, 4)


def _boom(*_a, **_k):
    raise AssertionError("S12.1 must read only the stored research roots")


def _gap_contracts(root: Path) -> list[HistoricalContract]:
    reqs = u6.requirements_from_rows(u6._read_csv(u6.Layout(root).contracts))
    return [r.contract for r in reqs if r.trading_date == GAP_DAY and r.role == u6.ATM
            and r.contract.instrument_type == "PE"]


def _files(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    b = tmp_path_factory.mktemp("s121")
    src = b / "src" / "upstox_6m"
    make_store(src)
    add_futures(src)
    gone = _gap_contracts(src)
    assert gone
    lay = u6.Layout(src)
    for c in gone:
        lay.option_file(u6.month_label(GAP_DAY), c).unlink()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s12, "PERIOD", SYNTH_PERIOD)
        mp.setattr(s12, "MIN_CLEAN_MONTHS", 1)
        mp.setattr(s12, "MIN_ATM_ELIGIBLE_SHARE", 0.5)
        a = s12.build_audit([src])
        s12.write_reports(a, b / "s12")
    return SimpleNamespace(base=b, src=src, gone=gone, s12_manifest=b / "s12" / "clean_history_manifest.json",
                           s12=a)


def _run(base, name: str, manifest_status: str | None):
    root = base.base / name / "upstox_6m"
    shutil.copytree(base.src, root)
    if manifest_status:
        m = u6.Manifest(u6.Layout(root).manifest)
        for c in base.gone:
            m.upsert({"instrument_key": c.instrument_key, "instrument_type": c.instrument_type,
                      "expiry": c.expiry.isoformat(), "strike": c.strike, "option_type": c.instrument_type,
                      "month": u6.month_label(GAP_DAY), "from_date": "2026-05-01", "to_date": "2026-05-29",
                      "candle_count": 0, "status": manifest_status, "http_status": 200, "source": "EXPIRED",
                      "trading_symbol": c.trading_symbol, "raw_rows": 0,
                      "detail": "Upstox returned no valid candles in the window"})
    files_before = _files(root)
    before = clean_script.production_hashes()
    out = base.base / name / "out"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(s121, "WINDOW", SYNTH_WINDOW)
        a = script.run(expanded_root=base.base / name / "upstox_expanded", data_root=root,
                       s12_manifest=base.s12_manifest, out_dir=out, secret_values=[TOKEN])
    return SimpleNamespace(a=a, root=root, out=out, files_before=files_before, files_after=_files(root),
                           prod_before=before)


@pytest.fixture(scope="module")
def limited(base):
    return _run(base, "limited", u6.EMPTY)


@pytest.fixture(scope="module")
def recoverable(base):
    return _run(base, "recoverable", None)


# 1 ------------------------------------------------------------------ session-level coverage
def test_session_level_coverage(limited):
    a = limited.a
    cols = ("date", "eligible_nifty_timestamps", "atm_CE_expected", "atm_CE_available", "atm_PE_expected",
            "atm_PE_available", "CE_coverage_pct", "PE_coverage_pct", "missing_CE_count", "missing_PE_count")
    days = {r["date"]: r for r in a["days"]}
    assert days and all(c in r for r in a["days"] for c in cols)
    assert min(days) >= SYNTH_WINDOW[0].isoformat() and max(days) <= SYNTH_WINDOW[1].isoformat()
    g = days[GAP_DAY.isoformat()]
    assert g["atm_PE_available"] == 0 and g["missing_PE_count"] == g["eligible_nifty_timestamps"] > 0
    assert g["atm_CE_eligible"] > 0 and not g["option_session_clean"]
    for r in a["days"]:
        for s in ("CE", "PE"):
            assert r[f"atm_{s}_expected"] == r[f"atm_{s}_available"] + r[f"missing_{s}_count"]
            assert r[f"atm_{s}_eligible"] + r[f"{s}_atr_warmup"] == r[f"atm_{s}_available"]
    t = a["totals"]["PE"]
    assert t["expected"] == sum(r["atm_PE_expected"] for r in a["days"])
    assert (limited.out / "pre_sep_atm_coverage.csv").read_text() == (limited.out / "coverage_by_day.csv").read_text()


# 2 ------------------------------------------------------------------ missing contract identification
def test_missing_contract_identification(limited, base):
    a, m = limited.a, limited.a["_market"]
    keys = {c.instrument_key for c in base.gone}
    assert a["missing"]
    for r in a["missing"]:
        assert r["side"] == "PE" and r["instrument_key"] in keys
        assert r["classification"] == s121.EXISTS_MISSING and r["contract_in_metadata"]
        d = date.fromisoformat(r["date"])
        chain = m.chains[(d, date.fromisoformat(r["expected_expiry"]), "PE")]
        assert float(atm_strike(sorted(chain), r["spot"])) == r["expected_atm_strike"]
        assert chain[r["expected_atm_strike"]].instrument_key == r["instrument_key"]
        for k in ("timestamp", "spot", "trading_symbol", "s12_reason"):
            assert r[k] is not None


# 3/4 ------------------------------------------------------------------ download audit + source limitation
def test_download_audit_source_empty(limited, base):
    a = limited.a
    keys = {c.instrument_key for c in base.gone}
    bad = [x for x in a["audit"] if x["verdict"] != s121.OK]
    assert bad and {x["instrument_key"] for x in bad} <= keys
    for x in bad:
        assert x["discovered"] and x["requested"] and x["returned_empty"] and not x["successfully_downloaded"]
        assert x["verdict"] == s121.SOURCE_EMPTY and x["source_limited"] and not x["recoverable"]
        assert x["bars_on_session"] == 0 and not x["file_exists"]
    assert all(r["root_cause"] == s121.SOURCE_LIMIT and r["cause_letter"] == "B" for r in a["missing"])
    assert a["audit_verdicts"][s121.OK] > 0


def _req(root: Path, d: date, key: str = "NSE_FO|1|28-04-2026") -> dict:
    c = HistoricalContract(instrument_key=key, underlying="NIFTY", expiry=date(2026, 4, 28), instrument_type="PE",
                           lot_size=65, source="EXPIRED", strike=24000.0, trading_symbol="NIFTY 24000 PE 28 APR 26")
    return {"root": root, "req": u6.Requirement(d, c, u6.ATM, (c.expiry - d).days)}


def test_download_verdicts(tmp_path):
    d = date(2026, 4, 6)
    full = bars([d], base=150.0)
    short = full.iloc[:70]
    cases = {  # status, stored bars, source-reported missing intervals -> verdict
        "ok": (u6.SUCCESS, full, False, s121.OK),
        "notreq": (None, None, False, s121.NOT_REQUESTED),
        "failed": (u6.FAILED, None, False, s121.REQUEST_FAILED),
        "empty": (u6.EMPTY, None, False, s121.SOURCE_EMPTY),
        "unavail": (u6.UNAVAILABLE, None, False, s121.SOURCE_EMPTY),
        "partial": (u6.SUCCESS, short, True, s121.SOURCE_INCOMPLETE),
        "lost": (u6.SUCCESS, None, False, s121.STORED_LOST),
    }
    for name, (status, frame, src_missing, want) in cases.items():
        key = f"NSE_FO|{name}|28-04-2026"
        meta = {"reqs": {(d, key): _req(tmp_path, d, key)},
                "manifest": {(key, "2026-04"): [{"status": status, "from_date": "2026-04-01", "to_date": "2026-04-30",
                                                 "http_status": 200, "raw_rows": 0}]} if status else {},
                "missing": {(key, d): [{"reason": "MISSING_INTERVALS", "count": 5}]} if src_missing else {}}
        mkt = SimpleNamespace(premium=lambda _k, f=frame: f)
        out = s121.download_audit(mkt, meta, {}, set())
        assert out[0]["verdict"] == want, name
        assert out[0]["recoverable"] == (want in s121.RECOVERABLE_VERDICTS), name
    meta = {"reqs": {}, "manifest": {}, "missing": {}}
    out = s121.download_audit(SimpleNamespace(premium=lambda _k: None), meta, {}, {(d, "NSE_FO|x|28-04-2026")})
    assert out[0]["verdict"] == s121.NOT_DISCOVERED and not out[0]["discovered"]


def test_no_substitution_no_fabrication(limited):
    a = limited.a
    assert a["raw_unchanged"] and a["fabricated_values"] == 0 and a["substituted_contracts"] == 0
    assert limited.files_after == {k: v for k, v in limited.files_after.items() if k in limited.files_before}
    assert {k: v for k, v in limited.files_before.items() if not k.startswith("metadata/")} == \
        {k: v for k, v in limited.files_after.items() if not k.startswith("metadata/")}
    m = a["_market"]
    for key in {r["instrument_key"] for r in a["missing"]}:
        assert m.premium(key) is None
    src = inspect.getsource(s121)
    for bad in ("ffill(", "bfill(", "interpolate(", ".resample(", "write_parquet(", "to_parquet("):
        assert bad not in src, bad


# 5 ------------------------------------------------------------------ point-in-time validation
def test_point_in_time_validation(limited):
    p = limited.a["pit"]
    assert p["passed"] and p["rows"] and not p["failed_sessions"]
    for r in p["rows"]:
        assert r["expiry"] >= r["date"] and r["contracts_from_expired_endpoint"] and r["key_encodes_same_expiry"]
    assert s121.key_expiry("NSE_FO|64695|30-09-2025") == date(2025, 9, 30)
    assert s121.key_expiry("NSE_FO|64695") is None


def test_point_in_time_detects_bad_metadata(tmp_path):
    d = date(2026, 4, 6)
    good = _req(tmp_path, d)["req"].contract
    wrong_key = HistoricalContract(instrument_key="NSE_FO|2|26-05-2026", underlying="NIFTY", expiry=good.expiry,
                                   instrument_type="CE", lot_size=65, source="EXPIRED", strike=24000.0,
                                   trading_symbol="X")
    live = HistoricalContract(instrument_key="NSE_FO|3", underlying="NIFTY", expiry=good.expiry, instrument_type="CE",
                              lot_size=65, source="ACTIVE", strike=24025.0, trading_symbol="Y")
    mkt = SimpleNamespace(calendar=[date(2026, 4, 28)])
    plans = {"r": {"calendar_from": "2026-04-01"}}
    ok = s121.pit_validation(mkt, [d], {"by_day": {d: [good]}, "plans": plans}, [])
    assert ok["passed"]
    bad = s121.pit_validation(mkt, [d], {"by_day": {d: [good, wrong_key, live]}, "plans": plans}, [])
    r = bad["rows"][0]
    assert not bad["passed"] and not r["key_encodes_same_expiry"] and not r["contracts_from_expired_endpoint"]
    assert not r["strikes_on_50_grid"]
    late = s121.pit_validation(mkt, [d], {"by_day": {d: [good]}, "plans": {"r": {"calendar_from": "2026-05-01"}}}, [])
    assert not late["passed"] and not late["rows"][0]["calendar_recorded_from_plan"]


def test_classify_reasons(tmp_path):
    d = date(2026, 4, 6)
    r = _req(tmp_path, d)
    c = r["req"].contract
    meta = {"reqs": {(d, c.instrument_key): r}}
    assert s121.classify("OPTION_DATA_MISSING", c, d, meta) == s121.EXISTS_MISSING
    assert s121.classify("OPTION_SIGNAL_CANDLE_MISSING", c, d, meta) == s121.EXISTS_MISSING
    assert s121.classify("OPTION_DATA_MISSING", c, d, {"reqs": {}}) == s121.NOT_AVAILABLE
    assert s121.classify("CONTRACT_LIST_UNAVAILABLE", None, d, meta) == s121.NOT_AVAILABLE
    assert s121.classify("OUTSIDE_PLANNED_STRIKES", c, d, meta) == s121.SELECTION
    assert s121.classify("ATM_NOT_BRACKETED", c, d, meta) == s121.SELECTION
    assert s121.classify("STRIKE_UNAVAILABLE:X", None, d, meta) == s121.SELECTION
    assert s121.classify("NO_LISTED_EXPIRY", None, d, meta) == s121.EXPIRY_META
    assert s121.classify("EXPIRY_NOT_POINT_IN_TIME", c, d, meta) == s121.EXPIRY_META


# 6 ------------------------------------------------------------------ decision
def test_decision_source_limited(limited):
    a = limited.a
    d, h = a["decision"], a["history"]
    assert d["decision"] == s121.SOURCE_LIMITED and GAP_DAY.isoformat() in d["blocking_sessions"]
    assert not d["recoverable_sessions"] and not d["contracts_to_download"]
    assert d["earliest_clean_usable_date"] == h["now"]["start"] > max(d["blocking_sessions"])
    assert h["if_recovered"] == h["now"]
    assert h["pre_window_segment"]["start"] == "2026-04-01"
    assert h["pre_window_segment"]["end"] < min(d["blocking_sessions"])
    assert d["required_months"] == s12.MIN_CLEAN_MONTHS == 12
    assert a["s12_consistent"]


def test_decision_recoverable(recoverable, base):
    a = recoverable.a
    d, h = a["decision"], a["history"]
    assert d["decision"] == s121.RECOVERABLE
    assert all(x["verdict"] == s121.NOT_REQUESTED for x in a["audit"] if x["verdict"] != s121.OK)
    w = d["contracts_to_download"]
    assert w and {x["instrument_key"] for x in w} <= {c.instrument_key for c in base.gone}
    assert {x["from"] for x in w} <= set(d["blocking_sessions"]) and all(x["reason"] == s121.NOT_REQUESTED for x in w)
    assert h["if_recovered"]["start"] == "2026-04-01" and h["if_recovered"]["months"] > h["now"]["months"]
    assert all(r["cause_letter"] == "A" for r in a["missing"])


def test_decide_rules():
    hist = {"now": {"start": "2025-09-26", "months": 9.1}, "if_recovered": {"start": "2025-01-01", "months": 18.0},
            "required_months": 12}
    rec = {"date": "2025-09-25", "recoverable": True}
    src = {"date": "2025-06-02", "recoverable": False}
    assert s121.decide([], hist, [])["decision"] == s121.NO_GAP
    assert s121.decide([rec], hist, [])["decision"] == s121.RECOVERABLE
    assert s121.decide([src], hist, [])["decision"] == s121.SOURCE_LIMITED
    p = s121.decide([src, rec], hist, [])
    assert p["decision"] == s121.PARTIAL and p["gate_passes_if_recovered"] and not p["gate_passes_now"]
    days = [{"date": "2025-09-25", "s12_clean": False, "option_session_clean": False, "s12_reasons": "LOW_ATM_COVERAGE",
             "both_share": 0.45, "missing_CE_count": 0, "missing_PE_count": 33},
            {"date": "2025-09-26", "s12_clean": False, "option_session_clean": False,
             "s12_reasons": "LOW_ATM_COVERAGE FUTURES_BARS_MISSING", "both_share": 0.4, "missing_CE_count": 0,
             "missing_PE_count": 30}]
    miss = [{"date": "2025-09-25", "cause_letter": "A", "instrument_key": "k", "trading_symbol": "s", "side": "PE",
             "root_cause": s121.EXISTS_MISSING},
            {"date": "2025-09-26", "cause_letter": "A", "instrument_key": "k", "trading_symbol": "s", "side": "PE",
             "root_cause": s121.EXISTS_MISSING}]
    b = s121.blocking_sessions(days, miss)
    assert [x["recoverable"] for x in b] == [True, False]


def test_root_cause_mapping():
    rec = {"date": "d", "instrument_key": "k", "classification": s121.EXISTS_MISSING}
    for verdict, want in ((s121.SOURCE_EMPTY, (s121.SOURCE_LIMIT, "B")), (s121.SOURCE_INCOMPLETE, (s121.SOURCE_LIMIT, "E")),
                          (s121.NOT_REQUESTED, (s121.EXISTS_MISSING, "A")), (s121.REQUEST_FAILED, (s121.EXISTS_MISSING, "A")),
                          (s121.STORED_LOST, (s121.EXISTS_MISSING, "A"))):
        assert s121.root_cause(rec, {("d", "k"): {"verdict": verdict}}) == want
    assert s121.root_cause(dict(rec, classification=s121.SELECTION), {}) == (s121.SELECTION, "C")
    assert s121.root_cause(dict(rec, classification=s121.EXPIRY_META), {}) == (s121.EXPIRY_META, "D")
    assert s121.root_cause(dict(rec, classification=s121.NOT_AVAILABLE), {}) == (s121.NOT_AVAILABLE, "B")


# 7 ------------------------------------------------------------------ reports / safety
def test_reports_and_terminal(limited):
    a = limited.a
    for n in s121.REPORT_FILES:
        p = limited.out / n
        assert p.is_file(), n
        assert TOKEN not in p.read_text(encoding="utf-8")
    assert a["token_leak"]["passed"] and a["production_unchanged"] is True
    assert clean_script.production_hashes() == limited.prod_before
    lines = s121.terminal_text(a).splitlines()
    assert lines[0] == s121.HEADING
    for k, v in (("DECISION:", s121.SOURCE_LIMITED), ("FABRICATED VALUES:", "0"), ("SUBSTITUTED CONTRACTS:", "0"),
                 ("LIVE ORDERS:", "0"), ("PRODUCTION CHANGED:", "NO"), ("OCTOBER LOADED:", "NO"), ("S11 RERUN:", "NO"),
                 ("POINT-IN-TIME VALIDATION:", "PASS"), ("S12 CONSISTENT:", "YES")):
        assert lines[lines.index(k) + 1] == v, k
    assert any(l.startswith("EARLIEST CLEAN USABLE DATE: ") for l in lines)
    assert a["max_timestamp_loaded"] <= "2026-05-31T23:59"


def test_october_and_inputs_guard(tmp_path, base):
    with pytest.raises(SystemExit):
        s121.build([base.src], base.s12_manifest, hi=date(2026, 10, 1))
    with pytest.raises(SystemExit):
        s121.build([base.src], tmp_path / "missing.json", lo=SYNTH_WINDOW[0], hi=SYNTH_WINDOW[1])


def test_no_ml_no_orders_no_network_code():
    for mod in (s121, script):
        src = inspect.getsource(mod)
        for bad in ("sklearn", "place_order", "requests.", "UpstoxHistoricalClient(", "expired_candles(", "fit(",
                    "fetch_item("):
            assert bad not in src, (mod.__name__, bad)
    assert pd.Timestamp(s121.WINDOW[1]) < pd.Timestamp("2025-10-01")
