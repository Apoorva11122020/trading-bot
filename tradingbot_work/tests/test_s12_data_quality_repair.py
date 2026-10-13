from __future__ import annotations

import inspect
import json
import shutil
from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s12_data_quality as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.research import s10_ml_data as sd
from src.research import s11_history as hx
from src.research import s12_data_quality as s12
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures, bars, fut

IST = s12.IST
STEP = s12.STEP
TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S12"
SYNTH_PERIOD = (date(2026, 4, 1), date(2026, 9, 30))


def _boom(*_a, **_k):
    raise AssertionError("S12 must read only the stored research roots")


def at(d: date, hhmm: str) -> pd.Timestamp:
    return s12.at(d, time.fromisoformat(hhmm))


def spot_frame(days, drop=(), seed: int = 3) -> pd.DataFrame:
    f = bars(days, seed=seed)
    f = f[~f["timestamp"].isin(set(drop))].reset_index(drop=True)
    return f.assign(tz_error=False)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s12")
    six, out = base / "upstox_6m", base / "out"
    make_store(six)
    add_futures(six)
    raw_before = s12.raw_hashes([six], max_day=SYNTH_PERIOD[1])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(s12, "PERIOD", SYNTH_PERIOD)
        mp.setattr(s12, "MIN_CLEAN_MONTHS", 1)
        mp.setattr(s12, "MIN_ATM_ELIGIBLE_SHARE", 0.5)
        a = script.run(expanded_root=base / "upstox_expanded", data_root=six, out_dir=out, secret_values=[TOKEN])
    return SimpleNamespace(a=a, base=base, six=six, out=out, before=before, raw_before=raw_before)


# 1 ------------------------------------------------------------------ timestamp validity
def test_timestamp_validity(tmp_path):
    d = date(2026, 4, 6)
    good = bars([d], seed=1)
    naive = good.assign(timestamp=good["timestamp"].dt.tz_localize(None))
    utc = good.assign(timestamp=good["timestamp"].dt.tz_convert("UTC"))
    for name, f, expect in (("good", good, 0), ("naive", naive, len(good)), ("utc", utc, len(good))):
        p = tmp_path / f"{name}.parquet"
        f.to_parquet(p, index=False)
        r, late = s12.read_raw(p)
        assert int(r["tz_error"].sum()) == expect and late == 0, name
        assert r["timestamp"].dt.tz is not None
    octo = bars([date(2026, 9, 30), date(2026, 10, 1)], seed=2)
    p = tmp_path / "oct.parquet"
    octo.to_parquet(p, index=False)
    r, late = s12.read_raw(p, date(2026, 9, 30))
    assert late == 75 and pd.DatetimeIndex(r["timestamp"]).date.max() == date(2026, 9, 30)
    with pytest.raises(SystemExit):
        s12.build_audit([tmp_path], hi=date(2026, 10, 1))


# 2 ------------------------------------------------------------------ session calendar (holiday / exception / gap)
def test_session_calendar_classification(monkeypatch):
    d_reg, d_exc, d_closed, d_missing, d_gap = (date(2026, 4, 6), date(2026, 4, 7), date(2026, 4, 8),
                                                date(2026, 4, 9), date(2026, 4, 10))
    monkeypatch.setitem(s12.SESSION_EXCEPTIONS, d_exc, {"name": "TEST_SPECIAL", "first": time(13, 45),
                                                        "last": time(14, 40), "source": "test exception"})
    spot = spot_frame([d_reg, d_exc, d_gap], drop={at(d_gap, "10:00"), at(d_gap, "10:05")})
    exc_day = pd.DatetimeIndex(spot["timestamp"]).date == d_exc
    keep = ~exc_day | ((spot["timestamp"] >= at(d_exc, "13:45")) & (spot["timestamp"] <= at(d_exc, "14:40")))
    spot = spot[keep].reset_index(drop=True)
    res = s12.nifty_audit(spot, d_reg, d_gap, activity_days={d_reg, d_exc, d_missing, d_gap},
                          fut_slots={at(d_gap, "10:00")})
    st = {r["date"]: r["status"] for r in res["days"]}
    assert st == {d_reg.isoformat(): s12.REGULAR, d_exc.isoformat(): s12.EXCEPTION, d_closed.isoformat(): s12.CLOSED,
                  d_missing.isoformat(): s12.MISSING_SESSION, d_gap.isoformat(): s12.REGULAR}
    reasons = {(m["date"], m["timestamp"]): m["reason"] for m in res["missing"]}
    assert reasons[(d_gap.isoformat(), at(d_gap, "10:00").isoformat())] == s12.GAP_A
    assert reasons[(d_gap.isoformat(), at(d_gap, "10:05").isoformat())] == s12.GAP_D
    assert sum(1 for m in res["missing"] if m["reason"] == s12.GAP_C) == 75 - 12
    s = res["summary"]
    assert s["missing_genuine"] == 2 and s["missing_session_exception"] == 63
    assert s["missing_sessions"] == [d_missing.isoformat()] and s["status"] == s12.FAIL
    assert s12.SESSION_EXCEPTIONS[date(2025, 10, 21)]["name"] == "MUHURAT_TRADING"
    assert len(s12.regular_grid(d_reg)) == 75 and len(s12.expected_grid(date(2025, 10, 21))) == 12


# 3 ------------------------------------------------------------------ 5-minute cadence
def test_five_minute_cadence():
    d = date(2026, 4, 6)
    spot = spot_frame([d])
    assert s12.nifty_audit(spot, d, d, {d}, set())["summary"]["status"] == s12.PASS
    extra = spot.iloc[[10]].copy()
    extra["timestamp"] = extra["timestamp"] + pd.Timedelta(minutes=1)
    bad = pd.concat([spot, extra], ignore_index=True)
    r = s12.nifty_audit(bad, d, d, {d}, set())["summary"]
    assert r["off_grid"] == 1 and r["status"] == s12.FAIL
    gap = spot_frame([d], drop={at(d, "11:00")})
    assert s12.nifty_audit(gap, d, d, {d}, set())["summary"]["spacing_gaps"] == 1
    assert hx.on_grid_mask(bad).tolist().count(False) == 1


# 4 ------------------------------------------------------------------ OHLC validity
def test_ohlc_validity():
    f = spot_frame([date(2026, 4, 6)])
    f.loc[0, "high"] = f.loc[0, "close"] - 5.0
    f.loc[1, "open"] = np.nan
    f.loc[2, ["open", "high", "low", "close"]] = [0.0, 0.0, 0.0, 0.0]
    q = s12.bar_checks(f)
    assert q["invalid_ohlc"] == 1 and q["missing_ohlc"] == 1 and q["non_positive_price"] == 1


# 5 ------------------------------------------------------------------ duplicate detection
def test_duplicate_detection():
    f = spot_frame([date(2026, 4, 6)])
    same = pd.concat([f, f.iloc[[5]]], ignore_index=True)
    q = s12.bar_checks(same)
    assert q["duplicate_timestamps"] == 1 and q["conflicting_duplicates"] == 0 and q["duplicate_rows"] == 1
    other = f.iloc[[5]].copy()
    other["close"] += 1.0
    other["high"] = other[["high", "close"]].max(axis=1)
    q2 = s12.bar_checks(pd.concat([f, other], ignore_index=True))
    assert q2["duplicate_timestamps"] == 1 and q2["conflicting_duplicates"] == 1


# 6 ------------------------------------------------------------------ off-grid detection / analysis
def _offgrid_file(d: date, copy: bool) -> pd.DataFrame:
    on = bars([d], base=150.0, seed=4).assign(instrument_key="K", tz_error=False)
    off = on.copy()
    off["timestamp"] = off["timestamp"] + pd.Timedelta(minutes=1)
    if not copy:
        rng = np.random.default_rng(1)
        for c in ("open", "high", "low", "close"):
            off[c] = off[c] * rng.uniform(0.97, 1.03, len(off))
        off["high"] = off[["open", "high", "low", "close"]].max(axis=1)
        off["low"] = off[["open", "high", "low", "close"]].min(axis=1)
    return pd.concat([on, off], ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def test_off_grid_detection_and_verdict():
    d = date(2025, 4, 21)
    rows, unusable = s12.off_grid_days(_offgrid_file(d, copy=False), "K", "f.parquet")
    r = rows[0]
    assert r["off_grid_rows"] == 75 and r["minute_offsets"] == "1" and r["second_offsets"] == "0"
    assert r["first_off_grid"] == "09:16:00" and r["last_off_grid"] == "15:26:00"
    assert r["on_grid_complete"] and not r["timezone_shift_explains"] and not r["normalization_safe"]
    assert r["verdict"] == "DIFFERENT_AGGREGATION_BUCKETS" and r["action"] == s12.UNUSABLE
    assert len(unusable) == 75 and all(u["reason"] == s12.UNUSABLE for u in unusable)
    dup, _ = s12.off_grid_days(_offgrid_file(d, copy=True), "K", "f.parquet")
    assert dup[0]["verdict"] == "RELABELLED_DUPLICATE_OF_ON_GRID_SERIES" and not dup[0]["normalization_safe"]
    summ = s12.off_grid_summary(rows)
    assert summ["verdict"] == "DIFFERENT_AGGREGATION_BUCKETS" and "NOT normalised" in summ["explanation"]


# 7 ------------------------------------------------------------------ contract validity (option files)
def _first_option_file(root: Path) -> Path:
    return u6.Layout(root).option_files()[0]


def test_option_contract_validity(env, tmp_path):
    a = env.a
    assert a["options"]["summary"]["status"] == s12.PASS and not a["options"]["summary"]["blocking_reasons"]
    bad_root = tmp_path / "upstox_6m"
    shutil.copytree(env.six, bad_root)
    p = _first_option_file(bad_root)
    f = u6.read_parquet(p)
    f["strike"] = f["strike"] + 50.0
    u6.write_parquet(p, f)
    r = s12.option_audit([bad_root], max_day=SYNTH_PERIOD[1])
    assert r["summary"]["status"] == s12.FAIL and "STRIKE_MISMATCH" in r["summary"]["blocking_reasons"]


def test_option_off_grid_is_quarantined_not_failed(env, tmp_path):
    root = tmp_path / "upstox_6m"
    shutil.copytree(env.six, root)
    p = _first_option_file(root)
    f = u6.read_parquet(p)
    off = f.copy()
    off["timestamp"] = off["timestamp"] + pd.Timedelta(minutes=1)
    off["close"] = off["close"] * 1.01
    off["high"] = off[["open", "high", "low", "close"]].max(axis=1)
    u6.write_parquet(p, pd.concat([f, off], ignore_index=True).sort_values("timestamp"))
    r = s12.option_audit([root], max_day=SYNTH_PERIOD[1])
    assert r["summary"]["status"] == s12.PASS_Q and r["summary"]["off_grid_rows"] == len(off)
    assert r["summary"]["off_grid_on_grid_complete"] is True
    assert len(r["unusable_rows"]) == len(off)
    fail = {x["reason"]: x for x in r["failures"]}
    assert not fail["OFF_GRID_TIMESTAMP"]["blocking"] and fail["OFF_GRID_TIMESTAMP"]["resolution"].startswith("MARKED")


# 8 ------------------------------------------------------------------ expiry causality
def test_expiry_causality(env):
    a = env.a
    cv = a["atm"]["contracts"]
    assert cv and a["atm"]["causal_passed"] and all(r["passed"] for r in cv)
    m = a["_market"]
    for r in cv[:: max(1, len(cv) // 25)]:
        d = date.fromisoformat(r["date"])
        assert date.fromisoformat(r["expiry"]) >= d
        assert date.fromisoformat(r["expiry"]) == sd.nearest_expiry(m.calendar, d)
    fs = a["futures"]["selection_checks"]
    assert not fs["violations"] and fs["independent_of_price_and_volume"]
    assert a["gate"]["conditions"]["causal_contract_selection_pass"]


def test_futures_missing_classification(monkeypatch):
    d1, d2, d3 = date(2026, 4, 6), date(2026, 4, 7), date(2026, 4, 8)
    monkeypatch.setitem(s12.SESSION_EXCEPTIONS, d3, {"name": "TEST_SPECIAL", "first": time(13, 45),
                                                     "last": time(14, 40), "source": "test exception"})
    c1, c2 = fut(date(2026, 4, 28)), fut(date(2026, 5, 26))
    f1 = bars([d1, d3], seed=5)
    keep = ~f1["timestamp"].isin({at(d1, "10:00"), at(d1, "10:05")})
    dd = pd.DatetimeIndex(f1["timestamp"]).date
    keep &= (dd != d3) | ((f1["timestamp"] >= at(d3, "13:45")) & (f1["timestamp"] <= at(d3, "14:40")))
    f1 = f1[keep].reset_index(drop=True)
    f2 = bars([d1, d2, d3], seed=6)
    frames = {c1.instrument_key: f1, c2.instrument_key: f2}
    spot_slots = set(bars([d1, d2], seed=7)["timestamp"]) - {at(d1, "10:05")}
    raw = [("r", c1, f1.assign(tz_error=False)), ("r", c2, f2.assign(tz_error=False))]
    res = s12.futures_audit(raw, [c1, c2], frames, [d1, d2, d3], spot_slots)
    by = {(m["date"], m["timestamp"]): m for m in res["missing"]}
    a = by[(d1.isoformat(), at(d1, "10:00").isoformat())]
    assert a["category"] == "A" and a["previous_candle"] == at(d1, "09:55").isoformat()
    assert a["next_candle"] == at(d1, "10:10").isoformat()
    assert by[(d1.isoformat(), at(d1, "10:05").isoformat())]["category"] == "D"
    assert all(m["category"] == "B" for m in res["missing"] if m["date"] == d2.isoformat())
    assert sum(m["category"] == "C" for m in res["missing"]) == 63
    assert res["summary"]["status"] == s12.FAIL


# 9 ------------------------------------------------------------------ ATM selection
def test_atm_selection(env):
    a, m = env.a, env.a["_market"]
    for r in a["atm"]["contracts"][:: max(1, len(a["atm"]["contracts"]) // 30)]:
        d = date.fromisoformat(r["date"])
        chain = m.chains[(d, date.fromisoformat(r["expiry"]), r["side"])]
        assert r["nearest_strike"] and r["bracketed"] and r["in_listed_chain"] and r["planned_atm"]
        assert chain[r["strike"]].instrument_key == r["instrument_key"]
    sess = a["atm"]["sessions"]
    assert sess and all(0 <= x["CE_eligible"] <= x["signal_timestamps"] for x in sess)
    tot = a["atm"]["totals"]
    assert sum(tot["CE"].values()) == sum(x["signal_timestamps"] for x in sess) == sum(tot["PE"].values())
    assert tot["CE"].get("ELIGIBLE", 0) == sum(x["CE_eligible"] for x in sess)


# 10 ------------------------------------------------------------------ no future filling
def test_no_future_filling(env):
    a = env.a
    m = a["_market"]
    raw = pd.concat([u6.read_parquet(f) for f in u6.Layout(env.six).option_files()], ignore_index=True)
    keys = sorted(m.atm.get(m.sessions[0], ()))[:3]
    assert keys
    for key in keys:
        p = m.premium(key)
        assert p is not None and set(p["timestamp"]) <= set(raw.loc[raw["instrument_key"] == key, "timestamp"])
    d = date(2026, 4, 6)
    spot = spot_frame([d], drop={at(d, "10:00")})
    n = len(spot)
    res = s12.nifty_audit(spot, d, d, {d}, set())
    assert len(spot) == n and res["days"][0]["bars"] == n and res["summary"]["missing_genuine"] == 1
    src = inspect.getsource(s12)
    for bad in ("ffill(", "bfill(", "interpolate(", "fillna(method", ".resample("):
        assert bad not in src, bad


# 11 ------------------------------------------------------------------ no fabricated values
def test_no_fabricated_values(env):
    a = env.a
    assert a["raw_unchanged"] and a["fabricated_values"] == 0 and a["repairs"]["count"] == 0
    assert s12.raw_hashes([env.six], max_day=SYNTH_PERIOD[1]) == env.raw_before
    d = date(2025, 4, 21)
    f = _offgrid_file(d, copy=False)
    m = SimpleNamespace(premium_fn=lambda _k: f)
    hx.quarantine_off_grid(m)
    kept = m.premium_fn("K")
    merged = kept.merge(f, on="timestamp", suffixes=("", "_raw"))
    assert len(merged) == len(kept) == 75
    assert np.allclose(merged["close"], merged["close_raw"]) and set(kept["timestamp"]) <= set(f["timestamp"])


# 12 ------------------------------------------------------------------ deterministic cleaning
def test_deterministic_cleaning(env):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s12, "PERIOD", SYNTH_PERIOD)
        mp.setattr(s12, "MIN_CLEAN_MONTHS", 1)
        mp.setattr(s12, "MIN_ATM_ELIGIBLE_SHARE", 0.5)
        b = s12.build_audit([env.six])
    x, y = s12.summary_obj(env.a), s12.summary_obj(b)
    for k in ("generated_at", "production_unchanged", "token_leak"):
        x.pop(k, None)
        y.pop(k, None)
    assert json.dumps(x, sort_keys=True, default=str) == json.dumps(y, sort_keys=True, default=str)


# ------------------------------------------------------------------ gate / reports / safety
def test_gate_and_clean_history(env):
    a = env.a
    assert a["gate"]["data_ready"] and all(a["gate"]["conditions"].values())
    h = a["history"]["ending_train_end"]
    assert h["start"] == "2026-04-01" and h["end"] <= "2026-06-30" and h["trading_days"] >= 3
    assert a["nifty"]["summary"]["status"] == s12.PASS and a["futures"]["summary"]["status"] == s12.PASS
    assert a["leakage"]["passed"] and a["leakage"]["max_day_loaded"] <= "2026-09-30"


def test_gate_blocks_and_stops(env):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s12, "PERIOD", SYNTH_PERIOD)
        b = s12.build_audit([env.six])
    assert not b["gate"]["data_ready"] and "clean_history_sufficient" in b["gate"]["blocking"]
    text = s12.terminal_text(b)
    lines = text.splitlines()
    assert lines[lines.index("DATA_READY:") + 1] == "NO" and "STOP - do NOT rerun S11 ML" in text
    assert any("CLEAN HISTORY" in x for x in b["blocking_detail"])


def test_status_rules():
    st = [{"date": "2026-01-01", "clean": True, "reasons": ""}, {"date": "2026-01-02", "clean": False, "reasons": "X"},
          {"date": "2026-01-05", "clean": True, "reasons": ""}, {"date": "2026-01-06", "clean": True, "reasons": ""}]
    r = s12.clean_runs(st, date(2026, 1, 6))
    assert r["longest"]["trading_days"] == 2 and r["ending_train_end"]["start"] == "2026-01-05"
    assert r["unclean_sessions"] == [{"date": "2026-01-02", "reasons": "X"}]


def test_reports_and_terminal(env):
    a = env.a
    for n in s12.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        t = p.read_text(encoding="utf-8")
        assert TOKEN not in t and "bearer " not in t.lower()
    assert a["token_leak"]["passed"] and a["production_unchanged"] is True
    assert clean_script.production_hashes() == env.before
    lines = s12.terminal_text(a).splitlines()
    assert lines[0] == s12.HEADING
    for needle in ("NIFTY:", "FUTURES:", "OPTIONS:", "ATM COVERAGE:", "CAUSAL CONTRACT SELECTION:", "DATA_READY:",
                   "CLEAN HISTORY:", "REPAIRS:", "FABRICATED VALUES:", "LIVE ORDERS:", "PRODUCTION CHANGED:",
                   "OCTOBER LOADED:"):
        assert needle in lines, needle
    for k in ("status:", "trading days:", "missing:", "duplicates:", "contracts:", "candles:", "off-grid:",
              "invalid:", "CE:", "PE:", "start:", "end:", "count:", "what was changed:"):
        assert any(l.startswith(k) for l in lines), k
    for k, v in (("FABRICATED VALUES:", "0"), ("LIVE ORDERS:", "0"), ("PRODUCTION CHANGED:", "NO"),
                 ("OCTOBER LOADED:", "NO"), ("DATA_READY:", "YES")):
        assert lines[lines.index(k) + 1] == v, k
    man = json.loads((env.out / "clean_history_manifest.json").read_text(encoding="utf-8"))
    assert man["data_ready"] is True and "2025-10-21" in man["rules"]["session_exceptions"]


def test_october_and_roots_guard(tmp_path):
    for name in ("upstox_fresh_oct", "upstox_6m"):
        (tmp_path / name / "metadata").mkdir(parents=True)
        (tmp_path / name / "metadata" / "plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        script.research_roots(tmp_path / "upstox_fresh_oct", tmp_path / "upstox_6m")
    assert script.research_roots(tmp_path / "missing", tmp_path / "upstox_6m") == [tmp_path / "upstox_6m"]


def test_no_ml_no_orders_no_network_code():
    for mod in (s12, script):
        src = inspect.getsource(mod)
        for bad in ("sklearn", "place_order", "requests.", "UpstoxHistoricalClient(", "fit(", "interpolate("):
            assert bad not in src, (mod.__name__, bad)
