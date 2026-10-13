from __future__ import annotations

import inspect
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s13_chain_audit as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.research import s13_chain_audit as s13
from tests.test_s4_six_month_research import make_store

IST = s13.IST
TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S13"
SYNTH_PERIODS = {"CLEAN_USABLE": (date(2026, 4, 1), date(2026, 6, 30)),
                 "EXISTING_2026": (date(2026, 4, 1), date(2026, 9, 30))}
D = date(2026, 5, 4)
E1, E2 = date(2026, 5, 5), date(2026, 5, 12)
CAL = [date(2026, 4, 28), E1, E2]
T = pd.Timestamp("2026-05-04 10:00", tz=IST)


def _boom(*_a, **_k):
    raise AssertionError("S13.1 must read only the stored research roots")


def _row(ts, strike, typ, expiry=E1, close=100.0, volume=1000.0, oi=5000.0, **kw):
    r = {"timestamp": ts, "tz_problem": False, "instrument_key": f"NSE_FO|{int(strike)}{typ}", "strike": float(strike),
         "expiry": expiry, "option_type": typ, "open": close, "high": close + 1, "low": close - 1, "close": close,
         "volume": volume, "open_interest": oi, "name_expiry": expiry.isoformat(), "name_strike": float(strike),
         "name_type": typ, "file": "synthetic"}
    r.update(kw)
    return r


def _frame(rows) -> pd.DataFrame:
    f = pd.DataFrame(rows)
    f["expiry_raw"] = f["expiry"].astype(str)
    f["date"] = f["timestamp"].dt.date
    return f


def _usable(rows, cal=CAL) -> pd.DataFrame:
    fl = s13.flag_candles(_frame(rows), cal, 50.0)
    return fl[fl["usable"]].drop_duplicates(["timestamp", "expiry", "strike", "option_type"])


def _stamps(*pairs) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": [p[0] for p in pairs], "close": [p[1] for p in pairs]})


# ------------------------------------------------------------------ point-in-time chain reconstruction
def test_point_in_time_chain_reconstruction():
    rows = [_row(T, k, "CE") for k in (23950, 24000, 24050)] + [_row(T, 24000, "PE")]
    R = s13.reconstruct(_stamps((T, 24010.0)), _usable(rows), CAL, 50.0)
    r = R.iloc[0]
    assert r["atm"] == 24000.0 and r["expiry"] == E1 and r["dte"] == 1
    assert r["CE0_p"] and r["CEm1_p"] and r["CEp1_p"] and not r["CEp2_p"]
    assert r["PE0_p"] and not r["PEm1_p"] and not r["PEp1_p"]
    assert r["chain_status"] == "partial_chain" and r["CE_strikes_present_pm5"] == 3
    s = s13.chain_summary(R)
    assert s["band_pm1_ce"] == 1.0 and s["band_pm1_pe"] == 0.0 and s["atm_both"] == 1.0
    full = [_row(T, 24000 + 50 * k, t) for k in range(-3, 4) for t in ("CE", "PE")]
    assert s13.reconstruct(_stamps((T, 24010.0)), _usable(full), CAL, 50.0)["chain_status"].iloc[0] == "complete_chain"
    assert s13.reconstruct(_stamps((T, 24010.0)), _usable([_row(T, 25000, "CE")]), CAL, 50.0)["chain_status"].iloc[0] \
        == "missing_chain"


def test_strike_correctness():
    assert s13.atm_of(24024.9, 50.0) == 24000.0
    assert s13.atm_of(24025.0, 50.0) == 24000.0  # tie -> lower strike
    assert s13.atm_of(24025.1, 50.0) == 24050.0
    assert s13.atm_of(23999.0, 50.0) == 24000.0
    f = _frame([_row(T, 24000, "CE"), _row(T, 24010, "CE", name_strike=24010.0), _row(T, 24050, "CE"),
                _row(T, 24100, "PE", name_strike=24150.0)])
    assert s13.strike_step(f.iloc[[0, 2]]) == 50.0
    fl = s13.flag_candles(f, CAL, 50.0)
    assert fl["f_strike"].tolist() == [False, True, False, True]


# ------------------------------------------------------------------ expiry causality
def test_expiry_causality():
    assert s13.nearest_expiry(CAL, D) == E1
    assert s13.nearest_expiry(CAL, E1) == E1           # expiry day itself
    assert s13.nearest_expiry(CAL, date(2026, 5, 6)) == E2
    assert s13.nearest_expiry(CAL, date(2026, 6, 1)) is None
    assert list(inspect.signature(s13.nearest_expiry).parameters) == ["calendar", "d"]
    # only the next expiry is stored at ATM: the nearest-expiry chain is NOT substituted
    R = s13.reconstruct(_stamps((T, 24000.0)), _usable([_row(T, 24000, "CE", expiry=E2)]), CAL, 50.0)
    assert R["expiry"].iloc[0] == E1 and not R["CE0_p"].iloc[0]


def test_expired_contract_is_invalid():
    f = s13.flag_candles(_frame([_row(T, 24000, "CE", expiry=date(2026, 4, 28))]), CAL, 50.0)
    assert f["f_expiry"].iloc[0] and not f["usable"].iloc[0]
    f = s13.flag_candles(_frame([_row(T, 24000, "CE", expiry=date(2026, 5, 7), name_expiry="2026-05-07")]), CAL, 50.0)
    assert f["f_expiry"].iloc[0]  # expiry not in the listed calendar


# ------------------------------------------------------------------ CE / PE separation
def test_ce_pe_separation():
    R = s13.reconstruct(_stamps((T, 24000.0)), _usable([_row(T, 24000, "CE")]), CAL, 50.0)
    assert R["CE0_p"].iloc[0] and not R["PE0_p"].iloc[0]
    f = s13.flag_candles(_frame([_row(T, 24000, "CE", name_type="PE"), _row(T, 24000, "XX", name_type=None)]),
                         CAL, 50.0)
    assert f["f_type"].tolist() == [True, True]


# ------------------------------------------------------------------ no future contracts / candles
def test_no_future_candles_or_contracts():
    later = T + pd.Timedelta(minutes=5)
    R = s13.reconstruct(_stamps((T, 24000.0)), _usable([_row(later, 24000, "CE"), _row(later, 24000, "PE")]),
                        CAL, 50.0)
    assert not R["CE0_p"].iloc[0] and not R["PE0_p"].iloc[0]
    prev = T - pd.Timedelta(minutes=5)
    R = s13.reconstruct(_stamps((T, 24000.0)), _usable([_row(prev, 24000, "CE"), _row(T, 24000, "CE")]), CAL, 50.0)
    assert R["CE0_p"].iloc[0] and R["CE0_prev_p"].iloc[0]
    rows = [_row(T + pd.Timedelta(minutes=5 * i), 24000, t) for i in range(-2, 4) for t in ("CE", "PE")]
    u = _usable(rows)
    st = _stamps(*[(T + pd.Timedelta(minutes=5 * i), 24000.0) for i in range(0, 3)])
    R = s13.reconstruct(st, u, CAL, 50.0)
    tc = s13.truncation_check(R, st, u, CAL, 50.0, 3)
    assert tc["passed"] and tc["checked"] == 3


# ------------------------------------------------------------------ no fabricated OI / volume
def test_no_fabricated_oi():
    rows = [_row(T, 24000, "CE", oi=np.nan), _row(T, 24000, "PE", oi=0.0)]
    u = _usable(rows)
    assert np.isnan(u.loc[u["option_type"] == "CE", "open_interest"]).all()
    R = s13.reconstruct(_stamps((T, 24000.0)), u, CAL, 50.0)
    assert R["CE0_p"].iloc[0] and not R["CE0_o"].iloc[0]
    assert R["PE0_o"].iloc[0] and not R["PE0_oi_pos"].iloc[0]
    fl = s13.flag_candles(_frame(rows), CAL, 50.0)
    oi = {r["group"]: r for r in s13.field_rows(fl, R, "open_interest", "X", D, D, 50.0)}
    assert oi["ALL"]["missing"] == 1 and oi["ALL"]["zero"] == 1 and oi["ALL"]["present_positive"] == 0
    assert oi["CE"]["coverage"] == 0.0 and oi["CE"]["status"] == s13.FIELD_UNAVAILABLE
    assert oi["ATM"]["candles"] == 2


def test_no_fabricated_volume_and_negative_values():
    rows = [_row(T, 24000, "CE", volume=np.nan), _row(T, 24000, "PE", volume=-5.0, oi=-1.0)]
    fl = s13.flag_candles(_frame(rows), CAL, 50.0)
    assert fl["f_negvol"].tolist() == [False, True] and fl["f_negoi"].tolist() == [False, True]
    assert np.isnan(fl["volume"].iloc[0]) and fl["volume"].iloc[1] == -5.0  # values never changed
    R = s13.reconstruct(_stamps((T, 24000.0)), fl[fl["usable"]], CAL, 50.0)
    assert R["CE0_p"].iloc[0] and not R["CE0_v"].iloc[0] and not R["PE0_p"].iloc[0]
    vol = {r["group"]: r for r in s13.field_rows(fl, R, "volume", "X", D, D, 50.0)}
    assert vol["ALL"]["missing"] == 1 and vol["ALL"]["invalid"] == 1 and vol["ALL"]["status"] == s13.FIELD_UNAVAILABLE
    assert s13.field_status(5, 10) == s13.FIELD_PARTIAL and s13.field_status(10, 10) == s13.FIELD_AVAILABLE


# ------------------------------------------------------------------ timestamp integrity / duplicates
def test_timestamp_integrity():
    ts, bad = s13._to_ist(pd.Series(pd.to_datetime(["2026-05-04 10:00", "2026-05-04 10:05"])))
    assert bad.all() and str(ts.dt.tz) == "Asia/Kolkata"
    ts, bad = s13._to_ist(pd.Series([T, T + pd.Timedelta(minutes=5)]))
    assert not bad.any()
    rows = [_row(T, 24000, "CE"), _row(T + pd.Timedelta(minutes=2), 24000, "CE"),
            _row(pd.Timestamp("2026-05-04 15:30", tz=IST), 24000, "CE"), _row(T, 24050, "CE", tz_problem=True)]
    fl = s13.flag_candles(_frame(rows), CAL, 50.0)
    assert fl["f_offgrid"].tolist() == [False, True, False, False]
    assert fl["f_session"].tolist() == [False, False, True, False]
    assert fl["f_tz"].tolist() == [False, False, False, True] and fl["usable"].tolist() == [True, False, False, False]


def test_duplicate_handling():
    rows = [_row(T, 24000, "CE"), _row(T, 24000, "CE"), _row(T, 24050, "CE", close=10.0),
            _row(T, 24050, "CE", close=11.0)]
    fl = s13.flag_candles(_frame(rows), CAL, 50.0)
    assert fl["f_duplicate"].sum() == 2
    assert fl["f_conflict"].tolist() == [False, False, True, True]
    assert fl.loc[fl["usable"], "strike"].tolist() == [24000.0]
    f = s13.flag_candles(_frame([_row(T, 24000, "CE", instrument_key="NSE_FO|1|06-05-2026")]), CAL, 50.0)
    assert f["f_key"].iloc[0]


def test_deterministic_reconstruction():
    rows = [_row(T + pd.Timedelta(minutes=5 * i), 24000 + 50 * k, t) for i in range(4) for k in (-1, 0, 1)
            for t in ("CE", "PE")]
    st = _stamps(*[(T + pd.Timedelta(minutes=5 * i), 24005.0 + i) for i in range(4)])
    a = s13.reconstruct(st, _usable(rows), CAL, 50.0)
    b = s13.reconstruct(st, _usable(list(reversed(rows))), CAL, 50.0)
    assert a.equals(b)


def test_feature_status_rules():
    assert s13.feature_status(0.99, 10, False)[0] == s13.AVAILABLE
    assert s13.feature_status(0.99, 10, True)[0] == s13.PARTIAL
    assert s13.feature_status(0.70, 10, False)[0] == s13.PARTIAL
    assert s13.feature_status(0.20, 10, False)[0] == s13.UNAVAILABLE
    assert s13.feature_status(0.99, 0, False)[0] == s13.UNAVAILABLE


def test_classify_rules():
    chain = {"atm_ce": 0.99, "atm_pe": 0.99, "band_pm3_both": 0.97}
    good = {"ATM": {"coverage": 0.99}}
    assert s13.classify(dq_ok=True, pit_ok=True, recon_ok=True, chain=chain, oi=good, vol=good)[0] == s13.READY
    assert s13.classify(dq_ok=True, pit_ok=True, recon_ok=True, chain={**chain, "band_pm3_both": 0.3}, oi=good,
                        vol=good)[0] == s13.PARTIAL_READY
    assert s13.classify(dq_ok=True, pit_ok=False, recon_ok=True, chain=chain, oi=good, vol=good)[0] == s13.NOT_READY
    assert s13.classify(dq_ok=True, pit_ok=True, recon_ok=True, chain={**chain, "atm_ce": 0.5}, oi=good,
                        vol=good)[0] == s13.NOT_READY
    low = {"ATM": {"coverage": 0.1}}
    assert s13.classify(dq_ok=True, pit_ok=True, recon_ok=True, chain=chain, oi=low, vol=low)[0] == s13.NOT_READY


def test_loader_refuses_october(tmp_path):
    for hi in (date(2026, 10, 1), date(2026, 10, 31)):
        with pytest.raises(SystemExit):
            s13.load_option_candles([tmp_path], date(2026, 4, 1), hi)
        with pytest.raises(SystemExit):
            s13.load_spot([tmp_path], date(2026, 4, 1), hi)


# ------------------------------------------------------------------ full pipeline on the synthetic store
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s13")
    six, out = base / "upstox_6m", base / "out"
    make_store(six)
    files_before = {p.relative_to(six).as_posix(): p.stat().st_size for p in six.rglob("*") if p.is_file()}
    opened: list[str] = []
    real = pd.read_parquet

    def spy(p, *a, **k):
        opened.append(Path(p).as_posix())
        return real(p, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(s13, "PERIODS", SYNTH_PERIODS)
        mp.setattr(pd, "read_parquet", spy)
        a = script.run(expanded_root=base / "upstox_expanded", data_root=six, out_dir=out, sample=6,
                       secret_values=[TOKEN])
    files_after = {p.relative_to(six).as_posix(): p.stat().st_size for p in six.rglob("*") if p.is_file()}
    return SimpleNamespace(a=a, out=out, six=six, before=before, opened=opened, files_before=files_before,
                           files_after=files_after)


def test_pipeline_reconstruction_and_pit(env):
    for name in SYNTH_PERIODS:
        P = env.a["periods"][name]
        assert P["ran"] and P["point_in_time"]["passed"] and P["reconstruction"]["passed"]
        assert P["point_in_time"]["expiry_causal_violations"] == 0 and P["point_in_time"]["expired_contracts_used"] == 0
        assert P["point_in_time"]["future_truncation"]["passed"] and P["point_in_time"]["deterministic"]
        c = P["chain"]
        assert c["atm_ce"] >= 0.99 and c["atm_pe"] >= 0.99 and c["band_pm1_ce"] >= 0.99
        assert c["complete_chain"] + c["partial_chain"] + c["missing_chain"] == c["timestamps"] > 0
        assert P["data_quality"]["passed"] and P["status"] in (s13.READY, s13.PARTIAL_READY, s13.NOT_READY)
    assert env.a["strike_step"] == 50.0 and env.a["status"] == env.a["periods"]["CLEAN_USABLE"]["status"]


def test_pipeline_oi_volume_and_features(env):
    P = env.a["periods"]["CLEAN_USABLE"]
    assert P["oi"]["ALL"]["coverage"] == 1.0 and P["volume"]["ALL"]["coverage"] == 1.0
    assert P["oi"]["ALL"]["missing"] == 0 and P["volume"]["ALL"]["invalid"] == 0
    feats = P["features"]
    assert len(feats) == 14 and all(not r["calculated"] for r in feats.values())
    assert feats["PCR_OI"]["status"] == s13.PARTIAL  # band-limited: full chain not stored
    assert feats["ATM_STRADDLE_PREMIUM"]["status"] == s13.AVAILABLE
    ex = P["expiry"]
    assert ex["no_expiry"] == 0 and ex["planned_expiry_mismatches"] == 0 and ex["expired_contract_sessions"] == 0


def test_raw_unchanged_and_no_october(env):
    assert env.files_before == env.files_after
    assert env.opened and all("2026-10" not in Path(p).parent.name for p in env.opened)
    assert env.a["load_info"]["max_raw"][:10] <= "2026-09-30" and env.a["october_loaded"] is False
    assert env.a["fabricated_values"] == 0 and env.a["repairs"] == 0


def test_reports_and_token(env):
    for n in s13.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        t = p.read_text(encoding="utf-8")
        assert TOKEN not in t and "bearer " not in t.lower()
    assert env.a["token_leak"]["passed"]
    ok, leaks = u6.token_leak_check([env.out], [TOKEN])
    assert ok and not leaks
    fa = pd.read_csv(env.out / "feature_availability.csv")
    assert set(fa["period"]) == set(SYNTH_PERIODS) and set(fa["status"]) <= {s13.AVAILABLE, s13.PARTIAL, s13.UNAVAILABLE}
    cc = pd.read_csv(env.out / "chain_coverage.csv")
    assert {"SUMMARY", "DAILY"} <= set(cc["level"])


def test_production_unchanged_and_paper(env):
    a = env.a
    assert a["production_unchanged"] is True and clean_script.production_hashes() == env.before
    assert a["live_orders"] == 0 and a["trading_mode"] == "PAPER" and a["live_trading_enabled"] is False


def test_terminal_format(env):
    lines = s13.terminal_text(env.a).splitlines()
    assert lines[0] == s13.HEADING
    order = ["PERIOD:", "start:", "end:", "ATM COVERAGE:", "CE:", "PE:", "ATM ±1:", "ATM ±3:", "OI:", "overall:",
             "VOLUME:", "FEATURE AVAILABILITY:", "PCR OI:", "PCR volume:", "OI change:", "volume imbalance:",
             "CE/PE relative strength:", "ATM straddle:", "ATM breadth:", "CHAIN RECONSTRUCTION:", "POINT-IN-TIME:",
             "DATA QUALITY:", "S13 STATUS:", "OCTOBER LOADED:", "PRODUCTION CHANGED:", "LIVE ORDERS:"]
    pos = -1
    for needle in order:
        pos = next(i for i, l in enumerate(lines) if i > pos and l.startswith(needle))
    assert lines[lines.index("S13 STATUS:") + 1] in (s13.READY, s13.PARTIAL_READY, s13.NOT_READY)
    assert lines[lines.index("OCTOBER LOADED:") + 1] == "NO"
    assert lines[lines.index("PRODUCTION CHANGED:") + 1] == "NO" and lines[lines.index("LIVE ORDERS:") + 1] == "0"
    for k in ("CHAIN RECONSTRUCTION:", "POINT-IN-TIME:", "DATA QUALITY:"):
        assert lines[lines.index(k) + 1] in ("PASS", "FAIL")


def test_no_strategy_ml_or_order_code():
    for mod in (s13, script):
        src = inspect.getsource(mod)
        for bad in ("sklearn", "fit(", "place_order", "fillna", "interpolate", "ffill", "bfill", "GridSearch"):
            assert bad not in src, (mod.__name__, bad)
