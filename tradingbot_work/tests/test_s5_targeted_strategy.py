from __future__ import annotations

import csv
import hashlib
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_s5_targeted_strategy as script
from src.backtest import s2_direction_research as dr
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.indicators.technical import rsi
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.research import s5_targeted_strategy as s5
from src.safety import live_gate
from tests.test_s4_six_month_research import make_store

ROOT = Path(__file__).resolve().parents[1]


def _boom(*_a, **_k):
    raise AssertionError("S5 must read only the 6-month store")


def fake(day: date, direction: str, win: bool, *, hhmm: str = "12:05", rsi14=55.0, sep=0.0, trend="RANGE",
         vol="NORMAL_VOLATILITY", ema="EMA20>EMA50", opt=True) -> dict:
    ts = f"{day.isoformat()}T{hhmm}:00+05:30"
    return {"session_date": day.isoformat(), "signal_time": ts, "direction": direction, "decision": "ACCEPTED",
            "outcome": "WIN" if win else "LOSS", "pnl": 200.0 if win else -100.0, "r_multiple": 2.0 if win else -1.0,
            "entry": 100.0, "stop_loss": 90.0, "target": 120.0, "time_bucket": "midday",
            "entry_time": (pd.Timestamp(ts) + pd.Timedelta(minutes=5)).isoformat(), "instrument_key": "K",
            "features": {"rsi14": rsi14, "ema_sep_atr": sep, "trend_regime": trend, "vol_regime": vol,
                         "time_window": dg.time_window(ts), "ema20_vs_ema50": ema}, "opt_confirm": opt}


def weekdays(lo: date, hi: date) -> list[date]:
    out, d = [], lo
    while d <= hi:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def good_sell_rows(lo=date(2026, 4, 1), hi=date(2026, 9, 30), per_day=3, win_every=5) -> list[dict]:
    """SELL rows matching S1..S7 that win 4 of every 5 trades, spread across days and two trend regimes."""
    rows, i = [], 0
    for d in weekdays(lo, hi):
        for k in range(per_day):
            rows.append(fake(d, "SELL", i % win_every != 0, hhmm=f"12:{5 + 5 * k:02d}",
                             trend="TREND_UP" if i % 2 else "RANGE"))
            i += 1
    return rows


# ------------------------------------------------------------------ conditions / candidates
def test_condition_semantics_and_no_signal():
    f = lambda **k: {"rsi14": None, "ema_sep_atr": None, "trend_regime": dg.UNAVAILABLE, "time_window": None,  # noqa
                     "ema20_vs_ema50": dg.UNAVAILABLE, **k}
    c = {k: v[1] for k, v in s5.CONDITIONS.items() if v[1]}
    assert c["RSI_50_60"](f(rsi14=50.0)) is True and c["RSI_50_60"](f(rsi14=60.0)) is False
    assert c["RSI_40_50"](f(rsi14=40.0)) is True and c["RSI_40_50"](f(rsi14=49.99)) is True
    assert c["EMA_SEP"](f(ema_sep_atr=-1.0)) is False and c["EMA_SEP"](f(ema_sep_atr=-1.01)) is True
    assert c["TREND_DOWN"](f(trend_regime="TREND_DOWN")) is True
    assert c["TIME_1200_1300"](f(time_window="12:00-13:00")) is True
    assert c["EMA20_GT_EMA50"](f(ema20_vs_ema50="EMA20>EMA50")) is True
    for k, fn in c.items():
        assert fn(f()) is None, k
    r = fake(date(2026, 4, 1), "SELL", True, rsi14=None)
    assert s5.candidate_rows(s5.BASE_CANDIDATES[6], [r]) == []
    assert s5.no_signal_count(s5.BASE_CANDIDATES[6], [r]) == 1


def test_exactly_the_predeclared_candidates():
    got = {c.cid: (c.direction, c.conditions) for c in s5.BASE_CANDIDATES}
    assert got == {
        "B1": ("BUY", ("EMA_SEP",)), "B2": ("BUY", ("TREND_DOWN",)), "B3": ("BUY", ("EMA_SEP", "TREND_DOWN")),
        "B4": ("BUY", ("EMA_SEP", "RSI_40_50")), "B5": ("BUY", ("TREND_DOWN", "RSI_40_50")),
        "B6": ("BUY", ("EMA_SEP", "TREND_DOWN", "RSI_40_50")),
        "S1": ("SELL", ("RSI_50_60",)), "S2": ("SELL", ("TIME_1200_1300",)), "S3": ("SELL", ("EMA20_GT_EMA50",)),
        "S4": ("SELL", ("RSI_50_60", "EMA20_GT_EMA50")), "S5": ("SELL", ("RSI_50_60", "TIME_1200_1300")),
        "S6": ("SELL", ("EMA20_GT_EMA50", "TIME_1200_1300")),
        "S7": ("SELL", ("RSI_50_60", "EMA20_GT_EMA50", "TIME_1200_1300"))}
    opt = s5.with_option(s5.BASE_CANDIDATES[0])
    assert opt.cid == "B1_OPT" and opt.conditions[-1] == "OPT_CONFIRM"


def test_buy_sell_separation_and_option_no_signal():
    d = date(2026, 4, 1)
    rows = [fake(d, "BUY", True, sep=-2.0, opt=None), fake(d, "SELL", True, opt=False), fake(d, "SELL", True)]
    assert all(r["direction"] == "BUY" for r in s5.candidate_rows(s5.BASE_CANDIDATES[0], rows))
    assert all(r["direction"] == "SELL" for r in s5.candidate_rows(s5.BASE_CANDIDATES[6], rows))
    assert s5.candidate_rows(s5.with_option(s5.BASE_CANDIDATES[0]), rows) == []
    assert len(s5.candidate_rows(s5.with_option(s5.BASE_CANDIDATES[6]), rows)) == 1
    assert s5.option_confirmation(None) is None
    assert s5.option_confirmation({"data_status": "UNAVAILABLE", "premium": None}) is None


# ------------------------------------------------------------------ chronology / selection isolation
def test_splits_and_walk_forward_windows():
    w = s5.wf_windows()
    assert w[0][0] == date(2026, 4, 1) and w[-1][1] == date(2026, 8, 31) and len(w) == 10
    assert all(b + timedelta(days=1) == c for (_, b), (c, _) in zip(w, w[1:]))
    assert all(hi < s5.SPLITS["HOLDOUT"][0] for _, hi in w)
    sel_w = [x for x in w if x[1] <= s5.SELECTION_END]
    assert len(sel_w) == 8


def test_selection_refuses_oos_and_holdout_rows():
    with pytest.raises(ValueError):
        s5.select([fake(date(2026, 8, 3), "SELL", True)])
    with pytest.raises(ValueError):
        s5.select([fake(date(2026, 9, 1), "SELL", True)])


def test_september_and_oos_cannot_change_selection():
    rows = good_sell_rows()
    pre = [r for r in rows if s5.day_of(r) <= s5.SELECTION_END]
    sel = s5.select(pre)
    assert sel["selected"]["SELL"] is not None and sel["selected"]["BUY"] is None
    assert sel["selected"]["SELL"].cid.startswith("S1")  # ties -> fewest conditions, then lowest id
    flipped = s5._flip(rows, s5.SELECTION_END)
    assert any(a["outcome"] != b["outcome"] for a, b in zip(rows, flipped) if s5.day_of(a) > s5.SELECTION_END)
    sel2 = s5.select([r for r in flipped if s5.day_of(r) <= s5.SELECTION_END])
    assert s5._ids(sel) == s5._ids(sel2) and s5.fingerprint(sel) == s5.fingerprint(sel2)


# ------------------------------------------------------------------ gate / sample size / holdout
def test_gate_robust_only_with_sufficient_stable_sample():
    s1 = s5.BASE_CANDIDATES[6]
    rows = good_sell_rows(hi=date(2026, 8, 31))
    g = s5.gate(s1, s5.candidate_rows(s1, rows), True, False)
    assert g["oos"]["trades"] >= s5.MIN_ROBUST_OOS and g["oos"]["accuracy"] >= 0.70
    assert g["status"] == s5.ROBUST, g["failed"]
    assert s5.gate(s1, s5.candidate_rows(s1, rows), False, False)["status"] != s5.ROBUST  # lookahead fail
    tiny = [r for r in rows if s5.day_of(r) < date(2026, 8, 1)] + \
        [r for r in rows if date(2026, 8, 1) <= s5.day_of(r) <= date(2026, 8, 4)]
    g2 = s5.gate(s1, s5.candidate_rows(s1, tiny), True, False)
    assert g2["oos"]["trades"] < s5.MIN_SUFFICIENT_OOS and g2["status"] == s5.INSUFFICIENT
    one_day = [fake(date(2026, 8, 3), "SELL", True, hhmm=f"12:{m:02d}") for m in range(0, 60, 1)][:59]
    g3 = s5.gate(s1, [r for r in rows if s5.day_of(r) < date(2026, 8, 1)] + one_day, True, False)
    assert g3["checks"]["no_day_dependency"] is False and g3["status"] != s5.ROBUST


def test_sample_grades_and_holdout_status():
    assert [s5.sample_grade(n) for n in (120, 60, 35, 10)] == ["A", "B", "C", "D"]
    m = {"trades": 40, "net_pnl": 1000.0, "profit_factor": 1.6, "accuracy": 0.55}
    assert s5.holdout_status(m, robust=False) == s5.PASSED_HO
    assert s5.holdout_status(m, robust=True) == s5.FAILED_HO
    assert s5.holdout_status(dict(m, trades=5), robust=False) == s5.FAILED_HO


def test_recommendation_rules():
    c = s5.BASE_CANDIDATES[6]
    best = {"BUY": None, "SELL": c}
    cand = {c.cid: {"gate": {"status": s5.ROBUST}}}
    ho = {"SELL": {"status": s5.PASSED_HO}, "COMBINED": {"status": s5.PASSED_HO}}
    assert s5.recommend(best, cand, None, ho, contaminated=False, clean=True) == s5.ROBUST
    assert s5.recommend(best, cand, None, ho, contaminated=True, clean=True) == s5.PROMISING
    assert s5.recommend(best, cand, None, dict(ho, SELL={"status": s5.FAILED_HO}), False, True) == s5.REJECTED
    assert s5.recommend(best, cand, None, ho, False, clean=False) == s5.REJECTED
    assert s5.recommend({"BUY": None, "SELL": None}, {}, None, ho, False, True) == s5.KEEP
    cand_i = {c.cid: {"gate": {"status": s5.INSUFFICIENT}}}
    assert s5.recommend(best, cand_i, None, ho, False, True) == s5.INSUFFICIENT


# ------------------------------------------------------------------ end-to-end on the synthetic 6-month store
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before, baseline = script.s2_hashes(), repr(dr.BASELINE)
    base = tmp_path_factory.mktemp("s5")
    make_store(base / "upstox_6m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=base / "upstox_6m", out_dir=base / "out", check_sample=6)
    store = s4.SixMonthStore(base / "upstox_6m")
    return SimpleNamespace(a=a, base=base, before=before, baseline=baseline, spot=store.spot())


def test_no_s2_mutation_paper_and_no_orders(env, monkeypatch):
    assert script.s2_hashes() == env.before and repr(dr.BASELINE) == env.baseline
    a = env.a
    assert a["s2_changed"] is False and a["orders"] == 0 and a["paper"] and a["live_trading_enabled"] is False
    assert script.module_safety()["ok"]
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))


def test_lookahead_checks_pass(env):
    la = env.a["lookahead"]
    assert env.a["lookahead_ok"]
    assert la["features"]["checked"] > 0 and la["features"]["passed"]
    assert la["time_window"]["passed"] and la["chronology"]["passed"] and la["option_confirmation"]["passed"]


def test_features_are_causal_on_truncated_history(env):
    """ema_sep_atr, RSI, EMA20/EMA50 and trend recomputed from candles <= signal only must match."""
    spot = env.spot
    t = dra._ist_series(spot["timestamp"])
    full_bars = dg.feature_bars(spot)
    picked = [{"signal_time": ts.isoformat(), "direction": "BUY", "session_date": ts.date().isoformat(),
               "break_atr": 0.5, "dte": 1, "strike": None, "option_type": None, "spot": None}
              for ts in list(full_bars.index[400::200])[:5]]
    assert picked
    for r in picked:
        ts = dra._ts(r["signal_time"])
        trunc = spot[(t <= ts).to_numpy()].reset_index(drop=True)
        a = dg.signal_features(r, full_bars, None)
        b = dg.signal_features(r, dg.feature_bars(trunc), None)
        for k in ("ema_sep_atr", "rsi14", "ema20", "ema50", "trend_regime", "ema20_vs_ema50", "time_window"):
            assert a[k] == (pytest.approx(b[k]) if isinstance(b[k], float) else b[k]), k
        closes = trunc["close"].astype(float)
        assert a["rsi14"] == pytest.approx(float(rsi(closes, 14).iloc[-1]))
        assert a["time_window"] == dg.time_window(ts)


def test_integrity_and_selection_isolation(env):
    ids = {x["id"]: x["status"] for x in env.a["integrity"]}
    assert ids["I1"] == "PASS" and ids["I4"] == "PASS" and ids["I5"] == "PASS" and ids["I6"] == "PASS"
    assert ids["I7"] == "PASS" and ids["I9"] == "WARN"
    assert env.a["integrity_ok"]
    assert env.a["invariance"]["oos_and_holdout_outcomes_flipped_same_selection"]
    assert env.a["holdout"]["fingerprint"] == env.a["fingerprint"]


def test_candidates_never_see_september(env):
    with (env.base / "out" / "s5_candidates.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {r["split"] for r in rows} <= {"TRAIN", "VALIDATION", "OOS"}
    with (env.base / "out" / "s5_walk_forward.csv").open(encoding="utf-8") as fh:
        wf = list(csv.DictReader(fh))
    assert wf and all(r["window"].split("..")[1] <= "2026-08-31" for r in wf)
    for v in env.a["candidates"].values():
        for split in ("train", "validation", "oos"):
            assert v["gate"][split]["trades"] >= 0


def test_reports_and_final_output(env):
    out = env.base / "out"
    for n in s5.REPORT_FILES:
        assert (out / n).is_file(), n
    text = s5.summary_text(env.a)
    for label in ("S5 TARGETED RESEARCH", "BEST BUY:", "BEST SELL:", "BEST COMBINED:", "candidate:",
                  "OOS accuracy:", "FINAL SEPTEMBER HOLDOUT:", "70% TARGET:", "ROBUST_70:", "LOOKAHEAD:\nPASS",
                  "INTEGRITY:\nPASS", "PRODUCTION S2 CHANGED:\nNO", "LIVE ORDERS:\n0", "FINAL RECOMMENDATION:"):
        assert label in text, label
    assert env.a["recommendation"] in (s5.ROBUST, s5.PROMISING, s5.KEEP, s5.INSUFFICIENT, s5.REJECTED)
    if not env.a["target_70_achieved"]:
        assert s5.NOT_ACHIEVED_TEXT in text
    for n in s5.REPORT_FILES:
        t = (out / n).read_text(encoding="utf-8").lower()
        assert "access_token" not in t and "bearer " not in t
    assert hashlib.sha256((out / "s5_integrity_audit.md").read_bytes()).hexdigest()
