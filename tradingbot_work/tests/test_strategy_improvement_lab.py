from __future__ import annotations

import csv
import json
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest import buy_dte_lab as dl
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.backtest import strategy_improvement_lab as lab
from src.backtest.live_vs_historical import time_bucket
from src.indicators.technical import ema

IST = imp.IST
S2_CFG = {"range_start": "09:15", "range_end": "09:30", "atr_period": 14}
TRAIN_DAYS = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09",
              "2026-09-10", "2026-09-11"]
VAL_DAYS = ["2026-09-14", "2026-09-15", "2026-09-16"]
OOS_DAYS = ["2026-09-17", "2026-09-18", "2026-09-21"]
SPLITS = imp.split_dates({"TRAIN": ("2026-09-01", "2026-09-11"), "VALIDATION": ("2026-09-14", "2026-09-16"),
                          "OOS": ("2026-09-17", "2026-09-21")})
SPLIT_DAYS = {"TRAIN": TRAIN_DAYS, "VALIDATION": VAL_DAYS, "OOS": OOS_DAYS}
REQUIRED = ["BASELINE", "BUY_EMA_CONFIRM", "BUY_VWAP_CONFIRM", "BUY_RSI_CONFIRM", "BUY_VOLUME_CONFIRM",
            "BUY_BREAKOUT_CONFIRM", "BUY_EMA_VWAP", "BUY_EMA_RSI", "BUY_VWAP_RSI", "BUY_VWAP_VOLUME", "BUY_EMA_VOLUME",
            "BUY_EMA_VWAP_RSI", "BUY_EMA_VWAP_VOLUME", "BUY_EMA_RSI_VOLUME", "BUY_VWAP_RSI_VOLUME",
            "BUY_STRONG_CONFIRMATION", "BUY_MORNING_ONLY", "BUY_MIDDAY_ONLY", "BUY_AFTERNOON_ONLY",
            "BUY_TRENDING_ONLY", "BUY_NON_CHOPPY_ONLY", "SELL_EMA_CONFIRM", "SELL_VWAP_CONFIRM", "SELL_RSI_CONFIRM",
            "SELL_VOLUME_CONFIRM", "SELL_BREAKOUT_CONFIRM", "SELL_EMA_VWAP", "SELL_EMA_RSI", "SELL_VWAP_RSI",
            "SELL_STRONG_CONFIRMATION"]


def _ts(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(date.fromisoformat(day), time.fromisoformat(hhmm)), tz=IST)


def _row(day, hhmm, direction, win, pnl=None, regimes=("range_bound",), dte=5):
    ts = _ts(day, hhmm)
    pnl = (650.0 if win else -325.0) if pnl is None else pnl
    return {"session_date": day, "signal_time": ts.isoformat(), "direction": direction, "decision": imp.ACCEPTED,
            "skip_reason": None, "state": sh.EVALUATED, "outcome": imp.WIN if win else imp.LOSS, "pnl": pnl,
            "r_multiple": 2.0 if win else -1.0, "time_bucket": time_bucket(ts), "dte": dte,
            "dte_bucket": sh.dte_bucket(dte), "regimes": list(regimes), "contract": "NIFTY CE", "break_atr": 0.3,
            "premium_atr": 2.0, "reason": None, "reason_raw": None, "strategy": lab.BASELINE}


def _fact(direction, minute=600, **kw):
    up = direction == "BUY"
    f = {"direction": direction, "minute": minute, "open": 100.0 if up else 101.0, "close": 101.0 if up else 99.0,
         "ema20": 100.5 if up else 99.5, "ema50": 100.0, "rsi14": 60.0 if up else 40.0, "vwap": None,
         "vol_ratio": None, "prev_open": 99.5 if up else 100.5, "prev_close": 100.2 if up else 100.0,
         "prev_orb_high": 100.1, "prev_orb_low": 100.2, "atr14": 10.0, "break_atr": 0.3, "atr_pct": 0.05,
         "vwap_dist_atr": None, "intraday_move": 0.002, "sign_changes": 3.0, "trending": False, "choppy": False,
         "pit_tags": [], "range_bound": False, "prev_day_labels": [], "has_prev_day": True, "premium_atr_pct": 5.0,
         "dte": 5, "dte_bucket": "5-7 DTE"}
    f.update(kw)
    return f


def _dataset():
    """Deterministic rows: BUY loses when RSI < 55 (and wins otherwise), SELL wins 2 of 3."""
    rows, facts = [], {}
    for i, d in enumerate(TRAIN_DAYS + VAL_DAYS + OOS_DAYS):
        for hhmm, direction, win, rsi_v in [("10:00", "BUY", True, 62.0), ("11:00", "BUY", False, 52.0),
                                            ("12:30", "BUY", (i % 3) != 0, 58.0),
                                            ("10:30", "SELL", (i + 1) % 3 != 0, 40.0),
                                            ("13:45", "SELL", i % 2 == 0, 44.0)]:
            r = _row(d, hhmm, direction, win, regimes=(sh.REGIME_LABELS[i % 4],))
            rows.append(r)
            ts = _ts(d, hhmm)
            facts[lab.row_key(r)] = _fact(direction, ts.hour * 60 + ts.minute, rsi14=rsi_v,
                                          atr_pct=0.04 + 0.01 * (i % 3), premium_atr_pct=3.0 + i % 5)
    return rows, facts


def _spot(days=10, start=date(2026, 9, 21), volume=0.0, seed=7):
    rng = np.random.default_rng(seed)
    rows, price, d = [], 25000.0, start
    while len({r["timestamp"].date() for r in rows}) < days:
        if d.weekday() < 5:
            t = pd.Timestamp(datetime.combine(d, time(9, 15)), tz=IST)
            for k in range(75):
                o = price
                c = o + rng.normal(0, 12)
                rows.append({"timestamp": t + pd.Timedelta(minutes=5 * k), "open": o, "high": max(o, c) + 4,
                             "low": min(o, c) - 4, "close": c, "volume": volume})
                price = c
        d += timedelta(days=1)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- candidate generation
def test_candidate_generation_is_predeclared_unique_and_complete():
    cands = lab.predeclared_candidates()
    names = [c.name for c in cands]
    assert len(names) == len(set(names))
    assert set(REQUIRED) <= set(names)
    assert {f"TIME_{a}_{b}" for a, b in lab.TIME_WINDOWS} <= set(names)
    assert lab.COMBINED not in names  # built later from TRAIN + VALIDATION only
    assert names == [c.name for c in lab.predeclared_candidates()]
    for c in cands:
        for d in lab.DIRECTIONS:
            for chk in c.checks(d) or ():
                lab.check(chk, d, _fact(d), {**lab.DEFAULT_PARAMS, "min_atr_pct": 0.01, "prem_lo": 1, "prem_hi": 9})
                assert lab.check_text(chk, d, lab.DEFAULT_PARAMS)
    by = {c.name: c for c in cands}
    assert by["BUY_EMA_VWAP_RSI"].buy == ("ema", "vwap", "rsi") and by["BUY_EMA_VWAP_RSI"].sell == ()
    assert by["SELL_EMA_RSI"].sell == ("ema", "rsi") and by["SELL_EMA_RSI"].buy == ()
    assert by["BASELINE"].n_checks == 0 and by["SELL_ONLY_REFERENCE"].buy is None


# --------------------------------------------------------------------------- feature calculation
def test_feature_calculation_matches_indicators_and_resets_per_session():
    spot = _spot(days=3)
    f = lab.bar_features(spot)
    assert np.allclose(f["ema20"].to_numpy(), ema(spot["close"], 20).to_numpy(), equal_nan=True)
    first_bars = f.groupby("day").head(1)
    assert first_bars["prev_close"].isna().all() and first_bars["sign_changes"].eq(0).all()
    assert f["vwap"].isna().all() and f["vol_ratio"].isna().all()  # zero index volume -> unavailable
    day = f[f["day"] == f["day"].iloc[0]]
    closes = day["close"].to_numpy()
    signs = np.sign(np.diff(closes))
    assert int(day["sign_changes"].iloc[-1]) == int((np.abs(np.diff(signs)) > 0).sum())
    with_vol = lab.bar_features(_spot(days=1, volume=1000.0))
    tp = (with_vol["high"] + with_vol["low"] + with_vol["close"]) / 3
    assert with_vol["vwap"].iloc[9] == pytest.approx(tp.iloc[:10].mean())


def test_checks_are_direction_aware():
    p = lab.DEFAULT_PARAMS
    assert lab.check("ema", "BUY", _fact("BUY"), p) and lab.check("ema", "SELL", _fact("SELL"), p)
    assert lab.check("ema", "SELL", _fact("BUY"), p) is False
    assert lab.check("rsi", "BUY", _fact("BUY", rsi14=55.0), p) and not lab.check("rsi", "BUY", _fact("BUY", rsi14=54.9), p)
    assert lab.check("rsi", "SELL", _fact("SELL", rsi14=45.0), p) and not lab.check("rsi", "SELL", _fact("SELL", rsi14=46), p)
    assert lab.check("vwap", "BUY", _fact("BUY"), p) is None and lab.check("volume", "SELL", _fact("SELL"), p) is None
    assert lab.check("breakout", "BUY", _fact("BUY", break_atr=0.5), p) and not lab.check("breakout", "BUY",
                                                                                          _fact("BUY", break_atr=0.51), p)
    assert lab.check("strong", "BUY", _fact("BUY"), p) and lab.check("strong", "SELL", _fact("SELL"), p)
    assert lab.check("candle", "SELL", _fact("SELL", close=102.0), p) is False


# --------------------------------------------------------------------------- no lookahead
def test_bar_features_are_invariant_to_truncation():
    spot = _spot(days=4)
    full = lab.bar_features(spot)
    for k in (40, 160, 230):
        part = lab.bar_features(spot.iloc[:k])
        pd.testing.assert_frame_equal(part, full.iloc[:k], check_freq=False)


def _la_rows(day):
    return [dict(_row(day, t, "BUY", True), break_atr=0.2) for t in ("10:00", "11:00", "12:55")]


def test_truncation_lookahead_passes_for_all_candidates():
    spot = _spot(days=8)
    day = sorted({t.date() for t in spot["timestamp"]})[-1].isoformat()
    rows = _la_rows(day)
    facts = lab.facts_from_spot(rows, spot, S2_CFG, {})
    p = {**lab.DEFAULT_PARAMS, "min_atr_pct": 0.01, "prem_lo": 1.0, "prem_hi": 9.0}
    res = lab.truncation_lookahead(rows, spot, S2_CFG, {}, facts, lab.predeclared_candidates(), p, [day])
    assert res["_features"]["passed"]
    assert all(v["passed"] for v in res.values())


def test_truncation_lookahead_detects_future_leak(monkeypatch):
    spot = _spot(days=8)
    day = sorted({t.date() for t in spot["timestamp"]})[-1].isoformat()
    rows = _la_rows(day)
    original = lab.bar_features

    def leaky(frame):
        f = original(frame)
        f["ema20"] = f["close"].shift(-1)
        return f

    monkeypatch.setattr(lab, "bar_features", leaky)
    facts = lab.facts_from_spot(rows, spot, S2_CFG, {})
    cands = [c for c in lab.predeclared_candidates() if c.name in ("BUY_EMA_CONFIRM", "BASELINE")]
    res = lab.truncation_lookahead(rows, spot, S2_CFG, {}, facts, cands, lab.DEFAULT_PARAMS, [day])
    assert not res["_features"]["passed"]


# --------------------------------------------------------------------------- chronological split
def test_chronological_split_and_walk_forward_windows():
    with pytest.raises(ValueError):
        imp.split_dates({"TRAIN": ("2026-09-01", "2026-09-20"), "VALIDATION": ("2026-09-18", "2026-09-24"),
                         "OOS": ("2026-09-25", "2026-10-01")})
    days = TRAIN_DAYS + VAL_DAYS + OOS_DAYS + ["2026-10-07"]
    split_days, unassigned = so.assign_days(list(reversed(days)), SPLITS)
    assert split_days == SPLIT_DAYS and unassigned == ["2026-10-07"]
    folds = lab.walk_forward_folds([f"2026-08-{i:02d}" for i in range(1, 23)])
    assert len(folds) == 3
    tests = [d for f in folds for d in f["test"]]
    assert len(tests) == len(set(tests))
    for f in folds:
        assert max(f["train"]) < min(f["validation"]) <= max(f["validation"]) < min(f["test"])


# --------------------------------------------------------------------------- time / regime filters
def test_time_filters_are_half_open_windows():
    p = lab.DEFAULT_PARAMS
    assert lab.check("time:0930-1000", "BUY", _fact("BUY", minute=9 * 60 + 30), p)
    assert not lab.check("time:0930-1000", "BUY", _fact("BUY", minute=10 * 60), p)
    assert lab.check("time:1330-1500", "SELL", _fact("SELL", minute=14 * 60 + 25), p)


def test_regime_filters_use_point_in_time_inputs_only():
    p = lab.DEFAULT_PARAMS
    assert lab.check("regime:trending", "BUY", _fact("BUY", trending=True), p)
    assert lab.check("regime:non_choppy", "BUY", _fact("BUY", choppy=True), p) is False
    assert lab.check("regime:non_choppy", "BUY", _fact("BUY", choppy=None), p) is None
    assert lab.check("regime:range", "SELL", _fact("SELL", range_bound=True), p)
    assert lab.check("regime:existing_avoid", "BUY", _fact("BUY", pit_tags=["low_volatility"]), p) is False
    assert lab.check("regime:prev_day_trend", "BUY", _fact("BUY", prev_day_labels=["strong_bearish_trend"]), p)
    assert lab.check("regime:prev_day_trend", "BUY", _fact("BUY", has_prev_day=False), p) is None
    # end-of-day labels on the row never influence a decision
    rows, facts = _dataset()
    cand = next(c for c in lab.predeclared_candidates() if c.name == "REGIME_TRENDING_ONLY")
    a = lab.apply_candidate(rows, facts, cand, p)
    b = lab.apply_candidate([dict(r, regimes=["strong_bullish_trend"]) for r in rows], facts, cand, p)
    assert [r["state"] for r in a] == [r["state"] for r in b]
    labels = {0: {"choppy_whipsaw"}, 1: {"strong_bullish_trend"}}
    prev = lab.previous_session_labels({date(2026, 9, 1): labels[0], date(2026, 9, 2): labels[1]})
    assert prev == {date(2026, 9, 2): {"choppy_whipsaw"}}


# --------------------------------------------------------------------------- BUY / SELL independence
def test_buy_and_sell_filters_are_independent():
    rows, facts = _dataset()
    by = {c.name: c for c in lab.predeclared_candidates()}
    p = lab.DEFAULT_PARAMS
    buy_f = lab.apply_candidate(rows, facts, by["BUY_RSI_CONFIRM"], p)
    sell_f = lab.apply_candidate(rows, facts, by["SELL_RSI_CONFIRM"], p)
    for base, b, s in zip(rows, buy_f, sell_f):
        if base["direction"] == "SELL":
            assert b["state"] == base["state"] and b["pnl"] == base["pnl"]
        else:
            assert s["state"] == base["state"] and s["pnl"] == base["pnl"]
            assert (b["state"] == sh.SKIPPED) == (facts[lab.row_key(base)]["rsi14"] < 55)
    ref = lab.apply_candidate(rows, facts, by["SELL_ONLY_REFERENCE"], p)
    assert all(r["skip_reason"] == dr.DIRECTION_SKIP for r in ref if r["direction"] == "BUY")
    kept = [r for r in buy_f if r["state"] == sh.EVALUATED]
    assert all(r["outcome"] == base["outcome"] for r, base in zip(buy_f, rows) if r["state"] == sh.EVALUATED)
    assert kept


# --------------------------------------------------------------------------- option expiry point-in-time
def test_option_expiry_selection_is_point_in_time():
    day = date(2026, 9, 22)
    expiries = [date(2026, 9, 16), date(2026, 9, 23), date(2026, 9, 30), date(2026, 10, 28)]
    for pol in dl.POLICIES:
        chosen = dl.eligible_expiries(expiries, day, pol)
        assert all(e >= day for e in chosen)
        assert all(dl.in_policy(pol, day, e) for e in chosen)
    nearest = next(p for p in dl.POLICIES if p.mode == "NEAREST")
    assert dl.eligible_expiries(expiries, day, nearest) == [date(2026, 9, 23)]
    assert dl.eligible_expiries(expiries, day, next(p for p in dl.POLICIES if p.requested == "0 DTE")) == []
    prem = pd.DataFrame({"timestamp": [_ts("2026-09-22", "10:00"), _ts("2026-09-22", "10:05")]})
    assert dl.listed_at(prem, _ts("2026-09-22", "10:00")) and not dl.listed_at(prem, _ts("2026-09-22", "09:55"))


# --------------------------------------------------------------------------- scoring / ranking
def _perf(n, wr, pf, pnl, stab=0.6):
    return {"evaluated": n, "win_rate": wr, "profit_factor": pf, "net_pnl": pnl, "max_drawdown": 1000.0,
            "pnl_excl_best_2_days": pnl * 0.5, "pnl_excl_largest_win": pnl * 0.8, "day_stability": stab,
            "max_consecutive_losses": 3}


def _score(oos, **kw):
    s = {"TRAIN": _perf(100, 0.55, 1.4, 5000), "VALIDATION": _perf(50, 0.56, 1.4, 3000), "OOS": oos,
         "ALL": _perf(200, 0.55, 1.4, 9000), "retention": 0.5, "regime_stability": 0.75, "regime_groups": 3,
         "profitable_regime_groups": 2, "time_stability": 0.6, "day_stability": 0.6, "wf_positive_share": 1.0,
         "robustness_ok": True, "lookahead_passed": True, "n_checks": 1, "family": "X"}
    s.update(kw)
    s.update(lab.candidate_status(s, kw.pop("criteria", lab.LabCriteria())))
    return s


def test_small_high_accuracy_candidate_never_beats_large_stable_candidate():
    scores = {"TINY_70PCT": _score(_perf(10, 0.70, 3.0, 8000, stab=1.0)),
              "STABLE_58PCT": _score(_perf(150, 0.58, 1.4, 12000))}
    assert scores["TINY_70PCT"]["status"] == lab.INSUFFICIENT_DATA
    assert scores["STABLE_58PCT"]["status"] == lab.PROMISING
    assert lab.rank(scores) == ["STABLE_58PCT", "TINY_70PCT"]


def test_status_rules():
    ok = _perf(150, 0.58, 1.4, 12000)
    assert _score(ok, lookahead_passed=False)["status"] == lab.REJECTED
    assert _score(ok, retention=0.05)["failed_checks"] == ["filter_removes_almost_all_signals"]
    assert _score(_perf(40, 0.95, 8.0, 9000))["failed_checks"] == ["suspiciously_perfect_oos"]
    assert "validation_collapses_vs_train" in _score(ok, VALIDATION=_perf(50, 0.3, 0.5, -2000))["failed_checks"]
    assert "oos_collapses_vs_validation" in _score(_perf(150, 0.4, 0.6, -3000))["failed_checks"]
    dep = dict(ok, pnl_excl_best_2_days=-10.0)
    assert _score(dep)["failed_checks"] == ["depends_on_one_or_two_days"]
    assert "works_in_only_one_regime" in _score(ok, profitable_regime_groups=1)["failed_checks"]
    assert _score(_perf(150, 0.5, 1.05, 500))["status"] == lab.NOT_PROMISING
    capped = _score(ok)
    assert capped["status"] == lab.PROMISING and "oos_window_previously_observed" in capped["failed_checks"]
    assert _score(ok, criteria=lab.LabCriteria(oos_previously_observed=False))["status"] == lab.ROBUST


# --------------------------------------------------------------------------- OOS isolation
def test_combined_selection_and_calibration_never_read_oos():
    rows, facts = _dataset()
    cands = lab.predeclared_candidates()
    p = lab.train_params(rows, facts, TRAIN_DAYS)
    by = {c.name: lab.apply_candidate(rows, facts, c, p) for c in cands}
    comb, pick = lab.combined_candidate(cands, by, SPLIT_DAYS)
    flipped = [dict(r, outcome=imp.LOSS, pnl=-5000.0) if r["session_date"] in OOS_DAYS else r for r in rows]
    oos_facts = {k: (dict(f, atr_pct=99.0, premium_atr_pct=99.0) if k[0][:10] in OOS_DAYS else f)
                 for k, f in facts.items()}
    assert lab.train_params(flipped, oos_facts, TRAIN_DAYS) == p
    by2 = {c.name: lab.apply_candidate(flipped, oos_facts, c, p) for c in cands}
    comb2, pick2 = lab.combined_candidate(cands, by2, SPLIT_DAYS)
    assert (comb2, pick2) == (comb, pick)
    assert pick["BUY"] is not None  # BUY_RSI_CONFIRM-type filters remove the RSI<55 losers on TRAIN/VALIDATION


# --------------------------------------------------------------------------- robustness
def test_robustness_marks_isolated_optimum_as_unstable():
    rows, facts = [], {}
    for d in TRAIN_DAYS[:1] + VAL_DAYS[:1] + OOS_DAYS[:1]:
        for hhmm, rsi_v, pnl in (("10:00", 53.0, -300.0), ("10:30", 56.0, 500.0), ("11:00", 59.0, -100.0),
                                 ("11:30", 61.0, -100.0)):
            r = _row(d, hhmm, "BUY", pnl > 0, pnl=pnl)
            rows.append(r)
            facts[lab.row_key(r)] = _fact("BUY", rsi14=rsi_v)
    grid = lab.robustness(rows, facts, lab.DEFAULT_PARAMS, SPLIT_DAYS, min_trades=1)
    rsi = {g["value"]: g for g in grid if g["family"] == "BUY_RSI"}
    assert [rsi[v]["profitable_train_validation"] for v in (52.0, 55.0, 58.0, 60.0)] == [False, True, False, False]
    assert not any(g["stable"] for g in rsi.values())
    by = {c.name: c for c in lab.predeclared_candidates()}
    assert lab.robustness_ok(by["BUY_RSI_CONFIRM"], grid) is False
    assert lab.robustness_ok(by["BUY_EMA_CONFIRM"], grid) is None


# --------------------------------------------------------------------------- report generation
def test_build_lab_and_report_generation(tmp_path):
    rows, facts = _dataset()
    rules = {"signal": "ORB", "buy": "CE", "sell": "PE"}
    forward = [{"date": "2026-10-06", "total": 41, "resolved": 33, "pending": 8, "wins": 22, "losses": 11,
                "invalid": 0, "accuracy": 66.67, "profit_factor": 4.25, "net_pnl": 30543.5, "max_drawdown": 6857.5}]
    a = lab.build_lab(base_rows=rows, facts=facts, days=TRAIN_DAYS + VAL_DAYS + OOS_DAYS, splits=SPLITS, rules=rules,
                      forward=forward)
    assert a["ranking"][0] in a["scores"] and lab.COMBINED in a["scores"]
    assert a["recommendation"] in ("KEEP_CURRENT", "TEST_CANDIDATE_IN_PAPER")
    assert a["scores"][lab.BASELINE]["OOS"]["evaluated"] == 15
    assert a["best_candidate"] and a["scores"][a["best_candidate"]]["family"] != lab.REFERENCE
    paths = lab.write_reports(a, tmp_path)
    for name in ("baseline.md", "candidate_summary.csv", "candidate_summary.md", "buy_analysis.csv",
                 "sell_analysis.csv", "time_analysis.csv", "regime_analysis.csv", "robustness.csv",
                 "walk_forward.csv", "final_ranking.md"):
        assert paths[name].is_file() and paths[name].stat().st_size > 0
    final = paths["final_ranking.md"].read_text(encoding="utf-8")
    for col in ("Train WR", "Validation WR", "OOS WR", "Train PF", "Validation PF", "OOS PF", "OOS P&L", "OOS DD",
                "OOS Trades", "Max Loss Streak", "Regime Stability", "Day Stability", "Lookahead", "Status"):
        assert col in final
    assert "CURRENT S2 vs BEST RESEARCH CANDIDATE" in final
    with paths["candidate_summary.csv"].open(encoding="utf-8") as fh:
        assert {r["candidate"] for r in csv.DictReader(fh)} == set(a["scores"])
    assert "2026-10-06" in paths["baseline.md"].read_text(encoding="utf-8")
    assert json.loads(paths["summary.json"].read_text(encoding="utf-8"))["live_trading_enabled"] is False
