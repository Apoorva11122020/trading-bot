from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s6_clean_strategy_lab as script
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.research import s2_baseline_drift_audit as dra
from src.research import s6_clean_strategy_lab as s6
from src.safety import live_gate
from tests.test_s4_six_month_research import make_store

STEP = pd.Timedelta(minutes=5)
SLOTS = (("10:00", "10:00-11:00"), ("11:30", "11:00-12:00"), ("13:15", "13:00-14:00"))
REGIMES = ("RANGE", "CHOPPY", "TREND_UP")


def _boom(*_a, **_k):
    raise AssertionError("S6 lab must read only the 6-month store")


def mk(day: date, t: str, window: str, direction: str, win: bool, vwap: str, regime: str, **feat) -> dict:
    st = f"{day.isoformat()}T{t}:00+05:30"
    f = {"ema20_vs_ema50": "EMA20>EMA50" if direction == "BUY" else "EMA20<=EMA50", "vwap_side": vwap,
         "break_atr": 0.5, "option_vol_ratio": 1.0, "rsi14": 55.0 if direction == "BUY" else 45.0,
         "ema_sep_atr": 0.5 if direction == "BUY" else -0.5, "time_window": window, "trend_regime": "RANGE",
         "vol_regime": "NORMAL_VOLATILITY", "premium_mom3_pct": 1.0, **feat}
    return {"session_date": day.isoformat(), "signal_time": st, "direction": direction, "state": sh.EVALUATED,
            "decision": "ACCEPTED", "outcome": "WIN" if win else "LOSS", "pnl": 200.0 if win else -100.0,
            "r_multiple": 2.0 if win else -1.0, "entry_time": (pd.Timestamp(st) + STEP).isoformat(),
            "instrument_key": f"K{day}{t}{direction}", "features": f, "s6": {"market_regime": regime},
            "opt_confirm": True}


def synth_rows() -> list[dict]:
    """VWAP-confirmed signals win 2 of 3; everything else loses. Apr..Sep weekdays."""
    rows, j, d = [], 0, date(2026, 4, 1)
    while d <= date(2026, 9, 30):
        if d.weekday() < 5:
            for t, w in SLOTS:
                for direction in ("BUY", "SELL"):
                    side_ok = "ABOVE" if direction == "BUY" else "BELOW_OR_AT"
                    vwap = side_ok if j % 2 == 0 else ("BELOW_OR_AT" if direction == "BUY" else "ABOVE")
                    rows.append(mk(d, t, w, direction, vwap == side_ok and j % 3 != 0, vwap, REGIMES[(j // 2) % 3]))
                j += 1
        d += timedelta(days=1)
    return rows


def universe_of(rows: list[dict]) -> dict:
    return {v: [dict(r) for r in rows] for v in s6.VARIANTS}


@pytest.fixture(scope="module")
def synth():
    rows = synth_rows()
    uni = universe_of(rows)
    sel = s6.select(s6.pre_universe(uni))
    return SimpleNamespace(rows=rows, uni=uni, sel=sel)


# ------------------------------------------------------------------ candidate definitions
def test_candidates_are_predeclared_separate_and_limited():
    for d in s6.DIRECTIONS:
        pool = [c for c in s6.STAGE1 if c.direction == d]
        assert len(pool) == 15 and len(pool) + 4 <= s6.MAX_CANDIDATES_PER_SIDE
        assert all(c.rationale for c in pool)
    assert len({c.cid for c in s6.STAGE1}) == len(s6.STAGE1)
    buy_conds = {k for c in s6.STAGE1 if c.direction == "BUY" for k in c.conditions}
    sell_conds = {k for c in s6.STAGE1 if c.direction == "SELL" for k in c.conditions}
    assert "EMA_UP" in buy_conds and "EMA_UP" not in sell_conds and "EMA_DOWN" in sell_conds


def test_missing_feature_is_no_signal(synth):
    c = next(c for c in s6.STAGE1 if c.cid == "S6-B5")
    rows = synth_rows()[:12]
    rows[0]["features"]["rsi14"] = None
    uni = universe_of(rows)
    taken = s6.candidate_rows(c, uni)
    assert dra.key_of(rows[0]) not in {dra.key_of(r) for r in taken}
    assert s6.no_signal_count(c, uni) == 1


def test_market_regime_uses_project_choppy_rule():
    assert s6.market_regime("TREND_UP", 0) == "TREND_UP"
    assert s6.market_regime("RANGE", 6) == "CHOPPY"
    assert s6.market_regime("RANGE", 5) == "RANGE"
    assert s6.market_regime("RANGE", None) == s6.UNAVAILABLE


# ------------------------------------------------------------------ selection isolation / determinism
def test_selection_picks_supported_candidate_on_train_validation(synth):
    assert s6.ids(synth.sel) == {"BUY": "S6-B2", "SELL": "S6-S2"}
    assert synth.sel["validation"]["S6-B2"]["verdict"] == s6.VALIDATED
    assert synth.sel["selected"]["BUY"] == next(c for c in s6.STAGE1 if c.cid == "S6-B2")


def test_selection_refuses_oos_and_holdout_rows(synth):
    with pytest.raises(ValueError):
        s6.select(synth.uni)


def test_oos_and_september_cannot_affect_selection(synth):
    proof = s6.anti_overfit(synth.uni, synth.sel)
    assert proof["passed"], proof["checks"]
    worse = {v: [dict(r, outcome="LOSS", pnl=-100.0, r_multiple=-1.0) if s6.cb.day_of(r) > s6.SELECTION_END else r
                 for r in rows] for v, rows in synth.uni.items()}
    assert s6.ids(s6.select(s6.pre_universe(worse))) == s6.ids(synth.sel)


def test_ranking_is_deterministic(synth):
    pre = s6.pre_universe(synth.uni)
    again = s6.select({v: list(reversed(rows)) for v, rows in pre.items()})
    assert s6.fingerprint(again) == s6.fingerprint(synth.sel) == s6.fingerprint(s6.select(pre))


def test_train_ranking_uses_only_train(synth):
    pre = s6.pre_universe(synth.uni)
    flipped = {v: s6.s5._flip(rows, s6.TRAIN_END) for v, rows in pre.items()}
    assert s6.select(flipped)["ranking"] == synth.sel["ranking"]


# ------------------------------------------------------------------ statuses
def _m(trades, wins, pnl, avg_r, pf):
    return {"total_trades": trades, "wins": wins, "losses": trades - wins, "net_pnl": pnl, "average_r": avg_r,
            "profit_factor": pf}


def test_degradation_and_holdout_and_final_status():
    tr = _m(60, 30, 3000.0, 0.5, 2.0)
    assert s6.degradation(tr, _m(20, 10, 900.0, 0.4, 1.8)) == s6.STABLE
    assert s6.degradation(tr, _m(20, 8, 100.0, 0.1, 1.05)) == s6.DEGRADED
    assert s6.degradation(tr, _m(20, 4, -500.0, -0.2, 0.6)) == s6.FAILED
    assert s6.degradation(tr, _m(0, 0, 0.0, None, None)) == s6.INSUFFICIENT
    assert s6.holdout_status(_m(10, 9, 900.0, 1.0, 9.0)) == s6.INSUFFICIENT
    assert s6.holdout_status(_m(25, 12, 900.0, 0.3, 1.4)) == s6.PASSED_HO
    ok = {"overfit_risk": False}
    oos = _m(40, 20, 1500.0, 0.3, 1.5)
    assert s6.final_status(True, oos, s6.STABLE, ok, s6.PASSED_HO, True) == s6.PROMISING
    assert s6.final_status(True, oos, s6.STABLE, ok, s6.FAILED_HO, True) == s6.VALIDATED
    assert s6.final_status(True, oos, s6.STABLE, {"overfit_risk": True}, s6.PASSED_HO, True) == s6.OVERFIT
    assert s6.final_status(True, _m(10, 8, 900.0, 1.0, 5.0), s6.STABLE, ok, s6.PASSED_HO, True) == s6.INSUFFICIENT
    assert s6.final_status(False, oos, s6.STABLE, ok, s6.PASSED_HO, True) == s6.REJECTED
    assert s6.final_status(True, oos, s6.STABLE, ok, s6.PASSED_HO, False) == s6.REJECTED


def test_profit_concentration_flags_overfit_risk(synth):
    rows = [r for r in synth.rows if r["direction"] == "BUY" and r["features"]["vwap_side"] == "ABOVE"]
    assert not s6.robustness(rows)["overfit_risk"]
    one_month = [dict(r, outcome="LOSS", pnl=-100.0, r_multiple=-1.0) if r["session_date"][:7] != "2026-05" else r
                 for r in rows]
    rb = s6.robustness(one_month)
    assert rb["overfit_risk"] and "monthly" in rb["overfit_flags"]


def test_confirmation_uses_only_the_completed_next_candle():
    idx = pd.date_range("2026-08-03 09:15", periods=12, freq="5min", tz="Asia/Kolkata")
    bars = pd.DataFrame({"close": [100.0 + i for i in range(12)]}, index=idx)
    r = {"direction": "BUY", "breakout_level": 104.5}
    at = idx[5]
    assert s6.confirmation(r, bars, at) is True
    assert s6.confirmation(r, bars[bars.index <= at], at) is True
    later = bars.copy()
    later.loc[later.index > at, "close"] = 0.0
    assert s6.confirmation(r, later, at) is True


# ------------------------------------------------------------------ end-to-end on the synthetic 6-month store
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before, baseline = clean_script.production_hashes(), repr(dr.BASELINE)
    base = tmp_path_factory.mktemp("s6")
    make_store(base / "upstox_6m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=base / "upstox_6m", out_dir=base / "out", mutation_days=2, check_sample=5)
    return SimpleNamespace(a=a, base=base, before=before, baseline=baseline)


def test_lookahead_checks_pass(env):
    lk = env.a["lookahead"]
    for k in ("feature_truncation", "s6_feature_truncation", "option_confirmation_truncation",
              "future_candle_scrambling", "chronological_replay", "entry_alignment_next_candle"):
        assert lk[k]["passed"], (k, lk[k])
    assert env.a["lookahead_ok"]


def test_future_option_candles_cannot_affect_entries_and_no_synthetic_values(env):
    v = env.a["lookahead"]["entry_exit_variant_causality"]
    assert v["checked"] > 0 and v["passed"] and v["synthetic_or_wrong_entry"] == [], v


def test_selection_isolated_from_oos_and_holdout(env):
    assert env.a["anti_overfit"]["passed"], env.a["anti_overfit"]["checks"]


def test_no_duplicates_impossible_timestamps_or_data_leakage(env):
    dc = env.a["data_checks"]
    assert dc["duplicates"]["passed"] and dc["timestamps"]["passed"], dc["timestamps"]
    assert dc["contract_validity"]["passed"]
    assert all(v["duplicates"]["passed"] and v["timestamps"]["passed"] for v in dc["frozen"].values())
    assert env.a["data_ok"]


def test_variant_trades_use_real_exact_s2_evaluation(env):
    a = env.a
    for r in a["rows"]:
        if r.get("state") == sh.EVALUATED:
            assert dra._ts(r["entry_time"]) == dra._ts(r["signal_time"]) + STEP


def test_no_strategy_or_production_change(env):
    assert clean_script.production_hashes() == env.before and repr(dr.BASELINE) == env.baseline
    assert env.a["production_unchanged"] is True and env.a["live_orders"] == 0
    assert env.a["trading_mode"] == "PAPER" and env.a["live_trading_enabled"] is False


def test_reports_terminal_and_holdout_labels(env):
    out = env.base / "out"
    for n in s6.REPORT_FILES:
        assert (out / n).is_file(), n
    json.loads((out / "summary.json").read_text(encoding="utf-8"))
    with (out / "candidate_results.csv").open(encoding="utf-8") as fh:
        cand = list(csv.DictReader(fh))
    chosen = {c.cid for c in env.a["selection"]["selected"].values() if c}
    for r in cand:
        if r["split"] in ("OOS", s6.HOLDOUT_LABEL):
            assert r["candidate"] in chosen
    text = s6.terminal_text(env.a)
    for label in ("S6 CLEAN STRATEGY LAB COMPLETE", "BASELINE S2:", "TRAIN:", "VALIDATION:", "OOS AUGUST:",
                  "HOLDOUT SEPTEMBER:", "BEST BUY CANDIDATE:", "BEST SELL CANDIDATE:", "BEST COMBINED:",
                  "70% TARGET:", "LOOKAHEAD:\nPASS", "DATA INTEGRITY:\nPASS", "OVERFITTING:",
                  "PRODUCTION CHANGED:\nNO", "LIVE ORDERS:\n0", "not live accuracy"):
        assert label in text, label
    for n in s6.REPORT_FILES:
        t = (out / n).read_text(encoding="utf-8").lower()
        assert "access_token" not in t and "bearer " not in t


def test_paper_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))
