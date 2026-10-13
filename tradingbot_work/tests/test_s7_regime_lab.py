from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s7_regime_lab as script
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.research import s2_baseline_drift_audit as dra
from src.research import s7_regime_lab as s7
from src.safety import live_gate
from tests.test_s4_six_month_research import make_store

STEP = pd.Timedelta(minutes=5)
SLOTS = (("10:00", "10:00-11:00"), ("11:30", "11:00-12:00"), ("12:15", "12:00-13:00"))
REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE")


def _boom(*_a, **_k):
    raise AssertionError("S7 lab must read only the 6-month store")


def mk(day: date, t: str, window: str, direction: str, win: bool, regime: str) -> dict:
    st = f"{day.isoformat()}T{t}:00+05:30"
    buy = direction == "BUY"
    f = {"ema20_vs_ema50": "EMA20>EMA50" if buy else "EMA20<=EMA50", "vwap_side": "ABOVE" if buy else "BELOW_OR_AT",
         "ema_sep_atr": 0.2 if buy else -0.2, "time_window": window, "vol_regime": "NORMAL_VOLATILITY",
         "break_atr": 0.5, "atr14": 20.0, "close": 24000.0, "ema20": 24010.0, "ema50": 24000.0, "rsi14": 55.0,
         "trend_regime": regime}
    return {"session_date": day.isoformat(), "signal_time": st, "direction": direction, "state": sh.EVALUATED,
            "decision": "ACCEPTED", "outcome": "WIN" if win else "LOSS", "pnl": 200.0 if win else -100.0,
            "r_multiple": 2.0 if win else -1.0, "entry_time": (pd.Timestamp(st) + STEP).isoformat(),
            "instrument_key": f"K{day}{t}{direction}", "features": f,
            "s6": {"market_regime": regime, "sign_changes": 2.0, "volume_level": "NORMAL"}}


def synth_rows() -> list[dict]:
    """BUY wins 2 of 3 in TREND_UP, SELL wins 2 of 3 in TREND_DOWN; every other signal loses. Apr..Sep."""
    rows, j, d = [], 0, date(2026, 4, 1)
    while d <= date(2026, 9, 30):
        if d.weekday() < 5:
            for t, w in SLOTS:
                reg = REGIMES[(j // 2) % 3]
                good = (j // 7) % 3 != 0
                rows.append(mk(d, t, w, "BUY", good and reg == "TREND_UP", reg))
                rows.append(mk(d, t, w, "SELL", good and reg == "TREND_DOWN", reg))
                j += 1
        d += timedelta(days=1)
    return rows


@pytest.fixture(scope="module")
def synth():
    rows = synth_rows()
    return SimpleNamespace(rows=rows, sel=s7.select(s7.pre_rows(rows)))


# ------------------------------------------------------------------ gates
def test_gates_predeclared_separate_and_limited():
    for d in s7.DIRECTIONS:
        pool = [g for g in s7.GATES if g.direction == d]
        assert 0 < len(pool) <= s7.MAX_GATES_PER_SIDE and all(g.rationale for g in pool)
    assert len({g.gid for g in s7.GATES}) == len(s7.GATES)
    buy = {c for g in s7.GATES if g.direction == "BUY" for c in g.conditions}
    sell = {c for g in s7.GATES if g.direction == "SELL" for c in g.conditions}
    assert "TREND_UP" in buy and "TREND_UP" not in sell and "TREND_DOWN" in sell


def test_gate_only_removes_signals_and_missing_feature_is_no_trade(synth):
    g = next(g for g in s7.GATES if g.gid == "S7-B1")
    rows = [dict(r) for r in synth.rows[:30]]
    rows[0] = dict(rows[0], s6=dict(rows[0]["s6"], market_regime=s7.UNAVAILABLE))
    kept = s7.gate_rows(g, rows)
    assert all(any(k is r for r in rows) for k in kept)
    assert dra.key_of(rows[0]) not in {dra.key_of(r) for r in kept}


def test_market_structure_and_choppiness():
    r = {"direction": "BUY", "features": {}, "s6": {"market_regime": "TREND_DOWN", "sign_changes": 7.0}}
    assert s7.market_structure(r) == "TREND_REVERSAL" and s7.choppiness(r) == "CHOPPY"
    r = {"direction": "SELL", "features": {}, "s6": {"market_regime": "TREND_DOWN", "sign_changes": 3.0}}
    assert s7.market_structure(r) == "TREND_CONTINUATION" and s7.choppiness(r) == "NON_CHOPPY"


# ------------------------------------------------------------------ coverage
def test_coverage_calculations():
    base = {"total_trades": 500, "win_rate": 0.35, "average_r": -0.1, "profit_factor": 0.8, "max_drawdown": 1000.0,
            "net_pnl": -5000.0}
    filt = {"total_trades": 280, "win_rate": 0.52, "average_r": 0.3, "profit_factor": 1.4, "max_drawdown": 400.0,
            "net_pnl": 4000.0}
    c = s7.compare(base, filt)
    assert c["retention"] == pytest.approx(0.56) and c["trade_reduction"] == pytest.approx(0.44)
    assert c["accuracy_change_pp"] == pytest.approx(17.0) and c["pf_change"] == pytest.approx(0.6)
    assert c["expectancy_change_r"] == pytest.approx(0.4) and c["dd_reduction"] == pytest.approx(0.6)


def test_tiny_sample_is_never_shortlisted(synth):
    tr = synth.sel["train"]
    for gid, t in tr.items():
        if t["metrics"]["total_trades"] < s7.MIN_TRAIN_TRADES:
            assert not t["eligible"]


# ------------------------------------------------------------------ selection isolation / determinism
def test_selection_uses_train_and_july(synth):
    assert s7.ids(synth.sel) == {"BUY": "S7-B1", "SELL": "S7-S1"}
    assert synth.sel["validation"]["S7-B1"]["label"] == s7.PASS
    assert synth.sel["train"]["S7-B1"]["stability"]["label"] == "MULTI_MONTH"


def test_oos_and_holdout_isolation(synth):
    iso = s7.isolation(synth.rows, synth.sel)
    assert iso["passed"], iso["checks"]
    with pytest.raises(ValueError):
        s7.select(synth.rows)
    worse = [dict(r, outcome="LOSS", pnl=-100.0, r_multiple=-1.0) if s7.cb.day_of(r) > s7.SELECTION_END else r
             for r in synth.rows]
    assert s7.ids(s7.select(s7.pre_rows(worse))) == s7.ids(synth.sel)


def test_deterministic_ranking(synth):
    pre = s7.pre_rows(synth.rows)
    assert s7.fingerprint(s7.select(list(reversed(pre)))) == s7.fingerprint(synth.sel)
    g = next(g for g in s7.GATES if g.gid == "S7-B1")
    assert s7.random_control(g, synth.rows) == s7.random_control(g, synth.rows)


def test_post_entry_information_cannot_change_gates(synth):
    assert s7.post_entry_independence(synth.rows)["passed"]


def test_one_month_profit_is_overfit_risk(synth):
    g = next(g for g in s7.GATES if g.gid == "S7-B1")
    only_may = [dict(r, outcome="LOSS", pnl=-100.0, r_multiple=-1.0)
                if r["direction"] == "BUY" and r["session_date"][:7] in ("2026-04", "2026-06") else r
                for r in synth.rows]
    assert s7.month_stability(g, only_may)["label"] == s7.OVERFIT


def test_final_status_rules():
    def blk(label, oos, cov, overfit=False):
        return {"validation_label": label, "per_split": {"OOS": oos}, "coverage": {"OOS": cov},
                "robustness": {"overfit_risk": overfit}}
    good = {"total_trades": 80, "profit_factor": 1.4, "average_r": 0.2}
    cov = {"expectancy_change_r": 0.4, "pf_change": 0.7, "dd_reduction": 0.5, "retention": 0.5}
    assert s7.final_status(blk(s7.PASS, good, cov), True) == s7.PROMISING
    assert s7.final_status(blk(s7.PASS, good, cov), False) == s7.REJECTED
    assert s7.final_status(blk(s7.PASS, good, cov, True), True) == s7.OVERFIT
    assert s7.final_status(blk(s7.PASS, dict(good, total_trades=8), cov), True) == s7.INSUFFICIENT
    assert s7.final_status(blk(s7.DEGRADED, good, cov), True) == s7.DEGRADED
    bad = {"total_trades": 80, "profit_factor": 0.7, "average_r": -0.2}
    assert s7.final_status(blk(s7.PASS, bad, dict(cov, dd_reduction=0.1, pf_change=-0.1)), True) == s7.REJECTED


# ------------------------------------------------------------------ end-to-end on the synthetic 6-month store
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before, baseline = clean_script.production_hashes(), repr(dr.BASELINE)
    base = tmp_path_factory.mktemp("s7")
    make_store(base / "upstox_6m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=base / "upstox_6m", out_dir=base / "out", mutation_days=2, check_sample=5)
    return SimpleNamespace(a=a, base=base, before=before, baseline=baseline)


def test_regime_features_causal_and_no_lookahead(env):
    lk = env.a["lookahead"]
    for k in ("feature_truncation", "regime_extra_truncation", "no_future_data_gate_mutation",
              "future_candle_scrambling", "chronological_replay", "entry_next_valid_candle",
              "no_post_entry_information"):
        assert lk[k]["passed"], (k, lk[k])
    assert env.a["lookahead_ok"]


def test_chronological_replay_and_next_candle_entry(env):
    ts = [dra._ts(r["signal_time"]) for r in env.a["rows"]]
    assert ts == sorted(ts)
    for r in env.a["rows"]:
        if r.get("state") == sh.EVALUATED:
            assert dra._ts(r["entry_time"]) == dra._ts(r["signal_time"]) + STEP


def test_selection_isolated_and_no_duplicate_trades(env):
    assert env.a["isolation"]["passed"], env.a["isolation"]["checks"]
    dc = env.a["data_checks"]
    assert dc["duplicates"]["passed"] and dc["timestamps"]["passed"] and dc["contract_validity"]["passed"]
    assert env.a["data_ok"]


def test_no_strategy_or_production_change(env):
    assert clean_script.production_hashes() == env.before and repr(dr.BASELINE) == env.baseline
    assert env.a["production_unchanged"] is True and env.a["live_orders"] == 0
    assert env.a["trading_mode"] == "PAPER" and env.a["live_trading_enabled"] is False


def test_reports_and_terminal_output(env):
    out = env.base / "out"
    for n in s7.REPORT_FILES:
        assert (out / n).is_file(), n
    json.loads((out / "summary.json").read_text(encoding="utf-8"))
    with (out / "regime_diagnostics.csv").open(encoding="utf-8") as fh:
        diag = list(csv.DictReader(fh))
    assert all(r["section"] in ("TRAIN_BUCKET", "TRAIN_MONTH") for r in diag)
    assert all(r.get("month", "") in ("", "2026-04", "2026-05", "2026-06") for r in diag)
    chosen = {g.gid for g in env.a["selection"]["selected"].values() if g}
    with (out / "candidate_results.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in ("OOS", s7.HOLDOUT_LABEL):
                assert r["gate"] in chosen
    text = s7.terminal_text(env.a)
    for label in ("S7 REGIME LAB COMPLETE", "BASELINE S2", "TRAIN:", "VALIDATION:", "OOS AUGUST:",
                  "HOLDOUT SEPTEMBER:", "BEST BUY REGIME FILTER:", "BEST SELL REGIME FILTER:", "BEST COMBINED:",
                  "retention:", "BASELINE vs FILTER:", "70% TARGET:", "LOOKAHEAD:\nPASS", "DATA INTEGRITY:\nPASS",
                  "OVERFITTING:", "PRODUCTION CHANGED:\nNO", "LIVE ORDERS:\n0", "FORWARD PAPER VALIDATION:",
                  "not live accuracy"):
        assert label in text, label
    for n in s7.REPORT_FILES:
        t = (out / n).read_text(encoding="utf-8").lower()
        assert "access_token" not in t and "bearer " not in t


def test_paper_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))
