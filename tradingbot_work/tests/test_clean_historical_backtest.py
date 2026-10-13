from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as script
from scripts import run_s4_six_month_research as s4script
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.options.strategy import round_to_tick
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s4_six_month_lab as s4
from src.safety import live_gate
from tests.test_s4_six_month_research import make_store

STEP = pd.Timedelta(minutes=5)


def _boom(*_a, **_k):
    raise AssertionError("clean backtest must read only the 6-month store")


def row(t: str, direction="BUY", outcome="WIN", state=sh.EVALUATED, pnl=None, r=None, day="2026-08-03", **kw):
    st = f"{day}T{t}:00+05:30"
    pnl = pnl if pnl is not None else {"WIN": 200.0, "LOSS": -100.0, "AMBIGUOUS": -100.0}.get(outcome, 0.0)
    base = {"session_date": day, "signal_time": st, "direction": direction, "state": state, "outcome": outcome,
            "pnl": pnl, "r_multiple": r if r is not None else (2.0 if outcome == "WIN" else -1.0),
            "entry_time": (pd.Timestamp(st) + STEP).isoformat(), "instrument_key": f"K{t}{direction}",
            "features": {"trend_regime": "RANGE", "vol_regime": "NORMAL_VOLATILITY"}}
    return {**base, **kw}


# ------------------------------------------------------------------ metric definitions
def test_accuracy_is_wins_over_wins_plus_losses():
    rows = [row("09:30"), row("09:35", outcome="LOSS"), row("09:40", outcome="AMBIGUOUS"),
            row("09:45", state=sh.DATA_UNAVAILABLE, outcome=None, pnl=0.0, reason="option_candle_missing"),
            row("09:50"), row("09:55", outcome="SCRATCH", pnl=0.0, r=0.0)]
    m = cb.metrics(rows)
    assert (m["wins"], m["losses"], m["ambiguous"], m["scratch"]) == (2, 1, 1, 1)
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["win_rate_denominator"] == "2 wins / 3 resolved (wins + losses)"
    assert m["conservative_win_rate"] == pytest.approx(2 / 5)
    assert m["unresolved_or_unavailable"] == 1 and m["signals"] == 6 and m["total_trades"] == 5
    assert m["gross_profit"] == 400.0 and m["gross_loss"] == 200.0 and m["net_pnl"] == 200.0
    assert m["profit_factor"] == pytest.approx(2.0)
    assert m["max_losing_streak"] == 2  # LOSS then AMBIGUOUS
    assert m["max_drawdown"] == pytest.approx(200.0)


def test_target_70_needs_sample_and_both_test_months():
    good = {"COMBINED": {"wins": 80, "losses": 20, "win_rate": 0.8}}
    bad = {"COMBINED": {"wins": 50, "losses": 50, "win_rate": 0.5}}
    tiny = {"COMBINED": {"wins": 9, "losses": 1, "win_rate": 0.9}}
    assert cb.target_70({"OOS": good, "FINAL_HOLDOUT": good}) == "ACHIEVED"
    assert cb.target_70({"OOS": good, "FINAL_HOLDOUT": bad}) == "NOT ACHIEVED"
    assert cb.target_70({"OOS": tiny, "FINAL_HOLDOUT": good}) == "INSUFFICIENT DATA"


# ------------------------------------------------------------------ duplicates / timestamps / contracts (unit)
def test_duplicate_trades_detected():
    assert cb.duplicate_check([row("09:30"), row("09:35")])["passed"]
    d = cb.duplicate_check([row("09:30"), row("09:30")])
    assert not d["passed"] and d["duplicate_signals"] == 1 and d["duplicate_trades"] == 1


def test_impossible_timestamps_detected():
    ok = row("10:00", exit_time="2026-08-03T11:00:00+05:30")
    assert cb.timestamp_checks([ok], "15:15", "14:30")["passed"]
    cases = {
        "WEEKEND_SIGNAL": row("10:00", day="2026-08-01"),
        "SIGNAL_AFTER_ENTRY_CUTOFF": row("14:30"),
        "SIGNAL_OUTSIDE_SESSION": row("09:10"),
        "ENTRY_NOT_AFTER_SIGNAL": row("10:00", entry_time="2026-08-03T10:00:00+05:30"),
        "EXIT_BEFORE_ENTRY": row("10:00", exit_time="2026-08-03T09:50:00+05:30"),
        "EXIT_AFTER_SQUARE_OFF": row("10:00", exit_time="2026-08-03T15:25:00+05:30"),
        "SESSION_DATE_MISMATCH": row("10:00", session_date="2026-08-04"),
    }
    for problem, r in cases.items():
        res = cb.timestamp_checks([r], "15:15", "14:30")
        assert not res["passed"] and problem in res["problems"], problem


def test_chronology_check():
    assert cb.chronology([row("09:30"), row("09:35")])["passed"]
    assert not cb.chronology([row("09:35"), row("09:30")])["passed"]


def test_split_labels_are_separate():
    assert cb.split_of(row("10:00", day="2026-06-30")) == "TRAIN"
    assert cb.split_of(row("10:00", day="2026-07-01")) == "VALIDATION"
    assert cb.split_of(row("10:00", day="2026-08-31")) == "OOS"
    assert cb.split_of(row("10:00", day="2026-09-01")) == "FINAL_HOLDOUT"


# ------------------------------------------------------------------ end-to-end on the synthetic 6-month store
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before, baseline = script.production_hashes(), repr(dr.BASELINE)
    base = tmp_path_factory.mktemp("clean")
    make_store(base / "upstox_6m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=base / "upstox_6m", out_dir=base / "out", mutation_days=2, check_sample=5)
    store = s4.SixMonthStore(base / "upstox_6m")
    rc, params, session, s2cfg, signal_fn = s4script.load_s2()
    return SimpleNamespace(a=a, base=base, before=before, baseline=baseline, store=store, params=params, rc=rc)


def test_signal_candle_cannot_access_future_candles(env):
    mut = env.a["checks"]["mutation"]
    assert mut is not None and mut["cuts"] > 0 and mut["passed"], mut
    assert env.a["checks"]["truncation"]["checked"] > 0 and env.a["checks"]["truncation"]["passed"]


def test_option_entry_is_next_candle_open_without_future_premium(env):
    assert env.a["checks"]["alignment"]["passed"], env.a["checks"]["alignment"]
    rows = [r for r in env.a["rows"] if r.get("state") == sh.EVALUATED]
    assert rows
    for r in rows:
        assert dra._ts(r["entry_time"]) == dra._ts(r["signal_time"]) + STEP
        prem = env.store.premium(r["instrument_key"])
        t = dra._ist_series(prem["timestamp"])
        o = float(prem.loc[(t == dra._ts(r["entry_time"])).to_numpy(), "open"].iat[0])
        assert r["entry"] == pytest.approx(round_to_tick(o, None, "nearest"))


def test_sl_target_do_not_use_future_information(env):
    rows = [r for r in env.a["rows"] if r.get("state") == sh.EVALUATED][:15]

    def warped(r):
        f = env.store.premium(r["instrument_key"]).copy()
        later = (dra._ist_series(f["timestamp"]) > dra._ts(r["entry_time"])).to_numpy()
        for col in ("open", "high", "low", "close"):
            f[col] = f[col].astype(float)
            f.loc[later, col] = f.loc[later, col] * 3.0
        return f

    assert rows
    features = imp.underlying_features(env.store.spot())
    for r in rows:
        x = dra.reevaluate([r], features, warped, lambda _r: None, env.params, env.rc.session.square_off)
        x = x[dra.key_of(r)]
        for k in ("entry", "premium_atr", "stop_loss", "target"):
            assert x[k] == pytest.approx(r[k]), k


def test_no_september_or_oos_used_for_selection(env):
    iso = env.a["checks"]["selection_isolation"]
    assert iso["passed"] and iso["refuses_post_validation_rows"] and iso["invariant_to_oos_and_holdout"]


def test_no_strategy_or_production_change(env):
    assert script.production_hashes() == env.before and repr(dr.BASELINE) == env.baseline
    assert env.a["production_unchanged"] is True and env.a["live_orders"] == 0
    assert env.a["trading_mode"] == "PAPER" and env.a["live_trading_enabled"] is False


def test_chronology_duplicates_timestamps_and_contracts(env):
    ck = env.a["checks"]
    assert ck["chronology"]["passed"] and ck["duplicates"]["passed"] and ck["timestamps"]["passed"], ck["timestamps"]
    assert ck["contract_validity"]["checked"] > 0 and ck["contract_validity"]["passed"], ck["contract_validity"]
    ts = [dra._ts(r["signal_time"]) for r in env.a["rows"]]
    assert ts == sorted(ts)
    for r in env.a["rows"]:
        if r.get("option_type"):
            assert r["option_type"] == ("CE" if r["direction"] == "BUY" else "PE")
            assert r["expiry_date"] >= date.fromisoformat(str(r["session_date"])[:10])


def test_contract_validity_flags_wrong_type(env):
    r = dict(next(x for x in env.a["rows"] if x.get("instrument_key")))
    bad = dict(r, option_type="PE" if r["option_type"] == "CE" else "CE")
    listed = lambda sd, e, k: sorted(env.store.chain.get((sd, e, k), {}))  # noqa: E731
    res = cb.contract_validity([bad], env.store.calendar, listed, lambda _r: True)
    assert not res["passed"] and res["invalid"] == 1


def test_splits_are_reported_separately_and_statuses_valid(env):
    a = env.a
    assert set(a["per_split"]) == {"TRAIN", "VALIDATION", "OOS", "FINAL_HOLDOUT"}
    total = sum(a["per_split"][k]["COMBINED"]["signals"] for k in a["per_split"])
    assert total == a["summary"]["generated_signals"]
    for k in ("OOS", "FINAL_HOLDOUT"):
        for s in cb.SCOPES:
            assert a["status"][k][s]["status"] in ("PROMISING", "REJECTED", "INSUFFICIENT_DATA")
    assert a["s5"]["contaminated"] is True
    assert a["lookahead_ok"] and a["data_ok"]


def test_reports_and_terminal_output(env):
    out = env.base / "out"
    for n in cb.REPORT_FILES:
        assert (out / n).is_file(), n
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["accuracy_definition"].startswith("wins / (wins + losses)")
    with (out / "trades.csv").open(encoding="utf-8") as fh:
        trades = list(csv.DictReader(fh))
    assert len(trades) == env.a["summary"]["generated_signals"]
    assert {t["split"] for t in trades} <= {"TRAIN", "VALIDATION", "OOS", "FINAL_HOLDOUT"}
    md = (out / "clean_backtest_report.md").read_text(encoding="utf-8")
    for label in ("A. TRAIN", "B. VALIDATION", "C. OOS", "D. FINAL HOLDOUT", "CONTAMINATED"):
        assert label in md
    text = cb.terminal_text(env.a)
    for label in ("CLEAN HISTORICAL BACKTEST COMPLETE", "TRAIN:", "VALIDATION:", "OOS AUGUST:",
                  "FINAL HOLDOUT SEPTEMBER:", "Trades", "Win Rate", "PF", "P&L", "Max DD", "70% TARGET:",
                  "LOOKAHEAD:\nPASS", "DATA INTEGRITY:\nPASS", "PRODUCTION CHANGED:\nNO", "LIVE ORDERS:\n0",
                  "historical backtest accuracy"):
        assert label in text, label
    for n in cb.REPORT_FILES:
        t = (out / n).read_text(encoding="utf-8").lower()
        assert "access_token" not in t and "bearer " not in t


def test_paper_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))
