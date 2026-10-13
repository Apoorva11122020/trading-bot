from __future__ import annotations

import inspect
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="S12.2 needs scikit-learn (pip install scikit-learn)")

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s12_2_constrained_ml as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.research import s4_six_month_lab as s4
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s11_ml_expanded as s11
from src.research import s12_2_constrained_ml as s122
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures
from tests.test_s10_ml_probability_strategy import FLAT, SIG, series

STEP = sd.STEP
TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S12-2"
DECISIONS = {s122.PROMISING, s122.REJECTED, s122.INSUFF, s122.UNSTABLE}
SYNTH = {"CLEAN_START": date(2026, 4, 1), "TRAIN_END": date(2026, 6, 30),
         "VALIDATION": (date(2026, 7, 1), date(2026, 7, 31)), "OOS": (date(2026, 8, 1), date(2026, 9, 30)),
         "MAX_LOAD": date(2026, 9, 30),
         "FOLDS": (("FOLD_1", date(2026, 4, 30), date(2026, 5, 1), date(2026, 5, 31)),
                   ("FOLD_2", date(2026, 5, 31), date(2026, 6, 1), date(2026, 6, 30))),
         "MIN_FOLD_TRAIN_SESSIONS": 3, "MIN_FOLD_VAL_SESSIONS": 3}


def _boom(*_a, **_k):
    raise AssertionError("S12.2 must read only the stored research roots")


def _s12_reports(d: Path, sessions, start: date, *, clean: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    summary = {"nifty": {"status": "PASS", "invalid_ohlc": 0}, "futures": {"status": "PASS", "invalid_ohlc": 0},
               "options": {"status": "PASS_WITH_QUARANTINE", "blocking_reasons": [], "invalid_rows": 0},
               "gate": {"conditions": {"causal_contract_selection_pass": True, "no_unresolved_timestamp_corruption": True,
                                       "no_material_leakage": True, "clean_history_sufficient": False}},
               "coverage_clean_history": {"options_ce": 0.96, "options_pe": 0.96},
               "clean_history": {"ending_train_end": {"start": start.isoformat(), "end": "2026-06-30"}},
               "raw_unchanged": True}
    man = {"sessions": [{"date": x.isoformat(), "clean": clean, "reasons": ""} for x in sessions]}
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (d / "clean_history_manifest.json").write_text(json.dumps(man), encoding="utf-8")


def _comparison_files(d: Path) -> dict[str, Path]:
    d.mkdir(parents=True, exist_ok=True)
    p = {k: d / f"{k}.csv" for k in ("S2", "S9", "S10", "S11")}
    p["S2"].write_text("month,scope,total_trades,wins,losses,win_rate,profit_factor,net_pnl,max_drawdown,average_r\n"
                       "2026-04,COMBINED,10,6,4,0.6,1.5,100,50,0.2\n", encoding="utf-8")
    p["S9"].write_text("month,scope,trades,wins,losses,accuracy,profit_factor,net_pnl,max_drawdown\n"
                       "MAY,COMBINED,8,3,5,0.375,0.7,-40,60\n", encoding="utf-8")
    p["S10"].write_text("month,split,scope,trades,resolved,wins,losses,accuracy,profit_factor,expectancy_r,net_pnl,"
                        "max_drawdown\nJUNE,TRAIN,COMBINED,5,5,3,2,0.6,1.2,0.1,20,10\n", encoding="utf-8")
    p["S11"].write_text("split,signal_time,session_date,side,outcome,pnl,r_multiple,holding_minutes\n"
                        "TRAIN (in-sample),2026-04-02T10:00:00+05:30,2026-04-02,CE,WIN,30,1.5,20\n", encoding="utf-8")
    return p


def _patch(mp) -> None:
    for k, v in SYNTH.items():
        mp.setattr(s122, k, v)
    mp.setattr(s11, "MIN_STABILITY_FOLDS", 2)
    mp.setattr(sm, "MIN_FIT_SAMPLES", 10)
    mp.setattr(sm, "MIN_CALIBRATION_SAMPLES", 5)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Full S12.2 pipeline on the synthetic store (Apr..Sep 2026, 3 sessions / month) with the window scaled to it:
    TRAIN Apr..Jun (2 inner folds), VALIDATION July, OOS Aug..Sep. The validation gate is forced to PASS so the OOS
    path and the chronology can be verified (the real gate is unit-tested separately)."""
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s122")
    six, out = base / "upstox_6m", base / "out"
    make_store(six)
    add_futures(six)
    cal = [d for d in s4.SixMonthStore(six).calendar if SYNTH["CLEAN_START"] <= d <= SYNTH["MAX_LOAD"]]
    _s12_reports(base / "s12", cal, SYNTH["CLEAN_START"])
    cmp = _comparison_files(base / "cmp")
    opened: list[str] = []
    real = u6.read_parquet

    def spy(p):
        opened.append(Path(p).as_posix())
        return real(p)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(u6, "read_parquet", spy)
        _patch(mp)
        mp.setattr(s122, "gate", lambda m, n: s122.PASS)
        a = script.run(expanded_root=base / "upstox_expanded", data_root=six, out_dir=out, s12_dir=base / "s12",
                       comparison_sources=cmp, mutation_days=2, check_sample=5, secret_values=[TOKEN])
    return SimpleNamespace(a=a, base=base, six=six, out=out, before=before, opened=opened, cmp=cmp)


# ------------------------------------------------------------------ bounded loading / max loaded timestamp (test 12)
def test_bounded_loader_never_opens_july_or_later(tmp_path):
    root = tmp_path / "upstox_6m"
    make_store(root)
    add_futures(root)
    opened: list[str] = []
    real = u6.read_parquet
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(u6, "read_parquet", lambda p: (opened.append(Path(p).as_posix()), real(p))[1])
        market, log = s122.load_clean_market([str(root)], "NIFTY", date(2026, 4, 1), s122.MAX_LOAD)
        for k in {c.instrument_key for v in market.chains.values() for c in v.values()}:
            market.premium(k)
    assert s122.MAX_LOAD == date(2026, 6, 30)
    assert opened and all("/2026-07" not in p and "/2026-08" not in p and "/2026-09" not in p for p in opened)
    fut = [Path(p).stem.split("_FUT_")[1] for p in opened if "_FUT_" in p]
    assert fut and all(e <= "2026-06-30" for e in fut)
    r = s122.read_summary(log)
    assert r["max_raw_timestamp"][:10] <= "2026-06-30" and not r["files_after_max_load"]
    assert max(market.sessions) <= date(2026, 6, 30) and market.spot["timestamp"].dt.date.max() <= date(2026, 6, 30)
    assert all(f["timestamp"].dt.date.max() <= date(2026, 6, 30) for f in market.frames.values())
    chk = s122.max_loaded_check(market, log)
    assert chk["passed"] and chk["max_loaded_date"] <= "2026-06-30"


def test_loader_refuses_after_june_and_october(tmp_path):
    for hi in (date(2026, 7, 1), date(2026, 9, 30), date(2026, 10, 5)):
        with pytest.raises(SystemExit):
            s122.load_clean_market([str(tmp_path)], "NIFTY", s122.CLEAN_START, hi)


def test_loader_drops_rows_before_window(tmp_path):
    root = tmp_path / "upstox_6m"
    make_store(root)
    add_futures(root)
    market, _ = s122.load_clean_market([str(root)], "NIFTY", date(2026, 5, 1), date(2026, 6, 30))
    assert min(market.sessions) >= date(2026, 5, 1) and market.spot["timestamp"].dt.date.min() >= date(2026, 5, 1)
    assert all(f["timestamp"].dt.date.min() >= date(2026, 5, 1) for f in market.frames.values())


def test_pipeline_max_loaded_and_files(env):
    la = env.a["lookahead"]["max_loaded_timestamp"]
    assert la["passed"] and la["max_loaded_date"] <= SYNTH["MAX_LOAD"].isoformat()
    assert all("/2026-10" not in p for p in env.opened)
    assert env.a["october"] == "NOT LOADED" and env.a["july_plus"] == "NOT LOADED"


def test_phase_view_hides_later_candles(env):
    m = env.a["_market"]
    v = s122.phase_view(m, date(2026, 5, 31))
    assert max(v.sessions) <= date(2026, 5, 31) and v.spot["timestamp"].dt.date.max() <= date(2026, 5, 31)
    key = next(iter(next(iter(m.chains.values())).values())).instrument_key
    p = v.premium(key)
    assert p is None or p["timestamp"].dt.date.max() <= date(2026, 5, 31)


# ------------------------------------------------------------------ data-quality gate
def test_s12_gate_rules(tmp_path):
    days = [date(2025, 9, 26), date(2025, 9, 29)]
    _s12_reports(tmp_path / "ok", days, date(2025, 9, 26))
    load = lambda d: (json.loads((d / "summary.json").read_text()),  # noqa: E731
                      json.loads((d / "clean_history_manifest.json").read_text()))
    s, m = load(tmp_path / "ok")
    assert s122.s12_gate(s, m, date(2025, 9, 26), date(2026, 6, 30))["passed"]
    bad = {**s, "nifty": {"status": "FAIL"}}
    assert not s122.s12_gate(bad, m, date(2025, 9, 26), date(2026, 6, 30))["passed"]
    bad = {**s, "options": {**s["options"], "invalid_rows": 3}}
    assert not s122.s12_gate(bad, m, date(2025, 9, 26), date(2026, 6, 30))["passed"]
    bad = {**s, "coverage_clean_history": {"options_ce": 0.5, "options_pe": 0.96}}
    assert not s122.s12_gate(bad, m, date(2025, 9, 26), date(2026, 6, 30))["passed"]
    _s12_reports(tmp_path / "dirty", days, date(2025, 9, 26), clean=False)
    assert not s122.s12_gate(*load(tmp_path / "dirty"), date(2025, 9, 26), date(2026, 6, 30))["passed"]
    _s12_reports(tmp_path / "late", days, date(2025, 11, 3))
    assert not s122.s12_gate(*load(tmp_path / "late"), date(2025, 9, 26), date(2026, 6, 30))["passed"]
    assert not s122.s12_gate(None, None)["passed"]


def test_data_quality_failure_stops_before_training(env, tmp_path):
    b = s122.build_study(market=env.a["_market"], read_log=[], availability=env.a["availability"],
                         data_quality={"passed": False, "reason": "synthetic failure"}, freeze_dir=tmp_path,
                         mutation_days=0, check_sample=2)
    assert b["ran"] is False and b["decision"] == s122.INSUFF and b["forward_paper"] == "NOT SAFE"
    assert not (tmp_path / "models").exists() and "S12.2 NOT RUN" in s122.terminal_text(b)


def test_data_quality_in_pipeline(env):
    dq = env.a["data_quality"]
    assert dq["passed"] and all(dq["checks"].values()) and "off_grid_quarantine" in dq
    assert env.a["data_ok"] and env.a["contract_validity"]["passed"] and env.a["dataset_checks"]["passed"]


# ------------------------------------------------------------------ lookahead tests 1-5
def test_future_scrambling_truncation_option_mutation(env):
    la = env.a["lookahead"]
    for k in ("future_candle_scrambling", "feature_truncation", "future_option_mutation"):
        assert la[k]["passed"], (k, la[k])


def test_rolling_feature_causality(env):
    r = env.a["lookahead"]["rolling_feature_causality"]
    assert r["passed"] and r["checked"] > 0


def test_contract_selection_causality(env):
    cs = env.a["lookahead"]["contract_selection_causality"]
    assert cs["passed"] and cs["options"]["passed"] and cs["futures"]["passed"]
    assert cs["futures"]["independent_of_price_and_volume"]


# ------------------------------------------------------------------ leakage tests 6-9
def test_scaler_and_calibration_leakage(env):
    lk = env.a["leakage"]
    assert lk["scaler_leakage"]["passed"] and lk["calibration_leakage"]["passed"]
    tc = lk["train_oos_contamination"]
    assert tc["passed"] and tc["selection_rows_and_labels_end_by_train_end"] and tc["fingerprint_unchanged"]
    assert tc["selection_max_option_read"] is None or tc["selection_max_option_read"][:10] <= "2026-06-30"
    meta = env.a["frozen"]["metadata"]
    assert all(d <= SYNTH["TRAIN_END"].isoformat() for d in meta["fit_days"] + meta["calibration_days"])


def test_threshold_isolation(env):
    a = env.a
    assert a["threshold"] in sm.THRESHOLDS and a["leakage"]["threshold_leakage"]["passed"]
    assert {r["threshold"] for r in a["threshold_rows"]} == set(sm.THRESHOLDS) == {0.55, 0.60, 0.65}
    st = a["stages"]
    assert st.index("threshold_selection") < st.index("validation_dataset") < st.index("freeze") \
        < st.index("validation_april")


def test_purge_embargo(env):
    p = env.a["leakage"]["purge_embargo_correctness"]
    assert p["passed"] and p["embargo_bars"] >= 6 and p["label_horizon_bars"] == 6 and p["fits_checked"] >= 3


# ------------------------------------------------------------------ tests 10-11
def test_deterministic_replay(env):
    d = env.a["lookahead"]["deterministic_replay"]
    assert d["passed"] and d["probabilities_identical"] and d["trades_identical"] and d["fingerprint_identical"]


def test_duplicate_trade_prevention(env):
    assert env.a["lookahead"]["duplicate_trade_prevention"]["passed"]
    for r in env.a["results"].values():
        ts = [t for t in r["trades"] if t["outcome"] in sd.EXECUTED]
        for x, y in zip(ts, ts[1:]):
            assert pd.Timestamp(y["entry_time"]) >= pd.Timestamp(x["exit_time"])
            assert pd.Timestamp(y["entry_time"]) == pd.Timestamp(y["signal_time"]) + STEP


def test_all_checks_pass(env):
    assert env.a["lookahead_ok"] and env.a["leakage_ok"]
    assert len(env.a["lookahead"]) + len(env.a["leakage"]) >= 12


# ------------------------------------------------------------------ OOS chronology
def test_oos_only_after_freeze_and_april(env):
    st = env.a["stages"]
    assert st.index("inner_folds") < st.index("freeze") < st.index("validation_april") < st.index("oos_dataset") \
        < st.index("oos_may_june")
    assert env.a["leakage"]["oos_untouched_during_selection"]["passed"]
    assert env.a["leakage"]["oos_untouched_during_selection"]["oos_rows_in_any_fit"] == 0
    assert set(env.a["results"]) == {"TRAIN", "VALIDATION", "OOS"}


def test_oos_not_run_when_validation_fails(env, tmp_path):
    with pytest.MonkeyPatch.context() as mp:
        _patch(mp)
        mp.setattr(s122, "gate", lambda m, n: s122.FAIL)
        b = s122.build_study(market=env.a["_market"], read_log=[], availability=env.a["availability"],
                             data_quality={"passed": True}, freeze_dir=tmp_path, mutation_days=0, check_sample=2)
    assert b["labels"]["OOS"] == s122.NOT_RUN and "oos_dataset" not in b["stages"] and "OOS" not in b["results"]
    assert b["dataset"]["oos_rows"] == 0 and b["decision"] == s122.REJECTED and b["forward_paper"] == "NOT SAFE"
    assert "NOT_RUN" in s122.terminal_text(b)


# ------------------------------------------------------------------ folds / models / label
def test_folds_chronological(env):
    folds = env.a["folds"]
    assert [f["fold"] for f in folds] == ["FOLD_1", "FOLD_2"] and all(f["status"] == s122.RUN for f in folds)
    for f in folds:
        assert f["train"][0] == SYNTH["CLEAN_START"].isoformat() and f["train"][1] < f["validation"][0]
    assert all(f["validation"][1] <= SYNTH["TRAIN_END"].isoformat() for f in folds)


def test_real_fold_plan():
    assert s122.CLEAN_START == date(2025, 9, 26) and s122.TRAIN_END == date(2026, 3, 31)
    assert s122.VALIDATION == (date(2026, 4, 1), date(2026, 4, 30))
    assert s122.OOS == (date(2026, 5, 1), date(2026, 6, 30)) and s122.MAX_LOAD == date(2026, 6, 30)
    assert [f[1:] for f in s122.FOLDS] == [(date(2025, 12, 31), date(2026, 1, 1), date(2026, 1, 31)),
                                         (date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 28)),
                                         (date(2026, 2, 28), date(2026, 3, 1), date(2026, 3, 31))]
    assert s122.FOLDS[-1][3] == s122.TRAIN_END


def test_only_lr_and_hgb(env):
    assert s122.FAMILIES == (sm.LR, sm.HGB)
    for f in env.a["folds"]:
        assert set(f["families"]) == {sm.LR, sm.HGB}
    assert set(env.a["families"].values()) <= {sm.LR, sm.HGB}
    lr, hgb = sm.make_estimator(sm.LR), sm.make_estimator(sm.HGB)
    assert lr.C == 1.0 and lr.class_weight == "balanced" and lr.penalty in ("l2", "deprecated")
    assert (hgb.max_depth, hgb.learning_rate, hgb.max_iter, hgb.min_samples_leaf) == (3, 0.05, 200, 50)


def test_l1_label_1_5r():
    s = series([FLAT, (100.0, 103.0, 99.0, 102.0), (102.0, 105.0, 101.0, 104.0), FLAT], SIG)
    r = sd.label_trade(s, SIG, 2.0, s122.HORIZON, target_r=s122.TARGET_R)
    assert (r["entry"], r["risk"], r["stop_loss"], r["target"]) == (100.0, 3.0, 97.0, 104.5)
    assert r["outcome"] == sd.WIN and s122.TARGET_R == 1.5 and s122.HORIZON == 6 and sd.STOP_ATR == 1.5


def test_pipeline_labels_use_1_5r(env):
    d = env.a["_data"]
    ok = d[d["outcome"].isin(sd.BINARY) & d["risk"].notna()]
    assert len(ok) and np.allclose(ok["target"], ok["entry"] + 1.5 * ok["risk"])
    assert np.allclose(ok["stop_loss"], ok["entry"] - ok["risk"])


# ------------------------------------------------------------------ decisions
def _m(resolved, pf, ex, dd=1.0, cov=0.1):
    return {"resolved": resolved, "profit_factor": pf, "expectancy_r": ex, "max_drawdown_r": dd, "coverage": cov}


def test_side_and_final_decisions():
    kw = dict(checks_ok=True, usable=True, stability=s122.STABLE, inner_positive=3, folds_run=3, oos_ran=True,
              degraded=False)
    assert s122.side_verdict(**kw, val=_m(40, 1.3, 0.1), oos=_m(60, 1.2, 0.08))[0] == s122.PROMISING
    assert s122.side_verdict(**{**kw, "stability": "MODEL_UNSTABLE"}, val=_m(40, 1.3, 0.1),
                             oos=_m(60, 1.2, 0.1))[0] == s122.UNSTABLE
    assert s122.side_verdict(**{**kw, "inner_positive": 1}, val=_m(40, 1.3, 0.1), oos=_m(60, 1.2, 0.1))[0] \
        == s122.UNSTABLE
    assert s122.side_verdict(**kw, val=_m(40, 0.8, -0.1), oos=None)[0] == s122.REJECTED
    assert s122.side_verdict(**kw, val=_m(12, 1.3, 0.1), oos=None)[0] == s122.INSUFF
    assert s122.side_verdict(**{**kw, "checks_ok": False}, val=None, oos=None)[0] == s122.REJECTED
    base = dict(checks_ok=True, any_usable=True, inner_trading_stable=True)
    sides = {"CE": s122.PROMISING, "PE": s122.REJECTED}
    assert s122.final_decision(**base, val_label=s122.PASS, oos_label=s122.PASS, sides=sides,
                               oos=_m(60, 1.2, 0.1))[0] == s122.PROMISING
    assert s122.final_decision(**base, val_label=s122.FAIL, oos_label=s122.NOT_RUN, sides=sides, oos=None)[0] \
        == s122.REJECTED
    assert s122.final_decision(**base, val_label=s122.INSUFFICIENT, oos_label=s122.NOT_RUN, sides=sides,
                               oos=None)[0] == s122.INSUFF
    assert s122.final_decision(**{**base, "inner_trading_stable": False}, val_label=s122.PASS,
                               oos_label=s122.PASS, sides=sides, oos=_m(60, 1.2, 0.1))[0] == s122.UNSTABLE
    assert s122.final_decision(**{**base, "checks_ok": False}, val_label=s122.PASS, oos_label=s122.PASS,
                               sides=sides, oos=_m(60, 1.2, 0.1))[0] == s122.REJECTED
    assert s122.gate(_m(29, 2.0, 0.5), 30) == s122.INSUFFICIENT and s122.gate(_m(30, 1.01, 0.01), 30) == s122.PASS


def test_sample_labels():
    assert sm.sample_label(29) == "INSUFFICIENT_DATA" and sm.sample_label(30) == "LOW_SAMPLE"
    assert sm.sample_label(50) == "MODERATE_SAMPLE" and sm.sample_label(100) == "STRONGER_SAMPLE"


# ------------------------------------------------------------------ reports / comparison / terminal / safety
def test_reports_and_token_leakage(env):
    for n in s122.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        t = p.read_text(encoding="utf-8")
        assert TOKEN not in t and "bearer " not in t.lower() and "AI predicts" not in t
    assert env.a["token_leak"]["passed"]
    ok, leaks = u6.token_leak_check([env.out], [TOKEN])
    assert ok and not leaks
    man = json.loads((env.out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert man["october"] == "NOT LOADED" and man["july_plus"] == "NOT LOADED" and man["option_contracts_used"] > 0


def test_coverage_funnel(env):
    r = env.a["results"]["VALIDATION"]["funnel"]
    assert r["final_trades"] <= r["passed_ev_gate"] <= r["passed_margin_gate"] <= r["passed_probability_gate"] \
        <= r["model_scored"] <= r["eligible_timestamps"]
    f = pd.read_csv(env.out / "coverage_funnel.csv")
    assert {"FOLD_1", "FOLD_2", s122.SPLIT_NAMES["VALIDATION"]} <= set(f["scope"])


def test_comparison_descriptive(env):
    cmp = pd.read_csv(env.out / "comparison_s2_s9_s10_s11_s12_2.csv")
    assert {"S2", "S9_FUTURES_VOLUME", "S10_SHORT_HISTORY", "S11_EXPANDED", s122.STRATEGY} <= set(cmp["system"])
    assert set(cmp["period"]) == {"APRIL", "MAY", "JUNE"}
    s2 = cmp[(cmp["system"] == "S2") & (cmp["period"] == "APRIL")].iloc[0]
    assert s2["trades"] == 10 and s2["resolved"] == 10 and s2["status"] == "AVAILABLE"
    s10r = cmp[(cmp["system"] == "S10_SHORT_HISTORY")].iloc[0]
    assert "IN-SAMPLE" in s10r["relation_to_period"]
    rows = s122.comparison_rows({}, {})
    assert all(r["status"] == "NOT_AVAILABLE" for r in rows)


def test_production_unchanged_and_paper(env):
    a = env.a
    assert a["production_unchanged"] is True and clean_script.production_hashes() == env.before
    assert a["live_orders"] == 0 and a["trading_mode"] == "PAPER" and a["live_trading_enabled"] is False
    assert a["decision"] in DECISIONS
    assert a["forward_paper"] == ("SAFE" if a["decision"] == s122.PROMISING else "NOT SAFE")


def test_terminal_format(env):
    text = s122.terminal_text(env.a)
    lines = text.splitlines()
    assert lines[0] == s122.HEADING
    order = ["CLEAN HISTORY:", "TRAIN:", "VALIDATION:", "OOS:", "JULY+:", "OCTOBER:", "INNER FOLDS:", "Fold 1:",
             "Fold 2:", "MODEL STABILITY:", "VALIDATION:", "OOS MAY-JUNE:", "70% TARGET:", "LOOKAHEAD:",
             "MODEL LEAKAGE:", "DATA INTEGRITY:", "MAX LOADED DATE:", "OCTOBER LOADED:", "PRODUCTION CHANGED:",
             "LIVE ORDERS:", "FORWARD PAPER:", "FINAL:"]
    pos = -1
    for needle in order:
        nxt = next(i for i, l in enumerate(lines) if i > pos and l.startswith(needle))
        pos = nxt
    assert "9.1 months" in lines and "NOT LOADED" in lines
    assert lines[lines.index("JULY+:") + 1] == "NOT LOADED" and lines[lines.index("OCTOBER:") + 1] == "NOT LOADED"
    assert lines[lines.index("OCTOBER LOADED:") + 1] == "NO"
    assert lines[lines.index("PRODUCTION CHANGED:") + 1] == "NO" and lines[lines.index("LIVE ORDERS:") + 1] == "0"
    assert lines[lines.index("FINAL:") + 1] in DECISIONS
    assert lines[lines.index("70% TARGET:") + 1].startswith(s122.TARGET_70_TEXT)
    for sc in ("CE:", "PE:", "COMBINED:"):
        assert any(l.startswith(sc) and "Win Rate" in l and "Max DD" in l and "Coverage" in l for l in lines)
    assert "AI predicts" not in text and "12-month validation" not in text


def test_no_tuning_resampling_or_order_code():
    for mod in (s122, script):
        src = inspect.getsource(mod)
        for bad in ("GridSearch", "RandomizedSearch", "train_test_split", "shuffle=True", "SMOTE", "imblearn",
                    "RandomOverSampler", "KFold(", "optuna", "place_order", "make_benchmarks", "LGBM"):
            assert bad not in src, (mod.__name__, bad)
    assert sm.THRESHOLDS == (0.55, 0.60, 0.65) and sm.MARGIN == 0.05
