from __future__ import annotations

import inspect
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="S11 needs scikit-learn (pip install scikit-learn)")

from scripts import download_s11_expanded_history as dl
from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s11_ml_expanded as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s11_history as hx
from src.research import s11_ml_expanded as s11
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures
from tests.test_s10_ml_probability_strategy import FLAT, SIG, series

STEP = sd.STEP
TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S11"
DECISIONS = {s11.PROMISING, s11.REJECTED, s11.INSUFF, s11.UNSTABLE}
SYNTH_FOLDS = (("FOLD_1", date(2026, 4, 30), date(2026, 5, 1), date(2026, 5, 31)),
               ("FOLD_2", date(2026, 5, 31), date(2026, 6, 1), date(2026, 6, 30)))


def _boom(*_a, **_k):
    raise AssertionError("S11 must read only the stored expanded / six-month research stores")


def _synthetic_store(root: Path) -> None:
    make_store(root)
    add_futures(root)
    lay = u6.Layout(root)
    lay.summary.write_text(json.dumps({"quality": {"pass": True}, "coverage": 1.0}), encoding="utf-8")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Full S11 pipeline on the synthetic six-month store (Apr..Sep 2026, 3 sessions / month). The history and fold
    rules are scaled down so the pipeline runs end to end; the July gate is forced to PASS so August / September
    execute and the chronology can be verified (the real gate is unit-tested separately)."""
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s11")
    six, out = base / "upstox_6m", base / "out"
    _synthetic_store(six)
    calls: list[dict] = []
    real_load = sd.load_market

    def spy(roots, underlying="NIFTY", max_day=None):
        calls.append({"roots": [Path(r).as_posix() for r in roots], "max_day": max_day})
        return real_load(roots, underlying, max_day)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(sd, "load_market", spy)
        mp.setattr(hx, "REQUESTED_START", date(2026, 4, 1))
        mp.setattr(hx, "MIN_HISTORY_MONTHS", 3)
        mp.setattr(hx, "MIN_MONTH_SESSIONS", 3)
        mp.setattr(hx, "MIN_MONTH_ELIGIBLE_SHARE", 0.0)
        mp.setattr(s11, "FOLDS", SYNTH_FOLDS)
        mp.setattr(s11, "MIN_FOLD_TRAIN_SESSIONS", 3)
        mp.setattr(s11, "MIN_FOLD_VAL_SESSIONS", 3)
        mp.setattr(s11, "MIN_STABILITY_FOLDS", 2)
        mp.setattr(sm, "MIN_FIT_SAMPLES", 10)
        mp.setattr(sm, "MIN_CALIBRATION_SAMPLES", 5)
        mp.setattr(s11, "gate", lambda m, n: s11.PASS)
        a = script.run(expanded_root=base / "upstox_expanded", data_root=six, out_dir=out, mutation_days=2,
                       check_sample=5, secret_values=[TOKEN], with_benchmarks=False)
    return SimpleNamespace(a=a, base=base, six=six, out=out, calls=calls, before=before)


# 1 ------------------------------------------------------------------ expanded date coverage
def test_expiry_coverage_and_resolvable_start():
    weekly = [date(2025, 3, 6) + timedelta(days=7 * i) for i in range(60)]
    cov = hx.expiry_coverage(weekly, date(2025, 1, 1), date(2026, 3, 31))
    assert cov["resolvable_start"] == "2025-03-07" and not cov["gaps"]
    assert cov["missing_periods"][0]["reason"].startswith("BEFORE_EARLIEST_LISTED_EXPIRY")
    gap = [e for e in weekly if not date(2025, 6, 1) <= e <= date(2025, 6, 30)]
    g = hx.expiry_coverage(gap, date(2025, 1, 1), date(2026, 3, 31))
    assert len(g["gaps"]) == 1 and date.fromisoformat(g["resolvable_start"]) > date(2025, 6, 30)
    assert hx.expiry_coverage([], date(2025, 1, 1), date(2026, 3, 31))["resolvable_start"] is None
    early = [date(2024, 12, 26) + timedelta(days=7 * i) for i in range(70)]
    assert hx.expiry_coverage(early, date(2025, 1, 1), date(2026, 3, 31))["resolvable_start"] == "2025-01-01"


def test_month_rows_and_history_range():
    days = [date(2025, 11, 3) + timedelta(days=i) for i in range(240) if (date(2025, 11, 3) + timedelta(days=i)).weekday() < 5]
    days = [d for d in days if d <= date(2026, 6, 30)]
    cand = {d: 60 for d in days}
    elig = {d: (10 if d.month == 1 and d.year == 2026 else 55) for d in days}
    rows = hx.month_rows(days, cand, elig, date(2025, 1, 1), date(2026, 6, 30))
    st = {r["month"]: r["status"] for r in rows}
    assert st["2025-01"] == hx.NO_DATA and st["2026-01"] == hx.LOW_ELIGIBILITY and st["2026-06"] == hx.CLEAN
    h = hx.history_range(rows, date(2025, 1, 1), date(2026, 6, 30))
    assert h["months"] == 5 and h["actual_start"].startswith("2026-02") and h["status"] == hx.HISTORY_INSUFFICIENT
    assert any(m["month"] == "2025-12" and m["status"] == hx.OUTSIDE_RUN for m in h["missing_periods"])


def test_history_and_manifest_in_pipeline(env):
    a = env.a
    h = a["history"]
    assert h["status"] == hx.HISTORY_OK and h["months"] == 3 and h["actual_start"] == "2026-04-01"
    assert a["train"]["range"] == [h["actual_start"], "2026-06-30"]
    man = json.loads((env.out / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert man["october"] == "NOT LOADED" and man["history_months"] == 3 and man["option_contracts_used"] > 0
    assert man["futures_candles"] > 0 and man["stores"]["upstox_6m"]["plan_sha256"]


def test_insufficient_history_stops_before_modelling(env, tmp_path):
    b = s11.build_study(market=env.a["_market"], availability=env.a["availability"], data_quality={"passed": True},
                        freeze_dir=tmp_path, benchmarks=lambda: {}, mutation_days=0, check_sample=2)
    assert b["ran"] is False and b["decision"] == s11.INSUFF and b["history_status"] == hx.HISTORY_INSUFFICIENT
    assert b["forward_paper"] == "NOT SAFE" and not (tmp_path / "models").exists()
    assert "S11 NOT RUN" in s11.terminal_text(b)


def test_off_grid_option_candles_are_dropped_not_shifted():
    ts = pd.DatetimeIndex([pd.Timestamp("2025-04-21 09:15", tz=sd.IST), pd.Timestamp("2025-04-21 09:16", tz=sd.IST),
                           pd.Timestamp("2025-04-21 09:20", tz=sd.IST), pd.Timestamp("2025-04-21 09:21", tz=sd.IST)])
    prem = pd.DataFrame({"timestamp": ts, "close": [1.0, 2.0, 3.0, 4.0]})
    m = SimpleNamespace(premium_fn=lambda _k: prem)
    stats = s11.quarantine_off_grid(m)
    got = m.premium_fn("K")
    assert got["timestamp"].tolist() == [ts[0], ts[2]] and got["close"].tolist() == [1.0, 3.0]
    rep = s11.quarantine_report(stats)
    assert rep["rows_dropped"] == 2 and rep["contracts_affected"] == 1 and rep["days"] == ["2025-04-21"]
    assert s11.on_grid_mask(prem).tolist() == [True, False, True, False]


# 2 ------------------------------------------------------------------ no October loading
def test_no_october_loading(env):
    assert env.calls and all(c["max_day"] == hx.RESEARCH_END for c in env.calls)
    assert all(env.six.as_posix() in c["roots"] and len(c["roots"]) == 1 for c in env.calls)
    m = env.a["_market"]
    assert max(m.sessions) <= hx.RESEARCH_END and m.spot["timestamp"].dt.date.max() <= hx.RESEARCH_END
    assert env.a["_data"]["session_date"].max() <= hx.RESEARCH_END.isoformat()
    assert env.a["lookahead"]["october_loading_guard"]["passed"]
    with pytest.raises(SystemExit):
        hx.guard_no_october(date(2026, 10, 1))
    with pytest.raises(SystemExit):
        dl.run(start=date(2026, 9, 1), end=date(2026, 10, 5), client=object(), catalog=object())
    with pytest.raises(SystemExit):
        dl.run(start=date(2025, 1, 1), end=date(2026, 4, 30), client=object(), catalog=object())


def test_research_roots_refuse_october(tmp_path):
    for name in ("upstox_fresh_oct", "upstox_6m"):
        (tmp_path / name / "metadata").mkdir(parents=True)
        (tmp_path / name / "metadata" / "plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        script.research_roots(tmp_path / "upstox_fresh_oct", tmp_path / "upstox_6m")
    assert script.research_roots(tmp_path / "missing", tmp_path / "upstox_6m") == [tmp_path / "upstox_6m"]


# 3 ------------------------------------------------------------------ causal features / rolling windows
def test_causal_features(env):
    la = env.a["lookahead"]
    for k in ("future_candle_scrambling", "feature_truncation", "future_option_price_mutation",
              "future_option_volume_mutation", "future_oi_mutation", "rolling_window_causality"):
        assert la[k]["passed"], (k, la[k])
    assert la["rolling_window_causality"]["checked"] > 0


# 4 ------------------------------------------------------------------ causal option / futures selection
def test_causal_option_selection(env):
    cs = env.a["lookahead"]["contract_selection_causality"]
    assert cs["passed"] and cs["options"]["passed"] and cs["futures"]["passed"]
    assert cs["futures"]["independent_of_price_and_volume"]


# 5 ------------------------------------------------------------------ L1 label correctness (1.5R target)
def test_l1_label_correctness():
    s = series([FLAT, (100.0, 103.0, 99.0, 102.0), (102.0, 105.0, 101.0, 104.0), FLAT], SIG)
    r = sd.label_trade(s, SIG, 2.0, s11.HORIZON, target_r=s11.TARGET_R)
    assert (r["entry"], r["risk"], r["stop_loss"], r["target"]) == (100.0, 3.0, 97.0, 104.5)
    assert r["outcome"] == sd.WIN and r["r_multiple"] == 1.5
    assert sd.label_trade(s, SIG, 2.0, s11.HORIZON)["outcome"] != sd.WIN  # S10 2R target not reached
    loss = series([FLAT, (100.0, 101.0, 96.0, 97.0)], SIG)
    assert sd.label_trade(loss, SIG, 2.0, s11.HORIZON, target_r=1.5)["outcome"] == sd.LOSS
    amb = series([FLAT, (100.0, 105.0, 96.0, 100.0)], SIG)
    assert sd.label_trade(amb, SIG, 2.0, s11.HORIZON, target_r=1.5)["outcome"] == sd.AMBIGUOUS
    assert s11.HORIZON == 6 and sd.STOP_ATR == 1.5


def test_pipeline_labels_use_1_5r(env):
    d = env.a["_data"]
    ok = d[d["outcome"].isin(sd.BINARY) & d["risk"].notna()]
    assert len(ok) and np.allclose(ok["target"], ok["entry"] + 1.5 * ok["risk"])
    assert np.allclose(ok["stop_loss"], ok["entry"] - ok["risk"])
    wins = d[d["outcome"] == sd.WIN]
    assert len(wins) and (wins["r_multiple"] >= 1.5 - 1e-9).all()
    assert np.isclose(wins.loc[wins["exit_reason"] == "TARGET", "r_multiple"], 1.5).all()


def test_ev_gate_uses_1_5():
    assert sm.decide(0.38, 0.10, 0.35, target_r=1.5)[1] == "EV_NOT_POSITIVE"
    assert sm.decide(0.38, 0.10, 0.35)[0] == "CE"
    side, why, p, ev = sm.decide(0.70, 0.20, 0.60, target_r=1.5)
    assert side == "CE" and why is None and ev == pytest.approx(1.5 * 0.7 - 0.3)


def test_s10_defaults_unchanged():
    assert sd.TARGET_R == 2.0
    assert inspect.signature(sd.build_samples).parameters["target_r"].default == sd.TARGET_R
    assert inspect.signature(sm.simulate).parameters["target_r"].default == sd.TARGET_R
    assert inspect.signature(sm.decide).parameters["target_r"].default == sd.TARGET_R


# 6 ------------------------------------------------------------------ purged split / embargo
def test_purged_split_and_embargo(env):
    lk = env.a["leakage"]
    assert lk["purge_correctness"]["passed"] and lk["purge_correctness"]["embargo_bars"] >= 6
    assert lk["purge_correctness"]["fits_checked"] >= 3


# 7 ------------------------------------------------------------------ scaler / calibration leakage
def test_scaler_and_calibration_leakage(env):
    lk = env.a["leakage"]
    assert lk["scaler_leakage"]["passed"] and lk["calibration_leakage"]["passed"]
    assert lk["train_oos_contamination"]["passed"]
    assert lk["train_oos_contamination"]["selection_rows_and_labels_end_by_train_end"]
    meta = env.a["frozen"]["metadata"]
    assert all(d <= "2026-06-30" for d in meta["fit_days"] + meta["calibration_days"])


# 8 ------------------------------------------------------------------ threshold isolation
def test_threshold_isolation(env):
    a = env.a
    assert a["threshold"] in sm.THRESHOLDS and a["leakage"]["threshold_leakage"]["passed"]
    assert {r["threshold"] for r in a["threshold_rows"]} == set(sm.THRESHOLDS)
    rows = [{"threshold": 0.55, "resolved": 40, "expectancy_r": 0.10},
            {"threshold": 0.60, "resolved": 31, "expectancy_r": 0.30},
            {"threshold": 0.65, "resolved": 5, "expectancy_r": 0.90}]
    assert s11.choose_threshold(rows)[0] == 0.60
    assert s11.choose_threshold([{**r, "resolved": 3} for r in rows])[0] == 0.55
    assert a["stages"].index("threshold_selection") < a["stages"].index("outer_dataset") \
        < a["stages"].index("freeze") < a["stages"].index("validation_july")


# 9 ------------------------------------------------------------------ fold ordering
def test_fold_ordering(env):
    folds = env.a["folds"]
    assert [f["fold"] for f in folds] == ["FOLD_1", "FOLD_2"] and all(f["status"] == s11.RUN for f in folds)
    for f in folds:
        assert f["train"][0] == env.a["history"]["actual_start"] and f["train"][1] < f["validation"][0]
    assert folds[0]["validation"][1] < folds[1]["validation"][0]
    assert all(f["validation"][1] <= "2026-06-30" for f in folds)
    assert len(s11.FOLDS) == 6 and [f[2] for f in s11.FOLDS] == sorted(f[2] for f in s11.FOLDS)
    assert all(tr < lo for _n, tr, lo, _hi in s11.FOLDS) and s11.FOLDS[-1][3] == hx.TRAIN_END


def test_family_stability_rules():
    def fold(name, auc, skill=0.01):
        m = {"roc_auc": auc, "brier": 0.24, "log_loss": 0.68, "ece": 0.02, "brier_skill": skill}
        t = {"profit_factor": 1.1, "expectancy_r": 0.05, "max_drawdown_r": 2.0, "resolved": 10}
        return {"fold": name, "status": s11.RUN, "_fits": {sm.LR: {"metrics": {"CE": m}, "trading": {0.55: {"CE": t}}}}}

    stable = s11.family_stability([fold(f"F{i}", 0.53) for i in range(6)], sm.LR, "CE", 0.55)
    assert stable["status"] == s11.STABLE
    one = s11.family_stability([fold("F0", 0.62)] + [fold(f"F{i}", 0.505) for i in range(1, 6)], sm.LR, "CE", 0.55)
    assert one["status"] == s11.UNSTABLE_MODEL
    few = s11.family_stability([fold("F0", 0.6)], sm.LR, "CE", 0.55)
    assert few["status"] == s11.FEW_FOLDS


# 10 ------------------------------------------------------------------ duplicate trade prevention
def test_duplicate_trade_prevention(env):
    assert env.a["lookahead"]["duplicate_trade_prevention"]["passed"]
    for r in env.a["results"].values():
        ts = [t for t in r["trades"] if t["outcome"] in sd.EXECUTED]
        for x, y in zip(ts, ts[1:]):
            assert pd.Timestamp(y["entry_time"]) >= pd.Timestamp(x["exit_time"])
            assert pd.Timestamp(y["entry_time"]) == pd.Timestamp(y["signal_time"]) + STEP


# 11 ------------------------------------------------------------------ deterministic results
def test_deterministic_results(env):
    d = env.a["lookahead"]["deterministic_replay"]
    assert d["passed"] and d["probabilities_identical"] and d["trades_identical"] and d["fingerprint_identical"]


# 12 ------------------------------------------------------------------ contract validity
def test_contract_validity(env):
    cv = env.a["contract_validity"]
    assert cv["passed"] and cv["expired_option_contracts"] == 0 and cv["expired_futures_contracts"] == 0
    assert cv["option_not_point_in_time_atm"] == 0


# 13 ------------------------------------------------------------------ chronology / September
def test_chronology_and_september_label(env):
    a = env.a
    st = a["stages"]
    assert st.index("inner_folds") < st.index("freeze") < st.index("validation_july") < st.index("oos_august") \
        < st.index("legacy_september")
    assert a["labels"]["LEGACY_HOLDOUT"] == hx.LEGACY_LABEL
    assert a["leakage"]["september_contamination_guard"]["passed"]
    md = (env.out / "s11_report.md").read_text(encoding="utf-8")
    assert hx.LEGACY_LABEL in md


def test_august_not_run_when_july_fails(env, tmp_path):
    with pytest.MonkeyPatch.context() as mp:
        for k, v in (("REQUESTED_START", date(2026, 4, 1)), ("MIN_HISTORY_MONTHS", 3), ("MIN_MONTH_SESSIONS", 3),
                     ("MIN_MONTH_ELIGIBLE_SHARE", 0.0)):
            mp.setattr(hx, k, v)
        mp.setattr(s11, "FOLDS", SYNTH_FOLDS)
        mp.setattr(s11, "MIN_FOLD_TRAIN_SESSIONS", 3)
        mp.setattr(s11, "MIN_FOLD_VAL_SESSIONS", 3)
        mp.setattr(sm, "MIN_FIT_SAMPLES", 10)
        mp.setattr(sm, "MIN_CALIBRATION_SAMPLES", 5)
        mp.setattr(s11, "gate", lambda m, n: s11.FAIL)
        b = s11.build_study(market=env.a["_market"], availability=env.a["availability"],
                            data_quality={"passed": True}, freeze_dir=tmp_path, benchmarks=lambda: {},
                            mutation_days=0, check_sample=2)
    assert b["labels"]["OOS"] == s11.NOT_RUN and "oos_august" not in b["stages"]
    assert not {"OOS", "LEGACY_HOLDOUT"} & set(b["results"]) and b["decision"] == s11.REJECTED


# 14 ------------------------------------------------------------------ reports, token leakage, production
def test_reports_and_token_leakage(env):
    for n in s11.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        t = p.read_text(encoding="utf-8")
        assert TOKEN not in t and "bearer " not in t.lower() and "AI predicts" not in t
    for n in ("ce_model.pkl", "pe_model.pkl", "preprocessor.pkl", "feature_columns.json", "thresholds.json",
              "model_metadata.json"):
        assert (env.out / "models" / n).is_file(), n
    assert env.a["token_leak"]["passed"]
    ok, leaks = u6.token_leak_check([env.out], [TOKEN])
    assert ok and not leaks


def test_production_unchanged_and_paper(env):
    a = env.a
    assert a["production_unchanged"] is True and clean_script.production_hashes() == env.before
    assert a["live_orders"] == 0 and a["trading_mode"] == "PAPER" and a["live_trading_enabled"] is False
    assert a["decision"] in DECISIONS and a["data_ok"]


def test_terminal_format(env):
    text = s11.terminal_text(env.a)
    lines = text.splitlines()
    assert lines[0] == s11.HEADING
    for needle in ("HISTORY:", "requested:", "actual:", "TRAIN:", "VALIDATION:", "OOS:", "LEGACY:", "OCTOBER:",
                   "NOT LOADED", "INNER WALK-FORWARD:", "Fold 1:", "Fold 2:", "MODEL STABILITY:",
                   "OUTER VALIDATION JULY:", "OOS AUGUST:", "S10 vs S11:", "70% TARGET:", "LOOKAHEAD:",
                   "MODEL LEAKAGE:", "DATA INTEGRITY:", "OCTOBER PROTECTED:", "PRODUCTION CHANGED:", "LIVE ORDERS:",
                   "FORWARD PAPER:", "FINAL:"):
        assert needle in lines or any(l.startswith(needle) for l in lines), needle
    assert lines[lines.index("PRODUCTION CHANGED:") + 1] == "NO"
    assert lines[lines.index("LIVE ORDERS:") + 1] == "0"
    assert lines[lines.index("OCTOBER PROTECTED:") + 1] == "YES"
    assert lines[lines.index("FINAL:") + 1] in DECISIONS
    assert "AI predicts" not in text


def test_coverage_funnel_and_comparison(env):
    funnel = pd.read_csv(env.out / "coverage_funnel.csv")
    assert {"FOLD_1", "FOLD_2", "JULY VALIDATION"} <= set(funnel["scope"])
    r = env.a["results"]["VALIDATION"]["funnel"]
    assert r["final_trades"] <= r["passed_ev_gate"] <= r["passed_margin_gate"] <= r["passed_probability_gate"] \
        <= r["model_scored"] <= r["eligible_timestamps"]
    cmp = pd.read_csv(env.out / "s2_vs_s9_vs_s10_vs_s11.csv")
    assert {"S2", "S9_FUTURES_VOLUME", "S10_ORIGINAL", "S10_SHORT_HISTORY_L1", s11.STRATEGY} <= set(cmp["system"])


# 15 ------------------------------------------------------------------ download probe (fake catalog, no network)
def test_probe_with_fake_catalog(tmp_path):
    class Catalog:
        def expired_expiries(self, _u):
            return [date(2025, 4, 3) + timedelta(days=7 * i) for i in range(52)]

    r = dl.run(start=date(2025, 1, 1), end=date(2026, 3, 31), root=tmp_path, probe_only=True, client=object(),
               catalog=Catalog())
    pr = r["probe"]
    assert pr["resolvable_start"] == "2025-04-04" and r["options"] is None and r["futures"] is None
    saved = json.loads(dl.probe_path(tmp_path).read_text(encoding="utf-8"))
    assert saved["resolvable_start"] == "2025-04-04" and TOKEN not in json.dumps(saved)


# 16 ------------------------------------------------------------------ no tuning / resampling / order code
def test_no_tuning_or_resampling_code():
    for mod in (hx, s11, script, dl):
        src = inspect.getsource(mod)
        for bad in ("GridSearch", "RandomizedSearch", "train_test_split", "shuffle=True", "SMOTE", "imblearn",
                    "RandomOverSampler", "KFold(", "optuna", "place_order"):
            assert bad not in src, (mod.__name__, bad)
    assert sm.THRESHOLDS == (0.55, 0.60, 0.65)
