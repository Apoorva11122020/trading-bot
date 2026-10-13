from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="S10.1 needs scikit-learn (pip install scikit-learn)")

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s10_ml_diagnostic as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.research import s10_ml_data as sd
from src.research import s10_ml_diagnostic as dx
from src.research import s10_ml_model as sm
from src.research import s10_ml_probability as s10
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures

TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S101"
IST = sd.IST
STEP = sd.STEP


def _boom(*_a, **_k):
    raise AssertionError("diagnostic must not re-fit models or read legacy stores")


def _hashes(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s101")
    six, s10dir = base / "upstox_6m", base / "s10_ml"
    make_store(six)
    add_futures(six)
    market = sd.load_market([six], max_day=sd.PRE_FREEZE_END)
    av = sd.availability_audit(market)
    flags = {"volume": av["use_option_volume"], "oi": av["use_option_oi"]}
    cols = sd.feature_columns(flags["volume"], flags["oi"])
    days = sd.days_in(market.sessions, sd.SPLITS["TRAIN"][0], dx.DIAG_END)
    data = sd.build_samples(market, days, **flags).data
    train_days = sorted(d for d in set(data["session_date"]) if d <= sd.SPLITS["TRAIN"][1].isoformat())
    val_days = sorted(d for d in set(data["session_date"]) if d > sd.SPLITS["TRAIN"][1].isoformat())
    fams = {"CE": sm.HGB, "PE": sm.LR}
    thr = 0.55
    with pytest.MonkeyPatch.context() as mp:
        # tiny synthetic sample: lower the fit minimums only for building the stand-in frozen model
        mp.setattr(sm, "MIN_FIT_SAMPLES", 10)
        mp.setattr(sm, "MIN_CALIBRATION_SAMPLES", 5)
        final = sm.fit_model(data, cols, train_days, fams, market.grid(), test_days=val_days)
    meta = {"families": fams, "calibration": {s: final.sides[s].calibration for s in sd.SIDES},
            "status": {s: final.sides[s].status for s in sd.SIDES},
            "fit_days": [final.fit_days[0], final.fit_days[-1]],
            "calibration_days": [final.calib_days[0], final.calib_days[-1]], "purge": final.purge,
            "fingerprint": sm.fingerprint(final, thr)}
    s10.save_artifacts(final, thr, meta, s10dir)
    models = {s: final for s in sd.SIDES}
    results = {k: {"trading": s10.evaluate(data[data["session_date"].isin(d)], models, thr)["trading"]}
               for k, d in (("TRAIN", train_days), ("VALIDATION", val_days))}
    (s10dir / "summary.json").write_text(json.dumps({"results": results}, default=str), encoding="utf-8")
    art_before = _hashes(s10dir / "models")
    calls = []
    real = sd.load_market

    def spy(roots, underlying="NIFTY", max_day=None):
        calls.append(max_day)
        return real(roots, underlying, max_day)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(sm, "fit_model", _boom)
        mp.setattr(sm, "fit_estimator", _boom)
        mp.setattr(sd, "load_market", spy)
        a = script.run(data_root=six, s10_dir=s10dir, secret_values=[TOKEN])
    return SimpleNamespace(a=a, base=base, s10dir=s10dir, out=s10dir / "diagnostic", final=final, data=data,
                           calls=calls, art_before=art_before, before=before, thr=thr)


def test_dataset_counts_no_silent_removal(env):
    a, cov, data = env.a, env.a["_coverage"], env.a["_data"]
    c = a["dataset_counts"]
    assert dx._count(c, "TOTAL", "raw_nifty_timestamps") == len(cov)
    assert sum(Counter(cov["reason"]).values()) == len(cov)
    eligible = int((cov["reason"] == "ELIGIBLE").sum())
    assert dx._count(c, "TOTAL", "eligible_prediction_timestamps") == eligible == data["signal_time"].nunique()
    for s in sd.SIDES:
        n = dx._count(c, "TOTAL", "eligible_samples", s)
        assert n == eligible
        assert n == sum(dx._count(c, "TOTAL", o, s) for o in (*sd.EXECUTED, sd.NO_ENTRY, sd.INVALID_LEVELS))
    days = sd.days_in(a["_coverage"]["session_date"].map(date.fromisoformat).unique(), date(2026, 4, 1),
                      dx.DIAG_END)
    assert len(cov) == 75 * len(days)


def test_coverage_matches_build_samples_funnel(env):
    cov = env.a["_coverage"]
    m = sd.load_market([env.base / "upstox_6m"], max_day=dx.DIAG_END)
    days = sd.days_in(m.sessions, sd.SPLITS["TRAIN"][0], dx.DIAG_END)
    built = sd.build_samples(m, days, **dx.flags_from_columns(env.final.columns))
    w = Counter(cov.loc[cov["in_window"], "reason"])
    assert w["ELIGIBLE"] == built.funnel["ELIGIBLE"]
    assert w[dx.ATR_HISTORY] + w[dx.ATR_INVALID] == built.funnel["OPTION_ATR_UNAVAILABLE"]
    for k in ("OPTION_SIGNAL_CANDLE_MISSING", "OPTION_DATA_MISSING", "OUTSIDE_PLANNED_STRIKES",
              "CONTRACT_LIST_UNAVAILABLE", "FUTURES_CANDLE_MISSING", "NO_LISTED_EXPIRY"):
        assert w[k] == built.funnel[k], k
    rows = env.a["coverage"]
    assert any(r["level"] == "ELIGIBLE_SAMPLE_KEPT" and r["reason"] == "MISSING_FEATURE" for r in rows)
    assert dx.main_exclusion(cov)[0] != "ELIGIBLE"


def test_excursions_unit():
    t0 = pd.Timestamp(datetime.combine(date(2026, 4, 6), time(10, 0)), tz=IST)
    rows = [(100, 101, 99, 100), (100, 104, 98, 103), (103, 109, 102, 108)] + [(100, 101, 99, 100)] * 6
    t = pd.DatetimeIndex([t0 + i * STEP for i in range(len(rows))])
    o, h, l, c = (np.array(v, dtype=float) for v in zip(*rows))
    s = sd.OptionSeries(pd.DataFrame(index=t), t, o, h, l, c)
    mfe, mae = dx.excursions(s, t0, 2.0)
    assert mfe == pytest.approx(3.0) and mae == pytest.approx(2 / 3)
    assert dx.excursions(s, t0 + 40 * STEP, 2.0) == (None, None)
    assert dx.excursions(None, t0, 2.0) == (None, None)


def test_label_diagnostic(env):
    a, data = env.a, env.a["_data"]
    for r in a["labels"]:
        d = data if r["split"] == "TOTAL" else data[data["split"] == r["split"]]
        g = d[d["side"] == r["side"]]
        assert r["target_before_stop"] == int((g["outcome"] == sd.WIN).sum())
        assert r["stop_before_target"] == int((g["outcome"] == sd.LOSS).sum())
        assert r["breakeven_win_rate"] == pytest.approx(1 / 3)
    win = data[(data["outcome"] == sd.WIN) & (data["exit_reason"] == "TARGET")]
    loss = data[(data["outcome"] == sd.LOSS) & (data["exit_reason"] == "STOP")]
    assert (win["mfe_r"] >= sd.TARGET_R - 1e-9).all() and (loss["mae_r"] >= 1.0 - 1e-9).all()
    assert env.a["frozen"]["horizon"] == sd.HORIZON == 6 and sd.STOP_ATR == 1.5 and sd.TARGET_R == 2.0


def test_probabilities_are_the_frozen_model(env):
    data = env.a["_data"]
    for s in sd.SIDES:
        g = data[data["side"] == s]
        assert np.allclose(g["p_cal"].to_numpy(), env.final.predict(s, g), equal_nan=True)
    for r in env.a["probability"]:
        if r["n"]:
            assert r["min"] <= r["p25"] <= r["median"] <= r["p75"] <= r["max"]
            assert r["count_ge_0.55"] >= r["count_ge_0.60"] >= r["count_ge_0.65"]


def test_funnel_and_reproduction(env):
    for r in env.a["funnel"]:
        assert r["eligible"] >= r["after_probability_gate"] >= r["after_margin_gate"] >= r["after_ev_gate"] \
            >= r["final_trades"] >= r["executed"] >= r["resolved"]
        assert r["after_ev_gate"] - r["active_trade_removed"] - r["horizon_undefined"] == r["final_trades"]
    assert env.a["reproduction"]["matches"] is True
    thr_rows = [r for r in env.a["funnel"] if r["primary_frozen_threshold"]]
    assert {r["threshold"] for r in thr_rows} == {env.thr}


def test_model_quality_and_calibration(env):
    q = env.a["quality"]
    assert {(r["split"], r["side"]) for r in q} == {(k, s) for k in dx.DIAG_SPLITS for s in (*sd.SIDES, "COMBINED")}
    for r in q:
        if r["roc_auc"] is not None and r["roc_auc_raw"] is not None:
            slope = env.a["frozen"]["calibrator_params"][r["side"]]["slope"]
            expect = r["roc_auc_raw"] if slope > 0 else 1 - r["roc_auc_raw"]
            assert r["roc_auc"] == pytest.approx(expect, abs=1e-9)
    data = env.a["_data"]
    for split in dx.DIAG_SPLITS:
        for s in sd.SIDES:
            n = int(((data["split"] == split) & (data["side"] == s) & data["outcome"].isin(sd.BINARY)
                     & data["p_cal"].notna()).sum())
            got = sum(r["count"] for r in env.a["calibration"] if r["split"] == split and r["side"] == s
                      and r["bucketed_by"] == "calibrated")
            assert got == n


def test_feature_health_and_importance(env):
    a = env.a
    assert [h["feature"] for h in a["feature_health"]] == env.final.columns
    assert all(h["inf_count_train"] == 0 for h in a["feature_health"])
    toy = pd.DataFrame({"a": [1.0] * 30, "b": np.arange(30.0), "side": ["CE", "PE"] * 15,
                        "outcome": [sd.WIN, sd.LOSS, sd.LOSS] * 10, "y": [1.0, 0.0, 0.0] * 10})
    h = {r["feature"]: r for r in dx.feature_health({"TRAIN": toy, "VALIDATION": toy}, ["a", "b"])}
    assert h["a"]["constant"] and not h["b"]["constant"] and h["a"]["top_value_share"] == 1.0
    assert set(a["stability"]) == set(sd.SIDES)
    assert all(r["wording"].startswith("predictive contribution") for r in a["importance"])


def test_contract_selection(env):
    for r in env.a["contracts"]:
        if r["side"] in sd.SIDES and r.get("samples"):
            assert r["atm_reresolution_mismatch"] == 0 and r["expiry_not_nearest_listed"] == 0
            assert r["expiry_before_session"] == 0 and r["spot_mismatch"] == 0


def test_root_causes_measured(env):
    rc = env.a["root_causes"]
    assert rc and [r["rank"] for r in rc] == list(range(1, len(rc) + 1))
    assert all(a["removed"] >= b["removed"] for a, b in zip(rc, rc[1:]))
    f = dx.primary_funnel(env.a["funnel"])
    by = {r["stage"]: r["removed"] for r in rc}
    if "probability gate" in by:
        assert by["probability gate"] == f["below_threshold"] + f["probability_unavailable"]
    d = env.a["diagnosis"]
    assert d["label"] in (*dx.LABELS, "MIXED")
    assert (d["label"] == "MIXED") == (len(d["components"]) > 1)
    assert {r["code"] for r in env.a["letters"]} == set("ABCDEFGHIJ")


def test_no_post_july_data_and_no_retraining(env):
    assert env.calls == [dx.DIAG_END]
    assert env.a["no_post_july_data"] and not env.a["october_used"]
    assert env.a["_data"]["session_date"].max() <= dx.DIAG_END.isoformat()
    assert _hashes(env.s10dir / "models") == env.art_before
    assert env.a["frozen"]["fingerprint_match"] in (True, False)


def test_reports_terminal_and_safety(env):
    for n in dx.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        t = p.read_text(encoding="utf-8")
        assert TOKEN not in t and "AI predicts" not in t
    text = dx.terminal_text(env.a)
    lines = text.splitlines()
    assert lines[0] == "S10.1 DIAGNOSTIC COMPLETE"
    for needle in ("CE samples:", "PE samples:", "CE WIN:", "CE LOSS:", "CE TIMEOUT:", "PE WIN:", "PE LOSS:",
                   "PE TIMEOUT:", "Main exclusion reason:", "percentage:", "CE probability range:",
                   "PE probability range:", "eligible:", "probability gate:", "margin gate:", "EV gate:",
                   "final trades:", "CE AUC:", "PE AUC:", "combined AUC:", "ROOT CAUSE:", "DIAGNOSIS:",
                   "PRODUCTION CHANGED:", "LIVE ORDERS:"):
        assert any(l.startswith(needle) for l in lines), needle
    assert lines[lines.index("PRODUCTION CHANGED:") + 1] == "NO"
    assert lines[lines.index("LIVE ORDERS:") + 1] == "0"
    assert lines[-1] == "NO OCTOBER DATA USED."
    assert env.a["production_unchanged"] is True and env.a["token_leak"]["passed"]
    assert clean_script.production_hashes() == env.before


def test_diagnostic_code_never_fits_or_tunes():
    for mod in (dx, script):
        src = inspect.getsource(mod)
        for bad in ("fit_model", "fit_estimator", ".fit(", "train_test_split", "GridSearch", "place_order",
                    "choose_threshold", "choose_family"):
            assert bad not in src, (mod.__name__, bad)
