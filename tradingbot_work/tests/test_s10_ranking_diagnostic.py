from __future__ import annotations

import hashlib
import inspect
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="S10.3 needs scikit-learn (pip install scikit-learn)")

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s10_label_geometry as lg_script
from scripts import run_s10_ranking_diagnostic as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.research import s10_label_geometry as lg
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s10_ml_probability as s10
from src.research import s10_ranking_diagnostic as rk
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures

TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S103"
IST = sd.IST


def _boom(*_a, **_k):
    raise AssertionError("S10.3 must not re-fit the frozen S10 model or read legacy stores")


def _hashes(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s103")
    six, s10dir = base / "upstox_6m", base / "s10_ml"
    make_store(six)
    add_futures(six)
    market = sd.load_market([six], max_day=sd.PRE_FREEZE_END)
    av = sd.availability_audit(market)
    flags = {"volume": av["use_option_volume"], "oi": av["use_option_oi"]}
    cols = sd.feature_columns(flags["volume"], flags["oi"])
    days = sd.days_in(market.sessions, sd.SPLITS["TRAIN"][0], rk.STUDY_END)
    data = sd.build_samples(market, days, **flags).data
    train_days = sorted(d for d in set(data["session_date"]) if d <= sd.SPLITS["TRAIN"][1].isoformat())
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sm, "MIN_FIT_SAMPLES", 10)
        mp.setattr(sm, "MIN_CALIBRATION_SAMPLES", 5)
        final = sm.fit_model(data, cols, train_days, {"CE": sm.HGB, "PE": sm.LR}, market.grid())
    meta = {"families": {"CE": sm.HGB, "PE": sm.LR}, "calibration": {s: final.sides[s].calibration for s in sd.SIDES},
            "status": {s: final.sides[s].status for s in sd.SIDES}, "fit_days": final.fit_days[:1],
            "calibration_days": final.calib_days[:1], "purge": final.purge, "fingerprint": sm.fingerprint(final, 0.55)}
    s10.save_artifacts(final, 0.55, meta, s10dir)
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
        lg_a = lg_script.run(data_root=six, s10_dir=s10dir, secret_values=[TOKEN])
        calls.clear()
        a = script.run(data_root=six, s10_dir=s10dir, secret_values=[TOKEN])
    return SimpleNamespace(a=a, lg_a=lg_a, out=s10dir / "ranking", s10dir=s10dir, calls=calls,
                           art_before=art_before, before=before)


def _frame(probs, outcomes, split="VALIDATION", side="CE"):
    r = {sd.WIN: 1.5, sd.LOSS: -1.0, sd.AMBIGUOUS: -1.0, sd.TIMEOUT: 0.1}
    t0 = pd.Timestamp(datetime.combine(date(2026, 7, 1), time(10, 0)), tz=IST)
    return pd.DataFrame({"split": split, "side": side, "prob": probs, "l1_outcome": outcomes,
                         "l1_r": [r[o] for o in outcomes],
                         "l1_y": [1.0 if o == sd.WIN else 0.0 if o == sd.LOSS else np.nan for o in outcomes],
                         "signal_time": [t0 + i * sd.STEP for i in range(len(probs))],
                         "instrument_key": [f"K{i}" for i in range(len(probs))]})


def test_bucket_math_unit():
    probs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    outs = [sd.WIN, sd.WIN, sd.LOSS, sd.TIMEOUT, sd.WIN, sd.LOSS, sd.LOSS, sd.AMBIGUOUS, sd.LOSS, sd.TIMEOUT]
    g = _frame(probs, outs)
    b = dict(rk.buckets(g))
    assert len(b["TOP10"]) == 1 and len(b["TOP20"]) == 2 and len(b["TOP30"]) == 3 and len(b["TOP50"]) == 5
    assert len(b["BOTTOM50"]) == 5 and len(b["ALL"]) == 10
    top20 = rk.bucket_stats(b["TOP20"])
    assert top20["WIN"] == 2 and top20["win_rate"] == 1.0
    allst = rk.bucket_stats(b["ALL"])
    assert allst["WIN"] == 3 and allst["LOSS"] == 4 and allst["TIMEOUT"] == 2 and allst["AMBIGUOUS"] == 1
    assert allst["win_rate"] == pytest.approx(3 / 7) and allst["raw_positive_rate"] == pytest.approx(0.3)
    assert allst["avg_realized_r"] == pytest.approx((3 * 1.5 - 5 + 0.2) / 10)
    assert rk.top_count(11, 0.10) == 2 and rk.top_count(0, 0.1) == 0
    assert rk.auc(g) == pytest.approx(11 / 12)
    rows = rk.decile_rows(g)
    assert rk._get(rows, split="VALIDATION", side="CE", bucket="TOP20")["lift_vs_overall"] == pytest.approx(7 / 3)


def test_ranking_order_and_ties_deterministic():
    g = _frame([0.5, 0.5, 0.9], [sd.LOSS, sd.WIN, sd.LOSS])
    r = rk.ranked(g)
    assert list(r["instrument_key"]) == ["K2", "K0", "K1"]


def test_precision_at_k_insufficient():
    g = _frame(list(np.linspace(0.9, 0.1, 25)), [sd.WIN if i % 3 == 0 else sd.LOSS for i in range(25)])
    rows = rk.precision_rows(g)
    by_k = {r["k"]: r for r in rows if r["side"] == "CE"}
    assert by_k[10]["status"] == "OK" and by_k[10]["precision"] == pytest.approx(4 / 10)
    assert by_k[20]["status"] == "OK"
    for k in (30, 50, 100):
        assert by_k[k]["status"] == "INSUFFICIENT_DATA" and by_k[k]["precision"] is None
    assert all(r["status"] == "INSUFFICIENT_DATA" for r in rows if r["side"] == "PE")


def test_probability_bucket_edges():
    g = _frame([0.19, 0.20, 0.35, 0.5, 0.69, 0.70, 0.95], [sd.WIN, sd.LOSS, sd.WIN, sd.LOSS, sd.WIN, sd.WIN, sd.TIMEOUT])
    rows = {r["bucket"]: r for r in rk.probability_rows(g) if r["side"] == "CE" and r["split"] == "VALIDATION"}
    assert [rows[n]["count"] for n in rk.PROB_NAMES] == [1, 1, 1, 0, 1, 1, 2]
    assert rows["0.70+"]["resolved"] == 1 and rows["0.70+"]["actual_win_rate"] == 1.0
    assert rows["0.40-0.50"]["actual_win_rate"] is None


def test_random_control_deterministic():
    g = _frame(list(np.linspace(0.9, 0.1, 40)), [sd.WIN if i < 10 else sd.LOSS for i in range(40)])
    d = rk.decile_rows(g)
    a, b = rk.random_rows(g, d), rk.random_rows(g, d)
    assert a == b
    ce = {r["bucket"]: r for r in a if r["side"] == "CE"}
    assert ce["TOP10"]["model_win_rate"] == 1.0 and ce["TOP10"]["seed"] == rk.RANDOM_SEED


def _tv(side, tr, va, va_auc=0.6, tr_auc=0.7, top10_res=50):
    return [{"side": side, "split": "TRAIN", "top10_lift": tr[0], "top20_lift": tr[1], "top30_lift": tr[2],
             "auc": tr_auc, "top10_resolved": 200},
            {"side": side, "split": "VALIDATION", "top10_lift": va[0], "top20_lift": va[1], "top30_lift": va[2],
             "auc": va_auc, "top10_resolved": top10_res}]


def test_side_quality_and_decision_rules():
    q = lambda tr, va, **k: rk.side_quality(_tv("CE", tr, va, **k), "CE")["quality"]  # noqa: E731
    assert q((1.3, 1.3, 1.2), (1.3, 1.25, 1.15)) == "GOOD"
    assert q((0.9, 1.3, 1.2), (1.3, 1.25, 1.15)) != "GOOD"
    assert q((1.5, 1.4, 1.3), (1.0, 1.05, 1.0)) == "OVERFIT"
    assert q((1.05, 1.05, 1.05), (1.15, 1.12, 1.0)) == "WEAK"
    assert q((1.05, 1.05, 1.05), (1.0, 1.02, 1.0)) == "NONE"
    assert q((1.3, 1.3, 1.2), (1.3, 1.25, 1.15), top10_res=10) == "INSUFFICIENT_DATA"
    assert q((1.3, 1.3, 1.2), (1.3, 1.25, 1.15), va_auc=0.52) == "WEAK"
    none = rk.side_quality(_tv("CE", (1, 1, 1), (1, 1, 1)), "CE")
    assert none["statement"] == "NO_MEANINGFUL_RANKING_EDGE"
    S = lambda a, b: rk.decide({"CE": {"quality": a}, "PE": {"quality": b}})  # noqa: E731
    assert S("GOOD", "GOOD") == "RANKING_EDGE_STRONG"
    assert S("GOOD", "NONE") == "RANKING_EDGE_WEAK" and S("WEAK", "OVERFIT") == "RANKING_EDGE_WEAK"
    assert S("OVERFIT", "NONE") == "OVERFIT_RANKING" and S("NONE", "INSUFFICIENT_DATA") == "NO_RANKING_EDGE"
    assert S("INSUFFICIENT_DATA", "INSUFFICIENT_DATA") == "INSUFFICIENT_DATA"


def test_study_consistency_and_reproduction(env):
    a, d = env.a, env.a["_data"]
    assert set(d["l1_outcome"]) <= set(sd.EXECUTED)
    assert all(t.time() <= lg.COMMON_LAST for t in d["signal_time"])
    assert set(d["split"]) <= {"TRAIN", "VALIDATION"}
    for r in a["deciles"]:
        assert r["samples"] == r["WIN"] + r["LOSS"] + r["TIMEOUT"] + r["AMBIGUOUS"]
        assert r["resolved"] == r["WIN"] + r["LOSS"]
    for split in ("TRAIN", "VALIDATION"):
        for s in sd.SIDES:
            g = {r["bucket"]: r for r in a["deciles"] if r["split"] == split and r["side"] == s}
            if g["ALL"]["samples"]:
                assert g["TOP50"]["samples"] + g["BOTTOM50"]["samples"] == g["ALL"]["samples"]
                assert g["TOP10"]["samples"] <= g["TOP20"]["samples"] <= g["TOP30"]["samples"]
    lg_l1 = {r["side"]: r for r in env.lg_a["learnability"] if r["scope"] == lg.COMMON and r["label"] == "L1"}
    for s in sd.SIDES:
        assert (d[(d["side"] == s) & (d["split"] == "VALIDATION")]["l1_y"].notna().sum()
                == lg_l1[s]["validation_binary"])
    if script.reference_auc(env.s10dir):
        assert a["reproduction"]["status"] == "REPRODUCED"
    else:
        assert a["reproduction"]["status"] == "NO_REFERENCE"
    assert a["decision"] in rk.SHORT
    assert _hashes(env.s10dir / "models") == env.art_before


def test_no_post_july_data(env):
    assert env.calls == [rk.STUDY_END] and env.a["no_post_july_data"] and not env.a["october_used"]
    assert env.a["_data"]["session_date"].max() <= rk.STUDY_END.isoformat()


def test_reports_terminal_and_safety(env):
    for n in rk.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        assert TOKEN not in p.read_text(encoding="utf-8")
    lines = rk.terminal_text(env.a).splitlines()
    assert lines[0] == "S10.3 RANKING DIAGNOSTIC COMPLETE"
    for needle in ("CE:", "PE:", "overall win rate:", "top10:", "top20:", "top30:", "AUC:", "lift:",
                   "TRAIN vs VALIDATION:", "top10 TRAIN:", "top10 VALIDATION:", "RANKING:"):
        assert any(l.startswith(needle) for l in lines), needle
    assert lines[lines.index("RANKING:") + 1] in rk.SHORT.values()
    assert "DO NOT RUN OCTOBER." in lines and "PRODUCTION CHANGED: NO" in lines and lines[-1] == "LIVE ORDERS: 0"
    assert env.a["production_unchanged"] is True and env.a["token_leak"]["passed"]
    assert clean_script.production_hashes() == env.before


def test_no_tuning_code():
    for mod in (rk, script):
        src = inspect.getsource(mod)
        for bad in ("choose_threshold", "choose_family", "GridSearch", "train_test_split", "shuffle=True",
                    "fit_model", "place_order", "SMOTE", "CalibratedClassifier"):
            assert bad not in src, (mod.__name__, bad)
