from __future__ import annotations

import hashlib
import inspect
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="S10.2 needs scikit-learn (pip install scikit-learn)")

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s10_label_geometry as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.research import s10_label_geometry as lg
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s10_ml_probability as s10
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures

TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S102"
IST, STEP = sd.IST, sd.STEP


def _boom(*_a, **_k):
    raise AssertionError("S10.2 must not re-fit the frozen S10 model or read legacy stores")


def _hashes(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s102")
    six, s10dir = base / "upstox_6m", base / "s10_ml"
    make_store(six)
    add_futures(six)
    market = sd.load_market([six], max_day=sd.PRE_FREEZE_END)
    av = sd.availability_audit(market)
    flags = {"volume": av["use_option_volume"], "oi": av["use_option_oi"]}
    cols = sd.feature_columns(flags["volume"], flags["oi"])
    days = sd.days_in(market.sessions, sd.SPLITS["TRAIN"][0], lg.STUDY_END)
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
        a = script.run(data_root=six, s10_dir=s10dir, secret_values=[TOKEN])
    return SimpleNamespace(a=a, out=s10dir / "label_geometry", s10dir=s10dir, calls=calls, art_before=art_before,
                           before=before, final=final)


def _series(rows, start):
    t = pd.DatetimeIndex([start + i * STEP for i in range(len(rows))])
    o, h, l, c = (np.array(v, dtype=float) for v in zip(*rows))
    return sd.OptionSeries(pd.DataFrame(index=t), t, o, h, l, c)


def test_target_r_parameter_unit():
    t0 = pd.Timestamp(datetime.combine(date(2026, 4, 6), time(10, 0)), tz=IST)
    flat = (100.0, 101.0, 99.0, 100.0)
    s = _series([flat, (100.0, 105.0, 99.0, 104.0)] + [flat] * 12, t0)
    assert sd.label_trade(s, t0, 2.0, 6, target_r=1.5)["outcome"] == sd.WIN
    assert sd.label_trade(s, t0, 2.0, 6, target_r=1.5)["target"] == pytest.approx(104.5)
    assert sd.label_trade(s, t0, 2.0, 6)["outcome"] == sd.TIMEOUT
    assert sd.label_trade(s, t0, 2.0, 6)["target"] == pytest.approx(106.0)
    assert [n for n, _, _ in lg.LABEL_DEFS] == ["L1", "L2", "L3", "L4"]
    assert [(t, h) for _, t, h in lg.LABEL_DEFS] == [(1.5, 6), (2.0, 6), (1.5, 12), (2.0, 12)]


def test_consistent_with_s10_labels(env):
    d = env.a["_data"]
    assert (d["L2_outcome"] == d["outcome"]).all()
    has12 = d["outcome_h12"].notna()
    assert has12.any() and (d.loc[has12, "L4_outcome"] == d.loc[has12, "outcome_h12"]).all()


def test_common_window_same_samples(env):
    a = env.a
    for split in ("TRAIN", "VALIDATION", "TOTAL"):
        for s in sd.SIDES:
            n = {lg._get(a["distribution"], scope=lg.COMMON, label=x, split=split, side=s)["eligible_samples"]
                 for x, _, _ in lg.LABEL_DEFS}
            assert len(n) == 1
    f = lg.scope_frame(a["_data"], lg.COMMON, 12)
    assert all(t.time() <= sd.last_signal(12) for t in f["signal_time"])
    for r in a["distribution"]:
        assert r["executed"] == r["WIN"] + r["LOSS"] + r["TIMEOUT"] + r["AMBIGUOUS"]
        assert r["eligible_samples"] == r["executed"] + r["NO_ENTRY"]


def test_label_geometry_row_wise_guarantees(env):
    d = env.a["_data"]
    f = lg.scope_frame(d, lg.COMMON, 12)
    for short, long_ in (("L1", "L3"), ("L2", "L4")):
        decided = f[f"{short}_outcome"].isin(sd.BINARY + (sd.AMBIGUOUS,))
        assert (f.loc[decided, f"{short}_outcome"] == f.loc[decided, f"{long_}_outcome"]).all()
        ok = f[f"{short}_mfe"].notna()
        assert (f.loc[ok, f"{long_}_mfe"] >= f.loc[ok, f"{short}_mfe"] - 1e-12).all()
        assert (f.loc[ok, f"{long_}_mae"] >= f.loc[ok, f"{short}_mae"] - 1e-12).all()
    for low, high in (("L1", "L2"), ("L3", "L4")):
        w = f[f"{high}_outcome"] == sd.WIN
        assert (f.loc[w, f"{low}_outcome"] == sd.WIN).all()


def test_learnability_rows(env):
    a, d = env.a, env.a["_data"]
    rows = a["learnability"]
    assert len(rows) == 4 * 2 * 2
    for r in rows:
        f = lg.scope_frame(d, r["scope"], r["horizon"])
        va = f[(f["side"] == r["side"]) & (f["split"] == "VALIDATION") & f[f"{r['label']}_y"].notna()]
        assert r["validation_binary"] == len(va)
        if r["status"] == "FITTED":
            assert r["roc_auc"] is None or 0.0 <= r["roc_auc"] <= 1.0
            assert 0.0 <= r["brier"] <= 1.0 and r["log_loss"] >= 0.0
    assert _hashes(env.s10dir / "models") == env.art_before


def test_positive_status_rules():
    assert lg.positive_status(99, 50, 0.5) == "INSUFFICIENT"
    assert lg.positive_status(200, 29, 0.5) == "INSUFFICIENT"
    assert lg.positive_status(200, 50, 0.05) == "SPARSE"
    assert lg.positive_status(200, 50, 0.12) == "SUFFICIENT"


def test_best_label_and_no_post_july(env):
    b = env.a["best"]
    assert b["name"] in {"L1", "L2", "L3", "L4"} and len(b["ranking"]) == 4
    assert b["verdict"] in ("LEARNABLE_FOR_FURTHER_RESEARCH", "NO_LABEL_MEANINGFULLY_LEARNABLE")
    assert env.calls == [lg.STUDY_END] and env.a["no_post_july_data"] and not env.a["october_used"]
    assert env.a["_data"]["session_date"].max() <= lg.STUDY_END.isoformat()


def test_reports_terminal_and_safety(env):
    for n in lg.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        assert TOKEN not in p.read_text(encoding="utf-8")
    lines = lg.terminal_text(env.a).splitlines()
    assert lines[0] == "S10.2 LABEL GEOMETRY COMPLETE"
    for needle in ("L1:", "L2:", "L3:", "L4:", "BEST LABEL FOR RESEARCH:", "name:", "Reason:",
                   "- positive sample rate", "- timeout rate", "- validation AUC", "- validation PR AUC",
                   "This is NOT strategy selection.", "This is NOT live accuracy."):
        assert any(l.startswith(needle) for l in lines), needle
    assert sum(l.startswith("CE:") for l in lines) == 4 and sum(l.startswith("PE:") for l in lines) == 4
    assert lines[lines.index("PRODUCTION CHANGED:") + 1] == "NO"
    assert lines[lines.index("LIVE ORDERS:") + 1] == "0" and lines[-1] == "NO OCTOBER DATA USED."
    assert env.a["production_unchanged"] is True and env.a["token_leak"]["passed"]
    assert clean_script.production_hashes() == env.before


def test_no_tuning_code():
    for mod in (lg, script):
        src = inspect.getsource(mod)
        for bad in ("choose_threshold", "choose_family", "GridSearch", "train_test_split", "shuffle=True",
                    "fit_model", "place_order", "SMOTE"):
            assert bad not in src, (mod.__name__, bad)
