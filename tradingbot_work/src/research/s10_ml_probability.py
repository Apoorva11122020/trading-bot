"""S10_ML_PROBABILITY (PAPER | RESEARCH ONLY | NO ORDERS).

A probability model, not a rule filter: for every eligible completed 5-minute candle a calibrated model per side
(CE_MODEL, PE_MODEL) estimates P(TARGET +2R before STOP -1R within 6 option candles) for the point-in-time ATM CE and
PE. The ML model estimates the historical probability of a predefined option outcome from causal features; it never
sees future prices. Historical performance is not live performance and no profit or accuracy is guaranteed.

Chronology (never reversed):
  1. OCTOBER data validated (data quality only - no features, labels or models touch it)
  2. dataset Apr..Sep built from causal features; labels use only the trade's own future option candles
  3. INNER selection inside TRAIN: fit Apr-May, choose model family per side + one threshold on June
  4. purged walk-forward folds inside TRAIN (stability only)
  5. FREEZE: final fit on TRAIN (Apr-Jun), artifacts written, fingerprint recorded
  6. VALIDATION July -> rejected candidates stop here (OOS NOT_RUN)
  7. OOS August (frozen) -> 8. September LEGACY_HOLDOUT_CONTAMINATED (report only)
  9. FRESH_UNSEEN_HOLDOUT October: loaded for the first time, frozen model run once (only if OOS passed)
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import sklearn

from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.research import s6_clean_strategy_lab as s6
from src.research import s9_orb_vwap_option_momentum as s9

HEADING = "S10 ML PROBABILITY STRATEGY COMPLETE"
STRATEGY = "S10_ML_PROBABILITY"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | historical backtest results (not live accuracy)"
IST = sd.IST
DISCLAIMER = ("ML model estimates the historical probability of a predefined option outcome from causal features. "
              "Historical performance != live performance. No guaranteed profit, no guaranteed 70% accuracy.")
SPLIT_NAMES = {"TRAIN": "TRAIN", "VALIDATION": "VALIDATION JULY", "OOS": "OOS AUGUST",
               "LEGACY_HOLDOUT": "SEPTEMBER", "FRESH_HOLDOUT": "FRESH OCTOBER HOLDOUT"}
SPLIT_LABELS = {"TRAIN": "IN_SAMPLE (frozen model on its own training period)", "VALIDATION": "OUTER_VALIDATION",
                "OOS": "FROZEN_OOS", "LEGACY_HOLDOUT": sd.LEGACY_LABEL, "FRESH_HOLDOUT": sd.FRESH_LABEL}
SCOPES = ("CE", "PE", "COMBINED")
WF_FOLDS = (("F1", (date(2026, 4, 1), date(2026, 4, 30)), (date(2026, 5, 1), date(2026, 5, 31))),
            ("F2", (date(2026, 4, 1), date(2026, 5, 31)), (date(2026, 6, 1), date(2026, 6, 30))),
            ("F3", (date(2026, 4, 1), date(2026, 6, 15)), (date(2026, 6, 16), date(2026, 6, 30))))

# Pre-declared gates (fixed before any model was fitted).
MIN_VAL_RESOLVED, MIN_OOS_RESOLVED, MIN_FRESH_RESOLVED = 20, 30, 30
MIN_STABLE_FOLDS = 2
MAX_DD_R, MIN_COVERAGE = 10.0, 0.005
MAX_STABLE_ECE, MIN_JACCARD = 0.10, 0.25
TARGET = 0.70
PASS, FAIL, INSUFFICIENT, NOT_RUN = "PASS", "FAIL", "INSUFFICIENT_DATA", "NOT_RUN"
PROMISING, REJECTED, INSUFF_DECISION, UNSTABLE = ("S10_PROMISING", "S10_REJECTED", "S10_INSUFFICIENT_DATA",
                                                  "S10_MODEL_UNSTABLE")
COST_MODEL = "COST_MODEL_UNAVAILABLE"
METRIC_KEYS = ("samples", "wins", "losses", "win_share", "roc_auc", "pr_auc", "brier", "log_loss", "ece",
               "climatology_brier", "brier_skill", "accuracy_at_threshold", "precision", "recall",
               "predicted_positive")
TRADE_KEYS = ("opportunities", "triggered", "trades", "wins", "losses", "ambiguous", "timeout", "unresolved_no_entry",
              "resolved", "accuracy", "accuracy_text", "coverage", "profit_factor", "gross_profit", "gross_loss",
              "net_pnl", "average_r", "expectancy_r", "total_r", "max_drawdown", "max_drawdown_r",
              "max_losing_streak", "average_holding_minutes", "sample")


def _days(data: pd.DataFrame, lo: date, hi: date) -> list[str]:
    return sorted(d for d in set(data["session_date"]) if lo.isoformat() <= d <= hi.isoformat())


def _rows(data: pd.DataFrame, days: Sequence[str]) -> pd.DataFrame:
    return data[data["session_date"].isin(sorted(set(days)))]


def usable(m: sm.S10Model, side: str) -> bool:
    s = m.sides[side]
    return s.status == sm.FITTED and s.calibrator is not None


def evaluate(data: pd.DataFrame, models: dict[str, sm.S10Model], threshold: float, *, horizon: int = sd.HORIZON,
             side_picker: Callable | None = None) -> dict[str, Any]:
    w = sm.wide_frame(data, models)
    trades, dec, opp = sm.simulate(w, threshold, horizon=horizon, side_picker=side_picker)
    return {"model": sm.evaluate_models(data, models, threshold), "trading": sm.scope_metrics(trades, opp),
            "trades": trades, "decisions": dict(dec), "opportunities": opp, "wide": w}


def random_picker(seed: int = sm.SEED) -> Callable[[pd.Timestamp, str], str]:
    rng = np.random.default_rng(seed)
    return lambda _ts, _side: "CE" if rng.random() < 0.5 else "PE"


def gate(m: dict, min_resolved: int) -> str:
    if m["resolved"] < min_resolved:
        return INSUFFICIENT
    pf, exp = m["profit_factor"], m["expectancy_r"]
    return PASS if exp is not None and exp > 0 and pf is not None and pf > 1 else FAIL


# ------------------------------------------------------------------ inner selection + walk-forward (TRAIN only)
def inner_selection(data: pd.DataFrame, cols: Sequence[str], grid: np.ndarray) -> dict[str, Any]:
    tr, va = _days(data, *sd.INNER_TRAIN), _days(data, *sd.INNER_VALIDATION)
    val = _rows(data, va)
    fits, rows = {}, {s: [] for s in sd.SIDES}
    for fam in sm.families():
        m = sm.fit_model(data, cols, tr, {s: fam for s in sd.SIDES}, grid, test_days=va)
        fits[fam] = m
        ev = sm.evaluate_models(val, {s: m for s in sd.SIDES}, sm.THRESHOLDS[1])
        for s in sd.SIDES:
            rows[s].append({"family": fam, "status": sm.FITTED if usable(m, s) else sm.INSUFFICIENT,
                            "calibration": m.sides[s].calibration, **{k: ev[s][k] for k in METRIC_KEYS}})
    chosen = {s: sm.choose_family(rows[s]) for s in sd.SIDES}
    combo = {s: fits[chosen[s][0]] for s in sd.SIDES}
    w = sm.wide_frame(val, combo)
    thr_rows = []
    for thr in sm.THRESHOLDS:
        trades, _, opp = sm.simulate(w, thr)
        thr_rows.append({"threshold": thr, **sm.trade_metrics(trades, opp)})
    thr, why = sm.choose_threshold(thr_rows)
    return {"inner_train_days": tr, "inner_validation_days": va, "family_rows": rows,
            "families": {s: chosen[s][0] for s in sd.SIDES}, "family_reason": {s: chosen[s][1] for s in sd.SIDES},
            "threshold_rows": thr_rows, "threshold": thr, "threshold_reason": why,
            "threshold_input_days": sorted(set(w["session_date"])) if len(w) else [], "_fits": fits}


def walk_forward(data: pd.DataFrame, cols: Sequence[str], grid: np.ndarray, fams: dict[str, str],
                 threshold: float) -> list[dict]:
    out = []
    for name, (a, b), (c, d) in WF_FOLDS:
        trd, ted = _days(data, a, b), _days(data, c, d)
        m = sm.fit_model(data, cols, trd, fams, grid, test_days=ted)
        ev = evaluate(_rows(data, ted), {s: m for s in sd.SIDES}, threshold)
        out.append({"fold": name, "train": f"{a}..{b}", "test": f"{c}..{d}", "train_days": len(trd),
                    "test_days": len(ted), "model": ev["model"], "trading": ev["trading"], "purge": m.purge,
                    "_model": m, "_test_days": ted})
    return out


def fold_stable(f: dict) -> bool:
    t, m = f["trading"]["COMBINED"], f["model"]["COMBINED"]
    return bool(t["expectancy_r"] is not None and t["expectancy_r"] > 0 and m["brier_skill"] is not None
                and m["brier_skill"] >= 0)


# ------------------------------------------------------------------ artifacts (research directory only)
def save_artifacts(model: sm.S10Model, threshold: float, meta: dict, out_dir: Path) -> dict[str, str]:
    d = out_dir / "models"
    d.mkdir(parents=True, exist_ok=True)
    blobs = {"ce_model.pkl": model.sides["CE"].estimator, "pe_model.pkl": model.sides["PE"].estimator,
             "preprocessor.pkl": model.preprocessor, "calibrator_ce.pkl": model.sides["CE"].calibrator,
             "calibrator_pe.pkl": model.sides["PE"].calibrator}
    paths = {}
    for name, obj in blobs.items():
        (d / name).write_bytes(pickle.dumps(obj, protocol=4))
        paths[name] = (d / name).as_posix()
    docs = {"feature_columns.json": list(model.columns),
            "thresholds.json": {"threshold": threshold, "candidates": list(sm.THRESHOLDS), "margin": sm.MARGIN,
                                "ev_gate": "EV = 2 * P_WIN - (1 - P_WIN) > 0 (secondary gate, pre-cost)",
                                "selected_on": "inner validation June 2026 (inside TRAIN only); frozen"},
            "model_metadata.json": meta}
    for name, obj in docs.items():
        (d / name).write_text(json.dumps(s6._jsonable(obj), indent=2, default=str), encoding="utf-8")
        paths[name] = (d / name).as_posix()
    return paths


# ------------------------------------------------------------------ anti-lookahead / leakage checks
def _vals(data: pd.DataFrame, cols: Sequence[str]) -> dict[tuple, tuple]:
    keep = ["instrument_key", "atr", *cols]
    return {(r["signal_time"], r["side"]): tuple(r[c] for c in keep) for r in data.to_dict("records")}


def _same_tuple(a: tuple, b: tuple) -> bool:
    for x, y in zip(a, b):
        if isinstance(x, (float, np.floating)) or isinstance(y, (float, np.floating)):
            fx, fy = float(x), float(y)
            if np.isnan(fx) and np.isnan(fy):
                continue
            if not np.isclose(fx, fy, rtol=1e-9, atol=1e-12):
                return False
        elif x != y:
            return False
    return len(a) == len(b)


def mutation_check(market: sd.MarketData, data: pd.DataFrame, cols: Sequence[str], days: Sequence[str],
                   flags: dict, model: sm.S10Model, *, seed: int, sources: dict, **kw) -> dict[str, Any]:
    """Candles after 12:00 mutated (or removed): every feature, contract, ATR and probability for signal candles
    completed by 12:00 must be identical to the full-data run."""
    bad, checked = [], 0
    for i, d in enumerate(days):
        day = date.fromisoformat(d)
        cut = pd.Timestamp(datetime.combine(day, datetime.min.time()).replace(hour=12), tz=IST)
        mm = sd.mutated_market(market, cut, seed=seed + i, **sources, **kw)
        b = sd.build_samples(mm, [day], volume=flags["volume"], oi=flags["oi"]).data
        a = data[data["session_date"] == d]
        a = a[a["signal_time"] + sd.STEP <= cut]
        b = b[b["signal_time"] + sd.STEP <= cut]
        va, vb = _vals(a, cols), _vals(b, cols)
        checked += len(va)
        if set(va) != set(vb):
            bad.append(f"{d}|row set differs")
            continue
        bad += [f"{k[0].isoformat()}|{k[1]}" for k in va if not _same_tuple(va[k], vb[k])]
        for side in sd.SIDES:
            pa = model.predict(side, a[a["side"] == side])
            pb = model.predict(side, b[b["side"] == side])
            if not np.allclose(pa, pb, equal_nan=True):
                bad.append(f"{d}|{side}|probability")
    return {"days": len(days), "rows_checked": checked, "mismatches": bad[:20], "passed": bool(checked) and not bad}


def truncation_check(market: sd.MarketData, data: pd.DataFrame, cols: Sequence[str], flags: dict,
                     sample: int) -> dict[str, Any]:
    """Each sampled row rebuilt from data truncated at its signal candle (every later candle removed)."""
    ts_all = sorted(set(data["signal_time"]))
    picked = ts_all[::max(1, len(ts_all) // max(sample, 1))][:sample]
    bad = []
    for ts in picked:
        tm = sd.mutated_market(market, ts, drop=True)
        b = sd.build_samples(tm, [ts.date()], volume=flags["volume"], oi=flags["oi"]).data
        va = _vals(data[data["signal_time"] == ts], cols)
        vb = _vals(b[b["signal_time"] == ts], cols)
        if set(va) != set(vb) or any(not _same_tuple(va[k], vb[k]) for k in va):
            bad.append(ts.isoformat())
    return {"checked": len(picked), "mismatches": bad[:20], "passed": bool(picked) and not bad}


def contract_causality_check(market: sd.MarketData, data: pd.DataFrame, sample: int) -> dict[str, Any]:
    rows = data.iloc[::max(1, len(data) // max(sample, 1))].head(sample).to_dict("records") if len(data) else []
    bad = []
    for r in rows:
        d = date.fromisoformat(r["session_date"])
        exp = sd.nearest_expiry(market.calendar, d)
        c, why = sd.resolve_contract(market, d, exp, r["side"], r["spot"])
        ok = (exp is not None and exp.isoformat() == r["expiry"] and date.fromisoformat(r["expiry"]) >= d
              and c is not None and c.instrument_key == r["instrument_key"]
              and abs(r["spot"] - market.spot_close()[r["signal_time"]]) < 1e-9)
        if not ok:
            bad.append(f"{r['signal_time'].isoformat()}|{r['side']}|{why}")
    return {"checked": len(rows), "mismatches": bad[:20], "passed": bool(rows) and not bad}


def leakage_checks(data: pd.DataFrame, final: sm.S10Model, inner: dict, fits: Sequence[tuple],
                   grid: np.ndarray) -> dict[str, dict]:
    lo, hi = sd.SPLITS["TRAIN"]
    later = set(data.index[data["session_date"] > hi.isoformat()])
    ids = {"preprocessor": final.pre_fit_ids,
           **{f"estimator_{s}": final.sides[s].fit_ids for s in sd.SIDES},
           **{f"calibrator_{s}": final.sides[s].calib_ids for s in sd.SIDES}}
    contamination = {k: sm.ids_within(data, v, lo, hi) and not (set(v) & later) for k, v in ids.items()}
    X = data.loc[final.pre_fit_ids, final.columns].to_numpy(dtype=float)
    imp = final.preprocessor.named_steps["imputer"]
    sc = final.preprocessor.named_steps["scaler"]
    with np.errstate(all="ignore"):
        med = np.nanmedian(X, axis=0) if len(X) else np.array([])
    med = np.where(np.isnan(med), 0.0, med)
    imp_ok = bool(len(X)) and np.allclose(imp.statistics_, med)
    scaler_ok = bool(len(X)) and np.allclose(sc.mean_, imp.transform(X).mean(axis=0)) \
        and int(sc.n_samples_seen_) == len(X)
    cal = {}
    for s in sd.SIDES:
        c_ids, f_ids = set(final.sides[s].calib_ids), set(final.sides[s].fit_ids)
        cd = set(data.loc[list(c_ids), "session_date"]) if c_ids else set()
        after_fit = all(d > max(final.fit_days) for d in cd) if final.fit_days else True
        cal[s] = bool(not (c_ids & f_ids) and cd <= set(final.calib_days) and after_fit
                      and sm.ids_within(data, list(c_ids), lo, hi))
    purge_bad = []
    for name, m, test_days in fits:
        test = _rows(data, test_days)
        for label, idx in (("fit", m.pre_fit_ids), *((f"calib_{s}", m.sides[s].calib_ids) for s in sd.SIDES)):
            if idx and sm.purge_violations(data.loc[list(idx)], test, grid):
                purge_bad.append(f"{name}|{label}")
    vlo, vhi = sd.INNER_VALIDATION
    thr_days = inner["threshold_input_days"]
    redo, _ = sm.choose_threshold(inner["threshold_rows"])
    return {
        "train_oos_contamination": {"checks": contamination, "passed": all(contamination.values())},
        "scaler_leakage": {"fit_rows": len(X), "passed": bool(scaler_ok)},
        "imputer_leakage": {"fit_rows": len(X), "passed": bool(imp_ok)},
        "probability_calibration_leakage": {"checks": cal, "passed": all(cal.values())},
        "threshold_leakage": {"input_days": f"{thr_days[0] if thr_days else '-'}..{thr_days[-1] if thr_days else '-'}",
                              "passed": bool(all(vlo.isoformat() <= d <= vhi.isoformat() for d in thr_days)
                                             and redo == inner["threshold"])},
        "purging_correctness": {"fits_checked": len(fits), "violations": purge_bad, "passed": not purge_bad},
    }


def dataset_checks(data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        return {"rows": 0, "passed": False, "reason": "NO_ELIGIBLE_SAMPLES"}
    st = pd.DatetimeIndex(data["signal_time"])
    ends = pd.DatetimeIndex(data["event_end"])
    last_end = pd.DatetimeIndex([pd.Timestamp(datetime.combine(t.date(), sd.LAST_EXIT_END), tz=IST) for t in st])
    per_ts = data.groupby("signal_time")["side"].nunique()
    y_ok = ((data["outcome"] == sd.WIN) == (data["y"] == 1)).all() and ((data["outcome"] == sd.LOSS)
                                                                        == (data["y"] == 0)).all()
    out = {"rows": int(len(data)),
           "duplicate_samples": int(data.duplicated(["signal_time", "side"]).sum()),
           "off_grid_signals": int(((st.minute % 5 != 0) | (st.second != 0)).sum()),
           "entry_not_next_candle": int((pd.DatetimeIndex(data["entry_time"]) != st + sd.STEP).sum()),
           "label_window_after_last_exit": int((ends > last_end).sum()),
           "timestamps_without_both_sides": int((per_ts != len(sd.SIDES)).sum()),
           "label_target_consistent": bool(y_ok)}
    out["passed"] = bool(not out["duplicate_samples"] and not out["off_grid_signals"]
                         and not out["entry_not_next_candle"] and not out["label_window_after_last_exit"]
                         and not out["timestamps_without_both_sides"] and y_ok)
    return out


def duplicate_trade_check(trade_lists: dict[str, list[dict]]) -> dict[str, Any]:
    bad = []
    for name, trades in trade_lists.items():
        ts = [t["signal_time"] for t in trades]
        if len(ts) != len(set(ts)):
            bad.append(f"{name}|duplicate signal")
        ex = sorted((t for t in trades if t["outcome"] in sd.EXECUTED), key=lambda t: t["entry_time"])
        if any(b["entry_time"] < a["exit_time"] for a, b in zip(ex, ex[1:])):
            bad.append(f"{name}|overlapping trades")
        if any(t["entry_time"] != t["signal_time"] + sd.STEP for t in trades):
            bad.append(f"{name}|entry not next candle")
    return {"lists": len(trade_lists), "violations": bad, "passed": not bad}


# ------------------------------------------------------------------ the study
def _view(m: dict | None) -> dict[str, Any]:
    if not m:
        return {"trades": None, "resolved": None, "accuracy": None, "profit_factor": None, "expectancy_r": None,
                "net_pnl": None, "max_drawdown": None, "max_losing_streak": None, "coverage": None}
    return {"trades": m["trades"], "resolved": m["resolved"], "accuracy": m["accuracy"],
            "profit_factor": m["profit_factor"], "expectancy_r": m["expectancy_r"], "net_pnl": m["net_pnl"],
            "max_drawdown": m["max_drawdown"], "max_losing_streak": m["max_losing_streak"],
            "coverage": m["coverage"]}


def build_study(*, market: sd.MarketData, availability: dict, data_checks: dict, fresh_quality: dict,
                fresh_loader: Callable[[], sd.MarketData | None], benchmarks: Callable[[], dict],
                freeze_dir: Path, mutation_days: int = 6, check_sample: int = 20,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    stages: list[str] = []
    flags = {"volume": bool(availability["use_option_volume"]), "oi": bool(availability["use_option_oi"])}
    cols = sd.feature_columns(flags["volume"], flags["oi"])
    grid = market.grid()

    say("building the causal feature / label dataset (Apr..Sep; October not loaded) ...")
    stages.append("dataset")
    pre_days = sd.days_in(market.sessions, sd.SPLITS["TRAIN"][0], sd.PRE_FREEZE_END)
    built = sd.build_samples(market, pre_days, volume=flags["volume"], oi=flags["oi"])
    data = built.data
    pre_freeze_max_read = market.reads["max_ts"]
    ds = dataset_checks(data)
    data_checks = {**data_checks, "dataset": ds, "passed": bool(data_checks.get("passed")) and ds["passed"]}
    if data.empty:
        return not_run(availability, data_checks, fresh_quality,
                       f"no eligible samples (DATA_UNAVAILABLE); funnel {dict(built.funnel)}")

    say("inner model / threshold selection (fit Apr-May, choose on June; TRAIN only) ...")
    stages.append("inner_selection")
    inner = inner_selection(data, cols, grid)
    fams, thr = inner["families"], inner["threshold"]

    say("purged walk-forward folds inside TRAIN ...")
    stages.append("walk_forward")
    wf = walk_forward(data, cols, grid, fams, thr)
    stable_folds = sum(fold_stable(f) for f in wf)

    say("freezing the S10 candidate (final fit on TRAIN Apr-Jun) ...")
    stages.append("freeze")
    train_days, val_days = _days(data, *sd.SPLITS["TRAIN"]), _days(data, *sd.SPLITS["VALIDATION"])
    oos_days, sep_days = _days(data, *sd.SPLITS["OOS"]), _days(data, *sd.SPLITS["LEGACY_HOLDOUT"])
    final = sm.fit_model(data, cols, train_days, fams, grid, test_days=val_days)
    frozen_fp = sm.fingerprint(final, thr)
    models = {s: final for s in sd.SIDES}
    meta = {"strategy": STRATEGY, "research_only": True, "families": fams, "family_reason": inner["family_reason"],
            "threshold": thr, "threshold_reason": inner["threshold_reason"], "margin": sm.MARGIN,
            "horizon_candles": sd.HORIZON, "stop_atr": sd.STOP_ATR, "target_r": sd.TARGET_R,
            "calibration": {s: final.sides[s].calibration for s in sd.SIDES},
            "status": {s: final.sides[s].status for s in sd.SIDES}, "fit_days": [final.fit_days[0], final.fit_days[-1]]
            if final.fit_days else [], "calibration_days": [final.calib_days[0], final.calib_days[-1]]
            if final.calib_days else [], "purge": final.purge, "fingerprint": frozen_fp,
            "sklearn_version": sklearn.__version__, "frozen_at": datetime.now(IST).isoformat(timespec="seconds"),
            "note": "research artifact only - never loaded by production execution"}
    artifacts = save_artifacts(final, thr, meta, freeze_dir)

    results: dict[str, dict] = {}
    stages.append("train_in_sample")
    results["TRAIN"] = evaluate(_rows(data, train_days), models, thr)

    say("VALIDATION July (frozen) ...")
    stages.append("validation_july")
    results["VALIDATION"] = evaluate(_rows(data, val_days), models, thr)
    val_label = gate(results["VALIDATION"]["trading"]["COMBINED"], MIN_VAL_RESOLVED)
    oos_ran = val_label == PASS
    oos_label = sep_label = fresh_label = NOT_RUN
    reasons = {"OOS": None, "LEGACY_HOLDOUT": None, "FRESH_HOLDOUT": None}
    if oos_ran:
        say("OOS August (frozen) ...")
        stages.append("oos_august")
        results["OOS"] = evaluate(_rows(data, oos_days), models, thr)
        oos_label = gate(results["OOS"]["trading"]["COMBINED"], MIN_OOS_RESOLVED)
        say(f"September {sd.LEGACY_LABEL} (report only) ...")
        stages.append("legacy_september")
        results["LEGACY_HOLDOUT"] = evaluate(_rows(data, sep_days), models, thr)
        sep_label = sd.LEGACY_LABEL
    else:
        reasons["OOS"] = reasons["LEGACY_HOLDOUT"] = f"July validation {val_label} - candidate not carried forward"

    fresh_info: dict[str, Any] = {"loaded": False, "period": None, "rows": 0}
    if oos_label == PASS and fresh_quality.get("usable"):
        say(f"{sd.FRESH_LABEL} October: loading for the first time, frozen model run once ...")
        stages.append("fresh_october")
        fm = fresh_loader()
        if fm is not None:
            lo = sd.SPLITS["FRESH_HOLDOUT"][0]
            fdays = [d for d in fm.sessions if d >= lo]
            fb = sd.build_samples(fm, fdays, volume=flags["volume"], oi=flags["oi"])
            results["FRESH_HOLDOUT"] = evaluate(fb.data, models, thr)
            fresh_label = gate(results["FRESH_HOLDOUT"]["trading"]["COMBINED"], MIN_FRESH_RESOLVED)
            fresh_info = {"loaded": True, "period": [fdays[0].isoformat(), fdays[-1].isoformat()] if fdays else None,
                          "rows": int(len(fb.data)), "funnel": dict(fb.funnel), "_market": fm, "_data": fb.data}
        else:
            reasons["FRESH_HOLDOUT"] = "October data could not be loaded"
    elif oos_label != PASS:
        reasons["FRESH_HOLDOUT"] = ("candidate did not pass OOS - October kept untouched for a future frozen "
                                    "candidate")
    else:
        reasons["FRESH_HOLDOUT"] = f"October data not usable: {fresh_quality.get('reason')}"
    fp_after = sm.fingerprint(final, thr)

    say("robustness (pre-declared variants, report only) ...")
    eval_frames = {"VALIDATION": _rows(data, val_days), **({"OOS": _rows(data, oos_days)} if oos_ran else {})}
    robustness = []
    for fam in sm.families():
        same = all(fams[s] == fam for s in sd.SIDES)
        m = final if same else sm.fit_model(data, cols, train_days, {s: fam for s in sd.SIDES}, grid,
                                            test_days=val_days)
        for split, frame in eval_frames.items():
            robustness.append({"variant": f"MODEL_{fam}_BOTH_SIDES", "kind": "model", "split": split,
                               "primary": same, **evaluate(frame, {s: m for s in sd.SIDES}, thr)["trading"]["COMBINED"]})
    for t in sm.THRESHOLDS:
        for split, frame in eval_frames.items():
            robustness.append({"variant": f"THRESHOLD_{t:.2f}", "kind": "threshold", "split": split,
                               "primary": t == thr, **evaluate(frame, models, t)["trading"]["COMBINED"]})
    for split, frame in eval_frames.items():
        robustness.append({"variant": "HORIZON_12_EXIT (frozen H6 model, 12-candle exit)", "kind": "horizon",
                           "split": split, "primary": False,
                           **evaluate(frame, models, thr, horizon=sd.ROBUST_HORIZON)["trading"]["COMBINED"]})
    for r in robustness:
        r["used_for_selection"] = False

    say("feature importance + stability (frozen model; never fed back) ...")
    importance, stability = [], {}
    train_frame = _rows(data, train_days)
    for side in sd.SIDES:
        if fams[side] == sm.LR:
            prim, method = sm.coefficient_importance(final, side), "ABS_STANDARDIZED_COEFFICIENT"
        else:
            prim, method = sm.permutation_importance(final, side, eval_frames["VALIDATION"]), \
                "PERMUTATION_BRIER_VALIDATION"
        for rank, c in enumerate(sm.top_k(prim, len(prim)), 1):
            importance.append({"side": side, "method": method, "rank": rank, "feature": c, "value": prim[c],
                               "wording": "predictive contribution in this model (not causation)"})
        perms = {"TRAIN": sm.permutation_importance(final, side, train_frame),
                 **{k: sm.permutation_importance(final, side, f) for k, f in eval_frames.items()}}
        tops = {k: sm.top_k(v) for k, v in perms.items() if v}
        jac = {f"TRAIN_vs_{k}": sm.jaccard(tops["TRAIN"], tops[k]) for k in tops if k != "TRAIN" and "TRAIN" in tops}
        stability[side] = {"top10": tops, "jaccard": jac,
                           "model_instability": any(j is not None and j < MIN_JACCARD for j in jac.values())}
    eces = {k: results[k]["model"]["COMBINED"]["ece"] for k in ("VALIDATION", "OOS") if k in results}
    calibration_stable = all(v is not None and v <= MAX_STABLE_ECE for v in eces.values()) and bool(eces)
    feature_unstable = any(v["model_instability"] for v in stability.values())

    say("benchmarks: random side, S2, S9 futures-volume ...")
    random_base = {k: evaluate(f, models, thr, side_picker=random_picker())["trading"] for k, f in eval_frames.items()}
    bench = benchmarks() or {}

    say("anti-lookahead and leakage checks ...")
    pre_iso = [d.isoformat() for d in pre_days]
    mdays = s4.sample_days(pre_iso, mutation_days) if mutation_days else []
    src_all = {"spot": True, "futures": True, "options": True}
    src_opt = {"spot": False, "futures": False, "options": True}
    src_vol = {"spot": True, "futures": True, "options": True}
    lookahead = {
        "future_candle_scrambling": mutation_check(market, data, cols, mdays, flags, final, seed=11, sources=src_all,
                                                   price=True, volume=True, oi=True),
        "feature_truncation": truncation_check(market, data, cols, flags, check_sample),
        "future_option_candle_mutation": mutation_check(market, data, cols, mdays, flags, final, seed=21,
                                                        sources=src_opt, drop=True),
        "future_option_price_mutation": mutation_check(market, data, cols, mdays, flags, final, seed=31,
                                                       sources=src_opt, price=True),
        "future_volume_mutation": mutation_check(market, data, cols, mdays, flags, final, seed=41, sources=src_vol,
                                                 volume=True),
        "future_oi_mutation": mutation_check(market, data, cols, mdays, flags, final, seed=51, sources=src_vol,
                                             oi=True),
    }
    fits = [(f"INNER_{k}", m, inner["inner_validation_days"]) for k, m in inner["_fits"].items()]
    fits += [(f"WF_{f['fold']}", f["_model"], f["_test_days"]) for f in wf]
    fits.append(("FINAL", final, val_days))
    leakage = leakage_checks(data, final, inner, fits, grid)
    fresh_loaded_after_freeze = "fresh_october" not in stages or stages.index("fresh_october") > \
        stages.index("freeze")
    leakage["holdout_contamination"] = {
        "market_roots_before_freeze": market.roots, "pre_freeze_max_read": pre_freeze_max_read,
        "fresh_loaded_after_freeze": fresh_loaded_after_freeze, "fingerprint_unchanged": fp_after == frozen_fp,
        "passed": bool(fresh_loaded_after_freeze and fp_after == frozen_fp
                       and (pre_freeze_max_read is None or pre_freeze_max_read.date() <= sd.PRE_FREEZE_END)
                       and not any(d > sd.SPLITS["TRAIN"][1].isoformat() for d in final.fit_days + final.calib_days)),
    }
    again = sm.fit_model(data, cols, train_days, fams, grid, test_days=val_days)
    rerun = evaluate(eval_frames["VALIDATION"], {s: again for s in sd.SIDES}, thr)
    w0, w1 = results["VALIDATION"]["wide"], rerun["wide"]
    same_p = all(np.array_equal(w0[f"p_{s}"].to_numpy(), w1[f"p_{s}"].to_numpy(), equal_nan=True) for s in sd.SIDES)
    key = lambda ts: [json.dumps(t, sort_keys=True, default=str) for t in ts]  # noqa: E731
    lookahead["deterministic_replay"] = {"probabilities_identical": same_p,
                                         "trades_identical": key(results["VALIDATION"]["trades"]) == key(rerun["trades"]),
                                         "fingerprint_identical": sm.fingerprint(again, thr) == frozen_fp,
                                         "passed": bool(same_p and key(results["VALIDATION"]["trades"])
                                                        == key(rerun["trades"]))}
    lookahead["duplicate_trades"] = duplicate_trade_check({k: v["trades"] for k, v in results.items()})
    lookahead["contract_selection_causality"] = contract_causality_check(market, data, check_sample)
    lookahead_ok = all(v["passed"] for v in lookahead.values())
    leakage_ok = all(v["passed"] for v in leakage.values())
    data_ok = bool(data_checks.get("passed")) and bool(availability["required_ok"])

    trading = {k: v["trading"] for k, v in results.items()}
    oos_m = trading.get("OOS", {}).get("COMBINED")
    if oos_m is None or oos_m["resolved"] < MIN_OOS_RESOLVED:
        target = INSUFFICIENT
    else:
        target = "ACHIEVED" if (oos_m["accuracy"] or 0) >= TARGET else "NOT ACHIEVED"
    train_stable = stable_folds >= MIN_STABLE_FOLDS
    models_ok = all(usable(final, s) for s in sd.SIDES)
    if not (lookahead_ok and leakage_ok and data_ok):
        decision, why = REJECTED, "lookahead / leakage / data-integrity check failed"
    elif not models_ok:
        decision, why = INSUFF_DECISION, "not enough TRAIN samples to fit and calibrate both side models"
    elif val_label == INSUFFICIENT:
        decision, why = INSUFF_DECISION, "too few resolved July trades"
    elif val_label == FAIL:
        decision, why = REJECTED, "July validation failed (PF <= 1 or expectancy <= 0); not rescued by tuning"
    elif oos_label == INSUFFICIENT:
        decision, why = INSUFF_DECISION, "too few resolved August OOS trades"
    elif oos_label == FAIL:
        decision, why = REJECTED, "negative August OOS expectancy or PF <= 1"
    elif fresh_label == FAIL:
        decision, why = REJECTED, "fresh October holdout failed"
    elif feature_unstable or not calibration_stable:
        decision, why = UNSTABLE, "feature importance or probability calibration unstable across splits"
    elif not train_stable:
        decision, why = REJECTED, f"TRAIN walk-forward unstable ({stable_folds}/{len(wf)} folds positive)"
    elif oos_m["max_drawdown_r"] > MAX_DD_R or (oos_m["coverage"] or 0) < MIN_COVERAGE:
        decision, why = REJECTED, "OOS drawdown too large or coverage too low"
    else:
        decision, why = PROMISING, "passed TRAIN stability, July, August OOS, calibration and leakage checks"
    forward = "SAFE" if decision == PROMISING else "NOT SAFE"
    fresh_status = fresh_label if fresh_label in (PASS, FAIL) else (INSUFFICIENT if fresh_label == INSUFFICIENT
                                                                     else NOT_RUN)

    comparison = []
    for split in ("VALIDATION", "OOS"):
        b = bench.get(split, {})
        s10 = trading.get(split, {}).get("COMBINED")
        comparison += [
            {"split": SPLIT_NAMES[split], "system": "S2", "data": "S2 frozen baseline (unchanged)", **b.get("S2", _view(None))},
            {"split": SPLIT_NAMES[split], "system": "S9_FUTURES_VOLUME",
             "data": "S9 rules on real NIFTY futures volume (REJECTED in S9.1)", **b.get("S9_FUTURES_VOLUME", _view(None))},
            {"split": SPLIT_NAMES[split], "system": STRATEGY, "data": "calibrated ML probability (gross, no costs)",
             "status": "RUN" if s10 else NOT_RUN, **_view(s10)},
            {"split": SPLIT_NAMES[split], "system": "RANDOM_SIDE_BASELINE",
             "data": "S10 trigger times, side chosen at random (fixed seed)",
             **_view(random_base.get(split, {}).get("COMBINED"))},
        ]
    return {
        "heading": HEADING, "strategy": STRATEGY, "banner": BANNER, "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"), "stages": stages,
        "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in sd.SPLITS.items()},
        "split_labels": SPLIT_LABELS, "inner_split": {"train": [x.isoformat() for x in sd.INNER_TRAIN],
                                                      "validation": [x.isoformat() for x in sd.INNER_VALIDATION]},
        "flags": flags, "feature_columns": cols, "availability": availability, "data_checks": data_checks,
        "fresh_quality": fresh_quality, "fresh": {k: v for k, v in fresh_info.items() if not k.startswith("_")},
        "dataset": {"rows": int(len(data)), "timestamps": int(data["signal_time"].nunique()) if len(data) else 0,
                    "funnel": dict(built.funnel), "labels": sd.label_counts(data),
                    "option_contracts": sorted(built.option_contracts),
                    "futures_contracts": sorted(built.futures_contracts),
                    "hash": sd.dataset_hash(data, cols), "missing_sessions": [d.isoformat() for d in
                                                                              built.missing_sessions]},
        "inner": {k: v for k, v in inner.items() if not k.startswith("_")},
        "walk_forward": [{k: v for k, v in f.items() if not k.startswith("_")} for f in wf],
        "stable_folds": stable_folds, "train_stable": train_stable, "families": fams, "threshold": thr,
        "frozen": {"fingerprint": frozen_fp, "artifacts": artifacts, "metadata": meta},
        "results": {k: {kk: vv for kk, vv in v.items() if kk not in ("wide",)} for k, v in results.items()},
        "labels": {"VALIDATION": val_label, "OOS": oos_label, "LEGACY_HOLDOUT": sep_label,
                   "FRESH_HOLDOUT": fresh_label}, "not_run_reasons": reasons,
        "robustness": robustness, "importance": importance, "stability": stability, "eces": eces,
        "calibration_stable": calibration_stable, "feature_unstable": feature_unstable,
        "random_baseline": random_base, "benchmarks": bench, "comparison": comparison,
        "lookahead": lookahead, "lookahead_ok": lookahead_ok, "leakage": leakage, "leakage_ok": leakage_ok,
        "data_ok": data_ok, "target_70": target, "decision": decision, "decision_reason": why,
        "forward_paper": forward, "paper_status": "CANDIDATE_FOR_FORWARD_PAPER" if forward == "SAFE" else None,
        "fresh_status": fresh_status, "cost_model": COST_MODEL, "production_unchanged": None, "live_orders": 0,
        "trading_mode": "PAPER", "live_trading_enabled": False,
        "_data": data, "_market": market, "_fresh": fresh_info,
    }


def not_run(availability: dict, data_checks: dict, fresh_quality: dict, reason: str) -> dict[str, Any]:
    return {"heading": HEADING, "strategy": STRATEGY, "banner": BANNER, "disclaimer": DISCLAIMER, "ran": False,
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"), "reason": reason,
            "availability": availability, "data_checks": data_checks, "fresh_quality": fresh_quality,
            "decision": INSUFF_DECISION, "decision_reason": reason, "forward_paper": "NOT SAFE",
            "lookahead_ok": None, "leakage_ok": None, "data_ok": False, "target_70": INSUFFICIENT,
            "fresh_status": NOT_RUN, "cost_model": COST_MODEL, "production_unchanged": None, "live_orders": 0,
            "trading_mode": "PAPER", "live_trading_enabled": False}


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "s10_report.md", "data_availability.json", "feature_dataset_summary.csv",
                "ce_model_metrics.csv", "pe_model_metrics.csv", "combined_model_metrics.csv", "train_metrics.csv",
                "validation_metrics.csv", "oos_metrics.csv", "legacy_holdout_september.csv",
                "fresh_holdout_october.csv", "trading_results.csv", "monthly_results.csv", "regime_results.csv",
                "probability_calibration.csv", "feature_importance.csv", "robustness.csv", "s2_vs_s9_vs_s10.csv",
                "trade_log.csv", "data_quality.json", "dataset_manifest.json")
TRADE_LOG_COLS = ("split", "label", "signal_time", "session_date", "direction", "side", "p_win", "p_other", "ev_r",
                  "threshold", "contract", "instrument_key", "strike", "expiry", "entry_time", "entry", "stop_loss",
                  "target", "risk", "exit_time", "exit_price", "exit_reason", "outcome", "r_multiple", "pnl",
                  "holding_minutes", "horizon")
MONTHS = (("2026-04", "APRIL"), ("2026-05", "MAY"), ("2026-06", "JUNE"), ("2026-07", "JULY"), ("2026-08", "AUGUST"),
          ("2026-09", "SEPTEMBER"), ("2026-10", "OCTOBER"))


def _model_row(base: dict, m: dict) -> dict:
    return {**base, **{k: m.get(k) for k in METRIC_KEYS}}


def _trade_row(base: dict, m: dict) -> dict:
    return {**base, **{k: m.get(k) for k in TRADE_KEYS}}


def _split_rows(a: dict, split: str) -> list[dict]:
    r = a["results"].get(split)
    lab = {"split": SPLIT_NAMES[split], "label": SPLIT_LABELS[split]}
    if r is None:
        return [{**lab, "status": NOT_RUN, "reason": a["not_run_reasons"].get(split) or "not evaluated"}]
    out = [_model_row({**lab, "kind": "model", "scope": s}, r["model"][s]) for s in SCOPES]
    out += [_trade_row({**lab, "kind": "trading", "scope": s}, r["trading"][s]) for s in SCOPES]
    out.append({**lab, "kind": "decisions", **r["decisions"]})
    return out


def model_metric_rows(a: dict, scope: str) -> list[dict]:
    out = []
    for fam_row in a["inner"]["family_rows"].get(scope, []) if scope in sd.SIDES else []:
        out.append({"stage": "INNER_VALIDATION_JUNE", "family": fam_row["family"], "status": fam_row["status"],
                    **{k: fam_row.get(k) for k in METRIC_KEYS}})
    for f in a["walk_forward"]:
        out.append(_model_row({"stage": f"WALK_FORWARD_{f['fold']}", "train": f["train"], "test": f["test"]},
                              f["model"][scope]))
    for split, r in a["results"].items():
        out.append(_model_row({"stage": SPLIT_NAMES[split], "label": SPLIT_LABELS[split],
                               "family": a["families"].get(scope, "PER_SIDE"), "threshold": a["threshold"]},
                              r["model"][scope]))
    return out


def dataset_summary_rows(a: dict) -> list[dict]:
    data, out = a["_data"], []
    frames = {k: sd.rows_between(data, *sd.SPLITS[k]) for k in ("TRAIN", "VALIDATION", "OOS", "LEGACY_HOLDOUT")}
    if a["_fresh"].get("loaded"):
        frames["FRESH_HOLDOUT"] = a["_fresh"]["_data"]
    for split, f in frames.items():
        for side in sd.SIDES:
            g = f[f["side"] == side]
            n = Counter(g["outcome"])
            res = n[sd.WIN] + n[sd.LOSS]
            out.append({"section": "labels", "split": SPLIT_NAMES[split], "side": side, "samples": len(g),
                        "timestamps": int(g["signal_time"].nunique()), **{o: n[o] for o in (*sd.EXECUTED, sd.NO_ENTRY,
                                                                                              sd.INVALID_LEVELS)},
                        "win_pct_of_resolved": n[sd.WIN] / res if res else None,
                        "loss_pct_of_resolved": n[sd.LOSS] / res if res else None})
    tr = frames["TRAIN"]
    for c in a["feature_columns"]:
        out.append({"section": "feature_missing_share_train", "feature": c,
                    "missing_share": float(tr[c].isna().mean()) if len(tr) else None})
    for k, v in sorted(a["dataset"]["funnel"].items()):
        out.append({"section": "timestamp_funnel", "reason": k, "count": v})
    return out


def trading_rows(a: dict) -> list[dict]:
    out = []
    for split, r in a["results"].items():
        out += [_trade_row({"system": STRATEGY, "split": SPLIT_NAMES[split], "label": SPLIT_LABELS[split],
                            "scope": s, "costs": COST_MODEL}, r["trading"][s]) for s in SCOPES]
    for split, t in a["random_baseline"].items():
        out += [_trade_row({"system": "RANDOM_SIDE_BASELINE", "split": SPLIT_NAMES[split], "scope": s}, t[s])
                for s in SCOPES]
    for split, reason in a["not_run_reasons"].items():
        if reason:
            out.append({"system": STRATEGY, "split": SPLIT_NAMES[split], "status": NOT_RUN, "reason": reason})
    return out


def all_trades(a: dict) -> list[dict]:
    out = []
    for split, r in a["results"].items():
        out += [{**t, "split": SPLIT_NAMES[split], "label": SPLIT_LABELS[split]} for t in r["trades"]]
    return out


def monthly_rows(a: dict) -> list[dict]:
    trades, out = all_trades(a), []
    for m, name in MONTHS:
        sub = [t for t in trades if str(t["session_date"])[:7] == m]
        if not sub:
            continue
        split = sub[0]["split"]
        for s in SCOPES:
            x = sm.trade_metrics([t for t in sub if s == "COMBINED" or t["side"] == s], 0)
            out.append({"month": name, "split": split, "scope": s, **{k: x[k] for k in TRADE_KEYS if k not in
                                                                       ("opportunities", "coverage")}})
    return out


def regime_rows(a: dict) -> list[dict]:
    bars = dg.feature_bars(a["_market"].spot)
    fresh_bars = dg.feature_bars(a["_fresh"]["_market"].spot) if a["_fresh"].get("loaded") else None
    lab: dict[tuple, str] = {}
    for t in all_trades(a):
        b = fresh_bars if (fresh_bars is not None and t["split"] == SPLIT_NAMES["FRESH_HOLDOUT"]) else bars
        lab[(t["split"], t["signal_time"])] = s9.regime_of(b, {"signal_time": t["signal_time"].isoformat()})
    out = []
    for split in {t["split"] for t in all_trades(a)}:
        sub = [t for t in all_trades(a) if t["split"] == split]
        for reg in sorted({lab[(split, t["signal_time"])] for t in sub}):
            g = [t for t in sub if lab[(split, t["signal_time"])] == reg]
            for s in SCOPES:
                x = sm.trade_metrics([t for t in g if s == "COMBINED" or t["side"] == s], 0)
                out.append({"split": split, "regime": reg, "scope": s, "descriptive_only": True,
                            **{k: x[k] for k in TRADE_KEYS if k not in ("opportunities", "coverage")}})
    return out


def calibration_rows(a: dict) -> list[dict]:
    out = []
    for split, r in a["results"].items():
        for s in SCOPES:
            for b in r["model"][s]["calibration"]:
                out.append({"split": SPLIT_NAMES[split], "scope": s, **b})
    return out


def manifest(a: dict) -> dict[str, Any]:
    d = a["dataset"]
    data = a["_data"]
    return {"strategy": STRATEGY, "generated_at": a["generated_at"],
            "date_range": {"from": data["session_date"].min() if len(data) else None,
                           "to": data["session_date"].max() if len(data) else None},
            "fresh_holdout": a["fresh"], "source": {"roots": a["_market"].roots,
                                                    "spot": "NSE_INDEX|Nifty 50 (6-month Upstox store)",
                                                    "futures": "real NIFTY FUT 5m (per contract, causal front month)",
                                                    "options": "real NIFTY CE/PE 5m (point-in-time ATM)"},
            "futures_instrument_keys": d["futures_contracts"], "option_contracts_used": len(d["option_contracts"]),
            "option_instrument_keys": d["option_contracts"], "rows": d["rows"], "timestamps": d["timestamps"],
            "feature_count": len(a["feature_columns"]), "label_counts": d["labels"],
            "missing_rows": {k: v for k, v in d["funnel"].items() if k != "ELIGIBLE"},
            "missing_sessions": d["missing_sessions"], "data_hash_sha256": d["hash"],
            "model_fingerprint_sha256": a["frozen"]["fingerprint"]}


def _strip(a: dict) -> dict:
    return {k: v for k, v in a.items() if not k.startswith("_")}


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(s6._jsonable(obj), indent=2, default=str), encoding="utf-8")


def _csv(path: Path, rows: Sequence[dict]) -> None:
    clean = [{k: (v.isoformat() if isinstance(v, (pd.Timestamp, datetime, date)) else v) for k, v in r.items()}
             for r in rows]
    s6._csv(path, clean)


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    _json(p["data_availability.json"], a["availability"])
    dq = {"data_checks": a["data_checks"], "fresh_october": a["fresh_quality"], "data_ok": a["data_ok"],
          "lookahead": a.get("lookahead"), "leakage": a.get("leakage"), "cost_model": COST_MODEL}
    _json(p["data_quality.json"], dq)
    p["s10_report.md"].write_text(markdown(a), encoding="utf-8")
    if a.get("ran") is False:
        _json(p["summary.json"], _strip(a))
        return {k: v for k, v in p.items() if k in ("summary.json", "s10_report.md", "data_availability.json",
                                                    "data_quality.json")}
    summary = _strip(a)
    summary["results"] = {k: {kk: vv for kk, vv in v.items() if kk != "trades"} for k, v in a["results"].items()}
    _json(p["summary.json"], summary)
    _csv(p["feature_dataset_summary.csv"], dataset_summary_rows(a))
    for scope, name in (("CE", "ce_model_metrics.csv"), ("PE", "pe_model_metrics.csv"),
                        ("COMBINED", "combined_model_metrics.csv")):
        _csv(p[name], model_metric_rows(a, scope))
    train = _split_rows(a, "TRAIN") + [
        {"split": "TRAIN", "label": f"PURGED_WALK_FORWARD_{f['fold']}", "kind": kind, "scope": s,
         "train": f["train"], "test": f["test"],
         **({k: f["model"][s].get(k) for k in METRIC_KEYS} if kind == "model" else
            {k: f["trading"][s].get(k) for k in TRADE_KEYS})}
        for f in a["walk_forward"] for kind in ("model", "trading") for s in SCOPES]
    _csv(p["train_metrics.csv"], train)
    _csv(p["validation_metrics.csv"], _split_rows(a, "VALIDATION"))
    _csv(p["oos_metrics.csv"], _split_rows(a, "OOS"))
    _csv(p["legacy_holdout_september.csv"], _split_rows(a, "LEGACY_HOLDOUT"))
    _csv(p["fresh_holdout_october.csv"], _split_rows(a, "FRESH_HOLDOUT"))
    _csv(p["trading_results.csv"], trading_rows(a))
    _csv(p["monthly_results.csv"], monthly_rows(a))
    _csv(p["regime_results.csv"], regime_rows(a))
    _csv(p["probability_calibration.csv"], calibration_rows(a))
    _csv(p["feature_importance.csv"], a["importance"] + [
        {"side": s, "method": "STABILITY_TOP10", "split": k, "features": " ".join(v)}
        for s, st in a["stability"].items() for k, v in st["top10"].items()] + [
        {"side": s, "method": "STABILITY_JACCARD", "split": k, "value": v,
         "model_instability": st["model_instability"]} for s, st in a["stability"].items()
        for k, v in st["jaccard"].items()])
    _csv(p["robustness.csv"], [{**r, "note": "pre-declared variant - REPORT ONLY, NOT USED FOR SELECTION"}
                               for r in a["robustness"]])
    _csv(p["s2_vs_s9_vs_s10.csv"], a["comparison"])
    _csv(p["trade_log.csv"], [{c: t.get(c) for c in TRADE_LOG_COLS} for t in all_trades(a)])
    _json(p["dataset_manifest.json"], manifest(a))
    return p


# ------------------------------------------------------------------ text
def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _pf(v) -> str:
    return "N/A" if v is None else ("INF" if v == float("inf") else f"{v:.2f}")


def _f(v, nd: int = 4) -> str:
    return "N/A" if v is None else f"{v:.{nd}f}"


def tline(m: dict) -> str:
    exp = "N/A" if m["expectancy_r"] is None else f"{m['expectancy_r']:+.3f}"
    return (f"Trades {m['trades']} | Resolved {m['resolved']} | Win Rate {_pct(m['accuracy'])} "
            f"({m['wins']}/{m['resolved']}) | PF {_pf(m['profit_factor'])} | P&L Rs {m['net_pnl']:,.2f} | "
            f"Max DD Rs {m['max_drawdown']:,.2f} | Max Losing Streak {m['max_losing_streak']} | "
            f"Expectancy R {exp} | Coverage {_pct(m['coverage'])}")


def _section(a: dict, split: str) -> list[str]:
    r = a.get("results", {}).get(split)
    if r is None:
        why = a.get("not_run_reasons", {}).get(split) or a.get("reason") or "not evaluated"
        return [f"{s}: NOT_RUN ({why})" for s in SCOPES]
    return [f"{s}: {tline(r['trading'][s])}" for s in SCOPES]


def _model_quality(a: dict) -> tuple[str, dict | None]:
    for k in ("OOS", "VALIDATION"):
        if k in a.get("results", {}):
            return SPLIT_NAMES[k], a["results"][k]["model"]["COMBINED"]
    return "N/A", None


def _cmp(a: dict, key: str, fmt: Callable) -> str:
    rows = {r["system"]: r for r in a.get("comparison", []) if r["split"] == SPLIT_NAMES["OOS"]}
    parts = []
    for name, label in (("S2", "S2"), ("S9_FUTURES_VOLUME", "S9"), (STRATEGY, "S10")):
        v = rows.get(name, {}).get(key)
        tail = " (NOT_RUN)" if name == STRATEGY and rows.get(name, {}).get("status") == NOT_RUN else ""
        parts.append(f"{label} {fmt(v) if v is not None else 'N/A'}{tail}")
    return " | ".join(parts)


def terminal_text(a: dict) -> str:
    ran = a.get("ran", True)
    data = a.get("_data")
    fresh = a.get("fresh", {})
    out = [HEADING, f"({DISCLAIMER})", "", "DATA:",
           f"historical rows: {len(data) if data is not None else 0}",
           f"option rows: {int(a['availability']['options']['train_atm_candles']) if a.get('availability') else 0} "
           f"TRAIN ATM option candles; {len(a['dataset']['option_contracts']) if ran else 0} option contracts used",
           f"fresh October rows: {fresh.get('rows', 0)}" + (f" ({fresh['period'][0]} -> {fresh['period'][1]})"
                                                            if fresh.get("period") else
                                                            f" ({a.get('fresh_quality', {}).get('reason') or 'not loaded'})"),
           "", "MODEL:", ""]
    for side in sd.SIDES:
        if ran:
            v = a["results"]["VALIDATION"]["model"][side]
            val = (f"July Brier {_f(v['brier'])} | Log Loss {_f(v['log_loss'])} | ROC AUC {_f(v['roc_auc'])} | "
                   f"ECE {_f(v['ece'])} | {a['labels']['VALIDATION']} (combined trading gate)")
            out += [f"{side} MODEL:", f"model: {a['families'][side]} ({a['frozen']['metadata']['calibration'][side]})",
                    f"threshold: {a['threshold']:.2f} (margin {sm.MARGIN}, EV > 0)", f"validation: {val}", ""]
        else:
            out += [f"{side} MODEL:", "model: NOT_RUN", "threshold: NOT_RUN", "validation: NOT_RUN", ""]
    for split, title in (("TRAIN", "TRAIN:"), ("VALIDATION", "VALIDATION JULY:"), ("OOS", "OOS AUGUST:")):
        out += [title, *_section(a, split), ""]
    out += ["SEPTEMBER:", sd.LEGACY_LABEL, "", *_section(a, "LEGACY_HOLDOUT"), "",
            "FRESH OCTOBER HOLDOUT:", *_section(a, "FRESH_HOLDOUT"), ""]
    qname, q = _model_quality(a)
    out += [f"MODEL QUALITY ({qname} combined):" if q else "MODEL QUALITY:",
            f"Brier: {_f(q['brier']) if q else 'N/A'}", f"Log Loss: {_f(q['log_loss']) if q else 'N/A'}",
            f"ROC AUC: {_f(q['roc_auc']) if q else 'N/A'}", f"PR AUC: {_f(q['pr_auc']) if q else 'N/A'}",
            f"Calibration: ECE {_f(q['ece']) if q else 'N/A'} "
            f"({'STABLE' if a.get('calibration_stable') else 'NOT STABLE / NOT PROVEN'})", ""]
    out += ["S2 vs S9 vs S10:", f"OOS accuracy: {_cmp(a, 'accuracy', _pct)}",
            f"OOS PF: {_cmp(a, 'profit_factor', _pf)}",
            f"OOS expectancy: {_cmp(a, 'expectancy_r', lambda v: f'{v:+.3f}R')}",
            f"OOS P&L: {_cmp(a, 'net_pnl', lambda v: f'Rs {v:,.2f}')}",
            f"OOS DD: {_cmp(a, 'max_drawdown', lambda v: f'Rs {v:,.2f}')}", ""]
    pf = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731
    out += ["FINAL:", "", "70% TARGET:", a["target_70"], "", "LOOKAHEAD:",
            pf(a["lookahead_ok"]) if a["lookahead_ok"] is not None else "NOT_RUN", "", "DATA INTEGRITY:",
            pf(a["data_ok"]), "", "MODEL LEAKAGE:", pf(a["leakage_ok"]) if a["leakage_ok"] is not None else "NOT_RUN",
            "", "FRESH HOLDOUT:", a["fresh_status"], "", "PRODUCTION CHANGED:",
            "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:", "0", "", "FORWARD PAPER:",
            a["forward_paper"], "", "FINAL DECISION:", a["decision"], f"({a['decision_reason']})", "",
            f"Costs: {COST_MODEL} - all P&L is GROSS (no validated brokerage / STT / slippage model in the project)."]
    return "\n".join(out)


def markdown(a: dict) -> str:
    md = [f"# {STRATEGY}", "", f"**{BANNER}**", "", f"> {DISCLAIMER}", "",
          f"Generated {a['generated_at']}. TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, live orders: 0. "
          f"Production changed: **{'NO' if a['production_unchanged'] is not False else 'YES'}**. "
          f"Costs: **{COST_MODEL}** (gross results only).", "",
          f"## Final decision: {a['decision']}", "", f"{a['decision_reason']}.", ""]
    if a.get("ran") is False:
        md += [f"S10 research NOT RUN: {a['reason']}."]
        return "\n".join(md) + "\n"
    md += [f"Forward paper: **{a['forward_paper']}** | 70% target: **{a['target_70']}** | lookahead "
           f"**{'PASS' if a['lookahead_ok'] else 'FAIL'}** | model leakage **{'PASS' if a['leakage_ok'] else 'FAIL'}** "
           f"| data integrity **{'PASS' if a['data_ok'] else 'FAIL'}** | fresh holdout **{a['fresh_status']}**", "",
           "## Splits", "", "| Split | Dates | Label |", "|---|---|---|"]
    md += [f"| {SPLIT_NAMES[k]} | {v[0]} -> {v[1]} | {SPLIT_LABELS[k]} |" for k, v in a["splits"].items()]
    md += ["", "September was examined in earlier project research: it is reported as "
           f"**{sd.LEGACY_LABEL}**, never as a clean holdout. October is the **{sd.FRESH_LABEL}**, loaded only after "
           "the model, threshold and features were frozen, and run at most once.", "",
           "## Model", "", f"- Families (inner validation June, TRAIN only): {a['families']}",
           f"- Reason: {a['inner']['family_reason']}",
           f"- Threshold: {a['threshold']:.2f} ({a['inner']['threshold_reason']}); margin {sm.MARGIN}; EV > 0",
           f"- Features: {len(a['feature_columns'])} (option volume {a['availability']['features']['option_volume']},"
           f" option OI change {a['availability']['features']['option_oi_change']}; IV / spread not available)",
           f"- Frozen fingerprint: `{a['frozen']['fingerprint'][:16]}...`",
           f"- TRAIN walk-forward stable folds: {a['stable_folds']}/{len(a['walk_forward'])}", "",
           "## Trading results (gross)", "", "| Split | Scope | Result |", "|---|---|---|"]
    for split in SPLIT_NAMES:
        for line in _section(a, split):
            scope, text = line.split(": ", 1)
            md.append(f"| {SPLIT_NAMES[split]} | {scope} | {text.replace(' | ', '; ')} |")
    md += ["", "## S2 vs S9 vs S10 (OOS August)", ""]
    for key, name in (("accuracy", "accuracy"), ("profit_factor", "PF"), ("expectancy_r", "expectancy"),
                      ("net_pnl", "P&L"), ("max_drawdown", "DD")):
        md.append(f"- OOS {name}: {_cmp(a, key, lambda v: f'{v:.4g}')}")
    md += ["", "## Checks", ""]
    md += [f"- lookahead / {k}: {'PASS' if v['passed'] else 'FAIL'}" for k, v in a["lookahead"].items()]
    md += [f"- leakage / {k}: {'PASS' if v['passed'] else 'FAIL'}" for k, v in a["leakage"].items()]
    md += ["", "## Wording", "", "Feature importance is reported as *predictive contribution in this model*, never as "
           "causation. A high ROC AUC without profitable trading does not mean the strategy works. Nothing is "
           "promoted or activated automatically; production S2, broker, OrderManager, RiskManager and live flags "
           "are unchanged."]
    return "\n".join(md) + "\n"
