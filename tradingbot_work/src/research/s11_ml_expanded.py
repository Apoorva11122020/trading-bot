"""S11_ML_EXPANDED - expanded-history ML NIFTY options research (PAPER | RESEARCH ONLY | NO ORDERS).

Question: did S10 fail because three months of training data were too short / not diverse enough, or because the
causal feature space carries little signal? Same S10 architecture (separate CE / PE models, LR + HGB, frozen S10
feature family, median imputer + scaler fit on each training fold only, chronological Platt calibration inside each
training fold, purge + 6-bar embargo), trained on the longest clean contiguous history ending 2026-06-30.

Label L1 is FIXED: entry = next option candle open, R = 1.5 x option ATR14, STOP = entry - R, TARGET = entry + 1.5R,
6 option candles; WIN / LOSS / TIMEOUT / AMBIGUOUS (both in one candle, booked at -1R).

Chronology (never reversed):
  1. data + history coverage (months with genuine data; < 12 clean months -> INSUFFICIENT_HISTORICAL_DATA, stop)
  2. TRAIN dataset (actual start .. 2026-06-30) only; six fixed chronological inner folds inside TRAIN
  3. model family per side (Brier, log loss, AUC, calibration, expectancy, PF, drawdown) and ONE threshold from
     {0.55, 0.60, 0.65} - both chosen on inner folds only
  4. FREEZE: final fit on the whole TRAIN, artifacts + fingerprint
  5. July OUTER VALIDATION -> 6. August OOS only if July passed -> 7. September LEGACY_HOLDOUT_CONTAMINATED (report)
October 2026 is never loaded. Historical results are not live accuracy; no profit is guaranteed.
"""
from __future__ import annotations

import json
import math
import pickle
import warnings
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import sklearn

from src.historical_data import nifty_futures as nf
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s10_ml_probability as s10
from src.research import s11_history as hx
from src.research import s4_six_month_lab as s4
from src.research import s6_clean_strategy_lab as s6

HEADING = "S11 EXPANDED ML RESEARCH COMPLETE"
STRATEGY = "S11_ML_EXPANDED"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | historical backtest results (not live accuracy)"
IST = sd.IST
DISCLAIMER = ("ML model estimates the historical probability of a predefined option outcome (L1) from causal "
              "features. Historical performance != live performance. No guaranteed profit, no guaranteed accuracy.")
TARGET_R, HORIZON = 1.5, sd.HORIZON
LABEL = {"name": "L1", "entry": "next option 5-minute candle OPEN", "risk": f"R = {sd.STOP_ATR} x option ATR14",
         "stop": "entry - 1R", "target": f"entry + {TARGET_R}R", "horizon_candles": HORIZON,
         "ambiguous": "stop and target in one candle -> AMBIGUOUS (-1R)"}
FOLDS = (("FOLD_1", date(2025, 6, 30), date(2025, 7, 1), date(2025, 8, 31)),
         ("FOLD_2", date(2025, 8, 31), date(2025, 9, 1), date(2025, 10, 31)),
         ("FOLD_3", date(2025, 10, 31), date(2025, 11, 1), date(2025, 12, 31)),
         ("FOLD_4", date(2025, 12, 31), date(2026, 1, 1), date(2026, 2, 28)),
         ("FOLD_5", date(2026, 2, 28), date(2026, 3, 1), date(2026, 4, 30)),
         ("FOLD_6", date(2026, 4, 30), date(2026, 5, 1), date(2026, 6, 30)))
SHORT_TRAIN = (date(2026, 4, 1), date(2026, 6, 30))       # the S10 three-month training window

# Pre-declared rules (fixed before any S11 model was fitted).
MIN_FOLD_TRAIN_SESSIONS, MIN_FOLD_VAL_SESSIONS = 60, 20
MIN_STABILITY_FOLDS = 3
MAX_AUC_STD = 0.08
EXCELLENT_AUC, WEAK_OTHER_AUC = 0.55, 0.52
THRESHOLD_MIN_RESOLVED = 30
MIN_VAL_RESOLVED, MIN_OOS_RESOLVED = 20, 30
MAX_DD_R, MIN_COVERAGE = 10.0, 0.005
MAX_AUC_DEGRADATION = 0.10
TOP_FEATURES = 10
TARGET_70 = 0.70
PASS, FAIL, INSUFFICIENT, NOT_RUN, RUN = "PASS", "FAIL", "INSUFFICIENT_DATA", "NOT_RUN", "RUN"
STABLE, UNSTABLE_MODEL, FEW_FOLDS = "STABLE", "MODEL_UNSTABLE", "INSUFFICIENT_FOLDS"
PROMISING, REJECTED, INSUFF, UNSTABLE = "S11_PROMISING", "S11_REJECTED", "S11_INSUFFICIENT_DATA", "S11_MODEL_UNSTABLE"
SCOPES = ("CE", "PE", "COMBINED")
SPLIT_NAMES = {"TRAIN": "TRAIN (in-sample)", "VALIDATION": "JULY VALIDATION", "OOS": "AUGUST OOS",
               "LEGACY_HOLDOUT": "SEPTEMBER LEGACY"}


# ------------------------------------------------------------------ small helpers
def _iso_days(data: pd.DataFrame, lo: date | str, hi: date | str) -> list[str]:
    lo, hi = str(lo), str(hi)
    return sorted(d for d in set(data["session_date"]) if lo <= d <= hi) if len(data) else []


def _rows(data: pd.DataFrame, days: Sequence[str]) -> pd.DataFrame:
    return data[data["session_date"].isin(sorted(set(days)))]


def usable(m: sm.S10Model, side: str) -> bool:
    s = m.sides[side]
    return s.status == sm.FITTED and s.calibrator is not None


def _mean(v: Sequence[float | None]) -> float | None:
    x = [float(a) for a in v if a is not None and np.isfinite(a)]
    return float(np.mean(x)) if x else None


def _std(v: Sequence[float | None]) -> float | None:
    x = [float(a) for a in v if a is not None and np.isfinite(a)]
    return float(np.std(x)) if len(x) >= 2 else None


def _finite_pf(v):
    return None if v is None else (1e9 if v == math.inf else v)


VIEW_KEYS = ("trades", "resolved", "accuracy", "profit_factor", "expectancy_r", "net_pnl", "max_drawdown",
             "max_losing_streak", "coverage")


def view(m: dict | None) -> dict[str, Any]:
    return {k: (m or {}).get(k) for k in VIEW_KEYS}


def coverage_funnel(w: pd.DataFrame, dec: Counter, trades: Sequence[dict]) -> dict[str, int]:
    """Eligible timestamps -> scored -> probability gate -> margin gate -> EV gate -> one-active-trade -> trades."""
    n = len(w)
    scored = int((np.isfinite(w["p_CE"].to_numpy(dtype=float)) & np.isfinite(w["p_PE"].to_numpy(dtype=float))).sum()) \
        if n else 0
    p_rm, m_rm, e_rm = dec.get("BELOW_THRESHOLD", 0), dec.get("MARGIN_TOO_SMALL", 0), dec.get("EV_NOT_POSITIVE", 0)
    a_rm, h_rm = dec.get("ACTIVE_TRADE_OPEN", 0), dec.get("HORIZON_NOT_DEFINED", 0)
    return {"eligible_timestamps": n, "model_scored": scored, "removed_by_probability_gate": p_rm,
            "passed_probability_gate": scored - p_rm, "removed_by_margin_gate": m_rm,
            "passed_margin_gate": scored - p_rm - m_rm, "removed_by_ev_gate": e_rm,
            "passed_ev_gate": scored - p_rm - m_rm - e_rm, "removed_by_active_trade": a_rm,
            "horizon_not_defined": h_rm, "final_trades": len(trades),
            "executed_trades": sum(t["outcome"] in sd.EXECUTED for t in trades)}


def evaluate(frame: pd.DataFrame, models: dict[str, sm.S10Model], threshold: float,
             side_picker: Callable | None = None) -> dict[str, Any]:
    w = sm.wide_frame(frame, models)
    trades, dec, opp = sm.simulate(w, threshold, side_picker=side_picker, target_r=TARGET_R)
    return {"model": sm.evaluate_models(frame, models, threshold), "trading": sm.scope_metrics(trades, opp),
            "trades": trades, "decisions": dict(dec), "opportunities": opp, "funnel": coverage_funnel(w, dec, trades),
            "wide": w}


def gate(m: dict | None, min_resolved: int) -> str:
    if not m or m["resolved"] < min_resolved:
        return INSUFFICIENT
    pf, ex = m["profit_factor"], m["expectancy_r"]
    return PASS if ex is not None and ex > 0 and pf is not None and pf > 1 else FAIL


def importance(m: sm.S10Model, side: str, frame: pd.DataFrame) -> tuple[str, dict[str, float]]:
    if m.sides[side].family == sm.LR:
        return "ABS_STANDARDIZED_COEFFICIENT", sm.coefficient_importance(m, side)
    return "PERMUTATION_BRIER_FOLD_VALIDATION", sm.permutation_importance(m, side, frame)


# ------------------------------------------------------------------ history / dataset
on_grid_mask, quarantine_off_grid, quarantine_report = hx.on_grid_mask, hx.quarantine_off_grid, hx.quarantine_report


def candidate_counts(market: sd.MarketData, days: Sequence[date]) -> dict[date, int]:
    last = sd.last_signal(HORIZON)
    bars = market.day_bars()
    return {d: sum(1 for t in bars.get(d, []) if sd.SIGNAL_FIRST <= t.time() <= last) for d in days}


def history(market: sd.MarketData, built: sd.BuildResult, days: Sequence[date]) -> dict[str, Any]:
    elig = {date.fromisoformat(k): v for k, v in built.opportunities.items()}
    rows = hx.month_rows(days, candidate_counts(market, days), elig, hx.REQUESTED_START, hx.TRAIN_END)
    return {**hx.history_range(rows, hx.REQUESTED_START, hx.TRAIN_END), "month_table": rows}


# ------------------------------------------------------------------ inner folds (TRAIN only)
def run_folds(data: pd.DataFrame, cols: Sequence[str], grid: np.ndarray, start: str,
              say: Callable[[str], None]) -> list[dict]:
    out = []
    for name, tr_end, va_lo, va_hi in FOLDS:
        trd, vad = _iso_days(data, start, tr_end), _iso_days(data, va_lo, va_hi)
        base = {"fold": name, "train": [start, tr_end.isoformat()], "validation": [va_lo.isoformat(), va_hi.isoformat()],
                "train_days": len(trd), "validation_days": len(vad)}
        if len(trd) < MIN_FOLD_TRAIN_SESSIONS or len(vad) < MIN_FOLD_VAL_SESSIONS:
            out.append({**base, "status": NOT_RUN, "reason": f"needs >= {MIN_FOLD_TRAIN_SESSIONS} train and >= "
                                                             f"{MIN_FOLD_VAL_SESSIONS} validation sessions"})
            continue
        say(f"  {name}: train {start}..{tr_end} ({len(trd)} days), validate {va_lo}..{va_hi} ({len(vad)} days)")
        val = _rows(data, vad)
        fits = {}
        for fam in sm.families():
            m = sm.fit_model(data, cols, trd, {s: fam for s in sd.SIDES}, grid, test_days=vad)
            models = {s: m for s in sd.SIDES}
            fits[fam] = {"model": m, "metrics": sm.evaluate_models(val, models, sm.THRESHOLDS[0]),
                         "trading": {t: evaluate(val, models, t)["trading"] for t in sm.THRESHOLDS},
                         "importance": {s: importance(m, s, val) for s in sd.SIDES}}
        out.append({**base, "status": RUN, "_fits": fits, "_train_days": trd, "_val_days": vad})
    return out


def family_stability(folds: Sequence[dict], fam: str, side: str, threshold: float) -> dict[str, Any]:
    run = [f for f in folds if f["status"] == RUN]
    met = [(f["fold"], f["_fits"][fam]["metrics"][side], f["_fits"][fam]["trading"][threshold][side]) for f in run]
    aucs = [(n, m["roc_auc"]) for n, m, _ in met if m["roc_auc"] is not None]
    good = sum(1 for _, m, _ in met if m["roc_auc"] is not None and m["roc_auc"] > 0.5
               and m["brier_skill"] is not None and m["brier_skill"] >= 0)
    n = len(met)
    vals = [a for _, a in aucs]
    std = _std(vals)
    excellent = [a for a in vals if a >= EXCELLENT_AUC]
    others = [a for a in vals if a < EXCELLENT_AUC]
    if n < MIN_STABILITY_FOLDS:
        status, why = FEW_FOLDS, f"only {n} inner folds run (< {MIN_STABILITY_FOLDS})"
    elif good < math.ceil(n / 2):
        status, why = UNSTABLE_MODEL, f"only {good}/{n} folds with AUC > 0.5 and Brier skill >= 0"
    elif std is not None and std > MAX_AUC_STD:
        status, why = UNSTABLE_MODEL, f"AUC std {std:.3f} > {MAX_AUC_STD}"
    elif len(excellent) == 1 and (_mean(others) or 0.0) < WEAK_OTHER_AUC:
        status, why = UNSTABLE_MODEL, f"excellent (AUC >= {EXCELLENT_AUC}) in one fold only"
    else:
        status, why = STABLE, f"{good}/{n} folds with AUC > 0.5 and Brier skill >= 0"
    best = max(aucs, key=lambda x: x[1]) if aucs else (None, None)
    worst = min(aucs, key=lambda x: x[1]) if aucs else (None, None)
    return {"family": fam, "side": side, "folds_run": n, "good_folds": good, "auc_mean": _mean(vals), "auc_std": std,
            "brier_mean": _mean([m["brier"] for _, m, _ in met]), "log_loss_mean": _mean([m["log_loss"] for _, m, _ in met]),
            "ece_mean": _mean([m["ece"] for _, m, _ in met]),
            "brier_skill_mean": _mean([m["brier_skill"] for _, m, _ in met]),
            "pf_mean": _mean([_finite_pf(t["profit_factor"]) for _, _, t in met]),
            "expectancy_mean": _mean([t["expectancy_r"] for _, _, t in met]),
            "max_drawdown_r_mean": _mean([t["max_drawdown_r"] for _, _, t in met]),
            "resolved_total": sum(t["resolved"] for _, _, t in met), "trading_threshold": threshold,
            "best_fold": best[0], "best_auc": best[1], "worst_fold": worst[0], "worst_auc": worst[1],
            "auc_spread": (best[1] - worst[1]) if aucs else None, "status": status, "reason": why}


def choose_family(rows: Sequence[dict]) -> tuple[str, str]:
    """Pre-declared: STABLE families first; then Brier, log loss, AUC (desc), ECE, expectancy (desc), PF (desc),
    drawdown, simpler family."""
    order = {sm.LR: 0, sm.HGB: 1, sm.LGBM: 2}
    have = [r for r in rows if r["brier_mean"] is not None]
    if not have:
        return sm.LR, "NO_FAMILY_SCORED (defaulted to the simplest family)"
    pool = [r for r in have if r["status"] == STABLE] or have
    big = lambda v, d: d if v is None else v  # noqa: E731
    best = min(pool, key=lambda r: (round(r["brier_mean"], 4), round(big(r["log_loss_mean"], 9), 4),
                                    -round(big(r["auc_mean"], 0), 3), round(big(r["ece_mean"], 9), 3),
                                    -big(r["expectancy_mean"], -9), -big(r["pf_mean"], -9),
                                    big(r["max_drawdown_r_mean"], 9e9), order[r["family"]]))
    why = "lowest mean inner-fold Brier (then log loss, AUC, ECE, expectancy, PF, drawdown)"
    if len(pool) < len(have):
        why += "; unstable families excluded"
    if best["status"] != STABLE:
        why += f" [{best['status']}]"
    return best["family"], why


def threshold_search(data: pd.DataFrame, folds: Sequence[dict], fams: dict[str, str]) -> tuple[list[dict], dict]:
    run = [f for f in folds if f["status"] == RUN]
    rows, per = [], {}
    for thr in sm.THRESHOLDS:
        pooled, opp, fold_rows = [], 0, []
        for f in run:
            combo = {s: f["_fits"][fams[s]]["model"] for s in sd.SIDES}
            ev = evaluate(_rows(data, f["_val_days"]), combo, thr)
            pooled += ev["trades"]
            opp += ev["opportunities"]
            fold_rows.append({"fold": f["fold"], "ev": ev})
        m = sm.trade_metrics(pooled, opp)
        pos = sum(1 for r in fold_rows if (r["ev"]["trading"]["COMBINED"]["expectancy_r"] or 0) > 0)
        rows.append({"threshold": thr, **m, "folds_run": len(run), "folds_positive_expectancy": pos,
                     "input_days": sorted(d for f in run for d in f["_val_days"])})
        per[thr] = fold_rows
    return rows, per


def choose_threshold(rows: Sequence[dict]) -> tuple[float, str]:
    ok = [r for r in rows if r["resolved"] >= THRESHOLD_MIN_RESOLVED and r["expectancy_r"] is not None]
    if not ok:
        return min(sm.THRESHOLDS), ("THRESHOLD_SELECTION_LOW_SAMPLE: no candidate reached "
                                    f"{THRESHOLD_MIN_RESOLVED} pooled resolved inner-fold trades (lowest kept)")
    best = max(ok, key=lambda r: (r["expectancy_r"], -r["threshold"]))
    return best["threshold"], (f"highest pooled inner-fold expectancy (R) among candidates with >= "
                               f"{THRESHOLD_MIN_RESOLVED} resolved trades")


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
                                "ev_gate": f"EV = {TARGET_R} * P_WIN - (1 - P_WIN) > 0",
                                "selected_on": "inner walk-forward folds inside expanded TRAIN only; frozen"},
            "model_metadata.json": meta}
    for name, obj in docs.items():
        (d / name).write_text(json.dumps(s6._jsonable(obj), indent=2, default=str), encoding="utf-8")
        paths[name] = (d / name).as_posix()
    return paths


# ------------------------------------------------------------------ checks
def _nan_equal(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or (not np.isnan(a) and not np.isnan(b)
                                             and np.isclose(a, b, rtol=1e-9, atol=1e-12))


def rolling_causality_check(market: sd.MarketData, data: pd.DataFrame, sample: int) -> dict[str, Any]:
    """Futures and option rolling features at t recomputed from candles <= t only must equal the full-data values."""
    rows = data.iloc[::max(1, len(data) // max(sample, 1))].head(sample).to_dict("records") if len(data) else []
    bad, fut_cols = [], [*sd.FUT_FEATURES]
    opt_cols = ["ret_1", "ret_3", "ret_6", "natr", "mom", "dist_high", "dist_low", "atr14", "vol_ratio"]
    for r in rows:
        ts = r["signal_time"]
        f = market.frames.get(r["futures_key"])
        if f is not None:
            full = sd.futures_feature_frame(f).loc[ts]
            cut = sd.futures_feature_frame(f[f["timestamp"] <= ts]).iloc[-1]
            if any(not _nan_equal(float(full[c]), float(cut[c])) for c in fut_cols):
                bad.append(f"{ts.isoformat()}|futures")
        p = market.premium(r["instrument_key"])
        if p is not None:
            a = sd.option_series(p).feat.loc[ts]
            b = sd.option_series(p[p["timestamp"] <= ts]).feat.iloc[-1]
            if any(not _nan_equal(float(a[c]), float(b[c])) for c in opt_cols):
                bad.append(f"{ts.isoformat()}|{r['side']}|option")
    return {"checked": len(rows), "mismatches": bad[:20], "passed": bool(rows) and not bad}


def futures_causality_check(market: sd.MarketData, data: pd.DataFrame, sessions: Sequence[date]) -> dict[str, Any]:
    sel = nf.select_contracts(market.contracts, market.frames, sessions)
    chk = nf.selection_checks(sel, market.contracts, market.frames, sessions)
    by_day = {r["session_date"]: r for r in sel}
    exp = {c.instrument_key: c.expiry for c in market.contracts}
    used = data.drop_duplicates("session_date")[["session_date", "futures_key"]].to_dict("records") if len(data) else []
    bad = [u["session_date"] for u in used
           if by_day.get(u["session_date"], {}).get("instrument_key") != u["futures_key"]
           or exp.get(u["futures_key"], date.min) < date.fromisoformat(u["session_date"])]
    return {"sessions": len(sel), "violations": chk["violations"] + bad[:20],
            "independent_of_price_and_volume": chk["independent_of_price_and_volume"],
            "rows_with_wrong_contract": len(bad),
            "passed": bool(used) and not chk["violations"] and not bad and chk["independent_of_price_and_volume"]}


def contract_validity(market: sd.MarketData, data: pd.DataFrame) -> dict[str, Any]:
    exp = {c.instrument_key: c.expiry for c in market.contracts}
    if data.empty:
        return {"rows": 0, "passed": False}
    sd_ = data["session_date"]
    expired_opt = int((data["expiry"] < sd_).sum())
    fut_exp = data["futures_key"].map(lambda k: exp.get(k, date.min).isoformat())
    expired_fut = int((fut_exp < sd_).sum())
    unplanned = int(sum(1 for k, d in zip(data["instrument_key"], sd_)
                        if k not in market.atm.get(date.fromisoformat(d), set())))
    out = {"rows": int(len(data)), "expired_option_contracts": expired_opt, "expired_futures_contracts": expired_fut,
           "option_not_point_in_time_atm": unplanned}
    out["passed"] = not (expired_opt or expired_fut or unplanned)
    return out


def preprocessor_check(data: pd.DataFrame, m: sm.S10Model) -> bool:
    if not m.pre_fit_ids:
        return False
    X = data.loc[m.pre_fit_ids, m.columns].to_numpy(dtype=float)
    imp, sc = m.preprocessor.named_steps["imputer"], m.preprocessor.named_steps["scaler"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return bool(np.allclose(imp.statistics_, med) and np.allclose(sc.mean_, imp.transform(X).mean(axis=0))
                and int(sc.n_samples_seen_) == len(X))


def calibration_check(data: pd.DataFrame, m: sm.S10Model, hi: str) -> bool:
    ok = True
    for s in sd.SIDES:
        c_ids, f_ids = set(m.sides[s].calib_ids), set(m.sides[s].fit_ids)
        cd = set(data.loc[list(c_ids), "session_date"]) if c_ids else set()
        ok &= not (c_ids & f_ids) and cd <= set(m.calib_days) and all(d <= hi for d in cd) \
            and (all(d > max(m.fit_days) for d in cd) if m.fit_days and cd else True)
    return bool(ok)


def ids_ok(data: pd.DataFrame, m: sm.S10Model, lo: str, hi: str) -> bool:
    ids = [*m.pre_fit_ids, *(i for s in sd.SIDES for i in m.sides[s].fit_ids),
           *(i for s in sd.SIDES for i in m.sides[s].calib_ids)]
    if not ids:
        return True
    d = data.loc[list(ids), "session_date"]
    return bool(((d >= lo) & (d <= hi)).all())


def leakage_checks(data: pd.DataFrame, *, final: sm.S10Model, fits: Sequence[tuple], start: str,
                   thr_rows: Sequence[dict], threshold: float, grid: np.ndarray, sep_days: Sequence[str],
                   oos_ran: bool, sep_ran: bool) -> dict[str, dict]:
    """fits: (name, model, train_end_iso, test_days)."""
    train_end = hx.TRAIN_END.isoformat()
    contamination = {"FINAL": ids_ok(data, final, start, train_end),
                     **{n: ids_ok(data, m, start, te) for n, m, te, _ in fits if n != "FINAL"}}
    scaler = {n: preprocessor_check(data, m) for n, m, _, _ in fits}
    calib = {n: calibration_check(data, m, te) for n, m, te, _ in fits}
    purge_bad = []
    for n, m, _, test_days in fits:
        test = _rows(data, test_days)
        for label, idx in (("fit", m.pre_fit_ids), *((f"calib_{s}", m.sides[s].calib_ids) for s in sd.SIDES)):
            if idx and sm.purge_violations(data.loc[list(idx)], test, grid):
                purge_bad.append(f"{n}|{label}")
    input_days = thr_rows[0]["input_days"] if thr_rows else []
    redo, _ = choose_threshold(thr_rows)
    sep_ids = set(data.index[data["session_date"].isin(set(sep_days))])
    used: set = set()
    for m in [final, *(f[1] for f in fits)]:
        used |= set(m.pre_fit_ids)
        for s in sd.SIDES:
            used |= set(m.sides[s].fit_ids) | set(m.sides[s].calib_ids)
    return {
        "train_oos_contamination": {"checks": contamination, "passed": all(contamination.values())},
        "scaler_leakage": {"checks": scaler, "note": "imputer medians and scaler means recomputed from the fit rows",
                           "passed": all(scaler.values())},
        "calibration_leakage": {"checks": calib, "passed": all(calib.values())},
        "threshold_leakage": {"input_days": f"{input_days[0] if input_days else '-'}..{input_days[-1] if input_days else '-'}",
                              "recomputed": redo, "frozen": threshold,
                              "passed": bool(input_days) and all(d <= train_end for d in input_days)
                              and redo == threshold},
        "purge_correctness": {"fits_checked": len(fits), "embargo_bars": sm.EMBARGO_BARS, "violations": purge_bad,
                              "passed": not purge_bad},
        "september_contamination_guard": {"september_rows_used_in_any_fit": len(sep_ids & used),
                                          "september_evaluated_only_after_august": (not sep_ran) or oos_ran,
                                          "label": hx.LEGACY_LABEL,
                                          "passed": not (sep_ids & used) and ((not sep_ran) or oos_ran)},
    }


def october_guard(market: sd.MarketData) -> dict[str, Any]:
    end = hx.RESEARCH_END
    mx = market.reads["max_ts"]
    spot_max = market.spot["timestamp"].dt.date.max() if len(market.spot) else None
    fut_max = max((f["timestamp"].dt.date.max() for f in market.frames.values() if len(f)), default=None)
    roots_ok = not any("fresh" in Path(r).name.lower() or "oct" in Path(r).name.lower() for r in market.roots)
    ok = bool(roots_ok and (mx is None or mx.date() <= end) and (spot_max is None or spot_max <= end)
              and (fut_max is None or fut_max <= end) and all(d <= end for d in market.sessions))
    return {"roots": market.roots, "max_option_read": mx.isoformat() if mx is not None else None,
            "max_spot_day": str(spot_max), "max_futures_day": str(fut_max), "passed": ok}


# ------------------------------------------------------------------ the study
def not_run(base: dict, reason: str, decision: str = INSUFF) -> dict[str, Any]:
    return {**base, "ran": False, "reason": reason, "decision": decision, "decision_reason": reason,
            "forward_paper": "NOT SAFE", "target_70": INSUFFICIENT, "lookahead_ok": None, "leakage_ok": None,
            "folds": [], "results": {}, "labels": {"VALIDATION": NOT_RUN, "OOS": NOT_RUN, "LEGACY_HOLDOUT": NOT_RUN}}


def build_study(*, market: sd.MarketData, availability: dict, data_quality: dict, freeze_dir: Path,
                benchmarks: Callable[[], dict], s10_reference: dict | None = None, mutation_days: int = 6,
                check_sample: int = 20, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    stages: list[str] = []
    flags = {"volume": bool(availability["use_option_volume"]), "oi": bool(availability["use_option_oi"])}
    cols = sd.feature_columns(flags["volume"], flags["oi"])
    grid = market.grid()
    caches: dict = {"fut": {}, "opt": {}}
    base = {"heading": HEADING, "strategy": STRATEGY, "banner": BANNER, "disclaimer": DISCLAIMER, "label": LABEL,
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"), "flags": flags, "feature_columns": cols,
            "availability": availability, "data_quality": data_quality, "october": "NOT LOADED",
            "production_unchanged": None, "live_orders": 0, "trading_mode": "PAPER", "live_trading_enabled": False,
            "splits": {"VALIDATION": [str(x) for x in hx.VALIDATION], "OOS": [str(x) for x in hx.OOS],
                       "LEGACY_HOLDOUT": [str(x) for x in hx.LEGACY], "LEGACY_LABEL": hx.LEGACY_LABEL,
                       "FRESH_HOLDOUT": "2026-10-01 -> latest (DO NOT LOAD)"}}

    say("building the TRAIN-period causal dataset (requested 2025-01-01 .. 2026-06-30; nothing after June) ...")
    stages.append("train_dataset")
    hist_days = sd.days_in(market.sessions, hx.REQUESTED_START, hx.TRAIN_END)
    built = sd.build_samples(market, hist_days, volume=flags["volume"], oi=flags["oi"], caches=caches,
                             target_r=TARGET_R)
    selection_max_read = market.reads["max_ts"]
    hist = history(market, built, hist_days)
    base.update(history=hist, build_funnel=dict(built.funnel), stages=stages)
    if hist["status"] != hx.HISTORY_OK:
        return not_run({**base, "history_status": hx.HISTORY_INSUFFICIENT},
                       f"{hx.HISTORY_INSUFFICIENT}: {hist['months']} clean contiguous months (< {hx.MIN_HISTORY_MONTHS})"
                       " - no modelling, no strategy promotion")
    start = hist["actual_start"]
    train = built.data[(built.data["session_date"] >= start)
                       & (built.data["session_date"] <= hx.TRAIN_END.isoformat())].reset_index(drop=True)
    n_train = len(train)
    if train.empty:
        return not_run({**base, "history_status": hist["status"]}, "no eligible TRAIN samples")

    say("inner walk-forward folds inside expanded TRAIN (July..October never touched) ...")
    stages.append("inner_folds")
    folds = run_folds(train, cols, grid, start, say)
    run = [f for f in folds if f["status"] == RUN]
    pre_thr = sm.THRESHOLDS[0]
    stab0 = {(fam, s): family_stability(folds, fam, s, pre_thr) for fam in sm.families() for s in sd.SIDES}
    chosen = {s: choose_family([stab0[(fam, s)] for fam in sm.families()]) for s in sd.SIDES}
    fams = {s: chosen[s][0] for s in sd.SIDES}
    stages.append("threshold_selection")
    thr_rows, thr_folds = threshold_search(train, folds, fams) if run else ([], {})
    thr, thr_why = choose_threshold(thr_rows) if thr_rows else (min(sm.THRESHOLDS), "NO_INNER_FOLDS")
    stability = [family_stability(folds, fam, s, thr) for fam in sm.families() for s in sd.SIDES]
    chosen_stab = {s: next(r for r in stability if r["family"] == fams[s] and r["side"] == s) for s in sd.SIDES}
    model_unstable = any(chosen_stab[s]["status"] != STABLE for s in sd.SIDES)
    fold_eval = {r["fold"]: r["ev"] for r in thr_folds.get(thr, [])}
    pos_folds = sum(1 for e in fold_eval.values() if (e["trading"]["COMBINED"]["expectancy_r"] or 0) > 0)
    inner_trading_stable = len(run) >= MIN_STABILITY_FOLDS and pos_folds >= math.ceil(len(run) / 2)

    say("building July / August / September rows (after selection; October never loaded) ...")
    stages.append("outer_dataset")
    outer_days = sd.days_in(market.sessions, hx.VALIDATION[0], hx.RESEARCH_END)
    outer = sd.build_samples(market, outer_days, volume=flags["volume"], oi=flags["oi"], caches=caches,
                             target_r=TARGET_R)
    data = pd.concat([train, outer.data], ignore_index=True)
    train_days = _iso_days(data, start, hx.TRAIN_END)
    val_days, oos_days = _iso_days(data, *hx.VALIDATION), _iso_days(data, *hx.OOS)
    sep_days = _iso_days(data, *hx.LEGACY)

    say("freezing S11 (final fit on the whole expanded TRAIN) ...")
    stages.append("freeze")
    final = sm.fit_model(data, cols, train_days, fams, grid, test_days=val_days)
    frozen_fp = sm.fingerprint(final, thr)
    models = {s: final for s in sd.SIDES}
    meta = {"strategy": STRATEGY, "research_only": True, "label": LABEL, "families": fams,
            "family_reason": {s: chosen[s][1] for s in sd.SIDES}, "threshold": thr, "threshold_reason": thr_why,
            "margin": sm.MARGIN, "ev_gate": f"EV = {TARGET_R} * P - (1 - P) > 0", "embargo_bars": sm.EMBARGO_BARS,
            "calibration": {s: final.sides[s].calibration for s in sd.SIDES},
            "status": {s: final.sides[s].status for s in sd.SIDES}, "train": [start, hx.TRAIN_END.isoformat()],
            "fit_days": [final.fit_days[0], final.fit_days[-1]] if final.fit_days else [],
            "calibration_days": [final.calib_days[0], final.calib_days[-1]] if final.calib_days else [],
            "purge": final.purge, "fingerprint": frozen_fp, "sklearn_version": sklearn.__version__,
            "frozen_at": datetime.now(IST).isoformat(timespec="seconds"),
            "note": "research artifact only - never loaded by production execution"}
    artifacts = save_artifacts(final, thr, meta, freeze_dir)

    results: dict[str, dict] = {"TRAIN": evaluate(_rows(data, train_days), models, thr)}
    say("July OUTER VALIDATION (frozen) ...")
    stages.append("validation_july")
    results["VALIDATION"] = evaluate(_rows(data, val_days), models, thr)
    val_label = gate(results["VALIDATION"]["trading"]["COMBINED"], MIN_VAL_RESOLVED)
    oos_ran = val_label == PASS
    oos_label = sep_label = NOT_RUN
    reasons = {"OOS": None, "LEGACY_HOLDOUT": None}
    if oos_ran:
        say("August OOS (frozen, no modification) ...")
        stages.append("oos_august")
        results["OOS"] = evaluate(_rows(data, oos_days), models, thr)
        oos_label = gate(results["OOS"]["trading"]["COMBINED"], MIN_OOS_RESOLVED)
        say(f"September {hx.LEGACY_LABEL} (historical comparison only) ...")
        stages.append("legacy_september")
        results["LEGACY_HOLDOUT"] = evaluate(_rows(data, sep_days), models, thr)
        sep_label = hx.LEGACY_LABEL
    else:
        reasons["OOS"] = reasons["LEGACY_HOLDOUT"] = f"July validation {val_label} - not carried forward"
    fp_after = sm.fingerprint(final, thr)

    say("S10-style three-month comparator (same pipeline, TRAIN Apr-Jun 2026 only; comparison only) ...")
    short_days = _iso_days(data, *SHORT_TRAIN)
    short = sm.fit_model(data, cols, short_days, fams, grid, test_days=val_days)
    short_res = {"VALIDATION": evaluate(_rows(data, val_days), {s: short for s in sd.SIDES}, thr)}
    if oos_ran:
        short_res["OOS"] = evaluate(_rows(data, oos_days), {s: short for s in sd.SIDES}, thr)

    say("feature stability / importance (reported only; nothing removed) ...")
    imp_rows, feat_rows = [], []
    for f in run:
        for fam, fit in f["_fits"].items():
            for s in sd.SIDES:
                method, imp = fit["importance"][s]
                for rank, c in enumerate(sm.top_k(imp, TOP_FEATURES), 1):
                    imp_rows.append({"scope": f["fold"], "family": fam, "side": s, "method": method, "rank": rank,
                                     "feature": c, "value": imp[c], "chosen_family": fam == fams[s]})
    for s in sd.SIDES:
        method, imp = importance(final, s, _rows(data, train_days))
        for rank, c in enumerate(sm.top_k(imp, TOP_FEATURES), 1):
            imp_rows.append({"scope": "FINAL_TRAIN", "family": fams[s], "side": s, "method": method, "rank": rank,
                             "feature": c, "value": imp[c], "chosen_family": True})
        cnt = Counter(r["feature"] for r in imp_rows if r["side"] == s and r["family"] == fams[s]
                      and r["scope"].startswith("FOLD_"))
        for feat, k in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            st = ("STABLE_FEATURE" if len(run) and k >= math.ceil(len(run) / 2) else
                  "UNSTABLE_FEATURE" if k == 1 else "MODERATE_FEATURE")
            feat_rows.append({"side": s, "family": fams[s], "feature": feat, "folds_in_top10": k,
                              "folds_run": len(run), "frequency": k / len(run) if run else None, "status": st,
                              "action": "none (reported only; not removed)"})

    say("anti-lookahead / leakage checks ...")
    mdays = s4.sample_days(train_days, mutation_days) if mutation_days else []
    src_all = {"spot": True, "futures": True, "options": True}
    src_opt = {"spot": False, "futures": False, "options": True}
    lookahead = {
        "future_candle_scrambling": s10.mutation_check(market, data, cols, mdays, flags, final, seed=11,
                                                       sources=src_all, price=True, volume=True, oi=True),
        "future_option_price_mutation": s10.mutation_check(market, data, cols, mdays, flags, final, seed=31,
                                                           sources=src_opt, price=True),
        "future_option_volume_mutation": s10.mutation_check(market, data, cols, mdays, flags, final, seed=41,
                                                            sources=src_opt, volume=True),
        "future_oi_mutation": s10.mutation_check(market, data, cols, mdays, flags, final, seed=51, sources=src_all,
                                                 oi=True),
        "feature_truncation": s10.truncation_check(market, _rows(data, train_days), cols, flags, check_sample),
        "rolling_window_causality": rolling_causality_check(market, _rows(data, train_days), check_sample),
        "contract_selection_causality": {
            "options": s10.contract_causality_check(market, _rows(data, train_days), check_sample),
            "futures": futures_causality_check(market, _rows(data, train_days),
                                               [date.fromisoformat(d) for d in train_days])},
    }
    cs = lookahead["contract_selection_causality"]
    cs["passed"] = bool(cs["options"]["passed"] and cs["futures"]["passed"])
    fits = [(f"{f['fold']}_{fam}", fit["model"], f["train"][1], f["_val_days"])
            for f in run for fam, fit in f["_fits"].items()]
    fits += [("FINAL", final, hx.TRAIN_END.isoformat(), val_days),
             ("S10_SHORT_COMPARATOR", short, SHORT_TRAIN[1].isoformat(), val_days)]
    leakage = leakage_checks(data, final=final, fits=fits, start=start, thr_rows=thr_rows, threshold=thr, grid=grid,
                             sep_days=sep_days, oos_ran=oos_ran, sep_ran="LEGACY_HOLDOUT" in results)
    leakage["train_oos_contamination"]["outer_rows_after_train"] = int(len(data) - n_train)
    leakage["train_oos_contamination"]["selection_max_option_file_read"] = selection_max_read.isoformat() \
        if selection_max_read is not None else None
    ends = pd.DatetimeIndex(train["event_end"])
    sel_ok = bool(len(train)) and train["session_date"].max() <= hx.TRAIN_END.isoformat() \
        and ends.max().date() <= hx.TRAIN_END
    leakage["train_oos_contamination"]["selection_rows_and_labels_end_by_train_end"] = sel_ok
    leakage["train_oos_contamination"]["passed"] = bool(leakage["train_oos_contamination"]["passed"] and sel_ok
                                                        and fp_after == frozen_fp)
    lookahead["october_loading_guard"] = october_guard(market)
    again = sm.fit_model(data, cols, train_days, fams, grid, test_days=val_days)
    rerun = evaluate(_rows(data, val_days), {s: again for s in sd.SIDES}, thr)
    w0, w1 = results["VALIDATION"]["wide"], rerun["wide"]
    same_p = all(np.array_equal(w0[f"p_{s}"].to_numpy(), w1[f"p_{s}"].to_numpy(), equal_nan=True) for s in sd.SIDES)
    key = lambda ts: [json.dumps(t, sort_keys=True, default=str) for t in ts]  # noqa: E731
    lookahead["deterministic_replay"] = {
        "probabilities_identical": same_p, "trades_identical": key(results["VALIDATION"]["trades"]) == key(rerun["trades"]),
        "fingerprint_identical": sm.fingerprint(again, thr) == frozen_fp,
        "passed": bool(same_p and key(results["VALIDATION"]["trades"]) == key(rerun["trades"])
                       and sm.fingerprint(again, thr) == frozen_fp)}
    lookahead["duplicate_trade_prevention"] = s10.duplicate_trade_check(
        {**{k: v["trades"] for k, v in results.items()}, **{f"FOLD_{k}": v["trades"] for k, v in fold_eval.items()}})
    lookahead_ok = all(v["passed"] for v in lookahead.values())
    leakage_ok = all(v["passed"] for v in leakage.values())
    ds = s10.dataset_checks(data)
    cv = contract_validity(market, data)
    data_ok = bool(data_quality.get("passed")) and ds["passed"] and cv["passed"] and bool(availability["required_ok"])

    trading = {k: v["trading"] for k, v in results.items()}
    oos_m = trading.get("OOS", {}).get("COMBINED")
    if oos_m is None or oos_m["resolved"] < MIN_OOS_RESOLVED:
        target = INSUFFICIENT
    else:
        target = "ACHIEVED" if (oos_m["accuracy"] or 0) >= TARGET_70 else "NOT ACHIEVED"
    val_auc = {s: results["VALIDATION"]["model"][s]["roc_auc"] for s in sd.SIDES}
    degradation = {s: (chosen_stab[s]["auc_mean"] - val_auc[s]) if chosen_stab[s]["auc_mean"] is not None
                   and val_auc[s] is not None else None for s in sd.SIDES}
    degraded = any(v is not None and v > MAX_AUC_DEGRADATION for v in degradation.values())
    models_ok = all(usable(final, s) for s in sd.SIDES)
    if not (lookahead_ok and leakage_ok and data_ok):
        decision, why = REJECTED, "lookahead / leakage / data-integrity check failed"
    elif not models_ok:
        decision, why = INSUFF, "not enough TRAIN samples to fit and calibrate both side models"
    elif val_label == FAIL:
        decision, why = REJECTED, "July validation failed (PF <= 1 or expectancy <= 0); not rescued by tuning"
    elif oos_label == FAIL:
        decision, why = REJECTED, "August OOS failed (expectancy <= 0 or PF <= 1)"
    elif model_unstable or not inner_trading_stable:
        decision, why = UNSTABLE, ("model unstable across inner folds" if model_unstable else
                                   f"inner-fold trading unstable ({pos_folds}/{len(run)} folds with positive expectancy)")
    elif val_label == INSUFFICIENT:
        decision, why = INSUFF, f"too few resolved July trades (< {MIN_VAL_RESOLVED})"
    elif oos_label == INSUFFICIENT:
        decision, why = INSUFF, f"too few resolved August trades (< {MIN_OOS_RESOLVED})"
    elif degraded:
        decision, why = REJECTED, f"inner-fold -> July AUC degradation > {MAX_AUC_DEGRADATION}"
    elif oos_m["max_drawdown_r"] > MAX_DD_R or (oos_m["coverage"] or 0) < MIN_COVERAGE:
        decision, why = REJECTED, "August drawdown too large or coverage too low"
    else:
        decision, why = PROMISING, "stable inner folds, July and August passed, leakage checks passed"
    forward = "SAFE" if decision == PROMISING else "NOT SAFE"

    bench = benchmarks() or {}
    comparison = comparison_rows(results, short_res, s10_reference, bench)
    return {
        **base, "ran": True, "history_status": hist["status"], "stages": stages,
        "train": {"range": [start, hx.TRAIN_END.isoformat()], "trading_days": len(train_days)},
        "dataset": {"train_rows": n_train, "outer_rows": int(len(outer.data)), "rows": int(len(data)),
                    "train_timestamps": int(train["signal_time"].nunique()),
                    "label_rows": int(data["outcome"].isin(sd.BINARY).sum()),
                    "labels": sd.label_counts(data), "class_balance": class_balance(data, train_days),
                    "build_funnel_train": dict(built.funnel), "build_funnel_outer": dict(outer.funnel),
                    "option_contracts": sorted(built.option_contracts | outer.option_contracts),
                    "futures_contracts": sorted(built.futures_contracts | outer.futures_contracts),
                    "hash": sd.dataset_hash(data, cols)},
        "folds": [fold_summary(f, fams, thr, fold_eval) for f in folds], "stability": stability,
        "chosen_stability": chosen_stab, "model_unstable": model_unstable, "inner_positive_folds": pos_folds,
        "inner_trading_stable": inner_trading_stable, "families": fams,
        "family_reason": {s: chosen[s][1] for s in sd.SIDES}, "threshold": thr, "threshold_reason": thr_why,
        "threshold_rows": [{k: v for k, v in r.items() if k != "input_days"} for r in thr_rows],
        "frozen": {"fingerprint": frozen_fp, "artifacts": artifacts, "metadata": meta},
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "wide"} for k, v in results.items()},
        "short_comparator": {k: {kk: vv for kk, vv in v.items() if kk not in ("wide", "trades")}
                             for k, v in short_res.items()},
        "labels": {"VALIDATION": val_label, "OOS": oos_label, "LEGACY_HOLDOUT": sep_label},
        "not_run_reasons": reasons, "degradation": degradation, "importance": imp_rows, "feature_stability": feat_rows,
        "calibration": calibration_rows(fold_eval, results), "comparison": comparison, "benchmarks": bench,
        "lookahead": lookahead, "lookahead_ok": lookahead_ok, "leakage": leakage, "leakage_ok": leakage_ok,
        "dataset_checks": ds, "contract_validity": cv, "data_ok": data_ok, "target_70": target,
        "decision": decision, "decision_reason": why, "forward_paper": forward,
        "_data": data, "_market": market, "_built": built, "_outer": outer,
    }


def class_balance(data: pd.DataFrame, train_days: Sequence[str]) -> dict[str, dict]:
    tr = _rows(data, train_days)
    out = {}
    for s in sd.SIDES:
        b = tr[(tr["side"] == s) & tr["outcome"].isin(sd.BINARY)]
        out[s] = {"train_binary": int(len(b)), "wins": int((b["y"] == 1).sum()), "losses": int((b["y"] == 0).sum()),
                  "win_share": float(b["y"].mean()) if len(b) else None,
                  "weighting": "balanced class weights (no synthetic samples, no oversampling, no shuffling)"}
    return out


def fold_summary(f: dict, fams: dict[str, str], thr: float, fold_eval: dict) -> dict[str, Any]:
    out = {k: v for k, v in f.items() if not k.startswith("_")}
    if f["status"] != RUN:
        return out
    out["families"] = {fam: {"metrics": {s: {k: fit["metrics"][s][k] for k in s10.METRIC_KEYS} for s in sd.SIDES},
                             "trading": {str(t): {sc: view(fit["trading"][t][sc]) for sc in SCOPES}
                                         for t in sm.THRESHOLDS}, "purge": fit["model"].purge}
                       for fam, fit in f["_fits"].items()}
    ev = fold_eval.get(f["fold"])
    if ev:
        out["chosen"] = {"families": fams, "threshold": thr,
                         "model": {s: {k: ev["model"][s][k] for k in s10.METRIC_KEYS} for s in SCOPES},
                         "trading": {sc: view(ev["trading"][sc]) for sc in SCOPES}, "funnel": ev["funnel"]}
    return out


def calibration_rows(fold_eval: dict, results: dict) -> list[dict]:
    out = []
    scopes = [(k, v["model"]) for k, v in fold_eval.items()]
    scopes += [(SPLIT_NAMES[k], results[k]["model"]) for k in ("VALIDATION", "OOS") if k in results]
    for name, m in scopes:
        for s in SCOPES:
            r = m[s]
            out.append({"scope": name, "side": s, "bin": "SUMMARY", "samples": r["samples"], "brier": r["brier"],
                        "log_loss": r["log_loss"], "ece": r["ece"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"]})
            for b in r["calibration"]:
                out.append({"scope": name, "side": s, "bin": b["bin"], "lo": b["lo"], "hi": b["hi"],
                            "samples": b["count"], "mean_predicted": b["mean_p"], "actual_win_rate": b["win_rate"]})
    return out


def _cmp_row(split: str, system: str, desc: str, model: dict | None, t: dict | None) -> dict[str, Any]:
    model = model or {}
    return {"split": split, "system": system, "description": desc, "auc": model.get("roc_auc"),
            "pr_auc": model.get("pr_auc"), "brier": model.get("brier"), "log_loss": model.get("log_loss"),
            **view(t), "status": RUN if t else NOT_RUN}


def comparison_rows(results: dict, short_res: dict, s10_ref: dict | None, bench: dict) -> list[dict]:
    out = []
    for split in ("VALIDATION", "OOS"):
        name = SPLIT_NAMES[split]
        b = bench.get(split, {})
        out.append(_cmp_row(name, "S2", "frozen S2 baseline (unchanged)", None, b.get("S2")))
        out.append(_cmp_row(name, "S9_FUTURES_VOLUME", "S9 rules on real futures volume (REJECTED)", None,
                            b.get("S9_FUTURES_VOLUME")))
        ref = ((s10_ref or {}).get("results") or {}).get(split) or {}
        out.append(_cmp_row(name, "S10_ORIGINAL", "S10 as run: Apr-Jun 2026 training, 2R label (reference only)",
                            (ref.get("model") or {}).get("COMBINED"), (ref.get("trading") or {}).get("COMBINED")))
        sr = short_res.get(split)
        out.append(_cmp_row(name, "S10_SHORT_HISTORY_L1", "same S11 pipeline / L1 / threshold, trained Apr-Jun 2026",
                            sr["model"]["COMBINED"] if sr else None, sr["trading"]["COMBINED"] if sr else None))
        r = results.get(split)
        out.append(_cmp_row(name, STRATEGY, "expanded-history ML (L1, gross, no costs)",
                            r["model"]["COMBINED"] if r else None, r["trading"]["COMBINED"] if r else None))
    return out


# ------------------------------------------------------------------ manifest / reports
def manifest(a: dict, market: sd.MarketData, store_info: dict) -> dict[str, Any]:
    h = a.get("history", {})
    ds = a.get("dataset", {})
    used = ds.get("option_contracts", [])
    opt_candles = sum(len(p) for p in (market.premium(k) for k in used) if p is not None)
    fut = [{"instrument_key": c.instrument_key, "symbol": c.trading_symbol, "expiry": c.expiry.isoformat(),
            "lot_size": c.lot_size, "first": str(market.frames[c.instrument_key]["timestamp"].min())
            if c.instrument_key in market.frames else None,
            "last": str(market.frames[c.instrument_key]["timestamp"].max())
            if c.instrument_key in market.frames else None} for c in market.contracts]
    return {"banner": BANNER, "requested_range": [hx.REQUESTED_START.isoformat(), hx.TRAIN_END.isoformat()],
            "research_range_end": hx.RESEARCH_END.isoformat(), "october": "NOT LOADED",
            "actual_range": [h.get("actual_start"), h.get("actual_end")], "history_months": h.get("months"),
            "history_status": h.get("status"), "trading_days_train": h.get("trading_days"),
            "sessions_loaded": len(market.sessions), "missing_periods": h.get("missing_periods"),
            "month_table": h.get("month_table"), "roots": market.roots,
            "futures_contracts": fut, "futures_candles": int(sum(len(f) for f in market.frames.values())),
            "option_contracts_used": len(used), "option_candles_used": int(opt_candles),
            "feature_rows": ds.get("rows"), "label_rows": ds.get("label_rows"), "dataset_hash": ds.get("hash"),
            "stores": store_info}


REPORT_FILES = ("summary.json", "s11_report.md", "dataset_manifest.json", "data_quality.json", "fold_results.csv",
                "model_comparison.csv", "feature_stability.csv", "feature_importance.csv", "calibration.csv",
                "threshold_results.csv", "july_validation.csv", "august_oos.csv", "september_legacy.csv",
                "s2_vs_s9_vs_s10_vs_s11.csv", "trade_log.csv", "coverage_funnel.csv")
TRADE_COLS = ("split", "signal_time", "session_date", "direction", "side", "p_win", "p_other", "ev_r", "threshold",
              "instrument_key", "contract", "strike", "expiry", "entry_time", "entry", "stop_loss", "target", "risk",
              "outcome", "exit_time", "exit_reason", "exit_price", "r_multiple", "pnl", "holding_minutes")


def fold_rows(a: dict) -> list[dict]:
    out = []
    for f in a.get("folds", []):
        base = {"fold": f["fold"], "train": "..".join(f["train"]), "validation": "..".join(f["validation"]),
                "train_days": f["train_days"], "validation_days": f["validation_days"], "status": f["status"]}
        if f["status"] != RUN:
            out.append({**base, "reason": f.get("reason")})
            continue
        for fam, d in f["families"].items():
            for s in sd.SIDES:
                for t, tr in d["trading"].items():
                    out.append({**base, "family": fam, "side": s, "threshold": t,
                                "chosen": fam == a["families"][s] and float(t) == a["threshold"],
                                **{k: d["metrics"][s][k] for k in ("samples", "roc_auc", "pr_auc", "brier", "log_loss",
                                                                   "ece", "brier_skill")}, **tr[s]})
    return out


def split_rows(a: dict, split: str) -> list[dict]:
    r = a.get("results", {}).get(split)
    if not r:
        return [{"split": SPLIT_NAMES[split], "status": NOT_RUN, "reason": a.get("not_run_reasons", {}).get(split)}]
    lab = hx.LEGACY_LABEL if split == "LEGACY_HOLDOUT" else a["labels"].get(split)
    return [{"split": SPLIT_NAMES[split], "label": lab, "scope": sc,
             **{k: r["model"][sc][k] for k in s10.METRIC_KEYS}, **{k: r["trading"][sc][k] for k in s10.TRADE_KEYS}}
            for sc in SCOPES]


def funnel_rows(a: dict) -> list[dict]:
    out = [{"scope": "DATASET_TRAIN_BUILD", **a.get("build_funnel", {})}]
    for f in a.get("folds", []):
        if f.get("chosen"):
            out.append({"scope": f["fold"], "threshold": a["threshold"], **f["chosen"]["funnel"]})
    for k, v in a.get("results", {}).items():
        out.append({"scope": SPLIT_NAMES[k], "threshold": a["threshold"], **v["funnel"]})
    return out


def trade_rows(a: dict) -> list[dict]:
    out = []
    for k, v in a.get("results", {}).items():
        for t in v["trades"]:
            out.append({c: (SPLIT_NAMES[k] if c == "split" else t.get(c)) for c in TRADE_COLS})
    return out


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not str(k).startswith("_") and k != "trades"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def write_reports(a: dict, out_dir: Path, manifest_obj: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    js = lambda path, obj: path.write_text(json.dumps(s6._jsonable(obj), indent=2, default=str),  # noqa: E731
                                           encoding="utf-8")
    js(p["summary.json"], _strip(a))
    js(p["dataset_manifest.json"], manifest_obj)
    js(p["data_quality.json"], {"stores": a.get("data_quality"), "dataset_checks": a.get("dataset_checks"),
                                "contract_validity": a.get("contract_validity"), "availability": a.get("availability"),
                                "history": a.get("history"), "data_ok": a.get("data_ok")})
    s6._csv(p["fold_results.csv"], fold_rows(a))
    s6._csv(p["model_comparison.csv"], a.get("stability", []))
    s6._csv(p["feature_stability.csv"], a.get("feature_stability", []))
    s6._csv(p["feature_importance.csv"], a.get("importance", []))
    s6._csv(p["calibration.csv"], a.get("calibration", []))
    s6._csv(p["threshold_results.csv"], [{**r, "chosen": r["threshold"] == a.get("threshold")}
                                         for r in a.get("threshold_rows", [])])
    s6._csv(p["july_validation.csv"], split_rows(a, "VALIDATION"))
    s6._csv(p["august_oos.csv"], split_rows(a, "OOS"))
    s6._csv(p["september_legacy.csv"], split_rows(a, "LEGACY_HOLDOUT"))
    s6._csv(p["s2_vs_s9_vs_s10_vs_s11.csv"], a.get("comparison", []))
    s6._csv(p["trade_log.csv"], trade_rows(a))
    s6._csv(p["coverage_funnel.csv"], funnel_rows(a))
    p["s11_report.md"].write_text(markdown(a), encoding="utf-8")
    return p


# ------------------------------------------------------------------ text
def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _f(v, nd: int = 4) -> str:
    return "N/A" if v is None else ("inf" if v == math.inf else f"{v:.{nd}f}")


def tline(m: dict | None) -> str:
    if not m:
        return "NOT_RUN"
    return (f"Trades {m['trades']} | Resolved {m['resolved']} | Win Rate {_pct(m['accuracy'])} | PF "
            f"{_f(m['profit_factor'], 2)} | Expectancy {_f(m['expectancy_r'], 3)}R | P&L Rs {m['net_pnl']:,.2f} | "
            f"Max DD Rs {m['max_drawdown']:,.2f} | Max Streak {m['max_losing_streak']} | Coverage "
            f"{_pct(m['coverage'])} [{m['sample']}]")


def fold_line(f: dict, side: str) -> str:
    if f["status"] != RUN:
        return f"NOT_RUN ({f.get('reason')})"
    c = f.get("chosen")
    if not c:
        return "RUN (no chosen-model evaluation)"
    m, t = c["model"][side], c["trading"][side]
    return (f"{c['families'][side] if side in c['families'] else ''} AUC {_f(m['roc_auc'], 3)} | Brier "
            f"{_f(m['brier'], 4)} | trades {t['trades']} | resolved {t['resolved']} | WR {_pct(t['accuracy'])} | PF "
            f"{_f(t['profit_factor'], 2)} | exp {_f(t['expectancy_r'], 3)}R")


def _cmp(a: dict, split: str, system: str) -> dict:
    return next((r for r in a.get("comparison", []) if r["split"] == SPLIT_NAMES[split] and r["system"] == system), {})


def terminal_text(a: dict) -> str:
    h = a.get("history", {})
    out = [HEADING, f"({DISCLAIMER})", "", "HISTORY:",
           f"requested: {hx.REQUESTED_START} -> {hx.TRAIN_END}",
           f"actual: {h.get('actual_start')} -> {h.get('actual_end')}",
           f"trading days: {h.get('trading_days')}", f"months: {h.get('months')} ({h.get('status')})"]
    miss = h.get("missing_periods") or []
    if miss:
        out.append("missing: " + ", ".join(f"{m['month']} {m['status']}" for m in miss))
    og = (a.get("data_quality") or {}).get("off_grid_quarantine")
    if og and og["rows_dropped"]:
        out.append(f"off-grid option candles dropped (not shifted): {og['rows_dropped']} rows, "
                   f"{og['contracts_affected']} contracts, days {', '.join(og['days'])}")
    tr = a.get("train", {})
    out += ["", "TRAIN:", f"date range: {' -> '.join(tr.get('range', [])) or 'NOT_RUN'}",
            f"trading days: {tr.get('trading_days', 0)}", "", "VALIDATION:", "July", "", "OOS:", "August", "",
            "LEGACY:", "September — CONTAMINATED", "", "OCTOBER:", "NOT LOADED", ""]
    if not a.get("ran"):
        out += [f"S11 NOT RUN: {a.get('reason')}", ""]
    else:
        out.append("INNER WALK-FORWARD:")
        for i, f in enumerate(a["folds"], 1):
            out += ["", f"Fold {i}:", f"CE: {fold_line(f, 'CE')}", f"PE: {fold_line(f, 'PE')}"]
        out += ["", "MODEL STABILITY:"]
        for s in sd.SIDES:
            c = a["chosen_stability"][s]
            out.append(f"{s}: {c['family']} {c['status']} (AUC mean {_f(c['auc_mean'], 3)} +/- {_f(c['auc_std'], 3)}, "
                       f"Brier {_f(c['brier_mean'], 4)}, good folds {c['good_folds']}/{c['folds_run']}, worst "
                       f"{c['worst_fold']} {_f(c['worst_auc'], 3)}, best {c['best_fold']} {_f(c['best_auc'], 3)})")
        out.append(f"inner trading: {a['inner_positive_folds']}/{len([f for f in a['folds'] if f['status'] == RUN])} "
                   f"folds positive expectancy; threshold {a['threshold']:.2f} ({a['threshold_reason']})")
        for split, title in (("VALIDATION", "OUTER VALIDATION JULY"), ("OOS", "OOS AUGUST")):
            r = a["results"].get(split)
            out += ["", f"{title}:" + ("" if r else f" NOT_RUN ({a['not_run_reasons'].get(split)})")]
            if r:
                out += [f"{sc}: {tline(r['trading'][sc])}" for sc in SCOPES]
        out += ["", "S10 vs S11:"]
        for split, title in (("VALIDATION", "JULY"), ("OOS", "AUGUST")):
            out += ["", f"{title}:"]
            for system in ("S10_ORIGINAL", "S10_SHORT_HISTORY_L1", STRATEGY):
                r = _cmp(a, split, system)
                out.append(f"{system}: AUC {_f(r.get('auc'), 4)} | PF {_f(r.get('profit_factor'), 2)} | Expectancy "
                           f"{_f(r.get('expectancy_r'), 3)}R | Trades {r.get('trades')}")
    ok = lambda v: "N/A (not run)" if v is None else ("PASS" if v else "FAIL")  # noqa: E731
    out += ["", "70% TARGET:", a.get("target_70", INSUFFICIENT), "", "LOOKAHEAD:", ok(a.get("lookahead_ok")), "",
            "MODEL LEAKAGE:", ok(a.get("leakage_ok")), "", "DATA INTEGRITY:", ok(a.get("data_ok")), "",
            "OCTOBER PROTECTED:", "YES", "", "PRODUCTION CHANGED:",
            "NO" if a.get("production_unchanged") is not False else "YES", "", "LIVE ORDERS:", "0", "",
            "FORWARD PAPER:", a.get("forward_paper", "NOT SAFE"), "", "FINAL:", a.get("decision", INSUFF),
            f"({a.get('decision_reason')})", "",
            "Historical backtest only - not live accuracy, no guaranteed profit. No October performance exists."]
    return "\n".join(out)


def markdown(a: dict) -> str:
    h = a.get("history", {})
    md = ["# S11 expanded-history ML NIFTY options research", "", f"**{BANNER}**", "", f"> {DISCLAIMER}", "",
          f"## Final: **{a.get('decision')}** - {a.get('decision_reason')}", "",
          f"Forward PAPER: **{a.get('forward_paper')}** | 70% target: {a.get('target_70')} | Lookahead "
          f"{a.get('lookahead_ok')} | Leakage {a.get('leakage_ok')} | Data integrity {a.get('data_ok')} | "
          "October: NOT LOADED | Production changed: NO | Live orders: 0", "",
          "## History", "", f"- Requested {hx.REQUESTED_START} -> {hx.TRAIN_END}; actual {h.get('actual_start')} -> "
          f"{h.get('actual_end')} ({h.get('months')} clean months, {h.get('trading_days')} sessions, {h.get('status')})",
          f"- Rule: {h.get('rule')}", "", "| Month | Sessions | Eligible share | Status |", "|---|---|---|---|"]
    md += [f"| {r['month']} | {r['sessions']} | {_pct(r['eligible_share'])} | {r['status']} |"
           for r in h.get("month_table", [])]
    md += ["", "## Label (fixed)", ""] + [f"- {k}: {v}" for k, v in LABEL.items()]
    if not a.get("ran"):
        md += ["", f"**Not run:** {a.get('reason')}"]
        return "\n".join(md) + "\n"
    md += ["", "## Inner walk-forward (expanded TRAIN only)", "",
           "| Fold | Train | Validation | Status | CE (chosen) | PE (chosen) |", "|---|---|---|---|---|---|"]
    md += [f"| {f['fold']} | {'..'.join(f['train'])} | {'..'.join(f['validation'])} | {f['status']} | "
           f"{fold_line(f, 'CE')} | {fold_line(f, 'PE')} |" for f in a["folds"]]
    md += ["", "## Model stability (per family and side)", "",
           "| Family | Side | Folds | Good | AUC mean | AUC std | Brier | Log loss | ECE | PF mean | Exp mean | Status |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['family']} | {r['side']} | {r['folds_run']} | {r['good_folds']} | {_f(r['auc_mean'], 3)} | "
           f"{_f(r['auc_std'], 3)} | {_f(r['brier_mean'])} | {_f(r['log_loss_mean'])} | {_f(r['ece_mean'], 3)} | "
           f"{_f(r['pf_mean'], 2)} | {_f(r['expectancy_mean'], 3)} | {r['status']} |" for r in a["stability"]]
    md += ["", f"Chosen families: {a['families']} ({a['family_reason']}). Threshold {a['threshold']:.2f}: "
           f"{a['threshold_reason']}. Class balance: {a['dataset']['class_balance']}.", "",
           "## Outer results (frozen)", ""]
    for split in ("TRAIN", "VALIDATION", "OOS", "LEGACY_HOLDOUT"):
        r = a["results"].get(split)
        lab = hx.LEGACY_LABEL if split == "LEGACY_HOLDOUT" else a["labels"].get(split, "IN_SAMPLE")
        md.append(f"**{SPLIT_NAMES[split]}** ({lab})" + ("" if r else f": NOT_RUN - {a['not_run_reasons'].get(split)}"))
        if r:
            md += [f"- {sc}: {tline(r['trading'][sc])} | AUC {_f(r['model'][sc]['roc_auc'])} | Brier "
                   f"{_f(r['model'][sc]['brier'])}" for sc in SCOPES]
        md.append("")
    md += ["## S2 vs S9 vs S10 vs S11", "",
           "| Split | System | AUC | PR AUC | Brier | Log loss | Trades | Win rate | PF | Expectancy | P&L | DD | Coverage |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['split']} | {r['system']} | {_f(r['auc'])} | {_f(r['pr_auc'])} | {_f(r['brier'])} | "
           f"{_f(r['log_loss'])} | {r.get('trades')} | {_pct(r.get('accuracy'))} | {_f(r.get('profit_factor'), 2)} | "
           f"{_f(r.get('expectancy_r'), 3)} | {r.get('net_pnl')} | {r.get('max_drawdown')} | {_pct(r.get('coverage'))} |"
           for r in a["comparison"]]
    md += ["", "S10_ORIGINAL used the 2R label (reference only); S10_SHORT_HISTORY_L1 is the like-for-like "
           "three-month comparator (same L1, families and threshold).", "", "## Checks", ""]
    md += [f"- lookahead `{k}`: {'PASS' if v['passed'] else 'FAIL'}" for k, v in a["lookahead"].items()]
    md += [f"- leakage `{k}`: {'PASS' if v['passed'] else 'FAIL'}" for k, v in a["leakage"].items()]
    md += ["", "Nothing was tuned on July, August or September. September is LEGACY_HOLDOUT_CONTAMINATED. October was "
           "not loaded. Production unchanged; live orders 0."]
    return "\n".join(md) + "\n"
