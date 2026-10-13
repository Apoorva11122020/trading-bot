"""S10.1 - ML signal / label / coverage diagnostic (PAPER | RESEARCH ONLY | NO ORDERS | DIAGNOSTIC ONLY).

Explains why the frozen S10 candidate produced so few trades and almost no validation edge. Uses ONLY TRAIN
(Apr-Jun) and VALIDATION (July) data and the already-frozen S10 artifacts. Nothing is re-fitted, re-calibrated,
re-thresholded or re-selected: horizon, stop, target, features, model families and thresholds are unchanged.
August, September and October are never loaded (market data is clipped at 2026-07-31).
"""
from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.historical_data import nifty_futures as nf
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s6_clean_strategy_lab as s6

HEADING = "S10.1 DIAGNOSTIC COMPLETE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | DIAGNOSTIC ONLY (no tuning, no retraining)"
DISCLAIMER = ("ML model estimates the historical probability of a predefined option outcome from causal features. "
              "This diagnostic explains the frozen S10 result; it does not change or optimise it.")
DIAG_SPLITS = ("TRAIN", "VALIDATION")
DIAG_END = sd.SPLITS["VALIDATION"][1]
BREAKEVEN_WIN_RATE = 1.0 / (1.0 + sd.TARGET_R)
BUCKETS = ((0.0, 0.4, "<0.4"), (0.4, 0.5, "0.4-0.5"), (0.5, 0.6, "0.5-0.6"), (0.6, 0.7, "0.6-0.7"),
           (0.7, 0.8, "0.7-0.8"), (0.8, 1.0000001, "0.8+"))
QUANTILES = (("min", 0.0), ("p10", 0.10), ("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90),
             ("max", 1.0))
OUTSIDE_WINDOW = "OUTSIDE_SIGNAL_WINDOW"
ATR_HISTORY, ATR_INVALID = "OPTION_ATR_INSUFFICIENT_HISTORY", "OPTION_ATR_INVALID"
CATEGORY = {
    OUTSIDE_WINDOW: "design signal window (09:45..14:40 candle start, exit by 15:15)",
    "CONTRACT_LIST_UNAVAILABLE": "no ATM option",
    "NO_LISTED_EXPIRY": "invalid expiry",
    "OUTSIDE_PLANNED_STRIKES": "contract unavailable (ATM strike not in the downloaded plan)",
    "OPTION_DATA_MISSING": "contract unavailable (no stored candles)",
    "OPTION_SIGNAL_CANDLE_MISSING": "missing option candle",
    ATR_HISTORY: "missing previous option candles (ATR14 needs 14 completed candles)",
    ATR_INVALID: "missing ATR (non-positive / undefined)",
    "FUTURES_CANDLE_MISSING": "other: futures candle missing",
    "ELIGIBLE": "eligible",
}
# Pre-declared diagnosis rules (fixed before the diagnostic was run on real data).
GATING_REMOVED_SHARE = 0.90
LABEL_WIN_SHARE_EXECUTED = 0.20
LABEL_UNRESOLVED_SHARE = 0.50
COVERAGE_ELIGIBLE_SHARE = 0.50
FEATURE_BAD_SHARE = 0.25
NEAR_CONSTANT_TOP_SHARE = 0.95
DUPLICATE_CORR = 0.99
SUSPICIOUS_LABEL_CORR = 0.30
OVERFIT_AUC_GAP = 0.15
LOW_EDGE_AUC = 0.55
IMPORTANCE_TOP = 20
MIN_JACCARD = 0.25
MIN_SPEARMAN = 0.20
LABELS = ("LABEL_PROBLEM", "DATA_COVERAGE_PROBLEM", "GATING_PROBLEM", "FEATURE_PROBLEM", "MODEL_PROBLEM",
          "GENUINE_LOW_EDGE")


# ------------------------------------------------------------------ frozen artifacts (read-only)
def load_frozen(models_dir: Path) -> tuple[sm.S10Model, float, dict]:
    need = ("ce_model.pkl", "pe_model.pkl", "preprocessor.pkl", "calibrator_ce.pkl", "calibrator_pe.pkl",
            "feature_columns.json", "thresholds.json", "model_metadata.json")
    missing = [n for n in need if not (models_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen S10 artifacts missing in {models_dir}: {missing}")
    rd = lambda n: json.loads((models_dir / n).read_text(encoding="utf-8"))  # noqa: E731
    ld = lambda n: pickle.loads((models_dir / n).read_bytes())  # noqa: E731
    cols, thr, meta = rd("feature_columns.json"), rd("thresholds.json"), rd("model_metadata.json")
    sides = {}
    for s in sd.SIDES:
        est, cal = ld(f"{s.lower()}_model.pkl"), ld(f"calibrator_{s.lower()}.pkl")
        status = meta["status"][s] if est is not None else sm.INSUFFICIENT
        sides[s] = sm.SideModel(meta["families"][s], status, est, cal, meta["calibration"][s])
    model = sm.S10Model(list(cols), ld("preprocessor.pkl"), sides, [], list(meta.get("fit_days") or []),
                        list(meta.get("calibration_days") or []), meta.get("purge") or {})
    return model, float(thr["threshold"]), meta


def raw_probability(model: sm.S10Model, side: str, frame: pd.DataFrame) -> np.ndarray:
    s = model.sides[side]
    if s.estimator is None or frame.empty:
        return np.full(len(frame), np.nan)
    X = model.preprocessor.transform(frame[model.columns].to_numpy(dtype=float))
    return s.estimator.predict_proba(X)[:, 1]


def flags_from_columns(cols: Sequence[str]) -> dict[str, bool]:
    return {"volume": "sel_vol_ratio" in cols, "oi": "sel_oi_chg_1" in cols}


# ------------------------------------------------------------------ coverage audit (same order as build_samples)
def coverage_audit(market: sd.MarketData, days: Sequence[date], caches: dict) -> pd.DataFrame:
    """One row per raw NIFTY timestamp: the timestamp-level exclusion reason (identical precedence to
    ``build_samples``: futures -> expiry -> CE leg -> PE leg) plus each leg's own independent reason."""
    want = [d for d in sorted(set(days)) if d in set(market.sessions)]
    sel = {date.fromisoformat(r["session_date"]): r for r in nf.select_contracts(market.contracts, market.frames,
                                                                                  want)}
    spot_close, day_bars = market.spot_close(), market.day_bars()
    last6 = sd.last_signal(sd.HORIZON)
    fut_ts: dict[str, set] = {}
    out = []

    def series(c):
        k = c.instrument_key
        if k not in caches["opt"]:
            caches["opt"][k] = sd.option_series(market.premium(k))
        return caches["opt"][k]

    def leg(d, expiry, kind, ts, sc) -> str | None:
        c, why = sd.resolve_contract(market, d, expiry, kind, sc)
        if c is None:
            return why.split(":")[0] if why else "CONTRACT_UNAVAILABLE"
        s = series(c)
        if s is None:
            return "OPTION_DATA_MISSING"
        j = int(s.feat.index.get_indexer([ts])[0])
        if j < 0:
            return "OPTION_SIGNAL_CANDLE_MISSING"
        a = float(s.feat["atr14"].iat[j])
        if not (np.isfinite(a) and a > 0):
            return ATR_HISTORY if j + 1 < sd.ATR_PERIOD else ATR_INVALID
        return None

    for d in want:
        r = sel[d]
        expiry = sd.nearest_expiry(market.calendar, d)
        key = r.get("instrument_key")
        if r["status"] == nf.SELECTED and key not in fut_ts:
            fut_ts[key] = set(sd._clean(market.frames[key])["timestamp"])
        for ts in day_bars.get(d, []):
            row = {"signal_time": ts, "session_date": d.isoformat(), "split": sd.split_of_day(d)}
            if not (sd.SIGNAL_FIRST <= ts.time() <= last6):
                out.append({**row, "reason": OUTSIDE_WINDOW, "CE": None, "PE": None, "in_window": False})
                continue
            if r["status"] != nf.SELECTED:
                why = f"FUTURES_{r['reason']}"
                out.append({**row, "reason": why, "CE": why, "PE": why, "in_window": True})
                continue
            if expiry is None:
                out.append({**row, "reason": "NO_LISTED_EXPIRY", "CE": "NO_LISTED_EXPIRY", "PE": "NO_LISTED_EXPIRY",
                            "in_window": True})
                continue
            if ts not in fut_ts[key]:
                out.append({**row, "reason": "FUTURES_CANDLE_MISSING", "CE": "FUTURES_CANDLE_MISSING",
                            "PE": "FUTURES_CANDLE_MISSING", "in_window": True})
                continue
            sc = spot_close[ts]
            legs = {k: leg(d, expiry, k, ts, sc) for k in sd.SIDES}
            first = next((legs[k] for k in sd.SIDES if legs[k]), None)
            out.append({**row, "reason": first or "ELIGIBLE", **legs, "in_window": True})
    return pd.DataFrame(out)


def category(reason: str) -> str:
    if reason in CATEGORY:
        return CATEGORY[reason]
    if reason.startswith("FUTURES_"):
        return "other: causal futures contract unavailable"
    if reason.startswith("STRIKE_UNAVAILABLE"):
        return "no ATM option"
    return "other"


# ------------------------------------------------------------------ label excursions (label diagnostic only)
def excursions(s: sd.OptionSeries | None, sig_ts: pd.Timestamp, atr_value: float,
               horizon: int = sd.HORIZON) -> tuple[float | None, float | None]:
    """Max favourable / adverse excursion in R over the full horizon window (entry candle included)."""
    if s is None:
        return None, None
    entry_ts = sig_ts + sd.STEP
    k = int(s.t.get_indexer([entry_ts])[0])
    if k < 0:
        return None, None
    entry, risk = float(s.o[k]), sd.STOP_ATR * float(atr_value)
    if not (risk > 0 and entry - risk > 0):
        return None, None
    j_end = int(s.t.searchsorted(entry_ts + horizon * sd.STEP, side="left"))
    hi, lo = float(np.max(s.h[k:j_end])), float(np.min(s.l[k:j_end]))
    return (hi - entry) / risk, (entry - lo) / risk


# ------------------------------------------------------------------ small helpers
def _pct(n: float, d: float) -> float | None:
    return float(n) / float(d) if d else None


def _q(values: np.ndarray) -> dict[str, float | None]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {k: (float(np.quantile(v, q)) if len(v) else None) for k, q in QUANTILES}


def auc_se(auc: float | None, n_pos: int, n_neg: int) -> float | None:
    """Hanley-McNeil standard error of a ROC AUC."""
    if auc is None or not n_pos or not n_neg:
        return None
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(math.sqrt(max(var, 0.0)))


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    ok = np.isfinite(p)
    y, p = np.asarray(y, dtype=int)[ok], np.asarray(p, dtype=float)[ok]
    return float(roc_auc_score(y, p)) if len(y) and 0 < y.sum() < len(y) else None


def _spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    ra = pd.Series([a[k] for k in keys]).rank()
    rb = pd.Series([b[k] for k in keys]).rank()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------ the diagnostic
def run_diagnostic(*, market: sd.MarketData, model: sm.S10Model, threshold: float, meta: dict,
                   s10_summary: dict | None = None,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    cols = model.columns
    flags = flags_from_columns(cols)
    days = sd.days_in(market.sessions, sd.SPLITS["TRAIN"][0], DIAG_END)
    caches: dict = {"fut": {}, "opt": {}}
    say(f"rebuilding the S10 dataset for TRAIN + VALIDATION ({len(days)} sessions; Aug..Oct never loaded) ...")
    built = sd.build_samples(market, days, volume=flags["volume"], oi=flags["oi"], caches=caches)
    data = built.data.copy()
    data["split"] = [sd.split_of_day(date.fromisoformat(d)) for d in data["session_date"]]
    say("coverage audit (every raw NIFTY timestamp) ...")
    cov = coverage_audit(market, days, caches)
    say("frozen-model probabilities (raw and Platt-calibrated) ...")
    data["p_raw"], data["p_cal"] = np.nan, np.nan
    for s in sd.SIDES:
        m = data["side"] == s
        data.loc[m, "p_raw"] = raw_probability(model, s, data[m])
        data.loc[m, "p_cal"] = model.predict(s, data[m])
    say("label excursions (MFE / MAE) ...")
    mfe, mae = [], []
    for r in data[["instrument_key", "signal_time", "atr"]].itertuples(index=False):
        f, a = excursions(caches["opt"].get(r.instrument_key), r.signal_time, r.atr)
        mfe.append(f)
        mae.append(a)
    data["mfe_r"], data["mae_r"] = pd.Series(mfe, dtype=float).values, pd.Series(mae, dtype=float).values
    max_read = market.reads["max_ts"]
    models = {s: model for s in sd.SIDES}
    frames = {k: data[data["split"] == k] for k in DIAG_SPLITS}
    base_rate = {}
    for s in sd.SIDES:
        b = frames["TRAIN"][(frames["TRAIN"]["side"] == s) & frames["TRAIN"]["outcome"].isin(sd.BINARY)]
        base_rate[s] = float(b["y"].mean()) if len(b) else None

    say("dataset counts / labels / coverage ...")
    counts = dataset_counts(data, cov)
    labels = label_rows(data)
    coverage = coverage_rows(cov, data, cols)
    say("prediction distribution + trade funnel ...")
    prob = probability_rows(data)
    funnel, trades = funnel_rows(frames, models, threshold)
    say("model quality by class ...")
    quality = quality_rows(frames, base_rate, threshold)
    say("feature health ...")
    health = feature_health(frames, cols)
    say("feature importance (frozen models, permutation on TRAIN and VALIDATION) ...")
    importance, stability = importance_rows(model, frames)
    say("calibration buckets ...")
    calibration = calibration_rows(data)
    say("contract selection audit ...")
    contracts = contract_rows(market, data, cov)
    calib_params = {s: ({"slope": float(model.sides[s].calibrator.coef_[0][0]),
                         "intercept": float(model.sides[s].calibrator.intercept_[0])}
                        if model.sides[s].calibrator is not None else None) for s in sd.SIDES}
    reproduce = reproduction(funnel, s10_summary)
    fp = sm.fingerprint(model, threshold) if all(model.sides[s].status == sm.FITTED for s in sd.SIDES) else None
    ranking = root_causes(cov, funnel)
    letters, flags_out, diagnosis = diagnose(counts=counts, labels=labels, cov=cov, funnel=funnel, quality=quality,
                                             health=health, stability=stability, calib_params=calib_params,
                                             contracts=contracts, reproduce=reproduce, data=data)
    return {
        "heading": HEADING, "banner": BANNER, "disclaimer": DISCLAIMER, "splits_used": list(DIAG_SPLITS),
        "period": [days[0].isoformat(), days[-1].isoformat()] if days else None,
        "max_market_read": max_read.isoformat() if max_read is not None else None,
        "no_post_july_data": bool(max_read is None or max_read.date() <= DIAG_END)
        and bool(data.empty or data["session_date"].max() <= DIAG_END.isoformat())
        and bool(not len(market.spot) or market.spot["timestamp"].dt.date.max() <= DIAG_END),
        "frozen": {"threshold": threshold, "margin": sm.MARGIN, "families": meta.get("families"),
                   "calibration": meta.get("calibration"), "fingerprint_metadata": meta.get("fingerprint"),
                   "fingerprint_reloaded": fp, "fingerprint_match": fp == meta.get("fingerprint"),
                   "horizon": sd.HORIZON, "stop": f"{sd.STOP_ATR} x option ATR14 (= 1R)",
                   "target": f"{sd.TARGET_R}R (= {sd.STOP_ATR * sd.TARGET_R} x ATR14)",
                   "calibrator_params": calib_params, "train_base_rate": base_rate},
        "dataset_counts": counts, "labels": labels, "coverage": coverage, "probability": prob, "funnel": funnel,
        "quality": quality, "feature_health": health, "importance": importance, "stability": stability,
        "calibration": calibration, "contracts": contracts, "reproduction": reproduce, "root_causes": ranking,
        "letters": letters, "flags": flags_out, "diagnosis": diagnosis, "funnel_trades": trades,
        "production_unchanged": None, "live_orders": 0, "october_used": False, "trading_mode": "PAPER",
        "_data": data, "_coverage": cov,
    }


# ------------------------------------------------------------------ 1. dataset counts
def dataset_counts(data: pd.DataFrame, cov: pd.DataFrame) -> list[dict]:
    out = []
    for split in (*DIAG_SPLITS, "TOTAL"):
        c = cov if split == "TOTAL" else cov[cov["split"] == split]
        d = data if split == "TOTAL" else data[data["split"] == split]
        base = {"split": split}
        out += [{**base, "metric": "raw_nifty_timestamps", "count": int(len(c))},
                {**base, "metric": "signal_window_timestamps", "count": int(c["in_window"].sum()) if len(c) else 0},
                {**base, "metric": "eligible_prediction_timestamps",
                 "count": int((c["reason"] == "ELIGIBLE").sum()) if len(c) else 0}]
        for s in sd.SIDES:
            g = d[d["side"] == s]
            n = Counter(g["outcome"])
            ex = sum(n[o] for o in sd.EXECUTED)
            res = n[sd.WIN] + n[sd.LOSS]
            out.append({**base, "side": s, "metric": "eligible_samples", "count": int(len(g))})
            for o in (*sd.EXECUTED, sd.NO_ENTRY, sd.INVALID_LEVELS):
                out.append({**base, "side": s, "metric": o, "count": int(n[o]),
                            "pct_of_samples": _pct(n[o], len(g)),
                            "pct_of_executed": _pct(n[o], ex) if o in sd.EXECUTED else None,
                            "pct_of_resolved": _pct(n[o], res) if o in sd.BINARY else None})
    return out


def _count(counts: list[dict], split: str, metric: str, side: str | None = None) -> int:
    return next((r["count"] for r in counts if r["split"] == split and r["metric"] == metric
                 and r.get("side") == side), 0)


# ------------------------------------------------------------------ 2. labels
def label_rows(data: pd.DataFrame) -> list[dict]:
    out = []
    for split in (*DIAG_SPLITS, "TOTAL"):
        d = data if split == "TOTAL" else data[data["split"] == split]
        for s in sd.SIDES:
            g = d[d["side"] == s]
            n = Counter(g["outcome"])
            why = Counter(g["exit_reason"].dropna())
            ex = sum(n[o] for o in sd.EXECUTED)
            res = n[sd.WIN] + n[sd.LOSS]
            e = g[g["outcome"].isin(sd.EXECUTED)]
            mfe, mae = e["mfe_r"].to_numpy(dtype=float), e["mae_r"].to_numpy(dtype=float)
            ok = np.isfinite(mfe)
            risk_pct = (e["risk"].astype(float) / e["entry"].astype(float)).replace([np.inf, -np.inf], np.nan)
            win_res = _pct(n[sd.WIN], res)
            win_ex = _pct(n[sd.WIN], ex)
            out.append({
                "split": split, "side": s, "samples": int(len(g)), "executed": int(ex), "resolved": int(res),
                "target_before_stop": int(n[sd.WIN]), "stop_before_target": int(n[sd.LOSS]),
                "timeout": int(n[sd.TIMEOUT]), "ambiguous": int(n[sd.AMBIGUOUS]), "no_entry": int(n[sd.NO_ENTRY]),
                "invalid_levels": int(n[sd.INVALID_LEVELS]),
                "target_pct_of_executed": win_ex, "stop_pct_of_executed": _pct(n[sd.LOSS], ex),
                "timeout_pct_of_executed": _pct(n[sd.TIMEOUT], ex), "ambiguous_pct_of_executed": _pct(n[sd.AMBIGUOUS], ex),
                "win_rate_of_resolved": win_res, "breakeven_win_rate": BREAKEVEN_WIN_RATE,
                "target_gap_exits": int(why["TARGET_GAP"]), "stop_gap_exits": int(why["STOP_GAP"]),
                "avg_mfe_r": float(np.mean(mfe[ok])) if ok.any() else None,
                "avg_mae_r": float(np.mean(mae[ok])) if ok.any() else None,
                "median_mfe_r": float(np.median(mfe[ok])) if ok.any() else None,
                "p75_mfe_r": float(np.quantile(mfe[ok], 0.75)) if ok.any() else None,
                "p90_mfe_r": float(np.quantile(mfe[ok], 0.90)) if ok.any() else None,
                "share_mfe_ge_target": float(np.mean(mfe[ok] >= sd.TARGET_R)) if ok.any() else None,
                "share_mae_ge_stop": float(np.mean(mae[ok] >= 1.0)) if ok.any() else None,
                "avg_risk_pct_of_entry": float(risk_pct.mean()) if len(risk_pct.dropna()) else None,
                "avg_target_pct_of_entry": float(risk_pct.mean() * sd.TARGET_R) if len(risk_pct.dropna()) else None,
                "win_label_verdict": ("INSUFFICIENT_DATA" if not ex else
                                      "EXTREMELY_RARE" if (win_ex or 0) < 0.10 else
                                      "RARE" if (win_ex or 0) < LABEL_WIN_SHARE_EXECUTED else "NOT_RARE"),
            })
    return out


# ------------------------------------------------------------------ 3. coverage
def coverage_rows(cov: pd.DataFrame, data: pd.DataFrame, cols: Sequence[str]) -> list[dict]:
    out = []
    for split in (*DIAG_SPLITS, "TOTAL"):
        c = cov if split == "TOTAL" else cov[cov["split"] == split]
        raw, win = len(c), int(c["in_window"].sum()) if len(c) else 0
        for reason, n in Counter(c["reason"]).most_common():
            out.append({"split": split, "level": "TIMESTAMP", "reason": reason, "category": category(reason),
                        "count": int(n), "pct_of_raw": _pct(n, raw),
                        "pct_of_window": _pct(n, win) if reason != OUTSIDE_WINDOW else None})
        w = c[c["in_window"]] if len(c) else c
        for s in sd.SIDES:
            for reason, n in Counter(w[s].fillna("OK")).most_common():
                out.append({"split": split, "level": f"{s}_LEG_INDEPENDENT", "reason": reason,
                            "category": "leg available" if reason == "OK" else category(reason), "count": int(n),
                            "pct_of_window": _pct(n, win)})
        d = data if split == "TOTAL" else data[data["split"] == split]
        for o in (sd.NO_ENTRY, sd.INVALID_LEVELS):
            n = int((d["outcome"] == o).sum())
            out.append({"split": split, "level": "ELIGIBLE_SAMPLE_UNRESOLVABLE", "reason": o,
                        "category": "eligible sample without a tradeable label", "count": n,
                        "pct_of_samples": _pct(n, len(d))})
        nan_rows = int(d[list(cols)].isna().any(axis=1).sum()) if len(d) else 0
        out.append({"split": split, "level": "ELIGIBLE_SAMPLE_KEPT", "reason": "MISSING_FEATURE",
                    "category": "missing feature (row kept; TRAIN-median imputed, never dropped)", "count": nan_rows,
                    "pct_of_samples": _pct(nan_rows, len(d))})
    return out


def main_exclusion(cov: pd.DataFrame) -> tuple[str | None, int, float | None]:
    w = cov[cov["in_window"] & (cov["reason"] != "ELIGIBLE")] if len(cov) else cov
    if not len(w):
        return None, 0, None
    reason, n = Counter(w["reason"]).most_common(1)[0]
    return reason, int(n), _pct(n, int(cov["in_window"].sum()))


# ------------------------------------------------------------------ 4. prediction distribution + funnel
def probability_rows(data: pd.DataFrame) -> list[dict]:
    out = []
    for split in (*DIAG_SPLITS, "TOTAL"):
        d = data if split == "TOTAL" else data[data["split"] == split]
        for s in sd.SIDES:
            g = d[d["side"] == s]
            for kind, col in (("calibrated", "p_cal"), ("raw", "p_raw")):
                for pop, sub in (("ALL_ELIGIBLE", g), ("BINARY_WIN_LOSS", g[g["outcome"].isin(sd.BINARY)])):
                    p = sub[col].to_numpy(dtype=float)
                    out.append({"split": split, "side": s, "probability": kind, "population": pop,
                                "n": int(np.isfinite(p).sum()), **_q(p),
                                **{f"count_ge_{t:.2f}": int((p >= t).sum()) for t in sm.THRESHOLDS}})
    return out


def funnel_rows(frames: dict[str, pd.DataFrame], models: dict, threshold: float) -> tuple[list[dict], dict]:
    out, trades = [], {}
    for split, f in frames.items():
        w = sm.wide_frame(f, models) if len(f) else pd.DataFrame(columns=["p_CE", "p_PE"])
        pce, ppe = w["p_CE"].to_numpy(dtype=float), w["p_PE"].to_numpy(dtype=float)
        best, other = np.fmax(pce, ppe), np.fmin(pce, ppe)
        margin_any = int(((best - other) >= sm.MARGIN).sum())
        best_side = Counter("CE" if a >= b else "PE" for a, b in zip(pce, ppe) if np.isfinite(a) and np.isfinite(b))
        for thr in sm.THRESHOLDS:
            tr, dec, opp = sm.simulate(w, thr)
            n = Counter(t["outcome"] for t in tr)
            unavailable, below = dec["PROBABILITY_UNAVAILABLE"], dec["BELOW_THRESHOLD"]
            after_p = opp - unavailable - below
            after_m = after_p - dec["MARGIN_TOO_SMALL"]
            after_ev = after_m - dec["EV_NOT_POSITIVE"]
            out.append({"split": split, "threshold": thr, "primary_frozen_threshold": thr == threshold,
                        "eligible": int(opp), "probability_unavailable": int(unavailable),
                        "below_threshold": int(below), "after_probability_gate": int(after_p),
                        "margin_removed": int(dec["MARGIN_TOO_SMALL"]), "after_margin_gate": int(after_m),
                        "ev_removed": int(dec["EV_NOT_POSITIVE"]), "after_ev_gate": int(after_ev),
                        "active_trade_removed": int(dec["ACTIVE_TRADE_OPEN"]),
                        "horizon_undefined": int(dec["HORIZON_NOT_DEFINED"]), "final_trades": len(tr),
                        "executed": int(sum(n[o] for o in sd.EXECUTED)), "resolved": int(n[sd.WIN] + n[sd.LOSS]),
                        "wins": int(n[sd.WIN]), "losses": int(n[sd.LOSS]), "timeout": int(n[sd.TIMEOUT]),
                        "ambiguous": int(n[sd.AMBIGUOUS]), "no_entry": int(n[sd.NO_ENTRY] + n[sd.INVALID_LEVELS]),
                        "timestamps_margin_ge_0.05_any_threshold": margin_any,
                        "timestamps_best_side_CE": int(best_side["CE"]), "timestamps_best_side_PE": int(best_side["PE"]),
                        "note": "frozen threshold" if thr == threshold else "report only - NOT a tuning candidate"})
            if thr == threshold:
                trades[split] = tr
    return out, trades


def primary_funnel(funnel: list[dict], split: str | None = None) -> dict[str, int]:
    rows = [r for r in funnel if r["primary_frozen_threshold"] and (split is None or r["split"] == split)]
    keys = ("eligible", "probability_unavailable", "below_threshold", "after_probability_gate", "margin_removed",
            "after_margin_gate", "ev_removed", "after_ev_gate", "active_trade_removed", "horizon_undefined",
            "final_trades", "executed", "resolved", "wins", "losses", "timeout", "ambiguous", "no_entry")
    return {k: int(sum(r[k] for r in rows)) for k in keys}


# ------------------------------------------------------------------ 5. model quality by class
def quality_rows(frames: dict[str, pd.DataFrame], base_rate: dict, threshold: float) -> list[dict]:
    out = []
    for split, f in frames.items():
        for s in (*sd.SIDES, "COMBINED"):
            b = f[f["outcome"].isin(sd.BINARY)] if s == "COMBINED" else f[(f["side"] == s)
                                                                         & f["outcome"].isin(sd.BINARY)]
            y = b["y"].to_numpy(dtype=int)
            br = (float(np.mean([v for v in base_rate.values() if v is not None])) if s == "COMBINED"
                  and any(v is not None for v in base_rate.values()) else base_rate.get(s))
            m = sm.model_metrics(y, b["p_cal"].to_numpy(dtype=float), threshold, br)
            auc_raw = _auc(y, b["p_raw"].to_numpy(dtype=float)) if s != "COMBINED" else None
            se = auc_se(m["roc_auc"], m["wins"], m["losses"])
            out.append({"split": split, "side": s, "samples": m["samples"], "wins": m["wins"], "losses": m["losses"],
                        "class_balance_win_share": m["win_share"], "roc_auc": m["roc_auc"], "roc_auc_raw": auc_raw,
                        "roc_auc_se": se,
                        "roc_auc_z_vs_0.5": ((m["roc_auc"] - 0.5) / se) if (se and m["roc_auc"] is not None) else None,
                        "pr_auc": m["pr_auc"], "pr_auc_baseline": m["win_share"], "brier": m["brier"],
                        "climatology_brier": m["climatology_brier"], "brier_skill": m["brier_skill"],
                        "log_loss": m["log_loss"], "ece": m["ece"], "threshold": threshold,
                        "predicted_positive": m["predicted_positive"], "precision": m["precision"],
                        "recall": m["recall"], "accuracy_at_threshold": m["accuracy_at_threshold"],
                        "note": "TRAIN = in-sample (frozen model fit on these days)" if split == "TRAIN" else
                        "VALIDATION July (frozen, out-of-sample)"})
    return out


def _quality(quality: list[dict], split: str, side: str) -> dict:
    return next((r for r in quality if r["split"] == split and r["side"] == side), {})


# ------------------------------------------------------------------ 6. feature health
def feature_health(frames: dict[str, pd.DataFrame], cols: Sequence[str]) -> list[dict]:
    tr, va = frames["TRAIN"], frames["VALIDATION"]
    X = tr[list(cols)].astype(float)
    corr = X.corr().abs() if len(X) > 2 else pd.DataFrame(index=cols, columns=cols, dtype=float)
    out = []
    for c in cols:
        v = X[c]
        fin = v[np.isfinite(v)]
        top = float(fin.value_counts(normalize=True).iloc[0]) if len(fin) else None
        dup = [o for o in cols if o != c and (corr.loc[c, o] >= DUPLICATE_CORR if pd.notna(corr.loc[c, o]) else False)]
        same = [o for o in cols if o != c and len(fin) and X[o].equals(v)]
        ycorr = {}
        for s in sd.SIDES:
            b = tr[(tr["side"] == s) & tr["outcome"].isin(sd.BINARY)]
            x, y = b[c].astype(float), b["y"].astype(float)
            ok = x.notna()
            ycorr[s] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 \
                else None
        n_unique = int(fin.nunique())
        out.append({
            "feature": c, "group": ("futures" if c.startswith("fut_") else "time" if c in sd.TIME_FEATURES else
                                    "selected_option" if c.startswith("sel_") else "opposite_option"
                                    if c.startswith("opp_") else "relative"),
            "missing_pct_train": float(v.isna().mean()) if len(v) else None,
            "missing_pct_validation": float(va[c].isna().mean()) if len(va) else None,
            "inf_count_train": int(np.isinf(tr[c].astype(float)).sum()), "n_unique_train": n_unique,
            "variance_train": float(fin.var()) if len(fin) > 1 else None,
            "std_train": float(fin.std()) if len(fin) > 1 else None, "top_value_share": top,
            "constant": n_unique <= 1, "near_constant": bool(n_unique > 1 and top is not None
                                                               and top >= NEAR_CONSTANT_TOP_SHARE),
            "highly_correlated_with": " ".join(dup), "identical_to": " ".join(same),
            "corr_with_win_CE": ycorr["CE"], "corr_with_win_PE": ycorr["PE"],
            "suspicious_future_derived": bool(any(abs(x) >= SUSPICIOUS_LABEL_CORR for x in ycorr.values()
                                                  if x is not None)),
            "construction": "causal: candles <= signal candle t only (verified by the S10 truncation / mutation tests)",
        })
    return out


# ------------------------------------------------------------------ 7. feature importance (frozen models)
def importance_rows(model: sm.S10Model, frames: dict[str, pd.DataFrame]) -> tuple[list[dict], dict]:
    out, stab = [], {}
    for s in sd.SIDES:
        perms = {k: sm.permutation_importance(model, s, f) for k, f in frames.items()}
        coef = sm.coefficient_importance(model, s)
        ranks = {k: {c: i for i, c in enumerate(sm.top_k(v, len(v)), 1)} for k, v in perms.items() if v}
        tops = {k: sm.top_k(v, IMPORTANCE_TOP) for k, v in perms.items() if v}
        jac = sm.jaccard(tops["TRAIN"], tops["VALIDATION"]) if {"TRAIN", "VALIDATION"} <= set(tops) else None
        rho = _spearman(perms.get("TRAIN", {}), perms.get("VALIDATION", {}))
        primary = coef or perms.get("TRAIN", {})
        for i, c in enumerate(sm.top_k(primary, len(primary)), 1):
            out.append({"side": s, "family": model.sides[s].family, "feature": c,
                        "primary_rank": i, "primary_method": "ABS_STANDARDIZED_COEFFICIENT" if coef
                        else "PERMUTATION_BRIER_TRAIN", "primary_value": primary[c],
                        "train_permutation": perms.get("TRAIN", {}).get(c),
                        "validation_permutation": perms.get("VALIDATION", {}).get(c),
                        "train_rank": ranks.get("TRAIN", {}).get(c), "validation_rank": ranks.get("VALIDATION", {}).get(c),
                        "top20_primary": i <= IMPORTANCE_TOP,
                        "major_rank_shift": bool(ranks.get("TRAIN", {}).get(c) and ranks.get("VALIDATION", {}).get(c)
                                                 and abs(ranks["TRAIN"][c] - ranks["VALIDATION"][c]) >= 20),
                        "wording": "predictive contribution in this frozen model (not causation)"})
        neg_val = sum(1 for v in perms.get("VALIDATION", {}).values() if v < 0)
        stab[s] = {"top20_train": tops.get("TRAIN", []), "top20_validation": tops.get("VALIDATION", []),
                   "jaccard_top20": jac, "spearman_train_vs_validation": rho,
                   "validation_negative_importance_features": neg_val,
                   "unstable": bool((jac is not None and jac < MIN_JACCARD) or (rho is not None and rho < MIN_SPEARMAN))}
    return out, stab


# ------------------------------------------------------------------ 8. calibration
def calibration_rows(data: pd.DataFrame) -> list[dict]:
    out = []
    for split in DIAG_SPLITS:
        d = data[(data["split"] == split) & data["outcome"].isin(sd.BINARY)]
        for s in (*sd.SIDES, "COMBINED"):
            g = d if s == "COMBINED" else d[d["side"] == s]
            for basis in ("p_raw", "p_cal"):
                p = g[basis].to_numpy(dtype=float)
                for lo, hi, name in BUCKETS:
                    m = (p >= lo) & (p < hi)
                    n = int(m.sum())
                    wr = float(g["y"].to_numpy(dtype=float)[m].mean()) if n else None
                    mp = float(p[m].mean()) if n else None
                    out.append({"split": split, "side": s, "bucketed_by": "raw" if basis == "p_raw" else "calibrated",
                                "bucket": name, "count": n, "mean_raw": float(g["p_raw"].to_numpy()[m].mean()) if n
                                else None, "mean_calibrated": float(g["p_cal"].to_numpy()[m].mean()) if n else None,
                                "actual_win_rate": wr, "gap_predicted_minus_actual": (mp - wr) if n else None})
    return out


# ------------------------------------------------------------------ 9. contract selection
def contract_rows(market: sd.MarketData, data: pd.DataFrame, cov: pd.DataFrame) -> list[dict]:
    out = []
    spot = market.spot_close()
    step_cache: dict = {}
    for split in (*DIAG_SPLITS, "TOTAL"):
        d = data if split == "TOTAL" else data[data["split"] == split]
        for s in sd.SIDES:
            g = d[d["side"] == s]
            if not len(g):
                out.append({"split": split, "side": s, "samples": 0})
                continue
            dist = (g["strike"].astype(float) - g["spot"].astype(float)).abs()
            mism_exp = mism_atm = mism_spot = 0
            steps = []
            for r in g[["session_date", "expiry", "strike", "spot", "signal_time", "instrument_key"]].itertuples(
                    index=False):
                day = date.fromisoformat(r.session_date)
                exp = sd.nearest_expiry(market.calendar, day)
                mism_exp += exp is None or exp.isoformat() != r.expiry
                c, _ = sd.resolve_contract(market, day, exp, s, spot[r.signal_time]) if exp else (None, None)
                mism_atm += c is None or c.instrument_key != r.instrument_key
                mism_spot += abs(spot[r.signal_time] - r.spot) > 1e-9
                key = (day, exp, s)
                if key not in step_cache:
                    ks = sorted(market.chains.get(key, {}))
                    diffs = np.diff(ks)
                    step_cache[key] = float(diffs[diffs > 0].min()) if len(diffs[diffs > 0]) else None
                steps.append(step_cache[key])
            step = pd.Series(steps, dtype=float)
            dte = (pd.to_datetime(g["expiry"]) - pd.to_datetime(g["session_date"])).dt.days
            per_contract = g.groupby("instrument_key").size()
            per_day = g.groupby("session_date")["instrument_key"].nunique()
            out.append({"split": split, "side": s, "samples": int(len(g)),
                        "distinct_contracts": int(per_contract.size),
                        "samples_per_contract_max": int(per_contract.max()),
                        "samples_per_contract_median": float(per_contract.median()),
                        "contracts_per_day_mean": float(per_day.mean()),
                        "atm_distance_points_mean": float(dist.mean()), "atm_distance_points_max": float(dist.max()),
                        "strike_step_points": float(step.median()) if step.notna().any() else None,
                        "atm_distance_within_half_step_share": float((dist.to_numpy() <= step.to_numpy() / 2 + 1e-9)
                                                                     .mean()) if step.notna().all() else None,
                        "expiry_not_nearest_listed": int(mism_exp), "atm_reresolution_mismatch": int(mism_atm),
                        "spot_mismatch": int(mism_spot), "dte_min": int(dte.min()), "dte_median": float(dte.median()),
                        "dte_max": int(dte.max()), "expiry_day_share": float((dte == 0).mean()),
                        "dte_le_1_share": float((dte <= 1).mean()),
                        "expiry_before_session": int((dte < 0).sum())})
    w = cov[cov["in_window"]] if len(cov) else cov
    for split in (*DIAG_SPLITS, "TOTAL"):
        c = w if split == "TOTAL" else w[w["split"] == split]
        ce_ok, pe_ok = c["CE"].isna(), c["PE"].isna()
        out.append({"split": split, "side": "CE_PE_AVAILABILITY", "samples": int(len(c)),
                    "both_available": int((ce_ok & pe_ok).sum()), "only_CE_available": int((ce_ok & ~pe_ok).sum()),
                    "only_PE_available": int((~ce_ok & pe_ok).sum()), "neither_available": int((~ce_ok & ~pe_ok).sum())})
    return out


# ------------------------------------------------------------------ reproduction of the frozen S10 run
def reproduction(funnel: list[dict], s10_summary: dict | None) -> dict[str, Any]:
    out: dict[str, Any] = {"available": bool(s10_summary)}
    if not s10_summary:
        return {**out, "matches": None}
    ok = True
    for split in DIAG_SPLITS:
        ref = (((s10_summary.get("results") or {}).get(split) or {}).get("trading") or {}).get("COMBINED")
        got = primary_funnel(funnel, split)
        if ref is None:
            out[split] = None
            continue
        same = ref.get("trades") == got["executed"] and ref.get("resolved") == got["resolved"]
        ok &= same
        out[split] = {"s10_trades": ref.get("trades"), "s10_resolved": ref.get("resolved"),
                      "diagnostic_trades": got["executed"], "diagnostic_resolved": got["resolved"], "match": same}
    out["matches"] = ok
    return out


# ------------------------------------------------------------------ 10. root causes (measured counts only)
def root_causes(cov: pd.DataFrame, funnel: list[dict]) -> list[dict]:
    window = int(cov["in_window"].sum()) if len(cov) else 0
    f = primary_funnel(funnel)
    items = [("model probability below the frozen threshold", "probability gate",
              f["below_threshold"] + f["probability_unavailable"], f["eligible"]),
             ("CE/PE margin < 0.05", "margin gate", f["margin_removed"], f["after_probability_gate"]),
             ("EV gate (EV <= 0)", "EV gate", f["ev_removed"], f["after_margin_gate"]),
             ("one active trade at a time", "position rule", f["active_trade_removed"], f["after_ev_gate"])]
    w = cov[cov["in_window"] & (cov["reason"] != "ELIGIBLE")] if len(cov) else cov
    grouped: dict[str, int] = defaultdict(int)
    for reason, n in Counter(w["reason"]).items():
        grouped[f"{category(reason)} [{reason}]"] += n
    items += [(k, "coverage (before the model)", v, window) for k, v in grouped.items()]
    items.append(("taken trades that did not resolve (TIMEOUT / AMBIGUOUS / NO_ENTRY)",
                  "label outcome (reduces resolved, not trades)", f["timeout"] + f["ambiguous"] + f["no_entry"],
                  f["final_trades"]))
    rows = [{"cause": c, "stage": st, "removed": int(n), "stage_input": int(base), "pct_of_stage_input": _pct(n, base)}
            for c, st, n, base in items if n]
    rows.sort(key=lambda r: -r["removed"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ------------------------------------------------------------------ diagnosis (pre-declared rules)
def diagnose(*, counts, labels, cov, funnel, quality, health, stability, calib_params, contracts, reproduce,
             data) -> tuple[list[dict], dict, dict]:
    f = primary_funnel(funnel)
    window = int(cov["in_window"].sum()) if len(cov) else 0
    eligible = int((cov["reason"] == "ELIGIBLE").sum()) if len(cov) else 0
    lab = {r["side"]: r for r in labels if r["split"] == "TRAIN"}
    win_ex = {s: lab[s]["target_pct_of_executed"] for s in sd.SIDES if s in lab}
    unres = {s: _pct(lab[s]["timeout"] + lab[s]["ambiguous"], lab[s]["executed"]) for s in sd.SIDES if s in lab}
    vq = {s: _quality(quality, "VALIDATION", s) for s in (*sd.SIDES, "COMBINED")}
    tq = {s: _quality(quality, "TRAIN", s) for s in (*sd.SIDES, "COMBINED")}
    bad_feat = sum(1 for h in health if h["constant"] or h["near_constant"] or (h["missing_pct_train"] or 0) > 0.5)
    prob_removed = _pct(f["below_threshold"] + f["probability_unavailable"], f["eligible"])
    label_checks = bool(len(data) and ((data["outcome"] == sd.WIN) == (data["y"] == 1)).all()
                        and ((data["outcome"] == sd.LOSS) == (data["y"] == 0)).all())
    contract_bad = sum(r.get("atm_reresolution_mismatch", 0) + r.get("expiry_not_nearest_listed", 0)
                       + r.get("expiry_before_session", 0) for r in contracts if r.get("split") == "TOTAL")
    gap = {s: (tq[s].get("roc_auc") - vq[s].get("roc_auc")) if tq[s].get("roc_auc") is not None
           and vq[s].get("roc_auc") is not None else None for s in sd.SIDES}
    inverted = {s: bool(vq[s].get("roc_auc_z_vs_0.5") is not None and vq[s]["roc_auc_z_vs_0.5"] <= -2.0)
                for s in sd.SIDES}
    flags = {
        "GATING_PROBLEM": bool(prob_removed is not None and prob_removed >= GATING_REMOVED_SHARE),
        "LABEL_PROBLEM": bool(any(v is not None and v < LABEL_WIN_SHARE_EXECUTED for v in win_ex.values())
                              or any(v is not None and v > LABEL_UNRESOLVED_SHARE for v in unres.values())),
        "DATA_COVERAGE_PROBLEM": bool(window and _pct(eligible, window) < COVERAGE_ELIGIBLE_SHARE),
        "FEATURE_PROBLEM": bool(health and bad_feat / len(health) > FEATURE_BAD_SHARE),
        "MODEL_PROBLEM": bool(any(g is not None and g > OVERFIT_AUC_GAP for g in gap.values())
                              or any(inverted.values()) or reproduce.get("matches") is False),
        "GENUINE_LOW_EDGE": bool(vq["COMBINED"].get("roc_auc") is not None
                                 and vq["COMBINED"]["roc_auc"] < LOW_EDGE_AUC),
    }
    evidence = {
        "GATING_PROBLEM": f"probability gate removes {_fmt_pct(prob_removed)} of eligible timestamps "
                          f"(rule: >= {GATING_REMOVED_SHARE:.0%})",
        "LABEL_PROBLEM": f"TRAIN WIN share of executed CE {_fmt_pct(win_ex.get('CE'))} / PE {_fmt_pct(win_ex.get('PE'))};"
                         f" TIMEOUT+AMBIGUOUS CE {_fmt_pct(unres.get('CE'))} / PE {_fmt_pct(unres.get('PE'))} "
                         f"(rules: WIN < {LABEL_WIN_SHARE_EXECUTED:.0%} or unresolved > {LABEL_UNRESOLVED_SHARE:.0%})",
        "DATA_COVERAGE_PROBLEM": f"eligible {eligible} / {window} in-window timestamps "
                                 f"({_fmt_pct(_pct(eligible, window))}; rule < {COVERAGE_ELIGIBLE_SHARE:.0%})",
        "FEATURE_PROBLEM": f"{bad_feat}/{len(health)} features constant / near-constant / >50% missing "
                           f"(rule > {FEATURE_BAD_SHARE:.0%})",
        "MODEL_PROBLEM": f"TRAIN-VALIDATION AUC gap CE {_fmt(gap.get('CE'))} / PE {_fmt(gap.get('PE'))} "
                         f"(rule > {OVERFIT_AUC_GAP}); significantly inverted AUC (z <= -2): {inverted}; "
                         f"reproduces frozen S10 trades: {reproduce.get('matches')}",
        "GENUINE_LOW_EDGE": f"VALIDATION combined AUC {_fmt(vq['COMBINED'].get('roc_auc'))} (rule < {LOW_EDGE_AUC})",
    }
    ece = vq["COMBINED"].get("ece")
    max_cal = float(np.nanmax(data["p_cal"])) if len(data) and np.isfinite(data["p_cal"]).any() else None
    minority = min((tq[s].get("class_balance_win_share") or 0) for s in sd.SIDES)
    stab_bad = [s for s, v in stability.items() if v["unstable"]]
    st = lambda b: "SUPPORTED" if b else "NOT_SUPPORTED"  # noqa: E731
    letters = [
        {"code": "A", "hypothesis": "label construction", "status": st(not label_checks),
         "evidence": f"label/target consistency {'PASS' if label_checks else 'FAIL'}; first-touch order, gap and "
                     "same-candle rules unit-tested in S10"},
        {"code": "B", "hypothesis": "class imbalance", "status": "PARTIAL" if minority < 0.30 else "NOT_SUPPORTED",
         "evidence": f"TRAIN WIN share of resolved: CE {_fmt_pct(tq['CE'].get('class_balance_win_share'))}, PE "
                     f"{_fmt_pct(tq['PE'].get('class_balance_win_share'))} (balanced class weights already used); a "
                     "calibrated model then rarely outputs P >= 0.55"},
        {"code": "C", "hypothesis": "option data coverage", "status": st(flags["DATA_COVERAGE_PROBLEM"]),
         "evidence": evidence["DATA_COVERAGE_PROBLEM"]},
        {"code": "D", "hypothesis": "feature quality", "status": st(flags["FEATURE_PROBLEM"]),
         "evidence": evidence["FEATURE_PROBLEM"]},
        {"code": "E", "hypothesis": "probability calibration",
         "status": "NOT_SUPPORTED" if ece is not None and ece <= 0.10 else "SUPPORTED",
         "evidence": f"VALIDATION ECE {_fmt(ece)}; max calibrated P {_fmt(max_cal)}; Platt params {calib_params}"},
        {"code": "F", "hypothesis": "threshold / margin gating", "status": st(flags["GATING_PROBLEM"]),
         "evidence": evidence["GATING_PROBLEM"] + f"; margin removes {f['margin_removed']}, EV removes "
                                                  f"{f['ev_removed']}"},
        {"code": "G", "hypothesis": "target / stop definition", "status": st(flags["LABEL_PROBLEM"]),
         "evidence": evidence["LABEL_PROBLEM"]},
        {"code": "H", "hypothesis": "contract selection", "status": st(contract_bad > 0),
         "evidence": f"{contract_bad} ATM / expiry mismatches on re-resolution"},
        {"code": "I", "hypothesis": "model implementation", "status": st(reproduce.get("matches") is False),
         "evidence": f"frozen artifacts reproduce the S10 trade counts: {reproduce.get('matches')}; "
                     f"unstable importance: {stab_bad or 'none'}"},
        {"code": "J", "hypothesis": "genuinely weak predictive signal", "status": st(flags["GENUINE_LOW_EDGE"]),
         "evidence": evidence["GENUINE_LOW_EDGE"]},
    ]
    active = [k for k in LABELS if flags[k]]
    diagnosis = {"label": active[0] if len(active) == 1 else ("MIXED" if active else "GENUINE_LOW_EDGE"),
                 "components": active, "evidence": {k: evidence[k] for k in LABELS}}
    return letters, flags, diagnosis


def _fmt(v, nd: int = 4) -> str:
    return "N/A" if v is None else f"{v:.{nd}f}"


def _fmt_pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


# ------------------------------------------------------------------ reports
REPORT_FILES = ("dataset_counts.csv", "label_distribution.csv", "coverage_reasons.csv", "probability_distribution.csv",
                "trade_funnel.csv", "feature_health.csv", "feature_importance_diagnostic.csv",
                "calibration_diagnostic.csv", "contract_selection_diagnostic.csv", "diagnostic_report.md",
                "summary.json")


def _csv(path: Path, rows: Sequence[dict]) -> None:
    s6._csv(path, [{k: (v.isoformat() if isinstance(v, (pd.Timestamp, date)) else v) for k, v in r.items()}
                   for r in rows])


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    _csv(p["dataset_counts.csv"], a["dataset_counts"])
    _csv(p["label_distribution.csv"], a["labels"])
    _csv(p["coverage_reasons.csv"], a["coverage"])
    _csv(p["probability_distribution.csv"], a["probability"])
    _csv(p["trade_funnel.csv"], a["funnel"])
    _csv(p["feature_health.csv"], a["feature_health"])
    imp = a["importance"] + [{"side": s, "summary": "STABILITY", "jaccard_top20": v["jaccard_top20"],
                              "spearman_train_vs_validation": v["spearman_train_vs_validation"],
                              "validation_negative_importance_features": v["validation_negative_importance_features"],
                              "unstable": v["unstable"], "top20_train": " ".join(v["top20_train"]),
                              "top20_validation": " ".join(v["top20_validation"])} for s, v in a["stability"].items()]
    _csv(p["feature_importance_diagnostic.csv"], imp)
    _csv(p["calibration_diagnostic.csv"], a["calibration"])
    _csv(p["contract_selection_diagnostic.csv"], a["contracts"])
    p["diagnostic_report.md"].write_text(markdown(a), encoding="utf-8")
    summary = {k: v for k, v in a.items() if not k.startswith("_") and k != "funnel_trades"}
    summary["funnel_trades"] = {k: len(v) for k, v in a["funnel_trades"].items()}
    p["summary.json"].write_text(json.dumps(s6._jsonable(summary), indent=2, default=str), encoding="utf-8")
    return p


def _range(a: dict, side: str) -> str:
    r = next((x for x in a["probability"] if x["split"] == "TOTAL" and x["side"] == side
              and x["probability"] == "calibrated" and x["population"] == "ALL_ELIGIBLE"), None)
    if not r or r["min"] is None:
        return "N/A"
    return f"{r['min']:.3f} .. {r['max']:.3f} (median {r['median']:.3f}, p90 {r['p90']:.3f})"


def terminal_text(a: dict) -> str:
    c = a["dataset_counts"]
    tot = {s: {o: _count(c, "TOTAL", o, s) for o in (*sd.EXECUTED, "eligible_samples")} for s in sd.SIDES}
    reason, n, pct = main_exclusion(a["_coverage"])
    f = primary_funnel(a["funnel"])
    q = {s: _quality(a["quality"], "VALIDATION", s) for s in (*sd.SIDES, "COMBINED")}
    out = [HEADING, f"({DISCLAIMER})", "", "DATASET (TRAIN Apr-Jun + VALIDATION July):",
           f"CE samples: {tot['CE']['eligible_samples']}", f"PE samples: {tot['PE']['eligible_samples']}", "",
           "LABELS:"]
    for s in sd.SIDES:
        ex = sum(tot[s][o] for o in sd.EXECUTED)
        for o in (sd.WIN, sd.LOSS, sd.TIMEOUT):
            out.append(f"{s} {o}: {tot[s][o]} ({_fmt_pct(_pct(tot[s][o], ex))} of executed)")
    out += ["", "COVERAGE:", "", "Main exclusion reason:",
            f"{reason} - {category(reason)}" if reason else "none", "percentage:",
            f"{_fmt_pct(pct)} of in-window timestamps ({n})", "", "PREDICTION:", "",
            f"CE probability range: {_range(a, 'CE')}", f"PE probability range: {_range(a, 'PE')}", "",
            f"TRADE FUNNEL (frozen threshold {a['frozen']['threshold']:.2f}, TRAIN + VALIDATION timestamps):", "",
            f"eligible: {f['eligible']}", f"probability gate: {f['after_probability_gate']}",
            f"margin gate: {f['after_margin_gate']}", f"EV gate: {f['after_ev_gate']}",
            f"final trades: {f['final_trades']} (executed {f['executed']}, resolved {f['resolved']}; one-active-trade "
            f"rule removed {f['active_trade_removed']})", "", "MODEL QUALITY (VALIDATION July, frozen):", "",
            f"CE AUC: {_fmt(q['CE'].get('roc_auc'))} (SE {_fmt(q['CE'].get('roc_auc_se'))}, n {q['CE'].get('samples')})",
            f"PE AUC: {_fmt(q['PE'].get('roc_auc'))} (SE {_fmt(q['PE'].get('roc_auc_se'))}, n {q['PE'].get('samples')})",
            f"combined AUC: {_fmt(q['COMBINED'].get('roc_auc'))}", "", "ROOT CAUSE:", ""]
    for r in a["root_causes"][:3]:
        out.append(f"{r['rank']}. {r['cause']} - {r['removed']} removed at {r['stage']} "
                   f"({_fmt_pct(r['pct_of_stage_input'])} of {r['stage_input']})")
    d = a["diagnosis"]
    out += ["", "DIAGNOSIS:", "", d["label"] + (f" ({', '.join(d['components'])})" if d["label"] == "MIXED" else "")]
    out += [f"  {k}: {'YES' if a['flags'][k] else 'no'} - {d['evidence'][k]}" for k in LABELS]
    out += ["", "PRODUCTION CHANGED:", "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:",
            "0", "", "NO OCTOBER DATA USED." if not a["october_used"] and a["no_post_july_data"] else
            "WARNING: post-July data detected"]
    return "\n".join(out)


def markdown(a: dict) -> str:
    f = primary_funnel(a["funnel"])
    md = ["# S10.1 ML diagnostic", "", f"**{BANNER}**", "", f"> {DISCLAIMER}", "",
          f"Data: TRAIN {sd.SPLITS['TRAIN'][0]}..{sd.SPLITS['TRAIN'][1]} and VALIDATION "
          f"{sd.SPLITS['VALIDATION'][0]}..{sd.SPLITS['VALIDATION'][1]} only (latest market read: "
          f"{a['max_market_read']}). August, September and October were not loaded. Frozen S10 artifacts reused "
          f"(fingerprint match: {a['frozen']['fingerprint_match']}); nothing re-fitted or re-tuned.", "",
          f"Stop = {a['frozen']['stop']}; target = {a['frozen']['target']}; horizon {sd.HORIZON} option candles; "
          f"threshold {a['frozen']['threshold']:.2f}; margin {sm.MARGIN}; EV = 2P - (1 - P) > 0.", "",
          f"## Diagnosis: {a['diagnosis']['label']}", ""]
    md += [f"- **{k}**: {'YES' if a['flags'][k] else 'no'} - {a['diagnosis']['evidence'][k]}" for k in LABELS]
    md += ["", "## Hypotheses A-J", "", "| Code | Hypothesis | Status | Evidence |", "|---|---|---|---|"]
    md += [f"| {r['code']} | {r['hypothesis']} | {r['status']} | {r['evidence']} |" for r in a["letters"]]
    md += ["", "## Why are trades so few? (measured)", "", "| Rank | Cause | Stage | Removed | Of |", "|---|---|---|---|---|"]
    md += [f"| {r['rank']} | {r['cause']} | {r['stage']} | {r['removed']} | {r['stage_input']} |"
           for r in a["root_causes"]]
    md += ["", "## Trade funnel (frozen threshold, TRAIN + VALIDATION)", "",
           f"eligible {f['eligible']} -> probability gate {f['after_probability_gate']} -> margin gate "
           f"{f['after_margin_gate']} -> EV gate {f['after_ev_gate']} -> final trades {f['final_trades']} "
           f"(executed {f['executed']}, resolved {f['resolved']})", "",
           "Note: with a 2R target and 1R stop the EV gate (EV = 3P - 1 > 0) is implied by any P >= 0.34, so it "
           "cannot remove samples once P >= 0.55.", "", "## Labels (TRAIN + VALIDATION)", "",
           "| Split | Side | Executed | WIN | LOSS | TIMEOUT | AMBIG | WIN % exec | avg MFE R | avg MAE R | MFE>=2R |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["labels"]:
        md.append(f"| {r['split']} | {r['side']} | {r['executed']} | {r['target_before_stop']} | "
                  f"{r['stop_before_target']} | {r['timeout']} | {r['ambiguous']} | "
                  f"{_fmt_pct(r['target_pct_of_executed'])} | {_fmt(r['avg_mfe_r'], 2)} | {_fmt(r['avg_mae_r'], 2)} | "
                  f"{_fmt_pct(r['share_mfe_ge_target'])} |")
    md += ["", "## Model quality by class", "", "| Split | Side | n | win share | AUC | AUC SE | PR AUC | Brier | "
           "Brier skill | log loss | ECE | precision | recall |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["quality"]:
        md.append(f"| {r['split']} | {r['side']} | {r['samples']} | {_fmt_pct(r['class_balance_win_share'])} | "
                  f"{_fmt(r['roc_auc'])} | {_fmt(r['roc_auc_se'])} | {_fmt(r['pr_auc'])} | {_fmt(r['brier'])} | "
                  f"{_fmt(r['brier_skill'])} | {_fmt(r['log_loss'])} | {_fmt(r['ece'])} | {_fmt(r['precision'])} | "
                  f"{_fmt(r['recall'])} |")
    md += ["", "## Feature importance stability (frozen models)", ""]
    md += [f"- {s}: Jaccard(top20 TRAIN, VALIDATION) {_fmt(v['jaccard_top20'])}, Spearman "
           f"{_fmt(v['spearman_train_vs_validation'])}, unstable {v['unstable']}" for s, v in a["stability"].items()]
    md += ["", "Feature importance = predictive contribution in this frozen model, never causation.", "",
           "## Reproduction of the frozen S10 run", "", f"{a['reproduction']}", "",
           "## Not done (by design)", "", "No threshold, horizon, stop, target, feature, model or hyper-parameter "
           "change. No August / September / October data. Production S2, broker, OrderManager, RiskManager and live "
           "flags unchanged; live orders 0."]
    return "\n".join(md) + "\n"
