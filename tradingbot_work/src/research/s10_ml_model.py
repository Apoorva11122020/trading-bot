"""S10_ML_PROBABILITY - model layer (PAPER | RESEARCH ONLY | NO ORDERS).

Small fixed models only (no deep learning, no hyper-parameter search, no random splits):

  LOGISTIC_REGRESSION      median imputer + StandardScaler (fit on the model-fit fold only), L2, balanced weights
  HIST_GRADIENT_BOOSTING   max_depth 3, learning_rate 0.05, max_iter 200, min_samples_leaf 50, balanced weights
  LIGHTGBM                 same fixed shape, only if already installed (never installed automatically)

Each fit is chronological: the last 20% of the training days form the calibration fold (Platt / sigmoid), the rest
the model-fit fold. Every training sample whose label window [t, entry + H candles] overlaps the next block (or starts
within the 6-bar embargo after it) is purged. The calibrated probability is P(TARGET before STOP).
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import pickle
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.research import s10_ml_data as sd

LR, HGB, LGBM = "LOGISTIC_REGRESSION", "HIST_GRADIENT_BOOSTING", "LIGHTGBM"
THRESHOLDS = (0.55, 0.60, 0.65)
MARGIN = 0.05
EMBARGO_BARS = 6
CALIBRATION_SHARE = 0.20
SEED = 20261006

# Pre-declared sample / selection rules (fixed before any model was fitted).
MIN_FIT_SAMPLES = 50
MIN_CALIBRATION_SAMPLES = 30
MIN_INNER_VAL_SAMPLES = 100
MAX_INNER_ECE = 0.15
MIN_THRESHOLD_RESOLVED = 20
N_BINS = 10
FITTED, INSUFFICIENT = "FITTED", "INSUFFICIENT_DATA"
PLATT, UNCALIBRATED = "PLATT_SIGMOID", "UNCALIBRATED_INSUFFICIENT_DATA"


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def families() -> list[str]:
    return [LR, HGB, *([LGBM] if lightgbm_available() else [])]


def make_estimator(family: str):
    if family == LR:
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
    if family == HGB:
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
                                              early_stopping=False, random_state=SEED)
    if family == LGBM:
        import lightgbm
        return lightgbm.LGBMClassifier(max_depth=3, learning_rate=0.05, n_estimators=200, min_child_samples=50,
                                       class_weight="balanced", random_state=SEED, deterministic=True, n_jobs=1,
                                       verbose=-1)
    raise ValueError(f"unknown model family {family}")


def fit_estimator(family: str, X: np.ndarray, y: np.ndarray):
    est = make_estimator(family)
    if family == HGB:
        est.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        est.fit(X, y)
    return est


def make_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler())])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


# ------------------------------------------------------------------ purged chronological splits
def _ns(values) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(values))).asi8


def purge_mask(train: pd.DataFrame, test: pd.DataFrame, grid: np.ndarray, embargo_bars: int = EMBARGO_BARS,
               end_col: str = "event_end") -> tuple[np.ndarray, dict[str, int]]:
    """Keep-mask over ``train``: drop samples whose [signal_time, event_end] overlaps the test block
    [first test signal, last test event end] or that start within ``embargo_bars`` bars after the block."""
    n = len(train)
    if n == 0 or test.empty:
        return np.ones(n, dtype=bool), {"train": n, "purged": 0, "embargoed": 0, "kept": n}
    s, e = _ns(train["signal_time"]), _ns(train[end_col])
    t0, t1 = int(_ns(test["signal_time"]).min()), int(_ns(test[end_col]).max())
    i = int(np.searchsorted(grid, t1, side="left")) if len(grid) else 0
    emb_end = int(grid[min(i + embargo_bars, len(grid) - 1)]) if len(grid) else t1
    overlap = (s <= t1) & (e >= t0)
    embargo = ~overlap & (s > t1) & (s <= emb_end)
    keep = ~(overlap | embargo)
    return keep, {"train": n, "purged": int(overlap.sum()), "embargoed": int(embargo.sum()), "kept": int(keep.sum())}


def purge_violations(train: pd.DataFrame, test: pd.DataFrame, grid: np.ndarray,
                     embargo_bars: int = EMBARGO_BARS) -> int:
    keep, _ = purge_mask(train, test, grid, embargo_bars)
    return int((~keep).sum())


# ------------------------------------------------------------------ fitting
@dataclass
class SideModel:
    family: str
    status: str
    estimator: Any = None
    calibrator: Any = None
    calibration: str = UNCALIBRATED
    fit_ids: list[int] = field(default_factory=list)
    calib_ids: list[int] = field(default_factory=list)
    base_rate: float | None = None


@dataclass
class S10Model:
    columns: list[str]
    preprocessor: Pipeline
    sides: dict[str, SideModel]
    pre_fit_ids: list[int]
    fit_days: list[str]
    calib_days: list[str]
    purge: dict[str, dict]

    def predict_X(self, side: str, X: np.ndarray) -> np.ndarray:
        sm = self.sides[side]
        if sm.status != FITTED or sm.calibrator is None or not len(X):
            return np.full(len(X), np.nan)
        raw = sm.estimator.predict_proba(self.preprocessor.transform(X))[:, 1]
        return sm.calibrator.predict_proba(_logit(raw))[:, 1]

    def predict(self, side: str, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_X(side, frame[self.columns].to_numpy(dtype=float))


def chrono_calibration_split(days: Sequence[str]) -> tuple[list[str], list[str]]:
    days = sorted(days)
    n_cal = max(1, int(round(len(days) * CALIBRATION_SHARE))) if len(days) >= 2 else 0
    return (days[:-n_cal], days[-n_cal:]) if n_cal else (days, [])


def fit_model(data: pd.DataFrame, columns: Sequence[str], train_days: Sequence[str], fams: dict[str, str],
              grid: np.ndarray, test_days: Sequence[str] = ()) -> S10Model:
    """Everything learned from data (imputer, scaler, estimator, calibrator) is fit on ``train_days`` only."""
    tr = data[data["session_date"].isin(sorted(set(train_days)))]
    fit_days, calib_days = chrono_calibration_split(sorted(set(tr["session_date"])))
    test = data[data["session_date"].isin(sorted(set(test_days)))]
    fit = tr[tr["session_date"].isin(fit_days)]
    cal = tr[tr["session_date"].isin(calib_days)]
    k1, p1 = purge_mask(fit, cal, grid)
    k2, p2 = purge_mask(fit, test, grid)
    k3, p3 = purge_mask(cal, test, grid)
    fit, cal = fit[k1 & k2], cal[k3]
    cols = list(columns)
    pre = make_preprocessor()
    if len(fit):
        pre.fit(fit[cols].to_numpy(dtype=float))
    sides: dict[str, SideModel] = {}
    for side in sd.SIDES:
        fam = fams[side]
        fb = fit[(fit["side"] == side) & fit["outcome"].isin(sd.BINARY)]
        cb = cal[(cal["side"] == side) & cal["outcome"].isin(sd.BINARY)]
        y = fb["y"].to_numpy(dtype=int)
        if len(fb) < MIN_FIT_SAMPLES or len(set(y)) < 2:
            sides[side] = SideModel(fam, INSUFFICIENT, fit_ids=list(fb.index))
            continue
        est = fit_estimator(fam, pre.transform(fb[cols].to_numpy(dtype=float)), y)
        sm = SideModel(fam, FITTED, est, fit_ids=list(fb.index), calib_ids=list(cb.index),
                       base_rate=float(y.mean()))
        yc = cb["y"].to_numpy(dtype=int)
        if len(cb) >= MIN_CALIBRATION_SAMPLES and len(set(yc)) == 2:
            raw = est.predict_proba(pre.transform(cb[cols].to_numpy(dtype=float)))[:, 1]
            sm.calibrator = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(raw), yc)
            sm.calibration = PLATT
        sides[side] = sm
    return S10Model(cols, pre, sides, list(fit.index), fit_days, calib_days,
                    {"fit_vs_calibration": p1, "fit_vs_test": p2, "calibration_vs_test": p3})


def model_bytes(m: S10Model, threshold: float | None = None) -> bytes:
    payload = (m.columns, threshold, {s: (v.family, v.status, v.calibration) for s, v in m.sides.items()},
               m.preprocessor, {s: (v.estimator, v.calibrator) for s, v in m.sides.items()})
    return pickle.dumps(payload, protocol=4)


def fingerprint(m: S10Model, threshold: float | None = None) -> str:
    return hashlib.sha256(model_bytes(m, threshold)).hexdigest()


# ------------------------------------------------------------------ evaluation of probabilities
def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple[float | None, list[dict]]:
    if not len(p):
        return None, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        m = b == i
        if not m.any():
            rows.append({"bin": i, "lo": edges[i], "hi": edges[i + 1], "count": 0, "mean_p": None, "win_rate": None})
            continue
        mp, wr = float(p[m].mean()), float(y[m].mean())
        ece += abs(mp - wr) * m.sum() / len(p)
        rows.append({"bin": i, "lo": round(edges[i], 2), "hi": round(edges[i + 1], 2), "count": int(m.sum()),
                     "mean_p": round(mp, 4), "win_rate": round(wr, 4)})
    return float(ece), rows


def model_metrics(y: np.ndarray, p: np.ndarray, threshold: float, base_rate: float | None = None) -> dict[str, Any]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    n, pos = len(y), int(y.sum()) if len(y) else 0
    out: dict[str, Any] = {"samples": n, "wins": pos, "losses": n - pos, "win_share": pos / n if n else None,
                           "roc_auc": None, "pr_auc": None, "brier": None, "log_loss": None, "ece": None,
                           "climatology_brier": None, "brier_skill": None, "accuracy_at_threshold": None,
                           "precision": None, "recall": None, "predicted_positive": 0, "calibration": []}
    if not n:
        return out
    out["brier"] = float(brier_score_loss(y, p))
    out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    if 0 < pos < n:
        out["roc_auc"] = float(roc_auc_score(y, p))
    if pos > 0:
        out["pr_auc"] = float(average_precision_score(y, p))
    out["ece"], out["calibration"] = calibration_bins(y, p)
    if base_rate is not None:
        clim = float(np.mean((y - base_rate) ** 2))
        out["climatology_brier"] = clim
        out["brier_skill"] = 1.0 - out["brier"] / clim if clim > 0 else None
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    out.update(accuracy_at_threshold=float((pred == (y == 1)).mean()), predicted_positive=int(pred.sum()),
               precision=tp / int(pred.sum()) if pred.any() else None, recall=tp / pos if pos else None)
    return out


def side_predictions(data: pd.DataFrame, models: dict[str, S10Model]) -> dict[str, pd.DataFrame]:
    """Per side: the binary-labelled rows (WIN / LOSS) with the calibrated probability of that side's model."""
    out = {}
    for side in sd.SIDES:
        b = data[(data["side"] == side) & data["outcome"].isin(sd.BINARY)]
        out[side] = b.assign(p=models[side].predict(side, b) if len(b) else np.array([]))
    return out


def evaluate_models(data: pd.DataFrame, models: dict[str, S10Model], threshold: float) -> dict[str, dict]:
    preds = side_predictions(data, models)
    res = {side: model_metrics(preds[side]["y"].to_numpy(), preds[side]["p"].to_numpy(), threshold,
                               models[side].sides[side].base_rate) for side in sd.SIDES}
    both = pd.concat([preds[s] for s in sd.SIDES])
    rates = [models[s].sides[s].base_rate for s in sd.SIDES if models[s].sides[s].base_rate is not None]
    res["COMBINED"] = model_metrics(both["y"].to_numpy(), both["p"].to_numpy(), threshold,
                                    float(np.mean(rates)) if rates else None)
    return res


# ------------------------------------------------------------------ decision policy + trade simulation
def decide(p_ce: float, p_pe: float, threshold: float, margin: float = MARGIN,
           target_r: float = sd.TARGET_R) -> tuple[str | None, str | None, float | None, float | None]:
    if not (np.isfinite(p_ce) and np.isfinite(p_pe)):
        return None, "PROBABILITY_UNAVAILABLE", None, None
    best, other = ("CE", "PE") if p_ce >= p_pe else ("PE", "CE")
    pb, po = (p_ce, p_pe) if best == "CE" else (p_pe, p_ce)
    if pb < threshold:
        return None, "BELOW_THRESHOLD", pb, None
    if pb - po < margin:
        return None, "MARGIN_TOO_SMALL", pb, None
    ev = target_r * pb - 1.0 * (1.0 - pb)
    if ev <= 0:
        return None, "EV_NOT_POSITIVE", pb, ev
    return best, None, pb, ev


LABEL_FIELDS = ("instrument_key", "contract", "strike", "expiry", "entry_time", "entry", "stop_loss", "target",
                "risk", "outcome", "exit_time", "exit_price", "exit_reason", "r_multiple", "pnl", "holding_minutes",
                "outcome_h12", "r_h12", "pnl_h12", "exit_time_h12", "holding_h12", "exit_reason_h12",
                "exit_price_h12")


def wide_frame(data: pd.DataFrame, models: dict[str, S10Model]) -> pd.DataFrame:
    """One row per eligible timestamp with P_WIN_CE, P_WIN_PE and both sides' (future) trade outcomes."""
    parts = []
    for side in sd.SIDES:
        sub = data[data["side"] == side].set_index("signal_time")
        p = models[side].predict(side, sub) if len(sub) else np.array([])
        part = sub[list(LABEL_FIELDS)].add_suffix(f"_{side}")
        part[f"p_{side}"] = p
        if side == sd.SIDES[0]:
            part["session_date"] = sub["session_date"]
        parts.append(part)
    return parts[0].join(parts[1], how="inner").sort_index()


def simulate(w: pd.DataFrame, threshold: float, *, horizon: int = sd.HORIZON, margin: float = MARGIN,
             side_picker: Callable[[pd.Timestamp, str], str] | None = None,
             target_r: float = sd.TARGET_R) -> tuple[list[dict], Counter, int]:
    """Chronological, one active S10 trade at a time; entry at the next option candle open."""
    sfx = "" if horizon == sd.HORIZON else "_h12"
    trades: list[dict] = []
    dec: Counter = Counter()
    active_until = None
    for ts, row in zip(w.index, w.to_dict("records")):
        side, why, pb, ev = decide(row["p_CE"], row["p_PE"], threshold, margin, target_r)
        if side is None:
            dec[why] += 1
            continue
        model_side = side
        if side_picker is not None:
            side = side_picker(ts, side)
        entry_ts = ts + sd.STEP
        if active_until is not None and entry_ts < active_until:
            dec["ACTIVE_TRADE_OPEN"] += 1
            continue
        outcome = row[f"outcome{sfx}_{side}"]
        if outcome is None or (isinstance(outcome, float) and math.isnan(outcome)):
            dec["HORIZON_NOT_DEFINED"] += 1
            continue
        rec = {"signal_time": ts, "session_date": row["session_date"], "side": side,
               "direction": f"BUY {side}", "model_side": model_side, "p_win": pb,
               "p_other": row[f"p_{sd.OTHER[model_side]}"], "ev_r": ev, "threshold": threshold, "horizon": horizon,
               **{k: row[f"{k}_{side}"] for k in ("instrument_key", "contract", "strike", "expiry", "entry_time",
                                                  "entry", "stop_loss", "target", "risk")},
               "outcome": outcome, "exit_time": row[f"exit_time{sfx}_{side}"],
               "exit_reason": row[f"exit_reason{sfx}_{side}"], "exit_price": row[f"exit_price{sfx}_{side}"],
               "r_multiple": row[f"{'r_multiple' if not sfx else 'r_h12'}_{side}"],
               "pnl": row[f"pnl{sfx}_{side}"],
               "holding_minutes": row[f"{'holding_minutes' if not sfx else 'holding_h12'}_{side}"]}
        trades.append(rec)
        dec[outcome] += 1
        if outcome in sd.EXECUTED:
            active_until = rec["exit_time"]
    return trades, dec, len(w)


def sample_label(resolved: int) -> str:
    if resolved < 30:
        return "INSUFFICIENT_DATA"
    if resolved < 50:
        return "LOW_SAMPLE"
    if resolved < 100:
        return "MODERATE_SAMPLE"
    return "STRONGER_SAMPLE"


def _pf(gp: float, gl: float) -> float | None:
    return gp / gl if gl > 0 else (math.inf if gp > 0 else None)


def trade_metrics(trades: Sequence[dict], opportunities: int) -> dict[str, Any]:
    trades = sorted(trades, key=lambda t: t["signal_time"])
    ex = [t for t in trades if t["outcome"] in sd.EXECUTED]
    n = Counter(t["outcome"] for t in trades)
    wins, losses = n[sd.WIN], n[sd.LOSS]
    pnl = [float(t["pnl"]) for t in ex]
    rs = [float(t["r_multiple"]) for t in ex]
    gp, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    eq = peak = dd = eq_r = peak_r = dd_r = 0.0
    streak = worst = 0
    for p, r in zip(pnl, rs):
        eq += p
        peak, dd = max(peak, eq), max(dd, peak - eq)
        eq_r += r
        peak_r, dd_r = max(peak_r, eq_r), max(dd_r, peak_r - eq_r)
        streak = streak + 1 if r < 0 else 0
        worst = max(worst, streak)
    resolved = wins + losses
    hold = [float(t["holding_minutes"]) for t in ex if t.get("holding_minutes") is not None]
    return {"opportunities": int(opportunities), "triggered": len(trades), "trades": len(ex), "wins": wins,
            "losses": losses, "ambiguous": n[sd.AMBIGUOUS], "timeout": n[sd.TIMEOUT],
            "unresolved_no_entry": n[sd.NO_ENTRY] + n[sd.INVALID_LEVELS], "resolved": resolved,
            "accuracy": wins / resolved if resolved else None,
            "accuracy_text": f"{wins} wins / {resolved} resolved (WIN + LOSS; TIMEOUT and AMBIGUOUS excluded)",
            "coverage": len(ex) / opportunities if opportunities else None,
            "profit_factor": _pf(gp, gl) if ex else None, "gross_profit": round(gp, 2), "gross_loss": round(gl, 2),
            "net_pnl": round(gp - gl, 2), "average_r": round(sum(rs) / len(rs), 4) if rs else None,
            "expectancy_r": round(sum(rs) / len(rs), 4) if rs else None, "total_r": round(sum(rs), 4),
            "max_drawdown": round(dd, 2), "max_drawdown_r": round(dd_r, 4), "max_losing_streak": worst,
            "average_holding_minutes": round(sum(hold) / len(hold), 1) if hold else None,
            "sample": sample_label(resolved)}


def scope_metrics(trades: Sequence[dict], opportunities: int) -> dict[str, dict]:
    out = {s: trade_metrics([t for t in trades if t["side"] == s], opportunities) for s in sd.SIDES}
    out["COMBINED"] = trade_metrics(trades, opportunities)
    return out


# ------------------------------------------------------------------ pre-declared selection rules (TRAIN only)
def choose_family(rows: Sequence[dict]) -> tuple[str, str]:
    """rows: one per family with inner-validation metrics for ONE side. (1) enough samples, (2) ECE <= 0.15,
    (3) lowest Brier, (4) lowest log loss, (5) simpler family first. Returns (family, reason)."""
    order = {LR: 0, HGB: 1, LGBM: 2}
    usable = [r for r in rows if r["status"] == FITTED and r["brier"] is not None]
    if not usable:
        return LR, "NO_FAMILY_FITTED_INNER (defaulted to the simplest family)"
    big = [r for r in usable if r["samples"] >= MIN_INNER_VAL_SAMPLES] or usable
    cal = [r for r in big if r["ece"] is not None and r["ece"] <= MAX_INNER_ECE] or big
    best = min(cal, key=lambda r: (round(r["brier"], 6), round(r["log_loss"], 6), order[r["family"]]))
    why = "lowest inner-validation Brier"
    if len(big) < len(usable):
        why += " (families below the minimum sample excluded)"
    if len(cal) < len(big):
        why += f" (families with ECE > {MAX_INNER_ECE} excluded)"
    if best["samples"] < MIN_INNER_VAL_SAMPLES:
        why += " [LOW_SAMPLE]"
    return best["family"], why


def choose_threshold(rows: Sequence[dict]) -> tuple[float, str]:
    """rows: one per threshold with inner-validation COMBINED trade metrics. Highest expectancy among thresholds with
    >= MIN_THRESHOLD_RESOLVED resolved trades (ties -> lower threshold); otherwise the lowest threshold."""
    ok = [r for r in rows if r["resolved"] >= MIN_THRESHOLD_RESOLVED and r["expectancy_r"] is not None]
    if not ok:
        return min(THRESHOLDS), "THRESHOLD_SELECTION_LOW_SAMPLE (no threshold reached the minimum resolved trades)"
    best = max(ok, key=lambda r: (r["expectancy_r"], -r["threshold"]))
    return best["threshold"], "highest inner-validation expectancy (R) with the minimum resolved trades"


# ------------------------------------------------------------------ feature importance (frozen model only)
def coefficient_importance(m: S10Model, side: str) -> dict[str, float]:
    sm = m.sides[side]
    if sm.status != FITTED or not hasattr(sm.estimator, "coef_"):
        return {}
    return {c: float(abs(v)) for c, v in zip(m.columns, sm.estimator.coef_[0])}


def permutation_importance(m: S10Model, side: str, frame: pd.DataFrame, seed: int = SEED,
                           repeats: int = 3) -> dict[str, float]:
    """Mean Brier increase when one feature column is permuted (evaluation only; the model is not changed)."""
    b = frame[(frame["side"] == side) & frame["outcome"].isin(sd.BINARY)]
    if len(b) < 20 or m.sides[side].status != FITTED or m.sides[side].calibrator is None:
        return {}
    X = b[m.columns].to_numpy(dtype=float)
    y = b["y"].to_numpy(dtype=int)
    base = brier_score_loss(y, m.predict_X(side, X))
    rng = np.random.default_rng(seed)
    out = {}
    for j, col in enumerate(m.columns):
        inc = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]
            inc.append(brier_score_loss(y, m.predict_X(side, Xp)) - base)
        out[col] = float(np.mean(inc))
    return out


def top_k(imp: dict[str, float], k: int = 10) -> list[str]:
    return [c for c, _ in sorted(imp.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float | None:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else None


def ids_within(data: pd.DataFrame, ids: Sequence[int], lo: date, hi: date) -> bool:
    if not len(ids):
        return True
    d = data.loc[list(ids), "session_date"]
    return bool(((d >= lo.isoformat()) & (d <= hi.isoformat())).all())
