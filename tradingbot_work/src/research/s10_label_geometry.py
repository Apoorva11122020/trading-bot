"""S10.2 - label geometry / outcome feasibility study (PAPER | RESEARCH ONLY | NO ORDERS | NOT strategy selection).

Four PRE-DEFINED label definitions only (R = 1.5 x option ATR14, stop always 1R, entry = next option candle open):

  L1  1R stop / 1.5R target /  6 candles
  L2  1R stop / 2.0R target /  6 candles   (current S10 label)
  L3  1R stop / 1.5R target / 12 candles
  L4  1R stop / 2.0R target / 12 candles

Primary comparison uses the SAME samples for all four (signal candles whose 12-candle horizon ends by 15:15), so the
only difference is the label geometry. Learnability is descriptive: one fixed logistic regression per label and side
on the EXISTING frozen S10 feature pipeline (frozen preprocessor, frozen feature list), fit on TRAIN (Apr-Jun),
scored on VALIDATION (July). No shuffling, no tuning, no threshold, no new features. Aug / Sep / Oct never loaded.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.research import s10_ml_data as sd
from src.research import s10_ml_diagnostic as dx
from src.research import s10_ml_model as sm
from src.research import s6_clean_strategy_lab as s6

HEADING = "S10.2 LABEL GEOMETRY COMPLETE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | label feasibility study - NOT strategy selection, NOT live accuracy"
DISCLAIMER = ("ML model estimates the historical probability of a predefined option outcome from causal features. "
              "This study only describes how learnable each pre-defined label is; it selects no strategy.")
LABEL_DEFS = (("L1", 1.5, 6), ("L2", 2.0, 6), ("L3", 1.5, 12), ("L4", 2.0, 12))
STUDY_END = sd.SPLITS["VALIDATION"][1]
COMMON, OWN = "COMMON_WINDOW", "OWN_WINDOW"
COMMON_LAST = sd.last_signal(max(h for _, _, h in LABEL_DEFS))
# Pre-declared rules (fixed before running on real data).
MIN_TRAIN_POSITIVES, MIN_VAL_POSITIVES, MIN_WIN_SHARE = 100, 30, 0.10
MEANINGFUL_AUC = 0.55


def make_lr() -> LogisticRegression:
    """Fixed, untuned logistic regression (unweighted so Brier / log loss stay on the natural probability scale)."""
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")


def label_all(data: pd.DataFrame, caches: dict) -> pd.DataFrame:
    """Adds outcome / MFE / MAE columns for every label definition (future option candles used for labels only)."""
    out = data.copy()
    for name, target_r, horizon in LABEL_DEFS:
        oc, mfe, mae = [], [], []
        for r in out[["instrument_key", "signal_time", "atr"]].itertuples(index=False):
            s = caches["opt"].get(r.instrument_key)
            lab = sd.label_trade(s, r.signal_time, r.atr, horizon, target_r=target_r)
            f, a = dx.excursions(s, r.signal_time, r.atr, horizon)
            oc.append(lab["outcome"])
            mfe.append(f)
            mae.append(a)
        out[f"{name}_outcome"] = oc
        out[f"{name}_mfe"] = pd.Series(mfe, dtype=float).values
        out[f"{name}_mae"] = pd.Series(mae, dtype=float).values
        out[f"{name}_y"] = np.where(out[f"{name}_outcome"] == sd.WIN, 1.0,
                                    np.where(out[f"{name}_outcome"] == sd.LOSS, 0.0, np.nan))
    return out


def scope_frame(data: pd.DataFrame, scope: str, horizon: int) -> pd.DataFrame:
    last = COMMON_LAST if scope == COMMON else sd.last_signal(horizon)
    return data[pd.DatetimeIndex(data["signal_time"]).time <= last] if len(data) else data


def _pct(n, d):
    return float(n) / float(d) if d else None


def positive_status(train_pos: int, val_pos: int, win_share: float | None) -> str:
    if train_pos < MIN_TRAIN_POSITIVES or val_pos < MIN_VAL_POSITIVES:
        return "INSUFFICIENT"
    if win_share is None or win_share < MIN_WIN_SHARE:
        return "SPARSE"
    return "SUFFICIENT"


def distribution_rows(data: pd.DataFrame) -> list[dict]:
    out = []
    for scope in (COMMON, OWN):
        for name, target_r, horizon in LABEL_DEFS:
            f = scope_frame(data, scope, horizon)
            for split in ("TRAIN", "VALIDATION", "TOTAL"):
                g0 = f if split == "TOTAL" else f[f["split"] == split]
                for side in sd.SIDES:
                    g = g0[g0["side"] == side]
                    n = Counter(g[f"{name}_outcome"])
                    ex = sum(n[o] for o in sd.EXECUTED)
                    e = g[g[f"{name}_outcome"].isin(sd.EXECUTED)]
                    mfe, mae = e[f"{name}_mfe"].dropna(), e[f"{name}_mae"].dropna()
                    out.append({"scope": scope, "label": name, "stop_r": 1.0, "target_r": target_r,
                                "horizon": horizon, "split": split, "side": side, "eligible_samples": int(len(g)),
                                "executed": int(ex), "WIN": int(n[sd.WIN]), "LOSS": int(n[sd.LOSS]),
                                "TIMEOUT": int(n[sd.TIMEOUT]), "AMBIGUOUS": int(n[sd.AMBIGUOUS]),
                                "NO_ENTRY": int(n[sd.NO_ENTRY] + n[sd.INVALID_LEVELS]),
                                "win_pct": _pct(n[sd.WIN], ex), "loss_pct": _pct(n[sd.LOSS], ex),
                                "timeout_pct": _pct(n[sd.TIMEOUT], ex), "ambiguous_pct": _pct(n[sd.AMBIGUOUS], ex),
                                "win_rate_of_resolved": _pct(n[sd.WIN], n[sd.WIN] + n[sd.LOSS]),
                                "breakeven_win_rate": 1.0 / (1.0 + target_r),
                                "mfe_mean_r": float(mfe.mean()) if len(mfe) else None,
                                "mae_mean_r": float(mae.mean()) if len(mae) else None,
                                "mfe_median_r": float(mfe.median()) if len(mfe) else None,
                                "mae_median_r": float(mae.median()) if len(mae) else None})
    return out


def learnability_rows(data: pd.DataFrame, model: sm.S10Model) -> list[dict]:
    """One fixed LR per label x side on the frozen preprocessor; fit TRAIN binary rows, score July binary rows."""
    cols, pre = model.columns, model.preprocessor
    out = []
    for scope in (COMMON, OWN):
        for name, target_r, horizon in LABEL_DEFS:
            f = scope_frame(data, scope, horizon)
            for side in sd.SIDES:
                g = f[(f["side"] == side) & f[f"{name}_y"].notna()]
                tr, va = g[g["split"] == "TRAIN"], g[g["split"] == "VALIDATION"]
                ytr, yva = tr[f"{name}_y"].to_numpy(dtype=int), va[f"{name}_y"].to_numpy(dtype=int)
                base = {"scope": scope, "label": name, "target_r": target_r, "horizon": horizon, "side": side,
                        "train_binary": int(len(tr)), "train_positives": int(ytr.sum()),
                        "train_positive_rate": float(ytr.mean()) if len(ytr) else None,
                        "validation_binary": int(len(va)), "validation_positives": int(yva.sum()),
                        "validation_positive_rate": float(yva.mean()) if len(yva) else None}
                base["positive_class"] = positive_status(base["train_positives"], base["validation_positives"],
                                                         _win_share(f, name, side))
                if len(set(ytr)) < 2 or not len(yva):
                    out.append({**base, "status": "NOT_FITTED_INSUFFICIENT_DATA"})
                    continue
                est = make_lr().fit(pre.transform(tr[cols].to_numpy(dtype=float)), ytr)
                p = est.predict_proba(pre.transform(va[cols].to_numpy(dtype=float)))[:, 1]
                m = sm.model_metrics(yva, p, 0.5, base["train_positive_rate"])
                out.append({**base, "status": "FITTED", "roc_auc": m["roc_auc"],
                            "roc_auc_se": dx.auc_se(m["roc_auc"], m["wins"], m["losses"]),
                            "pr_auc": m["pr_auc"],
                            "pr_auc_lift": (m["pr_auc"] / base["validation_positive_rate"])
                            if m["pr_auc"] is not None and base["validation_positive_rate"] else None,
                            "brier": m["brier"], "climatology_brier": m["climatology_brier"],
                            "brier_skill": m["brier_skill"], "log_loss": m["log_loss"], "ece": m["ece"],
                            "mean_predicted": float(np.mean(p)),
                            "note": "descriptive learnability only - fixed LR, frozen S10 preprocessor, no tuning"})
    return out


def _win_share(f: pd.DataFrame, name: str, side: str) -> float | None:
    g = f[(f["side"] == side) & f[f"{name}_outcome"].isin(sd.EXECUTED)]
    return float((g[f"{name}_outcome"] == sd.WIN).mean()) if len(g) else None


def _get(rows: list[dict], **kw) -> dict:
    return next((r for r in rows if all(r.get(k) == v for k, v in kw.items())), {})


def best_label(dist: list[dict], learn: list[dict]) -> dict[str, Any]:
    """Pre-declared descriptive ranking (COMMON_WINDOW): labels whose positive class is SUFFICIENT for both sides
    first; then highest mean validation AUC (CE, PE); ties -> higher mean PR-AUC lift, then lower timeout rate."""
    rows = []
    for name, target_r, horizon in LABEL_DEFS:
        lr = {s: _get(learn, scope=COMMON, label=name, side=s) for s in sd.SIDES}
        dr = {s: _get(dist, scope=COMMON, label=name, split="TOTAL", side=s) for s in sd.SIDES}
        aucs = [lr[s].get("roc_auc") for s in sd.SIDES]
        lifts = [lr[s].get("pr_auc_lift") for s in sd.SIDES]
        prs = [lr[s].get("pr_auc") for s in sd.SIDES]
        mean = lambda v: float(np.mean(v)) if all(x is not None for x in v) else None  # noqa: E731
        rows.append({"label": name, "target_r": target_r, "horizon": horizon,
                     "sufficient": all(lr[s].get("positive_class") == "SUFFICIENT" for s in sd.SIDES),
                     "mean_auc": mean(aucs), "mean_pr_auc": mean(prs), "mean_pr_lift": mean(lifts),
                     "win_pct": mean([dr[s].get("win_pct") for s in sd.SIDES]),
                     "timeout_pct": mean([dr[s].get("timeout_pct") for s in sd.SIDES]),
                     "val_positive_rate": mean([lr[s].get("validation_positive_rate") for s in sd.SIDES]),
                     "positive_class": {s: lr[s].get("positive_class") for s in sd.SIDES}})
    key = lambda r: (r["sufficient"], r["mean_auc"] if r["mean_auc"] is not None else -1,  # noqa: E731
                     r["mean_pr_lift"] if r["mean_pr_lift"] is not None else -1,
                     -(r["timeout_pct"] if r["timeout_pct"] is not None else 1))
    ranked = sorted(rows, key=key, reverse=True)
    best = ranked[0]
    learnable = bool(best["mean_auc"] is not None and best["mean_auc"] >= MEANINGFUL_AUC and best["sufficient"])
    return {"name": best["label"], "row": best, "ranking": ranked,
            "meaningfully_learnable": learnable,
            "verdict": "LEARNABLE_FOR_FURTHER_RESEARCH" if learnable else "NO_LABEL_MEANINGFULLY_LEARNABLE",
            "rule": "COMMON_WINDOW; SUFFICIENT positives both sides first, then mean July AUC, then PR-AUC lift, then "
                    f"lower timeout; 'meaningfully learnable' requires mean AUC >= {MEANINGFUL_AUC}",
            "caveat": "chosen on July VALIDATION only for further research; NOT a strategy, NOT tested on OOS"}


def run_study(*, market: sd.MarketData, model: sm.S10Model, progress: Callable[[str], None] | None = None
              ) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    flags = dx.flags_from_columns(model.columns)
    days = sd.days_in(market.sessions, sd.SPLITS["TRAIN"][0], STUDY_END)
    caches: dict = {"fut": {}, "opt": {}}
    say(f"building the S10 sample set for TRAIN + VALIDATION ({len(days)} sessions; Aug..Oct never loaded) ...")
    built = sd.build_samples(market, days, volume=flags["volume"], oi=flags["oi"], caches=caches)
    data = built.data.copy()
    data["split"] = [sd.split_of_day(date.fromisoformat(d)) for d in data["session_date"]]
    say("labelling L1..L4 (same entry, same 1R stop) ...")
    data = label_all(data, caches)
    say("label distributions ...")
    dist = distribution_rows(data)
    say("descriptive learnability (fixed LR on the frozen S10 preprocessor) ...")
    learn = learnability_rows(data, model)
    best = best_label(dist, learn)
    mx = market.reads["max_ts"]
    no_post = bool((mx is None or mx.date() <= STUDY_END)
                   and (data.empty or data["session_date"].max() <= STUDY_END.isoformat())
                   and market.spot["timestamp"].dt.date.max() <= STUDY_END)
    return {"heading": HEADING, "banner": BANNER, "disclaimer": DISCLAIMER,
            "labels": [{"name": n, "stop_r": 1.0, "target_r": t, "horizon": h} for n, t, h in LABEL_DEFS],
            "r_definition": f"R = {sd.STOP_ATR} x option ATR14; entry = next option candle open",
            "common_window_last_signal": COMMON_LAST.isoformat(timespec="minutes"),
            "period": [days[0].isoformat(), days[-1].isoformat()] if days else None,
            "max_market_read": mx.isoformat() if mx is not None else None, "no_post_july_data": no_post,
            "distribution": dist, "learnability": learn, "best": best, "production_unchanged": None,
            "live_orders": 0, "october_used": False, "_data": data}


# ------------------------------------------------------------------ reports
REPORT_FILES = ("label_distribution.csv", "learnability.csv", "label_ranking.csv", "label_geometry_report.md",
                "summary.json")


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    s6._csv(p["label_distribution.csv"], a["distribution"])
    s6._csv(p["learnability.csv"], a["learnability"])
    s6._csv(p["label_ranking.csv"], [{**{k: v for k, v in r.items() if k != "positive_class"},
                                      "positive_class_CE": r["positive_class"]["CE"],
                                      "positive_class_PE": r["positive_class"]["PE"], "rank": i}
                                     for i, r in enumerate(a["best"]["ranking"], 1)])
    p["label_geometry_report.md"].write_text(markdown(a), encoding="utf-8")
    summary = {k: v for k, v in a.items() if not k.startswith("_")}
    p["summary.json"].write_text(json.dumps(s6._jsonable(summary), indent=2, default=str), encoding="utf-8")
    return p


def _f(v, nd: int = 4) -> str:
    return "N/A" if v is None else f"{v:.{nd}f}"


def _p(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def side_line(a: dict, name: str, side: str, scope: str = COMMON) -> str:
    d = _get(a["distribution"], scope=scope, label=name, split="TOTAL", side=side)
    l = _get(a["learnability"], scope=scope, label=name, side=side)
    return (f"eligible {d.get('eligible_samples')} | WIN {d.get('WIN')} ({_p(d.get('win_pct'))}) | LOSS {d.get('LOSS')} "
            f"({_p(d.get('loss_pct'))}) | TIMEOUT {d.get('TIMEOUT')} ({_p(d.get('timeout_pct'))}) | AMBIGUOUS "
            f"{d.get('AMBIGUOUS')} ({_p(d.get('ambiguous_pct'))}) | MFE {_f(d.get('mfe_mean_r'), 2)}R (median "
            f"{_f(d.get('mfe_median_r'), 2)}R) | MAE {_f(d.get('mae_mean_r'), 2)}R (median {_f(d.get('mae_median_r'), 2)}R)"
            f" | positives {l.get('positive_class')} | July AUC {_f(l.get('roc_auc'))} | PR AUC {_f(l.get('pr_auc'))} | "
            f"Brier {_f(l.get('brier'))} | Log Loss {_f(l.get('log_loss'))} | July positive rate "
            f"{_p(l.get('validation_positive_rate'))}")


def terminal_text(a: dict) -> str:
    out = [HEADING, f"({DISCLAIMER})", "",
           f"Samples: identical for all labels (signal candles <= {a['common_window_last_signal']}, TRAIN Apr-Jun + "
           f"VALIDATION July). {a['r_definition']}. WIN/LOSS/TIMEOUT % are of executed samples.", ""]
    for name, target_r, horizon in LABEL_DEFS:
        out += [f"{name}: (1R SL / {target_r}R TARGET / {horizon} candles)"]
        out += [f"{s}: {side_line(a, name, s)}" for s in sd.SIDES]
        out.append("")
    b = a["best"]
    r = b["row"]
    out += ["BEST LABEL FOR RESEARCH:", f"name: {b['name']} (1R SL / {r['target_r']}R TARGET / {r['horizon']} candles)"
            f" - {b['verdict']}", "", "Reason:",
            f"- positive sample rate: WIN {_p(r['win_pct'])} of executed; July WIN share of WIN+LOSS "
            f"{_p(r['val_positive_rate'])}; positive class CE {r['positive_class']['CE']} / PE "
            f"{r['positive_class']['PE']}",
            f"- timeout rate: {_p(r['timeout_pct'])}",
            f"- validation AUC: mean CE/PE {_f(r['mean_auc'])}",
            f"- validation PR AUC: mean CE/PE {_f(r['mean_pr_auc'])} (lift over positive rate {_f(r['mean_pr_lift'], 2)}x)",
            f"- rule: {b['rule']}", "", "IMPORTANT:", "This is NOT strategy selection.", "This is NOT live accuracy.",
            "No threshold optimisation; no new model or features selected; S10 production / paper execution unchanged.",
            "", "PRODUCTION CHANGED:", "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:",
            "0", "", "NO OCTOBER DATA USED." if a["no_post_july_data"] and not a["october_used"] else
            "WARNING: post-July data detected"]
    return "\n".join(out)


def markdown(a: dict) -> str:
    md = ["# S10.2 Label geometry / outcome feasibility", "", f"**{BANNER}**", "", f"> {DISCLAIMER}", "",
          f"TRAIN Apr-Jun + VALIDATION July only (latest market read {a['max_market_read']}). {a['r_definition']}. "
          f"Primary comparison: identical samples for all labels (signal candles <= {a['common_window_last_signal']}).",
          "", "| Label | Side | Eligible | WIN % | LOSS % | TIMEOUT % | AMBIG % | MFE mean/median R | MAE mean/median R |"
          " Positives | July AUC (SE) | PR AUC (lift) | Brier (skill) | Log loss |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, _, _ in LABEL_DEFS:
        for s in sd.SIDES:
            d = _get(a["distribution"], scope=COMMON, label=name, split="TOTAL", side=s)
            l = _get(a["learnability"], scope=COMMON, label=name, side=s)
            md.append(f"| {name} | {s} | {d.get('eligible_samples')} | {_p(d.get('win_pct'))} | {_p(d.get('loss_pct'))} | "
                      f"{_p(d.get('timeout_pct'))} | {_p(d.get('ambiguous_pct'))} | {_f(d.get('mfe_mean_r'), 2)} / "
                      f"{_f(d.get('mfe_median_r'), 2)} | {_f(d.get('mae_mean_r'), 2)} / {_f(d.get('mae_median_r'), 2)} | "
                      f"{l.get('positive_class')} | {_f(l.get('roc_auc'))} ({_f(l.get('roc_auc_se'))}) | "
                      f"{_f(l.get('pr_auc'))} ({_f(l.get('pr_auc_lift'), 2)}x) | {_f(l.get('brier'))} "
                      f"({_f(l.get('brier_skill'))}) | {_f(l.get('log_loss'))} |")
    b = a["best"]
    md += ["", f"## Best label for further research: {b['name']} ({b['verdict']})", "", f"Rule: {b['rule']}.", "",
           f"Caveat: {b['caveat']}.", "", "Positive class rule: SUFFICIENT needs >= "
           f"{MIN_TRAIN_POSITIVES} TRAIN and >= {MIN_VAL_POSITIVES} July positives and WIN >= {MIN_WIN_SHARE:.0%} "
           "of executed samples; otherwise SPARSE / INSUFFICIENT.", "",
           "OWN_WINDOW rows (each label on its own full signal window) are in the CSVs for reference.", "",
           "This is NOT strategy selection and NOT live accuracy. No threshold optimisation. No October access. "
           "S10 production / paper execution unchanged; live orders 0."]
    return "\n".join(md) + "\n"
