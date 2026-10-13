"""S10.3 - L1 model ranking / precision diagnostic (PAPER | RESEARCH ONLY | NO ORDERS | DIAGNOSTIC ONLY).

Question: does the frozen L1 learnability model (S10.2: fixed LogisticRegression(C=1.0) on the frozen S10 preprocessor
and feature list, fit on TRAIN binary rows, COMMON_WINDOW samples) rank the better option outcomes above the weaker
ones? The same deterministic model is re-created exactly as in S10.2 (no new features, no tuning, no threshold) and
its July AUC is checked against the S10.2 summary. TRAIN Apr-Jun + VALIDATION July only; Aug / Sep / Oct never loaded.

L1 = 1R stop / 1.5R target / 6 completed option candles (R = 1.5 x option ATR14, entry = next option candle open).
Ranked population = executed samples (WIN / LOSS / TIMEOUT / AMBIGUOUS); NO_ENTRY / INVALID_LEVELS rows never
traded and are counted separately. Win rate = WIN / (WIN + LOSS).
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.research import s10_label_geometry as lg
from src.research import s10_ml_data as sd
from src.research import s10_ml_diagnostic as dx
from src.research import s10_ml_model as sm
from src.research import s6_clean_strategy_lab as s6

HEADING = "S10.3 RANKING DIAGNOSTIC COMPLETE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | ranking diagnostic - NOT strategy selection, NOT live accuracy"
LABEL, TARGET_R, HORIZON = "L1", 1.5, 6
STUDY_END = lg.STUDY_END
TOP_FRACTIONS = (("TOP10", 0.10), ("TOP20", 0.20), ("TOP30", 0.30), ("TOP40", 0.40), ("TOP50", 0.50))
PRECISION_K = (10, 20, 30, 50, 100)
PROB_EDGES = (0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
PROB_NAMES = ("<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70+")
RANDOM_SEED = 20261006
# Pre-declared rules (fixed before running on real data).
MIN_TOP10_RESOLVED = 30
MATERIAL_LIFT = 1.10
STRONG_LIFT = 1.20
MEANINGFUL_AUC = 0.55
REPRO_TOL = 1e-6


# ------------------------------------------------------------------ data / model
def label_l1(data: pd.DataFrame, caches: dict) -> pd.DataFrame:
    """L1 outcome + realized R per sample (future option candles used for the label only)."""
    out = data.copy()
    oc, rr = [], []
    for r in out[["instrument_key", "signal_time", "atr"]].itertuples(index=False):
        lab = sd.label_trade(caches["opt"].get(r.instrument_key), r.signal_time, r.atr, HORIZON, target_r=TARGET_R)
        oc.append(lab["outcome"])
        rr.append(lab["r_multiple"])
    out["l1_outcome"] = oc
    out["l1_r"] = pd.Series(rr, dtype=float).values
    out["l1_y"] = np.where(out["l1_outcome"] == sd.WIN, 1.0, np.where(out["l1_outcome"] == sd.LOSS, 0.0, np.nan))
    return out


def score(data: pd.DataFrame, model: sm.S10Model) -> tuple[pd.DataFrame, dict]:
    """Re-creates the S10.2 L1 model per side (TRAIN binary rows, COMMON_WINDOW) and scores every executed sample."""
    cols, pre = model.columns, model.preprocessor
    f = lg.scope_frame(data, lg.COMMON, max(h for _, _, h in lg.LABEL_DEFS))
    f = f[f["l1_outcome"].isin(sd.EXECUTED)].copy()
    f["prob"] = np.nan
    fits: dict[str, Any] = {}
    for side in sd.SIDES:
        g = f[f["side"] == side]
        tr = g[(g["split"] == "TRAIN") & g["l1_y"].notna()]
        ytr = tr["l1_y"].to_numpy(dtype=int)
        if len(set(ytr)) < 2:
            fits[side] = {"status": "NOT_FITTED_INSUFFICIENT_DATA", "train_binary": int(len(tr))}
            continue
        est = lg.make_lr().fit(pre.transform(tr[cols].to_numpy(dtype=float)), ytr)
        f.loc[g.index, "prob"] = est.predict_proba(pre.transform(g[cols].to_numpy(dtype=float)))[:, 1]
        fits[side] = {"status": "FITTED", "train_binary": int(len(tr)), "train_positives": int(ytr.sum())}
    return f, fits


def ranked(g: pd.DataFrame) -> pd.DataFrame:
    """Highest probability first; ties broken by time then instrument so the order is deterministic."""
    g = g[g["prob"].notna()]
    return g.sort_values(["prob", "signal_time", "instrument_key"], ascending=[False, True, True], kind="mergesort")


def top_count(n: int, frac: float) -> int:
    return int(math.ceil(frac * n)) if n else 0


def _rate(n, d):
    return float(n) / float(d) if d else None


def auc(g: pd.DataFrame) -> float | None:
    b = g[g["l1_y"].notna() & g["prob"].notna()]
    y = b["l1_y"].to_numpy(dtype=int)
    return float(roc_auc_score(y, b["prob"].to_numpy())) if len(set(y)) == 2 else None


def bucket_stats(g: pd.DataFrame) -> dict[str, Any]:
    o = g["l1_outcome"]
    w, l, t, a = (int((o == x).sum()) for x in (sd.WIN, sd.LOSS, sd.TIMEOUT, sd.AMBIGUOUS))
    r = g["l1_r"].dropna()
    return {"samples": int(len(g)), "WIN": w, "LOSS": l, "TIMEOUT": t, "AMBIGUOUS": a, "resolved": w + l,
            "win_rate": _rate(w, w + l), "raw_positive_rate": _rate(w, len(g)),
            "avg_predicted_probability": float(g["prob"].mean()) if len(g) else None,
            "avg_realized_r": float(r.mean()) if len(r) else None}


def buckets(g: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    r = ranked(g)
    out = [("ALL", r)] + [(name, r.iloc[:top_count(len(r), frac)]) for name, frac in TOP_FRACTIONS]
    out.append(("BOTTOM50", r.iloc[top_count(len(r), 0.50):]))
    return out


def _sub(scored: pd.DataFrame, split: str, side: str) -> pd.DataFrame:
    return scored[(scored["split"] == split) & (scored["side"] == side)]


# ------------------------------------------------------------------ tables
def decile_rows(scored: pd.DataFrame) -> list[dict]:
    out = []
    for split in ("TRAIN", "VALIDATION"):
        for side in sd.SIDES:
            g = _sub(scored, split, side)
            overall = bucket_stats(ranked(g))["win_rate"]
            for name, b in buckets(g):
                st = bucket_stats(b)
                out.append({"split": split, "side": side, "bucket": name, **st,
                            "lift_vs_overall": (st["win_rate"] / overall) if st["win_rate"] is not None and overall
                            else None})
    return out


def lift_rows(deciles: list[dict]) -> list[dict]:
    out = []
    for r in deciles:
        if r["bucket"] not in ("TOP10", "TOP20", "TOP30"):
            continue
        overall = _get(deciles, split=r["split"], side=r["side"], bucket="ALL")
        out.append({"split": r["split"], "side": r["side"], "bucket": r["bucket"], "bucket_samples": r["samples"],
                    "bucket_resolved": r["resolved"], "bucket_win_rate": r["win_rate"],
                    "overall_win_rate": overall.get("win_rate"), "lift": r["lift_vs_overall"],
                    "material": bool(r["lift_vs_overall"] is not None and r["lift_vs_overall"] >= MATERIAL_LIFT)})
    return out


def precision_rows(scored: pd.DataFrame) -> list[dict]:
    """Precision@K = WIN share of the K highest-ranked resolved (WIN / LOSS) VALIDATION samples."""
    out = []
    for side in sd.SIDES:
        r = ranked(_sub(scored, "VALIDATION", side))
        b = r[r["l1_y"].notna()]
        base = _rate(int(b["l1_y"].sum()), len(b))
        for k in PRECISION_K:
            if len(b) < k:
                out.append({"side": side, "k": k, "status": "INSUFFICIENT_DATA", "available_resolved": int(len(b)),
                            "wins": None, "precision": None, "overall_win_rate": base, "lift": None})
                continue
            w = int(b["l1_y"].iloc[:k].sum())
            out.append({"side": side, "k": k, "status": "OK", "available_resolved": int(len(b)), "wins": w,
                        "precision": w / k, "overall_win_rate": base, "lift": (w / k) / base if base else None})
    return out


def probability_rows(scored: pd.DataFrame) -> list[dict]:
    out = []
    for split in ("TRAIN", "VALIDATION"):
        for side in sd.SIDES:
            g = _sub(scored, split, side)
            g = g[g["prob"].notna()]
            idx = np.searchsorted(np.array(PROB_EDGES[1:]), g["prob"].to_numpy(), side="right")
            for i, name in enumerate(PROB_NAMES):
                b = g[idx == i]
                res = b[b["l1_y"].notna()]
                pred = float(res["prob"].mean()) if len(res) else None
                act = float(res["l1_y"].mean()) if len(res) else None
                out.append({"split": split, "side": side, "bucket": name, "count": int(len(b)),
                            "resolved": int(len(res)), "wins": int(res["l1_y"].sum()) if len(res) else 0,
                            "mean_predicted_all": float(b["prob"].mean()) if len(b) else None,
                            "mean_predicted_resolved": pred, "actual_win_rate": act,
                            "actual_minus_predicted": (act - pred) if act is not None and pred is not None else None})
    return out


def train_val_rows(scored: pd.DataFrame, deciles: list[dict]) -> list[dict]:
    out = []
    for side in sd.SIDES:
        for split in ("TRAIN", "VALIDATION"):
            d = {b: _get(deciles, split=split, side=side, bucket=b) for b in ("ALL", "TOP10", "TOP20", "TOP30")}
            out.append({"side": side, "split": split, "samples": d["ALL"].get("samples"),
                        "resolved": d["ALL"].get("resolved"), "overall_win_rate": d["ALL"].get("win_rate"),
                        **{f"{b.lower()}_win_rate": d[b].get("win_rate") for b in ("TOP10", "TOP20", "TOP30")},
                        **{f"{b.lower()}_lift": d[b].get("lift_vs_overall") for b in ("TOP10", "TOP20", "TOP30")},
                        "top10_resolved": d["TOP10"].get("resolved"), "auc": auc(_sub(scored, split, side)),
                        "note": "TRAIN is in-sample (model fit on TRAIN)" if split == "TRAIN" else "out-of-sample"})
    return out


def random_rows(scored: pd.DataFrame, deciles: list[dict]) -> list[dict]:
    """Descriptive control only: one deterministic random order of the same VALIDATION samples."""
    out = []
    for side in sd.SIDES:
        g = ranked(_sub(scored, "VALIDATION", side))
        perm = g.iloc[np.random.default_rng(RANDOM_SEED).permutation(len(g))]
        overall = _get(deciles, split="VALIDATION", side=side, bucket="ALL").get("win_rate")
        for name, frac in TOP_FRACTIONS[:2]:
            m = _get(deciles, split="VALIDATION", side=side, bucket=name)
            rnd = bucket_stats(perm.iloc[:top_count(len(perm), frac)])
            out.append({"side": side, "bucket": name, "seed": RANDOM_SEED, "samples": m.get("samples"),
                        "model_resolved": m.get("resolved"), "model_win_rate": m.get("win_rate"),
                        "random_resolved": rnd["resolved"], "random_win_rate": rnd["win_rate"],
                        "overall_win_rate": overall,
                        "model_minus_random": (m["win_rate"] - rnd["win_rate"])
                        if m.get("win_rate") is not None and rnd["win_rate"] is not None else None,
                        "note": "descriptive control; no statistical claim"})
    return out


def _get(rows: list[dict], **kw) -> dict:
    return next((r for r in rows if all(r.get(k) == v for k, v in kw.items())), {})


# ------------------------------------------------------------------ classification
def side_quality(tv: list[dict], side: str) -> dict[str, Any]:
    """Pre-declared per-side rule.
    INSUFFICIENT_DATA: < MIN_TOP10_RESOLVED resolved July samples in the top 10%.
    GOOD: July top10 & top20 lift >= STRONG_LIFT, top30 lift >= MATERIAL_LIFT, July AUC >= MEANINGFUL_AUC, and TRAIN
          top10/20/30 lifts all > 1 (same direction).
    OVERFIT: TRAIN top10 & top20 lift >= STRONG_LIFT but July top10 & top20 lift < MATERIAL_LIFT.
    WEAK: at least two of July top10/20/30 lifts >= MATERIAL_LIFT and July AUC > 0.5.
    NONE: otherwise (NO_MEANINGFUL_RANKING_EDGE)."""
    tr, va = _get(tv, side=side, split="TRAIN"), _get(tv, side=side, split="VALIDATION")
    L = lambda r, b: r.get(f"{b}_lift") if r.get(f"{b}_lift") is not None else 0.0  # noqa: E731
    vl = {b: L(va, b) for b in ("top10", "top20", "top30")}
    tl = {b: L(tr, b) for b in ("top10", "top20", "top30")}
    v_auc = va.get("auc")
    consistent = all(x > 1.0 for x in tl.values())
    train_strong = tl["top10"] >= STRONG_LIFT and tl["top20"] >= STRONG_LIFT
    collapse = vl["top10"] < MATERIAL_LIFT and vl["top20"] < MATERIAL_LIFT
    material = sum(x >= MATERIAL_LIFT for x in vl.values())
    if (va.get("top10_resolved") or 0) < MIN_TOP10_RESOLVED or v_auc is None:
        q, why = "INSUFFICIENT_DATA", f"July top-10% has < {MIN_TOP10_RESOLVED} resolved samples or no AUC"
    elif (vl["top10"] >= STRONG_LIFT and vl["top20"] >= STRONG_LIFT and vl["top30"] >= MATERIAL_LIFT
          and v_auc >= MEANINGFUL_AUC and consistent):
        q, why = "GOOD", "July top buckets clearly beat the July average and TRAIN points the same way"
    elif train_strong and collapse:
        q, why = "OVERFIT", "TRAIN top buckets strong but July top10/top20 lift below the material level"
    elif material >= 2 and v_auc > 0.5:
        q, why = "WEAK", f"{material}/3 July top buckets reach lift >= {MATERIAL_LIFT}"
    else:
        q, why = "NONE", "July top buckets do not materially beat the July average"
    return {"side": side, "quality": q, "reason": why, "validation_lifts": vl, "train_lifts": tl,
            "validation_auc": v_auc, "train_auc": tr.get("auc"), "train_consistent": consistent,
            "overfit_flag": bool(train_strong and collapse),
            "auc_gap": (tr["auc"] - v_auc) if tr.get("auc") is not None and v_auc is not None else None,
            "meaningful_ranking": q in ("GOOD", "WEAK"),
            "statement": None if q in ("GOOD", "WEAK") else "NO_MEANINGFUL_RANKING_EDGE"}


def decide(sides: dict[str, dict]) -> str:
    q = [sides[s]["quality"] for s in sd.SIDES]
    if all(x == "GOOD" for x in q):
        return "RANKING_EDGE_STRONG"
    if any(x in ("GOOD", "WEAK") for x in q):
        return "RANKING_EDGE_WEAK"
    if any(x == "OVERFIT" for x in q):
        return "OVERFIT_RANKING"
    if any(x == "NONE" for x in q):
        return "NO_RANKING_EDGE"
    return "INSUFFICIENT_DATA"


SHORT = {"RANKING_EDGE_STRONG": "STRONG", "RANKING_EDGE_WEAK": "WEAK", "NO_RANKING_EDGE": "NONE",
         "OVERFIT_RANKING": "OVERFIT", "INSUFFICIENT_DATA": "INSUFFICIENT_DATA"}


def reproduction(tv: list[dict], reference: dict[str, float] | None) -> dict[str, Any]:
    if not reference:
        return {"status": "NO_REFERENCE", "sides": {}}
    rows = {}
    for s in sd.SIDES:
        got, exp = _get(tv, side=s, split="VALIDATION").get("auc"), reference.get(s)
        ok = got is not None and exp is not None and abs(got - exp) <= REPRO_TOL
        rows[s] = {"expected_s10_2": exp, "reproduced": got, "match": ok}
    return {"status": "REPRODUCED" if all(r["match"] for r in rows.values()) else "MISMATCH", "sides": rows}


# ------------------------------------------------------------------ study
def run_study(*, market: sd.MarketData, model: sm.S10Model, reference_auc: dict[str, float] | None = None,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    flags = dx.flags_from_columns(model.columns)
    days = sd.days_in(market.sessions, sd.SPLITS["TRAIN"][0], STUDY_END)
    caches: dict = {"fut": {}, "opt": {}}
    say(f"building the S10 sample set for TRAIN + VALIDATION ({len(days)} sessions; Aug..Oct never loaded) ...")
    data = sd.build_samples(market, days, volume=flags["volume"], oi=flags["oi"], caches=caches).data.copy()
    data["split"] = [sd.split_of_day(date.fromisoformat(d)) for d in data["session_date"]]
    say("labelling L1 (1R SL / 1.5R target / 6 candles) ...")
    data = label_l1(data, caches)
    say("re-creating the frozen L1 model (fixed LR, frozen S10 preprocessor) and scoring ...")
    scored, fits = score(data, model)
    deciles = decile_rows(scored)
    tv = train_val_rows(scored, deciles)
    sides = {s: side_quality(tv, s) for s in sd.SIDES}
    mx = market.reads["max_ts"]
    no_post = bool((mx is None or mx.date() <= STUDY_END)
                   and (data.empty or data["session_date"].max() <= STUDY_END.isoformat())
                   and market.spot["timestamp"].dt.date.max() <= STUDY_END)
    lg_scope = lg.scope_frame(data, lg.COMMON, max(h for _, _, h in lg.LABEL_DEFS))
    excluded = {s: int(((lg_scope["side"] == s) & ~lg_scope["l1_outcome"].isin(sd.EXECUTED)).sum()) for s in sd.SIDES}
    return {"heading": HEADING, "banner": BANNER,
            "label": {"name": LABEL, "stop_r": 1.0, "target_r": TARGET_R, "horizon": HORIZON},
            "model": "fixed LogisticRegression(C=1.0) on the frozen S10 preprocessor / feature list, fit on TRAIN "
                     "binary rows (COMMON_WINDOW) - identical to S10.2; not tuned",
            "fits": fits, "population": "executed samples (WIN/LOSS/TIMEOUT/AMBIGUOUS), signal <= "
            f"{lg.COMMON_LAST.isoformat(timespec='minutes')}", "not_executed_excluded": excluded,
            "period": [days[0].isoformat(), days[-1].isoformat()] if days else None,
            "max_market_read": mx.isoformat() if mx is not None else None, "no_post_july_data": no_post,
            "deciles": deciles, "lift": lift_rows(deciles), "precision": precision_rows(scored),
            "probability_buckets": probability_rows(scored), "train_vs_validation": tv,
            "random_control": random_rows(scored, deciles), "sides": sides, "decision": decide(sides),
            "reproduction": reproduction(tv, reference_auc),
            "rules": {"min_top10_resolved": MIN_TOP10_RESOLVED, "material_lift": MATERIAL_LIFT,
                      "strong_lift": STRONG_LIFT, "meaningful_auc": MEANINGFUL_AUC, "random_seed": RANDOM_SEED},
            "production_unchanged": None, "live_orders": 0, "october_used": False, "_data": scored}


# ------------------------------------------------------------------ reports
REPORT_FILES = ("ranking_deciles.csv", "precision_at_k.csv", "probability_buckets.csv", "lift_analysis.csv",
                "train_vs_validation.csv", "random_control.csv", "ranking_report.md", "summary.json")


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    s6._csv(p["ranking_deciles.csv"], a["deciles"])
    s6._csv(p["precision_at_k.csv"], a["precision"])
    s6._csv(p["probability_buckets.csv"], a["probability_buckets"])
    s6._csv(p["lift_analysis.csv"], a["lift"])
    s6._csv(p["train_vs_validation.csv"], a["train_vs_validation"])
    s6._csv(p["random_control.csv"], a["random_control"])
    p["ranking_report.md"].write_text(markdown(a), encoding="utf-8")
    summary = {k: v for k, v in a.items() if not k.startswith("_")}
    p["summary.json"].write_text(json.dumps(s6._jsonable(summary), indent=2, default=str), encoding="utf-8")
    return p


def _f(v, nd: int = 4) -> str:
    return "N/A" if v is None else f"{v:.{nd}f}"


def _p(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _wr(a: dict, split: str, side: str, bucket: str) -> str:
    r = _get(a["deciles"], split=split, side=side, bucket=bucket)
    return f"{_p(r.get('win_rate'))} ({r.get('WIN')}/{r.get('resolved')})" if r else "N/A"


def terminal_text(a: dict) -> str:
    out = [HEADING, "", f"L1 = 1R SL / {TARGET_R}R TARGET / {HORIZON} candles | win rate = WIN / (WIN + LOSS) on "
           "July VALIDATION executed samples", f"Frozen L1 model reproduction vs S10.2: {a['reproduction']['status']}",
           ""]
    for s in sd.SIDES:
        tv = _get(a["train_vs_validation"], side=s, split="VALIDATION")
        sq = a["sides"][s]
        out += [f"{s}:", f"overall win rate: {_wr(a, 'VALIDATION', s, 'ALL')}",
                f"top10: {_wr(a, 'VALIDATION', s, 'TOP10')}", f"top20: {_wr(a, 'VALIDATION', s, 'TOP20')}",
                f"top30: {_wr(a, 'VALIDATION', s, 'TOP30')}", f"AUC: {_f(tv.get('auc'))}",
                f"lift: top10 {_f(tv.get('top10_lift'), 2)}x | top20 {_f(tv.get('top20_lift'), 2)}x | top30 "
                f"{_f(tv.get('top30_lift'), 2)}x",
                f"ranking quality: {sq['quality']}" + (f" - {sq['statement']}" if sq["statement"] else ""), ""]
    out += ["TRAIN vs VALIDATION:", ""]
    for s in sd.SIDES:
        tr = _get(a["train_vs_validation"], side=s, split="TRAIN")
        out += [f"{s}:", f"top10 TRAIN: {_wr(a, 'TRAIN', s, 'TOP10')} (AUC {_f(tr.get('auc'))}, in-sample)",
                f"top10 VALIDATION: {_wr(a, 'VALIDATION', s, 'TOP10')}", ""]
    out += [f"RANDOM CONTROL (seed {RANDOM_SEED}, descriptive only):"]
    for r in a["random_control"]:
        out.append(f"{r['side']} {r['bucket']}: model {_p(r['model_win_rate'])} vs random {_p(r['random_win_rate'])}")
    out += ["", "RANKING:", SHORT[a["decision"]], f"({a['decision']})", "",
            "This is diagnostic only: no threshold, feature, model, target, SL or horizon change.", "",
            "DO NOT RUN OCTOBER." if a["no_post_july_data"] and not a["october_used"] else
            "WARNING: post-July data detected",
            "PRODUCTION CHANGED: " + ("NO" if a["production_unchanged"] is not False else "YES"), "LIVE ORDERS: 0"]
    return "\n".join(out)


def markdown(a: dict) -> str:
    md = ["# S10.3 L1 model ranking / precision diagnostic", "", f"**{BANNER}**", "",
          f"TRAIN Apr-Jun + VALIDATION July only (latest market read {a['max_market_read']}). Label L1 = 1R SL / "
          f"{TARGET_R}R target / {HORIZON} candles. Model: {a['model']}. Population: {a['population']}.", "",
          f"Reproduction of the S10.2 July AUC: **{a['reproduction']['status']}**.", "",
          f"## Decision: **{a['decision']}**", ""]
    for s in sd.SIDES:
        q = a["sides"][s]
        md.append(f"- {s}: **{q['quality']}** - {q['reason']}" + (f" ({q['statement']})" if q["statement"] else ""))
    md += ["", "## Ranking buckets (VALIDATION)", "",
           "| Side | Bucket | Samples | WIN | LOSS | TIMEOUT | AMBIG | Win rate | Raw positive | Avg prob | Avg R | Lift |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["deciles"]:
        if r["split"] == "VALIDATION":
            md.append(f"| {r['side']} | {r['bucket']} | {r['samples']} | {r['WIN']} | {r['LOSS']} | {r['TIMEOUT']} | "
                      f"{r['AMBIGUOUS']} | {_p(r['win_rate'])} | {_p(r['raw_positive_rate'])} | "
                      f"{_f(r['avg_predicted_probability'])} | {_f(r['avg_realized_r'], 3)} | "
                      f"{_f(r['lift_vs_overall'], 2)} |")
    md += ["", "## Precision at top K (VALIDATION resolved samples)", "", "| Side | K | Status | Precision | Lift |",
           "|---|---|---|---|---|"]
    md += [f"| {r['side']} | {r['k']} | {r['status']} | {_p(r['precision'])} | {_f(r['lift'], 2)} |"
           for r in a["precision"]]
    md += ["", "## Calibration (VALIDATION)", "", "| Side | Bucket | Count | Resolved | Mean predicted | Actual win rate |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['side']} | {r['bucket']} | {r['count']} | {r['resolved']} | {_f(r['mean_predicted_resolved'])} | "
           f"{_p(r['actual_win_rate'])} |" for r in a["probability_buckets"] if r["split"] == "VALIDATION"]
    md += ["", "## TRAIN vs VALIDATION", "", "| Side | Split | Overall | Top10 | Top20 | Top30 | AUC |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['side']} | {r['split']} | {_p(r['overall_win_rate'])} | {_p(r['top10_win_rate'])} | "
           f"{_p(r['top20_win_rate'])} | {_p(r['top30_win_rate'])} | {_f(r['auc'])} |" for r in a["train_vs_validation"]]
    md += ["", f"## Random control (seed {RANDOM_SEED}, descriptive only, no statistical claim)", "",
           "| Side | Bucket | Model win rate | Random win rate |", "|---|---|---|---|"]
    md += [f"| {r['side']} | {r['bucket']} | {_p(r['model_win_rate'])} | {_p(r['random_win_rate'])} |"
           for r in a["random_control"]]
    md += ["", "## Pre-declared rules", "",
           "- Win rate = WIN / (WIN + LOSS); lift = bucket win rate / overall July win rate.",
           f"- Material lift >= {MATERIAL_LIFT}; strong lift >= {STRONG_LIFT}; meaningful AUC >= {MEANINGFUL_AUC}; "
           f"top-10% needs >= {MIN_TOP10_RESOLVED} resolved July samples.",
           "- GOOD: July top10 & top20 lift strong, top30 material, AUC meaningful, TRAIN lifts all > 1.",
           "- OVERFIT: TRAIN top10 & top20 strong but July top10 & top20 below material. WEAK: >= 2 of 3 July lifts "
           "material and AUC > 0.5. NONE otherwise.",
           "- Overall: STRONG only if both sides GOOD; WEAK if any side GOOD/WEAK; else OVERFIT, NONE, INSUFFICIENT.",
           "- TRAIN numbers are in-sample (model fit on TRAIN).", "",
           "Diagnostic only. No threshold, feature, model, target, SL or horizon change. August, September and October "
           "not loaded. Production unchanged; live orders 0."]
    return "\n".join(md) + "\n"
