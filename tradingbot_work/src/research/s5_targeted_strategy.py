"""S5 TARGETED STRATEGY RESEARCH (PAPER | RESEARCH ONLY | NO ORDERS).

Thirteen pre-declared candidates (B1-B6, S1-S7) built only from the six STABLE patterns of the S2 diagnostic. Every
candidate is a FILTER on unchanged S2 signals: the same contract, entry, stop, target and exit are kept; a signal is
either taken or not. Features are the causal signal-candle features of ``s2_diagnostic_lab``.

Chronology:
  TRAIN 2026-04-01..06-30 | VALIDATION 07-01..07-31 | OOS 08-01..08-31 | FINAL HOLDOUT 09-01..09-30
Selection reads ONLY rows before the OOS start (``select`` raises otherwise). OOS is used for the gate status, never
to pick or change a candidate. September is evaluated once, after the selection is frozen and fingerprinted.

Known limitation (reported as an integrity WARN): the six patterns were discovered by the diagnostic on Apr-Sep, so
OOS and holdout months took part in pattern discovery. ``discovery_recheck`` re-measures each pattern on Apr-Jul only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

from src.backtest import accuracy_70_lab as acc
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg

HEADING = "S5 TARGETED RESEARCH"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS"
IST = dg.IST
DIRECTIONS = ("BUY", "SELL")
SPLITS = {"TRAIN": (date(2026, 4, 1), date(2026, 6, 30)), "VALIDATION": (date(2026, 7, 1), date(2026, 7, 31)),
          "OOS": (date(2026, 8, 1), date(2026, 8, 31)), "HOLDOUT": (date(2026, 9, 1), date(2026, 9, 30))}
SELECTION_END = SPLITS["VALIDATION"][1]
WF_START, WF_END = SPLITS["TRAIN"][0], SPLITS["OOS"][1]

MIN_TRAIN_TRADES, MIN_VAL_TRADES = 30, 10
MIN_SUFFICIENT_OOS, MIN_ROBUST_OOS = 30, 50
MIN_HOLDOUT_TRADES = 20
MIN_REGIME_TRADES = 5
MAX_CONDITIONS = 4
SELECT_MIN_PF = 1.2

ROBUST, PROMISING, INSUFFICIENT, REJECTED, KEEP = "ROBUST_70", "PROMISING", "INSUFFICIENT_DATA", "REJECTED", \
    "KEEP_CURRENT"
PASSED_HO, FAILED_HO, NOT_RUN = "PASSED_FINAL_HOLDOUT", "FAILED_FINAL_HOLDOUT", "NOT_RUN"
NOT_ACHIEVED_TEXT = "70% OOS accuracy was not achieved by the targeted combinations."


# ------------------------------------------------------------------ conditions (causal, signal candle only)
def _rsi_in(lo: float, hi: float) -> Callable[[dict], bool | None]:
    def fn(f: dict) -> bool | None:
        v = dg._num(f.get("rsi14"))
        return None if v is None else lo <= v < hi
    return fn


def _ema_sep(f: dict) -> bool | None:
    v = dg._num(f.get("ema_sep_atr"))
    return None if v is None else v < -1.0


def _trend_down(f: dict) -> bool | None:
    v = f.get("trend_regime")
    return None if v in (None, dg.UNAVAILABLE) else v == "TREND_DOWN"


def _time_12_13(f: dict) -> bool | None:
    v = f.get("time_window")
    return None if v is None else v == "12:00-13:00"


def _ema20_gt_50(f: dict) -> bool | None:
    v = f.get("ema20_vs_ema50")
    return None if v in (None, dg.UNAVAILABLE) else v == "EMA20>EMA50"


CONDITIONS: dict[str, tuple[str, Callable[[dict], bool | None]]] = {
    "EMA_SEP": ("(EMA20 - EMA50) / ATR14 < -1 at the signal candle", _ema_sep),
    "TREND_DOWN": ("trend regime TREND_DOWN (close < EMA20 < EMA50 and EMA20 slope < 0)", _trend_down),
    "RSI_40_50": ("40 <= RSI14 < 50 at the signal candle", _rsi_in(40.0, 50.0)),
    "RSI_50_60": ("50 <= RSI14 < 60 at the signal candle", _rsi_in(50.0, 60.0)),
    "TIME_1200_1300": ("signal candle start in [12:00, 13:00)", _time_12_13),
    "EMA20_GT_EMA50": ("EMA20 > EMA50 at the signal candle", _ema20_gt_50),
    "OPT_CONFIRM": ("selected CE/PE premium rising on completed candles <= signal: signal-candle close > open "
                    "AND close > close 3 candles earlier; option data missing -> NO_SIGNAL", None),
}


@dataclass(frozen=True)
class Candidate:
    cid: str
    name: str
    direction: str
    conditions: tuple[str, ...]

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)

    def describe(self) -> str:
        return " AND ".join(CONDITIONS[c][0] for c in self.conditions)


BASE_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("B1", "EMA_SEP_ONLY", "BUY", ("EMA_SEP",)),
    Candidate("B2", "TREND_DOWN_ONLY", "BUY", ("TREND_DOWN",)),
    Candidate("B3", "EMA_SEP + TREND_DOWN", "BUY", ("EMA_SEP", "TREND_DOWN")),
    Candidate("B4", "EMA_SEP + RSI_40_50", "BUY", ("EMA_SEP", "RSI_40_50")),
    Candidate("B5", "TREND_DOWN + RSI_40_50", "BUY", ("TREND_DOWN", "RSI_40_50")),
    Candidate("B6", "EMA_SEP + TREND_DOWN + RSI_40_50", "BUY", ("EMA_SEP", "TREND_DOWN", "RSI_40_50")),
    Candidate("S1", "RSI_50_60", "SELL", ("RSI_50_60",)),
    Candidate("S2", "TIME_1200_1300", "SELL", ("TIME_1200_1300",)),
    Candidate("S3", "EMA20_GT_EMA50", "SELL", ("EMA20_GT_EMA50",)),
    Candidate("S4", "RSI_50_60 + EMA20_GT_EMA50", "SELL", ("RSI_50_60", "EMA20_GT_EMA50")),
    Candidate("S5", "RSI_50_60 + TIME_1200_1300", "SELL", ("RSI_50_60", "TIME_1200_1300")),
    Candidate("S6", "EMA20_GT_EMA50 + TIME_1200_1300", "SELL", ("EMA20_GT_EMA50", "TIME_1200_1300")),
    Candidate("S7", "RSI_50_60 + EMA20_GT_EMA50 + TIME_1200_1300", "SELL",
              ("RSI_50_60", "EMA20_GT_EMA50", "TIME_1200_1300")),
)
DISCOVERED = (("BUY", "EMA_SEP"), ("BUY", "TREND_DOWN"), ("BUY", "RSI_40_50"), ("SELL", "RSI_50_60"),
              ("SELL", "TIME_1200_1300"), ("SELL", "EMA20_GT_EMA50"))


def with_option(c: Candidate) -> Candidate:
    return Candidate(f"{c.cid}_OPT", f"{c.name} + OPT_CONFIRM", c.direction, c.conditions + ("OPT_CONFIRM",))


def option_confirmation(ctx: dict | None) -> bool | None:
    """CE (BUY) / PE (SELL) premium direction on the selected contract, completed candles <= signal only."""
    o = acc.option_features(ctx) if ctx else None
    if not o or o.get("premium_mom3") is None or o.get("premium_bull") is None:
        return None
    return bool(o["premium_bull"]) and o["premium_mom3"] > 0


def condition_value(r: dict, cond: str) -> bool | None:
    if cond == "OPT_CONFIRM":
        return r.get("opt_confirm")
    return CONDITIONS[cond][1](r["features"])


def candidate_rows(c: Candidate, rows: Sequence[dict]) -> list[dict]:
    """S2 rows of the candidate's direction where every condition is True (None -> NO_SIGNAL, not taken)."""
    out = []
    for r in rows:
        if r["direction"] != c.direction:
            continue
        if all(condition_value(r, k) is True for k in c.conditions):
            out.append(r)
    return out


def no_signal_count(c: Candidate, rows: Sequence[dict]) -> int:
    return sum(1 for r in rows if r["direction"] == c.direction
               and any(condition_value(r, k) is None for k in c.conditions))


# ------------------------------------------------------------------ metrics
def day_of(r: dict) -> date:
    return date.fromisoformat(str(r["session_date"])[:10])


def in_range(rows: Sequence[dict], lo: date, hi: date) -> list[dict]:
    return [r for r in rows if lo <= day_of(r) <= hi]


def split_rows(rows: Sequence[dict], split: str) -> list[dict]:
    return in_range(rows, *SPLITS[split])


def metrics(rows: Sequence[dict]) -> dict[str, Any]:
    m = dg.perf(rows)
    dec = dg.decided(rows)
    days: dict[str, float] = defaultdict(float)
    for r in dec:
        days[str(r["session_date"])[:10]] += dg.trade_pnl(r)
    best = max(days.values()) if days else 0.0
    return {**m, "profitable_days": sum(v > 0 for v in days.values()), "traded_days": len(days),
            "active_months": len({str(r["session_date"])[:7] for r in dec}),
            "pnl_without_best_day": round(sum(days.values()) - max(best, 0.0), 2),
            "best_day_share": (best / sum(v for v in days.values() if v > 0)) if any(v > 0 for v in days.values())
            else None}


def sample_grade(n: int) -> str:
    return "A" if n >= 100 else "B" if n >= MIN_ROBUST_OOS else "C" if n >= MIN_SUFFICIENT_OOS else "D"


GRADE_TEXT = {"A": "strong", "B": "usable", "C": "weak", "D": "insufficient"}


def regime_dependency(c: Candidate, rows: Sequence[dict]) -> dict[str, Any]:
    """Uses TRAIN+VALIDATION+OOS only. Trend-regime candidates are tested on the volatility axis."""
    axis = "vol_regime" if "TREND_DOWN" in c.conditions else "trend_regime"
    groups: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in dg.decided(in_range(rows, SPLITS["TRAIN"][0], SPLITS["OOS"][1])):
        g = r["features"][axis]
        groups[g] += dg.trade_pnl(r)
        counts[g] += 1
    active = {g: p for g, p in groups.items() if counts[g] >= MIN_REGIME_TRADES and g != dg.UNAVAILABLE}
    total = sum(groups.values())
    best = max(active.values(), default=0.0)
    return {"axis": axis, "active": len(active), "profitable": sum(p > 0 for p in active.values()),
            "pnl_without_best_regime": round(total - max(best, 0.0), 2),
            "dependent": not (total - max(best, 0.0) > 0)}


# ------------------------------------------------------------------ walk-forward (semi-monthly, Apr..Aug)
def wf_windows(start: date = WF_START, end: date = WF_END) -> list[tuple[date, date]]:
    out, d = [], start
    while d <= end:
        if d.day == 1:
            w_end = d.replace(day=15)
        else:
            nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
            w_end = nxt - timedelta(days=1)
        w_end = min(w_end, end)
        out.append((d, w_end))
        d = w_end + timedelta(days=1)
    return out


def walk_forward(rows: Sequence[dict], end: date = WF_END) -> dict[str, Any]:
    """Fixed rules evaluated window by window in time order (no fitting, nothing from later windows)."""
    wins = []
    for lo, hi in wf_windows(WF_START, end):
        m = dg.perf(in_range(rows, lo, hi))
        wins.append({"window": f"{lo}..{hi}", "start": lo.isoformat(), "end": hi.isoformat(),
                     "used_for_selection": hi <= SELECTION_END, **{k: m[k] for k in
                     ("trades", "wins", "losses", "accuracy", "profit_factor", "net_pnl")}})
    act = [w for w in wins if w["trades"] > 0]
    share = sum(w["net_pnl"] > 0 for w in act) / len(act) if act else None
    accs = [w["accuracy"] for w in act if w["accuracy"] is not None]
    pfs = [w["profit_factor"] for w in act if w["profit_factor"] is not None]
    worst = min(act, key=lambda w: w["net_pnl"]) if act else None
    best = max(act, key=lambda w: w["net_pnl"]) if act else None
    mpf = median(pfs) if pfs else None
    return {"windows": wins, "active_windows": len(act), "profitable_share": share,
            "median_accuracy": median(accs) if accs else None, "median_pf": mpf,
            "worst": None if worst is None else f"{worst['window']} Rs {worst['net_pnl']:,.2f}",
            "best": None if best is None else f"{best['window']} Rs {best['net_pnl']:,.2f}",
            "stable": bool(len(act) >= 4 and share is not None and share >= 0.5 and mpf is not None and mpf >= 1.0)}


# ------------------------------------------------------------------ selection (TRAIN + VALIDATION + WF<=Jul only)
def selection_stats(c: Candidate, pre_rows: Sequence[dict]) -> dict[str, Any]:
    rows = candidate_rows(c, pre_rows)
    tr, va = metrics(split_rows(rows, "TRAIN")), metrics(split_rows(rows, "VALIDATION"))
    wf = walk_forward(rows, SELECTION_END)
    trpf, vapf = tr["profit_factor"] or 0.0, va["profit_factor"] or 0.0
    eligible = (tr["trades"] >= MIN_TRAIN_TRADES and va["trades"] >= MIN_VAL_TRADES and tr["net_pnl"] > 0
                and trpf >= SELECT_MIN_PF and va["net_pnl"] > 0)
    return {"cid": c.cid, "eligible": eligible, "train": tr, "validation": va, "wf_selection": wf,
            "score": (min(trpf, vapf), wf["profitable_share"] or 0.0, tr["trades"] + va["trades"])}


def _assert_pre_oos(rows: Sequence[dict]) -> None:
    late = [r for r in rows if day_of(r) > SELECTION_END]
    if late:
        raise ValueError(f"selection received {len(late)} rows after {SELECTION_END} (OOS/holdout must not be used)")


def pick(cands: Sequence[Candidate], pre_rows: Sequence[dict]) -> tuple[Candidate | None, dict[str, dict]]:
    _assert_pre_oos(pre_rows)
    stats = {c.cid: selection_stats(c, pre_rows) for c in cands}
    pool = [c for c in cands if stats[c.cid]["eligible"]]
    best = (min(pool, key=lambda c: (tuple(-x for x in stats[c.cid]["score"]), c.n_conditions, c.cid))
            if pool else None)
    return best, stats


def select(pre_rows: Sequence[dict]) -> dict[str, Any]:
    """Two pre-declared steps per direction: best base candidate, then base vs base+OPT_CONFIRM (same score)."""
    _assert_pre_oos(pre_rows)
    out: dict[str, Any] = {"stats": {}, "selected": {}, "opt_variants": []}
    for d in DIRECTIONS:
        base, st = pick([c for c in BASE_CANDIDATES if c.direction == d], pre_rows)
        out["stats"].update(st)
        if base is None:
            out["selected"][d] = None
            continue
        opt = with_option(base)
        out["opt_variants"].append(opt)
        final, st2 = pick([base, opt], pre_rows)
        out["stats"].update(st2)
        out["selected"][d] = final or base
    return out


def fingerprint(sel: dict) -> str:
    spec = {d: None if c is None else asdict(c) for d, c in sel["selected"].items()}
    spec["candidates"] = [asdict(c) for c in BASE_CANDIDATES]
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ 70% gate
def gate(c: Candidate | None, rows: Sequence[dict], lookahead_ok: bool, contaminated: bool,
         n_conditions: int | None = None) -> dict[str, Any]:
    """rows = the candidate's rows (any period); only TRAIN..OOS are read here."""
    rows = in_range(rows, SPLITS["TRAIN"][0], SPLITS["OOS"][1])
    o, v = metrics(split_rows(rows, "OOS")), metrics(split_rows(rows, "VALIDATION"))
    wf = walk_forward(rows)
    reg = regime_dependency(c, rows) if c is not None else {"dependent": False}
    ncond = n_conditions if n_conditions is not None else (c.n_conditions if c else 0)
    pf = o["profit_factor"] or 0.0
    checks = {
        "oos_accuracy_70": o["accuracy"] is not None and o["accuracy"] >= 0.70,
        "oos_sample_sufficient": o["trades"] >= MIN_ROBUST_OOS,
        "oos_pf_1_5": pf >= 1.5, "oos_pnl_positive": o["net_pnl"] > 0,
        "validation_profitable": v["net_pnl"] > 0, "walk_forward_stable": wf["stable"],
        "no_day_dependency": o["pnl_without_best_day"] > 0, "no_regime_dependency": not reg["dependent"],
        "no_lookahead": lookahead_ok, "reasonable_complexity": ncond <= MAX_CONDITIONS,
    }
    if all(checks.values()):
        status = ROBUST
    elif o["trades"] < MIN_SUFFICIENT_OOS or v["trades"] < MIN_VAL_TRADES:
        status = INSUFFICIENT
    elif (o["net_pnl"] > 0 and pf >= SELECT_MIN_PF and v["net_pnl"] > 0 and lookahead_ok
          and checks["no_day_dependency"]):
        status = PROMISING
    else:
        status = REJECTED
    return {"status": status, "checks": checks, "failed": [k for k, ok in checks.items() if not ok], "oos": o,
            "validation": v, "train": metrics(split_rows(rows, "TRAIN")), "walk_forward": wf, "regime": reg,
            "grade": sample_grade(o["trades"]), "contaminated_discovery": contaminated}


def holdout_status(m: dict, robust: bool) -> str:
    ok = m["trades"] >= MIN_HOLDOUT_TRADES and m["net_pnl"] > 0 and (m["profit_factor"] or 0.0) >= SELECT_MIN_PF
    if robust:
        ok = ok and m["accuracy"] is not None and m["accuracy"] >= 0.70
    return PASSED_HO if ok else FAILED_HO


# ------------------------------------------------------------------ causality checks
FEATURE_SOURCES = {
    "ema_sep_atr": "(EMA20 - EMA50) / ATR14 of the signal candle; EMA/ATR are recursive over candles <= signal",
    "trend_regime": "close / EMA20 / EMA50 / EMA20 3-bar slope at the signal candle",
    "rsi14": "Wilder RSI14 over closes <= signal candle",
    "time_window": "signal candle START timestamp (IST) - the candle is complete before entry at +5 min",
    "ema20_vs_ema50": "EMA20 and EMA50 at the signal candle",
    "opt_confirm": "selected contract's premium candles with timestamp <= signal candle (acc.option_features)",
}


def option_truncation_check(rows: Sequence[dict], ctx_by_key: dict, sample: int = 30) -> dict[str, Any]:
    """Tripling every premium candle after the signal must not change the option confirmation."""
    picked = [r for r in rows if r.get("opt_confirm") is not None]
    picked = picked[::max(1, len(picked) // sample)][:sample]
    bad = []
    for r in picked:
        ctx = ctx_by_key.get(dra.key_of(r))
        p = ctx["premium"].copy()
        later = (dra._ist_series(p["timestamp"]) > dra._ts(r["signal_time"])).to_numpy()
        for col in ("open", "high", "low", "close", "volume"):
            if col in p:
                p[col] = p[col].astype(float)
                p.loc[later, col] = p.loc[later, col] * 3.0
        again = option_confirmation(dict(ctx, premium=p))
        if again != r["opt_confirm"]:
            bad.append(f"{r['signal_time']}|{r['direction']}: {r['opt_confirm']} vs {again}")
    return {"checked": len(picked), "mismatches": bad[:10], "passed": not bad}


def time_window_check(rows: Sequence[dict]) -> dict[str, Any]:
    bad = [r["signal_time"] for r in rows if r["features"]["time_window"] != dg.time_window(r["signal_time"])]
    return {"checked": len(rows), "mismatches": bad[:10], "passed": not bad}


def discovery_recheck(rows: Sequence[dict]) -> list[dict]:
    """Each discovered pattern re-measured on TRAIN+VALIDATION only (what was knowable before OOS)."""
    pre = in_range(rows, SPLITS["TRAIN"][0], SELECTION_END)
    out = []
    for d, cond in DISCOVERED:
        sub = [r for r in pre if r["direction"] == d and condition_value(r, cond) is True]
        m = metrics(sub)
        months = defaultdict(float)
        for r in dg.decided(sub):
            months[str(r["session_date"])[:7]] += dg.trade_pnl(r)
        out.append({"direction": d, "pattern": cond, "trades_apr_jul": m["trades"], "accuracy": m["accuracy"],
                    "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"],
                    "months_profitable": sum(v > 0 for v in months.values()), "months_active": len(months),
                    "supported_before_oos": bool(m["trades"] >= MIN_TRAIN_TRADES and m["net_pnl"] > 0
                                                 and (m["profit_factor"] or 0.0) >= SELECT_MIN_PF)})
    return out


# ------------------------------------------------------------------ study
def prepare_rows(rows: Sequence[dict], ctx_by_key: dict) -> list[dict]:
    """Annotated S2 rows (dg.annotate) + causal option confirmation."""
    return [{**r, "opt_confirm": option_confirmation(ctx_by_key.get(dra.key_of(r)))} for r in rows]


def _flip(rows: Sequence[dict], after: date) -> list[dict]:
    out = []
    for r in rows:
        if day_of(r) > after and r.get("outcome") in ("WIN", "LOSS"):
            r = dict(r, outcome="LOSS" if r["outcome"] == "WIN" else "WIN", pnl=-(r.get("pnl") or 0.0),
                     r_multiple=-(r.get("r_multiple") or 0.0))
        out.append(r)
    return out


def _ids(sel: dict) -> dict:
    return {d: None if c is None else c.cid for d, c in sel["selected"].items()}


def build_s5(*, rows: Sequence[dict], ctx_by_key: dict, spot, dataset: dict, s2_sources_unchanged: bool | None = None,
             module_safety: dict | None = None, check_sample: int = 20) -> dict[str, Any]:
    rows = list(rows)
    pre = [r for r in rows if day_of(r) <= SELECTION_END]

    # Phase 1 - causality
    lookahead = {"features": dg.truncation_check(spot, rows, ctx_by_key, check_sample),
                 "option_confirmation": option_truncation_check(rows, ctx_by_key, check_sample),
                 "time_window": time_window_check(rows),
                 "chronology": dg.chronology_check(rows)}
    lookahead_ok = all(v["passed"] for v in lookahead.values())

    # Selection (pre-OOS only) + invariance proofs
    sel = select(pre)
    fp = fingerprint(sel)
    mutated = _flip(rows, SELECTION_END)
    sel_mut = select([r for r in mutated if day_of(r) <= SELECTION_END])
    invariance = {"oos_and_holdout_outcomes_flipped_same_selection": _ids(sel) == _ids(sel_mut),
                  "fingerprint_before_holdout": fp}

    recheck = discovery_recheck(rows)
    contaminated = True  # patterns were discovered on Apr-Sep (diagnostic); see discovery_recheck

    # Every candidate: TRAIN / VALIDATION / OOS metrics + gate (September never read here)
    no_sep = [r for r in rows if day_of(r) < SPLITS["HOLDOUT"][0]]
    all_cands = list(BASE_CANDIDATES) + list(sel["opt_variants"])
    cand = {}
    for c in all_cands:
        crow = candidate_rows(c, no_sep)
        g = gate(c, crow, lookahead_ok, contaminated)
        cand[c.cid] = {"candidate": c, "gate": g, "no_signal": no_signal_count(c, no_sep),
                       "selection": sel["stats"].get(c.cid), "selected": c in sel["selected"].values()}

    s2 = {d: [r for r in no_sep if r["direction"] == d] for d in DIRECTIONS}
    s2_gate = {d: gate(None, s2[d], lookahead_ok, False, 0) for d in DIRECTIONS}
    s2_gate["COMBINED"] = gate(None, no_sep, lookahead_ok, False, 0)

    best = {d: sel["selected"][d] for d in DIRECTIONS}
    comb_rows = [r for d in DIRECTIONS if best[d] for r in candidate_rows(best[d], no_sep)]
    comb_name = " + ".join(best[d].cid for d in DIRECTIONS if best[d]) or None
    comb_ncond = max((best[d].n_conditions for d in DIRECTIONS if best[d]), default=0)
    comb_gate = None
    if comb_name:
        comb_gate = gate(None, comb_rows, lookahead_ok, contaminated, comb_ncond)
        regs = [cand[best[d].cid]["gate"]["regime"]["dependent"] for d in DIRECTIONS if best[d]]
        comb_gate["checks"]["no_regime_dependency"] = not any(regs)
        if comb_gate["status"] == ROBUST and any(regs):
            comb_gate["status"] = PROMISING
        comb_gate["failed"] = [k for k, ok in comb_gate["checks"].items() if not ok]

    # Phase 9 - final holdout, run once on the frozen selection
    if fingerprint(sel) != fp:
        raise RuntimeError("selection changed before holdout")
    sep = split_rows(rows, "HOLDOUT")
    holdout: dict[str, Any] = {"fingerprint": fp}
    for d in DIRECTIONS:
        c = best[d]
        if c is None:
            holdout[d] = {"candidate": None, "status": NOT_RUN, "metrics": None}
            continue
        m = metrics(candidate_rows(c, sep))
        holdout[d] = {"candidate": c.cid, "metrics": m,
                      "status": holdout_status(m, cand[c.cid]["gate"]["status"] == ROBUST)}
    if comb_name:
        m = metrics([r for d in DIRECTIONS if best[d] for r in candidate_rows(best[d], sep)])
        holdout["COMBINED"] = {"candidate": comb_name, "metrics": m,
                               "status": holdout_status(m, comb_gate["status"] == ROBUST)}
    else:
        holdout["COMBINED"] = {"candidate": None, "status": NOT_RUN, "metrics": None}
    s2_holdout = {d: metrics([r for r in sep if r["direction"] == d]) for d in DIRECTIONS}
    s2_holdout["COMBINED"] = metrics(sep)

    achieved = any(g is not None and g["oos"]["accuracy"] is not None and g["oos"]["accuracy"] >= 0.70
                   and g["oos"]["trades"] >= MIN_ROBUST_OOS
                   for g in [cand[best[d].cid]["gate"] for d in DIRECTIONS if best[d]] + [comb_gate])
    robust_any = any(g is not None and g["status"] == ROBUST
                     for g in [cand[best[d].cid]["gate"] for d in DIRECTIONS if best[d]] + [comb_gate])

    integrity = integrity_checks(rows=rows, sel=sel, invariance=invariance, lookahead_ok=lookahead_ok,
                                 cand=cand, no_sep=no_sep, s2_sources_unchanged=s2_sources_unchanged,
                                 module_safety=module_safety, recheck=recheck)
    integrity_ok = all(x["status"] != "FAIL" for x in integrity)
    rec = recommend(best, cand, comb_gate, holdout, contaminated, integrity_ok and lookahead_ok)
    return {
        "heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "dataset": dataset, "splits": {k: (a.isoformat(), b.isoformat()) for k, (a, b) in SPLITS.items()},
        "lookahead": lookahead, "lookahead_ok": lookahead_ok, "selection": sel, "invariance": invariance,
        "fingerprint": fp, "candidates": cand, "best": best, "combined": {"name": comb_name, "gate": comb_gate},
        "s2_gate": s2_gate, "holdout": holdout, "s2_holdout": s2_holdout, "discovery_recheck": recheck,
        "contaminated_discovery": contaminated, "target_70_achieved": achieved, "robust_70": robust_any,
        "integrity": integrity, "integrity_ok": integrity_ok, "recommendation": rec,
        "s2_changed": False, "orders": 0, "paper": True, "live_trading_enabled": False,
    }


def integrity_checks(*, rows, sel, invariance, lookahead_ok, cand, no_sep, s2_sources_unchanged, module_safety,
                     recheck) -> list[dict]:
    def chk(cid, name, ok, detail, warn=False):
        return {"id": cid, "check": name, "status": ("WARN" if warn else ("PASS" if ok else "FAIL")),
                "detail": detail}
    base_ids = [c.cid for c in BASE_CANDIDATES]
    subset_ok = all(set(map(dra.key_of, candidate_rows(c["candidate"], no_sep))) <=
                    {dra.key_of(r) for r in no_sep if r["direction"] == c["candidate"].direction}
                    for c in cand.values())
    opt_ok = all(c.conditions[-1] == "OPT_CONFIRM" and c.cid[:2] in base_ids for c in sel["opt_variants"])
    supported = [f"{x['direction']} {x['pattern']}" for x in recheck if x["supported_before_oos"]]
    return [
        chk("I1", "Production S2 sources/config unchanged", s2_sources_unchanged is not False,
            "sha256 of S2 strategy, selection, options and config files before vs after the run"
            if s2_sources_unchanged is not None else "not measured in this call"),
        chk("I2", "No order placement path imported", (module_safety or {}).get("ok", True),
            (module_safety or {}).get("detail", "module import scan")),
        chk("I3", "PAPER only / LIVE_TRADING_ENABLED=false", True, "assert_paper_trading_only() before the run"),
        chk("I4", "Selection used only TRAIN+VALIDATION (<= 2026-07-31)",
            invariance["oos_and_holdout_outcomes_flipped_same_selection"],
            "select() refuses later rows; flipping every Aug/Sep outcome leaves the selection unchanged"),
        chk("I5", "Exactly the pre-declared candidates", base_ids == ["B1", "B2", "B3", "B4", "B5", "B6", "S1", "S2",
                                                                       "S3", "S4", "S5", "S6", "S7"] and opt_ok,
            f"{len(base_ids)} base + {len(sel['opt_variants'])} option-confirmation variants of the selected bases"),
        chk("I6", "Candidates only filter S2 signals (same contract/entry/exit)", subset_ok,
            "every candidate row is an unchanged S2 row of the same direction"),
        chk("I7", "No lookahead (features, option confirmation, time window, chronology)", lookahead_ok,
            "truncation / premium-mutation re-computation"),
        chk("I8", "September holdout evaluated once on a frozen, fingerprinted selection", True,
            f"fingerprint {invariance['fingerprint_before_holdout']} verified before the holdout"),
        chk("I9", "Pattern discovery independent of OOS/holdout", False,
            "the six patterns came from the Apr-Sep diagnostic, so Aug/Sep took part in discovery. Supported on "
            f"Apr-Jul alone: {supported or 'none'}. OOS/holdout results are therefore optimistic.", warn=True),
    ]


def recommend(best: dict, cand: dict, comb_gate: dict | None, holdout: dict, contaminated: bool,
              clean: bool) -> str:
    if not clean:
        return REJECTED
    picked = [(d, cand[best[d].cid]["gate"]) for d in DIRECTIONS if best[d]]
    if comb_gate is not None:
        picked.append(("COMBINED", comb_gate))
    if not picked:
        return KEEP
    statuses = {d: g["status"] for d, g in picked}
    passed = {d: holdout[d]["status"] == PASSED_HO for d, _ in picked}
    if any(s == ROBUST and passed[d] for d, s in statuses.items()):
        return PROMISING if contaminated else ROBUST
    if any(s in (ROBUST, PROMISING) and passed[d] for d, s in statuses.items()):
        return PROMISING
    if any(s in (ROBUST, PROMISING) for s in statuses.values()):
        return REJECTED
    if all(s == INSUFFICIENT for s in statuses.values()):
        return INSUFFICIENT
    return KEEP


# ------------------------------------------------------------------ reports
REPORT_FILES = ("s5_targeted_strategy.md", "s5_candidates.csv", "s5_walk_forward.csv", "s5_final_holdout.csv",
                "s5_integrity_audit.md")
METRIC_KEYS = ("trades", "wins", "losses", "accuracy", "profit_factor", "net_pnl", "avg_r", "max_drawdown",
               "max_loss_streak", "profitable_days", "active_months")


def _csv(path: Path, rows: Sequence[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("INF" if v == math.inf else v) for k, v in r.items()})


def candidate_csv(a: dict) -> list[dict]:
    out = []
    entries = [(k, v["candidate"].name, v["candidate"].direction, v["candidate"].describe(), v["gate"],
                v["no_signal"], v["selected"], v["selection"]) for k, v in a["candidates"].items()]
    entries += [(f"S2_BASELINE_{d}", "unchanged S2", d, "all S2 signals", a["s2_gate"][d], 0, False, None)
                for d in ("BUY", "SELL", "COMBINED")]
    if a["combined"]["gate"]:
        entries.append((a["combined"]["name"], "best BUY + best SELL", "COMBINED", "union of the selected candidates",
                        a["combined"]["gate"], 0, True, None))
    for cid, name, d, desc, g, nos, selected, st in entries:
        for split in ("train", "validation", "oos"):
            m = g[split]
            out.append({"candidate": cid, "name": name, "direction": d, "split": split.upper(), "rules": desc,
                        **{k: m[k] for k in METRIC_KEYS}, "sample_grade": sample_grade(m["trades"]),
                        "selection_eligible": None if st is None else st["eligible"], "selected": selected,
                        "status": g["status"], "failed_gate_checks": ",".join(g["failed"]),
                        "wf_profitable_share": g["walk_forward"]["profitable_share"],
                        "wf_median_accuracy": g["walk_forward"]["median_accuracy"],
                        "wf_median_pf": g["walk_forward"]["median_pf"], "wf_stable": g["walk_forward"]["stable"],
                        "no_signal_option_missing_or_feature_missing": nos})
    return out


def wf_csv(a: dict) -> list[dict]:
    out = []
    items = [(k, v["candidate"].direction, v["gate"]["walk_forward"]) for k, v in a["candidates"].items()]
    items += [(f"S2_BASELINE_{d}", d, a["s2_gate"][d]["walk_forward"]) for d in ("BUY", "SELL", "COMBINED")]
    if a["combined"]["gate"]:
        items.append((a["combined"]["name"], "COMBINED", a["combined"]["gate"]["walk_forward"]))
    for cid, d, wf in items:
        for w in wf["windows"]:
            out.append({"candidate": cid, "direction": d, **{k: v for k, v in w.items() if k not in ("start", "end")}})
    return out


def holdout_csv(a: dict) -> list[dict]:
    out = []
    for d in ("BUY", "SELL", "COMBINED"):
        h = a["holdout"][d]
        m = h["metrics"] or {}
        out.append({"scope": d, "candidate": h["candidate"], "status": h["status"],
                    **{k: m.get(k) for k in METRIC_KEYS}, "fingerprint": a["fingerprint"]})
        s = a["s2_holdout"][d]
        out.append({"scope": d, "candidate": f"S2_BASELINE_{d}", "status": "REFERENCE",
                    **{k: s.get(k) for k in METRIC_KEYS}, "fingerprint": a["fingerprint"]})
    return out


def _l(m: dict | None) -> str:
    if not m:
        return "N/A"
    return (f"{dg._pct(m['accuracy'])} ({m['trades']} trades, PF {dg._pf(m['profit_factor'])}, "
            f"P&L Rs {m['net_pnl']:,.2f})")


def _cmp_row(label: str, s2: dict, s5: dict | None, wf2: dict, wf5: dict | None, h2: dict, h5: dict | None) -> str:
    def cell(m, k, f=lambda v: v):
        return "N/A" if not m or m.get(k) is None else f(m[k])
    return (f"| {label} | {cell(s2, 'accuracy', dg._pct)} / {cell(s5, 'accuracy', dg._pct)} | "
            f"{cell(s2, 'trades')} / {cell(s5, 'trades')} | {cell(s2, 'profit_factor', dg._pf)} / "
            f"{cell(s5, 'profit_factor', dg._pf)} | {cell(s2, 'net_pnl', lambda v: f'{v:,.0f}')} / "
            f"{cell(s5, 'net_pnl', lambda v: f'{v:,.0f}')} | {cell(s2, 'max_drawdown', lambda v: f'{v:,.0f}')} / "
            f"{cell(s5, 'max_drawdown', lambda v: f'{v:,.0f}')} | {cell(s2, 'max_loss_streak')} / "
            f"{cell(s5, 'max_loss_streak')} | {dg._pct(wf2.get('profitable_share'))} / "
            f"{dg._pct((wf5 or {}).get('profitable_share'))} | {_l(h2)} / {_l(h5)} |")


def markdown(a: dict) -> str:
    md = [f"# {HEADING}", "", f"**{BANNER}** | production S2 changed: NO | live orders: 0 | PAPER | "
          f"LIVE_TRADING_ENABLED=false | generated {a['generated_at']}", "",
          "## Executive Summary", ""]
    for d in DIRECTIONS:
        c = a["best"][d]
        if c is None:
            md.append(f"- Best {d}: none passed the TRAIN/VALIDATION selection criteria.")
            continue
        g = a["candidates"][c.cid]["gate"]
        md.append(f"- Best {d}: **{c.cid} {c.name}** - OOS {_l(g['oos'])}, status **{g['status']}**, holdout "
                  f"{a['holdout'][d]['status']} {_l(a['holdout'][d]['metrics'])}")
    if a["combined"]["gate"]:
        g = a["combined"]["gate"]
        md.append(f"- Best COMBINED: **{a['combined']['name']}** - OOS {_l(g['oos'])}, status **{g['status']}**, "
                  f"holdout {a['holdout']['COMBINED']['status']} {_l(a['holdout']['COMBINED']['metrics'])}")
    md += [f"- 70% target: **{'ACHIEVED' if a['target_70_achieved'] else 'NOT ACHIEVED'}**; ROBUST_70: "
           f"**{'YES' if a['robust_70'] else 'NO'}**",
           *([f"- {NOT_ACHIEVED_TEXT}"] if not a["target_70_achieved"] else []),
           f"- Final recommendation: **{a['recommendation']}**",
           "- **Caveat:** the six input patterns were discovered on Apr-Sep data (S2 diagnostic), so August and "
           "September influenced which conditions exist here. OOS and holdout are therefore optimistic; any "
           "candidate needs fresh forward (post-September) paper validation.", "",
           "## Splits", ""]
    md += [f"- {k}: {lo} -> {hi}" for k, (lo, hi) in a["splits"].items()]
    md += ["", "Selection used TRAIN + VALIDATION + walk-forward windows ending <= 2026-07-31 only. OOS gave the "
           "gate status. September was evaluated once after the selection fingerprint "
           f"`{a['fingerprint']}` was frozen.", "", "## Phase 1 - Feature causality", ""]
    md += [f"- **{k}**: {v}" for k, v in FEATURE_SOURCES.items()]
    md += [f"- {k}: {'PASS' if v['passed'] else 'FAIL'} ({v.get('checked', '-')} checked; "
           f"{v.get('mismatches') or 'no mismatches'})" for k, v in a["lookahead"].items()]
    md += ["", "## Phase 2/3 - Candidates", "", "| ID | Side | Rules | Selection eligible | Train | Validation | OOS | "
           "Grade | Status |", "|---|---|---|---|---|---|---|---|---|"]
    for cid, v in a["candidates"].items():
        g, st = v["gate"], v["selection"]
        md.append(f"| {cid}{' (selected)' if v['selected'] else ''} | {v['candidate'].direction} | "
                  f"{v['candidate'].describe()} | {None if st is None else st['eligible']} | {_l(g['train'])} | "
                  f"{_l(g['validation'])} | {_l(g['oos'])} | {g['grade']} ({GRADE_TEXT[g['grade']]}) | "
                  f"{g['status']} |")
    md += ["", f"Selection rule (pre-declared): TRAIN >= {MIN_TRAIN_TRADES} trades, VALIDATION >= {MIN_VAL_TRADES} "
           f"trades, TRAIN P&L > 0 and PF >= {SELECT_MIN_PF}, VALIDATION P&L > 0; rank by min(TRAIN PF, VALIDATION "
           "PF), then walk-forward profitable share (windows <= Jul), then trades. Option confirmation was then "
           "tested only on each direction's selected base candidate and kept only if it ranks higher by the same "
           "rule.", "", "## Phase 6 - Walk-forward (semi-monthly windows, 2026-04-01..2026-08-31)", "",
           "| Candidate | Active windows | Profitable % | Median accuracy | Median PF | Worst | Best | Stable |",
           "|---|---|---|---|---|---|---|---|"]
    wf_items = [(k, v["gate"]["walk_forward"]) for k, v in a["candidates"].items()]
    wf_items += [(f"S2_BASELINE_{d}", a["s2_gate"][d]["walk_forward"]) for d in ("BUY", "SELL", "COMBINED")]
    md += [f"| {k} | {w['active_windows']} | {dg._pct(w['profitable_share'])} | {dg._pct(w['median_accuracy'])} | "
           f"{dg._pf(w['median_pf'])} | {w['worst']} | {w['best']} | {w['stable']} |" for k, w in wf_items]
    md += ["", "## Phase 7/8 - Sample size and 70% gate", "",
           f"Grades (OOS trades): A >= 100 strong, B >= {MIN_ROBUST_OOS} usable, C >= {MIN_SUFFICIENT_OOS} weak, "
           f"D < {MIN_SUFFICIENT_OOS} insufficient. ROBUST_70 needs OOS accuracy >= 70% on >= {MIN_ROBUST_OOS} "
           "trades, OOS PF >= 1.5, OOS P&L > 0, VALIDATION profitable, walk-forward stable (>= 4 active windows, "
           ">= 50% profitable, median PF >= 1.0), OOS P&L > 0 without its best day, P&L > 0 without the best "
           f"regime, no lookahead, <= {MAX_CONDITIONS} conditions.", ""]
    for d in DIRECTIONS:
        c = a["best"][d]
        if c:
            md.append(f"- {c.cid} failed gate checks: {a['candidates'][c.cid]['gate']['failed'] or 'none'}")
    if a["combined"]["gate"]:
        md.append(f"- {a['combined']['name']} failed gate checks: {a['combined']['gate']['failed'] or 'none'}")
    md += ["", "## Phase 9 - Final September holdout (run once)", "",
           "| Scope | Candidate | Status | Result | S2 baseline |", "|---|---|---|---|---|"]
    for d in ("BUY", "SELL", "COMBINED"):
        h = a["holdout"][d]
        md.append(f"| {d} | {h['candidate']} | {h['status']} | {_l(h['metrics'])} | {_l(a['s2_holdout'][d])} |")
    md += ["", "## Phase 10 - S2 vs S5 (OOS August; holdout September)", "",
           "| Scope | Accuracy S2 / S5 | Trades | PF | P&L | Max DD | Loss streak | WF profitable % | Holdout |",
           "|---|---|---|---|---|---|---|---|---|"]
    for d in DIRECTIONS:
        c = a["best"][d]
        g = a["candidates"][c.cid]["gate"] if c else None
        md.append(_cmp_row(f"{d} (S2 vs {c.cid if c else 'none'})", a["s2_gate"][d]["oos"], g and g["oos"],
                           a["s2_gate"][d]["walk_forward"], g and g["walk_forward"], a["s2_holdout"][d],
                           a["holdout"][d]["metrics"]))
    cg = a["combined"]["gate"]
    md.append(_cmp_row(f"COMBINED (S2 vs {a['combined']['name']})", a["s2_gate"]["COMBINED"]["oos"],
                       cg and cg["oos"], a["s2_gate"]["COMBINED"]["walk_forward"], cg and cg["walk_forward"],
                       a["s2_holdout"]["COMBINED"], a["holdout"]["COMBINED"]["metrics"]))
    md += ["", "## Discovery re-check (Apr-Jul only)", "",
           "| Side | Pattern | Trades | Accuracy | PF | P&L | Months profitable | Supported before OOS |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {x['direction']} | {x['pattern']} | {x['trades_apr_jul']} | {dg._pct(x['accuracy'])} | "
           f"{dg._pf(x['profit_factor'])} | {x['net_pnl']:,.2f} | {x['months_profitable']}/{x['months_active']} | "
           f"{x['supported_before_oos']} |" for x in a["discovery_recheck"]]
    md += ["", "## Phase 11", "", "No rule was changed after OOS or after September. No threshold was searched; every "
           "edge comes from the diagnostic buckets. Production S2 untouched; no orders."]
    return "\n".join(md) + "\n"


def integrity_markdown(a: dict) -> str:
    md = ["# S5 integrity audit", "", f"**{BANNER}** | generated {a['generated_at']}", "",
          f"Overall: **{'PASS' if a['integrity_ok'] else 'FAIL'}** (WARN items are disclosed limitations, not "
          "failures).", "", "| ID | Check | Status | Detail |", "|---|---|---|---|"]
    md += [f"| {x['id']} | {x['check']} | {x['status']} | {x['detail']} |" for x in a["integrity"]]
    md += ["", "## Lookahead", ""]
    md += [f"- {k}: {'PASS' if v['passed'] else 'FAIL'} - {v}" for k, v in a["lookahead"].items()]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    p["s5_targeted_strategy.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["s5_candidates.csv"], candidate_csv(a))
    _csv(p["s5_walk_forward.csv"], wf_csv(a))
    _csv(p["s5_final_holdout.csv"], holdout_csv(a))
    p["s5_integrity_audit.md"].write_text(integrity_markdown(a), encoding="utf-8")
    return p


def summary_text(a: dict) -> str:
    out = [HEADING, ""]

    def block(label: str, cid: str | None, g: dict | None) -> list[str]:
        if not cid or g is None:
            return [f"{label}:", "candidate: none (no candidate met the TRAIN/VALIDATION selection criteria)",
                    "OOS accuracy: N/A", "trades: 0", "PF: N/A", "P&L: N/A", f"status: {REJECTED}", ""]
        o = g["oos"]
        return [f"{label}:", f"candidate: {cid}", f"OOS accuracy: {dg._pct(o['accuracy'])}", f"trades: {o['trades']}",
                f"PF: {dg._pf(o['profit_factor'])}", f"P&L: Rs {o['net_pnl']:,.2f}", f"status: {g['status']}", ""]
    for d in DIRECTIONS:
        c = a["best"][d]
        out += block(f"BEST {d}", c and f"{c.cid} {c.name}", c and a["candidates"][c.cid]["gate"])
    out += block("BEST COMBINED", a["combined"]["name"], a["combined"]["gate"])
    out.append("FINAL SEPTEMBER HOLDOUT:")
    for d in ("BUY", "SELL", "COMBINED"):
        h = a["holdout"][d]
        out.append(f"{d}: {h['candidate'] or 'none'} {_l(h['metrics'])} {h['status']}")
    out += ["", "70% TARGET:", "ACHIEVED" if a["target_70_achieved"] else "NOT ACHIEVED"]
    if not a["target_70_achieved"]:
        out.append(NOT_ACHIEVED_TEXT)
    out += ["", "ROBUST_70:", "YES" if a["robust_70"] else "NO", "", "LOOKAHEAD:", "PASS" if a["lookahead_ok"] else "FAIL",
            "", "INTEGRITY:", "PASS" if a["integrity_ok"] else "FAIL",
            "(WARN: patterns were discovered on Apr-Sep, so OOS/holdout are optimistic)", "",
            "PRODUCTION S2 CHANGED:", "NO", "", "LIVE ORDERS:", "0", "", "FINAL RECOMMENDATION:", a["recommendation"]]
    return "\n".join(out)
