"""S7 CLEAN MARKET REGIME / NO-TRADE FILTER LAB (PAPER | RESEARCH ONLY | NO ORDERS).

Question: WHEN should the unchanged S2 strategy trade, and when should it skip the market?

A regime gate only REMOVES S2 signals: signal, contract, entry (next candle open), stop, target and exit are the exact
S2 replay. Gates read causal signal-candle regime features only (completed NIFTY candles <= signal; the S6 extras are
reused). BUY and SELL gates are separate families (max 20 each, pre-declared below).

Chronology (historical backtest accuracy, never live accuracy):
  TRAIN 2026-04-01..06-30      regime discovery + shortlist (TRAIN only)
  VALIDATION 07-01..07-31      PASS / DEGRADED / REJECTED / INSUFFICIENT_DATA; gates are never modified
  OOS 08-01..08-31             frozen gates only, once
  FINAL HOLDOUT 09-01..09-30   frozen gates only, once, NOT USED FOR SELECTION
``select`` raises if it receives any row after 2026-07-31.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import strategy_improvement_lab as lab
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s5_targeted_strategy as s5
from src.research import s6_clean_strategy_lab as s6

HEADING = "S7 REGIME LAB COMPLETE"
BANNER = cb.BANNER
IST = s6.IST
SPLITS = cb.SPLITS
SPLIT_NAMES = s6.SPLIT_NAMES
SCOPES, DIRECTIONS = cb.SCOPES, s6.DIRECTIONS
PRE_HOLDOUT = s6.PRE_HOLDOUT
TRAIN_END, SELECTION_END = s6.TRAIN_END, s6.SELECTION_END
TRAIN_MONTHS = ("2026-04", "2026-05", "2026-06")
UNAVAILABLE = s6.UNAVAILABLE
HOLDOUT_LABEL = s6.HOLDOUT_LABEL

# Pre-declared thresholds (fixed before any result was computed; never fitted).
MAX_GATES_PER_SIDE = 20
MIN_TRAIN_TRADES, MIN_VAL_TRADES, MIN_OOS_TRADES = 30, 10, 30
MIN_RETENTION = 0.20            # a gate must keep >= 20% of the side's S2 trades
MIN_EXPECTANCY_GAIN = 0.05      # avg R vs the S2 baseline of the same side and split
MIN_PF_GAIN = 0.10
MIN_DD_REDUCTION = 0.25         # drawdown reduction counted as meaningful
STRONG_DD_REDUCTION = 0.50      # risk benefit that can justify OOS PF < 1
MIN_MONTH_BEATS = 2             # gate expectancy beats the baseline in >= 2 of the 3 TRAIN months
MIN_GROUP_TRADES = s6.MIN_GROUP_TRADES
COLLAPSE_PF_SHARE = 0.5
RANDOM_DRAWS, RANDOM_SEED = 1000, 7
TARGET = 0.70

PASS, DEGRADED, REJECTED, INSUFFICIENT = "PASS", "DEGRADED", "REJECTED", "INSUFFICIENT_DATA"
PROMISING, OVERFIT = "PROMISING", "OVERFIT_RISK"
NO_VALID = "NO VALID REGIME FILTER"

_pf, _ev, fv, split_m, _not = s6._pf, s6._ev, s6.fv, s6.split_m, s6._not


# ------------------------------------------------------------------ causal regime features
ATR_PCT_EDGES = (0.05, 0.08, 0.12)   # ATR14 / close, % (5m NIFTY)
EMA_POINTS_EDGES = (-20.0, 0.0, 20.0)


def breakout_class(r: dict) -> str:
    v = dg._num(r["features"].get("break_atr"))
    return UNAVAILABLE if v is None else "WEAK" if v < 0.25 else "STRONG" if v >= 0.75 else "NORMAL"


def choppiness(r: dict) -> str:
    sc = dg._num(fv(r, "sign_changes"))
    return UNAVAILABLE if sc is None else "CHOPPY" if sc >= lab.CHOPPY_SIGN_CHANGES else "NON_CHOPPY"


def market_structure(r: dict) -> str:
    """Signal relative to the causal trend regime: continuation / reversal / range-chop."""
    reg = fv(r, "market_regime")
    if reg in (None, UNAVAILABLE):
        return UNAVAILABLE
    if reg in ("RANGE", "CHOPPY"):
        return "RANGE_CHOP"
    with_trend = (reg == "TREND_UP") == (r["direction"] == "BUY")
    return "TREND_CONTINUATION" if with_trend else "TREND_REVERSAL"


def atr_pct(r: dict) -> float | None:
    a, c = dg._num(r["features"].get("atr14")), dg._num(r["features"].get("close"))
    return a / c * 100 if a is not None and c else None


def ema_points(r: dict) -> float | None:
    e20, e50 = dg._num(r["features"].get("ema20")), dg._num(r["features"].get("ema50"))
    return None if e20 is None or e50 is None else e20 - e50


DIMENSIONS: dict[str, Callable[[dict], Any]] = {
    "trend_direction": lambda r: fv(r, "market_regime"),
    "ema_relationship": lambda r: r["features"].get("ema20_vs_ema50"),
    "ema_separation_points": lambda r: dg.bucket(ema_points(r), EMA_POINTS_EDGES),
    "ema_separation_atr": lambda r: r["features"].get("ema_sep_atr_bucket"),
    "vwap_side": lambda r: r["features"].get("vwap_side"),
    "vwap_distance_atr": lambda r: r["features"].get("vwap_dist_atr_bucket"),
    "rsi": lambda r: fv(r, "rsi_bucket"),
    "atr_regime": lambda r: r["features"].get("vol_regime"),
    "atr_pct_of_price": lambda r: dg.bucket(atr_pct(r), ATR_PCT_EDGES),
    "atr_percentile": lambda r: r["features"].get("atr_pctile_bucket"),
    "volume_level": lambda r: fv(r, "volume_level"),
    "breakout_strength": breakout_class,
    "previous_range_atr": lambda r: fv(r, "orb_range_bucket"),
    "time_of_day": lambda r: r["features"].get("time_window"),
    "choppiness": choppiness,
    "market_structure": market_structure,
}
NOT_AVAILABLE_FEATURES = {"vwap_slope": "no causal VWAP slope exists in the project; not invented here"}


# ------------------------------------------------------------------ gate conditions (signal candle only)
def _volume_ok(r: dict) -> bool | None:
    v = fv(r, "volume_level")
    return None if v in (None, UNAVAILABLE) else v != "LOW"


def _non_choppy(r: dict) -> bool | None:
    c = choppiness(r)
    return None if c == UNAVAILABLE else c == "NON_CHOPPY"


def _not_high_vol(r: dict) -> bool | None:
    v = r["features"].get("vol_regime")
    return None if v in (None, UNAVAILABLE) else v != "HIGH_VOLATILITY"


def _before_1300(r: dict) -> bool:
    return dra._ts(r["signal_time"]).time() < time(13, 0)


CONDITIONS: dict[str, tuple[str, Callable[[dict], bool | None]]] = {
    **{k: s6.CONDITIONS[k] for k in ("EMA_UP", "EMA_DOWN", "VWAP_ABOVE", "VWAP_BELOW", "ATR_TREND_UP",
                                     "ATR_TREND_DOWN")},
    "TREND_UP": ("market regime TREND_UP (close > EMA20 > EMA50 and EMA20 rising)", s6._eq("market_regime",
                                                                                           "TREND_UP")),
    "TREND_DOWN": ("market regime TREND_DOWN (close < EMA20 < EMA50 and EMA20 falling)",
                   s6._eq("market_regime", "TREND_DOWN")),
    "EMA_SEP_POS": ("(EMA20 - EMA50) / ATR14 >= 0.5", s6._between("ema_sep_atr", 0.5)),
    "EMA_SEP_NEG": ("(EMA20 - EMA50) / ATR14 < -0.5", s6._between("ema_sep_atr", None, -0.5)),
    "VOLUME_OK": ("bought option relative volume NORMAL or HIGH (>= 0.8; index volume is zero)", _volume_ok),
    "NON_CHOPPY": (f"fewer than {lab.CHOPPY_SIGN_CHANGES} close-to-close sign changes so far today", _non_choppy),
    "NOT_HIGH_VOL": ("ATR regime is not HIGH_VOLATILITY", _not_high_vol),
    "BEFORE_1300": ("signal candle starts before 13:00", _before_1300),
}


@dataclass(frozen=True)
class Gate:
    gid: str
    name: str
    direction: str
    rationale: str
    conditions: tuple[str, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.conditions)

    def describe(self) -> str:
        return ("TRADE only if " + " AND ".join(CONDITIONS[c][0] for c in self.conditions)) if self.conditions \
            else "all S2 signals (baseline)"


def _gates() -> tuple[Gate, ...]:
    out = []
    for d, p, t, v, e, s, st, w in (("BUY", "S7-B", "TREND_UP", "VWAP_ABOVE", "EMA_UP", "EMA_SEP_POS",
                                     "ATR_TREND_UP", "bullish"),
                                    ("SELL", "S7-S", "TREND_DOWN", "VWAP_BELOW", "EMA_DOWN", "EMA_SEP_NEG",
                                     "ATR_TREND_DOWN", "bearish")):
        spec = [("0", "S2_ALL", "baseline reference", ()),
                ("1", t, f"breakouts work in a {w} trend", (t,)),
                ("2", v, "price already on the trade side of the session average", (v,)),
                ("3", e, f"{w} EMA structure", (e,)),
                ("4", s, "clear EMA separation in ATR units", (s,)),
                ("5", f"{t}+{v}", "trend and VWAP agree", (t, v)),
                ("6", f"{t}+{e}", "trend and EMA structure agree", (t, e)),
                ("7", f"{t}+VOLUME_OK", "trend with participation", (t, "VOLUME_OK")),
                ("8", "NON_CHOPPY", "skip whipsaw sessions", ("NON_CHOPPY",)),
                ("9", f"{t}+NON_CHOPPY", "trend without whipsaw", (t, "NON_CHOPPY")),
                ("10", st, "strong trend regime (|EMA separation| >= 1 ATR)", (st,)),
                ("11", "NOT_HIGH_VOL", "skip disorderly high-volatility conditions", ("NOT_HIGH_VOL",)),
                ("12", "BEFORE_1300", "skip the low-follow-through afternoon", ("BEFORE_1300",))]
        out += [Gate(f"{p}{n}", f"{d}_{name}", d, why, conds) for n, name, why, conds in spec]
    return tuple(out)


GATES = _gates()


def gate_decision(g: Gate, r: dict) -> tuple:
    return tuple(CONDITIONS[c][1](r) for c in g.conditions)


def gate_rows(g: Gate, rows: Sequence[dict]) -> list[dict]:
    """S2 rows of the gate's side where every condition is True (None -> NO-TRADE). Rows are never modified."""
    return [r for r in rows if r["direction"] == g.direction and all(v is True for v in gate_decision(g, r))]


def side_rows(rows: Sequence[dict], d: str) -> list[dict]:
    return list(rows) if d == "COMBINED" else [r for r in rows if r["direction"] == d]


# ------------------------------------------------------------------ coverage / comparisons
def compare(base: dict, filt: dict) -> dict[str, Any]:
    bt, ft = base["total_trades"], filt["total_trades"]
    bw, fw, br, fr = base["win_rate"], filt["win_rate"], base["average_r"], filt["average_r"]
    bd, fd = base["max_drawdown"], filt["max_drawdown"]
    return {"baseline_trades": bt, "filtered_trades": ft, "retention": round(ft / bt, 4) if bt else None,
            "trade_reduction": round(1 - ft / bt, 4) if bt else None,
            "accuracy_change_pp": None if bw is None or fw is None else round((fw - bw) * 100, 2),
            "pf_change": round(_pf(filt) - _pf(base), 4),
            "expectancy_change_r": None if br is None or fr is None else round(fr - br, 4),
            "dd_reduction": round((bd - fd) / bd, 4) if bd > 0 else None,
            "pnl_change": round(filt["net_pnl"] - base["net_pnl"], 2)}


def month_stability(g: Gate, rows: Sequence[dict]) -> dict[str, Any]:
    out, beats = {}, 0
    for m in TRAIN_MONTHS:
        base = cb.metrics([r for r in side_rows(rows, g.direction) if str(r["session_date"])[:7] == m])
        filt = cb.metrics([r for r in gate_rows(g, rows) if str(r["session_date"])[:7] == m])
        better = (filt["total_trades"] >= MIN_GROUP_TRADES and filt["average_r"] is not None
                  and base["average_r"] is not None and filt["average_r"] > base["average_r"])
        beats += better
        out[m] = {"trades": filt["total_trades"], "win_rate": filt["win_rate"], "profit_factor": filt["profit_factor"],
                  "average_r": filt["average_r"], "net_pnl": filt["net_pnl"], "baseline_average_r": base["average_r"],
                  "beats_baseline": better}
    pnl = {m: v["net_pnl"] for m, v in out.items()}
    total, best = sum(pnl.values()), max(pnl.values(), default=0.0)
    one_month = total > 0 and total - max(best, 0.0) <= 0
    return {"months": out, "beats": beats, "profitable_months": sum(p > 0 for p in pnl.values()),
            "one_month_dependent": one_month,
            "label": OVERFIT if beats < MIN_MONTH_BEATS or one_month else "MULTI_MONTH"}


# ------------------------------------------------------------------ TRAIN shortlist / VALIDATION / selection
def train_eval(g: Gate, rows: Sequence[dict]) -> dict[str, Any]:
    base = split_m(side_rows(rows, g.direction), "TRAIN")
    m = split_m(gate_rows(g, rows), "TRAIN")
    cov, ms = compare(base, m), month_stability(g, rows)
    checks = {
        "not_baseline": bool(g.conditions),
        "meaningful_trade_count": m["total_trades"] >= MIN_TRAIN_TRADES,
        "reasonable_retention": (cov["retention"] or 0.0) >= MIN_RETENTION,
        "expectancy_improved": (cov["expectancy_change_r"] or -math.inf) >= MIN_EXPECTANCY_GAIN,
        "pf_or_dd_improved": cov["pf_change"] >= MIN_PF_GAIN or (cov["dd_reduction"] or 0.0) >= MIN_DD_REDUCTION,
        "multi_month_edge": ms["label"] != OVERFIT,
    }
    ar = m["average_r"] if m["average_r"] is not None else -math.inf
    return {"gid": g.gid, "metrics": m, "baseline": base, "coverage": cov, "stability": ms, "checks": checks,
            "eligible": all(checks.values()), "failed": [k for k, ok in checks.items() if not ok],
            "score": (ar, _pf(m), m["total_trades"])}


def rank_key(g: Gate, tr: dict) -> tuple:
    s = tr[g.gid]["score"]
    return (-s[0], -s[1], -s[2], g.complexity, g.gid)


def validation_eval(g: Gate, rows: Sequence[dict], train_m: dict) -> dict[str, Any]:
    base = split_m(side_rows(rows, g.direction), "VALIDATION")
    m = split_m(gate_rows(g, rows), "VALIDATION")
    cov = compare(base, m)
    if m["total_trades"] < MIN_VAL_TRADES:
        label = INSUFFICIENT
    elif (cov["expectancy_change_r"] or -math.inf) < MIN_EXPECTANCY_GAIN or \
            _pf(m) < COLLAPSE_PF_SHARE * _pf(train_m):
        label = REJECTED
    elif _pf(m) >= 1.0 and (m["average_r"] or 0.0) > 0:
        label = PASS
    else:
        label = DEGRADED
    return {"gid": g.gid, "metrics": m, "baseline": base, "coverage": cov, "label": label}


def _assert_pre(rows: Sequence[dict]) -> None:
    late = sum(1 for r in rows if cb.day_of(r) > SELECTION_END)
    if late:
        raise ValueError(f"selection received {late} rows after {SELECTION_END} (OOS/holdout must not be used)")


def pre_rows(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if cb.day_of(r) <= SELECTION_END]


def select(rows: Sequence[dict]) -> dict[str, Any]:
    """Per side: shortlist = TRAIN-eligible gates (frozen after TRAIN), ranked on TRAIN; July labels them; the
    frozen gate is the best-ranked PASS, else the best-ranked DEGRADED, else NO TRADE improvement (None)."""
    _assert_pre(rows)
    out: dict[str, Any] = {"train": {}, "validation": {}, "shortlist": {}, "selected": {}, "counts": {}}
    for d in DIRECTIONS:
        pool = [g for g in GATES if g.direction == d]
        tr = {g.gid: train_eval(g, rows) for g in pool}
        short = sorted((g for g in pool if tr[g.gid]["eligible"]), key=lambda g: rank_key(g, tr))
        va = {g.gid: validation_eval(g, rows, tr[g.gid]["metrics"]) for g in short}
        pick = next((g for g in short if va[g.gid]["label"] == PASS), None) or \
            next((g for g in short if va[g.gid]["label"] == DEGRADED), None)
        out["train"].update(tr)
        out["validation"].update(va)
        out["shortlist"][d] = [g.gid for g in short]
        out["selected"][d] = pick
        out["counts"][d] = len(pool)
    return out


def ids(sel: dict) -> dict[str, str | None]:
    return {d: None if g is None else g.gid for d, g in sel["selected"].items()}


def fingerprint(sel: dict) -> str:
    spec = {"selected": {d: None if g is None else asdict(g) for d, g in sel["selected"].items()},
            "gates": [asdict(g) for g in GATES], "shortlist": sel["shortlist"]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def isolation(rows: Sequence[dict], sel: dict) -> dict[str, Any]:
    flipped = select(pre_rows(s5._flip(rows, SELECTION_END)))
    sep_only = select(pre_rows(s5._flip(rows, SPLITS["OOS"][1])))
    refuses = False
    try:
        select(rows)
    except ValueError:
        refuses = True
    pre = pre_rows(rows)
    rev = select(list(reversed(pre)))
    val_flip = select(s5._flip(pre, TRAIN_END))
    checks = {
        "august_cannot_affect_selection": ids(sel) == ids(flipped),
        "september_cannot_affect_selection": ids(sel) == ids(sep_only),
        "selection_refuses_post_july_rows": refuses,
        "ranking_deterministic": fingerprint(sel) == fingerprint(rev),
        "shortlist_uses_train_only": sel["shortlist"] == val_flip["shortlist"],
        "gate_count_within_limit": all(n <= MAX_GATES_PER_SIDE for n in sel["counts"].values()),
    }
    return {"checks": checks, "passed": all(checks.values())}


# ------------------------------------------------------------------ random control (descriptive, TRAIN only)
def random_control(g: Gate, rows: Sequence[dict], draws: int = RANDOM_DRAWS, seed: int = RANDOM_SEED) -> dict:
    base = [lab.r_value(r) for r in _ev(cb.split_rows(side_rows(rows, g.direction), "TRAIN"))]
    sub = [lab.r_value(r) for r in _ev(cb.split_rows(gate_rows(g, rows), "TRAIN"))]
    k = len(sub)
    if not k or k > len(base) or not g.conditions:
        return {"gid": g.gid, "subset_size": k, "draws": 0}
    rng = random.Random(f"{seed}:{g.gid}")
    means = sorted(sum(rng.sample(base, k)) / k for _ in range(draws))
    obs = sum(sub) / k
    pct = sum(m < obs for m in means) / draws
    return {"gid": g.gid, "subset_size": k, "draws": draws, "filter_avg_r": round(obs, 4),
            "random_mean_avg_r": round(sum(means) / draws, 4), "random_p05": round(means[int(0.05 * draws)], 4),
            "random_p95": round(means[int(0.95 * draws) - 1], 4), "filter_percentile": round(pct, 4),
            "plausibly_sample_selection": pct < 0.95}


# ------------------------------------------------------------------ lookahead (regime gate specific)
POST_ENTRY_FIELDS = ("outcome", "pnl", "r_multiple", "exit_time", "exit_price", "exit_reason", "stop_loss", "target",
                     "entry", "entry_time", "premium_atr")


def gate_truncation_check(spot: pd.DataFrame, rows: Sequence[dict], ctx_by_key: dict, range_start: str,
                          range_end: str, sample: int = 20) -> dict[str, Any]:
    """No-future-data mutation: every gate decision recomputed with all later NIFTY candles removed and all later
    premium candles tripled must be identical."""
    ann = [r for r in rows if r.get("instrument_key")]
    picked = ann[::max(1, len(ann) // max(sample, 1))][:sample]
    t_all = dra._ist_series(spot["timestamp"])
    bad = []
    for r in picked:
        ts = dra._ts(r["signal_time"])
        tb = dg.feature_bars(spot[(t_all <= ts).to_numpy()].reset_index(drop=True))
        ctx = ctx_by_key.get(dra.key_of(r))
        prem = (ctx or {}).get("premium")
        warped = dict(ctx, premium=s6._warp_after(prem, ts)) if prem is not None and not isinstance(prem, str) \
            else ctx
        again = dict(r, features=dg.signal_features(r, tb, warped))
        again["s6"] = s6.extra_features(again, tb, s6.orb_ranges(tb, range_start, range_end), warped)
        bad += [f"{r['signal_time']}|{r['direction']}|{g.gid}" for g in GATES if g.direction == r["direction"]
                and gate_decision(g, r) != gate_decision(g, again)]
        bad += [f"{r['signal_time']}|{r['direction']}|{k}" for k, fn in DIMENSIONS.items() if fn(r) != fn(again)]
    return {"checked": len(picked), "mismatches": bad[:20], "passed": bool(picked) and not bad}


def post_entry_independence(rows: Sequence[dict]) -> dict[str, Any]:
    """Wiping every post-entry field (outcome, P&L, exit, levels) must not change any gate decision."""
    bad = 0
    for r in rows:
        wiped = dict(r, **{k: None for k in POST_ENTRY_FIELDS})
        bad += sum(gate_decision(g, r) != gate_decision(g, wiped) for g in GATES if g.direction == r["direction"])
    return {"checked": len(rows), "mismatches": bad, "passed": bool(rows) and bad == 0}


# ------------------------------------------------------------------ frozen evaluation / status
def generalization(rows: Sequence[dict]) -> dict[str, Any]:
    pre = [r for r in rows if cb.split_of(r) in PRE_HOLDOUT]
    axes = {"months": lambda r: str(r["session_date"])[:7], "volatility": lambda r: r["features"].get("vol_regime"),
            "trend_state": lambda r: fv(r, "market_regime"), "time_window": lambda r: r["features"]["time_window"]}
    res = {k: s6.concentration(pre, fn) for k, fn in axes.items()}
    res["generalizes"] = all(v["profitable"] >= 2 for v in res.values())
    return res


def robustness(rows: Sequence[dict], combined: bool = False) -> dict[str, Any]:
    rb = s6.robustness(rows, combined)
    pre = [r for r in rows if cb.split_of(r) in PRE_HOLDOUT]
    rb["day"] = s6.concentration(pre, lambda r: str(r["session_date"])[:10])
    rb["generalization"] = generalization(rows)
    if rb["day"]["concentrated"] and "day" not in rb["overfit_flags"]:
        rb["overfit_flags"] = rb["overfit_flags"] + ["day"]
    rb["overfit_risk"] = bool(rb["overfit_flags"])
    return rb


def frozen_block(name: str, rows: Sequence[dict], base_rows: Sequence[dict], val_label: str,
                 combined: bool = False) -> dict[str, Any]:
    per = {k: split_m(rows, k) for k in PRE_HOLDOUT}
    base = {k: split_m(base_rows, k) for k in PRE_HOLDOUT}
    return {"name": name, "rows": list(rows), "base_rows": list(base_rows), "validation_label": val_label,
            "per_split": per, "baseline": base, "coverage": {k: compare(base[k], per[k]) for k in PRE_HOLDOUT},
            "robustness": robustness(rows, combined)}


def oos_improved(cov: dict, m: dict) -> bool:
    return ((cov["expectancy_change_r"] or -math.inf) >= MIN_EXPECTANCY_GAIN
            and (cov["pf_change"] >= MIN_PF_GAIN or (cov["dd_reduction"] or 0.0) >= MIN_DD_REDUCTION)
            and (_pf(m) >= 1.0 or (cov["dd_reduction"] or 0.0) >= STRONG_DD_REDUCTION))


def final_status(blk: dict, clean: bool) -> str:
    oos, cov = blk["per_split"]["OOS"], blk["coverage"]["OOS"]
    if not clean or blk["validation_label"] == REJECTED:
        return REJECTED
    if blk["validation_label"] == INSUFFICIENT or oos["total_trades"] < MIN_OOS_TRADES:
        return INSUFFICIENT
    if blk["robustness"]["overfit_risk"]:
        return OVERFIT
    improved = oos_improved(cov, oos) and (cov["retention"] or 0.0) >= MIN_RETENTION
    if improved and blk["validation_label"] == PASS:
        return PROMISING
    return DEGRADED if improved or blk["validation_label"] == DEGRADED else REJECTED


# ------------------------------------------------------------------ Phase 1 diagnostics (TRAIN only)
def diagnostics(rows: Sequence[dict]) -> list[dict]:
    train = cb.split_rows(rows, "TRAIN")
    out = []
    for d in DIRECTIONS:
        dir_rows = side_rows(train, d)
        n_dir = len(dir_rows)
        for dim, fn in DIMENSIONS.items():
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in dir_rows:
                groups[str(fn(r))].append(r)
            for b in sorted(groups):
                m = cb.metrics(groups[b])
                out.append({"section": "TRAIN_BUCKET", "direction": d, "dimension": dim, "bucket": b,
                            "total_signals": m["signals"], "executable_trades": m["executable"], **s6._cols(m),
                            "expectancy_rs": s6.expectancy_rs(m),
                            "retention_of_side_signals": round(m["signals"] / n_dir, 4) if n_dir else None,
                            "sample_ok": m["wins"] + m["losses"] >= dg.MIN_SAMPLE})
                for mo in TRAIN_MONTHS:
                    mm = cb.metrics([r for r in groups[b] if str(r["session_date"])[:7] == mo])
                    out.append({"section": "TRAIN_MONTH", "direction": d, "dimension": dim, "bucket": b, "month": mo,
                                "total_signals": mm["signals"], "executable_trades": mm["executable"],
                                **s6._cols(mm)})
    return out


# ------------------------------------------------------------------ study
def build(*, rows: Sequence[dict], contexts: Sequence[dict], ctx_by_key: dict, spot: pd.DataFrame,
          bars: pd.DataFrame, params, square_off: str, stop_new_entries: str, range_start: str, range_end: str,
          calendar, listed: Callable, exists: Callable, integrity: dict, mutation: dict | None,
          sessions: Sequence[str], dataset: dict, check_sample: int = 20,
          production_unchanged: bool | None = None) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
    if not all("s6" in r for r in rows):
        rows = s6.prepare(rows, bars, ctx_by_key, range_start, range_end)

    # Phase 12 - causality
    lookahead = {
        "feature_truncation": dg.truncation_check(spot, rows, ctx_by_key, check_sample),
        "regime_extra_truncation": s6.extra_truncation_check(spot, rows, ctx_by_key, range_start, range_end,
                                                             check_sample),
        "no_future_data_gate_mutation": gate_truncation_check(spot, rows, ctx_by_key, range_start, range_end,
                                                              check_sample),
        "future_candle_scrambling": mutation or {"passed": False, "detail": "NOT RUN"},
        "chronological_replay": dg.chronology_check(rows),
        "entry_next_valid_candle": dra.alignment_checks(contexts, params),
        "no_post_entry_information": post_entry_independence(rows),
    }
    lookahead_ok = all(bool(v.get("passed")) for v in lookahead.values())

    # Phases 1-6 - TRAIN discovery, shortlist frozen after TRAIN, July labels
    sel = select(pre_rows(rows))
    fp = fingerprint(sel)
    iso = isolation(rows, sel)
    controls = {g.gid: random_control(g, rows) for g in GATES if g.conditions}

    # integrity
    dup = cb.duplicate_check(rows)
    ts_chk = cb.timestamp_checks(rows, square_off, stop_new_entries)
    cval = cb.contract_validity(rows, calendar, listed, exists)
    data_ok = bool(integrity.get("ok")) and dup["passed"] and ts_chk["passed"] and cval["passed"]
    clean = lookahead_ok and data_ok

    # Phase 7 - frozen OOS (once)
    frozen: dict[str, dict | None] = {}
    for d in DIRECTIONS:
        g = sel["selected"][d]
        frozen[d] = None if g is None else frozen_block(f"{g.gid} {g.name}", gate_rows(g, rows), side_rows(rows, d),
                                                        sel["validation"][g.gid]["label"])
    picked = [sel["selected"][d] for d in DIRECTIONS if sel["selected"][d]]
    if picked:
        comb = sorted((r for g in picked for r in gate_rows(g, rows)),
                      key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
        no_trade = [d for d in DIRECTIONS if sel["selected"][d] is None]
        name = " + ".join(g.gid for g in picked) + "".join(f" ({d} = NO TRADE)" for d in no_trade)
        labels = [sel["validation"][g.gid]["label"] for g in picked]
        frozen["COMBINED"] = frozen_block(name, comb, rows, PASS if all(x == PASS for x in labels) else DEGRADED,
                                          combined=True)
    else:
        frozen["COMBINED"] = None

    # Phase 8 - September, once, after the fingerprint check
    if fingerprint(sel) != fp:
        raise RuntimeError("selection changed before the final holdout")
    for blk in frozen.values():
        if blk is not None:
            blk["holdout"] = split_m(blk["rows"], "FINAL_HOLDOUT")
            blk["holdout_baseline"] = split_m(blk["base_rows"], "FINAL_HOLDOUT")
            blk["holdout_coverage"] = compare(blk["holdout_baseline"], blk["holdout"])
            blk["holdout_status"] = s6.holdout_status(blk["holdout"])
            blk["status"] = final_status(blk, clean)
            blk["duplicates"] = cb.duplicate_check(blk["rows"])
    frozen_dup_ok = all(blk["duplicates"]["passed"] for blk in frozen.values() if blk is not None)
    data_ok = data_ok and frozen_dup_ok

    statuses = [blk["status"] for blk in frozen.values() if blk is not None]
    overfitting_ok = iso["passed"] and OVERFIT not in statuses
    target = "NOT ACHIEVED"
    cblk = frozen["COMBINED"]
    if cblk is not None and all(m["wins"] + m["losses"] >= MIN_OOS_TRADES and (m["win_rate"] or 0) >= TARGET
                                for m in (cblk["per_split"]["OOS"], cblk["holdout"])):
        target = "ACHIEVED"
    promising = [k for k, blk in frozen.items() if blk is not None and blk["status"] == PROMISING]
    safe = bool(promising) and lookahead_ok and data_ok and overfitting_ok and all(
        frozen[k]["holdout_status"] == s6.PASSED_HO for k in promising)

    return {
        "heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "accuracy_definition": "wins / (wins + losses); ambiguous (-1R), scratch and unavailable are never wins",
        "dataset": dataset, "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        "sessions": len(sessions), "signals": len(rows),
        "s2": {k: cb.scoped(cb.split_rows(rows, k)) for k in SPLITS},
        "diagnostics": diagnostics(rows), "not_available_features": NOT_AVAILABLE_FEATURES,
        "selection": sel, "fingerprint": fp, "isolation": iso, "random_control": controls,
        "frozen": frozen, "target_70": target,
        "lookahead": lookahead, "lookahead_ok": lookahead_ok,
        "data_integrity": integrity, "data_checks": {"duplicates": dup, "timestamps": ts_chk,
                                                     "contract_validity": cval},
        "data_ok": data_ok, "overfitting_ok": overfitting_ok,
        "final_result": ("PROMISING regime filter(s): " + ", ".join(f"{k}: {frozen[k]['name']}" for k in promising))
        if promising else NO_VALID,
        "forward_paper": "SAFE" if safe else "NOT SAFE",
        "disclosure": ("August and September S2 results were already seen in earlier labs (S2 diagnostic, S5, clean "
                       "backtest, S6). S7 gates are the pre-declared structures of the S7 specification, shortlisted "
                       "on TRAIN and labelled on July only, but the researcher was not blind to those months; OOS and "
                       "holdout are therefore optimistic and fresh forward PAPER data is required."),
        "production_unchanged": production_unchanged, "live_orders": 0, "trading_mode": "PAPER",
        "live_trading_enabled": False, "rows": rows,
    }


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "s7_report.md", "baseline_results.csv", "regime_diagnostics.csv",
                "buy_regime_results.csv", "sell_regime_results.csv", "candidate_results.csv",
                "validation_results.csv", "oos_results.csv", "holdout_results.csv", "coverage_analysis.csv",
                "robustness.csv", "data_quality.json")
_m, _csv, _jsonable, line = s6._m, s6._csv, s6._jsonable, s6.line


def _status(a: dict, gid: str) -> str | None:
    for k, blk in a["frozen"].items():
        if blk is not None and k != "COMBINED" and blk["name"].split(" ")[0] == gid:
            return blk["status"]
    return None


def baseline_csv(a: dict) -> list[dict]:
    return [{"split": SPLIT_NAMES[k], "scope": s, **_m(a["s2"][k][s])} for k in SPLITS for s in SCOPES]


def side_results(a: dict, d: str) -> list[dict]:
    sel, out = a["selection"], []
    for g in (g for g in GATES if g.direction == d):
        tr, rc = sel["train"][g.gid], a["random_control"].get(g.gid, {})
        row = {"gate": g.gid, "name": g.name, "rule": g.describe(), "rationale": g.rationale, **_m(tr["metrics"]),
               **{f"cov_{k}": v for k, v in tr["coverage"].items()}, **tr["checks"], "train_eligible": tr["eligible"],
               "stability": tr["stability"]["label"], "train_months_beating_baseline": tr["stability"]["beats"],
               "one_month_dependent": tr["stability"]["one_month_dependent"],
               "train_rank": sel["shortlist"][d].index(g.gid) + 1 if g.gid in sel["shortlist"][d] else None,
               "random_filter_percentile": rc.get("filter_percentile"),
               "random_mean_avg_r": rc.get("random_mean_avg_r"),
               "plausibly_sample_selection": rc.get("plausibly_sample_selection")}
        for mo, v in tr["stability"]["months"].items():
            row.update({f"{mo}_{k}": v[k] for k in ("trades", "win_rate", "profit_factor", "average_r", "net_pnl")})
        out.append(row)
    return out


def candidate_csv(a: dict) -> list[dict]:
    sel, out = a["selection"], []
    chosen = {g.gid for g in sel["selected"].values() if g}
    for g in GATES:
        tr, va = sel["train"][g.gid], sel["validation"].get(g.gid)
        base = {"gate": g.gid, "name": g.name, "direction": g.direction, "rule": g.describe(),
                "train_eligible": tr["eligible"], "train_failed": ",".join(tr["failed"]),
                "validation_label": va["label"] if va else "NOT_SHORTLISTED", "selected": g.gid in chosen,
                "final_status": _status(a, g.gid)}
        out.append({**base, "split": "TRAIN", **_m(tr["metrics"]), "retention": tr["coverage"]["retention"]})
        if va:
            out.append({**base, "split": "VALIDATION", **_m(va["metrics"]), "retention": va["coverage"]["retention"]})
        if g.gid in chosen:
            blk = a["frozen"][g.direction]
            out.append({**base, "split": "OOS", **_m(blk["per_split"]["OOS"]),
                        "retention": blk["coverage"]["OOS"]["retention"]})
            out.append({**base, "split": HOLDOUT_LABEL, **_m(blk["holdout"]),
                        "retention": blk["holdout_coverage"]["retention"]})
    return out


def validation_csv(a: dict) -> list[dict]:
    sel = a["selection"]
    return [{"gate": gid, "name": next(g.name for g in GATES if g.gid == gid), "label": va["label"],
             **_m(va["metrics"]), **{f"cov_{k}": v for k, v in va["coverage"].items()},
             "baseline_average_r": va["baseline"]["average_r"], "baseline_pf": va["baseline"]["profit_factor"]}
            for gid, va in sel["validation"].items()]


def _frozen_split(a: dict, split: str) -> list[dict]:
    out = []
    for k in SCOPES:
        blk = a["frozen"].get(k)
        base = a["s2"][split][k]
        out.append({"scope": k, "system": f"BASELINE_S2_{k}", **_m(base)})
        if blk is None:
            out.append({"scope": k, "system": f"S7_FILTERED_{k}", "gate": "none (NO TRADE improvement survived)"})
            continue
        m = blk["per_split"]["OOS"] if split == "OOS" else blk["holdout"]
        cov = blk["coverage"]["OOS"] if split == "OOS" else blk["holdout_coverage"]
        row = {"scope": k, "system": f"S7_FILTERED_{k}", "gate": blk["name"], **_m(m), **cov,
               "status": blk["status"]}
        if split != "OOS":
            row.update(label=HOLDOUT_LABEL, holdout_status=blk["holdout_status"])
        out.append(row)
    return out


def coverage_csv(a: dict) -> list[dict]:
    sel, out = a["selection"], []
    for g in GATES:
        if not g.conditions:
            continue
        out.append({"gate": g.gid, "direction": g.direction, "split": "TRAIN", **sel["train"][g.gid]["coverage"]})
        if g.gid in sel["validation"]:
            out.append({"gate": g.gid, "direction": g.direction, "split": "VALIDATION",
                        **sel["validation"][g.gid]["coverage"]})
    for k, blk in a["frozen"].items():
        if blk is not None:
            out.append({"gate": blk["name"], "direction": k, "split": "OOS", **blk["coverage"]["OOS"]})
            out.append({"gate": blk["name"], "direction": k, "split": HOLDOUT_LABEL, **blk["holdout_coverage"]})
    return out


def robustness_csv(a: dict) -> list[dict]:
    out = []
    for k, blk in a["frozen"].items():
        if blk is None:
            continue
        rb = blk["robustness"]
        for comp in ("monthly", "day", "regime", "time_of_day"):
            out.append({"scope": k, "gate": blk["name"], "component": f"concentration_{comp}",
                        **{x: rb[comp][x] for x in ("groups", "profitable", "best_group", "best_share_of_profit",
                                                    "total_pnl", "pnl_without_best", "concentrated")}})
        for ax in ("months", "volatility", "trend_state", "time_window"):
            g = rb["generalization"][ax]
            out.append({"scope": k, "gate": blk["name"], "component": f"generalization_{ax}", "groups": g["groups"],
                        "profitable": g["profitable"], "best_group": g["best_group"]})
        out.append({"scope": k, "gate": blk["name"], "component": "few_trades", **rb["few_trades"]})
        out.append({"scope": k, "gate": blk["name"], "component": "summary",
                    "generalizes": rb["generalization"]["generalizes"], "overfit_flags": ",".join(rb["overfit_flags"]),
                    "overfit_risk": rb["overfit_risk"], "max_losing_streak": rb["max_losing_streak"],
                    "max_drawdown": rb["max_drawdown"], "final_status": blk["status"]})
    return out


def _summary_json(a: dict) -> dict:
    frozen = {k: None if blk is None else {x: v for x, v in blk.items() if x not in ("rows", "base_rows")}
              for k, blk in a["frozen"].items()}
    skip = ("rows", "frozen", "diagnostics", "data_integrity")
    return {**{k: v for k, v in a.items() if k not in skip}, "frozen": frozen}


def _best_cmp(a: dict) -> tuple[str, dict] | None:
    for k in ("COMBINED", "SELL", "BUY"):
        if a["frozen"].get(k) is not None:
            return k, a["frozen"][k]["coverage"]["OOS"]
    return None


def comparison_lines(a: dict) -> list[str]:
    best = _best_cmp(a)
    if best is None:
        return ["N/A - no regime gate survived TRAIN + VALIDATION (BUY = NO TRADE improvement, SELL = NO TRADE "
                "improvement)"]
    k, c = best

    def f(v, fmt):
        return "N/A" if v is None else fmt.format(v)
    return [f"(OOS AUGUST, {k}: {a['frozen'][k]['name']})",
            f"accuracy change: {f(c['accuracy_change_pp'], '{:+.1f} pp')}",
            f"PF change: {f(c['pf_change'], '{:+.2f}')}",
            f"expectancy change: {f(c['expectancy_change_r'], '{:+.3f}R')}",
            f"DD change: {f(None if c['dd_reduction'] is None else -c['dd_reduction'] * 100, '{:+.1f}%')}",
            f"trade reduction: {f(None if c['trade_reduction'] is None else c['trade_reduction'] * 100, '{:.1f}%')}"]


def markdown(a: dict) -> str:
    sel = a["selection"]
    md = ["# S7 market regime / no-trade filter lab", "", f"**{BANNER}**", "",
          f"Generated {a['generated_at']}. Dataset {a['dataset'].get('from')} -> {a['dataset'].get('to')} "
          "(data/historical/upstox_6m, NSE_INDEX|Nifty 50, real option premium candles). TRADING_MODE=PAPER, "
          "LIVE_TRADING_ENABLED=false, live orders: 0. Production changed: "
          f"**{'NO' if a['production_unchanged'] is not False else 'YES'}**.", "",
          "All numbers are **historical backtest accuracy**, not live accuracy; no profit is guaranteed. A gate only "
          "removes S2 signals; the S2 signal, contract, entry, stop, target and exit are unchanged.", "",
          f"> {a['disclosure']}", "", "## Result", "", f"- **{a['final_result']}**",
          f"- Forward PAPER validation: **{a['forward_paper']}**",
          f"- Lookahead **{'PASS' if a['lookahead_ok'] else 'FAIL'}** | data integrity "
          f"**{'PASS' if a['data_ok'] else 'FAIL'}** | overfitting **{'PASS' if a['overfitting_ok'] else 'FAIL'}** | "
          f"70% target **{a['target_70']}**", ""]
    md += ["## Baseline S2", "", "| Split | Scope | Result |", "|---|---|---|"]
    md += [f"| {SPLIT_NAMES[k]} | {s} | {line(a['s2'][k][s])} |" for k in SPLITS for s in SCOPES]
    md += ["", "## Phase 1/2 - TRAIN regime diagnostics", "",
           "Full tables in regime_diagnostics.csv (TRAIN only, with April/May/June rows). VWAP slope: "
           f"{NOT_AVAILABLE_FEATURES['vwap_slope']}.", "",
           "## Phase 3/4 - gates, TRAIN shortlist and coverage", "",
           f"Shortlist rule (TRAIN only): >= {MIN_TRAIN_TRADES} trades, retention >= {MIN_RETENTION:.0%}, expectancy "
           f">= baseline + {MIN_EXPECTANCY_GAIN}R, PF >= baseline + {MIN_PF_GAIN} or drawdown reduced >= "
           f"{MIN_DD_REDUCTION:.0%}, expectancy beats the baseline in >= {MIN_MONTH_BEATS} of the 3 TRAIN months and "
           "profit not from one month. Ranked by expectancy, PF, trades, fewer conditions.", "",
           "| Gate | Rule | TRAIN | Retention | dExp R | dPF | DD red. | Months | Random pct | Shortlisted | July |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in GATES:
        tr, va, rc = sel["train"][g.gid], sel["validation"].get(g.gid), a["random_control"].get(g.gid, {})
        c = tr["coverage"]
        md.append(f"| {g.gid} | {g.describe()} | {line(tr['metrics'])} | {c['retention']} | "
                  f"{c['expectancy_change_r']} | {c['pf_change']} | {c['dd_reduction']} | {tr['stability']['beats']}/3 "
                  f"({tr['stability']['label']}) | {rc.get('filter_percentile', '-')} | {tr['eligible']} | "
                  f"{va['label'] if va else '-'} |")
    md += ["", f"Frozen after July: {ids(sel)} (fingerprint `{a['fingerprint']}`).", "",
           "## Phase 7/8 - frozen OOS and single September holdout", "",
           "| Scope | Gate | OOS filtered | OOS baseline | OOS coverage | Holdout filtered | Holdout baseline | Status |",
           "|---|---|---|---|---|---|---|---|"]
    for k in SCOPES:
        blk = a["frozen"].get(k)
        if blk is None:
            md.append(f"| {k} | none | - | {line(a['s2']['OOS'][k])} | - | - | {line(a['s2']['FINAL_HOLDOUT'][k])} | "
                      "- |")
            continue
        md.append(f"| {k} | {blk['name']} | {line(blk['per_split']['OOS'])} | {line(blk['baseline']['OOS'])} | "
                  f"{blk['coverage']['OOS']} | {line(blk['holdout'])} | {line(blk['holdout_baseline'])} | "
                  f"**{blk['status']}** |")
    md += ["", f"September rows: **{HOLDOUT_LABEL}**.", "", "## Phase 9/10 - generalization and concentration", ""]
    for k, blk in a["frozen"].items():
        if blk is not None:
            rb = blk["robustness"]
            md.append(f"- {k}: generalizes {rb['generalization']['generalizes']}; overfit flags "
                      f"{rb['overfit_flags'] or 'none'}")
    md += ["", "## Phase 11 - random control (TRAIN, descriptive)", "",
           f"{RANDOM_DRAWS} equal-size random subsets of the side's TRAIN S2 trades (seeded). A filter percentile "
           "below 0.95 means the improvement is plausibly explained by sample selection alone.", "",
           "## Phase 12 - anti-lookahead and isolation", ""]
    md += [f"- {k}: {'PASS' if v.get('passed') else 'FAIL'}" for k, v in a["lookahead"].items()]
    md += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in a["isolation"]["checks"].items()]
    md += ["", "## Baseline vs filter", ""] + [f"- {x}" for x in comparison_lines(a)]
    md += ["", "## Phase 13/14 - promotion and safety", "", a["final_result"] + ".",
           "No candidate is promoted or activated automatically. Production S2, broker, OrderManager, risk and "
           "safety settings are unchanged."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    p["summary.json"].write_text(json.dumps(_jsonable(_summary_json(a)), indent=2, default=str), encoding="utf-8")
    p["s7_report.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["baseline_results.csv"], baseline_csv(a))
    _csv(p["regime_diagnostics.csv"], a["diagnostics"])
    _csv(p["buy_regime_results.csv"], side_results(a, "BUY"))
    _csv(p["sell_regime_results.csv"], side_results(a, "SELL"))
    _csv(p["candidate_results.csv"], candidate_csv(a))
    _csv(p["validation_results.csv"], validation_csv(a))
    _csv(p["oos_results.csv"], _frozen_split(a, "OOS"))
    _csv(p["holdout_results.csv"], _frozen_split(a, "FINAL_HOLDOUT"))
    _csv(p["coverage_analysis.csv"], coverage_csv(a))
    _csv(p["robustness.csv"], robustness_csv(a))
    dq = {"data_integrity": a["data_integrity"], "checks": a["data_checks"], "data_ok": a["data_ok"],
          "unavailable_reasons": {k: a["s2"][k]["COMBINED"]["unavailable_reasons"] for k in SPLITS},
          "lookahead": a["lookahead"], "missing_option_data": "marked DATA_UNAVAILABLE, never interpolated"}
    p["data_quality.json"].write_text(json.dumps(_jsonable(dq), indent=2, default=str), encoding="utf-8")
    return p


def _gate_block(label: str, blk: dict | None, full: bool) -> list[str]:
    out = [f"{label}:"]
    if blk is None:
        out += ["name: none (NO TRADE improvement survived TRAIN + VALIDATION)"]
        out += (["TRAIN: N/A", "VALIDATION: N/A"] if full else []) + ["OOS: N/A", "HOLDOUT: N/A", "retention: N/A",
                                                                      f"status: {REJECTED}", ""]
        return out
    p = blk["per_split"]
    out.append(f"name: {blk['name']}")
    if full:
        out += [f"TRAIN: {line(p['TRAIN'])}", f"VALIDATION: {line(p['VALIDATION'])} ({blk['validation_label']})"]
    ret = [blk["coverage"][k]["retention"] for k in PRE_HOLDOUT] + [blk["holdout_coverage"]["retention"]]
    out += [f"OOS: {line(p['OOS'])}", f"HOLDOUT: {line(blk['holdout'])} ({HOLDOUT_LABEL})",
            "retention: " + " / ".join(f"{n} {'N/A' if v is None else f'{v:.0%}'}"
                                       for n, v in zip(("TRAIN", "VAL", "OOS", "HOLDOUT"), ret)),
            f"status: {blk['status']}", ""]
    return out


def terminal_text(a: dict) -> str:
    out = [HEADING, "(historical backtest accuracy - not live accuracy, no guaranteed profit)", "", "BASELINE S2", ""]
    for k in SPLITS:
        out.append(f"{SPLIT_NAMES[k]}:")
        out += [f"{s}: {line(a['s2'][k][s])}" for s in SCOPES]
        out.append("")
    out += _gate_block("BEST BUY REGIME FILTER", a["frozen"].get("BUY"), True)
    out += _gate_block("BEST SELL REGIME FILTER", a["frozen"].get("SELL"), True)
    out += _gate_block("BEST COMBINED", a["frozen"].get("COMBINED"), False)
    out += ["BASELINE vs FILTER:"] + comparison_lines(a) + [""]
    out += ["70% TARGET:", a["target_70"], "", "LOOKAHEAD:", "PASS" if a["lookahead_ok"] else "FAIL", "",
            "DATA INTEGRITY:", "PASS" if a["data_ok"] else "FAIL", "", "OVERFITTING:",
            "PASS" if a["overfitting_ok"] else "FAIL", "", "PRODUCTION CHANGED:",
            "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:", "0", "",
            "FORWARD PAPER VALIDATION:", a["forward_paper"], "", "FINAL RESULT:", a["final_result"]]
    return "\n".join(out)
