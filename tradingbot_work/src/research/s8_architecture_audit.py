"""S8 STRATEGY ARCHITECTURE AUDIT & CLEAN ALTERNATIVE RESEARCH (PAPER | RESEARCH ONLY | NO ORDERS).

Question: does the S2 architecture itself have an edge, and if not, does a fundamentally different family?

Layer by layer, on the exact S2 replay (production S2 is never modified):
  A  underlying signal  - NIFTY forward returns / MFE / MAE after every S2 signal, before any option is touched
  B  option translation - does the bought CE/PE premium follow the NIFTY move (response, lag, premium MFE/MAE)
  C  entry timing       - four predefined causal entries on the same signals and the same S2 exit
  D  exit architecture  - current S2 (2R) / 1.5R / 2R / 2.5R on the same entries
  E  alternatives       - only if S2 lacks an underlying edge: 4 families x 3 predefined variants per side
  G  premium confirmation of the shortlisted alternative (completed option candles only)

Chronology (historical backtest accuracy, never live accuracy):
  TRAIN 2026-04-01..06-30      ranking / shortlist only
  VALIDATION 07-01..07-31      VALIDATED / REJECTED / INSUFFICIENT_DATA; nothing is modified afterwards
  OOS 08-01..08-31             frozen (fingerprinted) choices only, once
  FINAL HOLDOUT 09-01..09-30   frozen choices only, once, NOT USED FOR SELECTION
``select`` raises if it receives any row after 2026-07-31; entry / exit choices read TRAIN and VALIDATION only.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.backtest import current_day_s2 as cds
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest.research_strategies import ResearchSignal
from src.options.strategy import round_to_tick
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s5_targeted_strategy as s5
from src.research import s6_clean_strategy_lab as s6

HEADING = "S8 COMPLETE"
BANNER = cb.BANNER
IST = s6.IST
SPLITS = cb.SPLITS
SPLIT_NAMES = s6.SPLIT_NAMES
SCOPES, DIRECTIONS = cb.SCOPES, s6.DIRECTIONS
PRE_OOS, PRE_HOLDOUT = s6.PRE_OOS, s6.PRE_HOLDOUT
TRAIN_END, SELECTION_END = s6.TRAIN_END, s6.SELECTION_END
TRAIN_MONTHS = ("2026-04", "2026-05", "2026-06")
HOLDOUT_LABEL = s6.HOLDOUT_LABEL
STEP = s6.STEP
STOP_ATR, BASE_TARGET = s6.STOP_ATR, s6.BASE_TARGET
UNAVAILABLE = s6.UNAVAILABLE
TARGET = 0.70

# Pre-declared thresholds (fixed before any result was computed; never fitted).
HORIZONS = (1, 2, 3, 5)               # candles after the signal candle
MAX_H = 5                             # MFE / MAE window (candles)
EDGE_H = 5                            # horizon of the underlying edge verdict
MIN_DIRECTION_ACC = 0.52
MIN_EDGE_VS_UNCONDITIONAL = 0.02      # accuracy above the same-window unconditional drift
MIN_DAY_T = 2.0                       # day-clustered t-statistic (signals of one day are not independent)
RESPONSE_OK = 0.60                    # option up when NIFTY moved the signal's way
TRANSLATION_GAP = 0.05                # option accuracy may trail underlying accuracy by at most 5 pp
MIN_VARIANT_GAIN = 0.05               # entry / exit change: +0.05R on BOTH TRAIN and VALIDATION
PULLBACK_ENTRY_BARS = 3
MFE_LEVELS = (1.0, 1.5, 2.0, 2.5)

PULLBACK_BARS = 6                     # alternative families: look-back for a completed pullback
STRETCH_BARS = 3
STRONG_TREND_ATR = 1.0
STRETCH_ATR = 1.5
RSI_EXTREME = 30.0
RSI_MOMENTUM = (50.0, 70.0)
COOLDOWN_BARS = 6
MAX_SIGNALS_PER_DAY = 3
MAX_FAMILIES, MAX_VARIANTS = 4, 3

MIN_TRAIN_TRADES, MIN_VAL_TRADES, MIN_OOS_TRADES = 30, 10, s6.MIN_OOS_TRADES
MIN_TRAIN_PF = 1.10
MAX_DD_SHARE = 0.5
MAX_TRAIN_STREAK = 12
MIN_PROFITABLE_MONTHS = 2
MIN_VAL_PF_RETENTION = 0.5
MAX_CANDIDATES_PER_SIDE = 20

UNDERLYING_EDGE, WEAK_EDGE, NO_EDGE = "UNDERLYING_EDGE", "WEAK_UNDERLYING_EDGE", "NO UNDERLYING EDGE"
T_EDGE, T_LOSS, T_NONE = "OPTION_TRANSLATION_EDGE", "OPTION_TRANSLATION_LOSS", "NO_EDGE_TO_TRANSLATE"
VALIDATED, REJECTED, INSUFFICIENT = s6.VALIDATED, s6.REJECTED, s6.INSUFFICIENT
PROMISING, OVERFIT = s6.PROMISING, s6.OVERFIT
D_S2_EDGE = "S2 HAS EDGE"
D_OPTION = "S2 ENTRY HAS EDGE BUT OPTION LAYER DAMAGES IT"
D_EXIT = "S2 EXIT ARCHITECTURE DAMAGES IT"
D_NO_EDGE = "S2 HAS NO SUFFICIENT EDGE"
D_ALT = "ALTERNATIVE STRATEGY PROMISING"
D_NONE = "NO VALID STRATEGY FOUND"

_pf, _ev, fv, split_m = s6._pf, s6._ev, s6.fv, s6.split_m


def _sign(d: str) -> float:
    return 1.0 if d == "BUY" else -1.0


def _div(a, b) -> float | None:
    a, b = dg._num(a), dg._num(b)
    return None if a is None or not b else a / b


def _avg(v: Sequence) -> float | None:
    v = [x for x in v if x is not None]
    return round(mean(v), 4) if v else None


def _med(v: Sequence) -> float | None:
    v = [x for x in v if x is not None]
    return round(median(v), 4) if v else None


def _share(v: Sequence[bool | None]) -> float | None:
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


class BarIndex:
    """Completed NIFTY 5m candles as arrays; position lookup by IST candle label."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self.ts = bars.index
        self.pos = {t: i for i, t in enumerate(bars.index)}
        self.o, self.h, self.l, self.c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.atr = bars["atr14"].to_numpy(float)
        self.day = np.array([t.date() for t in bars.index])

    def __len__(self) -> int:
        return len(self.ts)

    def same_day(self, i: int, j: int) -> bool:
        return 0 <= j < len(self) and self.day[j] == self.day[i]


# ------------------------------------------------------------------ PART A - underlying signal audit
def underlying_item(r: dict, bx: BarIndex) -> dict | None:
    """Signed NIFTY move after the signal candle's close (outcome measurement only, never a decision input)."""
    i = bx.pos.get(dra._ts(r["signal_time"]))
    if i is None:
        return None
    s, c0, a = _sign(r["direction"]), bx.c[i], dg._num(bx.atr[i])
    out: dict[str, Any] = {"key": dra.key_of(r), "direction": r["direction"], "split": cb.split_of(r),
                           "day": str(r["session_date"])[:10]}
    for k in range(1, MAX_H + 1):
        j = i + k
        v = s * (bx.c[j] - c0) if bx.same_day(i, j) else None
        out[f"ret{k}"], out[f"ret{k}_atr"] = v, _div(v, a)
    js = [j for j in range(i + 1, i + MAX_H + 1) if bx.same_day(i, j)]
    fav = bx.h if s > 0 else -bx.l
    adv = bx.l if s > 0 else -bx.h
    mfe = float(max(fav[j] for j in js) - s * c0) if js else None
    mae = float(s * c0 - min(adv[j] for j in js)) if js else None
    out.update(mfe=mfe, mae=mae, mfe_atr=_div(mfe, a), mae_atr=_div(mae, a))
    return out


def day_t(items: Sequence[dict], key: str) -> float | None:
    per: dict[str, list[float]] = defaultdict(list)
    for x in items:
        if x.get(key) is not None:
            per[x["day"]].append(x[key])
    means = [sum(v) / len(v) for v in per.values()]
    if len(means) < 3:
        return None
    sd = stdev(means)
    return round(mean(means) / (sd / math.sqrt(len(means))), 3) if sd > 0 else None


def underlying_stats(items: Sequence[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"signals": len(items), "days": len({x["day"] for x in items})}
    for k in HORIZONS:
        out[f"mean_ret{k}_pts"] = _avg([x[f"ret{k}"] for x in items])
        out[f"mean_ret{k}_atr"] = _avg([x[f"ret{k}_atr"] for x in items])
        out[f"direction_acc{k}"] = _share([None if x[f"ret{k}"] in (None, 0) else x[f"ret{k}"] > 0 for x in items])
    out.update(avg_mfe_pts=_avg([x["mfe"] for x in items]), avg_mae_pts=_avg([x["mae"] for x in items]),
               avg_mfe_atr=_avg([x["mfe_atr"] for x in items]), avg_mae_atr=_avg([x["mae_atr"] for x in items]))
    out["mfe_mae_ratio"] = None if not out["avg_mae_pts"] or out["avg_mfe_pts"] is None else \
        round(out["avg_mfe_pts"] / out["avg_mae_pts"], 4)
    out[f"day_t{EDGE_H}"] = day_t(items, f"ret{EDGE_H}_atr")
    return out


def unconditional(bx: BarIndex, first: time, cutoff: time) -> dict[str, dict[str, float | None]]:
    """Share of ALL candles in the signal window whose NIFTY close rose / fell over EDGE_H candles, per split."""
    out: dict[str, dict[str, float | None]] = {}
    for k, (lo, hi) in SPLITS.items():
        up = down = 0
        for i, t in enumerate(bx.ts):
            if not (lo <= bx.day[i] <= hi) or not (first < t.time() < cutoff) or not bx.same_day(i, i + EDGE_H):
                continue
            d = bx.c[i + EDGE_H] - bx.c[i]
            up, down = up + (d > 0), down + (d < 0)
        n = up + down
        out[k] = {"BUY": round(up / n, 4) if n else None, "SELL": round(down / n, 4) if n else None}
    return out


def first_per_day(items: Sequence[dict]) -> list[dict]:
    seen, out = set(), []
    for x in sorted(items, key=lambda x: x["key"][0]):
        if (x["day"], x["direction"]) not in seen:
            seen.add((x["day"], x["direction"]))
            out.append(x)
    return out


def underlying_verdict(items: Sequence[dict], uncond: dict) -> dict[str, Any]:
    """Pre-declared rule on TRAIN and VALIDATION only (OOS / holdout are reported as confirmation)."""
    d = items[0]["direction"] if items else None
    per = {k: underlying_stats([x for x in items if x["split"] == k]) for k in SPLITS}
    pooled = underlying_stats([x for x in items if x["split"] in PRE_OOS])
    acc, ret = f"direction_acc{EDGE_H}", f"mean_ret{EDGE_H}_atr"

    def split_edge(k: str) -> bool:
        a, base = per[k][acc], (uncond.get(k) or {}).get(d)
        return (a is not None and a >= MIN_DIRECTION_ACC and (per[k][ret] or 0.0) > 0
                and (base is None or a - base >= MIN_EDGE_VS_UNCONDITIONAL))
    checks = {"train_edge": split_edge("TRAIN"), "validation_edge": split_edge("VALIDATION"),
              "day_clustered_t": (pooled[f"day_t{EDGE_H}"] or 0.0) >= MIN_DAY_T}
    if all(checks.values()):
        verdict = UNDERLYING_EDGE
    elif (pooled[acc] or 0.0) >= 0.5 and (pooled[ret] or 0.0) > 0 and (checks["train_edge"] or
                                                                         checks["validation_edge"]):
        verdict = WEAK_EDGE
    else:
        verdict = NO_EDGE
    later = [k for k in ("OOS", "FINAL_HOLDOUT") if split_edge(k)]
    return {"direction": d, "verdict": verdict, "checks": checks, "per_split": per, "train_validation": pooled,
            "confirmed_later_in": later}


# ------------------------------------------------------------------ PART B - option translation audit
def option_item(r: dict, ctx: dict | None, und: dict | None) -> dict | None:
    """Bought CE/PE premium after entry (open of the candle after the signal) vs the NIFTY move. Missing premium
    candles stay missing (never interpolated)."""
    if und is None or ctx is None or ctx.get("data_status") != "OK" or not ctx.get("premium_atr"):
        return None
    prem = ctx["premium"]
    t = pd.Index(dra._ist_series(prem["timestamp"]))
    sig, entry, risk = ctx["signal_time"], float(ctx["entry"]), STOP_ATR * float(ctx["premium_atr"])
    closes, highs, lows = {}, {}, {}
    for k in range(1, MAX_H + 1):
        lab = sig + k * STEP
        j = int(t.get_indexer([lab])[0]) if lab.date() == sig.date() else -1
        if j >= 0:
            closes[k], highs[k], lows[k] = (float(prem[c].iat[j]) for c in ("close", "high", "low"))
    out = {**{k: und[k] for k in ("key", "direction", "split", "day")}, "risk": risk}
    for k in range(1, MAX_H + 1):
        v = closes[k] - entry if k in closes else None
        out[f"und{k}"], out[f"opt{k}"], out[f"opt{k}_r"] = und[f"ret{k}"], v, _div(v, risk)
    out["opt_mfe_r"] = _div(max(highs.values()) - entry, risk) if highs else None
    out["opt_mae_r"] = _div(entry - min(lows.values()), risk) if lows else None
    uf = next((k for k in range(1, MAX_H + 1) if (und[f"ret{k}"] or 0.0) > 0), None)
    of = next((k for k in range(1, MAX_H + 1) if k in closes and closes[k] > entry), None)
    out.update(und_first=uf, opt_first=of, lag=None if uf is None or of is None else of - uf)
    u3, o3 = und["ret3"], out["opt3"]
    out["premium_per_point"] = _div(o3, u3) if u3 and u3 > 0 and o3 is not None else None
    return out


def translation_stats(items: Sequence[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"signals": len(items)}
    for k in (3, 5):
        out[f"underlying_acc{k}"] = _share([None if x[f"und{k}"] in (None, 0) else x[f"und{k}"] > 0 for x in items])
        out[f"option_acc{k}"] = _share([None if x[f"opt{k}"] in (None, 0) else x[f"opt{k}"] > 0 for x in items])
        out[f"option_mean_ret{k}_r"] = _avg([x[f"opt{k}_r"] for x in items])
    moved = [x for x in items if (x["und3"] or 0) > 0 and x["opt3"] is not None]
    out["option_response_rate"] = _share([x["opt3"] > 0 for x in moved])
    out.update(option_mfe_r=_avg([x["opt_mfe_r"] for x in items]), option_mae_r=_avg([x["opt_mae_r"] for x in items]),
               option_mfe_ge_2r_share=_share([None if x["opt_mfe_r"] is None else x["opt_mfe_r"] >= 2.0
                                              for x in items]),
               avg_option_lag_candles=_avg([x["lag"] for x in items]),
               median_premium_per_point=_med([x["premium_per_point"] for x in items]))
    return out


def translation_verdict(items: Sequence[dict], und_verdict: str) -> dict[str, Any]:
    per = {k: translation_stats([x for x in items if x["split"] == k]) for k in SPLITS}
    pooled = translation_stats([x for x in items if x["split"] in PRE_OOS])
    ua, oa = pooled[f"underlying_acc{EDGE_H}"], pooled[f"option_acc{EDGE_H}"]
    gap = None if ua is None or oa is None else round(ua - oa, 4)
    loss = ((pooled["option_response_rate"] is not None and pooled["option_response_rate"] < RESPONSE_OK)
            or (gap is not None and gap > TRANSLATION_GAP)
            or (und_verdict == UNDERLYING_EDGE and (pooled[f"option_mean_ret{EDGE_H}_r"] or 0.0) <= 0))
    has_edge = und_verdict == UNDERLYING_EDGE
    label = T_LOSS if loss else T_EDGE if has_edge and (pooled[f"option_mean_ret{EDGE_H}_r"] or 0.0) > 0 else T_NONE
    return {"label": label, "underlying_edge": has_edge, "option_translation_edge": label == T_EDGE,
            "option_translation_loss": label == T_LOSS, "accuracy_gap": gap, "per_split": per,
            "train_validation": pooled}


# ------------------------------------------------------------------ PART C - entry timing audit
ENTRY_METHODS = {
    "NEXT_OPEN": "next available option candle open after the S2 signal (unchanged S2)",
    "NEXT_CONFIRM": "the NIFTY candle after the signal closes beyond the breakout level; entry at the following open",
    "BREAKOUT_CONTINUATION": "the NIFTY candle after the signal closes beyond the signal candle's high (BUY) / low "
                             "(SELL); entry at the following open",
    "PULLBACK": f"within {PULLBACK_ENTRY_BARS} candles after the signal a NIFTY candle trades back to the breakout "
                "level (low <= level BUY / high >= level SELL) and closes beyond it; entry at the following open",
}
BASE_ENTRY = "NEXT_OPEN"
_OUTCOME_KEYS = ("outcome", "pnl", "r_multiple", "exit_reason", "exit_time", "exit_price", "stop_loss", "target",
                 "entry", "entry_time", "premium_atr")


def entry_ref(method: str, r: dict, bx: BarIndex, cutoff: time) -> tuple[pd.Timestamp | None, str | None]:
    """Decision candle (completed NIFTY candle after the signal) or (None, skip reason)."""
    i = bx.pos.get(dra._ts(r["signal_time"]))
    if i is None:
        return None, "SIGNAL_BAR_MISSING"
    s, lvl = _sign(r["direction"]), dg._num(r.get("breakout_level"))
    if method in ("NEXT_CONFIRM", "PULLBACK") and lvl is None:
        return None, "LEVEL_UNAVAILABLE"
    for k in range(1, (PULLBACK_ENTRY_BARS if method == "PULLBACK" else 1) + 1):
        j = i + k
        if not bx.same_day(i, j):
            return None, "SESSION_END"
        if bx.ts[j].time() >= cutoff:
            return None, "ENTRY_AFTER_CUTOFF"
        c = s * bx.c[j]
        if method == "NEXT_CONFIRM":
            ok = c > s * lvl
        elif method == "BREAKOUT_CONTINUATION":
            ok = c > (bx.h[i] if s > 0 else -bx.l[i])
        else:
            ok = (bx.l[j] <= lvl if s > 0 else bx.h[j] >= lvl) and c > s * lvl
        if ok:
            return bx.ts[j], None
        if method != "PULLBACK":
            return None, "NO_CONFIRMATION"
    return None, "NO_PULLBACK"


def _skipped(row: dict, reason: str) -> dict:
    return dict(row, taken=False, decision="SKIPPED", state=sh.SKIPPED, skip_reason=reason,
                **{k: None for k in _OUTCOME_KEYS})


def shifted_row(r: dict, ctx: dict | None, bx: BarIndex, params, square_off: str, cutoff: time, method: str) -> dict:
    """Same signal / contract / S2 stop (1.5 x premium ATR) and 2R target; only the entry candle moves later."""
    row = dict(r, entry_method=method, taken=True)
    if method == BASE_ENTRY or r.get("decision") != imp.ACCEPTED or ctx is None or ctx.get("data_status") != "OK":
        return row
    ref, why = entry_ref(method, r, bx, cutoff)
    if ref is None:
        return _skipped(row, why)
    sig = replace(ctx["signal"], timestamp=ref.to_pydatetime())
    lv = cds.option_levels(sig, dra._contract(r, ctx.get("tick")), ctx["premium"], params)
    if lv.get("premium_atr") is None or lv.get("entry") is None or lv.get("entry_time") is None:
        raw = dict(row, decision=imp.ACCEPTED, outcome=imp.UNAVAILABLE, data_reason=lv.get("reason"),
                   **{k: None for k in _OUTCOME_KEYS if k != "outcome"})
        return sh.classify_row(raw)
    c = dict(ctx, entry_time=imp._ist(lv["entry_time"]), entry=float(lv["entry"]), premium_atr=float(lv["premium_atr"]))
    res = imp.evaluate(c, STOP_ATR, BASE_TARGET, square_off, sh.LOT_SIZE)
    raw = dict(row, decision=imp.ACCEPTED, entry_time=c["entry_time"].isoformat(), entry=c["entry"],
               premium_atr=c["premium_atr"], data_reason=None, decision_time=ref.isoformat(), **res)
    return sh.classify_row(raw)


def dedupe(rows: Sequence[dict]) -> list[dict]:
    """Two signals mapped onto the same option entry candle = one trade (the earlier signal keeps it)."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda x: (dra._ts(x["signal_time"]), x["direction"])):
        k = (r.get("instrument_key"), r.get("entry_time"), r["direction"])
        if r.get("state") == sh.EVALUATED and k in seen:
            r = _skipped(r, "DUPLICATE_ENTRY")
        elif r.get("state") == sh.EVALUATED:
            seen.add(k)
        out.append(r)
    return out


def entry_universe(rows: Sequence[dict], ctx_by_key: dict, bx: BarIndex, params, square_off: str,
                   cutoff: time) -> dict[str, list[dict]]:
    return {m: dedupe([shifted_row(r, ctx_by_key.get(dra.key_of(r)), bx, params, square_off, cutoff, m)
                       for r in rows]) for m in ENTRY_METHODS}


# ------------------------------------------------------------------ PART D - exit architecture audit
EXITS = {"CURRENT_S2": BASE_TARGET, "1.5R": 1.5, "2R": 2.0, "2.5R": 2.5}
BASE_EXIT = "CURRENT_S2"
EXIT_ALIASES = {"2R": BASE_EXIT}      # the current S2 target IS 2R; re-evaluated to prove the engine reproduces S2
TARGET_REASONS, STOP_REASONS = ("TARGET", "TARGET_GAP"), ("STOP_LOSS", "STOP_LOSS_GAP")


def exit_row(r: dict, ctx: dict | None, square_off: str, target_r: float, name: str) -> dict:
    row = dict(r, exit_variant=name)
    if name == BASE_EXIT or r.get("decision") != imp.ACCEPTED or ctx is None or ctx.get("data_status") != "OK":
        return row
    res = imp.evaluate(ctx, STOP_ATR, target_r, square_off, sh.LOT_SIZE)
    return sh.classify_row(dict(row, data_reason=None, **res))


def exit_universe(rows: Sequence[dict], ctx_by_key: dict, square_off: str) -> dict[str, list[dict]]:
    return {n: [exit_row(r, ctx_by_key.get(dra.key_of(r)), square_off, t, n) for r in rows] for n, t in EXITS.items()}


def excursion(r: dict, ctx: dict | None, square_off: str) -> dict | None:
    """Premium MFE / MAE in R from entry to the square-off (independent of the exit rule) and until the S2 exit."""
    if r.get("state") != sh.EVALUATED or ctx is None or not r.get("stop_loss") or ctx.get("premium") is None:
        return None
    prem = ctx["premium"]
    t = dra._ist_series(prem["timestamp"])
    et, so = dra._ts(r["entry_time"]), time.fromisoformat(square_off)
    risk = float(r["entry"]) - float(r["stop_loss"])
    held = prem[((t >= et) & (t.dt.date == et.date()) & (t.dt.time < so)).to_numpy()]
    if held.empty or risk <= 0:
        return None
    pre = held[(dra._ist_series(held["timestamp"]) <= dra._ts(r["exit_time"])).to_numpy()] if r.get("exit_time") \
        else held
    e = float(r["entry"])
    return {"key": dra.key_of(r), "direction": r["direction"], "split": cb.split_of(r),
            "mfe_r_eod": (float(held["high"].max()) - e) / risk, "mae_r_eod": (e - float(held["low"].min())) / risk,
            "mfe_r_before_exit": (float(pre["high"].max()) - e) / risk if len(pre) else None}


def excursion_stats(items: Sequence[dict]) -> dict[str, Any]:
    out = {"trades": len(items), "avg_mfe_r_eod": _avg([x["mfe_r_eod"] for x in items]),
           "median_mfe_r_eod": _med([x["mfe_r_eod"] for x in items]),
           "avg_mae_r_eod": _avg([x["mae_r_eod"] for x in items]),
           "avg_mfe_r_before_exit": _avg([x["mfe_r_before_exit"] for x in items])}
    for lv in MFE_LEVELS:
        out[f"mfe_reaches_{lv:g}R"] = _share([x["mfe_r_eod"] >= lv for x in items])
    return out


def exit_timing(rows: Sequence[dict]) -> dict[str, Any]:
    ev = _ev(rows)

    def minutes(r: dict) -> float:
        return (dra._ts(r["exit_time"]) - dra._ts(r["entry_time"])).total_seconds() / 60

    tgt = [minutes(r) for r in ev if r.get("exit_reason") in TARGET_REASONS and r.get("exit_time")]
    stp = [minutes(r) for r in ev if r.get("exit_reason") in STOP_REASONS and r.get("exit_time")]
    n = len(ev)
    return {"target_hit_rate": round(len(tgt) / n, 4) if n else None,
            "stop_hit_rate": round(len(stp) / n, 4) if n else None,
            "eod_exit_rate": round(sum(r.get("exit_reason") == "EOD_SQUARE_OFF" for r in ev) / n, 4) if n else None,
            "ambiguous_rate": round(sum(r.get("outcome") == imp.AMBIGUOUS for r in ev) / n, 4) if n else None,
            "avg_minutes_to_target": _avg(tgt), "median_minutes_to_target": _med(tgt),
            "avg_minutes_to_stop": _avg(stp), "median_minutes_to_stop": _med(stp)}


# ------------------------------------------------------------------ entry / exit tables and TRAIN+VALIDATION choice
def variant_tables(universe: dict[str, list[dict]]) -> dict[str, dict[str, dict[str, dict]]]:
    return {n: {s: {k: cb.metrics(cb.split_rows(cb.side(rows, s), k)) for k in SPLITS} for s in SCOPES}
            for n, rows in universe.items()}


def best_variant(tables: dict, base: str, scope: str, aliases: dict | None = None) -> dict[str, Any]:
    """Reads TRAIN and VALIDATION only: a variant replaces the base if it adds >= MIN_VARIANT_GAIN R on both."""
    b, gains = tables[base][scope], {}
    for n, t in tables.items():
        if n == base or n in (aliases or {}):
            continue
        g = [None if t[scope][k]["average_r"] is None or b[k]["average_r"] is None
             else t[scope][k]["average_r"] - b[k]["average_r"] for k in PRE_OOS]
        gains[n] = None if None in g else round(min(g), 4)
    beaters = [n for n, g in gains.items() if g is not None and g >= MIN_VARIANT_GAIN]
    best = max(beaters, key=lambda n: (tables[n][scope]["TRAIN"]["average_r"], n)) if beaters else base
    positive = all((tables[best][scope][k]["average_r"] or -1.0) > 0 and _pf(tables[best][scope][k]) >= 1.0
                   for k in PRE_OOS)
    return {"best": best, "improved": best != base, "positive_on_train_and_validation": positive,
            "min_gain_train_validation_r": gains}


def choice_text(choice: dict[str, dict], base: str, what: str) -> str:
    parts = []
    for s in DIRECTIONS:
        c = choice[s]
        if not c["improved"]:
            parts.append(f"{s}: no predefined {what} added >= {MIN_VARIANT_GAIN}R expectancy on BOTH TRAIN and "
                         f"VALIDATION vs {base}")
        else:
            parts.append(f"{s}: {c['best']} adds >= {MIN_VARIANT_GAIN}R on TRAIN and VALIDATION"
                         + ("" if c["positive_on_train_and_validation"] else
                            " but expectancy / PF stay negative (does not create an edge)"))
    return "; ".join(parts)


# ------------------------------------------------------------------ PART E - alternative strategy families
@dataclass(frozen=True)
class Spec:
    sid: str
    family: str
    variant: int
    name: str
    rationale: str
    rule: str


SPECS: tuple[Spec, ...] = (
    Spec("S8-TC1", "TREND_CONTINUATION", 1, "TREND_PULLBACK_RESUME",
         "trends resume after shallow pullbacks; buy strength after weakness inside the trend",
         f"TREND_UP (close > EMA20 > EMA50, EMA20 rising) AND a low touched EMA20 in the last {PULLBACK_BARS} candles "
         "AND the candle closes above the previous candle's high and above its open"),
    Spec("S8-TC2", "TREND_CONTINUATION", 2, "STRONG_TREND_PULLBACK_RESUME",
         "only established trends (EMA separation >= 1 ATR) have enough momentum to resume",
         f"S8-TC1 AND (EMA20 - EMA50) / ATR14 >= {STRONG_TREND_ATR:g}"),
    Spec("S8-TC3", "TREND_CONTINUATION", 3, "TREND_PULLBACK_6BAR_BREAK",
         "continuation is confirmed when the pullback's swing high is taken out",
         f"TREND_UP AND a low touched EMA20 in the last {PULLBACK_BARS} candles AND close > highest high of the "
         "previous 6 candles"),
    Spec("S8-VP1", "VWAP_PULLBACK", 1, "VWAP_PULLBACK_RESUME",
         "institutional reference price acts as support in an up-day; buy the bounce away from it",
         f"close > session VWAP (TWAP; index volume is zero) AND EMA20 > EMA50 AND a low touched VWAP in the last "
         f"{PULLBACK_BARS} candles AND close > previous high AND close > open"),
    Spec("S8-VP2", "VWAP_PULLBACK", 2, "VWAP_PULLBACK_IN_TREND",
         "VWAP bounces work best when the trend regime agrees", "S8-VP1 AND TREND_UP"),
    Spec("S8-VP3", "VWAP_PULLBACK", 3, "VWAP_PULLBACK_MOMENTUM",
         "the bounce needs momentum but not exhaustion",
         f"S8-VP1 AND {RSI_MOMENTUM[0]:g} <= RSI14 < {RSI_MOMENTUM[1]:g} (SELL mirrored)"),
    Spec("S8-BR1", "BREAKOUT_RETEST", 1, "ORB_BREAKOUT_RETEST",
         "a breakout that survives a retest of the broken level is less likely to be false (S2 enters the raw break)",
         "close crosses above the S2 opening-range high, a later candle's low retests it while closing above, then a "
         "close above that retest candle's high; any close back inside resets"),
    Spec("S8-BR2", "BREAKOUT_RETEST", 2, "ORB30_BREAKOUT_RETEST",
         "a wider 30-minute opening range filters early noise", "same as S8-BR1 on the 09:15-09:45 range"),
    Spec("S8-BR3", "BREAKOUT_RETEST", 3, "PREV_DAY_LEVEL_RETEST",
         "previous-day high / low are widely watched levels", "same as S8-BR1 on the previous session's high / low"),
    Spec("S8-MR1", "MEAN_REVERSION", 1, "VWAP_STRETCH_REVERSAL",
         "statistically extended moves away from the session mean tend to revert",
         f"(close - VWAP) / ATR14 <= -{STRETCH_ATR:g} in the last {STRETCH_BARS} candles AND the candle closes above "
         "the previous high and above its open (BUY; SELL mirrored)"),
    Spec("S8-MR2", "MEAN_REVERSION", 2, "RSI_EXTREME_REVERSAL",
         "momentum extremes exhaust",
         f"RSI14 <= {RSI_EXTREME:g} in the last {STRETCH_BARS} candles AND reversal candle (SELL: RSI >= "
         f"{100 - RSI_EXTREME:g})"),
    Spec("S8-MR3", "MEAN_REVERSION", 3, "VWAP_STRETCH_REVERSAL_NOT_AGAINST_TREND",
         "do not fade an established opposite trend", "S8-MR1 AND NOT TREND_DOWN (SELL: NOT TREND_UP)"),
)
FAMILIES = tuple(dict.fromkeys(s.family for s in SPECS))


def signal_frame(bars: pd.DataFrame, orb: dict) -> pd.DataFrame:
    """Per-candle inputs known at the candle's close (bars come from dg.feature_bars, all causal)."""
    f = pd.DataFrame(index=bars.index)
    for c in ("open", "high", "low", "close", "ema20", "ema50", "atr14", "ema20_slope_atr", "rsi14", "hh6", "ll6",
              "orb30_high", "orb30_low", "pdh", "pdl"):
        f[c] = bars[c].astype(float) if c in bars else np.nan
    vw = bars["vwap"].astype(float) if "vwap" in bars else pd.Series(np.nan, index=bars.index)
    f["ref"] = vw.where(vw.notna(), bars["twap"].astype(float))
    days = pd.Series([t.date() for t in f.index], index=f.index)
    last = days.map(lambda d: orb[d][2] if d in orb else pd.NaT)
    known = pd.Series([pd.notna(x) and t > x for t, x in zip(f.index, last)], index=f.index)
    f["orb_high"] = days.map(lambda d: orb[d][0] if d in orb else np.nan).where(known)
    f["orb_low"] = days.map(lambda d: orb[d][1] if d in orb else np.nan).where(known)
    return f


def _mirror(g: pd.DataFrame, s: float) -> dict[str, np.ndarray]:
    v = lambda c: g[c].to_numpy(float)  # noqa: E731
    if s > 0:
        return {"c": v("close"), "o": v("open"), "h": v("high"), "l": v("low"), "e20": v("ema20"), "e50": v("ema50"),
                "atr": v("atr14"), "slope": v("ema20_slope_atr"), "rsi": v("rsi14"), "ref": v("ref"),
                "hh6": v("hh6"), "BR1": v("orb_high"), "BR2": v("orb30_high"), "BR3": v("pdh")}
    return {"c": -v("close"), "o": -v("open"), "h": -v("low"), "l": -v("high"), "e20": -v("ema20"),
            "e50": -v("ema50"), "atr": v("atr14"), "slope": -v("ema20_slope_atr"), "rsi": 100.0 - v("rsi14"),
            "ref": -v("ref"), "hh6": -v("ll6"), "BR1": -v("orb_low"), "BR2": -v("orb30_low"), "BR3": -v("pdl")}


def _trend_with(a: dict, i: int) -> bool:
    return bool(a["c"][i] > a["e20"][i] > a["e50"][i] and a["slope"][i] > 0)


def _trend_against(a: dict, i: int) -> bool:
    return bool(a["c"][i] < a["e20"][i] < a["e50"][i] and a["slope"][i] < 0)


def _resume(a: dict, i: int) -> bool:
    return bool(i >= 1 and a["c"][i] > a["h"][i - 1] and a["c"][i] > a["o"][i])


def _touched(a: dict, i: int, key: str, n: int = PULLBACK_BARS) -> bool:
    return any(a["l"][j] <= a[key][j] for j in range(max(0, i - n), i))


def _rule_tc(a: dict, i: int, v: int, state: dict) -> tuple[bool, float]:
    if not (_trend_with(a, i) and _touched(a, i, "e20")):
        return False, a["e20"][i]
    if v == 3:
        return bool(a["c"][i] > a["hh6"][i]), a["e20"][i]
    ok = _resume(a, i)
    if v == 2:
        sep = _div(a["e20"][i] - a["e50"][i], a["atr"][i])
        ok = ok and sep is not None and sep >= STRONG_TREND_ATR
    return ok, a["e20"][i]


def _rule_vp(a: dict, i: int, v: int, state: dict) -> tuple[bool, float]:
    ok = bool(a["c"][i] > a["ref"][i] and a["e20"][i] > a["e50"][i]) and _touched(a, i, "ref") and _resume(a, i)
    if v == 2:
        ok = ok and _trend_with(a, i)
    elif v == 3:
        ok = ok and bool(RSI_MOMENTUM[0] <= a["rsi"][i] < RSI_MOMENTUM[1])
    return ok, a["ref"][i]


def _rule_br(a: dict, i: int, v: int, state: dict) -> tuple[bool, float]:
    lv, c = a[f"BR{v}"][i], a["c"][i]
    if not np.isfinite(lv):
        state.clear()
        return False, lv
    phase = state.get("phase", 0)
    if phase == 0:
        if i >= 1 and a["c"][i - 1] <= lv < c:
            state["phase"] = 1
        return False, lv
    if c <= lv:
        state["phase"] = 0
        return False, lv
    if phase == 1:
        if a["l"][i] <= lv:
            state.update(phase=2, retest_high=a["h"][i])
        return False, lv
    if c > state["retest_high"]:
        state["phase"] = 0
        return True, lv
    if a["l"][i] <= lv:
        state["retest_high"] = a["h"][i]
    return False, lv


def _rule_mr(a: dict, i: int, v: int, state: dict) -> tuple[bool, float]:
    lo = max(0, i - STRETCH_BARS)
    if v == 2:
        stretched = any(a["rsi"][j] <= RSI_EXTREME for j in range(lo, i))
    else:
        stretched = any((_div(a["c"][j] - a["ref"][j], a["atr"][j]) or 0.0) <= -STRETCH_ATR for j in range(lo, i))
    ok = stretched and _resume(a, i)
    if v == 3:
        ok = ok and not _trend_against(a, i)
    return ok, a["ref"][i]


RULES: dict[str, Callable[[dict, int, int, dict], tuple[bool, float]]] = {
    "TREND_CONTINUATION": _rule_tc, "VWAP_PULLBACK": _rule_vp, "BREAKOUT_RETEST": _rule_br, "MEAN_REVERSION": _rule_mr}


def generate(frame: pd.DataFrame, spec: Spec, direction: str, first: time, cutoff: time) -> list[tuple]:
    """(timestamp, close, level) of every signal candle. Rules read the current and earlier candles of the same
    session only; signals fire strictly after the opening range and before the entry cutoff, at most
    MAX_SIGNALS_PER_DAY per side and COOLDOWN_BARS apart."""
    s, rule, out = _sign(direction), RULES[spec.family], []
    for _, g in frame.groupby([t.date() for t in frame.index], sort=True):
        a, ts, close = _mirror(g, s), g.index, g["close"].to_numpy(float)
        state: dict = {}
        last, n = -10 ** 9, 0
        for i in range(len(g)):
            fire, lv = rule(a, i, spec.variant, state)
            t = ts[i].time()
            if not fire or not (first < t < cutoff) or i - last < COOLDOWN_BARS or n >= MAX_SIGNALS_PER_DAY:
                continue
            out.append((ts[i], float(close[i]), None if not np.isfinite(lv) else float(s * lv)))
            last, n = i, n + 1
    return out


def alt_signals(bars: pd.DataFrame, specs: Sequence[Spec], range_start: str, range_end: str,
                cutoff: time) -> dict[str, list[ResearchSignal]]:
    frame = signal_frame(bars, s6.orb_ranges(bars, range_start, range_end))
    first = time.fromisoformat(range_end)
    out: dict[str, list[ResearchSignal]] = {}
    for sp in specs:
        sigs = []
        for d in DIRECTIONS:
            for ts, close, lv in generate(frame, sp, d, first, cutoff):
                cond = {"family": sp.family} if lv is None else \
                    {"family": sp.family, ("above" if d == "BUY" else "below"): lv}
                sigs.append(ResearchSignal(ts.to_pydatetime(), d, sp.sid, "NIFTY", close, cond))
        out[sp.sid] = sorted(sigs, key=lambda x: (x.timestamp, x.direction))
    return out


def replay_signals(sigs: Sequence[ResearchSignal], feats: pd.DataFrame, resolve: Callable, premium_for: Callable,
                   params, square_off: str, strategy: str) -> tuple[list[dict], list[dict], dict]:
    """The exact S2 execution path (build_contexts + run_candidate + classify_row) for another signal stream."""
    ctxs = imp.build_contexts(sigs, feats, resolve, premium_for, params)
    raw = dr.run_candidate(ctxs, feats, dr.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE,
                           cache={}) if ctxs else []
    ctx_by_key = {(c["signal_time"].isoformat(), c["direction"]): c for c in ctxs}
    rows = []
    for r in raw:
        row = dict(sh.classify_row(r, {}), strategy=strategy)
        ctx = ctx_by_key.get(dra.key_of(row), {})
        meta = dra.parse_symbol(row.get("contract"))
        row.update(strike=meta["strike"], option_type=meta["option_type"],
                   expiry_date=ctx.get("expiry") or meta["expiry"], tick_inferred=ctx.get("tick"))
        rows.append(row)
    return rows, ctxs, ctx_by_key


def alt_pipeline(specs: Sequence[Spec], resolve: Callable, params, range_start: str, range_end: str,
                 cutoff: time) -> Callable:
    """pipeline(spot, premium_for, days) -> (contexts, features) for the future-candle scrambling test."""
    def pipeline(spot: pd.DataFrame, premium_for: Callable, days: set | None = None):
        feats = imp.underlying_features(spot, range_start, range_end, params.atr_period)
        sigs = alt_signals(dg.feature_bars(spot), specs, range_start, range_end, cutoff)
        ctxs: list[dict] = []
        for sp in specs:
            keep = [x for x in sigs[sp.sid] if days is None or imp._ist(x.timestamp).date() in days]
            ctxs += imp.build_contexts(keep, feats, resolve, premium_for, params)
        return ctxs, feats
    return pipeline


# ------------------------------------------------------------------ PART G - option premium confirmation
CONFIRMS: dict[str, tuple[str, Callable[[dict], bool | None]]] = {
    "PREM_CANDLE_UP": ("completed bought-option candle at the signal closes above its open",
                       lambda r: None if fv(r, "premium_candle") in (None, UNAVAILABLE)
                       else fv(r, "premium_candle") == "BULLISH"),
    "PREM_ABOVE_PREV": ("bought-option close above the previous completed candle's close",
                        lambda r: fv(r, "premium_above_prev")),
    "PREM_MOMENTUM": ("bought-option close > open and > close 3 candles earlier", lambda r: r.get("opt_confirm")),
    "PREM_ATR_NORMAL": ("premium ATR14 / premium close between 5% and 15%", s6._between("premium_atr_pct", 5.0, 15.0)),
}


@dataclass(frozen=True)
class Alt:
    cid: str
    sid: str
    direction: str
    confirm: str | None = None

    @property
    def spec(self) -> Spec:
        return next(s for s in SPECS if s.sid == self.sid)

    @property
    def complexity(self) -> int:
        return 1 + (self.confirm is not None)

    @property
    def name(self) -> str:
        return f"{self.direction}_{self.spec.name}" + (f"+{self.confirm}" if self.confirm else "")

    def describe(self) -> str:
        return self.spec.rule + (f" AND {CONFIRMS[self.confirm][0]}" if self.confirm else "") + \
            (" (SELL mirrored)" if self.direction == "SELL" else "")


ALTS: tuple[Alt, ...] = tuple(Alt(f"{s.sid}-{d[0]}", s.sid, d) for s in SPECS for d in DIRECTIONS)


def confirm_variants(c: Alt | None) -> list[Alt]:
    if c is None or c.confirm is not None:
        return []
    return [replace(c, cid=f"{c.cid}+{k}", confirm=k) for k in CONFIRMS]


def cand_rows(c: Alt, alt_rows: dict[str, list[dict]]) -> list[dict]:
    """Rows of the candidate's family variant and side; a confirmation must be True (missing option data -> NO
    SIGNAL)."""
    return [r for r in alt_rows.get(c.sid, []) if r["direction"] == c.direction
            and (c.confirm is None or CONFIRMS[c.confirm][1](r) is True)]


# ------------------------------------------------------------------ PART H / I - clean selection with quality scoring
def train_eval(c: Alt, alt_rows: dict) -> dict[str, Any]:
    rows = cb.split_rows(cand_rows(c, alt_rows), "TRAIN")
    m = cb.metrics(rows)
    gp, pf, n = m["gross_profit"], _pf(m), m["total_trades"]
    months = {mo: cb.metrics([r for r in rows if str(r["session_date"])[:7] == mo])["net_pnl"] for mo in TRAIN_MONTHS}
    checks = {
        "meaningful_trade_count": n >= MIN_TRAIN_TRADES,
        "positive_expectancy": (m["average_r"] or 0.0) > 0,
        "profit_factor": pf >= MIN_TRAIN_PF,
        "controlled_drawdown": gp > 0 and m["max_drawdown"] <= MAX_DD_SHARE * gp,
        "losing_streak": m["max_losing_streak"] <= MAX_TRAIN_STREAK,
        "monthly_stability": sum(v > 0 for v in months.values()) >= MIN_PROFITABLE_MONTHS,
        "regime_stability": not s6.concentration(rows, lambda r: fv(r, "market_regime"))["concentrated"],
        "no_profit_concentration": not s6.top_trade_dependency(rows)["concentrated"],
    }
    ar = m["average_r"] if m["average_r"] is not None else -math.inf
    return {"cid": c.cid, "metrics": m, "months": months, "checks": checks, "eligible": all(checks.values()),
            "failed": [k for k, ok in checks.items() if not ok],
            "score": (ar * math.sqrt(n) if n and ar != -math.inf else -math.inf, pf, n)}


def rank_key(c: Alt, tr: dict) -> tuple:
    s = tr[c.cid]["score"]
    return (-s[0], -s[1], -s[2], c.complexity, c.cid)


def validation_eval(c: Alt, alt_rows: dict, train_m: dict) -> dict[str, Any]:
    v = split_m(cand_rows(c, alt_rows), "VALIDATION")
    if v["total_trades"] < MIN_VAL_TRADES:
        verdict, failed = INSUFFICIENT, ["sample_not_tiny"]
    else:
        checks = {"no_collapse": v["net_pnl"] > 0 and _pf(v) >= 1.0,
                  "no_extreme_degradation": _pf(v) >= MIN_VAL_PF_RETENTION * _pf(train_m)}
        failed = [k for k, ok in checks.items() if not ok]
        verdict = VALIDATED if not failed else REJECTED
    return {"cid": c.cid, "metrics": v, "verdict": verdict, "failed": failed}


def _assert_pre(alt_rows: dict) -> None:
    late = sum(1 for rows in alt_rows.values() for r in rows if cb.day_of(r) > SELECTION_END)
    if late:
        raise ValueError(f"selection received {late} rows after {SELECTION_END} (OOS/holdout must not be used)")


def pre_rows(alt_rows: dict) -> dict:
    return {k: [r for r in rows if cb.day_of(r) <= SELECTION_END] for k, rows in alt_rows.items()}


def select(alt_rows: dict) -> dict[str, Any]:
    """Per side: rank on TRAIN quality; the best TRAIN-eligible candidate that VALIDATES is picked; its premium
    confirmation variants (Part G) replace it only if TRAIN-eligible, VALIDATED and better ranked on TRAIN."""
    _assert_pre(alt_rows)
    out: dict[str, Any] = {"train": {}, "validation": {}, "ranking": {}, "stage2": {}, "candidates": {},
                           "selected": {}, "counts": {}}
    for d in DIRECTIONS:
        pool = [c for c in ALTS if c.direction == d]
        tr = {c.cid: train_eval(c, alt_rows) for c in pool}
        order = sorted((c for c in pool if tr[c.cid]["eligible"]), key=lambda c: rank_key(c, tr))
        va = {c.cid: validation_eval(c, alt_rows, tr[c.cid]["metrics"]) for c in order}
        pick = next((c for c in order if va[c.cid]["verdict"] == VALIDATED), None)
        variants = confirm_variants(pick)
        for v in variants:
            tr[v.cid] = train_eval(v, alt_rows)
            if tr[v.cid]["eligible"]:
                va[v.cid] = validation_eval(v, alt_rows, tr[v.cid]["metrics"])
        better = [v for v in variants if tr[v.cid]["eligible"] and va[v.cid]["verdict"] == VALIDATED
                  and rank_key(v, tr)[:3] < rank_key(pick, tr)[:3]]
        if better:
            pick = min(better, key=lambda c: rank_key(c, tr))
        out["train"].update(tr)
        out["validation"].update(va)
        out["ranking"][d] = [c.cid for c in order]
        out["stage2"][d] = [v.cid for v in variants]
        out["candidates"].update({c.cid: c for c in pool + variants})
        out["counts"][d] = len(pool) + len(variants)
        out["selected"][d] = pick
    return out


def ids(sel: dict) -> dict[str, str | None]:
    return {d: None if c is None else c.cid for d, c in sel["selected"].items()}


def fingerprint(sel: dict, entry_best: dict | None = None, exit_best: dict | None = None) -> str:
    spec = {"selected": {d: None if c is None else asdict(c) for d, c in sel["selected"].items()},
            "alts": [asdict(c) for c in ALTS], "specs": [asdict(s) for s in SPECS], "ranking": sel["ranking"],
            "entry": entry_best, "exit": exit_best}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _best_ids(tables: dict, base: str, aliases: dict | None = None) -> dict[str, str]:
    return {s: best_variant(tables, base, s, aliases)["best"] for s in SCOPES}


def isolation(alt_rows: dict, sel: dict, entries: dict, exits: dict) -> dict[str, Any]:
    """OOS / holdout isolation + determinism for the alternative selection and the entry / exit choices."""
    checks: dict[str, bool] = {}
    if alt_rows:
        aug_sep = select(pre_rows({k: s5._flip(v, SELECTION_END) for k, v in alt_rows.items()}))
        sep = select(pre_rows({k: s5._flip(v, SPLITS["OOS"][1]) for k, v in alt_rows.items()}))
        refuses = False
        try:
            select(alt_rows)
        except ValueError:
            refuses = True
        pre = pre_rows(alt_rows)
        rev = select({k: list(reversed(v)) for k, v in pre.items()})
        val = select({k: s5._flip(v, TRAIN_END) for k, v in pre.items()})
        checks.update({
            "oos_cannot_affect_alternative_selection": ids(sel) == ids(aug_sep),
            "holdout_cannot_affect_alternative_selection": ids(sel) == ids(sep),
            "selection_refuses_post_july_rows": refuses,
            "alternative_ranking_deterministic": fingerprint(sel) == fingerprint(rev),
            "train_ranking_uses_train_only": sel["ranking"] == val["ranking"],
            "candidate_count_within_limit": all(n <= MAX_CANDIDATES_PER_SIDE for n in sel["counts"].values()),
        })
    for name, uni, base, al in (("entry", entries, BASE_ENTRY, None), ("exit", exits, BASE_EXIT, EXIT_ALIASES)):
        best = _best_ids(variant_tables(uni), base, al)
        flipped = _best_ids(variant_tables({k: s5._flip(v, SELECTION_END) for k, v in uni.items()}), base, al)
        rev = _best_ids(variant_tables({k: list(reversed(v)) for k, v in uni.items()}), base, al)
        checks[f"{name}_choice_ignores_oos_and_holdout"] = best == flipped
        checks[f"{name}_choice_deterministic"] = best == rev
    checks["family_and_variant_limits"] = len(FAMILIES) <= MAX_FAMILIES and all(
        sum(s.family == f for s in SPECS) <= MAX_VARIANTS for f in FAMILIES)
    return {"checks": checks, "passed": all(checks.values())}


# ------------------------------------------------------------------ PART K - lookahead checks specific to S8
def signal_truncation_check(spot: pd.DataFrame, sigs: dict[str, list[ResearchSignal]], range_start: str,
                            range_end: str, cutoff: time, sample: int = 20) -> dict[str, Any]:
    """No future signal leakage: with every NIFTY candle after a signal candle removed, the signal (and the signal /
    no-signal state of the two earlier candles) must be identical."""
    full = {sid: {(imp._ist(x.timestamp), x.direction) for x in v} for sid, v in sigs.items()}
    items = sorted((imp._ist(x.timestamp), x.direction, sid) for sid, v in sigs.items() for x in v)
    picked = items[::max(1, len(items) // max(sample, 1))][:sample]
    t_all = dra._ist_series(spot["timestamp"])
    bad = []
    for ts, d, sid in picked:
        tb = dg.feature_bars(spot[(t_all <= ts).to_numpy()].reset_index(drop=True))
        spec = next(s for s in SPECS if s.sid == sid)
        again = {(imp._ist(x.timestamp), x.direction) for x in alt_signals(tb, [spec], range_start, range_end,
                                                                             cutoff)[sid]}
        for k in range(3):
            key = (ts - k * STEP, d)
            if (key in full[sid]) != (key in again):
                bad.append(f"{sid}|{key[0].isoformat()}|{d}")
    return {"checked": len(picked), "mismatches": bad[:20], "passed": not bad, "vacuous": not picked}


def entry_causality_check(entries: dict[str, list[dict]], ctx_by_key: dict, bars: pd.DataFrame, params,
                          square_off: str, cutoff: time, sample: int = 20) -> dict[str, Any]:
    """Shifted entries: entry, premium ATR, stop and target are unchanged with every NIFTY candle after the decision
    candle removed and every premium candle after entry tripled; entry = a real premium candle open."""
    bad, synthetic, checked = [], [], 0
    for m in ENTRY_METHODS:
        if m == BASE_ENTRY:
            continue
        ev = _ev(entries[m])
        for r in ev[::max(1, len(ev) // max(sample, 1))][:sample]:
            ctx = ctx_by_key.get(dra.key_of(r))
            if ctx is None:
                continue
            checked += 1
            et, prem = dra._ts(r["entry_time"]), ctx["premium"]
            hit = prem[(dra._ist_series(prem["timestamp"]) == et).to_numpy()]
            if hit.empty or not math.isclose(round_to_tick(float(hit["open"].iat[0]), ctx.get("tick"), "nearest"),
                                             float(r["entry"]), abs_tol=1e-9):
                synthetic.append(f"{r['signal_time']}|{r['direction']}|{m}")
            tb = BarIndex(bars[bars.index <= et - STEP])
            again = shifted_row(r, dict(ctx, premium=s6._warp_after(prem, et)), tb, params, square_off, cutoff, m)
            bad += [f"{r['signal_time']}|{r['direction']}|{m}|{k}" for k in ("entry", "entry_time", "premium_atr",
                                                                            "stop_loss", "target")
                    if not s6._same(r.get(k), again.get(k))]
    return {"checked": checked, "mismatches": bad[:20], "synthetic_or_wrong_entry": synthetic[:20],
            "passed": not bad and not synthetic, "vacuous": checked == 0}


def exit_engine_check(base: Sequence[dict], two_r: Sequence[dict]) -> dict[str, Any]:
    """The 2R re-evaluation must reproduce every current S2 outcome and P&L exactly."""
    a = {dra.key_of(r): (r.get("outcome"), r.get("pnl")) for r in base}
    b = {dra.key_of(r): (r.get("outcome"), r.get("pnl")) for r in two_r}
    diff = [str(k) for k in a if a[k] != b.get(k)]
    return {"checked": len(a), "mismatches": diff[:10], "passed": bool(a) and not diff}


def _skip_check(why: str) -> dict[str, Any]:
    return {"passed": True, "not_applicable": True, "detail": why}


# ------------------------------------------------------------------ PART L - decision
def s2_option_edge(s2: dict) -> dict[str, bool]:
    return {d: all(_pf(s2[k][d]) >= 1.0 and (s2[k][d]["average_r"] or -1.0) > 0 for k in PRE_OOS) for d in DIRECTIONS}


def decide(und: dict, trans: dict, exit_choice: dict, s2: dict, frozen: dict, clean: bool) -> dict[str, Any]:
    edge = s2_option_edge(s2)
    if any(edge.values()):
        s2_decision = D_S2_EDGE
    elif any(und[d]["verdict"] == UNDERLYING_EDGE and trans[d]["label"] == T_LOSS for d in DIRECTIONS):
        s2_decision = D_OPTION
    elif any(exit_choice[d]["improved"] and exit_choice[d]["positive_on_train_and_validation"] for d in DIRECTIONS):
        s2_decision = D_EXIT
    else:
        s2_decision = D_NO_EDGE
    promising = [k for k, b in frozen.items() if b is not None and b.get("status") == PROMISING]
    alt_decision = D_ALT if promising and clean else D_NONE
    return {"s2": s2_decision, "alternative": alt_decision, "promising": promising,
            "s2_option_level_edge": edge, "outcomes": [s2_decision, alt_decision]}


def cause_text(und: dict, entry_choice: dict, exit_choice: dict) -> str:
    if all(und[d]["verdict"] == NO_EDGE for d in DIRECTIONS):
        return ("ENTRY SIGNAL: the S2 NIFTY signal itself has no directional edge; no predefined entry or exit "
                "change can create one")

    def best_gain(ch: dict) -> float:
        return max((g for d in DIRECTIONS for g in ch[d]["min_gain_train_validation_r"].values() if g is not None),
                   default=0.0)
    e, x = best_gain(entry_choice), best_gain(exit_choice)
    if max(e, x) < MIN_VARIANT_GAIN:
        return "NEITHER: no predefined entry or exit change improves expectancy on both TRAIN and VALIDATION"
    return f"{'ENTRY' if e >= x else 'EXIT'} (largest TRAIN+VALIDATION gain: entry {e:+.3f}R vs exit {x:+.3f}R)"


# ------------------------------------------------------------------ study
def build(*, rows: Sequence[dict], contexts: Sequence[dict], ctx_by_key: dict, spot: pd.DataFrame,
          bars: pd.DataFrame, feats: pd.DataFrame, params, square_off: str, stop_new_entries: str, range_start: str,
          range_end: str, calendar, listed: Callable, exists: Callable, resolve: Callable, premium_for: Callable,
          integrity: dict, mutation: dict | None, mutation_days: Sequence[str], sessions: Sequence[str],
          dataset: dict, check_sample: int = 20, force_alternatives: bool = False,
          production_unchanged: bool | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    rows = sorted(rows, key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
    if not all("s6" in r for r in rows):
        rows = s6.prepare(rows, bars, ctx_by_key, range_start, range_end)
    cutoff, first = time.fromisoformat(stop_new_entries), time.fromisoformat(range_end)
    bx = BarIndex(bars)
    session_days = set(sessions)

    # PART A / B
    say("Part A/B: underlying signal and option translation audit ...")
    und_items = {dra.key_of(r): underlying_item(r, bx) for r in rows}
    uncond = unconditional(bx, first, cutoff)
    und: dict[str, dict] = {}
    trans: dict[str, dict] = {}
    for d in DIRECTIONS:
        items = [x for x in und_items.values() if x is not None and x["direction"] == d]
        und[d] = underlying_verdict(items, uncond)
        und[d]["first_per_day"] = underlying_verdict(first_per_day(items), uncond)
        opt = [o for o in (option_item(r, ctx_by_key.get(dra.key_of(r)), und_items[dra.key_of(r)])
                           for r in rows if r["direction"] == d) if o is not None]
        trans[d] = translation_verdict(opt, und[d]["verdict"])

    # PART C / D
    say("Part C/D: predefined entry methods and exit matrix on the same S2 signals ...")
    entries = entry_universe(rows, ctx_by_key, bx, params, square_off, cutoff)
    exits = exit_universe(rows, ctx_by_key, square_off)
    entry_tables, exit_tables = variant_tables(entries), variant_tables(exits)
    entry_choice = {s: best_variant(entry_tables, BASE_ENTRY, s) for s in SCOPES}
    exit_choice = {s: best_variant(exit_tables, BASE_EXIT, s, EXIT_ALIASES) for s in SCOPES}
    exc = [x for x in (excursion(r, ctx_by_key.get(dra.key_of(r)), square_off) for r in rows) if x is not None]
    excursions = {d: {k: excursion_stats([x for x in exc if x["direction"] == d and x["split"] == k])
                      for k in SPLITS} for d in DIRECTIONS}
    timing = {n: {s: {k: exit_timing(cb.split_rows(cb.side(v, s), k)) for k in SPLITS} for s in SCOPES}
              for n, v in exits.items()}

    # PART E / G
    lacks_edge = any(und[d]["verdict"] != UNDERLYING_EDGE for d in DIRECTIONS)
    run_alt = lacks_edge or force_alternatives
    alt_rows: dict[str, list[dict]] = {}
    alt_ctx: dict = {}
    alt_contexts: list[dict] = []
    sigs: dict[str, list[ResearchSignal]] = {}
    if run_alt:
        say(f"Part E: {len(SPECS)} predefined alternative variants ({len(FAMILIES)} families) through the S2 "
            "execution path ...")
        sigs = alt_signals(bars, SPECS, range_start, range_end, cutoff)
        for sp in SPECS:
            keep = [x for x in sigs[sp.sid] if imp._ist(x.timestamp).date().isoformat() in session_days]
            sigs[sp.sid] = keep
            arows, actx, akey = replay_signals(keep, feats, resolve, premium_for, params, square_off, sp.sid)
            arows = s5.prepare_rows(dg.annotate(arows, bars, akey), akey)
            alt_rows[sp.sid] = s6.prepare(arows, bars, akey, range_start, range_end)
            alt_contexts += actx
            alt_ctx.update(akey)

    # PART K - causality
    say("Part K: anti-lookahead checks ...")
    alt_flat = sorted((r for v in alt_rows.values() for r in v), key=lambda r: (dra._ts(r["signal_time"]),
                                                                                r["direction"]))
    na = "alternatives not researched (S2 underlying edge present)"
    alt_mut: dict[str, Any] = _skip_check(na)
    if run_alt:
        cuts = [pd.Timestamp(datetime.combine(datetime.fromisoformat(d).date(), time(12, 0)), tz=IST)
                for d in mutation_days]
        alt_mut = dr.mutation_test(alt_pipeline(SPECS, resolve, params, range_start, range_end, cutoff), spot,
                                   premium_for, [dr.BASELINE], cuts)[dr.BASELINE.name] if cuts else \
            {"passed": False, "detail": "NOT RUN"}
    lookahead = {
        "s2_feature_truncation": dg.truncation_check(spot, rows, ctx_by_key, check_sample),
        "s2_future_candle_scrambling": mutation or {"passed": False, "detail": "NOT RUN"},
        "s2_chronological_replay": dg.chronology_check(rows),
        "s2_no_future_option_price": dra.alignment_checks(contexts, params),
        "entry_methods_no_future_data": entry_causality_check(entries, ctx_by_key, bars, params, square_off, cutoff,
                                                              check_sample),
        "alt_future_candle_scrambling": alt_mut,
        "alt_no_future_signal_leakage": signal_truncation_check(spot, sigs, range_start, range_end, cutoff,
                                                                check_sample) if run_alt else _skip_check(na),
        "alt_feature_truncation": dg.truncation_check(spot, alt_flat, alt_ctx, check_sample)
        if alt_flat else _skip_check(na if not run_alt else "no alternative signals"),
        "alt_extra_truncation": s6.extra_truncation_check(spot, alt_flat, alt_ctx, range_start, range_end,
                                                          check_sample) if alt_flat else _skip_check("no rows"),
        "alt_no_future_option_price": dra.alignment_checks(alt_contexts, params)
        if any(c.get("data_status") == "OK" for c in alt_contexts) else _skip_check("no option contexts"),
        "alt_premium_confirmation_truncation": s5.option_truncation_check(alt_flat, alt_ctx, check_sample)
        if alt_flat else _skip_check("no rows"),
        "alt_chronological_replay": {"passed": all(dg.chronology_check(v)["passed"] for v in alt_rows.values())}
        if alt_rows else _skip_check(na),
    }
    lookahead_ok = all(bool(v.get("passed")) for v in lookahead.values())

    # PART H - selection (TRAIN ranking, July validation), frozen before OOS
    say("Part H: TRAIN ranking, July validation, freeze ...")
    sel = select(pre_rows(alt_rows)) if run_alt else None
    entry_best = {s: entry_choice[s]["best"] for s in SCOPES}
    exit_best = {s: exit_choice[s]["best"] for s in SCOPES}
    fp = fingerprint(sel, entry_best, exit_best) if sel else fingerprint({"selected": {}, "ranking": {}}, entry_best,
                                                                         exit_best)
    iso = isolation(alt_rows, sel, entries, exits) if sel else isolation({}, {}, entries, exits)

    # integrity
    dup = cb.duplicate_check(rows)
    ts_chk = cb.timestamp_checks(rows, square_off, stop_new_entries)
    cval = cb.contract_validity(rows, calendar, listed, exists)
    alt_checks = {
        "duplicates": {sid: cb.duplicate_check(v) for sid, v in alt_rows.items()},
        "timestamps": cb.timestamp_checks(alt_flat, square_off, stop_new_entries) if alt_flat else {"passed": True},
        "contract_validity": cb.contract_validity(alt_flat, calendar, listed, exists) if alt_flat else
        {"passed": True},
    }
    entry_dups = {m: cb.duplicate_check(_ev(v)) for m, v in entries.items()}
    engine = exit_engine_check(exits[BASE_EXIT], exits["2R"])
    data_ok = (bool(integrity.get("ok")) and dup["passed"] and ts_chk["passed"] and cval["passed"]
               and all(x["passed"] for x in alt_checks["duplicates"].values()) and alt_checks["timestamps"]["passed"]
               and alt_checks["contract_validity"]["passed"] and all(x["passed"] for x in entry_dups.values())
               and engine["passed"])
    clean = lookahead_ok and data_ok

    # OOS (once) - frozen alternatives
    frozen: dict[str, dict | None] = {k: None for k in SCOPES}
    if sel:
        for d in DIRECTIONS:
            c = sel["selected"][d]
            if c is not None:
                frozen[d] = s6.frozen_block(f"{c.cid} {c.name}", cand_rows(c, alt_rows), True)
        picked = [sel["selected"][d] for d in DIRECTIONS if sel["selected"][d]]
        if picked:
            comb = sorted((r for c in picked for r in cand_rows(c, alt_rows)),
                          key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
            no_trade = [d for d in DIRECTIONS if sel["selected"][d] is None]
            name = " + ".join(c.cid for c in picked) + "".join(f" ({d} = NO TRADE)" for d in no_trade)
            frozen["COMBINED"] = s6.frozen_block(name, comb, True, combined=True)

    # FINAL HOLDOUT - once, after the fingerprint check
    if sel and fingerprint(sel, entry_best, exit_best) != fp:
        raise RuntimeError("selection changed before the final holdout")
    for blk in frozen.values():
        if blk is not None:
            blk["holdout"] = split_m(blk["rows"], "FINAL_HOLDOUT")
            blk["holdout_status"] = s6.holdout_status(blk["holdout"])
            blk["status"] = s6.final_status(blk["validated"], blk["per_split"]["OOS"], blk["validation_to_oos"],
                                            blk["robustness"], blk["holdout_status"], clean)
            blk["duplicates"] = cb.duplicate_check(blk["rows"])
    data_ok = data_ok and all(b["duplicates"]["passed"] for b in frozen.values() if b is not None)
    clean = lookahead_ok and data_ok

    statuses = [b["status"] for b in frozen.values() if b is not None]
    overfitting_ok = iso["passed"] and OVERFIT not in statuses
    s2 = {k: cb.scoped(cb.split_rows(rows, k)) for k in SPLITS}
    decision = decide(und, trans, exit_choice, s2, frozen, clean)
    best_key = next((k for k in ("COMBINED", "SELL", "BUY") if frozen.get(k) is not None), None)
    target = "NOT ACHIEVED"
    if best_key and all(m["wins"] + m["losses"] >= MIN_OOS_TRADES and (m["win_rate"] or 0) >= TARGET
                        for m in (frozen[best_key]["per_split"]["OOS"], frozen[best_key]["holdout"])):
        target = "ACHIEVED"
    safe = decision["alternative"] == D_ALT and lookahead_ok and data_ok and overfitting_ok

    return {
        "heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "accuracy_definition": "wins / (wins + losses); ambiguous (-1R), scratch and unavailable are never wins",
        "dataset": dataset, "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        "sessions": len(sessions), "signals": len(rows), "s2": s2,
        "underlying": und, "unconditional": uncond, "translation": trans,
        "entry_tables": entry_tables, "entry_choice": entry_choice,
        "entry_reason": choice_text(entry_choice, BASE_ENTRY, "entry"),
        "exit_tables": exit_tables, "exit_choice": exit_choice, "exit_timing": timing, "excursions": excursions,
        "exit_reason": choice_text(exit_choice, BASE_EXIT, "exit"),
        "cause": cause_text(und, entry_choice, exit_choice),
        "alternatives_run": run_alt, "alternatives_forced": force_alternatives and not lacks_edge,
        "alternative_signals": {sid: len(v) for sid, v in sigs.items()},
        "selection": sel, "fingerprint": fp, "isolation": iso, "frozen": frozen, "best_key": best_key,
        "target_70": target, "lookahead": lookahead, "lookahead_ok": lookahead_ok,
        "data_integrity": integrity,
        "data_checks": {"duplicates": dup, "timestamps": ts_chk, "contract_validity": cval, "alternatives": alt_checks,
                        "entry_duplicates": entry_dups, "exit_engine_reproduces_s2": engine},
        "data_ok": data_ok, "overfitting_ok": overfitting_ok, "decision": decision,
        "forward_paper": "SAFE" if safe else "NOT SAFE",
        "disclosure": ("August and September S2 results were already seen in earlier labs (S2 diagnostic, S5, clean "
                       "backtest, S6, S7). S8 families, variants, entries and exits are the pre-declared structures of "
                       "the S8 specification, ranked on TRAIN and validated on July only, but the researcher was not "
                       "blind to those months; OOS and holdout are therefore optimistic and fresh forward PAPER data "
                       "is required."),
        "production_unchanged": production_unchanged, "live_orders": 0, "trading_mode": "PAPER",
        "live_trading_enabled": False, "rows": rows, "alt_rows": alt_rows,
    }


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "s8_report.md", "underlying_signal_audit.csv", "option_translation_audit.csv",
                "entry_audit.csv", "exit_audit.csv", "alternative_candidates.csv", "train_results.csv",
                "validation_results.csv", "oos_results.csv", "holdout_results.csv", "robustness.csv",
                "data_quality.json")
_m, _csv, line = s6._m, s6._csv, s6.line


def _jsonable(v):
    if is_dataclass(v) and not isinstance(v, type):
        return asdict(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return s6._jsonable(v)


def underlying_csv(a: dict) -> list[dict]:
    out = []
    for d in DIRECTIONS:
        for sample, v in (("ALL_SIGNALS", a["underlying"][d]), ("FIRST_SIGNAL_PER_DAY",
                                                                a["underlying"][d]["first_per_day"])):
            for k in SPLITS:
                out.append({"direction": d, "sample": sample, "split": SPLIT_NAMES[k], **v["per_split"][k],
                            f"unconditional_acc{EDGE_H}": (a["unconditional"].get(k) or {}).get(d),
                            "verdict_uses": "TRAIN+VALIDATION" if k in PRE_OOS else "confirmation only",
                            "verdict": v["verdict"]})
            out.append({"direction": d, "sample": sample, "split": "TRAIN+VALIDATION", **v["train_validation"],
                        "verdict": v["verdict"], **v["checks"]})
    return out


def translation_csv(a: dict) -> list[dict]:
    out = []
    for d in DIRECTIONS:
        t = a["translation"][d]
        for k in SPLITS:
            out.append({"direction": d, "split": SPLIT_NAMES[k], **t["per_split"][k], "label": t["label"]})
        out.append({"direction": d, "split": "TRAIN+VALIDATION", **t["train_validation"], "label": t["label"],
                    "underlying_edge": t["underlying_edge"], "option_translation_edge": t["option_translation_edge"],
                    "option_translation_loss": t["option_translation_loss"], "accuracy_gap": t["accuracy_gap"]})
    return out


def entry_csv(a: dict) -> list[dict]:
    return [{"entry_method": n, "rule": ENTRY_METHODS[n], "scope": s, "split": SPLIT_NAMES[k],
             **_m(a["entry_tables"][n][s][k]), "selected_on": "TRAIN+VALIDATION",
             "chosen_for_scope": a["entry_choice"][s]["best"] == n}
            for n in ENTRY_METHODS for s in SCOPES for k in SPLITS]


def exit_csv(a: dict) -> list[dict]:
    out = []
    for n, tr in EXITS.items():
        for s in SCOPES:
            for k in SPLITS:
                row = {"exit": n, "target_r": tr, "stop": f"{STOP_ATR:g} x premium ATR (unchanged)", "scope": s,
                       "split": SPLIT_NAMES[k], **_m(a["exit_tables"][n][s][k]), **a["exit_timing"][n][s][k],
                       "alias_of": EXIT_ALIASES.get(n), "chosen_for_scope": a["exit_choice"][s]["best"] == n}
                if s in DIRECTIONS:
                    row.update({f"s2_trades_{x}": v for x, v in a["excursions"][s][k].items()})
                out.append(row)
    return out


def _status_of(a: dict, cid: str) -> str | None:
    for k, blk in a["frozen"].items():
        if blk is not None and k != "COMBINED" and blk["name"].split(" ")[0] == cid:
            return blk["status"]
    return None


def alternatives_csv(a: dict) -> list[dict]:
    sel = a["selection"]
    if not sel:
        return [{"note": "alternatives not researched: S2 underlying edge present on both sides"}]
    chosen = {c.cid for c in sel["selected"].values() if c}
    out = []
    for cid, c in sel["candidates"].items():
        tr, va = sel["train"].get(cid), sel["validation"].get(cid)
        base = {"candidate": cid, "family": c.spec.family, "variant": c.spec.variant, "name": c.name,
                "direction": c.direction, "premium_confirmation": c.confirm, "rule": c.describe(),
                "rationale": c.spec.rationale, "signals": a["alternative_signals"].get(c.sid),
                "train_eligible": tr["eligible"], "train_failed": ",".join(tr["failed"]),
                "validation_verdict": va["verdict"] if va else "NOT_EVALUATED (not TRAIN-eligible)",
                "selected": cid in chosen, "final_status": _status_of(a, cid)}
        out.append({**base, "split": "TRAIN", **_m(tr["metrics"]), "quality_score": tr["score"][0]})
        if va:
            out.append({**base, "split": "VALIDATION", **_m(va["metrics"])})
        if cid in chosen:
            blk = a["frozen"][c.direction]
            out.append({**base, "split": "OOS", **_m(blk["per_split"]["OOS"])})
            out.append({**base, "split": HOLDOUT_LABEL, **_m(blk["holdout"])})
    return out


def train_csv(a: dict) -> list[dict]:
    out = [{"system": "CURRENT_S2", "scope": s, **_m(a["s2"]["TRAIN"][s])} for s in SCOPES]
    sel = a["selection"]
    if sel:
        for cid, tr in sel["train"].items():
            c = sel["candidates"][cid]
            out.append({"system": cid, "name": c.name, "scope": c.direction, **_m(tr["metrics"]),
                        **{f"month_{k}_pnl": v for k, v in tr["months"].items()}, **tr["checks"],
                        "train_eligible": tr["eligible"], "quality_score": tr["score"][0],
                        "train_rank": sel["ranking"][c.direction].index(cid) + 1
                        if cid in sel["ranking"][c.direction] else None})
    return out


def validation_csv(a: dict) -> list[dict]:
    out = [{"system": "CURRENT_S2", "scope": s, **_m(a["s2"]["VALIDATION"][s])} for s in SCOPES]
    sel = a["selection"]
    if sel:
        out += [{"system": cid, "name": sel["candidates"][cid].name, "scope": sel["candidates"][cid].direction,
                 "verdict": va["verdict"], "failed": ",".join(va["failed"]), **_m(va["metrics"])}
                for cid, va in sel["validation"].items()]
    return out


def _split_csv(a: dict, split: str) -> list[dict]:
    out = []
    for s in SCOPES:
        out.append({"scope": s, "system": "CURRENT_S2", **_m(a["s2"][split][s])})
        blk = a["frozen"].get(s)
        if blk is None:
            out.append({"scope": s, "system": "BEST_ALTERNATIVE", "name": "none (NO TRADE: nothing survived TRAIN + "
                                                                          "VALIDATION)"})
            continue
        m = blk["holdout"] if split == "FINAL_HOLDOUT" else blk["per_split"][split]
        row = {"scope": s, "system": "BEST_ALTERNATIVE", "name": blk["name"], **_m(m), "status": blk["status"]}
        if split == "FINAL_HOLDOUT":
            row.update(label=HOLDOUT_LABEL, holdout_status=blk["holdout_status"])
        out.append(row)
    return out


def robustness_csv(a: dict) -> list[dict]:
    out = []
    for k, blk in a["frozen"].items():
        if blk is None:
            continue
        rb = blk["robustness"]
        for comp in ("monthly", "regime", "time_of_day"):
            out.append({"scope": k, "candidate": blk["name"], "component": f"concentration_{comp}",
                        **{x: rb[comp][x] for x in ("groups", "profitable", "best_group", "best_share_of_profit",
                                                    "total_pnl", "pnl_without_best", "concentrated")}})
        out.append({"scope": k, "candidate": blk["name"], "component": "few_trades", **rb["few_trades"]})
        out.append({"scope": k, "candidate": blk["name"], "component": "summary",
                    "train_to_validation": blk["train_to_validation"], "validation_to_oos": blk["validation_to_oos"],
                    "overfit_flags": ",".join(rb["overfit_flags"]), "max_losing_streak": rb["max_losing_streak"],
                    "max_drawdown": rb["max_drawdown"], "final_status": blk["status"]})
    if not out:
        out.append({"note": "no frozen alternative - nothing to stress"})
    return out


def _summary_json(a: dict) -> dict:
    frozen = {k: None if b is None else {x: v for x, v in b.items() if x != "rows"} for k, b in a["frozen"].items()}
    skip = ("rows", "alt_rows", "frozen", "data_integrity")
    return {**{k: v for k, v in a.items() if k not in skip}, "frozen": frozen}


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _num(v, fmt: str = "{:+.3f}") -> str:
    return "N/A" if v is None else fmt.format(v)


def underlying_line(u: dict) -> str:
    p, tv = u["per_split"], u["train_validation"]
    acc = f"direction_acc{EDGE_H}"
    return (f"{u['verdict']} ({EDGE_H}-candle direction accuracy TRAIN {_pct(p['TRAIN'][acc])} / VALIDATION "
            f"{_pct(p['VALIDATION'][acc])}, mean {_num(tv[f'mean_ret{EDGE_H}_atr'])} ATR, MFE/MAE "
            f"{_num(tv['mfe_mae_ratio'], '{:.2f}')}, day-clustered t {_num(tv[f'day_t{EDGE_H}'], '{:.2f}')}; "
            f"OOS {_pct(p['OOS'][acc])} / HOLDOUT {_pct(p['FINAL_HOLDOUT'][acc])} confirmation only)")


def translation_line(t: dict) -> str:
    tv = t["train_validation"]
    return (f"{t['label']} (TRAIN+VALIDATION: NIFTY {_pct(tv[f'underlying_acc{EDGE_H}'])} -> option "
            f"{_pct(tv[f'option_acc{EDGE_H}'])} at {EDGE_H} candles, response {_pct(tv['option_response_rate'])}, "
            f"lag {_num(tv['avg_option_lag_candles'], '{:+.2f}')} candles, premium MFE "
            f"{_num(tv['option_mfe_r'], '{:.2f}')}R / MAE {_num(tv['option_mae_r'], '{:.2f}')}R)")


def markdown(a: dict) -> str:
    md = ["# S8 strategy architecture audit & clean alternative research", "", f"**{BANNER}**", "",
          f"Generated {a['generated_at']}. Dataset {a['dataset'].get('from')} -> {a['dataset'].get('to')} "
          "(data/historical/upstox_6m, NSE_INDEX|Nifty 50, real option premium candles). TRADING_MODE=PAPER, "
          "LIVE_TRADING_ENABLED=false, live orders: 0. Production changed: "
          f"**{'NO' if a['production_unchanged'] is not False else 'YES'}**.", "",
          "All numbers are **historical backtest accuracy**, not live accuracy; no profit is guaranteed. Accuracy = "
          "wins / (wins + losses); ambiguous candles are booked at -1R.", "", f"> {a['disclosure']}", "",
          "## Final decision", "", *[f"- **{x}**" for x in a["decision"]["outcomes"]],
          f"- Forward PAPER validation: **{a['forward_paper']}**",
          f"- Lookahead **{'PASS' if a['lookahead_ok'] else 'FAIL'}** | data integrity "
          f"**{'PASS' if a['data_ok'] else 'FAIL'}** | overfitting **{'PASS' if a['overfitting_ok'] else 'FAIL'}** | "
          f"70% target **{a['target_70']}**", f"- Main cause of poor S2 results: {a['cause']}", ""]
    md += ["## Current S2 (option results)", "", "| Split | Scope | Result |", "|---|---|---|"]
    md += [f"| {SPLIT_NAMES[k]} | {s} | {line(a['s2'][k][s])} |" for k in SPLITS for s in SCOPES]
    md += ["", "## Part A - underlying signal audit (before any option)", "",
           f"Verdict rule (TRAIN and VALIDATION only): {EDGE_H}-candle direction accuracy >= {MIN_DIRECTION_ACC:.0%}, "
           f"mean signed return > 0 and >= {MIN_EDGE_VS_UNCONDITIONAL:.0%} above the unconditional drift of all "
           f"candles in the same window, in BOTH splits, plus day-clustered t >= {MIN_DAY_T:g} on TRAIN+VALIDATION.",
           ""]
    md += [f"- {d}: {underlying_line(a['underlying'][d])}; first signal per day: "
           f"{a['underlying'][d]['first_per_day']['verdict']}" for d in DIRECTIONS]
    md += ["", "Per-split tables (1/2/3/5-candle returns, MFE, MAE): underlying_signal_audit.csv.", "",
           "## Part B - option translation audit", ""]
    md += [f"- {d}: {translation_line(a['translation'][d])}" for d in DIRECTIONS]
    md += ["", "## Part C - entry timing audit (same signals, same S2 stop and 2R target)", "",
           "| Entry | Scope | TRAIN | VALIDATION |", "|---|---|---|---|"]
    md += [f"| {n} | {s} | {line(a['entry_tables'][n][s]['TRAIN'])} | {line(a['entry_tables'][n][s]['VALIDATION'])} |"
           for n in ENTRY_METHODS for s in DIRECTIONS]
    md += ["", f"Choice (TRAIN+VALIDATION only): {a['entry_reason']}.", "",
           "## Part D - exit architecture audit (same entries)", "",
           "| Exit | Scope | TRAIN | VALIDATION | Target hit | Stop hit |", "|---|---|---|---|---|---|"]
    for n in EXITS:
        for s in DIRECTIONS:
            t = a["exit_timing"][n][s]["TRAIN"]
            md.append(f"| {n} | {s} | {line(a['exit_tables'][n][s]['TRAIN'])} | "
                      f"{line(a['exit_tables'][n][s]['VALIDATION'])} | {_pct(t['target_hit_rate'])} | "
                      f"{_pct(t['stop_hit_rate'])} |")
    for d in DIRECTIONS:
        e = a["excursions"][d]["TRAIN"]
        md.append(f"\n{d} TRAIN premium excursion to square-off: avg MFE {_num(e['avg_mfe_r_eod'], '{:.2f}')}R, avg "
                  f"MAE {_num(e['avg_mae_r_eod'], '{:.2f}')}R, MFE reaches 1R {_pct(e['mfe_reaches_1R'])} / 2R "
                  f"{_pct(e['mfe_reaches_2R'])} / 2.5R {_pct(e['mfe_reaches_2.5R'])}.")
    md += ["", f"Choice (TRAIN+VALIDATION only): {a['exit_reason']}. 2R re-evaluation reproduces current S2: "
           f"{'PASS' if a['data_checks']['exit_engine_reproduces_s2']['passed'] else 'FAIL'}.", "",
           "## Part E/G/H - alternative strategy families", ""]
    sel = a["selection"]
    if not sel:
        md.append("Not researched: the underlying audit found an edge on both sides.")
    else:
        md += [f"{len(SPECS)} predefined variants ({len(FAMILIES)} families x <= {MAX_VARIANTS}), BUY and SELL "
               "separately, executed through the exact S2 execution path (next option candle open, 1.5 x premium "
               "ATR stop, 2R target, EOD square-off). TRAIN eligibility: >= "
               f"{MIN_TRAIN_TRADES} trades, expectancy > 0, PF >= {MIN_TRAIN_PF}, drawdown <= {MAX_DD_SHARE:.0%} of "
               f"gross profit, losing streak <= {MAX_TRAIN_STREAK}, profitable in >= {MIN_PROFITABLE_MONTHS} of 3 "
               "TRAIN months, profit not from one regime or the top 5 trades. Ranked by expectancy x sqrt(trades).",
               "", "| Candidate | Signals | TRAIN | Eligible | July |", "|---|---|---|---|---|"]
        for cid, c in sel["candidates"].items():
            tr, va = sel["train"][cid], sel["validation"].get(cid)
            md.append(f"| {cid} {c.name} | {a['alternative_signals'].get(c.sid)} | {line(tr['metrics'])} | "
                      f"{tr['eligible']} ({','.join(tr['failed']) or 'all checks'}) | {va['verdict'] if va else '-'} |")
        md += ["", f"Frozen after July: {ids(sel)} (fingerprint `{a['fingerprint']}`)."]
    md += ["", "## Part J - current S2 vs best alternative", "",
           "| Split | Scope | Current S2 | Best alternative | Status |", "|---|---|---|---|---|"]
    for k in SPLITS:
        for s in SCOPES:
            blk = a["frozen"].get(s)
            alt = "none" if blk is None else line(blk["holdout"] if k == "FINAL_HOLDOUT" else blk["per_split"][k])
            md.append(f"| {SPLIT_NAMES[k]} | {s} | {line(a['s2'][k][s])} | {alt} | "
                      f"{'-' if blk is None else blk['status']} |")
    md += ["", f"September rows: **{HOLDOUT_LABEL}**.", "", "## Part K - anti-lookahead and isolation", ""]
    md += [f"- {k}: {'PASS' if v.get('passed') else 'FAIL'}" + (" (not applicable)" if v.get("not_applicable") else "")
           for k, v in a["lookahead"].items()]
    md += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in a["isolation"]["checks"].items()]
    md += ["", "## Safety", "", "No candidate is promoted or activated automatically. Production S2, broker, "
           "OrderManager, risk and safety settings are unchanged."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    p["summary.json"].write_text(json.dumps(_jsonable(_summary_json(a)), indent=2, default=str), encoding="utf-8")
    p["s8_report.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["underlying_signal_audit.csv"], underlying_csv(a))
    _csv(p["option_translation_audit.csv"], translation_csv(a))
    _csv(p["entry_audit.csv"], entry_csv(a))
    _csv(p["exit_audit.csv"], exit_csv(a))
    _csv(p["alternative_candidates.csv"], alternatives_csv(a))
    _csv(p["train_results.csv"], train_csv(a))
    _csv(p["validation_results.csv"], validation_csv(a))
    _csv(p["oos_results.csv"], _split_csv(a, "OOS"))
    _csv(p["holdout_results.csv"], _split_csv(a, "FINAL_HOLDOUT"))
    _csv(p["robustness.csv"], robustness_csv(a))
    dq = {"data_integrity": a["data_integrity"], "checks": a["data_checks"], "data_ok": a["data_ok"],
          "unavailable_reasons": {k: a["s2"][k]["COMBINED"]["unavailable_reasons"] for k in SPLITS},
          "lookahead": a["lookahead"], "missing_option_data": "DATA_UNAVAILABLE / NO SIGNAL, never interpolated"}
    p["data_quality.json"].write_text(json.dumps(_jsonable(dq), indent=2, default=str), encoding="utf-8")
    return p


def _block(label: str, blk: dict | None) -> list[str]:
    if blk is None:
        return [f"{label}:", "name: none (NO TRADE - nothing survived TRAIN + VALIDATION)", f"status: {REJECTED}", ""]
    return [f"{label}:", f"name: {blk['name']}", f"status: {blk['status']} (holdout {blk['holdout_status']})", ""]


def terminal_text(a: dict) -> str:
    out = [HEADING, "(historical backtest accuracy - not live accuracy, no guaranteed profit)", "",
           "CURRENT S2 UNDERLYING EDGE:"]
    out += [f"{d}: {underlying_line(a['underlying'][d])}" for d in DIRECTIONS]
    out += ["", "OPTION TRANSLATION:"] + [f"{d}: {translation_line(a['translation'][d])}" for d in DIRECTIONS]
    out += ["", "ENTRY AUDIT:", "best: " + " | ".join(f"{s} {a['entry_choice'][s]['best']}" for s in DIRECTIONS),
            f"reason: {a['entry_reason']}", "", "EXIT AUDIT:",
            "best: " + " | ".join(f"{s} {a['exit_choice'][s]['best']}" for s in DIRECTIONS),
            f"reason: {a['exit_reason']}", f"main cause: {a['cause']}", "", "BEST ALTERNATIVE:"]
    k = a["best_key"]
    if k is None:
        out += ["name: none (" + ("no alternative survived TRAIN + VALIDATION" if a["alternatives_run"] else
                                  "alternatives not researched - S2 underlying edge present") + ")",
                "TRAIN: N/A", "VALIDATION: N/A", "OOS: N/A", "HOLDOUT: N/A"]
    else:
        blk = a["frozen"][k]
        p = blk["per_split"]
        out += [f"name: {blk['name']} ({k})", f"TRAIN: {line(p['TRAIN'])}", f"VALIDATION: {line(p['VALIDATION'])}",
                f"OOS: {line(p['OOS'])}", f"HOLDOUT: {line(blk['holdout'])} ({HOLDOUT_LABEL})"]
    out += ["", "CURRENT S2 (same splits):"]
    out += [f"{SPLIT_NAMES[x]} {s}: {line(a['s2'][x][s])}" for x in SPLITS for s in SCOPES]
    out += [""] + _block("BEST BUY", a["frozen"].get("BUY")) + _block("BEST SELL", a["frozen"].get("SELL")) + \
        _block("COMBINED", a["frozen"].get("COMBINED"))
    out += ["70% TARGET:", a["target_70"], "", "LOOKAHEAD:", "PASS" if a["lookahead_ok"] else "FAIL", "",
            "DATA INTEGRITY:", "PASS" if a["data_ok"] else "FAIL", "", "PRODUCTION CHANGED:",
            "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:", "0", "",
            "FORWARD PAPER VALIDATION:", a["forward_paper"], "", "FINAL DECISION:",
            " | ".join(a["decision"]["outcomes"])]
    return "\n".join(out)
