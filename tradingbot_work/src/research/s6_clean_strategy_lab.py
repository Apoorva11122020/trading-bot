"""S6 CLEAN STRATEGY IMPROVEMENT LAB (PAPER | RESEARCH ONLY | NO ORDERS).

Searches for a repeatable, causal edge on top of the unchanged S2 signal stream. Every candidate is pre-declared in
this file (economic rationale, coarse fixed thresholds) before any result is computed:
  * a FILTER on S2 signals of ONE direction (same contract, entry, stop, target and exit as S2), or
  * an ENTRY variant (B = next-bar confirmation, C = one-bar delay) or EXIT variant (1.5R / 2.5R target), evaluated
    through the exact S2 evaluation code (``s2_1dte_improvement.evaluate``) on real premium candles.
BUY and SELL candidates are separate families and are ranked independently.

Chronology (historical backtest accuracy, never live accuracy):
  TRAIN 2026-04-01..06-30      candidates ranked on TRAIN metrics only
  VALIDATION 07-01..07-31      VALIDATED / REJECTED; a candidate is never modified after validation
  OOS 08-01..08-31             frozen (fingerprinted) candidates only
  FINAL HOLDOUT 09-01..09-30   frozen candidates only, run once, NOT USED FOR SELECTION
``select`` raises if it receives any row after 2026-07-31.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import current_day_s2 as cds
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.options.strategy import round_to_tick
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s5_targeted_strategy as s5

HEADING = "S6 CLEAN STRATEGY LAB COMPLETE"
BANNER = cb.BANNER
IST = imp.IST
SPLITS = cb.SPLITS
SPLIT_NAMES = {"TRAIN": "TRAIN", "VALIDATION": "VALIDATION", "OOS": "OOS AUGUST", "FINAL_HOLDOUT": "HOLDOUT SEPTEMBER"}
SCOPES, DIRECTIONS = cb.SCOPES, ("BUY", "SELL")
PRE_OOS = ("TRAIN", "VALIDATION")
PRE_HOLDOUT = ("TRAIN", "VALIDATION", "OOS")
TRAIN_END, SELECTION_END = SPLITS["TRAIN"][1], SPLITS["VALIDATION"][1]
UNAVAILABLE = dg.UNAVAILABLE
STEP = pd.Timedelta(minutes=5)

STOP_ATR = imp.BASELINE.stop_atr
BASE_TARGET = imp.BASELINE.target_r
BASE_VARIANT = ("A", BASE_TARGET)
VARIANTS = (BASE_VARIANT, ("B", BASE_TARGET), ("C", BASE_TARGET), ("A", 1.5), ("A", 2.5))
ENTRY_TEXT = {"A": "immediate next-bar open (unchanged S2)",
              "B": "next-bar confirmation: the 5m candle after the signal must close beyond the breakout level; "
                   "entry at the following candle's open, premium ATR from candles <= the confirmation candle",
              "C": "one-bar delay: entry at the open two candles after the signal, premium ATR from candles <= the "
                   "candle before entry"}

# Pre-declared thresholds (fixed before any result was computed; never fitted).
MAX_CANDIDATES_PER_SIDE = 30
MIN_TRAIN_TRADES, MIN_VAL_TRADES, MIN_OOS_TRADES, MIN_HOLDOUT_TRADES = 30, 10, 30, 20
MIN_GROUP_TRADES = 5
PF_IMPROVEMENT = 0.10          # "improving" = TRAIN PF >= S2 TRAIN PF + 0.10
MAX_DD_SHARE = 0.5             # TRAIN max drawdown <= 50% of TRAIN gross profit
MAX_BEST_TRADE_SHARE = 0.25    # best single trade <= 25% of TRAIN gross profit
MAX_TRAIN_STREAK = 10
MIN_VAL_PF_RETENTION = 0.5     # VALIDATION PF >= 50% of TRAIN PF
PROMISING_OOS_PF = 1.1
TOP_TRADES = 5
TARGET = 0.70
PF_CAP = 10.0

PROMISING, VALIDATED, REJECTED, INSUFFICIENT, OVERFIT = ("PROMISING", "VALIDATED", "REJECTED", "INSUFFICIENT_DATA",
                                                         "OVERFIT_RISK")
STABLE, DEGRADED, FAILED = "STABLE", "DEGRADED", "FAILED"
PASSED_HO, FAILED_HO = "PASSED", "FAILED"
HOLDOUT_LABEL = "FINAL HOLDOUT - NOT USED FOR SELECTION"


# ------------------------------------------------------------------ helpers
def _pf(m: dict) -> float:
    v = m.get("profit_factor")
    return 0.0 if v is None else min(float(v), PF_CAP)


def split_m(rows: Sequence[dict], split: str) -> dict[str, Any]:
    return cb.metrics(cb.split_rows(rows, split))


def _ev(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if r.get("state") == sh.EVALUATED]


def fv(r: dict, key: str):
    s6 = r.get("s6") or {}
    return s6[key] if key in s6 else r["features"].get(key)


def expectancy_rs(m: dict) -> float | None:
    return round(m["net_pnl"] / m["total_trades"], 2) if m["total_trades"] else None


# ------------------------------------------------------------------ extra causal features (signal candle only)
RSI_EDGES = (30.0, 40.0, 50.0, 60.0, 70.0)
VOLUME_EDGES = (0.8, 1.2)              # selected option volume / mean of previous 10 completed candles
ORB_RANGE_EDGES = (1.0, 2.0, 3.0)      # opening-range size / ATR14
PREMIUM_ATR_EDGES = (5.0, 10.0, 20.0)  # premium ATR14 in rupees
SL_EST_EXCESSIVE = 20.0                # estimated stop distance (1.5 x premium ATR / premium close) in %
LATE_ENTRY = time(13, 0)
EXTRA_CHECK_KEYS = ("market_regime", "sign_changes", "orb_range_atr", "premium_candle", "premium_above_prev",
                    "volume_level", "rsi_bucket")


def orb_ranges(bars: pd.DataFrame, start: str, end: str) -> dict[date, tuple[float, float, pd.Timestamp]]:
    t = bars.index.time
    s, e = time.fromisoformat(start), time.fromisoformat(end)
    sub = bars.loc[(t >= s) & (t < e)]
    return {d: (float(g["high"].max()), float(g["low"].min()), g.index.max())
            for d, g in sub.groupby(sub.index.date)}


def market_regime(trend: str | None, sign_changes: float | None) -> str:
    """Project trend regime plus the project's CHOPPY rule (>= 6 close-to-close sign changes so far today)."""
    if trend in ("TREND_UP", "TREND_DOWN"):
        return trend
    if trend in (None, UNAVAILABLE) or sign_changes is None:
        return UNAVAILABLE
    return "CHOPPY" if sign_changes >= lab.CHOPPY_SIGN_CHANGES else "RANGE"


def extra_features(r: dict, bars: pd.DataFrame, orb: dict, ctx: dict | None) -> dict[str, Any]:
    ts = dra._ts(r["signal_time"])
    f = r["features"]
    b = dg.bar_values(bars, ts)
    sc, a = b.get("sign_changes"), b.get("atr14")
    o = orb.get(ts.date())
    orb_atr = (o[0] - o[1]) / a if o and a and o[2] < ts else None
    candle = above_prev = None
    prem = (ctx or {}).get("premium")
    if prem is not None and not isinstance(prem, str) and not prem.empty:
        hist = prem[prem["timestamp"] <= ts]
        if len(hist) and hist["timestamp"].iloc[-1] == ts:
            c, op = float(hist["close"].iloc[-1]), float(hist["open"].iloc[-1])
            candle = "BULLISH" if c > op else "BEARISH" if c < op else "DOJI"
            if len(hist) >= 2:
                above_prev = c > float(hist["close"].iloc[-2])
    vr = dg._num(f.get("option_vol_ratio"))
    patr, patr_pct = dg._num(r.get("premium_atr")), dg._num(f.get("premium_atr_pct"))
    return {
        "market_regime": market_regime(f.get("trend_regime"), sc), "sign_changes": sc,
        "orb_range_atr": orb_atr, "orb_range_bucket": dg.bucket(orb_atr, ORB_RANGE_EDGES),
        "rsi_bucket": dg.bucket(f.get("rsi14"), RSI_EDGES),
        "volume_level": UNAVAILABLE if vr is None else "LOW" if vr < VOLUME_EDGES[0] else
        "HIGH" if vr >= VOLUME_EDGES[1] else "NORMAL",
        "premium_candle": candle or UNAVAILABLE, "premium_above_prev": above_prev,
        "premium_atr": patr, "premium_atr_bucket": dg.bucket(patr, PREMIUM_ATR_EDGES),
        "sl_est_pct": None if patr_pct is None else STOP_ATR * patr_pct,
    }


def prepare(rows: Sequence[dict], bars: pd.DataFrame, ctx_by_key: dict, range_start: str,
            range_end: str) -> list[dict]:
    """Annotated S2 rows (dg.annotate + s5.prepare_rows) plus the S6 causal extras."""
    orb = orb_ranges(bars, range_start, range_end)
    return [{**r, "s6": extra_features(r, bars, orb, ctx_by_key.get(dra.key_of(r)))} for r in rows]


def post_entry(rows: Sequence[dict], ctx_by_key: dict) -> dict[tuple, dict]:
    """POST-ENTRY diagnostics (immediate adverse premium move, max favourable excursion). Never used as a filter."""
    out = {}
    for r in _ev(rows):
        ctx = ctx_by_key.get(dra.key_of(r))
        prem = (ctx or {}).get("premium")
        if prem is None or isinstance(prem, str) or not r.get("exit_time") or not r.get("stop_loss"):
            continue
        t = dra._ist_series(prem["timestamp"])
        held = prem[((t >= dra._ts(r["entry_time"])) & (t <= dra._ts(r["exit_time"]))).to_numpy()]
        if held.empty:
            continue
        risk = float(r["entry"]) - float(r["stop_loss"])
        out[dra.key_of(r)] = {"immediate_adverse": float(held["close"].iloc[0]) < float(r["entry"]),
                              "mfe_r": (float(held["high"].max()) - float(r["entry"])) / risk if risk > 0 else None}
    return out


# ------------------------------------------------------------------ conditions (causal, signal candle only)
def _eq(key: str, value: str) -> Callable[[dict], bool | None]:
    def fn(r: dict) -> bool | None:
        v = fv(r, key)
        return None if v in (None, UNAVAILABLE) else v == value
    return fn


def _between(key: str, lo: float | None = None, hi: float | None = None) -> Callable[[dict], bool | None]:
    def fn(r: dict) -> bool | None:
        v = dg._num(fv(r, key))
        if v is None:
            return None
        return (lo is None or v >= lo) and (hi is None or v < hi)
    return fn


def _at_most(key: str, hi: float) -> Callable[[dict], bool | None]:
    def fn(r: dict) -> bool | None:
        v = dg._num(fv(r, key))
        return None if v is None else v <= hi
    return fn


CONDITIONS: dict[str, tuple[str, Callable[[dict], bool | None]]] = {
    "EMA_UP": ("EMA20 > EMA50 at the signal candle", _eq("ema20_vs_ema50", "EMA20>EMA50")),
    "EMA_DOWN": ("EMA20 <= EMA50 at the signal candle", _eq("ema20_vs_ema50", "EMA20<=EMA50")),
    "VWAP_ABOVE": ("close > session VWAP (TWAP when the index has no volume)", _eq("vwap_side", "ABOVE")),
    "VWAP_BELOW": ("close <= session VWAP (TWAP when the index has no volume)", _eq("vwap_side", "BELOW_OR_AT")),
    "BREAK_STRONG": ("breakout distance beyond the opening-range level >= 0.25 ATR14", _between("break_atr", 0.25)),
    "OPT_VOLUME_HIGH": ("bought option volume / mean of its previous 10 completed candles >= 1.2 (index volume is "
                        "zero)", _between("option_vol_ratio", 1.2)),
    "RSI_BULL": ("50 <= RSI14 < 70 (momentum, not overbought)", _between("rsi14", 50.0, 70.0)),
    "RSI_BEAR": ("30 <= RSI14 < 50 (momentum, not oversold)", _between("rsi14", 30.0, 50.0)),
    "ATR_TREND_UP": ("(EMA20 - EMA50) / ATR14 >= 1", _between("ema_sep_atr", 1.0)),
    "ATR_TREND_DOWN": ("(EMA20 - EMA50) / ATR14 < -1", _between("ema_sep_atr", None, -1.0)),
    "OPT_CONFIRM": ("bought CE/PE premium rising on completed candles <= signal (close > open and close > close 3 "
                    "candles earlier); option data missing -> NO_SIGNAL", lambda r: r.get("opt_confirm")),
}


@dataclass(frozen=True)
class Candidate:
    cid: str
    name: str
    direction: str
    family: str
    rationale: str
    conditions: tuple[str, ...] = ()
    entry: str = "A"
    target_r: float = BASE_TARGET

    @property
    def variant(self) -> tuple[str, float]:
        return (self.entry, self.target_r)

    @property
    def complexity(self) -> int:
        return len(self.conditions) + (self.entry != "A") + (self.target_r != BASE_TARGET)

    def describe(self) -> str:
        rules = " AND ".join(CONDITIONS[c][0] for c in self.conditions) or "all S2 signals of this direction"
        return (f"{rules} | entry {self.entry}: {ENTRY_TEXT[self.entry]} | stop {STOP_ATR:g} x premium ATR, target "
                f"{self.target_r:g}R")


_SIDE = {"BUY": {"p": "S6-B", "trend": "EMA_UP", "vwap": "VWAP_ABOVE", "rsi": "RSI_BULL", "atr": "ATR_TREND_UP",
                 "word": "bullish"},
         "SELL": {"p": "S6-S", "trend": "EMA_DOWN", "vwap": "VWAP_BELOW", "rsi": "RSI_BEAR", "atr": "ATR_TREND_DOWN",
                  "word": "bearish"}}


def stage1_candidates() -> tuple[Candidate, ...]:
    out = []
    for d, s in _SIDE.items():
        spec = [
            ("0", "S2_ALL", "BASELINE", "unchanged S2 signals (reference)", ()),
            ("1", "TREND_ALIGNED", "TREND", f"trade only with the {s['word']} EMA trend", (s["trend"],)),
            ("2", "VWAP_CONFIRMED", "VWAP", "price already on the trade side of the session average price",
             (s["vwap"],)),
            ("3", "STRONG_BREAKOUT", "BREAKOUT", "decisive close beyond the opening range, not a marginal poke",
             ("BREAK_STRONG",)),
            ("4", "VOLUME_CONFIRMED", "VOLUME", "participation in the bought option confirms the move",
             ("OPT_VOLUME_HIGH",)),
            ("5", "RSI_CONFIRMED", "RSI", "momentum in the trade direction without exhaustion", (s["rsi"],)),
            ("6", "ATR_NORMALIZED_TREND", "TREND", "trend strength measured in ATR units", (s["atr"],)),
            ("7", "OPTION_CONFIRMED", "OPTION", "the bought option's own premium is already rising", ("OPT_CONFIRM",)),
            ("8", "TREND_VWAP", "COMBO", "EMA trend and VWAP side agree", (s["trend"], s["vwap"])),
            ("9", "TREND_BREAKOUT", "COMBO", "EMA trend plus a decisive breakout", (s["trend"], "BREAK_STRONG")),
            ("10", "TREND_OPTION", "COMBO", "EMA trend plus a rising bought premium", (s["trend"], "OPT_CONFIRM")),
        ]
        out += [Candidate(f"{s['p']}{n}", f"{d}_{name}", d, fam, why, conds) for n, name, fam, why, conds in spec]
        out += [Candidate(f"{s['p']}11", f"{d}_S2_ENTRY_CONFIRM", d, "ENTRY",
                          "avoid false breakouts: the next candle must hold beyond the breakout level", entry="B"),
                Candidate(f"{s['p']}12", f"{d}_S2_ENTRY_DELAY", d, "ENTRY",
                          "avoid the first-bar spike after a breakout", entry="C"),
                Candidate(f"{s['p']}13", f"{d}_S2_TARGET_1.5R", d, "EXIT",
                          "intraday option moves often fade before 2R", target_r=1.5),
                Candidate(f"{s['p']}14", f"{d}_S2_TARGET_2.5R", d, "EXIT",
                          "let trend days pay for more losers", target_r=2.5)]
    return tuple(out)


STAGE1 = stage1_candidates()


def stage2_variants(c: Candidate | None) -> list[Candidate]:
    """Entry/exit variants of the TRAIN+VALIDATION-picked filter candidate (pre-declared, at most 4)."""
    if c is None or not c.conditions or c.variant != BASE_VARIANT:
        return []
    return [replace(c, cid=f"{c.cid}-EB", name=f"{c.name}+ENTRY_CONFIRM", entry="B"),
            replace(c, cid=f"{c.cid}-EC", name=f"{c.name}+ENTRY_DELAY", entry="C"),
            replace(c, cid=f"{c.cid}-T1.5", name=f"{c.name}+TARGET_1.5R", target_r=1.5),
            replace(c, cid=f"{c.cid}-T2.5", name=f"{c.name}+TARGET_2.5R", target_r=2.5)]


# ------------------------------------------------------------------ entry / exit variants (exact S2 evaluation)
def confirmation(r: dict, bars: pd.DataFrame, at: pd.Timestamp) -> bool | None:
    """Underlying candle starting at ``at`` (signal + 5 min) closes beyond the breakout level; known at entry time."""
    c, lvl = dg.bar_values(bars, at).get("close"), dg._num(r.get("breakout_level"))
    if c is None or lvl is None:
        return None
    return c > lvl if r["direction"] == "BUY" else c < lvl


def variant_row(r: dict, ctx: dict | None, bars: pd.DataFrame, params, square_off: str, cutoff: time, entry: str,
                target_r: float) -> dict:
    if (entry, target_r) == BASE_VARIANT:
        return r
    row = dict(r, variant=f"{entry}/{target_r:g}R", taken=True)
    if r.get("decision") != imp.ACCEPTED or ctx is None or ctx.get("data_status") != "OK":
        return row
    c = ctx
    if entry in ("B", "C"):
        ref = ctx["signal_time"] + STEP
        if ref.time() >= cutoff:
            return dict(row, taken=False, skip_reason="ENTRY_AFTER_CUTOFF")
        if entry == "B":
            ok = confirmation(r, bars, ref)
            if ok is not True:
                return dict(row, taken=False, skip_reason="NO_CONFIRMATION" if ok is False else
                            "CONFIRMATION_UNAVAILABLE")
        sig = replace(ctx["signal"], timestamp=ref.to_pydatetime())
        lv = cds.option_levels(sig, dra._contract(r, ctx.get("tick")), ctx["premium"], params)
        if lv.get("premium_atr") is None or lv.get("entry") is None or lv.get("entry_time") is None:
            raw = dict(row, decision=imp.ACCEPTED, outcome=imp.UNAVAILABLE, data_reason=lv.get("reason"), pnl=None,
                       r_multiple=None, exit_reason=None, exit_time=None, exit_price=None, stop_loss=None, target=None,
                       entry=None, entry_time=None, premium_atr=None)
            return sh.classify_row(raw)
        c = dict(ctx, entry_time=imp._ist(lv["entry_time"]), entry=float(lv["entry"]),
                 premium_atr=float(lv["premium_atr"]))
    res = imp.evaluate(c, STOP_ATR, target_r, square_off, sh.LOT_SIZE)
    raw = dict(row, decision=imp.ACCEPTED, entry_time=c["entry_time"].isoformat(), entry=c["entry"],
               premium_atr=c["premium_atr"], data_reason=None, **res)
    return sh.classify_row(raw)


def build_universe(rows: Sequence[dict], ctx_by_key: dict, bars: pd.DataFrame, params, square_off: str,
                   stop_new_entries: str) -> dict[tuple[str, float], list[dict]]:
    cutoff = time.fromisoformat(stop_new_entries)
    out: dict[tuple[str, float], list[dict]] = {BASE_VARIANT: list(rows)}
    for v in VARIANTS:
        if v != BASE_VARIANT:
            out[v] = [variant_row(r, ctx_by_key.get(dra.key_of(r)), bars, params, square_off, cutoff, *v)
                      for r in rows]
    return out


def candidate_rows(c: Candidate, universe: dict) -> list[dict]:
    """Rows of the candidate's direction, taken by its entry rule, with every condition True (None -> NO_SIGNAL)."""
    out = []
    for r in universe[c.variant]:
        if r["direction"] != c.direction or r.get("taken") is False:
            continue
        if all(CONDITIONS[k][1](r) is True for k in c.conditions):
            out.append(r)
    return out


def no_signal_count(c: Candidate, universe: dict) -> int:
    return sum(1 for r in universe[c.variant] if r["direction"] == c.direction
               and (r.get("taken") is False or any(CONDITIONS[k][1](r) is None for k in c.conditions)))


# ------------------------------------------------------------------ concentration / robustness
def concentration(rows: Sequence[dict], key: Callable[[dict], Any]) -> dict[str, Any]:
    ev = _ev(rows)
    groups: dict[str, list[float]] = defaultdict(list)
    for r in ev:
        groups[str(key(r))].append(float(r["pnl"]))
    act = {k: sum(v) for k, v in groups.items() if len(v) >= MIN_GROUP_TRADES and k != UNAVAILABLE}
    total = sum(float(r["pnl"]) for r in ev)
    pos = {k: p for k, p in act.items() if p > 0}
    best = max(sorted(pos), key=lambda k: pos[k]) if pos else None
    without = total - (pos[best] if best else 0.0)
    return {"groups": len(act), "profitable": len(pos), "best_group": best,
            "best_share_of_profit": round(pos[best] / sum(pos.values()), 4) if best else None,
            "total_pnl": round(total, 2), "pnl_without_best": round(without, 2),
            "concentrated": total <= 0 or without <= 0}


def top_trade_dependency(rows: Sequence[dict], n: int = TOP_TRADES) -> dict[str, Any]:
    pnl = sorted((float(r["pnl"]) for r in _ev(rows)), reverse=True)
    total = sum(pnl)
    top = sum(p for p in pnl[:n] if p > 0)
    return {"top_n": n, "pnl_top_trades": round(top, 2), "total_pnl": round(total, 2),
            "pnl_without_top_trades": round(total - top, 2), "concentrated": total <= 0 or total - top <= 0}


def robustness(rows: Sequence[dict], combined: bool = False) -> dict[str, Any]:
    """TRAIN + VALIDATION + OOS only (September is never part of a status input)."""
    pre = [r for r in rows if cb.split_of(r) in PRE_HOLDOUT]
    m = cb.metrics(pre)
    res: dict[str, Any] = {
        "monthly": concentration(pre, lambda r: str(r["session_date"])[:7]),
        "regime": concentration(pre, lambda r: fv(r, "market_regime")),
        "time_of_day": concentration(pre, lambda r: r["features"]["time_window"]),
        "few_trades": top_trade_dependency(pre),
        "max_losing_streak": m["max_losing_streak"], "max_drawdown": m["max_drawdown"],
        "trades_train_to_oos": m["total_trades"],
        "oos_trade_count_sufficient": split_m(rows, "OOS")["total_trades"] >= MIN_OOS_TRADES,
    }
    if combined:
        res["buy_sell"] = {d: cb.metrics([r for r in pre if r["direction"] == d])["net_pnl"] for d in DIRECTIONS}
        res["buy_sell_both_profitable"] = all(v > 0 for v in res["buy_sell"].values())
    res["overfit_flags"] = [k for k in ("monthly", "regime", "time_of_day", "few_trades") if res[k]["concentrated"]]
    res["overfit_risk"] = bool(res["overfit_flags"])
    return res


# ------------------------------------------------------------------ TRAIN selection / VALIDATION
def train_eval(c: Candidate, universe: dict, base_m: dict, is_base: bool) -> dict[str, Any]:
    rows = cb.split_rows(candidate_rows(c, universe), "TRAIN")
    m = cb.metrics(rows)
    best = max((float(r["pnl"]) for r in _ev(rows)), default=0.0)
    pf, gp = _pf(m), m["gross_profit"]
    checks = {
        "meaningful_trade_count": m["total_trades"] >= MIN_TRAIN_TRADES,
        "pf_positive_or_improving": pf > 1.0 or (not is_base and pf >= _pf(base_m) + PF_IMPROVEMENT),
        "reasonable_drawdown": gp > 0 and m["max_drawdown"] <= MAX_DD_SHARE * gp,
        "no_single_trade_dependency": gp > 0 and max(best, 0.0) / gp <= MAX_BEST_TRADE_SHARE,
        "no_extreme_losing_streak": m["max_losing_streak"] <= MAX_TRAIN_STREAK,
        "economic_rationale": bool(c.rationale),
    }
    ar = m["average_r"] if m["average_r"] is not None else -math.inf
    return {"cid": c.cid, "metrics": m, "checks": checks, "eligible": all(checks.values()),
            "failed": [k for k, ok in checks.items() if not ok], "score": (ar, pf, m["total_trades"])}


def rank_key(c: Candidate, tr: dict) -> tuple:
    s = tr[c.cid]["score"]
    return (-s[0], -s[1], -s[2], c.complexity, c.cid)


def validation_eval(c: Candidate, universe: dict, train_m: dict) -> dict[str, Any]:
    rows = candidate_rows(c, universe)
    v = split_m(rows, "VALIDATION")
    pre = [r for r in rows if cb.split_of(r) in PRE_OOS]
    tw = concentration(pre, lambda r: r["features"]["time_window"])
    rg = concentration(pre, lambda r: fv(r, "market_regime"))
    checks = {
        "sample_not_tiny": v["total_trades"] >= MIN_VAL_TRADES,
        "no_collapse": v["net_pnl"] > 0 and _pf(v) >= 1.0,
        "no_extreme_degradation": _pf(v) >= MIN_VAL_PF_RETENTION * _pf(train_m),
        "not_one_time_bucket": tw["profitable"] >= 2,
        "not_one_tiny_regime": rg["profitable"] >= 2,
    }
    return {"cid": c.cid, "metrics": v, "checks": checks, "verdict": VALIDATED if all(checks.values()) else REJECTED,
            "failed": [k for k, ok in checks.items() if not ok], "time_windows": tw, "regimes": rg}


def _assert_pre(universe: dict) -> None:
    late = sum(1 for rows in universe.values() for r in rows if cb.day_of(r) > SELECTION_END)
    if late:
        raise ValueError(f"selection received {late} rows after {SELECTION_END} (OOS/holdout must not be used)")


def pre_universe(universe: dict) -> dict:
    return {v: [r for r in rows if cb.day_of(r) <= SELECTION_END] for v, rows in universe.items()}


def select(universe: dict) -> dict[str, Any]:
    """Per direction: rank stage-1 candidates on TRAIN; the best TRAIN-eligible one that is VALIDATED is picked;
    its pre-declared entry/exit variants replace it only if they are VALIDATED and rank higher on TRAIN."""
    _assert_pre(universe)
    out: dict[str, Any] = {"train": {}, "validation": {}, "ranking": {}, "stage2": {}, "candidates": {},
                           "selected": {}, "counts": {}}
    for d in DIRECTIONS:
        pool = [c for c in STAGE1 if c.direction == d]
        base = pool[0]
        base_m = split_m(candidate_rows(base, universe), "TRAIN")
        tr = {c.cid: train_eval(c, universe, base_m, c is base) for c in pool}
        order = sorted((c for c in pool if tr[c.cid]["eligible"]), key=lambda c: rank_key(c, tr))
        va = {c.cid: validation_eval(c, universe, tr[c.cid]["metrics"]) for c in order}
        pick = next((c for c in order if va[c.cid]["verdict"] == VALIDATED), None)
        variants = stage2_variants(pick)
        for v in variants:
            tr[v.cid] = train_eval(v, universe, base_m, False)
            if tr[v.cid]["eligible"]:
                va[v.cid] = validation_eval(v, universe, tr[v.cid]["metrics"])
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


def fingerprint(sel: dict) -> str:
    spec = {"selected": {d: None if c is None else asdict(c) for d, c in sel["selected"].items()},
            "stage1": [asdict(c) for c in STAGE1], "ranking": sel["ranking"]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ anti-overfitting proofs
def anti_overfit(universe: dict, sel: dict) -> dict[str, Any]:
    flipped = {v: s5._flip(rows, SELECTION_END) for v, rows in universe.items()}
    sel_flip = select(pre_universe(flipped))
    refuses = False
    try:
        select(universe)
    except ValueError:
        refuses = True
    pre = pre_universe(universe)
    sel_rev = select({v: list(reversed(rows)) for v, rows in pre.items()})
    sel_val = select({v: s5._flip(rows, TRAIN_END) for v, rows in pre.items()})
    checks = {
        "oos_and_holdout_cannot_affect_selection": ids(sel) == ids(sel_flip),
        "selection_refuses_post_july_rows": refuses,
        "ranking_deterministic": fingerprint(sel) == fingerprint(sel_rev),
        "train_ranking_uses_train_only": sel["ranking"] == sel_val["ranking"],
        "candidate_count_within_limit": all(n <= MAX_CANDIDATES_PER_SIDE for n in sel["counts"].values()),
    }
    return {"checks": checks, "passed": all(checks.values()), "counts": sel["counts"],
            "flipped_selection": ids(sel_flip), "reversed_order_fingerprint": fingerprint(sel_rev)}


# ------------------------------------------------------------------ lookahead checks specific to S6
def _warp_after(prem: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    p = prem.copy()
    later = (dra._ist_series(p["timestamp"]) > cut).to_numpy()
    for col in ("open", "high", "low", "close", "volume"):
        if col in p:
            p[col] = p[col].astype(float)
            p.loc[later, col] = p.loc[later, col] * 3.0
    return p


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return (a is None) == (b is None) and (a is None or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9))
    return a == b


def extra_truncation_check(spot: pd.DataFrame, rows: Sequence[dict], ctx_by_key: dict, range_start: str,
                           range_end: str, sample: int = 20) -> dict[str, Any]:
    """S6 extras recomputed with every later NIFTY candle removed and every later premium candle tripled."""
    ann = [r for r in rows if r.get("instrument_key")]
    picked = ann[::max(1, len(ann) // max(sample, 1))][:sample]
    t_all = dra._ist_series(spot["timestamp"])
    bad = []
    for r in picked:
        ts = dra._ts(r["signal_time"])
        tb = dg.feature_bars(spot[(t_all <= ts).to_numpy()].reset_index(drop=True))
        ctx = ctx_by_key.get(dra.key_of(r))
        prem = (ctx or {}).get("premium")
        warped = dict(ctx, premium=_warp_after(prem, ts)) if prem is not None and not isinstance(prem, str) else ctx
        again = extra_features(r, tb, orb_ranges(tb, range_start, range_end), warped)
        bad += [f"{r['signal_time']}|{r['direction']}|{k}: {r['s6'][k]} vs {again[k]}" for k in EXTRA_CHECK_KEYS
                if not _same(r["s6"][k], again[k])]
    return {"checked": len(picked), "mismatches": bad[:20], "passed": bool(picked) and not bad}


def variant_causality_check(universe: dict, ctx_by_key: dict, bars: pd.DataFrame, params, square_off: str,
                            stop_new_entries: str, sample: int = 20) -> dict[str, Any]:
    """Entry B/C and 1.5R/2.5R: entry, premium ATR, stop and target are unchanged when every NIFTY candle after the
    decision candle is removed and every premium candle after entry is tripled; entry = real premium candle open."""
    cutoff = time.fromisoformat(stop_new_entries)
    bad, synthetic, checked = [], [], 0
    for v in VARIANTS:
        if v == BASE_VARIANT:
            continue
        ev = [r for r in _ev(universe[v]) if r.get("taken") is not False]
        for r in ev[::max(1, len(ev) // max(sample, 1))][:sample]:
            ctx = ctx_by_key.get(dra.key_of(r))
            if ctx is None:
                continue
            checked += 1
            et = dra._ts(r["entry_time"])
            prem = ctx["premium"]
            hit = prem[(dra._ist_series(prem["timestamp"]) == et).to_numpy()]
            if hit.empty or not math.isclose(round_to_tick(float(hit["open"].iat[0]), ctx.get("tick"), "nearest"),
                                             float(r["entry"]), abs_tol=1e-9):
                synthetic.append(f"{r['signal_time']}|{r['direction']}|{v}")
            tb = bars[bars.index <= et - STEP]
            again = variant_row(r, dict(ctx, premium=_warp_after(prem, et)), tb, params, square_off, cutoff, *v)
            for k in ("entry", "entry_time", "premium_atr", "stop_loss", "target"):
                if not _same(r.get(k), again.get(k)):
                    bad.append(f"{r['signal_time']}|{r['direction']}|{v}|{k}: {r.get(k)} vs {again.get(k)}")
    return {"checked": checked, "mismatches": bad[:20], "synthetic_or_wrong_entry": synthetic[:20],
            "passed": checked > 0 and not bad and not synthetic}


# ------------------------------------------------------------------ status helpers
def degradation(a: dict, b: dict) -> str:
    if b["total_trades"] == 0:
        return INSUFFICIENT
    ra, rb = a["average_r"], b["average_r"]
    if rb is None or rb <= 0 or _pf(b) < 1.0:
        return FAILED
    if ra is not None and ra > 0 and rb < 0.5 * ra:
        return DEGRADED
    return STABLE


def holdout_status(m: dict) -> str:
    if m["total_trades"] < MIN_HOLDOUT_TRADES:
        return INSUFFICIENT
    return PASSED_HO if m["net_pnl"] > 0 and _pf(m) >= 1.0 else FAILED_HO


def final_status(validated: bool, oos: dict, val_to_oos: str, robust: dict, holdout: str, lookahead_ok: bool) -> str:
    if not lookahead_ok or not validated:
        return REJECTED
    if oos["total_trades"] < MIN_OOS_TRADES:
        return INSUFFICIENT
    if val_to_oos == FAILED:
        return REJECTED
    if robust["overfit_risk"]:
        return OVERFIT
    if _pf(oos) >= PROMISING_OOS_PF and oos["net_pnl"] > 0 and holdout == PASSED_HO:
        return PROMISING
    return VALIDATED


def frozen_block(name: str, rows: Sequence[dict], validated: bool, combined: bool = False) -> dict[str, Any]:
    """TRAIN / VALIDATION / OOS of a frozen candidate (holdout is added later, once, after the fingerprint check)."""
    per = {k: split_m(rows, k) for k in PRE_HOLDOUT}
    return {"name": name, "rows": list(rows), "validated": validated, "per_split": per,
            "train_to_validation": degradation(per["TRAIN"], per["VALIDATION"]),
            "validation_to_oos": degradation(per["VALIDATION"], per["OOS"]),
            "robustness": robustness(rows, combined)}


# ------------------------------------------------------------------ Phase 1 / 2 / 3 / 6 / 7 diagnostics
DIMENSIONS: dict[str, Callable[[dict], Any]] = {
    "direction": lambda r: r["direction"],
    "market_regime": lambda r: fv(r, "market_regime"),
    "ema_relationship": lambda r: r["features"].get("ema20_vs_ema50"),
    "ema_sep_atr": lambda r: r["features"].get("ema_sep_atr_bucket"),
    "vwap_side": lambda r: r["features"].get("vwap_side"),
    "vwap_dist_atr": lambda r: r["features"].get("vwap_dist_atr_bucket"),
    "rsi": lambda r: fv(r, "rsi_bucket"),
    "volume_level": lambda r: fv(r, "volume_level"),
    "option_vol_ratio": lambda r: r["features"].get("option_vol_ratio_bucket"),
    "breakout_strength_body_frac": lambda r: r["features"].get("body_frac_bucket"),
    "breakout_distance_atr": lambda r: r["features"].get("break_atr_bucket"),
    "previous_range_atr": lambda r: fv(r, "orb_range_bucket"),
    "atr_regime": lambda r: r["features"].get("vol_regime"),
    "time_of_day": lambda r: r["features"].get("time_window"),
    "premium_volatility_pct": lambda r: r["features"].get("premium_atr_pct_bucket"),
    "premium_atr_rs": lambda r: fv(r, "premium_atr_bucket"),
    "premium_candle_direction": lambda r: fv(r, "premium_candle"),
    "premium_momentum_pct": lambda r: r["features"].get("premium_mom3_pct_bucket"),
}
REGIME_DIMS = ("market_regime", "atr_regime", "ema_relationship", "vwap_side")
METRIC_COLS = ("total_trades", "wins", "losses", "ambiguous", "win_rate", "profit_factor", "net_pnl", "max_drawdown",
               "max_losing_streak", "average_r")


def _cols(m: dict, prefix: str = "") -> dict[str, Any]:
    return {f"{prefix}{k}": m[k] for k in METRIC_COLS}


def phase1(rows: Sequence[dict]) -> list[dict]:
    by_split = {k: cb.split_rows(rows, k) for k in SPLITS}
    by_split["ALL"] = list(rows)
    out = []
    for dim, key in DIMENSIONS.items():
        for sp, srows in by_split.items():
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in srows:
                groups[str(key(r))].append(r)
            for b in sorted(groups):
                for s in DIRECTIONS:
                    sub = [r for r in groups[b] if r["direction"] == s]
                    if sub:
                        m = cb.metrics(sub)
                        out.append({"section": "PHASE1_BUCKET", "dimension": dim, "bucket": b, "scope": s,
                                    "split": sp, **_cols(m), "sample_ok": m["wins"] + m["losses"] >= dg.MIN_SAMPLE,
                                    "descriptive_only": True})
    return out


def _not(v):
    return None if v is None else not v


def factor_fns(post: dict) -> dict[str, tuple[str, Callable[[dict], bool | None], bool]]:
    side = lambda r, k: CONDITIONS[_SIDE[r["direction"]][k]][1](r)  # noqa: E731
    p = lambda r, k: (post.get(dra.key_of(r)) or {}).get(k)  # noqa: E731
    return {
        "against_ema_trend": ("BUY with EMA20 <= EMA50 / SELL with EMA20 > EMA50", lambda r: _not(side(r, "trend")),
                              False),
        "against_vwap": ("BUY at/below VWAP / SELL above VWAP", lambda r: _not(side(r, "vwap")), False),
        "weak_breakout": ("breakout distance < 0.25 ATR14", lambda r: _not(CONDITIONS["BREAK_STRONG"][1](r)), False),
        "rsi_outside_momentum_band": ("BUY RSI outside 50-70 / SELL RSI outside 30-50",
                                      lambda r: _not(side(r, "rsi")), False),
        "low_volume": ("bought option volume ratio < 0.8", _between("option_vol_ratio", None, VOLUME_EDGES[0]), False),
        "high_volatility": ("ATR regime HIGH_VOLATILITY", _eq("vol_regime", "HIGH_VOLATILITY"), False),
        "choppy_regime": ("market regime CHOPPY", _eq("market_regime", "CHOPPY"), False),
        "trend_up_regime": ("market regime TREND_UP", _eq("market_regime", "TREND_UP"), False),
        "trend_down_regime": ("market regime TREND_DOWN", _eq("market_regime", "TREND_DOWN"), False),
        "late_entry": ("signal at or after 13:00", lambda r: dra._ts(r["signal_time"]).time() >= LATE_ENTRY, False),
        "weak_option_confirmation": ("bought premium NOT rising at the signal candle",
                                     lambda r: _not(r.get("opt_confirm")), False),
        "weak_premium_momentum": ("bought premium 3-candle momentum <= 0", _at_most("premium_mom3_pct", 0.0), False),
        "excessive_sl_distance": (f"estimated stop distance >= {SL_EST_EXCESSIVE:g}% of premium",
                                  _between("sl_est_pct", SL_EST_EXCESSIVE), False),
        "immediate_adverse_premium": ("POST-ENTRY: first held premium candle closes below entry",
                                      lambda r: p(r, "immediate_adverse"), True),
        "target_too_far": ("POST-ENTRY: not a win although the premium reached >= 1R in favour",
                           lambda r: None if p(r, "mfe_r") is None else (r.get("outcome") != imp.WIN
                                                                        and p(r, "mfe_r") >= 1.0), True),
    }


def _train_consistency(srows: Sequence[dict], absent: Sequence[dict]) -> tuple[int, bool]:
    months = sorted({str(r["session_date"])[:7] for r in srows})
    beats = 0
    for m in months:
        b = cb.metrics([r for r in srows if str(r["session_date"])[:7] == m])
        a = cb.metrics([r for r in absent if str(r["session_date"])[:7] == m])
        if a["total_trades"] >= MIN_GROUP_TRADES and _pf(a) > _pf(b):
            beats += 1
    return beats, beats >= 2 and cb.metrics(absent)["total_trades"] >= MIN_TRAIN_TRADES


def root_cause(rows: Sequence[dict], direction: str, post: dict) -> list[dict]:
    out = []
    fns = factor_fns(post)
    for split in SPLITS:
        srows = [r for r in cb.split_rows(rows, split) if r["direction"] == direction]
        base = cb.metrics(srows)
        out.append({"section": "OPTION_COVERAGE", "direction": direction, "split": split, "signals": base["signals"],
                    "unavailable": base["unresolved_or_unavailable"],
                    "coverage": round(1 - base["unresolved_or_unavailable"] / base["signals"], 4)
                    if base["signals"] else None, "unavailable_reasons": json.dumps(base["unavailable_reasons"])})
        for name, (desc, fn, post_only) in fns.items():
            pres = [r for r in srows if fn(r) is True]
            absent = [r for r in srows if fn(r) is False]
            row = {"section": "ROOT_CAUSE", "direction": direction, "factor": name, "description": desc,
                   "split": split, "post_entry_diagnostic_only": post_only, "used_for_selection": False,
                   **_cols(base, "baseline_"), **_cols(cb.metrics(pres), "present_"),
                   **_cols(cb.metrics(absent), "absent_")}
            if split == "TRAIN" and not post_only:
                beats, ok = _train_consistency(srows, absent)
                row.update(train_months_absent_beats_baseline=beats, train_consistent=ok)
            out.append(row)
    return out


def entry_exit_analysis(universe: dict, rows: Sequence[dict], direction: str) -> list[dict]:
    """Phase 6/7 on unchanged S2 signals - TRAIN and VALIDATION only (exits are never judged on OOS/holdout)."""
    out = []
    for split in PRE_OOS:
        for v in VARIANTS:
            sub = [r for r in cb.split_rows(universe[v], split) if r["direction"] == direction
                   and r.get("taken") is not False]
            sec = "BASELINE" if v == BASE_VARIANT else "ENTRY_ANALYSIS" if v[1] == BASE_TARGET else "EXIT_ANALYSIS"
            out.append({"section": sec, "direction": direction, "split": split,
                        "variant": f"entry {v[0]} / target {v[1]:g}R", **_cols(cb.metrics(sub)),
                        "not_taken": sum(1 for r in cb.split_rows(universe[v], split)
                                         if r["direction"] == direction and r.get("taken") is False)})
        ev = _ev([r for r in cb.split_rows(rows, split) if r["direction"] == direction])
        reasons = Counter(str(r.get("exit_reason")) for r in ev)
        for reason, n in sorted(reasons.items()):
            pnl = sum(float(r["pnl"]) for r in ev if str(r.get("exit_reason")) == reason)
            out.append({"section": "EXIT_REASON", "direction": direction, "split": split, "exit_reason": reason,
                        "total_trades": n, "share": round(n / len(ev), 4), "net_pnl": round(pnl, 2)})
    return out


def _binom_upper(k: int, n: int, p: float) -> float | None:
    if n == 0 or p is None:
        return None
    if p <= 0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1:
        return 1.0
    total = 0.0
    for i in range(k, n + 1):
        lg = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + \
            (n - i) * math.log(1 - p)
        total += math.exp(lg)
    return min(1.0, total)


def sell_september(rows: Sequence[dict]) -> dict[str, Any]:
    """Phase 3: was September's SELL result repeatable / regime-specific / random / overfit? Computed after the
    selection was frozen; descriptive only, never used for selection."""
    sell = [r for r in rows if r["direction"] == "SELL"]
    pre = [r for r in sell if cb.split_of(r) in PRE_OOS]
    sep, aug = cb.split_rows(sell, "FINAL_HOLDOUT"), cb.split_rows(sell, "OOS")
    mp, ms, ma = cb.metrics(pre), cb.metrics(sep), cb.metrics(aug)
    months = {m: cb.metrics([r for r in pre if str(r["session_date"])[:7] == m])
              for m in sorted({str(r["session_date"])[:7] for r in pre})}
    prof_months = sum(1 for m in months.values() if m["net_pnl"] > 0)
    n, k = ms["wins"] + ms["losses"], ms["wins"]
    pval = _binom_upper(k, n, mp["win_rate"]) if mp["win_rate"] is not None else None
    regimes = []
    pre_ev, sep_ev = _ev(pre), _ev(sep)
    for g in sorted({fv(r, "market_regime") for r in pre_ev + sep_ev}):
        gp = cb.metrics([r for r in pre if fv(r, "market_regime") == g])
        gs = cb.metrics([r for r in sep if fv(r, "market_regime") == g])
        regimes.append({"regime": g, "pre_share": round(gp["total_trades"] / len(pre_ev), 4) if pre_ev else None,
                        "sep_share": round(gs["total_trades"] / len(sep_ev), 4) if sep_ev else None,
                        "pre_trades": gp["total_trades"], "pre_pf": gp["profit_factor"], "pre_win_rate": gp["win_rate"],
                        "sep_trades": gs["total_trades"], "sep_pnl": gs["net_pnl"], "sep_win_rate": gs["win_rate"]})
    top = max(regimes, key=lambda x: x["sep_pnl"], default=None)
    if _pf(mp) >= PROMISING_OOS_PF and mp["net_pnl"] > 0 and prof_months >= 3:
        answer = "A. REPEATABLE - SELL was already profitable before August in most months"
    elif pval is not None and pval >= 0.05:
        answer = "C. RANDOM - September's SELL win rate is within normal variation of the pre-August rate"
    elif (top and top["pre_trades"] >= MIN_TRAIN_TRADES and (top["pre_pf"] or 0) >= 1.0
          and (top["sep_share"] or 0) - (top["pre_share"] or 0) >= 0.15):
        answer = (f"B. REGIME-SPECIFIC - September had far more {top['regime']} SELL signals than before and that "
                  "regime was already profitable before August")
    else:
        answer = ("D. OVERFIT RISK - September's SELL result is not explained by any condition that was profitable "
                  "before August; any rule tuned to it would be overfit")
    return {"answer": answer, "pre_aug": mp, "august": ma, "september": ms, "pre_months": months,
            "pre_profitable_months": prof_months, "september_pvalue_vs_pre_rate": pval, "regimes": regimes,
            "used_for_selection": False}


# ------------------------------------------------------------------ study
def build(*, rows: Sequence[dict], contexts: Sequence[dict], ctx_by_key: dict, spot: pd.DataFrame,
          bars: pd.DataFrame, params, square_off: str, stop_new_entries: str, range_start: str, range_end: str,
          calendar, listed: Callable, exists: Callable, integrity: dict, mutation: dict | None,
          sessions: Sequence[str], dataset: dict, check_sample: int = 20,
          production_unchanged: bool | None = None) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
    if not all("s6" in r for r in rows):
        rows = prepare(rows, bars, ctx_by_key, range_start, range_end)
    post = post_entry(rows, ctx_by_key)
    universe = build_universe(rows, ctx_by_key, bars, params, square_off, stop_new_entries)

    # Phase 13 - causality
    lookahead = {
        "feature_truncation": dg.truncation_check(spot, rows, ctx_by_key, check_sample),
        "s6_feature_truncation": extra_truncation_check(spot, rows, ctx_by_key, range_start, range_end, check_sample),
        "option_confirmation_truncation": s5.option_truncation_check(rows, ctx_by_key, check_sample),
        "future_candle_scrambling": mutation or {"passed": False, "detail": "NOT RUN"},
        "chronological_replay": dg.chronology_check(rows),
        "entry_alignment_next_candle": dra.alignment_checks(contexts, params),
        "entry_exit_variant_causality": variant_causality_check(universe, ctx_by_key, bars, params, square_off,
                                                                stop_new_entries, check_sample),
    }
    lookahead_ok = all(bool(v.get("passed")) for v in lookahead.values())

    # Phase 4/8/9 - selection on TRAIN + VALIDATION only, then frozen
    sel = select(pre_universe(universe))
    fp = fingerprint(sel)
    overfit_proof = anti_overfit(universe, sel)

    # Phase 10 - OOS for frozen candidates only
    frozen: dict[str, dict | None] = {}
    for d in DIRECTIONS:
        c = sel["selected"][d]
        frozen[d] = None if c is None else frozen_block(f"{c.cid} {c.name}", candidate_rows(c, universe), True)
    picked = [sel["selected"][d] for d in DIRECTIONS if sel["selected"][d]]
    if picked:
        comb_rows = sorted((r for c in picked for r in candidate_rows(c, universe)),
                           key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
        frozen["COMBINED"] = frozen_block(" + ".join(c.cid for c in picked), comb_rows, True, combined=True)
        frozen["COMBINED"]["both_sides"] = len(picked) == 2
    else:
        frozen["COMBINED"] = None

    # Phase 11 - September, once, on the frozen selection
    if fingerprint(sel) != fp:
        raise RuntimeError("selection changed before the final holdout")
    for blk in frozen.values():
        if blk is not None:
            blk["holdout"] = split_m(blk["rows"], "FINAL_HOLDOUT")
            blk["holdout_status"] = holdout_status(blk["holdout"])
            blk["status"] = final_status(blk["validated"], blk["per_split"]["OOS"], blk["validation_to_oos"],
                                         blk["robustness"], blk["holdout_status"], lookahead_ok)

    s2 = {k: cb.scoped(cb.split_rows(rows, k)) for k in SPLITS}
    s2_deg = {s: {"train_to_validation": degradation(s2["TRAIN"][s], s2["VALIDATION"][s]),
                  "validation_to_oos": degradation(s2["VALIDATION"][s], s2["OOS"][s])} for s in SCOPES}

    # integrity
    dup = cb.duplicate_check(rows)
    ts_chk = cb.timestamp_checks(rows, square_off, stop_new_entries)
    cval = cb.contract_validity(rows, calendar, listed, exists)
    frozen_checks = {k: {"duplicates": cb.duplicate_check(blk["rows"]),
                         "timestamps": cb.timestamp_checks(blk["rows"], square_off, stop_new_entries)}
                     for k, blk in frozen.items() if blk is not None}
    frozen_ok = all(v["duplicates"]["passed"] and v["timestamps"]["passed"] for v in frozen_checks.values())
    data_ok = bool(integrity.get("ok")) and dup["passed"] and ts_chk["passed"] and cval["passed"] and frozen_ok

    statuses = [blk["status"] for blk in frozen.values() if blk is not None]
    overfitting_ok = overfit_proof["passed"] and OVERFIT not in statuses
    comb = frozen["COMBINED"]
    target = "NOT ACHIEVED"
    if comb is not None:
        tests = [comb["per_split"]["OOS"], comb["holdout"]]
        if all(m["wins"] + m["losses"] >= MIN_OOS_TRADES and (m["win_rate"] or 0) >= TARGET for m in tests):
            target = "ACHIEVED"
    safe = [f"{k}: {blk['name']}" for k, blk in frozen.items() if blk is not None and blk["status"] == PROMISING]
    if not (lookahead_ok and data_ok and overfitting_ok):
        safe = []

    return {
        "heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "accuracy_definition": "wins / (wins + losses); ambiguous (-1R), scratch and unavailable are never wins",
        "dataset": dataset, "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        "sessions": len(sessions), "signals": len(rows),
        "s2": s2, "s2_degradation": s2_deg,
        "phase1": phase1(rows),
        "root_cause": {d: root_cause(rows, d, post) for d in DIRECTIONS},
        "entry_exit": {d: entry_exit_analysis(universe, rows, d) for d in DIRECTIONS},
        "sell_september": sell_september(rows),
        "selection": sel, "fingerprint": fp, "anti_overfit": overfit_proof,
        "no_signal": {c.cid: no_signal_count(c, universe) for c in sel["candidates"].values()},
        "frozen": frozen, "target_70": target,
        "lookahead": lookahead, "lookahead_ok": lookahead_ok,
        "data_integrity": integrity, "data_checks": {"duplicates": dup, "timestamps": ts_chk,
                                                     "contract_validity": cval, "frozen": frozen_checks},
        "data_ok": data_ok, "overfitting_ok": overfitting_ok,
        "forward_paper_candidates": safe,
        "recommendation": ("Forward PAPER validation candidate(s): " + "; ".join(safe)) if safe else
        "NO candidate is safe to move to forward PAPER validation.",
        "disclosure": ("August and September S2 baseline results were already seen in earlier labs (S2 diagnostic, "
                       "S5, clean backtest). S6 candidates are the pre-declared families of the S6 specification with "
                       "coarse fixed thresholds, selected on TRAIN/VALIDATION only, but the researcher was not blind "
                       "to those months; treat OOS and holdout as optimistic and require fresh forward PAPER data."),
        "production_unchanged": production_unchanged, "live_orders": 0, "trading_mode": "PAPER",
        "live_trading_enabled": False, "rows": rows,
    }


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "s6_report.md", "candidate_results.csv", "train_results.csv",
                "validation_results.csv", "oos_results.csv", "holdout_results.csv", "regime_analysis.csv",
                "time_analysis.csv", "buy_analysis.csv", "sell_analysis.csv", "robustness.csv", "data_quality.json")
CSV_METRICS = METRIC_COLS + ("gross_profit", "gross_loss", "unresolved_or_unavailable")


def _m(m: dict) -> dict[str, Any]:
    return {**{k: m[k] for k in CSV_METRICS}, "expectancy_rs": expectancy_rs(m)}


def _status_of(a: dict, cid: str) -> str | None:
    for k, blk in a["frozen"].items():
        if blk is not None and k != "COMBINED" and blk["name"].split(" ")[0] == cid:
            return blk["status"]
    return None


def candidate_csv(a: dict) -> list[dict]:
    sel, out = a["selection"], []
    chosen = {c.cid for c in sel["selected"].values() if c}
    for cid, c in sel["candidates"].items():
        tr, va = sel["train"].get(cid), sel["validation"].get(cid)
        base = {"candidate": cid, "name": c.name, "direction": c.direction, "family": c.family, "rules": c.describe(),
                "entry": c.entry, "target_r": c.target_r, "train_eligible": tr["eligible"] if tr else None,
                "train_failed": ",".join(tr["failed"]) if tr else None,
                "validation_verdict": va["verdict"] if va else "NOT_EVALUATED (not TRAIN-eligible)",
                "validation_failed": ",".join(va["failed"]) if va else None, "selected": cid in chosen,
                "final_status": _status_of(a, cid), "no_signal": a["no_signal"].get(cid)}
        if tr:
            out.append({**base, "split": "TRAIN", **_m(tr["metrics"])})
        if va:
            out.append({**base, "split": "VALIDATION", **_m(va["metrics"])})
        if cid in chosen:
            blk = a["frozen"][c.direction]
            out.append({**base, "split": "OOS", **_m(blk["per_split"]["OOS"])})
            out.append({**base, "split": HOLDOUT_LABEL, **_m(blk["holdout"])})
    return out


def train_csv(a: dict) -> list[dict]:
    sel, out = a["selection"], []
    for cid, tr in sel["train"].items():
        c = sel["candidates"][cid]
        rank = sel["ranking"][c.direction].index(cid) + 1 if cid in sel["ranking"][c.direction] else None
        out.append({"candidate": cid, "name": c.name, "direction": c.direction, "family": c.family,
                    "rationale": c.rationale, "rules": c.describe(), **_m(tr["metrics"]), **tr["checks"],
                    "eligible": tr["eligible"], "train_rank": rank,
                    "stage": "STAGE2_VARIANT" if cid in sel["stage2"][c.direction] else "STAGE1"})
    return out


def validation_csv(a: dict) -> list[dict]:
    sel = a["selection"]
    return [{"candidate": cid, "name": sel["candidates"][cid].name, "direction": sel["candidates"][cid].direction,
             **_m(va["metrics"]), **va["checks"], "profitable_time_windows_pre_oos": va["time_windows"]["profitable"],
             "profitable_regimes_pre_oos": va["regimes"]["profitable"], "verdict": va["verdict"]}
            for cid, va in sel["validation"].items()]


def _frozen_rows(a: dict, split: str) -> list[dict]:
    out = []
    for k in SCOPES:
        m = a["s2"][split][k]
        out.append({"strategy": f"S2_BASELINE_{k}", "scope": k, **_m(m),
                    **({"train_to_validation": a["s2_degradation"][k]["train_to_validation"],
                        "validation_to_oos": a["s2_degradation"][k]["validation_to_oos"]} if split == "OOS" else {})})
        blk = a["frozen"].get(k)
        if blk is None:
            out.append({"strategy": f"S6_BEST_{k}", "scope": k, "candidate": "none"})
            continue
        m = blk["per_split"]["OOS"] if split == "OOS" else blk["holdout"]
        row = {"strategy": f"S6_BEST_{k}", "scope": k, "candidate": blk["name"], **_m(m), "status": blk["status"]}
        if split == "OOS":
            row.update(train_to_validation=blk["train_to_validation"], validation_to_oos=blk["validation_to_oos"])
        else:
            row.update(holdout_status=blk["holdout_status"], label=HOLDOUT_LABEL)
        out.append(row)
    return out


def robustness_csv(a: dict) -> list[dict]:
    out = []
    for k, blk in a["frozen"].items():
        if blk is None:
            continue
        rb = blk["robustness"]
        for comp in ("monthly", "regime", "time_of_day"):
            out.append({"scope": k, "candidate": blk["name"], "component": comp,
                        **{x: rb[comp][x] for x in ("groups", "profitable", "best_group", "best_share_of_profit",
                                                    "total_pnl", "pnl_without_best", "concentrated")}})
        out.append({"scope": k, "candidate": blk["name"], "component": "few_trades", **rb["few_trades"]})
        out.append({"scope": k, "candidate": blk["name"], "component": "summary",
                    "max_losing_streak": rb["max_losing_streak"], "max_drawdown": rb["max_drawdown"],
                    "trades_train_to_oos": rb["trades_train_to_oos"],
                    "oos_trade_count_sufficient": rb["oos_trade_count_sufficient"],
                    "buy_sell": json.dumps(rb.get("buy_sell")) if rb.get("buy_sell") else None,
                    "overfit_flags": ",".join(rb["overfit_flags"]), "overfit_risk": rb["overfit_risk"],
                    "final_status": blk["status"]})
    return out


def side_csv(a: dict, d: str) -> list[dict]:
    out = [r for r in a["phase1"] if r["scope"] == d] + a["root_cause"][d] + a["entry_exit"][d]
    if d == "SELL":
        s = a["sell_september"]
        out.append({"section": "SEPTEMBER_SELL_ASSESSMENT", "direction": "SELL", "answer": s["answer"],
                    "pre_aug": json.dumps(_m(s["pre_aug"])), "august": json.dumps(_m(s["august"])),
                    "september": json.dumps(_m(s["september"])),
                    "pvalue_vs_pre_rate": s["september_pvalue_vs_pre_rate"],
                    "regimes": json.dumps(s["regimes"], default=str), "used_for_selection": False})
    return out


def _jsonable(v):
    if isinstance(v, Candidate):
        return asdict(v)
    if isinstance(v, float) and math.isinf(v):
        return "INF" if v > 0 else "-INF"
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (date, datetime, pd.Timestamp)):
        return v.isoformat()
    return v


def _csv(path: Path, rows: Sequence[dict]) -> None:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("INF" if isinstance(v, float) and math.isinf(v) else v) for k, v in r.items()})


def line(m: dict | None) -> str:
    if not m:
        return "N/A"
    exp = "N/A" if m["average_r"] is None else f"{m['average_r']:+.3f}R"
    return f"{cb.line(m)} | Max streak {m['max_losing_streak']} | Expectancy {exp}"


def _summary_json(a: dict) -> dict:
    frozen = {k: None if blk is None else {x: v for x, v in blk.items() if x != "rows"}
              for k, blk in a["frozen"].items()}
    return {**{k: v for k, v in a.items() if k not in ("rows", "frozen", "phase1", "root_cause", "entry_exit",
                                                         "data_integrity")}, "frozen": frozen}


def markdown(a: dict) -> str:
    sel = a["selection"]
    md = ["# S6 clean strategy improvement lab", "", f"**{BANNER}**", "",
          f"Generated {a['generated_at']}. Dataset {a['dataset'].get('from')} -> {a['dataset'].get('to')} "
          f"(data/historical/upstox_6m, NSE_INDEX|Nifty 50, real option premium candles). TRADING_MODE=PAPER, "
          "LIVE_TRADING_ENABLED=false, live orders: 0. Production S2 changed: "
          f"**{'NO' if a['production_unchanged'] is not False else 'YES'}**.", "",
          "All numbers are **historical backtest accuracy**, not live accuracy, and imply no guaranteed profit. "
          "Accuracy = wins / (wins + losses); ambiguous candles are booked at -1R.", "",
          f"> {a['disclosure']}", "", "## Result", "",
          f"- Lookahead: **{'PASS' if a['lookahead_ok'] else 'FAIL'}** | Data integrity: "
          f"**{'PASS' if a['data_ok'] else 'FAIL'}** | Overfitting: **{'PASS' if a['overfitting_ok'] else 'FAIL'}** "
          f"| 70% target: **{a['target_70']}**", f"- {a['recommendation']}", ""]
    for k in SCOPES:
        blk = a["frozen"].get(k)
        if blk is None:
            md.append(f"- Best {k}: none passed TRAIN selection and VALIDATION.")
            continue
        md.append(f"- Best {k}: **{blk['name']}** - status **{blk['status']}**; OOS {line(blk['per_split']['OOS'])}; "
                  f"holdout ({HOLDOUT_LABEL}) {line(blk['holdout'])}")
    md += ["", "## Splits", ""] + [f"- {k}: {lo} -> {hi}" for k, (lo, hi) in a["splits"].items()]
    md += ["", f"Selection fingerprint `{a['fingerprint']}` (verified before September).", "",
           "## Baseline S2 (exact replay)", "", "| Split | Scope | Result |", "|---|---|---|"]
    md += [f"| {SPLIT_NAMES[k]} | {s} | {line(a['s2'][k][s])} |" for k in SPLITS for s in SCOPES]
    md += ["", "## Phase 1 - diagnostics (descriptive only)", "",
           "Full tables: regime_analysis.csv, time_analysis.csv, buy_analysis.csv, sell_analysis.csv. No bucket was "
           "turned into a rule; candidates are the pre-declared families below.", "",
           "## Phase 2/3 - root causes (TRAIN-consistent factors)", ""]
    for d in DIRECTIONS:
        cons = [r["factor"] for r in a["root_cause"][d] if r.get("train_consistent")]
        md.append(f"- {d}: removing these factors helped in >= 2 of 3 TRAIN months with >= {MIN_TRAIN_TRADES} "
                  f"remaining trades: {cons or 'none'}")
    s = a["sell_september"]
    md += [f"- September SELL assessment (after freezing, not used for selection): **{s['answer']}** "
           f"(pre-August SELL {line(s['pre_aug'])}; August {line(s['august'])}; September {line(s['september'])}; "
           f"p-value vs pre-August win rate {s['september_pvalue_vs_pre_rate']})", "",
           "## Phase 4/8 - candidates and TRAIN selection", "",
           f"TRAIN-eligible: >= {MIN_TRAIN_TRADES} trades; PF > 1 or PF >= S2 PF + {PF_IMPROVEMENT}; max DD <= "
           f"{MAX_DD_SHARE:.0%} of gross profit; best trade <= {MAX_BEST_TRADE_SHARE:.0%} of gross profit; losing "
           f"streak <= {MAX_TRAIN_STREAK}; economic rationale. Ranked by expectancy (avg R), then PF, then trades, "
           "then fewer rules.", "", "| ID | Side | Rules | TRAIN | Eligible | Failed | VALIDATION | Verdict |",
           "|---|---|---|---|---|---|---|---|"]
    for cid, c in sel["candidates"].items():
        tr, va = sel["train"][cid], sel["validation"].get(cid)
        md.append(f"| {cid} | {c.direction} | {c.describe()} | {line(tr['metrics'])} | {tr['eligible']} | "
                  f"{','.join(tr['failed']) or '-'} | {line(va['metrics']) if va else '-'} | "
                  f"{va['verdict'] if va else 'not evaluated'} |")
    md += ["", f"Selected: {ids(sel)}. Candidates per side: {sel['counts']} (max {MAX_CANDIDATES_PER_SIDE}).", "",
           "## Phase 9/10/11 - frozen candidates", "",
           "| Scope | Candidate | TRAIN | VALIDATION | OOS | Train->Val | Val->OOS | Holdout | Status |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k in SCOPES:
        blk = a["frozen"].get(k)
        if blk is not None:
            p = blk["per_split"]
            md.append(f"| {k} | {blk['name']} | {line(p['TRAIN'])} | {line(p['VALIDATION'])} | {line(p['OOS'])} | "
                      f"{blk['train_to_validation']} | {blk['validation_to_oos']} | {line(blk['holdout'])} "
                      f"({blk['holdout_status']}) | **{blk['status']}** |")
    md += ["", f"September rows are labelled **{HOLDOUT_LABEL}**.", "", "## Phase 12 - robustness", ""]
    for k, blk in a["frozen"].items():
        if blk is not None:
            rb = blk["robustness"]
            md.append(f"- {k} {blk['name']}: overfit flags {rb['overfit_flags'] or 'none'}; months "
                      f"{rb['monthly']['profitable']}/{rb['monthly']['groups']} profitable; regimes "
                      f"{rb['regime']['profitable']}/{rb['regime']['groups']}; time windows "
                      f"{rb['time_of_day']['profitable']}/{rb['time_of_day']['groups']}; P&L without top "
                      f"{TOP_TRADES} trades Rs {rb['few_trades']['pnl_without_top_trades']:,.2f}")
    md += ["", "## Phase 13 - anti-overfitting and lookahead", ""]
    md += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in a["anti_overfit"]["checks"].items()]
    md += [f"- {k}: {'PASS' if v.get('passed') else 'FAIL'} ({v.get('checked', v.get('cuts', '-'))} checked)"
           for k, v in a["lookahead"].items()]
    dc = a["data_checks"]
    md += ["", "## Data quality", "", f"- 6-month store integrity: {'PASS' if a['data_integrity'].get('ok') else 'FAIL'}",
           f"- duplicates: {dc['duplicates']}", f"- timestamps: {dc['timestamps']['problems'] or 'none'}",
           f"- contract validity: {dc['contract_validity']['invalid']} invalid of {dc['contract_validity']['checked']}",
           "", "## Phase 14 - recommendation", "", a["recommendation"],
           "No candidate is promoted automatically. Production S2 is unchanged; any change would need a separate, "
           "explicit decision after forward PAPER validation."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    p["summary.json"].write_text(json.dumps(_jsonable(_summary_json(a)), indent=2, default=str), encoding="utf-8")
    p["s6_report.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["candidate_results.csv"], candidate_csv(a))
    _csv(p["train_results.csv"], train_csv(a))
    _csv(p["validation_results.csv"], validation_csv(a))
    _csv(p["oos_results.csv"], _frozen_rows(a, "OOS"))
    _csv(p["holdout_results.csv"], _frozen_rows(a, "FINAL_HOLDOUT"))
    _csv(p["regime_analysis.csv"], [r for r in a["phase1"] if r["dimension"] in REGIME_DIMS])
    _csv(p["time_analysis.csv"], [r for r in a["phase1"] if r["dimension"] == "time_of_day"])
    _csv(p["buy_analysis.csv"], side_csv(a, "BUY"))
    _csv(p["sell_analysis.csv"], side_csv(a, "SELL"))
    _csv(p["robustness.csv"], robustness_csv(a))
    dq = {"data_integrity": a["data_integrity"], "checks": a["data_checks"], "data_ok": a["data_ok"],
          "unavailable_reasons": {k: a["s2"][k]["COMBINED"]["unavailable_reasons"] for k in SPLITS},
          "lookahead": a["lookahead"], "no_synthetic_values": a["lookahead"]["entry_exit_variant_causality"]
          .get("synthetic_or_wrong_entry") == []}
    p["data_quality.json"].write_text(json.dumps(_jsonable(dq), indent=2, default=str), encoding="utf-8")
    return p


def terminal_text(a: dict) -> str:
    out = [HEADING, "(historical backtest accuracy - not live accuracy, no guaranteed profit)", "", "BASELINE S2:", ""]
    for k in SPLITS:
        out.append(f"{SPLIT_NAMES[k]}:")
        out += [f"{s}: {line(a['s2'][k][s])}" for s in SCOPES]
        out.append("")
    for k, label in (("BUY", "BEST BUY CANDIDATE"), ("SELL", "BEST SELL CANDIDATE")):
        blk = a["frozen"].get(k)
        out.append(f"{label}:")
        if blk is None:
            out += ["name: none (no candidate passed TRAIN selection + VALIDATION)", "TRAIN: N/A", "VALIDATION: N/A",
                    "OOS: N/A", "HOLDOUT: N/A", f"status: {REJECTED}", ""]
            continue
        p = blk["per_split"]
        out += [f"name: {blk['name']}", f"TRAIN: {line(p['TRAIN'])}", f"VALIDATION: {line(p['VALIDATION'])}",
                f"OOS: {line(p['OOS'])}", f"HOLDOUT: {line(blk['holdout'])} ({HOLDOUT_LABEL})",
                f"status: {blk['status']}", ""]
    blk = a["frozen"].get("COMBINED")
    out.append("BEST COMBINED:")
    if blk is None:
        out += ["name: none", "OOS: N/A", "HOLDOUT: N/A", f"status: {REJECTED}", ""]
    else:
        out += [f"name: {blk['name']}", f"OOS: {line(blk['per_split']['OOS'])}",
                f"HOLDOUT: {line(blk['holdout'])} ({HOLDOUT_LABEL})", f"status: {blk['status']}", ""]
    out += ["70% TARGET:", a["target_70"], "", "LOOKAHEAD:", "PASS" if a["lookahead_ok"] else "FAIL", "",
            "DATA INTEGRITY:", "PASS" if a["data_ok"] else "FAIL", "", "OVERFITTING:",
            "PASS" if a["overfitting_ok"] else "FAIL", "", "PRODUCTION CHANGED:",
            "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:", "0", "",
            "FORWARD PAPER VALIDATION:", a["recommendation"]]
    return "\n".join(out)
