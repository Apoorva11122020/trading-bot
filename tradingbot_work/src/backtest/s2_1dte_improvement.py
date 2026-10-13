"""Controlled S2 1-DTE improvement research (research only, PAPER, S2 unchanged).

Signals come from the unchanged S2 generator. Research variants only remove
signals (repeated-signal and breakout-distance filters) or change the option
exit levels (premium-ATR stop multiple, target R). Every surviving signal is
evaluated independently with the live tracker's method: next option candle
open entry, look-ahead-safe premium ATR (candles <= signal candle), first
touch of stop/target on real premium candles, square-off at 15:15.

All variants are predeclared in ``predeclared_variants``. Combined candidates
are built by a fixed rule from TRAIN metrics only. Historical option contracts
are 5-33 DTE and labelled NON-COMPARABLE_1DTE; only the current day is 1-DTE.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest.current_day_s2 import LevelParams, option_levels
from src.backtest.live_vs_historical import TIME_BUCKETS, time_bucket, wilson
from src.indicators.technical import atr
from src.options.models import OptionResolutionError
from src.options.strategy import OptionSignalError, option_stop_loss, option_target
from src.paper_trading.signal_tracker import AMBIGUOUS, INVALID, LOSS, WIN, SignalRecord, evaluate_candle

IST = ZoneInfo("Asia/Kolkata")
NON_COMPARABLE = "NON-COMPARABLE_1DTE"
COMPARABLE = "COMPARABLE_1DTE_CURRENT_DAY"
SPLITS = ("TRAIN", "VALIDATION", "OOS")
DEFAULT_SPLITS = {"TRAIN": ("2026-09-01", "2026-09-17"), "VALIDATION": ("2026-09-18", "2026-09-24"),
                  "OOS": ("2026-09-25", "2026-10-01")}
STATUS_ORDER = ("ROBUST", "PROMISING", "CURRENT_DAY_ONLY", "OVERFIT", "NOT_ROBUST", "LOOKAHEAD_FAIL")
SCRATCH, PENDING, UNAVAILABLE = "SCRATCH", "PENDING", "DATA_UNAVAILABLE"
ACCEPTED = "ACCEPTED"
GAP_WINDOWS = (5, 10, 15, 20, 30)


# ------------------------------------------------------------------- variants
@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    repeat: str = "NONE"
    cooldown_min: int = 0
    max_break_pct: float | None = None
    max_break_atr: float | None = None
    stop_atr: float = 1.5
    target_r: float = 2.0

    def describe(self) -> list[str]:
        rep = {"NONE": "all S2 signals (no repeat filter)",
               "FIRST_SIGNAL_ONLY": "first signal per direction until the opposite ORB side breaks",
               "COOLDOWN": f"{self.cooldown_min}-minute same-direction cooldown",
               "BREAKOUT_REARM": "re-arm only after a close back inside the opening range"}[self.repeat]
        entry = ("no breakout-distance limit" if self.max_break_pct is None and self.max_break_atr is None else
                 f"breakout distance <= {self.max_break_pct:.2f}%" if self.max_break_pct is not None else
                 f"breakout distance <= {self.max_break_atr:.2f} ATR")
        return [rep, entry, f"stop {self.stop_atr:.2f} x premium ATR", f"target {self.target_r:g}R"]

    @property
    def dims(self) -> tuple:
        return ((self.repeat, self.cooldown_min), (self.max_break_pct, self.max_break_atr),
                (self.stop_atr, self.target_r))


BASELINE = Variant("BASELINE_S2", "BASELINE")


def predeclared_variants() -> list[Variant]:
    """Phases 2-4, fixed before any result is computed (baseline + 26 single-change variants)."""
    out = [BASELINE, Variant("REPEAT_FIRST_SIGNAL_ONLY", "REPEAT", repeat="FIRST_SIGNAL_ONLY")]
    out += [Variant(f"REPEAT_COOLDOWN_{m}M", "REPEAT", repeat="COOLDOWN", cooldown_min=m) for m in (10, 15, 20, 30)]
    out.append(Variant("REPEAT_BREAKOUT_REARM", "REPEAT", repeat="BREAKOUT_REARM"))
    out += [Variant(f"ENTRY_MAX_BREAK_{p:.2f}PCT", "ENTRY", max_break_pct=p) for p in (0.10, 0.15, 0.20, 0.25, 0.30)]
    out += [Variant(f"ENTRY_MAX_BREAK_{a:.2f}ATR", "ENTRY", max_break_atr=a) for a in (0.25, 0.50, 0.75, 1.00)]
    out += [Variant(f"EXIT_STOP_{s:.2f}ATR_T{t:g}R", "EXIT", stop_atr=s, target_r=t)
            for s in (1.25, 1.50, 1.75, 2.00) for t in (1.0, 1.5, 2.0) if (s, t) != (1.5, 2.0)]
    return out


# ------------------------------------------------------------------ features
def _ist(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def underlying_features(candles: pd.DataFrame, range_start: str = "09:15", range_end: str = "09:30",
                        atr_period: int = 14) -> pd.DataFrame:
    """Causal per-bar ATR and opening-range levels (NaN until the range is complete)."""
    f = candles[["timestamp", "open", "high", "low", "close"]].copy()
    ts = pd.to_datetime(f["timestamp"])
    f["timestamp"] = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    f = f.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    f["day"] = f["timestamp"].dt.date
    f["atr14"] = atr(f, atr_period)
    t = f["timestamp"].dt.time
    start, end = time.fromisoformat(range_start), time.fromisoformat(range_end)
    in_range = (t >= start) & (t < end)
    hi = f["high"].where(in_range).groupby(f["day"]).transform("max")
    lo = f["low"].where(in_range).groupby(f["day"]).transform("min")
    f["orb_high"] = hi.where(t >= end)
    f["orb_low"] = lo.where(t >= end)
    return f


# ------------------------------------------------------------ signal context
def build_contexts(signals: Iterable, features: pd.DataFrame, resolve: Callable, premium_for: Callable,
                   params: LevelParams) -> list[dict[str, Any]]:
    """Per-signal point-in-time facts: breakout distance, contract, entry, premium ATR (candles <= signal)."""
    by_ts = features.set_index("timestamp")
    out = []
    for sig in sorted(signals, key=lambda s: (s.timestamp, s.direction)):
        ts = _ist(sig.timestamp)
        level = sig.conditions.get("above", sig.conditions.get("below"))
        sign = 1.0 if sig.direction == "BUY" else -1.0
        bar = by_ts.loc[ts] if ts in by_ts.index else None
        a = float(bar["atr14"]) if bar is not None and pd.notna(bar["atr14"]) else None
        dist = sign * (sig.spot - level) if level is not None else None
        ctx: dict[str, Any] = {
            "signal": sig, "signal_time": ts, "session_date": ts.date(), "direction": sig.direction,
            "spot": float(sig.spot), "breakout_level": None if level is None else float(level), "atr14": a,
            "break_pct": None if dist is None else dist / sig.spot * 100,
            "break_atr": None if dist is None or not a else dist / a, "time_bucket": time_bucket(ts),
            "data_status": "OK", "reason": None, "contract": None, "instrument_key": None, "expiry": None,
            "dte": None, "entry_time": None, "entry": None, "premium_atr": None, "tick": None, "premium": None,
        }
        out.append(ctx)
        try:
            contract = resolve(sig)
        except OptionResolutionError as exc:
            ctx.update(data_status=UNAVAILABLE, reason=f"CONTRACT_UNRESOLVED:{exc.code}")
            continue
        ctx.update(contract=getattr(contract, "trading_symbol", None), instrument_key=contract.instrument_key,
                   expiry=contract.expiry, dte=(contract.expiry - ts.date()).days,
                   tick=getattr(contract, "tick_size", None))
        premium = premium_for(contract)
        if premium is None or isinstance(premium, str) or premium.empty:
            ctx.update(data_status=UNAVAILABLE, reason=premium if isinstance(premium, str) else "NO_PREMIUM_DATA")
            continue
        levels = option_levels(sig, contract, premium, params)
        if levels.get("premium_atr") is None or levels.get("entry") is None:
            ctx.update(data_status=UNAVAILABLE, reason=levels.get("reason"))
            continue
        ctx.update(entry_time=_ist(levels["entry_time"]), entry=float(levels["entry"]),
                   premium_atr=float(levels["premium_atr"]), premium=premium)
    return out


# ------------------------------------------------------------------- filters
def repeat_filter(contexts: Sequence[dict], features: pd.DataFrame, variant: Variant) -> list[str | None]:
    """Skip reason per context (None = accepted). Decisions read only bars/signals at or before each signal."""
    if variant.repeat == "NONE":
        return [None] * len(contexts)
    closes = features.set_index("timestamp")[["close", "orb_high", "orb_low"]]
    out: list[str | None] = []
    last: dict[str, pd.Timestamp] = {}
    day = None
    for ctx in contexts:
        ts, d = ctx["signal_time"], ctx["direction"]
        if ctx["session_date"] != day:
            day, last = ctx["session_date"], {}
        prev = last.get(d)
        skip = None
        if prev is not None:
            if variant.repeat == "COOLDOWN":
                if ts - prev < pd.Timedelta(minutes=variant.cooldown_min):
                    skip = f"COOLDOWN_{variant.cooldown_min}M"
            else:
                between = closes[(closes.index > prev) & (closes.index < ts)]
                if variant.repeat == "FIRST_SIGNAL_ONLY":
                    reset = (between["close"] < between["orb_low"]).any() if d == "BUY" else \
                        (between["close"] > between["orb_high"]).any()
                    skip = None if reset else "FIRST_SIGNAL_ONLY"
                else:
                    rearmed = (between["close"] <= between["orb_high"]).any() if d == "BUY" else \
                        (between["close"] >= between["orb_low"]).any()
                    skip = None if rearmed else "BREAKOUT_NOT_REARMED"
        if skip is None:
            last[d] = ts
        out.append(skip)
    return out


def entry_filter(ctx: dict, variant: Variant) -> str | None:
    if variant.max_break_pct is not None:
        if ctx["break_pct"] is None:
            return "BREAK_DISTANCE_UNKNOWN"
        if ctx["break_pct"] > variant.max_break_pct:
            return f"BREAK_PCT>{variant.max_break_pct:.2f}"
    if variant.max_break_atr is not None:
        if ctx["break_atr"] is None:
            return "BREAK_ATR_UNKNOWN"
        if ctx["break_atr"] > variant.max_break_atr:
            return f"BREAK_ATR>{variant.max_break_atr:.2f}"
    return None


# ---------------------------------------------------------------- evaluation
def exit_levels(ctx: dict, stop_atr: float, target_r: float) -> tuple[float, float]:
    stop = option_stop_loss(ctx["entry"], stop_mode="premium_atr", multiplier=stop_atr,
                            premium_atr=ctx["premium_atr"], tick_size=ctx["tick"])
    return stop, option_target(ctx["entry"], stop, target_r, ctx["tick"])


def evaluate(ctx: dict, stop_atr: float, target_r: float, square_off: str, lot_size: int,
             cache: dict | None = None) -> dict[str, Any]:
    """First-touch outcome on same-day real premium candles. Ambiguous candles are kept as AMBIGUOUS."""
    key = (ctx["signal_time"], ctx["direction"], stop_atr, target_r)
    if cache is not None and key in cache:
        return cache[key]
    res: dict[str, Any] = {"outcome": None, "pnl": None, "r_multiple": None, "exit_reason": None,
                           "exit_time": None, "exit_price": None, "stop_loss": None, "target": None}
    if ctx["data_status"] != "OK":
        res["outcome"] = UNAVAILABLE
    else:
        try:
            stop, target = exit_levels(ctx, stop_atr, target_r)
        except OptionSignalError as exc:
            res.update(outcome=UNAVAILABLE, exit_reason=exc.code)
            stop = target = None
        if stop is not None:
            res.update(stop_loss=stop, target=target)
            sig = ctx["signal"]
            record = SignalRecord.create(timestamp=ctx["signal_time"], underlying=sig.underlying,
                                         direction=sig.direction, active_from=ctx["entry_time"],
                                         instrument_key=ctx["instrument_key"], entry_price=ctx["entry"],
                                         stop_loss=stop, target=target, quantity=lot_size)
            prem = ctx["premium"]
            future = prem[(prem["timestamp"] >= ctx["entry_time"])
                          & (prem["timestamp"].dt.date == ctx["session_date"])]
            cutoff = time.fromisoformat(square_off)
            out = None
            for bar in future.to_dict("records"):
                candle = {"timestamp": _ist(bar["timestamp"]).to_pydatetime(),
                          **{k: float(bar[k]) for k in ("open", "high", "low", "close")}}
                out = evaluate_candle(record, candle, cutoff)
                if out is not None:
                    break
            if out is None:
                res.update(outcome=PENDING, exit_reason="NO_EXIT_EVIDENCE")
            else:
                status = {WIN: WIN, LOSS: LOSS, AMBIGUOUS: AMBIGUOUS, INVALID: SCRATCH}[out["status"]]
                res.update(outcome=status, pnl=out["pnl"], r_multiple=out["r_multiple"], exit_reason=out["reason"],
                           exit_time=_ist(out["exit_time"]).isoformat(), exit_price=out["exit_price"])
                if status == SCRATCH:
                    res.update(pnl=0.0, r_multiple=0.0)
    if cache is not None:
        cache[key] = res
    return res


def run_variant(contexts: Sequence[dict], features: pd.DataFrame, variant: Variant, *, square_off: str,
                lot_size: int = 65, cache: dict | None = None) -> list[dict[str, Any]]:
    """One row per S2 signal: accepted or skipped (with reason) plus the outcome if accepted."""
    skips = repeat_filter(contexts, features, variant)
    rows = []
    for ctx, skip in zip(contexts, skips):
        skip = skip or entry_filter(ctx, variant)
        row = {"variant": variant.name, "session_date": ctx["session_date"].isoformat(),
               "signal_time": ctx["signal_time"].isoformat(), "direction": ctx["direction"], "spot": ctx["spot"],
               "breakout_level": ctx["breakout_level"], "break_pct": ctx["break_pct"], "break_atr": ctx["break_atr"],
               "time_bucket": ctx["time_bucket"], "contract": ctx["contract"], "instrument_key": ctx["instrument_key"],
               "dte": ctx["dte"], "entry_time": None if ctx["entry_time"] is None else ctx["entry_time"].isoformat(),
               "entry": ctx["entry"], "premium_atr": ctx["premium_atr"],
               "decision": ACCEPTED if skip is None else "SKIPPED", "skip_reason": skip,
               "data_reason": ctx["reason"]}
        if skip is None:
            row.update(evaluate(ctx, variant.stop_atr, variant.target_r, square_off, lot_size, cache))
        rows.append(row)
    return rows


# ------------------------------------------------------------------- metrics
def _pf(gp: float, gl: float) -> float | None:
    return gp / gl if gl > 0 else (math.inf if gp > 0 else None)


def _conservative(r: dict) -> tuple[float, float]:
    """Ambiguous counts as a stop-loss (-1R); otherwise the tracker's P&L."""
    if r["outcome"] == AMBIGUOUS:
        return (r["stop_loss"] - r["entry"]) * 65, -1.0
    return float(r["pnl"]), float(r["r_multiple"])


def direction_stats(rows: Sequence[dict]) -> dict[str, Any]:
    decided = [r for r in rows if r.get("outcome") in (WIN, LOSS, AMBIGUOUS, SCRATCH)]
    wins = sum(1 for r in decided if r["outcome"] == WIN)
    pnl = [_conservative(r)[0] for r in decided]
    return {"signals": len(rows), "evaluated": len(decided), "wins": wins, "losses": len(decided) - wins,
            "accuracy": wins / len(decided) if decided else None,
            "profit_factor": _pf(sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)) if decided else None,
            "net_pnl": round(sum(pnl), 2)}


def compute_metrics(rows: Sequence[dict]) -> dict[str, Any]:
    """Win rate = wins / evaluated (wins, losses, ambiguous-as-loss and scratches); tracker WR also reported."""
    acc = sorted((r for r in rows if r["decision"] == ACCEPTED), key=lambda r: r["signal_time"])
    decided = [r for r in acc if r["outcome"] in (WIN, LOSS, AMBIGUOUS, SCRATCH)]
    wins = sum(1 for r in decided if r["outcome"] == WIN)
    losses = sum(1 for r in decided if r["outcome"] == LOSS)
    amb = sum(1 for r in decided if r["outcome"] == AMBIGUOUS)
    pnl_r = [_conservative(r) for r in decided]
    gp, gl = sum(p for p, _ in pnl_r if p > 0), -sum(p for p, _ in pnl_r if p < 0)
    eq = peak = dd = 0.0
    streak = max_streak = 0
    for (p, _), r in zip(pnl_r, decided):
        eq += p
        peak, dd = max(peak, eq), max(dd, peak - eq)
        streak = streak + 1 if r["outcome"] != WIN else 0
        max_streak = max(max_streak, streak)
    n = len(decided)
    ci = wilson(wins, n)
    skip_reasons: dict[str, int] = {}
    for r in rows:
        if r["decision"] != ACCEPTED:
            key = str(r["skip_reason"])
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
    days = sorted({r["session_date"] for r in rows})
    return {
        "s2_signals": len(rows), "accepted_signals": len(acc), "skipped_signals": len(rows) - len(acc),
        "skip_reasons": skip_reasons, "evaluated": n, "wins": wins, "losses": losses, "ambiguous": amb,
        "scratches": n - wins - losses - amb,
        "pending_or_unavailable": len(acc) - n,
        "win_rate": wins / n if n else None,
        "win_rate_ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "tracker_win_rate": wins / (wins + losses) if wins + losses else None,
        "profit_factor": _pf(gp, gl) if n else None, "net_pnl": round(gp - gl, 2), "max_drawdown": round(dd, 2),
        "average_r": round(sum(r for _, r in pnl_r) / n, 4) if n else None, "max_consecutive_losses": max_streak,
        "sessions": len(days), "accepted_per_day": round(len(acc) / len(days), 2) if days else None,
        "BUY": direction_stats([r for r in acc if r["direction"] == "BUY"]),
        "SELL": direction_stats([r for r in acc if r["direction"] == "SELL"]),
        "time_of_day": {lab: direction_stats([r for r in acc if r["time_bucket"] == lab])
                        for _, _, lab in TIME_BUCKETS if any(r["time_bucket"] == lab for r in acc)},
    }


def _summary(values: Iterable) -> dict[str, Any] | None:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return None
    return {"n": len(v), "median": round(median(v), 4), "min": round(min(v), 4), "max": round(max(v), 4)}


def baseline_profile(rows: Sequence[dict]) -> dict[str, Any]:
    """Phase 1 descriptive fields for the unchanged S2 (contract, DTE, premium ATR, stop/target distance)."""
    acc = [r for r in rows if r["decision"] == ACCEPTED]
    ok = [r for r in acc if r.get("stop_loss") is not None]
    return {
        "dte_distribution": {str(k): sum(1 for r in acc if r["dte"] == k) for k in sorted({r["dte"] for r in acc
                                                                                         if r["dte"] is not None})},
        "contracts": {c: sum(1 for r in acc if r["contract"] == c) for c in sorted({r["contract"] for r in acc
                                                                                    if r["contract"]})},
        "premium_atr": _summary(r["premium_atr"] for r in ok),
        "premium_atr_pct": _summary(r["premium_atr"] / r["entry"] * 100 for r in ok),
        "stop_distance": _summary(r["entry"] - r["stop_loss"] for r in ok),
        "stop_distance_pct": _summary((r["entry"] - r["stop_loss"]) / r["entry"] * 100 for r in ok),
        "target_distance": _summary(r["target"] - r["entry"] for r in ok),
        "target_distance_pct": _summary((r["target"] - r["entry"]) / r["entry"] * 100 for r in ok),
        "unavailable_reasons": _count(r["data_reason"] or r.get("exit_reason") for r in acc
                                      if r.get("outcome") in (UNAVAILABLE, PENDING)),
    }


def _count(values: Iterable) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


# ----------------------------------------------------- repeated-signal study
def annotate_sequence(rows: Sequence[dict]) -> list[dict]:
    """Ordinal of each signal within its direction and day, previous-signal relation and gap (all causal)."""
    out, day, count, prev, prev_same = [], None, {}, None, {}
    for r in sorted(rows, key=lambda x: x["signal_time"]):
        ts = _ist(r["signal_time"])
        if r["session_date"] != day:
            day, count, prev, prev_same = r["session_date"], {}, None, {}
        count[r["direction"]] = count.get(r["direction"], 0) + 1
        n = count[r["direction"]]
        gap = None if r["direction"] not in prev_same else (ts - prev_same[r["direction"]]).total_seconds() / 60
        out.append({**r, "same_direction_number": n,
                    "ordinal": "FIRST" if n == 1 else "SECOND" if n == 2 else "THIRD_PLUS",
                    "previous_relation": None if prev is None else ("SAME_DIRECTION" if prev == r["direction"]
                                                                    else "OPPOSITE_DIRECTION"),
                    "minutes_since_same_direction": gap})
        prev, prev_same[r["direction"]] = r["direction"], ts
    return out


def repeated_signal_analysis(rows: Sequence[dict]) -> dict[str, Any]:
    seq = annotate_sequence([r for r in rows if r["decision"] == ACCEPTED])
    stats = lambda sub: direction_stats(sub)
    within = {}
    for w in GAP_WINDOWS:
        sub = [r for r in seq if r["minutes_since_same_direction"] is not None
               and r["minutes_since_same_direction"] <= w]
        within[f"<= {w} min"] = stats(sub)
    return {
        "by_ordinal": {k: stats([r for r in seq if r["ordinal"] == k]) for k in ("FIRST", "SECOND", "THIRD_PLUS")},
        "previous_signal_same_direction": stats([r for r in seq if r["previous_relation"] == "SAME_DIRECTION"]),
        "previous_signal_opposite_direction": stats([r for r in seq if r["previous_relation"] ==
                                                      "OPPOSITE_DIRECTION"]),
        "first_of_day": stats([r for r in seq if r["previous_relation"] is None]),
        "within_minutes_of_previous_same_direction": within,
    }


# ------------------------------------------------------------- splits, status
def split_dates(spec: dict[str, tuple[str, str]] = DEFAULT_SPLITS) -> dict[str, tuple[date, date]]:
    out = {k: (date.fromisoformat(a), date.fromisoformat(b)) for k, (a, b) in spec.items()}
    ordered = [out[k] for k in SPLITS]
    for (a1, b1), (a2, _) in zip(ordered, ordered[1:]):
        if not (a1 <= b1 < a2):
            raise ValueError("Splits must be chronological and disjoint")
    return out


def split_of(day: str | date, splits: dict[str, tuple[date, date]]) -> str | None:
    d = date.fromisoformat(day) if isinstance(day, str) else day
    return next((k for k, (a, b) in splits.items() if a <= d <= b), None)


@dataclass(frozen=True)
class Criteria:
    robust_wr: float = 0.55
    promising_oos_wr: float = 0.50
    min_train: int = 20
    min_validation: int = 8
    min_oos: int = 8
    overfit_gap: float = 0.10
    current_day_gain: float = 0.10
    min_current: int = 3


def classify(m: dict[str, dict], current: dict | None, baseline_current_wr: float | None, lookahead_ok: bool,
             c: Criteria = Criteria()) -> tuple[str, list[str]]:
    if not lookahead_ok:
        return "LOOKAHEAD_FAIL", ["mutation test changed point-in-time decisions"]
    wr = {k: m[k]["win_rate"] or 0.0 for k in SPLITS}
    n = {k: m[k]["evaluated"] for k in SPLITS}
    pf = m["OOS"]["profit_factor"] or 0.0
    enough = n["TRAIN"] >= c.min_train and n["VALIDATION"] >= c.min_validation and n["OOS"] >= c.min_oos
    notes = [f"{k} {wr[k]:.1%} (n={n[k]})" for k in SPLITS] + [f"OOS PF {_fmt_pf(m['OOS']['profit_factor'])}"]
    if not enough:
        notes.append("insufficient trades")
    if enough and all(wr[k] >= c.robust_wr for k in SPLITS) and pf > 1:
        return "ROBUST", notes
    if wr["TRAIN"] >= c.robust_wr and min(wr["VALIDATION"], wr["OOS"]) <= wr["TRAIN"] - c.overfit_gap \
            and min(wr["VALIDATION"], wr["OOS"]) < c.robust_wr:
        return "OVERFIT", notes
    if wr["OOS"] >= c.promising_oos_wr and pf > 1:
        return "PROMISING", notes
    if current and current["win_rate"] is not None and baseline_current_wr is not None \
            and current["evaluated"] >= c.min_current and current["win_rate"] >= baseline_current_wr + c.current_day_gain:
        return "CURRENT_DAY_ONLY", notes
    return "NOT_ROBUST", notes


def improves_on_train(train: dict, base_train: dict, min_trades: int) -> bool:
    """Evidence rule for Phase 5 (TRAIN only): higher win rate AND higher profit factor, enough trades."""
    pf, bpf = train["profit_factor"] or 0.0, base_train["profit_factor"] or 0.0
    return train["evaluated"] >= min_trades and (train["win_rate"] or 0) > (base_train["win_rate"] or 0) and pf > bpf


def combined_candidates(single: Sequence[Variant], train: dict[str, dict], c: Criteria = Criteria(),
                        top: dict[str, int] | None = None, cap: int = 15) -> tuple[list[Variant], dict[str, list[str]]]:
    """Combine only TRAIN-improving variants: top-N per group by TRAIN WR then PF; never reads VALIDATION/OOS."""
    top = top or {"REPEAT": 2, "ENTRY": 2, "EXIT": 3}
    base = train[BASELINE.name]
    chosen: dict[str, list[Variant]] = {}
    for group, k in top.items():
        good = [v for v in single if v.group == group and improves_on_train(train[v.name], base, c.min_train)]
        good.sort(key=lambda v: (-(train[v.name]["win_rate"] or 0), -min(train[v.name]["profit_factor"] or 0, 1e6),
                                 v.name))
        chosen[group] = good[:k]
    reps = chosen["REPEAT"] or [BASELINE]
    ents = chosen["ENTRY"] or [BASELINE]
    exits = chosen["EXIT"] or [BASELINE]
    out: list[Variant] = []
    for r in reps:
        for e in ents:
            for x in exits:
                changed = sum([r is not BASELINE, e is not BASELINE, x is not BASELINE])
                if changed < 2:
                    continue
                v = Variant(f"CANDIDATE_{len(out) + 1:02d}", "COMBINED", repeat=r.repeat, cooldown_min=r.cooldown_min,
                            max_break_pct=e.max_break_pct, max_break_atr=e.max_break_atr, stop_atr=x.stop_atr,
                            target_r=x.target_r)
                out.append(v)
    return out[:cap], {g: [v.name for v in vs] for g, vs in chosen.items()}


# ---------------------------------------------------------------- look-ahead
def mutate_after(frame: pd.DataFrame, cut: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Scale every candle strictly after ``cut`` by a random factor (0.85-1.15); earlier candles untouched."""
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"])
    ts = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    mask = (ts > cut).to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.85, 1.15, int(mask.sum()))
    for col in ("open", "high", "low", "close"):
        out.loc[mask, col] = out.loc[mask, col].to_numpy() * k
    if "volume" in out:
        out.loc[mask, "volume"] = out.loc[mask, "volume"].to_numpy() * rng.uniform(0.2, 5.0, int(mask.sum()))
    return out


def decision_fingerprint(contexts: Sequence[dict], features: pd.DataFrame, variant: Variant,
                         upto: pd.Timestamp, day: date) -> list[tuple]:
    """Point-in-time decisions for one day's signals at or before ``upto``."""
    skips = repeat_filter(contexts, features, variant)
    out = []
    for ctx, skip in zip(contexts, skips):
        if ctx["session_date"] != day or ctx["signal_time"] > upto:
            continue
        skip = skip or entry_filter(ctx, variant)
        stop_dist = None if ctx["premium_atr"] is None else round(variant.stop_atr * ctx["premium_atr"], 6)
        out.append((ctx["signal_time"].isoformat(), ctx["direction"], skip, ctx["instrument_key"],
                    None if ctx["break_atr"] is None else round(ctx["break_atr"], 9),
                    None if ctx["premium_atr"] is None else round(ctx["premium_atr"], 9), stop_dist,
                    ctx["entry_time"] is None or ctx["entry_time"] > ctx["signal_time"]))
    return out


def mutation_test(pipeline: Callable[[pd.DataFrame, Callable, set], tuple[list[dict], pd.DataFrame]],
                  spot: pd.DataFrame, premium_for: Callable, variants: Sequence[Variant],
                  cuts: Sequence[pd.Timestamp], seed: int = 11) -> dict[str, dict]:
    """For each cut, mutate all underlying and premium candles after it and re-run the pipeline.

    ``pipeline(spot, premium_for, days)`` returns (contexts, features). A variant fails if any decision for a
    signal at or before the cut changes, or if any entry is not strictly after its signal.
    """
    result = {v.name: {"cuts": 0, "failures": []} for v in variants}
    for i, cut in enumerate(cuts):
        cut = _ist(cut)
        day = cut.date()
        base_ctx, base_feat = pipeline(spot, premium_for, {day})
        mut_spot = mutate_after(spot, cut, seed + i)
        mut_ctx, mut_feat = pipeline(mut_spot, lambda c, _cut=cut, _i=i: _mut_prem(premium_for(c), _cut, seed + _i),
                                     {day})
        for v in variants:
            a = decision_fingerprint(base_ctx, base_feat, v, cut, day)
            b = decision_fingerprint(mut_ctx, mut_feat, v, cut, day)
            result[v.name]["cuts"] += 1
            if a != b or not all(x[-1] for x in a):
                result[v.name]["failures"].append(cut.isoformat())
    for r in result.values():
        r["passed"] = not r["failures"]
    return result


def _mut_prem(frame, cut, seed):
    return frame if frame is None or isinstance(frame, str) else mutate_after(frame, cut, seed)


def _ist_series(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values)
    return ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)


# ---------------------------------------------------------------- reporting
def _fmt_pf(v) -> str:
    return "n/a" if v is None else ("inf" if v == math.inf else f"{v:.2f}")


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _jsonable(v):
    if isinstance(v, float):
        return "inf" if v == math.inf else (None if math.isnan(v) else v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (date, datetime, pd.Timestamp)):
        return v.isoformat()
    if isinstance(v, np.generic):
        return _jsonable(v.item())
    return v


CSV_SPLITS = ("TRAIN", "VALIDATION", "OOS", "CURRENT_DAY")


def candidate_csv_rows(results: Sequence[dict]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        row = {"variant": r["name"], "group": r["group"], "configuration": " | ".join(r["configuration"]),
               "status": r["status"], "lookahead_passed": r["lookahead_passed"]}
        for s in CSV_SPLITS:
            m = r["metrics"].get(s)
            p = s.lower()
            label = COMPARABLE if s == "CURRENT_DAY" else NON_COMPARABLE
            row[f"{p}_label"] = label
            for k in ("s2_signals", "accepted_signals", "skipped_signals", "evaluated", "wins", "losses", "ambiguous",
                      "pending_or_unavailable", "win_rate", "profit_factor", "net_pnl", "max_drawdown", "average_r"):
                row[f"{p}_{k}"] = None if m is None else _jsonable(m[k])
            for d in ("BUY", "SELL"):
                ds = None if m is None else m[d]
                for k in ("evaluated", "wins", "losses", "accuracy", "profit_factor"):
                    row[f"{p}_{d.lower()}_{k}"] = None if ds is None else _jsonable(ds[k])
        out.append(row)
    return out


def write_reports(report: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "s2_1dte_improvement_baseline.json",
             "csv": out_dir / "s2_1dte_improvement_candidates.csv",
             "md": out_dir / "s2_1dte_improvement_report.md"}
    paths["json"].write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    rows = candidate_csv_rows(report["results"])
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    return paths


def final_table_row(r: dict) -> str:
    m = r["metrics"]
    cur = m.get("CURRENT_DAY") or {}
    hist = m["HISTORICAL_ALL"]
    return (f"| {r['name']} | {_pct(m['TRAIN']['win_rate'])} ({m['TRAIN']['evaluated']}) | "
            f"{_pct(m['VALIDATION']['win_rate'])} ({m['VALIDATION']['evaluated']}) | "
            f"{_pct(m['OOS']['win_rate'])} ({m['OOS']['evaluated']}) | "
            f"{_pct(cur.get('win_rate'))} ({cur.get('evaluated', 'n/a')}) | {_fmt_pf(m['OOS']['profit_factor'])} | "
            f"{hist['net_pnl']:,.0f} | {hist['max_drawdown']:,.0f} | {hist['evaluated']} | {r['status']} |")


TABLE_HEAD = ["| Variant | Train WR (n) | Validation WR (n) | OOS WR (n) | Current 1-DTE WR (n) | OOS PF | "
              "Net P&L (hist) | Max DD (hist) | Trades (hist) | Status |", "|---|---|---|---|---|---|---:|---:|---:|---|"]


def _metric_block(name: str, m: dict) -> list[str]:
    avg_r = "n/a" if m["average_r"] is None else f"{m['average_r']:.2f}"
    return [f"| {name} | {m['s2_signals']} | {m['accepted_signals']} | {m['evaluated']} | {m['wins']} | "
            f"{m['losses']} | {m['ambiguous']} | {m['pending_or_unavailable']} | {_pct(m['win_rate'])} | "
            f"{_fmt_pf(m['profit_factor'])} | {m['net_pnl']:,.0f} | {m['max_drawdown']:,.0f} | {avg_r} | "
            f"{_pct(m['BUY']['accuracy'])} ({m['BUY']['evaluated']}) | {_pct(m['SELL']['accuracy'])} "
            f"({m['SELL']['evaluated']}) | {m['accepted_per_day']} |"]


def _dir_line(name: str, m: dict) -> str:
    return (f"| {name} | " + " | ".join(
        f"{m[d]['evaluated']} | {m[d]['wins']}/{m[d]['losses']} | {_pct(m[d]['accuracy'])} | "
        f"{_fmt_pf(m[d]['profit_factor'])}" for d in ("BUY", "SELL")) + " |")


def _stats_table(title: str, groups: dict[str, dict]) -> list[str]:
    md = [f"| {title} | Count | Wins/Losses | Win rate | PF | P&L |", "|---|---:|---|---|---|---:|"]
    for k, s in groups.items():
        md.append(f"| {k} | {s['evaluated']} | {s['wins']}/{s['losses']} | {_pct(s['accuracy'])} | "
                  f"{_fmt_pf(s['profit_factor'])} | {s['net_pnl']:,.0f} |")
    return md


def render_markdown(rep: dict) -> str:
    res = {r["name"]: r for r in rep["results"]}
    base = res[BASELINE.name]
    sd = rep["splits"]
    summ = rep["summary"]
    md = ["# S2 1-DTE improvement research", "",
          f"Generated {rep['generated_at']}. {rep['mode']}.", "",
          "## 1. Executive summary", ""]
    md += [f"- {line}" for line in summ["lines"]]
    md += ["", "## 2. Baseline (current corrected S2, unchanged)", "",
           f"Historical splits are **{NON_COMPARABLE}** (option DTE {rep['historical_dte']}). The current day is "
           f"**{COMPARABLE}**.", "",
           "| Sample | S2 signals | Accepted | Evaluated | Wins | Losses | Ambiguous | Pending/unavailable | "
           "Win rate | PF | Net P&L | Max DD | Avg R | BUY acc (n) | SELL acc (n) | Signals/day |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---|"]
    for name in ("TRAIN", "VALIDATION", "OOS", "HISTORICAL_ALL", "CURRENT_DAY"):
        if base["metrics"].get(name):
            md += _metric_block(name, base["metrics"][name])
    bp = rep["baseline_profile"]
    md += ["", "Current-day 1-DTE profile:", ""]
    for k in ("dte_distribution", "contracts", "premium_atr", "premium_atr_pct", "stop_distance", "stop_distance_pct",
              "target_distance", "target_distance_pct", "unavailable_reasons"):
        md.append(f"- {k}: {bp['current_day'].get(k)}")
    md += ["", "Historical profile:", ""]
    for k in ("dte_distribution", "premium_atr_pct", "stop_distance_pct", "target_distance_pct", "unavailable_reasons"):
        md.append(f"- {k}: {bp['historical'].get(k)}")
    md += ["", "Baseline time of day (current day / historical):", ""]
    md += _stats_table("Bucket (current day)", base["metrics"]["CURRENT_DAY"]["time_of_day"]) + [""]
    md += _stats_table("Bucket (historical)", base["metrics"]["HISTORICAL_ALL"]["time_of_day"])
    md += ["", "## 3. Repeated signal analysis (baseline S2)", ""]
    for sample in ("current_day", "historical"):
        ra = rep["repeated_analysis"][sample]
        md += [f"### {sample.replace('_', ' ').title()}", ""]
        md += _stats_table("Ordinal", ra["by_ordinal"]) + [""]
        md += _stats_table("Previous signal", {"same direction": ra["previous_signal_same_direction"],
                                               "opposite direction": ra["previous_signal_opposite_direction"],
                                               "first of day": ra["first_of_day"]}) + [""]
        md += _stats_table("Gap to previous same-direction signal", ra["within_minutes_of_previous_same_direction"])
        md.append("")
    for num, group, title in ((None, "REPEAT", "Repeated-signal filters (Phase 2)"),
                              ("4", "ENTRY", "Entry filter analysis (Phase 3)"),
                              ("5", "EXIT", "Exit analysis (Phase 4)")):
        if group == "REPEAT":
            md += ["### Repeated-signal filter variants", ""]
        else:
            md += [f"## {num}. {title}", ""]
        md += TABLE_HEAD + [final_table_row(base)] + [final_table_row(r) for r in rep["results"] if r["group"] == group]
        md += ["", "Skipped signals (historical / current): " + "; ".join(
            f"{r['name']} {r['metrics']['HISTORICAL_ALL']['skipped_signals']}/"
            f"{(r['metrics'].get('CURRENT_DAY') or {}).get('skipped_signals', 'n/a')}"
            for r in rep["results"] if r["group"] == group), ""]
    md += ["## 6. Combined candidates (Phase 5)", "",
           "Rule (fixed before running): a single-change variant qualifies only if its TRAIN win rate AND TRAIN PF "
           f"beat the baseline's TRAIN values with >= {rep['criteria']['min_train']} TRAIN trades. Top 2 repeat, "
           "top 2 entry and top 3 exit qualifiers (by TRAIN WR, then PF) are crossed; combinations changing fewer "
           "than two dimensions are dropped; cap 15. VALIDATION/OOS/current day are not read.", "",
           f"Qualifiers: {rep['phase5_qualifiers']}", ""]
    combos = [r for r in rep["results"] if r["group"] == "COMBINED"]
    if combos:
        for r in combos:
            md.append(f"- **{r['name']}**: " + "; ".join(r["configuration"]))
        md += [""] + TABLE_HEAD + [final_table_row(r) for r in combos]
    else:
        md.append("No combined candidates: fewer than two dimensions had TRAIN-qualifying variants.")
    md += ["", "## 7. Train / validation / OOS (all variants, ranked)", "",
           f"TRAIN {sd['TRAIN'][0]}..{sd['TRAIN'][1]}, VALIDATION {sd['VALIDATION'][0]}..{sd['VALIDATION'][1]}, OOS "
           f"{sd['OOS'][0]}..{sd['OOS'][1]} - all {NON_COMPARABLE}. Ranking: status, then VALIDATION WR, then TRAIN "
           "WR (OOS is not used to rank).", ""]
    md += TABLE_HEAD + [final_table_row(res[n]) for n in rep["ranking"]]
    md += ["", "### Direction analysis (Phase 10)", "",
           "| Variant | Split | BUY n | BUY W/L | BUY acc | BUY PF | SELL n | SELL W/L | SELL acc | SELL PF |",
           "|---|---|---:|---|---|---|---:|---|---|---|"]
    for n in rep["ranking"]:
        for s in ("TRAIN", "VALIDATION", "OOS", "CURRENT_DAY"):
            m = res[n]["metrics"].get(s)
            if m:
                md.append(_dir_line(f"{n} | {s}", m))
    md += ["", f"BUY removal: {summ['buy_removal']}", "",
           f"## 8. Current 1-DTE result ({COMPARABLE})", ""]
    md += ["| Variant | Accepted | Evaluated | W/L | Win rate | PF | Net P&L | Max DD |", "|---|---:|---:|---|---|---|---:|---:|"]
    for n in rep["ranking"]:
        m = res[n]["metrics"]["CURRENT_DAY"]
        md.append(f"| {n} | {m['accepted_signals']} | {m['evaluated']} | {m['wins']}/{m['losses'] + m['ambiguous']} | "
                  f"{_pct(m['win_rate'])} | {_fmt_pf(m['profit_factor'])} | {m['net_pnl']:,.0f} | "
                  f"{m['max_drawdown']:,.0f} |")
    la = rep["lookahead"]
    md += ["", "## 9. Look-ahead test", "",
           f"- Current-day S2 point-in-time signals equal the full-day run: {rep['point_in_time_matches_full_run']}.",
           f"- Mutation test: every underlying and premium candle after each cut was randomly rescaled and the full "
           f"pipeline re-run ({la['cuts']} cuts: {la['cut_summary']}). Compared for every signal at or before the "
           "cut: accept/skip decision, contract, breakout distance in ATR, premium ATR and stop distance; entry must "
           "be strictly after the signal.",
           f"- Result: {la['passed']} of {la['variants']} variants passed; failed: {la['failed'] or 'none'}.", "",
           "## 10. Best candidate", ""]
    best = summ["best"]
    if best:
        md += [f"**{best['name']}** - {best['status']}", ""] + [f"- {x}" for x in best["configuration"]]
        md += ["", TABLE_HEAD[0], TABLE_HEAD[1], final_table_row(res[best['name']])]
    md += ["", "## 11. Improvement over baseline", ""] + [f"- {x}" for x in summ["improvement_lines"]]
    md += ["", "## 12. Risks and limitations", ""] + [f"- {x}" for x in rep["limitations"]]
    md += ["", "## 13. Final recommendation", "", f"**{summ['recommendation']}**", ""]
    md += ["```", summ["final_block"], "```", ""]
    return "\n".join(md) + "\n"

# ------------------------------------------------------------------ the study
LIMITATIONS = [
    "Historical option outcomes use the 2026-10-06 weekly at 5-33 DTE (NON-COMPARABLE_1DTE). Expired weekly "
    "(including 1-DTE) candles are unavailable on this Upstox plan, so no historical result measures 1-DTE behaviour.",
    "Only one comparable 1-DTE session (2026-10-05) exists; its 40 signals are highly correlated (same contracts, "
    "shared exit bars), so the effective sample is far smaller than 40.",
    "Historical sample is 22 sessions in one month of mostly falling markets; Wilson 95% intervals on split win "
    "rates are wide.",
    "Signal-level evaluation: every accepted signal is evaluated independently with 1 lot and overlapping positions "
    "are allowed (as in the live tracker); costs and slippage are excluded.",
    "Ambiguous candles (stop and target inside one 5-minute candle) are counted as losses in the win rate.",
    "Lower target R mechanically raises win rate; exit variants are therefore judged on win rate AND profit factor.",
]


def _pipeline_factory(signal_fn: Callable[[pd.DataFrame], list], resolve: Callable, params: LevelParams,
                      range_start: str, range_end: str):
    def pipeline(spot: pd.DataFrame, premium_for: Callable, days: set | None = None):
        feats = underlying_features(spot, range_start, range_end, params.atr_period)
        sigs = [s for s in signal_fn(spot) if days is None or _ist(s.timestamp).date() in days]
        return build_contexts(sigs, feats, resolve, premium_for, params), feats
    return pipeline


def _evaluate_all(variant: Variant, hist: tuple, cur: tuple | None, splits: dict, square_off: str, cache: dict,
                  current_day: date | None) -> tuple[dict, list[dict]]:
    rows_h = run_variant(hist[0], hist[1], variant, square_off=square_off, cache=cache)
    for r in rows_h:
        r["split"], r["comparability"] = split_of(r["session_date"], splits), NON_COMPARABLE
    metrics = {s: compute_metrics([r for r in rows_h if r["split"] == s]) for s in SPLITS}
    metrics["HISTORICAL_ALL"] = compute_metrics([r for r in rows_h if r["split"] in SPLITS])
    rows_c: list[dict] = []
    if cur is not None:
        rows_c = run_variant(cur[0], cur[1], variant, square_off=square_off, cache=cache)
        for r in rows_c:
            r["split"], r["comparability"] = "CURRENT_DAY", COMPARABLE
        metrics["CURRENT_DAY"] = compute_metrics(rows_c)
    return metrics, rows_h + rows_c


def run_research(*, hist_spot: pd.DataFrame, signal_fn: Callable[[pd.DataFrame], list], resolve: Callable,
                 hist_premium_for: Callable, params: LevelParams, square_off: str = "15:15",
                 range_start: str = "09:15", range_end: str = "09:30", current_spot: pd.DataFrame | None = None,
                 current_premium_for: Callable | None = None, current_day: date | None = None,
                 splits: dict[str, tuple[str, str]] = DEFAULT_SPLITS, criteria: Criteria = Criteria(),
                 hist_cut_time: str = "12:00", current_cut_every_min: int = 15,
                 point_in_time_matches_full_run: bool | None = None) -> dict[str, Any]:
    sp = split_dates(splits)
    hist_days = {d for d in pd.to_datetime(hist_spot["timestamp"]).dt.date if split_of(d, sp)}
    pipeline = _pipeline_factory(signal_fn, resolve, params, range_start, range_end)
    hist = pipeline(hist_spot, hist_premium_for, hist_days)
    cur = pipeline(current_spot, current_premium_for, {current_day}) if current_spot is not None else None
    cache: dict = {}

    single = predeclared_variants()
    results, all_rows = [], []
    for v in single:
        m, rows = _evaluate_all(v, hist, cur, sp, square_off, cache, current_day)
        results.append({"variant": v, "metrics": m})
        all_rows += rows
    combos, qualifiers = combined_candidates(single, {r["variant"].name: r["metrics"]["TRAIN"] for r in results},
                                             criteria)
    for v in combos:
        m, rows = _evaluate_all(v, hist, cur, sp, square_off, cache, current_day)
        results.append({"variant": v, "metrics": m})
        all_rows += rows
    variants = single + combos

    hist_cuts = [pd.Timestamp(datetime.combine(d, time.fromisoformat(hist_cut_time)), tz=IST) for d in sorted(hist_days)]
    la_hist = mutation_test(pipeline, hist_spot, hist_premium_for, variants, hist_cuts)
    la_cur = {v.name: {"cuts": 0, "failures": [], "passed": True} for v in variants}
    cur_cuts: list = []
    if cur is not None:
        start = datetime.combine(current_day, time(9, 35))
        cur_cuts = [pd.Timestamp(start, tz=IST) + pd.Timedelta(minutes=current_cut_every_min * i)
                    for i in range(0, 300 // current_cut_every_min + 1)
                    if (start + pd.Timedelta(minutes=current_cut_every_min * i)).time() <= time(14, 30)]
        la_cur = mutation_test(pipeline, current_spot, current_premium_for, variants, cur_cuts)

    base_cur_wr = results[0]["metrics"].get("CURRENT_DAY", {}).get("win_rate")
    out = []
    for r in results:
        v = r["variant"]
        ok = la_hist[v.name]["passed"] and la_cur[v.name]["passed"]
        status, notes = classify(r["metrics"], r["metrics"].get("CURRENT_DAY"), base_cur_wr, ok, criteria)
        if v.name == BASELINE.name and status != "LOOKAHEAD_FAIL":
            status = "BASELINE"
        out.append({"name": v.name, "group": v.group, "configuration": v.describe(), "variant": asdict(v),
                    "metrics": r["metrics"], "status": status, "status_notes": notes, "lookahead_passed": ok,
                    "lookahead": {"historical": la_hist[v.name], "current_day": la_cur[v.name]}})

    order = {s: i for i, s in enumerate(STATUS_ORDER + ("BASELINE",))}
    ranking = [r["name"] for r in sorted(out, key=lambda r: (
        order[r["status"]], -(r["metrics"]["VALIDATION"]["win_rate"] or 0), -(r["metrics"]["TRAIN"]["win_rate"] or 0),
        r["name"]))]
    base = out[0]
    hist_dte = sorted({r["dte"] for r in all_rows if r["comparability"] == NON_COMPARABLE and r["dte"] is not None})
    report = {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "mode": "Research only - TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, no orders, S2 unchanged",
        "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in sp.items()},
        "labels": {"historical": NON_COMPARABLE, "current_day": COMPARABLE},
        "current_day": None if current_day is None else current_day.isoformat(),
        "historical_dte": hist_dte,
        "criteria": asdict(criteria),
        "evaluation_method": ("Every accepted S2 signal evaluated independently: option entry at next candle open, "
                              "premium ATR from candles <= signal candle, stop = k x premium ATR, target = R x risk, "
                              "first touch on real premium candles, square-off at " + square_off + ", 1 lot (65), "
                              "costs excluded, ambiguous counted as loss in win rate."),
        "point_in_time_matches_full_run": point_in_time_matches_full_run,
        "predeclared_variants": [{"name": v.name, "group": v.group, "configuration": v.describe()} for v in single],
        "phase5_qualifiers": qualifiers,
        "results": out,
        "ranking": ranking,
        "baseline_profile": {
            "historical": baseline_profile([r for r in all_rows if r["variant"] == BASELINE.name
                                            and r["comparability"] == NON_COMPARABLE and r["split"]]),
            "current_day": baseline_profile([r for r in all_rows if r["variant"] == BASELINE.name
                                             and r["comparability"] == COMPARABLE]),
        },
        "repeated_analysis": {
            "historical": repeated_signal_analysis([r for r in all_rows if r["variant"] == BASELINE.name
                                                    and r["comparability"] == NON_COMPARABLE and r["split"]]),
            "current_day": repeated_signal_analysis([r for r in all_rows if r["variant"] == BASELINE.name
                                                     and r["comparability"] == COMPARABLE]),
        },
        "lookahead": {"cuts": len(hist_cuts) + len(cur_cuts),
                      "cut_summary": f"{len(hist_cuts)} historical at {hist_cut_time}, {len(cur_cuts)} current-day "
                                     f"every {current_cut_every_min} min",
                      "variants": len(out), "passed": sum(1 for r in out if r["lookahead_passed"]),
                      "failed": [r["name"] for r in out if not r["lookahead_passed"]]},
        "limitations": LIMITATIONS,
        "baseline_signals_current_day": [r for r in all_rows if r["variant"] == BASELINE.name
                                         and r["comparability"] == COMPARABLE],
    }
    report["summary"] = summarize(report, base)
    return report


def _buy_removal(base: dict) -> str:
    m = base["metrics"]
    weak = [s for s in SPLITS if m[s]["BUY"]["evaluated"] and (m[s]["BUY"]["profit_factor"] or 0) < 1
            and (m[s]["BUY"]["accuracy"] or 0) < (m[s]["SELL"]["accuracy"] or 0)]
    if len(weak) == len(SPLITS):
        return ("BUY underperformed SELL with PF < 1 in TRAIN, VALIDATION and OOS; removing BUY is supported by "
                "historical (non-comparable) evidence only and is not applied here.")
    return (f"Not recommended: BUY underperformed with PF < 1 only in {weak or 'no'} split(s), which does not survive "
            "validation/OOS.")


def summarize(rep: dict, base: dict) -> dict[str, Any]:
    res = {r["name"]: r for r in rep["results"]}
    bm = base["metrics"]
    base_cur = (bm.get("CURRENT_DAY") or {}).get("win_rate")
    robust = [n for n in rep["ranking"] if res[n]["status"] == "ROBUST"]
    candidates = [n for n in rep["ranking"] if n != BASELINE.name]
    best_name = robust[0] if robust else (candidates[0] if candidates else None)
    best = res.get(best_name)
    lines, imp = [], []
    pp = lambda a, b: "n/a" if a is None or b is None else f"{(a - b) * 100:+.1f} percentage points"
    if best:
        bc = (best["metrics"].get("CURRENT_DAY") or {}).get("win_rate")
        imp.append(f"Current-day 1-DTE ({COMPARABLE}): baseline {_pct(base_cur)} -> {best_name} {_pct(bc)} "
                   f"({pp(bc, base_cur)}).")
        imp.append(f"OOS ({NON_COMPARABLE}): baseline {_pct(bm['OOS']['win_rate'])} -> "
                   f"{_pct(best['metrics']['OOS']['win_rate'])} ({pp(best['metrics']['OOS']['win_rate'], bm['OOS']['win_rate'])}).")
        imp.append(f"VALIDATION ({NON_COMPARABLE}): baseline {_pct(bm['VALIDATION']['win_rate'])} -> "
                   f"{_pct(best['metrics']['VALIDATION']['win_rate'])}.")
    if robust:
        rec = "PROMOTE CANDIDATE FOR FURTHER PAPER VALIDATION"
        imp.insert(0, f"Best validated candidate: {best_name} (ROBUST on {NON_COMPARABLE} history).")
    else:
        rec = "NO ROBUST IMPROVEMENT FOUND - KEEP S2"
        imp.insert(0, "No robust improvement found.")
    counts = _count(r["status"] for r in rep["results"])
    lines.append(f"Baseline S2 current-day 1-DTE win rate {_pct(base_cur)} "
                 f"({(bm.get('CURRENT_DAY') or {}).get('wins')}/{(bm.get('CURRENT_DAY') or {}).get('evaluated')}); "
                 f"historical ({NON_COMPARABLE}) TRAIN/VALIDATION/OOS {_pct(bm['TRAIN']['win_rate'])}/"
                 f"{_pct(bm['VALIDATION']['win_rate'])}/{_pct(bm['OOS']['win_rate'])}.")
    lines.append(f"{len(rep['results']) - 1} research variants evaluated (26 predeclared single changes + "
                 f"{sum(1 for r in rep['results'] if r['group'] == 'COMBINED')} TRAIN-qualified combinations). "
                 f"Status counts: {counts}.")
    lines.append(imp[0])
    if best:
        lines.append(f"Top-ranked candidate: {best_name} ({best['status']}): " + "; ".join(best["configuration"]))
    lines.append(f"Recommendation: {rec}.")
    bc_m = (best["metrics"].get("CURRENT_DAY") or {}) if best else {}
    block = "\n".join([
        "BASELINE:", _pct(base_cur), "",
        "BEST CANDIDATE:", f"{best_name}: current-day 1-DTE {_pct(bc_m.get('win_rate'))} "
                           f"({bc_m.get('wins')}/{bc_m.get('evaluated')})" if best else "none", "",
        "IMPROVEMENT:", pp(bc_m.get("win_rate"), base_cur) + " (current day, single session)" if best else "n/a", "",
        "VALIDATION:", (f"{_pct(best['metrics']['VALIDATION']['win_rate'])} (n={best['metrics']['VALIDATION']['evaluated']}, "
                        f"{NON_COMPARABLE})") if best else "n/a", "",
        "OOS:", (f"{_pct(best['metrics']['OOS']['win_rate'])} (n={best['metrics']['OOS']['evaluated']}, "
                 f"{NON_COMPARABLE})") if best else "n/a", "",
        "PF:", (f"OOS {_fmt_pf(best['metrics']['OOS']['profit_factor'])}; current day "
                f"{_fmt_pf(bc_m.get('profit_factor'))}") if best else "n/a", "",
        "STATUS:", best["status"] if best else "n/a", "",
        "RECOMMENDATION:", "- " + ("PROMOTE CANDIDATE FOR FURTHER PAPER VALIDATION" if robust
                                   else "NO ROBUST IMPROVEMENT FOUND (KEEP S2)"),
    ])
    return {"lines": lines, "improvement_lines": imp, "recommendation": rec, "buy_removal": _buy_removal(base),
            "best": None if not best else {"name": best_name, "status": best["status"],
                                           "configuration": best["configuration"]},
            "robust_candidates": robust, "final_block": block}
