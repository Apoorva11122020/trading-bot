"""S2 MONTHLY + REGIME DIAGNOSTIC LAB (PAPER | RESEARCH ONLY | NO ORDERS).

Explains WHEN the unchanged S2 baseline works and when it fails on the 6-month dataset. Nothing here creates S2
parameters, filters, thresholds, exits or weights: every bucket edge is a fixed, coarse, pre-declared value and the
outcome of a trade is only ever used for retrospective grouping, never to build a feature.

Causality: every feature is read from the 5m candle at the signal timestamp (values known at that candle's close) or
from option premium candles <= the signal candle. ``truncation_check`` recomputes the features with all later data
removed / mutated and requires identical values.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import accuracy_70_lab as acc
from src.backtest import s2_1dte_improvement as imp
from src.research import s2_baseline_drift_audit as dra

HEADING = "S2 MONTHLY DIAGNOSTIC"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS"
IST = imp.IST
DIRECTIONS = ("BUY", "SELL")
SCOPES = ("BUY", "SELL", "COMBINED")
MONTHS = (("2026-04", "APRIL"), ("2026-05", "MAY"), ("2026-06", "JUNE"), ("2026-07", "JULY"),
          ("2026-08", "AUGUST"), ("2026-09", "SEPTEMBER"))
TIME_WINDOWS = (("09:15", "10:00"), ("10:00", "11:00"), ("11:00", "12:00"), ("12:00", "13:00"),
                ("13:00", "14:00"), ("14:00", "14:30"))
TREND_REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE")
VOL_REGIMES = ("HIGH_VOLATILITY", "LOW_VOLATILITY", "NORMAL_VOLATILITY")
REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOLATILITY", "LOW_VOLATILITY", "NORMAL_VOLATILITY")
HIGH_VOL_PCTILE, LOW_VOL_PCTILE = 2 / 3, 1 / 3
UNAVAILABLE = "UNAVAILABLE"

# Coarse pre-declared bucket edges (never fitted to outcomes).
NUMERIC_BUCKETS: dict[str, tuple[float, ...]] = {
    "ema_sep_atr": (-1.0, 0.0, 1.0),          # (EMA20 - EMA50) / ATR14
    "ema20_slope_atr": (-0.5, 0.0, 0.5),      # EMA20 change over 3 bars / ATR14
    "vwap_dist_atr": (-1.0, 0.0, 1.0),        # (close - VWAP) / ATR14 (TWAP when the index has no volume)
    "rsi14": (40.0, 50.0, 60.0, 70.0),
    "atr14": (10.0, 15.0, 20.0, 30.0),        # NIFTY points
    "atr_pctile": (0.25, 0.5, 0.75),          # trailing 375-bar percentile of ATR14/close (current bar excluded)
    "underlying_vol_ratio": (0.8, 1.2, 2.0),
    "option_vol_ratio": (0.8, 1.2, 2.0),
    "break_atr": (0.25, 0.5, 1.0),            # breakout distance beyond the ORB level / ATR14
    "body_frac": (0.3, 0.6),                  # |close - open| / (high - low) of the signal candle
    "range_atr": (0.75, 1.25, 2.0),           # signal candle range / ATR14
    "premium_mom3_pct": (-5.0, 0.0, 5.0, 15.0),
    "premium_atr_pct": (5.0, 10.0, 15.0),     # premium ATR14 / premium close, %
}
DTE_BUCKETS = ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 10_000, ">7"))
CATEGORICAL = ("ema20_vs_ema50", "vwap_side", "time_window", "moneyness", "dte_bucket")
FEATURES = tuple(NUMERIC_BUCKETS) + CATEGORICAL
WIN_LOSS_FEATURES = ("rsi14", "atr14", "atr_pctile", "option_vol_ratio", "underlying_vol_ratio", "vwap_dist_atr",
                     "ema_sep_atr", "ema20_slope_atr", "break_atr", "body_frac", "range_atr", "premium_mom3_pct",
                     "premium_atr_pct", "dte")
OPTION_ANALYSES = (("moneyness", "moneyness"), ("dte", "dte_bucket"),
                   ("premium_volatility", "premium_atr_pct_bucket"), ("premium_momentum", "premium_mom3_pct_bucket"))

MIN_SAMPLE = 30           # below this a bucket is reported but never interpreted
MIN_MONTH_TRADES = 5      # a condition is "active" in a month / regime with at least this many trades
STABLE_MIN_TRADES = 60
STATUS_ORDER = ("STABLE", "PROMISING", "UNSTABLE", "WEAK_SAMPLE")


# ------------------------------------------------------------------ helpers
def _num(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _fmt(v: float) -> str:
    return f"{v:g}"


def bucket(v, edges: Sequence[float]) -> str:
    v = _num(v)
    if v is None:
        return UNAVAILABLE
    if v < edges[0]:
        return f"<{_fmt(edges[0])}"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"{_fmt(lo)}..{_fmt(hi)}"
    return f">={_fmt(edges[-1])}"


def dte_bucket(dte) -> str:
    d = _num(dte)
    if d is None:
        return UNAVAILABLE
    return next((lab for lo, hi, lab in DTE_BUCKETS if lo <= d <= hi), UNAVAILABLE)


def time_window(ts) -> str:
    hhmm = dra._ts(ts).strftime("%H:%M")
    return next((f"{a}-{b}" for a, b in TIME_WINDOWS if a <= hhmm < b), "OUTSIDE_WINDOWS")


def month_of(r: dict) -> str:
    return str(r["session_date"])[:7]


def perf(rows: Sequence[dict]) -> dict[str, Any]:
    """Exact S2 metric definitions (s2_1dte_improvement.compute_metrics). Ambiguous counts as a -1R loss."""
    if not rows:
        return {"trades": 0, "wins": 0, "losses": 0, "ambiguous": 0, "accuracy": None, "profit_factor": None,
                "net_pnl": 0.0, "avg_r": None, "max_drawdown": 0.0, "max_loss_streak": 0}
    m = imp.compute_metrics(list(rows))
    return {"trades": m["evaluated"], "wins": m["wins"], "losses": m["evaluated"] - m["wins"],
            "ambiguous": m["ambiguous"], "accuracy": m["win_rate"], "profit_factor": m["profit_factor"],
            "net_pnl": m["net_pnl"], "avg_r": m["average_r"], "max_drawdown": m["max_drawdown"],
            "max_loss_streak": m["max_consecutive_losses"]}


def scoped(rows: Sequence[dict]) -> dict[str, dict]:
    return {"BUY": perf([r for r in rows if r["direction"] == "BUY"]),
            "SELL": perf([r for r in rows if r["direction"] == "SELL"]), "COMBINED": perf(rows)}


def decided(rows: Sequence[dict]) -> list[dict]:
    return [r for r in rows if r.get("decision") == imp.ACCEPTED
            and r.get("outcome") in (imp.WIN, imp.LOSS, imp.AMBIGUOUS, imp.SCRATCH)]


def trade_pnl(r: dict) -> float:
    return imp._conservative(r)[0]


def sample_grade(n: int) -> str:
    return "A" if n >= 100 else "B" if n >= 60 else "C" if n >= MIN_SAMPLE else "D"


def _pct(v) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def _pf(v) -> str:
    return "N/A" if v is None else ("INF" if v == math.inf else f"{v:.2f}")


def line(m: dict) -> str:
    return (f"{_pct(m['accuracy'])} ({m['trades']} trades, {m['wins']} W / {m['losses']} L, PF {_pf(m['profit_factor'])}, "
            f"P&L Rs {m['net_pnl']:,.2f})")


# ------------------------------------------------------------------ causal features
def feature_bars(spot: pd.DataFrame) -> pd.DataFrame:
    """Per-bar values known at each completed 5m candle's close, indexed by IST timestamp."""
    return acc.family_bars(spot).set_index("timestamp")


def classify_regime(f: dict) -> tuple[str, str]:
    """Trend axis from close / EMA20 / EMA50 / EMA20 slope; volatility axis from the trailing ATR percentile."""
    c, e20, e50, sl, ap = (_num(f.get(k)) for k in ("close", "ema20", "ema50", "ema20_slope_atr", "atr_pctile"))
    if None in (c, e20, e50, sl):
        trend = UNAVAILABLE
    elif c > e20 > e50 and sl > 0:
        trend = "TREND_UP"
    elif c < e20 < e50 and sl < 0:
        trend = "TREND_DOWN"
    else:
        trend = "RANGE"
    if ap is None:
        vol = UNAVAILABLE
    else:
        vol = "HIGH_VOLATILITY" if ap >= HIGH_VOL_PCTILE else "LOW_VOLATILITY" if ap <= LOW_VOL_PCTILE \
            else "NORMAL_VOLATILITY"
    return trend, vol


def moneyness(option_type: str | None, strike, spot) -> str:
    s, k = _num(spot), _num(strike)
    if option_type not in ("CE", "PE") or s is None or k is None:
        return UNAVAILABLE
    d = k - s
    if abs(d) <= 12.5:
        return "ATM"
    itm = d < 0 if option_type == "CE" else d > 0
    return "SLIGHT_ITM" if itm else "SLIGHT_OTM"


def bar_values(bars: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float | None]:
    if ts not in bars.index:
        return {}
    b = bars.loc[ts]
    cols = ("close", "atr14", "ema20", "ema50", "rsi14", "ema20_slope_atr", "atr_pctile", "vwap", "twap",
            "vol_ratio", "body_frac", "range_atr", "intraday_move", "sign_changes")
    return {c: _num(b.get(c)) for c in cols}


def signal_features(r: dict, bars: pd.DataFrame, ctx: dict | None) -> dict[str, Any]:
    ts = dra._ts(r["signal_time"])
    b = bar_values(bars, ts)
    close, a = b.get("close"), b.get("atr14")
    ref, src = (b.get("vwap"), "VWAP") if b.get("vwap") is not None else (b.get("twap"), "TWAP_NO_INDEX_VOLUME")
    opt = acc.option_features(ctx) if ctx else None
    opt = opt or {}
    e20, e50 = b.get("ema20"), b.get("ema50")
    mom = _num(opt.get("premium_mom3"))
    f = {
        "close": close, "atr14": a, "ema20": e20, "ema50": e50, "rsi14": b.get("rsi14"),
        "ema20_slope_atr": b.get("ema20_slope_atr"), "atr_pctile": b.get("atr_pctile"),
        "ema_sep_atr": None if None in (e20, e50) or not a else (e20 - e50) / a,
        "ema20_vs_ema50": UNAVAILABLE if None in (e20, e50) else ("EMA20>EMA50" if e20 > e50 else "EMA20<=EMA50"),
        "vwap_ref": ref, "vwap_source": src,
        "vwap_dist_atr": None if ref is None or close is None or not a else (close - ref) / a,
        "vwap_side": UNAVAILABLE if ref is None or close is None else ("ABOVE" if close > ref else "BELOW_OR_AT"),
        "underlying_vol_ratio": b.get("vol_ratio"), "option_vol_ratio": _num(opt.get("premium_vol_ratio")),
        "break_atr": _num(r.get("break_atr")), "body_frac": b.get("body_frac"), "range_atr": b.get("range_atr"),
        "premium_mom3_pct": None if mom is None else mom * 100, "premium_atr_pct": _num(opt.get("premium_atr_pct")),
        "dte": _num(r.get("dte")), "time_window": time_window(ts),
        "moneyness": moneyness(r.get("option_type"), r.get("strike"), r.get("spot")),
    }
    f["dte_bucket"] = dte_bucket(f["dte"])
    for name, edges in NUMERIC_BUCKETS.items():
        f[f"{name}_bucket"] = bucket(f[name], edges)
    f["trend_regime"], f["vol_regime"] = classify_regime(f)
    return f


def annotate(rows: Sequence[dict], bars: pd.DataFrame, ctx_by_key: dict) -> list[dict]:
    """Rows (chronological) with causal features attached. Outcome fields are copied, never read here."""
    out = []
    for r in sorted(rows, key=lambda x: (dra._ts(x["signal_time"]), x["direction"])):
        k = dra.key_of(r)
        out.append({**r, "month": month_of(r), "features": signal_features(r, bars, ctx_by_key.get(k))})
    return out


def bucket_of(r: dict, feature: str) -> str:
    f = r["features"]
    return f.get(f"{feature}_bucket") if feature in NUMERIC_BUCKETS else f.get(feature, UNAVAILABLE)


# ------------------------------------------------------------------ leakage checks
FEATURE_CHECK_COLUMNS = ("close", "atr14", "ema20", "ema50", "rsi14", "ema20_slope_atr", "atr_pctile", "vwap_ref",
                         "body_frac", "range_atr", "option_vol_ratio", "premium_mom3_pct", "premium_atr_pct")


def truncation_check(spot: pd.DataFrame, rows: Sequence[dict], ctx_by_key: dict, sample: int = 20) -> dict[str, Any]:
    """Recompute features with every later underlying candle removed and every later premium candle tripled."""
    ann = [r for r in rows if r.get("instrument_key")]
    if not ann:
        return {"checked": 0, "mismatches": [], "passed": False}
    step = max(1, len(ann) // sample)
    picked = ann[::step][:sample]
    t_all = dra._ist_series(spot["timestamp"])
    bad = []
    for r in picked:
        ts = dra._ts(r["signal_time"])
        full = r["features"]
        trunc_bars = feature_bars(spot[(t_all <= ts).to_numpy()].reset_index(drop=True))
        ctx = ctx_by_key.get(dra.key_of(r))
        warped = None
        if ctx and ctx.get("premium") is not None and not isinstance(ctx["premium"], str):
            p = ctx["premium"].copy()
            later = (dra._ist_series(p["timestamp"]) > ts).to_numpy()
            for col in ("open", "high", "low", "close", "volume"):
                if col in p:
                    p[col] = p[col].astype(float)
                    p.loc[later, col] = p.loc[later, col] * 3.0
            warped = dict(ctx, premium=p)
        again = signal_features(r, trunc_bars, warped)
        for c in FEATURE_CHECK_COLUMNS:
            a, b = full.get(c), again.get(c)
            if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)):
                bad.append(f"{r['signal_time']}|{r['direction']}|{c}: {a} vs {b}")
        for c in ("trend_regime", "vol_regime", "time_window", "moneyness", "dte_bucket"):
            if full.get(c) != again.get(c):
                bad.append(f"{r['signal_time']}|{r['direction']}|{c}: {full.get(c)} vs {again.get(c)}")
    return {"checked": len(picked), "mismatches": bad[:20], "passed": not bad}


def monthly_consistency(full_rows: Sequence[dict], monthly: dict[str, list[dict]]) -> dict[str, Any]:
    """Each separate monthly replay must equal the full-period replay sliced to that month."""
    out = {}
    for m, rows in monthly.items():
        a = {dra.key_of(r): (r.get("outcome"), r.get("pnl")) for r in full_rows if month_of(r) == m}
        b = {dra.key_of(r): (r.get("outcome"), r.get("pnl")) for r in rows}
        out[m] = {"full_slice": len(a), "separate_run": len(b), "identical": a == b,
                  "wrong_month": sum(1 for r in rows if month_of(r) != m)}
    return {"months": out, "passed": all(v["identical"] and not v["wrong_month"] for v in out.values())}


def chronology_check(rows: Sequence[dict]) -> dict[str, Any]:
    ts = [dra._ts(r["signal_time"]) for r in rows]
    bad_entry = sum(1 for r in rows if r.get("entry_time")
                    and dra._ts(r["entry_time"]) != dra._ts(r["signal_time"]) + pd.Timedelta(minutes=5))
    wrong_day = sum(1 for r in rows if dra._ts(r["signal_time"]).date().isoformat() != str(r["session_date"])[:10])
    late_exit = sum(1 for r in rows if r.get("exit_time") and r.get("entry_time")
                    and dra._ts(r["exit_time"]) < dra._ts(r["entry_time"]))
    ordered = all(a <= b for a, b in zip(ts, ts[1:]))
    return {"ordered": ordered, "entry_not_next_bar": bad_entry, "session_date_mismatch": wrong_day,
            "exit_before_entry": late_exit, "passed": ordered and not bad_entry and not wrong_day and not late_exit}


# ------------------------------------------------------------------ phase tables
def monthly_table(monthly: dict[str, list[dict]]) -> list[dict]:
    out = []
    for m, label in MONTHS:
        for s, v in scoped(monthly.get(m, [])).items():
            out.append({"month": m, "label": label, "scope": s, **v})
    return out


def daily_table(rows: Sequence[dict]) -> list[dict]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[str(r["session_date"])[:10]].append(r)
    out = []
    for d in sorted(by_day):
        sc = scoped(by_day[d])
        out.append({"date": d, "month": d[:7],
                    **{f"{s.lower()}_{k}": sc[s][k] for s in SCOPES for k in ("trades", "accuracy", "net_pnl")}})
    return out


def daily_summary(days: Sequence[dict]) -> dict[str, Any]:
    traded = [d for d in days if d["combined_trades"]]
    pnl = [d["combined_net_pnl"] for d in traded]
    if not traded:
        return {"days": len(days), "traded_days": 0}
    by_pnl = sorted(traded, key=lambda d: d["combined_net_pnl"])
    gross = sum(p for p in pnl if p > 0)
    best = by_pnl[-1]
    return {"days": len(days), "traded_days": len(traded), "profitable_days": sum(p > 0 for p in pnl),
            "losing_days": sum(p < 0 for p in pnl), "flat_days": sum(p == 0 for p in pnl),
            "median_day_pnl": round(median(pnl), 2), "total_pnl": round(sum(pnl), 2),
            "best_days": [(d["date"], d["combined_net_pnl"]) for d in by_pnl[::-1][:5]],
            "worst_days": [(d["date"], d["combined_net_pnl"]) for d in by_pnl[:5]],
            "largest_day": best["date"], "largest_day_pnl": best["combined_net_pnl"],
            "largest_day_share_of_gross_profit": best["combined_net_pnl"] / gross if gross > 0 else None,
            "pnl_without_best_day": round(sum(pnl) - best["combined_net_pnl"], 2)}


def group_table(rows: Sequence[dict], key: Callable[[dict], str], labels: Sequence[str] | None = None,
                name: str = "group") -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    order = list(labels or []) + sorted(k for k in groups if k not in (labels or []))
    out = []
    for g in order:
        if g not in groups and labels is None:
            continue
        for s, v in scoped(groups.get(g, [])).items():
            out.append({name: g, "scope": s, **v, "sample_ok": v["trades"] >= MIN_SAMPLE})
    return out


def regime_table(rows: Sequence[dict]) -> list[dict]:
    out = []
    for reg in REGIMES + (UNAVAILABLE,):
        axis = "trend_regime" if reg in TREND_REGIMES or reg == UNAVAILABLE else "vol_regime"
        sub = [r for r in rows if r["features"][axis] == reg]
        if reg == UNAVAILABLE:
            sub = [r for r in rows if UNAVAILABLE in (r["features"]["trend_regime"], r["features"]["vol_regime"])]
            if not sub:
                continue
        for s, v in scoped(sub).items():
            out.append({"regime": reg, "axis": axis, "scope": s, **v, "sample_ok": v["trades"] >= MIN_SAMPLE})
    return out


def time_table(rows: Sequence[dict]) -> list[dict]:
    labels = [f"{a}-{b}" for a, b in TIME_WINDOWS]
    return group_table(rows, lambda r: r["features"]["time_window"], labels, "time_window")


def feature_table(rows: Sequence[dict]) -> list[dict]:
    out = []
    for feat in FEATURES:
        for g in group_table(rows, lambda r, f=feat: bucket_of(r, f), None, "bucket"):
            if g["scope"] == "COMBINED":
                continue
            out.append({"feature": feat, **g})
    return out


def option_table(rows: Sequence[dict]) -> list[dict]:
    out = []
    for analysis, col in OPTION_ANALYSES:
        for g in group_table(rows, lambda r, c=col: r["features"].get(c, UNAVAILABLE), None, "bucket"):
            if g["scope"] == "COMBINED":
                continue
            side = "BUY / CALL" if g["scope"] == "BUY" else "SELL / PUT"
            out.append({"analysis": analysis, "side": side, **g,
                        "interpretation": "OK" if g["sample_ok"] else f"REJECTED: fewer than {MIN_SAMPLE} trades"})
    return out


def win_loss_table(rows: Sequence[dict]) -> list[dict]:
    out = []
    for d in DIRECTIONS:
        dec = [r for r in decided(rows) if r["direction"] == d]
        wins = [r for r in dec if r["outcome"] == imp.WIN]
        losses = [r for r in dec if r["outcome"] != imp.WIN]
        for feat in WIN_LOSS_FEATURES:
            w = [v for v in (_num(r["features"].get(feat)) for r in wins) if v is not None]
            lo = [v for v in (_num(r["features"].get(feat)) for r in losses) if v is not None]
            wm = sum(w) / len(w) if w else None
            lm = sum(lo) / len(lo) if lo else None
            out.append({"direction": d, "feature": feat, "win_n": len(w), "loss_n": len(lo),
                        "win_mean": None if wm is None else round(wm, 4),
                        "loss_mean": None if lm is None else round(lm, 4),
                        "win_median": round(median(w), 4) if w else None,
                        "loss_median": round(median(lo), 4) if lo else None,
                        "mean_diff_win_minus_loss": None if wm is None or lm is None else round(wm - lm, 4),
                        "sample_ok": len(w) >= MIN_SAMPLE and len(lo) >= MIN_SAMPLE})
    return out


# ------------------------------------------------------------------ phase 8/9 repeatable patterns
def conditions() -> list[tuple[str, str]]:
    """(feature, regime axis used for the regime-dependence test). Single pre-declared conditions only."""
    return [(f, "trend_regime") for f in FEATURES] + [("trend_regime", "vol_regime"), ("vol_regime", "trend_regime")]


def condition_values(rows: Sequence[dict], feat: str) -> list[str]:
    if feat in ("trend_regime", "vol_regime"):
        vals = {r["features"][feat] for r in rows}
    else:
        vals = {bucket_of(r, feat) for r in rows}
    return sorted(v for v in vals if v not in (UNAVAILABLE, None))


def stability(rows: Sequence[dict], regime_axis: str) -> dict[str, Any]:
    dec = decided(rows)
    m = perf(rows)
    months: dict[str, list[dict]] = defaultdict(list)
    days: dict[str, float] = defaultdict(float)
    regs: dict[str, list[dict]] = defaultdict(list)
    for r in dec:
        months[month_of(r)].append(r)
        days[str(r["session_date"])[:10]] += trade_pnl(r)
        regs[r["features"][regime_axis]].append(r)
    active = {k: v for k, v in months.items() if len(v) >= MIN_MONTH_TRADES}
    prof_months = [k for k, v in active.items() if sum(trade_pnl(x) for x in v) > 0]
    reg_active = {k: v for k, v in regs.items() if len(v) >= MIN_MONTH_TRADES and k != UNAVAILABLE}
    reg_prof = [k for k, v in reg_active.items() if sum(trade_pnl(x) for x in v) > 0]
    best_day = max(days.values()) if days else 0.0
    best_reg = max((sum(trade_pnl(x) for x in v) for v in reg_active.values()), default=0.0)
    total = sum(days.values())
    return {**m, "months_active": len(active), "months_profitable": len(prof_months),
            "months_list": ",".join(sorted(active)), "regimes_active": len(reg_active),
            "regimes_profitable": len(reg_prof), "regimes_list": ",".join(sorted(reg_active)),
            "days_traded": len(days), "pnl_without_best_day": round(total - max(best_day, 0.0), 2),
            "pnl_without_best_regime": round(total - max(best_reg, 0.0), 2)}


def status_of(s: dict) -> str:
    pf = s["profit_factor"] or 0.0
    if s["trades"] < MIN_SAMPLE or s["months_active"] < 3:
        return "WEAK_SAMPLE"
    robust_days = s["pnl_without_best_day"] > 0
    robust_regime = s["regimes_profitable"] >= 2 and s["pnl_without_best_regime"] > 0
    if (s["trades"] >= STABLE_MIN_TRADES and s["net_pnl"] > 0 and pf >= 1.2 and s["months_profitable"] >= 3
            and s["months_profitable"] / s["months_active"] >= 0.6 and robust_days and robust_regime):
        return "STABLE"
    if s["net_pnl"] > 0 and pf >= 1.1 and s["months_profitable"] >= 3 and robust_days:
        return "PROMISING"
    return "UNSTABLE"


def pattern_table(rows: Sequence[dict]) -> list[dict]:
    out = []
    for feat, axis in conditions():
        for val in condition_values(rows, feat):
            for d in DIRECTIONS:
                if feat in ("trend_regime", "vol_regime"):
                    sub = [r for r in rows if r["direction"] == d and r["features"][feat] == val]
                else:
                    sub = [r for r in rows if r["direction"] == d and bucket_of(r, feat) == val]
                if not sub:
                    continue
                s = stability(sub, axis)
                out.append({"condition": f"{feat} = {val}", "feature": feat, "value": val, "direction": d, **s,
                            "uses_future_information": False, "sample_grade": sample_grade(s["trades"]),
                            "status": status_of(s)})
    return out


def rank_patterns(patterns: Sequence[dict], direction: str | None = None) -> list[dict]:
    pool = [p for p in patterns if direction is None or p["direction"] == direction]
    return sorted(pool, key=lambda p: (STATUS_ORDER.index(p["status"]), -(p["profit_factor"] or 0.0),
                                       -p["trades"], p["condition"]))


def best_group(table: Sequence[dict], name: str, scope: str) -> dict | None:
    pool = [r for r in table if r["scope"] == scope and r["sample_ok"] and r[name] != UNAVAILABLE]
    return max(pool, key=lambda r: ((r["profit_factor"] or 0.0), r["accuracy"] or 0.0)) if pool else None


# ------------------------------------------------------------------ study
def build_diagnostic(*, full_rows: Sequence[dict], monthly_rows: dict[str, list[dict]], spot: pd.DataFrame,
                     ctx_by_key: dict, dataset: dict, implementation: dict | None = None,
                     check_sample: int = 20) -> dict[str, Any]:
    bars = feature_bars(spot)
    rows = annotate(full_rows, bars, ctx_by_key)
    monthly = {m: annotate(v, bars, ctx_by_key) for m, v in monthly_rows.items()}
    days = daily_table(rows)
    regimes = regime_table(rows)
    times = time_table(rows)
    patterns = pattern_table(rows)
    checks = {"truncation": truncation_check(spot, rows, ctx_by_key, check_sample),
              "monthly_consistency": monthly_consistency(full_rows, monthly_rows),
              "chronology": chronology_check(rows),
              "buy_sell_separation": all((r["direction"] == "BUY") == (r.get("option_type") == "CE")
                                         for r in rows if r.get("option_type"))}
    vsrc = Counter(r["features"]["vwap_source"] for r in rows)
    return {
        "heading": HEADING, "banner": BANNER, "s2_changed": False, "orders": 0, "paper": True,
        "live_trading_enabled": False, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "dataset": dataset, "implementation": implementation or {}, "rows": rows,
        "overall": scoped(rows), "monthly": monthly_table(monthly), "daily": days,
        "daily_summary": daily_summary(days), "regimes": regimes, "times": times,
        "features": feature_table(rows), "options": option_table(rows), "win_loss": win_loss_table(rows),
        "patterns": patterns, "checks": checks, "vwap_source": dict(vsrc),
        "best_regime": {d: best_group(regimes, "regime", d) for d in DIRECTIONS},
        "best_time": {d: best_group(times, "time_window", d) for d in DIRECTIONS},
        "ranked": {d: rank_patterns(patterns, d) for d in DIRECTIONS}, "ranked_all": rank_patterns(patterns),
    }


# ------------------------------------------------------------------ reports
REPORT_FILES = ("s2_monthly_diagnostic.csv", "s2_daily_diagnostic.csv", "s2_regime_diagnostic.csv",
                "s2_time_diagnostic.csv", "s2_feature_diagnostic.csv", "s2_option_diagnostic.csv",
                "s2_win_loss_diagnostic.csv", "s2_pattern_diagnostic.csv", "s2_diagnostic.md")


def _csv(path: Path, rows: Sequence[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("INF" if v == math.inf else v) for k, v in r.items()})


def _row_md(label: str, m: dict) -> str:
    return (f"| {label} | {m['trades']} | {_pct(m['accuracy'])} | {_pf(m['profit_factor'])} | {m['net_pnl']:,.2f} | "
            f"{'-' if m['avg_r'] is None else m['avg_r']} | {m['max_drawdown']:,.2f} | {m['max_loss_streak']} |")


HEAD = ["| Group | Trades | Accuracy | PF | P&L (Rs) | Avg R | Max DD | Max loss streak |",
        "|---|---|---|---|---|---|---|---|"]


def describe_pattern(p: dict | None) -> str:
    if not p:
        return "none"
    return (f"{p['direction']} {p['condition']} - {_pct(p['accuracy'])} ({p['trades']} trades, PF "
            f"{_pf(p['profit_factor'])}, P&L Rs {p['net_pnl']:,.2f}, months {p['months_profitable']}/"
            f"{p['months_active']} profitable, regimes {p['regimes_active']}, grade {p['sample_grade']}) "
            f"[{p['status']}]")


def best_features(a: dict, n: int = 3) -> list[str]:
    good = [p for p in a["ranked_all"] if p["status"] in ("STABLE", "PROMISING")][:n]
    return [describe_pattern(p) for p in good]


def strongest(a: dict, d: str, n: int = 3) -> list[str]:
    return [describe_pattern(p) for p in a["ranked"][d] if p["status"] in ("STABLE", "PROMISING")][:n]


def markdown(a: dict) -> str:
    ds, ov, dsum, ck = a["dataset"], a["overall"], a["daily_summary"], a["checks"]
    md = [f"# {HEADING}", "", f"**{BANNER}** | S2 changed: NO | Orders: 0 | PAPER: YES | "
          f"LIVE_TRADING_ENABLED=false | generated {a['generated_at']}", "",
          "Diagnostic only. No S2 parameter, filter, threshold, exit or weight was created or changed. Bucket edges "
          "are coarse pre-declared values. A condition is never called profitable unless it passes the stability "
          f"requirement (>= 3 active months, >= {MIN_SAMPLE} trades, not dependent on one day or one regime, causal).",
          "", "## Executive Summary", ""]
    md += [f"- Dataset: {ds.get('from')} -> {ds.get('to')}, {ds.get('sessions')} sessions, {ds.get('spot_candles')} "
           f"NIFTY candles, {ds.get('option_candles')} option candles, {ds.get('contracts')} contracts.",
           f"- Exact S2 over the full period: BUY {line(ov['BUY'])}; SELL {line(ov['SELL'])}; COMBINED "
           f"{line(ov['COMBINED'])}."]
    for d in DIRECTIONS:
        br, bt = a["best_regime"][d], a["best_time"][d]
        md.append(f"- Best regime {d}: " + (f"{br['regime']} {line(br)}" if br else f"no regime with >= {MIN_SAMPLE} trades"))
        md.append(f"- Best time window {d}: " + (f"{bt['time_window']} {line(bt)}" if bt else "insufficient sample"))
    bf = best_features(a)
    md.append("- Repeatable conditions (STABLE / PROMISING): " + ("; ".join(bf) if bf else
              "**none** - no single condition passes the stability requirement."))
    md += ["", "## Checks", "",
           f"- No future feature leakage (truncate later candles / triple later premium): "
           f"{'PASS' if ck['truncation']['passed'] else 'FAIL'} ({ck['truncation']['checked']} signals; "
           f"{ck['truncation']['mismatches'] or 'no mismatches'})",
           f"- Separate monthly replays equal the full replay sliced by month: "
           f"{'PASS' if ck['monthly_consistency']['passed'] else 'FAIL'} {ck['monthly_consistency']['months']}",
           f"- Chronology (ordered signals, entry = signal + 5 min, exit >= entry, IST session date): "
           f"{'PASS' if ck['chronology']['passed'] else 'FAIL'} {ck['chronology']}",
           f"- BUY -> CE and SELL -> PE only: {'PASS' if ck['buy_sell_separation'] else 'FAIL'}",
           f"- VWAP reference source: {a['vwap_source']} (the NIFTY index has no traded volume, so a session TWAP "
           "of typical price is used when VWAP is undefined; underlying volume ratio is then UNAVAILABLE and the "
           "option volume ratio is reported instead)."]
    if a["implementation"]:
        md += ["", "## Exact S2 (unchanged)", ""] + [f"- **{k}**: {v}" for k, v in a["implementation"].items()]
    md += ["", "## Phase 1 - Monthly breakdown (separate exact-S2 replay per month)", ""] + HEAD
    md += [_row_md(f"{r['label']} {r['scope']}", r) for r in a["monthly"]]
    md += ["", "## Phase 2 - Daily breakdown", "",
           f"- Sessions: {dsum.get('days')}; traded: {dsum.get('traded_days')}; profitable: "
           f"{dsum.get('profitable_days')}; losing: {dsum.get('losing_days')}; flat: {dsum.get('flat_days')}",
           f"- Median day P&L: Rs {dsum.get('median_day_pnl')}; total: Rs {dsum.get('total_pnl')}",
           f"- Best days: {dsum.get('best_days')}", f"- Worst days: {dsum.get('worst_days')}",
           f"- Largest single-day contribution: {dsum.get('largest_day')} Rs {dsum.get('largest_day_pnl')} "
           f"({_pct(dsum.get('largest_day_share_of_gross_profit'))} of gross profit); P&L without it: Rs "
           f"{dsum.get('pnl_without_best_day')}"]
    md += ["", "## Phase 3 - Regimes (signal candle only)", "",
           "Trend: TREND_UP = close > EMA20 > EMA50 and EMA20 slope > 0; TREND_DOWN = mirror; otherwise RANGE. "
           f"Volatility: trailing ATR14/close percentile >= {HIGH_VOL_PCTILE:.2f} HIGH, <= {LOW_VOL_PCTILE:.2f} LOW, "
           "else NORMAL. Both axes use candles <= the signal candle only.", ""] + HEAD
    md += [_row_md(f"{r['regime']} {r['scope']}", r) for r in a["regimes"]]
    md += ["", "## Phase 4 - Time of day (no filter created)", ""] + HEAD
    md += [_row_md(f"{r['time_window']} {r['scope']}", r) for r in a["times"]]
    md += ["", "## Phase 5 - Feature buckets", "",
           f"Full table in s2_feature_diagnostic.csv. Buckets with fewer than {MIN_SAMPLE} trades are listed but "
           "not interpreted. Rows below: buckets with sample_ok.", "",
           "| Feature | Bucket | Side | Trades | Accuracy | PF | P&L |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['feature']} | {r['bucket']} | {r['scope']} | {r['trades']} | {_pct(r['accuracy'])} | "
           f"{_pf(r['profit_factor'])} | {r['net_pnl']:,.2f} |" for r in a["features"] if r["sample_ok"]]
    md += ["", "## Phase 6 - Winning vs losing trades (retrospective only)", "",
           "| Side | Feature | WIN mean (n) | LOSS mean (n) | WIN - LOSS |", "|---|---|---|---|---|"]
    md += [f"| {r['direction']} | {r['feature']} | {r['win_mean']} ({r['win_n']}) | {r['loss_mean']} ({r['loss_n']}) "
           f"| {r['mean_diff_win_minus_loss']} |" for r in a["win_loss"]]
    md += ["", "## Phase 7 - Option analysis", "",
           "S2 always buys the listed strike nearest to the signal close, so every contract is within half a strike "
           "step of spot: ATM = |strike - spot| <= 12.5, otherwise SLIGHT_ITM / SLIGHT_OTM (no deep ITM/OTM exists).",
           "", "| Analysis | Side | Bucket | Trades | Accuracy | PF | P&L | Interpretation |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['analysis']} | {r['side']} | {r['bucket']} | {r['trades']} | {_pct(r['accuracy'])} | "
           f"{_pf(r['profit_factor'])} | {r['net_pnl']:,.2f} | {r['interpretation']} |" for r in a["options"]]
    md += ["", "## Phase 8/9 - Repeatable patterns and stability", "",
           f"STABLE: >= {STABLE_MIN_TRADES} trades, >= 3 active months (>= {MIN_MONTH_TRADES} trades each), >= 3 "
           "profitable months and >= 60% of active months profitable, PF >= 1.2, P&L > 0 without the best day, "
           "profitable in >= 2 regimes and P&L > 0 without the best regime. PROMISING: P&L > 0, PF >= 1.1, >= 3 "
           f"profitable months, P&L > 0 without the best day. WEAK_SAMPLE: < {MIN_SAMPLE} trades or < 3 active "
           "months. UNSTABLE: everything else. All conditions are single, causal and pre-declared.", "",
           "| Condition | Side | Trades | Accuracy | Months active (profitable) | Regimes active | PF | P&L | Grade | "
           "Status |", "|---|---|---|---|---|---|---|---|---|---|"]
    shown = [p for p in a["ranked_all"] if p["status"] in ("STABLE", "PROMISING")]
    shown += [p for p in a["ranked_all"] if p["status"] == "UNSTABLE"][:15]
    md += [f"| {p['condition']} | {p['direction']} | {p['trades']} | {_pct(p['accuracy'])} | {p['months_active']} "
           f"({p['months_profitable']}) | {p['regimes_active']} | {_pf(p['profit_factor'])} | {p['net_pnl']:,.2f} | "
           f"{p['sample_grade']} | {p['status']} |" for p in shown]
    st = Counter(p["status"] for p in a["patterns"])
    md += ["", f"Status counts: {dict(st)}. Full list in s2_pattern_diagnostic.csv.", "",
           "## Phase 10 - No strategy modification", "",
           "No S2 parameter, filter, threshold, exit or weight was created or changed; production code untouched. "
           "Conditions above are hypotheses for the NEXT research phase and must be validated out-of-sample "
           "before any use.", "", "## Limitations", "",
           "- Six months of 5m data; per-month samples per bucket are small, so most buckets are WEAK_SAMPLE.",
           "- Many buckets are inspected; some will look good by chance (multiple comparisons). The stability "
           "requirement reduces but does not remove this risk.",
           "- NIFTY index volume is zero, so underlying VWAP / volume ratio are unavailable (TWAP proxy used)."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    _csv(p["s2_monthly_diagnostic.csv"], a["monthly"])
    _csv(p["s2_daily_diagnostic.csv"], a["daily"])
    _csv(p["s2_regime_diagnostic.csv"], a["regimes"])
    _csv(p["s2_time_diagnostic.csv"], a["times"])
    _csv(p["s2_feature_diagnostic.csv"], a["features"])
    _csv(p["s2_option_diagnostic.csv"], a["options"])
    _csv(p["s2_win_loss_diagnostic.csv"], a["win_loss"])
    _csv(p["s2_pattern_diagnostic.csv"], a["patterns"])
    p["s2_diagnostic.md"].write_text(markdown(a), encoding="utf-8")
    return p


def summary_text(a: dict) -> str:
    by = {(r["month"], r["scope"]): r for r in a["monthly"]}
    out = [HEADING, ""]
    for m, label in MONTHS:
        out.append(f"{label}:")
        out += [f"{s}: {line(by[(m, s)])}" for s in SCOPES]
        out.append("")
    out.append("BEST REGIME:")
    for d in DIRECTIONS:
        br = a["best_regime"][d]
        out.append(f"{d}: " + (f"{br['regime']} {line(br)}" if br else f"none with >= {MIN_SAMPLE} trades"))
    out += ["", "BEST TIME WINDOW:"]
    for d in DIRECTIONS:
        bt = a["best_time"][d]
        out.append(f"{d}: " + (f"{bt['time_window']} {line(bt)}" if bt else f"none with >= {MIN_SAMPLE} trades"))
    out += ["", "BEST REPEATABLE FEATURES:"]
    bf = best_features(a)
    if bf:
        out += [f"{i}. {t}" for i, t in enumerate(bf, 1)]
    else:
        out.append("1. none - no condition passes the stability requirement (not claiming any profitable rule)")
    for d in DIRECTIONS:
        out += ["", f"{d}:"]
        s = strongest(a, d)
        out += s if s else [f"no STABLE or PROMISING {d} condition (sample / multi-month stability insufficient)"]
    out += ["", "FINAL:", "DIAGNOSTIC COMPLETE", "", "S2 CHANGED:", "NO", "", "ORDERS:", "0", "", "PAPER:", "YES"]
    return "\n".join(out)
