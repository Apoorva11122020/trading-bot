"""STRATEGY IMPROVEMENT LAB for the existing S2 NIFTY options strategy (PAPER, research only, no orders).

Every candidate is the unchanged S2 signal stream with pre-declared skip rules per direction. Entry, contract, stop,
target and outcome come from the existing historical shadow replay (``shadow_historical.replay``), and each signal is
evaluated independently there, so a candidate's accepted rows are exactly the baseline rows it keeps. All filter
inputs are read from the completed signal candle or earlier (verified by a truncation test); end-of-day regime labels
are used for reporting and stability scoring only, never as a filter. Thresholds that need calibration are fixed from
TRAIN signals only. Candidate selection (COMBINED_DIRECTION_FILTER, walk-forward picks) never reads OOS.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from src.backtest import buy_dte_lab as dl
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.backtest.live_vs_historical import wilson
from src.backtest.regime_classifier import ABS_TREND_RETURN, build_regime_labels
from src.backtest.regime_filter import AVOID_REGIMES, PointInTimeRegimeClassifier
from src.indicators.technical import ema, rsi

HEADING = "STRATEGY IMPROVEMENT LAB"
ROBUST, PROMISING, NOT_PROMISING, INSUFFICIENT_DATA, REJECTED = (
    "ROBUST", "PROMISING", "NOT_PROMISING", "INSUFFICIENT_DATA", "REJECTED")
STATUS_RANK = {ROBUST: 0, PROMISING: 1, NOT_PROMISING: 2, INSUFFICIENT_DATA: 3, REJECTED: 4}
BASELINE, COMBINED, REFERENCE = "BASELINE", "COMBINED_DIRECTION_FILTER", "REFERENCE"
DIRECTIONS = ("BUY", "SELL")
CHOPPY_SIGN_CHANGES = 6  # same threshold as regime_classifier.classify_day, applied to bars up to the signal
STRONG_TREND_LABELS = ("strong_bullish_trend", "strong_bearish_trend")
DEFAULT_PARAMS = {"rsi_buy": 55.0, "rsi_sell": 45.0, "volume_mult": 1.2, "break_atr": 0.50, "vwap_dist_atr": 1.0}
SPLITS = imp.SPLITS
PERIODS = (*SPLITS, "ALL")


# ------------------------------------------------------------------ candidates
@dataclass(frozen=True)
class Candidate:
    """``buy``/``sell``: checks applied to that direction; () = unfiltered, None = direction excluded."""
    name: str
    family: str
    buy: tuple[str, ...] | None = ()
    sell: tuple[str, ...] | None = ()
    note: str = ""

    def checks(self, direction: str) -> tuple[str, ...] | None:
        return self.buy if direction == "BUY" else self.sell

    @property
    def n_checks(self) -> int:
        return len(set(self.buy or ()) | set(self.sell or ()))


def _b(name, *checks, family="BUY_CONFIRM"):
    return Candidate(name, family, buy=tuple(checks))


def _s(name, *checks, family="SELL_CONFIRM"):
    return Candidate(name, family, sell=tuple(checks))


def _both(name, *checks, family):
    return Candidate(name, family, buy=tuple(checks), sell=tuple(checks))


TIME_WINDOWS = (("0930", "1000"), ("1000", "1100"), ("1100", "1200"), ("1200", "1330"), ("1330", "1430"),
                ("1430", "1500"), ("0930", "1100"), ("1000", "1200"), ("1330", "1500"))


def predeclared_candidates() -> list[Candidate]:
    """Fixed before any result is computed; COMBINED_DIRECTION_FILTER is appended by ``combined_candidate``."""
    out = [Candidate(BASELINE, REFERENCE, note="current S2, unchanged"),
           Candidate("SELL_ONLY_REFERENCE", REFERENCE, buy=None, note="existing shadow candidate; reference only")]
    out += [_b("BUY_EMA_CONFIRM", "ema"), _b("BUY_VWAP_CONFIRM", "vwap"), _b("BUY_RSI_CONFIRM", "rsi"),
            _b("BUY_VOLUME_CONFIRM", "volume"), _b("BUY_BREAKOUT_CONFIRM", "breakout"),
            _b("BUY_CANDLE_CONFIRM", "candle"),
            _b("BUY_EMA_VWAP", "ema", "vwap"), _b("BUY_EMA_RSI", "ema", "rsi"), _b("BUY_VWAP_RSI", "vwap", "rsi"),
            _b("BUY_VWAP_VOLUME", "vwap", "volume"), _b("BUY_EMA_VOLUME", "ema", "volume"),
            _b("BUY_EMA_VWAP_RSI", "ema", "vwap", "rsi"), _b("BUY_EMA_VWAP_VOLUME", "ema", "vwap", "volume"),
            _b("BUY_EMA_RSI_VOLUME", "ema", "rsi", "volume"), _b("BUY_VWAP_RSI_VOLUME", "vwap", "rsi", "volume"),
            _b("BUY_STRONG_CONFIRMATION", "strong"),
            _b("BUY_MORNING_ONLY", "time:0930-1100", family="BUY_TIME"),
            _b("BUY_MIDDAY_ONLY", "time:1100-1330", family="BUY_TIME"),
            _b("BUY_AFTERNOON_ONLY", "time:1330-1500", family="BUY_TIME"),
            _b("BUY_TRENDING_ONLY", "regime:trending", family="BUY_REGIME"),
            _b("BUY_NON_CHOPPY_ONLY", "regime:non_choppy", family="BUY_REGIME")]
    out += [_s("SELL_EMA_CONFIRM", "ema"), _s("SELL_VWAP_CONFIRM", "vwap"), _s("SELL_RSI_CONFIRM", "rsi"),
            _s("SELL_VOLUME_CONFIRM", "volume"), _s("SELL_BREAKOUT_CONFIRM", "breakout"),
            _s("SELL_CANDLE_CONFIRM", "candle"),
            _s("SELL_EMA_VWAP", "ema", "vwap"), _s("SELL_EMA_RSI", "ema", "rsi"), _s("SELL_VWAP_RSI", "vwap", "rsi"),
            _s("SELL_STRONG_CONFIRMATION", "strong")]
    out += [_both(f"TIME_{a}_{b}", f"time:{a}-{b}", family="TIME") for a, b in TIME_WINDOWS]
    out += [_both("REGIME_TRENDING_ONLY", "regime:trending", family="REGIME"),
            _both("REGIME_RANGE_BOUND_ONLY", "regime:range", family="REGIME"),
            _both("REGIME_NOT_RANGE_BOUND", "regime:not_range", family="REGIME"),
            _both("REGIME_CHOPPY_EXCLUDED", "regime:non_choppy", family="REGIME"),
            _both("REGIME_CHOPPY_ONLY", "regime:choppy", family="REGIME"),
            _both("REGIME_EXISTING_AVOID", "regime:existing_avoid", family="REGIME"),
            _both("REGIME_PREV_DAY_TREND", "regime:prev_day_trend", family="REGIME")]
    out += [_both("ENTRY_EMA_ALIGN", "ema", family="ENTRY"), _both("ENTRY_VWAP_ALIGN", "vwap", family="ENTRY"),
            _both("ENTRY_RSI_CONFIRM", "rsi", family="ENTRY"), _both("ENTRY_VOLUME_CONFIRM", "volume", family="ENTRY"),
            _both("ENTRY_BREAKOUT_0.50ATR", "breakout", family="ENTRY"),
            _both("ENTRY_CANDLE_CONFIRM", "candle", family="ENTRY"),
            _both("ENTRY_VWAP_DISTANCE", "vwap_dist", family="ENTRY"),
            _both("ENTRY_MIN_ATR", "min_atr", family="ENTRY"),
            _both("ENTRY_PREMIUM_NORMAL", "premium_normal", family="ENTRY")]
    return out


def check_text(name: str, direction: str, p: dict) -> str:
    up = direction == "BUY"
    if name.startswith("time:"):
        a, b = name[5:].split("-")
        return f"signal candle time {a[:2]}:{a[2:]}-{b[:2]}:{b[2:]}"
    return {
        "ema": "close > EMA20 > EMA50" if up else "close < EMA20 < EMA50",
        "vwap": "close > session VWAP" if up else "close < session VWAP",
        "rsi": f"RSI14 >= {p['rsi_buy']:g}" if up else f"RSI14 <= {p['rsi_sell']:g}",
        "volume": f"volume >= {p['volume_mult']:g} x previous-20 average",
        "breakout": f"breakout distance <= {p['break_atr']:.2f} x ATR14",
        "candle": ("previous and signal candle bullish" if up else "previous and signal candle bearish"),
        "strong": ("previous close > ORB high, bullish signal candle, close > EMA20 > EMA50" if up else
                   "previous close < ORB low, bearish signal candle, close < EMA20 < EMA50"),
        "vwap_dist": f"|close - VWAP| <= {p['vwap_dist_atr']:g} x ATR14",
        "min_atr": f"ATR14/close >= {p.get('min_atr_pct', float('nan')):.4f}% (TRAIN median)",
        "premium_normal": (f"premium ATR % of premium in [{p.get('prem_lo', float('nan')):.2f}, "
                           f"{p.get('prem_hi', float('nan')):.2f}] (TRAIN 10th-90th pct)"),
        "regime:trending": f"|close / day open - 1| >= {ABS_TREND_RETURN:.1%} at the signal candle",
        "regime:range": "point-in-time range_bound tag (regime_filter)",
        "regime:not_range": "no point-in-time range_bound tag",
        "regime:non_choppy": f"fewer than {CHOPPY_SIGN_CHANGES} close-to-close sign changes so far today",
        "regime:choppy": f">= {CHOPPY_SIGN_CHANGES} close-to-close sign changes so far today",
        "regime:existing_avoid": f"no point-in-time tag in {', '.join(AVOID_REGIMES)}",
        "regime:prev_day_trend": "previous session labelled strong bullish/bearish trend",
    }[name]


def describe(c: Candidate, p: dict) -> dict[str, str]:
    def side(d):
        ch = c.checks(d)
        return "EXCLUDED" if ch is None else ("unchanged S2" if not ch else
                                              "; ".join(check_text(x, d, p) for x in ch))
    return {"BUY": side("BUY"), "SELL": side("SELL")}


# ------------------------------------------------------------------ causal features
def bar_features(spot: pd.DataFrame) -> pd.DataFrame:
    """Per-bar values known at each bar's close (indexed by IST timestamp)."""
    f = imp.underlying_features(spot)
    vol = spot[["timestamp"]].copy()
    vol["volume"] = spot["volume"].astype(float) if "volume" in spot else float("nan")
    ts = pd.to_datetime(vol["timestamp"])
    vol["timestamp"] = ts.dt.tz_localize(imp.IST) if ts.dt.tz is None else ts.dt.tz_convert(imp.IST)
    f = f.merge(vol.drop_duplicates("timestamp", keep="last"), on="timestamp", how="left")
    f["ema20"], f["ema50"], f["rsi14"] = ema(f["close"], 20), ema(f["close"], 50), rsi(f["close"], 14)
    v = f["volume"].where(f["volume"] > 0)
    f["vol_avg"] = v.shift(1).rolling(20, min_periods=20).mean()
    f["vol_ratio"] = v / f["vol_avg"]
    g = f.groupby("day")
    tp = (f["high"] + f["low"] + f["close"]) / 3
    f["vwap"] = (tp * v).groupby(f["day"]).cumsum() / v.groupby(f["day"]).cumsum()
    for col in ("open", "close", "orb_high", "orb_low"):
        f[f"prev_{col}"] = g[col].shift(1)
    f["day_open"] = g["open"].transform("first")
    sign = np.sign(g["close"].pct_change())
    flip = (sign.groupby(f["day"]).diff().abs() > 0).astype(int)
    f["sign_changes"] = flip.groupby(f["day"]).cumsum()
    return f.set_index("timestamp")


def previous_session_labels(regimes: dict[date, set[str]]) -> dict[date, set[str]]:
    days = sorted(regimes)
    return {d: set(regimes[p]) for p, d in zip(days, days[1:])}


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else float(v)


def row_key(r: dict) -> tuple[str, str]:
    return (str(r["signal_time"]), r["direction"])


def premium_close_at_signal(contexts: Sequence[dict]) -> dict[tuple[str, str], float]:
    out = {}
    for ctx in contexts:
        prem = ctx.get("premium")
        if prem is None or isinstance(prem, str) or prem.empty:
            continue
        hit = prem.loc[prem["timestamp"] == ctx["signal_time"], "close"]
        if len(hit):
            out[(ctx["signal_time"].isoformat(), ctx["direction"])] = float(hit.iloc[0])
    return out


def build_facts(rows: Sequence[dict], bars: pd.DataFrame, pit: PointInTimeRegimeClassifier,
                prev_labels: dict[date, set[str]], prem_close: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for r in rows:
        ts = imp._ist(r["signal_time"])
        bar = bars.loc[ts] if ts in bars.index else None
        val = (lambda c: None if bar is None else _num(bar[c]))
        close, atr14, vwap = val("close"), val("atr14"), val("vwap")
        day_open, sc = val("day_open"), val("sign_changes")
        move = None if close is None or not day_open else close / day_open - 1
        tags = pit.regimes_at(ts.to_pydatetime())
        prem, patr = prem_close.get(row_key(r)), _num(r.get("premium_atr"))
        out[row_key(r)] = {
            "direction": r["direction"], "minute": ts.hour * 60 + ts.minute,
            **{c: val(c) for c in ("open", "close", "ema20", "ema50", "rsi14", "vwap", "vol_ratio", "prev_open",
                                    "prev_close", "prev_orb_high", "prev_orb_low", "atr14")},
            "break_atr": _num(r.get("break_atr")),
            "atr_pct": None if close is None or atr14 is None else atr14 / close * 100,
            "vwap_dist_atr": None if vwap is None or not atr14 else abs(close - vwap) / atr14,
            "intraday_move": move, "sign_changes": sc,
            "trending": None if move is None else abs(move) >= ABS_TREND_RETURN,
            "choppy": None if sc is None else sc >= CHOPPY_SIGN_CHANGES,
            "pit_tags": sorted(tags), "range_bound": "range_bound" in tags,
            "prev_day_labels": sorted(prev_labels.get(ts.date(), ())),
            "has_prev_day": ts.date() in prev_labels,
            "premium_atr_pct": None if prem is None or patr is None or prem <= 0 else patr / prem * 100,
            "dte": r.get("dte"), "dte_bucket": r.get("dte_bucket"),
        }
    return out


def facts_from_spot(rows, spot, s2_cfg, prem_close) -> dict:
    return build_facts(rows, bar_features(spot), PointInTimeRegimeClassifier(spot),
                       previous_session_labels(build_regime_labels(spot, s2_cfg)), prem_close)


# ------------------------------------------------------------------ checks / decisions
def check(name: str, direction: str, f: dict, p: dict) -> bool | None:
    """True/False, or None when the input was not available at the signal candle."""
    up = direction == "BUY"

    def need(*keys):
        return all(f.get(k) is not None for k in keys)

    if name == "ema":
        if not need("close", "ema20", "ema50"):
            return None
        return (f["close"] > f["ema20"] > f["ema50"]) if up else (f["close"] < f["ema20"] < f["ema50"])
    if name == "vwap":
        return None if not need("close", "vwap") else (f["close"] > f["vwap"] if up else f["close"] < f["vwap"])
    if name == "rsi":
        return None if not need("rsi14") else (f["rsi14"] >= p["rsi_buy"] if up else f["rsi14"] <= p["rsi_sell"])
    if name == "volume":
        return None if not need("vol_ratio") else f["vol_ratio"] >= p["volume_mult"]
    if name == "breakout":
        return None if not need("break_atr") else f["break_atr"] <= p["break_atr"]
    if name == "candle":
        if not need("open", "close", "prev_open", "prev_close"):
            return None
        return ((f["prev_close"] > f["prev_open"] and f["close"] > f["open"]) if up else
                (f["prev_close"] < f["prev_open"] and f["close"] < f["open"]))
    if name == "strong":
        lvl = "prev_orb_high" if up else "prev_orb_low"
        if not need("open", "close", "prev_close", lvl, "ema20", "ema50"):
            return None
        if up:
            return f["prev_close"] > f[lvl] and f["close"] > f["open"] and f["close"] > f["ema20"] > f["ema50"]
        return f["prev_close"] < f[lvl] and f["close"] < f["open"] and f["close"] < f["ema20"] < f["ema50"]
    if name == "vwap_dist":
        return None if not need("vwap_dist_atr") else f["vwap_dist_atr"] <= p["vwap_dist_atr"]
    if name == "min_atr":
        return None if not need("atr_pct") or p.get("min_atr_pct") is None else f["atr_pct"] >= p["min_atr_pct"]
    if name == "premium_normal":
        if not need("premium_atr_pct") or p.get("prem_lo") is None:
            return None
        return p["prem_lo"] <= f["premium_atr_pct"] <= p["prem_hi"]
    if name.startswith("time:"):
        a, b = name[5:].split("-")
        lo, hi = int(a[:2]) * 60 + int(a[2:]), int(b[:2]) * 60 + int(b[2:])
        return lo <= f["minute"] < hi
    if name == "regime:trending":
        return f.get("trending")
    if name == "regime:choppy":
        return f.get("choppy")
    if name == "regime:non_choppy":
        return None if f.get("choppy") is None else not f["choppy"]
    if name == "regime:range":
        return f["range_bound"]
    if name == "regime:not_range":
        return not f["range_bound"]
    if name == "regime:existing_avoid":
        return not (set(f["pit_tags"]) & set(AVOID_REGIMES))
    if name == "regime:prev_day_trend":
        return None if not f.get("has_prev_day") else bool(set(f["prev_day_labels"]) & set(STRONG_TREND_LABELS))
    raise ValueError(f"unknown check {name}")


def decide(c: Candidate, f: dict | None, p: dict) -> str | None:
    """Skip reason, or None when the signal is accepted."""
    if f is None:
        return f"{dr.CONFIRM_UNAVAILABLE}:SIGNAL_BAR_MISSING"
    checks = c.checks(f["direction"])
    if checks is None:
        return dr.DIRECTION_SKIP
    results = {k: check(k, f["direction"], f, p) for k in checks}
    missing = [k for k, v in results.items() if v is None]
    failed = [k for k, v in results.items() if v is False]
    if missing:
        return f"{dr.CONFIRM_UNAVAILABLE}:{','.join(missing)}"
    return "FILTERED:" + ",".join(failed) if failed else None


def apply_candidate(base_rows: Sequence[dict], facts: dict, c: Candidate, p: dict) -> list[dict]:
    """Baseline rows with the candidate's skips applied; accepted rows keep the baseline outcome unchanged."""
    out = []
    for r in base_rows:
        row = dict(r, strategy=c.name)
        skip = decide(c, facts.get(row_key(r)), p)
        if skip is not None:
            row.update(decision="SKIPPED", skip_reason=skip, state=sh.SKIPPED, outcome=None, pnl=None,
                       r_multiple=None, reason=None, reason_raw=None)
        out.append(row)
    return out


# ------------------------------------------------------------------ metrics
def r_value(r: dict) -> float:
    return -1.0 if r.get("outcome") == imp.AMBIGUOUS else float(r.get("r_multiple") or 0.0)


def perf(rows: Iterable[dict]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["signal_time"])
    m = sh.stats(rows)
    ev = [r for r in rows if r["state"] == sh.EVALUATED]
    pnl, rs = [float(r["pnl"]) for r in ev], [r_value(r) for r in ev]
    win = cw = cl = mw = ml = 0
    for r in ev:
        win = r["outcome"] == imp.WIN
        cw, cl = (cw + 1, 0) if win else (0, cl + 1)
        mw, ml = max(mw, cw), max(ml, cl)
    daily: dict[str, float] = {}
    for r in ev:
        daily[r["session_date"]] = daily.get(r["session_date"], 0.0) + float(r["pnl"])
    ordered = sorted(daily.items(), key=lambda kv: kv[1], reverse=True)
    net = m["net_pnl"]
    largest = max(pnl) if pnl else None
    gl = -sum(p for p in pnl if p < 0)
    m.update(
        gross_profit=round(sum(p for p in pnl if p > 0), 2), gross_loss=round(gl, 2),
        average_r=round(sum(rs) / len(rs), 4) if rs else None, median_r=round(median(rs), 4) if rs else None,
        max_consecutive_wins=mw, max_consecutive_losses=ml,
        traded_days=len(daily), profitable_days=sum(v > 0 for v in daily.values()),
        day_stability=(sum(v > 0 for v in daily.values()) / len(daily)) if daily else None,
        best_day=ordered[0][0] if ordered else None, best_day_pnl=round(ordered[0][1], 2) if ordered else None,
        worst_day=ordered[-1][0] if ordered else None, worst_day_pnl=round(ordered[-1][1], 2) if ordered else None,
        best_day_share=(ordered[0][1] / net) if ordered and net > 0 else None,
        worst_day_share=(-ordered[-1][1] / gl) if ordered and gl > 0 and ordered[-1][1] < 0 else None,
        pnl_excl_best_day=round(net - (ordered[0][1] if ordered else 0.0), 2),
        pnl_excl_best_2_days=round(net - sum(v for _, v in ordered[:2]), 2),
        largest_win=largest, pnl_excl_largest_win=round(net - max(largest or 0.0, 0.0), 2),
        wr_ci95=None if not ev else [round(x, 4) for x in wilson(m["wins"], len(ev))],
    )
    return m


def group_stability(rows: Sequence[dict], labels: Callable[[dict], Iterable[str]],
                    min_trades: int) -> tuple[float | None, dict[str, dict]]:
    groups: dict[str, list] = {}
    for r in rows:
        if r["state"] == sh.EVALUATED:
            for lab in labels(r):
                groups.setdefault(lab, []).append(r)
    stats = {k: perf(v) for k, v in groups.items()}
    big = {k: v for k, v in stats.items() if v["evaluated"] >= min_trades}
    share = sum(v["net_pnl"] > 0 for v in big.values()) / len(big) if big else None
    return share, stats


def period_rows(rows: Sequence[dict], days: Iterable[str]) -> list[dict]:
    wanted = set(days)
    return [r for r in rows if r["session_date"] in wanted]


# ------------------------------------------------------------------ status / ranking
@dataclass(frozen=True)
class LabCriteria:
    min_oos_trades: int = 30
    min_retention: float = 0.15
    min_oos_pf: float = 1.10
    collapse_ratio: float = 0.50
    suspicious_wr: float = 0.90
    min_group_trades: int = 5
    robust_wf_share: float = 0.75
    oos_previously_observed: bool = True


def candidate_status(s: dict, c: LabCriteria = LabCriteria()) -> dict[str, Any]:
    """``s``: scoring dict with train/validation/oos perf, retention, stability, lookahead, walk-forward, robustness."""
    t, v, o = s["TRAIN"], s["VALIDATION"], s["OOS"]
    pf = lambda m: m["profit_factor"]  # noqa: E731
    reasons: list[str] = []
    if not s["lookahead_passed"]:
        return {"status": REJECTED, "failed_checks": ["lookahead"]}
    if s["retention"] is None or s["retention"] < c.min_retention:
        return {"status": REJECTED, "failed_checks": ["filter_removes_almost_all_signals"]}
    if o["evaluated"] < c.min_oos_trades:
        return {"status": INSUFFICIENT_DATA, "failed_checks": [f"oos_trades<{c.min_oos_trades}"]}
    if (o["win_rate"] or 0) >= c.suspicious_wr or (pf(o) is not None and math.isinf(pf(o))):
        return {"status": REJECTED, "failed_checks": ["suspiciously_perfect_oos"]}
    if pf(t) is not None and pf(t) > 1 and (pf(v) is None or pf(v) < c.collapse_ratio * min(pf(t), 10.0)):
        reasons.append("validation_collapses_vs_train")
    if pf(v) is not None and pf(v) > 1 and (pf(o) is None or pf(o) < c.collapse_ratio * min(pf(v), 10.0)):
        reasons.append("oos_collapses_vs_validation")
    if o["net_pnl"] > 0 and o["pnl_excl_best_2_days"] <= 0:
        reasons.append("depends_on_one_or_two_days")
    if o["net_pnl"] > 0 and o["pnl_excl_largest_win"] <= 0:
        reasons.append("depends_on_one_extreme_winner")
    if s["regime_groups"] >= 2 and s["profitable_regime_groups"] <= 1 and o["net_pnl"] > 0:
        reasons.append("works_in_only_one_regime")
    if reasons:
        return {"status": REJECTED, "failed_checks": reasons}
    weak = []
    if pf(o) is None or pf(o) <= c.min_oos_pf:
        weak.append(f"oos_pf<={c.min_oos_pf}")
    if o["net_pnl"] <= 0:
        weak.append("oos_pnl<=0")
    if pf(v) is None or pf(v) <= 1.0 or v["net_pnl"] <= 0:
        weak.append("validation_not_profitable")
    if weak:
        return {"status": NOT_PROMISING, "failed_checks": weak}
    cap = []
    if s["wf_positive_share"] is None or s["wf_positive_share"] < c.robust_wf_share:
        cap.append("walk_forward_not_stable")
    if s["robustness_ok"] is False:
        cap.append("parameter_not_in_stable_region")
    if c.oos_previously_observed:
        cap.append("oos_window_previously_observed")
    return {"status": PROMISING if cap else ROBUST, "failed_checks": cap}


def _n(v, missing=-1.0):
    return missing if v is None else (1e9 if isinstance(v, float) and math.isinf(v) else v)


def rank(scores: dict[str, dict]) -> list[str]:
    """Status, OOS stability, OOS PF, OOS P&L, OOS DD, OOS WR, trades, validation/regime/day stability, simplicity."""
    def key(n):
        s, o = scores[n], scores[n]["OOS"]
        return (STATUS_RANK[s["status"]], -_n(o["day_stability"]), -_n(o["profit_factor"]), -o["net_pnl"],
                o["max_drawdown"], -_n(o["win_rate"]), -o["evaluated"], -_n(scores[n]["VALIDATION"]["day_stability"]),
                -_n(s["regime_stability"]), -_n(s["day_stability"]), s["n_checks"], n)
    return sorted(scores, key=key)


# ------------------------------------------------------------------ TRAIN-only calibration
def train_params(base_rows: Sequence[dict], facts: dict, train_days: Sequence[str]) -> dict:
    tr = period_rows(base_rows, train_days)
    atr = [facts[row_key(r)]["atr_pct"] for r in tr if facts.get(row_key(r), {}).get("atr_pct") is not None]
    prem = [facts[row_key(r)]["premium_atr_pct"] for r in tr
            if facts.get(row_key(r), {}).get("premium_atr_pct") is not None]
    p = dict(DEFAULT_PARAMS)
    if atr:
        p["min_atr_pct"] = float(np.median(atr))
        p["atr_terciles"] = [float(x) for x in np.percentile(atr, [33.33, 66.67])]
    if prem:
        p["prem_lo"], p["prem_hi"] = (float(x) for x in np.percentile(prem, [10, 90]))
        p["prem_terciles"] = [float(x) for x in np.percentile(prem, [33.33, 66.67])]
    return p


# ------------------------------------------------------------------ combined (TRAIN + VALIDATION only)
def _dir_perf(rows, days, d):
    return perf(r for r in period_rows(rows, days) if r["direction"] == d)


def combined_candidate(cands: Sequence[Candidate], rows: dict[str, list[dict]], split_days: dict,
                       min_train: int = 10, min_val: int = 5) -> tuple[Candidate, dict]:
    """Per direction: best VALIDATION PF among that direction's single-direction candidates that are profitable on
    TRAIN and VALIDATION; otherwise that direction stays unfiltered. OOS rows are never read."""
    pick: dict[str, str | None] = {}
    for d in DIRECTIONS:
        fam = "BUY_" if d == "BUY" else "SELL_"
        pool = [c for c in cands if c.family.startswith(fam) and c.checks(d)]
        best, best_key = None, None
        for c in pool:
            t, v = _dir_perf(rows[c.name], split_days["TRAIN"], d), _dir_perf(rows[c.name], split_days["VALIDATION"], d)
            if t["evaluated"] < min_train or v["evaluated"] < min_val:
                continue
            if not (_n(t["profit_factor"], 0) > 1 and _n(v["profit_factor"], 0) > 1 and v["net_pnl"] > 0):
                continue
            k = (_n(v["profit_factor"]), v["net_pnl"], -c.n_checks)
            if best_key is None or k > best_key:
                best, best_key = c, k
        pick[d] = best.name if best else None
    by = {c.name: c for c in cands}
    buy = by[pick["BUY"]].buy if pick["BUY"] else ()
    sell = by[pick["SELL"]].sell if pick["SELL"] else ()
    note = f"BUY from {pick['BUY'] or 'none (unfiltered)'}, SELL from {pick['SELL'] or 'none (unfiltered)'}"
    return Candidate(COMBINED, "COMBINED", buy=buy, sell=sell, note=note), pick


# ------------------------------------------------------------------ robustness (Phase 13)
ROBUSTNESS_GRID = (
    ("BUY_RSI", "BUY", "rsi", "rsi_buy", (52.0, 55.0, 58.0, 60.0)),
    ("SELL_RSI", "SELL", "rsi", "rsi_sell", (48.0, 45.0, 42.0, 40.0)),
    ("BUY_VOLUME", "BUY", "volume", "volume_mult", (1.0, 1.2, 1.5)),
    ("SELL_VOLUME", "SELL", "volume", "volume_mult", (1.0, 1.2, 1.5)),
    ("BUY_BREAKOUT_ATR", "BUY", "breakout", "break_atr", (0.25, 0.50, 0.75, 1.00)),
    ("SELL_BREAKOUT_ATR", "SELL", "breakout", "break_atr", (0.25, 0.50, 0.75, 1.00)),
)


def robustness(base_rows, facts, p: dict, split_days: dict, min_trades: int = 5) -> list[dict]:
    """Stable value = it and its grid neighbours are profitable on TRAIN and VALIDATION (OOS shown, never used)."""
    out = []
    for fam, d, chk, key, values in ROBUSTNESS_GRID:
        cand = Candidate(fam, "ROBUSTNESS", **({"buy": (chk,)} if d == "BUY" else {"sell": (chk,)}))
        res = []
        for v in values:
            rows = apply_candidate(base_rows, facts, cand, {**p, key: v})
            m = {s: _dir_perf(rows, split_days[s], d) for s in SPLITS}
            ok = all(m[s]["evaluated"] >= min_trades and _n(m[s]["profit_factor"], 0) > 1 and m[s]["net_pnl"] > 0
                     for s in ("TRAIN", "VALIDATION"))
            res.append((v, m, ok))
        for i, (v, m, ok) in enumerate(res):
            near = [res[j][2] for j in (i - 1, i + 1) if 0 <= j < len(res)]
            out.append({"family": fam, "direction": d, "check": chk, "param": key, "value": v,
                        "is_default": v == DEFAULT_PARAMS[key], "profitable_train_validation": ok,
                        "stable": ok and all(near), **{f"{s.lower()}_{k}": m[s][k] for s in SPLITS
                                                       for k in ("evaluated", "win_rate", "profit_factor", "net_pnl")}})
    return out


def robustness_ok(c: Candidate, grid: Sequence[dict]) -> bool | None:
    """None when the candidate has no tunable threshold; otherwise every default threshold must sit in a stable
    region of its grid."""
    seen = []
    for d in DIRECTIONS:
        for chk in c.checks(d) or ():
            rows = [g for g in grid if g["direction"] == d and g["check"] == chk and g["is_default"]]
            if rows:
                seen.append(rows[0]["stable"])
    return None if not seen else all(seen)


# ------------------------------------------------------------------ walk-forward (Phase 14)
def walk_forward_folds(days: Sequence[str], train: int = 8, val: int = 3, test: int = 3) -> list[dict]:
    """Rolling windows anchored at the last session; test windows never overlap; earlier days only for selection."""
    days = sorted(days)
    folds, end = [], len(days)
    while end - test - val - train >= 0:
        s = end - test - val - train
        folds.append({"train": days[s:s + train], "validation": days[s + train:s + train + val],
                      "test": days[end - test:end]})
        end -= test
    return list(reversed(folds))


def wf_select(cands: Sequence[Candidate], rows: dict[str, list[dict]], fold: dict,
              min_train: int = 10, min_val: int = 5) -> str:
    best, best_key = BASELINE, None
    for c in cands:
        t, v = perf(period_rows(rows[c.name], fold["train"])), perf(period_rows(rows[c.name], fold["validation"]))
        if t["evaluated"] < min_train or v["evaluated"] < min_val:
            continue
        if not (_n(t["profit_factor"], 0) > 1 and _n(v["profit_factor"], 0) > 1 and v["net_pnl"] > 0):
            continue
        k = (_n(v["profit_factor"]), v["net_pnl"], -c.n_checks)
        if best_key is None or k > best_key:
            best, best_key = c.name, k
    return best


def walk_forward(cands: Sequence[Candidate], rows: dict[str, list[dict]], days: Sequence[str],
                 split_of_day: dict[str, str]) -> dict[str, Any]:
    folds = walk_forward_folds(days)
    out_folds, per_cand = [], {c.name: [] for c in cands}
    for i, f in enumerate(folds, 1):
        # COMBINED was assembled from the full TRAIN+VALIDATION window, so it cannot be picked inside a fold.
        chosen = wf_select([c for c in cands if c.family not in (REFERENCE, "COMBINED")], rows, f)
        test = {n: perf(period_rows(rows[n], f["test"])) for n in rows}
        out_folds.append({"fold": i, **{k: f"{v[0]}..{v[-1]}" for k, v in f.items()},
                          "test_splits": sorted({split_of_day.get(d, "?") for d in f["test"]}),
                          "selected": chosen, "selected_test": test[chosen], "baseline_test": test[BASELINE]})
        for c in cands:
            per_cand[c.name].append(test[c.name])
    share = {}
    for n, tests in per_cand.items():
        traded = [t for t in tests if t["evaluated"]]
        share[n] = sum(t["net_pnl"] > 0 for t in traded) / len(traded) if traded else None
    return {"folds": out_folds, "per_candidate": per_cand, "positive_share": share}


# ------------------------------------------------------------------ lookahead (truncation test)
def truncation_lookahead(base_rows: Sequence[dict], spot: pd.DataFrame, s2_cfg: dict, prem_close: dict,
                         full_facts: dict, cands: Sequence[Candidate], p: dict, days: Sequence[str],
                         cut_times: Sequence[str] = ("11:00", "13:00")) -> dict[str, dict]:
    """Recompute every feature from candles <= cut only; facts and decisions for signals <= cut must not change."""
    ts_all = pd.to_datetime(spot["timestamp"])
    ts_all = ts_all.dt.tz_localize(imp.IST) if ts_all.dt.tz is None else ts_all.dt.tz_convert(imp.IST)
    res = {c.name: {"cuts": 0, "failures": []} for c in cands}
    res["_features"] = {"cuts": 0, "failures": []}
    for d in days:
        for t in cut_times:
            cut = pd.Timestamp(datetime.combine(date.fromisoformat(d), time.fromisoformat(t)), tz=imp.IST)
            sub = [r for r in base_rows if r["session_date"] == d and imp._ist(r["signal_time"]) <= cut]
            trunc = facts_from_spot(sub, spot[ts_all <= cut].reset_index(drop=True), s2_cfg, prem_close)
            same = all(_facts_equal(trunc.get(row_key(r)), full_facts.get(row_key(r))) for r in sub)
            res["_features"]["cuts"] += 1
            if not same:
                res["_features"]["failures"].append(cut.isoformat())
            for c in cands:
                res[c.name]["cuts"] += 1
                if any(decide(c, trunc.get(row_key(r)), p) != decide(c, full_facts.get(row_key(r)), p) for r in sub):
                    res[c.name]["failures"].append(cut.isoformat())
    for r in res.values():
        r["passed"] = bool(r["cuts"]) and not r["failures"]
    return res


def _facts_equal(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a is b
    for k in a:
        x, y = a[k], b.get(k)
        if isinstance(x, float) or isinstance(y, float):
            if (x is None) != (y is None) or (x is not None and not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)):
                return False
        elif x != y:
            return False
    return True


# ------------------------------------------------------------------ bucket analysis (Phases 4, 5, 7, 8)
def _bucket(v, edges, labels):
    if v is None:
        return "UNAVAILABLE"
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


def factor_buckets(f: dict | None, r: dict, p: dict) -> dict[str, str]:
    if f is None:
        return {}
    up = f["direction"] == "BUY"
    ema_ok = None if f["ema20"] is None or f["ema50"] is None else f["ema20"] > f["ema50"]
    at, pt = p.get("atr_terciles"), p.get("prem_terciles")
    candle = (None if f["open"] is None else "BULLISH" if f["close"] > f["open"] else
              "BEARISH" if f["close"] < f["open"] else "DOJI")
    regime = ("TRENDING" if f["trending"] else "RANGE_BOUND" if f["range_bound"] else
              "CHOPPY" if f["choppy"] else "OTHER") if f["trending"] is not None else "UNAVAILABLE"
    return {
        "01_ema_trend": "UNAVAILABLE" if ema_ok is None else ("EMA20>EMA50" if ema_ok else "EMA20<=EMA50"),
        "02_vwap": ("UNAVAILABLE (index volume is zero)" if f["vwap"] is None else
                    "close>VWAP" if f["close"] > f["vwap"] else "close<=VWAP"),
        "03_rsi14": _bucket(f["rsi14"], (45, 50, 55, 60, 70), ("<45", "45-50", "50-55", "55-60", "60-70", ">70")),
        "04_volume_ratio": ("UNAVAILABLE (index volume is zero)" if f["vol_ratio"] is None else
                            _bucket(f["vol_ratio"], (0.8, 1.0, 1.2, 1.5), ("<0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5",
                                                                          ">1.5"))),
        "05_breakout_atr": _bucket(f["break_atr"], (0.25, 0.5, 0.75, 1.0),
                                   ("0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00", ">1.00")),
        "06_atr_pct": "UNAVAILABLE" if not at else _bucket(f["atr_pct"], at, ("LOW (TRAIN T1)", "MID (TRAIN T2)",
                                                                              "HIGH (TRAIN T3)")),
        "07_vwap_distance_atr": ("UNAVAILABLE (index volume is zero)" if f["vwap_dist_atr"] is None else
                                 _bucket(f["vwap_dist_atr"], (0.5, 1.0, 2.0), ("<0.5", "0.5-1.0", "1.0-2.0", ">2.0"))),
        "08_candle_direction": "UNAVAILABLE" if candle is None else
                               (candle + (" (with signal)" if (candle == "BULLISH") == up and candle != "DOJI"
                                          else " (against signal)" if candle != "DOJI" else "")),
        "09_time_of_day": r.get("time_bucket") or "other",
        "10_regime_point_in_time": regime,
        "11_dte": r.get("dte_bucket") or "UNAVAILABLE",
        "12_premium_volatility": "UNAVAILABLE" if not pt else _bucket(f["premium_atr_pct"], pt,
                                                                      ("LOW (TRAIN T1)", "MID (TRAIN T2)",
                                                                       "HIGH (TRAIN T3)")),
    }


def bucket_analysis(base_rows, facts, p, split_days, direction: str, min_weak: int = 15) -> list[dict]:
    """Weak zone = TRAIN bucket with >= ``min_weak`` trades, PF < 1 and Wilson WR upper bound below the
    direction's TRAIN WR; confirmed when the VALIDATION bucket also has PF < 1. OOS is reported, never used."""
    rows = [r for r in base_rows if r["direction"] == direction]
    labelled = [(r, factor_buckets(facts.get(row_key(r)), r, p)) for r in rows]
    base = {s: perf(period_rows(rows, split_days[s])) for s in SPLITS}
    factors = sorted({k for _, b in labelled for k in b})
    out = []
    for fac in factors:
        for lab in sorted({b.get(fac) for _, b in labelled if b.get(fac)}):
            grp = [r for r, b in labelled if b.get(fac) == lab]
            m = {s: perf(period_rows(grp, split_days[s])) for s in SPLITS}
            m["ALL"] = perf(grp)
            t, v = m["TRAIN"], m["VALIDATION"]
            weak = (t["evaluated"] >= min_weak and _n(t["profit_factor"], 0) < 1 and t["wr_ci95"] is not None
                    and base["TRAIN"]["win_rate"] is not None and t["wr_ci95"][1] < base["TRAIN"]["win_rate"])
            zone = ("" if not weak else "WEAK_TRAIN_CONFIRMED_VALIDATION"
                    if v["evaluated"] and _n(v["profit_factor"], 0) < 1 else "WEAK_TRAIN_ONLY")
            for s in PERIODS:
                x = m[s]
                out.append({"direction": direction, "factor": fac, "bucket": lab, "split": s,
                            "trades": x["evaluated"], "wins": x["wins"], "losses": x["losses"],
                            "win_rate": x["win_rate"], "profit_factor": x["profit_factor"], "net_pnl": x["net_pnl"],
                            "max_drawdown": x["max_drawdown"],
                            "wr_ci_low": None if not x["wr_ci95"] else x["wr_ci95"][0],
                            "wr_ci_high": None if not x["wr_ci95"] else x["wr_ci95"][1],
                            "weak_zone": zone})
    return out


def window_analysis(base_rows, facts, p, split_days, filters: Sequence[tuple[str, str, str]]) -> list[dict]:
    """(label, check, kind) filters applied per direction and in total; coverage = trades kept / baseline trades."""
    out = []
    for label, chk, kind in filters:
        cand = Candidate(label, kind, buy=(chk,), sell=(chk,))
        rows = apply_candidate(base_rows, facts, cand, p)
        for d in ("BUY", "SELL", "TOTAL"):
            sel = (lambda rs: rs) if d == "TOTAL" else (lambda rs, _d=d: [r for r in rs if r["direction"] == _d])
            for s in PERIODS:
                days = sum((split_days[x] for x in SPLITS), []) if s == "ALL" else split_days[s]
                m, b = perf(sel(period_rows(rows, days))), perf(sel(period_rows(base_rows, days)))
                out.append({"filter": label, "kind": kind, "direction": d, "split": s, "trades": m["evaluated"],
                            "wins": m["wins"], "losses": m["losses"], "win_rate": m["win_rate"],
                            "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"],
                            "max_drawdown": m["max_drawdown"],
                            "coverage": m["evaluated"] / b["evaluated"] if b["evaluated"] else None})
    return out


def eod_regime_analysis(base_rows, split_days) -> list[dict]:
    """End-of-day labels (look-ahead if used as a filter): reporting only."""
    out = []
    for lab in sh.REGIME_LABELS:
        for d in ("BUY", "SELL", "TOTAL"):
            for s in PERIODS:
                days = sum((split_days[x] for x in SPLITS), []) if s == "ALL" else split_days[s]
                rows = [r for r in period_rows(base_rows, days) if lab in r["regimes"]
                        and (d == "TOTAL" or r["direction"] == d)]
                m = perf(rows)
                out.append({"filter": f"EOD_LABEL:{lab}", "kind": "EOD_LABEL_REPORTING_ONLY_LOOKAHEAD",
                            "direction": d, "split": s, "trades": m["evaluated"], "wins": m["wins"],
                            "losses": m["losses"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                            "net_pnl": m["net_pnl"], "max_drawdown": m["max_drawdown"], "coverage": None})
    return out


TIME_FILTERS = [(f"{a[:2]}:{a[2:]}-{b[:2]}:{b[2:]}", f"time:{a}-{b}", "TIME_WINDOW") for a, b in TIME_WINDOWS]
REGIME_FILTERS = [("TRENDING_ONLY", "regime:trending", "CAUSAL_REGIME"),
                  ("RANGE_BOUND_ONLY", "regime:range", "CAUSAL_REGIME"),
                  ("NOT_RANGE_BOUND", "regime:not_range", "CAUSAL_REGIME"),
                  ("CHOPPY_EXCLUDED", "regime:non_choppy", "CAUSAL_REGIME"),
                  ("CHOPPY_ONLY", "regime:choppy", "CAUSAL_REGIME"),
                  ("EXISTING_AVOID_LIST", "regime:existing_avoid", "CAUSAL_REGIME"),
                  ("PREV_DAY_STRONG_TREND", "regime:prev_day_trend", "PREVIOUS_SESSION_LABEL")]


# ------------------------------------------------------------------ DTE policies (Phase 10)
def dte_study(x, hist_days: Sequence[str], split_days: dict, progress: Callable[[str], None] | None = None) -> list[dict]:
    """Existing point-in-time DTE policies (``buy_dte_lab``) applied to both directions (CE for BUY, PE for SELL)."""
    out, day_set = [], set(hist_days)
    for i, pol in enumerate(dl.POLICIES[1:], 1):
        if progress:
            progress(f"DTE policy {i}/{len(dl.POLICIES) - 1}: {pol.requested}")
        resolve = dl.policy_resolver(pol, x.pit, x.rc, x.premium_for)
        pipeline = imp._pipeline_factory(x.signal_fn, resolve, x.params, x.s2["range_start"], x.s2["range_end"])
        contexts, feats = pipeline(x.spot, x.premium_for, {date.fromisoformat(d) for d in hist_days})
        rows, seen = [], set()
        for ctx, r in zip(contexts, imp.run_variant(contexts, feats, imp.BASELINE, square_off=x.rc.session.square_off,
                                                     lot_size=sh.LOT_SIZE, cache={})):
            key = (r["signal_time"], r["direction"])
            if key in seen or r["session_date"] not in day_set:
                continue
            seen.add(key)
            r["expiry"] = ctx["expiry"].isoformat() if ctx.get("expiry") else None
            rows.append(sh.classify_row(r, x.regimes))
        resolved = [r for r in rows if r.get("expiry")]
        before = sum(r["dte"] is not None and r["dte"] < 0 for r in resolved)
        outside = sum(not dl.in_policy(pol, date.fromisoformat(r["session_date"]), date.fromisoformat(r["expiry"][:10]))
                      for r in resolved)
        for d in ("BUY", "SELL", "TOTAL"):
            for s in PERIODS:
                days = hist_days if s == "ALL" else split_days[s]
                sel = [r for r in period_rows(rows, days) if d == "TOTAL" or r["direction"] == d]
                m = perf(sel)
                out.append({"policy": pol.requested, "direction": d, "split": s, "signals": m["signals"],
                            "trades": m["evaluated"], "expiry_or_data_unavailable": m["data_unavailable"],
                            "win_rate": m["win_rate"], "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"],
                            "max_drawdown": m["max_drawdown"], "expiry_before_signal_date": before,
                            "outside_requested_range": outside, "point_in_time_ok": before == 0 and outside == 0})
    return out


# ------------------------------------------------------------------ S2 rules (Phase 1)
def s2_rules(rc, params, s2_cfg: dict) -> dict[str, str]:
    om = rc.option_mapping
    return {
        "signal": (f"Opening range {s2_cfg['range_start']}-{s2_cfg['range_end']} on NIFTY 5m candles. Every completed "
                   f"candle after {s2_cfg['range_end']} that closes above the ORB high is a BUY signal, below the ORB "
                   "low a SELL signal (no repeat/cooldown filter; consecutive candles can each signal)."),
        "buy": "BUY signal -> long ATM CE of the nearest listed expiry.",
        "sell": "SELL signal -> long ATM PE of the nearest listed expiry (option buying, no writing).",
        "option_selection": (f"expiry_mode={om.get('expiry_mode')}, strike_mode={om.get('strike_mode')}, "
                             f"offset={om.get('strike_offset')}; historical replay resolves the expiry point-in-time "
                             "(official expired expiries + expired contracts)."),
        "entry_timing": (f"entry_timing={params.entry_timing}: open of the option candle after the signal candle, "
                         f"rounded to tick; new entries only before {params.stop_new_entries}."),
        "stop_loss": (f"entry - {params.stop_atr_multiplier:g} x premium ATR{params.atr_period} (option candles up to "
                      "and including the signal candle)."),
        "target": f"entry + {params.target_r:g} x (entry - stop) (= {params.target_r:g}R).",
        "square_off": f"{rc.session.square_off} (first candle at/after square-off exits at its open).",
        "outcome": ("first touch of stop/target on later real option candles; stop and target inside one candle = "
                    "AMBIGUOUS, counted as -1R (conservative)."),
        "existing_filters": ("none beyond ORB completeness and the entry window; the research replay evaluates every "
                             "signal independently (no position limit, daily loss limit or trade cap applied)."),
        "costs": "excluded (no brokerage/slippage).",
    }


# ------------------------------------------------------------------ forward (Phase 15)
def load_forward(directory: Path, after: date) -> list[dict]:
    out = []
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            d = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if d <= after:
            continue
        j = json.loads(path.read_text(encoding="utf-8"))
        out.append({k: j.get(k) for k in ("date", "total", "resolved", "pending", "wins", "losses", "invalid",
                                          "accuracy", "profit_factor", "net_pnl", "max_drawdown")})
    return out


# ------------------------------------------------------------------ study
def build_lab(*, base_rows: Sequence[dict], facts: dict, days: Sequence[str], splits: dict, rules: dict,
              lookahead: Callable[[Sequence[Candidate], dict], dict] | None = None,
              baseline_lookahead_passed: bool = True, criteria: LabCriteria = LabCriteria(),
              dte: list[dict] | None = None, forward: list[dict] | None = None) -> dict[str, Any]:
    split_days, unassigned = so.assign_days(days, splits)
    hist = [d for s in SPLITS for d in split_days[s]]
    base_rows = period_rows(base_rows, hist)
    p = train_params(base_rows, facts, split_days["TRAIN"])
    cands = predeclared_candidates()
    rows = {c.name: apply_candidate(base_rows, facts, c, p) for c in cands}
    comb, pick = combined_candidate(cands, rows, split_days)
    cands.append(comb)
    rows[COMBINED] = apply_candidate(base_rows, facts, comb, p)
    la = lookahead(cands, p) if lookahead else {c.name: {"cuts": 0, "failures": [], "passed": True} for c in cands}
    grid = robustness(base_rows, facts, p, split_days)
    split_of_day = {d: s for s in SPLITS for d in split_days[s]}
    wf = walk_forward(cands, rows, hist, split_of_day)
    base_all = perf(rows[BASELINE])
    periods = {**split_days, "ALL": hist}
    scores = {}
    for c in cands:
        m = {s: perf(period_rows(rows[c.name], d)) for s, d in periods.items()}
        reg_share, reg = group_stability(rows[c.name], lambda r: r["regimes"], criteria.min_group_trades)
        time_share, _ = group_stability(rows[c.name], lambda r: [r.get("time_bucket") or "other"],
                                        criteria.min_group_trades)
        big = {k: v for k, v in reg.items() if v["evaluated"] >= criteria.min_group_trades}
        passed = la.get(c.name, {}).get("passed", False) and (baseline_lookahead_passed if c.name == BASELINE
                                                               else la.get("_features", {}).get("passed", True))
        s = {**m, "retention": m["ALL"]["evaluated"] / base_all["evaluated"] if base_all["evaluated"] else None,
             "regime_stability": reg_share, "regime_groups": len(big),
             "profitable_regime_groups": sum(v["net_pnl"] > 0 for v in big.values()),
             "time_stability": time_share, "day_stability": m["ALL"]["day_stability"],
             "wf_positive_share": wf["positive_share"].get(c.name), "robustness_ok": robustness_ok(c, grid),
             "lookahead_passed": bool(passed), "n_checks": c.n_checks, "family": c.family,
             "direction": {d: {s: _dir_perf(rows[c.name], dd, d) for s, dd in periods.items()} for d in DIRECTIONS}}
        s.update(candidate_status(s, criteria))
        if c.family == REFERENCE and c.name != BASELINE:
            s["failed_checks"] = s["failed_checks"] + ["reference_only_direction_disabled"]
        scores[c.name] = s
    ranking = rank(scores)
    research = [n for n in ranking if scores[n]["family"] != REFERENCE]
    best = research[0] if research else None
    base_o = scores[BASELINE]["OOS"]
    deployable = bool(best and scores[best]["status"] in (ROBUST, PROMISING)
                      and _n(scores[best]["OOS"]["profit_factor"]) > _n(base_o["profit_factor"])
                      and scores[best]["OOS"]["net_pnl"] > base_o["net_pnl"])
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"),
        "rules": rules, "params": p,
        "splits": {s: {"label": so.SPLIT_LABELS[s], "from": splits[s][0].isoformat(), "to": splits[s][1].isoformat(),
                       "days": split_days[s]} for s in SPLITS},
        "unassigned_days": unassigned, "criteria": asdict(criteria),
        "candidates": {c.name: {"family": c.family, "rules": describe(c, p), "n_checks": c.n_checks, "note": c.note}
                       for c in cands},
        "combined_pick": pick, "scores": scores, "ranking": ranking, "best_candidate": best,
        "recommendation": "TEST_CANDIDATE_IN_PAPER" if deployable else "KEEP_CURRENT",
        "lookahead": la, "robustness": grid, "walk_forward": wf,
        "buy_analysis": bucket_analysis(base_rows, facts, p, split_days, "BUY"),
        "sell_analysis": bucket_analysis(base_rows, facts, p, split_days, "SELL"),
        "time_analysis": window_analysis(base_rows, facts, p, split_days, TIME_FILTERS),
        "regime_analysis": (window_analysis(base_rows, facts, p, split_days, REGIME_FILTERS)
                            + eod_regime_analysis(base_rows, split_days)),
        "baseline": baseline_report(base_rows, facts, p, split_days),
        "dte": dte, "forward": forward or [],
        "notes": {
            "oos": ("OOS (2026-09-25..10-01) was already visible in earlier project research, so no candidate can be "
                    "ROBUST here; PROMISING is the ceiling and means forward PAPER observation only."),
            "vwap_volume": ("NIFTY index candles carry zero volume: VWAP, VWAP distance and volume-ratio checks are "
                            "DATA_UNAVAILABLE and those candidates skip every signal they apply to."),
            "forward": ("2026-10-05 and later (including today's live PAPER tracker) are forward data: reported "
                        "separately and never used for selection."),
            "calibration": ("ENTRY_MIN_ATR, ENTRY_PREMIUM_NORMAL and the ATR/premium buckets use thresholds fixed from "
                            "TRAIN signals only."),
            "regime": ("Filters use point-in-time regime inputs only; end-of-day labels appear in reports and "
                       "stability scores, never in a filter."),
        },
        "_rows": rows,
    }


def baseline_report(base_rows, facts, p, split_days) -> dict[str, Any]:
    hist = [d for s in SPLITS for d in split_days[s]]
    periods = {**split_days, "ALL": hist}
    out: dict[str, Any] = {"metrics": {}, "breakdown": {}}
    for s, days in periods.items():
        rows = period_rows(base_rows, days)
        out["metrics"][s] = {d: perf(r for r in rows if d == "TOTAL" or r["direction"] == d)
                             for d in ("BUY", "SELL", "TOTAL")}
    rows = period_rows(base_rows, hist)
    keys = {
        "date": lambda r, f: [r["session_date"]],
        "time_of_day": lambda r, f: [r.get("time_bucket") or "other"],
        "regime_eod_label": lambda r, f: r["regimes"] or ["unlabelled"],
        "regime_point_in_time": lambda r, f: [factor_buckets(f, r, p).get("10_regime_point_in_time", "UNAVAILABLE")],
        "dte": lambda r, f: [r.get("dte_bucket") or "UNAVAILABLE"],
        "option_contract": lambda r, f: [str(r.get("contract") or "UNRESOLVED")],
        "atr_bucket": lambda r, f: [factor_buckets(f, r, p).get("06_atr_pct", "UNAVAILABLE")],
    }
    for name, fn in keys.items():
        groups: dict[tuple[str, str], list] = {}
        for r in rows:
            for lab in fn(r, facts.get(row_key(r))):
                for d in (r["direction"], "TOTAL"):
                    groups.setdefault((lab, d), []).append(r)
        out["breakdown"][name] = {f"{lab}|{d}": perf(g) for (lab, d), g in sorted(groups.items())}
    return out


# ------------------------------------------------------------------ reports
_pct, _pf, _rs = sh._pct, sh._pf_s, sh._rs
METRIC_KEYS = ("signals", "evaluated", "wins", "losses", "win_rate", "profit_factor", "gross_profit", "gross_loss",
               "net_pnl", "max_drawdown", "average_r", "median_r", "max_consecutive_losses", "max_consecutive_wins")


def _fmt(v):
    if isinstance(v, float) and math.isinf(v):
        return "INF"
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(sh._jsonable(v))
    return v


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def summary_rows(a: dict) -> list[dict]:
    out = []
    for i, n in enumerate(a["ranking"], 1):
        s, c = a["scores"][n], a["candidates"][n]
        row = {"rank": i, "candidate": n, "family": c["family"], "buy_rule": c["rules"]["BUY"],
               "sell_rule": c["rules"]["SELL"], "n_checks": c["n_checks"]}
        for p in PERIODS:
            m = s[p]
            row.update({f"{p.lower()}_{k}": m[k] for k in ("evaluated", "wins", "losses", "win_rate", "profit_factor",
                                                          "net_pnl", "max_drawdown")})
        o = s["OOS"]
        row.update(max_loss_streak=s["ALL"]["max_consecutive_losses"], regime_stability=s["regime_stability"],
                   day_stability=s["day_stability"], time_stability=s["time_stability"],
                   oos_day_stability=o["day_stability"], oos_best_day_share=o["best_day_share"],
                   oos_worst_day_share=o["worst_day_share"], oos_pnl_excl_best_2_days=o["pnl_excl_best_2_days"],
                   oos_pnl_excl_largest_win=o["pnl_excl_largest_win"], retention=s["retention"],
                   wf_positive_share=s["wf_positive_share"], robustness_ok=s["robustness_ok"],
                   lookahead="PASS" if s["lookahead_passed"] else "FAIL", status=s["status"],
                   failed_checks=";".join(s["failed_checks"]))
        out.append(row)
    return out


def _metric_table(title: str, by: dict[str, dict]) -> list[str]:
    md = [f"### {title}", "", "| Scope | Signals | Resolved | Wins | Losses | Accuracy | PF | Gross profit | Gross loss | "
          "Net P&L | Max DD | Avg R | Median R | Max loss streak | Max win streak |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in by.items():
        md.append(f"| {k} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | {_pct(m['win_rate'])} | "
                  f"{_pf(m['profit_factor'])} | {_rs(m['gross_profit'])} | {_rs(m['gross_loss'])} | "
                  f"{_rs(m['net_pnl'])} | {_rs(m['max_drawdown'])} | {m['average_r']} | {m['median_r']} | "
                  f"{m['max_consecutive_losses']} | {m['max_consecutive_wins']} |")
    return md + [""]


def baseline_markdown(a: dict) -> str:
    md = [f"# {HEADING} - BASELINE (current S2, unchanged)", "",
          "PAPER mode | research only | no orders | production S2 not modified", "", "## Phase 1 - current S2 rules", ""]
    md += [f"- **{k}**: {v}" for k, v in a["rules"].items()]
    md += ["", "## Splits (existing project split, chronological)", "", "| Split | From | To | Sessions |",
           "|---|---|---|---|"]
    md += [f"| {sp['label']} | {sp['from']} | {sp['to']} | {len(sp['days'])} |" for sp in a["splits"].values()]
    md += ["", "## Phase 2 - baseline metrics", ""]
    for s, by in a["baseline"]["metrics"].items():
        md += _metric_table(s, by)
    md += ["## Breakdowns (TRAIN + VALIDATION + OOS)", ""]
    for name, groups in a["baseline"]["breakdown"].items():
        md += [f"### {name}", "", "| Group | Direction | Resolved | WR | PF | Net P&L | Max DD |",
               "|---|---|---|---|---|---|---|"]
        for k, m in groups.items():
            lab, d = k.rsplit("|", 1)
            md.append(f"| {lab} | {d} | {m['evaluated']} | {_pct(m['win_rate'])} | {_pf(m['profit_factor'])} | "
                      f"{_rs(m['net_pnl'])} | {_rs(m['max_drawdown'])} |")
        md.append("")
    if a["forward"]:
        md += ["## Forward data (NOT used for research; live PAPER tracker premium replay)", "",
               "| Date | Total | Resolved | Pending | Wins | Losses | Accuracy | PF | Net P&L |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {f['date']} | {f['total']} | {f['resolved']} | {f['pending']} | {f['wins']} | {f['losses']} | "
               f"{f['accuracy']} | {f['profit_factor']} | {f['net_pnl']} |" for f in a["forward"]]
    md += ["", "Notes:"] + [f"- {v}" for v in a["notes"].values()]
    return "\n".join(md) + "\n"


def _rank_line(i, n, s):
    t, v, o = s["TRAIN"], s["VALIDATION"], s["OOS"]
    return (f"| {i} | {n} | {_pct(t['win_rate'])} | {_pct(v['win_rate'])} | {_pct(o['win_rate'])} | "
            f"{_pf(t['profit_factor'])} | {_pf(v['profit_factor'])} | {_pf(o['profit_factor'])} | {_rs(o['net_pnl'])} | "
            f"{_rs(o['max_drawdown'])} | {o['evaluated']} | {s['ALL']['max_consecutive_losses']} | "
            f"{_pct(s['regime_stability'])} | {_pct(s['day_stability'])} | "
            f"{'PASS' if s['lookahead_passed'] else 'FAIL'} | {s['status']} |")


RANK_HEAD = ["| # | Candidate | Train WR | Validation WR | OOS WR | Train PF | Validation PF | OOS PF | OOS P&L | OOS DD | "
             "OOS Trades | Max Loss Streak | Regime Stability | Day Stability | Lookahead | Status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def comparison(a: dict) -> dict[str, Any] | None:
    best = a["best_candidate"]
    if not best:
        return None
    b, x = a["scores"][BASELINE], a["scores"][best]
    out = {"candidate": best, "rules_current": a["candidates"][BASELINE]["rules"],
           "rules_candidate": a["candidates"][best]["rules"], "periods": {}}
    for p in PERIODS:
        out["periods"][p] = {k: (b[p][k], x[p][k]) for k in ("evaluated", "win_rate", "profit_factor", "net_pnl",
                                                             "max_drawdown")}
    return out


def final_markdown(a: dict) -> str:
    md = [f"# {HEADING} - FINAL RANKING", "", "PAPER mode | research only | no orders | production S2 NOT modified", "",
          "Ranking: status (ROBUST > PROMISING > NOT_PROMISING > INSUFFICIENT_DATA > REJECTED), then OOS day stability, "
          "OOS PF, OOS P&L, OOS DD, OOS WR, OOS trades, validation stability, regime stability, day stability, "
          "simplicity. A candidate below the OOS trade minimum can never outrank a sufficiently-sampled one.", ""]
    md += RANK_HEAD + [_rank_line(i, n, a["scores"][n]) for i, n in enumerate(a["ranking"], 1)]
    md += ["", "## Status reasons", ""]
    md += [f"- **{n}** ({a['scores'][n]['status']}): {', '.join(a['scores'][n]['failed_checks']) or 'all checks passed'}"
           for n in a["ranking"]]
    cmp_ = comparison(a)
    md += ["", "## CURRENT S2 vs BEST RESEARCH CANDIDATE", ""]
    if not cmp_:
        md.append("- No research candidate.")
    else:
        md += [f"Best research candidate: **{cmp_['candidate']}** (status {a['scores'][cmp_['candidate']]['status']})",
               "", "| Direction | CURRENT S2 | Candidate |", "|---|---|---|"]
        md += [f"| {d} | {cmp_['rules_current'][d]} | {cmp_['rules_candidate'][d]} |" for d in DIRECTIONS]
        md += ["", "| Period | Metric | CURRENT S2 | Candidate |", "|---|---|---|---|"]
        for p, ms in cmp_["periods"].items():
            for k, (bv, xv) in ms.items():
                f = _pct if k == "win_rate" else _pf if k == "profit_factor" else _rs if k != "evaluated" else str
                md.append(f"| {p} | {k} | {f(bv)} | {f(xv)} |")
    md += ["", f"## Recommendation: **{a['recommendation']}**", "",
           "Nothing is deployed automatically; production S2 stays unchanged until explicitly approved.", "",
           "## COMBINED_DIRECTION_FILTER construction (TRAIN + VALIDATION only)", "",
           f"- BUY component: {a['combined_pick']['BUY'] or 'none qualified (BUY unfiltered)'}",
           f"- SELL component: {a['combined_pick']['SELL'] or 'none qualified (SELL unfiltered)'}", "",
           "## Walk-forward (train 8 / validation 3 / test 3 sessions, rolling)", "",
           "| Fold | Train | Validation | Test | Test split | Selected | Selected test P&L | Selected test PF | "
           "Baseline test P&L | Baseline test PF |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f in a["walk_forward"]["folds"]:
        md.append(f"| {f['fold']} | {f['train']} | {f['validation']} | {f['test']} | {'/'.join(f['test_splits'])} | "
                  f"{f['selected']} | {_rs(f['selected_test']['net_pnl'])} | {_pf(f['selected_test']['profit_factor'])} "
                  f"| {_rs(f['baseline_test']['net_pnl'])} | {_pf(f['baseline_test']['profit_factor'])} |")
    md += ["", "Notes:"] + [f"- {v}" for v in a["notes"].values()]
    md += ["", "---", "Research only. TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, zero orders placed. Costs excluded."]
    return "\n".join(md) + "\n"


def candidate_markdown(a: dict) -> str:
    md = [f"# {HEADING} - CANDIDATES", "", "All candidates are pre-declared skip rules on the unchanged S2 signal.", "",
          "| Candidate | Family | BUY rule | SELL rule | Retention | Status |", "|---|---|---|---|---|---|"]
    for n in a["ranking"]:
        c, s = a["candidates"][n], a["scores"][n]
        md.append(f"| {n} | {c['family']} | {c['rules']['BUY']} | {c['rules']['SELL']} | {_pct(s['retention'])} | "
                  f"{s['status']} |")
    md += ["", "## Per-direction OOS", "", "| Candidate | BUY OOS trades | BUY OOS WR | BUY OOS PF | BUY OOS P&L | "
           "SELL OOS trades | SELL OOS WR | SELL OOS PF | SELL OOS P&L |", "|---|---|---|---|---|---|---|---|---|"]
    for n in a["ranking"]:
        b, s = a["scores"][n]["direction"]["BUY"]["OOS"], a["scores"][n]["direction"]["SELL"]["OOS"]
        md.append(f"| {n} | {b['evaluated']} | {_pct(b['win_rate'])} | {_pf(b['profit_factor'])} | {_rs(b['net_pnl'])} | "
                  f"{s['evaluated']} | {_pct(s['win_rate'])} | {_pf(s['profit_factor'])} | {_rs(s['net_pnl'])} |")
    weak = [r for r in a["buy_analysis"] + a["sell_analysis"] if r["weak_zone"] and r["split"] == "TRAIN"]
    md += ["", "## Weak zones (identified on TRAIN; VALIDATION confirmation shown; not auto-converted to filters)", ""]
    md += [f"- {r['direction']} {r['factor']} = {r['bucket']}: {r['trades']} trades, WR {_pct(r['win_rate'])}, "
           f"PF {_pf(r['profit_factor'])} -> {r['weak_zone']}" for r in weak] or ["- None met the threshold."]
    md += ["", "## Lookahead (truncation test: features recomputed from candles <= cut)", "",
           f"- Feature recomputation: {'PASS' if a['lookahead'].get('_features', {}).get('passed', True) else 'FAIL'}",
           f"- Candidates failing: {[n for n, v in a['lookahead'].items() if not n.startswith('_') and not v['passed']]}"]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ("baseline.md", "candidate_summary.csv", "candidate_summary.md", "buy_analysis.csv", "sell_analysis.csv",
             "time_analysis.csv", "regime_analysis.csv", "robustness.csv", "walk_forward.csv", "final_ranking.md",
             "dte_analysis.csv", "summary.json")
    p = {n: out_dir / n for n in names}
    p["baseline.md"].write_text(baseline_markdown(a), encoding="utf-8")
    _csv(p["candidate_summary.csv"], summary_rows(a))
    p["candidate_summary.md"].write_text(candidate_markdown(a), encoding="utf-8")
    for k in ("buy_analysis", "sell_analysis", "time_analysis", "regime_analysis", "robustness"):
        _csv(p[f"{k}.csv"], a[k])
    wf_rows = [{"row_type": "FOLD_SELECTION", "fold": f["fold"], "train": f["train"], "validation": f["validation"],
                "test": f["test"], "test_splits": "/".join(f["test_splits"]), "candidate": f["selected"],
                "test_trades": f["selected_test"]["evaluated"], "test_win_rate": f["selected_test"]["win_rate"],
                "test_profit_factor": f["selected_test"]["profit_factor"], "test_net_pnl": f["selected_test"]["net_pnl"],
                "baseline_test_trades": f["baseline_test"]["evaluated"],
                "baseline_test_net_pnl": f["baseline_test"]["net_pnl"]} for f in a["walk_forward"]["folds"]]
    for n, tests in a["walk_forward"]["per_candidate"].items():
        for f, t in zip(a["walk_forward"]["folds"], tests):
            wf_rows.append({"row_type": "CANDIDATE_TEST", "fold": f["fold"], "train": f["train"],
                            "validation": f["validation"], "test": f["test"], "test_splits": "/".join(f["test_splits"]),
                            "candidate": n, "test_trades": t["evaluated"], "test_win_rate": t["win_rate"],
                            "test_profit_factor": t["profit_factor"], "test_net_pnl": t["net_pnl"],
                            "baseline_test_trades": f["baseline_test"]["evaluated"],
                            "baseline_test_net_pnl": f["baseline_test"]["net_pnl"]})
    _csv(p["walk_forward.csv"], wf_rows)
    _csv(p["dte_analysis.csv"], a["dte"] or [{"note": "DTE study not run (--skip-dte)"}])
    p["final_ranking.md"].write_text(final_markdown(a), encoding="utf-8")
    public = {k: v for k, v in a.items() if not k.startswith("_")}
    p["summary.json"].write_text(json.dumps(sh._jsonable(public), indent=2, default=str), encoding="utf-8")
    return p
