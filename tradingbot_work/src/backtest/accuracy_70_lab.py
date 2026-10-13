"""70%+ OOS ACCURACY STRATEGY RESEARCH LAB (research only, PAPER, no orders, production S2 untouched).

Fundamentally different, pre-declared strategy families generate their own causal signals on completed NIFTY 5m
candles. Every signal goes through the unchanged S2 execution model (point-in-time nearest-expiry ATM contract, entry
at the next option candle open, stop = 1.5 x premium ATR14 from option candles up to the signal candle, 2R target,
first touch on later real option candles, 15:15 square-off, AMBIGUOUS = -1R). BUY and SELL are separate candidates.

Selection uses TRAIN + VALIDATION only; OOS is evaluated once for the selected finalists. The final score that also
uses OOS only orders the report and never picks the finalist.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.backtest.regime_classifier import ABS_TREND_RETURN
from src.backtest.research_strategies import ResearchSignal
from src.indicators.technical import atr, ema

HEADING = "70%+ OOS ACCURACY STRATEGY RESEARCH LAB"
DIRECTIONS = ("BUY", "SELL")
SPLITS = imp.SPLITS
PERIODS = (*SPLITS, "ALL")
ACCEPT, PROMISING, KEEP, REJECT = ("ACCEPT_RESEARCH_CANDIDATE", "PROMISING_NEEDS_MORE_DATA", "KEEP_CURRENT",
                                   "REJECT_RESEARCH")
NOT_ACHIEVED_TEXT = "70% OOS accuracy was not achieved with the available data without unacceptable overfitting."
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
LOOKAHEAD_FAIL = "LOOKAHEAD_FAIL"
ALL_GATES_PASS = "ALL_GATES_PASS"
QUALITY_PASS_BELOW_70 = "QUALITY_PASS_BELOW_70"
PROMISING_SMALL_SAMPLE = "PROMISING_SMALL_SAMPLE"
INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
REJECTED = "REJECTED"
STATUS_ORDER = {ALL_GATES_PASS: 0, PROMISING_SMALL_SAMPLE: 1, QUALITY_PASS_BELOW_70: 2, REJECTED: 3,
                INSUFFICIENT_SAMPLE: 4, DATA_UNAVAILABLE: 5, LOOKAHEAD_FAIL: 6}
PASS, FAIL, INFO = "PASS", "FAIL", "INFO"
TRAILING_BARS = 375  # five 75-bar sessions
SESSION_BARS = 75
STOP_ATR, TARGET_R = imp.BASELINE.stop_atr, imp.BASELINE.target_r


@dataclass(frozen=True)
class Criteria:
    target_accuracy: float = 0.70
    min_train_trades: int = 20
    min_val_trades: int = 10
    min_val_pf: float = 1.2
    wf_train_days: int = 8
    wf_val_days: int = 3
    wf_test_days: int = 3
    wf_min_train_trades: int = 10
    wf_min_val_trades: int = 5
    tier_a: int = 50
    tier_b: int = 30
    tier_c: int = 15
    min_oos_pf: float = 1.5
    max_oos_best_day_share: float = 0.50
    max_all_best_day_share: float = 0.35
    min_profitable_day_share: float = 0.50
    max_loss_streak: int = 8
    min_wf_share: float = 0.60
    min_wf_pf: float = 1.2
    min_regime_share: float = 0.60
    min_group_trades: int = 5
    suspicious_wr: float = 0.95
    oos_previously_observed: bool = True


# ------------------------------------------------------------------ strategies
@dataclass(frozen=True)
class Strategy:
    cid: str
    family: str
    direction: str
    conditions: tuple[str, ...]
    params: tuple[tuple[str, float], ...] = ()
    overlay: str | None = None
    base: str | None = None

    @property
    def p(self) -> dict[str, float]:
        return dict(self.params)

    @property
    def n_conditions(self) -> int:
        return len(self.conditions) + (1 if self.overlay else 0)

    @property
    def signal_source(self) -> str:
        return self.base or self.cid


FAMILIES = {
    "A_TREND_MOMENTUM": "EMA relation/slope/distance, RSI, candle body, consecutive candles",
    "B_BREAKOUT_MOMENTUM": "N-candle high/low break, breakout distance, ATR expansion, close strength, retest",
    "C_MEAN_REVERSION": "stretch from the session mean (TWAP proxy), RSI extreme, reversal candle, failed ORB",
    "D_VOLATILITY_REGIME": "ATR percentile, range expansion/contraction (squeeze), volatility-conditioned entries",
    "E_PRICE_ACTION": "engulfing, wick rejection, inside/outside bars, multi-candle pushes",
    "F_SESSION_STRUCTURE": "first ORB break, 30-minute ORB, previous-day high/low, gap continuation, late trend",
}

# (name, family, conditions, coarse bounded grid). Every grid point is a separate, reported candidate.
SPECS: tuple[tuple[str, str, tuple[str, ...], dict[str, tuple]], ...] = (
    ("TREND_MOMENTUM", "A_TREND_MOMENTUM", ("ema_stack", "ema_slope", "ema_dist", "rsi_mom", "body_dir"),
     {"slope": (0.0, 0.25), "rsi": (55, 60), "max_dist": (1.5,), "body": (0.5,)}),
    ("TREND_STREAK", "A_TREND_MOMENTUM", ("ema_stack", "streak"), {"streak": (3, 4)}),
    ("TREND_PULLBACK", "A_TREND_MOMENTUM", ("ema_stack", "ema_slope", "pullback_resume"), {"slope": (0.0,)}),
    ("BREAKOUT_NBAR", "B_BREAKOUT_MOMENTUM", ("nbar_break", "break_dist", "close_strength", "ema_side"),
     {"n": (6, 12), "close_pos": (0.6, 0.75), "max_break": (0.5,)}),
    ("BREAKOUT_ATR_EXPANSION", "B_BREAKOUT_MOMENTUM", ("nbar_break", "range_expansion", "close_strength"),
     {"n": (12,), "rx": (1.2,), "close_pos": (0.7,)}),
    ("BREAKOUT_RETEST", "B_BREAKOUT_MOMENTUM", ("retest",), {"n": (12,), "tol": (0.15,)}),
    ("BREAKOUT_VOLUME", "B_BREAKOUT_MOMENTUM", ("nbar_break", "volume_expansion"), {"n": (12,), "vol_mult": (1.5,)}),
    ("BREAKOUT_VWAP", "B_BREAKOUT_MOMENTUM", ("nbar_break", "vwap_side"), {"n": (12,)}),
    ("TWAP_REVERSION", "C_MEAN_REVERSION", ("twap_stretch", "rsi_extreme", "reversal_candle"),
     {"k": (1.0, 1.5), "rsi_lo": (30, 35)}),
    ("ORB_RECLAIM_FADE", "C_MEAN_REVERSION", ("orb_reclaim",), {}),
    ("VWAP_REVERSION", "C_MEAN_REVERSION", ("vwap_stretch", "reversal_candle"), {"k": (1.0,)}),
    ("VOL_EXPANSION_MOMENTUM", "D_VOLATILITY_REGIME", ("range_expansion", "body_dir", "close_strength"),
     {"rx": (1.2, 1.6), "body": (0.6,), "close_pos": (0.75,)}),
    ("VOL_SQUEEZE_BREAK", "D_VOLATILITY_REGIME", ("squeeze", "nbar_break"), {"sq": (0.4,), "n": (6,)}),
    ("HIGH_VOL_TREND", "D_VOLATILITY_REGIME", ("atr_high", "ema_stack", "body_dir"), {"atr_hi": (0.6,), "body": (0.5,)}),
    ("LOW_VOL_REVERSION", "D_VOLATILITY_REGIME", ("atr_low", "twap_stretch", "reversal_candle"),
     {"atr_lo": (0.4,), "k": (1.0,)}),
    ("VOLUME_PERCENTILE_TREND", "D_VOLATILITY_REGIME", ("volume_pctile", "ema_stack"), {"vol_pct": (0.7,)}),
    ("ENGULFING", "E_PRICE_ACTION", ("engulfing",), {}),
    ("ENGULFING_WITH_TREND", "E_PRICE_ACTION", ("engulfing", "ema_stack"), {}),
    ("WICK_REJECTION", "E_PRICE_ACTION", ("wick_rejection",), {"wick": (0.5, 0.6)}),
    ("INSIDE_BAR_BREAK", "E_PRICE_ACTION", ("inside_break",), {}),
    ("OUTSIDE_BAR", "E_PRICE_ACTION", ("outside_bar",), {"close_pos": (0.7,)}),
    ("THREE_CANDLE_PUSH", "E_PRICE_ACTION", ("streak", "body_dir"), {"streak": (3,), "body": (0.5,)}),
    ("ORB_FIRST_BREAK", "F_SESSION_STRUCTURE", ("orb_first",), {}),
    ("ORB_FIRST_BREAK_MORNING", "F_SESSION_STRUCTURE", ("orb_first", "window"), {"win_from": (930,), "win_to": (1100,)}),
    ("ORB30_FIRST_BREAK", "F_SESSION_STRUCTURE", ("orb30_first",), {}),
    ("PDH_PDL_BREAK", "F_SESSION_STRUCTURE", ("pd_break",), {}),
    ("GAP_CONTINUATION", "F_SESSION_STRUCTURE", ("gap_go", "window"),
     {"gap": (0.002,), "win_from": (930,), "win_to": (1100,)}),
    ("LATE_SESSION_TREND", "F_SESSION_STRUCTURE", ("late_trend",), {"move": (0.004,)}),
)


def _param_tag(p: dict) -> str:
    return "_".join(f"{k}{v:g}" for k, v in p.items()) if p else ""


def predeclared_strategies() -> list[Strategy]:
    """Fixed before any result is computed. One candidate per grid point and direction."""
    out = []
    for name, family, conds, grid in SPECS:
        keys = list(grid)
        for values in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            p = dict(zip(keys, (float(v) for v in values)))
            tag = _param_tag(p)
            for d in DIRECTIONS:
                cid = f"{d}_{name}" + (f"[{tag}]" if tag else "")
                out.append(Strategy(cid, family, d, conds, tuple(p.items())))
    return out


OVERLAYS: dict[str, str] = {
    "PREM_MOMENTUM": "option premium close above its close 3 candles earlier (underlying move confirmed by premium)",
    "PREM_ABOVE_VWAP": "option premium close above the session premium VWAP (option volume)",
    "PREM_VOLUME": "option candle volume >= 1.5x the previous 10 option candles",
    "NON_EXPIRY_DAY": "signal not on the contract expiry day (DTE >= 1)",
}


def overlay_strategies(bases: Iterable[Strategy]) -> list[Strategy]:
    return [Strategy(f"{b.cid}+{o}", b.family, b.direction, b.conditions, b.params, overlay=o, base=b.cid)
            for b in bases for o in OVERLAYS]


def overlay_check(name: str, o: dict | None) -> bool | None:
    if o is None:
        return None
    if name == "PREM_MOMENTUM":
        return None if o["premium_mom3"] is None else o["premium_mom3"] > 0
    if name == "PREM_ABOVE_VWAP":
        return o["premium_above_vwap"]
    if name == "PREM_VOLUME":
        return None if o["premium_vol_ratio"] is None else o["premium_vol_ratio"] >= 1.5
    if name == "NON_EXPIRY_DAY":
        return None if o["expiry_day"] is None else not o["expiry_day"]
    raise ValueError(f"unknown overlay {name}")


CONDITION_TEXT = {
    "ema_stack": "BUY close>EMA20>EMA50 / SELL close<EMA20<EMA50",
    "ema_slope": "EMA20 3-bar slope / ATR beyond +-{slope:g}",
    "ema_dist": "0 <= signed (close-EMA20)/ATR <= {max_dist:g}",
    "rsi_mom": "RSI14 >= {rsi:g} (BUY) / <= 100-{rsi:g} (SELL)",
    "body_dir": "candle in trade direction with body >= {body:g} of range",
    "streak": ">= {streak:g} consecutive candles in trade direction",
    "pullback_resume": "previous candle pulled back to EMA20, current close beyond previous high/low",
    "nbar_break": "close beyond the previous {n:g}-candle high/low (same session)",
    "break_dist": "breakout distance <= {max_break:g} ATR",
    "close_strength": "close in the top/bottom {close_pos:g} of the candle range",
    "ema_side": "close on the trade side of EMA20",
    "range_expansion": "candle range >= {rx:g} x ATR14",
    "retest": "{n:g}-candle breakout in the last 6 candles, retest within {tol:g} ATR and close back beyond the level",
    "volume_expansion": "underlying volume >= {vol_mult:g}x 20-candle average (DATA_UNAVAILABLE: index volume 0)",
    "vwap_side": "close on the trade side of VWAP (DATA_UNAVAILABLE: index volume 0)",
    "vwap_stretch": "close >= {k:g} ATR away from VWAP (DATA_UNAVAILABLE: index volume 0)",
    "twap_stretch": "close >= {k:g} ATR beyond the session TWAP (volume-free mean, VWAP proxy)",
    "rsi_extreme": "RSI14 <= {rsi_lo:g} (BUY) / >= 100-{rsi_lo:g} (SELL)",
    "reversal_candle": "reversal candle in trade direction closing in the outer 40% of its range",
    "orb_reclaim": "previous close outside the ORB, current close back inside (failed breakout fade)",
    "squeeze": "previous 6 candle ranges all <= {sq:g} trailing percentile",
    "atr_high": "ATR%/close >= {atr_hi:g} trailing 5-session percentile",
    "atr_low": "ATR%/close <= {atr_lo:g} trailing 5-session percentile",
    "volume_pctile": "volume >= {vol_pct:g} trailing percentile (DATA_UNAVAILABLE: index volume 0)",
    "engulfing": "engulfing candle in trade direction",
    "wick_rejection": "wick >= {wick:g} of range sweeping the 6-candle low/high, close in the outer 40%",
    "inside_break": "previous candle inside its predecessor, current close beyond it",
    "outside_bar": "outside candle in trade direction closing in the outer {close_pos:g}",
    "orb_first": "first close beyond the 09:15-09:30 ORB of the session",
    "orb30_first": "first close beyond the 09:15-09:45 range of the session",
    "pd_break": "first close beyond the previous session high/low",
    "gap_go": "opening gap >= {gap:g} in trade direction and price still beyond the open",
    "late_trend": "after 11:00, day move >= {move:g} in trade direction and a new session extreme",
    "window": "signal candle from {win_from:04.0f} to before {win_to:04.0f}",
}


def describe(s: Strategy) -> str:
    p = s.p
    txt = "; ".join(CONDITION_TEXT[c].format(**p) for c in s.conditions)
    return txt + (f"; overlay: {OVERLAYS[s.overlay]}" if s.overlay else "")


# ------------------------------------------------------------------ causal bar features
FEATURE_COLUMNS = ("close", "atr14", "ema20", "ema50", "rsi14", "ema9", "ema20_slope_atr", "dist_ema20_atr",
                   "body_frac", "close_pos", "upper_wick", "lower_wick", "range_atr", "up_streak", "down_streak",
                   "hh6", "ll6", "hh12", "ll12", "atr_pctile", "range_pctile", "squeeze6", "twap", "dist_twap_atr",
                   "intraday_move", "pdh", "pdl", "pdc", "gap_pct", "prev_high", "prev_low", "prev2_high",
                   "prev2_low", "prev_ema20", "day_high_before", "day_low_before", "orb_high", "orb_low",
                   "orb30_high", "orb30_low", "sign_changes", "vwap", "vol_ratio", "vol_pctile")


def _streak(flag: pd.Series, day: pd.Series) -> pd.Series:
    flag = flag.fillna(False).astype(bool)
    block = (~flag).astype(int).groupby(day).cumsum()
    return flag.astype(int).groupby([day, block]).cumsum()


def trailing_pctile(s: pd.Series, window: int = TRAILING_BARS) -> pd.Series:
    """Share of the previous ``window`` values strictly below the current value (current bar excluded)."""
    x = s.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) > window:
        w = sliding_window_view(x[:-1], window)
        cur = x[window:]
        valid = ~np.isnan(w).any(axis=1) & ~np.isnan(cur)
        out[window:] = np.where(valid, (w < cur[:, None]).mean(axis=1), np.nan)
    return pd.Series(out, index=s.index)


def family_bars(spot: pd.DataFrame) -> pd.DataFrame:
    """Per-bar features known at each completed candle's close (rows ordered by IST timestamp)."""
    f = lab.bar_features(spot).reset_index()
    day = f["day"]
    g = f.groupby("day", sort=False)
    rng = f["high"] - f["low"]
    pos = rng.where(rng > 0)
    t = f["timestamp"].dt
    f["minute"] = t.hour * 60 + t.minute
    f["ema9"] = ema(f["close"], 9)
    f["ema20_slope_atr"] = (f["ema20"] - f["ema20"].shift(3)) / f["atr14"]
    f["dist_ema20_atr"] = (f["close"] - f["ema20"]) / f["atr14"]
    f["body_frac"] = (f["close"] - f["open"]).abs() / pos
    f["close_pos"] = (f["close"] - f["low"]) / pos
    f["upper_wick"] = (f["high"] - f[["open", "close"]].max(axis=1)) / pos
    f["lower_wick"] = (f[["open", "close"]].min(axis=1) - f["low"]) / pos
    f["range_atr"] = rng / f["atr14"]
    f["up_streak"] = _streak(f["close"] > f["open"], day)
    f["down_streak"] = _streak(f["close"] < f["open"], day)
    for n in (6, 12):
        f[f"hh{n}"] = g["high"].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=n).max())
        f[f"ll{n}"] = g["low"].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=n).min())
    for k, sfx in ((1, ""), (2, "2")):
        f[f"prev{sfx}_high"] = g["high"].shift(k)
        f[f"prev{sfx}_low"] = g["low"].shift(k)
    f["prev_ema20"] = g["ema20"].shift(1)
    f["atr_pctile"] = trailing_pctile(f["atr14"] / f["close"])
    f["range_pctile"] = trailing_pctile(rng)
    f["squeeze6"] = f["range_pctile"].groupby(day).transform(lambda s: s.shift(1).rolling(6, min_periods=6).max())
    f["vol_pctile"] = trailing_pctile(f["volume"].where(f["volume"] > 0))
    tp = (f["high"] + f["low"] + f["close"]) / 3
    f["twap"] = tp.groupby(day).cumsum() / (g.cumcount() + 1)
    f["dist_twap_atr"] = (f["close"] - f["twap"]) / f["atr14"]
    f["intraday_move"] = f["close"] / f["day_open"] - 1
    daily = g.agg(dh=("high", "max"), dl=("low", "min"), dc=("close", "last")).shift(1)
    f["pdh"], f["pdl"], f["pdc"] = day.map(daily["dh"]), day.map(daily["dl"]), day.map(daily["dc"])
    f["gap_pct"] = f["day_open"] / f["pdc"] - 1
    f["day_high_before"] = g["high"].transform(lambda s: s.cummax().shift(1))
    f["day_low_before"] = g["low"].transform(lambda s: s.cummin().shift(1))
    tt = t.time
    in30 = (tt >= time(9, 15)) & (tt < time(9, 45))
    done = tt >= time(9, 45)
    f["orb30_high"] = f["high"].where(in30).groupby(day).transform("max").where(done)
    f["orb30_low"] = f["low"].where(in30).groupby(day).transform("min").where(done)
    return f


def _have(f: pd.DataFrame, *cols: str) -> pd.Series:
    return f[list(cols)].notna().all(axis=1)


def _lvl(f, up, p):
    return f[f"hh{int(p['n'])}"] if up else f[f"ll{int(p['n'])}"]


def _first_of_day(flag: pd.Series, day: pd.Series) -> pd.Series:
    flag = flag.fillna(False).astype(bool)
    return flag & (flag.astype(int).groupby(day).cumsum() == 1)


def _dir_candle(f, up):
    return (f["close"] > f["open"]) if up else (f["close"] < f["open"])


def _c_close_strength(f, up, p):
    return ((f["close_pos"] >= p["close_pos"]) if up else (f["close_pos"] <= 1 - p["close_pos"])), _have(f, "close_pos")


def _c_retest(f, up, p):
    lvl = _lvl(f, up, p)
    brk = ((f["close"] > lvl) if up else (f["close"] < lvl)).fillna(False)
    broken = lvl.where(brk).groupby(f["day"]).transform(lambda s: s.ffill().shift(1))
    recent = brk.astype(float).groupby(f["day"]).transform(lambda s: s.shift(1).rolling(6, min_periods=1).max()) > 0
    tol = p["tol"] * f["atr14"]
    if up:
        v = recent & (f["low"] <= broken + tol) & (f["close"] > broken) & (f["close"] > f["open"])
    else:
        v = recent & (f["high"] >= broken - tol) & (f["close"] < broken) & (f["close"] < f["open"])
    return v, broken.notna() & f["atr14"].notna()


def _hhmm(v: float) -> int:
    v = int(v)
    return (v // 100) * 60 + v % 100


CONDITIONS: dict[str, Callable[[pd.DataFrame, bool, dict], tuple[pd.Series, pd.Series]]] = {
    "ema_stack": lambda f, up, p: ((((f["close"] > f["ema20"]) & (f["ema20"] > f["ema50"])) if up else
                                    ((f["close"] < f["ema20"]) & (f["ema20"] < f["ema50"]))),
                                   _have(f, "close", "ema20", "ema50")),
    "ema_slope": lambda f, up, p: ((1 if up else -1) * f["ema20_slope_atr"] > p["slope"], _have(f, "ema20_slope_atr")),
    "ema_dist": lambda f, up, p: (((1 if up else -1) * f["dist_ema20_atr"]).between(0, p["max_dist"]),
                                  _have(f, "dist_ema20_atr")),
    "rsi_mom": lambda f, up, p: ((f["rsi14"] >= p["rsi"]) if up else (f["rsi14"] <= 100 - p["rsi"]), _have(f, "rsi14")),
    "body_dir": lambda f, up, p: (_dir_candle(f, up) & (f["body_frac"] >= p["body"]), _have(f, "body_frac")),
    "streak": lambda f, up, p: (f["up_streak" if up else "down_streak"] >= p["streak"],
                                _have(f, "up_streak" if up else "down_streak")),
    "pullback_resume": lambda f, up, p: (
        ((f["prev_close"] < f["prev_open"]) & (f["prev_low"] <= f["prev_ema20"]) & (f["close"] > f["prev_high"])) if up
        else ((f["prev_close"] > f["prev_open"]) & (f["prev_high"] >= f["prev_ema20"]) & (f["close"] < f["prev_low"])),
        _have(f, "prev_close", "prev_open", "prev_low", "prev_high", "prev_ema20")),
    "nbar_break": lambda f, up, p: (((f["close"] > _lvl(f, up, p)) if up else (f["close"] < _lvl(f, up, p))),
                                    _lvl(f, up, p).notna()),
    "break_dist": lambda f, up, p: ((f["close"] - _lvl(f, up, p)) * (1 if up else -1) / f["atr14"] <= p["max_break"],
                                    _lvl(f, up, p).notna() & f["atr14"].notna()),
    "close_strength": _c_close_strength,
    "ema_side": lambda f, up, p: ((f["close"] > f["ema20"]) if up else (f["close"] < f["ema20"]), _have(f, "ema20")),
    "range_expansion": lambda f, up, p: (f["range_atr"] >= p["rx"], _have(f, "range_atr")),
    "retest": _c_retest,
    "volume_expansion": lambda f, up, p: (f["vol_ratio"] >= p["vol_mult"], _have(f, "vol_ratio")),
    "vwap_side": lambda f, up, p: ((f["close"] > f["vwap"]) if up else (f["close"] < f["vwap"]), _have(f, "vwap")),
    "vwap_stretch": lambda f, up, p: ((((f["close"] - f["vwap"]) / f["atr14"] <= -p["k"]) if up else
                                       ((f["close"] - f["vwap"]) / f["atr14"] >= p["k"])), _have(f, "vwap", "atr14")),
    "twap_stretch": lambda f, up, p: ((f["dist_twap_atr"] <= -p["k"]) if up else (f["dist_twap_atr"] >= p["k"]),
                                      _have(f, "dist_twap_atr")),
    "rsi_extreme": lambda f, up, p: ((f["rsi14"] <= p["rsi_lo"]) if up else (f["rsi14"] >= 100 - p["rsi_lo"]),
                                     _have(f, "rsi14")),
    "reversal_candle": lambda f, up, p: ((_dir_candle(f, up) & ((f["close_pos"] >= 0.6) if up else
                                                                 (f["close_pos"] <= 0.4))), _have(f, "close_pos")),
    "orb_reclaim": lambda f, up, p: (
        ((f["prev_close"] < f["prev_orb_low"]) & (f["close"] > f["orb_low"])) if up else
        ((f["prev_close"] > f["prev_orb_high"]) & (f["close"] < f["orb_high"])),
        _have(f, "prev_close", "prev_orb_low", "prev_orb_high", "orb_low", "orb_high")),
    "squeeze": lambda f, up, p: (f["squeeze6"] <= p["sq"], _have(f, "squeeze6")),
    "atr_high": lambda f, up, p: (f["atr_pctile"] >= p["atr_hi"], _have(f, "atr_pctile")),
    "atr_low": lambda f, up, p: (f["atr_pctile"] <= p["atr_lo"], _have(f, "atr_pctile")),
    "volume_pctile": lambda f, up, p: (f["vol_pctile"] >= p["vol_pct"], _have(f, "vol_pctile")),
    "engulfing": lambda f, up, p: (
        ((f["close"] > f["open"]) & (f["prev_close"] < f["prev_open"]) & (f["close"] >= f["prev_open"])
         & (f["open"] <= f["prev_close"])) if up else
        ((f["close"] < f["open"]) & (f["prev_close"] > f["prev_open"]) & (f["close"] <= f["prev_open"])
         & (f["open"] >= f["prev_close"])), _have(f, "prev_close", "prev_open")),
    "wick_rejection": lambda f, up, p: (
        ((f["lower_wick"] >= p["wick"]) & (f["close_pos"] >= 0.6) & (f["low"] <= f["ll6"])) if up else
        ((f["upper_wick"] >= p["wick"]) & (f["close_pos"] <= 0.4) & (f["high"] >= f["hh6"])),
        _have(f, "lower_wick", "upper_wick", "ll6", "hh6")),
    "inside_break": lambda f, up, p: (
        (f["prev_high"] < f["prev2_high"]) & (f["prev_low"] > f["prev2_low"])
        & ((f["close"] > f["prev_high"]) if up else (f["close"] < f["prev_low"])),
        _have(f, "prev_high", "prev_low", "prev2_high", "prev2_low")),
    "outside_bar": lambda f, up, p: (
        (f["high"] > f["prev_high"]) & (f["low"] < f["prev_low"]) & _dir_candle(f, up)
        & _c_close_strength(f, up, p)[0], _have(f, "prev_high", "prev_low", "close_pos")),
    "orb_first": lambda f, up, p: (_first_of_day((f["close"] > f["orb_high"]) if up else (f["close"] < f["orb_low"]),
                                                 f["day"]), _have(f, "orb_high", "orb_low")),
    "orb30_first": lambda f, up, p: (_first_of_day((f["close"] > f["orb30_high"]) if up else
                                                   (f["close"] < f["orb30_low"]), f["day"]),
                                     _have(f, "orb30_high", "orb30_low")),
    "pd_break": lambda f, up, p: (_first_of_day((f["close"] > f["pdh"]) if up else (f["close"] < f["pdl"]), f["day"]),
                                  _have(f, "pdh", "pdl")),
    "gap_go": lambda f, up, p: (((1 if up else -1) * f["gap_pct"] >= p["gap"])
                                & ((1 if up else -1) * (f["close"] - f["day_open"]) > 0), _have(f, "gap_pct", "day_open")),
    "late_trend": lambda f, up, p: (
        (f["minute"] >= 660) & ((1 if up else -1) * f["intraday_move"] >= p["move"])
        & ((f["close"] > f["day_high_before"]) if up else (f["close"] < f["day_low_before"])),
        _have(f, "intraday_move", "day_high_before", "day_low_before")),
    "window": lambda f, up, p: (f["minute"].between(_hhmm(p["win_from"]), _hhmm(p["win_to"]) - 1),
                                pd.Series(True, index=f.index)),
}


def default_allowed(ts: pd.Timestamp) -> bool:
    return time(9, 30) <= ts.time() < time(14, 30)


def allowed_mask(f: pd.DataFrame, allowed: Callable[[pd.Timestamp], bool] = default_allowed,
                 days: set[str] | None = None) -> pd.Series:
    ok = f["timestamp"].map(lambda ts: bool(allowed(ts)))
    if days is not None:
        ok &= f["day"].map(lambda d: d.isoformat() in days)
    return ok.astype(bool)


def strategy_signals(f: pd.DataFrame, s: Strategy, mask: pd.Series) -> tuple[list[tuple[str, str]], float]:
    """Edge-triggered signal keys (first candle on which every condition holds) and the input availability share."""
    up = s.direction == "BUY"
    cond = pd.Series(True, index=f.index)
    avail = pd.Series(True, index=f.index)
    for name in s.conditions:
        v, a = CONDITIONS[name](f, up, s.p)
        cond &= v.fillna(False).astype(bool)
        avail &= a.fillna(False).astype(bool)
    cond &= avail
    prev = cond.groupby(f["day"]).shift(1, fill_value=False).astype(bool)
    trig = cond & ~prev & mask
    n = int(mask.sum())
    keys = [(ts.isoformat(), s.direction) for ts in f.loc[trig, "timestamp"]]
    return keys, (float((avail & mask).sum()) / n if n else 0.0)


def union_signals(f: pd.DataFrame, keys: Iterable[tuple[str, str]], underlying: str) -> list[ResearchSignal]:
    look = {ts.isoformat(): (ts, float(c)) for ts, c in zip(f["timestamp"], f["close"])}
    out = []
    for t, d in sorted(set(keys)):
        ts, close = look[t]
        out.append(ResearchSignal(ts.to_pydatetime(), d, "R70", underlying, close, {}))
    return out


# ------------------------------------------------------------------ option features (premium candles <= signal)
def option_features(ctx: dict, premium: pd.DataFrame | None = None) -> dict[str, Any] | None:
    prem = ctx.get("premium") if premium is None else premium
    if ctx.get("data_status") != "OK" or prem is None or isinstance(prem, str) or prem.empty:
        return None
    ts = ctx["signal_time"]
    hist = prem[prem["timestamp"] <= ts]
    if hist.empty or hist["timestamp"].iloc[-1] != ts:
        return None
    last = hist.iloc[-1]
    close, opn, hi, lo = (float(last[k]) for k in ("close", "open", "high", "low"))
    c = hist["close"].astype(float)
    mom = float(c.iloc[-1] / c.iloc[-4] - 1) if len(c) >= 4 and c.iloc[-4] > 0 else None
    pv = vr = None
    if "volume" in hist:
        day = hist[hist["timestamp"].dt.date == ts.date()]
        vol = day["volume"].astype(float)
        if vol.sum() > 0:
            pv = float((((day["high"] + day["low"] + day["close"]) / 3) * vol).sum() / vol.sum())
        if len(hist) >= 11:
            avg = float(hist["volume"].iloc[-11:-1].astype(float).mean())
            vr = float(last["volume"]) / avg if avg > 0 else None
    patr, dte = ctx.get("premium_atr"), ctx.get("dte")
    return {"premium_close": close, "premium_mom3": mom,
            "premium_body": abs(close - opn) / (hi - lo) if hi > lo else None, "premium_bull": close > opn,
            "premium_vwap": pv, "premium_above_vwap": None if pv is None else close > pv,
            "premium_vol_ratio": vr, "premium_atr_pct": patr / close * 100 if patr and close > 0 else None,
            "dte": dte, "expiry_day": None if dte is None else dte == 0}


# ------------------------------------------------------------------ candidate rows
def row_key(r: dict) -> tuple[str, str]:
    return (str(r["signal_time"]), r["direction"])


def candidate_rows(s: Strategy, keys: Sequence[tuple[str, str]], row_by_key: dict, opt: dict) -> list[dict]:
    """Fresh row copies for one candidate; only its own direction's signals are ever included."""
    out = []
    for k in keys:
        if k[1] != s.direction:
            raise ValueError(f"{s.cid}: foreign-direction signal {k}")
        base = row_by_key.get(k)
        if base is None:
            continue
        row = dict(base, strategy=s.cid)
        if s.overlay and row["state"] == sh.EVALUATED:
            ok = overlay_check(s.overlay, opt.get(k))
            if ok is not True:
                row.update(decision="SKIPPED", state=sh.SKIPPED, outcome=None, pnl=None, r_multiple=None,
                           skip_reason=f"{'OPTION_FEATURE_UNAVAILABLE' if ok is None else 'FILTERED'}:{s.overlay}")
        out.append(row)
    return out


# ------------------------------------------------------------------ metrics
def _num(v, default: float = 0.0) -> float:
    if v is None:
        return default
    return 1e9 if isinstance(v, float) and math.isinf(v) else float(v)


def _cap(v: float) -> float:
    return max(0.0, min(1.0, v))


def day_profile(rows: Sequence[dict]) -> dict[str, Any]:
    daily: dict[str, float] = {}
    for r in rows:
        if r["state"] == sh.EVALUATED:
            daily[r["session_date"]] = daily.get(r["session_date"], 0.0) + float(r["pnl"])
    if not daily:
        return {"days": 0, "best_day": None, "best_day_pnl": None, "worst_day": None, "worst_day_pnl": None,
                "median_day_pnl": None, "profitable_day_share": None, "best_day_share_of_net": None,
                "best_day_share_of_gross_profit": None, "daily": {}}
    net = sum(daily.values())
    gp = sum(v for v in daily.values() if v > 0)
    best = max(daily.items(), key=lambda kv: kv[1])
    worst = min(daily.items(), key=lambda kv: kv[1])
    return {"days": len(daily), "best_day": best[0], "best_day_pnl": round(best[1], 2), "worst_day": worst[0],
            "worst_day_pnl": round(worst[1], 2), "median_day_pnl": round(median(daily.values()), 2),
            "profitable_day_share": sum(v > 0 for v in daily.values()) / len(daily),
            "best_day_share_of_net": best[1] / net if net > 0 else None,
            "best_day_share_of_gross_profit": best[1] / gp if gp > 0 and best[1] > 0 else None,
            "daily": {d: round(v, 2) for d, v in sorted(daily.items())}}


def wf_metrics(rows: Sequence[dict], folds: Sequence[dict], min_fold_trades: int = 5) -> dict[str, Any]:
    per = []
    for i, f in enumerate(folds, 1):
        m = lab.perf(lab.period_rows(rows, f["test"]))
        per.append({"fold": i, "train": f"{f['train'][0]}..{f['train'][-1]}",
                    "validation": f"{f['validation'][0]}..{f['validation'][-1]}",
                    "test": f"{f['test'][0]}..{f['test'][-1]}", "evaluated": m["evaluated"], "wins": m["wins"],
                    "win_rate": m["win_rate"], "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"]})
    pooled = lab.perf([r for f in folds for r in lab.period_rows(rows, f["test"])])
    traded = [p for p in per if p["evaluated"]]
    big = [p["win_rate"] for p in per if p["evaluated"] >= min_fold_trades]
    return {"folds": per, "wf_accuracy": pooled["win_rate"], "wf_trades": pooled["evaluated"],
            "wf_pf": pooled["profit_factor"], "wf_pnl": pooled["net_pnl"],
            "wf_profitable_share": sum(p["net_pnl"] > 0 for p in traded) / len(traded) if traded else None,
            "wf_consistency": 1 - (max(big) - min(big)) if len(big) >= 2 else None}


def selection_score(train: dict, val: dict, c: Criteria, min_train: int, min_val: int) -> float | None:
    """TRAIN + VALIDATION only. None = not eligible for selection."""
    if train["evaluated"] < min_train or val["evaluated"] < min_val:
        return None
    if not (train["net_pnl"] > 0 and val["net_pnl"] > 0 and _num(val["profit_factor"]) >= c.min_val_pf):
        return None
    t = c.target_accuracy
    return round(0.60 * _cap(val["win_rate"] / t) + 0.25 * _cap(_num(val["profit_factor"]) / 2.0)
                 + 0.15 * _cap(train["win_rate"] / t), 6)


def sample_tier(n: int, c: Criteria) -> str:
    return "A" if n >= c.tier_a else "B" if n >= c.tier_b else "C" if n >= c.tier_c else "D"


QUALITY_GATES = ("lookahead_pass", "oos_sample_meaningful", "oos_pf_1_5", "oos_pnl_positive", "oos_drawdown_ok",
                 "no_single_day_dependency", "no_single_trade_dependency", "losing_streak_ok",
                 "validation_profitable", "walk_forward_stable", "day_stable", "regime_stable", "not_suspicious")
SAMPLE_GATES = ("oos_sample_meaningful", "walk_forward_stable", "regime_stable")


def gates(s: dict, c: Criteria) -> dict[str, bool]:
    o, v, a = s["OOS"], s["VALIDATION"], s["ALL"]
    wf = s["walk_forward"]
    sus = any(m["evaluated"] >= 10 and _num(m["win_rate"]) >= c.suspicious_wr for m in (v, o))
    return {
        "lookahead_pass": bool(s["lookahead_passed"]),
        "oos_sample_meaningful": o["evaluated"] >= c.tier_b,
        "oos_accuracy_70": _num(o["win_rate"]) >= c.target_accuracy,
        "oos_pf_1_5": _num(o["profit_factor"]) >= c.min_oos_pf,
        "oos_pnl_positive": o["net_pnl"] > 0,
        "oos_drawdown_ok": o["net_pnl"] > 0 and o["max_drawdown"] <= o["net_pnl"],
        "no_single_day_dependency": (o["net_pnl"] > 0 and o["best_day_share"] is not None
                                     and o["best_day_share"] <= c.max_oos_best_day_share and o["pnl_excl_best_day"] > 0),
        "no_single_trade_dependency": o["pnl_excl_largest_win"] > 0,
        "losing_streak_ok": a["max_consecutive_losses"] <= c.max_loss_streak,
        "validation_profitable": (v["evaluated"] >= c.min_val_trades and v["net_pnl"] > 0
                                  and _num(v["profit_factor"]) > 1),
        "walk_forward_stable": (_num(wf["wf_profitable_share"]) >= c.min_wf_share
                                and _num(wf["wf_pf"]) >= c.min_wf_pf),
        "day_stable": (_num(s["day_all"]["profitable_day_share"]) >= c.min_profitable_day_share
                       and s["day_all"]["best_day_share_of_net"] is not None
                       and s["day_all"]["best_day_share_of_net"] <= c.max_all_best_day_share),
        "regime_stable": _num(s["regime_share"]) >= c.min_regime_share,
        "not_suspicious": not sus,
    }


def candidate_status(s: dict, c: Criteria) -> tuple[str, list[str]]:
    if s["data_status"] == DATA_UNAVAILABLE:
        return DATA_UNAVAILABLE, ["required input never available (index volume is zero)"]
    g = s["gates"]
    if not g["lookahead_pass"]:
        return LOOKAHEAD_FAIL, ["lookahead_pass"]
    failed = [k for k in QUALITY_GATES if not g[k]]
    if s["tier"] == "D":
        return INSUFFICIENT_SAMPLE, [f"oos_trades<{c.tier_c}"] + failed
    if not failed:
        return (ALL_GATES_PASS, []) if g["oos_accuracy_70"] else (QUALITY_PASS_BELOW_70, ["oos_accuracy_70"])
    core = g["oos_accuracy_70"] and g["oos_pf_1_5"] and g["oos_pnl_positive"]
    if core and set(failed) <= set(SAMPLE_GATES):
        return PROMISING_SMALL_SAMPLE, failed
    return REJECTED, failed + ([] if g["oos_accuracy_70"] else ["oos_accuracy_70"])


SCORE_FORMULA = (
    "score = 100 x (0.30 x min(OOS_accuracy / 0.70, 1) + 0.20 x min(OOS_PF / 2.0, 1) "
    "+ 0.15 x [0.5 x min(VAL_accuracy / 0.70, 1) + 0.5 x min(VAL_PF / 2.0, 1)] (0 if VAL P&L <= 0) "
    "+ 0.15 x WF_profitable_window_share + 0.10 x ALL_profitable_day_share x (1 if best-day share of net P&L <= 0.35 "
    "else 0.5) + 0.05 x regime_positive_share + 0.05 x max(0, 1 - 0.15 x (conditions - 1))); "
    "capped at 40 when the OOS gate fails (OOS trades < 30, OOS PF < 1.5 or OOS P&L <= 0); "
    "0 for DATA_UNAVAILABLE / LOOKAHEAD_FAIL. The score only orders the report; finalists are chosen by the "
    "TRAIN+VALIDATION selection score.")
SELECTION_FORMULA = (
    "selection = 0.60 x min(VAL_accuracy / 0.70, 1) + 0.25 x min(VAL_PF / 2.0, 1) + 0.15 x min(TRAIN_accuracy / 0.70, 1); "
    "eligible only with TRAIN >= 20 and VALIDATION >= 10 evaluated trades, TRAIN and VALIDATION P&L > 0 and "
    "VALIDATION PF >= 1.2 (walk-forward folds: >= 10 / >= 5 trades). Ties: fewer conditions, then name.")


def final_score(s: dict, c: Criteria) -> float:
    if s["status"] in (DATA_UNAVAILABLE, LOOKAHEAD_FAIL):
        return 0.0
    o, v, g = s["OOS"], s["VALIDATION"], s["gates"]
    t = c.target_accuracy
    val = 0.5 * _cap(_num(v["win_rate"]) / t) + 0.5 * _cap(_num(v["profit_factor"]) / 2.0)
    day = _num(s["day_all"]["profitable_day_share"]) * (
        1.0 if s["day_all"]["best_day_share_of_net"] is not None
        and s["day_all"]["best_day_share_of_net"] <= c.max_all_best_day_share else 0.5)
    score = 100 * (0.30 * _cap(_num(o["win_rate"]) / t) + 0.20 * _cap(_num(o["profit_factor"]) / 2.0)
                   + 0.15 * (val if v["net_pnl"] > 0 else 0.0)
                   + 0.15 * _num(s["walk_forward"]["wf_profitable_share"]) + 0.10 * day
                   + 0.05 * _num(s["regime_share"]) + 0.05 * max(0.0, 1 - 0.15 * (s["n_conditions"] - 1)))
    if not (g["oos_sample_meaningful"] and g["oos_pf_1_5"] and g["oos_pnl_positive"]):
        score = min(score, 40.0)
    return round(score, 2)


def regime_labels(bar: pd.Series | None, opt: dict | None) -> list[str]:
    """Causal labels at the signal candle (bars <= signal)."""
    out = []
    if bar is not None:
        mv, sc, ap = bar.get("intraday_move"), bar.get("sign_changes"), bar.get("atr_pctile")
        if pd.notna(mv):
            out.append("trending" if abs(mv) >= ABS_TREND_RETURN else "sideways")
        if pd.notna(sc):
            out.append("choppy" if sc >= lab.CHOPPY_SIGN_CHANGES else "non_choppy")
        if pd.notna(ap):
            out.append("high_volatility" if ap >= 0.5 else "low_volatility")
    if opt is not None and opt.get("premium_vol_ratio") is not None:
        out.append("strong_option_volume" if opt["premium_vol_ratio"] >= 1.0 else "weak_option_volume")
    return out


def score_rows(rows: Sequence[dict], periods: dict[str, Sequence[str]], folds: Sequence[dict],
               labels: dict, c: Criteria) -> dict[str, Any]:
    m = {p: lab.perf(lab.period_rows(rows, d)) for p, d in periods.items()}
    share, groups = lab.group_stability(rows, lambda r: labels.get(row_key(r), []), c.min_group_trades)
    return {**m, "walk_forward": wf_metrics(rows, folds), "regime_share": share, "regimes": groups,
            "day_all": day_profile(lab.period_rows(rows, periods["ALL"])),
            "day_oos": day_profile(lab.period_rows(rows, periods["OOS"]))}


def pick(scores: dict[str, dict], names: Iterable[str]) -> str | None:
    pool = [n for n in names if scores[n]["selection_score"] is not None]
    return min(pool, key=lambda n: (-scores[n]["selection_score"], scores[n]["n_conditions"], n)) if pool else None


def wf_procedure(names: Sequence[str], rows: dict[str, list[dict]], n_cond: dict[str, int], folds: Sequence[dict],
                 c: Criteria) -> dict[str, Any]:
    """Per fold: select on the fold's TRAIN+VALIDATION days only, then record the next (test) days."""
    out, tests = [], []
    for i, f in enumerate(folds, 1):
        cand = {}
        for n in names:
            t, v = lab.perf(lab.period_rows(rows[n], f["train"])), lab.perf(lab.period_rows(rows[n], f["validation"]))
            sc = selection_score(t, v, c, c.wf_min_train_trades, c.wf_min_val_trades)
            if sc is not None:
                cand[n] = sc
        chosen = min(cand, key=lambda n: (-cand[n], n_cond[n], n)) if cand else None
        test_rows = lab.period_rows(rows[chosen], f["test"]) if chosen else []
        tests += test_rows
        m = lab.perf(test_rows)
        out.append({"fold": i, "test": f"{f['test'][0]}..{f['test'][-1]}", "selected": chosen,
                    "evaluated": m["evaluated"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                    "net_pnl": m["net_pnl"]})
    pooled = lab.perf(tests)
    traded = [o for o in out if o["evaluated"]]
    return {"folds": out, "wf_accuracy": pooled["win_rate"], "wf_trades": pooled["evaluated"],
            "wf_pf": pooled["profit_factor"], "wf_pnl": pooled["net_pnl"],
            "wf_profitable_share": sum(o["net_pnl"] > 0 for o in traded) / len(traded) if traded else None}


def recommend(finalists: dict[str, str | None], scores: dict[str, dict], integrity_ok: bool) -> str:
    if not integrity_ok:
        return REJECT
    st = [scores[n]["status"] for n in finalists.values() if n]
    if ALL_GATES_PASS in st:
        return ACCEPT
    if PROMISING_SMALL_SAMPLE in st:
        return PROMISING
    if not any(scores[n]["OOS"]["net_pnl"] > 0 for n in finalists.values() if n):
        return REJECT
    return KEEP


# ------------------------------------------------------------------ lookahead / integrity
def _ist_series(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values)
    return ts.dt.tz_localize(imp.IST) if ts.dt.tz is None else ts.dt.tz_convert(imp.IST)


def feature_truncation_test(spot: pd.DataFrame, strategies: Sequence[Strategy], days: Sequence[str],
                            allowed: Callable = default_allowed,
                            cut_times: Sequence[str] = ("11:00", "13:00")) -> dict[str, Any]:
    """Recompute every feature and signal from candles <= cut; nothing at or before the cut may change."""
    ts_all = _ist_series(spot["timestamp"])
    full = family_bars(spot)
    fmask = allowed_mask(full, allowed)
    full_sig = {s.cid: set(strategy_signals(full, s, fmask)[0]) for s in strategies}
    res = {"cuts": 0, "feature_failures": [], "signal_failures": []}
    for d in days:
        dd = date.fromisoformat(d)
        for t in cut_times:
            cut = pd.Timestamp(datetime.combine(dd, time.fromisoformat(t)), tz=imp.IST)
            tb = family_bars(spot[ts_all <= cut].reset_index(drop=True))
            a = full[(full["timestamp"] <= cut) & (full["day"] == dd)].reset_index(drop=True)
            b = tb[tb["day"] == dd].reset_index(drop=True)
            res["cuts"] += 1
            same = len(a) == len(b) and (a["timestamp"] == b["timestamp"]).all() and all(
                np.allclose(a[col].astype(float), b[col].astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
                for col in FEATURE_COLUMNS)
            if not same:
                res["feature_failures"].append(cut.isoformat())
            tmask = allowed_mask(tb, allowed)
            for s in strategies:
                got = {k for k in strategy_signals(tb, s, tmask)[0] if k[0][:10] == d}
                want = {k for k in full_sig[s.cid] if k[0][:10] == d and pd.Timestamp(k[0]) <= cut}
                if got != want:
                    res["signal_failures"].append(f"{cut.isoformat()}|{s.cid}")
    res["passed"] = bool(res["cuts"]) and not res["feature_failures"] and not res["signal_failures"]
    return res


def option_feature_mutation_test(contexts: Sequence[dict], atr_period: int = 14, seed: int = 23) -> dict[str, Any]:
    """Option features and premium ATR must not change when every premium candle after the signal is scrambled."""
    checked, failures = 0, []
    for i, ctx in enumerate(contexts):
        if ctx.get("data_status") != "OK":
            continue
        prem = ctx["premium"]
        mut = imp.mutate_after(prem, ctx["signal_time"], seed + i)
        trunc = prem[prem["timestamp"] <= ctx["signal_time"]].reset_index(drop=True)
        patr = float(atr(trunc, atr_period).iloc[-1]) if len(trunc) else float("nan")
        checked += 1
        same_atr = ctx["premium_atr"] is not None and math.isclose(patr, ctx["premium_atr"], rel_tol=1e-9, abs_tol=1e-9)
        if option_features(ctx, mut) != option_features(ctx) or not same_atr:
            failures.append(f"{ctx['signal_time'].isoformat()}|{ctx['direction']}")
    return {"checked": checked, "failures": failures[:50], "failure_count": len(failures),
            "passed": checked > 0 and not failures}


def pipeline_mutation_test(spot, strategies, allowed, resolve, premium_for, params, underlying, days,
                           cut: str = "12:00") -> dict[str, Any]:
    """Existing ``imp.mutation_test``: scramble underlying + premium candles after the cut and rebuild the contexts."""
    def signal_fn(frame):
        b = family_bars(frame)
        m = allowed_mask(b, allowed)
        return union_signals(b, [k for s in strategies for k in strategy_signals(b, s, m)[0]], underlying)

    pipeline = imp._pipeline_factory(signal_fn, resolve, params, "09:15", "09:30")
    cuts = [pd.Timestamp(datetime.combine(date.fromisoformat(d), time.fromisoformat(cut)), tz=imp.IST) for d in days]
    return imp.mutation_test(pipeline, spot, premium_for, [imp.BASELINE], cuts)[imp.BASELINE.name]


def _check(cid: str, name: str, ok: bool | None, detail: str) -> dict[str, str]:
    return {"id": cid, "check": name, "status": INFO if ok is None else (PASS if ok else FAIL), "detail": detail}


def data_checks(spot: pd.DataFrame, days: Sequence[str], interval: int = 5) -> list[dict]:
    ts = _ist_series(spot["timestamp"])
    want = set(days)
    sel = ts[ts.dt.date.map(lambda d: d.isoformat() in want)].sort_values()
    tz_ok = str(sel.dt.tz) in ("Asia/Kolkata", "IST") and len(sel) > 0
    dup = int(sel.duplicated().sum())
    grid = bool(((sel.dt.minute % interval == 0) & (sel.dt.second == 0)).all())
    counts = sel.groupby(sel.dt.date).agg(["count", "min", "max"])
    full = (counts["count"] == SESSION_BARS) & (counts["min"].dt.time == time(9, 15)) & (counts["max"].dt.time == time(15, 25))
    gaps = sel.groupby(sel.dt.date).diff().dropna()
    native = bool((gaps == pd.Timedelta(minutes=interval)).all())
    return [
        _check("D1", "Timestamps are tz-aware IST, unique, on the 5-minute grid", tz_ok and dup == 0 and grid,
               f"tz={sel.dt.tz}, bars={len(sel)}, duplicates={dup}, on_grid={grid}"),
        _check("D2", "Only complete sessions (09:15..15:25, 75 candles) are replayed", bool(full.all()),
               f"{int(full.sum())}/{len(full)} sessions complete"),
        _check("D3", "Native 5-minute candles, no resampling (every in-session gap = 5 minutes)", native,
               "candles are stored Upstox 5m bars; the pipeline never resamples"),
    ]


def context_checks(contexts: Sequence[dict], row_by_key: dict, interval: int = 5) -> list[dict]:
    ok = [c for c in contexts if c["data_status"] == "OK"]
    step = pd.Timedelta(minutes=interval)
    nxt = sum(c["entry_time"] == c["signal_time"] + step for c in ok)
    at_sig = sum(bool((c["premium"]["timestamp"] == c["signal_time"]).any()) for c in ok)
    resolved = [c for c in contexts if c.get("expiry") is not None]
    exp_ok = sum(c["expiry"] >= c["session_date"] for c in resolved)
    reasons = Counter(sh.reason_code(c["reason"]) for c in contexts if c["data_status"] != "OK")
    ev = [(k, r) for k, r in row_by_key.items() if r["state"] == sh.EVALUATED]
    ctx_by = {(c["signal_time"].isoformat(), c["direction"]): c for c in ok}
    sl_ok = exit_ok = 0
    for k, r in ev:
        ctx = ctx_by.get(k)
        if ctx is None:
            continue
        stop, target = imp.exit_levels(ctx, STOP_ATR, TARGET_R)
        sl_ok += math.isclose(stop, r["stop_loss"]) and math.isclose(target, r["target"])
        exit_ok += r.get("exit_time") is not None and pd.Timestamp(r["exit_time"]) >= ctx["entry_time"]
    return [
        _check("C1", "Signal decided on the completed signal candle; entry strictly later", all(
            c["entry_time"] > c["signal_time"] for c in ok) and bool(ok), f"{len(ok)} signals with full data"),
        _check("C2", "Entry = open of the next option candle (signal + 5 min)", nxt == len(ok) and bool(ok),
               f"{nxt}/{len(ok)} entries exactly at signal + {interval} min"),
        _check("C3", "Option candle exists at the underlying signal timestamp (aligned 5m grid)",
               at_sig == len(ok) and bool(ok), f"{at_sig}/{len(ok)}"),
        _check("C4", "Point-in-time expiry: resolved expiry is listed and not before the signal date",
               exp_ok == len(resolved) and bool(resolved),
               f"{exp_ok}/{len(resolved)} contracts; unresolved/unavailable by reason: {dict(reasons)}"),
        _check("C5", "SL/target computed from entry and premium ATR (known at entry); exits only after entry",
               bool(ev) and sl_ok == len(ev) and exit_ok == len(ev),
               f"SL/target recomputed {sl_ok}/{len(ev)}, exit after entry {exit_ok}/{len(ev)}"),
    ]


def split_checks(split_days: dict[str, list[str]], splits: dict, today: date) -> list[dict]:
    order = [d for s in SPLITS for d in split_days[s]]
    chrono = order == sorted(order) and len(order) == len(set(order))
    bounds = all(split_days[a] and split_days[b] and split_days[a][-1] < split_days[b][0]
                 for a, b in zip(SPLITS, SPLITS[1:]))
    latest = max(order) if order else None
    return [
        _check("S1", "TRAIN < VALIDATION < OOS, disjoint and chronological", chrono and bounds,
               "; ".join(f"{s} {split_days[s][0] if split_days[s] else '-'}..{split_days[s][-1] if split_days[s] else '-'}"
                         f" ({len(split_days[s])})" for s in SPLITS)),
        _check("S2", "Current day / forward data excluded from every split", latest is not None
               and latest < today.isoformat() and latest <= splits["OOS"][1].isoformat(),
               f"latest replayed session {latest}, today {today.isoformat()}"),
        _check("S3", "Selection uses TRAIN+VALIDATION only; OOS evaluated once for the finalists", True,
               "enforced in code: pick() reads selection_score (TRAIN/VALIDATION metrics) only"),
    ]


# ------------------------------------------------------------------ Phase 13: duplicate BUY metrics in the old lab
def _fp(m: dict) -> tuple:
    return (m["evaluated"], m["wins"], m["net_pnl"], m["max_drawdown"])


def _sig(rows: Iterable[dict]) -> list[tuple]:
    return [(r["signal_time"], r["direction"], r["decision"], r.get("outcome"), r.get("pnl")) for r in rows]


def legacy_duplicate_audit(base_rows: Sequence[dict], facts: dict, train_days: Sequence[str]) -> dict[str, Any]:
    """Re-run the STRATEGY IMPROVEMENT LAB candidates and explain metrics that duplicate SELL_ONLY_REFERENCE."""
    base_rows = list(base_rows)
    snapshot = json.dumps(base_rows, sort_keys=True, default=str)
    p = lab.train_params(base_rows, facts, train_days)
    cands = lab.predeclared_candidates()
    rows = {c.name: lab.apply_candidate(base_rows, facts, c, p) for c in cands}
    unchanged = json.dumps(base_rows, sort_keys=True, default=str) == snapshot
    names = [c.name for c in cands]
    by_dir = {d: [r for r in rows[lab.BASELINE] if r["direction"] == d] for d in DIRECTIONS}
    sell_only_fp = _fp(lab.perf(rows["SELL_ONLY_REFERENCE"]))
    buy_only_fp = _fp(lab.perf(by_dir["BUY"]))
    table, sell_leak, buy_leak, unexplained = [], [], [], []
    for c in cands:
        rs = rows[c.name]
        b = [r for r in rs if r["direction"] == "BUY"]
        s = [r for r in rs if r["direction"] == "SELL"]
        mb, ms, mc = lab.perf(b), lab.perf(s), lab.perf(rs)
        reasons = Counter(r.get("skip_reason") or "NO_SKIP_REASON" for r in b if r["state"] == sh.SKIPPED)
        sreasons = Counter(r.get("skip_reason") or "NO_SKIP_REASON" for r in s if r["state"] == sh.SKIPPED)
        dup_sell = _fp(mc) == sell_only_fp and c.name != "SELL_ONLY_REFERENCE"
        dup_buy = _fp(mc) == buy_only_fp
        if c.sell == () and _sig(s) != _sig(by_dir["SELL"]):
            sell_leak.append(c.name)
        if c.buy == () and _sig(b) != _sig(by_dir["BUY"]):
            buy_leak.append(c.name)
        why = None
        if dup_sell:
            all_unavail = bool(b) and all(r["state"] == sh.SKIPPED and (r.get("skip_reason") or "").startswith(
                (dr.CONFIRM_UNAVAILABLE, dr.DIRECTION_SKIP)) for r in b)
            why = ("every BUY skipped because a required input is unavailable (" + ", ".join(sorted(reasons)) + ")"
                   if all_unavail else None)
            if why is None and mb["evaluated"] == 0:
                why = "no BUY signal passed the filter: " + ", ".join(sorted(reasons))
            if why is None:
                unexplained.append(c.name)
        elif dup_buy:
            why = "every SELL skipped: " + ", ".join(sorted(sreasons))
        table.append({"candidate": c.name, "buy_checks": "EXCLUDED" if c.buy is None else ",".join(c.buy) or "-",
                      "sell_checks": "EXCLUDED" if c.sell is None else ",".join(c.sell) or "-",
                      "buy_evaluated": mb["evaluated"], "buy_pnl": mb["net_pnl"], "sell_evaluated": ms["evaluated"],
                      "sell_pnl": ms["net_pnl"], "combined_evaluated": mc["evaluated"], "combined_pnl": mc["net_pnl"],
                      "duplicates_sell_only": dup_sell, "duplicates_buy_only": dup_buy, "explanation": why})
    so_buy = [r for r in rows["SELL_ONLY_REFERENCE"] if r["direction"] == "BUY"]
    again = {n: _fp(lab.perf(rows[n])) for n in names}
    shuffled = {n: _fp(lab.perf(list(reversed(rows[n])))) for n in names}
    checks = [
        _check("P13-1", "Candidate IDs are unique", len(names) == len(set(names)), f"{len(names)} candidates"),
        _check("P13-2", "BUY-only candidates never change SELL rows", not sell_leak,
               f"leaks: {sell_leak or 'none'}"),
        _check("P13-3", "SELL-only candidates never change BUY rows", not buy_leak, f"leaks: {buy_leak or 'none'}"),
        _check("P13-4", "SELL_ONLY_REFERENCE cannot populate BUY metrics",
               bool(so_buy) and all(r["state"] == sh.SKIPPED for r in so_buy),
               f"{len(so_buy)} BUY rows, all skipped ({dict(Counter(r.get('skip_reason') for r in so_buy))})"),
        _check("P13-5", "Base rows are never mutated (rows are copied per candidate)", unchanged,
               "JSON snapshot of base rows before/after applying every candidate"),
        _check("P13-6", "Metrics recompute identically from scratch and independent of row order",
               again == {n: _fp(lab.perf(rows[n])) for n in names} and again == shuffled, "perf() is pure"),
        _check("P13-7", "Every duplicate of SELL_ONLY metrics is explained by BUY data unavailability",
               not unexplained, f"unexplained: {unexplained or 'none'}"),
    ]
    dups = [t["candidate"] for t in table if t["duplicates_sell_only"]]
    return {"table": table, "checks": checks, "duplicates_of_sell_only": dups,
            "duplicates_of_buy_only": [t["candidate"] for t in table if t["duplicates_buy_only"]],
            "root_cause": (
                "Not cache contamination or shared state. NIFTY index candles carry zero volume, so VWAP and "
                "volume-ratio are NaN; check() returns None (unavailable) and decide() skips every BUY with "
                "CONFIRMATION_DATA_UNAVAILABLE. The old lab ranked candidates on COMBINED (BUY+SELL) metrics, so a "
                "BUY candidate whose BUY side is entirely unavailable reports exactly the SELL-only numbers. "
                "Fix in this lab: BUY and SELL are separate candidates with separate metrics, and candidates whose "
                "inputs are never available are DATA_UNAVAILABLE and never ranked.")}


def snapshot_rows(row_by_key: dict) -> str:
    return json.dumps([[list(k), v] for k, v in sorted(row_by_key.items())], sort_keys=True, default=str)


def isolation_checks(strategies: Sequence[Strategy], rows: dict[str, list[dict]], row_by_key: dict,
                     snapshot: str) -> list[dict]:
    ids = [s.cid for s in strategies]
    pure = [s.cid for s in strategies if any(r["direction"] != s.direction for r in rows[s.cid])]
    again = all(_fp(lab.perf(rows[n])) == _fp(lab.perf(list(reversed(rows[n])))) for n in ids)
    return [
        _check("I1", "Candidate IDs are unique", len(ids) == len(set(ids)), f"{len(ids)} candidates"),
        _check("I2", "Each candidate contains only its own direction (BUY and SELL fully independent)", not pure,
               f"violations: {pure[:10] or 'none'}"),
        _check("I3", "Shared evaluated rows are never mutated by candidates",
               snapshot_rows(row_by_key) == snapshot, "JSON snapshot before/after"),
        _check("I4", "Metrics recompute independently of row order", again, "perf() recomputed per candidate"),
    ]


# ------------------------------------------------------------------ study
def build_lab70(*, spot: pd.DataFrame, split_days: dict[str, list[str]], splits: dict, resolve: Callable,
                premium_for: Callable, params, square_off: str, underlying: str = "NIFTY",
                allowed: Callable = default_allowed, base_rows: Sequence[dict] = (), facts: dict | None = None,
                baseline_lookahead_passed: bool = True, rules: dict | None = None, today: date | None = None,
                criteria: Criteria = Criteria(), progress: Callable[[str], None] | None = None,
                run_mutation: bool = True, cut_times: Sequence[str] = ("11:00", "13:00"),
                lookahead_days: Sequence[str] | None = None, interval: int = 5) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    c = criteria
    hist = [d for s in SPLITS for d in split_days[s]]
    today = today or datetime.now(sh.IST).date()
    periods = {**{s: split_days[s] for s in SPLITS}, "ALL": hist}
    bases = predeclared_strategies()
    say(f"computing causal features and {len(bases)} base signal sets ...")
    bars = family_bars(spot)
    mask = allowed_mask(bars, allowed, set(hist))
    sig = {s.cid: strategy_signals(bars, s, mask) for s in bases}
    union = union_signals(bars, [k for ks, _ in sig.values() for k in ks], underlying)
    say(f"resolving contracts / premiums for {len(union)} unique signals (point-in-time) ...")
    feats = imp.underlying_features(spot)
    contexts = imp.build_contexts(union, feats, resolve, premium_for, params)
    cache: dict = {}
    raw = imp.run_variant(contexts, feats, imp.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE, cache=cache)
    row_by_key = {row_key(r): sh.classify_row(r, {}) for r in raw}
    raw_by_key = {row_key(r): r for r in raw}
    ctx_by_key = {(x["signal_time"].isoformat(), x["direction"]): x for x in contexts}
    opt = {k: option_features(x) for k, x in ctx_by_key.items()}
    bar_at = bars.set_index("timestamp")
    labels = {}
    for k in [*row_by_key, *(row_key(r) for r in base_rows)]:
        if k not in labels:
            ts = pd.Timestamp(k[0]).tz_convert(imp.IST)
            labels[k] = regime_labels(bar_at.loc[ts] if ts in bar_at.index else None, opt.get(k))
    available = {s.cid: sig[s.cid][1] > 0 for s in bases}
    strategies = bases + overlay_strategies([s for s in bases if available[s.cid]])
    snapshot = snapshot_rows(row_by_key)
    rows = {s.cid: candidate_rows(s, sig[s.signal_source][0], row_by_key, opt) for s in strategies}

    la_days = list(lookahead_days) if lookahead_days is not None else hist
    say(f"truncation lookahead test ({len(la_days)} sessions x {len(cut_times)} cuts) ...")
    la_feat = feature_truncation_test(spot, bases, la_days, allowed, cut_times)
    la_opt = option_feature_mutation_test(contexts, params.atr_period)
    la_mut = None
    if run_mutation:
        say("pipeline mutation test (scrambled candles after 12:00, contexts rebuilt) ...")
        la_mut = pipeline_mutation_test(spot, bases, allowed, resolve, premium_for, params, underlying, la_days)
    lookahead_ok = la_feat["passed"] and la_opt["passed"] and (la_mut is None or la_mut["passed"])

    folds = lab.walk_forward_folds(hist, c.wf_train_days, c.wf_val_days, c.wf_test_days)
    scores = {}
    for s in strategies:
        sc = score_rows(rows[s.cid], periods, folds, labels, c)
        sc.update(cid=s.cid, family=s.family, direction=s.direction, n_conditions=s.n_conditions,
                  overlay=s.overlay, base=s.base, rule=describe(s), params=s.p,
                  availability=sig[s.signal_source][1], lookahead_passed=lookahead_ok,
                  data_status="OK" if available[s.signal_source] else DATA_UNAVAILABLE,
                  tier=sample_tier(sc["OOS"]["evaluated"], c))
        sc["selection_score"] = (selection_score(sc["TRAIN"], sc["VALIDATION"], c, c.min_train_trades,
                                                 c.min_val_trades) if sc["data_status"] == "OK" else None)
        sc["gates"] = gates(sc, c)
        sc["status"], sc["failed"] = candidate_status(sc, c)
        sc["final_score"] = final_score(sc, c)
        scores[s.cid] = sc
    groups: dict[tuple, list[str]] = {}
    for s in strategies:
        if scores[s.cid]["data_status"] == "OK":
            acc = tuple(sorted(row_key(r) for r in rows[s.cid] if r["state"] == sh.EVALUATED))
            groups.setdefault((s.direction, acc), []).append(s.cid)
    for members in groups.values():
        for n in members:
            scores[n]["identical_to"] = [m for m in members if m != n]

    finalists = {d: pick(scores, [s.cid for s in strategies if s.direction == d]) for d in DIRECTIONS}
    family_finalists = {f"{d}|{fam}": pick(scores, [s.cid for s in strategies if s.direction == d and s.family == fam])
                        for d in DIRECTIONS for fam in FAMILIES}
    n_cond = {s.cid: s.n_conditions for s in strategies}
    procedure = {d: wf_procedure([s.cid for s in strategies if s.direction == d and scores[s.cid]["data_status"] == "OK"],
                                 rows, n_cond, folds, c) for d in DIRECTIONS}
    hist_base = lab.period_rows(list(base_rows), hist)
    baseline = {d: score_rows([r for r in hist_base if r["direction"] == d], periods, folds, labels, c)
                for d in DIRECTIONS}
    comb_rows = [r for d in DIRECTIONS if finalists[d] for r in rows[finalists[d]]]
    combined = score_rows(comb_rows, periods, folds, labels, c) if comb_rows else None
    exit_info = {}
    for d, n in finalists.items():
        if not n:
            continue
        alt = []
        for r in rows[n]:
            if r["state"] != sh.EVALUATED:
                continue
            k = row_key(r)
            row = dict(raw_by_key[k])
            row.update(imp.evaluate(ctx_by_key[k], STOP_ATR, 1.0, square_off, sh.LOT_SIZE, cache))
            alt.append(sh.classify_row(row, {}))
        exit_info[d] = {p: lab.perf(lab.period_rows(alt, dd)) for p, dd in periods.items()}

    say("integrity audit (Phase 2) and duplicate-metric audit (Phase 13) ...")
    legacy = legacy_duplicate_audit(hist_base, facts or {}, split_days["TRAIN"]) if base_rows else None
    integrity = (data_checks(spot, hist, interval) + context_checks(contexts, row_by_key, interval)
                 + [_check("L1", "Indicators/features/signals unchanged when recomputed from candles <= cut",
                           la_feat["passed"], f"{la_feat['cuts']} cuts; feature failures "
                           f"{len(la_feat['feature_failures'])}, signal failures {len(la_feat['signal_failures'])}"),
                    _check("L2", "Option features + premium ATR unchanged when later premium candles are scrambled",
                           la_opt["passed"], f"{la_opt['checked']} signals, failures {la_opt['failure_count']}"),
                    _check("L3", "Pipeline mutation test (contract, premium ATR, stop distance, entry > signal)",
                           None if la_mut is None else la_mut["passed"],
                           "skipped" if la_mut is None else f"{la_mut['cuts']} cuts, failures {len(la_mut['failures'])}"),
                    _check("L4", "Existing S2 baseline replay mutation test", baseline_lookahead_passed, "12:00 cut"),
                    _check("L5", "No end-of-day regime labels in any entry decision", True,
                           "regime inputs are causal (bars <= signal); EOD labels are not loaded by this lab")]
                 + split_checks(split_days, splits, today)
                 + isolation_checks(strategies, rows, row_by_key, snapshot)
                 + (legacy["checks"] if legacy else []))
    integrity_ok = all(x["status"] != FAIL for x in integrity)
    rec = recommend(finalists, scores, integrity_ok)
    achieved = rec == ACCEPT
    ranking = sorted(scores, key=lambda n: (STATUS_ORDER[scores[n]["status"]], -scores[n]["final_score"], n))
    best_any = max((n for n in scores if scores[n]["OOS"]["evaluated"] >= c.tier_b and scores[n]["data_status"] == "OK"),
                   key=lambda n: (_num(scores[n]["OOS"]["win_rate"]), n), default=None)
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "production_s2_changed": False, "live_orders": 0,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"), "today": today.isoformat(),
        "rules": rules or {}, "criteria": asdict(c), "specs": [
            {"name": n, "family": f, "conditions": list(cs), "grid": {k: list(v) for k, v in g.items()}}
            for n, f, cs, g in SPECS],
        "splits": {s: {"from": splits[s][0].isoformat(), "to": splits[s][1].isoformat(), "days": split_days[s]}
                   for s in SPLITS},
        "counts": {"base_candidates": len(bases), "overlay_candidates": len(strategies) - len(bases),
                   "candidates": len(strategies), "unique_signals": len(union),
                   "data_unavailable": sum(not v for v in available.values()),
                   "evaluated_signals": sum(r["state"] == sh.EVALUATED for r in row_by_key.values())},
        "scores": scores, "ranking": ranking, "finalists": finalists, "family_finalists": family_finalists,
        "wf_procedure": procedure, "baseline": baseline, "combined": combined, "exit_sensitivity": exit_info,
        "best_oos_any": best_any, "lookahead": {"features": la_feat, "option": la_opt, "mutation": la_mut,
                                                "passed": lookahead_ok},
        "integrity": integrity, "integrity_ok": integrity_ok, "legacy_audit": legacy,
        "recommendation": rec, "target_achieved": achieved,
        "formulas": {"score": SCORE_FORMULA, "selection": SELECTION_FORMULA},
        "_rows": rows,
    }


# ------------------------------------------------------------------ reports
_pct, _pf, _rs = sh._pct, sh._pf_s, sh._rs


def _fmt(v):
    if isinstance(v, float) and math.isinf(v):
        return "INF"
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(sh._jsonable(v))
    return v


def _csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})


def _m(m: dict | None) -> str:
    if not m:
        return "-"
    return f"{m['evaluated']} / {_pct(m['win_rate'])} / PF {_pf(m['profit_factor'])} / Rs {_rs(m['net_pnl'])}"


def candidate_csv_rows(a: dict) -> list[dict]:
    out = []
    fin = {v: f"DIRECTION_FINALIST_{k}" for k, v in a["finalists"].items() if v}
    fam = {v: "FAMILY_FINALIST" for v in a["family_finalists"].values() if v}
    for n in a["ranking"]:
        s = a["scores"][n]
        row = {"candidate": n, "family": s["family"], "direction": s["direction"], "rule": s["rule"],
               "params": s["params"], "overlay": s["overlay"], "n_conditions": s["n_conditions"],
               "data_status": s["data_status"], "input_availability": s["availability"]}
        for p in PERIODS:
            m = s[p]
            row.update({f"{p.lower()}_trades": m["evaluated"], f"{p.lower()}_wins": m["wins"],
                        f"{p.lower()}_losses": m["losses"], f"{p.lower()}_accuracy": m["win_rate"],
                        f"{p.lower()}_pf": m["profit_factor"], f"{p.lower()}_pnl": m["net_pnl"]})
        o, w = s["OOS"], s["walk_forward"]
        row.update({"oos_avg_r": o["average_r"], "oos_max_dd": o["max_drawdown"],
                    "oos_max_losing_streak": o["max_consecutive_losses"], "oos_best_day": o["best_day"],
                    "oos_best_day_pnl": o["best_day_pnl"], "oos_worst_day": o["worst_day"],
                    "oos_worst_day_pnl": o["worst_day_pnl"], "oos_wr_ci95": o["wr_ci95"], "sample_tier": s["tier"],
                    "wf_accuracy": w["wf_accuracy"], "wf_trades": w["wf_trades"], "wf_pf": w["wf_pf"],
                    "wf_profitable_share": w["wf_profitable_share"], "wf_consistency": w["wf_consistency"],
                    "all_profitable_day_share": s["day_all"]["profitable_day_share"],
                    "all_best_day_share": s["day_all"]["best_day_share_of_net"], "regime_share": s["regime_share"],
                    "selection_score": s["selection_score"], "final_score": s["final_score"], "status": s["status"],
                    "failed_gates": s["failed"], "selected_as": fin.get(n) or fam.get(n) or "",
                    "identical_signal_set_to": s.get("identical_to", [])})
        out.append(row)
    return out


def _named(a: dict) -> dict[str, dict]:
    """Finalists, family finalists, combined and the S2 baseline per direction (for per-day/regime tables)."""
    out = {f"S2_BASELINE_{d}": a["baseline"][d] for d in DIRECTIONS}
    for n in [*a["finalists"].values(), *a["family_finalists"].values()]:
        if n:
            out[n] = a["scores"][n]
    if a["combined"]:
        out["COMBINED_FINALISTS"] = a["combined"]
    return out


def walk_forward_csv_rows(a: dict) -> list[dict]:
    out = []
    for n in a["ranking"]:
        s = a["scores"][n]
        if s["data_status"] != "OK":
            continue
        for f in s["walk_forward"]["folds"]:
            out.append({"kind": "CANDIDATE", "candidate": n, "direction": s["direction"], **f})
    for d, p in a["wf_procedure"].items():
        for f in p["folds"]:
            out.append({"kind": "SELECTION_PROCEDURE", "candidate": f["selected"] or "NONE_ELIGIBLE", "direction": d,
                        "fold": f["fold"], "test": f["test"], "evaluated": f["evaluated"], "win_rate": f["win_rate"],
                        "profit_factor": f["profit_factor"], "net_pnl": f["net_pnl"]})
    return out


def day_csv_rows(a: dict) -> list[dict]:
    out = []
    split_of = {d: s for s in SPLITS for d in a["splits"][s]["days"]}
    for n, s in _named(a).items():
        prof = s["day_all"]
        for d, v in prof["daily"].items():
            out.append({"candidate": n, "session_date": d, "split": split_of.get(d), "pnl": v,
                        "share_of_net": v / s["ALL"]["net_pnl"] if s["ALL"]["net_pnl"] > 0 else None,
                        "best_day": prof["best_day"], "worst_day": prof["worst_day"],
                        "median_day_pnl": prof["median_day_pnl"], "profitable_day_share": prof["profitable_day_share"],
                        "best_day_share_of_net": prof["best_day_share_of_net"]})
    return out


def regime_csv_rows(a: dict) -> list[dict]:
    out = []
    for n, s in _named(a).items():
        for lab_name, m in sorted(s["regimes"].items()):
            out.append({"candidate": n, "regime": lab_name, "trades": m["evaluated"], "wins": m["wins"],
                        "accuracy": m["win_rate"], "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"],
                        "regime_positive_share": s["regime_share"]})
    return out


def _dir_table(a: dict, d: str, limit: int = 15) -> list[str]:
    names = sorted((n for n, s in a["scores"].items() if s["direction"] == d and s["data_status"] == "OK"),
                   key=lambda n: (-_num(a["scores"][n]["selection_score"], -1), n))[:limit]
    md = ["| Candidate | Selection | TRAIN n/acc/PF/P&L | VALIDATION n/acc/PF/P&L | OOS n/acc/PF/P&L | Status |",
          "|---|---|---|---|---|---|"]
    for n in names:
        s = a["scores"][n]
        md.append(f"| {n} | {s['selection_score'] if s['selection_score'] is not None else 'not eligible'} | "
                  f"{_m(s['TRAIN'])} | {_m(s['VALIDATION'])} | {_m(s['OOS'])} | {s['status']} |")
    return md


def _full_line(name: str, m: dict) -> str:
    return (f"| {name} | {m['evaluated']} | {m['wins']} | {m['losses']} | {_pct(m['win_rate'])} | "
            f"{_pf(m['profit_factor'])} | {_rs(m['net_pnl'])} | {m['average_r']} | {_rs(m['max_drawdown'])} | "
            f"{m['max_consecutive_losses']} | {m['best_day'] or '-'} ({_rs(m['best_day_pnl'])}) | "
            f"{m['worst_day'] or '-'} ({_rs(m['worst_day_pnl'])}) |")


FULL_HEAD = ["| Strategy | Trades | Wins | Losses | Accuracy | PF | P&L | Avg R | Max DD | Max losing streak | "
             "Best day | Worst day |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]


def main_markdown(a: dict) -> str:
    sc, fin, c = a["scores"], a["finalists"], a["criteria"]
    md = [f"# {HEADING}", "", "PAPER mode | research only | no orders | production S2 NOT modified | "
          f"LIVE_TRADING_ENABLED=false | generated {a['generated_at']}", ""]
    md += ["## 1. Baseline (current S2)", "", "Full description: `research_70_baseline.md`.", "", *FULL_HEAD]
    for d in DIRECTIONS:
        for p in PERIODS:
            md.append(_full_line(f"S2 {d} {p}", a["baseline"][d][p]))
    md += ["", "## 2. Data integrity", "", f"Overall: **{'PASS' if a['integrity_ok'] else 'FAIL'}** "
           "(details in `70_accuracy_integrity_audit.md`).", "", "| ID | Check | Status |", "|---|---|---|"]
    md += [f"| {x['id']} | {x['check']} | {x['status']} |" for x in a["integrity"]]
    k = a["counts"]
    md += ["", "## 3. Candidates", "",
           f"{k['base_candidates']} base candidates (6 families x coarse grids x BUY/SELL) + {k['overlay_candidates']} "
           f"option-premium overlay candidates = {k['candidates']} candidates; {k['unique_signals']} unique signals "
           f"resolved, {k['evaluated_signals']} with a real-premium outcome. {k['data_unavailable']} base candidates "
           "need underlying volume/VWAP (NIFTY index volume is zero) and are DATA_UNAVAILABLE (never ranked).", "",
           "| Family | Idea |", "|---|---|"]
    md += [f"| {f} | {t} |" for f, t in FAMILIES.items()]
    md += ["", "| Spec | Family | Conditions | Parameter grid (all points reported) |", "|---|---|---|---|"]
    md += [f"| {s['name']} | {s['family']} | {', '.join(s['conditions'])} | {s['grid'] or '-'} |" for s in a["specs"]]
    md += ["", "Option overlays (Phase 5, premium candles <= signal only): "
           + "; ".join(f"{k} = {v}" for k, v in OVERLAYS.items()) + ".",
           "Strike is ATM by the unchanged S2 contract rule, so ITM/OTM variants were not evaluated (would change the "
           "contract rule). Signals are edge-triggered: the first candle on which every condition holds.", ""]
    for i, d in ((4, "BUY"), (5, "SELL")):
        md += [f"## {i}. {d} results (top 15 by TRAIN+VALIDATION selection score)", "", *_dir_table(a, d), ""]
    md += ["## 6. Validation (selection)", "", SELECTION_FORMULA, "", "| Direction | Finalist | Selection score | "
           "VALIDATION n/acc/PF/P&L |", "|---|---|---|---|"]
    for d in DIRECTIONS:
        n = fin[d]
        md.append(f"| {d} | {n or 'NONE ELIGIBLE'} | {sc[n]['selection_score'] if n else '-'} | "
                  f"{_m(sc[n]['VALIDATION']) if n else '-'} |")
    md += ["", "Family finalists (validation-selected per family and direction):", "",
           "| Direction / family | Finalist | VALIDATION | OOS | Status |", "|---|---|---|---|---|"]
    for key, n in a["family_finalists"].items():
        md.append(f"| {key} | {n or 'none eligible'} | {_m(sc[n]['VALIDATION']) if n else '-'} | "
                  f"{_m(sc[n]['OOS']) if n else '-'} | {sc[n]['status'] if n else '-'} |")
    md += ["", "## 7. OOS (evaluated once, never used for selection)", "", *FULL_HEAD]
    for d in DIRECTIONS:
        if fin[d]:
            md.append(_full_line(f"{d} finalist {fin[d]}", sc[fin[d]]["OOS"]))
        md.append(_full_line(f"S2 baseline {d}", a["baseline"][d]["OOS"]))
    if a["combined"]:
        md.append(_full_line("COMBINED finalists", a["combined"]["OOS"]))
    md += ["", "## 8. Walk-forward", "", f"Folds: train {c['wf_train_days']} / validation {c['wf_val_days']} / test "
           f"{c['wf_test_days']} sessions, anchored at the last session, test windows never overlap.", "",
           "| Strategy | WF accuracy | WF trades | WF PF | WF P&L | Profitable windows | Consistency |",
           "|---|---|---|---|---|---|---|"]
    for n, s in _named(a).items():
        w = s["walk_forward"]
        md.append(f"| {n} | {_pct(w['wf_accuracy'])} | {w['wf_trades']} | {_pf(w['wf_pf'])} | {_rs(w['wf_pnl'])} | "
                  f"{_pct(w['wf_profitable_share'])} | {_pct(w['wf_consistency'])} |")
    for d, p in a["wf_procedure"].items():
        md.append(f"| SELECTION PROCEDURE {d} (re-selected each fold) | {_pct(p['wf_accuracy'])} | {p['wf_trades']} | "
                  f"{_pf(p['wf_pf'])} | {_rs(p['wf_pnl'])} | {_pct(p['wf_profitable_share'])} | - |")
    md += ["", "## 9. Day stability", "", "| Strategy | Days | Best day | Worst day | Median day | Profitable days | "
           "Best-day share of net | Best-day share of gross profit |", "|---|---|---|---|---|---|---|---|"]
    for n, s in _named(a).items():
        p = s["day_all"]
        md.append(f"| {n} | {p['days']} | {p['best_day'] or '-'} ({_rs(p['best_day_pnl'])}) | {p['worst_day'] or '-'} "
                  f"({_rs(p['worst_day_pnl'])}) | {_rs(p['median_day_pnl'])} | {_pct(p['profitable_day_share'])} | "
                  f"{_pct(p['best_day_share_of_net'])} | {_pct(p['best_day_share_of_gross_profit'])} |")
    md += ["", "## 10. Regime stability", "", "Causal labels at the signal candle: trending/sideways "
           f"(|day move| >= {ABS_TREND_RETURN:.1%}), choppy/non_choppy (>= {lab.CHOPPY_SIGN_CHANGES} sign changes), "
           "high/low volatility (ATR percentile >= 0.5), strong/weak option volume. Underlying volume regime: "
           "DATA_UNAVAILABLE (index volume is zero).", "", "| Strategy | Regime | Trades | Accuracy | PF | P&L |",
           "|---|---|---|---|---|---|"]
    for n, s in _named(a).items():
        for r, m in sorted(s["regimes"].items()):
            md.append(f"| {n} | {r} | {m['evaluated']} | {_pct(m['win_rate'])} | {_pf(m['profit_factor'])} | "
                      f"{_rs(m['net_pnl'])} |")
    ident = sum(1 for s in sc.values() if s.get("identical_to"))
    best = a["best_oos_any"]
    md += ["", "## 11. Overfitting checks", "",
           f"- {k['candidates']} candidates tested -> multiple-testing risk is high; therefore only the validation-"
           "selected finalist per direction can support a claim.",
           f"- Complexity penalty: 5% of the score, -0.15 per condition beyond the first. All parameter ranges are coarse "
           "and fixed in SPECS (section 3); no thresholds were tuned on OOS.",
           f"- {ident} candidates share an identical evaluated-signal set with another candidate (see CSV column).",
           f"- Gates: OOS >= {c['tier_b']} trades, OOS PF >= {c['min_oos_pf']}, OOS P&L > 0, DD <= OOS P&L, "
           f"best OOS day <= {c['max_oos_best_day_share']:.0%} of net, P&L without the largest win > 0, max losing streak "
           f"<= {c['max_loss_streak']}, VALIDATION profitable, WF profitable windows >= {c['min_wf_share']:.0%} and WF "
           f"PF >= {c['min_wf_pf']}, profitable days >= {c['min_profitable_day_share']:.0%} with best day <= "
           f"{c['max_all_best_day_share']:.0%} of net, regime share >= {c['min_regime_share']:.0%}.",
           f"- Sample tiers (OOS trades): A >= {c['tier_a']} strong, B >= {c['tier_b']} usable, C >= {c['tier_c']} weak, "
           "D insufficient.",
           "- OOS 2026-09-25..10-01 was visible in earlier project research (S2 filter labs); these families were not "
           "tuned on it, but the window is not pristine.",
           (f"- Highest OOS accuracy of ANY candidate with >= {c['tier_b']} OOS trades: {best} "
            f"({_m(sc[best]['OOS'])}). This is a max over {k['candidates']} candidates chosen by looking at OOS and is "
            "NOT a valid claim." if best else "- No candidate reached 30 OOS trades.")]
    md += ["", "## 12. Top candidates (report ordering by final score)", "", SCORE_FORMULA, "",
           "| # | Candidate | Tier | OOS n/acc/PF/P&L | VALIDATION | WF acc / windows | Score | Status |",
           "|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate([n for n in a["ranking"] if sc[n]["data_status"] == "OK"][:20], 1):
        s = sc[n]
        md.append(f"| {i} | {n} | {s['tier']} | {_m(s['OOS'])} | {_m(s['VALIDATION'])} | "
                  f"{_pct(s['walk_forward']['wf_accuracy'])} / {_pct(s['walk_forward']['wf_profitable_share'])} | "
                  f"{s['final_score']} | {s['status']} |")
    reasons = Counter(g for s in sc.values() if s["status"] not in (ALL_GATES_PASS,) for g in s["failed"])
    md += ["", "## 13. Rejected candidates and why", "", "| Failed gate | Candidates |", "|---|---|"]
    md += [f"| {g} | {n} |" for g, n in reasons.most_common()]
    md += ["", "| Status | Candidates |", "|---|---|"]
    md += [f"| {st} | {n} |" for st, n in Counter(s["status"] for s in sc.values()).most_common()]
    md += ["", "Finalist failures: " + "; ".join(f"{d}: {fin[d]} -> {', '.join(sc[fin[d]]['failed']) or 'none'}"
                                                 if fin[d] else f"{d}: no candidate eligible on TRAIN+VALIDATION"
                                                 for d in DIRECTIONS)]
    md += ["", "## 14. 70% achieved?", "", f"**{'ACHIEVED' if a['target_achieved'] else 'NOT ACHIEVED'}**", ""]
    if not a["target_achieved"]:
        md.append(NOT_ACHIEVED_TEXT)
    md += ["", f"With the unchanged 2R target the breakeven accuracy is ~33%; 70% accuracy at 2R would mean an "
           "average of about +1.1R per trade, far above anything observed. Information only (not eligible, it changes "
           "the risk rule): the same finalists with a 1R target:", ""]
    for d, by in a["exit_sensitivity"].items():
        md.append(f"- {d} {fin[d]} @1R: OOS {_m(by['OOS'])}; VALIDATION {_m(by['VALIDATION'])}")
    md += ["", "## 15. Recommendation", "", f"**{a['recommendation']}**", "", {
        ACCEPT: "A validation-selected finalist passed every gate including >= 70% OOS accuracy. Research candidate "
                "only: forward PAPER observation before any production discussion; nothing is deployed.",
        PROMISING: "A finalist reached >= 70% OOS accuracy with PF/P&L gates, but the sample is too small for the "
                   "stability gates. Needs more data (forward PAPER sessions).",
        KEEP: "No finalist passed the gates for a 70% claim. Keep the current S2 (SELL leg remains its source of edge).",
        REJECT: "Integrity/lookahead failure or no finalist profitable in OOS; this research line is rejected.",
    }[a["recommendation"]], "", "Production S2 changed: NO. Live orders: 0. TRADING_MODE=PAPER, "
           "LIVE_TRADING_ENABLED=false."]
    return "\n".join(md) + "\n"


def integrity_markdown(a: dict) -> str:
    md = [f"# {HEADING} - integrity audit", "", f"Overall: **{'PASS' if a['integrity_ok'] else 'FAIL'}**", "",
          "| ID | Check | Status | Detail |", "|---|---|---|---|"]
    md += [f"| {x['id']} | {x['check']} | {x['status']} | {x['detail']} |" for x in a["integrity"]]
    la = a["lookahead"]
    md += ["", "## Lookahead detail", "", f"- Feature/signal truncation: {la['features']['cuts']} cuts, feature failures "
           f"{la['features']['feature_failures'][:10] or 'none'}, signal failures "
           f"{la['features']['signal_failures'][:10] or 'none'}",
           f"- Option feature mutation: {la['option']['checked']} signals, failures {la['option']['failures'][:10] or 'none'}",
           f"- Pipeline mutation: {'skipped' if la['mutation'] is None else la['mutation']}"]
    lg = a["legacy_audit"]
    if lg:
        md += ["", "## Phase 13 - duplicate BUY metrics in the previous lab", "", f"Root cause: {lg['root_cause']}", "",
               f"Candidates whose COMBINED metrics equal SELL_ONLY_REFERENCE: {', '.join(lg['duplicates_of_sell_only']) or 'none'}",
               f"Candidates whose COMBINED metrics equal BUY-only: {', '.join(lg['duplicates_of_buy_only']) or 'none'}", "",
               "| Candidate | BUY checks | SELL checks | BUY n | BUY P&L | SELL n | SELL P&L | = SELL-only | Explanation |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {t['candidate']} | {t['buy_checks']} | {t['sell_checks']} | {t['buy_evaluated']} | {_rs(t['buy_pnl'])} | "
               f"{t['sell_evaluated']} | {_rs(t['sell_pnl'])} | {'YES' if t['duplicates_sell_only'] else ''} | "
               f"{t['explanation'] or ''} |" for t in lg["table"]]
    return "\n".join(md) + "\n"


def baseline_markdown(a: dict) -> str:
    md = [f"# {HEADING} - Phase 1 baseline (actual S2 implementation)", "",
          "Documented from the code paths used by the replay (`research_strategies.signals_s2`, "
          "`s2_1dte_improvement.build_contexts/evaluate`, `current_day_s2.option_levels`).", ""]
    md += [f"- **{k}**: {v}" for k, v in a["rules"].items()]
    md += ["", "Notes:", "- Every S2 signal is evaluated independently (no position limit, daily loss limit or trade cap).",
           "- Index volume is zero, so S2 has no volume/VWAP inputs.",
           f"- 2R target -> breakeven accuracy ~33%; the 70% target is therefore extremely demanding at this payoff.",
           "", *FULL_HEAD]
    for d in DIRECTIONS:
        for p in PERIODS:
            md.append(_full_line(f"S2 {d} {p}", a["baseline"][d][p]))
    return "\n".join(md) + "\n"


REPORT_FILES = ("70_accuracy_lab.md", "70_accuracy_candidates.csv", "70_accuracy_walk_forward.csv",
                "70_accuracy_day_stability.csv", "70_accuracy_regime_stability.csv", "70_accuracy_integrity_audit.md",
                "research_70_baseline.md")


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {n: out_dir / n for n in REPORT_FILES}
    paths["70_accuracy_lab.md"].write_text(main_markdown(a), encoding="utf-8")
    _csv(paths["70_accuracy_candidates.csv"], candidate_csv_rows(a))
    _csv(paths["70_accuracy_walk_forward.csv"], walk_forward_csv_rows(a))
    _csv(paths["70_accuracy_day_stability.csv"], day_csv_rows(a))
    _csv(paths["70_accuracy_regime_stability.csv"], regime_csv_rows(a))
    paths["70_accuracy_integrity_audit.md"].write_text(integrity_markdown(a), encoding="utf-8")
    paths["research_70_baseline.md"].write_text(baseline_markdown(a), encoding="utf-8")
    return paths
