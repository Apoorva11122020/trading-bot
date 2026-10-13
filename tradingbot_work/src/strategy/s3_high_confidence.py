"""S3_HIGH_CONFIDENCE - research-only confirmation-score strategy (PAPER | RESEARCH ONLY | NO ORDERS).

Not registered with the runtime, the signal generator or ``research_strategies``; nothing here places orders.
Every input is known at the close of the completed signal candle: underlying features come from
``accuracy_70_lab.family_bars`` (bars <= signal) and option features from premium candles <= the signal candle.

Decision per completed candle and direction: BUY / SELL / NO_SIGNAL. "No trade is a valid decision":
missing data, an uncertain or choppy regime, missing option premium data or a score below the threshold all
produce NO_SIGNAL. BUY and SELL use separate profiles with deliberately different assumptions (pre-declared
research hypotheses, not fitted): BUY guards against chasing extended upside breakouts and needs every premium
confirmation; SELL tolerates more extension (downside moves expand put premium faster) and credits upper-wick
rejection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import accuracy_70_lab as acc

STRATEGY_ID = "S3_HIGH_CONFIDENCE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS"
BUY, SELL, NO_SIGNAL = "BUY", "SELL", "NO_SIGNAL"
DIRECTIONS = (BUY, SELL)

# Starting research weights (documented, NOT optimised; never tuned on OOS).
WEIGHTS = {"trend": 20, "vwap": 15, "ema": 15, "momentum": 10, "volume": 10, "price_action": 10, "breakout": 10,
           "option_premium": 10}
UNDERLYING_COMPONENTS = ("trend", "vwap", "ema", "momentum", "price_action", "breakout")
OPTION_COMPONENTS = ("volume", "option_premium")
THRESHOLDS = (60, 65, 70, 75, 80, 85, 90)
TIERS = (("S3-A", 60, 69), ("S3-B", 70, 79), ("S3-C", 80, 89), ("S3-D", 90, 100))
PRE_FILTER = 40  # underlying points (max 80) needed before the option is looked at; 40 + 20 option points = 60

REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE_CHOP", "HIGH_VOLATILITY", "LOW_VOLATILITY", "UNCERTAIN")
NO_TRADE_REGIMES = ("RANGE_CHOP", "UNCERTAIN")
TREND_SLOPE = 0.15      # EMA20 3-candle slope in ATR units
FLAT_SLOPE = 0.05
FLAT_DIST = 0.5         # |close - EMA20| / ATR
CHOP_FLIPS = 8          # close-to-close direction flips within the last 12 candles of the session
HIGH_VOL, LOW_VOL = 0.80, 0.20  # trailing 5-session ATR/close percentile
REGIME_INPUTS = ("close", "ema20", "ema50", "ema20_slope_atr", "dist_ema20_atr", "atr_pctile", "flips12")
SCORE_INPUTS = ("open", "close", "ema9", "ema20", "ema50", "ema20_slope_atr", "dist_ema20_atr", "atr14", "rsi14",
                "body_frac", "close_pos", "upper_wick", "hh6", "ll6", "hh12", "ll12", "vwap_ref", "vwap_slope_atr",
                "intraday_move")


@dataclass(frozen=True)
class DirectionProfile:
    direction: str
    trend_regime: str
    rsi_full: tuple[float, float]
    rsi_half: tuple[float, float]
    max_ext_atr: float
    max_vwap_ext_atr: float
    max_break_atr: float
    max_streak: int | None
    rejection_wick: bool
    premium_rule: str
    notes: tuple[str, ...] = field(default=())


BUY_PROFILE = DirectionProfile(
    BUY, "TREND_UP", rsi_full=(55, 68), rsi_half=(50, 72), max_ext_atr=1.0, max_vwap_ext_atr=1.0,
    max_break_atr=0.3, max_streak=3, rejection_wick=False, premium_rule="ALL_THREE",
    notes=("no price-action credit after more than 3 consecutive up candles (anti-chasing)",
           "breakout must be within 0.3 ATR of the 12-candle high",
           "full premium credit needs CE momentum AND bullish CE candle AND CE above its VWAP"))
SELL_PROFILE = DirectionProfile(
    SELL, "TREND_DOWN", rsi_full=(28, 45), rsi_half=(22, 50), max_ext_atr=2.0, max_vwap_ext_atr=2.0,
    max_break_atr=0.6, max_streak=None, rejection_wick=True, premium_rule="MOMENTUM_PLUS_ONE",
    notes=("extension up to 2 ATR tolerated", "upper-wick rejection counts as full price action",
           "full premium credit needs PE momentum plus a bullish PE candle OR PE above its VWAP"))
PROFILES = {BUY: BUY_PROFILE, SELL: SELL_PROFILE}


# ------------------------------------------------------------------ causal bars + regime
def classify_regimes(f: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Primary regime and volatility label per completed candle (inputs at or before that candle only)."""
    have = f[list(REGIME_INPUTS)].notna().all(axis=1) & ~f["gap_so_far"].astype(bool)
    up = (f["close"] > f["ema20"]) & (f["ema20"] > f["ema50"]) & (f["ema20_slope_atr"] >= TREND_SLOPE)
    dn = (f["close"] < f["ema20"]) & (f["ema20"] < f["ema50"]) & (f["ema20_slope_atr"] <= -TREND_SLOPE)
    chop = (f["flips12"] >= CHOP_FLIPS) | ((f["ema20_slope_atr"].abs() < FLAT_SLOPE)
                                           & (f["dist_ema20_atr"].abs() < FLAT_DIST))
    hi, lo = f["atr_pctile"] >= HIGH_VOL, f["atr_pctile"] <= LOW_VOL
    vol = np.select([~have, hi, lo], ["UNKNOWN", "HIGH_VOLATILITY", "LOW_VOLATILITY"], "NORMAL_VOLATILITY")
    regime = np.select([~have, chop, up, dn, hi, lo],
                       ["UNCERTAIN", "RANGE_CHOP", "TREND_UP", "TREND_DOWN", "HIGH_VOLATILITY", "LOW_VOLATILITY"],
                       "UNCERTAIN")
    return pd.Series(regime, index=f.index), pd.Series(vol, index=f.index)


def s3_bars(spot: pd.DataFrame, interval: int = 5) -> pd.DataFrame:
    f = acc.family_bars(spot)
    day = f["day"]
    ts = f.groupby("day")["timestamp"]
    step = pd.Timedelta(minutes=interval)
    diff = ts.diff()
    bad = (diff.notna() & (diff != step)) | (ts.transform("first").dt.time != time(9, 15))
    f["gap_so_far"] = bad.astype(int).groupby(day).cummax().astype(bool)
    sgn = np.sign(f.groupby("day")["close"].diff())
    flip = ((sgn * sgn.groupby(day).shift(1)) < 0).astype(int)
    f["flips12"] = flip.groupby(day).transform(lambda s: s.rolling(12, min_periods=12).sum())
    f["vwap_ref"] = f["vwap"].where(f["vwap"].notna(), f["twap"])
    f["vwap_source"] = np.where(f["vwap"].notna(), "VWAP", "TWAP_PROXY")
    f["vwap_slope_atr"] = (f["vwap_ref"] - f.groupby("day")["vwap_ref"].shift(3)) / f["atr14"]
    f["regime"], f["vol_regime"] = classify_regimes(f)
    return f


# ------------------------------------------------------------------ direction engine
def underlying_points(f: pd.DataFrame, prof: DirectionProfile) -> pd.DataFrame:
    """Underlying component points (max 80) for one direction; every input is known at the candle close."""
    up = prof.direction == BUY
    s = 1.0 if up else -1.0
    out = pd.DataFrame(index=f.index)
    trend_half = ((f["ema20"] - f["ema50"]) * s > 0) & (f["intraday_move"] * s > 0)
    out["trend"] = np.where(f["regime"] == prof.trend_regime, WEIGHTS["trend"],
                            np.where(trend_half, WEIGHTS["trend"] // 2, 0))
    d = (f["close"] - f["vwap_ref"]) * s / f["atr14"]
    slope_ok = f["vwap_slope_atr"] * s > 0
    out["vwap"] = np.where((d > 0) & (d <= prof.max_vwap_ext_atr) & slope_ok, WEIGHTS["vwap"],
                           np.where(d > 0, 7, 0))
    stack = (((f["close"] > f["ema9"]) & (f["ema9"] > f["ema20"]) & (f["ema20"] > f["ema50"])) if up else
             ((f["close"] < f["ema9"]) & (f["ema9"] < f["ema20"]) & (f["ema20"] < f["ema50"])))
    ext = f["dist_ema20_atr"] * s
    out["ema"] = (np.where(stack & (ext <= prof.max_ext_atr), 10, 0)
                  + np.where(f["ema20_slope_atr"] * s >= TREND_SLOPE, 5, 0))
    out["momentum"] = np.where(f["rsi14"].between(*prof.rsi_full), WEIGHTS["momentum"],
                               np.where(f["rsi14"].between(*prof.rsi_half), 5, 0))
    dirc = (f["close"] > f["open"]) if up else (f["close"] < f["open"])
    strong = (f["close_pos"] >= 0.7) if up else (f["close_pos"] <= 0.3)
    pa = np.where(dirc & (f["body_frac"] >= 0.5) & strong, WEIGHTS["price_action"],
                  np.where(dirc & (f["body_frac"] >= 0.3), 5, 0))
    if prof.rejection_wick:
        pa = np.where((f["upper_wick"] >= 0.5) & (f["close_pos"] <= 0.4), WEIGHTS["price_action"], pa)
    if prof.max_streak is not None:
        pa = np.where(f["up_streak" if up else "down_streak"] > prof.max_streak, 0, pa)
    out["price_action"] = pa
    lvl12, lvl6 = (f["hh12"], f["hh6"]) if up else (f["ll12"], f["ll6"])
    d12, d6 = (f["close"] - lvl12) * s / f["atr14"], (f["close"] - lvl6) * s / f["atr14"]
    out["breakout"] = np.where((d12 > 0) & (d12 <= prof.max_break_atr), WEIGHTS["breakout"],
                               np.where((d6 > 0) & (d6 <= 2 * prof.max_break_atr), 5, 0))
    out["underlying"] = out[list(UNDERLYING_COMPONENTS)].sum(axis=1)
    out["available"] = (f[list(SCORE_INPUTS)].notna().all(axis=1) & ~f["gap_so_far"].astype(bool)
                        & (f["regime"] != "UNCERTAIN"))
    out["tradeable_regime"] = ~f["regime"].isin(NO_TRADE_REGIMES)
    out["regime"] = f["regime"]
    return out


def option_points(opt: dict | None, prof: DirectionProfile) -> dict[str, Any] | None:
    """Option points (volume 10 + premium 10) from premium candles <= signal. None = required data missing."""
    if opt is None or opt.get("premium_mom3") is None or opt.get("premium_body") is None:
        return None
    vr = opt.get("premium_vol_ratio")
    volume = 0 if vr is None else (WEIGHTS["volume"] if vr >= 1.5 else 5 if vr >= 1.0 else 0)
    mom = opt["premium_mom3"] > 0
    bull = bool(opt["premium_bull"]) and opt["premium_body"] >= 0.4
    above = opt.get("premium_above_vwap") is True
    if prof.premium_rule == "ALL_THREE":
        hits = mom + bull + above
        prem = WEIGHTS["option_premium"] if hits == 3 else 5 if hits == 2 else 0
    else:
        prem = WEIGHTS["option_premium"] if mom and (bull or above) else 5 if mom else 0
    return {"volume": volume, "option_premium": prem, "volume_available": vr is not None}


def score_tier(score: float | None) -> str | None:
    if score is None:
        return None
    for name, lo, hi in TIERS:
        if lo <= score <= hi:
            return name
    return None


@dataclass(frozen=True)
class S3Decision:
    decision: str
    score: float | None
    tier: str | None
    reason: str | None
    components: dict[str, float]


def decide(direction: str, points: dict | pd.Series, opt: dict | None, threshold: float) -> S3Decision:
    """One direction at one completed candle. NO_SIGNAL is returned whenever confidence is not established."""
    prof = PROFILES[direction]
    comps = {k: float(points[k]) for k in UNDERLYING_COMPONENTS}
    if not bool(points["available"]):
        return S3Decision(NO_SIGNAL, None, None, "DATA_UNAVAILABLE", comps)
    if not bool(points["tradeable_regime"]):
        return S3Decision(NO_SIGNAL, None, None, f"REGIME_{points['regime']}", comps)
    if float(points["underlying"]) < PRE_FILTER:
        return S3Decision(NO_SIGNAL, float(points["underlying"]), None, "UNDERLYING_BELOW_PRE_FILTER", comps)
    op = option_points(opt, prof)
    if op is None:
        return S3Decision(NO_SIGNAL, None, None, "PREMIUM_DATA_UNAVAILABLE", comps)
    comps.update(volume=float(op["volume"]), option_premium=float(op["option_premium"]))
    score = sum(comps.values())
    if score < threshold:
        return S3Decision(NO_SIGNAL, score, score_tier(score), "SCORE_BELOW_THRESHOLD", comps)
    return S3Decision(direction, score, score_tier(score), None, comps)


def resolve_conflict(buy: S3Decision, sell: S3Decision) -> str:
    """Both directions qualifying on the same candle is ambiguous -> NO_SIGNAL."""
    b, s = buy.decision == BUY, sell.decision == SELL
    return NO_SIGNAL if b == s else (BUY if b else SELL)
