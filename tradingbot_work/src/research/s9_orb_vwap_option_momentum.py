"""S9_ORB_VWAP_OPTION_MOMENTUM (PAPER | RESEARCH ONLY | NO ORDERS).

A new, separate strategy family (S2 is not used, modified or extended). 5-minute NIFTY 50, BUY -> ATM CE, SELL -> ATM
PE; both are BOUGHT options.

  opening range   09:15-09:30 candles only (09:15, 09:20, 09:25); finalized before any signal
  entry window    signal candle 09:30..13:55 (entry at the next candle's open, never after 14:00)
  signal          completed candle: close beyond OR by >= 0.05% of price, beyond VWAP, candle direction,
                  body/range >= 0.50, close in the outer 30% of the range, volume ratio >= 1.20
                  (current volume / mean of the PRIOR 20 completed candles)
  option          causal ATM strike from the signal close, nearest listed expiry on the signal date, contract must
                  exist in the dataset; confirmation on the option candle at the signal label (completed before
                  entry): CE close > open and > previous close; PE close < open and < previous close (spec rule 9,
                  literal - confirmed by the user)
  entry           OPEN of the option candle at signal + 5 min (missing -> UNRESOLVED)
  stop / target   1.5 x option ATR14 (completed candles <= signal candle; project framework) / 2R
  exit            first touch; SL and target in one candle -> AMBIGUOUS (booked at -1R, never a win);
                  otherwise close of the last candle ending at or before 14:30
  one trade       one active S9 trade at a time; a breakout re-arms only after a close back inside the range

DATA LIMITATION (disclosed): the NSE_INDEX|Nifty 50 candles carry zero volume. EXACT mode applies the specification
literally (VWAP and volume ratio from index volume -> NO_SIGNAL). PROXY mode (the evaluated base, fixed before any
result was seen) uses the project's existing fallbacks: VWAP -> session TWAP of the typical price, volume ratio ->
selected option's signal-candle volume / mean of its prior 20 completed candles.

Chronology (historical backtest accuracy, never live accuracy): TRAIN Apr-Jun discovery / eligibility, VALIDATION
July, OOS August (frozen), FINAL HOLDOUT September (once, NOT USED FOR SELECTION).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest.research_strategies import ResearchSignal
from src.indicators.technical import atr
from src.options.models import OptionResolutionError
from src.options.strategy import OptionSignalError, option_stop_loss, option_target, round_to_tick
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s5_targeted_strategy as s5
from src.research import s6_clean_strategy_lab as s6

HEADING = "S9 ORB + VWAP + OPTION MOMENTUM COMPLETE"
STRATEGY = "S9_ORB_VWAP_OPTION_MOMENTUM"
BANNER = cb.BANNER
IST = imp.IST
SPLITS = cb.SPLITS
SPLIT_NAMES = {"TRAIN": "TRAIN", "VALIDATION": "VALIDATION", "OOS": "OOS AUGUST",
               "FINAL_HOLDOUT": "FINAL HOLDOUT SEPTEMBER"}
SCOPES, DIRECTIONS = ("BUY", "SELL", "COMBINED"), ("BUY", "SELL")
SCOPE_NAMES = {"BUY": "BUY CE", "SELL": "SELL PE", "COMBINED": "COMBINED"}
PRE_OOS = ("TRAIN", "VALIDATION")
SELECTION_END = SPLITS["VALIDATION"][1]
MONTHS = (("2026-04", "APRIL"), ("2026-05", "MAY"), ("2026-06", "JUNE"), ("2026-07", "JULY"),
          ("2026-08", "AUGUST"), ("2026-09", "SEPTEMBER"))
REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "CHOPPY")
HOLDOUT_LABEL = "FINAL HOLDOUT — NOT USED FOR SELECTION"
STEP = pd.Timedelta(minutes=5)

OR_START, OR_END, OR_CANDLES = time(9, 15), time(9, 30), 3
ENTRY_START, LAST_ENTRY, SQUARE_OFF, SESSION_END = time(9, 30), time(14, 0), time(14, 30), time(15, 30)
VOLUME_LOOKBACK = 20
ATR_PERIOD, STOP_ATR, TARGET_R = 14, 1.5, 2.0
LOT = sh.LOT_SIZE
TARGET = 0.70

PROXY, EXACT, FUTURES = "PROXY", "EXACT", "FUTURES"
LITERAL_FALLING_PE, RISING_PREMIUM = "LITERAL_FALLING_PE", "RISING_PREMIUM"
WIN, LOSS, AMBIGUOUS, UNRESOLVED, NO_SIGNAL, SCRATCH = "WIN", "LOSS", "AMBIGUOUS", "UNRESOLVED", "NO_SIGNAL", \
    "SCRATCH"
EXECUTED = (WIN, LOSS, AMBIGUOUS, SCRATCH)
READY = "READY"

# Pre-declared selection thresholds (fixed before any result was computed).
MIN_TRAIN_RESOLVED, MIN_VAL_RESOLVED, MIN_OOS_RESOLVED, MIN_HOLDOUT_RESOLVED = 30, 10, 20, 20
MAX_DD_SHARE, MAX_STREAK = 0.5, 12
DEGRADED_PF, PROMISING_OOS_PF = 0.8, 1.1
ELIGIBLE, PASS, DEGRADED, REJECTED, INSUFFICIENT = "ELIGIBLE", "PASS", "DEGRADED", "REJECTED", "INSUFFICIENT_DATA"
PROMISING, FAILED, PASSED_HO, FAILED_HO = "PROMISING", "FAILED", "PASSED", "FAILED"


@dataclass(frozen=True)
class Config:
    name: str = "BASE"
    breakout_pct: float = 0.05
    volume_ratio: float = 1.20
    body_frac: float = 0.50
    close_zone: float = 0.30
    mode: str = PROXY
    option_confirmation: str = LITERAL_FALLING_PE


BASE = Config()
EXACT_BASE = Config("BASE_EXACT", mode=EXACT)
VARIANTS: dict[str, tuple[str, Config]] = {
    "A": ("breakout threshold 0.05% (identical to base)", BASE),
    "B": ("breakout threshold 0.10%", Config("B", breakout_pct=0.10)),
    "C": ("volume ratio 1.20 (identical to base)", BASE),
    "D": ("volume ratio 1.50", Config("D", volume_ratio=1.50)),
    "E": ("body/range 0.50 (identical to base)", BASE),
    "F": ("body/range 0.60", Config("F", body_frac=0.60)),
}
DISTINCT = tuple(dict.fromkeys(c for _, c in VARIANTS.values()))


def _num(v) -> float | None:
    return dg._num(v)


def _ts(v) -> pd.Timestamp:
    return dra._ts(v)


# ------------------------------------------------------------------ NIFTY features (causal, per completed candle)
def nifty_features(spot: pd.DataFrame) -> pd.DataFrame:
    """Every value at candle t uses candles <= t only. OR = 09:15/09:20/09:25 candles, published from 09:30."""
    f = spot[["timestamp", "open", "high", "low", "close"]].copy()
    f["volume"] = spot["volume"].astype(float) if "volume" in spot else np.nan
    f["timestamp"] = dra._ist_series(f["timestamp"])
    f = f.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    t = f["timestamp"].dt.time
    day = f["timestamp"].dt.date
    f["day"] = day
    in_or = (t >= OR_START) & (t < OR_END)
    published = t >= OR_END
    n_or = in_or.astype(int).groupby(day).transform("sum")
    f["or_complete"] = published & (n_or == OR_CANDLES)
    f["or_high"] = f["high"].where(in_or).groupby(day).transform("max").where(f["or_complete"])
    f["or_low"] = f["low"].where(in_or).groupby(day).transform("min").where(f["or_complete"])
    session = (t >= OR_START) & (t < SESSION_END)
    tp = (f["high"] + f["low"] + f["close"]) / 3
    f["twap"] = tp.where(session).groupby(day).cumsum() / session.astype(int).groupby(day).cumsum()
    v = f["volume"].where(session)
    cum_v = v.fillna(0.0).groupby(day).cumsum()
    f["vwap"] = ((tp * v).fillna(0.0).groupby(day).cumsum() / cum_v).where(cum_v > 0)
    prior = f["volume"].shift(1).rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean()
    f["volume_ratio_index"] = (f["volume"] / prior).where(prior > 0)
    f["range"] = f["high"] - f["low"]
    return f.set_index("timestamp")


FEATURE_COLS = ("or_high", "or_low", "or_complete", "twap", "vwap", "volume_ratio_index", "range")


def in_window(ts: pd.Timestamp) -> bool:
    return ts.time() >= ENTRY_START and (ts + STEP).time() <= LAST_ENTRY and (ts + STEP).date() == ts.date()


def candle_strength(row: dict, side: str) -> tuple[float | None, float | None]:
    """(body / range, close position from the trade side's extreme as a share of range)."""
    rng = row["high"] - row["low"]
    if not rng > 0:
        return None, None
    if side == "BUY":
        return (row["close"] - row["open"]) / rng, (row["close"] - row["low"]) / rng
    return (row["open"] - row["close"]) / rng, (row["high"] - row["close"]) / rng


def vwap_ref(row: dict, cfg: Config) -> float | None:
    return _num(row["twap"] if cfg.mode == PROXY else row["vwap"])


def strike_reference(row: dict) -> float | None:
    """NIFTY spot close used for ATM strike selection. Futures-driven rows carry it in ``spot_close``."""
    return _num(row["spot_close"]) if "spot_close" in row else _num(row["close"])


VWAP_SOURCES = {PROXY: "SESSION_TWAP (index volume is zero)", EXACT: "INDEX_VWAP",
                FUTURES: "FUTURES_SESSION_VWAP (real futures volume)"}
VOLUME_SOURCES = {PROXY: "SELECTED_OPTION_VOLUME", EXACT: "INDEX_VOLUME", FUTURES: "FUTURES_VOLUME"}


def underlying_check(row: dict, side: str, cfg: Config) -> tuple[bool, str | None]:
    """Signal-candle conditions on the completed NIFTY candle only. Returns (passed, first failing reason)."""
    if not bool(row.get("or_complete")):
        return False, "OR_INCOMPLETE"
    c, o = row["close"], row["open"]
    hi, lo = row["or_high"], row["or_low"]
    if side == "BUY" and not c > hi:
        return False, "NO_BREAKOUT"
    if side == "SELL" and not c < lo:
        return False, "NO_BREAKOUT"
    ref = vwap_ref(row, cfg)
    if ref is None:
        return False, "VWAP_UNAVAILABLE"
    if (side == "BUY" and not c > ref) or (side == "SELL" and not c < ref):
        return False, "WRONG_SIDE_OF_VWAP"
    if (side == "BUY" and not c > o) or (side == "SELL" and not c < o):
        return False, "WRONG_CANDLE_DIRECTION"
    dist = c - hi if side == "BUY" else lo - c
    if dist < cfg.breakout_pct / 100.0 * c:
        return False, "BREAKOUT_TOO_SMALL"
    body, pos = candle_strength(row, side)
    if body is None:
        return False, "ZERO_RANGE"
    if body < cfg.body_frac:
        return False, "WEAK_BODY"
    if pos < 1.0 - cfg.close_zone:
        return False, "CLOSE_NOT_AT_EXTREME"
    if cfg.mode in (EXACT, FUTURES):
        vr = _num(row.get("volume_ratio_index"))
        if vr is None:
            return False, "VOLUME_UNAVAILABLE"
        if vr < cfg.volume_ratio:
            return False, "LOW_VOLUME"
    return True, None


# ------------------------------------------------------------------ option leg (completed option candles only)
def premium_frame(contract, premium_for: Callable, cache: dict) -> tuple | None:
    k = contract.instrument_key
    if k not in cache:
        p = premium_for(contract)
        if p is None or isinstance(p, str) or p.empty:
            cache[k] = None
        else:
            p = p.copy()
            p["timestamp"] = dra._ist_series(p["timestamp"])
            dups = int(p["timestamp"].duplicated().sum())
            p = p.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
            cache[k] = (p, pd.DatetimeIndex(p["timestamp"]), dups)
    return cache[k]


def _candle_text(p: pd.DataFrame, j: int, t: pd.DatetimeIndex) -> str:
    vals = {c: float(p[c].iat[j]) for c in ("open", "high", "low", "close")}
    vol = float(p["volume"].iat[j]) if "volume" in p else float("nan")
    return f"{t[j].isoformat()} O={vals['open']:g} H={vals['high']:g} L={vals['low']:g} C={vals['close']:g} V={vol:g}"


def option_leg(prem: pd.DataFrame, t: pd.DatetimeIndex, sig_ts: pd.Timestamp, side: str, cfg: Config,
               tick: float | None) -> dict[str, Any]:
    """Confirmation on the option candle at the signal label (completed when the entry candle opens), entry at the
    next candle's OPEN, ATR14 from candles <= the signal candle, SL 1.5 x ATR, target 2R."""
    j = int(t.get_indexer([sig_ts])[0])
    if j < 0:
        return {"reason": "OPTION_SIGNAL_CANDLE_MISSING"}
    if j < 1:
        return {"reason": "OPTION_PREVIOUS_CANDLE_MISSING"}
    o, c = float(prem["open"].iat[j]), float(prem["close"].iat[j])
    pc = float(prem["close"].iat[j - 1])
    out: dict[str, Any] = {"option_signal_candle": _candle_text(prem, j, t), "option_prev_close": pc}
    if cfg.mode == PROXY:
        if j < VOLUME_LOOKBACK or "volume" not in prem:
            return {**out, "reason": "OPTION_VOLUME_HISTORY_UNAVAILABLE"}
        avg = float(prem["volume"].iloc[j - VOLUME_LOOKBACK:j].mean())
        v = _num(prem["volume"].iat[j])
        if v is None or not avg > 0:
            return {**out, "reason": "OPTION_VOLUME_UNAVAILABLE"}
        out["option_volume_ratio"] = v / avg
        if v / avg < cfg.volume_ratio:
            return {**out, "reason": "LOW_VOLUME"}
    rising = side == "BUY" or cfg.option_confirmation == RISING_PREMIUM
    confirmed = (c > o and c > pc) if rising else (c < o and c < pc)
    if not confirmed:
        return {**out, "reason": "OPTION_NOT_CONFIRMED"}
    out.update(fired=True, outcome=UNRESOLVED)
    entry_ts = sig_ts + STEP
    k = int(t.get_indexer([entry_ts])[0])
    if k < 0:
        return {**out, "reason": "NO_OPTION_ENTRY_CANDLE"}
    entry = round_to_tick(float(prem["open"].iat[k]), tick, "nearest")
    out.update(entry=entry, entry_time=entry_ts.isoformat(), option_entry_candle=_candle_text(prem, k, t))
    a = atr(prem.iloc[:j + 1].reset_index(drop=True), ATR_PERIOD).iloc[-1]
    if pd.isna(a):
        return {**out, "reason": "ATR_UNAVAILABLE"}
    try:
        stop = option_stop_loss(entry, stop_mode="premium_atr", multiplier=STOP_ATR, premium_atr=float(a),
                                tick_size=tick)
        target = option_target(entry, stop, TARGET_R, tick)
    except OptionSignalError as exc:
        return {**out, "atr": float(a), "reason": f"LEVELS_INVALID:{exc.code}"}
    return {**out, "atr": float(a), "stop_loss": stop, "target": target, "outcome": READY, "reason": None}


def simulate(prem: pd.DataFrame, t: pd.DatetimeIndex, entry_ts: pd.Timestamp, entry: float, stop: float,
             target: float) -> dict[str, Any]:
    """First touch on real option candles from the entry candle; exit by the last candle ending <= 14:30."""
    cut = pd.Timestamp(datetime.combine(entry_ts.date(), SQUARE_OFF), tz=IST)
    idx = np.where((t >= entry_ts) & (t < cut))[0]
    if not len(idx):
        return {"outcome": UNRESOLVED, "reason": "NO_OPTION_CANDLES_AFTER_ENTRY"}
    risk = entry - stop

    def done(outcome: str, price: float, j: int, why: str) -> dict[str, Any]:
        end = t[j] + STEP
        r = -1.0 if outcome == AMBIGUOUS else (price - entry) / risk
        px = stop if outcome == AMBIGUOUS else price
        return {"outcome": outcome, "exit_price": px, "exit_time": end.isoformat(), "exit_reason": why,
                "r_multiple": round(r, 4), "pnl": round((px - entry) * LOT, 2),
                "holding_minutes": (end - entry_ts).total_seconds() / 60, "reason": why}

    for j in idx:
        o, h, l = float(prem["open"].iat[j]), float(prem["high"].iat[j]), float(prem["low"].iat[j])
        if o <= stop:
            return done(LOSS, o, j, "STOP_LOSS_GAP")
        if o >= target:
            return done(WIN, o, j, "TARGET_GAP")
        hit_s, hit_t = l <= stop, h >= target
        if hit_s and hit_t:
            return done(AMBIGUOUS, stop, j, "SL_AND_TARGET_SAME_CANDLE")
        if hit_s:
            return done(LOSS, stop, j, "STOP_LOSS")
        if hit_t:
            return done(WIN, target, j, "TARGET")
    j = int(idx[-1])
    c = float(prem["close"].iat[j])
    return done(WIN if c > entry else LOSS if c < entry else SCRATCH, c, j, "EOD_EXIT_1430")


# ------------------------------------------------------------------ replay (chronological, one active trade)
def _record(row: dict, side: str, cfg: Config) -> dict[str, Any]:
    ts = row["timestamp"]
    body, pos = candle_strength(row, side)
    return {"strategy": STRATEGY, "variant": cfg.name, "mode": cfg.mode, "signal_time": ts.isoformat(),
            "session_date": ts.date().isoformat(), "direction": side, "option_type": "CE" if side == "BUY" else "PE",
            "spot": strike_reference(row), "underlying_close": float(row["close"]),
            "futures_contract": row.get("futures_contract"), "or_high": _num(row["or_high"]),
            "or_low": _num(row["or_low"]), "vwap": vwap_ref(row, cfg), "vwap_source": VWAP_SOURCES[cfg.mode],
            "nifty_volume_ratio": _num(row.get("volume_ratio_index")), "option_volume_ratio": None,
            "volume_source": VOLUME_SOURCES[cfg.mode], "option_confirmation_rule": cfg.option_confirmation,
            "candle_body_frac": body, "candle_close_position": pos, "contract": None, "instrument_key": None,
            "strike": None, "expiry_date": None, "tick": None, "option_signal_candle": None,
            "option_entry_candle": None, "entry_time": None, "entry": None, "atr": None, "stop_loss": None,
            "target": None, "exit_price": None, "exit_time": None, "exit_reason": None, "r_multiple": None,
            "pnl": None, "holding_minutes": None, "fired": False, "outcome": NO_SIGNAL, "reason": None,
            "state": None}


def attempt(row: dict, side: str, cfg: Config, resolve: Callable, premium_for: Callable, cache: dict) -> dict:
    rec = _record(row, side, cfg)
    if rec["spot"] is None:
        return dict(rec, reason="SPOT_CANDLE_MISSING")
    sig = ResearchSignal(row["timestamp"].to_pydatetime(), side, "S9", "NIFTY", rec["spot"], {})
    try:
        c = resolve(sig)
    except (OptionResolutionError, ValueError) as exc:
        return dict(rec, reason=f"CONTRACT_UNAVAILABLE:{getattr(exc, 'code', type(exc).__name__)}")
    rec.update(contract=getattr(c, "trading_symbol", None), instrument_key=c.instrument_key,
               strike=float(c.strike), expiry_date=c.expiry, tick=getattr(c, "tick_size", None))
    pf = premium_frame(c, premium_for, cache)
    if pf is None:
        return dict(rec, reason="OPTION_DATA_MISSING")
    return {**rec, **option_leg(pf[0], pf[1], row["timestamp"], side, cfg, rec["tick"])}


def replay(feat: pd.DataFrame, days: Sequence, resolve: Callable, premium_for: Callable, cfg: Config,
           cache: dict | None = None) -> tuple[list[dict], Counter, dict]:
    """Candle by candle, day by day. Underlying-only failures are counted (funnel); every candle that passes the
    underlying conditions becomes a record (NO_SIGNAL with reason, UNRESOLVED, or a trade)."""
    cache = {} if cache is None else cache
    want = {d if isinstance(d, date) else date.fromisoformat(str(d)[:10]) for d in days}
    out: list[dict] = []
    funnel: Counter = Counter()
    active_until: pd.Timestamp | None = None
    for day, g in feat[feat["day"].isin(sorted(want))].groupby("day", sort=True):
        armed = {"BUY": True, "SELL": True}
        for row in g.reset_index().to_dict("records"):
            ts = row["timestamp"]
            if _num(row["or_high"]) is not None and row["close"] <= row["or_high"]:
                armed["BUY"] = True
            if _num(row["or_low"]) is not None and row["close"] >= row["or_low"]:
                armed["SELL"] = True
            if not in_window(ts):
                continue
            for side in DIRECTIONS:
                ok, why = underlying_check(row, side, cfg)
                if not ok:
                    funnel[f"{side}:{why}"] += 1
                    continue
                if not armed[side]:
                    funnel[f"{side}:REPEATED_BREAKOUT"] += 1
                    continue
                rec = attempt(row, side, cfg, resolve, premium_for, cache)
                out.append(rec)
                if not rec["fired"]:
                    funnel[f"{side}:{rec['reason']}"] += 1
                    continue
                armed[side] = False
                if active_until is not None and ts + STEP < active_until:
                    rec.update(outcome=NO_SIGNAL, reason="ACTIVE_TRADE_OPEN", entry=None, entry_time=None,
                               stop_loss=None, target=None)
                    funnel[f"{side}:ACTIVE_TRADE_OPEN"] += 1
                    continue
                if rec["outcome"] != READY:
                    funnel[f"{side}:{rec['reason']}"] += 1
                    continue
                prem, t, _ = cache[rec["instrument_key"]]
                rec.update(simulate(prem, t, ts + STEP, rec["entry"], rec["stop_loss"], rec["target"]))
                if rec["outcome"] in EXECUTED:
                    rec["state"] = sh.EVALUATED
                    active_until = _ts(rec["exit_time"])
                funnel[f"{side}:{rec['outcome']}"] += 1
    return out, funnel, cache


# ------------------------------------------------------------------ metrics
def side_of(recs: Sequence[dict], scope: str) -> list[dict]:
    return list(recs) if scope == "COMBINED" else [r for r in recs if r["direction"] == scope]


def split_of(recs: Sequence[dict], split: str) -> list[dict]:
    lo, hi = SPLITS[split]
    return [r for r in recs if lo <= cb.day_of(r) <= hi]


def _pf(gp: float, gl: float) -> float | None:
    return gp / gl if gl > 0 else (math.inf if gp > 0 else None)


def metrics(recs: Sequence[dict]) -> dict[str, Any]:
    recs = sorted(recs, key=lambda r: (_ts(r["signal_time"]), r["direction"]))
    fired = [r for r in recs if r["fired"]]
    ex = [r for r in fired if r["outcome"] in EXECUTED]
    n = Counter(r["outcome"] for r in ex)
    wins, losses = n[WIN], n[LOSS]
    pnl = [float(r["pnl"]) for r in ex]
    rs = [float(r["r_multiple"]) for r in ex]
    gp, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    eq = peak = dd = 0.0
    streak = worst = 0
    for r, p in zip(ex, pnl):
        eq += p
        peak, dd = max(peak, eq), max(dd, peak - eq)
        streak = streak + 1 if r["outcome"] in (LOSS, AMBIGUOUS) else 0
        worst = max(worst, streak)
    resolved = wins + losses
    hold = [float(r["holding_minutes"]) for r in ex if r.get("holding_minutes") is not None]
    return {
        "total_signals": len(fired), "executable_trades": len(ex), "wins": wins, "losses": losses,
        "ambiguous": n[AMBIGUOUS], "scratch": n[SCRATCH],
        "unresolved": sum(r["outcome"] == UNRESOLVED for r in fired),
        "blocked_active_trade": sum(r["reason"] == "ACTIVE_TRADE_OPEN" for r in fired),
        "resolved": resolved, "accuracy": wins / resolved if resolved else None,
        "accuracy_text": f"{wins} wins / {resolved} resolved trades" + (f" = {wins / resolved:.1%}" if resolved
                                                                         else ""),
        "profit_factor": _pf(gp, gl) if ex else None, "gross_profit": round(gp, 2), "gross_loss": round(gl, 2),
        "net_pnl": round(gp - gl, 2), "max_drawdown": round(dd, 2), "max_losing_streak": worst,
        "expectancy_r": round(sum(rs) / len(rs), 4) if rs else None,
        "average_r": round(sum(rs) / len(rs), 4) if rs else None, "median_r": round(median(rs), 4) if rs else None,
        "average_holding_minutes": round(sum(hold) / len(hold), 1) if hold else None,
        "trade_retention": round(len(ex) / len(fired), 4) if fired else None,
    }


def table(recs: Sequence[dict]) -> dict[str, dict[str, dict]]:
    return {k: {s: metrics(side_of(split_of(recs, k), s)) for s in SCOPES} for k in SPLITS}


def _pfv(m: dict) -> float:
    v = m.get("profit_factor")
    return 0.0 if v is None else min(float(v), 10.0)


# ------------------------------------------------------------------ TRAIN / VALIDATION / OOS / HOLDOUT labels
def train_label(m: dict) -> dict[str, Any]:
    checks = {"meaningful_sample": m["resolved"] >= MIN_TRAIN_RESOLVED,
              "positive_expectancy": (m["expectancy_r"] or 0.0) > 0, "profit_factor_above_1": _pfv(m) > 1.0,
              "controlled_drawdown": m["gross_profit"] > 0 and m["max_drawdown"] <= MAX_DD_SHARE * m["gross_profit"],
              "reasonable_losing_streak": m["max_losing_streak"] <= MAX_STREAK}
    label = INSUFFICIENT if not checks["meaningful_sample"] else ELIGIBLE if all(checks.values()) else REJECTED
    return {"label": label, "checks": checks, "failed": [k for k, ok in checks.items() if not ok]}


def validation_label(m: dict) -> str:
    if m["resolved"] < MIN_VAL_RESOLVED:
        return INSUFFICIENT
    if _pfv(m) >= 1.0 and (m["expectancy_r"] or 0.0) > 0:
        return PASS
    return DEGRADED if _pfv(m) >= DEGRADED_PF else REJECTED


def oos_label(m: dict) -> str:
    if m["resolved"] < MIN_OOS_RESOLVED:
        return INSUFFICIENT
    return PROMISING if _pfv(m) >= PROMISING_OOS_PF and (m["expectancy_r"] or 0.0) > 0 and m["net_pnl"] > 0 \
        else FAILED


def holdout_label(m: dict) -> str:
    if m["resolved"] < MIN_HOLDOUT_RESOLVED:
        return INSUFFICIENT
    return PASSED_HO if m["net_pnl"] > 0 and _pfv(m) >= 1.0 else FAILED_HO


def s9_status(train: dict, val: str, oos: str, clean: bool) -> str:
    if not clean:
        return REJECTED
    if train["label"] == INSUFFICIENT:
        return INSUFFICIENT
    if train["label"] != ELIGIBLE:
        return REJECTED
    if val == INSUFFICIENT:
        return INSUFFICIENT
    if val != PASS:
        return REJECTED
    if oos == INSUFFICIENT:
        return INSUFFICIENT
    return PROMISING if oos == PROMISING else REJECTED


def _assert_pre(recs_by_cfg: dict) -> None:
    late = sum(1 for v in recs_by_cfg.values() for r in v if cb.day_of(r) > SELECTION_END)
    if late:
        raise ValueError(f"selection received {late} records after {SELECTION_END} (OOS/holdout must not be used)")


def pre(recs_by_cfg: dict) -> dict:
    return {k: [r for r in v if cb.day_of(r) <= SELECTION_END] for k, v in recs_by_cfg.items()}


def select(recs_by_cfg: dict[str, list[dict]]) -> dict[str, Any]:
    """TRAIN eligibility + July label per configuration (COMBINED scope). The base is the frozen candidate; a
    variant is only reported as a validated alternative when it is TRAIN-ELIGIBLE and July PASS on its own."""
    _assert_pre(recs_by_cfg)
    out: dict[str, Any] = {"train": {}, "validation": {}, "train_metrics": {}, "validation_metrics": {}}
    for name, recs in recs_by_cfg.items():
        tm = metrics(split_of(recs, "TRAIN"))
        vm = metrics(split_of(recs, "VALIDATION"))
        out["train"][name], out["validation"][name] = train_label(tm), validation_label(vm)
        out["train_metrics"][name], out["validation_metrics"][name] = tm, vm

    def ok(n: str) -> bool:
        return out["train"][n]["label"] == ELIGIBLE and out["validation"][n] == PASS

    def score(n: str) -> tuple:
        m = out["train_metrics"][n]
        return ((m["expectancy_r"] or -math.inf) * math.sqrt(max(m["resolved"], 1)), _pfv(m), n)

    others = sorted((n for n in recs_by_cfg if n != BASE.name and ok(n)), key=score, reverse=True)
    out["base_passes_train_and_validation"] = BASE.name in recs_by_cfg and ok(BASE.name)
    out["validated_variant"] = others[0] if others else None
    return out


def fingerprint(sel: dict) -> str:
    spec = {"base": asdict(BASE), "variants": {k: asdict(c) for k, (_, c) in VARIANTS.items()},
            "train": {k: v["label"] for k, v in sel["train"].items()}, "validation": sel["validation"],
            "validated_variant": sel["validated_variant"]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def _sel_ids(sel: dict) -> tuple:
    return (tuple(sorted((k, v["label"]) for k, v in sel["train"].items())),
            tuple(sorted(sel["validation"].items())), sel["validated_variant"])


def isolation(recs_by_cfg: dict, sel: dict) -> dict[str, Any]:
    aug = select(pre({k: s5._flip(v, SELECTION_END) for k, v in recs_by_cfg.items()}))
    sep = select(pre({k: s5._flip(v, SPLITS["OOS"][1]) for k, v in recs_by_cfg.items()}))
    refuses = False
    try:
        select(recs_by_cfg)
    except ValueError:
        refuses = True
    rev = select({k: list(reversed(v)) for k, v in pre(recs_by_cfg).items()})
    checks = {"august_cannot_affect_selection": _sel_ids(aug) == _sel_ids(sel),
              "september_cannot_affect_selection": _sel_ids(sep) == _sel_ids(sel),
              "selection_refuses_post_july_records": refuses,
              "selection_deterministic": fingerprint(rev) == fingerprint(sel),
              "at_most_6_predefined_variants": len(VARIANTS) <= 6}
    return {"checks": checks, "passed": all(checks.values())}


# ------------------------------------------------------------------ anti-lookahead checks
def _same(a, b) -> bool:
    return s6._same(a, b)


def feature_truncation_check(spot: pd.DataFrame, feat: pd.DataFrame, recs: Sequence[dict],
                             sample: int = 20) -> dict[str, Any]:
    """Features and signal decisions at candle t recomputed with every later NIFTY candle removed."""
    window = [ts for ts in feat.index if in_window(ts)]
    sig = sorted({_ts(r["signal_time"]) for r in recs})
    picked = sorted(set(window[::max(1, len(window) // max(sample, 1))][:sample]) |
                    set(sig[::max(1, len(sig) // max(sample, 1))][:sample]))
    t_all = dra._ist_series(spot["timestamp"])
    bad = []
    for ts in picked:
        tf = nifty_features(spot[(t_all <= ts).to_numpy()].reset_index(drop=True))
        a, b = feat.loc[ts], tf.loc[ts]
        bad += [f"{ts.isoformat()}|{c}" for c in FEATURE_COLS
                if not _same(_num(a[c]) if c != "or_complete" else bool(a[c]),
                             _num(b[c]) if c != "or_complete" else bool(b[c]))]
        ra, rb = {**a.to_dict(), "timestamp": ts}, {**b.to_dict(), "timestamp": ts}
        for cfg in (*DISTINCT, EXACT_BASE):
            for side in DIRECTIONS:
                if underlying_check(ra, side, cfg) != underlying_check(rb, side, cfg):
                    bad.append(f"{ts.isoformat()}|{cfg.name}|{side}|decision")
    return {"checked": len(picked), "mismatches": bad[:20], "passed": bool(picked) and not bad}


def _decision(r: dict, cut: pd.Timestamp) -> tuple:
    base = (r["signal_time"], r["direction"], r["fired"], r["instrument_key"], r["option_signal_candle"],
            None if r["option_volume_ratio"] is None else round(r["option_volume_ratio"], 9))
    if r.get("entry_time") and _ts(r["entry_time"]) <= cut:
        base += (r["entry"], None if r["atr"] is None else round(r["atr"], 9), r["stop_loss"], r["target"])
    return base


def scrambling_check(spot: pd.DataFrame, days: Sequence[str], resolve: Callable, premium_for: Callable,
                     cfg: Config, seed: int = 11) -> dict[str, Any]:
    """Every NIFTY and option candle after 12:00 rescaled at random: all decisions at or before 12:00 identical."""
    fails = []
    feat = nifty_features(spot)
    for i, d in enumerate(days):
        cut = pd.Timestamp(datetime.combine(date.fromisoformat(d), time(12, 0)), tz=IST)
        a = replay(feat, [d], resolve, premium_for, cfg)[0]
        b = replay(nifty_features(imp.mutate_after(spot, cut, seed + i)), [d], resolve,
                   lambda c, _cut=cut, _i=i: imp._mut_prem(premium_for(c), _cut, seed + _i), cfg)[0]
        pa = [_decision(r, cut) for r in a if _ts(r["signal_time"]) <= cut]
        pb = [_decision(r, cut) for r in b if _ts(r["signal_time"]) <= cut]
        if pa != pb:
            fails.append(cut.isoformat())
    return {"cuts": len(days), "failures": fails, "passed": bool(days) and not fails}


def option_truncation_check(recs: Sequence[dict], cache: dict, sample: int = 20,
                            configs: Sequence[Config] = ()) -> dict[str, Any]:
    """Option candles after the entry candle tripled: confirmation, option volume ratio, entry, ATR, SL and target
    unchanged (they use only option candles completed before entry plus the entry candle's open)."""
    ent = [r for r in recs if r.get("entry_time") and r["instrument_key"] in cache and cache[r["instrument_key"]]]
    picked = ent[::max(1, len(ent) // max(sample, 1))][:sample]
    bad = []
    cfgs = {c.name: c for c in (*DISTINCT, EXACT_BASE, *configs)}
    for r in picked:
        prem, t, _ = cache[r["instrument_key"]]
        warped = s6._warp_after(prem, _ts(r["entry_time"]))
        keys = ("option_volume_ratio", "entry", "atr", "stop_loss", "target", "option_signal_candle")
        again = {"fired": False, **dict.fromkeys(keys), **option_leg(warped, t, _ts(r["signal_time"]),
                                                                     r["direction"], cfgs[r["variant"]], r["tick"])}
        bad += [f"{r['signal_time']}|{r['direction']}|{k}" for k in ("fired", *keys)
                if not _same(r.get(k), again.get(k))]
    return {"checked": len(picked), "mismatches": bad[:20], "passed": not bad, "vacuous": not picked}


def chronology_check(recs: Sequence[dict]) -> dict[str, Any]:
    ts = [_ts(r["signal_time"]) for r in recs]
    ex = [r for r in recs if r["outcome"] in EXECUTED]
    bad_entry = sum(1 for r in recs if r.get("entry_time") and _ts(r["entry_time"]) != _ts(r["signal_time"]) + STEP)
    late = sum(1 for r in recs if r.get("entry_time") and _ts(r["entry_time"]).time() > LAST_ENTRY)
    exit_bad = sum(1 for r in ex if not (_ts(r["entry_time"]) < _ts(r["exit_time"]) <=
                                         pd.Timestamp(datetime.combine(_ts(r["entry_time"]).date(), SQUARE_OFF),
                                                      tz=IST)))
    overlap = sum(1 for a, b in zip(ex, ex[1:]) if _ts(b["entry_time"]) < _ts(a["exit_time"]))
    ordered = all(x <= y for x, y in zip(ts, ts[1:]))
    return {"ordered": ordered, "entry_not_next_candle": bad_entry, "entry_after_1400": late,
            "exit_outside_entry_to_1430": exit_bad, "overlapping_active_trades": overlap,
            "passed": ordered and not bad_entry and not late and not exit_bad and not overlap}


def determinism_check(recs: Sequence[dict], feat: pd.DataFrame, days: Sequence[str], resolve: Callable,
                      premium_for: Callable, cfg: Config) -> dict[str, Any]:
    """The same days replayed twice (fresh caches) give identical records, equal to the full-run slice."""
    a = replay(feat, days, resolve, premium_for, cfg)[0]
    b = replay(feat, days, resolve, premium_for, cfg)[0]
    want = set(days)
    full = [r for r in recs if r["session_date"] in want]
    key = lambda rs: [json.dumps({k: v for k, v in r.items()}, sort_keys=True, default=str) for r in rs]  # noqa: E731
    return {"days": len(days), "repeatable": key(a) == key(b), "equals_full_run": key(a) == key(full),
            "passed": bool(days) and key(a) == key(b) == key(full)}


def contract_checks(recs: Sequence[dict], calendar, listed: Callable, exists: Callable) -> dict[str, Any]:
    res = [dra.contract_check(r, calendar, listed, exists) for r in recs if r.get("instrument_key")]
    bad = [x for x in res if not x["valid"]]
    fails = Counter(k for x in bad for k, v in x.items() if isinstance(v, bool) and not v and k != "valid")
    return {"checked": len(res), "invalid": len(bad), "failed_fields": dict(fails),
            "expired_contracts": fails.get("expiry_not_before_signal", 0), "invalid_strikes": fails.get("atm_strike", 0),
            "examples": [f"{x['timestamp']}|{x['direction']}|{x['contract']}" for x in bad[:5]], "passed": not bad}


# ------------------------------------------------------------------ descriptive analyses
def regime_of(bars: pd.DataFrame, r: dict) -> str:
    b = dg.bar_values(bars, _ts(r["signal_time"]))
    if not b:
        return dg.UNAVAILABLE
    return s6.market_regime(dg.classify_regime(b)[0], b.get("sign_changes"))


def monthly_rows(recs: Sequence[dict]) -> list[dict]:
    out = []
    for m, label in MONTHS:
        sub = [r for r in recs if r["session_date"][:7] == m]
        for s in SCOPES:
            x = metrics(side_of(sub, s))
            out.append({"month": label, "scope": SCOPE_NAMES[s], "trades": x["executable_trades"], "wins": x["wins"],
                        "losses": x["losses"], "accuracy": x["accuracy"], "accuracy_text": x["accuracy_text"],
                        "profit_factor": x["profit_factor"], "net_pnl": x["net_pnl"],
                        "max_drawdown": x["max_drawdown"]})
    return out


def regime_rows(recs: Sequence[dict], bars: pd.DataFrame) -> list[dict]:
    lab = {(r["signal_time"], r["direction"]): regime_of(bars, r) for r in recs if r["fired"]}
    out = []
    for split in ("ALL", *SPLITS):
        sub = recs if split == "ALL" else split_of(recs, split)
        for reg in (*REGIMES, dg.UNAVAILABLE):
            g = [r for r in sub if lab.get((r["signal_time"], r["direction"])) == reg]
            for s in SCOPES:
                x = metrics(side_of(g, s))
                out.append({"split": SPLIT_NAMES.get(split, split), "regime": reg, "scope": SCOPE_NAMES[s],
                            "descriptive_only": True, **{k: x[k] for k in RESULT_COLS}})
    return out


RESULT_COLS = ("total_signals", "executable_trades", "wins", "losses", "ambiguous", "scratch", "unresolved",
               "blocked_active_trade", "resolved", "accuracy", "accuracy_text", "profit_factor", "gross_profit",
               "gross_loss", "net_pnl", "max_drawdown", "max_losing_streak", "expectancy_r", "average_r", "median_r",
               "average_holding_minutes", "trade_retention")


def s2_view(m: dict) -> dict[str, Any]:
    return {"trades": m["total_trades"], "resolved": m["wins"] + m["losses"], "accuracy": m["win_rate"],
            "profit_factor": m["profit_factor"], "expectancy_r": m["average_r"], "net_pnl": m["net_pnl"],
            "max_drawdown": m["max_drawdown"], "max_losing_streak": m["max_losing_streak"]}


def s9_view(m: dict) -> dict[str, Any]:
    return {"trades": m["executable_trades"], "resolved": m["resolved"], "accuracy": m["accuracy"],
            "profit_factor": m["profit_factor"], "expectancy_r": m["expectancy_r"], "net_pnl": m["net_pnl"],
            "max_drawdown": m["max_drawdown"], "max_losing_streak": m["max_losing_streak"]}


# ------------------------------------------------------------------ study
def build(*, spot: pd.DataFrame, sessions: Sequence[str], resolve: Callable, premium_for: Callable,
          s2_rows: Sequence[dict], calendar, listed: Callable, exists: Callable, integrity: dict,
          excluded_sessions: dict, dataset: dict, mutation_days: Sequence[str], check_sample: int = 20,
          production_unchanged: bool | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    feat = nifty_features(spot)
    bars = dg.feature_bars(spot)

    say("S9 base (PROXY) chronological replay ...")
    base, funnel, cache = replay(feat, sessions, resolve, premium_for, BASE)
    say("S9 EXACT specification replay (index volume) ...")
    exact, exact_funnel, _ = replay(feat, sessions, resolve, premium_for, EXACT_BASE, cache)
    say("predefined robustness variants B / D / F (A, C, E are identical to the base) ...")
    by_cfg: dict[str, list[dict]] = {BASE.name: base}
    for c in DISTINCT:
        if c.name != BASE.name:
            by_cfg[c.name] = replay(feat, sessions, resolve, premium_for, c, cache)[0]

    say("anti-lookahead checks ...")
    lookahead = {
        "feature_truncation": feature_truncation_check(spot, feat, base, check_sample),
        "future_candle_scrambling": scrambling_check(spot, mutation_days, resolve, premium_for, BASE),
        "option_candle_truncation": option_truncation_check([r for v in by_cfg.values() for r in v], cache,
                                                            check_sample),
        "chronological_replay": chronology_check(base),
        "determinism": determinism_check(base, feat, mutation_days, resolve, premium_for, BASE),
        **{f"chronological_replay_variant_{k}": chronology_check(v) for k, v in by_cfg.items() if k != BASE.name},
    }
    lookahead_ok = all(bool(v.get("passed")) for v in lookahead.values())

    say("TRAIN eligibility and July validation (OOS / holdout untouched) ...")
    sel = select(pre(by_cfg))
    fp = fingerprint(sel)
    iso = isolation(by_cfg, sel)

    executed = [r for v in by_cfg.values() for r in v if r["outcome"] in EXECUTED]
    dup = {k: cb.duplicate_check([r for r in v if r["outcome"] in EXECUTED]) for k, v in by_cfg.items()}
    ts_chk = cb.timestamp_checks(executed, SQUARE_OFF.isoformat(timespec="minutes"),
                                 LAST_ENTRY.isoformat(timespec="minutes"))
    cval = contract_checks([r for v in by_cfg.values() for r in v], calendar, listed, exists)
    data_ok = bool(integrity.get("ok")) and all(d["passed"] for d in dup.values()) and ts_chk["passed"] and \
        cval["passed"]
    clean = lookahead_ok and data_ok and iso["passed"]

    tables = {"BASE": table(base), "EXACT": table(exact)}
    oos = tables["BASE"]["OOS"]["COMBINED"]
    oos_lab = oos_label(oos)
    status = s9_status(sel["train"][BASE.name], sel["validation"][BASE.name], oos_lab, clean)
    variant_frozen = None
    vv = sel["validated_variant"]
    if vv:
        variant_frozen = {"name": vv, "oos": table(by_cfg[vv])["OOS"], "oos_label": None}
        variant_frozen["oos_label"] = oos_label(variant_frozen["oos"]["COMBINED"])

    if fingerprint(sel) != fp:
        raise RuntimeError("selection changed before the final holdout")
    holdout = tables["BASE"]["FINAL_HOLDOUT"]["COMBINED"]
    ho_lab = holdout_label(holdout)
    if variant_frozen:
        variant_frozen["holdout"] = table(by_cfg[vv])["FINAL_HOLDOUT"]
        variant_frozen["holdout_label"] = holdout_label(variant_frozen["holdout"]["COMBINED"])

    if oos["resolved"] < MIN_OOS_RESOLVED or holdout["resolved"] < MIN_HOLDOUT_RESOLVED:
        target = "INSUFFICIENT DATA"
    elif (oos["accuracy"] or 0) >= TARGET and (holdout["accuracy"] or 0) >= TARGET:
        target = "ACHIEVED"
    else:
        target = "NOT ACHIEVED"
    safe = status == PROMISING and ho_lab == PASSED_HO and clean

    s2 = {k: {s: s2_view(cb.metrics(cb.side(cb.split_rows(s2_rows, k), s))) for s in SCOPES} for k in SPLITS}
    comparison = [{"split": SPLIT_NAMES[k], "scope": SCOPE_NAMES[s], "system": name, **view}
                  for k in SPLITS for s in SCOPES
                  for name, view in (("S2", s2[k][s]), ("S9", s9_view(tables["BASE"][k][s])))]
    reasons = Counter(r["reason"] for r in base if r["outcome"] in (NO_SIGNAL, UNRESOLVED))
    quality = {
        "data_integrity_ok": bool(integrity.get("ok")),
        "missing_nifty_candles": {"sessions_excluded": excluded_sessions,
                                  "sessions_used": len(sessions)},
        "missing_option_candles": {k: v for k, v in reasons.items() if k and ("MISSING" in k or "ENTRY_CANDLE" in k
                                                                              or "OPTION_DATA" in k)},
        "invalid_timestamps": ts_chk, "invalid_strikes": cval["invalid_strikes"],
        "expired_contracts": cval["expired_contracts"],
        "duplicate_nifty_candles": int(dra._ist_series(spot["timestamp"]).duplicated().sum()),
        "duplicate_option_candles": sum(v[2] for v in cache.values() if v),
        "duplicate_trades": {k: d["duplicate_trades"] for k, d in dup.items()},
        "unresolved_signals": sum(r["outcome"] == UNRESOLVED for r in base),
        "contract_coverage": {
            "attempted": len(base), "resolved_contract": sum(1 for r in base if r["instrument_key"]),
            "premium_available": sum(1 for r in base if r["instrument_key"] and cache.get(r["instrument_key"])),
            "unavailable_reasons": dict(Counter(r["reason"] for r in base if not r["instrument_key"]))},
        "index_volume_zero_share": float((spot["volume"].fillna(0) == 0).mean()) if "volume" in spot else None,
        "contract_validity": cval, "no_fabrication": "missing candles -> NO_SIGNAL / UNRESOLVED; nothing is "
                                                      "interpolated or synthesized",
    }
    return {
        "heading": HEADING, "strategy": STRATEGY, "banner": BANNER,
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "accuracy_definition": "historical backtest accuracy = wins / (wins + losses); AMBIGUOUS (booked at -1R), "
                               "SCRATCH and UNRESOLVED are never wins or losses",
        "dataset": dataset, "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        "sessions": len(sessions), "rules": rules_text(), "base_config": asdict(BASE),
        "exact_config": asdict(EXACT_BASE), "tables": tables, "funnel": dict(funnel),
        "exact_funnel": dict(exact_funnel), "exact_signals": sum(r["fired"] for r in exact),
        "variants": {k: {"description": d, "config": c.name} for k, (d, c) in VARIANTS.items()},
        "variant_tables": {k: table(v) for k, v in by_cfg.items() if k != BASE.name},
        "selection": sel, "fingerprint": fp, "isolation": iso, "variant_frozen": variant_frozen,
        "train_label": sel["train"][BASE.name], "validation_label": sel["validation"][BASE.name],
        "oos_label": oos_lab, "holdout_label": ho_lab, "status": status, "target_70": target,
        "lookahead": lookahead, "lookahead_ok": lookahead_ok, "data_integrity": integrity,
        "data_checks": {"duplicates": dup, "timestamps": ts_chk, "contract_validity": cval}, "data_ok": data_ok,
        "data_quality": quality, "s2": s2, "comparison": comparison, "monthly": monthly_rows(base),
        "regimes": regime_rows(base, bars), "forward_paper": "SAFE" if safe else "NOT SAFE",
        "disclosure": ("S2's August and September results were seen in earlier labs. S9 rules are the user's "
                       "specification, implemented before any S9 result was computed; the PROXY substitutions for "
                       "the missing index volume were fixed before any result was seen. Fresh forward PAPER data is "
                       "still required."),
        "production_unchanged": production_unchanged, "live_orders": 0, "trading_mode": "PAPER",
        "live_trading_enabled": False, "records": base, "exact_records": exact, "variant_records": by_cfg,
    }


def rules_text() -> dict[str, str]:
    return {
        "opening_range": "OR_HIGH / OR_LOW = max high / min low of the 09:15, 09:20, 09:25 candles (all three "
                         "required); published from the 09:30 candle",
        "entry_window": "signal candle 09:30..13:55 IST; entry at the next candle's open, never after 14:00",
        "vwap": "EXACT: session VWAP from index volume; PROXY: session TWAP of typical price (index volume is zero)",
        "breakout": "BUY close > OR_HIGH and close - OR_HIGH >= 0.05% of close, close > VWAP, close > open; SELL "
                    "mirrored",
        "volume": "ratio = current completed candle volume / mean of the PRIOR 20 completed candles >= 1.20; "
                  "EXACT: NIFTY index volume; PROXY: selected ATM option's own volume",
        "candle_strength": "body / range >= 0.50 and close in the upper (BUY) / lower (SELL) 30% of the range; "
                           "range <= 0 -> NO_SIGNAL",
        "option_selection": "ATM strike nearest the signal close, nearest listed expiry on the signal date "
                            "(project resolver, expired / unlisted contracts rejected); missing -> NO_SIGNAL",
        "option_confirmation": "option candle at the signal label (completed before entry): CE close > open and > "
                               "previous completed close; PE close < open and < previous completed close (spec "
                               "rule 9, literal)",
        "entry": "OPEN of the option candle at signal + 5 min; missing -> UNRESOLVED",
        "stop_target": f"SL = entry - {STOP_ATR:g} x option ATR{ATR_PERIOD} (completed candles <= signal candle, "
                       f"project premium-ATR framework, rounded to tick); target = entry + {TARGET_R:g}R",
        "exit": "first touch on real candles; SL and target in one candle -> AMBIGUOUS (-1R, never a win); "
                "otherwise close of the last candle ending at or before 14:30",
        "duplicates": "one active S9 trade at a time; a breakout re-arms only after a close back inside the range",
    }


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "s9_report.md", "trade_log.csv", "monthly_results.csv", "buy_results.csv",
                "sell_results.csv", "combined_results.csv", "regime_results.csv", "comparison_s2_vs_s9.csv",
                "data_quality.json", "variant_results.csv")
TRADE_COLS = ("signal_time", "direction", "spot", "or_high", "or_low", "vwap", "vwap_source", "nifty_volume_ratio",
              "option_volume_ratio", "volume_source", "candle_body_frac", "candle_close_position", "contract",
              "strike", "expiry_date", "option_type", "instrument_key", "option_signal_candle", "option_entry_candle",
              "entry_time", "entry", "atr", "stop_loss", "target", "exit_price", "exit_time", "exit_reason",
              "outcome", "r_multiple", "pnl", "holding_minutes", "reason", "session_date", "mode", "variant")


def _csv(path: Path, rows: Sequence[dict]) -> None:
    s6._csv(path, rows)


def _jsonable(v):
    return s6._jsonable(v)


def scope_csv(a: dict, scope: str) -> list[dict]:
    out = []
    for mode in ("BASE", "EXACT"):
        for k in SPLITS:
            m = a["tables"][mode][k][scope]
            out.append({"mode": "PROXY (evaluated base)" if mode == "BASE" else "EXACT (index volume)",
                        "split": SPLIT_NAMES[k], "scope": SCOPE_NAMES[scope],
                        "label": HOLDOUT_LABEL if k == "FINAL_HOLDOUT" else "", **{c: m[c] for c in RESULT_COLS}})
    return out


def variant_csv(a: dict) -> list[dict]:
    sel, out = a["selection"], []
    for k, (desc, c) in VARIANTS.items():
        t = a["tables"]["BASE"] if c.name == BASE.name else a["variant_tables"][c.name]
        for split in PRE_OOS:
            m = t[split]["COMBINED"]
            out.append({"variant": k, "description": desc, "config": c.name, "identical_to_base": c is BASE,
                        "split": split, "train_label": sel["train"][c.name]["label"],
                        "train_failed": ",".join(sel["train"][c.name]["failed"]),
                        "validation_label": sel["validation"][c.name], **{x: m[x] for x in RESULT_COLS}})
    vf = a["variant_frozen"]
    if vf:
        for split, key in (("OOS AUGUST", "oos"), (HOLDOUT_LABEL, "holdout")):
            m = vf[key]["COMBINED"]
            out.append({"variant": vf["name"], "split": split, "selected_from": "TRAIN + VALIDATION only",
                        "label": vf[f"{key}_label"], **{x: m[x] for x in RESULT_COLS}})
    return out


def _summary_json(a: dict) -> dict:
    skip = ("records", "exact_records", "variant_records", "data_integrity")
    return {k: v for k, v in a.items() if k not in skip}


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _pfs(v) -> str:
    return "N/A" if v is None else ("INF" if v == math.inf else f"{v:.2f}")


def line(m: dict) -> str:
    exp = "N/A" if m["expectancy_r"] is None else f"{m['expectancy_r']:+.3f}R"
    return (f"Trades {m['executable_trades']} | Resolved {m['resolved']} | Win Rate {_pct(m['accuracy'])} "
            f"({m['wins']}/{m['resolved']}) | PF {_pfs(m['profit_factor'])} | P&L Rs {m['net_pnl']:,.2f} | "
            f"Max DD Rs {m['max_drawdown']:,.2f} | Max Losing Streak {m['max_losing_streak']} | Expectancy {exp}")


def md_line(m: dict) -> str:
    return line(m).replace(" | ", "; ")


def view_text(v: dict) -> str:
    exp = "N/A" if v["expectancy_r"] is None else f"{v['expectancy_r']:+.3f}R"
    return (f"{v['trades']} trades, acc {_pct(v['accuracy'])} ({v['resolved']} resolved), PF "
            f"{_pfs(v['profit_factor'])}, exp {exp}, P&L {v['net_pnl']:,.2f}, DD {v['max_drawdown']:,.2f}, "
            f"streak {v['max_losing_streak']}")


def markdown(a: dict) -> str:
    q = a["data_quality"]
    md = [f"# {STRATEGY}", "", f"**{BANNER}**", "",
          f"Generated {a['generated_at']}. Dataset {a['dataset'].get('from')} -> {a['dataset'].get('to')} "
          "(data/historical/upstox_6m, NSE_INDEX|Nifty 50, real option premium candles). TRADING_MODE=PAPER, "
          "LIVE_TRADING_ENABLED=false, live orders: 0. Production changed: "
          f"**{'NO' if a['production_unchanged'] is not False else 'YES'}**.", "",
          "All numbers are **historical backtest accuracy**, not live accuracy; no profit is guaranteed. "
          f"{a['accuracy_definition']}.", "", f"> {a['disclosure']}", "",
          "## Result", "", f"- S9 status: **{a['status']}** (TRAIN {a['train_label']['label']}, July "
          f"{a['validation_label']}, August {a['oos_label']}, September {a['holdout_label']} - {HOLDOUT_LABEL})",
          f"- 70% target: **{a['target_70']}** | Lookahead **{'PASS' if a['lookahead_ok'] else 'FAIL'}** | Data "
          f"integrity **{'PASS' if a['data_ok'] else 'FAIL'}** | Forward PAPER validation **{a['forward_paper']}**",
          "", "## Rules (implemented exactly; no parameter search)", ""]
    md += [f"- **{k}**: {v}" for k, v in a["rules"].items()]
    md += ["", "## Index volume limitation", "",
           f"NIFTY index volume is zero on {_pct(q['index_volume_zero_share'])} of candles. EXACT specification "
           f"produced **{a['exact_signals']}** signals (funnel: {a['exact_funnel']}). The PROXY base (TWAP + selected "
           "option volume) is the evaluated strategy.", "", "## Base S9 (PROXY)", "",
           "| Split | Scope | Result |", "|---|---|---|"]
    md += [f"| {SPLIT_NAMES[k]} | {SCOPE_NAMES[s]} | {md_line(a['tables']['BASE'][k][s])} |" for k in SPLITS
           for s in SCOPES]
    md += ["", f"September rows: **{HOLDOUT_LABEL}**.", "", "## Signal funnel (base)", ""]
    md += [f"- {k}: {v}" for k, v in sorted(a["funnel"].items())]
    md += ["", "## Monthly", "", "| Month | Scope | Trades | Accuracy | PF | P&L | Max DD |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['month']} | {r['scope']} | {r['trades']} | {r['accuracy_text']} | {_pfs(r['profit_factor'])} | "
           f"{r['net_pnl']:,.2f} | {r['max_drawdown']:,.2f} |" for r in a["monthly"]]
    md += ["", "## Market regimes (descriptive only - never used to modify S9)", "",
           "| Regime | Scope | Trades | Accuracy | PF | P&L |", "|---|---|---|---|---|---|"]
    md += [f"| {r['regime']} | {r['scope']} | {r['executable_trades']} | {r['accuracy_text']} | "
           f"{_pfs(r['profit_factor'])} | {r['net_pnl']:,.2f} |" for r in a["regimes"] if r["split"] == "ALL"]
    md += ["", "## Robustness variants (selection from TRAIN + VALIDATION only)", "",
           "| Variant | Description | TRAIN | July | TRAIN label | July label |", "|---|---|---|---|---|---|"]
    for k, (desc, c) in VARIANTS.items():
        t = a["tables"]["BASE"] if c.name == BASE.name else a["variant_tables"][c.name]
        md.append(f"| {k} | {desc} | {md_line(t['TRAIN']['COMBINED'])} | {md_line(t['VALIDATION']['COMBINED'])} | "
                  f"{a['selection']['train'][c.name]['label']} | {a['selection']['validation'][c.name]} |")
    vf = a["variant_frozen"]
    md += ["", f"Validated alternative variant: {vf['name'] + ' (OOS ' + vf['oos_label'] + ', holdout ' + vf['holdout_label'] + ')' if vf else 'none'}. "
           "The base result stands on its own; a variant is never assumed better.", "",
           "## S2 vs S9", "", "| Split | Scope | S2 | S9 |", "|---|---|---|---|"]
    for k in SPLITS:
        for s in SCOPES:
            s2, s9 = a["s2"][k][s], s9_view(a["tables"]["BASE"][k][s])
            md.append(f"| {SPLIT_NAMES[k]} | {SCOPE_NAMES[s]} | {view_text(s2)} | {view_text(s9)} |")
    md += ["", "## Anti-lookahead and isolation", ""]
    md += [f"- {k}: {'PASS' if v.get('passed') else 'FAIL'}" for k, v in a["lookahead"].items()]
    md += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in a["isolation"]["checks"].items()]
    md += ["", "## Data quality", "", "```", json.dumps(_jsonable({k: v for k, v in q.items()
                                                                  if k != "contract_validity"}), indent=2,
                                                       default=str), "```", "", "## Safety", "",
           "S9 is research-only. Production S2, broker, OrderManager, RiskManager, credentials and live flags are "
           "unchanged. Nothing is promoted or activated automatically."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    p["summary.json"].write_text(json.dumps(_jsonable(_summary_json(a)), indent=2, default=str), encoding="utf-8")
    p["s9_report.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["trade_log.csv"], [{**{c: r.get(c) for c in TRADE_COLS},
                               "split": SPLIT_NAMES.get(cb.split_of(r) or "", ""),
                               "label": HOLDOUT_LABEL if cb.split_of(r) == "FINAL_HOLDOUT" else ""}
                              for r in a["records"]])
    _csv(p["monthly_results.csv"], a["monthly"])
    _csv(p["buy_results.csv"], scope_csv(a, "BUY"))
    _csv(p["sell_results.csv"], scope_csv(a, "SELL"))
    _csv(p["combined_results.csv"], scope_csv(a, "COMBINED"))
    _csv(p["regime_results.csv"], a["regimes"])
    _csv(p["comparison_s2_vs_s9.csv"], a["comparison"])
    _csv(p["variant_results.csv"], variant_csv(a))
    dq = {**a["data_quality"], "lookahead": a["lookahead"], "data_checks": a["data_checks"], "data_ok": a["data_ok"],
          "store_integrity": a["data_integrity"]}
    p["data_quality.json"].write_text(json.dumps(_jsonable(dq), indent=2, default=str), encoding="utf-8")
    return p


def terminal_text(a: dict) -> str:
    out = [HEADING, "(historical backtest accuracy - not live accuracy, no guaranteed profit)", "", "BASE S9", ""]
    for k in SPLITS:
        out.append(f"{SPLIT_NAMES[k]}:" + (f"  ({HOLDOUT_LABEL})" if k == "FINAL_HOLDOUT" else ""))
        out += [f"{s}: {line(a['tables']['BASE'][k][s])}" for s in SCOPES]
        out.append("")
    s2, s9 = a["s2"]["OOS"]["COMBINED"], s9_view(a["tables"]["BASE"]["OOS"]["COMBINED"])
    out += ["DIRECT S2 vs S9:", ""]
    for name, v in (("S2 OOS", s2), ("S9 OOS", s9)):
        out += [f"{name}:", f"accuracy: {_pct(v['accuracy'])} ({v['resolved']} resolved)",
                f"PF: {_pfs(v['profit_factor'])}", f"P&L: Rs {v['net_pnl']:,.2f}", f"DD: Rs {v['max_drawdown']:,.2f}",
                ""]
    out += ["S9 STATUS:", a["status"], "", "70% TARGET:", a["target_70"], "", "LOOKAHEAD:",
            "PASS" if a["lookahead_ok"] else "FAIL", "", "DATA INTEGRITY:", "PASS" if a["data_ok"] else "FAIL", "",
            "PRODUCTION CHANGED:", "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:", "0",
            "", "FORWARD PAPER VALIDATION:", a["forward_paper"], "", "NOTES:",
            f"base = PROXY (VWAP -> session TWAP, volume -> selected option volume; NIFTY index volume is zero); "
            f"EXACT specification signals: {a['exact_signals']}",
            f"stages: TRAIN {a['train_label']['label']} | July {a['validation_label']} | August {a['oos_label']} | "
            f"September {a['holdout_label']}",
            "validated variant: " + (a["variant_frozen"]["name"] if a["variant_frozen"] else "none")]
    return "\n".join(out)
