"""S10_ML_PROBABILITY - data layer (PAPER | RESEARCH ONLY | NO ORDERS).

Point-in-time samples for the S10 probability model. At every completed 5-minute candle t (signal candle start t,
closed at t + 5 min) inside the signal window:

  futures      causal front-month NIFTY future of the session (``nifty_futures.select_contracts``; nearest unexpired
               expiry from metadata, no substitution, no stitching); features from that contract's own candles <= t
  option       NIFTY SPOT close at t -> ATM listed strike (``atm_strike``), nearest listed expiry >= session date
               from the official expiry calendar; CE and PE candidates must both have a real option candle at t and
               a finite ATR14 from completed candles <= t, otherwise the timestamp is DATA_UNAVAILABLE (never rolled)
  entry        OPEN of the option candle at t + 5 min (missing -> NO_ENTRY_CANDLE, never a later candle)
  risk         R = 1.5 x option ATR14(candles <= t); STOP = entry - R; TARGET = entry + 2R (long CE / long PE only)
  label        first touch over the H option candles starting with the entry candle (H = 6 primary, 12 robustness):
               WIN / LOSS / AMBIGUOUS (both in one candle) / TIMEOUT (neither; exit at the last horizon close)

Nothing is interpolated, forward-filled or synthesised. Optional option volume / OI features are used only when the
raw fields are genuinely populated in TRAIN data (rule fixed before modelling).
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from src.backtest import shadow_historical as sh
from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import HistoricalContract
from src.indicators.technical import atr, ema
from src.options.models import OptionResolutionError
from src.options.selection import atm_strike
from src.research import s4_six_month_lab as s4

IST = u6.IST
STEP = pd.Timedelta(minutes=5)
SESSION_OPEN, FIRST_30_LAST, FIRST_30_END = time(9, 15), time(9, 40), time(9, 45)
SIGNAL_FIRST = time(9, 45)
LAST_EXIT_END = time(15, 15)
HORIZON, ROBUST_HORIZON = 6, 12
ATR_PERIOD, STOP_ATR, TARGET_R = 14, 1.5, 2.0
RANGE_LOOKBACK, VOLUME_LOOKBACK = 12, 20
LOT = sh.LOT_SIZE
SIDES = ("CE", "PE")
OTHER = {"CE": "PE", "PE": "CE"}

WIN, LOSS, AMBIGUOUS, TIMEOUT = "WIN", "LOSS", "AMBIGUOUS", "TIMEOUT"
NO_ENTRY, INVALID_LEVELS = "NO_ENTRY_CANDLE", "INVALID_LEVELS"
EXECUTED = (WIN, LOSS, AMBIGUOUS, TIMEOUT)
BINARY = (WIN, LOSS)

SPLITS = {"TRAIN": (date(2026, 4, 1), date(2026, 6, 30)),
          "VALIDATION": (date(2026, 7, 1), date(2026, 7, 31)),
          "OOS": (date(2026, 8, 1), date(2026, 8, 31)),
          "LEGACY_HOLDOUT": (date(2026, 9, 1), date(2026, 9, 30)),
          "FRESH_HOLDOUT": (date(2026, 10, 1), date(2026, 10, 6))}
INNER_TRAIN = (date(2026, 4, 1), date(2026, 5, 31))
INNER_VALIDATION = (date(2026, 6, 1), date(2026, 6, 30))
PRE_FREEZE_END = SPLITS["LEGACY_HOLDOUT"][1]
LEGACY_LABEL, FRESH_LABEL = "LEGACY_HOLDOUT_CONTAMINATED", "FRESH_UNSEEN_HOLDOUT"

# Optional raw-field availability rule (fixed before modelling, evaluated on TRAIN ATM option candles only).
OPTIONAL_MIN_PRESENT, OPTIONAL_MIN_POSITIVE = 0.95, 0.50

FUT_FEATURES = ("fut_ret_1", "fut_ret_3", "fut_ret_6", "fut_ret_12", "fut_body", "fut_range", "fut_body_range",
                "fut_upper_wick", "fut_lower_wick", "fut_natr", "fut_vol20", "fut_close_ema20", "fut_close_ema50",
                "fut_ema_sep", "fut_vwap_dist", "fut_vwap_slope", "fut_log_volume", "fut_volume_ratio",
                "fut_volume_z", "fut_range_pos", "fut_breakout_up", "fut_breakout_down", "fut_from_open", "fut_gap",
                "fut_prev_ret", "fut_or30_high_dist", "fut_or30_low_dist", "fut_day_range_atr", "fut_trend_persist")
TIME_FEATURES = ("minutes_since_open", "hour_bucket")
OPT_BASE = ("ret_1", "ret_2", "ret_3", "ret_6", "body", "range", "body_range", "natr", "mom", "dist_high", "dist_low")
OPT_VOLUME = ("vol_ratio", "log_volume")
OPT_OI = ("oi_chg_1", "oi_chg_6")
OPT_CONTRACT = ("dte", "moneyness")
REL_BASE = ("rel_ret_1", "rel_ret_3", "rel_mom", "rel_strength", "rel_natr")
REL_VOLUME = ("rel_vol_ratio",)


def feature_columns(volume: bool, oi: bool) -> list[str]:
    opt = [*OPT_BASE, *(OPT_VOLUME if volume else ()), *(OPT_OI if oi else ()), *OPT_CONTRACT]
    return [*FUT_FEATURES, *TIME_FEATURES, *(f"sel_{c}" for c in opt), *(f"opp_{c}" for c in opt), *REL_BASE,
            *(REL_VOLUME if volume else ())]


def last_signal(horizon: int) -> time:
    """Latest signal candle start whose entry candle + horizon candles end by LAST_EXIT_END."""
    return (datetime.combine(date(2000, 1, 1), LAST_EXIT_END) - (horizon + 1) * timedelta(minutes=5)).time()


def split_of_day(d: date) -> str | None:
    return next((k for k, (a, b) in SPLITS.items() if a <= d <= b), None)


def days_in(days: Iterable[date], lo: date, hi: date) -> list[date]:
    return sorted(d for d in days if lo <= d <= hi)


def _ist(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values)
    return ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)


def _clean(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    f = frame.copy()
    f["timestamp"] = _ist(f["timestamp"])
    return f.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------------ market data (read-only, point-in-time access)
@dataclass
class MarketData:
    spot: pd.DataFrame
    contracts: list[HistoricalContract]
    frames: dict[str, pd.DataFrame]
    calendar: list[date]
    chains: dict[tuple[date, date, str], dict[float, HistoricalContract]]
    atm: dict[date, set[str]]
    premium_fn: Callable[[str], pd.DataFrame | None]
    sessions: list[date]
    roots: list[str]
    reads: dict = field(default_factory=lambda: {"max_ts": None, "premium_calls": 0})
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def premium(self, key: str) -> pd.DataFrame | None:
        p = self.premium_fn(key)
        self.reads["premium_calls"] += 1
        if p is not None and not p.empty:
            mx = p["timestamp"].iloc[-1]
            if self.reads["max_ts"] is None or mx > self.reads["max_ts"]:
                self.reads["max_ts"] = mx
        return p

    def spot_close(self) -> dict[pd.Timestamp, float]:
        if "spot_close" not in self._cache:
            self._cache["spot_close"] = dict(zip(self.spot["timestamp"], self.spot["close"].astype(float)))
        return self._cache["spot_close"]

    def day_bars(self) -> dict[date, list[pd.Timestamp]]:
        if "day_bars" not in self._cache:
            out: dict[date, list[pd.Timestamp]] = defaultdict(list)
            for ts in self.spot["timestamp"]:
                out[ts.date()].append(ts)
            self._cache["day_bars"] = dict(out)
        return self._cache["day_bars"]

    def grid(self) -> np.ndarray:
        """Every stored NIFTY 5-minute bar start (int64 ns), the bar clock used for the embargo."""
        if "grid" not in self._cache:
            self._cache["grid"] = np.sort(pd.DatetimeIndex(self.spot["timestamp"]).asi8)
        return self._cache["grid"]


def load_market(roots: Sequence[str | Path], underlying: str = "NIFTY", max_day: date | None = None) -> MarketData:
    """Six-month store(s) + futures folder(s), merged chronologically. Futures contracts are merged by expiry (one
    listed NIFTY future per expiry) so a contract's own earlier candles are available as feature history."""
    stores = [s4.SixMonthStore(r, underlying) for r in roots]
    spots = [s.spot() for s in stores]
    parts = [p for p in spots if not p.empty]
    spot = _clean(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame(columns=["timestamp", "close"])
    sessions: set[date] = set()
    for s, sp in zip(stores, spots):
        if not sp.empty:
            sessions |= set(sh.complete_sessions(sp, s.start, s.end)[0])
    chains: dict = {}
    atm: dict[date, set[str]] = defaultdict(set)
    calendar: set[date] = set()
    for s in stores:
        for k, v in s.chain.items():
            chains.setdefault(k, {}).update(v)
        for d, keys in s.atm.items():
            atm[d] |= set(keys)
        calendar |= set(s.calendar)
    by_expiry: dict[date, tuple[HistoricalContract, list[pd.DataFrame]]] = {}
    for r in roots:
        cs, fr = nf.load_futures(r, underlying)
        for c in cs:
            got = [fr[c.instrument_key]] if c.instrument_key in fr else []
            prev = by_expiry.get(c.expiry)
            by_expiry[c.expiry] = (c, (prev[1] if prev else []) + got)
    contracts = sorted((c for c, _ in by_expiry.values()), key=lambda c: (c.expiry, c.instrument_key))
    frames = {c.instrument_key: _clean(pd.concat(fs, ignore_index=True)) for c, fs in by_expiry.values() if fs}
    frames = {k: v for k, v in frames.items() if v is not None}
    cache: dict[str, pd.DataFrame | None] = {}

    def premium_fn(key: str) -> pd.DataFrame | None:
        if key not in cache:
            got = [p for p in (s.premium(key) for s in stores) if p is not None and not p.empty]
            cache[key] = _clean(pd.concat(got, ignore_index=True)) if got else None
            if cache[key] is not None and max_day is not None:
                f = cache[key]
                cache[key] = _clean(f[f["timestamp"].dt.date <= max_day])
        return cache[key]

    if max_day is not None:
        spot = spot[spot["timestamp"].dt.date <= max_day].reset_index(drop=True)
        frames = {k: v[v["timestamp"].dt.date <= max_day].reset_index(drop=True) for k, v in frames.items()}
        frames = {k: v for k, v in frames.items() if not v.empty}
        sessions = {d for d in sessions if d <= max_day}
    return MarketData(spot=spot, contracts=contracts, frames=frames, calendar=sorted(calendar), chains=chains,
                      atm=dict(atm), premium_fn=premium_fn, sessions=sorted(sessions),
                      roots=[Path(r).as_posix() for r in roots])


def mutate_frame(frame: pd.DataFrame | None, cut: pd.Timestamp, *, price: bool = False, volume: bool = False,
                 oi: bool = False, drop: bool = False, seed: int = 0) -> pd.DataFrame | None:
    """Copy with every candle strictly after ``cut`` removed (drop) or rescaled at random. Earlier rows untouched."""
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    later = (_ist(out["timestamp"]) > cut).to_numpy()
    if drop:
        return out[~later].reset_index(drop=True)
    n = int(later.sum())
    rng = np.random.default_rng(seed)
    if price and n:
        k = rng.uniform(0.85, 1.15, n)
        for col in ("open", "high", "low", "close"):
            out[col] = out[col].astype(float)
            out.loc[later, col] = out.loc[later, col].to_numpy() * k
    for flag, col in ((volume, "volume"), (oi, "open_interest")):
        if flag and n and col in out:
            out[col] = out[col].astype(float)
            out.loc[later, col] = out.loc[later, col].to_numpy() * rng.uniform(0.2, 5.0, n)
    return out


def mutated_market(market: MarketData, cut: pd.Timestamp, *, spot: bool = True, futures: bool = True,
                   options: bool = True, seed: int = 0, **kw) -> MarketData:
    """Same market with candles after ``cut`` mutated (kw: price / volume / oi / drop) in the chosen sources."""
    base_fn = market.premium_fn
    cache: dict[str, pd.DataFrame | None] = {}

    def premium_fn(key: str) -> pd.DataFrame | None:
        if key not in cache:
            p = base_fn(key)
            cache[key] = mutate_frame(p, cut, seed=seed + 7, **kw) if options else p
        return cache[key]

    new_spot = mutate_frame(market.spot, cut, seed=seed, **kw) if spot else market.spot
    new_frames = {k: mutate_frame(f, cut, seed=seed + 3, **kw) for k, f in market.frames.items()} if futures \
        else market.frames
    return replace(market, spot=new_spot, frames=new_frames, premium_fn=premium_fn,
                   reads={"max_ts": None, "premium_calls": 0})


# ------------------------------------------------------------------ causal features
def futures_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per candle of ONE futures contract; every value at t uses that contract's candles <= t only."""
    f = _clean(frame)
    o, h, l, c = (f[k].astype(float) for k in ("open", "high", "low", "close"))
    v = f["volume"].astype(float) if "volume" in f else pd.Series(np.nan, index=f.index)
    day, t = f["timestamp"].dt.date, f["timestamp"].dt.time
    out = pd.DataFrame(index=f.index)
    out["timestamp"], out["day"] = f["timestamp"], day
    for k in (1, 3, 6, 12):
        out[f"fut_ret_{k}"] = c / c.shift(k) - 1.0
    rng = h - l
    out["fut_body"] = (c - o) / c
    out["fut_range"] = rng / c
    out["fut_body_range"] = ((c - o) / rng).where(rng > 0)
    out["fut_upper_wick"] = (h - np.maximum(o, c)) / c
    out["fut_lower_wick"] = (np.minimum(o, c) - l) / c
    a = atr(pd.DataFrame({"open": o, "high": h, "low": l, "close": c}), ATR_PERIOD)
    a = a.where(a > 0)
    out["fut_natr"] = a / c
    out["fut_vol20"] = c.pct_change().rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).std()
    e20, e50 = ema(c, 20), ema(c, 50)
    out["fut_close_ema20"] = (c - e20) / a
    out["fut_close_ema50"] = (c - e50) / a
    out["fut_ema_sep"] = (e20 - e50) / a
    tp = (h + l + c) / 3.0
    cum_v = v.groupby(day).cumsum()
    vw = ((tp * v).groupby(day).cumsum() / cum_v).where(cum_v > 0)
    out["fut_vwap_dist"] = (c - vw) / a
    out["fut_vwap_slope"] = (vw - vw.groupby(day).shift(3)) / a
    prior = v.shift(1).rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK)
    pm, ps = prior.mean(), prior.std()
    out["fut_log_volume"] = np.log1p(v.clip(lower=0))
    out["fut_volume_ratio"] = (v / pm).where(pm > 0)
    out["fut_volume_z"] = ((v - pm) / ps).where(ps > 0)
    hi = h.rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).max()
    lo = l.rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).min()
    out["fut_range_pos"] = ((c - lo) / (hi - lo)).where(hi > lo)
    phi = h.shift(1).rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).max()
    plo = l.shift(1).rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).min()
    out["fut_breakout_up"] = (c - phi) / a
    out["fut_breakout_down"] = (plo - c) / a
    day_open = o.groupby(day).transform("first")
    out["fut_from_open"] = c / day_open - 1.0
    closes = c.groupby(day).last()
    prev_close, prev2 = day.map(closes.shift(1)), day.map(closes.shift(2))
    out["fut_gap"] = day_open / prev_close - 1.0
    out["fut_prev_ret"] = prev_close / prev2 - 1.0
    in30 = (t >= SESSION_OPEN) & (t < FIRST_30_END)
    published = (t >= FIRST_30_LAST) & (in30.astype(int).groupby(day).transform("sum") == 6)
    hi30 = h.where(in30).groupby(day).transform("max")
    lo30 = l.where(in30).groupby(day).transform("min")
    out["fut_or30_high_dist"] = ((c - hi30) / a).where(published)
    out["fut_or30_low_dist"] = ((c - lo30) / a).where(published)
    out["fut_day_range_atr"] = (h.groupby(day).cummax() - l.groupby(day).cummin()) / a
    out["fut_trend_persist"] = (c.diff() > 0).astype(float).rolling(RANGE_LOOKBACK,
                                                                     min_periods=RANGE_LOOKBACK).mean()
    mins = (f["timestamp"] - f["timestamp"].dt.normalize()).dt.total_seconds() / 60.0 - (9 * 60 + 15)
    out["minutes_since_open"] = mins + 5.0
    out["hour_bucket"] = f["timestamp"].dt.hour.astype(float)
    num = [*FUT_FEATURES, *TIME_FEATURES]
    out[num] = out[num].astype(float).replace([np.inf, -np.inf], np.nan)
    return out.set_index("timestamp")


@dataclass
class OptionSeries:
    feat: pd.DataFrame
    t: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray


def option_series(premium: pd.DataFrame | None) -> OptionSeries | None:
    """Option features on ONE contract's stored candles; values at t use candles <= t only."""
    p = _clean(premium)
    if p is None:
        return None
    o, h, l, c = (p[k].astype(float) for k in ("open", "high", "low", "close"))
    nan = pd.Series(np.nan, index=p.index)
    v = p["volume"].astype(float) if "volume" in p else nan
    oi = p["open_interest"].astype(float) if "open_interest" in p else nan
    f = pd.DataFrame(index=p.index)
    for k in (1, 2, 3, 6):
        f[f"ret_{k}"] = c / c.shift(k) - 1.0
    rng = h - l
    f["body"], f["range"] = (c - o) / c, rng / c
    f["body_range"] = ((c - o) / rng).where(rng > 0)
    a = atr(pd.DataFrame({"open": o, "high": h, "low": l, "close": c}), ATR_PERIOD)
    f["atr14"] = a
    a = a.where(a > 0)
    f["natr"] = a / c
    f["mom"] = (c - c.shift(3)) / a
    f["dist_high"] = (c - h.rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).max()) / a
    f["dist_low"] = (c - l.rolling(RANGE_LOOKBACK, min_periods=RANGE_LOOKBACK).min()) / a
    prior = v.shift(1).rolling(VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean()
    f["vol_ratio"] = (v / prior).where(prior > 0)
    f["log_volume"] = np.log1p(v.clip(lower=0))
    for k in (1, 6):
        f[f"oi_chg_{k}"] = (oi / oi.shift(k) - 1.0).where(oi.shift(k) > 0)
    f = f.astype(float).replace([np.inf, -np.inf], np.nan)
    t = pd.DatetimeIndex(p["timestamp"])
    f.index = t
    return OptionSeries(f, t, o.to_numpy(), h.to_numpy(), l.to_numpy(), c.to_numpy())


def option_values(c: HistoricalContract, s: OptionSeries, j: int, day: date, spot: float, kind: str, volume: bool,
                  oi: bool) -> dict[str, float]:
    row = s.feat.iloc[j]
    cols = [*OPT_BASE, *(OPT_VOLUME if volume else ()), *(OPT_OI if oi else ())]
    out = {k: float(row[k]) for k in cols}
    k = float(c.strike)
    out["dte"] = float((c.expiry - day).days)
    out["moneyness"] = (spot - k) / spot if kind == "CE" else (k - spot) / spot
    return out


def relative(a: dict[str, float], b: dict[str, float], volume: bool) -> dict[str, float]:
    strength = math.log((1.0 + a["ret_3"]) / (1.0 + b["ret_3"])) \
        if np.isfinite(a["ret_3"]) and np.isfinite(b["ret_3"]) and a["ret_3"] > -1 and b["ret_3"] > -1 else np.nan
    out = {"rel_ret_1": a["ret_1"] - b["ret_1"], "rel_ret_3": a["ret_3"] - b["ret_3"], "rel_mom": a["mom"] - b["mom"],
           "rel_strength": strength, "rel_natr": a["natr"] - b["natr"]}
    if volume:
        out["rel_vol_ratio"] = a["vol_ratio"] - b["vol_ratio"]
    return out


# ------------------------------------------------------------------ labels (future option candles: label only)
def label_trade(s: OptionSeries, sig_ts: pd.Timestamp, atr_value: float, horizon: int,
                lot: int = LOT, target_r: float = TARGET_R) -> dict[str, Any]:
    """Entry at the OPEN of the option candle at sig_ts + 5 min; first touch over the next ``horizon`` candles
    (entry candle included). Both levels in one candle -> AMBIGUOUS, booked at the stop (-1R)."""
    entry_ts = sig_ts + STEP
    end = entry_ts + horizon * STEP
    out: dict[str, Any] = {"entry_time": entry_ts, "event_end": end, "entry": None, "stop_loss": None,
                           "target": None, "risk": None, "exit_time": None, "exit_price": None, "exit_reason": None,
                           "r_multiple": None, "pnl": None, "holding_minutes": None, "horizon_candles": 0,
                           "horizon_complete": False}
    k = int(s.t.get_indexer([entry_ts])[0])
    if k < 0:
        return {**out, "outcome": NO_ENTRY}
    entry = float(s.o[k])
    risk = STOP_ATR * float(atr_value)
    stop, target = entry - risk, entry + target_r * risk
    out.update(entry=entry, stop_loss=stop, target=target, risk=risk)
    if not (risk > 0 and stop > 0):
        return {**out, "outcome": INVALID_LEVELS}
    j_end = int(s.t.searchsorted(end, side="left"))
    idx = range(k, j_end)
    out.update(horizon_candles=len(idx), horizon_complete=len(idx) == horizon)

    def done(outcome: str, price: float, j: int, why: str) -> dict[str, Any]:
        exit_end = s.t[j] + STEP
        r = -1.0 if outcome == AMBIGUOUS else (price - entry) / risk
        return {**out, "outcome": outcome, "exit_time": exit_end, "exit_price": price, "exit_reason": why,
                "r_multiple": round(r, 6), "pnl": round((price - entry) * lot, 2),
                "holding_minutes": (exit_end - entry_ts).total_seconds() / 60.0}

    for j in idx:
        o, hi, lo = float(s.o[j]), float(s.h[j]), float(s.l[j])
        if o <= stop:
            return done(LOSS, o, j, "STOP_GAP")
        if o >= target:
            return done(WIN, o, j, "TARGET_GAP")
        hit_s, hit_t = lo <= stop, hi >= target
        if hit_s and hit_t:
            return done(AMBIGUOUS, stop, j, "STOP_AND_TARGET_SAME_CANDLE")
        if hit_s:
            return done(LOSS, stop, j, "STOP")
        if hit_t:
            return done(WIN, target, j, "TARGET")
    return done(TIMEOUT, float(s.c[idx[-1]]), idx[-1], "TIMEOUT_HORIZON_CLOSE")


# ------------------------------------------------------------------ point-in-time contract resolution
def nearest_expiry(calendar: Sequence[date], day: date) -> date | None:
    return next((e for e in sorted(calendar) if e >= day), None)


def resolve_contract(market: MarketData, day: date, expiry: date, kind: str,
                     spot_close: float) -> tuple[HistoricalContract | None, str | None]:
    chain = market.chains.get((day, expiry, kind)) or {}
    if not chain:
        return None, "CONTRACT_LIST_UNAVAILABLE"
    try:
        k = atm_strike(sorted(chain), spot_close)
    except OptionResolutionError as exc:
        return None, f"STRIKE_UNAVAILABLE:{exc.code}"
    c = chain[float(k)]
    if c.instrument_key not in market.atm.get(day, set()):
        return None, "OUTSIDE_PLANNED_STRIKES"
    return c, None


# ------------------------------------------------------------------ sample construction
@dataclass
class BuildResult:
    data: pd.DataFrame
    funnel: Counter
    opportunities: Counter
    missing_sessions: list[date]
    option_contracts: set[str]
    futures_contracts: set[str]


SAMPLE_COLS = ("signal_time", "session_date", "side", "instrument_key", "contract", "strike", "expiry", "spot",
               "futures_key", "futures_contract", "atr", "outcome", "y", "entry_time", "entry", "stop_loss",
               "target", "risk", "exit_time", "exit_price", "exit_reason", "r_multiple", "pnl", "holding_minutes",
               "horizon_complete", "event_end", "outcome_h12", "r_h12", "pnl_h12", "exit_time_h12", "holding_h12",
               "event_end_h12", "exit_reason_h12", "exit_price_h12")


def build_samples(market: MarketData, days: Iterable[date], *, volume: bool, oi: bool,
                  caches: dict | None = None, target_r: float = TARGET_R) -> BuildResult:
    caches = {"fut": {}, "opt": {}} if caches is None else caches
    days = sorted(set(days))
    want = [d for d in days if d in set(market.sessions)]
    sel = {date.fromisoformat(r["session_date"]): r for r in nf.select_contracts(market.contracts, market.frames,
                                                                                  want)}
    spot_close, day_bars = market.spot_close(), market.day_bars()
    last6, last12 = last_signal(HORIZON), last_signal(ROBUST_HORIZON)
    rows: list[dict] = []
    funnel: Counter = Counter()
    opp: Counter = Counter()
    used_opt, used_fut = set(), set()

    def series(c: HistoricalContract) -> OptionSeries | None:
        k = c.instrument_key
        if k not in caches["opt"]:
            caches["opt"][k] = option_series(market.premium(k))
        return caches["opt"][k]

    for d in want:
        sts = [ts for ts in day_bars.get(d, []) if SIGNAL_FIRST <= ts.time() <= last6]
        r = sel[d]
        if r["status"] != nf.SELECTED:
            funnel[f"FUTURES_{r['reason']}"] += len(sts)
            continue
        key = r["instrument_key"]
        if key not in caches["fut"]:
            caches["fut"][key] = futures_feature_frame(market.frames[key])
        ff = caches["fut"][key]
        expiry = nearest_expiry(market.calendar, d)
        if expiry is None:
            funnel["NO_LISTED_EXPIRY"] += len(sts)
            continue
        for ts in sts:
            if ts not in ff.index:
                funnel["FUTURES_CANDLE_MISSING"] += 1
                continue
            sc = spot_close[ts]
            legs: dict[str, tuple] = {}
            why = None
            for kind in SIDES:
                c, why = resolve_contract(market, d, expiry, kind, sc)
                if c is None:
                    break
                s = series(c)
                if s is None:
                    why = "OPTION_DATA_MISSING"
                    break
                j = int(s.feat.index.get_indexer([ts])[0])
                if j < 0:
                    why = "OPTION_SIGNAL_CANDLE_MISSING"
                    break
                a = float(s.feat["atr14"].iat[j])
                if not (np.isfinite(a) and a > 0):
                    why = "OPTION_ATR_UNAVAILABLE"
                    break
                legs[kind] = (c, s, j, a)
            if why:
                funnel[why] += 1
                continue
            funnel["ELIGIBLE"] += 1
            opp[d.isoformat()] += 1
            fr = ff.loc[ts]
            base = {k: float(fr[k]) for k in (*FUT_FEATURES, *TIME_FEATURES)}
            vals = {kind: option_values(c, s, j, d, sc, kind, volume, oi) for kind, (c, s, j, _) in legs.items()}
            used_fut.add(key)
            for kind in SIDES:
                c, s, j, a = legs[kind]
                used_opt.add(c.instrument_key)
                lab = label_trade(s, ts, a, HORIZON, target_r=target_r)
                l12 = label_trade(s, ts, a, ROBUST_HORIZON, target_r=target_r) if ts.time() <= last12 else None
                rec = {"signal_time": ts, "session_date": d.isoformat(), "side": kind,
                       "instrument_key": c.instrument_key, "contract": c.trading_symbol, "strike": float(c.strike),
                       "expiry": c.expiry.isoformat(), "spot": sc, "futures_key": key,
                       "futures_contract": r["trading_symbol"], "atr": a, **lab,
                       "y": 1.0 if lab["outcome"] == WIN else 0.0 if lab["outcome"] == LOSS else np.nan,
                       "outcome_h12": l12["outcome"] if l12 else None, "r_h12": l12["r_multiple"] if l12 else None,
                       "pnl_h12": l12["pnl"] if l12 else None, "exit_time_h12": l12["exit_time"] if l12 else None,
                       "holding_h12": l12["holding_minutes"] if l12 else None,
                       "event_end_h12": l12["event_end"] if l12 else None,
                       "exit_reason_h12": l12["exit_reason"] if l12 else None,
                       "exit_price_h12": l12["exit_price"] if l12 else None,
                       **base, **{f"sel_{k}": v for k, v in vals[kind].items()},
                       **{f"opp_{k}": v for k, v in vals[OTHER[kind]].items()},
                       **relative(vals[kind], vals[OTHER[kind]], volume)}
                rows.append(rec)
    cols = [*SAMPLE_COLS, *feature_columns(volume, oi)]
    data = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    data = data.sort_values(["signal_time", "side"]).reset_index(drop=True)
    return BuildResult(data, funnel, opp, [d for d in days if d not in set(market.sessions)], used_opt, used_fut)


def rows_between(data: pd.DataFrame, lo: date, hi: date) -> pd.DataFrame:
    return data[(data["session_date"] >= lo.isoformat()) & (data["session_date"] <= hi.isoformat())]


def label_counts(data: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {side: {o: int(((data["side"] == side) & (data["outcome"] == o)).sum())
                   for o in (*EXECUTED, NO_ENTRY, INVALID_LEVELS)} for side in SIDES}


def dataset_hash(data: pd.DataFrame, columns: Sequence[str]) -> str:
    if data.empty:
        return hashlib.sha256(b"").hexdigest()
    cols = ["signal_time", "side", "outcome", *columns]
    h = pd.util.hash_pandas_object(data[cols].astype(str), index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


# ------------------------------------------------------------------ Phase 1: data availability
def _share(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def availability_audit(market: MarketData, train: tuple[date, date] = SPLITS["TRAIN"]) -> dict[str, Any]:
    """What genuinely exists. Optional option volume / OI are used only if populated on TRAIN ATM candles."""
    sp = market.spot
    spot_vol = sp["volume"].astype(float) if "volume" in sp else pd.Series(dtype=float)
    fut_rows = sum(len(f) for f in market.frames.values())
    fut_cols = set().union(*(set(f.columns) for f in market.frames.values())) if market.frames else set()
    sessions = days_in(market.sessions, *train)
    sel = nf.select_contracts(market.contracts, market.frames, market.sessions)
    fm = []
    for r in sel:
        if r["status"] == nf.SELECTED:
            f = market.frames[r["instrument_key"]]
            fm.append(f[f["timestamp"].dt.date == date.fromisoformat(r["session_date"])])
    front = pd.concat(fm, ignore_index=True) if fm else pd.DataFrame(columns=["volume"])
    keys = sorted(set().union(*(market.atm.get(d, set()) for d in sessions))) if sessions else []
    frames = []
    for k in keys:
        p = market.premium(k)
        if p is not None:
            p = p[p["timestamp"].dt.date.map(lambda x: train[0] <= x <= train[1])]
            frames.append(p)
    opt = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    have = set(opt.columns)

    def field_stats(col: str) -> dict[str, Any]:
        if col not in have or opt.empty:
            return {"present": False, "non_null_share": 0.0, "positive_share": 0.0, "available": False}
        v = pd.to_numeric(opt[col], errors="coerce")
        nn, pos = _share(v.notna()), _share(v.fillna(0) > 0)
        return {"present": True, "non_null_share": nn, "positive_share": pos,
                "available": bool(nn is not None and nn >= OPTIONAL_MIN_PRESENT and pos is not None
                                  and pos >= OPTIONAL_MIN_POSITIVE)}

    kinds = Counter(str(x) for x in opt["option_type"]) if "option_type" in have else Counter()
    ohlc = {"open", "high", "low", "close", "timestamp"}
    req = {
        "futures_ohlc": ohlc <= fut_cols,
        "futures_volume": "volume" in fut_cols and bool(len(front)) and float((front["volume"] > 0).mean()) >= 0.98,
        "options_ce_ohlc": ohlc <= have and kinds.get("CE", 0) > 0,
        "options_pe_ohlc": ohlc <= have and kinds.get("PE", 0) > 0,
        "options_strike_expiry": {"strike", "expiry", "option_type"} <= have,
    }
    vol, oi = field_stats("volume"), field_stats("open_interest")
    return {
        "required": req, "required_ok": all(req.values()),
        "nifty_index": {"candles": int(len(sp)), "sessions": len(market.sessions),
                        "volume_zero_share": _share(spot_vol.fillna(0) == 0),
                        "note": "index volume is zero; volume features come from real NIFTY futures only"},
        "futures": {"contracts": len(market.contracts), "candles": int(fut_rows),
                    "front_month_candles": int(len(front)),
                    "front_month_zero_volume_share": _share(front["volume"].astype(float) == 0) if len(front) else None,
                    "open_interest_column": "open_interest" in fut_cols},
        "options": {"train_atm_contracts": len(keys), "train_atm_candles": int(len(opt)),
                    "ce_candles": int(kinds.get("CE", 0)), "pe_candles": int(kinds.get("PE", 0)),
                    "volume": vol, "open_interest": oi},
        "features": {"option_volume": "AVAILABLE" if vol["available"] else "UNAVAILABLE",
                     "option_oi_change": "AVAILABLE" if oi["available"] else "UNAVAILABLE",
                     "implied_volatility": "NOT_AVAILABLE (no raw IV field; not derived)",
                     "bid_ask_spread": "NOT_AVAILABLE (no quote data)",
                     "rule": f"optional raw field used only if non-null >= {OPTIONAL_MIN_PRESENT:.0%} and positive "
                             f">= {OPTIONAL_MIN_POSITIVE:.0%} on TRAIN ATM option candles (fixed before modelling)"},
        "use_option_volume": bool(vol["available"]), "use_option_oi": bool(oi["available"]),
    }
