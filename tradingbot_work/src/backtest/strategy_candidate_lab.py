"""NIFTY 5-minute strategy candidate lab (research only, PAPER, no orders).

Eight independent strategy families are evaluated on a small, predeclared
parameter grid. Every rule reads only the completed signal bar and earlier
bars. Option outcomes use real premium candles via the existing resolver and
the live tracker's first-touch rules; nothing is synthesised.

Chronology: TRAIN -> VALIDATION -> OOS are disjoint, ordered blocks of whole
sessions. The variant representing each family is chosen from TRAIN metrics
only; VALIDATION and OOS are never read during selection. The current-day
session is evaluated separately and never used for selection or status.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest.current_day_s2 import LevelParams, option_levels
from src.backtest.live_vs_historical import TIME_BUCKETS, time_bucket, wilson
from src.indicators.technical import atr, ema, rsi
from src.options.models import OptionResolutionError
from src.paper_trading.signal_tracker import AMBIGUOUS, INVALID, LOSS, WIN, SignalRecord, evaluate_candle

IST = ZoneInfo("Asia/Kolkata")
SPLITS = ("TRAIN", "VALIDATION", "OOS")
STATUSES = ("ROBUST", "PROMISING", "OVERFIT", "INSUFFICIENT_DATA", "REJECTED")
HISTORICAL_1DTE_UNAVAILABLE = "HISTORICAL_1DTE_DATA_UNAVAILABLE"
SCRATCH = "SCRATCH"
TRADED, SKIPPED_OPEN, SKIPPED_CAP, UNRESOLVED = "TRADED", "SKIPPED_POSITION_OPEN", "SKIPPED_DAILY_CAP", "UNRESOLVED"
SESSION_START, SESSION_LAST_BAR = time(9, 15), time(15, 25)
BREAKOUT_LOOKBACK = 12
TREND_SPREAD_ATR = 0.5
CURRENT_AUDIT_TIMES = tuple(f"{m // 60:02d}:{m % 60:02d}" for m in range(9 * 60 + 45, 14 * 60 + 31, 15))

FAMILIES = {
    "ORB_EMA": "ORB breakout + EMA20/EMA50 trend confirmation",
    "ORB_VWAP": "ORB breakout + VWAP confirmation",
    "ORB_RSI": "ORB breakout + RSI momentum confirmation",
    "EMA_VWAP_PULLBACK": "EMA trend + VWAP pullback",
    "BREAKOUT_ATR": "Breakout + ATR volatility confirmation",
    "TREND_RSI_VWAP": "Trend + RSI + VWAP confirmation",
    "PULLBACK_AFTER_BREAKOUT": "Pullback after breakout",
    "FIRST_BREAKOUT_ONLY": "Conservative first-breakout-only variant",
}


# ------------------------------------------------------------------ candidates
@dataclass(frozen=True)
class CandidateSpec:
    family: str
    params: tuple[tuple[str, Any], ...]
    target_r: float
    max_trades_per_day: int = 6

    @property
    def p(self) -> dict[str, Any]:
        return dict(self.params)

    @property
    def orb_minutes(self) -> int:
        return int(self.p.get("orb_minutes", 15))

    @property
    def candidate_id(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.family}[{args};tR={self.target_r:g}]"


def candidate_grid() -> list[CandidateSpec]:
    """The complete, predeclared research grid: 8 families x 2 rule values x 2 target R = 32 variants."""
    family_params: dict[str, list[dict[str, Any]]] = {
        "ORB_EMA": [{"orb_minutes": 15}, {"orb_minutes": 30}],
        "ORB_VWAP": [{"orb_minutes": 15}, {"orb_minutes": 30}],
        "ORB_RSI": [{"orb_minutes": 15, "rsi": 55}, {"orb_minutes": 15, "rsi": 60}],
        "EMA_VWAP_PULLBACK": [{"pullback_to": "EMA20"}, {"pullback_to": "VWAP"}],
        "BREAKOUT_ATR": [{"range_atr": 1.2}, {"range_atr": 1.5}],
        "TREND_RSI_VWAP": [{"rsi": 55}, {"rsi": 60}],
        "PULLBACK_AFTER_BREAKOUT": [{"orb_minutes": 15, "retest_atr": 0.0}, {"orb_minutes": 15, "retest_atr": 0.3}],
        "FIRST_BREAKOUT_ONLY": [{"orb_minutes": 15, "buffer_atr": 0.0}, {"orb_minutes": 15, "buffer_atr": 0.2}],
    }
    specs = []
    for family, variants in family_params.items():
        cap = 1 if family == "FIRST_BREAKOUT_ONLY" else 6
        for params in variants:
            for target_r in (1.0, 2.0):
                specs.append(CandidateSpec(family, tuple(params.items()), target_r, cap))
    return specs


@dataclass(frozen=True)
class CandidateSignal:
    timestamp: datetime
    direction: str
    underlying: str
    spot: float
    candidate_id: str
    family: str
    conditions: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


# -------------------------------------------------------------------- features
def _ist_series(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values)
    return ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)


def _ist(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def _session_only(frame: pd.DataFrame) -> pd.DataFrame:
    t = frame["timestamp"].dt.time
    return frame[(t >= SESSION_START) & (t <= SESSION_LAST_BAR)]


def build_features(spot: pd.DataFrame, futures: pd.DataFrame | None = None, *, orb_minutes: int = 15) -> pd.DataFrame:
    """Causal per-bar features: each row uses only that bar and earlier bars.

    ORB levels are NaN until the opening range is complete; VWAP comes from real
    futures volume (index volume is zero) and is NaN where futures are missing.
    """
    f = spot[["timestamp", "open", "high", "low", "close"]].copy()
    f["timestamp"] = _ist_series(f["timestamp"])
    f = _session_only(f).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        f[col] = f[col].astype(float)
    f["day"] = f["timestamp"].dt.date
    f["ema20"] = ema(f["close"], 20)
    f["ema50"] = ema(f["close"], 50)
    f["rsi14"] = rsi(f["close"], 14)
    f["atr14"] = atr(f, 14)
    prev_close = f["close"].shift(1)
    f["tr"] = pd.concat([f["high"] - f["low"], (f["high"] - prev_close).abs(), (f["low"] - prev_close).abs()],
                        axis=1).max(axis=1)
    f["atr14_prev"] = f["atr14"].shift(1)
    f["rsi_prev"] = f["rsi14"].shift(1)
    by_day = f.groupby("day", sort=False)
    f["prev_close_day"] = by_day["close"].shift(1)
    f["prior_high"] = by_day["high"].transform(
        lambda s: s.shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).max())
    f["prior_low"] = by_day["low"].transform(
        lambda s: s.shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).min())

    orb_end = (datetime.combine(date(2000, 1, 1), SESSION_START) + timedelta(minutes=orb_minutes)).time()
    t = f["timestamp"].dt.time
    in_orb = (t >= SESSION_START) & (t < orb_end)
    orb_hi = f["high"].where(in_orb).groupby(f["day"]).transform("max")
    orb_lo = f["low"].where(in_orb).groupby(f["day"]).transform("min")
    orb_n = in_orb.groupby(f["day"]).transform("sum")
    complete = (t >= orb_end) & (orb_n == orb_minutes // 5)
    f["orb_high"] = orb_hi.where(complete)
    f["orb_low"] = orb_lo.where(complete)

    fut_cols = ["fut_open", "fut_high", "fut_low", "fut_close", "fut_vwap"]
    if futures is not None and not futures.empty:
        fu = futures[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        fu["timestamp"] = _ist_series(fu["timestamp"])
        fu = _session_only(fu).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        day = fu["timestamp"].dt.date
        vol = fu["volume"].astype(float)
        pv = (fu["high"] + fu["low"] + fu["close"]).astype(float) / 3.0 * vol
        fu["fut_vwap"] = pv.groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0, np.nan)
        fu = fu.rename(columns={"open": "fut_open", "high": "fut_high", "low": "fut_low", "close": "fut_close"})
        f = f.merge(fu[["timestamp"] + fut_cols], on="timestamp", how="left")
    else:
        for col in fut_cols:
            f[col] = np.nan
    return f


# --------------------------------------------------------------------- signals
def _parse(value: str | time) -> time:
    return value if isinstance(value, time) else time.fromisoformat(value)


def _ok(*values) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in values)


def _snapshot(r) -> dict[str, Any]:
    keys = ("close", "ema20", "ema50", "rsi14", "atr14", "orb_high", "orb_low", "fut_close", "fut_vwap", "tr")
    return {k: (None if not _ok(getattr(r, k)) else round(float(getattr(r, k)), 4)) for k in keys}


def generate_signals(features: pd.DataFrame, spec: CandidateSpec, *, underlying: str = "NIFTY",
                     entry_start: str | time = "09:30", stop_new_entries: str | time = "14:30") -> list[CandidateSignal]:
    """Signals on completed bars; a decision at row i reads rows <= i only (state is carried forward)."""
    p = spec.p
    start, stop = _parse(entry_start), _parse(stop_new_entries)
    out: list[CandidateSignal] = []
    seen: set[tuple] = set()
    cur_day = None
    st: dict[str, Any] = {}
    for r in features.itertuples(index=False):
        if r.day != cur_day:
            cur_day = r.day
            st = {"break_dir": None, "pulled": False, "first_done": False}
        direction = _decide(spec.family, p, r, st)
        t = r.timestamp.time()
        if direction is None or not (start <= t < stop):
            continue
        key = (r.timestamp, direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(CandidateSignal(r.timestamp.to_pydatetime(), direction, underlying.upper(), float(r.close),
                                   spec.candidate_id, spec.family, _snapshot(r)))
    return out


def _orb_cross(r) -> str | None:
    if not _ok(r.orb_high, r.orb_low, r.prev_close_day):
        return None
    if r.close > r.orb_high and r.prev_close_day <= r.orb_high:
        return "BUY"
    if r.close < r.orb_low and r.prev_close_day >= r.orb_low:
        return "SELL"
    return None


def _ema_trend(r) -> str | None:
    if not _ok(r.ema20, r.ema50):
        return None
    if r.ema20 > r.ema50 and r.close > r.ema20:
        return "BUY"
    if r.ema20 < r.ema50 and r.close < r.ema20:
        return "SELL"
    return None


def _vwap_side(r) -> str | None:
    if not _ok(r.fut_close, r.fut_vwap):
        return None
    return "BUY" if r.fut_close > r.fut_vwap else "SELL" if r.fut_close < r.fut_vwap else None


def _decide(family: str, p: dict[str, Any], r, st: dict[str, Any]) -> str | None:
    if family == "ORB_EMA":
        d = _orb_cross(r)
        return d if d and _ema_trend(r) == d else None
    if family == "ORB_VWAP":
        d = _orb_cross(r)
        return d if d and _vwap_side(r) == d else None
    if family == "ORB_RSI":
        d = _orb_cross(r)
        if not d or not _ok(r.rsi14):
            return None
        return d if (r.rsi14 >= p["rsi"] if d == "BUY" else r.rsi14 <= 100 - p["rsi"]) else None
    if family == "EMA_VWAP_PULLBACK":
        d = _ema_trend(r)
        if d is None or _vwap_side(r) != d:
            return None
        if p["pullback_to"] == "EMA20":
            touched = r.low <= r.ema20 if d == "BUY" else r.high >= r.ema20
            confirm = r.close > r.open if d == "BUY" else r.close < r.open
        else:
            if not _ok(r.fut_low, r.fut_high, r.fut_open):
                return None
            touched = r.fut_low <= r.fut_vwap if d == "BUY" else r.fut_high >= r.fut_vwap
            confirm = r.fut_close > r.fut_open if d == "BUY" else r.fut_close < r.fut_open
        return d if touched and confirm else None
    if family == "BREAKOUT_ATR":
        if not _ok(r.prior_high, r.prior_low, r.atr14_prev, r.tr) or r.tr < p["range_atr"] * r.atr14_prev:
            return None
        if r.close > r.prior_high and r.close > r.open:
            return "BUY"
        if r.close < r.prior_low and r.close < r.open:
            return "SELL"
        return None
    if family == "TREND_RSI_VWAP":
        if not _ok(r.ema20, r.ema50, r.rsi14, r.rsi_prev):
            return None
        thr = p["rsi"]
        if r.ema20 > r.ema50 and r.rsi_prev < thr <= r.rsi14 and _vwap_side(r) == "BUY":
            return "BUY"
        if r.ema20 < r.ema50 and r.rsi_prev > 100 - thr >= r.rsi14 and _vwap_side(r) == "SELL":
            return "SELL"
        return None
    if family == "PULLBACK_AFTER_BREAKOUT":
        return _pullback_after_breakout(p, r, st)
    if family == "FIRST_BREAKOUT_ONLY":
        if st["first_done"] or not _ok(r.orb_high, r.orb_low, r.atr14, r.ema20, r.ema50):
            return None
        d = "BUY" if r.close > r.orb_high else "SELL" if r.close < r.orb_low else None
        if d is None:
            return None
        st["first_done"] = True
        dist = r.close - r.orb_high if d == "BUY" else r.orb_low - r.close
        aligned = r.ema20 > r.ema50 if d == "BUY" else r.ema20 < r.ema50
        return d if dist >= p["buffer_atr"] * r.atr14 and aligned else None
    raise ValueError(f"Unknown family {family}")


def _pullback_after_breakout(p: dict[str, Any], r, st: dict[str, Any]) -> str | None:
    if not _ok(r.orb_high, r.orb_low, r.atr14):
        return None
    tol = p["retest_atr"] * r.atr14
    d = st["break_dir"]
    if d is None:
        cross = _orb_cross(r)
        if cross:
            st.update(break_dir=cross, pulled=False)
        return None
    if d == "BUY" and r.close < r.orb_low:
        st.update(break_dir="SELL", pulled=False)
        return None
    if d == "SELL" and r.close > r.orb_high:
        st.update(break_dir="BUY", pulled=False)
        return None
    if d == "BUY":
        st["pulled"] = st["pulled"] or r.low <= r.orb_high + tol
        if st["pulled"] and r.close > r.orb_high and r.close > r.open:
            st["pulled"] = False
            return "BUY"
    else:
        st["pulled"] = st["pulled"] or r.high >= r.orb_low - tol
        if st["pulled"] and r.close < r.orb_low and r.close < r.open:
            st["pulled"] = False
            return "SELL"
    return None


def dedupe_signals(signals: Iterable[CandidateSignal]) -> list[CandidateSignal]:
    """One signal per (candidate, timestamp, direction), chronological."""
    seen, out = set(), []
    for s in sorted(signals, key=lambda x: (x.timestamp, x.direction)):
        key = (s.candidate_id, pd.Timestamp(s.timestamp), s.direction)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def regime_label(conditions: dict[str, Any], pit_tags: Iterable[str] = ()) -> str:
    e20, e50, a = conditions.get("ema20"), conditions.get("ema50"), conditions.get("atr14")
    if e20 is None or e50 is None or not a:
        trend = "UNKNOWN"
    else:
        spread = (e20 - e50) / a
        trend = "TREND_UP" if spread >= TREND_SPREAD_ATR else "TREND_DOWN" if spread <= -TREND_SPREAD_ATR else "RANGE"
    return trend + ("+LOW_VOL" if "low_volatility" in set(pit_tags) else "")


# ---------------------------------------------------------------- simulation
def _outcome(sig_ts: pd.Timestamp, sig: CandidateSignal, contract, levels: dict, premium: pd.DataFrame,
             cutoff: time, lot_size: int) -> dict[str, Any] | None:
    entry_ts = _ist(levels["entry_time"])
    record = SignalRecord.create(timestamp=sig_ts, underlying=sig.underlying, direction=sig.direction,
                                 active_from=entry_ts, strategy=sig.family, instrument_key=contract.instrument_key,
                                 entry_price=levels["entry"], stop_loss=levels["stop_loss"], target=levels["target"],
                                 quantity=lot_size)
    future = premium[(premium["timestamp"] >= entry_ts) & (premium["timestamp"].dt.date == sig_ts.date())]
    for bar in future.to_dict("records"):
        candle = {"timestamp": _ist(bar["timestamp"]).to_pydatetime(),
                  **{k: float(bar[k]) for k in ("open", "high", "low", "close")}}
        out = evaluate_candle(record, candle, cutoff)
        if out is not None:
            return out
    return None


def simulate(signals: Sequence[CandidateSignal], *, resolve: Callable, premium_for: Callable, params: LevelParams,
             square_off: str, max_trades_per_day: int, lot_size: int = 65,
             regime_for: Callable[[CandidateSignal], str] | None = None) -> list[dict[str, Any]]:
    """One position at a time. Every signal yields a row; nothing is dropped.

    Same-candle SL+target is counted as a stop-loss LOSS (conservative). An EOD
    exit exactly at entry is a SCRATCH (not a win). Signals without real
    premium evidence are UNRESOLVED and never counted as trades.
    """
    cutoff = time.fromisoformat(square_off)
    rows: list[dict[str, Any]] = []
    busy_until: pd.Timestamp | None = None
    day, trades_today = None, 0
    for sig in dedupe_signals(signals):
        sig_ts = _ist(sig.timestamp)
        if sig_ts.date() != day:
            day, trades_today = sig_ts.date(), 0
        row: dict[str, Any] = {
            "candidate_id": sig.candidate_id, "family": sig.family, "session_date": sig_ts.date().isoformat(),
            "signal_time": sig_ts.isoformat(), "direction": sig.direction, "spot": sig.spot,
            "time_bucket": time_bucket(sig_ts), "regime": regime_for(sig) if regime_for else None,
            "status": None, "outcome": None, "reason": None, "exit_reason": None, "contract": None,
            "instrument_key": None, "expiry": None, "dte": None, "entry_time": None, "entry_price": None,
            "stop_loss": None, "target": None, "exit_time": None, "exit_price": None, "pnl": None,
            "r_multiple": None, **{f"f_{k}": v for k, v in sig.conditions.items()},
        }
        rows.append(row)
        if busy_until is not None and sig_ts <= busy_until:
            row["status"] = SKIPPED_OPEN
            continue
        if trades_today >= max_trades_per_day:
            row["status"] = SKIPPED_CAP
            continue
        row["status"] = UNRESOLVED
        try:
            contract = resolve(sig)
        except OptionResolutionError as exc:
            row["reason"] = f"CONTRACT_UNRESOLVED:{exc.code}"
            continue
        row.update(contract=getattr(contract, "trading_symbol", None), instrument_key=contract.instrument_key,
                   expiry=contract.expiry.isoformat(), dte=(contract.expiry - sig_ts.date()).days)
        premium = premium_for(contract)
        if premium is None or isinstance(premium, str) or premium.empty:
            row["reason"] = premium if isinstance(premium, str) else "NO_PREMIUM_DATA"
            continue
        levels = option_levels(sig, contract, premium, params)
        if levels["status"] != "RESOLVED":
            row["reason"] = levels["reason"]
            continue
        entry_ts = _ist(levels["entry_time"])
        row.update(entry_time=entry_ts.isoformat(), entry_price=levels["entry"], stop_loss=levels["stop_loss"],
                   target=levels["target"])
        out = _outcome(sig_ts, sig, contract, levels, premium, cutoff, lot_size)
        trades_today += 1
        if out is None:
            row["reason"] = "NO_EXIT_EVIDENCE"
            busy_until = pd.Timestamp(datetime.combine(sig_ts.date(), time(23, 59)), tz=IST)
            continue
        exit_ts = _ist(out["exit_time"])
        entry, stop = levels["entry"], levels["stop_loss"]
        if out["status"] == AMBIGUOUS:
            pnl, r_mult, outcome, exit_price = (stop - entry) * lot_size, (stop - entry) / (entry - stop), LOSS, stop
            exit_reason = "SL_AND_TARGET_SAME_CANDLE_COUNTED_AS_LOSS"
        elif out["status"] == INVALID:
            pnl, r_mult, outcome, exit_price, exit_reason = 0.0, 0.0, SCRATCH, out["exit_price"], out["reason"]
        else:
            pnl, r_mult, exit_price, exit_reason = out["pnl"], out["r_multiple"], out["exit_price"], out["reason"]
            outcome = WIN if out["status"] == WIN and pnl > 0 else LOSS if pnl < 0 else SCRATCH
        row.update(status=TRADED, outcome=outcome, exit_reason=exit_reason, exit_time=exit_ts.isoformat(),
                   exit_price=exit_price, pnl=round(float(pnl), 2), r_multiple=round(float(r_mult), 4))
        busy_until = exit_ts
    return rows


# -------------------------------------------------------------------- metrics
def _pf(gp: float, gl: float) -> float:
    return gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)


def _small(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for t in trades if t["outcome"] == WIN)
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    return {"trades": len(trades), "wins": wins, "win_rate": wins / len(trades) if trades else None,
            "profit_factor": _pf(gp, gl) if trades else None, "net_pnl": round(sum(t["pnl"] for t in trades), 2)}


def compute_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Metrics over every signal row; win rate = wins / resolved trades (scratches count as non-wins)."""
    trades = sorted((r for r in rows if r["status"] == TRADED), key=lambda r: r["signal_time"])
    wins = [t for t in trades if t["outcome"] == WIN]
    losses = [t for t in trades if t["outcome"] == LOSS]
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    equity = peak = max_dd = 0.0
    streak = max_streak = 0
    for t in trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        streak = streak + 1 if t["outcome"] != WIN else 0
        max_streak = max(max_streak, streak)
    unresolved = [r for r in rows if r["status"] == UNRESOLVED]
    reasons: dict[str, int] = {}
    for r in unresolved:
        reasons[str(r["reason"])] = reasons.get(str(r["reason"]), 0) + 1
    n = len(trades)
    ci = wilson(len(wins), n)
    by = lambda key, labels=None: {lab: _small([t for t in trades if t[key] == lab])
                                   for lab in (labels or sorted({str(t[key]) for t in trades}))}
    return {
        "total_signals": len(rows),
        "resolved_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "scratches": n - len(wins) - len(losses),
        "win_rate": len(wins) / n if n else None,
        "win_rate_ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        "profit_factor": _pf(gp, gl) if n else None,
        "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2),
        "net_pnl": round(sum(t["pnl"] for t in trades), 2),
        "max_drawdown": round(max_dd, 2),
        "average_r": round(sum(t["r_multiple"] for t in trades) / n, 4) if n else None,
        "max_consecutive_losses": max_streak,
        "trading_days": len({t["session_date"] for t in trades}),
        "skipped_position_open": sum(1 for r in rows if r["status"] == SKIPPED_OPEN),
        "skipped_daily_cap": sum(1 for r in rows if r["status"] == SKIPPED_CAP),
        "unresolved": len(unresolved),
        "unresolved_reasons": reasons,
        "dte_distribution": {str(k): v for k, v in sorted(_count(t["dte"] for t in trades).items())},
        "by_direction": by("direction", ["BUY", "SELL"]),
        "by_time_of_day": {lab: _small([t for t in trades if t["time_bucket"] == lab])
                           for _, _, lab in TIME_BUCKETS if any(t["time_bucket"] == lab for t in trades)},
        "by_regime": by("regime"),
    }


def _count(values: Iterable) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


# ---------------------------------------------------------------- validation
def chronological_split(days: Iterable[date], fractions: tuple[float, float, float] = (0.55, 0.225, 0.225)
                        ) -> dict[str, list[date]]:
    """Ordered, disjoint blocks of whole sessions: TRAIN < VALIDATION < OOS."""
    ordered = sorted(set(days))
    n = len(ordered)
    if n < 3:
        raise ValueError("Need at least 3 sessions for a TRAIN/VALIDATION/OOS split")
    n_oos = max(1, round(n * fractions[2]))
    n_val = max(1, round(n * fractions[1]))
    n_train = n - n_val - n_oos
    if n_train < 1:
        raise ValueError("Split leaves no TRAIN sessions")
    return {"TRAIN": ordered[:n_train], "VALIDATION": ordered[n_train:n_train + n_val],
            "OOS": ordered[n_train + n_val:]}


def split_of(day: date | str, split_days: dict[str, list[date]]) -> str | None:
    d = date.fromisoformat(day) if isinstance(day, str) else day
    for name, days in split_days.items():
        if d in days:
            return name
    return None


@dataclass(frozen=True)
class AcceptanceCriteria:
    train_win_rate: float = 0.60
    validation_win_rate: float = 0.55
    oos_win_rate: float = 0.55
    oos_profit_factor: float = 1.0
    min_train_trades: int = 20
    min_validation_trades: int = 8
    min_oos_trades: int = 8
    promising_win_rate: float = 0.50


def classify(train: dict, validation: dict, oos: dict, lookahead_ok: bool,
             c: AcceptanceCriteria = AcceptanceCriteria()) -> tuple[str, list[str]]:
    """Status from measured metrics; the current-day session is deliberately not an input."""
    wr = lambda m: m.get("win_rate") or 0.0
    if not lookahead_ok:
        return "REJECTED", ["look-ahead detected"]
    short = [f"{name} trades {m['resolved_trades']} < {need}" for name, m, need in (
        ("TRAIN", train, c.min_train_trades), ("VALIDATION", validation, c.min_validation_trades),
        ("OOS", oos, c.min_oos_trades)) if m["resolved_trades"] < need]
    if short:
        return "INSUFFICIENT_DATA", short
    checks = [
        (wr(train) >= c.train_win_rate, f"TRAIN WR {wr(train):.1%} < {c.train_win_rate:.0%}"),
        (wr(validation) >= c.validation_win_rate, f"VALIDATION WR {wr(validation):.1%} < {c.validation_win_rate:.0%}"),
        (wr(oos) >= c.oos_win_rate, f"OOS WR {wr(oos):.1%} < {c.oos_win_rate:.0%}"),
        ((oos.get("profit_factor") or 0.0) > c.oos_profit_factor,
         f"OOS PF {_fmt_pf(oos.get('profit_factor'))} <= {c.oos_profit_factor:g}"),
        (oos["net_pnl"] > 0, f"OOS P&L {oos['net_pnl']:,.0f} <= 0"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if not failed:
        return "ROBUST", []
    if wr(train) >= c.train_win_rate:
        return "OVERFIT", failed
    if (min(wr(train), wr(validation), wr(oos)) >= c.promising_win_rate
            and (oos.get("profit_factor") or 0.0) > c.oos_profit_factor and oos["net_pnl"] > 0):
        return "PROMISING", failed
    return "REJECTED", failed


def select_by_train(train_metrics: dict[str, dict], min_trades: int) -> str:
    """Pick one variant per family from TRAIN metrics only (WR, then PF, then trade count)."""
    def key(item):
        m = item[1]
        pf = m.get("profit_factor") or 0.0
        return (m["resolved_trades"] >= min_trades, m.get("win_rate") or 0.0, min(pf, 1e6), m["resolved_trades"])
    return max(sorted(train_metrics.items()), key=key)[0]


def lookahead_audit(spot: pd.DataFrame, futures: pd.DataFrame | None, spec: CandidateSpec,
                    cut_points: Sequence[pd.Timestamp], **signal_kw) -> dict[str, Any]:
    """Recompute signals on data truncated at each cut point; they must equal the full-run signals up to it."""
    full = generate_signals(build_features(spot, futures, orb_minutes=spec.orb_minutes), spec, **signal_kw)
    full_keys = [(pd.Timestamp(s.timestamp), s.direction) for s in full]
    s_ts, f_ts = _ist_series(spot["timestamp"]), None if futures is None else _ist_series(futures["timestamp"])
    mismatches = []
    for cut in cut_points:
        cut = _ist(cut)
        fut = None if futures is None else futures[f_ts <= cut]
        part = generate_signals(build_features(spot[s_ts <= cut], fut, orb_minutes=spec.orb_minutes), spec,
                                **signal_kw)
        got = [(pd.Timestamp(s.timestamp), s.direction) for s in part]
        want = [k for k in full_keys if k[0] <= cut]
        if got != want:
            mismatches.append(cut.isoformat())
    return {"cut_points": len(cut_points), "mismatches": mismatches, "passed": not mismatches}


def entry_after_signal(rows: Iterable[dict[str, Any]]) -> bool:
    return all(_ist(r["entry_time"]) > _ist(r["signal_time"]) for r in rows if r.get("entry_time"))


def audit_cut_points(spot: pd.DataFrame, days: Iterable[date], times: Sequence[str] = ("10:30", "13:00")
                     ) -> list[pd.Timestamp]:
    ts = set(_ist_series(spot["timestamp"]))
    out = []
    for d in sorted(set(days)):
        for t in times:
            cut = pd.Timestamp(datetime.combine(d, time.fromisoformat(t)), tz=IST)
            if cut in ts:
                out.append(cut)
    return out


# ------------------------------------------------------------------- the lab
@dataclass
class Dataset:
    """Real candles plus resolver/premium accessors for one evaluation sample."""
    spot: pd.DataFrame
    futures: pd.DataFrame | None
    resolve: Callable
    premium_for: Callable
    days: list[date]
    regime_classifier: Any = None


def _run_candidate(spec: CandidateSpec, data: Dataset, params: LevelParams, square_off: str, stop_new_entries: str,
                   lot_size: int, features_cache: dict) -> list[dict[str, Any]]:
    key = (id(data), spec.orb_minutes)
    if key not in features_cache:
        features_cache[key] = build_features(data.spot, data.futures, orb_minutes=spec.orb_minutes)
    signals = [s for s in generate_signals(features_cache[key], spec, stop_new_entries=stop_new_entries)
               if pd.Timestamp(s.timestamp).date() in set(data.days)]

    def regime_for(sig):
        tags = data.regime_classifier.regimes_at(sig.timestamp) if data.regime_classifier else ()
        return regime_label(sig.conditions, tags)

    return simulate(signals, resolve=data.resolve, premium_for=data.premium_for,
                    params=replace(params, target_r=spec.target_r), square_off=square_off,
                    max_trades_per_day=spec.max_trades_per_day, lot_size=lot_size, regime_for=regime_for)


def run_lab(historical: Dataset, *, params: LevelParams, square_off: str, stop_new_entries: str,
            current: Dataset | None = None, current_day: date | None = None,
            specs: Sequence[CandidateSpec] | None = None, criteria: AcceptanceCriteria = AcceptanceCriteria(),
            fractions: tuple[float, float, float] = (0.55, 0.225, 0.225), lot_size: int = 65,
            audit_times: Sequence[str] = ("10:30", "13:00"), current_audit_times: Sequence[str] | None = None,
            historical_1dte_available: bool = False) -> dict[str, Any]:
    specs = list(specs or candidate_grid())
    split_days = chronological_split(historical.days, fractions)
    cache: dict = {}
    hist_cuts = audit_cut_points(historical.spot, historical.days, audit_times)
    cur_times = current_audit_times or CURRENT_AUDIT_TIMES
    cur_cuts = audit_cut_points(current.spot, current.days, cur_times) if current else []
    results, all_rows = [], []
    for spec in specs:
        rows = _run_candidate(spec, historical, params, square_off, stop_new_entries, lot_size, cache)
        for r in rows:
            r["split"] = split_of(r["session_date"], split_days)
        per_split = {s: compute_metrics([r for r in rows if r["split"] == s]) for s in SPLITS}
        audit = lookahead_audit(historical.spot, historical.futures, spec, hist_cuts,
                                stop_new_entries=stop_new_entries)
        cur_rows, cur_metrics, cur_audit = [], None, None
        if current is not None:
            cur_rows = _run_candidate(spec, current, params, square_off, stop_new_entries, lot_size, cache)
            for r in cur_rows:
                r["split"] = "CURRENT_DAY"
            cur_metrics = compute_metrics(cur_rows)
            cur_audit = lookahead_audit(current.spot, current.futures, spec, cur_cuts,
                                        stop_new_entries=stop_new_entries)
        entries_ok = entry_after_signal(rows) and entry_after_signal(cur_rows)
        lookahead_ok = audit["passed"] and entries_ok and (cur_audit is None or cur_audit["passed"])
        status, reasons = classify(per_split["TRAIN"], per_split["VALIDATION"], per_split["OOS"], lookahead_ok,
                                   criteria)
        results.append({"candidate_id": spec.candidate_id, "family": spec.family,
                        "family_description": FAMILIES[spec.family], "params": spec.p, "target_r": spec.target_r,
                        "max_trades_per_day": spec.max_trades_per_day, "metrics": per_split,
                        "full_history": compute_metrics(rows), "current_day": cur_metrics,
                        "lookahead": {"historical": audit, "current_day": cur_audit,
                                      "entry_after_signal": entries_ok, "passed": lookahead_ok},
                        "status": status, "status_reasons": reasons, "selected_for_family": False})
        all_rows += rows + cur_rows

    for family in FAMILIES:
        fam = [r for r in results if r["family"] == family]
        if not fam:
            continue
        chosen = select_by_train({r["candidate_id"]: r["metrics"]["TRAIN"] for r in fam}, criteria.min_train_trades)
        next(r for r in fam if r["candidate_id"] == chosen)["selected_for_family"] = True

    selected = sorted((r for r in results if r["selected_for_family"]), key=rank_key)
    hist_dtes = sorted({r["dte"] for r in all_rows if r["status"] == TRADED and r["split"] in SPLITS})
    return {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "mode": "PAPER research only - no orders, S2 unchanged, live settings untouched",
        "split_days": {k: [d.isoformat() for d in v] for k, v in split_days.items()},
        "split_fractions": list(fractions),
        "current_day": current_day.isoformat() if current_day else None,
        "criteria": asdict(criteria),
        "exit_model": {"entry": params.entry_timing, "stop": f"{params.stop_atr_multiplier}x premium ATR"
                       f"({params.atr_period}) from candles <= signal bar", "target_r": "per candidate (1.0 or 2.0)",
                       "square_off": square_off, "stop_new_entries": stop_new_entries, "lot_size": lot_size,
                       "costs": "excluded", "slippage_bps": params.slippage_bps,
                       "position_rule": "one open position per candidate; later signals recorded as skipped"},
        "historical_option_dte": hist_dtes,
        "data_flags": [] if historical_1dte_available else [HISTORICAL_1DTE_UNAVAILABLE],
        "grid_size": len(specs),
        "candidates": results,
        "ranked": [table_row(r) for r in selected],
        "best_candidate": selected[0]["candidate_id"] if selected else None,
        "trades": all_rows,
    }


def rank_key(r: dict) -> tuple:
    """Status first, then VALIDATION then TRAIN win rate; OOS is never used to order candidates."""
    v, t = r["metrics"]["VALIDATION"], r["metrics"]["TRAIN"]
    return (STATUSES.index(r["status"]), -(v.get("win_rate") or 0.0), -(t.get("win_rate") or 0.0),
            -t["resolved_trades"])


def table_row(r: dict) -> dict[str, Any]:
    m, cur = r["metrics"], r["current_day"] or {}
    return {"candidate": r["candidate_id"], "train_wr": m["TRAIN"]["win_rate"],
            "train_trades": m["TRAIN"]["resolved_trades"], "validation_wr": m["VALIDATION"]["win_rate"],
            "validation_trades": m["VALIDATION"]["resolved_trades"], "oos_wr": m["OOS"]["win_rate"],
            "oos_trades": m["OOS"]["resolved_trades"], "oos_pf": m["OOS"]["profit_factor"],
            "oos_pnl": m["OOS"]["net_pnl"], "oos_max_dd": m["OOS"]["max_drawdown"],
            "current_day_wr": cur.get("win_rate"), "current_day_trades": cur.get("resolved_trades"),
            "status": r["status"]}


# -------------------------------------------------------------------- output
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


SUMMARY_FIELDS = ("candidate_id", "family", "selected_for_family", "status", "target_r", "params",
                  *(f"{s.lower()}_{k}" for s in SPLITS for k in (
                      "signals", "trades", "wins", "losses", "win_rate", "profit_factor", "net_pnl", "max_drawdown",
                      "average_r", "max_consecutive_losses")),
                  "current_day_signals", "current_day_trades", "current_day_win_rate", "current_day_profit_factor",
                  "current_day_net_pnl", "lookahead_passed", "status_reasons")


def summary_rows(report: dict) -> list[dict[str, Any]]:
    out = []
    for r in report["candidates"]:
        row = {"candidate_id": r["candidate_id"], "family": r["family"], "selected_for_family": r["selected_for_family"],
               "status": r["status"], "target_r": r["target_r"], "params": json.dumps(r["params"])}
        for s in SPLITS:
            m = r["metrics"][s]
            for k, src in (("signals", "total_signals"), ("trades", "resolved_trades"), ("wins", "wins"),
                           ("losses", "losses"), ("win_rate", "win_rate"), ("profit_factor", "profit_factor"),
                           ("net_pnl", "net_pnl"), ("max_drawdown", "max_drawdown"), ("average_r", "average_r"),
                           ("max_consecutive_losses", "max_consecutive_losses")):
                row[f"{s.lower()}_{k}"] = _jsonable(m[src])
        cur = r["current_day"] or {}
        row.update(current_day_signals=cur.get("total_signals"), current_day_trades=cur.get("resolved_trades"),
                   current_day_win_rate=cur.get("win_rate"), current_day_profit_factor=_jsonable(cur.get("profit_factor")),
                   current_day_net_pnl=cur.get("net_pnl"), lookahead_passed=r["lookahead"]["passed"],
                   status_reasons="; ".join(r["status_reasons"]))
        out.append(row)
    return out


def write_outputs(report: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "candidate_summary.json", "csv": out_dir / "candidate_summary.csv",
             "md": out_dir / "candidate_summary.md", "trades": out_dir / "candidate_trades.csv"}
    body = {k: v for k, v in report.items() if k != "trades"}
    paths["json"].write_text(json.dumps(_jsonable(body), indent=2), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(summary_rows(report))
    trades = report["trades"]
    cols: list[str] = []
    for r in trades:
        cols += [k for k in r if k not in cols]
    with paths["trades"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols or ["candidate_id"])
        writer.writeheader()
        for r in trades:
            writer.writerow({k: "" if r.get(k) is None else _jsonable(r.get(k)) for k in cols})
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    return paths


def _table(rows: list[dict]) -> list[str]:
    md = ["| Candidate | Train WR (n) | Validation WR (n) | OOS WR (n) | OOS PF | OOS P&L | OOS Max DD | "
          "Current Day WR (n) | Status |", "|---|---|---|---|---|---:|---:|---|---|"]
    for t in rows:
        md.append(f"| {t['candidate']} | {_pct(t['train_wr'])} ({t['train_trades']}) | {_pct(t['validation_wr'])} "
                  f"({t['validation_trades']}) | {_pct(t['oos_wr'])} ({t['oos_trades']}) | {_fmt_pf(t['oos_pf'])} | "
                  f"{t['oos_pnl']:,.0f} | {t['oos_max_dd']:,.0f} | {_pct(t['current_day_wr'])} "
                  f"({t['current_day_trades'] if t['current_day_trades'] is not None else 'n/a'}) | {t['status']} |")
    return md


def render_markdown(report: dict) -> str:
    c = report["criteria"]
    sd = report["split_days"]
    md = ["# NIFTY 5-minute strategy candidate lab", "",
          f"Generated {report['generated_at']}. Mode: {report['mode']}.", ""]
    for flag in report["data_flags"]:
        md += [f"**{flag}**: historical option outcomes use the contracts the existing resolver returns for each "
               f"historical day (DTE {report['historical_option_dte']}). No historical 1-DTE premium candles exist "
               "in this dataset, so historical results are not equivalent to 1-DTE trading and none were substituted.",
               ""]
    md += ["## Method", "",
           f"- Chronological split by whole sessions: TRAIN {sd['TRAIN'][0]}..{sd['TRAIN'][-1]} ({len(sd['TRAIN'])}), "
           f"VALIDATION {sd['VALIDATION'][0]}..{sd['VALIDATION'][-1]} ({len(sd['VALIDATION'])}), "
           f"OOS {sd['OOS'][0]}..{sd['OOS'][-1]} ({len(sd['OOS'])}).",
           f"- Predeclared grid of {report['grid_size']} variants (8 families x 2 rule values x target R 1.0/2.0). "
           "One variant per family is selected on TRAIN metrics only; VALIDATION/OOS/current day never influence "
           "selection. Ranking uses status, then VALIDATION, then TRAIN win rate.",
           f"- Acceptance (ROBUST): TRAIN WR >= {c['train_win_rate']:.0%}, VALIDATION WR >= "
           f"{c['validation_win_rate']:.0%}, OOS WR >= {c['oos_win_rate']:.0%}, OOS PF > {c['oos_profit_factor']:g}, "
           f"OOS P&L > 0, trades >= {c['min_train_trades']}/{c['min_validation_trades']}/{c['min_oos_trades']} "
           "(TRAIN/VALIDATION/OOS), look-ahead audit passed.",
           "- Signals use completed 5-minute NIFTY bars only. VWAP is session VWAP of the front-month NIFTY future "
           "(real futures volume; index volume is zero). ORB levels exist only after the opening range completes.",
           "- Outcomes: real option premium candles, existing resolver (nearest expiry, ATM, BUY->CE, SELL->PE), "
           f"{report['exit_model']['entry']} entry, {report['exit_model']['stop']}, first touch of SL/target, EOD "
           f"square-off {report['exit_model']['square_off']}; same-candle SL+target counted as a loss; 1 lot "
           f"({report['exit_model']['lot_size']}); costs excluded. One open position per candidate; overlapping "
           "signals are kept and recorded as skipped. Unresolved signals (missing real premium) are reported, not "
           "counted.", "",
           "## Ranked family candidates (TRAIN-selected variant per family)", ""]
    md += _table(report["ranked"])
    md += ["", "Max DD is the OOS peak-to-trough on cumulative 1-lot P&L. Current-day results are shown for "
           "information only and do not affect status.", "", "## All variants", ""]
    rows = [table_row(r) for r in sorted(report["candidates"], key=rank_key)]
    md += _table(rows)
    md += ["", "## Candidate details", ""]
    for r in sorted(report["candidates"], key=rank_key):
        if not r["selected_for_family"]:
            continue
        md += [f"### {r['candidate_id']} - {r['status']}", "", f"{r['family_description']}.", ""]
        if r["status_reasons"]:
            md += ["Failed: " + "; ".join(r["status_reasons"]), ""]
        md += ["| Split | Signals | Trades | W/L/S | WR (95% CI) | PF | Net P&L | Max DD | Avg R | Max consec. L | "
               "Unresolved |", "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, m in [*r["metrics"].items(), ("CURRENT_DAY", r["current_day"])]:
            if not m:
                continue
            ci = m["win_rate_ci95"]
            ci_s = f" ({ci[0] * 100:.0f}-{ci[1] * 100:.0f}%)" if ci else ""
            avg_r = "n/a" if m["average_r"] is None else f"{m['average_r']:.2f}"
            md.append(f"| {name} | {m['total_signals']} | {m['resolved_trades']} | {m['wins']}/{m['losses']}/"
                      f"{m['scratches']} | {_pct(m['win_rate'])}{ci_s} | {_fmt_pf(m['profit_factor'])} | "
                      f"{m['net_pnl']:,.0f} | {m['max_drawdown']:,.0f} | {avg_r} | "
                      f"{m['max_consecutive_losses']} | {m['unresolved']} |")
        md.append("")
        full = r["full_history"]
        md.append("BUY/SELL (all historical): " + ", ".join(
            f"{k} {v['wins']}/{v['trades']} ({_pct(v['win_rate'])}, {v['net_pnl']:,.0f})"
            for k, v in full["by_direction"].items()))
        md.append("")
        md.append("Time of day (all historical): " + (", ".join(
            f"{k} {v['wins']}/{v['trades']}" for k, v in full["by_time_of_day"].items()) or "none"))
        md.append("")
        md.append("Regime (all historical): " + (", ".join(
            f"{k} {v['wins']}/{v['trades']}" for k, v in full["by_regime"].items()) or "none"))
        md.append("")
    md += ["## Notes", "", "- No live settings were changed; S2 is unchanged.",
           "- Small samples: Wilson 95% intervals are wide; a single month cannot establish a robust edge.",
           "- Historical option DTE differs from live 1-DTE trading; see data flags."]
    return "\n".join(md) + "\n"
