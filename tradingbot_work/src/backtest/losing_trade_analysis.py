"""Descriptive analysis of resolved S2 trades (winners vs losers). Historical research only.

Every value is computed from stored Upstox candles or the replayed trade
records. Fields that the data cannot support (e.g. index volume, which Upstox
reports as zero) are returned as ``None`` and reported as unavailable.
Hypothetical filters are naive trade removals, not re-simulations, and are
never applied to the strategy.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest.regime_classifier import ABS_GAP_RETURN, build_regime_labels
from src.backtest.regime_filter import PointInTimeRegimeClassifier
from src.backtest.research_strategies import enrich_s2
from src.backtest.robustness import entry_time_bucket
from src.indicators.technical import atr, ema, rsi

IST = ZoneInfo("Asia/Kolkata")
EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD, VOLUME_AVG_PERIOD = 20, 50, 14, 14, 20
INITIAL_MOVE_BARS = 3
INITIAL_MOVE_R = 0.5
FALSE_BREAKOUT_BARS = 3
MIN_PATTERN_TRADES = 5
MIN_FILTER_SUPPORT = 10
MIN_REMAINING_TRADES = 30
UNAVAILABLE = "unavailable"
WIN, LOSS, BREAKEVEN = "WIN", "LOSS", "BREAKEVEN"
_SYMBOL_STRIKE = re.compile(r"\s(\d+(?:\.\d+)?)\s(CE|PE)\b")


def _ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def classify_outcome(pnl: float) -> str:
    if pnl > 0:
        return WIN
    if pnl < 0:
        return LOSS
    return BREAKEVEN


# ----------------------------------------------------------------- enrichment
def enrich_underlying(candles: pd.DataFrame, s2_cfg: dict[str, Any]) -> pd.DataFrame:
    frame = candles.sort_values("timestamp").reset_index(drop=True).copy()
    frame["timestamp"] = frame["timestamp"].map(_ts)
    frame = enrich_s2(frame, s2_cfg)
    frame["ema20"] = ema(frame["close"], EMA_FAST)
    frame["ema50"] = ema(frame["close"], EMA_SLOW)
    frame["rsi14"] = rsi(frame["close"], RSI_PERIOD)
    frame["atr14"] = atr(frame, ATR_PERIOD)
    frame["day"] = frame["timestamp"].dt.date
    has_volume = "volume" in frame and bool((frame["volume"].fillna(0) > 0).any())
    if has_volume:
        typical = (frame["high"] + frame["low"] + frame["close"]) / 3
        pv = (typical * frame["volume"]).groupby(frame["day"]).cumsum()
        vol = frame["volume"].groupby(frame["day"]).cumsum()
        frame["vwap"] = pv / vol.replace(0, np.nan)
        frame["volume_avg20"] = frame["volume"].rolling(VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean()
        frame["volume_value"] = frame["volume"]
    else:
        frame["vwap"] = np.nan
        frame["volume_avg20"] = np.nan
        frame["volume_value"] = np.nan
    frame.attrs["has_volume"] = has_volume
    return frame


def _day_gaps(frame: pd.DataFrame) -> dict[date, float | None]:
    gaps: dict[date, float | None] = {}
    prev_close: float | None = None
    for day, group in frame.groupby("day", sort=True):
        day_open = float(group.iloc[0]["open"])
        gaps[day] = (day_open - prev_close) / prev_close if prev_close else None
        prev_close = float(group.iloc[-1]["close"])
    return gaps


def _orb_state(close: float, hi: float, lo: float) -> str:
    return "above" if close > hi else ("below" if close < lo else "inside")


def _choppiness_before(day_bars: pd.DataFrame, signal_ts: pd.Timestamp, direction: str,
                       range_end: time) -> tuple[int | None, int | None]:
    """Point-in-time: ORB state changes and opposite-side breaks before the signal bar."""
    post = day_bars[(day_bars["timestamp"].dt.time > range_end) & (day_bars["timestamp"] < signal_ts)]
    if day_bars.empty or pd.isna(day_bars.iloc[0]["orb_high"]):
        return None, None
    hi, lo = float(day_bars.iloc[0]["orb_high"]), float(day_bars.iloc[0]["orb_low"])
    states = [_orb_state(float(c), hi, lo) for c in post["close"]]
    changes = sum(1 for a, b in zip(states, states[1:]) if a != b)
    opposite = "below" if direction == "BUY" else "above"
    return changes, sum(1 for s in states if s == opposite)


def _false_breakout_after(day_bars: pd.DataFrame, signal_ts: pd.Timestamp, direction: str) -> bool | None:
    """Hindsight: underlying closed back inside the opening range within N bars after the signal."""
    after = day_bars[day_bars["timestamp"] > signal_ts].head(FALSE_BREAKOUT_BARS)
    if after.empty or pd.isna(day_bars.iloc[0]["orb_high"]):
        return None
    if direction == "BUY":
        return bool((after["close"] <= float(day_bars.iloc[0]["orb_high"])).any())
    return bool((after["close"] >= float(day_bars.iloc[0]["orb_low"])).any())


def _option_features(premium: pd.DataFrame | None, trade: dict[str, Any]) -> dict[str, Any]:
    keys = ("strike", "entry_premium_open", "option_volume", "option_volume_avg20", "option_volume_ratio",
            "option_vwap", "option_vs_vwap", "mfe_r", "mae_r", "initial_favorable_move",
            "sl_vs_target_first", "stop_width_pct")
    out: dict[str, Any] = {k: None for k in keys}
    entry, stop, target = (_num(trade.get(k)) for k in ("entry_price", "stop_loss", "target"))
    if entry and stop is not None:
        out["stop_width_pct"] = (entry - stop) / entry * 100.0
    match = _SYMBOL_STRIKE.search(str(trade.get("trading_symbol", "")))
    if match:
        out["strike"] = float(match.group(1))
    if premium is None or premium.empty:
        return out
    frame = premium.sort_values("timestamp").reset_index(drop=True).copy()
    frame["timestamp"] = frame["timestamp"].map(_ts)
    if "strike" in frame and _num(frame["strike"].iloc[0]) is not None:
        out["strike"] = float(frame["strike"].iloc[0])
    signal_ts, entry_ts, exit_ts = (_ts(trade[k]) for k in ("signal_time", "entry_time", "exit_time"))

    at_signal = frame[frame["timestamp"] <= signal_ts]
    if not at_signal.empty and "volume" in frame:
        vol = frame["volume"].astype(float)
        avg = vol.rolling(VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean()
        idx = at_signal.index[-1]
        out["option_volume"] = _num(vol.loc[idx])
        out["option_volume_avg20"] = _num(avg.loc[idx])
        if out["option_volume"] is not None and out["option_volume_avg20"]:
            out["option_volume_ratio"] = out["option_volume"] / out["option_volume_avg20"]
        session = at_signal[at_signal["timestamp"].dt.date == signal_ts.date()]
        sv = session["volume"].astype(float)
        if sv.sum() > 0:
            typical = (session["high"] + session["low"] + session["close"]) / 3
            out["option_vwap"] = float((typical * sv).sum() / sv.sum())
            out["option_vs_vwap"] = "above" if float(session.iloc[-1]["close"]) > out["option_vwap"] else "below_or_equal"

    held = frame[(frame["timestamp"] >= entry_ts) & (frame["timestamp"] <= exit_ts)]
    if held.empty or not entry or stop is None or target is None:
        return out
    out["entry_premium_open"] = _num(held.iloc[0]["open"])
    risk = entry - stop
    if risk > 0:
        out["mfe_r"] = (float(held["high"].max()) - entry) / risk
        out["mae_r"] = (entry - float(held["low"].min())) / risk
        first = held.head(INITIAL_MOVE_BARS)
        out["initial_favorable_move"] = bool(float(first["high"].max()) >= entry + INITIAL_MOVE_R * risk)
    order = "NEITHER"
    for _, bar in held.iterrows():
        hit_stop, hit_target = float(bar["low"]) <= stop, float(bar["high"]) >= target
        if hit_stop and hit_target:
            order = "SAME_BAR_AMBIGUOUS"
            break
        if hit_stop:
            order = "SL_FIRST"
            break
        if hit_target:
            order = "TARGET_FIRST"
            break
    out["sl_vs_target_first"] = order
    return out


def build_trade_rows(trades: list[dict[str, Any]], candles: pd.DataFrame,
                     premium_history: dict[str, pd.DataFrame], s2_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per resolved trade; unavailable values stay ``None``."""
    frame = enrich_underlying(candles, s2_cfg)
    by_ts = frame.set_index("timestamp")
    gaps = _day_gaps(frame)
    day_labels = build_regime_labels(candles, s2_cfg)
    pit = PointInTimeRegimeClassifier(candles)
    range_end = time(*map(int, str(s2_cfg.get("range_end", "09:30")).split(":")))
    trades_per_day: dict[date, int] = {}
    rows: list[dict[str, Any]] = []
    for trade in sorted(trades, key=lambda t: t["entry_time"]):
        signal_ts, entry_ts, exit_ts = (_ts(trade[k]) for k in ("signal_time", "entry_time", "exit_time"))
        day = signal_ts.date()
        trades_per_day[day] = trades_per_day.get(day, 0) + 1
        bar = by_ts.loc[signal_ts] if signal_ts in by_ts.index else None
        entry_bar = by_ts.loc[entry_ts] if entry_ts in by_ts.index else None
        day_bars = frame[frame["day"] == day]
        direction = trade["direction"]
        cond = trade.get("conditions") or {}
        level = _num(cond.get("above") if direction == "BUY" else cond.get("below"))
        spot = _num(bar["close"]) if bar is not None else None
        ema20 = _num(bar["ema20"]) if bar is not None else None
        ema50 = _num(bar["ema50"]) if bar is not None else None
        atr14 = _num(bar["atr14"]) if bar is not None else None
        vwap = _num(bar["vwap"]) if bar is not None else None
        sign = 1.0 if direction == "BUY" else -1.0
        distance = sign * (spot - level) if spot is not None and level is not None else None
        changes, opposite = _choppiness_before(day_bars, signal_ts, direction, range_end)
        labels = sorted(day_labels.get(day, set()))
        pit_tags = sorted(pit.regimes_at(signal_ts.to_pydatetime()))
        gap = gaps.get(day)
        orb_hi = _num(bar["orb_high"]) if bar is not None else None
        orb_lo = _num(bar["orb_low"]) if bar is not None else None
        pnl = float(trade["pnl"])
        row: dict[str, Any] = {
            "signal_time": signal_ts.isoformat(),
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "entry_date": entry_ts.date().isoformat(),
            "direction": direction,
            "option_contract": trade.get("trading_symbol"),
            "instrument_key": trade.get("instrument_key"),
            "option_type": trade.get("option_type"),
            "quantity": trade.get("quantity"),
            "entry_price": _num(trade.get("entry_price")),
            "exit_price": _num(trade.get("exit_price")),
            "stop_loss": _num(trade.get("stop_loss")),
            "target": _num(trade.get("target")),
            "pnl": pnl,
            "r_multiple": _num(trade.get("r_multiple")),
            "outcome": classify_outcome(pnl),
            "exit_reason": trade.get("exit_reason"),
            "holding_minutes": (exit_ts - entry_ts).total_seconds() / 60.0,
            "entry_time_bucket": entry_time_bucket(trade["entry_time"]) or "outside_configured_buckets",
            "trade_number_in_day": trades_per_day[day],
            "underlying_at_signal": spot,
            "underlying_at_entry": _num(entry_bar["open"]) if entry_bar is not None else None,
            "underlying_vwap": vwap,
            "vwap_relationship": (None if vwap is None or spot is None
                                  else ("above" if spot > vwap else "below_or_equal")),
            "ema20": ema20,
            "ema50": ema50,
            "ema_trend_aligned": (None if ema20 is None or ema50 is None
                                  else bool(sign * (ema20 - ema50) > 0)),
            "ema_spread_atr": (None if ema20 is None or ema50 is None or not atr14
                               else sign * (ema20 - ema50) / atr14),
            "rsi14": _num(bar["rsi14"]) if bar is not None else None,
            "atr14": atr14,
            "atr_pct": atr14 / spot * 100.0 if atr14 is not None and spot else None,
            "underlying_volume": _num(bar["volume_value"]) if bar is not None else None,
            "underlying_volume_avg20": _num(bar["volume_avg20"]) if bar is not None else None,
            "breakout_level": level,
            "distance_from_breakout": distance,
            "distance_from_breakout_pct": distance / spot * 100.0 if distance is not None and spot else None,
            "breakout_strength_atr": distance / atr14 if distance is not None and atr14 else None,
            "orb_width_pct": ((orb_hi - orb_lo) / spot * 100.0
                              if orb_hi is not None and orb_lo is not None and spot else None),
            "orb_state_changes_before": changes,
            "opposite_breaks_before": opposite,
            "gap_pct": gap * 100.0 if gap is not None else None,
            "gap_condition": (None if gap is None else
                              "gap_up" if gap >= ABS_GAP_RETURN else "gap_down" if gap <= -ABS_GAP_RETURN
                              else "flat"),
            "day_regimes": labels,
            "pit_regimes": pit_tags,
            "volatility_condition": ("high_volatility" if "high_volatility" in labels else
                                     "low_volatility" if "low_volatility" in labels else
                                     "normal" if day in day_labels else None),
            "false_breakout_after_entry": _false_breakout_after(day_bars, signal_ts, direction),
        }
        row.update(_option_features(premium_history.get(trade.get("instrument_key")), trade))
        row["sl_hit_before_target"] = (None if row["sl_vs_target_first"] is None
                                       else row["sl_vs_target_first"] == "SL_FIRST")
        rows.append(row)
    return rows


# -------------------------------------------------------------------- metrics
def _max_drawdown(rows: list[dict[str, Any]], initial_equity: float) -> float:
    equity = peak = initial_equity
    worst = 0.0
    for row in sorted(rows, key=lambda r: r["entry_time"]):
        equity += row["pnl"]
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def group_metrics(rows: list[dict[str, Any]], initial_equity: float = 1_000_000.0) -> dict[str, Any]:
    n = len(rows)
    wins = [r for r in rows if r["outcome"] == WIN]
    losses = [r for r in rows if r["outcome"] == LOSS]
    gross_profit = sum(r["pnl"] for r in wins)
    gross_loss = abs(sum(r["pnl"] for r in losses))
    r_values = [r["r_multiple"] for r in rows if r["r_multiple"] is not None]
    pf = (gross_profit / gross_loss) if gross_loss else (float("inf") if gross_profit > 0 else None)
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": n - len(wins) - len(losses),
        "win_rate": len(wins) / n if n else None,
        "loss_rate": len(losses) / n if n else None,
        "net_pnl": sum(r["pnl"] for r in rows),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": pf,
        "average_r": sum(r_values) / len(r_values) if r_values else None,
        "max_drawdown": _max_drawdown(rows, initial_equity),
        "sample": sample_flag(n),
    }


def sample_flag(n: int) -> str:
    if n < MIN_PATTERN_TRADES:
        return "TOO_FEW_TRADES"
    if n < MIN_FILTER_SUPPORT:
        return "SMALL_SAMPLE"
    if n < 20:
        return "LIMITED_SAMPLE"
    return "ADEQUATE"


def _bucket(value: float | None, edges: list[float], labels: list[str]) -> str:
    if value is None:
        return UNAVAILABLE
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


@dataclass(frozen=True)
class Factor:
    factor_id: str
    title: str
    key: Callable[[dict[str, Any]], Any]
    point_in_time: bool
    note: str = ""
    order: tuple[str, ...] = ()


def _flag(value: bool | None, yes: str, no: str) -> str:
    return UNAVAILABLE if value is None else (yes if value else no)


def _tercile_factor(rows: list[dict[str, Any]], field_name: str) -> Callable[[dict[str, Any]], str]:
    values = sorted(r[field_name] for r in rows if r.get(field_name) is not None)
    if len(values) < 3:
        return lambda r: UNAVAILABLE
    lo, hi = float(np.percentile(values, 33.33)), float(np.percentile(values, 66.67))

    def key(r: dict[str, Any]) -> str:
        v = r.get(field_name)
        if v is None:
            return UNAVAILABLE
        return f"low (<{lo:.3f})" if v < lo else (f"mid ({lo:.3f}-{hi:.3f})" if v < hi else f"high (>={hi:.3f})")
    return key


def _binned(field_name: str, edges: list[float], labels: list[str]) -> dict[str, Any]:
    return {"key": lambda r: _bucket(r[field_name], edges, labels), "order": tuple(labels)}


def build_factors(rows: list[dict[str, Any]]) -> list[Factor]:
    terciles = ("low", "mid", "high")
    return [
        Factor("time_of_day", "1. Time-of-day (entry bucket)", lambda r: r["entry_time_bucket"], True),
        Factor("direction", "Direction (BUY CE / SELL PE)", lambda r: r["direction"], True),
        Factor("trade_number_in_day", "Trade number within the day",
               lambda r: str(r["trade_number_in_day"]) if r["trade_number_in_day"] < 3 else "3+", True,
               order=("1", "2", "3+")),
        Factor("rsi_range", "2. RSI14 range", point_in_time=True,
               **_binned("rsi14", [30, 40, 50, 60, 70], ["<30", "30-40", "40-50", "50-60", "60-70", ">=70"])),
        Factor("rsi_alignment", "2b. RSI aligned with direction (BUY>=50 / SELL<=50)",
               lambda r: UNAVAILABLE if r["rsi14"] is None else
               ("aligned" if (r["rsi14"] >= 50) == (r["direction"] == "BUY") else "against"), True),
        Factor("underlying_vwap", "3. Underlying VWAP position", lambda r: r["vwap_relationship"] or UNAVAILABLE,
               True, "NIFTY index candles carry zero volume, so underlying VWAP cannot be computed."),
        Factor("option_vwap", "3b. Option premium vs session option VWAP (at signal bar)",
               lambda r: r["option_vs_vwap"] or UNAVAILABLE, True),
        Factor("ema_alignment", "4. EMA20/EMA50 trend aligned with direction",
               lambda r: _flag(r["ema_trend_aligned"], "aligned", "against"), True),
        Factor("ema_strength", "4b. EMA20-EMA50 spread in ATR (signed by direction)", point_in_time=True,
               **_binned("ema_spread_atr", [-1.5, 0, 1.5, 3],
                         ["< -1.5 (strong against)", "-1.5..0 (weak against)", "0..1.5 (weak aligned)",
                          "1.5..3 (aligned)", ">=3 (strong aligned)"])),
        Factor("underlying_volume", "5. Underlying volume vs 20-candle average",
               lambda r: UNAVAILABLE if r["underlying_volume"] is None or not r["underlying_volume_avg20"]
               else ("above_avg" if r["underlying_volume"] >= r["underlying_volume_avg20"] else "below_avg"),
               True, "NIFTY index candles carry zero volume."),
        Factor("option_volume", "5b. Option volume / 20-candle average (at signal bar)", point_in_time=True,
               **_binned("option_volume_ratio", [0.75, 1.0, 1.5], ["<0.75", "0.75-1.0", "1.0-1.5", ">=1.5"])),
        Factor("atr_pct", "6. ATR14 as % of spot (sample terciles)", _tercile_factor(rows, "atr_pct"), True,
               "Tercile thresholds use the full sample (descriptive).", order=terciles),
        Factor("volatility_condition", "6b. Day volatility label", lambda r: r["volatility_condition"] or UNAVAILABLE,
               False, "Uses the full session range (hindsight)."),
        Factor("breakout_strength", "7. Breakout strength (distance / ATR14)", point_in_time=True,
               **_binned("breakout_strength_atr", [0.25, 0.5, 1.0, 2.0],
                         ["<0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", ">=2.0"])),
        Factor("breakout_distance", "8. Distance from breakout level (% of spot)", point_in_time=True,
               **_binned("distance_from_breakout_pct", [0.05, 0.15, 0.30, 0.60],
                         ["<0.05%", "0.05-0.15%", "0.15-0.30%", "0.30-0.60%", ">=0.60%"])),
        Factor("orb_width", "8b. Opening-range width (% of spot, sample terciles)",
               _tercile_factor(rows, "orb_width_pct"), True, "Tercile thresholds use the full sample.",
               order=terciles),
        Factor("day_regime", "9. Market regime (day label, multi-label)", lambda r: r["day_regimes"] or ["none"],
               False, "Day labels use the full session (hindsight); one trade can carry several labels."),
        Factor("pit_regime", "9b. Point-in-time regime at signal (avoid-regime detector)",
               lambda r: r["pit_regimes"] or ["none"], True),
        Factor("gap", "10. Gap condition (open vs previous close)", lambda r: r["gap_condition"] or UNAVAILABLE, True),
        Factor("gap_alignment", "10b. Gap direction vs trade direction",
               lambda r: UNAVAILABLE if r["gap_condition"] is None else
               "no_gap" if r["gap_condition"] == "flat" else
               ("with_gap" if (r["gap_condition"] == "gap_up") == (r["direction"] == "BUY") else "against_gap"),
               True),
        Factor("chop_before", "11. Choppiness before signal (ORB state changes)", point_in_time=True,
               **_binned("orb_state_changes_before", [1, 3, 6], ["0", "1-2", "3-5", "6+"])),
        Factor("opposite_breaks", "11b. Opposite-side ORB closes before signal", point_in_time=True,
               **_binned("opposite_breaks_before", [1], ["0", "1+"])),
        Factor("false_breakout", "12. False breakout (close back inside ORB within 3 bars)",
               lambda r: _flag(r["false_breakout_after_entry"], "false_breakout", "held"), False,
               "Outcome descriptor (hindsight)."),
        Factor("entry_premium", "13. Entry premium level (Rs)", point_in_time=True,
               **_binned("entry_price", [100, 150, 200, 300], ["<100", "100-150", "150-200", "200-300", ">=300"])),
        Factor("stop_width", "13b. Stop width (% of entry premium)", point_in_time=True,
               **_binned("stop_width_pct", [5, 10, 15, 20], ["<5%", "5-10%", "10-15%", "15-20%", ">=20%"])),
        Factor("initial_move", "13c. Initial favorable move (>=0.5R within 3 bars)",
               lambda r: _flag(r["initial_favorable_move"], "yes", "no"), False, "Outcome descriptor (hindsight)."),
        Factor("sl_first", "13d. Stop vs target touched first (initial levels)",
               lambda r: r["sl_vs_target_first"] or UNAVAILABLE, False, "Outcome descriptor (hindsight)."),
        Factor("holding", "14. Holding duration (minutes)", point_in_time=False,
               note="Outcome descriptor (hindsight).",
               **_binned("holding_minutes", [15, 60, 120, 240], ["<15", "15-60", "60-120", "120-240", ">=240"])),
        Factor("exit_reason", "Exit reason", lambda r: r["exit_reason"] or UNAVAILABLE, False, "Outcome descriptor."),
    ]


def factor_table(rows: list[dict[str, Any]], factor: Factor, initial_equity: float) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        keys = factor.key(row)
        for key in (keys if isinstance(keys, list) else [keys]):
            groups.setdefault(str(key), []).append(row)
    def rank(key: str) -> tuple[int, int, str]:
        if key == UNAVAILABLE:
            return (2, 0, key)
        pos = next((i for i, p in enumerate(factor.order) if key == p or key.startswith(p + " ")), None)
        return (0, pos, key) if pos is not None else (1, 0, key)
    return [{"bucket": k, **group_metrics(groups[k], initial_equity)} for k in sorted(groups, key=rank)]


def is_non_monotonic(table: list[dict[str, Any]], bucket: str) -> bool:
    """True when an ordered bucket is losing but both neighbours are profitable (likely noise)."""
    ordered = [row for row in table if row["bucket"] != UNAVAILABLE]
    idx = next((i for i, row in enumerate(ordered) if row["bucket"] == bucket), None)
    if idx is None or idx == 0 or idx == len(ordered) - 1:
        return False
    return ordered[idx - 1]["net_pnl"] > 0 and ordered[idx + 1]["net_pnl"] > 0


NUMERIC_FIELDS = ("rsi14", "ema_spread_atr", "atr_pct", "breakout_strength_atr", "distance_from_breakout_pct",
                  "orb_width_pct", "orb_state_changes_before", "gap_pct", "option_volume_ratio", "entry_price",
                  "stop_width_pct", "holding_minutes", "mfe_r", "mae_r", "r_multiple")


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}


def winners_vs_losers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_name in NUMERIC_FIELDS:
        entry: dict[str, Any] = {}
        for label, outcome in (("winners", WIN), ("losers", LOSS)):
            entry[label] = _stats([r[field_name] for r in rows if r["outcome"] == outcome and r[field_name] is not None])
        entry["unavailable"] = sum(1 for r in rows if r[field_name] is None)
        out[field_name] = entry
    return out


def losing_breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losers = [r for r in rows if r["outcome"] == LOSS]

    def counts(fn) -> dict[str, int]:
        result: dict[str, int] = {}
        for r in losers:
            key = str(fn(r))
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items(), key=lambda kv: -kv[1]))
    return {
        "losing_trades": len(losers),
        "total_loss": sum(r["pnl"] for r in losers),
        "by_exit_reason": counts(lambda r: r["exit_reason"]),
        "by_direction": counts(lambda r: r["direction"]),
        "by_sl_vs_target_first": counts(lambda r: r["sl_vs_target_first"] or UNAVAILABLE),
        "had_initial_favorable_move": counts(lambda r: _flag(r["initial_favorable_move"], "yes", "no")),
        "false_breakout_after_entry": counts(lambda r: _flag(r["false_breakout_after_entry"], "yes", "no")),
        "losses_beyond_1r": sum(1 for r in losers if r["r_multiple"] is not None and r["r_multiple"] < -1.0),
    }


SETUP_FIELDS = ("entry_time", "direction", "option_contract", "entry_price", "exit_price", "pnl", "r_multiple",
                "exit_reason", "holding_minutes", "rsi14", "ema_trend_aligned", "breakout_strength_atr",
                "distance_from_breakout_pct", "gap_condition", "pit_regimes", "day_regimes", "mfe_r", "mae_r")


def top_setups(rows: list[dict[str, Any]], *, worst: bool, n: int = 10) -> list[dict[str, Any]]:
    pool = [r for r in rows if r["outcome"] == (LOSS if worst else WIN)]
    pool.sort(key=lambda r: r["pnl"], reverse=not worst)
    return [{k: r[k] for k in SETUP_FIELDS} for r in pool[:n]]


def _pf_value(pf: float | None) -> float:
    return -1.0 if pf is None else pf


def hypothetical_filters(rows: list[dict[str, Any]], factors: list[Factor],
                         initial_equity: float) -> list[dict[str, Any]]:
    """Naive removal of each point-in-time bucket. Not a re-simulation; never applied to S2."""
    base = group_metrics(rows, initial_equity)
    candidates: list[dict[str, Any]] = []
    for factor in factors:
        if not factor.point_in_time:
            continue
        table = factor_table(rows, factor, initial_equity)
        for bucket in table:
            if bucket["bucket"] == UNAVAILABLE or bucket["trades"] in (0, len(rows)):
                continue

            def matches(r, f=factor, b=bucket["bucket"]) -> bool:
                keys = f.key(r)
                return b in [str(k) for k in (keys if isinstance(keys, list) else [keys])]
            removed = [r for r in rows if matches(r)]
            kept = [r for r in rows if not matches(r)]
            after = group_metrics(kept, initial_equity)
            improves = (after["net_pnl"] > base["net_pnl"] and _pf_value(after["profit_factor"]) > _pf_value(base["profit_factor"])
                        and after["max_drawdown"] <= base["max_drawdown"])
            if len(removed) < MIN_FILTER_SUPPORT:
                assessment = "TOO_FEW_TRADES"
            elif len(kept) < MIN_REMAINING_TRADES:
                assessment = "REMOVES_TOO_MANY_TRADES"
            elif improves:
                assessment = "CANDIDATE_FOR_FURTHER_TESTING"
            elif after["net_pnl"] < base["net_pnl"]:
                assessment = "REDUCES_NET_PNL"
            else:
                assessment = "MIXED"
            candidates.append({
                "factor_id": factor.factor_id,
                "condition": f"exclude {factor.title.split('. ', 1)[-1]} = {bucket['bucket']}",
                "trades_affected": len(removed),
                "trades_removed": len(removed),
                "wins_removed": sum(1 for r in removed if r["outcome"] == WIN),
                "losses_removed": sum(1 for r in removed if r["outcome"] == LOSS),
                "trades_remaining": len(kept),
                "original_win_rate": base["win_rate"],
                "filtered_win_rate": after["win_rate"],
                "original_pf": base["profit_factor"],
                "filtered_pf": after["profit_factor"],
                "original_net_pnl": base["net_pnl"],
                "filtered_net_pnl": after["net_pnl"],
                "original_max_drawdown": base["max_drawdown"],
                "filtered_max_drawdown": after["max_drawdown"],
                "net_pnl_change": after["net_pnl"] - base["net_pnl"],
                "non_monotonic": bool(factor.order) and is_non_monotonic(table, bucket["bucket"]),
                "assessment": assessment,
            })
    candidates.sort(key=lambda c: (c["assessment"] != "CANDIDATE_FOR_FURTHER_TESTING", -c["net_pnl_change"]))
    return candidates


def biggest_losing_pattern(tables: dict[str, list[dict[str, Any]]], titles: dict[str, str],
                           point_in_time: set[str]) -> dict[str, Any] | None:
    """Most negative net-P&L bucket among entry-time factors (outcome descriptors would be circular)."""
    best = None
    for fid, table in tables.items():
        if fid not in point_in_time:
            continue
        for row in table:
            if row["bucket"] == UNAVAILABLE or row["trades"] < MIN_PATTERN_TRADES or row["net_pnl"] >= 0:
                continue
            if best is None or row["net_pnl"] < best["net_pnl"]:
                best = {"factor_id": fid, "factor": titles[fid], **row}
    return best


def analyze(trades: list[dict[str, Any]], candles: pd.DataFrame, premium_history: dict[str, pd.DataFrame],
            s2_cfg: dict[str, Any], initial_equity: float) -> dict[str, Any]:
    rows = build_trade_rows(trades, candles, premium_history, s2_cfg)
    factors = build_factors(rows)
    tables = {f.factor_id: factor_table(rows, f, initial_equity) for f in factors}
    titles = {f.factor_id: f.title for f in factors}
    filters = hypothetical_filters(rows, factors, initial_equity)
    strongest = next((c for c in filters if c["assessment"] == "CANDIDATE_FOR_FURTHER_TESTING"), None)
    unavailable_fields = sorted({k for r in rows for k, v in r.items() if v is None})
    return {
        "overall": group_metrics(rows, initial_equity),
        "winners": group_metrics([r for r in rows if r["outcome"] == WIN], initial_equity),
        "losers": group_metrics([r for r in rows if r["outcome"] == LOSS], initial_equity),
        "winners_vs_losers": winners_vs_losers(rows),
        "losing_breakdown": losing_breakdown(rows),
        "worst_losing_setups": top_setups(rows, worst=True),
        "best_winning_setups": top_setups(rows, worst=False),
        "factors": [{"factor_id": f.factor_id, "title": f.title, "point_in_time": f.point_in_time,
                     "note": f.note, "buckets": tables[f.factor_id]} for f in factors],
        "hypothetical_filters": filters,
        "biggest_losing_pattern": biggest_losing_pattern(tables, titles,
                                                         {f.factor_id for f in factors if f.point_in_time}),
        "strongest_candidate": strongest,
        "fields_with_unavailable_values": {k: sum(1 for r in rows if r[k] is None) for k in unavailable_fields},
        "trades": rows,
    }


# -------------------------------------------------------------------- reports
def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _fmt(value: Any, kind: str = "num") -> str:
    if value is None:
        return UNAVAILABLE
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "money":
        return f"{value:,.0f}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "-"
    return str(value)


def _metric_table(rows: Iterable[dict[str, Any]], label: str) -> list[str]:
    lines = [f"| {label} | Trades | Win% | Loss% | Net P&L | PF | Avg R | Sample |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['bucket']} | {r['trades']} | {_fmt(r['win_rate'], 'pct')} | {_fmt(r['loss_rate'], 'pct')} | "
                     f"{_fmt(r['net_pnl'], 'money')} | {_fmt(r['profit_factor'])} | {_fmt(r['average_r'])} | {r['sample']} |")
    return lines


def _setup_table(setups: list[dict[str, Any]]) -> list[str]:
    lines = ["| Entry | Dir | Contract | Entry Rs | Exit Rs | P&L | R | Exit | Hold min | RSI | EMA aligned | "
             "Brk/ATR | Dist % | Gap | PIT regimes | MFE R | MAE R |",
             "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|"]
    for s in setups:
        lines.append(" | ".join([
            "", s["entry_time"][:16], s["direction"], str(s["option_contract"]), _fmt(s["entry_price"]),
            _fmt(s["exit_price"]), _fmt(s["pnl"], "money"), _fmt(s["r_multiple"]), str(s["exit_reason"]),
            _fmt(s["holding_minutes"]), _fmt(s["rsi14"]), _fmt(s["ema_trend_aligned"]),
            _fmt(s["breakout_strength_atr"]), _fmt(s["distance_from_breakout_pct"]),
            _fmt(s["gap_condition"]), _fmt(s["pit_regimes"]), _fmt(s["mfe_r"]), _fmt(s["mae_r"]), ""]).strip())
    return lines


def _factor_section(report: dict[str, Any], ids: Iterable[str]) -> list[str]:
    lines: list[str] = []
    by_id = {f["factor_id"]: f for f in report["analysis"]["factors"]}
    for fid in ids:
        f = by_id[fid]
        tag = "point-in-time" if f["point_in_time"] else "hindsight / outcome"
        lines += ["", f"### {f['title']} ({tag})"]
        if f["note"]:
            lines.append(f"_{f['note']}_")
        lines += [""] + _metric_table(f["buckets"], "Bucket")
    return lines


def write_reports(report: dict[str, Any], out_dir: Path, stem: str = "NIFTY_S2_losing_trade_analysis") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {ext: out_dir / f"{stem}.{ext}" for ext in ("json", "csv", "md")}
    paths["json"].write_text(json.dumps(_json_safe(report), indent=2, default=str), encoding="utf-8")

    rows = report["analysis"]["trades"]
    if rows:
        with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (UNAVAILABLE if v is None else "|".join(v) if isinstance(v, list) else v)
                                 for k, v in row.items()})
    else:
        paths["csv"].write_text("no resolved trades\n", encoding="utf-8")

    a = report["analysis"]
    o, w, l = a["overall"], a["winners"], a["losers"]
    md = [f"# NIFTY S2 losing-trade analysis", "",
          f"Source run `{report['source_run_id']}` | replay `{report['replay_run_id']}` | "
          f"{report['from_date']} .. {report['to_date']} | {report['mode']}", "",
          "> Analysis only. S2 logic, configuration and risk rules are unchanged. "
          "Nothing here is applied to the strategy.", ""]
    md += [f"- {n}" for n in report["notes"]]
    md += ["", "## 1. Overall S2 summary", ""]
    md += ["| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in [
        ("Resolved trades", o["trades"]), ("Wins / losses / breakeven", f"{o['wins']} / {o['losses']} / {o['breakeven']}"),
        ("Win rate", _fmt(o["win_rate"], "pct")), ("Profit factor", _fmt(o["profit_factor"])),
        ("Net P&L", _fmt(o["net_pnl"], "money")), ("Gross profit / loss", f"{_fmt(o['gross_profit'], 'money')} / {_fmt(o['gross_loss'], 'money')}"),
        ("Average R", _fmt(o["average_r"])), ("Max drawdown", _fmt(o["max_drawdown"], "money")),
        ("Source run summary (stored)", report["source_row_summary"])]]

    md += ["", "## 2. Winning vs losing comparison", "",
           "| Feature | Winners n | Winners mean | Winners median | Losers n | Losers mean | Losers median | Unavailable |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in a["winners_vs_losers"].items():
        md.append(f"| {name} | {s['winners']['n']} | {_fmt(s['winners']['mean'])} | {_fmt(s['winners']['median'])} | "
                  f"{s['losers']['n']} | {_fmt(s['losers']['mean'])} | {_fmt(s['losers']['median'])} | {s['unavailable']} |")
    md += ["", f"Winners: {w['trades']} trades, net {_fmt(w['net_pnl'], 'money')}, avg R {_fmt(w['average_r'])}. "
           f"Losers: {l['trades']} trades, net {_fmt(l['net_pnl'], 'money')}, avg R {_fmt(l['average_r'])}."]

    lb = a["losing_breakdown"]
    md += ["", "## 3. Losing-trade breakdown", "",
           f"- Losing trades: {lb['losing_trades']} (total {_fmt(lb['total_loss'], 'money')})",
           f"- Losses worse than -1R: {lb['losses_beyond_1r']}"]
    for key in ("by_exit_reason", "by_direction", "by_sl_vs_target_first", "had_initial_favorable_move",
                "false_breakout_after_entry"):
        md.append(f"- {key.replace('_', ' ')}: " + ", ".join(f"{k}={v}" for k, v in lb[key].items()))
    md += ["", "## 4. Worst losing setups", ""] + _setup_table(a["worst_losing_setups"])
    md += ["", "## 5. Best winning setups", ""] + _setup_table(a["best_winning_setups"])
    md += ["", "## 6. Time-of-day analysis"] + _factor_section(report, ["time_of_day", "trade_number_in_day", "direction"])
    md += ["", "## 7. Market-regime analysis"] + _factor_section(
        report, ["day_regime", "pit_regime", "volatility_condition", "gap", "gap_alignment", "chop_before",
                 "opposite_breaks", "false_breakout"])
    md += ["", "## 8. Indicator-condition analysis"] + _factor_section(
        report, ["rsi_range", "rsi_alignment", "underlying_vwap", "ema_alignment", "ema_strength",
                 "underlying_volume", "atr_pct", "breakout_strength", "breakout_distance", "orb_width"])
    md += ["", "## 9. Option behavior analysis"] + _factor_section(
        report, ["option_vwap", "option_volume", "entry_premium", "stop_width", "initial_move", "sl_first",
                 "holding", "exit_reason"])

    md += ["", "## 10. Potential improvement candidates (NOT strategy changes)", "",
           "Each row removes one point-in-time bucket from the realised trade list. This is a naive removal, "
           "not a re-simulation: skipping a trade can change later entries, sizing and the daily loss budget. "
           f"Buckets removing fewer than {MIN_FILTER_SUPPORT} trades are marked TOO_FEW_TRADES; filters leaving "
           f"fewer than {MIN_REMAINING_TRADES} trades are marked REMOVES_TOO_MANY_TRADES. A candidate must raise "
           "net P&L and PF without increasing max drawdown; win rate alone is not sufficient.", "",
           "| Condition | Affected | Removed | Wins rm | Losses rm | Win% orig | Win% after | PF orig | PF after | "
           "Net orig | Net after | MaxDD orig | MaxDD after | Assessment |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in a["hypothetical_filters"]:
        md.append(f"| {c['condition']} | {c['trades_affected']} | {c['trades_removed']} | {c['wins_removed']} | "
                  f"{c['losses_removed']} | {_fmt(c['original_win_rate'], 'pct')} | {_fmt(c['filtered_win_rate'], 'pct')} | "
                  f"{_fmt(c['original_pf'])} | {_fmt(c['filtered_pf'])} | {_fmt(c['original_net_pnl'], 'money')} | "
                  f"{_fmt(c['filtered_net_pnl'], 'money')} | {_fmt(c['original_max_drawdown'], 'money')} | "
                  f"{_fmt(c['filtered_max_drawdown'], 'money')} | {c['assessment']} |")
    md += ["", "### Conclusions", ""]
    md += [f"- {line}" for line in report["conclusions"]]
    md += ["", "### Unavailable fields", ""]
    md += [f"- `{k}`: unavailable for {v} trade(s)" for k, v in a["fields_with_unavailable_values"].items()] or ["- None"]
    paths["md"].write_text("\n".join(md) + "\n", encoding="utf-8")
    return paths
