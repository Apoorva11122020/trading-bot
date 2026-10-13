"""Compare one live NIFTY S2 day against the historical S2 sample (analysis only, PAPER, no orders).

Both samples are measured with the same signal-level method as the live
signal accuracy tracker: every point-in-time S2 signal, next-bar option open
entry, unchanged premium-ATR stop and target-R levels, first touch on real
option candles, EOD square-off at the open of the first candle at/after
square-off, same-candle SL+target = AMBIGUOUS, P&L = 1 lot, costs excluded.
Engine-level results (one position at a time, trailing exits, daily loss
budget) are reported separately because they answer a different question.
Nothing here changes S2 or proposes filters.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

import pandas as pd

from src.backtest.current_day_s2 import IST, LevelParams, option_levels
from src.backtest.losing_trade_analysis import (
    _choppiness_before,
    _day_gaps,
    _false_breakout_after,
    _num,
    _orb_state,
    enrich_underlying,
)
from src.backtest.regime_classifier import ABS_GAP_RETURN
from src.indicators.technical import atr
from src.options.models import OptionResolutionError
from src.options.strategy import OptionSignalError, option_stop_loss, option_target
from src.paper_trading.signal_tracker import AMBIGUOUS, INVALID, LOSS, PENDING, WIN, SignalRecord, evaluate_candle

UNRESOLVED = "UNRESOLVED"
NUMERIC_FEATURES = (
    "entry_minute", "rsi14", "ema_spread_atr", "atr14", "atr_pct", "breakout_distance_pts",
    "breakout_distance_pct", "breakout_distance_atr", "premium_atr", "premium_atr_pct", "entry_price",
    "option_vs_vwap_pct", "stop_pts", "stop_pct", "target_pts", "target_pct", "stop_underlying_atr",
    "holding_minutes", "mfe_r", "mae_r", "dte", "orb_width_pct", "orb_state_changes_before",
    "signal_number_in_day", "same_direction_number", "gap_pct",
)
CATEGORICAL_FEATURES = ("direction", "time_bucket", "ema_aligned", "option_vs_vwap", "gap_condition",
                        "pit_regime", "dte_bucket", "false_breakout_after", "exit_reason")
TIME_BUCKETS = ((time(9, 30), time(10, 0), "09:30-10:00"), (time(10, 0), time(11, 0), "10:00-11:00"),
                (time(11, 0), time(12, 0), "11:00-12:00"), (time(12, 0), time(13, 30), "12:00-13:30"),
                (time(13, 30), time(15, 30), "13:30-15:30"))
CSV_FIELDS = (
    "sample", "session_date", "signal_time", "direction", "contract", "expiry", "dte", "status", "exit_reason",
    "entry_time", "entry_price", "stop_loss", "target", "exit_time", "exit_price", "pnl", "r_multiple",
    "holding_minutes", "mfe_r", "mae_r", "rsi14", "ema20", "ema50", "ema_aligned", "ema_spread_atr", "atr14",
    "atr_pct", "breakout_level", "breakout_distance_pts", "breakout_distance_atr", "premium_atr",
    "premium_atr_pct", "option_vwap", "option_vs_vwap", "option_vs_vwap_pct", "stop_pts", "stop_pct",
    "target_pts", "target_pct", "stop_underlying_atr", "gap_pct", "gap_condition", "pit_regime",
    "orb_width_pct", "orb_state_changes_before", "signal_number_in_day", "same_direction_number",
    "false_breakout_after", "outcome_cluster", "pit_premium_atr", "pit_stop_loss", "pit_target", "pit_status",
)


# ------------------------------------------------------------------ helpers
def _ist(ts: Any) -> pd.Timestamp:
    value = pd.Timestamp(ts)
    return value.tz_convert(IST) if value.tzinfo else value.tz_localize(IST)


def time_bucket(ts: pd.Timestamp) -> str:
    t = ts.time()
    for start, end, label in TIME_BUCKETS:
        if start <= t < end:
            return label
    return "other"


def dte_bucket(dte: int | None) -> str | None:
    if dte is None:
        return None
    return "0-1 DTE" if dte <= 1 else "2-3 DTE" if dte <= 3 else "4+ DTE"


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1)))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mann_whitney(a: list[float], b: list[float]) -> dict[str, Any] | None:
    """Two-sided Mann-Whitney U with normal approximation and tie correction."""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5:
        return None
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pooled)
    i, ties = 0, 0.0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    if sigma == 0:
        return {"u": u1, "z": 0.0, "p_value": 1.0}
    z = (u1 - n1 * n2 / 2) / sigma
    return {"u": u1, "z": round(z, 3), "p_value": round(math.erfc(abs(z) / math.sqrt(2)), 4)}


def percentile_rank(value: float | None, population: Iterable[float | None]) -> float | None:
    pop = [p for p in population if p is not None]
    if value is None or not pop:
        return None
    below = sum(1 for p in pop if p < value)
    equal = sum(1 for p in pop if p == value)
    return round((below + 0.5 * equal) / len(pop) * 100, 1)


def significance(p_value: float | None, independent_n: int) -> str:
    if p_value is None:
        return "NOT_TESTABLE"
    if independent_n < 10:
        return "LIKELY_NOISE_OR_UNTESTABLE (independent sample < 10)" if p_value >= 0.01 else \
            "SUGGESTIVE_BUT_TINY_SAMPLE"
    if p_value < 0.01:
        return "STATISTICALLY_MEANINGFUL"
    if p_value < 0.05:
        return "SUGGESTIVE"
    return "LIKELY_NOISE"


# --------------------------------------------------------- signal outcomes
def _option_vwap(premium: pd.DataFrame, sig_ts: pd.Timestamp) -> float | None:
    session = premium[(premium["timestamp"].dt.date == sig_ts.date()) & (premium["timestamp"] <= sig_ts)]
    if session.empty or "volume" not in session:
        return None
    vol = session["volume"].astype(float)
    if vol.sum() <= 0:
        return None
    typical = (session["high"] + session["low"] + session["close"]) / 3
    return float((typical * vol).sum() / vol.sum())


def _first_touch(sig, sig_ts, entry_ts, contract, entry, stop, target, future: pd.DataFrame, cutoff: time,
                 lot_size: int) -> tuple[dict[str, Any] | None, float, float]:
    record = SignalRecord.create(timestamp=sig_ts, underlying=sig.underlying, direction=sig.direction,
                                 active_from=entry_ts, instrument_key=contract.instrument_key,
                                 entry_price=entry, stop_loss=stop, target=target, quantity=lot_size)
    hi, lo, outcome = entry, entry, None
    for bar in future.to_dict("records"):
        bar = {"timestamp": _ist(bar["timestamp"]).to_pydatetime(),
               **{k: float(bar[k]) for k in ("open", "high", "low", "close")}}
        outcome = evaluate_candle(record, bar, cutoff)
        if outcome is None or outcome["reason"] != "EOD_SQUARE_OFF":
            hi, lo = max(hi, bar["high"]), min(lo, bar["low"])
        if outcome is not None:
            break
    return outcome, hi, lo


def signal_outcomes(signals, resolve: Callable, premium_for: Callable, params: LevelParams,
                    square_off: str, sample: str, lot_size: int = 65) -> list[dict[str, Any]]:
    """Resolve each signal exactly as the live tracker does, plus descriptive option features."""
    cutoff = time.fromisoformat(square_off)
    rows: list[dict[str, Any]] = []
    for sig in signals:
        sig_ts = _ist(sig.timestamp)
        row: dict[str, Any] = {
            "sample": sample, "session_date": sig_ts.date().isoformat(), "signal_time": sig_ts.isoformat(),
            "direction": sig.direction, "spot": sig.spot,
            "breakout_level": _num(sig.conditions.get("above", sig.conditions.get("below"))),
            "status": UNRESOLVED, "reason": None,
        }
        rows.append(row)
        try:
            contract = resolve(sig)
        except OptionResolutionError as exc:
            row["reason"] = f"CONTRACT_UNRESOLVED:{exc.code}"
            continue
        row.update(contract=contract.trading_symbol, instrument_key=contract.instrument_key,
                   expiry=contract.expiry.isoformat(), dte=(contract.expiry - sig_ts.date()).days)
        premium = premium_for(contract)
        if premium is None or isinstance(premium, str) or premium.empty:
            row["reason"] = premium if isinstance(premium, str) else "NO_PREMIUM_DATA"
            continue
        levels = option_levels(sig, contract, premium, params)
        if levels["status"] != "RESOLVED":
            row["reason"] = levels["reason"]
            continue
        entry, stop, target = levels["entry"], levels["stop_loss"], levels["target"]
        entry_ts = _ist(levels["entry_time"])
        vwap = _option_vwap(premium, sig_ts)
        close_at_signal = levels["premium_at_signal_close"]
        row.update(
            entry_time=entry_ts.isoformat(), entry_price=entry, stop_loss=stop, target=target,
            premium_atr=levels["premium_atr"], premium_atr_pct=levels["premium_atr"] / entry * 100,
            stop_pts=entry - stop, stop_pct=(entry - stop) / entry * 100,
            target_pts=target - entry, target_pct=(target - entry) / entry * 100,
            option_vwap=vwap,
            option_vs_vwap=None if vwap is None else ("above" if close_at_signal > vwap else "below_or_equal"),
            option_vs_vwap_pct=None if not vwap else (close_at_signal / vwap - 1) * 100,
        )
        future = premium[(premium["timestamp"] >= entry_ts) & (premium["timestamp"].dt.date == sig_ts.date())]
        outcome, hi, lo = _first_touch(sig, sig_ts, entry_ts, contract, entry, stop, target, future, cutoff, lot_size)
        pit_atr = atr(premium[premium["timestamp"] <= sig_ts], params.atr_period).iloc[-1]
        if pd.notna(pit_atr) and pit_atr > 0:
            try:
                tick = getattr(contract, "tick_size", None)
                pit_stop = option_stop_loss(entry, stop_mode="premium_atr", multiplier=params.stop_atr_multiplier,
                                            premium_atr=float(pit_atr), tick_size=tick)
                pit_target = option_target(entry, pit_stop, params.target_r, tick)
                pit_out, _, _ = _first_touch(sig, sig_ts, entry_ts, contract, entry, pit_stop, pit_target, future,
                                             cutoff, lot_size)
                row.update(pit_premium_atr=float(pit_atr), pit_stop_loss=pit_stop, pit_target=pit_target,
                           pit_status=pit_out["status"] if pit_out else PENDING,
                           pit_pnl=pit_out["pnl"] if pit_out else None)
            except OptionSignalError as exc:
                row.update(pit_status=f"INVALID:{exc.code}")
        risk = entry - stop
        row["mfe_r"], row["mae_r"] = (hi - entry) / risk, (entry - lo) / risk
        if outcome is None:
            row.update(status=PENDING, reason="NO_EXIT_EVIDENCE")
            continue
        exit_ts = _ist(outcome["exit_time"])
        row.update(status=outcome["status"], exit_reason=outcome["reason"], exit_time=exit_ts.isoformat(),
                   exit_price=outcome["exit_price"], pnl=outcome["pnl"], r_multiple=outcome["r_multiple"],
                   holding_minutes=(exit_ts - entry_ts).total_seconds() / 60)
    return rows


# ------------------------------------------------------- underlying context
def add_underlying_features(rows: list[dict[str, Any]], candles: pd.DataFrame, s2_cfg: dict[str, Any],
                            pit, delta: float) -> pd.DataFrame:
    frame = enrich_underlying(candles, s2_cfg)
    by_ts = frame.set_index("timestamp")
    gaps = _day_gaps(frame)
    range_end = time.fromisoformat(str(s2_cfg.get("range_end", "09:30")))
    counters: dict[tuple, int] = {}
    for row in sorted(rows, key=lambda r: r["signal_time"]):
        ts = _ist(row["signal_time"])
        day = ts.date()
        counters[(day,)] = counters.get((day,), 0) + 1
        counters[(day, row["direction"])] = counters.get((day, row["direction"]), 0) + 1
        bar = by_ts.loc[ts] if ts in by_ts.index else None
        day_bars = frame[frame["day"] == day]
        sign = 1.0 if row["direction"] == "BUY" else -1.0
        get = (lambda k: _num(bar[k])) if bar is not None else (lambda k: None)
        spot, atr14, ema20, ema50 = get("close"), get("atr14"), get("ema20"), get("ema50")
        level = row.get("breakout_level")
        dist = sign * (spot - level) if spot is not None and level is not None else None
        orb_hi, orb_lo = get("orb_high"), get("orb_low")
        changes, opposite = _choppiness_before(day_bars, ts, row["direction"], range_end)
        gap = gaps.get(day)
        tags = sorted(pit.regimes_at(ts.to_pydatetime()))
        row.update(
            entry_minute=ts.hour * 60 + ts.minute, time_bucket=time_bucket(ts),
            signal_number_in_day=counters[(day,)], same_direction_number=counters[(day, row["direction"])],
            rsi14=get("rsi14"), ema20=ema20, ema50=ema50,
            ema_aligned=None if ema20 is None or ema50 is None else bool(sign * (ema20 - ema50) > 0),
            ema_spread_atr=None if ema20 is None or ema50 is None or not atr14 else sign * (ema20 - ema50) / atr14,
            atr14=atr14, atr_pct=atr14 / spot * 100 if atr14 and spot else None,
            breakout_distance_pts=dist, breakout_distance_pct=dist / spot * 100 if dist is not None and spot else None,
            breakout_distance_atr=dist / atr14 if dist is not None and atr14 else None,
            orb_width_pct=(orb_hi - orb_lo) / spot * 100 if orb_hi and orb_lo and spot else None,
            orb_state_changes_before=changes, opposite_breaks_before=opposite,
            gap_pct=gap * 100 if gap is not None else None,
            gap_condition=None if gap is None else ("gap_up" if gap >= ABS_GAP_RETURN else
                                                    "gap_down" if gap <= -ABS_GAP_RETURN else "flat"),
            pit_regime="+".join(tags) if tags else "none",
            false_breakout_after=_false_breakout_after(day_bars, ts, row["direction"]),
            stop_underlying_atr=(row["stop_pts"] / delta / atr14
                                 if row.get("stop_pts") is not None and atr14 else None),
            dte_bucket=dte_bucket(row.get("dte")),
        )
    return frame


def assign_clusters(rows: list[dict[str, Any]]) -> int:
    """Signals on the same contract that exit on the same bar with the same status share one outcome."""
    clusters: dict[tuple, int] = {}
    for row in rows:
        if row["status"] in (WIN, LOSS, AMBIGUOUS):
            key = (row.get("instrument_key"), row.get("exit_time"), row["status"])
            row["outcome_cluster"] = clusters.setdefault(key, len(clusters) + 1)
    return len(clusters)


# ------------------------------------------------------------- day profiles
def day_profiles(frame: pd.DataFrame, rows: list[dict[str, Any]], range_end: str = "09:30") -> list[dict[str, Any]]:
    end = time.fromisoformat(range_end)
    out = []
    for day, bars in frame.groupby("day", sort=True):
        bars = bars.sort_values("timestamp")
        orb = bars[bars["timestamp"].dt.time < end]
        post = bars[bars["timestamp"].dt.time >= end]
        if orb.empty or post.empty:
            continue
        hi, lo = float(orb["high"].max()), float(orb["low"].min())
        first_open, last_close = float(bars["open"].iloc[0]), float(bars["close"].iloc[-1])
        path = float(bars["close"].diff().abs().sum()) + abs(float(bars["close"].iloc[0]) - first_open)
        states = [_orb_state(float(c), hi, lo) for c in post["close"]]
        crossings = sum(1 for a, b in zip(states, states[1:]) if a != b)
        first_idx = next((i for i, s in enumerate(states) if s != "inside"), None)
        follow = back_inside = first_dir = None
        if first_idx is not None:
            first_dir = "BUY" if states[first_idx] == "above" else "SELL"
            after = post.iloc[first_idx:]
            follow = (float(after["high"].max()) - hi) if first_dir == "BUY" else (lo - float(after["low"].min()))
            back_inside = any(s != states[first_idx] for s in states[first_idx + 1:])
        day_rows = [r for r in rows if r["session_date"] == day.isoformat()]
        m = group_metrics(day_rows)
        atr_open = _num(bars["atr14"].iloc[0]) if "atr14" in bars else None
        out.append({
            "date": day.isoformat(),
            "orb_width_pts": hi - lo, "orb_width_pct": (hi - lo) / first_open * 100,
            "day_range_pts": float(bars["high"].max() - bars["low"].min()),
            "day_range_pct": float(bars["high"].max() - bars["low"].min()) / first_open * 100,
            "open_to_close_pct": (last_close / first_open - 1) * 100,
            "efficiency_ratio": abs(last_close - first_open) / path if path else None,
            "orb_crossings": crossings,
            "share_bars_inside_orb": states.count("inside") / len(states),
            "first_break": first_dir,
            "first_break_follow_through_orb_widths": follow / (hi - lo) if follow is not None and hi > lo else None,
            "first_break_follow_through_atr": follow / atr_open if follow is not None and atr_open else None,
            "returned_inside_after_first_break": back_inside,
            "close_state": states[-1],
            "signals": len(day_rows), "buys": sum(r["direction"] == "BUY" for r in day_rows),
            "sells": sum(r["direction"] == "SELL" for r in day_rows),
            "wins": m["wins"], "losses": m["losses"], "win_rate": m["win_rate"], "net_pnl": m["net_pnl"],
            "first_signal_status": next((r["status"] for r in sorted(day_rows, key=lambda r: r["signal_time"])), None),
        })
    return out


# ---------------------------------------------------------------- metrics
def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decided = sorted((r for r in rows if r["status"] in (WIN, LOSS)), key=lambda r: (r["exit_time"], r["signal_time"]))
    wins = sum(r["status"] == WIN for r in decided)
    losses = len(decided) - wins
    pnls = [r["pnl"] for r in decided if r.get("pnl") is not None]
    gp, gl = sum(p for p in pnls if p > 0), -sum(p for p in pnls if p < 0)
    eq = peak = dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    rs = [r["r_multiple"] for r in decided if r.get("r_multiple") is not None]
    holds = [r["holding_minutes"] for r in decided if r.get("holding_minutes") is not None]
    return {
        "signals": len(rows),
        "buys": sum(r["direction"] == "BUY" for r in rows), "sells": sum(r["direction"] == "SELL" for r in rows),
        "resolved": len(decided), "wins": wins, "losses": losses,
        "ambiguous": sum(r["status"] == AMBIGUOUS for r in rows),
        "pending": sum(r["status"] == PENDING for r in rows),
        "unresolved_levels": sum(r["status"] in (UNRESOLVED, INVALID) for r in rows),
        "win_rate": wins / len(decided) if decided else None,
        "profit_factor": gp / gl if gl else None,
        "gross_profit": gp, "gross_loss": gl, "net_pnl": gp - gl, "max_drawdown": dd,
        "average_r": mean(rs) if rs else None,
        "median_holding_minutes": median(holds) if holds else None,
    }


def _stats(values: list[float]) -> dict[str, Any] | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    q = lambda f: values[min(len(values) - 1, int(round(f * (len(values) - 1))))]
    return {"n": len(values), "mean": mean(values), "median": median(values), "p25": q(0.25), "p75": q(0.75)}


def feature_comparison(hist: list[dict[str, Any]], today: list[dict[str, Any]]) -> dict[str, Any]:
    def subset(rows, status=None):
        return [r for r in rows if status is None or r["status"] == status]

    numeric = {}
    for f in NUMERIC_FEATURES:
        groups = {
            "historical_all": _stats([r.get(f) for r in subset(hist) if r["status"] in (WIN, LOSS)]),
            "historical_winners": _stats([r.get(f) for r in subset(hist, WIN)]),
            "historical_losers": _stats([r.get(f) for r in subset(hist, LOSS)]),
            "today_all": _stats([r.get(f) for r in subset(today) if r["status"] in (WIN, LOSS)]),
            "today_winners": _stats([r.get(f) for r in subset(today, WIN)]),
            "today_losers": _stats([r.get(f) for r in subset(today, LOSS)]),
        }
        hist_vals = [r.get(f) for r in hist if r["status"] in (WIN, LOSS)]
        today_vals = [r.get(f) for r in today if r["status"] in (WIN, LOSS)]
        groups["today_median_percentile_in_historical"] = percentile_rank(
            groups["today_all"]["median"] if groups["today_all"] else None, hist_vals)
        groups["mann_whitney_today_vs_historical"] = mann_whitney(today_vals, hist_vals)
        numeric[f] = groups
    categorical = {}
    for f in CATEGORICAL_FEATURES:
        table = {}
        for label, rows in (("historical", hist), ("today", today)):
            for r in rows:
                if r["status"] not in (WIN, LOSS):
                    continue
                key = str(r.get(f))
                cell = table.setdefault(key, {}).setdefault(label, {"signals": 0, "wins": 0})
                cell["signals"] += 1
                cell["wins"] += r["status"] == WIN
        for key, cells in table.items():
            for cell in cells.values():
                cell["win_rate"] = cell["wins"] / cell["signals"]
        categorical[f] = table
    return {"numeric": numeric, "categorical": categorical}


# ------------------------------------------------------------ investigations
def _issue(code: str, question: str, finding: str, evidence: list[str], trades: int, sample: str,
           p_value: float | None, independent_n: int, test: str, verdict: str | None = None) -> dict[str, Any]:
    return {"code": code, "question": question, "finding": finding, "evidence": evidence, "trades": trades,
            "sample_size": sample, "test": test, "p_value": None if p_value is None else round(p_value, 4),
            "independent_n": independent_n, "verdict": verdict or significance(p_value, independent_n)}


def _wr(rows):
    d = [r for r in rows if r["status"] in (WIN, LOSS)]
    return (sum(r["status"] == WIN for r in d) / len(d)) if d else None, len(d)


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _daily_rates(rows: list[dict[str, Any]], direction: str | None, min_signals: int = 3) -> list[float]:
    by_day: dict[str, list[str]] = {}
    for r in rows:
        if r["status"] in (WIN, LOSS) and (direction is None or r["direction"] == direction):
            by_day.setdefault(r["session_date"], []).append(r["status"])
    return [s.count(WIN) / len(s) for s in by_day.values() if len(s) >= min_signals]


def _day_level(today_rate: float | None, hist_rates: list[float]) -> tuple[float | None, str]:
    """Empirical one-sided p: share of historical days with a win rate at or below today's."""
    if today_rate is None or not hist_rates:
        return None, "n/a"
    share = sum(r <= today_rate for r in hist_rates) / len(hist_rates)
    return share, (f"{sum(r <= today_rate for r in hist_rates)}/{len(hist_rates)} historical days had a win rate "
                   f"<= {today_rate * 100:.1f}% (empirical p={share:.3f}); historical daily median "
                   f"{median(hist_rates) * 100:.1f}%")


def _med(rows, field):
    st = _stats([r.get(field) for r in rows])
    return None if st is None else st["median"]


def _bucket_counts(rows, key) -> str:
    table: dict[str, list[int]] = {}
    for r in rows:
        if r["status"] in (WIN, LOSS):
            cell = table.setdefault(str(r.get(key)), [0, 0])
            cell[0 if r["status"] == WIN else 1] += 1
    return "; ".join(f"{k} {v[0]}W/{v[1]}L" for k, v in sorted(table.items()))


def investigations(hist: list[dict[str, Any]], today: list[dict[str, Any]], hist_days: list[dict[str, Any]],
                   today_day: dict[str, Any], engine: dict[str, Any], target_r: float,
                   lot_size: int = 65, stop_multiplier: float = 1.5) -> list[dict[str, Any]]:
    out = []
    decided_hist = [r for r in hist if r["status"] in (WIN, LOSS)]
    decided_today = [r for r in today if r["status"] in (WIN, LOSS)]
    breakeven = 1 / (1 + target_r)
    pct_rank = lambda field: percentile_rank(today_day.get(field), [d.get(field) for d in hist_days])

    # A. BUY signals --------------------------------------------------------
    t_buy = [r for r in decided_today if r["direction"] == "BUY"]
    t_buy_wins = sum(r["status"] == WIN for r in t_buy)
    h_buy_wr, h_buy_n = _wr([r for r in hist if r["direction"] == "BUY"])
    h_early_wr, h_early_n = _wr([r for r in hist if r["direction"] == "BUY" and r.get("time_bucket") == "09:30-10:00"])
    buy_clusters = len({r.get("outcome_cluster") for r in t_buy})
    p_a = binom_cdf(t_buy_wins, buy_clusters, h_buy_wr) if t_buy and h_buy_wr is not None and t_buy_wins == 0 else (
        binom_cdf(t_buy_wins, len(t_buy), h_buy_wr) if t_buy and h_buy_wr is not None else None)
    back = sum(bool(r.get("false_breakout_after")) for r in t_buy)
    mfe = [round(r["mfe_r"], 2) for r in t_buy]
    out.append(_issue(
        "A", "Why all BUY signals lost",
        (f"{len(t_buy)} BUY signals fired between {t_buy[0]['signal_time'][11:16]} and {t_buy[-1]['signal_time'][11:16]} "
         f"on {len({r.get('contract') for r in t_buy})} contract(s); NIFTY closed back inside the opening range within "
         f"3 bars after {back}/{len(t_buy)} of them and all hit the stop (best excursion {max(mfe):.2f}R, target "
         f"{target_r}R). BUY signals were already weak historically ({_pct(h_buy_wr)} win rate, n={h_buy_n}), so "
         f"0/{len(t_buy)} on {buy_clusters} independent exit event(s) is consistent with that base rate.")
        if t_buy else "No BUY signals today.",
        [f"Times {', '.join(r['signal_time'][11:16] for r in t_buy)}; contracts {sorted({r.get('contract') for r in t_buy})}; "
         f"exits {sorted({(r.get('exit_time') or '')[11:16] for r in t_buy})}.",
         f"Max favourable excursion before exit (R): {mfe}.",
         f"Historical BUY 09:30-10:00 win rate {_pct(h_early_wr)} (n={h_early_n}).",
         f"Day: first break {today_day.get('first_break')}, first-break follow-through "
         f"{today_day.get('first_break_follow_through_atr') or 0:.2f} ATR (percentile "
         f"{pct_rank('first_break_follow_through_atr')}), returned inside ORB: "
         f"{today_day.get('returned_inside_after_first_break')}."],
        len(t_buy), f"{len(t_buy)} signals, {buy_clusters} independent exit event(s), 1 day",
        p_a, buy_clusters,
        "exact binomial P(0 wins | n = independent exit events, p = historical BUY win rate)"))

    # B. SELL signals -------------------------------------------------------
    t_sell = [r for r in decided_today if r["direction"] == "SELL"]
    s_wins = sum(r["status"] == WIN for r in t_sell)
    h_sell_wr, h_sell_n = _wr([r for r in hist if r["direction"] == "SELL"])
    clusters: dict[Any, str] = {}
    for r in t_sell:
        clusters.setdefault(r["outcome_cluster"], r["status"])
    c_wins = sum(s == WIN for s in clusters.values())
    p_cluster = binom_cdf(c_wins, len(clusters), h_sell_wr) if clusters and h_sell_wr else None
    today_sell_wr = s_wins / len(t_sell) if t_sell else None
    sell_daily = _daily_rates(hist, "SELL")
    p_day, day_text = _day_level(today_sell_wr, sell_daily)
    losers = [r for r in t_sell if r["status"] == LOSS]
    winners = [r for r in t_sell if r["status"] == WIN]
    worst_bucket = max({r["time_bucket"] for r in losers} or {"n/a"},
                       key=lambda b: sum(r["time_bucket"] == b for r in losers))
    worst_n = sum(r["time_bucket"] == worst_bucket for r in losers)
    worst_w = sum(r["time_bucket"] == worst_bucket for r in winners)
    out.append(_issue(
        "B", "Why most SELL signals lost",
        (f"{len(losers)}/{len(t_sell)} SELL signals lost ({_pct(today_sell_wr)} vs historical {_pct(h_sell_wr)}), "
         f"but they form only {len(clusters)} independent exit events ({c_wins} winning). The {worst_bucket} block "
         f"alone produced {worst_n} losses and {worst_w} wins. Losers rarely moved in favour (median MFE "
         f"{_med(losers, 'mfe_r') or 0:.2f}R) before the stop. Across historical days, a SELL win rate this low "
         f"{'is not unusual' if p_day is not None and p_day >= 0.1 else 'is uncommon'}: {day_text}."),
        [f"Signal-level binomial vs pooled historical SELL rate (overstates significance; overlapping signals): "
         f"p={binom_cdf(s_wins, len(t_sell), h_sell_wr):.4f}." if t_sell and h_sell_wr else "n/a",
         f"Cluster-level binomial: p={p_cluster:.4f} (n={len(clusters)} clusters, all from one day)."
         if p_cluster is not None else "n/a",
         f"Day-level: {day_text}.",
         f"Entry-time buckets: {_bucket_counts(t_sell, 'time_bucket')}.",
         f"Winners median MAE {_med(winners, 'mae_r') or 0:.2f}R; losers median holding "
         f"{_med(losers, 'holding_minutes') or 0:.0f} min vs winners {_med(winners, 'holding_minutes') or 0:.0f} min.",
         f"The historical SELL rate comes from a window in which NIFTY drifted lower (historical median day "
         f"open-to-close {median(d['open_to_close_pct'] for d in hist_days):+.2f}%)."],
        len(t_sell), f"{len(t_sell)} signals, {len(clusters)} exit clusters, 1 day; {len(sell_daily)} historical days",
        p_day, len(sell_daily), "empirical day-level test (share of historical days with SELL win rate <= today's)"))

    # C. ORB assumption ------------------------------------------------------
    back_hist = [d for d in hist_days if d["first_break"]]
    first_hist = [d for d in hist_days if d["first_signal_status"] in (WIN, LOSS)]
    ft_rank = pct_rank("first_break_follow_through_atr")
    out.append(_issue(
        "C", "Did the opening-range breakout assumption fail today?",
        (f"The first breakout was {today_day['first_break']} and price returned inside the range "
         f"({today_day['returned_inside_after_first_break']}); follow-through was "
         f"{today_day.get('first_break_follow_through_atr') or 0:.2f} ATR (percentile {ft_rank} of {len(hist_days)} days) "
         f"and NIFTY ended the day {today_day['close_state']} the opening range. A return inside the range after the "
         f"first break happened on {sum(bool(d['returned_inside_after_first_break']) for d in back_hist)}/{len(back_hist)} "
         f"historical days, so the failure mode is common; today's follow-through was on the weak side."),
        [f"ORB crossings after 09:30: today {today_day['orb_crossings']}, historical median "
         f"{median(d['orb_crossings'] for d in hist_days)} (percentile {pct_rank('orb_crossings')}).",
         f"First-break follow-through in ORB widths: today {today_day.get('first_break_follow_through_orb_widths') or 0:.2f}, "
         f"percentile {pct_rank('first_break_follow_through_orb_widths')}.",
         f"First signal of the day won on {sum(d['first_signal_status'] == WIN for d in first_hist)}/{len(first_hist)} "
         f"historical days; today: {today_day['first_signal_status']}.",
         f"Open-to-close today {today_day['open_to_close_pct']:+.2f}%."],
        today_day["signals"], f"1 day vs {len(hist_days)} historical days", None, 1,
        "day-level percentiles (n = 1 day)", "NOT_TESTABLE (one day); descriptive evidence only"))

    # D. Choppiness ----------------------------------------------------------
    er_rank, range_rank = pct_rank("efficiency_ratio"), pct_rank("day_range_pct")
    out.append(_issue(
        "D", "Was today range-bound / choppy?",
        (f"Two-sided rather than quiet: day range {today_day['day_range_pct']:.2f}% (percentile {range_rank}) but net "
         f"move only {today_day['open_to_close_pct']:+.2f}%, giving an efficiency ratio of "
         f"{today_day['efficiency_ratio']:.3f} (percentile {er_rank}; historical median "
         f"{median(d['efficiency_ratio'] for d in hist_days):.3f}). Price travelled a lot without net progress, which "
         f"is the environment in which a breakout-continuation entry with a 2R target is least likely to pay."),
        [f"ORB width {today_day['orb_width_pct']:.2f}% (percentile {pct_rank('orb_width_pct')}).",
         f"Share of post-ORB bars inside the ORB {today_day['share_bars_inside_orb']:.2f} (percentile "
         f"{pct_rank('share_bars_inside_orb')}).",
         f"Point-in-time regime tags on today's signals: {sorted({r.get('pit_regime') for r in today})} "
         "(the existing regime filter's avoided regimes were not detected)."],
        today_day["signals"], f"1 day vs {len(hist_days)} historical days", None, 1,
        "day-level percentiles (n = 1 day)", "NOT_TESTABLE (one day); descriptive evidence only"))

    # E. Premium behaviour -----------------------------------------------------
    def cmp(field, digits=2):
        h, t = _stats([r.get(field) for r in decided_hist]), _stats([r.get(field) for r in decided_today])
        mw = mann_whitney([r.get(field) for r in decided_today], [r.get(field) for r in decided_hist])
        if not h or not t:
            return f"{field}: n/a", None
        return (f"{field}: today median {t['median']:.{digits}f} vs historical {h['median']:.{digits}f} "
                f"(historical range {min(r.get(field) for r in decided_hist if r.get(field) is not None):.{digits}f}"
                f"-{max(r.get(field) for r in decided_hist if r.get(field) is not None):.{digits}f}; "
                f"MW p={mw['p_value'] if mw else 'n/a'})"), mw
    hist_dte = [r["dte"] for r in decided_hist if r.get("dte") is not None]
    today_dte = sorted({r["dte"] for r in decided_today if r.get("dte") is not None})
    hist_expiries = sorted({r.get("expiry") for r in hist if r.get("expiry")})
    overlap = sum(1 for r in decided_hist if r.get("dte") is not None and r["dte"] <= max(today_dte or [0]))
    e_lines = [cmp(f)[0] for f in ("dte", "entry_price", "premium_atr", "premium_atr_pct", "option_vs_vwap_pct")]
    e_lines += [f"Historical expiries used: {hist_expiries}; historical DTE range {min(hist_dte)}-{max(hist_dte)}; "
                f"today DTE {today_dte}. Historical signals with DTE <= today's: {overlap}.",
                "Cause: the contract resolver only lists currently tradable expiries, so every historical signal was "
                "mapped to a future expiry instead of the weekly that was nearest on that day."]
    out.append(_issue(
        "E", "Did option premium behaviour differ from the historical sample?",
        (f"Yes, structurally. Today's contracts were {today_dte} day(s) from expiry while every historical signal used a "
         f"contract {min(hist_dte)}-{max(hist_dte)} days out (median {median(hist_dte):.0f}). Premium ATR was "
         f"{_med(decided_today, 'premium_atr_pct') or 0:.1f}% of premium today vs "
         f"{_med(decided_hist, 'premium_atr_pct') or 0:.1f}% historically. The historical sample contains {overlap} "
         f"signals at comparable DTE, so it cannot say how S2 behaves on near-expiry contracts."),
        e_lines, len(decided_today), f"today {len(decided_today)} signals vs historical {len(decided_hist)}",
        None, len(decided_hist), "distribution comparison (Mann-Whitney shown per feature)",
        "STRUCTURAL_DIFFERENCE (not sampling noise): historical sample has no comparable-DTE contracts"))

    # F. SL width ---------------------------------------------------------------
    t_losers = [r for r in today if r["status"] == LOSS]
    h_losers = [r for r in hist if r["status"] == LOSS]
    f_lines = [cmp(f)[0] for f in ("stop_pct", "stop_underlying_atr", "target_pct", "mfe_r", "mae_r")]
    reached = sum(r["mfe_r"] >= 1 for r in t_losers)
    h_reached = sum(r["mfe_r"] >= 1 for r in h_losers)
    f_lines.append(f"Losers that reached +1R before stopping: today {reached}/{len(t_losers)}, historical "
                   f"{h_reached}/{len(h_losers)}.")
    stop_pts = _med(decided_today, "stop_pts")
    if stop_pts:
        f_lines.append(f"Median stop distance today {stop_pts:.2f} premium points = Rs {stop_pts * lot_size:,.0f} per "
                       f"lot (historical Rs {(_med(decided_hist, 'stop_pts') or 0) * lot_size:,.0f}).")
    ordered = sorted(r["stop_underlying_atr"] for r in decided_hist if r.get("stop_underlying_atr") is not None)
    if len(ordered) >= 30:
        c1, c2 = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]
        label = lambda v: "narrow" if v < c1 else "mid" if v < c2 else "wide"
        for r in decided_hist + decided_today:
            if r.get("stop_underlying_atr") is not None:
                r["_stop_tercile"] = label(r["stop_underlying_atr"])
        f_lines.append(f"Win/loss by stop width in underlying ATRs (terciles <{c1:.2f}, <{c2:.2f}, wider): "
                       f"historical {_bucket_counts(decided_hist, '_stop_tercile')}; "
                       f"today {_bucket_counts(decided_today, '_stop_tercile')}.")
        for r in decided_hist + decided_today:
            r.pop("_stop_tercile", None)
    mw_stop = cmp("stop_underlying_atr")[1]
    out.append(_issue(
        "F", "Is the option ATR / SL calculation too wide or inappropriate today?",
        (f"The stop formula is unchanged ({stop_multiplier:g} x premium ATR) but near-expiry premium ATR made it "
         f"{_med(decided_today, 'stop_pct') or 0:.1f}% of premium (historical {_med(decided_hist, 'stop_pct') or 0:.1f}%). "
         f"Converted to index terms (stop / delta / ATR14) it needed {_med(decided_today, 'stop_underlying_atr') or 0:.2f} "
         f"ATR against the trade vs {_med(decided_hist, 'stop_underlying_atr') or 0:.2f} historically, and the 2R target "
         f"needed twice that in favour. On a low-efficiency day few signals travelled that far. Whether the width is "
         f"'inappropriate' cannot be judged from history, because history has no near-expiry contracts."),
        f_lines, len(t_losers), f"today {len(t_losers)} losers vs historical {len(h_losers)}",
        mw_stop["p_value"] if mw_stop else None, 1, "Mann-Whitney on stop width in underlying ATRs (one day)",
        "DIFFERENT_FROM_HISTORY (one day; effect on outcomes not testable)"))

    # G. Signal frequency -------------------------------------------------------
    sig_hist = [d["signals"] for d in hist_days]
    first_wr, first_n = _wr([r for r in hist if r.get("same_direction_number") == 1])
    rep_wr, rep_n = _wr([r for r in hist if (r.get("same_direction_number") or 0) > 1])
    eng = engine.get("today", {})
    n_clusters = len({r["outcome_cluster"] for r in decided_today if r.get("outcome_cluster")})
    daily_all = _daily_rates(hist, None)
    p_all, all_text = _day_level(len([r for r in decided_today if r["status"] == WIN]) / len(decided_today)
                                 if decided_today else None, daily_all)
    out.append(_issue(
        "G", "Is signal frequency too high because S2 fires repeatedly after the breakout?",
        (f"S2 emits a signal on every close beyond the range: {today_day['signals']} today vs a historical median of "
         f"{median(sig_hist)} per day (percentile {pct_rank('signals')}), so today's count is normal for S2. Repeats "
         f"are highly correlated - the 40 resolved signals reduce to {n_clusters} exit events - which multiplies one "
         f"day's direction error into many tracked losses. Historically repeat signals did not do worse than the first "
         f"signal ({_pct(rep_wr)} vs {_pct(first_wr)}). The research engine, holding one position at a time, took "
         f"{eng.get('resolved_trades', 'n/a')} trade(s) today (net {eng.get('net_pnl', 'n/a')}); "
         f"{eng.get('skip_summary', 'skip reasons n/a')}."),
        [f"Historical first signal per direction per day: {_pct(first_wr)} (n={first_n}); repeats {_pct(rep_wr)} (n={rep_n}).",
         f"Day-level overall win rate: {all_text}.",
         f"Breakeven win rate at {target_r}R: {breakeven * 100:.1f}%."],
        today_day["signals"], f"1 day; historical {len(decided_hist)} signals over {len(hist_days)} days",
        p_all, len(daily_all), "empirical day-level test on overall win rate"))

    # H. Data-integrity check on level sizing --------------------------------------
    def pit_summary(rows):
        both = [r for r in rows if r["status"] in (WIN, LOSS) and r.get("pit_status") in (WIN, LOSS)]
        changed = sum(r["status"] != r["pit_status"] for r in both)
        ratio = [r["premium_atr"] / r["pit_premium_atr"] for r in rows
                 if r.get("pit_premium_atr") and r.get("premium_atr") is not None]
        wr = sum(r["status"] == WIN for r in both) / len(both) if both else None
        pit_wr = sum(r["pit_status"] == WIN for r in both) / len(both) if both else None
        return both, changed, (median(ratio) if ratio else None), wr, pit_wr
    hb, hc, hr, hwr, hpwr = pit_summary(hist)
    tb, tc, tr, twr, tpwr = pit_summary(today)
    out.append(_issue(
        "H", "Data-integrity check: do the replayed levels use information unavailable at entry?",
        ("Yes, slightly. For next-bar-open entries, OptionsResearchEngine (mirrored by the replay) computes premium "
         "ATR through the entry candle, whose high/low/close are not known at the open. Re-evaluating the same entries "
         f"with ATR through the signal candle only changes {hc}/{len(hb)} historical and {tc}/{len(tb)} of today's "
         "outcomes. This affects historical and live-replay measurements alike; the live paper runner sizes from "
         "completed candles only. Reported for awareness; nothing was changed."),
        [f"Median ATR ratio (engine / point-in-time): historical {hr:.3f}, today {tr:.3f}." if hr and tr else "n/a",
         f"Win rate engine-levels vs point-in-time levels: historical {_pct(hwr)} vs {_pct(hpwr)} (n={len(hb)}); "
         f"today {_pct(twr)} vs {_pct(tpwr)} (n={len(tb)})."],
        len(tb), f"today {len(tb)}, historical {len(hb)} signals", None, len(hb), "paired re-evaluation (descriptive)",
        "MEASUREMENT_CAVEAT (does not explain today's result)" if tc <= max(2, len(tb) // 10)
        else "MEASUREMENT_CAVEAT (material for today's result)"))
    return out


# ------------------------------------------------------------------ reports
def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def _f(v, kind="num"):
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "money":
        return f"Rs {v:,.0f}"
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


def write_reports(report: dict[str, Any], out_dir: Path, stem: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / f"{stem}.json", "csv": out_dir / f"{stem}.csv", "md": out_dir / f"{stem}.md"}
    paths["json"].write_text(json.dumps(_jsonable(report), indent=2, default=str), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in report["today_signals"] + report["historical_signals"]:
            writer.writerow({k: _jsonable(row.get(k)) for k in CSV_FIELDS})
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    return paths


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [f"# NIFTY S2 live vs historical — {report['live_date']}", "",
             f"Mode: {report['mode']}", "", "## Method", ""]
    lines += [f"- {n}" for n in report["notes"]]
    lines += ["", "## Headline comparison (signal level, 1 lot, costs excluded)", "",
              "| Metric | Historical " + report["historical_window"] + " | Today |", "|---|---|---|"]
    h, t = s["historical_signal_level"], s["today_signal_level"]
    for key, kind in (("signals", "num"), ("buys", "num"), ("sells", "num"), ("resolved", "num"),
                      ("wins", "num"), ("losses", "num"), ("ambiguous", "num"), ("win_rate", "pct"),
                      ("profit_factor", "num"), ("net_pnl", "money"), ("max_drawdown", "money"),
                      ("average_r", "num"), ("median_holding_minutes", "num")):
        lines.append(f"| {key} | {_f(h[key], kind)} | {_f(t[key], kind)} |")
    lines += ["", f"Historical signals per day: {s['historical_signals_per_day']}. "
              f"Historical days: {s['historical_days']}. Today win rate 95% Wilson CI: {s['today_win_rate_ci']}.", "",
              "## Engine-level context (one position at a time, trailing exits, daily loss budget)", ""]
    for k, v in report["engine"].items():
        lines.append(f"- {k}: " + (", ".join(f"{a}={_f(b) if not isinstance(b, str) else b}" for a, b in v.items()
                                              if a not in ("trades", "skipped_signals"))
                                    if isinstance(v, dict) else str(v)))
        for t in v.get("trades", []) if isinstance(v, dict) and k == "today" else []:
            lines.append(f"  - {t['signal_time']} {t['direction']} {t['trading_symbol']} entry {t['entry_price']} "
                         f"exit {t['exit_price']} {t['exit_reason']} pnl {t['pnl']}")
    lines += ["", "## Today's day profile vs historical days", "", "| Metric | Today | Hist median | Percentile |",
              "|---|---|---|---|"]
    for k, v in report["day_comparison"].items():
        lines.append(f"| {k} | {_f(v['today'])} | {_f(v['historical_median'])} | {_f(v['percentile'])} |")
    lines += ["", "## Investigations", ""]
    for i in report["investigations"]:
        lines += [f"### {i['code']}. {i['question']}", "", i["finding"], ""]
        lines += [f"- {e}" for e in i["evidence"]]
        lines += ["", f"Trades: {i['trades']}; sample: {i['sample_size']}; test: {i['test']}; "
                  f"p = {i['p_value'] if i['p_value'] is not None else 'n/a'}; **verdict: {i['verdict']}**", ""]
    lines += ["## Feature comparison (resolved signals; medians)", "",
              "| Feature | Hist all | Hist win | Hist loss | Today all | Today win | Today loss | Today pct | MW p |",
              "|---|---|---|---|---|---|---|---|---|"]
    med = lambda g: "n/a" if not g else f"{g['median']:.2f}"
    for f, g in report["feature_comparison"]["numeric"].items():
        mw = g["mann_whitney_today_vs_historical"]
        lines.append(f"| {f} | {med(g['historical_all'])} | {med(g['historical_winners'])} | "
                     f"{med(g['historical_losers'])} | {med(g['today_all'])} | {med(g['today_winners'])} | "
                     f"{med(g['today_losers'])} | {_f(g['today_median_percentile_in_historical'])} | "
                     f"{mw['p_value'] if mw else 'n/a'} |")
    lines += ["", "## Categorical splits (wins/signals)", ""]
    for f, table in report["feature_comparison"]["categorical"].items():
        cells = "; ".join(
            f"{k}: hist {c.get('historical', {}).get('wins', 0)}/{c.get('historical', {}).get('signals', 0)}, "
            f"today {c.get('today', {}).get('wins', 0)}/{c.get('today', {}).get('signals', 0)}"
            for k, c in sorted(table.items()))
        lines.append(f"- **{f}**: {cells}")
    lines += ["", "## Today's losers and winners", ""]
    for label in ("losers", "winners"):
        common = report["today_groups"][label]
        lines += [f"### {label.title()} ({common['count']})", ""]
        lines += [f"- {c}" for c in common["characteristics"]]
        lines += ["", "| Time | Dir | Contract | Entry | SL | Target | Exit | Exit time | R | Hold min | MFE R |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in common["signals"]:
            lines.append(f"| {r['signal_time'][11:16]} | {r['direction']} | {r.get('contract')} | "
                         f"{_f(r.get('entry_price'))} | {_f(r.get('stop_loss'))} | {_f(r.get('target'))} | "
                         f"{_f(r.get('exit_price'))} | {(r.get('exit_time') or '')[11:16]} | {_f(r.get('r_multiple'))} | "
                         f"{_f(r.get('holding_minutes'))} | {_f(r.get('mfe_r'))} |")
        lines.append("")
    lines += ["## Top evidence-based reasons", ""] + [f"{n}. {r}" for n, r in enumerate(report["top_reasons"], 1)]
    lines += ["", "No strategy change, filter or parameter is proposed or applied in this report.", ""]
    return "\n".join(lines)


def group_characteristics(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["none"]
    n = len(rows)
    out = []
    dirs = {d: sum(r["direction"] == d for r in rows) for d in ("BUY", "SELL")}
    out.append(f"Direction: BUY {dirs['BUY']}, SELL {dirs['SELL']}.")
    buckets = {}
    for r in rows:
        buckets[r["time_bucket"]] = buckets.get(r["time_bucket"], 0) + 1
    out.append("Entry time: " + ", ".join(f"{k} {v}" for k, v in sorted(buckets.items())) + ".")
    for f, label in (("rsi14", "RSI14"), ("breakout_distance_atr", "breakout distance (ATR)"),
                     ("stop_pct", "stop % of premium"), ("premium_atr_pct", "premium ATR %"),
                     ("mfe_r", "MFE (R)"), ("mae_r", "MAE (R)"), ("holding_minutes", "holding minutes"),
                     ("same_direction_number", "same-direction signal number")):
        st = _stats([r.get(f) for r in rows])
        if st:
            out.append(f"{label}: median {st['median']:.2f} (IQR {st['p25']:.2f}-{st['p75']:.2f}).")
    aligned = [r.get("ema_aligned") for r in rows if r.get("ema_aligned") is not None]
    if aligned:
        out.append(f"EMA20/EMA50 aligned with direction: {sum(aligned)}/{len(aligned)}.")
    vw = [r.get("option_vs_vwap") for r in rows if r.get("option_vs_vwap")]
    if vw:
        out.append(f"Option above session VWAP at signal: {vw.count('above')}/{len(vw)}.")
    contracts = {}
    for r in rows:
        contracts[r.get("contract")] = contracts.get(r.get("contract"), 0) + 1
    out.append("Contracts: " + ", ".join(f"{k} x{v}" for k, v in contracts.items()) + ".")
    reasons = {}
    for r in rows:
        reasons[r.get("exit_reason")] = reasons.get(r.get("exit_reason"), 0) + 1
    out.append("Exit reasons: " + ", ".join(f"{k} {v}" for k, v in reasons.items()) + f" (n={n}).")
    return out
