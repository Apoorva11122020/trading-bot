"""Classify trading days into market regimes from real 5m OHLC (no look-ahead thresholds)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest.research_strategies import enrich_s2

IST = ZoneInfo("Asia/Kolkata")

REGIME_CATALOG: list[tuple[str, str]] = [
    ("strong_bullish_trend", "Strong bullish trend"),
    ("strong_bearish_trend", "Strong bearish trend"),
    ("range_bound", "Range-bound / sideways"),
    ("sudden_upward_spike", "Sudden upward spike"),
    ("sudden_downward_spike", "Sudden downward spike/crash"),
    ("high_volatility", "High volatility"),
    ("low_volatility", "Low volatility"),
    ("gap_up", "Gap-up"),
    ("gap_down", "Gap-down"),
    ("choppy_whipsaw", "Choppy / whipsaw"),
    ("false_breakout", "False breakout"),
    ("opening_volatility", "Opening volatility"),
    ("midday_low_momentum", "Midday low momentum"),
    ("late_session_volatility", "Late-session volatility"),
]

MIN_HISTORY_DAYS = 5
LOOKBACK_DAYS = 20
ABS_TREND_RETURN = 0.006
ABS_GAP_RETURN = 0.002
ABS_BAR_SPIKE = 0.0035


@dataclass(frozen=True)
class DayFeatures:
    day: date
    open: float
    high: float
    low: float
    close: float
    prev_close: float | None
    day_return: float
    day_range_pct: float
    max_bar_up: float
    max_bar_down: float
    sign_changes: int
    opening_range_pct: float
    midday_range_pct: float
    late_range_pct: float
    false_breakout: bool


def _ts_local(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize(IST)
    return ts.dt.tz_convert(IST)


def _between(times: pd.Series, start: time, end: time) -> pd.Series:
    local = times.dt.tz_convert(IST)
    minutes = local.dt.hour * 60 + local.dt.minute
    start_m = start.hour * 60 + start.minute
    end_m = end.hour * 60 + end.minute
    return (minutes >= start_m) & (minutes < end_m)


def _day_features(day_df: pd.DataFrame, prev_close: float | None, s2_cfg: dict[str, Any]) -> DayFeatures:
    frame = day_df.sort_values("timestamp")
    times = _ts_local(frame["timestamp"])
    open_ = float(frame.iloc[0]["open"])
    high = float(frame["high"].max())
    low = float(frame["low"].min())
    close = float(frame.iloc[-1]["close"])
    day_return = (close - open_) / open_ if open_ else 0.0
    day_range_pct = (high - low) / close if close else 0.0

    closes = frame["close"].astype(float)
    bar_rets = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    max_bar_up = float(bar_rets.max()) if not bar_rets.empty else 0.0
    max_bar_down = float(bar_rets.min()) if not bar_rets.empty else 0.0

    signs = np.sign(bar_rets)
    sign_changes = int((signs.diff().abs() > 0).sum()) if len(signs) > 1 else 0

    open_mask = _between(times, time(9, 15), time(9, 30))
    mid_mask = _between(times, time(11, 0), time(13, 0))
    late_mask = _between(times, time(14, 0), time(15, 30))

    def _range_pct(mask: pd.Series) -> float:
        if not mask.any():
            return 0.0
        sub = frame.loc[mask]
        c = float(sub["close"].iloc[-1]) or close
        return (float(sub["high"].max()) - float(sub["low"].min())) / c if c else 0.0

    false_breakout = _detect_false_breakout(frame, s2_cfg)

    return DayFeatures(
        day=times.iloc[0].date(),
        open=open_,
        high=high,
        low=low,
        close=close,
        prev_close=prev_close,
        day_return=day_return,
        day_range_pct=day_range_pct,
        max_bar_up=max_bar_up,
        max_bar_down=max_bar_down,
        sign_changes=sign_changes,
        opening_range_pct=_range_pct(open_mask),
        midday_range_pct=_range_pct(mid_mask),
        late_range_pct=_range_pct(late_mask),
        false_breakout=false_breakout,
    )


def _detect_false_breakout(day_df: pd.DataFrame, s2_cfg: dict[str, Any]) -> bool:
    enriched = enrich_s2(day_df.copy(), s2_cfg)
    if enriched.empty:
        return False
    orb_high = enriched["orb_high"].iloc[-1]
    orb_low = enriched["orb_low"].iloc[-1]
    if pd.isna(orb_high) or pd.isna(orb_low):
        return False
    range_end = s2_cfg.get("range_end", "09:30")
    hour, minute = map(int, range_end.split(":"))
    after = _ts_local(enriched["timestamp"]).dt.time > time(hour, minute)
    post = enriched.loc[after]
    if post.empty:
        return False
    broke_high = (post["high"] > orb_high).any()
    broke_low = (post["low"] < orb_low).any()
    back_inside_high = broke_high and (post["close"] < orb_high).any()
    back_inside_low = broke_low and (post["close"] > orb_low).any()
    return bool(back_inside_high or back_inside_low)


def _percentile_threshold(history: list[float], percentile: float, fallback: float) -> float:
    if len(history) < MIN_HISTORY_DAYS:
        return fallback
    window = history[-LOOKBACK_DAYS:]
    return float(np.percentile(window, percentile))


def classify_day(features: DayFeatures, history: list[DayFeatures]) -> set[str]:
    """Label a day using only prior days in *history* for percentile thresholds."""
    regimes: set[str] = set()
    past_returns = [h.day_return for h in history]
    past_ranges = [h.day_range_pct for h in history]
    past_max_up = [h.max_bar_up for h in history]
    past_max_down = [abs(h.max_bar_down) for h in history]
    past_open_rng = [h.opening_range_pct for h in history]
    past_mid_rng = [h.midday_range_pct for h in history]
    past_late_rng = [h.late_range_pct for h in history]

    bull_thr = max(ABS_TREND_RETURN, _percentile_threshold(past_returns, 80, ABS_TREND_RETURN))
    bear_thr = min(-ABS_TREND_RETURN, _percentile_threshold(past_returns, 20, -ABS_TREND_RETURN))
    if features.day_return >= bull_thr:
        regimes.add("strong_bullish_trend")
    if features.day_return <= bear_thr:
        regimes.add("strong_bearish_trend")

    low_range_thr = _percentile_threshold(past_ranges, 25, 0.004)
    high_range_thr = _percentile_threshold(past_ranges, 80, 0.012)
    if features.day_range_pct <= low_range_thr:
        regimes.add("range_bound")
    if features.day_range_pct >= high_range_thr:
        regimes.add("high_volatility")
    if features.day_range_pct <= _percentile_threshold(past_ranges, 20, 0.003):
        regimes.add("low_volatility")

    up_spike_thr = max(ABS_BAR_SPIKE, _percentile_threshold(past_max_up, 90, ABS_BAR_SPIKE))
    down_spike_thr = max(ABS_BAR_SPIKE, _percentile_threshold(past_max_down, 90, ABS_BAR_SPIKE))
    if features.max_bar_up >= up_spike_thr:
        regimes.add("sudden_upward_spike")
    if abs(features.max_bar_down) >= down_spike_thr:
        regimes.add("sudden_downward_spike")

    if features.prev_close and features.prev_close > 0:
        gap = (features.open - features.prev_close) / features.prev_close
        if gap >= ABS_GAP_RETURN:
            regimes.add("gap_up")
        if gap <= -ABS_GAP_RETURN:
            regimes.add("gap_down")

    if features.sign_changes >= 6:
        regimes.add("choppy_whipsaw")
    if features.false_breakout:
        regimes.add("false_breakout")

    if features.opening_range_pct >= _percentile_threshold(past_open_rng, 70, 0.004):
        regimes.add("opening_volatility")
    if features.midday_range_pct <= _percentile_threshold(past_mid_rng, 30, 0.0015):
        regimes.add("midday_low_momentum")
    if features.late_range_pct >= _percentile_threshold(past_late_rng, 70, 0.003):
        regimes.add("late_session_volatility")

    return regimes


def build_regime_labels(candles: pd.DataFrame, s2_cfg: dict[str, Any]) -> dict[date, set[str]]:
    """Map each session date to one or more regime tags (thresholds use prior days only)."""
    if candles is None or candles.empty:
        return {}
    frame = candles.sort_values("timestamp").reset_index(drop=True)
    frame["_day"] = _ts_local(frame["timestamp"]).dt.date
    labels: dict[date, set[str]] = {}
    history: list[DayFeatures] = []
    prev_close: float | None = None
    for day, group in frame.groupby("_day", sort=True):
        features = _day_features(group, prev_close, s2_cfg)
        labels[day] = classify_day(features, history)
        history.append(features)
        prev_close = features.close
    return labels
