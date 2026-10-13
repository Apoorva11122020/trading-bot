"""Point-in-time market-regime filter for S2 research signals.

At a signal bar timestamp T only bars with timestamp <= T are used (the signal
bar is complete when S2 fires on its close). Thresholds come from prior
sessions only, measured at the same time of day so partial-day ranges are
compared like for like.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.backtest.regime_classifier import ABS_BAR_SPIKE, LOOKBACK_DAYS, MIN_HISTORY_DAYS, REGIME_CATALOG
from src.backtest.research_strategies import ResearchSignal

IST = ZoneInfo("Asia/Kolkata")

AVOID_REGIMES: tuple[str, ...] = ("range_bound", "low_volatility", "sudden_upward_spike")
ALLOW_REGIMES: tuple[str, ...] = (
    "strong_bearish_trend",
    "high_volatility",
    "gap_up",
    "gap_down",
    "choppy_whipsaw",
    "false_breakout",
    "opening_volatility",
    "midday_low_momentum",
    "late_session_volatility",
)
RANGE_BOUND_PERCENTILE = 25
LOW_VOLATILITY_PERCENTILE = 20
SPIKE_PERCENTILE = 90


@dataclass
class _DayBars:
    minutes: np.ndarray
    cum_high: np.ndarray
    cum_low: np.ndarray
    closes: np.ndarray
    cum_max_bar_up: np.ndarray

    def index_at(self, minute_of_day: int) -> int:
        return int(np.searchsorted(self.minutes, minute_of_day, side="right")) - 1

    def range_pct_at(self, minute_of_day: int) -> float | None:
        idx = self.index_at(minute_of_day)
        if idx < 0 or self.closes[idx] <= 0:
            return None
        return float((self.cum_high[idx] - self.cum_low[idx]) / self.closes[idx])

    def max_bar_up_at(self, minute_of_day: int) -> float | None:
        idx = self.index_at(minute_of_day)
        if idx < 0:
            return None
        return float(self.cum_max_bar_up[idx])

    @property
    def full_day_max_bar_up(self) -> float:
        return float(self.cum_max_bar_up[-1]) if len(self.cum_max_bar_up) else 0.0


def _minute_of_day(ts: datetime) -> int:
    local = ts.astimezone(IST)
    return local.hour * 60 + local.minute


class PointInTimeRegimeClassifier:
    """Detects the avoided regimes at a timestamp without using later data."""

    def __init__(
        self,
        candles: pd.DataFrame,
        *,
        lookback_days: int = LOOKBACK_DAYS,
        min_history_days: int = MIN_HISTORY_DAYS,
    ) -> None:
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days
        self._days: dict[date, _DayBars] = {}
        if candles is None or candles.empty:
            self._ordered: list[date] = []
            return
        frame = candles.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(frame["timestamp"])
        ts = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
        frame = frame.assign(_ts=ts, _day=ts.dt.date, _minute=ts.dt.hour * 60 + ts.dt.minute)
        for day, group in frame.groupby("_day", sort=True):
            closes = group["close"].astype(float).to_numpy()
            bar_up = np.zeros(len(closes))
            if len(closes) > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rets = np.where(closes[:-1] > 0, closes[1:] / closes[:-1] - 1.0, 0.0)
                bar_up[1:] = np.maximum.accumulate(np.maximum(rets, 0.0))
            self._days[day] = _DayBars(
                minutes=group["_minute"].to_numpy(),
                cum_high=np.maximum.accumulate(group["high"].astype(float).to_numpy()),
                cum_low=np.minimum.accumulate(group["low"].astype(float).to_numpy()),
                closes=closes,
                cum_max_bar_up=bar_up,
            )
        self._ordered = sorted(self._days)

    def _prior_days(self, day: date) -> list[date]:
        prior = [d for d in self._ordered if d < day]
        return prior[-self.lookback_days:]

    def features_at(self, ts: datetime) -> dict[str, Any]:
        day = ts.astimezone(IST).date()
        bars = self._days.get(day)
        minute = _minute_of_day(ts)
        if bars is None:
            return {"day": day.isoformat(), "range_pct": None, "max_bar_up": None, "history_days": 0}
        prior = self._prior_days(day)
        return {
            "day": day.isoformat(),
            "range_pct": bars.range_pct_at(minute),
            "max_bar_up": bars.max_bar_up_at(minute),
            "history_days": len(prior),
        }

    def regimes_at(self, ts: datetime) -> set[str]:
        day = ts.astimezone(IST).date()
        bars = self._days.get(day)
        if bars is None:
            return set()
        minute = _minute_of_day(ts)
        prior = self._prior_days(day)
        tags: set[str] = set()

        range_now = bars.range_pct_at(minute)
        if range_now is not None and len(prior) >= self.min_history_days:
            same_time = [r for r in (self._days[d].range_pct_at(minute) for d in prior) if r is not None]
            if len(same_time) >= self.min_history_days:
                if range_now <= float(np.percentile(same_time, RANGE_BOUND_PERCENTILE)):
                    tags.add("range_bound")
                if range_now <= float(np.percentile(same_time, LOW_VOLATILITY_PERCENTILE)):
                    tags.add("low_volatility")

        spike_now = bars.max_bar_up_at(minute)
        if spike_now is not None:
            threshold = ABS_BAR_SPIKE
            if len(prior) >= self.min_history_days:
                past = [self._days[d].full_day_max_bar_up for d in prior]
                threshold = max(ABS_BAR_SPIKE, float(np.percentile(past, SPIKE_PERCENTILE)))
            if spike_now >= threshold:
                tags.add("sudden_upward_spike")
        return tags


@dataclass
class RegimeSignalFilter:
    """Blocks S2 signals whose point-in-time regime is in the avoid list."""

    classifier: PointInTimeRegimeClassifier
    avoid: tuple[str, ...] = AVOID_REGIMES
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        known = {rid for rid, _ in REGIME_CATALOG}
        unknown = set(self.avoid) - known
        if unknown:
            raise ValueError(f"Unknown regimes: {sorted(unknown)}")

    def __call__(self, signal: ResearchSignal) -> list[str]:
        tags = self.classifier.regimes_at(signal.timestamp)
        blocked = sorted(tags & set(self.avoid))
        self.decisions.append({
            "signal_time": signal.timestamp.isoformat(),
            "direction": signal.direction,
            "regimes": sorted(tags),
            "blocked_by": blocked,
        })
        return blocked


def validate_hypothesis(regime_report: dict[str, Any], avoid: Iterable[str], allow: Iterable[str]) -> dict[str, Any]:
    """Cross-check the filter hypothesis against the completed regime study."""
    rows = {row["regime_id"]: row for row in regime_report.get("regimes", [])}
    missing = [rid for rid in list(avoid) + list(allow) if rid not in rows]
    if missing:
        raise ValueError(f"Regimes missing from regime study: {missing}")
    return {rid: rows[rid] for rid in avoid}
