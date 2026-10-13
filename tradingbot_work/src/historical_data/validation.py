"""Candle validation. Problems are reported; missing candles are never filled."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, time
from typing import Any

import pandas as pd

from .models import IST

SESSION_START = time(9, 15)
SESSION_END = time(15, 30)


@dataclass
class ValidationReport:
    rows: int = 0
    valid_rows: int = 0
    rejected_rows: int = 0
    duplicates_removed: int = 0
    outside_session_rows: int = 0
    was_sorted: bool = True
    issues: dict[str, int] = field(default_factory=dict)
    trading_days: int = 0
    expected_candles: int = 0
    missing_candles: int = 0
    missing_by_day: dict[str, int] = field(default_factory=dict)
    first: str | None = None
    last: str | None = None

    @property
    def ok(self) -> bool:
        return self.rejected_rows == 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def expected_slots(day: date, interval: int = 5, start: time = SESSION_START, end: time = SESSION_END) -> list[pd.Timestamp]:
    first = pd.Timestamp.combine(day, start).tz_localize(IST)
    last = pd.Timestamp.combine(day, end).tz_localize(IST) - pd.Timedelta(minutes=interval)
    return list(pd.date_range(first, last, freq=f"{int(interval)}min"))


def validate_candles(
    frame: pd.DataFrame,
    *,
    interval: int = 5,
    expiry: date | None = None,
    option_type: str | None = None,
    session_start: time = SESSION_START,
    session_end: time = SESSION_END,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Return (clean_frame, report).

    clean_frame is sorted, de-duplicated and excludes rows that fail price
    checks. Rejected rows are counted in the report, never silently dropped.
    Missing candles are counted against the regular session grid of each
    trading day present in the data; they are not filled.
    """
    report = ValidationReport(rows=len(frame))
    if frame.empty:
        return frame.copy(), report

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if data["timestamp"].dt.tz is None:
        data["timestamp"] = data["timestamp"].dt.tz_localize(IST)
    data["timestamp"] = data["timestamp"].dt.tz_convert(IST)
    issues: dict[str, int] = {}

    report.was_sorted = bool(data["timestamp"].is_monotonic_increasing)
    if not report.was_sorted:
        issues["UNSORTED_TIMESTAMPS"] = 1
    data = data.sort_values("timestamp", kind="stable")
    dup = data["timestamp"].duplicated(keep="last")
    if dup.any():
        issues["DUPLICATE_TIMESTAMP"] = int(dup.sum())
        report.duplicates_removed = int(dup.sum())
        data = data[~dup]

    times = data["timestamp"].dt.time
    outside = (times < session_start) | (times >= session_end)
    report.outside_session_rows = int(outside.sum())
    # Exclude out-of-session rows before price checks so they cannot inflate
    # rejected-row counts or contaminate issue metrics.
    data = data[~outside].copy()

    checks = {
        "NON_POSITIVE_PRICE": (data[["open", "high", "low", "close"]] <= 0).any(axis=1),
        "INVALID_OHLC": (data["high"] < data[["open", "close", "low"]].max(axis=1))
        | (data["low"] > data[["open", "close", "high"]].min(axis=1)),
        "NEGATIVE_VOLUME": data["volume"] < 0,
        "MISSING_PRICE": data[["open", "high", "low", "close"]].isna().any(axis=1),
        "MISSING_VOLUME": data["volume"].isna(),
    }
    if "open_interest" in data:
        checks["NEGATIVE_OPEN_INTEREST"] = data["open_interest"].fillna(0) < 0
    if expiry is not None:
        checks["AFTER_EXPIRY"] = data["timestamp"].dt.date > expiry
    bad = pd.Series(False, index=data.index)
    if option_type is not None and str(option_type).upper() not in ("CE", "PE"):
        issues["INVALID_OPTION_TYPE"] = len(data)
        bad[:] = True
    for code, mask in checks.items():
        count = int(mask.sum())
        if count:
            issues[code] = count
            bad |= mask

    clean = data[~bad].reset_index(drop=True)
    report.rejected_rows = int(bad.sum())
    report.valid_rows = len(clean)
    report.issues = issues

    if not clean.empty:
        report.first = clean["timestamp"].iloc[0].isoformat()
        report.last = clean["timestamp"].iloc[-1].isoformat()
        present = set(clean["timestamp"])
        days = sorted(set(clean["timestamp"].dt.date))
        report.trading_days = len(days)
        for day in days:
            slots = expected_slots(day, interval, session_start, session_end)
            report.expected_candles += len(slots)
            missing = sum(1 for slot in slots if slot not in present)
            if missing:
                report.missing_by_day[day.isoformat()] = missing
                report.missing_candles += missing
    return clean, report
