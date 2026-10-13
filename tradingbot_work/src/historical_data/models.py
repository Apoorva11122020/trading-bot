"""Normalised historical candle models.

Upstox returns candles as ``[timestamp, open, high, low, close, volume, oi]``.
Rows are converted 1:1; nothing is interpolated or estimated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

IST = "Asia/Kolkata"
CANDLE_COLUMNS = ["timestamp", "instrument_key", "open", "high", "low", "close", "volume", "open_interest"]
OPTION_COLUMNS = CANDLE_COLUMNS + ["underlying", "expiry", "strike", "option_type"]


class HistoricalDataError(RuntimeError):
    """A historical-data request or validation failed. ``code`` is machine readable."""

    def __init__(self, code: str, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "status": self.status}


@dataclass(frozen=True)
class HistoricalCandle:
    timestamp: datetime
    instrument_key: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptionHistoricalCandle(HistoricalCandle):
    underlying: str = ""
    expiry: date | None = None
    strike: float | None = None
    option_type: str | None = None


def parse_candle_rows(rows: Iterable[Any], instrument_key: str) -> list[HistoricalCandle]:
    """Convert raw Upstox candle arrays. Malformed rows raise; they are never skipped silently."""
    candles: list[HistoricalCandle] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise HistoricalDataError("MALFORMED_CANDLE", f"{instrument_key}: candle #{index} has unexpected shape")
        try:
            ts = pd.Timestamp(row[0])
            if ts.tzinfo is None:
                ts = ts.tz_localize(IST)
            candles.append(HistoricalCandle(
                timestamp=ts.tz_convert(IST).to_pydatetime(),
                instrument_key=instrument_key,
                open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]),
                volume=float(row[5]),
                open_interest=float(row[6]) if len(row) > 6 and row[6] is not None else None,
            ))
        except (TypeError, ValueError) as exc:
            raise HistoricalDataError(
                "MALFORMED_CANDLE", f"{instrument_key}: candle #{index} could not be parsed ({exc})") from exc
    return candles


def candles_to_frame(candles: Iterable[HistoricalCandle], **extra: Any) -> pd.DataFrame:
    rows = [c.as_dict() for c in candles]
    frame = pd.DataFrame(rows, columns=CANDLE_COLUMNS) if not rows else pd.DataFrame(rows)[CANDLE_COLUMNS]
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST)
    for key, value in extra.items():
        frame[key] = value
    return frame.sort_values("timestamp").reset_index(drop=True) if not frame.empty else frame
