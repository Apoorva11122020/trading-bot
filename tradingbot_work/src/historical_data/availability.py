"""Discover the historical date range actually available from Upstox V3.

Index (NSE_INDEX) minute candles are available from January 2022.
Option candles are only returned for **currently listed** contracts and
only from their listing date forward. Expired contracts require the Upstox
Plus expired-instruments API (not available with the sandbox token).
"""
from __future__ import annotations

import calendar
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

from src.historical_data.models import HistoricalDataError
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.options.instrument_resolver import OptionInstrumentResolver
from src.options.selection import atm_strike, select_expiry

EARLIEST_INDEX_MINUTE = date(2022, 1, 1)


@dataclass
class AvailabilityRange:
    underlying: str
    requested_from: date
    requested_to: date
    index_available_from: date | None
    index_available_to: date | None
    option_available_from: date | None
    option_available_to: date | None
    research_from: date | None
    research_to: date | None
    limitation: str
    index_candle_probe: dict[str, Any] | None = None
    option_candle_probe: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "requested_from", "requested_to", "index_available_from", "index_available_to",
            "option_available_from", "option_available_to", "research_from", "research_to",
        ):
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _probe_index_month(client: UpstoxHistoricalClient, instrument_key: str, year: int, month: int) -> int:
    start = date(year, month, 1)
    end = min(_month_end(year, month), date.today())
    if start > end:
        return 0
    try:
        return len(client.candles(instrument_key, start, end, interval=5))
    except HistoricalDataError:
        return 0


def discover_index_range(client: UpstoxHistoricalClient, instrument_key: str) -> tuple[date | None, date | None]:
    today = date.today()
    first_month = None
    for year in range(EARLIEST_INDEX_MINUTE.year, today.year + 1):
        for month in range(1, 13):
            if date(year, month, 1) > today:
                break
            if _probe_index_month(client, instrument_key, year, month) > 0:
                first_month = date(year, month, 1)
                break
        if first_month:
            break
    if not first_month:
        return None, None
    last_month = None
    for year in range(today.year, first_month.year - 1, -1):
        months = range(12, 0, -1) if year != today.year else range(today.month, 0, -1)
        for month in months:
            if date(year, month, 1) < first_month:
                continue
            if _probe_index_month(client, instrument_key, year, month) > 0:
                last_month = min(_month_end(year, month), today)
                return first_month, last_month
    return first_month, today


def _probe_option_week(
    client: UpstoxHistoricalClient,
    resolver: OptionInstrumentResolver,
    underlying: str,
    sample_day: date,
) -> dict[str, Any]:
    resolver.today = lambda: sample_day
    try:
        expiry = select_expiry(resolver.expiries(underlying), "nearest")
        strike = atm_strike(resolver.strikes(underlying, expiry, "CE"), 100000.0)
        contract = resolver.resolve(underlying, expiry, strike, "CE")
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
    start = sample_day
    end = min(sample_day + timedelta(days=4), date.today())
    try:
        count = len(client.candles(contract.instrument_key, start, end, interval=5))
    except HistoricalDataError as exc:
        return {"ok": False, "instrument_key": contract.instrument_key, "code": exc.code, "message": exc.message}
    return {
        "ok": count > 0,
        "candles": count,
        "instrument_key": contract.instrument_key,
        "trading_symbol": contract.trading_symbol,
        "expiry": contract.expiry.isoformat(),
        "strike": contract.strike,
        "sample_day": sample_day.isoformat(),
    }


def discover_option_range(
    client: UpstoxHistoricalClient,
    resolver: OptionInstrumentResolver,
    underlying: str,
) -> tuple[date | None, date | None, dict[str, Any]]:
    today = date.today()
    first = None
    probe: dict[str, Any] = {}
    for year in range(EARLIEST_INDEX_MINUTE.year, today.year + 1):
        for month in range(1, 13):
            sample = date(year, month, 15)
            if sample > today:
                break
            result = _probe_option_week(client, resolver, underlying.upper(), sample)
            if result.get("ok"):
                first = date(year, month, 1)
                probe = result
                break
        if first:
            break
    if not first:
        return None, None, {"ok": False, "reason": "No listed option contract returned V3 candles in any probed month"}
    # Latest day: today (active contracts trade until expiry)
    return first, today, probe


def discover_research_range(
    client: UpstoxHistoricalClient,
    underlying: str,
    instrument_key: str,
    resolver: OptionInstrumentResolver,
    *,
    requested_from: date | None = None,
    requested_to: date | None = None,
) -> AvailabilityRange:
    today = date.today()
    req_from = requested_from or EARLIEST_INDEX_MINUTE
    req_to = min(requested_to or today, today)
    index_from, index_to = discover_index_range(client, instrument_key)
    option_from, option_to, option_probe = discover_option_range(client, resolver, underlying.upper())

    limitation = (
        "Upstox V3 returns index candles from January 2022. Option candles are available only for "
        "currently listed contracts from their listing date; expired contracts require Upstox Plus "
        "and are not used. Research range is the overlap of index data and retrievable option premiums."
    )
    research_from = research_to = None
    if index_from and option_from:
        research_from = max(index_from, option_from, req_from)
        research_to = min(index_to or today, option_to or today, req_to)
        if research_from > research_to:
            research_from = research_to = None
            limitation += " No overlapping index and option window was found for the requested range."

    return AvailabilityRange(
        underlying=underlying.upper(),
        requested_from=req_from,
        requested_to=req_to,
        index_available_from=index_from,
        index_available_to=index_to,
        option_available_from=option_from,
        option_available_to=option_to,
        research_from=research_from,
        research_to=research_to,
        limitation=limitation,
        option_candle_probe=option_probe,
    )
