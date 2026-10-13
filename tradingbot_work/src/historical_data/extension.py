"""Extend stored NIFTY historical data with real Upstox candles (research only, PAPER mode).

Underlying 5m candles and the exact option contracts S2 needs for the new
dates are downloaded read-only, validated and merged append-only into the
existing Parquet store. Nothing is interpolated, estimated or synthesised.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from .downloader import FATAL_CODES, month_chunks
from .models import HistoricalCandle, HistoricalDataError, candles_to_frame
from .option_coverage import (
    RequiredOptionContract,
    extract_required_contracts,
    has_stored_option_data,
    research_signals,
    signal_has_premium_coverage,
)
from .storage import ParquetCandleStore
from .upstox_client import FORBIDDEN_TOKEN_ENV_NAMES, TOKEN_ENV_NAMES, configured_token, token_source
from .validation import SESSION_END, validate_candles

IST = ZoneInfo("Asia/Kolkata")
IST_OFFSET = timedelta(hours=5, minutes=30)
SAFE_TOKEN_ENV_NAMES = TOKEN_ENV_NAMES
FORBIDDEN_TOKEN_ENV = FORBIDDEN_TOKEN_ENV_NAMES[0]
SIGNAL_HISTORY_DAYS = 30
REPORT_STEM = "NIFTY_historical_extension"

safe_token_source = token_source
safe_token = configured_token


def latest_completed_day(now: datetime) -> date:
    """Most recent weekday whose regular session has closed (exchange holidays are found from data)."""
    local = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
    day = local.date() if local.time() >= SESSION_END else local.date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def resolve_range(start: date, end: date, now: datetime) -> tuple[date, date, list[str]]:
    """Clamp ``end`` to the latest completed session; partial sessions are never stored."""
    if start > end:
        raise HistoricalDataError("INVALID_RANGE", f"start {start} is after end {end}")
    notes: list[str] = []
    latest = latest_completed_day(now)
    effective_end = min(end, latest)
    if effective_end < end:
        notes.append(f"End date clamped from {end} to {effective_end}: later sessions are not complete yet.")
    return start, effective_end, notes


def weekdays(start: date, end: date) -> list[date]:
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def non_ist_timestamps(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    ts = pd.to_datetime(frame["timestamp"])
    if ts.dt.tz is None:
        return len(ts)
    return int(sum(1 for value in ts if value.utcoffset() != IST_OFFSET))


def append_only_write(store: ParquetCandleStore, kind: str, underlying: str, instrument_key: str,
                      frame: pd.DataFrame, *, expiry: date | None = None) -> dict[str, int]:
    """Write only timestamps not already stored, so existing rows are never replaced."""
    existing = store.read(kind, underlying, instrument_key, expiry=expiry)
    before = len(existing)
    if frame.empty:
        return {"existing_rows_before": before, "rows_written": 0, "rows_skipped_existing": 0,
                "existing_rows_after": before}
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.tz_convert(IST)
    known = set(existing["timestamp"]) if not existing.empty else set()
    overlap = data["timestamp"].isin(known)
    fresh = data[~overlap]
    written = store.write(kind, underlying, instrument_key, fresh, expiry=expiry) if not fresh.empty else 0
    after = len(store.read(kind, underlying, instrument_key, expiry=expiry))
    if after != before + written:
        raise HistoricalDataError("STORE_INTEGRITY", f"{instrument_key}: row count {after} != {before} + {written}")
    return {"existing_rows_before": before, "rows_written": int(written),
            "rows_skipped_existing": int(overlap.sum()), "existing_rows_after": after}


def _limitation(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **extra}


@dataclass
class UnderlyingResult:
    instrument_key: str
    requested_start: str
    requested_end: str
    effective_end: str
    candles_downloaded: int = 0
    candles_written: int = 0
    candles_skipped_existing: int = 0
    existing_rows_before: int = 0
    existing_rows_after: int = 0
    non_ist_timestamps: int = 0
    duplicates_removed: int = 0
    was_sorted: bool = True
    missing_candles: int = 0
    missing_by_day: dict[str, int] = field(default_factory=dict)
    validation_errors: dict[str, int] = field(default_factory=dict)
    rejected_rows: int = 0
    new_trading_days: list[str] = field(default_factory=list)
    weekdays_without_data: list[str] = field(default_factory=list)
    actual_first: str | None = None
    actual_last: str | None = None
    latest_stored_candle: str | None = None
    chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OptionResult:
    total_signals: int = 0
    contracts_required: int = 0
    contracts_already_stored: int = 0
    contracts_downloaded: int = 0
    contracts_unavailable: int = 0
    signals_covered: int = 0
    coverage_percent: float | None = None
    candles_written: int = 0
    validation_errors: dict[str, int] = field(default_factory=dict)
    required: list[dict[str, Any]] = field(default_factory=list)
    downloaded: list[dict[str, Any]] = field(default_factory=list)
    unavailable: list[dict[str, Any]] = field(default_factory=list)
    resolution_errors: list[dict[str, Any]] = field(default_factory=list)


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for code, count in source.items():
        target[code] = target.get(code, 0) + int(count)


def download_underlying(client, store: ParquetCandleStore, *, underlying: str, instrument_key: str,
                        start: date, end: date, requested_end: date, interval: int,
                        limitations: list[dict[str, Any]]) -> UnderlyingResult:
    result = UnderlyingResult(instrument_key=instrument_key, requested_start=start.isoformat(),
                              requested_end=requested_end.isoformat(), effective_end=end.isoformat())
    candles: list[HistoricalCandle] = []
    if start <= end:
        for chunk_start, chunk_end in month_chunks(start, end):
            try:
                rows = client.candles(instrument_key, chunk_start, chunk_end, interval=interval)
            except HistoricalDataError as exc:
                result.chunks.append({"start": chunk_start.isoformat(), "end": chunk_end.isoformat(),
                                      "status": "FAILED", "code": exc.code})
                limitations.append(_limitation(exc.code, exc.message, scope="underlying",
                                               window=f"{chunk_start}..{chunk_end}"))
                if exc.code in FATAL_CODES:
                    raise
                continue
            result.chunks.append({"start": chunk_start.isoformat(), "end": chunk_end.isoformat(),
                                  "status": "OK" if rows else "EMPTY", "rows": len(rows)})
            candles.extend(rows)

    frame = candles_to_frame(candles)
    if not frame.empty:
        dates = frame["timestamp"].dt.date
        frame = frame[(dates >= start) & (dates <= end)].reset_index(drop=True)
    result.candles_downloaded = len(frame)
    result.non_ist_timestamps = non_ist_timestamps(frame)
    clean, report = validate_candles(frame, interval=interval)
    result.duplicates_removed = report.duplicates_removed
    result.was_sorted = report.was_sorted
    result.missing_candles = report.missing_candles
    result.missing_by_day = dict(report.missing_by_day)
    result.validation_errors = {k: v for k, v in report.issues.items()
                                if k not in ("DUPLICATE_TIMESTAMP", "UNSORTED_TIMESTAMPS")}
    result.rejected_rows = report.rejected_rows
    if report.outside_session_rows:
        result.validation_errors["OUTSIDE_SESSION"] = report.outside_session_rows

    stats = append_only_write(store, "underlying", underlying, instrument_key, clean)
    result.candles_written = stats["rows_written"]
    result.candles_skipped_existing = stats["rows_skipped_existing"]
    result.existing_rows_before = stats["existing_rows_before"]
    result.existing_rows_after = stats["existing_rows_after"]

    present = sorted({d.isoformat() for d in clean["timestamp"].dt.date}) if not clean.empty else []
    result.new_trading_days = present
    result.weekdays_without_data = [d.isoformat() for d in weekdays(start, end) if d.isoformat() not in present]
    if not clean.empty:
        result.actual_first = clean["timestamp"].min().isoformat()
        result.actual_last = clean["timestamp"].max().isoformat()
    stored = store.read("underlying", underlying, instrument_key)
    if not stored.empty:
        result.latest_stored_candle = stored["timestamp"].max().isoformat()
    return result


def _expired_key(client, underlying_key: str, contract: RequiredOptionContract) -> str:
    rows = client.expired_option_contracts(underlying_key, contract.expiry)
    for row in rows:
        strike = row.get("strike_price", row.get("strike"))
        kind = str(row.get("instrument_type") or row.get("option_type") or "").upper()
        if strike is not None and float(strike) == float(contract.strike) and kind == contract.option_type:
            return str(row["instrument_key"])
    raise HistoricalDataError("NOT_FOUND", f"Expired contract {contract.trading_symbol} not listed by Upstox")


def download_option_contract(client, store: ParquetCandleStore, contract: RequiredOptionContract, *,
                             underlying_key: str, start: date, end: date, interval: int,
                             today: date) -> dict[str, Any]:
    """Fetch real premium candles for one required contract and merge them append-only."""
    expired = contract.expiry < today
    key = _expired_key(client, underlying_key, contract) if expired else contract.instrument_key
    window_end = min(end, contract.expiry)
    candles: list[HistoricalCandle] = []
    for chunk_start, chunk_end in month_chunks(start, window_end):
        if expired:
            candles.extend(client.expired_candles(key, chunk_start, chunk_end, interval=interval))
        else:
            candles.extend(client.candles(key, chunk_start, chunk_end, interval=interval))
    # Stored under the listed instrument key so existing readers find it.
    for i, candle in enumerate(candles):
        if candle.instrument_key != contract.instrument_key:
            candles[i] = HistoricalCandle(**{**candle.as_dict(), "instrument_key": contract.instrument_key})
    frame = candles_to_frame(candles, underlying=contract.underlying, expiry=contract.expiry.isoformat(),
                             strike=contract.strike, option_type=contract.option_type)
    if frame.empty:
        raise HistoricalDataError("NO_DATA", f"Upstox returned no candles for {contract.trading_symbol}")
    clean, report = validate_candles(frame, interval=interval, expiry=contract.expiry,
                                     option_type=contract.option_type)
    if clean.empty:
        raise HistoricalDataError("OPTION_DATA_INVALID", f"{contract.trading_symbol} failed validation")
    stats = append_only_write(store, "options", contract.underlying, contract.instrument_key, clean,
                              expiry=contract.expiry)
    return {"instrument_key": contract.instrument_key, "trading_symbol": contract.trading_symbol,
            "source": "EXPIRED" if expired else "ACTIVE", "candles": len(clean),
            "validation": report.as_dict(), **stats}


def extend_options(client, store: ParquetCandleStore, *, underlying: str, underlying_key: str,
                   strategy_id: str, research_config, strategy_config,
                   resolver_factory: Callable[[date], Any], start: date, end: date, interval: int,
                   today: date, limitations: list[dict[str, Any]]) -> OptionResult:
    from src.session.session_manager import SessionManager

    result = OptionResult()
    history = store.read("underlying", underlying, underlying_key,
                         start=start - timedelta(days=SIGNAL_HISTORY_DAYS), end=end)
    if history.empty or history[history["timestamp"].dt.date >= start].empty:
        return result
    history = history.copy()
    history["symbol"] = underlying.upper()
    session = SessionManager(strategy_config.session)
    signals = research_signals(history, underlying, strategy_id, research_config, start, end, session)
    required, errors = extract_required_contracts(signals, resolver_factory, research_config, start, end)
    result.total_signals = len(signals)
    result.contracts_required = len(required)
    result.required = [c.as_dict() for c in required]
    result.resolution_errors = errors
    for err in errors:
        limitations.append(_limitation(err.get("reason", "RESOLUTION_FAILED"), err.get("message", ""),
                                       scope="option_resolution", signal_time=err.get("signal_time")))

    for contract in required:
        if has_stored_option_data(store, contract, start, end):
            result.contracts_already_stored += 1
            continue
        try:
            info = download_option_contract(client, store, contract, underlying_key=underlying_key,
                                            start=start, end=end, interval=interval, today=today)
        except HistoricalDataError as exc:
            result.contracts_unavailable += 1
            result.unavailable.append({"instrument_key": contract.instrument_key,
                                       "trading_symbol": contract.trading_symbol,
                                       "expiry": contract.expiry.isoformat(), "code": exc.code,
                                       "message": exc.message})
            limitations.append(_limitation(exc.code, exc.message, scope="option",
                                           contract=contract.trading_symbol))
            if exc.code in FATAL_CODES:
                raise
            continue
        result.contracts_downloaded += 1
        result.candles_written += info["rows_written"]
        _merge_counts(result.validation_errors, info["validation"].get("issues", {}))
        result.downloaded.append(info)

    entry_timing = str(research_config.option_mapping.get("entry_timing", "next_open"))
    for signal in signals:
        contract = next((c for c in required if signal.timestamp.isoformat() in c.signal_times), None)
        if contract is not None and signal_has_premium_coverage(store, contract, signal,
                                                                entry_timing=entry_timing)[0]:
            result.signals_covered += 1
    if signals:
        result.coverage_percent = round(100.0 * result.signals_covered / len(signals), 2)
    return result


def build_report(*, run_id: str, underlying: str, strategy_id: str, token_source: str | None,
                 requested: tuple[date, date], effective_end: date, underlying_result: UnderlyingResult,
                 option_result: OptionResult, limitations: list[dict[str, Any]], notes: list[str],
                 generated_at: datetime) -> dict[str, Any]:
    u, o = underlying_result, option_result
    data_available = bool(u.new_trading_days)
    if not data_available:
        limitations = limitations + [_limitation(
            "NO_NEW_DATA",
            f"Upstox returned no completed {underlying} candles between {requested[0]} and {effective_end}. "
            "No synthetic data was generated.",
            weekdays_without_data=u.weekdays_without_data)]
    return {
        "run_id": run_id,
        "generated_at": generated_at.isoformat(),
        "underlying": underlying.upper(),
        "strategy_id": strategy_id,
        "mode": "PAPER research only - no broker orders",
        "token_source": token_source,
        "requested_range": {"start": requested[0].isoformat(), "end": requested[1].isoformat()},
        "effective_range": {"start": requested[0].isoformat(), "end": effective_end.isoformat()},
        "actual_data_range": {"first": u.actual_first, "last": u.actual_last},
        "data_available": data_available,
        "underlying": asdict(u),
        "options": asdict(o),
        "summary": {
            "latest_candle": u.latest_stored_candle,
            "new_trading_days": len(u.new_trading_days),
            "candles_downloaded": u.candles_downloaded,
            "candles_written": u.candles_written,
            "missing_candles": u.missing_candles,
            "duplicate_candles": u.duplicates_removed,
            "underlying_validation_errors": sum(u.validation_errors.values()),
            "option_contracts_required": o.contracts_required,
            "option_contracts_downloaded": o.contracts_downloaded,
            "option_contracts_already_stored": o.contracts_already_stored,
            "option_contracts_unavailable": o.contracts_unavailable,
            "option_validation_errors": sum(o.validation_errors.values()),
            "coverage_percent": o.coverage_percent,
        },
        "notes": notes,
        "limitations": limitations,
    }


def _md_table(rows: Iterable[tuple[str, Any]]) -> list[str]:
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return lines


def write_extension_reports(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {ext: out_dir / f"{REPORT_STEM}.{ext}" for ext in ("json", "csv", "md")}
    paths["json"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    s = report["summary"]
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["requested_start", report["requested_range"]["start"]])
        writer.writerow(["requested_end", report["requested_range"]["end"]])
        writer.writerow(["effective_end", report["effective_range"]["end"]])
        writer.writerow(["actual_first", report["actual_data_range"]["first"]])
        writer.writerow(["actual_last", report["actual_data_range"]["last"]])
        writer.writerow(["data_available", report["data_available"]])
        for key, value in s.items():
            writer.writerow([key, value])
        writer.writerow(["limitations", "; ".join(f"{l['code']}: {l['message']}" for l in report["limitations"])])

    lines = [f"# {report['underlying']} historical data extension", "",
             f"Run `{report['run_id']}` at {report['generated_at']} ({report['mode']}).",
             f"Token source variable: `{report['token_source']}` (value never logged).", "",
             "## Ranges", ""]
    lines += _md_table([
        ("Requested", f"{report['requested_range']['start']} .. {report['requested_range']['end']}"),
        ("Effective (completed sessions)", f"{report['effective_range']['start']} .. {report['effective_range']['end']}"),
        ("Actual data", f"{report['actual_data_range']['first']} .. {report['actual_data_range']['last']}"),
        ("Latest stored candle", s["latest_candle"]),
    ])
    lines += ["", "## Underlying candles", ""]
    lines += _md_table([
        ("New trading days", s["new_trading_days"]),
        ("Candles downloaded", s["candles_downloaded"]),
        ("Candles written (new)", s["candles_written"]),
        ("Missing candles", s["missing_candles"]),
        ("Duplicate candles removed", s["duplicate_candles"]),
        ("Validation errors", s["underlying_validation_errors"]),
        ("Weekdays without data", ", ".join(report["underlying"]["weekdays_without_data"]) or "-"),
    ])
    lines += ["", "## S2 option contracts", ""]
    lines += _md_table([
        ("Signals in range", report["options"]["total_signals"]),
        ("Contracts required", s["option_contracts_required"]),
        ("Already stored", s["option_contracts_already_stored"]),
        ("Downloaded", s["option_contracts_downloaded"]),
        ("Unavailable", s["option_contracts_unavailable"]),
        ("Option validation errors", s["option_validation_errors"]),
        ("Signal premium coverage %", s["coverage_percent"] if s["coverage_percent"] is not None else "n/a"),
    ])
    lines += ["", "## Notes and limitations", ""]
    lines += [f"- {n}" for n in report["notes"]]
    lines += [f"- **{l['code']}**: {l['message']}" for l in report["limitations"]] or ["- None"]
    paths["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
