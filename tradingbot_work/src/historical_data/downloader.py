"""Month-by-month historical downloader with resume and validation.

Upstox serves 1-15 minute candles in windows of at most one month, so every
request is a calendar-month chunk. A chunk that completed (and is not the
current, still-growing month) is skipped on the next run. Every API error is
recorded in the job result; credential errors stop the job immediately.
"""
from __future__ import annotations

import calendar
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Iterable

import pandas as pd

from src.logging.logger import log_event

from .contracts import ACTIVE, ContractCatalog, HistoricalContract
from .models import HistoricalDataError, candles_to_frame
from .storage import ParquetCandleStore
from .validation import validate_candles

logger = logging.getLogger("historical_data.downloader")
FATAL_CODES = {"CREDENTIALS_MISSING", "AUTH_FAILED"}
JOB_PREFIX = "historical_job:"
CHUNK_PREFIX = "historical_chunk:"
EARLIEST_MINUTE_DATA = date(2022, 1, 1)


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("end date must be on or after start date")
    chunks = []
    cursor = start
    while cursor <= end:
        last_day = date(cursor.year, cursor.month, calendar.monthrange(cursor.year, cursor.month)[1])
        chunks.append((cursor, min(last_day, end)))
        cursor = last_day + timedelta(days=1)
    return chunks


class MetadataStore:
    """Job/chunk metadata in the application database (system_state), or memory."""

    def __init__(self, database=None) -> None:
        self.database = database
        self._memory: dict[str, Any] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        if self.database is not None:
            self.database.set_state(key, value)
        else:
            with self._lock:
                self._memory[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if self.database is not None:
            return self.database.get_state(key, default)
        with self._lock:
            return self._memory.get(key, default)

    def values(self, prefix: str) -> list[Any]:
        if self.database is not None:
            return self.database.get_states(prefix)
        with self._lock:
            return [v for k, v in reversed(list(self._memory.items())) if k.startswith(prefix)]


@dataclass
class DownloadJob:
    job_id: str
    underlyings: list[str]
    from_date: date
    to_date: date
    interval: int = 5
    include_options: bool = True
    include_futures: bool = True
    status: str = "PENDING"
    started_at: str | None = None
    finished_at: str | None = None
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_skipped: int = 0
    chunks_empty: int = 0
    chunks_failed: int = 0
    rows_written: int = 0
    rows_rejected: int = 0
    missing_candles: int = 0
    contracts: dict[str, int] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    availability: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["from_date"] = self.from_date.isoformat()
        data["to_date"] = self.to_date.isoformat()
        data["failures"] = self.failures[-50:]
        data["failures_total"] = len(self.failures)
        return data


def nearest_windows(expiries: Iterable[date], start: date, end: date) -> dict[date, tuple[date, date]]:
    """For each expiry, the dates in [start, end] on which it was the nearest expiry."""
    windows = {}
    previous: date | None = None
    for expiry in sorted(set(expiries)):
        lo = max(start, previous + timedelta(days=1)) if previous else start
        hi = min(end, expiry)
        if lo <= hi:
            windows[expiry] = (lo, hi)
        previous = expiry
        if expiry >= end:
            break
    return windows


def strike_step(strikes: list[float]) -> float:
    values = sorted(set(strikes))
    diffs = [b - a for a, b in zip(values, values[1:]) if b - a > 0]
    if not diffs:
        raise HistoricalDataError("NO_STRIKES", "Cannot infer strike step")
    return pd.Series(diffs).mode().iloc[0]


class HistoricalDownloader:
    def __init__(
        self,
        client,
        store: ParquetCandleStore,
        catalog: ContractCatalog,
        metadata: MetadataStore | None = None,
        *,
        interval: int = 5,
        strike_window: int = 4,
        today: Callable[[], date] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.catalog = catalog
        self.metadata = metadata or MetadataStore()
        self.interval = int(interval)
        self.strike_window = int(strike_window)
        self.today = today or catalog.today
        self.cancel_event = cancel_event or threading.Event()

    # ----------------------------------------------------------------- utils
    def _save(self, job: DownloadJob) -> None:
        self.metadata.set(JOB_PREFIX + job.job_id, job.as_dict())

    def _chunk_key(self, kind: str, key: str, start: date) -> str:
        return f"{CHUNK_PREFIX}{kind}:{key}:{start.isoformat()}:{self.interval}"

    def _fetch(self, contract_source: str, key: str, start: date, end: date):
        if contract_source == ACTIVE:
            return self.client.candles(key, start, end, interval=self.interval)
        return self.client.expired_candles(key, start, end, interval=self.interval)

    def download_chunk(self, job: DownloadJob, *, kind: str, underlying: str, key: str, start: date, end: date,
                       source: str = ACTIVE, contract: HistoricalContract | None = None) -> dict:
        chunk_id = self._chunk_key(kind, key, start)
        previous = self.metadata.get(chunk_id)
        complete_month = end < self.today()
        expiry = contract.expiry if contract else None
        if previous and previous.get("status") in ("OK", "EMPTY") and previous.get("complete") and \
                previous.get("end") == end.isoformat() and \
                (previous["status"] == "EMPTY" or self.store.has_month(kind, underlying, key, start.year,
                                                                      start.month, expiry)):
            job.chunks_skipped += 1
            return previous
        record: dict[str, Any] = {"kind": kind, "instrument_key": key, "underlying": underlying,
                                  "start": start.isoformat(), "end": end.isoformat(), "source": source,
                                  "complete": complete_month}
        try:
            candles = self._fetch(source, key, start, end)
        except HistoricalDataError as exc:
            record.update(status="FAILED", error=exc.as_dict())
            job.chunks_failed += 1
            job.failures.append({"instrument_key": key, "start": start.isoformat(), "end": end.isoformat(),
                                 **exc.as_dict()})
            self.metadata.set(chunk_id, record)
            if exc.code in FATAL_CODES:
                raise
            return record

        extra: dict[str, Any] = {}
        if contract is not None:
            extra = {"underlying": underlying, "expiry": contract.expiry.isoformat(),
                     "strike": contract.strike, "option_type": contract.instrument_type}
        frame = candles_to_frame(candles, **extra)
        clean, report = validate_candles(
            frame, interval=self.interval, expiry=contract.expiry if contract else None,
            option_type=contract.instrument_type if contract and contract.is_option else None)
        written = self.store.write(kind, underlying, key, clean, expiry=contract.expiry if contract else None)
        record.update(status="OK" if len(clean) else "EMPTY", rows=len(clean), written=written,
                      validation=report.as_dict())
        job.rows_written += written
        job.rows_rejected += report.rejected_rows
        job.missing_candles += report.missing_candles
        if len(clean):
            job.chunks_done += 1
        else:
            job.chunks_empty += 1
        self.metadata.set(chunk_id, record)
        return record

    # ------------------------------------------------------------ per kind
    def _underlying_frame(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        key = self.catalog.underlying_key(underlying)
        return self.store.read("underlying", underlying, key, start=start, end=end)

    def download_underlying(self, job: DownloadJob, underlying: str) -> None:
        key = self.catalog.underlying_key(underlying)
        for start, end in month_chunks(job.from_date, job.to_date):
            if self.cancel_event.is_set():
                return
            job.chunks_total += 1
            self.download_chunk(job, kind="underlying", underlying=underlying, key=key, start=start, end=end)
            self._save(job)

    def _windows(self, contracts: list[HistoricalContract], job: DownloadJob) -> dict[date, tuple[date, date]]:
        return nearest_windows([c.expiry for c in contracts], job.from_date, job.to_date)

    def download_futures(self, job: DownloadJob, underlying: str) -> None:
        contracts = self.catalog.contracts(underlying, job.from_date, job.to_date, kind="FUT")
        windows = self._windows(contracts, job)
        job.contracts[f"{underlying}:FUT"] = len([c for c in contracts if c.expiry in windows])
        for contract in contracts:
            if contract.expiry not in windows:
                continue
            lo, hi = windows[contract.expiry]
            for start, end in month_chunks(lo, hi):
                if self.cancel_event.is_set():
                    return
                job.chunks_total += 1
                self.download_chunk(job, kind="futures", underlying=underlying, key=contract.instrument_key,
                                    start=start, end=end, source=contract.source, contract=contract)
                self._save(job)

    def option_targets(self, underlying: str, job: DownloadJob) -> list[tuple[HistoricalContract, date, date]]:
        """Contracts near the money while their expiry was the nearest one.

        Strikes come from the listed contracts for that expiry and the real
        underlying daily range; nothing is assumed about strike spacing.
        """
        contracts = self.catalog.contracts(underlying, job.from_date, job.to_date, kind="OPTION")
        windows = self._windows(contracts, job)
        spot = self._underlying_frame(underlying, job.from_date, job.to_date)
        if spot.empty:
            raise HistoricalDataError("UNDERLYING_DATA_MISSING",
                                      f"{underlying}: download underlying candles before options")
        daily = spot.groupby(spot["timestamp"].dt.date).agg(low=("low", "min"), high=("high", "max"))
        targets = []
        for expiry, (lo, hi) in windows.items():
            chain = [c for c in contracts if c.expiry == expiry]
            if not chain:
                continue
            days = daily[(daily.index >= lo) & (daily.index <= hi)]
            if days.empty:
                continue
            step = strike_step([c.strike for c in chain])
            low = float(days["low"].min()) - self.strike_window * step
            high = float(days["high"].max()) + self.strike_window * step
            for contract in chain:
                if low <= float(contract.strike) <= high:
                    targets.append((contract, lo, hi))
        return targets

    def download_options(self, job: DownloadJob, underlying: str) -> None:
        targets = self.option_targets(underlying, job)
        job.contracts[f"{underlying}:OPTIONS"] = len(targets)
        for contract, lo, hi in targets:
            for start, end in month_chunks(lo, hi):
                if self.cancel_event.is_set():
                    return
                job.chunks_total += 1
                self.download_chunk(job, kind="options", underlying=underlying, key=contract.instrument_key,
                                    start=start, end=end, source=contract.source, contract=contract)
            self._save(job)

    # ----------------------------------------------------------------- jobs
    def new_job(self, underlyings: list[str], from_date: date, to_date: date, *,
                include_options: bool = True, include_futures: bool = True) -> DownloadJob:
        if to_date < from_date:
            raise HistoricalDataError("INVALID_RANGE", "to date must be on or after from date")
        if from_date < EARLIEST_MINUTE_DATA:
            raise HistoricalDataError("INVALID_RANGE", "Upstox minute data starts in January 2022")
        if to_date > self.today():
            raise HistoricalDataError("INVALID_RANGE", "to date cannot be in the future")
        job = DownloadJob(job_id=uuid.uuid4().hex[:12], underlyings=[u.upper() for u in underlyings],
                          from_date=from_date, to_date=to_date, interval=self.interval,
                          include_options=include_options, include_futures=include_futures)
        self._save(job)
        return job

    def run(self, job: DownloadJob) -> DownloadJob:
        job.status = "RUNNING"
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._save(job)
        log_event(logger, logging.INFO, "historical_download_started", job_id=job.job_id,
                  underlyings=job.underlyings, start=job.from_date.isoformat(), end=job.to_date.isoformat())
        try:
            for underlying in job.underlyings:
                self.download_underlying(job, underlying)
                if job.include_futures:
                    self.download_futures(job, underlying)
                if job.include_options:
                    self.download_options(job, underlying)
                job.availability[underlying] = self.catalog.availability.get(underlying, {})
            if self.cancel_event.is_set():
                job.status, job.message = "CANCELLED", "Download cancelled"
            elif job.chunks_failed:
                job.status = "COMPLETED_WITH_ERRORS"
                job.message = f"{job.chunks_failed} chunk(s) failed; see failures"
            else:
                job.status, job.message = "COMPLETED", "Download completed"
        except HistoricalDataError as exc:
            job.status, job.message = "FAILED", f"{exc.code}: {exc.message}"
        except Exception as exc:  # recorded, never swallowed silently
            job.status, job.message = "FAILED", f"{type(exc).__name__}: {exc}"
        job.finished_at = datetime.now(timezone.utc).isoformat()
        for underlying in job.underlyings:
            job.availability.setdefault(underlying, self.catalog.availability.get(underlying, {}))
        self._save(job)
        log_event(logger, logging.INFO, "historical_download_finished", job_id=job.job_id, status=job.status,
                  chunks_failed=job.chunks_failed, rows_written=job.rows_written)
        return job

    def jobs(self) -> list[dict]:
        return self.metadata.values(JOB_PREFIX)
