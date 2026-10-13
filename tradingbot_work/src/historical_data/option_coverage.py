"""Map research signals to required option contracts and ensure premium coverage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.research_strategies import ResearchSignal, generate_signals
from src.historical_data.contracts import HistoricalContract
from src.historical_data.downloader import HistoricalDownloader, DownloadJob, month_chunks
from src.historical_data.models import HistoricalDataError, candles_to_frame
from src.historical_data.storage import ParquetCandleStore
from src.historical_data.validation import validate_candles
from src.options.instrument_resolver import OptionInstrumentResolver
from src.options.models import OptionContract, OptionResolutionError, OptionType
from src.options.selection import atm_strike, select_expiry, select_strike

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class RequiredOptionContract:
    instrument_key: str
    underlying: str
    expiry: date
    strike: float
    option_type: str
    trading_symbol: str
    lot_size: int
    freeze_quantity: int | None
    tick_size: float | None
    from_date: date
    to_date: date
    signal_count: int = 0
    signal_times: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expiry"] = self.expiry.isoformat()
        data["from_date"] = self.from_date.isoformat()
        data["to_date"] = self.to_date.isoformat()
        return data


@dataclass
class CoverageReport:
    underlying: str
    strategy_id: str
    from_date: date
    to_date: date
    total_signals: int = 0
    unique_required_contracts: int = 0
    contracts_already_stored: int = 0
    contracts_downloaded: int = 0
    contracts_unavailable: int = 0
    signals_with_complete_coverage: int = 0
    signals_missing_coverage: int = 0
    coverage_percent: float = 0.0
    required: list[dict[str, Any]] = field(default_factory=list)
    downloaded: list[dict[str, Any]] = field(default_factory=list)
    unavailable: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["from_date"] = self.from_date.isoformat()
        data["to_date"] = self.to_date.isoformat()
        return data


def _ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def _parse_time(value: str):
    hour, minute = value.split(":")
    from datetime import time
    return time(int(hour), int(minute))


def entry_allowed(ts: datetime, session, stop_new_entries: str) -> bool:
    local = ts.astimezone(IST) if ts.tzinfo else ts.replace(tzinfo=IST)
    return session.is_entry_allowed(local) and local.time() < _parse_time(stop_new_entries)


def resolve_signal_contract(
    signal: ResearchSignal,
    resolver: OptionInstrumentResolver,
    research_config,
) -> OptionContract:
    """Same selection rules as OptionsResearchEngine (BUY->CE, SELL->PE, nearest, ATM)."""
    resolver.today = lambda: signal.timestamp.date()
    option_type = OptionType.CE if signal.direction == "BUY" else OptionType.PE
    expiry = select_expiry(
        resolver.expiries(signal.underlying),
        research_config.option_mapping.get("expiry_mode", "nearest"),
    )
    strike = atm_strike(
        resolver.strikes(signal.underlying, expiry, option_type),
        signal.spot,
    )
    offset = int(research_config.option_mapping.get("strike_offset", 0))
    if offset:
        strike = select_strike(
            resolver.strikes(signal.underlying, expiry, option_type),
            option_type=option_type,
            mode=research_config.option_mapping.get("strike_mode", "ATM"),
            spot=signal.spot,
            offset=offset,
        )
    return resolver.resolve(signal.underlying, expiry, strike, option_type)


def research_signals(
    underlying_candles: pd.DataFrame,
    underlying: str,
    strategy_id: str,
    research_config,
    from_date: date,
    to_date: date,
    session,
) -> list[ResearchSignal]:
    stop = research_config.session.stop_new_entries
    raw = generate_signals(strategy_id, underlying_candles, underlying.upper(), research_config)
    return [
        s for s in raw
        if from_date <= s.timestamp.date() <= to_date and entry_allowed(s.timestamp, session, stop)
    ]


def extract_required_contracts(
    signals: Iterable[ResearchSignal],
    resolver_factory: Callable[[date], OptionInstrumentResolver],
    research_config,
    from_date: date,
    to_date: date,
) -> tuple[list[RequiredOptionContract], list[dict[str, Any]]]:
    """Return (unique required contracts, per-signal resolution errors)."""
    grouped: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for signal in signals:
        try:
            contract = resolve_signal_contract(signal, resolver_factory(signal.timestamp.date()), research_config)
        except OptionResolutionError as exc:
            errors.append({
                "signal_time": signal.timestamp.isoformat(),
                "direction": signal.direction,
                "spot": signal.spot,
                "reason": exc.code,
                "message": exc.message,
            })
            continue
        key = contract.instrument_key
        if key not in grouped:
            grouped[key] = {
                "contract": contract,
                "times": [],
                "dates": [],
            }
        grouped[key]["times"].append(signal.timestamp.isoformat())
        grouped[key]["dates"].append(signal.timestamp.date())

    required: list[RequiredOptionContract] = []
    for item in grouped.values():
        contract = item["contract"]
        dates = item["dates"]
        required.append(RequiredOptionContract(
            instrument_key=contract.instrument_key,
            underlying=contract.underlying,
            expiry=contract.expiry,
            strike=float(contract.strike),
            option_type=contract.option_type.value,
            trading_symbol=contract.trading_symbol,
            lot_size=int(contract.lot_size),
            freeze_quantity=contract.freeze_quantity,
            tick_size=contract.tick_size,
            from_date=min(min(dates), from_date),
            to_date=max(max(dates), to_date),
            signal_count=len(item["times"]),
            signal_times=tuple(item["times"]),
        ))
    return sorted(required, key=lambda c: (c.expiry, c.strike, c.option_type)), errors


def read_option_premium(
    store: ParquetCandleStore,
    contract: RequiredOptionContract,
    from_date: date,
    to_date: date,
) -> pd.DataFrame:
    return store.read(
        "options",
        contract.underlying,
        contract.instrument_key,
        expiry=contract.expiry,
        start=from_date,
        end=to_date,
    )


def verify_option_premium(
    store: ParquetCandleStore,
    contract: RequiredOptionContract,
    from_date: date,
    to_date: date,
    *,
    interval: int = 5,
) -> dict[str, Any]:
    frame = read_option_premium(store, contract, from_date, to_date)
    paths = store.files("options", contract.underlying, contract.instrument_key, contract.expiry)
    if frame.empty:
        raise HistoricalDataError("OPTION_DATA_MISSING", f"No stored candles for {contract.instrument_key}")
    clean, report = validate_candles(
        frame, interval=interval, expiry=contract.expiry, option_type=contract.option_type,
    )
    if clean.empty:
        raise HistoricalDataError("OPTION_DATA_INVALID", f"{contract.instrument_key} failed validation")
    ts = pd.to_datetime(clean["timestamp"])
    in_range = clean[(ts.dt.date >= from_date) & (ts.dt.date <= to_date)]
    if in_range.empty:
        raise HistoricalDataError(
            "OPTION_DATA_INCOMPLETE",
            f"{contract.instrument_key} has no candles between {from_date} and {to_date}",
        )
    return {
        "instrument_key": contract.instrument_key,
        "candle_count": len(clean),
        "files": [str(p) for p in paths],
        "first": ts.min().isoformat(),
        "last": ts.max().isoformat(),
        "validation": report.as_dict(),
    }


def signal_has_premium_coverage(
    store: ParquetCandleStore,
    contract: RequiredOptionContract,
    signal: ResearchSignal,
    *,
    entry_timing: str = "next_open",
) -> tuple[bool, str]:
    frame = read_option_premium(store, contract, contract.from_date, contract.to_date)
    if frame.empty:
        return False, "NO_PREMIUM_DATA"
    ts = _ts(signal.timestamp)
    indexed = frame.copy()
    indexed["timestamp"] = indexed["timestamp"].map(_ts)
    past = indexed[indexed["timestamp"] <= ts]
    if past.empty:
        return False, "NO_PREMIUM_AT_SIGNAL"
    signal_idx = past.index[-1]
    future = indexed[indexed.index > signal_idx]
    if entry_timing == "next_open" and future.empty:
        return False, "NO_ENTRY_BAR"
    return True, "OK"


def has_stored_option_data(
    store: ParquetCandleStore,
    contract: RequiredOptionContract,
    from_date: date,
    to_date: date,
) -> bool:
    try:
        verify_option_premium(store, contract, from_date, to_date)
        return True
    except HistoricalDataError:
        return False


def _to_historical_contract(contract: RequiredOptionContract) -> HistoricalContract:
    return HistoricalContract(
        instrument_key=contract.instrument_key,
        underlying=contract.underlying,
        expiry=contract.expiry,
        instrument_type=contract.option_type,
        lot_size=contract.lot_size,
        source="ACTIVE",
        strike=contract.strike,
        trading_symbol=contract.trading_symbol,
        freeze_quantity=contract.freeze_quantity,
        tick_size=contract.tick_size,
    )


def download_required_contract(
    downloader: HistoricalDownloader,
    contract: RequiredOptionContract,
    from_date: date,
    to_date: date,
) -> dict[str, Any]:
    """Download one listed option contract via Upstox V3 (read-only)."""
    job = DownloadJob(
        job_id="coverage",
        underlyings=[contract.underlying],
        from_date=from_date,
        to_date=to_date,
    )
    hist = _to_historical_contract(contract)
    records = []
    for start, end in month_chunks(from_date, to_date):
        records.append(downloader.download_chunk(
            job,
            kind="options",
            underlying=contract.underlying,
            key=contract.instrument_key,
            start=start,
            end=end,
            source="ACTIVE",
            contract=hist,
        ))
    failed = [r for r in records if r.get("status") == "FAILED"]
    if failed:
        error = failed[-1].get("error", {})
        raise HistoricalDataError(
            error.get("code", "DOWNLOAD_FAILED"),
            error.get("message", "Option download failed"),
        )
    return verify_option_premium(downloader.store, contract, from_date, to_date, interval=downloader.interval)


def ensure_option_coverage(
    *,
    store: ParquetCandleStore,
    downloader: HistoricalDownloader,
    resolver_factory: Callable[[date], OptionInstrumentResolver],
    research_config,
    strategy_config,
    underlying: str,
    strategy_id: str,
    underlying_candles: pd.DataFrame,
    from_date: date,
    to_date: date,
) -> CoverageReport:
    from src.session.session_manager import SessionManager

    session = SessionManager(strategy_config.session)
    signals = research_signals(underlying_candles, underlying, strategy_id, research_config, from_date, to_date, session)
    required, resolution_errors = extract_required_contracts(
        signals, resolver_factory, research_config, from_date, to_date,
    )
    report = CoverageReport(
        underlying=underlying.upper(),
        strategy_id=strategy_id,
        from_date=from_date,
        to_date=to_date,
        total_signals=len(signals),
        unique_required_contracts=len(required),
        required=[c.as_dict() for c in required],
    )
    entry_timing = str(research_config.option_mapping.get("entry_timing", "next_open"))

    for contract in required:
        if has_stored_option_data(store, contract, contract.from_date, contract.to_date):
            report.contracts_already_stored += 1
            continue
        try:
            verification = download_required_contract(
                downloader, contract, contract.from_date, contract.to_date,
            )
            report.contracts_downloaded += 1
            report.downloaded.append(verification)
        except HistoricalDataError as exc:
            report.contracts_unavailable += 1
            report.unavailable.append({
                "instrument_key": contract.instrument_key,
                "trading_symbol": contract.trading_symbol,
                "expiry": contract.expiry.isoformat(),
                "strike": contract.strike,
                "option_type": contract.option_type,
                "signal_count": contract.signal_count,
                "code": exc.code,
                "message": exc.message,
            })

    for err in resolution_errors:
        report.contracts_unavailable += 1
        report.unavailable.append(err)

    covered = 0
    for signal in signals:
        contract = next(
            (c for c in required if signal.timestamp.isoformat() in c.signal_times),
            None,
        )
        if contract is None:
            report.signals_missing_coverage += 1
            continue
        ok, reason = signal_has_premium_coverage(store, contract, signal, entry_timing=entry_timing)
        if ok:
            covered += 1
        else:
            report.signals_missing_coverage += 1
            report.unavailable.append({
                "signal_time": signal.timestamp.isoformat(),
                "instrument_key": contract.instrument_key,
                "reason": reason,
            })
    report.signals_with_complete_coverage = covered
    if signals:
        report.coverage_percent = round(100.0 * covered / len(signals), 2)
    return report
