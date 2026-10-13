#!/usr/bin/env python3
"""S2 1-DTE improvement research (read-only, PAPER only, no orders, S2 unchanged)."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.current_day_s2 import (
    IST,
    CandleValidation,
    LevelParams,
    point_in_time_signals,
    premium_frame,
    validate_intraday,
)
from src.backtest.research_strategies import generate_signals
from src.backtest.s2_1dte_improvement import DEFAULT_SPLITS, run_research, write_reports
from src.historical_data.models import HistoricalDataError
from src.historical_data.option_coverage import (
    RequiredOptionContract,
    download_required_contract,
    entry_allowed,
    resolve_signal_contract,
)
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.historical_data.validation import expected_slots
from src.safety.live_gate import assert_paper_trading_only
from src.session.session_manager import SessionManager

DEFAULT_OUT = ROOT / "reports" / "strategy_research"
INDEX_KEY = "NSE_INDEX|Nifty 50"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


MISSING_LOCAL_PARQUET = "MISSING_LOCAL_PARQUET"


def complete_stored_session(frame: pd.DataFrame | None, day: date, interval: int) -> pd.DataFrame | None:
    """Stored candles for ``day`` only when every session slot is present; partial days are never used."""
    if frame is None or frame.empty:
        return None
    got = frame[frame["timestamp"].dt.date == day].sort_values("timestamp").reset_index(drop=True)
    have = {pd.Timestamp(ts).value for ts in got["timestamp"]}
    if got.empty or not {pd.Timestamp(s).value for s in expected_slots(day, interval)} <= have:
        return None
    return got


def s2_components(service: HistoricalDataService, name: str) -> tuple:
    """(research_config, params, session, s2_config, signal_fn, resolve) - the unchanged S2 signal and contract rules."""
    rc, sc = service.research_config, service.strategy_config
    params = LevelParams.from_configs(rc, sc)
    session = SessionManager(sc.session)

    def signal_fn(frame: pd.DataFrame) -> list:
        return [s for s in generate_signals("S2", frame.assign(symbol=name), name, rc)
                if entry_allowed(s.timestamp, session, params.stop_new_entries)]

    def resolve(sig):
        return resolve_signal_contract(sig, service.resolver_factory(sig.timestamp.date()), rc)

    return rc, params, session, rc.strategies["S2"], signal_fn, resolve


def prepare_inputs(*, day: date, hist_from: date, hist_to: date, underlying: str = "NIFTY",
                   download_missing_options: bool = False, prefer_stored_current_day: bool = False) -> dict:
    """Read-only research inputs shared by S2 research runners (stored history + completed current-day candles).

    With ``prefer_stored_current_day`` a complete stored session for ``day`` is used instead of the Upstox
    intraday endpoint (NIFTY and each option contract). For a past day without stored candles nothing is
    downloaded: NIFTY raises ``SystemExit`` and an option premium is reported as ``MISSING_LOCAL_PARQUET``.
    """
    assert_paper_trading_only()
    _refuse_live()
    service = HistoricalDataService()
    client = service.client
    name = underlying.upper()
    rc, params, session, s2, signal_fn, resolve = s2_components(service, name)

    spot = service.read_underlying(name, hist_from, hist_to)
    if spot.empty:
        raise SystemExit(f"No stored {name} candles {hist_from}..{hist_to}")

    store_cache: dict[str, pd.DataFrame] = {}
    downloads: list[dict] = []

    def stored(contract) -> pd.DataFrame:
        key = contract.instrument_key
        if key not in store_cache:
            frame = service.store.read("options", name, key, expiry=contract.expiry)
            if frame.empty and download_missing_options and contract.expiry > hist_to:
                req = RequiredOptionContract(
                    instrument_key=key, underlying=name, expiry=contract.expiry, strike=float(contract.strike),
                    option_type=contract.option_type.value, trading_symbol=contract.trading_symbol,
                    lot_size=int(contract.lot_size), freeze_quantity=contract.freeze_quantity,
                    tick_size=contract.tick_size, from_date=hist_from, to_date=hist_to)
                try:
                    downloads.append(download_required_contract(service._downloader(), req, hist_from, hist_to))
                except HistoricalDataError as exc:
                    downloads.append({"instrument_key": key, "error": exc.code})
                frame = service.store.read("options", name, key, expiry=contract.expiry)
            store_cache[key] = frame
        return store_cache[key]

    def hist_premium(contract):
        frame = stored(contract)
        if frame.empty:
            return "NO_PREMIUM_DATA"
        return frame[frame["timestamp"].dt.date <= hist_to].reset_index(drop=True)

    now = datetime.now(IST)
    step = timedelta(minutes=params.interval)
    is_live_day = day == now.date()
    stored_today = None
    if prefer_stored_current_day:
        stored_today = complete_stored_session(
            service.store.read("underlying", name, service.underlying_key(name), start=day, end=day),
            day, params.interval)
        if stored_today is None and not is_live_day:
            raise SystemExit(f"DATA_UNAVAILABLE: {MISSING_LOCAL_PARQUET} for {name} {day} "
                             "(intraday API only serves the current session)")
    current_day_source = "LOCAL_PARQUET" if stored_today is not None else "UPSTOX_INTRADAY_V3"
    if stored_today is not None:
        today = stored_today
        validation = CandleValidation(requested_day=day.isoformat(), received=len(today),
                                      expected_candles=len(expected_slots(day, params.interval)),
                                      valid_candles=len(today))
    else:
        try:
            raw = client.intraday_candles(INDEX_KEY, interval=params.interval)
        except HistoricalDataError as exc:
            raise SystemExit(f"API_REQUEST_ERROR: {exc.code} (HTTP {exc.status})") from None
        today, validation = validate_intraday(raw, day, params.interval)
    today = today[today["timestamp"] + step <= now].reset_index(drop=True)
    if today.empty:
        raise SystemExit(f"No completed intraday candles for {day}")
    combined = pd.concat([spot[COLUMNS], today[COLUMNS]], ignore_index=True)
    pit_signals, pit_same = point_in_time_signals(today, name, rc, session, params.stop_new_entries)
    full_today = [s for s in signal_fn(combined) if s.timestamp.date() == day]
    pit_ok = pit_same and [(s.timestamp, s.direction) for s in pit_signals] == \
        [(s.timestamp, s.direction) for s in full_today]

    intraday_cache: dict[str, object] = {}

    def current_premium(contract):
        key = contract.instrument_key
        if key not in intraday_cache and prefer_stored_current_day:
            local = complete_stored_session(stored(contract), day, params.interval)
            if local is not None:
                intraday_cache[key] = local
            elif not is_live_day:
                intraday_cache[key] = MISSING_LOCAL_PARQUET
        if key not in intraday_cache:
            try:
                candles = client.intraday_candles(key, interval=params.interval)
                intraday_cache[key] = [c for c in candles if c.timestamp.astimezone(IST).date() == day
                                       and c.timestamp + step <= now]
            except HistoricalDataError as exc:
                intraday_cache[key] = f"PREMIUM_API_ERROR:{exc.code}"
        got = intraday_cache[key]
        if isinstance(got, str):
            return got
        prior = service.store.read("options", name, key, expiry=contract.expiry)
        prior = prior[prior["timestamp"].dt.date < day] if not prior.empty else None
        if isinstance(got, pd.DataFrame):
            return premium_frame(pd.concat([p for p in (prior, got) if p is not None], ignore_index=True), [])
        return premium_frame(prior, got)

    return {
        "research": dict(hist_spot=spot, signal_fn=signal_fn, resolve=resolve, hist_premium_for=hist_premium,
                         params=params, square_off=rc.session.square_off, range_start=s2["range_start"],
                         range_end=s2["range_end"], current_spot=combined, current_premium_for=current_premium,
                         current_day=day, splits=DEFAULT_SPLITS),
        "point_in_time_matches_full_run": pit_ok,
        "data": {"underlying": name, "historical_window": f"{hist_from}..{hist_to}",
                 "current_day_source": current_day_source,
                 "current_day_candles": {"valid": validation.valid_candles, "expected": validation.expected_candles,
                                         "missing": validation.missing_candles},
                 "option_downloads": downloads},
    }


def run(*, day: date, hist_from: date, hist_to: date, out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY",
        download_missing_options: bool = False) -> dict:
    inputs = prepare_inputs(day=day, hist_from=hist_from, hist_to=hist_to, underlying=underlying,
                            download_missing_options=download_missing_options)
    report = run_research(**inputs["research"],
                          point_in_time_matches_full_run=inputs["point_in_time_matches_full_run"])
    report["data"] = inputs["data"]
    paths = write_reports(report, out_dir)
    for p in paths.values():
        print(f"Wrote {p}")
    print("\n" + report["summary"]["final_block"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 1-DTE improvement research (read-only)")
    parser.add_argument("--date", type=date.fromisoformat, default=date(2026, 10, 5))
    parser.add_argument("--hist-from", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--hist-to", type=date.fromisoformat, default=date(2026, 10, 1))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download-missing-options", action="store_true")
    args = parser.parse_args()
    run(day=args.date, hist_from=args.hist_from, hist_to=args.hist_to, out_dir=args.out_dir,
        download_missing_options=args.download_missing_options)


if __name__ == "__main__":
    main()
