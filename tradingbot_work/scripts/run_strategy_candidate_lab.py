#!/usr/bin/env python3
"""NIFTY 5-minute strategy candidate lab (read-only research, PAPER only, no orders, S2 unchanged)."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.current_day_s2 import IST, LevelParams, premium_frame, validate_intraday
from src.backtest.regime_filter import PointInTimeRegimeClassifier
from src.backtest.strategy_candidate_lab import (
    HISTORICAL_1DTE_UNAVAILABLE,
    Dataset,
    _pct,
    _fmt_pf,
    run_lab,
    write_outputs,
)
from src.historical_data.contracts import load_instrument_master_rows
from src.historical_data.models import HistoricalDataError, candles_to_frame
from src.historical_data.option_coverage import RequiredOptionContract, download_required_contract, resolve_signal_contract
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_candidate_lab"
INDEX_KEY = "NSE_INDEX|Nifty 50"


def front_future(service: HistoricalDataService, underlying: str, on: date) -> tuple[str, date]:
    """Nearest listed (non-expired) index future on ``on`` from the cached Upstox instrument master."""
    rows = load_instrument_master_rows(service._research_resolver.source)
    futs = []
    for r in rows:
        if r.get("instrument_type") == "FUT" and str(r.get("underlying_symbol", "")).upper() == underlying.upper():
            expiry = datetime.fromtimestamp(r["expiry"] / 1000, IST).date()
            if expiry >= on:
                futs.append((expiry, r["instrument_key"]))
    if not futs:
        raise SystemExit(f"No listed {underlying} future in the instrument master")
    expiry, key = min(futs)
    return key, expiry


def futures_history(service, client, key: str, expiry: date, underlying: str, start: date, end: date) -> pd.DataFrame:
    stored = service.store.read("futures", underlying, key, expiry=expiry, start=start, end=end)
    have = set(pd.to_datetime(stored["timestamp"]).dt.date) if not stored.empty else set()
    need = set(pd.to_datetime(service.read_underlying(underlying, start, end)["timestamp"]).dt.date)
    if need - have:
        raw = client.candles(key, start, end, interval=service.research_config.interval_minutes)
        frame = candles_to_frame(raw)
        frame = frame[frame["timestamp"].dt.date <= end]
        if not frame.empty:
            service.store.write("futures", underlying, key, frame.assign(instrument_key=key), expiry=expiry)
        stored = service.store.read("futures", underlying, key, expiry=expiry, start=start, end=end)
    return stored


def completed(frame: pd.DataFrame, day: date, interval: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    now = pd.Timestamp(datetime.now(IST))
    ts = pd.to_datetime(frame["timestamp"])
    return frame[(ts.dt.date == day) & (ts + pd.Timedelta(minutes=interval) <= now)].reset_index(drop=True)


def run(*, day: date, hist_from: date, hist_to: date, underlying: str = "NIFTY", out_dir: Path = DEFAULT_OUT,
        download_missing_options: bool = False, skip_current_day: bool = False) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    service = HistoricalDataService()
    client = service.client
    rc, sc = service.research_config, service.strategy_config
    params = LevelParams.from_configs(rc, sc)
    name = underlying.upper()

    spot = service.read_underlying(name, hist_from, hist_to)
    if spot.empty:
        raise SystemExit(f"No stored {name} candles {hist_from}..{hist_to}")
    hist_days = sorted(set(pd.to_datetime(spot["timestamp"]).dt.date))
    fut_key, fut_expiry = front_future(service, name, day if not skip_current_day else hist_to)
    try:
        hist_fut = futures_history(service, client, fut_key, fut_expiry, name, hist_from, hist_to)
    except HistoricalDataError as exc:
        print(f"Futures history unavailable ({exc.code}); VWAP-based rules will not fire.")
        hist_fut = pd.DataFrame()

    store_cache: dict[str, pd.DataFrame] = {}
    downloads: list[dict] = []

    def stored_premium(contract) -> pd.DataFrame:
        key = contract.instrument_key
        if key not in store_cache:
            frame = service.store.read("options", name, key, expiry=contract.expiry)
            if frame.empty and download_missing_options:
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
        frame = stored_premium(contract)
        if frame.empty:
            return "NO_PREMIUM_DATA"
        return frame[frame["timestamp"].dt.date <= hist_to].reset_index(drop=True)

    def resolve(sig):
        return resolve_signal_contract(sig, service.resolver_factory(sig.timestamp.date()), rc)

    historical = Dataset(spot=spot, futures=hist_fut, resolve=resolve, premium_for=hist_premium, days=hist_days,
                         regime_classifier=PointInTimeRegimeClassifier(spot))

    current = None
    validation = None
    if not skip_current_day:
        try:
            raw = client.intraday_candles(INDEX_KEY, interval=params.interval)
        except HistoricalDataError as exc:
            raise SystemExit(f"API_REQUEST_ERROR: {exc.code} (HTTP {exc.status})") from None
        today, validation = validate_intraday(raw, day, params.interval)
        today = completed(today, day, params.interval)
        if today.empty:
            raise SystemExit(f"No completed intraday candles for {day}; use --skip-current-day")
        try:
            fut_today = completed(candles_to_frame(client.intraday_candles(fut_key, interval=params.interval)),
                                  day, params.interval)
        except HistoricalDataError:
            fut_today = pd.DataFrame()
        combined = pd.concat([spot[["timestamp", "open", "high", "low", "close", "volume"]],
                              today[["timestamp", "open", "high", "low", "close", "volume"]]], ignore_index=True)
        fut_parts = [f[["timestamp", "open", "high", "low", "close", "volume"]] for f in (hist_fut, fut_today)
                     if f is not None and not f.empty]
        combined_fut = pd.concat(fut_parts, ignore_index=True) if fut_parts else pd.DataFrame()
        intraday_cache: dict[str, object] = {}

        def today_premium(contract):
            key = contract.instrument_key
            if key not in intraday_cache:
                now = datetime.now(IST)
                step = timedelta(minutes=params.interval)
                try:
                    candles = client.intraday_candles(key, interval=params.interval)
                    intraday_cache[key] = [c for c in candles if c.timestamp.astimezone(IST).date() == day
                                           and c.timestamp + step <= now]
                except HistoricalDataError as exc:
                    intraday_cache[key] = f"PREMIUM_API_ERROR:{exc.code}"
            got = intraday_cache[key]
            if isinstance(got, str):
                return got
            stored = service.store.read("options", name, key, expiry=contract.expiry)
            stored = stored[stored["timestamp"].dt.date < day] if not stored.empty else None
            return premium_frame(stored, got)

        current = Dataset(spot=combined, futures=combined_fut, resolve=resolve, premium_for=today_premium,
                          days=[day], regime_classifier=PointInTimeRegimeClassifier(combined))

    report = run_lab(historical, params=params, square_off=rc.session.square_off,
                     stop_new_entries=rc.session.stop_new_entries, current=current,
                     current_day=None if skip_current_day else day, lot_size=65)
    report["data"] = {
        "underlying": name, "historical_window": f"{hist_from}..{hist_to}", "historical_sessions": len(hist_days),
        "vwap_source": f"{fut_key} (NIFTY FUT expiry {fut_expiry}), real 5-minute volume",
        "futures_sessions": int(pd.to_datetime(hist_fut["timestamp"]).dt.date.nunique()) if not hist_fut.empty else 0,
        "current_day_candles": None if validation is None else {
            "valid": validation.valid_candles, "expected": validation.expected_candles,
            "missing": validation.missing_candles, "duplicates": validation.duplicates},
        "option_downloads": downloads,
        "historical_1dte": ("No historical 1-DTE option candles: the Upstox expired-instrument API is not available "
                            "on this plan (PLAN_NOT_SUPPORTED) and the store only holds the 2026-10-06 weekly. "
                            f"Flag: {HISTORICAL_1DTE_UNAVAILABLE}"),
    }
    paths = write_outputs(report, out_dir)
    for p in paths.values():
        print(f"Wrote {p}")
    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    print("\nSTRATEGY CANDIDATE LAB COMPLETE")
    print(f"  Data flags: {', '.join(report['data_flags']) or 'none'}; historical option DTE "
          f"{report['historical_option_dte']}")
    for k, v in report["split_days"].items():
        print(f"  {k:<10}: {v[0]} .. {v[-1]} ({len(v)} sessions)")
    print(f"  {'Candidate':<58} {'TrainWR':>8} {'ValWR':>8} {'OosWR':>8} {'OosPF':>6} {'OosPnL':>9} "
          f"{'OosDD':>8} {'CurWR':>8}  Status")
    for t in report["ranked"]:
        print(f"  {t['candidate']:<58} {_pct(t['train_wr']):>8} {_pct(t['validation_wr']):>8} {_pct(t['oos_wr']):>8} "
              f"{_fmt_pf(t['oos_pf']):>6} {t['oos_pnl']:>9,.0f} {t['oos_max_dd']:>8,.0f} "
              f"{_pct(t['current_day_wr']):>8}  {t['status']}")
    print(f"  Best (by status, then VALIDATION/TRAIN WR): {report['best_candidate']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY strategy candidate lab (research only)")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date(),
                        help="current-day session evaluated separately")
    parser.add_argument("--hist-from", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--hist-to", type=date.fromisoformat, default=date(2026, 10, 1))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download-missing-options", action="store_true",
                        help="download real premium candles for resolved contracts missing from the store")
    parser.add_argument("--skip-current-day", action="store_true")
    args = parser.parse_args()
    if args.hist_to >= args.date and not args.skip_current_day:
        raise SystemExit("--hist-to must be before the current-day session")
    run(day=args.date, hist_from=args.hist_from, hist_to=args.hist_to, out_dir=args.out_dir,
        download_missing_options=args.download_missing_options, skip_current_day=args.skip_current_day)


if __name__ == "__main__":
    main()
