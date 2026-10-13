#!/usr/bin/env python3
"""Track NIFTY S2 option-premium signal accuracy from real Upstox V3 intraday candles (observation only, no orders).

Each run registers new point-in-time S2 signals (idempotent) and resolves
pending ones from completed option premium candles, then writes
reports/live_signal_accuracy/YYYY-MM-DD.{json,csv}. Use --poll to repeat
during the session; the dashboard reads the same database.

Also reachable as ``python scripts/run_live_signal_accuracy.py --mode premium-replay``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time as time_mod
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any, Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.current_day_s2 import (
    IST,
    LevelParams,
    option_levels,
    point_in_time_signals,
    premium_frame,
    validate_intraday,
)
from src.database import Database
from src.historical_data.models import HistoricalDataError, candles_to_frame
from src.historical_data.option_coverage import resolve_signal_contract
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.historical_data.upstox_client import token_source
from src.options.models import OptionResolutionError
from src.paper_trading.signal_tracker import (
    LiveSignalTracker,
    SignalRecord,
    compute_stats,
    verify_observation_only,
    write_daily_report,
)
from src.safety.live_gate import assert_paper_trading_only, live_trading_enabled, trading_mode
from src.session.session_manager import SessionManager
from src.signal_accuracy.paper_dashboard import PREMIUM_STATUS_FILE
from src.signal_accuracy.realtime import write_status

DEFAULT_OUT = ROOT / "reports" / "live_signal_accuracy"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
SESSION_CLOSE = "15:30"
NOTES = [
    "Observation only: no broker, engine or order code is used; nothing is ordered.",
    "Signals are generated bar by bar from completed NIFTY candles (no look-ahead).",
    "Entry = next-bar option open; SL = premium-ATR stop; target = target R (unchanged S2 levels).",
    "Outcome = first touch of the predefined SL or target on real option candles; trailing/breakeven not applied.",
    "SL and target inside the same candle -> AMBIGUOUS (excluded). EOD exit at the open of the first "
    "candle at/after square-off.",
    "P&L = gross premium move x 1 lot (lot size of the contract); costs and slippage excluded.",
    "Accuracy = wins / (wins + losses). It is a measurement, not a forecast.",
]


def completed(frame: pd.DataFrame, interval: int, now: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    ends = frame["timestamp"] + pd.Timedelta(minutes=interval)
    return frame[ends <= pd.Timestamp(now)].reset_index(drop=True)


def run_once(*, day: date, database: Database, tracker: LiveSignalTracker, service: HistoricalDataService,
             client, underlying: str = "NIFTY", strategy_id: str = "S2", out_dir: Path = DEFAULT_OUT,
             now: datetime | None = None, option_feed_ok: Callable[[], bool] | None = None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    now = now or datetime.now(IST)
    params = LevelParams.from_configs(service.research_config, service.strategy_config)
    try:
        raw = client.intraday_candles(INSTRUMENT_KEY, interval=params.interval)
    except HistoricalDataError as exc:
        raise SystemExit(f"API_REQUEST_ERROR: {exc.code} (HTTP {exc.status})") from None
    candles, validation = validate_intraday(raw, day, params.interval)
    candles = completed(candles, params.interval, now)

    session = SessionManager(service.strategy_config.session)
    signals = []
    if not candles.empty:
        signals, _ = point_in_time_signals(candles, underlying, service.research_config, session,
                                           params.stop_new_entries, strategy_id)

    premium_cache: dict[str, object] = {}
    today_cache: dict[str, pd.DataFrame] = {}

    def premium_for(contract):
        key = contract.instrument_key
        if key not in premium_cache:
            try:
                today = client.intraday_candles(key, interval=params.interval)
            except HistoricalDataError as exc:
                premium_cache[key] = f"PREMIUM_API_ERROR:{exc.code}"
                return premium_cache[key]
            frame = candles_to_frame(today)
            if not frame.empty:
                frame = frame[frame["timestamp"].dt.date == day]
            today_cache[key] = completed(frame.reset_index(drop=True), params.interval, now)
            stored = service.store.read("options", underlying.upper(), key, expiry=contract.expiry)
            premium_cache[key] = completed(premium_frame(stored, today), params.interval, now)
        return premium_cache[key]

    created = deferred = 0
    for sig in signals:
        fields = {"signal_price": sig.spot, "source": "INTRADAY_REPLAY"}
        reason = None
        try:
            contract = resolve_signal_contract(sig, service.resolver_factory(sig.timestamp.date()),
                                               service.research_config)
        except OptionResolutionError as exc:
            contract, reason = None, f"CONTRACT_UNRESOLVED:{exc.code}"
        active_from = pd.Timestamp(sig.timestamp) + pd.Timedelta(minutes=params.interval)
        if contract is not None:
            option_type = getattr(contract, "option_type", None)
            fields.update(
                instrument_key=contract.instrument_key, trading_symbol=contract.trading_symbol,
                strike=float(contract.strike), option_type=getattr(option_type, "value", option_type),
                expiry=contract.expiry.isoformat(), quantity=int(contract.lot_size),
            )
            premium = premium_for(contract)
            if isinstance(premium, str):
                reason = premium
            elif premium.empty:
                reason = "NO_PREMIUM_DATA"
            else:
                levels = option_levels(sig, contract, premium, params)
                if levels["status"] != "RESOLVED":
                    reason = levels["reason"]
                else:
                    fields.update(entry_price=levels["entry"], stop_loss=levels["stop_loss"],
                                  target=levels["target"])
                    active_from = pd.Timestamp(levels["entry_time"])
        entry_candle_end = pd.Timestamp(sig.timestamp) + pd.Timedelta(minutes=2 * params.interval)
        if reason == "NO_PREMIUM_ENTRY_CANDLE" and pd.Timestamp(now) < entry_candle_end:
            # Registration is permanent, so wait until the entry candle can have completed.
            deferred += 1
            continue
        if reason is None and fields.get("entry_price") and option_feed_ok is not None and not option_feed_ok():
            deferred += 1
            continue
        record = SignalRecord.create(timestamp=sig.timestamp, underlying=underlying, direction=sig.direction,
                                     strategy=strategy_id, active_from=active_from, **fields)
        created += tracker.register(record, invalid_reason=reason)

    for key in tracker.pending_keys():
        if key not in today_cache:
            continue
        for row in today_cache[key].sort_values("timestamp").to_dict("records"):
            tracker.on_candle(key, row)

    close = datetime.combine(day, datetime.strptime(SESSION_CLOSE, "%H:%M").time(), IST)
    if now >= close + timedelta(minutes=params.interval):
        tracker.finalize_session(day)

    rows = database.get_live_signals(day.isoformat())
    paths = write_daily_report(rows, day, out_dir, NOTES)
    return {"stats": compute_stats(rows), "paths": paths, "new_signals": created, "deferred_signals": deferred,
            "candles": len(candles), "received": validation.received}


def write_premium_status(out_dir: Path, *, day: date, running: bool, stats: Mapping[str, Any],
                         candles: int, now: datetime | None = None, **extra: Any) -> None:
    now = now or datetime.now(IST)
    write_status(out_dir / PREMIUM_STATUS_FILE, {
        "type": "PREMIUM_SIGNAL_TRACKER_STATUS",
        "updated_at": now.isoformat(),
        "running": running,
        "session_date": day.isoformat(),
        "mode": extra.get("mode", "REST_POLL"),
        "completed_candles": candles,
        "pending_signals": stats.get("pending", 0),
        "subscription_status": extra.get("subscription_status", "REST_ONLY"),
        "subscribed_instruments": extra.get("subscribed_instruments", []),
        "orders_placed": 0,
        "trading_mode": trading_mode(),
        "live_trading_enabled": live_trading_enabled(),
        **{k: v for k, v in extra.items() if k not in {"mode", "subscription_status", "subscribed_instruments"}},
    })


def print_ready(day: date, result: dict) -> None:
    s = result["stats"]
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else (
        "no losses" if s["resolved"] and s["gross_profit"] > 0 else "N/A")
    print("\nLIVE SIGNAL ACCURACY TRACKER READY")
    print(f"  Date / token source : {day} / {token_source()}")
    print(f"  Completed candles   : {result['candles']} (new signals this run: {result['new_signals']})")
    print(f"  Total signals       : {s['total']}")
    print(f"  Resolved            : {s['resolved']}")
    print(f"  Pending             : {s['pending']}")
    print(f"  Wins / Losses       : {s['wins']} / {s['losses']}")
    print(f"  Ambiguous / Invalid : {s['ambiguous']} / {s['invalid']}")
    print(f"  Current accuracy    : {s['accuracy_display']}")
    print(f"  Profit factor       : {pf}")
    print(f"  Net P&L (1 lot)     : Rs {s['net_pnl']:,.2f}")
    print(f"  Max drawdown        : Rs {s['max_drawdown']:,.2f}")
    print(f"  Average R           : {s['average_r'] if s['average_r'] is not None else 'N/A'}")
    for path in result["paths"].values():
        print(f"  Report              : {path}")


def run_realtime(args) -> None:
    from src.signal_accuracy.premium_realtime import LivePremiumRealtimeRunner, start_premium_session
    from src.signal_accuracy.realtime import TrackerSettings, create_feed

    database = Database(args.db)
    service = HistoricalDataService()
    tracker = LiveSignalTracker(database, square_off=service.research_config.session.square_off)
    settings = TrackerSettings.load()
    runner_ref: list[LivePremiumRealtimeRunner] = []

    def rest_cycle() -> None:
        feed_ok = runner_ref[0].option_feed_ok if runner_ref else None
        run_once(day=args.date, database=database, tracker=tracker, service=service,
                 client=service.client, out_dir=args.out_dir, option_feed_ok=feed_ok)

    print("PREMIUM REALTIME MODE — WebSocket option ticks + REST signal backfill")
    print(f"REST backfill every {args.poll or 30}s | subscribed keys refresh with pending signals")
    feed = create_feed(stale_after_seconds=settings.stale_data_seconds)
    runner = LivePremiumRealtimeRunner(feed, database, tracker, settings=settings, out_dir=args.out_dir,
                                       rest_poll=rest_cycle, poll_seconds=args.poll or 30)
    runner_ref.append(runner)
    runner.run(args.date.isoformat(), until=getattr(args, "until", None))


def premium_realtime_worker(*, db: str, out_dir: Path, day: date, poll: int = 30,
                            until: datetime | None = None) -> None:
    """Blocking premium WebSocket worker (for background threads)."""
    ns = argparse.Namespace(db=db, out_dir=out_dir, date=day, poll=poll, until=until)
    run_realtime(ns)


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 live signal accuracy tracker (observation only)")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date())
    parser.add_argument("--db", default=os.getenv("DATABASE_PATH", "data/trading.db"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll", type=int, default=0, help="REST repeat interval seconds (default 30 in --realtime)")
    parser.add_argument("--realtime", action="store_true",
                        help="WebSocket option ticks for pending contracts (+ REST backfill)")
    args = parser.parse_args()

    safety = verify_observation_only()
    print(f"TRADING_MODE={safety['trading_mode']} LIVE_TRADING_ENABLED={safety['live_trading_enabled']} "
          "| observation only, no orders")
    if args.realtime:
        run_realtime(args)
        return
    database = Database(args.db)
    service = HistoricalDataService()
    tracker = LiveSignalTracker(database, square_off=service.research_config.session.square_off)
    try:
        while True:
            now = datetime.now(IST)
            result = run_once(day=args.date, database=database, tracker=tracker, service=service,
                              client=service.client, out_dir=args.out_dir)
            keys = sorted(tracker.pending_keys())
            write_premium_status(args.out_dir, day=args.date, running=args.poll > 0,
                                 stats=result["stats"], candles=result["candles"], now=now,
                                 subscribed_instruments=keys, subscription_status="REST_ONLY")
            print_ready(args.date, result)
            if args.poll <= 0:
                break
            time_mod.sleep(args.poll)
    finally:
        write_premium_status(args.out_dir, day=args.date, running=False, stats=compute_stats(
            database.get_live_signals(args.date.isoformat())), candles=0)


if __name__ == "__main__":
    main()
