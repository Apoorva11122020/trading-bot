#!/usr/bin/env python3
"""S2 forward paper validation collector (observation only, no orders).

Each run records every point-in-time S2 signal of the session (idempotent:
a signal is never stored twice) and resolves pending signals from real Upstox
V3 option candles, then rewrites reports/forward_validation/. Use --poll to
repeat during the session. Signals whose option candles are unavailable are
stored as DATA_UNAVAILABLE; nothing is synthesised.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_mod
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.current_day_s2 import IST, LevelParams, point_in_time_signals, premium_frame, validate_intraday
from src.backtest.regime_filter import PointInTimeRegimeClassifier
from src.database import Database
from src.historical_data.models import HistoricalDataError
from src.historical_data.option_coverage import resolve_signal_contract
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.historical_data.upstox_client import token_source
from src.paper_trading import forward_validation as fv
from src.safety.live_gate import assert_paper_trading_only
from src.session.session_manager import SessionManager

DEFAULT_OUT = ROOT / "reports" / "forward_validation"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
SESSION_CLOSE = "15:30"
HISTORY_DAYS = 45
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
NOTES = [
    "Observation only: no broker, engine or order code is used; nothing is ordered.",
    "Production setup: NIFTY, 5-minute candles, nearest expiry, ATM strike, BUY -> CE, SELL -> PE; S2 unchanged.",
    "Every signal is recorded, including repeated same-direction signals; only real Upstox V3 candles are used.",
    "Signals are generated bar by bar from completed NIFTY candles; RSI/EMA/regime/VWAP use bars up to the signal bar.",
    "Outcome = first touch of the predefined SL or target on real option candles from the entry candle; trailing "
    "not applied. SL and target in the same candle -> AMBIGUOUS. EOD exit at the open of the first candle at/after "
    "square-off.",
    "Missing option candles -> DATA_UNAVAILABLE (no outcome). P&L = gross premium move x 1 lot; costs excluded.",
    "The 2026-10-05 live-signal-accuracy benchmark (live_signal_outcomes, reports/live_signal_accuracy) is a "
    "separate immutable record and is not modified by forward validation.",
    "Historical comparable 1-DTE data: UNAVAILABLE (Upstox expired-instrument API returns PLAN_NOT_SUPPORTED).",
]


def completed(frame: pd.DataFrame, interval: int, now: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    ends = frame["timestamp"] + pd.Timedelta(minutes=interval)
    return frame[ends <= pd.Timestamp(now)].reset_index(drop=True)


def _ist(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[CANDLE_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
    return frame


def underlying_history(service, client, underlying: str, day: date, interval: int) -> pd.DataFrame:
    """Stored candles before ``day`` plus V3 historical candles for the most recent sessions."""
    parts = []
    stored = service.read_underlying(underlying, day - timedelta(days=HISTORY_DAYS), day - timedelta(days=1))
    if stored is not None and not stored.empty:
        parts.append(_ist(stored))
    try:
        recent = premium_frame(None, client.candles(INSTRUMENT_KEY, day - timedelta(days=10),
                                                    day - timedelta(days=1), interval=interval))
        if not recent.empty:
            parts.append(_ist(recent))
    except HistoricalDataError:
        pass
    if not parts:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    frame = pd.concat(parts, ignore_index=True).drop_duplicates("timestamp", keep="last")
    return frame[frame["timestamp"].dt.date < day].sort_values("timestamp").reset_index(drop=True)


def run_once(*, day: date, database: Database, tracker: fv.ForwardValidationTracker,
             service: HistoricalDataService, client, underlying: str = "NIFTY", strategy_id: str = "S2",
             out_dir: Path = DEFAULT_OUT, now: datetime | None = None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    now = now or datetime.now(IST)
    rc, sc = service.research_config, service.strategy_config
    params = LevelParams.from_configs(rc, sc)
    try:
        raw = client.intraday_candles(INSTRUMENT_KEY, interval=params.interval)
    except HistoricalDataError as exc:
        raise SystemExit(f"API_REQUEST_ERROR: {exc.code} (HTTP {exc.status})") from None
    today_candles, validation = validate_intraday(raw, day, params.interval)
    today_candles = completed(today_candles, params.interval, now)

    session = SessionManager(sc.session)
    signals = []
    combined = pd.DataFrame(columns=CANDLE_COLUMNS)
    if not today_candles.empty:
        signals, _ = point_in_time_signals(today_candles, underlying, rc, session, params.stop_new_entries,
                                           strategy_id)
        history = underlying_history(service, client, underlying, day, params.interval)
        combined = pd.concat([history, _ist(today_candles)], ignore_index=True)
        combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        combined["symbol"] = underlying.upper()

    premium_cache: dict[str, object] = {}

    def premium_for(contract):
        return premium_for_key(contract.instrument_key, contract.expiry)

    def premium_for_key(key: str, expiry: date):
        if key in premium_cache:
            return premium_cache[key]
        try:
            today = [c for c in client.intraday_candles(key, interval=params.interval)
                     if c.timestamp.astimezone(IST).date() == day]
        except HistoricalDataError as exc:
            premium_cache[key] = f"PREMIUM_API_ERROR:{exc.code}"
            return premium_cache[key]
        stored = service.store.read("options", underlying.upper(), key, expiry=expiry)
        source = "STORED_UPSTOX_V3+UPSTOX_V3_INTRADAY"
        if stored is not None and not stored.empty:
            stored = stored[pd.to_datetime(stored["timestamp"], utc=True).dt.tz_convert(IST).dt.date < day]
        if stored is None or stored.empty:
            stored, source = None, "UPSTOX_V3_INTRADAY"
            try:
                prior = client.candles(key, day - timedelta(days=10), day - timedelta(days=1),
                                       interval=params.interval)
                if prior:
                    stored, source = premium_frame(None, prior), "UPSTOX_V3_HISTORICAL+UPSTOX_V3_INTRADAY"
            except HistoricalDataError:
                pass
        frame = completed(premium_frame(stored, today), params.interval, now)
        frame.attrs["source"] = source if today else source.replace("+UPSTOX_V3_INTRADAY", "")
        premium_cache[key] = frame
        return frame

    def resolve(sig):
        return resolve_signal_contract(sig, service.resolver_factory(sig.timestamp.date()), rc)

    pit = PointInTimeRegimeClassifier(combined) if not combined.empty else None
    records = fv.build_signals(
        signals, combined, resolve=resolve, premium_for=premium_for, params=params,
        s2_cfg=rc.strategies[strategy_id], regime_at=(pit.regimes_at if pit else (lambda _ts: [])),
        underlying=underlying, strategy=strategy_id,
    ) if signals else []
    created = tracker.record_all(records)

    close = datetime.combine(day, datetime.strptime(SESSION_CLOSE, "%H:%M").time(), IST)
    session_complete = now >= close + timedelta(minutes=params.interval)
    frames = {k: v for k, v in premium_cache.items() if isinstance(v, pd.DataFrame)}
    for row in database.get_forward_signals(day.isoformat()):
        key = row["instrument_key"]
        if row["status"] == fv.PENDING and key and key not in frames:
            frame = premium_for_key(key, date.fromisoformat(row["expiry"]))
            if isinstance(frame, pd.DataFrame):
                frames[key] = frame
    resolved = tracker.resolve_pending(day.isoformat(), frames, session_complete)

    all_rows = database.get_forward_signals()
    paths = fv.write_reports(all_rows, out_dir, day.isoformat(), NOTES)
    day_rows = database.get_forward_signals(day.isoformat())
    day_path = out_dir / f"{day.isoformat()}.json"
    day_path.write_text(json.dumps({
        "date": day.isoformat(),
        "mode": "PAPER observation only - no orders placed",
        "session_complete": session_complete,
        "stats": fv.metrics(day_rows),
        "signals": [fv.public_row(r) for r in day_rows],
    }, indent=2, default=str), encoding="utf-8")
    paths["day_json"] = day_path
    return {"stats": fv.forward_stats(all_rows, day.isoformat()), "paths": paths, "new_signals": created,
            "resolved_now": len(resolved), "candles": len(today_candles), "received": validation.received,
            "session_complete": session_complete}


def print_ready(day: date, result: dict) -> None:
    s, t = result["stats"], result["stats"]["today"]
    pf = f"{s['profit_factor']:.2f}" if s["profit_factor"] is not None else "N/A"
    print("\nS2 FORWARD PAPER VALIDATION READY")
    print(f"  Session / token source : {day} / {token_source()}")
    print(f"  Completed candles      : {result['candles']} (new signals {result['new_signals']}, "
          f"resolved this run {result['resolved_now']}, session complete {result['session_complete']})")
    print(f"  Today                  : {t['signals']} signals, {t['wins']}W / {t['losses']}L, "
          f"{t['data_unavailable']} data unavailable, {t['pending']} pending")
    print(f"  Cumulative             : {s['signals']} signals, {s['resolved']} resolved, {s['wins']}W / "
          f"{s['losses']}L over {s['days_tracked']} day(s)")
    print(f"  Win rate               : {s['accuracy_display']} - {s['accuracy_label']}")
    print(f"  Profit factor          : {pf}")
    print(f"  Net P&L (1 lot)        : Rs {s['net_pnl']:,.2f}   Max drawdown: Rs {s['max_drawdown']:,.2f}")
    print(f"  Sample status          : {s['sample_status']}")
    print(f"  {fv.HISTORICAL_COMPARABLE_STATUS}")
    for path in result["paths"].values():
        print(f"  Report                 : {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 forward paper validation (observation only)")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date())
    parser.add_argument("--db", default=os.getenv("DATABASE_PATH", "data/trading.db"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll", type=int, default=0, help="repeat every N seconds (0 = run once)")
    args = parser.parse_args()

    safety = fv.verify_observation_only()
    print(f"TRADING_MODE={safety['trading_mode']} LIVE_TRADING_ENABLED={safety['live_trading_enabled']} "
          "| observation only, no orders")
    database = Database(args.db)
    service = HistoricalDataService()
    tracker = fv.ForwardValidationTracker(database, square_off=service.research_config.session.square_off)
    while True:
        result = run_once(day=args.date, database=database, tracker=tracker, service=service,
                          client=service.client, out_dir=args.out_dir)
        print_ready(args.date, result)
        if args.poll <= 0:
            break
        time_mod.sleep(args.poll)


if __name__ == "__main__":
    main()
