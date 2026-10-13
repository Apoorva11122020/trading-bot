#!/usr/bin/env python3
"""Realtime NIFTY 50 5m S2 signal-accuracy tracker (PAPER, observation only, no orders).

Streams real ticks from the existing Upstox V3 WebSocket feed, builds 5-minute
IST candles, evaluates the unchanged S2 generator on completed candles only,
and resolves every signal from subsequent real prices. Reports go to
reports/realtime_signal_accuracy/.

    python scripts/run_live_signal_accuracy.py --seconds 3600
    python scripts/run_live_signal_accuracy.py --minutes 60
    python scripts/run_live_signal_accuracy.py --until 14:30

``--mode premium-replay`` runs the earlier option-premium REST tracker
(scripts/run_live_premium_signal_accuracy.py) with its own arguments.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import threading
import time as time_mod
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.safety.live_gate import LiveTradingSafetyError, assert_paper_trading_only, live_trading_enabled, trading_mode
from src.signal_accuracy.realtime import DEFAULT_OUT, LiveSignalAccuracyRunner, TrackerSettings, create_feed
from src.signal_accuracy.tracker import IST, pct, profit_factor_display

BAR = "=" * 50
LEGACY_SCRIPT = ROOT / "scripts" / "run_live_premium_signal_accuracy.py"
PREMIUM_OUT = ROOT / "reports" / "live_signal_accuracy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime NIFTY 5m S2 signal-accuracy tracker (PAPER, no orders)")
    parser.add_argument("--mode", choices=("realtime", "premium-replay"), default="realtime")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--seconds", type=int, help="run for N seconds")
    duration.add_argument("--minutes", type=float, help="run for N minutes")
    duration.add_argument("--until", help="run until HH:MM IST today")
    parser.add_argument("--session-id", help="defaults to the IST date (YYYYMMDD); reuse it to resume after a restart")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--instrument", default=None, help="instrument key (default from config/paper_trading.yaml)")
    parser.add_argument("--tracking-every", type=int, default=30, help="seconds between [TRACKING] lines")
    parser.add_argument("--with-premium", action=argparse.BooleanOptionalAction, default=True,
                        help="start option-premium WebSocket tracker alongside NIFTY spot (default: on)")
    parser.add_argument("--premium-poll", type=int, default=30, help="REST backfill interval for option premium")
    parser.add_argument("--db", default=os.getenv("DATABASE_PATH", "data/trading.db"))
    return parser


def deadline_for(args, now: datetime, market_close: str) -> datetime:
    if args.seconds is not None:
        if args.seconds <= 0:
            raise ValueError("--seconds must be > 0")
        return now + timedelta(seconds=args.seconds)
    if args.minutes is not None:
        if args.minutes <= 0:
            raise ValueError("--minutes must be > 0")
        return now + timedelta(minutes=args.minutes)
    until = args.until or market_close
    end = datetime.combine(now.date(), time.fromisoformat(until), IST)
    if end <= now:
        raise ValueError(f"{until} IST has already passed; use --seconds/--minutes or a later --until")
    return end


def run_legacy(args, rest: list[str]) -> int:
    argv = list(rest)
    if args.out_dir is not None:
        argv += ["--out-dir", str(args.out_dir)]
    sys.argv = [str(LEGACY_SCRIPT), *argv]
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")
    return 0


def print_final(runner: LiveSignalAccuracyRunner, summary: dict, paths: dict) -> None:
    s, dq = summary, runner.diagnostics()
    print("\n" + BAR)
    print("S2 LIVE PAPER SIGNAL ACCURACY")
    print(BAR)
    print(f"Session:        {runner.session_id} ({s['session_date']})")
    print(f"Instrument:     {s['instrument']} ({runner.settings.instrument_key})")
    print(f"Timeframe:      {s['timeframe']}")
    print(f"Strategy:       {s['strategy']}")
    print()
    print(f"Total Signals:  {s['total_signals']}")
    print(f"BUY:            {s['buy_signals']}")
    print(f"SELL:           {s['sell_signals']}")
    print()
    print(f"Resolved:       {s['resolved_signals']}")
    print(f"Unresolved:     {s['unresolved_signals']}")
    print(f"Wins:           {s['wins']}")
    print(f"Losses:         {s['losses']}")
    print(f"Ambiguous:      {s['ambiguous']}")
    print(f"Time Exit:      {s['time_exits']}")
    print()
    print(f"Accuracy:       {pct(s['accuracy'])}")
    print(f"BUY Accuracy:   {pct(s['buy_accuracy'])}")
    print(f"SELL Accuracy:  {pct(s['sell_accuracy'])}")
    print()
    print(f"Total Points:   {s['total_points']:.2f}")
    print(f"Profit Factor:  {profit_factor_display(s)}")
    print(f"Max Drawdown:   {s['max_drawdown_points']:.2f} points")
    print()
    print(f"Data quality:   ws_connected={dq['websocket_connected']} ticks={dq['ticks_received']} "
          f"candles={dq['completed_candles']} incomplete={dq['incomplete_candles']} "
          f"stale={dq['stale_events']} disconnects={dq['disconnect_events']}")
    print(f"No Orders:      YES - LIVE ORDERS = {dq['orders_placed']}")
    print(f"PAPER MODE:     TRADING_MODE={trading_mode()}")
    print(f"LIVE TRADING:   LIVE_TRADING_ENABLED={str(live_trading_enabled()).lower()}")
    print()
    print(f"Report:         {paths['summary_md']}")
    for key in ("summary_json", "signals_csv", "session_json", "jsonl"):
        print(f"                {paths[key]}")
    print(BAR)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if args.mode == "premium-replay":
        return run_legacy(args, rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    load_dotenv(ROOT / ".env", override=False)
    try:
        assert_paper_trading_only()
    except LiveTradingSafetyError as exc:
        print(f"REFUSED TO START (fail closed): {exc}")
        print("Set TRADING_MODE=PAPER and LIVE_TRADING_ENABLED=false.")
        return 2

    settings = TrackerSettings.load()
    if args.instrument:
        settings = replace(settings, instrument_key=args.instrument)
    now = datetime.now(IST)
    try:
        deadline = deadline_for(args, now, settings.market_close)
    except ValueError as exc:
        parser.error(str(exc))

    print(BAR)
    print("S2 LIVE PAPER SIGNAL ACCURACY TRACKER")
    print("PAPER ONLY")
    print("NO ORDERS WILL BE PLACED")
    print(BAR)
    print(f"TRADING_MODE={trading_mode()}")
    print(f"LIVE_TRADING_ENABLED={str(live_trading_enabled()).lower()}")
    print(f"instrument={settings.instrument_key} timeframe={settings.timeframe} strategy={settings.strategy_id}")
    print(f"session window={settings.market_open}-{settings.market_close} entries "
          f"{settings.allow_entries_from}-{settings.stop_new_entries} square_off={settings.square_off} IST")
    print(f"run until={deadline:%Y-%m-%d %H:%M:%S} IST")
    print(BAR + "\n")

    try:
        feed = create_feed(stale_after_seconds=settings.stale_data_seconds)
    except Exception as exc:  # noqa: BLE001 - report the failure without exposing credentials
        print(f"MARKET DATA FEED UNAVAILABLE: {type(exc).__name__}: {exc}")
        return 1

    runner = LiveSignalAccuracyRunner(feed, settings=settings, out_dir=args.out_dir or DEFAULT_OUT,
                                      session_id=args.session_id)
    premium_thread = None
    if args.with_premium:
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_live_premium_signal_accuracy", LEGACY_SCRIPT)
        premium_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(premium_mod)

        premium_thread = threading.Thread(
            target=premium_mod.premium_realtime_worker,
            kwargs={"db": args.db, "out_dir": PREMIUM_OUT, "day": now.date(),
                    "poll": args.premium_poll, "until": deadline},
            daemon=True,
            name="option-premium-realtime",
        )
        premium_thread.start()
        print("Option premium realtime tracker started (WebSocket + REST backfill).")
    runner.start()
    print(f"Session {runner.session_id} started; WebSocket worker running. Press Ctrl+C to stop early.\n")
    last_report = time_mod.monotonic()
    try:
        while datetime.now(IST) < deadline:
            time_mod.sleep(1)
            runner.poll()
            if time_mod.monotonic() - last_report >= max(1, args.tracking_every):
                last_report = time_mod.monotonic()
                runner.print_tracking()
                print(f"[DIAG] {runner.diagnostics_line()}", flush=True)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C)...")
    summary, paths = runner.finish()
    print_final(runner, summary, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
