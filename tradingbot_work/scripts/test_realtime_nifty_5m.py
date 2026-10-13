#!/usr/bin/env python3
"""Safe live validation: NIFTY 50 5m candles + S2 signals (no orders)."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from src.paper_trading.config import load_paper_trading_settings
from src.paper_trading.upstox_feed import create_upstox_v3_feed
from src.safety.live_gate import assert_paper_trading_only, live_trading_enabled, trading_mode
from src.validation.realtime_nifty_5m import RealtimeNifty5mValidator

NIFTY_INDEX = "NSE_INDEX|Nifty 50"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate real-time NIFTY 5m candles + S2 signals")
    parser.add_argument(
        "--seconds",
        type=int,
        default=int(os.getenv("UPSTOX_MARKET_DATA_SECONDS", "120")),
        help="How long to stream market data",
    )
    parser.add_argument("--instrument", default=NIFTY_INDEX)
    args = parser.parse_args()

    assert_paper_trading_only()
    settings = load_paper_trading_settings()
    if args.instrument != settings.instrument_key:
        settings = settings.model_copy(update={"instrument_key": args.instrument})

    print("=== Real-time NIFTY 5m validation (NO ORDERS) ===")
    print(f"TRADING_MODE={trading_mode()}")
    print(f"LIVE_TRADING_ENABLED={live_trading_enabled()}")
    print(f"instrument={settings.instrument_key}")
    print(f"timeframe={settings.timeframe}")
    print(f"strategy={settings.strategy_id}")
    print(f"duration_seconds={args.seconds}")
    print("PAPER ONLY | place_order() will NOT be called")
    print("===============================================\n")

    feed = create_upstox_v3_feed(stale_after_seconds=settings.stale_data_seconds)
    session_id = uuid.uuid4().hex[:12]
    validator = RealtimeNifty5mValidator(feed=feed, settings=settings)

    def _shutdown(*_):
        validator.stop()
        report = validator.write_report(session_id)
        summary = validator.summary()
        print("\n=== Validation summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print(f"report: {report}")
        print(f"[DIAG] {validator.format_diagnostics_line()}")
        print("orders_placed: 0 (confirmed)")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    validator.start()
    health = feed.health()
    print("WebSocket worker started.")
    print(f"subscription_symbols={[settings.instrument_key]}")
    print(f"initial_market_data_status={health.status.value}")
    print("Streaming… Press Ctrl+C to stop early.\n")
    deadline = time.time() + args.seconds
    last_diag = 0.0
    try:
        while time.time() < deadline:
            time.sleep(1)
            now = time.time()
            if now - last_diag >= 10:
                last_diag = now
                print(f"[DIAG] {validator.format_diagnostics_line()}", flush=True)
    except KeyboardInterrupt:
        pass
    _shutdown()


if __name__ == "__main__":
    main()
