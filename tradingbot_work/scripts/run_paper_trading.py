#!/usr/bin/env python3
"""Start real-time NIFTY S2 paper trading (PAPER mode only)."""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from src.database import Database
from src.paper_trading.config import load_paper_trading_settings
from src.paper_trading.nifty_s2_runner import NiftyS2PaperRunner
from src.paper_trading.upstox_feed import create_upstox_v3_feed
from src.safety.live_gate import assert_paper_trading_only


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 real-time paper trading")
    parser.add_argument("--database", default="data/trading.db")
    parser.add_argument("--feed", choices=["upstox"], default="upstox", help="Real-time feed (no synthetic prices)")
    args = parser.parse_args()

    assert_paper_trading_only()
    settings = load_paper_trading_settings()
    database = Database(args.database)
    feed = create_upstox_v3_feed(stale_after_seconds=settings.stale_data_seconds)
    runner = NiftyS2PaperRunner(database=database, feed=feed, settings=settings)

    def _shutdown(*_):
        runner.stop()
        database.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    runner.start()
    print("Paper trading running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
