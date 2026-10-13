"""Run full NIFTY + BANKNIFTY strategy research over the maximum V3-available window.

Uses only real Upstox historical candles for currently listed option contracts.
Expired contracts are not fabricated. TRADING_MODE must be PAPER.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.historical_data.models import HistoricalDataError
from src.historical_data.service import HistoricalDataService
from src.safety.live_gate import live_trading_enabled, trading_mode


def main() -> None:
    if live_trading_enabled() or trading_mode() != "PAPER":
        print("Refusing to run unless TRADING_MODE=PAPER and LIVE_TRADING_ENABLED=false")
        sys.exit(1)

    service = HistoricalDataService(store_root=PROJECT_ROOT / "data" / "historical")
    print("Discovering maximum available Upstox V3 research window...")
    availability = service.discover_availability(["NIFTY", "BANKNIFTY"])
    for name, info in availability.items():
        print(f"\n{name}:")
        print(f"  Requested range     : {info['requested_from']} .. {info['requested_to']}")
        print(f"  Index available       : {info.get('index_available_from')} .. {info.get('index_available_to')}")
        print(f"  Option available      : {info.get('option_available_from')} .. {info.get('option_available_to')}")
        print(f"  Research window       : {info.get('research_from')} .. {info.get('research_to')}")
    print(f"\nLimitation: {next(iter(availability.values())).get('limitation', '')}")

    report = service.run_full_research(["NIFTY", "BANKNIFTY"])
    print(f"\nFinal reports written to reports/strategy_research/")
    print(f"  final_comparison.json")
    print(f"  final_comparison.csv")
    print(f"  final_report.md")
    print(f"Run ID: {report.run_id}")


if __name__ == "__main__":
    try:
        main()
    except HistoricalDataError as exc:
        print(f"FAILED: {exc.code}: {exc.message}")
        sys.exit(1)
