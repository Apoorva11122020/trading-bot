#!/usr/bin/env python3
"""S2 direction comparison research (read-only, PAPER only, no orders, production S2 unchanged)."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s2_1dte_improvement import prepare_inputs
from src.backtest.s2_direction_research import run_direction_research, write_reports
from src.historical_data.service import _refuse_live
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_direction_research"


def run(*, day: date, hist_from: date, hist_to: date, out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY",
        download_missing_options: bool = False) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    inputs = prepare_inputs(day=day, hist_from=hist_from, hist_to=hist_to, underlying=underlying,
                            download_missing_options=download_missing_options)
    report = run_direction_research(**inputs["research"])
    report["data"] = inputs["data"]
    report["point_in_time_matches_full_run"] = inputs["point_in_time_matches_full_run"]
    paths = write_reports(report, out_dir)
    for p in paths.values():
        print(f"Wrote {p}")
    print()
    for line in report["summary"]["lines"]:
        print(line)
    print("\n" + report["summary"]["recommendation"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 direction comparison research (read-only)")
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
