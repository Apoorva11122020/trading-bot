#!/usr/bin/env python3
"""LIVE QUALITY GATE V1 daily + cumulative report from stored shadow rows (PAPER, observation only, no orders)."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.live_quality_gate import QualityGateConfig
from src.database import Database
from src.paper_trading import live_quality_report as rpt
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "live_quality_gate"


def build(day: str, db: Database, out_dir: Path, config: QualityGateConfig) -> tuple[dict, dict]:
    daily = rpt.daily_report(db, day, config, run_status=rpt.read_run_status(out_dir, day))
    rpt.write_daily(daily, out_dir)
    cum = rpt.cumulative_report(db, config)
    rpt.write_cumulative(cum, out_dir)
    return daily, cum


def main() -> None:
    parser = argparse.ArgumentParser(description="LIVE QUALITY GATE V1 report; observation only")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now().astimezone().date())
    parser.add_argument("--db", default="data/trading.db")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-live-samples", type=int, default=QualityGateConfig.min_live_samples)
    args = parser.parse_args()

    assert_paper_trading_only()
    config = QualityGateConfig(min_live_samples=args.min_live_samples)
    daily, _ = build(args.date.isoformat(), Database(args.db), args.out_dir, config)
    print(rpt.console_summary(daily))
    print(f"Reports: {args.out_dir / (args.date.isoformat() + '.md')} | {args.out_dir / 'cumulative.md'}")


if __name__ == "__main__":
    main()
