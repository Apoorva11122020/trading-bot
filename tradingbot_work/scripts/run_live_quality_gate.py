#!/usr/bin/env python3
"""Daily LIVE QUALITY GATE V1 shadow run (PAPER ONLY, observation only, no broker orders).

1. assert paper-only  2. load the unchanged S2 pipeline  3. generate normal S2 contexts
4. apply LIVE_QUALITY_GATE_V1  5. evaluate accepted signals with the existing shadow evaluation
6. persist to shadow_strategy_signals  7. daily report  8. cumulative report
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_quality_report import DEFAULT_OUT, build
from scripts.run_s2_1dte_improvement import prepare_inputs
from src.backtest.live_quality_gate import DATA_UNAVAILABLE, GATE_VERSION, QualityGateConfig
from src.database import Database
from src.paper_trading import live_quality_report as rpt
from src.paper_trading.shadow_validation import run_once
from src.safety.live_gate import assert_paper_trading_only


def run(day: date, db: Database, out_dir: Path, config: QualityGateConfig, prepare=prepare_inputs) -> dict:
    assert_paper_trading_only()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = rpt.run_status_path(out_dir, day.isoformat())
    try:
        result = run_once(day=day, database=db, prepare_inputs_fn=prepare, quality_gate=config)
        status = {"status": "VALIDATED", "data_status": result.get("data_status"),
                  "point_in_time_matches_full_run": result.get("point_in_time_matches_full_run")}
    except SystemExit as exc:
        status = {"status": DATA_UNAVAILABLE, "reason": str(exc.code or exc)}
    status.update(date=day.isoformat(), gate_version=GATE_VERSION, live_orders=0,
                  ran_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    daily, _ = build(day.isoformat(), db, out_dir, config)
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description="LIVE QUALITY GATE V1 daily shadow run; observation only")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now().astimezone().date())
    parser.add_argument("--db", default="data/trading.db")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    assert_paper_trading_only()
    daily = run(args.date, Database(args.db), args.out_dir, QualityGateConfig())
    print(rpt.console_summary(daily))
    print(f"Reports: {args.out_dir / (args.date.isoformat() + '.md')} | {args.out_dir / 'cumulative.md'}")


if __name__ == "__main__":
    main()
