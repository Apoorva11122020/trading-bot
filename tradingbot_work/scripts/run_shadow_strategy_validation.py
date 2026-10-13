#!/usr/bin/env python3
"""Run observation-only shadow comparison for S2, SELL-only and SELL-confirmed."""
from __future__ import annotations

import argparse
import sys
import time as time_mod
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s2_1dte_improvement import prepare_inputs
from src.database import Database
from src.paper_trading.shadow_validation import run_once, write_cumulative, write_report
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "shadow_strategy_validation"


def _wr(m: dict) -> str:
    return "N/A" if m["win_rate"] is None else f"{m['win_rate']:.1%}"


def _pf(m: dict) -> str:
    pf = m["profit_factor"]
    return "N/A" if pf is None else ("INF" if pf == float("inf") else f"{pf:.2f}")


def print_result(result: dict, cum: dict, path: Path) -> None:
    data = result["data_status"]
    print(f"PAPER shadow validation | date={result['date']} | no orders")
    print(f"  Current-day source: {data.get('current_day_source')} | history {data.get('historical_window')}")
    for e in data["upstox_events"]:
        print(f"  {e['event']}: path={e.get('path')} code={e.get('code')} http={e.get('status')} "
              f"attempt={e.get('attempt')} retryable={e.get('retryable')}")
    for name, m in result["comparison"]["candidates"].items():
        print(f"  {name}: {m['evaluated']} evaluated, {m['wins']}W/{m['losses']}L, WR {_wr(m)}, PF {_pf(m)}, "
              f"P&L Rs {m['net_pnl']:,.2f}, max DD Rs {m['max_drawdown']:,.2f} | pending {m['pending']}, "
              f"data unavailable {m['data_unavailable']}, skipped {m['skipped']}")
    print(f"  Cumulative ({len(cum['validated_days'])} day(s): {', '.join(cum['validated_days'])}):")
    for name, m in cum["candidates"].items():
        print(f"    {name}: {m['trading_days']}d, {m['signals']} signals, {m['evaluated']} evaluated, "
              f"{m['wins']}W/{m['losses']}L, WR {_wr(m)}, PF {_pf(m)}, P&L Rs {m['net_pnl']:,.2f}, "
              f"max DD Rs {m['max_drawdown']:,.2f}, data unavailable {m['data_unavailable']}")
    print(f"  Report: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 shadow strategy validation; observation only")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now().astimezone().date())
    parser.add_argument("--db", default="data/trading.db")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll", type=int, default=0)
    args = parser.parse_args()

    assert_paper_trading_only()
    db = Database(args.db)
    while True:
        result = run_once(day=args.date, database=db, prepare_inputs_fn=prepare_inputs)
        path = write_report(result, args.out_dir)
        cum = write_cumulative(db, args.out_dir)
        print_result(result, cum, path)
        if args.poll <= 0:
            break
        time_mod.sleep(args.poll)


if __name__ == "__main__":
    main()
