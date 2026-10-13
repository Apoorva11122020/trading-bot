#!/usr/bin/env python3
"""ROBUSTNESS COMPARISON LAB: CURRENT_S2 vs SELL + {BUY_BREAKOUT_CONFIRM, BUY_NON_CHOPPY_ONLY, BUY_MORNING_ONLY}.

PAPER / research only, no orders, production S2 untouched, no thresholds re-optimised. Same historical replay and
chronological split as the strategy improvement lab; 2026-10-05 onwards is never used. Reports go to
reports/strategy_improvement/ (robustness_comparison.*, walk_forward_comparison.csv, day_stability.csv,
regime_stability.csv).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strategy_improvement_lab import replay_inputs
from src.backtest import robustness_comparison as rc
from src.backtest import strategy_improvement_lab as lab
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement"


def run(*, out_dir: Path = DEFAULT_OUT, today=None) -> dict:
    r = replay_inputs(today=today)
    with order_placement_blocked(), capture_upstox_events():
        a = rc.build_comparison(base_rows=r.base_rows, facts=r.facts, days=r.result["days"], splits=r.splits,
                                lookahead=r.lookahead, baseline_lookahead_passed=r.baseline_lookahead_passed,
                                forward=r.forward)
    paths = rc.write_reports(a, out_dir)
    print_result(a, paths)
    return a


def print_result(a: dict, paths: dict) -> None:
    pct, pf, rs = lab._pct, lab._pf, lab._rs
    print(f"\n{rc.HEADING} | PAPER | no orders | production S2 unchanged | no re-optimisation")
    for sp in a["splits"].values():
        print(f"  {sp['label']}: {sp['from']}..{sp['to']} ({len(sp['days'])} sessions)")
    print("\nConfig | Scope | Train n/WR/PF | Val n/WR/PF | OOS n/WR/PF/P&L/DD | Avg R (OOS) | MaxLossStreak | "
          "Best-day share (OOS) | WF+ | Status")
    for key, r in a["results"].items():
        n, s = key.split("|")
        t, v, o = r["TRAIN"], r["VALIDATION"], r["OOS"]
        print(f"{n} | {s} | {t['evaluated']}/{pct(t['win_rate'])}/{pf(t['profit_factor'])} | "
              f"{v['evaluated']}/{pct(v['win_rate'])}/{pf(v['profit_factor'])} | "
              f"{o['evaluated']}/{pct(o['win_rate'])}/{pf(o['profit_factor'])}/{rs(o['net_pnl'])}/{rs(o['max_drawdown'])}"
              f" | {o['average_r']} | {r['ALL']['max_consecutive_losses']} | {pct(o['best_day_share'])} | "
              f"{pct(r['wf_positive_share'])} | {r['status']}")
    print("\nBUY leg (what the BUY filter adds on top of SELL-only):")
    for n, v in a["verdicts"]["configs"].items():
        print(f"  {n}: OOS BUY trades {v['buy_leg_trades']['OOS']}, BUY P&L Rs {rs(v['buy_leg_pnl']['OOS'])}; "
              f"ALL BUY trades {v['buy_leg_trades']['ALL']}, BUY P&L Rs {rs(v['buy_leg_pnl']['ALL'])} | "
              f"{'ELIGIBLE' if v['eligible'] else 'not eligible: ' + ', '.join(v['failed'])}")
    print("\nLookahead: " + ", ".join(f"{n}={'PASS' if ok else 'FAIL'}" for n, ok in a["lookahead_passed"].items()))
    print(f"\nRECOMMENDATION: {a['recommendation']}" + (f" ({a['verdicts']['candidate']})"
                                                       if a["verdicts"]["candidate"] else ""))
    print("Production S2 was NOT changed. TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, zero orders placed.")
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--today", type=date.fromisoformat, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    run(out_dir=a.out_dir, today=a.today)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
