#!/usr/bin/env python3
"""BUY STRATEGY RESEARCH LAB: 12 pre-declared BUY confirmation candidates on the original S2 BUY signal.

PAPER / research only, no orders. Uses the project's chronological split (TRAIN 2026-09-01..09-17, VALIDATION
09-18..09-24, OOS 09-25..10-01) and the historical shadow replay pipeline (real cached NIFTY candles and real expired
option contracts/candles). Reports go to reports/buy_strategy_lab/. Nothing is applied to production S2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_shadow_historical_validation as hv
from src.backtest import buy_strategy_lab as lab
from src.backtest import s2_1dte_improvement as imp
from src.historical_data.service import _refuse_live
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "buy_strategy_lab"


def run(*, out_dir: Path = DEFAULT_OUT, splits: dict | None = None, criteria: lab.LabCriteria = lab.LabCriteria(),
        today=None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    splits = imp.split_dates(splits or imp.DEFAULT_SPLITS)
    result, meta = hv.build_replay(from_date=splits["TRAIN"][0], to_date=splits["OOS"][1], today=today,
                                   keep_pipeline=True)
    with order_placement_blocked(), capture_upstox_events():
        a = lab.build_lab(result, meta, splits=splits, criteria=criteria)
    paths = lab.write_reports(a, out_dir)
    print_result(a, paths)
    return a


def print_result(a: dict, paths: dict) -> None:
    pct, pf, rs = lab._pct, lab._pf, lab._rs
    print(f"{lab.HEADING} | PAPER | no orders | production S2 unchanged")
    for sp in a["splits"].values():
        print(f"  {sp['label']}: {sp['from']}..{sp['to']} ({len(sp['days'])} sessions)")
    print(f"  BUY_BASELINE identical to original S2 BUY: {a['baseline_matches_original_s2_buy']}")
    print()
    print("Candidate | Train WR | Validation WR | OOS WR | OOS PF | OOS P&L | OOS DD | OOS Trades | Status")
    rows = {r["candidate"]: r for r in a["final_table"]}
    for n in a["ranking_by_oos"]:
        r = rows[n]
        print(f"{n} | {pct(r['train_wr'])} | {pct(r['validation_wr'])} | {pct(r['oos_wr'])} | {pf(r['oos_pf'])} | "
              f"{rs(r['oos_pnl'])} | {rs(r['oos_dd'])} | {r['oos_trades']} | {r['status']}")
    base_all = a["metrics"]["ALL"]["BUY_BASELINE"]
    base_oos = a["metrics"]["OOS"]["BUY_BASELINE"]
    print()
    print("BASE BUY (TRAIN+VALIDATION+OOS):")
    print(f"{pct(base_all['win_rate'])} WR / PF {pf(base_all['profit_factor'])} / Rs {rs(base_all['net_pnl'])}")
    print(f"BASE BUY (OOS): {pct(base_oos['win_rate'])} WR / PF {pf(base_oos['profit_factor'])} / "
          f"Rs {rs(base_oos['net_pnl'])} / {base_oos['evaluated']} trades")
    best = a["best_candidate"]
    print()
    print("BEST BUY CANDIDATE:")
    if not best:
        print(f"None - no candidate has >= {a['criteria']['min_oos_trades']} OOS trades")
    else:
        for period, c in a["comparison"].items():
            x = c["candidate"]
            print(f"{best} ({period}): {pct(x['win_rate'])} WR / PF {pf(x['profit_factor'])} / Rs {rs(x['net_pnl'])} "
                  f"/ DD Rs {rs(x['max_drawdown'])} / {x['evaluated']} trades")
            print(f"  WR improvement: {pct(c['baseline']['win_rate'])} -> {pct(x['win_rate'])} = "
                  f"{lab._signed(c['wr_change_pp'], 'pp')}")
            print(f"  PF improvement: {lab._signed(c['pf_change'], 'pf')}")
            print(f"  P&L improvement: Rs {lab._signed(c['pnl_change'])}")
            print(f"  DD change: Rs {lab._signed(c['dd_change'])} | trade-count change: {c['trade_count_change']:+d}")
    failed_la = [n for n, r in a["lookahead"].items() if not r["passed"]]
    print()
    print("Lookahead: " + ("PASS for all 12 candidates" if not failed_la else f"FAIL: {', '.join(failed_la)}"))
    if best:
        st = a["status"][best]
        print(f"Best candidate status: {st['status']}" + (f" (failed: {', '.join(st['failed_checks'])})"
                                                         if st["failed_checks"] else ""))
    print(a["never_robust_note"])
    print("Production S2 and SELL_ONLY were NOT changed. Zero live orders were placed. Best candidate NOT applied.")
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run(out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
