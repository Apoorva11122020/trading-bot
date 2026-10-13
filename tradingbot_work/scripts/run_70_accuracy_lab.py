#!/usr/bin/env python3
"""70%+ OOS ACCURACY STRATEGY RESEARCH LAB (research only, PAPER, no orders, production S2 untouched).

Pre-declared strategy families (trend, breakout, mean reversion, volatility, price action, session structure, plus
option-premium overlays) on real cached NIFTY candles and real option candles, evaluated with the unchanged S2
execution model. TRAIN 2026-09-01..09-17 and VALIDATION 09-18..09-24 select; OOS 09-25..10-01 only evaluates.
Current-day data is never loaded. Reports go to reports/strategy_improvement/.
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
from src.backtest import accuracy_70_lab as acc
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.historical_data.option_coverage import entry_allowed
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.safety.live_gate import assert_paper_trading_only
from src.session.session_manager import SessionManager

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement"


def run(*, out_dir: Path = DEFAULT_OUT, today=None, skip_mutation: bool = False) -> dict:
    assert_paper_trading_only()
    r = replay_inputs(today=today)
    x = r.x
    session = SessionManager(x.service.strategy_config.session)

    def allowed(ts):
        return entry_allowed(ts.to_pydatetime(), session, x.params.stop_new_entries)

    with order_placement_blocked(), capture_upstox_events():
        a = acc.build_lab70(
            spot=r.result["pipeline"]["spot"], split_days=r.split_days, splits=r.splits,
            resolve=sh.strict_resolver(x.resolve, x.check), premium_for=x.premium_for, params=x.params,
            square_off=x.rc.session.square_off, underlying=x.name, allowed=allowed, base_rows=r.base_rows,
            facts=r.facts, baseline_lookahead_passed=r.baseline_lookahead_passed,
            rules=lab.s2_rules(x.rc, x.params, x.s2), today=x.today, run_mutation=not skip_mutation,
            interval=x.params.interval, progress=lambda m: print(f"  {m}", flush=True))
    paths = acc.write_reports(a, out_dir)
    print_summary(a, paths)
    return a


def _scope(name: str, m: dict | None, status: str) -> str:
    if not m:
        return f"{name:<9} | no eligible finalist | Status: {status}"
    return (f"{name:<9} | OOS accuracy {sh._pct(m['win_rate'])} | trades {m['evaluated']} | "
            f"PF {sh._pf_s(m['profit_factor'])} | P&L Rs {sh._rs(m['net_pnl'])} | Status: {status}")


def print_summary(a: dict, paths: dict) -> None:
    sc, fin = a["scores"], a["finalists"]
    print(f"\n{acc.HEADING} | PAPER | research only | no orders")
    print(f"Candidates: {a['counts']['candidates']} ({a['counts']['data_unavailable']} base DATA_UNAVAILABLE) | "
          f"unique signals {a['counts']['unique_signals']}")
    for d in acc.DIRECTIONS:
        n = fin[d]
        print(_scope(d, sc[n]["OOS"] if n else None, sc[n]["status"] if n else "NO_ELIGIBLE_CANDIDATE")
              + (f" | {n}" if n else ""))
    comb = a["combined"]
    print(_scope("COMBINED", comb["OOS"] if comb else None, "FINALISTS_TOGETHER" if comb else "-"))
    for d in acc.DIRECTIONS:
        m = a["baseline"][d]["OOS"]
        print(f"  (S2 baseline {d}: OOS accuracy {sh._pct(m['win_rate'])}, trades {m['evaluated']}, "
              f"PF {sh._pf_s(m['profit_factor'])}, P&L Rs {sh._rs(m['net_pnl'])})")
    print(f"\n70% TARGET: {'ACHIEVED' if a['target_achieved'] else 'NOT ACHIEVED'}")
    if not a["target_achieved"]:
        print(acc.NOT_ACHIEVED_TEXT)
    final = {acc.ACCEPT: "ACCEPT", acc.PROMISING: "PROMISING", acc.KEEP: "KEEP_CURRENT", acc.REJECT: "REJECT"}
    print(f"FINAL: {final[a['recommendation']]} ({a['recommendation']})")
    print("Production S2 changed: NO")
    print("Live orders: 0")
    print(f"Lookahead: {'PASS' if a['lookahead']['passed'] else 'FAIL'} | Integrity: "
          f"{'PASS' if a['integrity_ok'] else 'FAIL'}")
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-mutation", action="store_true",
                   help="skip the slower pipeline mutation test (truncation and option tests still run)")
    p.add_argument("--today", type=date.fromisoformat, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    run(out_dir=a.out_dir, today=a.today, skip_mutation=a.skip_mutation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
