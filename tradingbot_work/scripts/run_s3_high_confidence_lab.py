#!/usr/bin/env python3
"""S3_HIGH_CONFIDENCE research lab (PAPER | RESEARCH ONLY | NO ORDERS; production S2 untouched).

Causal regime + separate BUY/SELL confidence score (0-100) + option premium confirmation + entry confirmation modes,
evaluated on real cached NIFTY and option candles with the unchanged S2 execution model. TRAIN 2026-09-01..09-17
and VALIDATION 09-18..09-24 select; OOS 09-25..10-01 only evaluates. Forward data is never loaded.
Reports go to reports/strategy_improvement/.
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
from src.backtest import s3_high_confidence_lab as s3lab
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
        a = s3lab.build_s3(
            spot=r.result["pipeline"]["spot"], split_days=r.split_days, splits=r.splits,
            resolve=sh.strict_resolver(x.resolve, x.check), premium_for=x.premium_for, params=x.params,
            square_off=x.rc.session.square_off, underlying=x.name, allowed=allowed, base_rows=r.base_rows,
            baseline_lookahead_passed=r.baseline_lookahead_passed, rules=lab.s2_rules(x.rc, x.params, x.s2),
            today=x.today, run_mutation=not skip_mutation, interval=x.params.interval,
            progress=lambda m: print(f"  {m}", flush=True))
    paths = s3lab.write_reports(a, out_dir)
    print_summary(a, paths)
    return a


def _scope(name: str, m: dict | None, status: str) -> list[str]:
    if not m:
        return [f"{name}:", "  no eligible candidate (TRAIN+VALIDATION selection)", f"  Status: {status}"]
    return [f"{name}:", f"  OOS accuracy: {sh._pct(m['win_rate'])}", f"  Trades: {m['evaluated']}",
            f"  PF: {sh._pf_s(m['profit_factor'])}", f"  P&L: Rs {sh._rs(m['net_pnl'])}",
            f"  Max DD: Rs {sh._rs(m['max_drawdown'])}", f"  Status: {status}"]


def print_summary(a: dict, paths: dict) -> None:
    sc, fin = a["scores"], a["finalists"]
    print(f"\n{s3lab.HEADING} | {s3lab.BANNER}")
    print(f"Candidates: {a['counts']['candidates']} | scored candles {a['counts']['scored_candles']} | "
          f"VWAP source {a['counts']['vwap_source']}")
    lines = []
    for d in ("BUY", "SELL"):
        n = fin[d]
        lines += _scope(f"S3 {d}" + (f" ({n})" if n else ""), sc[n]["OOS"] if n else None,
                        sc[n]["status"] if n else s3lab.INSUFFICIENT_DATA)
    comb = a["combined"]
    lines += _scope("S3 COMBINED", comb["OOS"] if comb else None, a["recommendation"] if comb else s3lab.INSUFFICIENT_DATA)
    print("\n".join(lines))
    for d in ("BUY", "SELL", "COMBINED"):
        m = a["s2"][d]["OOS"]
        print(f"  (S2 {d}: OOS accuracy {sh._pct(m['win_rate'])}, trades {m['evaluated']}, "
              f"PF {sh._pf_s(m['profit_factor'])}, P&L Rs {sh._rs(m['net_pnl'])})")
    bt = a["best_tier"]
    print("\nBEST CONFIDENCE TIER: " + (
        f"{bt['direction']} {bt['tier']} (TRAIN+VAL {bt['trades']} trades / {sh._pct(bt['accuracy'])}; "
        f"OOS {bt['oos']['trades']} trades / {sh._pct(bt['oos']['accuracy'])})" if bt else "NONE (too few trades)"))
    print(f"70% OOS TARGET: {'ACHIEVED' if a['target_achieved'] else 'NOT ACHIEVED'}")
    if not a["target_achieved"]:
        print(s3lab.NOT_ACHIEVED_TEXT)
    print("S2 PRODUCTION CHANGED: NO")
    print("LIVE ORDERS: 0")
    print(f"LOOKAHEAD: {'PASS' if a['lookahead']['passed'] else 'FAIL'}")
    print(f"INTEGRITY: {'PASS' if a['integrity_ok'] else 'FAIL'}")
    print(f"FINAL RECOMMENDATION: {a['recommendation']}"
          + (f" ({a['forward_validation']})" if a["forward_validation"] else ""))
    print(s3lab.BANNER)
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
