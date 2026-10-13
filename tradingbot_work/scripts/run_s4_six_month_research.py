#!/usr/bin/env python3
"""S4_SIX_MONTH_RESEARCH_LAB (PAPER | RESEARCH ONLY | NO ORDERS; production S2 untouched).

Reads only data/historical/upstox_6m (never re-downloads, never falls back to the old dataset). The exact S2 signal
and execution model is replayed; TRAIN 04-01..06-30 + VALIDATION 07-01..07-31 + walk-forward select one finalist per
direction, OOS 08-01..08-31 grades it, and the September holdout is loaded only afterwards and evaluated once.
Reports go to reports/strategy_improvement/.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s2_1dte_improvement import s2_components
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.config.loader import load_strategy_config
from src.config.research_loader import load_research_config
from src.historical_data.option_coverage import entry_allowed
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s4_six_month_lab as s4
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
RESEARCH_CONFIG = ROOT / "config" / "strategy_research.yaml"
STRATEGY_CONFIG = ROOT / "config" / "strategy.yaml"


def load_s2(name: str = "NIFTY") -> tuple:
    """The unchanged S2 rules via ``s2_components``; the stub service exposes configs only (no data store)."""
    rc, sc = load_research_config(RESEARCH_CONFIG), load_strategy_config(STRATEGY_CONFIG)
    service = SimpleNamespace(research_config=rc, strategy_config=sc, resolver_factory=None)
    rc, params, session, s2cfg, signal_fn, _ = s2_components(service, name)
    return rc, params, session, s2cfg, signal_fn


def run(*, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT, today: date | None = None,
        skip_mutation: bool = False, underlying: str = "NIFTY") -> dict:
    assert_paper_trading_only()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    store = s4.SixMonthStore(data_root, underlying)
    spot = store.spot()
    if spot.empty:
        raise SystemExit(f"No 6-month underlying candles under {data_root} (dataset missing; not downloading)")
    say(f"phase 1: data integrity of {data_root} ...")
    di = s4.data_integrity(store, spot)
    rc, params, session, s2cfg, signal_fn = load_s2(underlying)

    def allowed(ts):
        return entry_allowed(ts.to_pydatetime(), session, params.stop_new_entries)

    with order_placement_blocked(), capture_upstox_events():
        a = s4.build_s4(
            spot=spot, signal_fn=signal_fn, resolve=s4.make_resolve(store, rc), premium_for_factory=store.premium_for,
            params=params, square_off=rc.session.square_off, range_start=s2cfg["range_start"],
            range_end=s2cfg["range_end"], underlying=underlying, allowed=allowed, today=today,
            rules=lab.s2_rules(rc, params, s2cfg), run_mutation=not skip_mutation, interval=params.interval,
            progress=say, integrity_data=di, store=store)
    paths = s4.write_reports(a, out_dir)
    print_summary(a, paths)
    return a


def _base(a: dict, scope: str) -> str:
    m = a["perf"][f"S2_{scope}"]["OOS"]
    return (f"  {scope}: OOS accuracy {sh._pct(m['accuracy'])}, Trades {m['trades']}, "
            f"PF {sh._pf_s(m['profit_factor'])}, P&L Rs {sh._rs(m['net_pnl'])}")


def _best(a: dict, scope: str) -> str:
    f = a["final"].get(scope)
    name = f"S4_{scope}"
    if not f or name not in a["perf"]:
        return f"  {scope}: no eligible candidate (TRAIN+VALIDATION+WF selection), status {s4.INSUFFICIENT_DATA}"
    m = a["perf"][name]["OOS"]
    return (f"  {scope} ({f['cid']}): accuracy {sh._pct(m['accuracy'])}, trades {m['trades']}, "
            f"PF {sh._pf_s(m['profit_factor'])}, P&L Rs {sh._rs(m['net_pnl'])}, status {f['status']}")


def _hold(a: dict, scope: str) -> str:
    name = f"S4_{scope}"
    if name not in a["holdout"]:
        return f"  {scope}: not evaluated (no finalist)"
    h = a["holdout"][name]
    return (f"  {scope}: accuracy {sh._pct(h['accuracy'])}, trades {h['trades']}, holdout {h['verdict']}, "
            f"status {a['final'][scope]['status']}")


def print_summary(a: dict, paths: dict) -> None:
    ds = a["dataset"]
    lines = [f"\n{s4.HEADING}", f"Dataset: {ds['from']} -> {ds['to']}", f"Trading days: {ds['trading_days']}",
             f"Option candles: {ds['option_candles']}", "S2 BASELINE:"]
    lines += [_base(a, s) for s in ("BUY", "SELL", "COMBINED")]
    lines.append("BEST S4:")
    lines += [_best(a, s) for s in ("BUY", "SELL", "COMBINED")]
    lines.append("FINAL SEPTEMBER HOLDOUT:")
    lines += [_hold(a, s) for s in ("BUY", "SELL", "COMBINED")]
    lines += [f"70% OOS TARGET: {'ACHIEVED' if a['target_achieved'] else 'NOT ACHIEVED'}",
              f"ROBUST_70: {'YES' if a['robust_70'] else 'NO'}",
              f"LOOKAHEAD: {'PASS' if a['lookahead']['passed'] else 'FAIL'}",
              f"INTEGRITY: {'PASS' if a['integrity_ok'] else 'FAIL'}",
              f"SEPTEMBER USED FOR OPTIMIZATION: {'YES' if a['september_used_for_optimization'] else 'NO'}",
              f"PRODUCTION S2 CHANGED: {'YES' if a['production_s2_changed'] else 'NO'}",
              f"LIVE ORDERS: {a['live_orders']}",
              f"FINAL RECOMMENDATION: {a['recommendation']}"]
    print("\n".join(lines))
    if not a["target_achieved"]:
        print(s4.NOT_ACHIEVED_TEXT)
    print(s4.BANNER)
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-mutation", action="store_true",
                   help="skip the slower exact-S2 pipeline mutation test (truncation and option tests still run)")
    p.add_argument("--today", type=date.fromisoformat, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    run(data_root=a.data_root, out_dir=a.out_dir, today=a.today, skip_mutation=a.skip_mutation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
