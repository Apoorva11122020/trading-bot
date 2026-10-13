#!/usr/bin/env python3
"""STRATEGY IMPROVEMENT LAB: pre-declared BUY/SELL/time/regime/entry candidates on the unchanged S2 signal.

PAPER / research only, no orders, production S2 untouched. Uses the project's chronological split (TRAIN
2026-09-01..09-17, VALIDATION 09-18..09-24, OOS 09-25..10-01) and the historical shadow replay (real cached NIFTY
candles, point-in-time expiries, real option candles). 2026-10-05 onwards is forward data and is only reported.
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

from scripts import run_shadow_historical_validation as hv
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.historical_data.service import _refuse_live
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement"
FORWARD_DIR = ROOT / "reports" / "live_signal_accuracy"


def replay_inputs(*, today=None, download_active: bool = False) -> SimpleNamespace:
    """Existing S2 historical replay (TRAIN..OOS only) plus the causal per-signal facts used by the labs."""
    assert_paper_trading_only()
    _refuse_live()
    splits = imp.split_dates(imp.DEFAULT_SPLITS)
    x = hv.prepare_replay_inputs(from_date=splits["TRAIN"][0], to_date=splits["OOS"][1], today=today,
                                 download_active=download_active)
    with order_placement_blocked(), capture_upstox_events():
        print("  replaying S2 (baseline) with the existing mutation lookahead test ...", flush=True)
        result = sh.replay(spot=x.spot, signal_fn=x.signal_fn, resolve=sh.strict_resolver(x.resolve, x.check),
                           premium_for=x.premium_for, params=x.params, square_off=x.rc.session.square_off,
                           range_start=x.s2["range_start"], range_end=x.s2["range_end"], days=x.selected,
                           regimes=x.regimes, keep_pipeline=True)
    pipe = result["pipeline"]
    base_rows = [dict(r, strategy=lab.BASELINE) for r in result["rows"][dr.BASELINE.name]]
    prem_close = lab.premium_close_at_signal(pipe["contexts"])
    facts = lab.build_facts(base_rows, lab.bar_features(pipe["spot"]), lab.PointInTimeRegimeClassifier(pipe["spot"]),
                            lab.previous_session_labels(x.regimes), prem_close)
    split_days, _ = lab.so.assign_days(result["days"], splits)
    hist = [d for s in lab.SPLITS for d in split_days[s]]

    def lookahead(cands, p):
        print(f"  truncation lookahead test ({len(hist)} sessions x 2 cuts) ...", flush=True)
        return lab.truncation_lookahead(lab.period_rows(base_rows, hist), pipe["spot"], x.s2, prem_close, facts,
                                        cands, p, hist)

    return SimpleNamespace(x=x, splits=splits, result=result, base_rows=base_rows, facts=facts,
                           split_days=split_days, hist=hist, lookahead=lookahead,
                           baseline_lookahead_passed=result["lookahead"][dr.BASELINE.name]["passed"],
                           forward=lab.load_forward(FORWARD_DIR, splits["OOS"][1]))


def run(*, out_dir: Path = DEFAULT_OUT, skip_dte: bool = False, today=None) -> dict:
    r = replay_inputs(today=today, download_active=not skip_dte)
    x = r.x
    with order_placement_blocked(), capture_upstox_events():
        dte = None if skip_dte else lab.dte_study(x, r.hist, r.split_days,
                                                  progress=lambda m: print(f"  {m}", flush=True))
        a = lab.build_lab(base_rows=r.base_rows, facts=r.facts, days=r.result["days"], splits=r.splits,
                          rules=lab.s2_rules(x.rc, x.params, x.s2), lookahead=r.lookahead,
                          baseline_lookahead_passed=r.baseline_lookahead_passed, dte=dte, forward=r.forward)
    paths = lab.write_reports(a, out_dir)
    print_result(a, paths)
    return a


def _line(name: str, m: dict) -> str:
    pct, pf, rs = lab._pct, lab._pf, lab._rs
    return (f"{name}: trades {m['evaluated']} | WR {pct(m['win_rate'])} | PF {pf(m['profit_factor'])} | "
            f"P&L Rs {rs(m['net_pnl'])} | DD Rs {rs(m['max_drawdown'])} | avg R {m['average_r']}")


def print_result(a: dict, paths: dict) -> None:
    pct, pf, rs = lab._pct, lab._pf, lab._rs
    print(f"\n{lab.HEADING} | PAPER | no orders | production S2 unchanged")
    for sp in a["splits"].values():
        print(f"  {sp['label']}: {sp['from']}..{sp['to']} ({len(sp['days'])} sessions)")
    print("\nFINAL CANDIDATE RANKING")
    print("# | Candidate | Train WR | Val WR | OOS WR | Train PF | Val PF | OOS PF | OOS P&L | OOS DD | OOS n | "
          "MaxLossStreak | Lookahead | Status")
    for i, n in enumerate(a["ranking"], 1):
        s = a["scores"][n]
        t, v, o = s["TRAIN"], s["VALIDATION"], s["OOS"]
        print(f"{i} | {n} | {pct(t['win_rate'])} | {pct(v['win_rate'])} | {pct(o['win_rate'])} | "
              f"{pf(t['profit_factor'])} | {pf(v['profit_factor'])} | {pf(o['profit_factor'])} | {rs(o['net_pnl'])} | "
              f"{rs(o['max_drawdown'])} | {o['evaluated']} | {s['ALL']['max_consecutive_losses']} | "
              f"{'PASS' if s['lookahead_passed'] else 'FAIL'} | {s['status']}")
    print("\nCURRENT S2 BASELINE")
    for period, by in a["baseline"]["metrics"].items():
        for d in ("BUY", "SELL", "TOTAL"):
            print("  " + _line(f"{period} {d}", by[d]))
    best = a["best_candidate"]
    print(f"\nBEST RESEARCH CANDIDATE: {best} ({a['scores'][best]['status'] if best else '-'})")
    if best:
        c = a["candidates"][best]
        print(f"  BUY rule : {c['rules']['BUY']}\n  SELL rule: {c['rules']['SELL']}")
        for period in lab.PERIODS:
            print("  " + _line(f"{period} candidate", a["scores"][best][period]))
            print("  " + _line(f"{period} baseline ", a["scores"][lab.BASELINE][period]))
        print(f"  Status reasons: {', '.join(a['scores'][best]['failed_checks']) or 'none'}")
    print(f"\nRECOMMENDATION: {a['recommendation']}")
    print(a["notes"]["oos"])
    print("Production S2 was NOT changed. TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, zero orders placed.")
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-dte", action="store_true", help="skip the Phase 10 DTE policy study (no extra downloads)")
    p.add_argument("--today", type=date.fromisoformat, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    run(out_dir=a.out_dir, skip_dte=a.skip_dte, today=a.today)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
