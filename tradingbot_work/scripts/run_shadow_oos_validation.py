#!/usr/bin/env python3
"""OUT-OF-SAMPLE / FORWARD VALIDATION of BASELINE_S2_BUY_SELL, SELL_ONLY and SELL_CONFIRMED (PAPER, no orders).

Uses the project's existing chronological split (TRAIN 2026-09-01..09-17, VALIDATION 09-18..09-24,
OOS 09-25..10-01) and the historical shadow replay pipeline (real expired contracts and candles cached under
data/historical/). Reports go to reports/shadow_strategy_oos/; the historical replay reports are not touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_shadow_historical_validation as hv
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_oos as so
from src.historical_data.service import _refuse_live
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "shadow_strategy_oos"


def run(*, out_dir: Path = DEFAULT_OUT, current_dir: Path = hv.CURRENT_DIR, splits: dict | None = None,
        criteria: so.RobustnessCriteria = so.RobustnessCriteria(), oos_previously_observed: bool = True,
        today=None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    splits = imp.split_dates(splits or imp.DEFAULT_SPLITS)
    result, meta = hv.build_replay(from_date=splits["TRAIN"][0], to_date=splits["OOS"][1], current_dir=current_dir,
                                   today=today)
    study = so.build_study(result, meta, splits=splits, current=hv.load_current_day(current_dir), criteria=criteria,
                           oos_previously_observed=oos_previously_observed)
    paths = so.write_reports(study, out_dir)
    print_summary(study, paths)
    return study


def _f(v, kind):
    if v is None:
        return "N/A"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "pf":
        return "INF" if v == float("inf") else f"{v:.2f}"
    return f"{v:,.2f}"


def print_summary(s: dict, paths: dict) -> None:
    print(f"{so.HEADING} | PAPER | no orders")
    for name, sp in s["splits"].items():
        print(f"  {sp['label']}: {sp['from']}..{sp['to']} ({len(sp['days'])} sessions)")
    d = (s["replay_meta"] or {}).get("expired_candle_downloads") or {}
    print(f"  Expired contracts: downloaded {d.get('contracts_downloaded', 0)}, from cache "
          f"{d.get('contracts_from_cache', 0)}, failures {len(d.get('failures') or [])}")
    print("  Strategy | Train WR | Train PF | Val WR | Val PF | OOS WR | OOS PF | OOS P&L | OOS DD | OOS Trades | Status")
    for r in s["final_table"]:
        print(f"  {r['strategy']} | {_f(r['train_wr'], 'pct')} | {_f(r['train_pf'], 'pf')} | "
              f"{_f(r['validation_wr'], 'pct')} | {_f(r['validation_pf'], 'pf')} | {_f(r['oos_wr'], 'pct')} | "
              f"{_f(r['oos_pf'], 'pf')} | {_f(r['oos_pnl'], 'rs')} | {_f(r['oos_dd'], 'rs')} | {r['oos_trades']} | "
              f"{r['status']}")
    print("  Ranking by OOS: " + " > ".join(s["ranking_by_oos"]))
    for split, la in s["lookahead"].items():
        print(f"  Lookahead {split}: " + ", ".join(
            "%s=%s(%d)" % (n, "PASS" if r["passed"] else "FAIL", r["cuts"]) for n, r in la.items()))
    for n, st in s["status"].items():
        if st["failed_checks"]:
            print(f"  {n}: {st['status']} - failed {', '.join(st['failed_checks'])}")
    if s["oos_previously_observed_note"]:
        print(f"  NOTE: {s['oos_previously_observed_note']}")
    print("  Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-oos-trades", type=int, default=so.RobustnessCriteria.min_oos_trades)
    a = p.parse_args(argv)
    run(out_dir=a.out_dir, criteria=so.RobustnessCriteria(min_oos_trades=a.min_oos_trades))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
