#!/usr/bin/env python3
"""BUY DTE / EXPIRY SELECTION RESEARCH LAB: the original S2 BUY signal under 8 pre-declared expiry policies.

PAPER / research only, no orders. Splits: TRAIN 2026-09-01..09-17, VALIDATION 09-18..09-24, OOS 09-25..10-01
(project DEFAULT_SPLITS). Real cached NIFTY candles, point-in-time expiries (Upstox expired-instruments API + live
master) and real option candles (expired contracts via the expired API, active contracts via the historical candle
API), cached under data/historical/. Reports go to reports/buy_dte_lab/. No policy is applied to production S2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_shadow_historical_validation as hv
from src.backtest import buy_dte_lab as dl
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.historical_data.service import _refuse_live
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "buy_dte_lab"


def run(*, out_dir: Path = DEFAULT_OUT, splits: dict | None = None, criteria: dl.DteCriteria = dl.DteCriteria(),
        today=None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    splits = imp.split_dates(splits or imp.DEFAULT_SPLITS)
    x = hv.prepare_replay_inputs(from_date=splits["TRAIN"][0], to_date=splits["OOS"][1], today=today,
                                 download_active=True)
    with order_placement_blocked(), capture_upstox_events() as events:
        original = sh.replay(spot=x.spot, signal_fn=x.signal_fn, resolve=sh.strict_resolver(x.resolve, x.check),
                             premium_for=x.premium_for, params=x.params, square_off=x.rc.session.square_off,
                             range_start=x.s2["range_start"], range_end=x.s2["range_end"], days=x.selected,
                             regimes=x.regimes, mutation_cut=None)
        original_buy = [r for r in original["rows"][dl.lab.ORIGINAL] if r["direction"] == "BUY"
                        and imp.split_of(r["session_date"], splits)]
        a = dl.build_lab(x, splits=splits, criteria=criteria, original_buy_rows=original_buy,
                         progress=lambda msg: print(f"  {msg}", flush=True))
    a["replay_meta"] = hv.replay_meta(x, events)
    paths = dl.write_reports(a, out_dir)
    print_result(a, paths)
    return a


def print_result(a: dict, paths: dict) -> None:
    pct, pf, rs, signed = dl._pct, dl._pf, dl._rs, dl._signed
    print(f"{dl.HEADING} | PAPER | no orders | production S2 / SELL_ONLY / SELL_CONFIRMED unchanged")
    for sp in a["splits"].values():
        print(f"  {sp['label']}: {sp['from']}..{sp['to']} ({len(sp['days'])} sessions)")
    d = (a.get("replay_meta") or {}).get("expired_candle_downloads") or {}
    print(f"  Option candles: expired downloaded {d.get('contracts_downloaded', 0)} / cached "
          f"{d.get('contracts_from_cache', 0)}; active downloaded {d.get('active_contracts_downloaded', 0)} / cached "
          f"{d.get('active_contracts_from_cache', 0)}; failures {len(d.get('failures') or [])}")
    print(f"  BUY_NEAREST_EXPIRY identical to original S2 BUY: {a['baseline_matches_original_s2_buy']}")
    print()
    print("Candidate | Train WR | Validation WR | OOS WR | OOS PF | OOS P&L | OOS DD | OOS Trades | Status")
    t = {r["candidate"]: r for r in a["final_table"]}
    for n in a["ranking_by_oos"]:
        r = t[n]
        print(f"{n} | {pct(r['train_wr'])} | {pct(r['validation_wr'])} | {pct(r['oos_wr'])} | {pf(r['oos_pf'])} | "
              f"{rs(r['oos_pnl'])} | {rs(r['oos_dd'])} | {r['oos_trades']} | {r['status']}")
    print()
    print("DTE reality check (requested | signals | resolved | expiry unavailable | outside range | resolved DTEs):")
    for n, r in a["dte_resolution"].items():
        print(f"  {n} | {a['policies'][n]} | {r['requested']} | {r['resolved']} | {r['expiry_unavailable']} | "
              f"{r['outside_requested_range']} | {r['resolved_dte_histogram']}")
    print("Sample: Date | Requested DTE | Actual resolved expiry | Actual DTE")
    for s in dl.resolution_sample(a["_rows"]):
        print(f"  {s['policy']}: {s['date']} | {s['requested_dte']} | {s['expiry']} | {s['actual_dte']}")
    for period in ("OOS", "ALL"):
        b = a["metrics"][period][dl.BASELINE]
        print()
        print(f"BASELINE BUY ({period}): WR {pct(b['win_rate'])} | PF {pf(b['profit_factor'])} | P&L Rs "
              f"{rs(b['net_pnl'])} | DD Rs {rs(b['max_drawdown'])} | trades {b['evaluated']}")
    best = a["best_candidate"]
    print()
    print(f"BEST DTE POLICY: {best or 'None - no policy has >= ' + str(a['criteria']['min_oos_trades']) + ' OOS trades'}")
    if best:
        for period in ("OOS", "ALL"):
            c = a["comparison_vs_baseline"][best][period]
            x = c["candidate"]
            print(f"BEST DTE ({period}): WR {pct(x['win_rate'])} | PF {pf(x['profit_factor'])} | P&L Rs "
                  f"{rs(x['net_pnl'])} | DD Rs {rs(x['max_drawdown'])} | trades {x['evaluated']}")
            print(f"  WR improvement: {pct(c['baseline']['win_rate'])} -> {pct(x['win_rate'])} = "
                  f"{signed(c['wr_change_pp'], 'pp')}")
            print(f"  PF improvement: {signed(c['pf_change'], 'pf')} | P&L improvement: Rs {signed(c['pnl_change'])} "
                  f"| DD change: Rs {signed(c['dd_change'])} | trade diff: {c['trade_count_change']:+d}")
        st = a["status"][best]
        dr_ = a["day_removal_check_oos"][best]["best_day_removed"]
        print(f"  Status: {st['status']}" + (f" (failed: {', '.join(st['failed_checks'])})" if st["failed_checks"]
                                             else ""))
        print(f"  OOS P&L without best day ({dr_['excluded_date']}): Rs {rs(dr_['net_pnl'])}")
    failed = [n for n, r in a["lookahead"].items() if not r["passed"]]
    unavailable = sum(r["expiry_unavailable"] for n, r in a["dte_resolution"].items() if n != dl.BASELINE)
    print()
    print("Lookahead: " + ("PASS for all 8 policies" if not failed else f"FAIL: {', '.join(failed)}"))
    print(f"Expiry selections unavailable (all non-baseline policies): {unavailable}")
    print(a["never_robust_note"])
    print("Production S2, SELL_ONLY and SELL_CONFIRMED were NOT changed. Zero live orders were placed. "
          "No DTE policy was applied.")
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run(out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
