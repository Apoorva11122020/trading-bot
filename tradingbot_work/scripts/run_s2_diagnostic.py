#!/usr/bin/env python3
"""S2 MONTHLY + REGIME DIAGNOSTIC LAB (PAPER | RESEARCH ONLY | NO ORDERS; S2 and production code untouched).

Exact S2 on data/historical/upstox_6m (full period + one separate replay per month). Reads only the 6-month store,
never downloads. Reports go to reports/strategy_improvement/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s4_six_month_research import load_s2
from src.backtest import shadow_historical as sh
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"


def run(*, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY",
        check_sample: int = 20) -> dict:
    assert_paper_trading_only()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    store = s4.SixMonthStore(data_root, underlying)
    spot = store.spot()
    if spot.empty:
        raise SystemExit(f"No 6-month underlying candles under {data_root} (not downloading)")
    rc, params, session, s2cfg, signal_fn = load_s2(underlying)
    sessions, _ = sh.complete_sessions(spot, store.start, store.end, params.interval)
    days = [d.isoformat() for d in sessions]
    kw = dict(spot=spot, signal_fn=signal_fn, resolve=s4.make_resolve(store, rc), premium_for=store.premium_for(None),
              params=params, square_off=rc.session.square_off, range_start=s2cfg["range_start"],
              range_end=s2cfg["range_end"])
    with order_placement_blocked(), capture_upstox_events():
        say(f"exact S2 replay, full period ({len(days)} sessions) ...")
        full = dra.replay(days=days, **kw)
        monthly = {}
        for m, label in dg.MONTHS:
            md = [d for d in days if d.startswith(m)]
            say(f"exact S2 replay, {label} ({len(md)} sessions) ...")
            monthly[m] = dra.replay(days=md, **kw)["rows"] if md else []
        say("counting option candles ...")
        option_candles = sum(len(u6.read_parquet(p)) for p in store.layout.option_files())
        dataset = {"from": store.start.isoformat(), "to": store.end.isoformat(), "sessions": len(days),
                   "spot_candles": len(spot), "option_candles": option_candles, "contracts": len(store.contracts)}
        say("causal features, regimes, buckets, stability ...")
        a = dg.build_diagnostic(full_rows=full["rows"], monthly_rows=monthly, spot=spot,
                                ctx_by_key=full["ctx_by_key"], dataset=dataset,
                                implementation=dra.s2_implementation(rc, params, s2cfg), check_sample=check_sample)
    paths = dg.write_reports(a, out_dir)
    print("\n" + dg.summary_text(a))
    ck = a["checks"]
    print(f"\nChecks: leakage {'PASS' if ck['truncation']['passed'] else 'FAIL'} | monthly replay consistency "
          f"{'PASS' if ck['monthly_consistency']['passed'] else 'FAIL'} | chronology "
          f"{'PASS' if ck['chronology']['passed'] else 'FAIL'} | BUY/SELL separation "
          f"{'PASS' if ck['buy_sell_separation'] else 'FAIL'}")
    print(dg.BANNER)
    print("Reports: " + ", ".join(str(p) for p in paths.values()))
    return a


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run(data_root=a.data_root, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
