#!/usr/bin/env python3
"""S2 BASELINE DRIFT AUDIT (PAPER | RESEARCH ONLY | NO ORDERS; S2 and production code untouched).

OLD = per-signal rows saved by the previous limited-data replay (reports/shadow_strategy_oos).
NEW = exact S2 replayed on data/historical/upstox_6m. The legacy candle store is read offline only (never downloads).
Reports go to reports/strategy_improvement/.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s4_six_month_research import load_s2
from src.backtest import shadow_historical as sh
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s2_baseline_drift_audit as audit
from src.research import s4_six_month_lab as s4
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
OLD_DIR = ROOT / "reports" / "shadow_strategy_oos"


def _legacy_root() -> Path:
    p = Path(storage.DEFAULT_ROOT)
    return p if p.is_absolute() else ROOT / p


def run(*, data_root: Path = DEFAULT_DATA, old_dir: Path = OLD_DIR, out_dir: Path = DEFAULT_OUT,
        underlying: str = "NIFTY") -> dict:
    assert_paper_trading_only()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    store = s4.SixMonthStore(data_root, underlying)
    new_spot = store.spot()
    if new_spot.empty:
        raise SystemExit(f"No 6-month underlying candles under {data_root} (not downloading)")
    rc, params, session, s2cfg, signal_fn = load_s2(underlying)
    sessions, _ = sh.complete_sessions(new_spot, store.start, store.end, params.interval)
    days = [d.isoformat() for d in sessions]
    old_rows = audit.load_old_rows(old_dir)
    lroot = _legacy_root()
    legacy = audit.LegacyData(storage.ParquetCandleStore(lroot), underlying) if (lroot / "options").is_dir() else None

    with order_placement_blocked(), capture_upstox_events():
        say(f"exact S2 replay on the 6-month store ({len(days)} sessions) ...")
        new = audit.replay(spot=new_spot, days=days, signal_fn=signal_fn, resolve=s4.make_resolve(store, rc),
                           premium_for=store.premium_for(None), params=params, square_off=rc.session.square_off,
                           range_start=s2cfg["range_start"], range_end=s2cfg["range_end"])
        say("profiling both option datasets ...")
        legacy_profile = (audit.frame_profile(pd.read_parquet(p) for p in legacy.option_files())
                          if legacy is not None else None)
        six_profile = audit.frame_profile(u6.read_parquet(p) for p in store.layout.option_files())

        def listed(sd: date, expiry: date, kind: str) -> list[float]:
            return sorted(store.chain.get((sd, expiry, kind), {}))

        def exists(r: dict) -> bool:
            sd = date.fromisoformat(str(r["session_date"])[:10])
            chain = store.chain.get((sd, r.get("expiry_date"), r.get("option_type")), {})
            return any(c.instrument_key == r.get("instrument_key") for c in chain.values())

        a = audit.build_audit(
            old_rows=old_rows, new=new, new_spot=new_spot, legacy=legacy,
            six_premium=lambda key: store.premium(key) if key else None, calendar=store.calendar,
            listed_strikes=listed, new_exists=exists, params=params, square_off=rc.session.square_off,
            range_start=s2cfg["range_start"], range_end=s2cfg["range_end"],
            implementation=audit.s2_implementation(rc, params, s2cfg),
            old_spot=legacy.spot() if legacy is not None else None, legacy_profile=legacy_profile,
            six_profile=six_profile, progress=say)
    paths = audit.write_reports(a, out_dir)
    print_summary(a, paths)
    return a


def _line(m: dict) -> str:
    pf = m["profit_factor"]
    pfs = "N/A" if pf is None else ("INF" if pf == float("inf") else f"{pf:.2f}")
    return (f"{audit._pct(m['accuracy'])} ({m['trades']} trades, {m['wins']} W / {m['losses']} L, PF {pfs}, "
            f"P&L Rs {m['net_pnl']:,.2f})")


def print_summary(a: dict, paths: dict) -> None:
    h, ev = a["headline"], a["evidence"]
    sel = [r for r in a["option_selection"] if r["status"] == "BOTH"]
    e = a["entry_drift"]
    nc = [c for c in a["contracts"] if c["dataset"] == "NEW_6M"]
    oc = [c for c in a["contracts"] if c["dataset"] == "OLD_LEGACY"]
    gn = [g for g in a["gaps"] if g["dataset"] == "NEW_6M"]
    go = [g for g in a["gaps"] if g["dataset"] == "OLD_LEGACY"]
    ps = a["pairs_summary"]
    reasons = ", ".join(f"{k} {v}" for k, v in
                        pd.Series([r["reason"] for r in a["count_reasons"]]).value_counts().items()) or "none"
    lines = [f"\n{audit.HEADING}", "", "OLD (legacy, OOS 2026-09-25..10-01):"]
    lines += [f"{s}: {_line(h['old_oos'][s])}" for s in audit.SCOPES]
    lines += ["", "NEW 6-MONTH (OOS 2026-08-01..08-31):"]
    lines += [f"{s}: {_line(h['new_oos'][s])}" for s in audit.SCOPES]
    lines += ["", "NEW 6-MONTH on the OLD OOS window (2026-09-25..09-30):"]
    lines += [f"{s}: {_line(h['new_same_window_as_old_oos'][s])}" for s in audit.SCOPES]
    lines += ["", f"Overlapping sessions: {len(a['overlap_days'])} | OLD {_line(h['old_overlap']['COMBINED'])} | "
              f"NEW {_line(h['new_overlap']['COMBINED'])}", "",
              f"Signal count drift: overlap OLD {ps.get('BOTH', 0) + ps.get('OLD_ONLY', 0)} vs NEW "
              f"{ps.get('BOTH', 0) + ps.get('NEW_ONLY', 0)} (matched {ps.get('BOTH', 0)}, old-only "
              f"{ps.get('OLD_ONLY', 0)}, new-only {ps.get('NEW_ONLY', 0)}); reasons: {reasons}",
              f"Option selection drift: {sum(r['mismatch'] for r in sel)}/{len(sel)} matched signals differ",
              f"Entry price drift: >5% {sum(r['gt_5pct'] for r in e)}, >10% {sum(r['gt_10pct'] for r in e)}, "
              f">20% {sum(r['gt_20pct'] for r in e)} of {len(e)}; old entry/SL/target on Rs 5 grid "
              f"{audit._pct(ev['old_tick5_share'])}",
              f"Candle alignment: {'PASS' if a['alignment']['passed'] else 'FAIL'}",
              f"Contract validity: {'PASS' if all(c['valid'] for c in nc) else 'FAIL'} "
              f"(new {sum(c['valid'] for c in nc)}/{len(nc)} valid; old {sum(c['valid'] for c in oc)}/{len(oc)} valid, "
              f"impossible {sum(c['impossible'] for c in oc)})",
              f"Data gaps: new losses with gaps {sum(g['loss_caused_by_data_gap'] for g in gn)}/{len(gn)}; old "
              f"{sum(g['loss_caused_by_data_gap'] for g in go)}/{len(go)}",
              f"Primary root cause: {a['primary_root_cause']}",
              f"Severity: {a['severity']}",
              "S2 changed: NO", f"Orders: {a['orders']}", f"Final recommendation: {a['recommendation']}"]
    print("\n".join(lines))
    print(audit.BANNER)
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--old-dir", type=Path, default=OLD_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run(data_root=a.data_root, old_dir=a.old_dir, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
