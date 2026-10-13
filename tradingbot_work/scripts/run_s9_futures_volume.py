#!/usr/bin/env python3
"""S9.1 - S9_FUTURES_VOLUME research replay on the stored real NIFTY futures data (no downloads).

PAPER | RESEARCH ONLY | NO ORDERS. Re-validates data/historical/upstox_6m/futures offline; the S9 futures replay runs
ONLY when futures data quality PASSES. The original S9 proxy reports are never touched.
Reports: reports/strategy_improvement/s9_futures_volume/
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_nifty_futures_6m import secrets
from scripts.run_clean_historical_backtest import production_hashes
from scripts.run_s4_six_month_research import load_s2
from src.backtest import shadow_historical as sh
from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.research import s5_targeted_strategy as s5
from src.research import s9_futures_volume as s9f
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s9_futures_volume"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"


def run(*, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY",
        mutation_days: int = 10, check_sample: int = 20, secret_values: list[str] | None = None) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    store = s4.SixMonthStore(data_root, underlying)
    spot = store.spot()
    if spot.empty:
        raise SystemExit(f"No 6-month underlying candles under {data_root} (not downloading)")
    rc, params, session, s2cfg, signal_fn = load_s2(underlying)
    sessions, _ = sh.complete_sessions(spot, store.start, store.end, params.interval)
    days = [d.isoformat() for d in sessions]
    sec = secrets() if secret_values is None else secret_values

    def listed(sd: date, expiry: date, kind: str) -> list[float]:
        return sorted(store.chain.get((sd, expiry, kind), {}))

    def exists(r: dict) -> bool:
        sd = date.fromisoformat(str(r["session_date"])[:10])
        chain = store.chain.get((sd, r.get("expiry_date"), r.get("option_type")), {})
        return any(c.instrument_key == r.get("instrument_key") for c in chain.values())

    with order_placement_blocked(), capture_upstox_events():
        say("validating stored NIFTY futures data (offline) ...")
        fq = nf.validate_store(root=data_root, start=store.start, end=store.end, spot=spot, secrets=sec)
        integrity = s4.data_integrity(store, spot)
        if not fq["pass"]:
            say(f"futures data quality {fq['status']} - S9 futures replay NOT RUN")
            a = s9f.not_run(fq, integrity)
        else:
            premium_for = store.premium_for(None)
            resolve = s4.make_resolve(store, rc)
            say(f"frozen S2 replay for the comparison only ({len(days)} sessions) ...")
            full = dra.replay(spot=spot, days=days, signal_fn=signal_fn, resolve=resolve, premium_for=premium_for,
                              params=params, square_off=rc.session.square_off, range_start=s2cfg["range_start"],
                              range_end=s2cfg["range_end"])
            bars = dg.feature_bars(spot)
            s2_rows = s5.prepare_rows(dg.annotate(full["rows"], bars, full["ctx_by_key"]), full["ctx_by_key"])
            contracts, frames = nf.load_futures(data_root, underlying)
            dataset = {"root": str(data_root), "from": store.start.isoformat(), "to": store.end.isoformat(),
                       "spot": "NSE_INDEX|Nifty 50", "futures": [c.trading_symbol for c in contracts],
                       "sessions": len(days), "spot_candles": len(spot), "futures_candles": fq["total_candles"]}
            a = s9f.build(spot=spot, sessions=days, contracts=contracts, frames=frames, selection=fq["_selection"],
                          futures_quality=fq, resolve=resolve, premium_for=premium_for, s2_rows=s2_rows,
                          calendar=store.calendar, listed=listed, exists=exists, integrity=integrity,
                          dataset=dataset, mutation_days=s4.sample_days(days, mutation_days) if mutation_days else [],
                          check_sample=check_sample, progress=say)
    a["production_unchanged"] = production_hashes() == before
    if not a["production_unchanged"]:
        a["forward_paper"] = "NOT SAFE"
    paths = s9f.write_reports(a, out_dir)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    if not leak_ok:
        a["futures_quality"]["token_leak"] = {"passed": False, "files": leaks}
    print("\n" + s9f.terminal_text(a))
    print(f"\nReports: {out_dir}")
    for p in paths.values():
        print(f"  {p}")
    return a


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mutation-days", type=int, default=10)
    a = p.parse_args(argv)
    run(data_root=a.data_root, out_dir=a.out_dir, mutation_days=a.mutation_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
