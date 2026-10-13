#!/usr/bin/env python3
"""S8 STRATEGY ARCHITECTURE AUDIT & CLEAN ALTERNATIVE RESEARCH on data/historical/upstox_6m (2026-04-01..2026-09-30).

PAPER | RESEARCH ONLY | NO ORDERS. Reads only the downloaded 6-month store (never downloads). Production strategy,
broker, order, risk and safety code are not modified (sha256 verified before/after).
Reports: reports/strategy_improvement/s8_architecture_audit/
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_clean_historical_backtest import production_hashes
from scripts.run_s4_six_month_research import load_s2
from src.backtest import shadow_historical as sh
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.research import s5_targeted_strategy as s5
from src.research import s6_clean_strategy_lab as s6
from src.research import s8_architecture_audit as s8
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s8_architecture_audit"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"


def run(*, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY",
        mutation_days: int = 10, check_sample: int = 20, force_alternatives: bool = False) -> dict:
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

    def listed(sd: date, expiry: date, kind: str) -> list[float]:
        return sorted(store.chain.get((sd, expiry, kind), {}))

    def exists(r: dict) -> bool:
        sd = date.fromisoformat(str(r["session_date"])[:10])
        chain = store.chain.get((sd, r.get("expiry_date"), r.get("option_type")), {})
        return any(c.instrument_key == r.get("instrument_key") for c in chain.values())

    with order_placement_blocked(), capture_upstox_events():
        say("data integrity of the 6-month store ...")
        integrity = s4.data_integrity(store, spot)
        say(f"exact frozen S2 replay, chronological ({len(days)} sessions) ...")
        premium_for = store.premium_for(None)
        resolve = s4.make_resolve(store, rc)
        full = dra.replay(spot=spot, days=days, signal_fn=signal_fn, resolve=resolve, premium_for=premium_for,
                          params=params, square_off=rc.session.square_off, range_start=s2cfg["range_start"],
                          range_end=s2cfg["range_end"])
        bars = dg.feature_bars(spot)
        rows = s5.prepare_rows(dg.annotate(full["rows"], bars, full["ctx_by_key"]), full["ctx_by_key"])
        rows = s6.prepare(rows, bars, full["ctx_by_key"], s2cfg["range_start"], s2cfg["range_end"])
        cut_days = s4.sample_days(days, mutation_days) if mutation_days else []
        mutation = None
        if cut_days:
            say(f"future candle scrambling: S2 pipeline mutation test on {len(cut_days)} sessions ...")
            stage = {"pipeline": full["pipeline"], "spot": spot, "premium_for": premium_for}
            mutation = s4.baseline_mutation_test(stage, cut_days)
        dataset = {"root": str(data_root), "from": store.start.isoformat(), "to": store.end.isoformat(),
                   "underlying": "NSE_INDEX|Nifty 50", "sessions": len(days), "spot_candles": len(spot),
                   "contracts": len(store.contracts),
                   "option_candles": sum(len(u6.read_parquet(p)) for p in store.layout.option_files())}
        a = s8.build(rows=rows, contexts=full["contexts"], ctx_by_key=full["ctx_by_key"], spot=spot, bars=bars,
                     feats=full["features"], params=params, square_off=rc.session.square_off,
                     stop_new_entries=params.stop_new_entries, range_start=s2cfg["range_start"],
                     range_end=s2cfg["range_end"], calendar=store.calendar, listed=listed, exists=exists,
                     resolve=resolve, premium_for=premium_for, integrity=integrity, mutation=mutation,
                     mutation_days=cut_days, sessions=days, dataset=dataset, check_sample=check_sample,
                     force_alternatives=force_alternatives, progress=say)
    a["production_unchanged"] = production_hashes() == before
    if not a["production_unchanged"]:
        a["forward_paper"] = "NOT SAFE"
    paths = s8.write_reports(a, out_dir)
    print("\n" + s8.terminal_text(a))
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
