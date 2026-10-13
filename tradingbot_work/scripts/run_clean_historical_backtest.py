#!/usr/bin/env python3
"""STRICT CLEAN HISTORICAL BACKTEST of the frozen S2 strategy on data/historical/upstox_6m.

PAPER | RESEARCH ONLY | NO ORDERS. Reads only the downloaded 6-month store (never downloads). Production strategy,
broker, order, risk and safety code are not modified (sha256 verified before/after).
Reports: reports/strategy_improvement/clean_historical_backtest/
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s4_six_month_research import load_s2
from src.backtest import shadow_historical as sh
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.research import s5_targeted_strategy as s5
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "clean_historical_backtest"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
PRODUCTION_FILES = ("src/backtest/research_strategies.py", "src/backtest/s2_1dte_improvement.py",
                    "src/backtest/s2_direction_research.py", "src/backtest/current_day_s2.py",
                    "src/options/strategy.py", "src/options/selection.py", "config/strategy.yaml",
                    "config/strategy_research.yaml", "src/execution/order_manager.py", "src/risk/risk_manager.py",
                    "src/safety/live_gate.py")


def production_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in PRODUCTION_FILES if (ROOT / f).is_file()}


def run(*, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY",
        mutation_days: int = 10, check_sample: int = 20) -> dict:
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
        full = dra.replay(spot=spot, days=days, signal_fn=signal_fn, resolve=s4.make_resolve(store, rc),
                          premium_for=premium_for, params=params, square_off=rc.session.square_off,
                          range_start=s2cfg["range_start"], range_end=s2cfg["range_end"])
        rows = s5.prepare_rows(dg.annotate(full["rows"], dg.feature_bars(spot), full["ctx_by_key"]),
                               full["ctx_by_key"])
        mutation = None
        if mutation_days:
            say(f"anti-lookahead: pipeline mutation test on {mutation_days} sessions (candles after 12:00 scrambled) ...")
            stage = {"pipeline": full["pipeline"], "spot": spot, "premium_for": premium_for}
            mutation = s4.baseline_mutation_test(stage, s4.sample_days(days, mutation_days))
        say("anti-lookahead: feature truncation test ...")
        truncation = dg.truncation_check(spot, rows, full["ctx_by_key"], check_sample)
        dataset = {"root": str(data_root), "from": store.start.isoformat(), "to": store.end.isoformat(),
                   "underlying": "NSE_INDEX|Nifty 50", "sessions": len(days), "spot_candles": len(spot),
                   "contracts": len(store.contracts),
                   "option_candles": sum(len(u6.read_parquet(p)) for p in store.layout.option_files())}
        say("metrics, robustness, reports ...")
        a = cb.build(rows=rows, contexts=full["contexts"], params=params, square_off=rc.session.square_off,
                     stop_new_entries=params.stop_new_entries, calendar=store.calendar, listed=listed, exists=exists,
                     integrity=integrity, mutation=mutation, truncation=truncation, sessions=days, dataset=dataset)
    a["production_unchanged"] = production_hashes() == before
    paths = cb.write_reports(a, out_dir)
    print("\n" + cb.terminal_text(a))
    print("\nReports:")
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
