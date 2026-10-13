#!/usr/bin/env python3
"""S13.2 - option-chain structure strategy research (PAPER | RESEARCH ONLY | NO ORDERS | rules only, no ML).

  python scripts/run_s12_data_quality.py      # S12 verdicts (data-quality gate)
  python scripts/run_s13_chain_audit.py       # S13.1 chain audit (data-quality gate)
  python scripts/run_s13_chain_strategy.py

Clean window only (2025-09-26 -> 2026-06-30); files that could hold candles after 2026-06-30 are never opened;
October never loaded. TRAIN 2025-09-26..2026-03-31, VALIDATION April 2026, OOS May-June 2026 (frozen; only
April-passing candidates). Reports: reports/strategy_improvement/s13_chain_strategy/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_nifty_futures_6m import secrets
from scripts.run_clean_historical_backtest import production_hashes
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s11_history as hx
from src.safety.live_gate import assert_paper_trading_only

try:  # s12_2 (loader helpers only - nothing is trained here) imports scikit-learn at module level
    from src.research import s12_2_constrained_ml as s122
    from src.research import s13_chain_strategy as s13
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("sklearn"):
        raise SystemExit("scikit-learn is not installed. Install it yourself first: pip install scikit-learn") \
            from None
    raise

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s13_chain_strategy"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
DEFAULT_EXPANDED = ROOT / "data" / "historical" / "upstox_expanded"
REPORTS = ROOT / "reports" / "strategy_improvement"
S12_DIR = REPORTS / "s12_data_quality"
S131_SUMMARY = REPORTS / "s13_chain_audit" / "summary.json"
COMPARISON_SOURCES = {"S2": REPORTS / "clean_historical_backtest" / "monthly_results.csv",
                      "S9": REPORTS / "s9_futures_volume" / "monthly_results.csv"}
FORBIDDEN_ROOT_WORDS = ("fresh", "oct")


def research_roots(expanded_root: Path, data_root: Path) -> list[Path]:
    roots = [r for r in (expanded_root, data_root) if (r / "metadata" / "plan.json").is_file()]
    for r in roots:
        if any(w in r.name.lower() for w in FORBIDDEN_ROOT_WORDS):
            raise SystemExit(f"Refusing root {r.name}: October / fresh-holdout data is never loaded in S13.2")
    return roots


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def run(*, expanded_root: Path = DEFAULT_EXPANDED, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT,
        s12_dir: Path = S12_DIR, s131_summary: Path = S131_SUMMARY, comparison_sources: dict[str, Path] | None = None,
        underlying: str = "NIFTY", mutation_days: int = 4, check_sample: int = 20,
        secret_values: list[str] | None = None) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    roots = research_roots(expanded_root, data_root)
    if not roots:
        raise SystemExit(f"No stored history under {expanded_root} or {data_root} (not downloading)")
    lo, hi = s13.CLEAN_START, s13.MAX_LOAD
    with order_placement_blocked(), capture_upstox_events():
        say(f"loading ONLY {lo} .. {hi} from {[r.name for r in roots]} (July+ / October files never opened) ...")
        market, read_log = s122.load_clean_market([str(r) for r in roots], underlying, lo, hi)
        if market.spot.empty:
            raise SystemExit("No NIFTY candles in the clean window (not downloading)")
        hx.guard_no_october(*market.sessions)
        mx = s122.read_summary(read_log)["max_raw_timestamp"]
        assert mx is None or mx[:10] <= hi.isoformat(), f"max_loaded_timestamp {mx} > {hi}"
        off_grid = hx.quarantine_off_grid(market)
        g12 = s122.s12_gate(_load(s12_dir / "summary.json"), _load(s12_dir / "clean_history_manifest.json"), lo, hi)
        g131 = s13.s131_gate(_load(s131_summary), lo, hi)
        dq = {"passed": bool(g12["passed"] and g131["passed"]), "s12": g12, "s13_1": g131}
        a = s13.build_study(market=market, read_log=read_log, data_quality=dq,
                            comparison_sources=comparison_sources or COMPARISON_SOURCES,
                            mutation_days=mutation_days, check_sample=check_sample, progress=say)
        a["read_summary"] = s122.read_summary(read_log)
        mx = a["read_summary"]["max_raw_timestamp"]
        assert mx is None or mx[:10] <= hi.isoformat(), f"max_loaded_timestamp {mx} > {hi}"
        a["data_quality"]["off_grid_quarantine"] = hx.quarantine_report(off_grid)
    a["production_unchanged"] = production_hashes() == before
    if not a["production_unchanged"]:
        a["forward_paper"] = "NOT SAFE"
    paths = s13.write_reports(a, out_dir)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
    if not leak_ok:
        a["forward_paper"] = "NOT SAFE"
    paths = s13.write_reports(a, out_dir)
    print("\n" + s13.terminal_text(a))
    if not leak_ok:
        print("\nTOKEN LEAK: FAIL")
        for f in leaks:
            print(f"  token value found in: {f}")
    print(f"\nReports: {out_dir}")
    for p in paths.values():
        print(f"  {p}")
    return a


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--expanded-root", type=Path, default=DEFAULT_EXPANDED)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mutation-days", type=int, default=4)
    a = p.parse_args(argv)
    run(expanded_root=a.expanded_root, data_root=a.data_root, out_dir=a.out_dir, mutation_days=a.mutation_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
