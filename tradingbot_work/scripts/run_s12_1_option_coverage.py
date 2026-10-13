#!/usr/bin/env python3
"""S12.1 - pre-2025-09-26 ATM option coverage investigation (PAPER | RESEARCH ONLY | NO ORDERS | offline, no ML).

  python scripts/run_s12_1_option_coverage.py

Reads only stored data + stored download metadata from data/historical/upstox_expanded (+ upstox_6m) and the S12
session statuses (reports/strategy_improvement/s12_data_quality/clean_history_manifest.json). Nothing is downloaded,
modified, shifted or substituted; October 2026 is never read; S11 is not rerun.
Reports: reports/strategy_improvement/s12_1_option_coverage/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_nifty_futures_6m import secrets
from scripts.run_clean_historical_backtest import production_hashes
from scripts.run_s12_data_quality import DEFAULT_DATA, DEFAULT_EXPANDED, research_roots
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s12_1_option_coverage as s121
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s12_1_option_coverage"
DEFAULT_S12 = ROOT / "reports" / "strategy_improvement" / "s12_data_quality" / "clean_history_manifest.json"


def run(*, expanded_root: Path = DEFAULT_EXPANDED, data_root: Path = DEFAULT_DATA, s12_manifest: Path = DEFAULT_S12,
        out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY", secret_values: list[str] | None = None) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    roots = research_roots(expanded_root, data_root)
    if not roots:
        raise SystemExit(f"No stored data under {expanded_root} or {data_root} (not downloading)")
    with order_placement_blocked(), capture_upstox_events():
        a = s121.build(roots, s12_manifest, underlying=underlying, progress=say)
    a["production_unchanged"] = production_hashes() == before
    s121.write_reports(a, out_dir)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
    paths = s121.write_reports(a, out_dir)
    print("\n" + s121.terminal_text(a))
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
    p.add_argument("--s12-manifest", type=Path, default=DEFAULT_S12)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run(expanded_root=a.expanded_root, data_root=a.data_root, s12_manifest=a.s12_manifest, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
