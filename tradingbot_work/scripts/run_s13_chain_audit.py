#!/usr/bin/env python3
"""S13.1 - historical option-chain data availability audit (PAPER | RESEARCH ONLY | NO ORDERS | no downloads, no ML).

  python scripts/run_s13_chain_audit.py

Reads the stored option / NIFTY files in data/historical/upstox_expanded + data/historical/upstox_6m for 2025-09-26 ..
2026-09-30 (October never read), reconstructs the point-in-time option chain and measures OHLC / volume / OI / strike
coverage. Nothing is repaired, filled or inferred. Reports: reports/strategy_improvement/s13_chain_audit/
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
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s13_chain_audit as s13
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s13_chain_audit"
DEFAULT_EXPANDED = ROOT / "data" / "historical" / "upstox_expanded"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
FORBIDDEN_ROOT_WORDS = ("fresh", "oct")


def research_roots(expanded_root: Path, data_root: Path) -> list[Path]:
    roots = [r for r in (expanded_root, data_root) if (r / "metadata" / "plan.json").is_file()]
    for r in roots:
        if any(w in r.name.lower() for w in FORBIDDEN_ROOT_WORDS):
            raise SystemExit(f"Refusing root {r.name}: October / fresh-holdout data is never read in S13.1")
    return roots


def run(*, expanded_root: Path = DEFAULT_EXPANDED, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT,
        underlying: str = "NIFTY", sample: int = 40, secret_values: list[str] | None = None) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    roots = research_roots(expanded_root, data_root)
    if not roots:
        raise SystemExit(f"No stored data under {expanded_root} or {data_root} (not downloading)")
    with order_placement_blocked(), capture_upstox_events():
        a = s13.build_audit(roots, underlying=underlying, sample=sample, progress=say)
    mx = a["load_info"]["max_raw"]
    assert mx is None or mx[:10] <= s13.LOAD_END.isoformat(), f"read a candle after {s13.LOAD_END}: {mx}"
    a["production_unchanged"] = production_hashes() == before
    paths = s13.write_reports(a, out_dir)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
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
    p.add_argument("--sample", type=int, default=40, help="timestamps per period for the truncation check")
    a = p.parse_args(argv)
    run(expanded_root=a.expanded_root, data_root=a.data_root, out_dir=a.out_dir, sample=a.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
