#!/usr/bin/env python3
"""S10.1 - ML signal / label / coverage diagnostic (PAPER | RESEARCH ONLY | NO ORDERS | DIAGNOSTIC ONLY).

  python scripts/run_s10_ml_diagnostic.py

Reuses the frozen S10 artifacts (reports/strategy_improvement/s10_ml/models/) on TRAIN (Apr-Jun) + VALIDATION
(July) only. Market data is loaded with a hard cut at 2026-07-31: August, September and October are never read.
No re-fitting, re-calibration, threshold / horizon / stop / target / feature / model change.
Reports: reports/strategy_improvement/s10_ml/diagnostic/
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
from src.research import s10_ml_data as sd
from src.safety.live_gate import assert_paper_trading_only

try:
    from src.research import s10_ml_diagnostic as dx
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("sklearn"):
        raise SystemExit("scikit-learn is not installed. Install it yourself first: pip install scikit-learn") \
            from None
    raise

DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
DEFAULT_S10 = ROOT / "reports" / "strategy_improvement" / "s10_ml"


def run(*, data_root: Path = DEFAULT_DATA, s10_dir: Path = DEFAULT_S10, out_dir: Path | None = None,
        underlying: str = "NIFTY", secret_values: list[str] | None = None) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    out_dir = out_dir or s10_dir / "diagnostic"
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    try:
        model, threshold, meta = dx.load_frozen(s10_dir / "models")
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc} - run scripts/run_s10_ml_probability_strategy.py first (not re-training here)") \
            from None
    summary_path = s10_dir / "summary.json"
    s10_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    with order_placement_blocked(), capture_upstox_events():
        say(f"loading TRAIN + VALIDATION market data only (cut at {dx.DIAG_END}) ...")
        market = sd.load_market([data_root], underlying, max_day=dx.DIAG_END)
        if market.spot.empty:
            raise SystemExit(f"No six-month underlying candles under {data_root} (not downloading)")
        a = dx.run_diagnostic(market=market, model=model, threshold=threshold, meta=meta, s10_summary=s10_summary,
                              progress=say)
    a["production_unchanged"] = production_hashes() == before
    paths = dx.write_reports(a, out_dir)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
    print("\n" + dx.terminal_text(a))
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
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--s10-dir", type=Path, default=DEFAULT_S10)
    p.add_argument("--out-dir", type=Path, default=None)
    a = p.parse_args(argv)
    run(data_root=a.data_root, s10_dir=a.s10_dir, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
