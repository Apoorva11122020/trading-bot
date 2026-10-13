#!/usr/bin/env python3
"""S10.3 - L1 model ranking / precision diagnostic (PAPER | RESEARCH ONLY | NO ORDERS | DIAGNOSTIC ONLY).

  python scripts/run_s10_ranking_diagnostic.py

TRAIN (Apr-Jun) + VALIDATION (July) only; market data is cut at 2026-07-31 so August, September and October are
never read. The L1 model is the S10.2 one (fixed logistic regression on the frozen S10 preprocessor from
reports/strategy_improvement/s10_ml/models/); its July AUC is checked against label_geometry/summary.json.
Nothing is tuned or thresholded and the frozen S10 artifacts are not modified.
Reports: reports/strategy_improvement/s10_ml/ranking/
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
    from src.research import s10_ranking_diagnostic as rk
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("sklearn"):
        raise SystemExit("scikit-learn is not installed. Install it yourself first: pip install scikit-learn") \
            from None
    raise

DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
DEFAULT_S10 = ROOT / "reports" / "strategy_improvement" / "s10_ml"


def reference_auc(s10_dir: Path) -> dict[str, float] | None:
    """S10.2 COMMON_WINDOW L1 July AUC per side (None if S10.2 was not run)."""
    p = s10_dir / "label_geometry" / "summary.json"
    if not p.is_file():
        return None
    rows = json.loads(p.read_text(encoding="utf-8")).get("learnability", [])
    ref = {r["side"]: r["roc_auc"] for r in rows
           if r.get("scope") == "COMMON_WINDOW" and r.get("label") == rk.LABEL and r.get("roc_auc") is not None}
    return ref or None


def run(*, data_root: Path = DEFAULT_DATA, s10_dir: Path = DEFAULT_S10, out_dir: Path | None = None,
        underlying: str = "NIFTY", secret_values: list[str] | None = None) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    out_dir = out_dir or s10_dir / "ranking"
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    try:
        model, _threshold, _meta = dx.load_frozen(s10_dir / "models")
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc} - run scripts/run_s10_ml_probability_strategy.py first") from None
    with order_placement_blocked(), capture_upstox_events():
        say(f"loading TRAIN + VALIDATION market data only (cut at {rk.STUDY_END}) ...")
        market = sd.load_market([data_root], underlying, max_day=rk.STUDY_END)
        if market.spot.empty:
            raise SystemExit(f"No six-month underlying candles under {data_root} (not downloading)")
        a = rk.run_study(market=market, model=model, reference_auc=reference_auc(s10_dir), progress=say)
    a["production_unchanged"] = production_hashes() == before
    paths = rk.write_reports(a, out_dir)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
    print("\n" + rk.terminal_text(a))
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
