#!/usr/bin/env python3
"""S12.2 - constrained 9.1-month ML research (PAPER | RESEARCH ONLY | NO ORDERS | no downloads).

  python scripts/run_s12_data_quality.py        # first (S12 verdicts are the data-quality gate)
  python scripts/run_s12_2_constrained_ml.py

Clean window ONLY: 2025-09-26 -> 2026-06-30. TRAIN 2025-09-26..2026-03-31 (three inner folds), VALIDATION April 2026,
OOS May-June 2026 (only if April passes). Files that could hold candles after 2026-06-30 are never opened; July /
August / September / October are NOT loaded; the script asserts max_loaded_timestamp <= 2026-06-30. Not a 12-month or
18-month validation. Reports: reports/strategy_improvement/s12_2_constrained_ml/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_nifty_futures_6m import secrets
from scripts.run_clean_historical_backtest import production_hashes
from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s10_ml_data as sd
from src.research import s11_history as hx
from src.safety.live_gate import assert_paper_trading_only

try:
    from src.research import s12_2_constrained_ml as s122
except ModuleNotFoundError as exc:  # scikit-learn is an optional research dependency
    if exc.name and exc.name.startswith("sklearn"):
        raise SystemExit("scikit-learn is not installed. Install it yourself first: pip install scikit-learn") \
            from None
    raise

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s12_2_constrained_ml"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
DEFAULT_EXPANDED = ROOT / "data" / "historical" / "upstox_expanded"
REPORTS = ROOT / "reports" / "strategy_improvement"
S12_DIR = REPORTS / "s12_data_quality"
COMPARISON_SOURCES = {"S2": REPORTS / "clean_historical_backtest" / "monthly_results.csv",
                      "S9": REPORTS / "s9_futures_volume" / "monthly_results.csv",
                      "S10": REPORTS / "s10_ml" / "monthly_results.csv",
                      "S11": REPORTS / "s11_ml_expanded" / "trade_log.csv"}
FORBIDDEN_ROOT_WORDS = ("fresh", "oct")


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def research_roots(expanded_root: Path, data_root: Path) -> list[Path]:
    roots = [r for r in (expanded_root, data_root) if (r / "metadata" / "plan.json").is_file()]
    for r in roots:
        if any(w in r.name.lower() for w in FORBIDDEN_ROOT_WORDS):
            raise SystemExit(f"Refusing root {r.name}: October / fresh-holdout data is never loaded in S12.2")
    return roots


def store_info(roots: list[Path]) -> dict:
    out = {}
    for r in roots:
        lay = u6.Layout(r, "NIFTY")
        out[r.name] = {"plan_sha256": _sha(lay.plan), "contracts_sha256": _sha(lay.contracts),
                       "summary_sha256": _sha(lay.summary),
                       "futures_contracts_sha256": _sha(nf.FuturesLayout(r, "NIFTY").contracts)}
    return out


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def run(*, expanded_root: Path = DEFAULT_EXPANDED, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT,
        s12_dir: Path = S12_DIR, comparison_sources: dict[str, Path] | None = None, underlying: str = "NIFTY",
        mutation_days: int = 6, check_sample: int = 20, secret_values: list[str] | None = None) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    roots = research_roots(expanded_root, data_root)
    if not roots:
        raise SystemExit(f"No stored history under {expanded_root} or {data_root} (not downloading)")
    lo, hi = s122.CLEAN_START, s122.MAX_LOAD
    with order_placement_blocked(), capture_upstox_events():
        say(f"loading ONLY {lo} .. {hi} from {[r.name for r in roots]} (July+ / October files never opened) ...")
        market, read_log = s122.load_clean_market([str(r) for r in roots], underlying, lo, hi)
        if market.spot.empty:
            raise SystemExit("No NIFTY candles in the clean window (not downloading)")
        hx.guard_no_october(*market.sessions)
        mx = s122.read_summary(read_log)["max_raw_timestamp"]
        assert mx is None or mx[:10] <= hi.isoformat(), f"max_loaded_timestamp {mx} > {hi}"
        off_grid = hx.quarantine_off_grid(market)
        dq = s122.s12_gate(_load(s12_dir / "summary.json"), _load(s12_dir / "clean_history_manifest.json"), lo, hi)
        availability = sd.availability_audit(s122.phase_view(market, s122.TRAIN_END), train=(lo, s122.TRAIN_END))
        a = s122.build_study(market=market, read_log=read_log, availability=availability, data_quality=dq,
                             freeze_dir=out_dir, comparison_sources=comparison_sources or COMPARISON_SOURCES,
                             mutation_days=mutation_days, check_sample=check_sample, progress=say)
        a["read_summary"] = s122.read_summary(read_log)
        mx = a["read_summary"]["max_raw_timestamp"]
        assert mx is None or mx[:10] <= hi.isoformat(), f"max_loaded_timestamp {mx} > {hi}"
        a["data_quality"]["off_grid_quarantine"] = hx.quarantine_report(off_grid)
        man = s122.manifest(a, market, store_info(roots))
        man["off_grid_quarantine"] = a["data_quality"]["off_grid_quarantine"]
    a["production_unchanged"] = production_hashes() == before
    if not a["production_unchanged"]:
        a["forward_paper"] = "NOT SAFE"
    paths = s122.write_reports(a, out_dir, man)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
    if not leak_ok:
        a["forward_paper"] = "NOT SAFE"
    print("\n" + s122.terminal_text(a))
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
    p.add_argument("--mutation-days", type=int, default=6)
    a = p.parse_args(argv)
    run(expanded_root=a.expanded_root, data_root=a.data_root, out_dir=a.out_dir, mutation_days=a.mutation_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
