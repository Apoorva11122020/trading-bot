#!/usr/bin/env python3
"""S11 - expanded-history ML NIFTY options research (PAPER | RESEARCH ONLY | NO ORDERS | no downloads).

  python scripts/download_s11_expanded_history.py --probe        # first (expanded history, Jan 2025 .. Mar 2026)
  python scripts/run_s11_ml_expanded.py

Fixed order (never reversed): load expanded TRAIN history + six-month store (max 2026-09-30) -> clean contiguous
history check (>= 12 months or INSUFFICIENT_HISTORICAL_DATA) -> six inner walk-forward folds -> family / threshold
selection on inner folds only -> freeze -> July validation -> August OOS (only if July passes) -> September
(LEGACY_HOLDOUT_CONTAMINATED, comparison only). October is never loaded. Reports:
reports/strategy_improvement/s11_ml_expanded/
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
from src.research import s4_six_month_lab as s4
from src.research import s10_ml_data as sd
from src.research import s11_history as hx
from src.safety.live_gate import assert_paper_trading_only

try:
    from scripts.run_s10_ml_probability_strategy import make_benchmarks
    from src.research import s11_ml_expanded as s11
except ModuleNotFoundError as exc:  # scikit-learn is an optional research dependency
    if exc.name and exc.name.startswith("sklearn"):
        raise SystemExit("scikit-learn is not installed. Install it yourself first: pip install scikit-learn") \
            from None
    raise

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s11_ml_expanded"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
DEFAULT_EXPANDED = ROOT / "data" / "historical" / "upstox_expanded"
S10_SUMMARY = ROOT / "reports" / "strategy_improvement" / "s10_ml" / "summary.json"
FORBIDDEN_ROOT_WORDS = ("fresh", "oct")


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def research_roots(expanded_root: Path, data_root: Path) -> list[Path]:
    """Expanded history (if downloaded) + the six-month store. Never an October / fresh-holdout root."""
    roots = [r for r in (expanded_root, data_root) if (r / "metadata" / "plan.json").is_file()]
    for r in roots:
        if any(w in r.name.lower() for w in FORBIDDEN_ROOT_WORDS):
            raise SystemExit(f"Refusing root {r.name}: October / fresh-holdout data is never loaded in S11")
    return roots


def store_quality(root: Path, sec: list[str], underlying: str = "NIFTY") -> dict:
    """Stored-data validity of one root (coverage gaps are judged by the clean-month history rule, not here)."""
    base = {"root": root.as_posix()}
    try:
        store = s4.SixMonthStore(root, underlying)
        spot = store.spot()
        lay = store.layout
        summary = json.loads(lay.summary.read_text(encoding="utf-8")) if lay.summary.is_file() else {}
        integrity = s4.data_integrity(store, spot)
        fq = nf.validate_store(root=root, start=store.start, end=store.end, spot=spot, secrets=sec, write=False)
    except SystemExit:
        raise
    except Exception as exc:  # report, never fabricate or fall back
        return {**base, "passed": False, "reason": f"store could not be validated ({type(exc).__name__}: {exc})"}
    q = summary.get("quality") or {}
    sel = fq.get("contract_selection") or {}
    align = fq.get("timestamp_alignment") or {}
    vw = fq.get("vwap") or {}
    reasons = sorted({str(f.get("reason")) for f in q.get("failing_files") or []})
    agg = q.get("aggregate") or {}
    hard = {k: agg.get(k, 0) for k in ("duplicate_timestamps", "duplicate_rows", "invalid_ohlc", "negative_volume",
                                       "negative_oi", "outside_session", "after_expiry", "out_of_range",
                                       "rejected_rows")}
    off_grid_only = bool(q) and reasons == ["on_grid"] and not any(hard.values())
    option_ok = bool(q.get("pass")) or off_grid_only
    futures_ok = bool(fq.get("stored_files_ok")) and not sel.get("violations") \
        and bool(sel.get("independent_of_price_and_volume", True)) and int(align.get("mismatched", 0) or 0) == 0 \
        and int(vw.get("truncation_mismatches", 0) or 0) == 0
    failing = [c["id"] for c in integrity.get("checks", []) if c.get("status") == "FAIL"]
    return {**base, "period": [store.start.isoformat(), store.end.isoformat()], "spot_candles": int(len(spot)),
            "option_quality_pass": option_ok, "option_coverage": summary.get("coverage"),
            "option_quality_raw_pass": bool(q.get("pass")), "option_failing_reasons": reasons,
            "option_failing_files": len(q.get("failing_files") or []),
            "option_off_grid_quarantined": off_grid_only,
            "futures_status": fq.get("status"), "futures_pass": bool(fq.get("pass")),
            "futures_stored_files_ok": bool(fq.get("stored_files_ok")),
            "futures_selection_violations": len(sel.get("violations") or []),
            "futures_alignment_mismatched": align.get("mismatched"),
            "s4_integrity_ok": bool(integrity.get("ok")), "s4_failing_checks_info": failing,
            "passed": option_ok and futures_ok,
            "rule": "passed = stored option files pass quality (off-5-minute-grid candles alone are quarantined: "
                    "dropped at read time, never shifted) + futures files / causal selection / alignment valid"}


def store_info(roots: list[Path]) -> dict:
    out = {}
    for r in roots:
        lay = u6.Layout(r, "NIFTY")
        out[r.name] = {"plan_sha256": _sha(lay.plan), "contracts_sha256": _sha(lay.contracts),
                       "summary_sha256": _sha(lay.summary),
                       "futures_contracts_sha256": _sha(nf.FuturesLayout(r, "NIFTY").contracts)}
    return out


def run(*, expanded_root: Path = DEFAULT_EXPANDED, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT,
        underlying: str = "NIFTY", mutation_days: int = 6, check_sample: int = 20,
        secret_values: list[str] | None = None, with_benchmarks: bool = True) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    roots = research_roots(expanded_root, data_root)
    if not roots:
        raise SystemExit(f"No stored history under {expanded_root} or {data_root} (not downloading)")
    with order_placement_blocked(), capture_upstox_events():
        say(f"loading stored history from {[r.name for r in roots]} (max {hx.RESEARCH_END}; October never loaded) ...")
        market = sd.load_market([str(r) for r in roots], underlying, max_day=hx.RESEARCH_END)
        if market.spot.empty:
            raise SystemExit("No NIFTY candles in the research roots (not downloading)")
        hx.guard_no_october(*market.sessions)
        off_grid = s11.quarantine_off_grid(market)
        say("validating stored data (per root) ...")
        quality = {r.name: store_quality(r, sec, underlying) for r in roots}
        data_quality = {"roots": quality, "passed": all(q["passed"] for q in quality.values())}
        availability = sd.availability_audit(market, train=(hx.REQUESTED_START, hx.TRAIN_END))
        s10_ref = json.loads(S10_SUMMARY.read_text(encoding="utf-8")) if S10_SUMMARY.is_file() else None
        if not availability["required_ok"]:
            missing = [k for k, v in availability["required"].items() if not v]
            a = s11.not_run({"availability": availability, "data_quality": data_quality, "october": "NOT LOADED",
                             "heading": s11.HEADING, "banner": s11.BANNER, "live_orders": 0},
                            f"required data unavailable: {', '.join(missing)}")
        else:
            benchmarks = make_benchmarks(data_root, underlying, sec, say) if with_benchmarks else (lambda: {})
            a = s11.build_study(market=market, availability=availability, data_quality=data_quality,
                                freeze_dir=out_dir, benchmarks=benchmarks, s10_reference=s10_ref,
                                mutation_days=mutation_days, check_sample=check_sample, progress=say)
        man = s11.manifest(a, market, store_info(roots))
        a.setdefault("data_quality", data_quality)["off_grid_quarantine"] = s11.quarantine_report(off_grid)
        man["off_grid_quarantine"] = a["data_quality"]["off_grid_quarantine"]
    a["production_unchanged"] = production_hashes() == before
    if not a["production_unchanged"]:
        a["forward_paper"] = "NOT SAFE"
    paths = s11.write_reports(a, out_dir, man)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
    if not leak_ok:
        a["forward_paper"] = "NOT SAFE"
    print("\n" + s11.terminal_text(a))
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
    p.add_argument("--no-benchmarks", action="store_true", help="skip the S2 / S9 comparison replays")
    a = p.parse_args(argv)
    run(expanded_root=a.expanded_root, data_root=a.data_root, out_dir=a.out_dir, mutation_days=a.mutation_days,
        with_benchmarks=not a.no_benchmarks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
