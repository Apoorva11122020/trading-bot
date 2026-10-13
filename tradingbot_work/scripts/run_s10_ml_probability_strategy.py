#!/usr/bin/env python3
"""S10 - ML probability-based NIFTY options research (PAPER | RESEARCH ONLY | NO ORDERS | no downloads).

  python scripts/run_s10_ml_probability_strategy.py

ML model estimates the historical probability of a predefined option outcome from causal features.
Fixed order (never reversed): validate October data (data-level only) -> build Apr..Sep datasets -> train / select
on TRAIN (inner June selection, purged walk-forward) -> July validation -> freeze -> August OOS -> September
(LEGACY_HOLDOUT_CONTAMINATED, report only) -> fresh October holdout, run exactly once with the frozen model.
October candles are loaded only after the freeze. Reports: reports/strategy_improvement/s10_ml/
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
from scripts.run_s4_six_month_research import load_s2
from src.backtest import shadow_historical as sh
from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.research import s5_targeted_strategy as s5
from src.research import s9_futures_volume as s9f
from src.research import s9_orb_vwap_option_momentum as s9
from src.research import s10_ml_data as sd
from src.safety.live_gate import assert_paper_trading_only

try:
    from src.research import s10_ml_probability as s10
except ModuleNotFoundError as exc:  # scikit-learn is an optional research dependency
    if exc.name and exc.name.startswith("sklearn"):
        raise SystemExit("scikit-learn is not installed. Install it yourself first: pip install scikit-learn") \
            from None
    raise

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement" / "s10_ml"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
DEFAULT_FRESH = ROOT / "data" / "historical" / "upstox_fresh_oct"
BENCH_SPLITS = ("VALIDATION", "OOS")


def fresh_october_quality(fresh_root: Path, sec: list[str], underlying: str = "NIFTY") -> dict:
    """Data-level validation only (integrity / coverage). Never builds features, labels or predictions."""
    lo, hi = sd.SPLITS["FRESH_HOLDOUT"]
    base = {"root": fresh_root.as_posix(), "planned": [lo.isoformat(), hi.isoformat()], "usable": False}
    if not (fresh_root / "metadata" / "plan.json").is_file():
        return {**base, "reason": "October data not downloaded (run scripts/download_s10_fresh_holdout.py)"}
    try:
        store = s4.SixMonthStore(fresh_root, underlying)
        spot = store.spot()
        if spot.empty:
            return {**base, "reason": "no October NIFTY candles stored"}
        sessions = [d for d in sh.complete_sessions(spot, store.start, store.end)[0] if lo <= d <= hi]
        integrity = s4.data_integrity(store, spot)
        fq = nf.validate_store(root=fresh_root, start=store.start, end=store.end, spot=spot, secrets=sec,
                               write=False)
    except Exception as exc:  # report, never fabricate or fall back
        return {**base, "reason": f"October data could not be validated ({type(exc).__name__})"}
    ok = bool(sessions) and bool(integrity.get("ok")) and bool(fq["pass"])
    reason = None if ok else ("no complete October session" if not sessions else
                              "October option/index integrity failed" if not integrity.get("ok") else
                              f"October futures data quality {fq['status']}")
    return {**base, "usable": ok, "reason": reason, "sessions": [d.isoformat() for d in sessions],
            "period": [sessions[0].isoformat(), sessions[-1].isoformat()] if sessions else None,
            "spot_candles": int(len(spot)), "integrity_ok": bool(integrity.get("ok")),
            "futures_status": fq["status"], "futures_pass": bool(fq["pass"])}


def make_benchmarks(data_root: Path, underlying: str, sec: list[str], say):
    """S2 (frozen, unchanged) and S9_FUTURES_VOLUME replays on the six-month data, VALIDATION / OOS COMBINED."""
    def run() -> dict:
        store = s4.SixMonthStore(data_root, underlying)
        spot = store.spot()
        spot = spot[spot["timestamp"].dt.date <= sd.PRE_FREEZE_END].reset_index(drop=True)
        rc, params, _session, s2cfg, signal_fn = load_s2(underlying)
        sessions, _ = sh.complete_sessions(spot, store.start, min(store.end, sd.PRE_FREEZE_END), params.interval)
        days = [d.isoformat() for d in sessions]
        premium_for, resolve = store.premium_for(None), s4.make_resolve(store, rc)
        say("benchmark: frozen S2 replay (comparison only) ...")
        full = dra.replay(spot=spot, days=days, signal_fn=signal_fn, resolve=resolve, premium_for=premium_for,
                          params=params, square_off=rc.session.square_off, range_start=s2cfg["range_start"],
                          range_end=s2cfg["range_end"])
        bars = dg.feature_bars(spot)
        s2_rows = s5.prepare_rows(dg.annotate(full["rows"], bars, full["ctx_by_key"]), full["ctx_by_key"])
        out = {k: {"S2": s9.s2_view(cb.metrics(cb.side(cb.split_rows(s2_rows, k), "COMBINED")))}
               for k in BENCH_SPLITS}
        fq = nf.validate_store(root=data_root, start=store.start, end=store.end, spot=store.spot(), secrets=sec,
                               write=False)
        if fq["pass"]:
            say("benchmark: S9_FUTURES_VOLUME replay (comparison only) ...")
            _contracts, frames = nf.load_futures(data_root, underlying)
            feat = s9f.futures_features(frames, fq["_selection"], store.spot())
            recs = s9.replay(feat, days, resolve, premium_for, s9f.CFG)[0]
            tab = s9.table(recs)
            for k in BENCH_SPLITS:
                out[k]["S9_FUTURES_VOLUME"] = s9.s9_view(tab[k]["COMBINED"])
        return out
    return run


def run(*, data_root: Path = DEFAULT_DATA, fresh_root: Path = DEFAULT_FRESH, out_dir: Path = DEFAULT_OUT,
        underlying: str = "NIFTY", mutation_days: int = 6, check_sample: int = 20,
        secret_values: list[str] | None = None, with_benchmarks: bool = True) -> dict:
    assert_paper_trading_only()
    before = production_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    sec = secrets() if secret_values is None else secret_values
    with order_placement_blocked(), capture_upstox_events():
        say("step 1: validating fresh October data (data-level only; no features / labels / predictions) ...")
        fresh_quality = fresh_october_quality(fresh_root, sec, underlying)
        say(f"  October usable: {fresh_quality['usable']} ({fresh_quality.get('reason') or 'OK'})")

        say("step 2: loading Apr..Sep data (October excluded) and checking integrity ...")
        market = sd.load_market([data_root], underlying, max_day=sd.PRE_FREEZE_END)
        if market.spot.empty:
            raise SystemExit(f"No six-month underlying candles under {data_root} (not downloading)")
        store = s4.SixMonthStore(data_root, underlying)
        integrity = s4.data_integrity(store, store.spot())
        fq = nf.validate_store(root=data_root, start=store.start, end=store.end, spot=store.spot(), secrets=sec,
                               write=False)
        availability = sd.availability_audit(market)
        data_checks = {"store_integrity_ok": bool(integrity.get("ok")), "futures_quality": fq["status"],
                       "futures_pass": bool(fq["pass"]),
                       "passed": bool(integrity.get("ok")) and bool(fq["pass"])}
        if not availability["required_ok"]:
            missing = [k for k, v in availability["required"].items() if not v]
            a = s10.not_run(availability, data_checks, fresh_quality,
                            f"required data unavailable: {', '.join(missing)}")
        else:
            benchmarks = make_benchmarks(data_root, underlying, sec, say) if with_benchmarks else (lambda: {})
            a = s10.build_study(
                market=market, availability=availability, data_checks=data_checks, fresh_quality=fresh_quality,
                fresh_loader=lambda: sd.load_market([data_root, fresh_root], underlying,
                                                    max_day=sd.SPLITS["FRESH_HOLDOUT"][1]),
                benchmarks=benchmarks, freeze_dir=out_dir, mutation_days=mutation_days,
                check_sample=check_sample, progress=say)
    a["production_unchanged"] = production_hashes() == before
    if not a["production_unchanged"]:
        a["forward_paper"] = "NOT SAFE"
    paths = s10.write_reports(a, out_dir)
    leak_ok, leaks = u6.token_leak_check([out_dir], sec)
    a["token_leak"] = {"passed": leak_ok, "files": leaks}
    if not leak_ok:
        a["forward_paper"] = "NOT SAFE"
    print("\n" + s10.terminal_text(a))
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
    p.add_argument("--fresh-root", type=Path, default=DEFAULT_FRESH)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--mutation-days", type=int, default=6)
    p.add_argument("--no-benchmarks", action="store_true", help="skip the S2 / S9 comparison replays")
    a = p.parse_args(argv)
    run(data_root=a.data_root, fresh_root=a.fresh_root, out_dir=a.out_dir, mutation_days=a.mutation_days,
        with_benchmarks=not a.no_benchmarks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
