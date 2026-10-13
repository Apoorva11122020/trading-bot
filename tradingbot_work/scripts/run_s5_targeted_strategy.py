#!/usr/bin/env python3
"""S5 TARGETED STRATEGY RESEARCH (PAPER | RESEARCH ONLY | NO ORDERS; production S2 untouched).

Exact S2 on data/historical/upstox_6m, causal signal features, 13 pre-declared filter candidates, selection on
TRAIN + VALIDATION only, OOS gate, one frozen September holdout. Reports go to reports/strategy_improvement/.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s4_six_month_research import load_s2
from src.backtest import shadow_historical as sh
from src.historical_data import upstox_6m as u6
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.research import s5_targeted_strategy as s5
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_improvement"
DEFAULT_DATA = ROOT / "data" / "historical" / "upstox_6m"
S2_FILES = ("src/backtest/research_strategies.py", "src/backtest/s2_1dte_improvement.py",
            "src/backtest/s2_direction_research.py", "src/backtest/current_day_s2.py", "src/options/strategy.py",
            "src/options/selection.py", "config/strategy.yaml", "config/strategy_research.yaml")


def s2_hashes() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in S2_FILES if (ROOT / f).is_file()}


def module_safety() -> dict:
    order_call = "place" + "_order"
    bad = []
    for mod in (s5, dg, sys.modules[__name__]):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
        if order_call in src or any("OrderManager" in ln or "order_manager" in ln for ln in imports):
            bad.append(Path(mod.__file__).name)
    return {"ok": not bad, "detail": f"scanned s5 / diagnostic / runner sources: {bad or 'no order path'}"}


def run(*, data_root: Path = DEFAULT_DATA, out_dir: Path = DEFAULT_OUT, underlying: str = "NIFTY",
        check_sample: int = 20) -> dict:
    assert_paper_trading_only()
    before = s2_hashes()
    say = lambda m: print(f"  {m}", flush=True)  # noqa: E731
    store = s4.SixMonthStore(data_root, underlying)
    spot = store.spot()
    if spot.empty:
        raise SystemExit(f"No 6-month underlying candles under {data_root} (not downloading)")
    rc, params, session, s2cfg, signal_fn = load_s2(underlying)
    sessions, _ = sh.complete_sessions(spot, store.start, store.end, params.interval)
    days = [d.isoformat() for d in sessions]
    with order_placement_blocked(), capture_upstox_events():
        say(f"exact S2 replay ({len(days)} sessions) ...")
        full = dra.replay(spot=spot, days=days, signal_fn=signal_fn, resolve=s4.make_resolve(store, rc),
                          premium_for=store.premium_for(None), params=params, square_off=rc.session.square_off,
                          range_start=s2cfg["range_start"], range_end=s2cfg["range_end"])
        say("causal signal features + option confirmation ...")
        rows = s5.prepare_rows(dg.annotate(full["rows"], dg.feature_bars(spot), full["ctx_by_key"]),
                               full["ctx_by_key"])
        dataset = {"from": store.start.isoformat(), "to": store.end.isoformat(), "sessions": len(days),
                   "spot_candles": len(spot), "contracts": len(store.contracts),
                   "option_candles": sum(len(u6.read_parquet(p)) for p in store.layout.option_files())}
        say("selection (TRAIN+VALIDATION), OOS gate, walk-forward, frozen September holdout ...")
        a = s5.build_s5(rows=rows, ctx_by_key=full["ctx_by_key"], spot=spot, dataset=dataset,
                        s2_sources_unchanged=None, module_safety=module_safety(), check_sample=check_sample)
    unchanged = s2_hashes() == before
    a["integrity"][0] = dict(a["integrity"][0], status="PASS" if unchanged else "FAIL",
                             detail="sha256 of S2 strategy, selection, options and config files before vs after")
    a["integrity_ok"] = all(x["status"] != "FAIL" for x in a["integrity"])
    if not unchanged:
        a["recommendation"] = s5.REJECTED
    paths = s5.write_reports(a, out_dir)
    print("\n" + s5.summary_text(a))
    print(s5.BANNER)
    print("Reports: " + ", ".join(str(p) for p in paths.values()))
    return a


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run(data_root=a.data_root, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
