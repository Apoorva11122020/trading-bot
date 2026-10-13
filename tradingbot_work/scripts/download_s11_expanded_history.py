#!/usr/bin/env python3
"""S11 - download + validate the EXPANDED NIFTY history (PAPER | DATA ACQUISITION ONLY | NO ORDERS).

  python scripts/download_s11_expanded_history.py --probe      # expiry-calendar availability only (few requests)
  python scripts/download_s11_expanded_history.py              # NIFTY index + point-in-time options + futures
  python scripts/download_s11_expanded_history.py --resume     # skip windows already stored and valid
  python scripts/download_s11_expanded_history.py --validate   # no API calls: re-validate stored data

Requested range 2025-01-01 -> 2026-03-31 (2026-04..09 is reused from data/historical/upstox_6m; nothing is
re-downloaded there). The real start is the first session whose nearest expiry is provable from the official
Upstox expired-instrument listing (src/research/s11_history.py); earlier / gap periods are recorded as missing,
never filled. Reuses the project's downloaders (scripts/download_upstox_6m.py, scripts/download_nifty_futures_6m.py):
contracts from the instrument master / expired-instrument API, each contract in its own parquet file, no
continuous contract, incremental + resumable manifest. October 2026 is never requested.
Read-only GET requests; the access token is read from the environment / .env and never printed or written.
Output: data/historical/upstox_expanded/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from scripts import download_nifty_futures_6m as dfut
from scripts import download_upstox_6m as d6
from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.historical_data import upstox_client as uc
from src.historical_data.models import HistoricalDataError
from src.historical_data.service import _refuse_live, build_catalog
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.research import s11_history as hx
from src.safety.live_gate import assert_paper_trading_only, trading_mode

EXPANDED_ROOT = ROOT / "data" / "historical" / "upstox_expanded"
PROBE_FILE = "s11_history_probe.json"


def probe_path(root: Path) -> Path:
    return root / "metadata" / PROBE_FILE


def probe(catalog, start: date, end: date, root: Path, underlying: str = "NIFTY") -> dict:
    """Official expired-expiry listing -> point-in-time resolvable start (no candles requested)."""
    try:
        expiries = catalog.expired_expiries(underlying)
    except HistoricalDataError as exc:
        raise SystemExit(f"STOPPED: expired-instrument expiries unavailable ({exc.code}) - cannot prove past "
                         "expiries, nothing downloaded") from None
    cov = hx.expiry_coverage(expiries, start, end)
    rs = date.fromisoformat(cov["resolvable_start"]) if cov["resolvable_start"] else None
    out = {**cov, "underlying": underlying, "probed_at": datetime.now(u6.IST).isoformat(timespec="seconds"),
           "estimated_history_months_to_train_end": hx.estimated_months(rs),
           "min_history_months": hx.MIN_HISTORY_MONTHS,
           "estimated_status": hx.HISTORY_OK if hx.estimated_months(rs) >= hx.MIN_HISTORY_MONTHS
           else hx.HISTORY_INSUFFICIENT,
           "note": "final history status is decided by scripts/run_s11_ml_expanded.py from genuine data coverage"}
    p = probe_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run(*, start: date = hx.REQUESTED_START, end: date = hx.EXPANDED_END, root: Path = EXPANDED_ROOT,
        probe_only: bool = False, resume: bool = False, validate_only: bool = False, client=None, catalog=None,
        now: datetime | None = None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    hx.guard_no_october(start, end)
    if end > hx.EXPANDED_END:
        raise SystemExit(f"--to-date {end} is after {hx.EXPANDED_END}: 2026-04..09 is reused from upstox_6m")
    if start > end:
        raise SystemExit(f"--from-date {start} is after --to-date {end}")
    if validate_only:
        plan = u6.Layout(root).plan
        if not plan.is_file():
            raise SystemExit(f"{plan} missing - run without --validate first")
        p = json.loads(plan.read_text(encoding="utf-8"))
        dl_start, dl_end = date.fromisoformat(p["from"]), date.fromisoformat(p["to"])
        pr = json.loads(probe_path(root).read_text(encoding="utf-8")) if probe_path(root).is_file() else {}
    else:
        if client is None:
            source = uc.token_source()
            if source is None:
                raise SystemExit("No historical token configured (set one of: " + ", ".join(uc.TOKEN_ENV_NAMES)
                                 + " in .env). Sandbox tokens are never used for market data.")
            print(f"Token source variable: {source} (value never printed)")
            client = uc.UpstoxHistoricalClient(token_provider=uc.configured_token, max_retries=4,
                                               backoff_seconds=1.0, max_backoff_seconds=30.0,
                                               min_interval_seconds=0.35)
        if catalog is None:
            catalog = build_catalog(client)
        with order_placement_blocked(), capture_upstox_events():
            pr = probe(catalog, start, end, root)
        print(f"  expiry calendar: earliest listed {pr['earliest_expiry']}, latest {pr['latest_expiry']}, "
              f"gaps {len(pr['gaps'])}; point-in-time resolvable from {pr['resolvable_start']}")
        print(f"  estimated history to {hx.TRAIN_END}: {pr['estimated_history_months_to_train_end']} months "
              f"({pr['estimated_status']})")
        if probe_only or pr["resolvable_start"] is None:
            return {"probe": pr, "options": None, "futures": None}
        dl_start, dl_end = date.fromisoformat(pr["resolvable_start"]), end
    print(f"S11 expanded window: {dl_start} -> {dl_end} (requested {start} -> {end})")
    s = d6.run(start=dl_start, end=dl_end, root=root, report_dir=root / "reports", resume=resume,
               validate_only=validate_only, client=client, catalog=catalog, now=now)
    q = dfut.run(start=dl_start, end=dl_end, root=root, resume=resume, validate_only=validate_only, client=client,
                 catalog=catalog)
    return {"probe": pr, "options": s, "futures": q, "window": [dl_start, dl_end]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-date", type=date.fromisoformat, default=hx.REQUESTED_START)
    p.add_argument("--to-date", type=date.fromisoformat, default=hx.EXPANDED_END)
    p.add_argument("--root", type=Path, default=EXPANDED_ROOT)
    p.add_argument("--probe", action="store_true", help="only check the official expiry listing (no candles)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate", action="store_true", help="no API calls; validate the stored expanded data")
    a = p.parse_args(argv)
    r = run(start=a.from_date, end=a.to_date, root=a.root, probe_only=a.probe, resume=a.resume,
            validate_only=a.validate)
    pr = r["probe"] or {}
    print(f"\nS11 EXPANDED HISTORY DATA\n{u6.BANNER}\n")
    print(f"requested: {a.from_date} -> {a.to_date} (+ 2026-04..09 reused from upstox_6m)")
    print(f"earliest listed expiry: {pr.get('earliest_expiry')}")
    print(f"point-in-time resolvable from: {pr.get('resolvable_start')}")
    for m in pr.get("missing_periods", []):
        print(f"missing: {m['from']} -> {m['to']} ({m['reason']})")
    print(f"estimated history to {hx.TRAIN_END}: {pr.get('estimated_history_months_to_train_end')} months "
          f"({pr.get('estimated_status')}; minimum {hx.MIN_HISTORY_MONTHS})")
    s, q = r.get("options"), r.get("futures")
    if s is None:
        print("\nNo candles downloaded (probe only or no resolvable history).")
        print(f"\nORDERS:\n0\n\nTRADING MODE:\n{trading_mode()}")
        return 0
    layout = u6.Layout(a.root)
    leak_ok, leaks = u6.token_leak_check([layout.metadata, a.root / "reports", layout.root / "underlying",
                                          layout.root / "options", nf.FuturesLayout(a.root).dir], dfut.secrets())
    print(f"\nwindow: {s['period']['from']} -> {s['period']['to']}")
    print(f"NIFTY: candles {s['underlying']['candles']}, trading days {s['underlying']['trading_days']}, "
          f"status {s['underlying']['status']}")
    print(f"OPTIONS: contracts {s['options']['contracts']}, candles {s['options']['candles']}, "
          f"coverage {s['option_coverage']}, ATM coverage {s['options']['atm_coverage']}")
    print(f"OPTION DATA QUALITY: {'PASS' if s['quality']['pass'] else 'FAIL'}\n")
    print("\n".join(nf.data_text(q)))
    print(f"\nTOKEN LEAK:\n{'PASS' if leak_ok else 'FAIL'}")
    for f in leaks:
        print(f"  token value found in: {f}")
    print(f"\nFiles: {a.root}\n\nORDERS:\n0\n\nTRADING MODE:\n{trading_mode()}\nOCTOBER: NOT REQUESTED")
    print("\nMonth-level coverage gaps are expected in older data; the research script decides the usable "
          "contiguous history. Next: python scripts/run_s11_ml_expanded.py")
    return 0 if leak_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
