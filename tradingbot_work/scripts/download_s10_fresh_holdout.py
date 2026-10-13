#!/usr/bin/env python3
"""S10 - download + validate the FRESH_UNSEEN_HOLDOUT (PAPER | DATA ACQUISITION ONLY | NO ORDERS).

  python scripts/download_s10_fresh_holdout.py              # 2026-10-01 -> latest complete session (max 2026-10-06)
  python scripts/download_s10_fresh_holdout.py --resume     # skip windows already stored and valid
  python scripts/download_s10_fresh_holdout.py --validate   # no API calls: re-validate the stored October data

Reuses the project's Upstox downloaders (NIFTY index + point-in-time NIFTY options, then real NIFTY futures) and its
authentication. Read-only GET requests; the access token is read from the environment / .env and never printed.
Stored separately under data/historical/upstox_fresh_oct/ so October can never leak into S10 model building; the S10
research script loads it only after the model, threshold and features are frozen.
"""
from __future__ import annotations

import argparse
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
from src.historical_data.extension import latest_completed_day
from src.historical_data.service import _refuse_live, build_catalog
from src.research import s10_ml_data as sd
from src.safety.live_gate import assert_paper_trading_only, trading_mode

FRESH_ROOT = ROOT / "data" / "historical" / "upstox_fresh_oct"
FRESH_FROM, FRESH_TO = sd.SPLITS["FRESH_HOLDOUT"]


def run(*, start: date = FRESH_FROM, end: date = FRESH_TO, root: Path = FRESH_ROOT, resume: bool = False,
        validate_only: bool = False, client=None, catalog=None, now: datetime | None = None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    now = now or datetime.now(u6.IST)
    latest = latest_completed_day(now)
    if end > latest:
        print(f"NOTE: end date clamped from {end} to {latest} (later sessions are not complete).")
        end = latest
    if start > end:
        raise SystemExit(f"No complete session between {start} and {end} yet - nothing to download.")
    if not validate_only and client is None:
        source = uc.token_source()
        if source is None:
            raise SystemExit("No historical token configured (set one of: " + ", ".join(uc.TOKEN_ENV_NAMES)
                             + " in .env). Sandbox tokens are never used for market data.")
        print(f"Token source variable: {source} (value never printed)")
        client = uc.UpstoxHistoricalClient(token_provider=uc.configured_token, max_retries=4, backoff_seconds=1.0,
                                           max_backoff_seconds=30.0, min_interval_seconds=0.35)
    if not validate_only and catalog is None:
        catalog = build_catalog(client)
    print(f"FRESH_UNSEEN_HOLDOUT window: {start} -> {end}")
    s = d6.run(start=start, end=end, root=root, report_dir=root / "reports", resume=resume,
               validate_only=validate_only, client=client, catalog=catalog, now=now)
    q = dfut.run(start=start, end=end, root=root, resume=resume, validate_only=validate_only, client=client,
                 catalog=catalog)
    return {"period": [start, end], "options": s, "futures": q}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-date", type=date.fromisoformat, default=FRESH_FROM)
    p.add_argument("--to-date", type=date.fromisoformat, default=FRESH_TO)
    p.add_argument("--root", type=Path, default=FRESH_ROOT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate", action="store_true", help="no API calls; validate the stored October data")
    a = p.parse_args(argv)
    r = run(start=a.from_date, end=a.to_date, root=a.root, resume=a.resume, validate_only=a.validate)
    s, q = r["options"], r["futures"]
    layout = u6.Layout(a.root)
    leak_ok, leaks = u6.token_leak_check([layout.metadata, a.root / "reports", layout.root / "underlying",
                                          layout.root / "options", nf.FuturesLayout(a.root).dir], dfut.secrets())
    print(f"\nS10 FRESH OCTOBER HOLDOUT DATA\n{u6.BANNER}\n")
    print(f"Period:\n{s['period']['from']} -> {s['period']['to']}\n")
    print(f"NIFTY:\ncandles: {s['underlying']['candles']}\ntrading days: {s['underlying']['trading_days']}\n"
          f"status: {s['underlying']['status']}\n")
    print(f"OPTIONS:\ncontracts: {s['options']['contracts']}\ncandles: {s['options']['candles']}\n"
          f"coverage: {s['option_coverage']}\n")
    print(f"OPTION DATA QUALITY:\n{'PASS' if s['quality']['pass'] else 'FAIL'}\n")
    print("\n".join(nf.data_text(q)))
    print(f"\nTOKEN LEAK:\n{'PASS' if leak_ok else 'FAIL'}")
    for f in leaks:
        print(f"  token value found in: {f}")
    print(f"\nFiles: {a.root}\n\nORDERS:\n0\n\nTRADING MODE:\n{trading_mode()}")
    print("\nThis data is NOT used for feature, model or threshold selection. Next: "
          "python scripts/run_s10_ml_probability_strategy.py")
    return 0 if leak_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
