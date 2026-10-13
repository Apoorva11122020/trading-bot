#!/usr/bin/env python3
"""S9.1 - download + validate real NIFTY FUTURES 5-minute OHLCV (PAPER | DATA ACQUISITION ONLY | NO ORDERS).

  python scripts/download_nifty_futures_6m.py              # discover contracts, download, validate
  python scripts/download_nifty_futures_6m.py --resume     # skip contracts whose stored file is already valid
  python scripts/download_nifty_futures_6m.py --validate   # no API calls: re-validate stored futures data

Contracts come from the official Upstox instrument master (live FUT) and the expired-instrument API (expired FUT);
no instrument key is hard-coded. Read-only GET requests only; the access token is read from the environment / .env
by the existing historical client and is never printed or written. Output: data/historical/upstox_6m/futures/
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values, load_dotenv

load_dotenv(ROOT / ".env", override=False)

from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.historical_data import upstox_client as uc
from src.historical_data.models import HistoricalDataError
from src.historical_data.service import _refuse_live, build_catalog
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.safety.live_gate import assert_paper_trading_only, trading_mode

DEFAULT_ROOT = ROOT / u6.DEFAULT_ROOT


def secrets() -> list[str]:
    """Configured Upstox token values, in memory only, used solely for the leak scan."""
    names = (*uc.TOKEN_ENV_NAMES, *uc.FORBIDDEN_TOKEN_ENV_NAMES)
    values = list(uc._token_settings().values())
    if uc.ENV_FILE.is_file():
        values += [v for k, v in dotenv_values(uc.ENV_FILE).items() if k in names and v]
    values += [os.environ[k] for k in names if os.environ.get(k)]
    return [v.strip() for v in values if v and v.strip()]


def run(*, start: date = u6.DEFAULT_FROM, end: date = u6.DEFAULT_TO, root: Path = DEFAULT_ROOT,
        resume: bool = False, validate_only: bool = False, client=None, catalog=None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    if start > end:
        raise SystemExit(f"--from-date {start} is after --to-date {end}")
    spot = u6.load_underlying(u6.Layout(root), start, end)
    if spot.empty:
        raise SystemExit(f"No NIFTY spot candles under {root}/underlying (run scripts/download_upstox_6m.py first)")
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
    with order_placement_blocked(), capture_upstox_events():
        try:
            return nf.run_futures_download(client=client, catalog=catalog, root=root, start=start, end=end, spot=spot,
                                           secrets=secrets(), resume=resume, validate_only=validate_only,
                                           progress=lambda m: print(f"  {m}", flush=True))
        except HistoricalDataError as exc:
            raise SystemExit(f"STOPPED: {exc.code} - {exc.message}. Re-run with --resume after fixing it.") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-date", type=date.fromisoformat, default=u6.DEFAULT_FROM)
    p.add_argument("--to-date", type=date.fromisoformat, default=u6.DEFAULT_TO)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate", action="store_true", help="no API calls; validate stored futures data")
    a = p.parse_args(argv)
    q = run(start=a.from_date, end=a.to_date, root=a.root, resume=a.resume, validate_only=a.validate)
    print(f"\nNIFTY FUTURES 5-MIN DATA\n{nf.BANNER}\n")
    print("\n".join(nf.data_text(q)))
    if q["status"] == nf.FUTURES_VOLUME_UNAVAILABLE:
        print(f"\n{nf.FUTURES_VOLUME_UNAVAILABLE}")
    print(f"\nFiles: {nf.FuturesLayout(a.root).dir}")
    print(f"\nORDERS:\n0\n\nTRADING MODE:\n{trading_mode()}")
    if q["pass"]:
        print("\nNext: python scripts/run_s9_futures_volume.py")
    else:
        print("\nFutures data quality did not PASS - do NOT run the S9 futures replay. See data_quality.json.")
    return 0 if q["token_leak"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
