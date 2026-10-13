#!/usr/bin/env python3
"""Download real NIFTY / BANK NIFTY futures OHLCV for pivot-strategy research.

PAPER | DATA ACQUISITION ONLY | NO ORDERS.
Reads the existing spot-index CSV only for session timestamps, spot/futures alignment,
and daily price reference. The volume filter must use the downloaded FUTURES volume,
never the spot-index CSV's (typically zero) volume.

Examples (from tradingbot_work):
  python scripts/download_pivot_futures.py --symbol NIFTY --spot-csv ../data/NIFTY_50_5m_2025-10-01_to_2026-10-09.csv
  python scripts/download_pivot_futures.py --symbol BANKNIFTY --spot-csv ../data/BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv
  python scripts/download_pivot_futures.py --symbol NIFTY --spot-csv path/to/nifty.csv --resume

Requires a valid Upstox historical-data token in .env and expired-contract API access for
past expiries. Token values are never printed or written to output files.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

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

DEFAULT_START = date(2025, 10, 1)
DEFAULT_END = date(2026, 10, 8)
DEFAULT_NIFTY_CSV = ROOT.parent / "data" / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"
DEFAULT_BANKNIFTY_CSV = ROOT.parent / "data" / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"
DEFAULT_ROOT = ROOT / "data" / "historical" / "pivot_futures"


def configured_secrets() -> list[str]:
    """Return configured token values in memory only for output leak scanning."""
    names = (*uc.TOKEN_ENV_NAMES, *uc.FORBIDDEN_TOKEN_ENV_NAMES)
    values = list(uc._token_settings().values())
    if uc.ENV_FILE.is_file():
        values += [v for k, v in dotenv_values(uc.ENV_FILE).items() if k in names and v]
    values += [os.environ[k] for k in names if os.environ.get(k)]
    return [v.strip() for v in values if v and v.strip()]


def load_spot_csv(path: str | Path, start: date, end: date) -> pd.DataFrame:
    """Load real spot OHLC timestamps/prices as a reference; never use its volume."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Spot CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {"open", "high", "low", "close"}
    missing = required - set(frame.columns)
    timestamp_col = "timestamp_ist" if "timestamp_ist" in frame.columns else "timestamp"
    if timestamp_col not in frame.columns:
        missing.add("timestamp (or timestamp_ist)")
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    raw_ts = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if raw_ts.dt.tz is None:
        # Upstox exports use UTC timestamps; explicit *_ist columns are local.
        if timestamp_col == "timestamp_ist":
            raw_ts = raw_ts.dt.tz_localize("Asia/Kolkata")
        else:
            raw_ts = raw_ts.dt.tz_localize("UTC")
    frame["timestamp"] = raw_ts.dt.tz_convert("Asia/Kolkata")
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    local_dates = frame["timestamp"].dt.date
    frame = frame[(local_dates >= start) & (local_dates <= end)].copy()
    minutes = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame = frame[(minutes >= 9 * 60 + 15) & (minutes < 15 * 60 + 30)]
    if frame.empty:
        raise ValueError(f"No regular-session spot candles in {path} for {start} through {end}")
    # Volume is deliberately irrelevant to spot/futures alignment and is not copied.
    return frame[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)


def run(*, symbol: str, spot_csv: Path, start: date, end: date, root: Path = DEFAULT_ROOT,
        resume: bool = False, validate_only: bool = False, client=None, catalog=None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    if start > end:
        raise SystemExit(f"--from-date {start} is after --to-date {end}")
    symbol = symbol.upper()
    if symbol not in {"NIFTY", "BANKNIFTY"}:
        raise SystemExit("--symbol must be NIFTY or BANKNIFTY")

    spot = load_spot_csv(spot_csv, start, end)
    symbol_root = root / symbol.lower()
    if not validate_only and client is None:
        source = uc.token_source()
        if source is None:
            raise SystemExit(
                "No historical token configured (set one of: " + ", ".join(uc.TOKEN_ENV_NAMES)
                + " in tradingbot_work/.env). Sandbox tokens cannot fetch historical market data."
            )
        print(f"Token source variable: {source} (value never printed)")
        client = uc.UpstoxHistoricalClient(
            token_provider=uc.configured_token, max_retries=4, backoff_seconds=1.0,
            max_backoff_seconds=30.0, min_interval_seconds=0.35,
        )
    if not validate_only and catalog is None:
        catalog = build_catalog(client)

    with order_placement_blocked(), capture_upstox_events():
        try:
            result = nf.run_futures_download(
                client=client, catalog=catalog, root=symbol_root, start=start, end=end, spot=spot,
                secrets=configured_secrets(), resume=resume, validate_only=validate_only,
                underlying=symbol, progress=lambda message: print(f"  {message}", flush=True),
            )
        except HistoricalDataError as exc:
            raise SystemExit(f"STOPPED: {exc.code} - {exc.message}. Check access/plan, then retry with --resume.") from None

    print(f"\n{symbol} FUTURES DATA\n{nf.BANNER}")
    print("\n".join(nf.data_text(result)))
    print(f"\nData quality: {result['status']}")
    print(f"Futures files: {nf.FuturesLayout(symbol_root, symbol).dir}")
    print(f"Orders: 0\nTrading mode: {trading_mode()}")
    if not result["pass"]:
        print("STOP: do not use this dataset for strategy validation until every quality gate passes.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", choices=("NIFTY", "BANKNIFTY"), required=True)
    parser.add_argument("--spot-csv", type=Path, help="existing spot OHLC CSV; its volume column is ignored")
    parser.add_argument("--from-date", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--to-date", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true", help="offline validation only; no API calls")
    args = parser.parse_args(argv)
    default_csv = DEFAULT_NIFTY_CSV if args.symbol == "NIFTY" else DEFAULT_BANKNIFTY_CSV
    spot_csv = args.spot_csv or default_csv
    result = run(symbol=args.symbol, spot_csv=spot_csv, start=args.from_date, end=args.to_date,
                 root=args.root, resume=args.resume, validate_only=args.validate)
    return 0 if result.get("token_leak", {}).get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
