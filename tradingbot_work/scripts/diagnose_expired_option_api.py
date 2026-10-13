#!/usr/bin/env python3
"""Single-contract diagnostic for the Upstox Plus expired-instrument API (read-only, PAPER, no orders).

Flow (official Upstox docs):
  1. GET /v2/expired-instruments/expiries?instrument_key=NSE_INDEX|Nifty 50
  2. GET /v2/expired-instruments/option/contract?instrument_key=...&expiry_date=YYYY-MM-DD
  3. GET /v2/expired-instruments/historical-candle/{expired_instrument_key}/5minute/{to_date}/{from_date}

The expired_instrument_key is taken verbatim from step 2. The token is never printed.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_data.models import HistoricalDataError
from src.historical_data.upstox_client import UpstoxHistoricalClient, token_source
from src.safety.live_gate import assert_paper_trading_only

UNDERLYING = "NSE_INDEX|Nifty 50"
CANDLE_FIELDS = ("timestamp", "open", "high", "low", "close", "volume", "open_interest")


def fail(step: str, endpoint: str, exc: HistoricalDataError) -> None:
    print("EXPIRED_OPTION_API_TEST=FAIL")
    print(f"  step: {step}")
    print(f"  endpoint type: {endpoint}")
    print(f"  HTTP status: {exc.status}")
    print(f"  error code: {exc.code}")
    print(f"  error message: {exc.message}")
    sys.exit(1)


def pick_contract(contracts: list[dict], option_type: str) -> dict | None:
    rows = [c for c in contracts if str(c.get("instrument_type", "")).upper() == option_type
            and c.get("instrument_key") and c.get("strike_price") is not None]
    if not rows:
        return None
    strikes = sorted(float(c["strike_price"]) for c in rows)
    middle = strikes[len(strikes) // 2]
    return min(rows, key=lambda c: abs(float(c["strike_price"]) - middle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upstox expired option API single-contract diagnostic")
    parser.add_argument("--expiry", type=date.fromisoformat, default=None,
                        help="expired expiry to use (default: latest returned expiry in Sep/Oct 2026)")
    parser.add_argument("--day", type=date.fromisoformat, default=None,
                        help="trading day for candles (default: the expiry date)")
    parser.add_argument("--option-type", choices=["CE", "PE"], default="CE")
    args = parser.parse_args()

    assert_paper_trading_only()
    source = token_source()
    print(f"PAPER diagnostic | read-only GET requests | token variable: {source or 'NONE'} (value not shown)")
    if not source:
        print("EXPIRED_OPTION_API_TEST=FAIL\n  error code: CREDENTIALS_MISSING")
        sys.exit(1)
    client = UpstoxHistoricalClient(max_retries=1)

    try:
        expiries = client.expired_expiries(UNDERLYING)
    except HistoricalDataError as exc:
        fail("1 - expiries", "GET /v2/expired-instruments/expiries", exc)
    window = [e for e in expiries if date(2026, 9, 1) <= e <= date(2026, 10, 31)]
    print(f"Step 1: {len(expiries)} expired expiries returned; Sep/Oct 2026: {[e.isoformat() for e in window]}")
    expiry = args.expiry or (window[-1] if window else None)
    if expiry is None:
        print("EXPIRED_OPTION_API_TEST=FAIL\n  reason: no expired NIFTY expiry in Sep/Oct 2026 returned by Upstox")
        sys.exit(1)
    if expiry not in expiries:
        print(f"EXPIRED_OPTION_API_TEST=FAIL\n  reason: {expiry} is not in the expiry list returned by Upstox")
        sys.exit(1)

    try:
        contracts = client.expired_option_contracts(UNDERLYING, expiry)
    except HistoricalDataError as exc:
        fail("2 - option contracts", "GET /v2/expired-instruments/option/contract", exc)
    contract = pick_contract(contracts, args.option_type)
    print(f"Step 2: {len(contracts)} expired option contracts for {expiry}")
    if contract is None:
        print(f"EXPIRED_OPTION_API_TEST=FAIL\n  reason: no {args.option_type} contract with an instrument_key returned")
        sys.exit(1)
    key = str(contract["instrument_key"])
    print(f"  expiry: {contract.get('expiry')}")
    print(f"  strike: {contract.get('strike_price')}")
    print(f"  type: {contract.get('instrument_type')}")
    print(f"  trading_symbol: {contract.get('trading_symbol')}")
    print(f"  expired_instrument_key: {key}")

    day = args.day or expiry
    path = (f"/v2/expired-instruments/historical-candle/{quote(key, safe='')}/5minute/"
            f"{day.isoformat()}/{day.isoformat()}")
    try:
        data = client._get(path)
    except HistoricalDataError as exc:
        fail("3 - historical candles", "GET /v2/expired-instruments/historical-candle/{key}/5minute/{to}/{from}", exc)

    candles = data.get("candles") if isinstance(data, dict) else None
    print(f"Step 3: HTTP 200, status=success (enforced by client), day {day}")
    if not isinstance(candles, list):
        print("EXPIRED_OPTION_API_TEST=FAIL\n  reason: response has no data.candles list")
        sys.exit(1)
    bad = [c for c in candles if not isinstance(c, list) or len(c) != len(CANDLE_FIELDS)]
    rows = sorted(candles, key=lambda c: str(c[0])) if not bad else candles
    print(f"  candles: {len(candles)}")
    if not candles:
        print("EXPIRED_OPTION_API_TEST=FAIL\n  reason: data.candles is empty for this contract/day")
        sys.exit(1)
    if bad:
        print(f"EXPIRED_OPTION_API_TEST=FAIL\n  reason: {len(bad)} candle(s) do not have the 7 fields {CANDLE_FIELDS}")
        sys.exit(1)
    first, last = rows[0], rows[-1]
    print(f"  first timestamp: {first[0]}")
    print(f"  last timestamp: {last[0]}")
    print(f"  first OHLC: O={first[1]} H={first[2]} L={first[3]} C={first[4]} V={first[5]} OI={first[6]}")
    print(f"  last OHLC:  O={last[1]} H={last[2]} L={last[3]} C={last[4]} V={last[5]} OI={last[6]}")
    print("EXPIRED_OPTION_API_TEST=PASS")


if __name__ == "__main__":
    main()
