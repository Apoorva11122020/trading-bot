"""Small, read-only validation of Upstox historical option candles.

Fetches 5-minute candles for ONE real, currently listed NIFTY option contract
(resolved from the official instrument master: nearest expiry, ATM from the
real underlying close), validates them, saves them to Parquet and reads them
back. Also reports whether expired-contract history is accessible.

Prints SUCCESS only when Upstox actually returned candles. Never prints the
access token. Places no orders.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.config.loader import load_strategy_config  # noqa: E402
from src.historical_data.contracts import ContractCatalog, load_instrument_master_rows  # noqa: E402
from src.historical_data.models import HistoricalDataError, candles_to_frame  # noqa: E402
from src.historical_data.storage import ParquetCandleStore  # noqa: E402
from src.historical_data.upstox_client import UpstoxHistoricalClient, token_source  # noqa: E402
from src.historical_data.validation import validate_candles  # noqa: E402
from src.options.instrument_resolver import build_resolver  # noqa: E402
from src.options.selection import atm_strike, select_expiry  # noqa: E402
from src.safety.live_gate import live_trading_enabled, trading_mode  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")
UNDERLYING = os.getenv("HISTORICAL_TEST_UNDERLYING", "NIFTY").upper()
LOOKBACK_DAYS = 7


def stop(reason: str) -> None:
    print(f"\nRESULT: FAILED - {reason}")
    sys.exit(1)


def candle_text(row) -> str:
    return (f"{row['timestamp'].isoformat()} O={row['open']} H={row['high']} L={row['low']} "
            f"C={row['close']} V={row['volume']:.0f} OI={row['open_interest']}")


def main() -> None:
    print("=== Upstox historical option data validation (read-only) ===")
    if live_trading_enabled() or trading_mode() != "PAPER":
        stop("refusing to run unless TRADING_MODE=PAPER and LIVE_TRADING_ENABLED=false")
    print(f"Trading mode           : {trading_mode()} (live trading disabled)")
    source = token_source()
    if not source:
        stop("no Upstox access token configured")
    print(f"Credentials            : present (from {source}; value not shown)")

    config = load_strategy_config(PROJECT_ROOT / "config" / "strategy.yaml")
    resolver = build_resolver(config.options)
    client = UpstoxHistoricalClient(max_retries=2)
    catalog = ContractCatalog(client, lambda: load_instrument_master_rows(resolver.source))

    today = datetime.now(IST).date()
    start, end = today - timedelta(days=LOOKBACK_DAYS), today
    try:
        underlying_key = catalog.underlying_key(UNDERLYING)
        spot_candles = client.candles(underlying_key, start, end, interval=5)
    except HistoricalDataError as exc:
        stop(f"underlying candles unavailable ({exc.code}: {exc.message})")
    if not spot_candles:
        stop(f"Upstox returned no {UNDERLYING} underlying candles for {start}..{end}")
    spot_frame = candles_to_frame(spot_candles)
    spot = float(spot_frame["close"].iloc[-1])
    print(f"Underlying             : {UNDERLYING} ({underlying_key}) last close {spot} at "
          f"{spot_frame['timestamp'].iloc[-1].isoformat()}")

    try:
        expiry = select_expiry(resolver.expiries(UNDERLYING), "nearest")
        strike = atm_strike(resolver.strikes(UNDERLYING, expiry, "CE"), spot)
        contract = resolver.resolve(UNDERLYING, expiry, strike, "CE")
    except Exception as exc:
        stop(f"contract resolution failed ({exc})")

    try:
        candles = client.candles(contract.instrument_key, start, end, interval=5)
    except HistoricalDataError as exc:
        stop(f"option candles request failed ({exc.code}: {exc.message})")
    frame = candles_to_frame(candles, underlying=UNDERLYING, expiry=contract.expiry.isoformat(),
                             strike=contract.strike, option_type=contract.option_type.value)

    print(f"Instrument             : {contract.trading_symbol} ({contract.instrument_key})")
    print(f"Expiry / Strike / Type : {contract.expiry.isoformat()} / {contract.strike:g} / {contract.option_type.value}")
    print(f"From / To              : {start.isoformat()} / {end.isoformat()}")
    print(f"Candle count           : {len(frame)}")
    if frame.empty:
        stop("Upstox returned zero candles for this contract (it may have been listed after the window "
             "or the market was closed)")
    print(f"First candle           : {candle_text(frame.iloc[0])}")
    print(f"Last candle            : {candle_text(frame.iloc[-1])}")
    sample = frame.iloc[len(frame) // 2]
    print(f"Sample OHLCV/OI        : {candle_text(sample)}")

    clean, report = validate_candles(frame, interval=5, expiry=contract.expiry, option_type="CE")
    print(f"Validation             : valid={report.valid_rows} rejected={report.rejected_rows} "
          f"duplicates={report.duplicates_removed} issues={report.issues or 'none'}")
    print(f"Missing candles        : {report.missing_candles} of {report.expected_candles} expected "
          f"over {report.trading_days} trading day(s) (not filled)")

    store = ParquetCandleStore(PROJECT_ROOT / "data" / "historical")
    store.write("options", UNDERLYING, contract.instrument_key, clean, expiry=contract.expiry)
    back = store.read("options", UNDERLYING, contract.instrument_key, expiry=contract.expiry, start=start, end=end)
    print(f"Saved + read back      : {len(back)} rows from {store.root / 'options' / UNDERLYING}")
    if len(back) < len(clean):
        stop("Parquet read-back returned fewer rows than were saved")
    ohlc_ok = back[["open", "high", "low", "close", "volume"]].notna().all().all()
    oi_ok = back["open_interest"].notna().any()
    print(f"OHLCV present          : {bool(ohlc_ok)}; open interest present: {bool(oi_ok)}")

    print("\nExpired-contract history (needed for strategy backtests before today):")
    try:
        expiries = client.expired_expiries(underlying_key)
        print(f"  AVAILABLE - {len(expiries)} expired expiries listed")
    except HistoricalDataError as exc:
        print(f"  UNAVAILABLE - {exc.code}: {exc.message}")
        print("  Only currently listed contracts can be downloaded with this token. Upstox serves expired")
        print("  option contracts only on the Upstox Plus plan with a live-account access token")
        print("  (set UPSTOX_HISTORICAL_ACCESS_TOKEN; it is used for read-only data requests only).")

    print(f"\nRESULT: SUCCESS - real Upstox historical candles returned ({len(frame)} candles, "
          f"{client.requests_made} API requests)")


if __name__ == "__main__":
    main()
