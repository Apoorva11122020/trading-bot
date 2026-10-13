#!/usr/bin/env python3
"""Extend NIFTY historical 5m candles and S2-required option premiums (research only, PAPER mode).

Read-only Upstox historical requests; no broker order is ever placed.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from src.historical_data.extension import (
    IST,
    build_report,
    download_underlying,
    extend_options,
    resolve_range,
    safe_token,
    safe_token_source,
    write_extension_reports,
)
from src.historical_data.models import HistoricalDataError
from src.historical_data.service import HistoricalDataService, _refuse_live, build_catalog
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_research"


def run_extension(*, start: date, end: date, underlying: str = "NIFTY", strategy_id: str = "S2",
                  out_dir: Path = DEFAULT_OUT, service: HistoricalDataService | None = None,
                  client=None, now: datetime | None = None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    now = now or datetime.now(IST)
    source = safe_token_source()
    if client is None:
        if source is None:
            raise SystemExit("No market-data/historical token configured "
                             "(UPSTOX_HISTORICAL_ACCESS_TOKEN / UPSTOX_MARKET_DATA_ACCESS_TOKEN / ACCESS_TOKEN).")
        client = UpstoxHistoricalClient(token_provider=safe_token)
    service = service or HistoricalDataService()
    service.client = client
    service.catalog = build_catalog(client)

    start, effective_end, notes = resolve_range(start, end, now)
    interval = int(service.research_config.interval_minutes)
    key = service.underlying_key(underlying)
    limitations: list[dict] = []
    try:
        u = download_underlying(client, service.store, underlying=underlying.upper(), instrument_key=key,
                                start=start, end=effective_end, requested_end=end, interval=interval,
                                limitations=limitations)
        o = extend_options(client, service.store, underlying=underlying.upper(), underlying_key=key,
                           strategy_id=strategy_id, research_config=service.research_config,
                           strategy_config=service.strategy_config, resolver_factory=service.resolver_factory,
                           start=start, end=effective_end, interval=interval, today=now.date(),
                           limitations=limitations)
    except HistoricalDataError as exc:
        raise SystemExit(f"STOPPED: {exc.code} - {exc.message}") from None

    report = build_report(run_id=uuid.uuid4().hex[:12], underlying=underlying, strategy_id=strategy_id,
                          token_source=source, requested=(start, end),
                          effective_end=effective_end, underlying_result=u, option_result=o,
                          limitations=limitations, notes=notes, generated_at=now)
    for path in write_extension_reports(report, out_dir).values():
        print(f"Wrote {path}")
    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    s = report["summary"]
    if not report["data_available"]:
        print("\nDATA EXTENSION STOPPED - no new completed market data available")
    else:
        print("\nDATA EXTENSION COMPLETE")
    coverage = "n/a (no signals)" if s["coverage_percent"] is None else f"{s['coverage_percent']:.2f}%"
    print(f"  Latest NIFTY candle      : {s['latest_candle']}")
    print(f"  New trading days         : {s['new_trading_days']}")
    print(f"  New candles              : {s['candles_written']}")
    print(f"  Option contracts         : {s['option_contracts_downloaded']} downloaded, "
          f"{s['option_contracts_already_stored']} already stored, {s['option_contracts_unavailable']} unavailable")
    print(f"  Option coverage          : {coverage}")
    print("  Access limitations       :")
    for item in report["limitations"] or [{"code": "NONE", "message": "-"}]:
        print(f"    - {item['code']}: {item['message']}")
    for note in report["notes"]:
        print(f"    - NOTE: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend NIFTY historical data (research only)")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2026, 10, 2))
    parser.add_argument("--end-date", type=date.fromisoformat, default=datetime.now(IST).date())
    parser.add_argument("--underlying", default="NIFTY", choices=["NIFTY"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_extension(start=args.start_date, end=args.end_date, underlying=args.underlying, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
