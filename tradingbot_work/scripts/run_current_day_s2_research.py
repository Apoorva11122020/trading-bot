#!/usr/bin/env python3
"""Current-day NIFTY S2 signal research from Upstox V3 intraday candles (read-only, PAPER only, no orders)."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.current_day_s2 import (
    DISCLAIMER,
    IST,
    LevelParams,
    point_in_time_signals,
    premium_frame,
    resolve_signals,
    summarize,
    validate_intraday,
    validation_dict,
    write_reports,
)
from src.historical_data.models import HistoricalDataError
from src.historical_data.option_coverage import resolve_signal_contract
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.historical_data.upstox_client import token_source
from src.safety.live_gate import assert_paper_trading_only
from src.session.session_manager import SessionManager

DEFAULT_OUT = ROOT / "reports" / "strategy_research"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"


def run(*, day: date, underlying: str = "NIFTY", strategy_id: str = "S2", out_dir: Path = DEFAULT_OUT,
        service: HistoricalDataService | None = None, client=None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    service = service or HistoricalDataService()
    client = client or service.client
    params = LevelParams.from_configs(service.research_config, service.strategy_config)

    try:
        raw = client.intraday_candles(INSTRUMENT_KEY, interval=params.interval)
    except HistoricalDataError as exc:
        raise SystemExit(f"API_REQUEST_ERROR: {exc.code} (HTTP {exc.status})") from None
    candles, validation = validate_intraday(raw, day, params.interval)
    if candles.empty:
        raise SystemExit(f"No intraday candles for {day} (received {validation.received}, "
                         f"{validation.wrong_day_rows} from another day). Nothing generated.")

    session = SessionManager(service.strategy_config.session)
    signals, lookahead_ok = point_in_time_signals(candles, underlying, service.research_config, session,
                                                  params.stop_new_entries, strategy_id)

    premium_cache: dict[str, object] = {}

    def premium_for(contract):
        key = contract.instrument_key
        if key not in premium_cache:
            try:
                today = client.intraday_candles(key, interval=params.interval)
            except HistoricalDataError as exc:
                premium_cache[key] = f"PREMIUM_API_ERROR:{exc.code}"
                return premium_cache[key]
            stored = service.store.read("options", underlying.upper(), key, expiry=contract.expiry)
            premium_cache[key] = premium_frame(stored, today)
        return premium_cache[key]

    def resolve(signal):
        return resolve_signal_contract(signal, service.resolver_factory(signal.timestamp.date()),
                                       service.research_config)

    rows = resolve_signals(signals, resolve, premium_for, params)
    summary = summarize(validation, rows, lookahead_ok)
    report = {
        "date": day.isoformat(),
        "underlying": underlying.upper(),
        "strategy_id": strategy_id,
        "instrument_key": INSTRUMENT_KEY,
        "fetched_at": datetime.now(IST).isoformat(timespec="seconds"),
        "token_source": token_source(),
        "mode": "PAPER research only - no broker orders, nothing written to Parquet",
        "disclaimer": DISCLAIMER,
        "summary": summary,
        "validation": validation_dict(validation),
        "signals": rows,
        "notes": [
            DISCLAIMER,
            "S2 fires on every 5-minute close beyond the opening range after 09:30. All such signals are listed; "
            "the research engine would act on only some of them (one position at a time, daily loss budget).",
            "Signals were generated bar by bar using only candles up to each bar; the result is compared with a "
            "full-day run as a look-ahead check.",
            "Entry/SL/target follow OptionsResearchEngine (next-bar open entry, premium-ATR stop, target R). "
            "Premium ATR uses real stored option history plus today's intraday option candles, in memory only.",
            "A signal is unresolved when the exact option candle at the signal bar or entry bar is missing.",
            "Flat candles (open = high = low = close) are kept exactly as received.",
        ],
    }
    paths = write_reports(report, out_dir, f"{underlying.upper()}_{strategy_id}_current_day_{day.isoformat()}")
    for path in paths.values():
        print(f"Wrote {path}")
    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    s, v = report["summary"], report["validation"]
    cov = "n/a" if s["option_coverage_percent"] is None else f"{s['option_coverage_percent']}%"
    print("\nCURRENT DAY S2 RESEARCH COMPLETE")
    print(f"  Date / token source : {report['date']} / {report['token_source']}")
    print(f"  Candles             : {s['candle_count']} (expected {v['expected_candles']}), "
          f"{s['first_candle']} -> {s['last_candle']}")
    print(f"  Duplicates / missing: {v['duplicates']} / {len(v['missing_candles'])}; "
          f"flat candles kept: {len(v['flat_candles'])}")
    print(f"  Signals             : {s['signal_count']} (BUY {s['buy_signals']}, SELL {s['sell_signals']})")
    print(f"  Resolved / unresolved: {s['resolved_option_signals']} / {s['unresolved_signals']} (coverage {cov})")
    if s["unresolved_reasons"]:
        print("  Unresolved reasons  : " + ", ".join(f"{k}={n}" for k, n in s["unresolved_reasons"].items()))
    print(f"  Contracts           : {', '.join(s['unique_contracts']) or '-'}")
    print(f"  Look-ahead check    : {'PASS' if s['point_in_time_matches_full_run'] else 'MISMATCH'}")
    print(f"  Note                : {DISCLAIMER}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Current-day NIFTY S2 research (read-only)")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(day=args.date, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
