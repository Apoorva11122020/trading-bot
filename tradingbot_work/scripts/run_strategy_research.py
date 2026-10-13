"""Run strategy research backtests and write reports/strategy_research/ outputs.

Downloads missing underlying and signal-required option candles from Upstox
V3 before running the backtest. No live orders are placed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=False)

from src.config.research_loader import load_research_config
from src.historical_data.models import HistoricalDataError
from src.historical_data.service import UNDERLYING_MISSING_HINT, HistoricalDataService
from src.safety.live_gate import live_trading_enabled, trading_mode

MISSING_HINT = UNDERLYING_MISSING_HINT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run index-option strategy research backtests",
        epilog=MISSING_HINT,
    )
    parser.add_argument("--underlying", action="append", required=True, help="NIFTY or BANKNIFTY (repeatable)")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--strategy", default="all", help="Strategy id or 'all'")
    parser.add_argument("--timeframe", default="5", help="Minutes (must match stored data)")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download missing data (fail if underlying candles are absent)",
    )
    return parser.parse_args()


def print_coverage_report(report: dict) -> None:
    print("\nOption coverage report:")
    print(f"  Total signals                 : {report['total_signals']}")
    print(f"  Unique required contracts     : {report['unique_required_contracts']}")
    print(f"  Contracts already stored      : {report['contracts_already_stored']}")
    print(f"  Contracts downloaded        : {report['contracts_downloaded']}")
    print(f"  Contracts unavailable         : {report['contracts_unavailable']}")
    print(f"  Signals with premium coverage : {report['signals_with_complete_coverage']}")
    print(f"  Signals missing coverage      : {report['signals_missing_coverage']}")
    print(f"  Coverage                      : {report['coverage_percent']}%")
    if report.get("required"):
        print("\n  Required contracts:")
        for row in report["required"]:
            print(
                f"    {row['option_type']} {row['strike']} exp {row['expiry']} "
                f"key={row['instrument_key']} signals={row['signal_count']}"
            )
    if report.get("unavailable"):
        print("\n  Unavailable / unresolved:")
        for row in report["unavailable"][:10]:
            print(f"    {row}")
        if len(report["unavailable"]) > 10:
            print(f"    ... and {len(report['unavailable']) - 10} more")


def strategy_metrics(result: dict, strategy_id: str) -> dict | None:
    for row in result.get("results", []):
        if row.get("strategy_id") == strategy_id and row.get("split") == "FULL":
            metrics = row.get("metrics", {})
            resolved = int(metrics.get("resolved_trades", 0))
            total = int(metrics.get("total_signals", 0))
            skipped = int(metrics.get("skipped", 0))
            invalid = int(metrics.get("data_invalid", 0))
            return {
                "total_signals": total,
                "resolved_trades": resolved,
                "unresolved_signals": max(0, total - resolved),
                "skipped": skipped,
                "data_invalid": invalid,
                "coverage_percent": round(100.0 * resolved / total, 2) if total else 0.0,
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "average_r": metrics.get("average_r", 0),
            }
    return None


def print_strategy_result(strategy_id: str, stats: dict | None) -> None:
    print(f"\n{strategy_id} backtest (FULL):")
    if not stats:
        print("  No FULL-split result found.")
        return
    pf = stats["profit_factor"]
    pf_text = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  Total signals      : {stats['total_signals']}")
    print(f"  Resolved trades    : {stats['resolved_trades']}")
    print(f"  Unresolved signals : {stats['unresolved_signals']}")
    print(f"  Trade resolution % : {stats['coverage_percent']}%")
    print(f"  Win rate           : {stats['win_rate'] * 100:.1f}%")
    print(f"  Profit factor      : {pf_text}")
    print(f"  Max drawdown       : {stats['max_drawdown']:,.2f}")
    print(f"  Avg R              : {stats['average_r']:.2f}")


def main() -> None:
    args = parse_args()
    if live_trading_enabled() or trading_mode() != "PAPER":
        print("Refusing to run unless TRADING_MODE=PAPER and LIVE_TRADING_ENABLED=false")
        sys.exit(1)

    research = load_research_config(PROJECT_ROOT / "config" / "strategy_research.yaml")
    if int(args.timeframe) != research.interval_minutes:
        print(f"Warning: requested timeframe {args.timeframe}m differs from config {research.interval_minutes}m")

    strategies = [sid for sid, cfg in research.strategies.items() if cfg.get("enabled", True)]
    if args.strategy.lower() != "all":
        strategies = [args.strategy.upper()]

    start = date.fromisoformat(args.from_date)
    end = date.fromisoformat(args.to_date)
    service = HistoricalDataService(store_root=PROJECT_ROOT / "data" / "historical")

    for underlying in args.underlying:
        name = underlying.upper()
        if args.skip_download:
            if not service.has_underlying_data(name, start, end):
                print(f"FAILED: No stored underlying candles for {name}.")
                print(MISSING_HINT)
                sys.exit(1)
            verification = service.verify_underlying_data(name, start, end)
        else:
            print(f"Ensuring underlying candles for {name} ({start} .. {end})...")
            if not service.has_underlying_data(name, start, end):
                verification = service.download_underlying_sync(name, start, end)
            else:
                verification = service.verify_underlying_data(name, start, end)

        print(f"Underlying instrument : {verification['instrument_key']}")
        print(f"Underlying candles    : {verification['candle_count']}")
        print(f"Stored file path(s)   : {', '.join(verification['files']) or 'n/a'}")

        for strategy_id in strategies:
            print(f"\nEnsuring option coverage for {name} / {strategy_id}...")
            coverage = service.ensure_strategy_option_coverage(name, strategy_id, start, end)
            report_path = PROJECT_ROOT / "reports" / "strategy_research" / "coverage"
            report_path.mkdir(parents=True, exist_ok=True)
            out_file = report_path / f"{name}_{strategy_id}_{start}_{end}.json"
            out_file.write_text(json.dumps(coverage.as_dict(), indent=2), encoding="utf-8")
            print_coverage_report(coverage.as_dict())
            print(f"  Coverage report saved: {out_file}")

    print("\nRunning strategy research backtest...")
    try:
        result = service.run_backtest_sync(args.underlying, start, end, strategies=strategies)
    except HistoricalDataError as exc:
        print(f"FAILED: {exc.code}: {exc.message}")
        if exc.code == "UNDERLYING_DATA_MISSING":
            print(MISSING_HINT)
        sys.exit(1)

    print(f"\nBacktest run ID: {result['run_id']}")
    print(f"Status         : {result['status']}")
    print(f"Reports        : reports/strategy_research/{result['run_id']}/")

    for strategy_id in strategies:
        print_strategy_result(strategy_id, strategy_metrics(result, strategy_id))


if __name__ == "__main__":
    try:
        main()
    except HistoricalDataError as exc:
        print(f"FAILED: {exc.code}: {exc.message}")
        if exc.code in {"UNDERLYING_DATA_MISSING", "AUTH_FAILED", "CREDENTIALS_MISSING"}:
            print(MISSING_HINT)
        sys.exit(1)
