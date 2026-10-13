#!/usr/bin/env python3
"""Walk-forward validation of original vs regime-filtered NIFTY S2 (historical research, PAPER only)."""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.options_engine import REGIME_FILTERED, OptionsResearchEngine
from src.backtest.regime_filter import AVOID_REGIMES, PointInTimeRegimeClassifier, RegimeSignalFilter
from src.backtest.regime_filtered import filtered_coverage, summarize
from src.backtest.robustness import coverage_for, load_final_comparison
from src.backtest.walk_forward import (
    Period,
    coverage_for_period,
    slice_candles,
    slice_premium,
    validate_split,
    walk_forward_verdict,
    write_walk_forward_reports,
)
from src.historical_data.service import HistoricalDataService
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_COMPARISON = ROOT / "reports" / "strategy_research" / "final_comparison.json"
DEFAULT_OUT = ROOT / "reports" / "strategy_research"

CAVEATS = [
    "The avoided regimes were chosen from a regime study over 2026-09-01..2026-10-05, which overlaps the OOS "
    "window, so the filter's regime selection is not fully independent of OOS data.",
    "Each period starts from the configured initial equity; per-trade sizing does not carry equity across periods.",
]


def run_period(service, candles, premium, coverage, period: Period, history_start: date,
               underlying: str, strategy_id: str, run_id: str) -> dict:
    """Run both variants on *period* using only data up to period.end (prior history allowed)."""
    period_candles = slice_candles(candles, history_start, period.end)
    period_premium = slice_premium(premium, history_start, period.end)
    research, strategy = service.research_config, service.strategy_config
    trading_days = len(set(slice_candles(candles, period.start, period.end)["timestamp"].dt.date))
    contracts = service._contracts_from_store(underlying)

    def engine(signal_filter=None) -> OptionsResearchEngine:
        return OptionsResearchEngine(research, strategy, contracts, period_premium, run_id=run_id,
                                     split=period.name.upper(), master_resolver=service._research_resolver,
                                     signal_filter=signal_filter)

    original = engine().run(period_candles, underlying, strategy_id, period.start, period.end).as_dict()
    regime_filter = RegimeSignalFilter(PointInTimeRegimeClassifier(period_candles))
    filtered = engine(regime_filter).run(period_candles, underlying, strategy_id, period.start, period.end).as_dict()

    signal_times = [s["timestamp"] for s in original["signals"]]
    period_coverage = coverage_for_period(coverage, signal_times, period)
    blocked = {s["time"] for s in filtered["skipped"] if s.get("reason") == REGIME_FILTERED}
    common = dict(trading_days=trading_days, from_date=period.start, to_date=period.end,
                  initial_equity=float(research.risk.initial_equity),
                  daily_loss_limit_inr=float(research.risk.max_daily_loss_inr))
    out = {}
    for name, result, cov in (("original", original, period_coverage),
                              ("filtered", filtered, filtered_coverage(period_coverage, blocked))):
        summary = summarize(result, coverage=cov, label=name, **common)
        summary["unresolved_signals"] = max(0, summary["total_signals"] - summary["resolved_trades"])
        summary["trade_signal_times"] = [t["signal_time"] for t in result["trades"]]
        out[name] = summary
    return out


def run_walk_forward(
    *,
    development: Period,
    out_of_sample: Period,
    comparison_path: Path = DEFAULT_COMPARISON,
    out_dir: Path = DEFAULT_OUT,
    underlying: str = "NIFTY",
    strategy_id: str = "S2",
) -> dict:
    assert_paper_trading_only()
    validate_split(development, out_of_sample)
    coverage = coverage_for(load_final_comparison(comparison_path), underlying, strategy_id)

    service = HistoricalDataService()
    candles = service.read_underlying(underlying, development.start, out_of_sample.end)
    if candles.empty:
        raise SystemExit(f"Missing underlying data for {underlying} {development.start}..{out_of_sample.end}")
    candles["symbol"] = underlying.upper()
    premium = service._premium_history(underlying, development.start, out_of_sample.end)

    run_id = uuid.uuid4().hex[:12]
    dev = run_period(service, candles, premium, coverage, development, development.start,
                     underlying, strategy_id, run_id)
    oos = run_period(service, candles, premium, coverage, out_of_sample, development.start,
                     underlying, strategy_id, run_id)
    results = {name: {"development": dev[name], "oos": oos[name]} for name in ("original", "filtered")}
    verdict, reasons = walk_forward_verdict(results)
    report = {
        "run_id": run_id,
        "underlying": underlying.upper(),
        "strategy_id": strategy_id,
        "development": {"start": development.start.isoformat(), "end": development.end.isoformat()},
        "out_of_sample": {"start": out_of_sample.start.isoformat(), "end": out_of_sample.end.isoformat()},
        "data_last_candle": str(candles["timestamp"].max()),
        "avoid_regimes": list(AVOID_REGIMES),
        "max_daily_loss_inr": float(service.research_config.risk.max_daily_loss_inr),
        "results": results,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "caveats": CAVEATS,
    }
    paths = write_walk_forward_reports(report, out_dir)
    for path in paths.values():
        print(f"Wrote {path}")
    print_comparison(report)
    return report


def print_comparison(report: dict) -> None:
    print("\n=== NIFTY S2 walk-forward (historical simulation, PAPER only) ===")
    print(f"{'Period':<14}{'Strategy':<10}{'Days':>5}{'Trades':>8}{'Win%':>7}{'PF':>7}{'Net P&L':>13}{'MaxDD':>11}{'AvgR':>7}")
    for period in ("development", "oos"):
        for strategy in ("original", "filtered"):
            s = report["results"][strategy][period]
            pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
            print(f"{period:<14}{strategy:<10}{s['trading_days']:>5}{s['resolved_trades']:>8}"
                  f"{s['win_rate'] * 100:>6.1f}%{pf:>7}{s['net_pnl']:>13,.0f}{s['max_drawdown']:>11,.0f}"
                  f"{s['average_r']:>7.2f}")
    print(f"\nVERDICT: {report['verdict']}")
    for reason in report["verdict_reasons"]:
        print(f"  - {reason}")
    for caveat in report["caveats"]:
        print(f"  - Caveat: {caveat}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 walk-forward validation")
    parser.add_argument("--dev-start", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--dev-end", type=date.fromisoformat, default=date(2026, 9, 23))
    parser.add_argument("--oos-start", type=date.fromisoformat, default=date(2026, 9, 24))
    parser.add_argument("--oos-end", type=date.fromisoformat, default=date(2026, 10, 5))
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_walk_forward(
        development=Period("development", args.dev_start, args.dev_end),
        out_of_sample=Period("oos", args.oos_start, args.oos_end),
        comparison_path=args.comparison,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
