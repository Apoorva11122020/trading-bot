#!/usr/bin/env python3
"""Generate NIFTY S2 robustness report from an existing full research run."""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.robustness import (
    analyze_robustness,
    coverage_for,
    load_final_comparison,
    print_conclusion,
    row_for,
    write_robustness_reports,
)
from src.backtest.strategy_comparator import compare_strategies
from src.historical_data.service import HistoricalDataService

DEFAULT_COMPARISON = ROOT / "reports" / "strategy_research" / "final_comparison.json"
DEFAULT_OUT = ROOT / "reports" / "strategy_research"


def run_robustness(
    *,
    source_run_id: str,
    comparison_path: Path = DEFAULT_COMPARISON,
    out_dir: Path = DEFAULT_OUT,
    underlying: str = "NIFTY",
    strategy_id: str = "S2",
) -> None:
    final = load_final_comparison(comparison_path)
    if final.get("run_id") != source_run_id:
        raise SystemExit(
            f"Comparison file run_id {final.get('run_id')} does not match requested {source_run_id}"
        )

    row = row_for(final, underlying, strategy_id)
    coverage = coverage_for(final, underlying, strategy_id)
    from_date = date.fromisoformat(row["from_date"])
    to_date = date.fromisoformat(row["to_date"])

    service = HistoricalDataService()
    if not service.has_underlying_data(underlying, from_date, to_date):
        raise SystemExit(f"Missing underlying data for {underlying} {from_date}..{to_date}")

    candles = service.read_underlying(underlying, from_date, to_date)
    candles["symbol"] = underlying.upper()
    contracts = service._contracts_from_store(underlying)
    premium = service._premium_history(underlying, from_date, to_date)
    trading_days = service.count_trading_days(underlying, from_date, to_date)

    analysis_run_id = uuid.uuid4().hex[:12]
    results, _ = compare_strategies(
        research_config=service.research_config,
        strategy_config=service.strategy_config,
        underlying=underlying,
        underlying_candles=candles,
        premium_history=premium,
        contracts=contracts,
        strategy_ids=[strategy_id],
        run_id=analysis_run_id,
        master_resolver=service._research_resolver,
        splits={"FULL": (from_date, to_date)},
    )
    result = next(r for r in results if r.split == "FULL").as_dict()

    report = analyze_robustness(
        run_id=analysis_run_id,
        source_run_id=source_run_id,
        underlying=underlying,
        strategy_id=strategy_id,
        from_date=from_date,
        to_date=to_date,
        trading_days=trading_days,
        result=result,
        coverage=coverage,
        initial_equity=float(service.research_config.risk.initial_equity),
        daily_loss_limit_inr=float(service.research_config.risk.max_daily_loss_inr),
    )
    paths = write_robustness_reports(report, out_dir)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['csv']}")
    print(f"Wrote {paths['md']}")
    print_conclusion(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 robustness analysis")
    parser.add_argument("--run-id", default="b9202ebc90b7", help="Source full research run id")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_robustness(
        source_run_id=args.run_id,
        comparison_path=args.comparison,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
