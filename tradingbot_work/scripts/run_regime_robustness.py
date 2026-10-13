#!/usr/bin/env python3
"""NIFTY S2 market-regime robustness study (historical data only, no orders)."""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.regime_robustness import analyze_regime_robustness, write_regime_robustness_reports
from src.backtest.robustness import coverage_for, load_final_comparison, row_for
from src.backtest.strategy_comparator import compare_strategies
from src.historical_data.service import HistoricalDataService
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_COMPARISON = ROOT / "reports" / "strategy_research" / "final_comparison.json"
DEFAULT_OUT = ROOT / "reports" / "strategy_research"


def run_regime_study(
    *,
    source_run_id: str,
    comparison_path: Path = DEFAULT_COMPARISON,
    out_dir: Path = DEFAULT_OUT,
    underlying: str = "NIFTY",
    strategy_id: str = "S2",
) -> None:
    assert_paper_trading_only()
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
    s2_cfg = dict(service.research_config.strategies[strategy_id])

    report = analyze_regime_robustness(
        run_id=analysis_run_id,
        source_run_id=source_run_id,
        underlying=underlying,
        strategy_id=strategy_id,
        from_date=from_date,
        to_date=to_date,
        trading_days=trading_days,
        underlying_candles=candles,
        result=result,
        coverage=coverage,
        s2_cfg=s2_cfg,
        initial_equity=float(service.research_config.risk.initial_equity),
    )
    paths = write_regime_robustness_reports(report, out_dir)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['csv']}")
    print(f"Wrote {paths['md']}")
    print(f"Regimes analyzed: {len(report.regimes)}")
    for row in report.regimes:
        if row.trading_days:
            print(
                f"  {row.regime_label}: days={row.trading_days} trades={row.resolved_trades} "
                f"net_pnl={row.net_pnl:,.0f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 market-regime robustness")
    parser.add_argument("--run-id", default="b9202ebc90b7", help="Source full research run id")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--underlying", default="NIFTY")
    parser.add_argument("--strategy", default="S2")
    args = parser.parse_args()
    run_regime_study(
        source_run_id=args.run_id,
        comparison_path=args.comparison,
        out_dir=args.out_dir,
        underlying=args.underlying,
        strategy_id=args.strategy,
    )


if __name__ == "__main__":
    main()
