#!/usr/bin/env python3
"""Compare original NIFTY S2 with the regime-filtered S2 (historical research, PAPER only)."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.options_engine import REGIME_FILTERED, OptionsResearchEngine
from src.backtest.regime_filter import (
    ALLOW_REGIMES,
    AVOID_REGIMES,
    PointInTimeRegimeClassifier,
    RegimeSignalFilter,
    validate_hypothesis,
)
from src.backtest.regime_filtered import build_comparison, filtered_coverage, summarize, trade_diff, write_reports
from src.backtest.robustness import coverage_for, load_final_comparison
from src.historical_data.service import HistoricalDataService
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_COMPARISON = ROOT / "reports" / "strategy_research" / "final_comparison.json"
DEFAULT_REGIME_STUDY = ROOT / "reports" / "strategy_research" / "NIFTY_S2_regime_robustness.json"
DEFAULT_OUT = ROOT / "reports" / "strategy_research"


def _filter_reason(row: dict) -> str:
    pf = row.get("profit_factor", 0.0)
    pf_text = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"Regime study (in-sample): {row.get('resolved_trades', 0)} trades, "
            f"net P&L {row.get('net_pnl', 0.0):,.0f}, PF {pf_text}, "
            f"win rate {row.get('win_rate', 0.0) * 100:.1f}%")


def run_regime_filtered(
    *,
    from_date: date,
    to_date: date,
    comparison_path: Path = DEFAULT_COMPARISON,
    regime_study_path: Path = DEFAULT_REGIME_STUDY,
    out_dir: Path = DEFAULT_OUT,
    underlying: str = "NIFTY",
    strategy_id: str = "S2",
) -> dict:
    assert_paper_trading_only()
    regime_study = json.loads(regime_study_path.read_text(encoding="utf-8"))
    avoid_rows = validate_hypothesis(regime_study, AVOID_REGIMES, ALLOW_REGIMES)
    final = load_final_comparison(comparison_path)
    coverage = coverage_for(final, underlying, strategy_id)

    service = HistoricalDataService()
    candles = service.read_underlying(underlying, from_date, to_date)
    if candles.empty:
        raise SystemExit(f"Missing underlying data for {underlying} {from_date}..{to_date}")
    candles["symbol"] = underlying.upper()
    contracts = service._contracts_from_store(underlying)
    premium = service._premium_history(underlying, from_date, to_date)
    trading_days = service.count_trading_days(underlying, from_date, to_date)
    research, strategy = service.research_config, service.strategy_config

    run_id = uuid.uuid4().hex[:12]

    def engine(signal_filter=None) -> OptionsResearchEngine:
        return OptionsResearchEngine(research, strategy, contracts, premium, run_id=run_id, split="FULL",
                                     master_resolver=service._research_resolver, signal_filter=signal_filter)

    original = engine().run(candles, underlying, strategy_id, from_date, to_date).as_dict()
    regime_filter = RegimeSignalFilter(PointInTimeRegimeClassifier(candles))
    filtered = engine(regime_filter).run(candles, underlying, strategy_id, from_date, to_date).as_dict()

    blocked_times = {s["time"] for s in filtered["skipped"] if s.get("reason") == REGIME_FILTERED}
    common = dict(trading_days=trading_days, from_date=from_date, to_date=to_date,
                  initial_equity=float(research.risk.initial_equity),
                  daily_loss_limit_inr=float(research.risk.max_daily_loss_inr))
    comparison = build_comparison(
        original=summarize(original, coverage=coverage, label="original", **common),
        filtered=summarize(filtered, coverage=filtered_coverage(coverage, blocked_times), label="filtered", **common),
        diff=trade_diff(original, filtered),
        filtered_regimes={rid: {**row, "reason": _filter_reason(row)} for rid, row in avoid_rows.items()},
        meta={
            "run_id": run_id,
            "underlying": underlying.upper(),
            "strategy_id": strategy_id,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "regime_study_run_id": regime_study.get("run_id"),
            "avoid_regimes": list(AVOID_REGIMES),
            "allow_regimes": list(ALLOW_REGIMES),
            "max_daily_loss_inr": float(research.risk.max_daily_loss_inr),
            "filter_decisions": regime_filter.decisions,
        },
    )
    paths = write_reports(comparison, out_dir)
    for path in paths.values():
        print(f"Wrote {path}")
    o, f = comparison["original"], comparison["filtered"]
    print(f"Original S2: trades={o['resolved_trades']} win={o['win_rate'] * 100:.1f}% "
          f"PF={o['profit_factor']:.2f} net={o['net_pnl']:,.0f} maxDD={o['max_drawdown']:,.0f}")
    print(f"Filtered S2: trades={f['resolved_trades']} win={f['win_rate'] * 100:.1f}% "
          f"PF={f['profit_factor']:.2f} net={f['net_pnl']:,.0f} maxDD={f['max_drawdown']:,.0f}")
    print(f"VERDICT: {comparison['verdict']}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 original vs regime-filtered backtest")
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--to-date", type=date.fromisoformat, default=date(2026, 10, 5))
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--regime-study", type=Path, default=DEFAULT_REGIME_STUDY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_regime_filtered(
        from_date=args.from_date,
        to_date=args.to_date,
        comparison_path=args.comparison,
        regime_study_path=args.regime_study,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
