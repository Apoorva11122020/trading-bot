#!/usr/bin/env python3
"""NIFTY S2 production risk-control audit (paper only)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.risk_audit import run_risk_audit, write_risk_audit_reports
from src.historical_data.service import HistoricalDataService

DEFAULT_COMPARISON = ROOT / "reports" / "strategy_research" / "final_comparison.json"
DEFAULT_OUT = ROOT / "reports" / "strategy_research"


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 risk-control audit")
    parser.add_argument("--run-id", default="b9202ebc90b7")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    service = HistoricalDataService()
    report = run_risk_audit(
        service=service,
        source_run_id=args.run_id,
        comparison_path=args.comparison,
    )
    paths = write_risk_audit_reports(report, args.out_dir)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['md']}")
    print("\n=== Risk Audit Conclusion ===")
    print(f"Meets ₹3,000 daily loss requirement: {'YES' if report.meets_client_requirement else 'PARTIAL'}")
    print(f"Replay trades: {report.replay_metrics.get('resolved_trades')}")
    print(f"Max daily realized loss: {report.replay_metrics.get('max_daily_realized_loss'):,.2f}")
    print(f"Max single-trade loss: {report.replay_metrics.get('max_single_trade_loss'):,.2f}")
    print(f"Days above daily cap: {report.replay_metrics.get('days_realized_loss_above_daily_cap')}")


if __name__ == "__main__":
    main()
