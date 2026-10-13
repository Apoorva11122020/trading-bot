"""Final strategy research report helpers."""
from __future__ import annotations

from datetime import date

from src.backtest.final_research_report import (
    MIN_RESOLVED_TRADES,
    row_from_result,
    rank_strategies,
    StrategyResearchRow,
    _status,
)


def test_insufficient_sample_status():
    assert _status(29) == "INSUFFICIENT SAMPLE"
    assert _status(30) == "VALID RESEARCH SAMPLE"


def test_row_from_result_counts_unresolved_not_as_loss():
    result = {
        "metrics": {
            "total_signals": 100,
            "resolved_trades": 10,
            "win_rate": 0.5,
            "profit_factor": 1.2,
            "total_pnl": 5000,
            "max_drawdown": 1000,
            "average_r": 0.1,
            "average_trades_per_day": 0.5,
            "max_consecutive_losses": 2,
            "daily_loss_hits": 1,
        },
        "trades": [
            {"pnl": 1000, "r_multiple": 1.0},
            {"pnl": -500, "r_multiple": -0.5},
        ],
    }
    row = row_from_result(
        underlying="NIFTY",
        strategy_id="S2",
        from_date=date(2026, 9, 1),
        to_date=date(2026, 10, 1),
        result=result,
        coverage={"coverage_percent": 100, "unique_required_contracts": 5, "contracts_downloaded": 5,
                  "contracts_unavailable": 0},
        trading_days=20,
    )
    assert row.unresolved_signals == 90
    assert row.signals == 100
    assert row.resolved_trades == 10
    assert row.gross_profit == 1000
    assert row.gross_loss == 500


def test_rank_strategies_only_valid_samples():
    rows = [
        StrategyResearchRow(
            underlying="NIFTY", strategy_id="S1", from_date="a", to_date="b", trading_days=20,
            signals=50, resolved_trades=35, unresolved_signals=15, win_rate=0.5,
            gross_profit=1, gross_loss=1, profit_factor=1.5, net_pnl=100, max_drawdown=0,
            average_r=0, median_r=0, average_trades_per_day=1, max_consecutive_losses=0,
            daily_loss_limit_breaches=0, premium_coverage_percent=100, resolved_trade_percent=70,
            required_contracts=5, contracts_downloaded=5, contracts_unavailable=0,
            status="VALID RESEARCH SAMPLE",
        ),
        StrategyResearchRow(
            underlying="NIFTY", strategy_id="S2", from_date="a", to_date="b", trading_days=20,
            signals=50, resolved_trades=10, unresolved_signals=40, win_rate=0.4,
            gross_profit=1, gross_loss=1, profit_factor=0.8, net_pnl=200, max_drawdown=0,
            average_r=0, median_r=0, average_trades_per_day=1, max_consecutive_losses=0,
            daily_loss_limit_breaches=0, premium_coverage_percent=100, resolved_trade_percent=20,
            required_contracts=5, contracts_downloaded=5, contracts_unavailable=0,
            status="INSUFFICIENT SAMPLE",
        ),
    ]
    order = rank_strategies(rows, "NIFTY")
    assert order == ["S1"]
