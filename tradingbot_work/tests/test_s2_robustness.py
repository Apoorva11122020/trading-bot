"""NIFTY S2 robustness analysis calculations."""
from __future__ import annotations

from datetime import date

import pytest

from src.backtest.options_engine import DAILY_LOSS, DATA_INVALID
from src.backtest.robustness import (
    ENTRY_TIME_BUCKETS,
    analyze_robustness,
    assess_trade_concentration,
    daily_loss_limit_behavior,
    entry_time_bucket,
    option_coverage_summary,
    profit_factor_excluding_top_n_winners,
    robustness_verdict,
    write_robustness_reports,
)


def _trade(pnl: float, entry_time: str, signal_time: str | None = None) -> dict:
    return {
        "pnl": pnl,
        "r_multiple": pnl / 1000.0,
        "entry_time": entry_time,
        "signal_time": signal_time or entry_time,
        "exit_time": entry_time,
        "direction": "BUY",
        "option_type": "CE",
        "trading_symbol": "NIFTY TEST CE",
        "exit_reason": "TARGET",
    }


def _result(trades: list[dict], signals: int, skipped: list[dict] | None = None) -> dict:
    return {
        "trades": trades,
        "signals": [{}] * signals,
        "skipped": skipped or [],
        "metrics": {"total_signals": signals},
    }


def _coverage(total: int, covered: int, missing: int) -> dict:
    return {
        "total_signals": total,
        "signals_with_complete_coverage": covered,
        "signals_missing_coverage": missing,
        "coverage_percent": round(100.0 * covered / total, 2) if total else 0.0,
        "unique_required_contracts": 1,
        "contracts_downloaded": 1,
        "contracts_unavailable": 0,
        "unavailable": [],
    }


def test_entry_time_bucket_boundaries():
    assert entry_time_bucket("2026-09-01T09:15:00+05:30") == "09:15-10:00"
    assert entry_time_bucket("2026-09-01T09:59:00+05:30") == "09:15-10:00"
    assert entry_time_bucket("2026-09-01T10:00:00+05:30") == "10:00-11:00"
    assert entry_time_bucket("2026-09-01T13:29:00+05:30") == "13:00-14:30"
    assert entry_time_bucket("2026-09-01T14:30:00+05:30") is None
    assert len(ENTRY_TIME_BUCKETS) == 5


def test_profit_factor_excluding_top_winners():
    trades = [_trade(500, "2026-09-01T10:05:00+05:30"), _trade(300, "2026-09-01T11:05:00+05:30"),
              _trade(-200, "2026-09-01T12:05:00+05:30")]
    assert profit_factor_excluding_top_n_winners(trades, 1) == pytest.approx(300 / 200)
    assert profit_factor_excluding_top_n_winners(trades, 2) == 0.0


def test_daily_loss_limit_behavior():
    trades = [_trade(-3500, "2026-09-01T10:05:00+05:30"), _trade(100, "2026-09-02T10:05:00+05:30")]
    skipped = [
        {"time": "2026-09-01T11:00:00+05:30", "reason": DAILY_LOSS},
        {"time": "2026-09-01T11:05:00+05:30", "reason": DAILY_LOSS},
        {"time": "2026-09-02T12:00:00+05:30", "reason": DATA_INVALID},
    ]
    info = daily_loss_limit_behavior(skipped, trades, limit_inr=3000)
    assert info["signals_skipped_after_limit"] == 2
    assert info["days_with_limit_skip"] == 1
    assert info["days_with_realized_loss_at_or_below_limit"] == 1


def test_option_coverage_summary_marks_affected_trades():
    coverage = _coverage(3, 2, 1)
    coverage["unavailable"] = [{"signal_time": "2026-09-01T10:00:00+05:30"}]
    trades = [
        _trade(100, "2026-09-01T10:05:00+05:30", "2026-09-01T10:00:00+05:30"),
        _trade(50, "2026-09-02T10:05:00+05:30", "2026-09-02T10:00:00+05:30"),
    ]
    skipped = [{"time": "2026-09-03T10:00:00+05:30", "reason": DATA_INVALID}]
    summary = option_coverage_summary(coverage, trades, skipped)
    assert summary["signals_with_complete_coverage"] == 2
    assert summary["signals_missing_coverage"] == 1
    assert summary["resolved_trades_affected_by_missing_coverage"] == 1
    assert summary["data_invalid_skips"] == 1


def test_analyze_robustness_core_metrics():
    trades = [
        _trade(1000, "2026-09-01T10:05:00+05:30"),
        _trade(-500, "2026-09-01T11:05:00+05:30"),
        _trade(300, "2026-09-02T10:05:00+05:30"),
        _trade(-200, "2026-09-02T11:05:00+05:30"),
    ]
    report = analyze_robustness(
        run_id="test-run",
        source_run_id="source-run",
        underlying="NIFTY",
        strategy_id="S2",
        from_date=date(2026, 9, 1),
        to_date=date(2026, 9, 2),
        trading_days=2,
        result=_result(trades, signals=6),
        coverage=_coverage(6, 5, 1),
        initial_equity=100000,
        daily_loss_limit_inr=3000,
    )
    assert report.total_signals == 6
    assert report.resolved_trades == 4
    assert report.unresolved_signals == 2
    assert report.win_rate == 0.5
    assert report.net_pnl == 600
    assert report.gross_profit == 1300
    assert report.gross_loss == 700
    assert report.profit_factor == pytest.approx(1300 / 700)
    assert report.max_consecutive_wins == 1
    assert report.max_consecutive_losses == 1
    assert report.average_trades_per_day == 2.0
    assert len(report.pnl_by_day) == 2
    assert len(report.pnl_by_week) == 1
    assert len(report.performance_by_entry_bucket) == 5
    assert len(report.largest_winning_trades) == 2
    assert len(report.largest_losing_trades) == 2


def test_unresolved_signals_not_counted_as_losses():
    trades = [_trade(100, "2026-09-01T10:05:00+05:30")]
    report = analyze_robustness(
        run_id="r1",
        source_run_id="src",
        underlying="NIFTY",
        strategy_id="S2",
        from_date=date(2026, 9, 1),
        to_date=date(2026, 9, 1),
        trading_days=1,
        result=_result(trades, signals=10),
        coverage=_coverage(10, 8, 2),
    )
    assert report.unresolved_signals == 9
    assert report.win_rate == 1.0
    assert report.gross_loss == 0


def test_trade_concentration_flags_dependency():
    concentration = assess_trade_concentration(
        [], top_1_share=45.0, top_3_share=60.0, pf_ex_top_3=0.8,
    )
    assert concentration["highly_dependent_on_few_trades"] is True
    assert concentration["reasons"]


def test_robustness_verdict_needs_more_data_for_short_sample():
    verdict, reasons = robustness_verdict(
        resolved_trades=58,
        coverage_percent=88.6,
        trade_concentration={"highly_dependent_on_few_trades": False, "reasons": []},
        trading_days=22,
    )
    assert verdict == "NEEDS MORE DATA"
    assert any("22 trading days" in r for r in reasons)


def test_write_robustness_reports(tmp_path):
    trades = [_trade(100, "2026-09-01T10:05:00+05:30"), _trade(-50, "2026-09-02T10:05:00+05:30")]
    report = analyze_robustness(
        run_id="out-run",
        source_run_id="src-run",
        underlying="NIFTY",
        strategy_id="S2",
        from_date=date(2026, 9, 1),
        to_date=date(2026, 9, 2),
        trading_days=2,
        result=_result(trades, signals=3),
        coverage=_coverage(3, 3, 0),
    )
    paths = write_robustness_reports(report, tmp_path)
    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert paths["md"].exists()
    assert "NIFTY S2 Robustness Report" in paths["md"].read_text(encoding="utf-8")
