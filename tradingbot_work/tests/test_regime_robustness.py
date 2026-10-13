"""Market-regime labeling and S2 regime robustness aggregation."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.regime_classifier import (
    REGIME_CATALOG,
    build_regime_labels,
    classify_day,
    DayFeatures,
)
from src.backtest.regime_robustness import analyze_regime_robustness

IST = ZoneInfo("Asia/Kolkata")
S2_CFG = {"range_start": "09:15", "range_end": "09:30", "atr_period": 14}


def _bar(day: int, h: int, m: int, o: float, hi: float, lo: float, c: float) -> dict:
    return {
        "timestamp": datetime(2026, 9, day, h, m, tzinfo=IST),
        "open": o,
        "high": hi,
        "low": lo,
        "close": c,
        "volume": 1000.0,
    }


def _flat_day(day: int, base: float = 100.0, n: int = 20) -> list[dict]:
    rows = []
    minute = 15
    for i in range(n):
        h = 9 + (minute + i * 15) // 60
        m = (minute + i * 15) % 60
        rows.append(_bar(day, h, m, base, base + 0.1, base - 0.1, base))
    return rows


def test_regime_catalog_has_fourteen_regimes():
    assert len(REGIME_CATALOG) == 14


def test_gap_up_detected_without_future_data():
    history = [
        DayFeatures(
            day=datetime(2026, 9, 1, tzinfo=IST).date(),
            open=100,
            high=101,
            low=99,
            close=100,
            prev_close=None,
            day_return=0,
            day_range_pct=0.02,
            max_bar_up=0.001,
            max_bar_down=-0.001,
            sign_changes=2,
            opening_range_pct=0.01,
            midday_range_pct=0.005,
            late_range_pct=0.004,
            false_breakout=False,
        )
    ]
    feat = DayFeatures(
        day=datetime(2026, 9, 2, tzinfo=IST).date(),
        open=101.5,
        high=102,
        low=101,
        close=101.8,
        prev_close=100.0,
        day_return=0.018,
        day_range_pct=0.01,
        max_bar_up=0.002,
        max_bar_down=-0.001,
        sign_changes=2,
        opening_range_pct=0.008,
        midday_range_pct=0.004,
        late_range_pct=0.003,
        false_breakout=False,
    )
    tags = classify_day(feat, history)
    assert "gap_up" in tags
    assert "strong_bullish_trend" in tags


def test_rolling_labels_do_not_use_same_day_for_percentiles():
    rows = []
    for day in range(1, 8):
        base = 100 + day
        rows.extend(_flat_day(day, base))
    # Day 8: large range day
    rows.extend(
        [
            _bar(8, 9, 15, 200, 210, 190, 205),
            _bar(8, 9, 30, 205, 212, 198, 208),
            _bar(8, 10, 0, 208, 215, 200, 210),
        ]
    )
    frame = pd.DataFrame(rows)
    labels = build_regime_labels(frame, S2_CFG)
    assert labels[datetime(2026, 9, 8, tzinfo=IST).date()]
    # First days have limited history — should not crash
    assert isinstance(labels[datetime(2026, 9, 1, tzinfo=IST).date()], set)


def test_analyze_regime_robustness_splits_trades_by_day():
    candles = pd.DataFrame(_flat_day(1) + _flat_day(2, base=110))
    result = {
        "trades": [
            {
                "pnl": 500,
                "r_multiple": 0.5,
                "entry_time": "2026-09-01T10:05:00+05:30",
                "signal_time": "2026-09-01T10:00:00+05:30",
            },
            {
                "pnl": -200,
                "r_multiple": -0.2,
                "entry_time": "2026-09-02T10:05:00+05:30",
                "signal_time": "2026-09-02T10:00:00+05:30",
            },
        ],
        "signals": [
            {"timestamp": "2026-09-01T10:00:00+05:30"},
            {"timestamp": "2026-09-02T10:00:00+05:30"},
        ],
        "skipped": [{"time": "2026-09-02T11:00:00+05:30", "reason": "DATA_INVALID"}],
        "metrics": {"total_signals": 2},
    }
    coverage = {
        "total_signals": 2,
        "signals_with_complete_coverage": 2,
        "signals_missing_coverage": 0,
        "coverage_percent": 100.0,
        "unavailable": [],
    }
    report = analyze_regime_robustness(
        run_id="test",
        source_run_id="src",
        underlying="NIFTY",
        strategy_id="S2",
        from_date=datetime(2026, 9, 1, tzinfo=IST).date(),
        to_date=datetime(2026, 9, 2, tzinfo=IST).date(),
        trading_days=2,
        underlying_candles=candles,
        result=result,
        coverage=coverage,
        s2_cfg=S2_CFG,
    )
    assert len(report.regimes) == 14
    total_trades = sum(r.resolved_trades for r in report.regimes)
    assert total_trades >= 2
