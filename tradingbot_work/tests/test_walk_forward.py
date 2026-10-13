"""Walk-forward validation: date split, look-ahead prevention, frozen filter, verdict."""
from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.regime_filter import AVOID_REGIMES, PointInTimeRegimeClassifier
from src.backtest.walk_forward import (
    Period,
    coverage_for_period,
    slice_candles,
    slice_premium,
    validate_split,
    walk_forward_verdict,
)
from src.safety.live_gate import LiveTradingSafetyError

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
DEV = Period("development", date(2026, 9, 1), date(2026, 9, 23))
OOS = Period("oos", date(2026, 9, 24), date(2026, 10, 5))


def _frame(days: list[int], month: int = 9) -> pd.DataFrame:
    rows = []
    for day in days:
        start = datetime(2026, month, day, 9, 15, tzinfo=IST)
        for i in range(10):
            rows.append({"timestamp": start + timedelta(minutes=5 * i), "symbol": "NIFTY",
                         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 0.0})
    return pd.DataFrame(rows)


def test_default_split_is_contiguous_and_non_overlapping():
    validate_split(DEV, OOS)
    assert DEV.end < OOS.start
    assert (OOS.start - DEV.end).days == 1


def test_overlapping_or_inverted_split_rejected():
    with pytest.raises(ValueError):
        validate_split(DEV, Period("oos", date(2026, 9, 23), date(2026, 10, 5)))
    with pytest.raises(ValueError):
        validate_split(Period("development", date(2026, 9, 23), date(2026, 9, 1)), OOS)


def test_period_contains_uses_ist_dates():
    assert DEV.contains("2026-09-23T15:25:00+05:30")
    assert not DEV.contains("2026-09-24T09:15:00+05:30")
    # 2026-09-23 19:00 UTC is 2026-09-24 00:30 IST
    assert OOS.contains("2026-09-23T19:00:00+00:00")


def test_development_slice_has_no_oos_candles_or_premium():
    candles = pd.concat([_frame([22, 23, 24, 25])], ignore_index=True)
    dev_candles = slice_candles(candles, DEV.start, DEV.end)
    assert pd.to_datetime(dev_candles["timestamp"]).dt.date.max() == date(2026, 9, 23)
    premium = slice_premium({"K": candles.copy()}, DEV.start, DEV.end)
    assert pd.to_datetime(premium["K"]["timestamp"]).dt.date.max() == date(2026, 9, 23)


def test_dev_regime_decisions_identical_with_or_without_oos_data():
    full = _frame([15, 16, 17, 18, 19, 22, 23, 24, 25])
    dev_only = slice_candles(full, DEV.start, DEV.end)
    ts = datetime(2026, 9, 23, 10, 0, tzinfo=IST)
    assert PointInTimeRegimeClassifier(full).regimes_at(ts) == PointInTimeRegimeClassifier(dev_only).regimes_at(ts)


def test_regime_filter_is_frozen():
    assert AVOID_REGIMES == ("range_bound", "low_volatility", "sudden_upward_spike")


def test_coverage_split_by_period():
    coverage = {"unavailable": [{"signal_time": "2026-09-10T10:00:00+05:30"},
                                {"signal_time": "2026-09-25T10:00:00+05:30"}]}
    times = ["2026-09-10T10:00:00+05:30", "2026-09-11T10:00:00+05:30", "2026-09-25T10:00:00+05:30"]
    dev = coverage_for_period(coverage, times, DEV)
    oos = coverage_for_period(coverage, times, OOS)
    assert (dev["total_signals"], dev["signals_missing_coverage"]) == (2, 1)
    assert (oos["total_signals"], oos["signals_missing_coverage"]) == (1, 1)
    assert oos["coverage_percent"] == 0.0


def _s(**kw):
    base = {"trading_days": 25, "resolved_trades": 40, "profit_factor": 1.5, "net_pnl": 10000.0, "max_drawdown": 4000.0}
    return {**base, **kw}


def _results(o_dev, o_oos, f_dev, f_oos):
    return {"original": {"development": o_dev, "oos": o_oos}, "filtered": {"development": f_dev, "oos": f_oos}}


def test_verdict_needs_more_data_for_short_oos():
    verdict, _ = walk_forward_verdict(_results(_s(), _s(trading_days=6), _s(), _s(trading_days=6)))
    assert verdict == "NEEDS MORE DATA"


def test_verdict_prefer_filtered_only_when_better_in_both_periods():
    better = _s(profit_factor=2.0, net_pnl=12000.0, max_drawdown=3000.0)
    assert walk_forward_verdict(_results(_s(), _s(), better, better))[0] == "PREFER FILTERED"
    assert walk_forward_verdict(_results(_s(), _s(), _s(), better))[0] != "PREFER FILTERED"


def test_verdict_prefer_original_and_robust():
    worse = _s(profit_factor=1.2, net_pnl=5000.0)
    assert walk_forward_verdict(_results(_s(), _s(), _s(), worse))[0] == "PREFER ORIGINAL"
    assert walk_forward_verdict(_results(_s(), _s(), _s(), _s()))[0] == "ROBUST"


def test_script_refuses_live(monkeypatch):
    spec = importlib.util.spec_from_file_location("wf", ROOT / "scripts" / "run_walk_forward_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(LiveTradingSafetyError):
        module.run_walk_forward(development=DEV, out_of_sample=OOS)
    text = (ROOT / "scripts" / "run_walk_forward_validation.py").read_text(encoding="utf-8")
    assert "place_order" not in text and "create_broker" not in text
