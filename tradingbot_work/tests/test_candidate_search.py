import pandas as pd
from datetime import datetime, timedelta

from src.backtest.candidate_search import screen


def _historical():
    rows = []
    for i in range(30):
        day = (datetime(2026, 9, 10) + timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append({
            "signal_time": f"{day}T11:00:00+05:30",
            "outcome": "WIN" if i % 3 else "LOSS",
            "direction": "SELL",
            "rsi14": 45,
            "atr_pct": 0.10,
            "ema_trend_aligned": True,
            "entry_time_bucket": "11:00-12:00",
            "distance_from_breakout_pct": 0.10,
            "option_vs_vwap": "above",
            "gap_condition": "gap_up",
            "pnl": 100.0 if i % 3 else -50.0,
        })
    return pd.DataFrame(rows)


def _current():
    return pd.DataFrame([{
        "timestamp": "2026-10-05T11:00:00+05:30",
        "outcome": "WIN",
        "direction": "SELL",
        "rsi": 45,
        "ema_alignment": "BULLISH (OPPOSED)",
        "time_of_day": "11:00-12:00",
        "option_vs_vwap": "ABOVE",
        "pnl_1_lot": 100.0,
    } for _ in range(10)])


def test_candidate_screen_is_oos_and_current_aware():
    results = screen(_historical(), _current(), min_train=5, min_oos=5, min_current=5)
    assert results
    assert all(r.oos_n >= 5 for r in results)
    assert all(r.current_n >= 5 for r in results)


def test_no_hindsight_columns_are_required():
    h = _historical()
    c = _current()
    h["false_breakout_after_entry"] = True
    h["mfe_r"] = 99
    h["mae_r"] = 99
    results = screen(h, c, min_train=5, min_oos=5, min_current=5)
    assert results
