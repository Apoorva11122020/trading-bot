"""Research strategy signal generation."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.research_strategies import generate_signals, signals_s1
from src.config.research_loader import load_research_config

IST = ZoneInfo("Asia/Kolkata")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _frame(prices):
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    rows = []
    for i, close in enumerate(prices):
        rows.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "open": close - 1, "high": close + 2, "low": close - 2, "close": close,
            "volume": 1000 + i * 50, "symbol": "NIFTY",
        })
    return pd.DataFrame(rows)


def test_s1_generates_no_signals_on_flat_data():
    config = load_research_config(ROOT / "config" / "strategy_research.yaml")
    signals = generate_signals("S1", _frame([100] * 60), "NIFTY", config)
    assert signals == []


def test_strategy_catalog_lists_four():
    config = load_research_config(ROOT / "config" / "strategy_research.yaml")
    assert len(config.strategies) == 4
