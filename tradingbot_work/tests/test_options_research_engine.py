"""Options research backtest engine."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.options_engine import OptionsResearchEngine
from src.backtest.research_strategies import ResearchSignal
from src.config.loader import load_strategy_config
from src.config.research_loader import load_research_config

from tests.options_support import contracts

IST = ZoneInfo("Asia/Kolkata")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _underlying():
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    rows = []
    price = 23000.0
    for i in range(80):
        price += 5
        rows.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "symbol": "NIFTY", "open": price - 2, "high": price + 5, "low": price - 5, "close": price, "volume": 5000,
        })
    return pd.DataFrame(rows)


def _premium(key):
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    rows = []
    price = 120.0
    for i in range(80):
        price += 1.5
        rows.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "instrument_key": key, "open": price, "high": price + 2, "low": price - 1, "close": price,
            "volume": 100, "open_interest": 1000,
        })
    return pd.DataFrame(rows)


def test_engine_reports_no_data():
    research = load_research_config(ROOT / "config" / "strategy_research.yaml")
    strategy = load_strategy_config(ROOT / "config" / "strategy.yaml")
    engine = OptionsResearchEngine(research, strategy, contracts(), {}, run_id="t1")
    result = engine.run(pd.DataFrame(), "NIFTY", "S1", date(2026, 10, 1), date(2026, 10, 2))
    assert result.status == "NO_DATA"
