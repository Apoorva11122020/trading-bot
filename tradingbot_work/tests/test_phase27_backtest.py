from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine


class StubSession:
    def should_force_exit(self, ts):
        return ts.hour >= 15 and ts.minute >= 15


class StubSignal:
    blocked = False
    quantity = 10
    symbol = "DEMO"
    side = type("Side", (), {"value": "BUY"})()
    entry_price = 100.0
    stop_loss = 99.0
    target = 102.0


class StubStrategy:
    session_manager = StubSession()
    def evaluate_entry(self, history, account, timestamp):
        if len(history) == 1:
            return StubSignal()
        return None


def candles(rows):
    start = datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    return pd.DataFrame([
        {"timestamp": start + timedelta(minutes=5*i), "symbol": "DEMO",
         "open": o, "high": h, "low": l, "close": c, "volume": 1000}
        for i, (o, h, l, c) in enumerate(rows)
    ])


def test_backtest_applies_adverse_slippage_and_costs():
    data = candles([(100, 100, 100, 100), (100, 102, 100, 102)])
    result = BacktestEngine(StubStrategy(), slippage_bps=10, commission_per_order=5).run(data)
    assert result.total_trades == 1
    assert result.trades[0]["exit_reason"] == "TARGET"
    assert result.total_costs == 10
    assert result.net_pnl < 20


def test_stop_loss_wins_when_stop_and_target_hit_same_candle():
    data = candles([(100, 100, 100, 100), (100, 103, 98, 101)])
    result = BacktestEngine(StubStrategy()).run(data)
    assert result.total_trades == 1
    assert result.trades[0]["exit_reason"] == "STOP_LOSS"
    assert result.net_pnl < 0


def test_open_position_is_realized_at_data_end():
    data = candles([(100, 100, 100, 100), (100, 101, 99.5, 100.5)])
    result = BacktestEngine(StubStrategy()).run(data)
    assert result.total_trades == 1
    assert result.trades_closed_at_data_end == 1
    assert result.trades[0]["exit_reason"] == "DATA_END"


def test_invalid_backtest_parameters_are_rejected():
    with pytest.raises(ValueError):
        BacktestEngine(StubStrategy(), initial_equity=0)
    with pytest.raises(ValueError):
        BacktestEngine(StubStrategy(), slippage_bps=-1)
