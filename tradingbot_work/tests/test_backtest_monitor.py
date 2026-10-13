import csv
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, export_trade_journal
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager
from src.strategy.strategy_engine import StrategyEngine

from trading_support import CFG, IST


def _engine():
    return StrategyEngine(CFG, RiskManager(CFG.risk), SessionManager(CFG.session))


def _breakout_then(path, count=55):
    """Deterministic uptrend + breakout bar, followed by caller-defined bars
    given as (open, high, low, close)."""
    start = datetime(2026, 10, 1, 9, 30, tzinfo=IST)
    rows, price = [], 100.0
    for i in range(count - 1):
        price += 0.15
        rows.append({"timestamp": start + timedelta(minutes=5 * i), "symbol": "RELIANCE",
                     "open": price - 0.10, "high": price + 0.20, "low": price - 0.20,
                     "close": price, "volume": 100.0})
    price += 5.0
    rows.append({"timestamp": start + timedelta(minutes=5 * (count - 1)), "symbol": "RELIANCE",
                 "open": price - 0.20, "high": price + 0.20, "low": price - 0.30,
                 "close": price, "volume": 200.0})
    t = rows[-1]["timestamp"]
    for k, (o, h, l, c) in enumerate(path(price), start=1):
        rows.append({"timestamp": t + timedelta(minutes=5 * k), "symbol": "RELIANCE",
                     "open": o, "high": h, "low": l, "close": c, "volume": 100.0})
    return pd.DataFrame(rows), price


def _first_entry(data):
    engine = _engine()
    for i in range(len(data)):
        ts = data.iloc[i]["timestamp"].to_pydatetime()
        from src.models.risk import AccountState
        signal = engine.evaluate_entry(data.iloc[: i + 1], AccountState(equity=100_000), ts)
        if signal and not signal.blocked:
            return signal
    raise AssertionError("no entry")


def test_backtest_uses_runtime_position_monitor_and_strategy_levels():
    data, entry_price = _breakout_then(lambda p: [(p, p + 50, p - 0.1, p + 0.5)])
    signal = _first_entry(data)
    result = BacktestEngine(_engine()).run(data)
    assert result.exit_engine == "position_monitor"
    trade = result.trades[0]
    assert trade["exit_reason"] == "TARGET"
    assert trade["stop_loss"] == pytest.approx(signal.stop_loss)
    assert trade["target"] == pytest.approx(signal.target)
    assert trade["exit_price"] == pytest.approx(signal.target)


def test_backtest_stop_loss_is_checked_before_target_intrabar():
    data, _ = _breakout_then(lambda p: [(p, p + 50, p - 50, p)])
    result = BacktestEngine(_engine()).run(data)
    assert result.trades[0]["exit_reason"] == "STOP_LOSS"
    assert result.net_pnl < 0


def test_backtest_trailing_stop_matches_runtime_monitor():
    base, _ = _breakout_then(lambda p: [])
    signal = _first_entry(base)
    risk = signal.entry_price - signal.stop_loss

    def path(p):
        # Rally to 1.8R (below the 2R target, above breakeven activation),
        # then a pullback that crosses the trailing stop.
        return [
            (p, p + 1.8 * risk, p - 0.1, p + 1.7 * risk),
            (p + 1.7 * risk, p + 1.75 * risk, p + 0.2, p + 0.3),
        ]

    data, _ = _breakout_then(path)
    result = BacktestEngine(_engine()).run(data)
    trade = result.trades[0]
    assert trade["exit_reason"] == "TRAILING_STOP"
    assert signal.entry_price < trade["exit_price"] < signal.entry_price + 1.8 * risk
    assert trade["pnl"] > 0


def test_backtest_eod_force_exit():
    def path(p):
        bars = []
        for _ in range(60):
            bars.append((p, p + 0.05, p - 0.05, p))
        return bars
    data, _ = _breakout_then(path)
    result = BacktestEngine(_engine()).run(data)
    assert result.trades[0]["exit_reason"] == "EOD_FORCE_EXIT"
    exit_time = pd.Timestamp(result.trades[0]["exit_time"])
    assert (exit_time.hour, exit_time.minute) >= (15, 15)


def test_backtest_metrics_equity_curve_and_export(tmp_path):
    data, _ = _breakout_then(lambda p: [(p, p + 50, p - 0.1, p + 0.5)])
    result = BacktestEngine(_engine(), commission_per_order=5).run(data)
    assert result.equity_curve[0]["equity"] == 100_000
    assert result.equity_curve[-1]["equity"] == pytest.approx(result.ending_equity)
    assert result.total_costs == 10
    assert result.win_rate == 100.0
    assert result.max_drawdown == 0.0
    assert "do not indicate future profitability" in result.disclaimer
    path = export_trade_journal(result, tmp_path / "journal.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == result.total_trades
    assert rows[0]["exit_reason"] == "TARGET"


def test_backtest_is_deterministic():
    data, _ = _breakout_then(lambda p: [(p, p + 50, p - 0.1, p + 0.5)])
    a = BacktestEngine(_engine()).run(data).as_dict()
    b = BacktestEngine(_engine()).run(data).as_dict()
    assert a == b


def test_drawdown_and_profit_factor():
    engine = BacktestEngine(_engine())
    from src.backtest.engine import BacktestTrade
    trades = [BacktestTrade("X", "BUY", 1, 100, 110, "a", "b", 10, 0, 10, "TARGET"),
              BacktestTrade("X", "BUY", 1, 100, 95, "a", "b", -5, 0, -5, "STOP_LOSS"),
              BacktestTrade("X", "BUY", 1, 100, 97, "a", "b", -3, 0, -3, "STOP_LOSS")]
    result = engine._result(trades)
    assert result.max_drawdown == pytest.approx(8.0)
    assert result.profit_factor == pytest.approx(10 / 8)
    assert result.winning_trades == 1 and result.losing_trades == 2
