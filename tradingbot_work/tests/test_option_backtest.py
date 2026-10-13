"""Option backtest: underlying signals, P&L only from historical premiums."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.options import DATA_UNAVAILABLE, NO_SIGNALS, OK, OptionBacktestEngine
from src.models.market import Candle
from src.options.demo_data import demo_premium_candles, demo_underlying_candles

from tests.options_support import contracts, strategy_stack

SPOT = 23000.0


def frame(candles):
    return pd.DataFrame([c.model_dump() for c in candles])


def nifty_atm_ce_key():
    return next(c.instrument_key for c in contracts()
                if c.underlying == "NIFTY" and c.strike == 23000.0 and c.option_type.value == "CE"
                and c.expiry.isoformat() == "2026-10-06")


def engine_with(premium_history):
    config, risk_manager, session_manager, strategy_engine, _ = strategy_stack()
    return OptionBacktestEngine(config, strategy_engine, risk_manager, session_manager, contracts(),
                                premium_history, initial_equity=1_000_000)


def premium_path(key, underlying):
    """Historical premium candles for the contract, rising after the signal bar."""
    premium = demo_premium_candles(key, underlying, "CE", SPOT, 0.05)
    last = premium[-1]
    future = []
    for step in range(1, 8):
        price = last.close + 6.0 * step
        future.append(Candle(timestamp=last.timestamp + timedelta(minutes=5 * step), symbol=key,
                             open=price - 3, high=price + 1, low=price - 4, close=price, volume=100.0))
    return premium + future


def test_without_premium_history_reports_data_unavailable():
    result = engine_with({}).run(frame(demo_underlying_candles("NIFTY", SPOT)), "NIFTY")
    assert result.status == DATA_UNAVAILABLE
    assert result.message == "Option historical data unavailable"
    assert result.backtest_type == "OPTION" and result.result is None


def test_premium_history_for_another_contract_is_not_substituted():
    underlying = demo_underlying_candles("NIFTY", SPOT)
    result = engine_with({"NSE_FO|999999": frame(premium_path("NSE_FO|999999", underlying))}).run(
        frame(underlying), "NIFTY")
    assert result.status == DATA_UNAVAILABLE
    assert result.signals == 1 and result.skipped[0]["reason"] == DATA_UNAVAILABLE


def test_option_trade_pnl_comes_from_premium_candles():
    underlying = demo_underlying_candles("NIFTY", SPOT)
    key = nifty_atm_ce_key()
    result = engine_with({key: frame(premium_path(key, underlying))}).run(frame(underlying), "NIFTY")
    assert result.status == OK and result.backtest_type == "OPTION"
    bt = result.result
    assert bt.backtest_type == "OPTION" and bt.exit_engine == "position_monitor"
    (trade,) = bt.trades
    assert trade["symbol"] == "NIFTY 23000 CE 06 OCT 26"
    assert trade["side"] == "BUY" and trade["quantity"] % 65 == 0
    assert trade["exit_reason"] == "TARGET"
    assert trade["exit_price"] == pytest.approx(trade["target"])
    assert trade["pnl"] == pytest.approx((trade["exit_price"] - trade["entry_price"]) * trade["quantity"])
    assert trade["pnl"] > 0
    assert result.as_dict()["result"]["backtest_type"] == "OPTION"


def test_no_signal_period_reports_no_signals():
    underlying = demo_underlying_candles("NIFTY", SPOT)[:-1]
    key = nifty_atm_ce_key()
    result = engine_with({key: frame(premium_path(key, underlying))}).run(frame(underlying), "NIFTY")
    assert result.status == NO_SIGNALS and result.signals == 0


def test_underlying_backtests_are_labelled_underlying():
    _, _, _, strategy_engine, _ = strategy_stack()
    result = BacktestEngine(strategy_engine).run(pd.DataFrame())
    assert result.backtest_type == "UNDERLYING"
