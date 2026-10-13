import numpy as np
import pandas as pd
import pytest

from scripts import classic_pivot_historical_backtest as full_backtest
from scripts import classic_pivot_short_oos as oos_backtest
from scripts import classic_pivot_short_only as short_backtest
from scripts.pivot_indicators import atr, ema, rma, rsi


@pytest.mark.parametrize("module", [full_backtest, short_backtest, oos_backtest])
def test_all_research_scripts_use_shared_pine_compatible_indicators(module):
    values = pd.Series([10.0, 12.0, 14.0, 16.0, 15.0, 17.0] * 8)
    assert np.allclose(module.ema(values, 3), ema(values, 3), equal_nan=True)
    assert np.allclose(module.rsi(values, 5), rsi(values, 5), equal_nan=True)
    candles = pd.DataFrame({
        "high": values + 1.0,
        "low": values - 1.0,
        "close": values,
    })
    assert np.allclose(module.atr(candles, 5), atr(candles, 5), equal_nan=True)


def test_rma_uses_sma_seed_then_wilder_recursion():
    actual = rma(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    assert actual.iloc[:2].isna().all()
    assert actual.iloc[2] == 2.0
    assert np.isclose(actual.iloc[3], 8.0 / 3.0)
    assert np.isclose(actual.iloc[4], 31.0 / 9.0)


def test_ema_matches_recursive_pine_initialization():
    actual = ema(pd.Series([10.0, 12.0, 14.0, 16.0]), 3)
    assert np.allclose(actual.to_numpy(), [10.0, 11.0, 12.5, 14.25])


def test_rsi_all_gains_and_all_losses_have_expected_limits():
    rising = pd.Series(np.arange(30, dtype=float))
    falling = pd.Series(np.arange(30, dtype=float)[::-1])
    assert np.isclose(rsi(rising, 14).iloc[-1], 100.0)
    assert np.isclose(rsi(falling, 14).iloc[-1], 0.0)


def test_atr_uses_wilder_rma_seed():
    candles = pd.DataFrame({
        "high": [11.0, 11.0, 11.0, 11.0],
        "low": [10.0, 10.0, 10.0, 10.0],
        "close": [10.5, 10.5, 10.5, 10.5],
    })
    actual = atr(candles, 3)
    assert np.isnan(actual.iloc[0:2]).all()
    assert np.isclose(actual.iloc[2], 1.0)
    assert np.isclose(actual.iloc[3], 1.0)
