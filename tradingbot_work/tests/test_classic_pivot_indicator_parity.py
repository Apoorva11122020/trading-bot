import numpy as np
import pandas as pd

from scripts.classic_pivot_short_oos import atr, rma, rsi


def test_rma_uses_sma_seed_then_wilder_recursion():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = rma(values, 3)

    assert actual.iloc[:2].isna().all()
    assert actual.iloc[2] == 2.0
    assert np.isclose(actual.iloc[3], 8.0 / 3.0)
    assert np.isclose(actual.iloc[4], 31.0 / 9.0)


def test_rsi_all_gains_and_all_losses_have_expected_limits():
    rising = pd.Series(np.arange(30, dtype=float))
    falling = pd.Series(np.arange(30, dtype=float)[::-1])

    rising_rsi = rsi(rising, 14)
    falling_rsi = rsi(falling, 14)

    assert np.isclose(rising_rsi.iloc[-1], 100.0)
    assert np.isclose(falling_rsi.iloc[-1], 0.0)


def test_atr_uses_wilder_rma_seed():
    # Keep every previous close inside each candle's range so true range is 1.0
    # on every bar (no overnight/gap component inflates the first RMA seed).
    candles = pd.DataFrame({
        "high": [11.0, 11.0, 11.0, 11.0],
        "low": [10.0, 10.0, 10.0, 10.0],
        "close": [10.5, 10.5, 10.5, 10.5],
    })

    actual = atr(candles, 3)

    assert np.isnan(actual.iloc[0:2]).all()
    assert np.isclose(actual.iloc[2], 1.0)
    assert np.isclose(actual.iloc[3], 1.0)
