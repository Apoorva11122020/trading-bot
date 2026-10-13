import pandas as pd
import numpy as np
from src.indicators.technical import ema, previous_high, previous_low


def test_ema_has_expected_length():
    s = pd.Series(np.arange(1, 30, dtype=float))
    result = ema(s, 5)
    assert len(result) == len(s)
    assert result.iloc[-1] > result.iloc[5]


def test_breakout_levels_exclude_current_candle():
    df = pd.DataFrame({
        "high": [1,2,3,4,5,100],
        "low": [0,0,1,2,3,4],
        "close": [1,2,3,4,5,99],
    })
    high = previous_high(df, 5)
    low = previous_low(df, 5)
    assert high.iloc[-1] == 5
    assert low.iloc[-1] == 0
