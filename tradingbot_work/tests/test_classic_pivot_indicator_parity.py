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

def test_pivots_ignore_out_of_session_candles():
    timestamps = pd.to_datetime([
        "2026-10-01 09:15:00+05:30",
        "2026-10-01 15:25:00+05:30",
        "2026-10-01 15:35:00+05:30",
        "2026-10-02 09:20:00+05:30",
    ])
    candles = pd.DataFrame({
        "ts": timestamps,
        "session": timestamps.date,
        "open": [100.0, 100.0, 500.0, 100.0],
        "high": [105.0, 110.0, 1000.0, 102.0],
        "low": [95.0, 90.0, 1.0, 98.0],
        "close": [100.0, 100.0, 500.0, 100.0],
        "volume": [0.0, 0.0, 0.0, 0.0],
    })
    prepared = oos_backtest.prepare(candles, volume_filter=False)
    second_day = prepared.loc[prepared["session"] == timestamps[3].date()].iloc[0]
    # Previous regular-session H/L/C = 110/90/100, so PP = 100.
    assert np.isclose(second_day["pp"], 100.0)



def test_oos_backtest_force_closes_at_last_candle_in_window():
    # A SHORT signal is injected on the first eligible candle. The next
    # available candle is after the requested test window and must not be used
    # as the position's closing price.
    timestamps = pd.date_range(
        "2026-06-01 09:15", periods=14, freq="5min", tz="Asia/Kolkata"
    )
    timestamps = timestamps.append(pd.DatetimeIndex([
        pd.Timestamp("2026-06-03 09:15", tz="Asia/Kolkata")
    ]))
    closes = np.linspace(100.0, 90.0, len(timestamps))
    frame = pd.DataFrame({
        "ts": timestamps,
        "session": timestamps.date,
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": 1.0,
    })

    # Make the signal deterministic while leaving the actual exit manager
    # under test. Signals outside the requested window remain false.
    original_prepare = oos_backtest.prepare
    def prepared_with_signal(df, volume_filter):
        result = original_prepare(df, volume_filter)
        result["short_signal"] = False
        result.loc[result.index[0], "short_signal"] = True
        result["atr14"] = 1.0
        result["flat_time"] = False
        result["flat_window_end"] = (
            result["ts"].dt.date == pd.Timestamp("2026-06-01").date()
        ) & (result["ts"].dt.hour * 60 + result["ts"].dt.minute >= 15 * 60 + 15)
        return result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(oos_backtest, "prepare", prepared_with_signal)
    try:
        trades, _ = oos_backtest.backtest(
            frame, "NIFTY_50", False,
            pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-01"),
            commission_pct=0.0, slippage_points=0.0,
        )
    finally:
        monkeypatch.undo()

    assert len(trades) == 1
    assert pd.Timestamp(trades.iloc[0]["exit_ts"]).date() == pd.Timestamp("2026-06-01").date()


def test_oos_strategy_flat_time_matches_pine_strategy_1500_close():
    timestamps = pd.to_datetime([
        "2026-10-01 14:55:00+05:30",
        "2026-10-01 15:00:00+05:30",
        "2026-10-01 15:15:00+05:30",
    ])
    candles = pd.DataFrame({
        "ts": timestamps,
        "session": timestamps.date,
        "open": [100.0, 100.0, 100.0],
        "high": [101.0, 101.0, 101.0],
        "low": [99.0, 99.0, 99.0],
        "close": [100.0, 100.0, 100.0],
        "volume": [0.0, 0.0, 0.0],
    })
    prepared = oos_backtest.prepare(candles, volume_filter=False)
    assert prepared["flat_time"].tolist() == [False, True, True]
