"""Indicator helpers matching TradingView's built-in EMA, RSI and ATR conventions.

EMA uses the first non-null source value as its recursive seed.
RMA uses an SMA seed of the first n valid values, then Wilder recursion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    if n < 1:
        raise ValueError("length must be >= 1")
    return s.astype(float).ewm(span=n, adjust=False, min_periods=1).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    if n < 1:
        raise ValueError("length must be >= 1")
    values = s.astype(float)
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    positions = np.flatnonzero(valid.to_numpy())
    if len(positions) < n:
        return out
    seed_end = int(positions[n - 1])
    seed_start = int(positions[0])
    seed_values = values.iloc[positions[:n]]
    previous = float(seed_values.mean())
    out.iloc[seed_end] = previous
    for i in range(seed_end + 1, len(values)):
        value = values.iloc[i]
        if pd.isna(value):
            out.iloc[i] = np.nan
            continue
        previous = (previous * (n - 1) + float(value)) / n
        out.iloc[i] = previous
    return out


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = rma(gain, n)
    avg_loss = rma(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    result = result.mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
    return result


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ATR input missing columns: {sorted(missing)}")
    previous_close = df["close"].astype(float).shift(1)
    true_range = pd.concat(
        [
            df["high"].astype(float) - df["low"].astype(float),
            (df["high"].astype(float) - previous_close).abs(),
            (df["low"].astype(float) - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(true_range, n)



def classic_pivots_from_intraday(
    df: pd.DataFrame, min_session_bars: int = 75
) -> pd.DataFrame:
    """Build prior-session classic pivots, rejecting incomplete source sessions.

    Five-minute NSE regular sessions normally contain 75 candles from 09:15
    through 15:25 IST. If a session is incomplete, the next session's pivot is
    left NaN rather than silently deriving levels from a partial H/L/C.
    Pass a smaller threshold only in focused synthetic tests.
    """
    required = {"ts", "session", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pivot input missing columns: {sorted(missing)}")
    if min_session_bars < 1:
        raise ValueError("min_session_bars must be >= 1")

    minutes = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    regular = df.loc[(minutes >= 9 * 60 + 15) & (minutes < 15 * 60 + 30)].copy()
    regular["_minute"] = minutes.loc[regular.index]
    daily = regular.groupby("session", sort=True).agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
        bar_count=("close", "size"),
        first_minute=("_minute", "min"),
        last_minute=("_minute", "max"),
    )
    complete = (
        daily["bar_count"].ge(min_session_bars)
        & daily["first_minute"].eq(9 * 60 + 15)
        & daily["last_minute"].eq(15 * 60 + 25)
    )
    valid_previous = complete.shift(1, fill_value=False)
    daily["prev_high"] = daily["day_high"].shift(1).where(valid_previous)
    daily["prev_low"] = daily["day_low"].shift(1).where(valid_previous)
    daily["prev_close"] = daily["day_close"].shift(1).where(valid_previous)
    daily["pp"] = (daily["prev_high"] + daily["prev_low"] + daily["prev_close"]) / 3.0
    daily["r1"] = 2.0 * daily["pp"] - daily["prev_low"]
    daily["s1"] = 2.0 * daily["pp"] - daily["prev_high"]
    return daily[["pp", "r1", "s1"]]
