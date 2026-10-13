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
