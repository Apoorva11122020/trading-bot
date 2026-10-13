import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask(avg_loss.eq(0) & avg_gain.gt(0), 100)
    result = result.mask(avg_gain.eq(0) & avg_loss.gt(0), 0)
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_volume = df["volume"].cumsum()
    cumulative_value = (typical_price * df["volume"]).cumsum()
    return cumulative_value / cumulative_volume.replace(0, np.nan)


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(period, min_periods=period).mean()


def previous_high(df: pd.DataFrame, period: int) -> pd.Series:
    return df["high"].shift(1).rolling(period, min_periods=period).max()


def previous_low(df: pd.DataFrame, period: int) -> pd.Series:
    return df["low"].shift(1).rolling(period, min_periods=period).min()


def enrich_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    result = df.copy()
    result["ema_fast"] = ema(result["close"], cfg.ema_fast)
    result["ema_slow"] = ema(result["close"], cfg.ema_slow)
    result["rsi"] = rsi(result["close"], cfg.rsi_period)
    result["atr"] = atr(result, cfg.atr_period)
    result["vwap"] = vwap(result)
    result["volume_sma"] = volume_sma(result, cfg.volume_sma_period)
    result["previous_high"] = previous_high(result, cfg.breakout_period)
    result["previous_low"] = previous_low(result, cfg.breakout_period)
    return result
