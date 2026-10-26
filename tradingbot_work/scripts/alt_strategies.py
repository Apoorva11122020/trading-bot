"""Spot-based alternative intraday strategies (day loop, same exit model as doc guide)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from scripts.classic_pivot_historical_backtest import finish, prepare


def _session_vwap(x: pd.DataFrame) -> pd.Series:
    typical = (x["high"] + x["low"] + x["close"]) / 3.0
    pv = typical * x["volume"].fillna(0)
    return pv.groupby(x["session"], sort=False).cumsum() / (
        x["volume"].fillna(0).groupby(x["session"], sort=False).cumsum().replace(0, np.nan)
    )


def add_opening_range_signals(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    minutes = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    opening = out.loc[(minutes >= 9 * 60 + 15) & (minutes < 9 * 60 + 45)]
    levels = opening.groupby("session", sort=True).agg(
        opening_range_high=("high", "max"),
        opening_range_low=("low", "min"),
        opening_bars=("high", "size"),
    )
    out = out.join(levels, on="session")
    prev_close = out.groupby("session", sort=False)["close"].shift(1)
    after_opening = (minutes >= 9 * 60 + 45) & (minutes < 15 * 60)
    out["long_signal"] = (
        after_opening
        & (out["opening_bars"] == 6)
        & (prev_close <= out["opening_range_high"])
        & (out["close"] > out["opening_range_high"])
    )
    out["short_signal"] = (
        after_opening
        & (out["opening_bars"] == 6)
        & (prev_close >= out["opening_range_low"])
        & (out["close"] < out["opening_range_low"])
    )
    return out


def add_vwap_reversion_signals(x: pd.DataFrame, excursion_atr: float = 1.0) -> pd.DataFrame:
    out = x.copy()
    out["vwap"] = _session_vwap(out)
    prev_close = out.groupby("session", sort=False)["close"].shift(1)
    prev_vwap = out.groupby("session", sort=False)["vwap"].shift(1)
    prev_atr = out.groupby("session", sort=False)["atr14"].shift(1)
    prev_deviation = (prev_close - prev_vwap) / prev_atr
    minutes = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    valid_time = (minutes >= 9 * 60 + 45) & (minutes < 15 * 60)
    out["long_signal"] = (
        valid_time
        & (prev_deviation <= -excursion_atr)
        & (out["close"] > out["vwap"])
    )
    out["short_signal"] = (
        valid_time
        & (prev_deviation >= excursion_atr)
        & (out["close"] < out["vwap"])
    )
    return out


def add_ema_pullback_signals(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    prev_close = out.groupby("session", sort=False)["close"].shift(1)
    prev_ema20 = out.groupby("session", sort=False)["ema20"].shift(1)
    minutes = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    in_window = (minutes >= 9 * 60 + 20) & (minutes < 15 * 60)
    out["long_signal"] = (
        in_window
        & (out["ema20"] > out["ema50"])
        & (prev_close <= prev_ema20)
        & (out["close"] > out["ema20"])
        & (out["rsi14"] > 50)
    )
    out["short_signal"] = (
        in_window
        & (out["ema20"] < out["ema50"])
        & (prev_close >= prev_ema20)
        & (out["close"] < out["ema20"])
        & (out["rsi14"] < 50)
    )
    return out


def backtest_signal_day(
    frame: pd.DataFrame,
    label: str,
    day: date,
    *,
    signal_builder,
    volume_filter: bool = False,
    commission_pct: float = 0.03,
    slippage_points: float = 2.0,
    atr_buffer: float = 0.1,
    rr: float = 1.5,
    already_prepared: bool = False,
    one_trade_per_day: bool = False,
) -> pd.DataFrame:
    base = frame if already_prepared else prepare(frame, volume_filter)
    x = signal_builder(base)
    day_rows = x.index[x.session == day].tolist()
    trades: list[dict] = []
    pos = None
    trade_taken = False

    for idx in day_rows:
        row = x.loc[idx]
        ts = row.ts

        if pos is not None:
            exit_price = None
            reason = None
            if pos["side"] == "LONG":
                if row.low <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop (stop-first)"
                elif row.high >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            else:
                if row.high >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop (stop-first)"
                elif row.low <= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            if exit_price is None and bool(row.flat_time):
                exit_price, reason = float(row.close), "intraday_flat"
            if exit_price is not None:
                trades.append(finish(pos, ts, float(exit_price), reason, commission_pct, slippage_points))
                pos = None

        if pos is None and not (trades and trades[-1].get("exit_ts") == ts):
            if bool(row.long_signal) or bool(row.short_signal):
                if one_trade_per_day and trade_taken:
                    continue
                side = "LONG" if bool(row.long_signal) else "SHORT"
                entry = float(row.close)
                atr_val = row.atr14
                if pd.notna(atr_val):
                    if side == "LONG":
                        stop = float(row.low - atr_val * atr_buffer)
                        risk = entry - stop
                        target = entry + risk * rr
                    else:
                        stop = float(row.high + atr_val * atr_buffer)
                        risk = stop - entry
                        target = entry - risk * rr
                    if risk > 0 and np.isfinite(risk):
                        pos = {
                            "symbol": label,
                            "side": side,
                            "session": row.session,
                            "entry_ts": ts,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "risk_points": risk,
                            "volume_filter": volume_filter,
                        }
                        trade_taken = True

    if pos is not None and day_rows:
        last = x.loc[day_rows[-1]]
        trades.append(finish(pos, last.ts, float(last.close), "period_end", commission_pct, slippage_points))
    return pd.DataFrame(trades)
