"""Four configurable research strategies on underlying 5-minute candles.

S1 mirrors the production EMA/VWAP/RSI/breakout/volume rules.
S2-S4 are separate candidate strategies for objective comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd

from src.indicators.technical import atr, ema, previous_high, previous_low, rsi, volume_sma, vwap

IST = "Asia/Kolkata"
STRATEGY_IDS = ("S1", "S2", "S3", "S4")


@dataclass(frozen=True)
class ResearchSignal:
    timestamp: datetime
    direction: str
    strategy_id: str
    underlying: str
    spot: float
    conditions: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_values = atr(df, period)
    upper = hl2 + multiplier * atr_values
    lower = hl2 - multiplier * atr_values
    direction = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0 or pd.isna(atr_values.iloc[i]):
            direction.iloc[i] = 1.0
            trend.iloc[i] = upper.iloc[i]
            continue
        prev = direction.iloc[i - 1]
        if df["close"].iloc[i] > trend.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif df["close"].iloc[i] < trend.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = prev
        if direction.iloc[i] > 0:
            trend.iloc[i] = max(lower.iloc[i], trend.iloc[i - 1] if prev > 0 else lower.iloc[i])
        else:
            trend.iloc[i] = min(upper.iloc[i], trend.iloc[i - 1] if prev < 0 else upper.iloc[i])
    return direction


def enrich_s1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ind = cfg["indicators"]
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ind["ema_fast"])
    out["ema_slow"] = ema(out["close"], ind["ema_slow"])
    out["rsi"] = rsi(out["close"], ind["rsi_period"])
    out["atr"] = atr(out, ind["atr_period"])
    out["vwap"] = vwap(out)
    out["volume_sma"] = volume_sma(out, ind["volume_sma_period"])
    out["previous_high"] = previous_high(out, ind["breakout_period"])
    out["previous_low"] = previous_low(out, ind["breakout_period"])
    return out


def enrich_s2(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, cfg.get("atr_period", 14))
    start, end = _parse_time(cfg["range_start"]), _parse_time(cfg["range_end"])
    out["orb_high"] = np.nan
    out["orb_low"] = np.nan
    for day, group in out.groupby(out["timestamp"].dt.date):
        mask = (group["timestamp"].dt.time >= start) & (group["timestamp"].dt.time < end)
        if not mask.any():
            continue
        hi = group.loc[mask, "high"].max()
        lo = group.loc[mask, "low"].min()
        out.loc[group.index, "orb_high"] = hi
        out.loc[group.index, "orb_low"] = lo
    return out


def enrich_s3(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg["rsi_period"])
    out["atr"] = atr(out, cfg["atr_period"])
    out["vwap"] = vwap(out)
    out["recent_high"] = previous_high(out, cfg["breakout_period"])
    out["recent_low"] = previous_low(out, cfg["breakout_period"])
    return out


def enrich_s4(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], cfg["ema_fast"])
    out["ema_slow"] = ema(out["close"], cfg["ema_slow"])
    out["atr"] = atr(out, cfg["atr_period"])
    out["vwap"] = vwap(out)
    out["supertrend_dir"] = supertrend(out, cfg["atr_period"], cfg["supertrend_multiplier"])
    return out


def _long_s1(row, cfg) -> dict[str, bool]:
    return {
        "ema_trend": row["ema_fast"] > row["ema_slow"],
        "vwap": row["close"] > row["vwap"],
        "rsi": row["rsi"] >= cfg["long"]["rsi_min"],
        "breakout": row["close"] > row["previous_high"],
        "volume": row["volume"] >= row["volume_sma"] * cfg["long"]["volume_multiplier"],
    }


def _short_s1(row, cfg) -> dict[str, bool]:
    return {
        "ema_trend": row["ema_fast"] < row["ema_slow"],
        "vwap": row["close"] < row["vwap"],
        "rsi": row["rsi"] <= cfg["short"]["rsi_max"],
        "breakout": row["close"] < row["previous_low"],
        "volume": row["volume"] >= row["volume_sma"] * cfg["short"]["volume_multiplier"],
    }


def signals_s1(df: pd.DataFrame, underlying: str, cfg: dict) -> list[ResearchSignal]:
    frame = enrich_s1(df, cfg)
    out = []
    for _, row in frame.iterrows():
        long_c = _long_s1(row, cfg)
        if all(long_c.values()):
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "BUY", "S1", underlying,
                                      float(row["close"]), long_c))
            continue
        short_c = _short_s1(row, cfg)
        if all(short_c.values()):
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "SELL", "S1", underlying,
                                      float(row["close"]), short_c))
    return out


def signals_s2(df: pd.DataFrame, underlying: str, cfg: dict) -> list[ResearchSignal]:
    frame = enrich_s2(df, cfg)
    range_end = _parse_time(cfg["range_end"])
    out = []
    for _, row in frame.iterrows():
        if row["timestamp"].time() <= range_end or pd.isna(row["orb_high"]):
            continue
        if row["close"] > row["orb_high"]:
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "BUY", "S2", underlying,
                                      float(row["close"]), {"orb_breakout": True, "above": float(row["orb_high"])}))
        elif row["close"] < row["orb_low"]:
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "SELL", "S2", underlying,
                                      float(row["close"]), {"orb_breakdown": True, "below": float(row["orb_low"])}))
    return out


def signals_s3(df: pd.DataFrame, underlying: str, cfg: dict) -> list[ResearchSignal]:
    frame = enrich_s3(df, cfg)
    out = []
    for _, row in frame.iterrows():
        long_c = {
            "vwap": row["close"] > row["vwap"],
            "rsi": row["rsi"] >= cfg["long_rsi_min"],
            "breakout": row["close"] > row["recent_high"],
        }
        if all(long_c.values()):
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "BUY", "S3", underlying,
                                      float(row["close"]), long_c))
            continue
        short_c = {
            "vwap": row["close"] < row["vwap"],
            "rsi": row["rsi"] <= cfg["short_rsi_max"],
            "breakdown": row["close"] < row["recent_low"],
        }
        if all(short_c.values()):
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "SELL", "S3", underlying,
                                      float(row["close"]), short_c))
    return out


def signals_s4(df: pd.DataFrame, underlying: str, cfg: dict) -> list[ResearchSignal]:
    frame = enrich_s4(df, cfg)
    out = []
    for _, row in frame.iterrows():
        long_c = {
            "ema_trend": row["ema_fast"] > row["ema_slow"],
            "supertrend": row["supertrend_dir"] > 0,
            "vwap": row["close"] > row["vwap"],
        }
        if all(long_c.values()):
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "BUY", "S4", underlying,
                                      float(row["close"]), long_c))
            continue
        short_c = {
            "ema_trend": row["ema_fast"] < row["ema_slow"],
            "supertrend": row["supertrend_dir"] < 0,
            "vwap": row["close"] < row["vwap"],
        }
        if all(short_c.values()):
            out.append(ResearchSignal(_ts(row["timestamp"]).to_pydatetime(), "SELL", "S4", underlying,
                                      float(row["close"]), short_c))
    return out


_GENERATORS = {
    "S1": signals_s1,
    "S2": signals_s2,
    "S3": signals_s3,
    "S4": signals_s4,
}


def strategy_catalog(config) -> list[dict[str, Any]]:
    rows = []
    for sid, cfg in config.strategies.items():
        rows.append({"id": sid, "name": cfg.get("name", sid), "enabled": bool(cfg.get("enabled", True))})
    return rows


def generate_signals(strategy_id: str, candles: pd.DataFrame, underlying: str, config) -> list[ResearchSignal]:
    sid = strategy_id.upper()
    if sid not in _GENERATORS:
        raise ValueError(f"Unknown strategy {strategy_id}")
    cfg = config.strategies[sid]
    if not cfg.get("enabled", True):
        return []
    frame = candles.sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        return []
    if "symbol" not in frame:
        frame["symbol"] = underlying
    return _GENERATORS[sid](frame, underlying.upper(), cfg)
