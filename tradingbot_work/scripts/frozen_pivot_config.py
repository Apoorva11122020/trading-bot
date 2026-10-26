"""Frozen classic pivot production rules (paper/research only).

Active default (v5_short_only): both NIFTY and BANKNIFTY short-only (no morning longs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import pandas as pd

Side = Literal["LONG", "SHORT"]

LONG_ENTRY_WINDOWS_MINUTES: Final[dict[str, tuple[int, int]]] = {
    "NIFTY": (9 * 60 + 30, 11 * 60),
    "BANKNIFTY": (9 * 60 + 30, 11 * 60),
}
SHORT_ENTRY_WINDOWS_MINUTES: Final[dict[str, tuple[int, int]]] = {
    "NIFTY": (11 * 60, 13 * 60),
    "BANKNIFTY": (13 * 60, 15 * 60),
}
MAX_RISK_POINTS: Final[dict[str, float]] = {
    "NIFTY": 40.0,
    "BANKNIFTY": 100.0,
}

RR: Final[float] = 1.5
ATR_BUFFER: Final[float] = 0.1
VOLUME_MULTIPLIER: Final[float] = 1.2
BREAKEVEN_TRIGGER_R: Final[float] = 0.75
SIGNAL_SESSION_END_MINUTE: Final[int] = 15 * 60
FLAT_SESSION_MINUTE: Final[int] = 15 * 60 + 15
V3_HYBRID_VERSION: Final[str] = "v3_hybrid"
V4_SPLIT_VERSION: Final[str] = "v4_split"
V5_SHORT_ONLY_VERSION: Final[str] = "v5_short_only"
CONFIG_VERSION: Final[str] = V5_SHORT_ONLY_VERSION


@dataclass(frozen=True)
class FrozenPivotConfig:
    allow_long: bool = True
    allow_short: bool = True
    require_volume: bool = True
    require_vwap_below_for_short: bool = True
    require_vwap_above_for_long: bool = True
    one_trade_per_session: bool = False
    separate_long_short_quota: bool = True
    volume_multiplier: float = VOLUME_MULTIPLIER
    rr: float = RR
    atr_buffer: float = ATR_BUFFER
    breakeven_trigger_r: float = BREAKEVEN_TRIGGER_R
    max_risk_points: dict[str, float] = field(default_factory=lambda: dict(MAX_RISK_POINTS))


V3_HYBRID_CONFIG = FrozenPivotConfig()
DEFAULT_CONFIG = V3_HYBRID_CONFIG

V2_CONFIG = FrozenPivotConfig(one_trade_per_session=True, separate_long_short_quota=False)

V1_SHORT_ONLY_CONFIG = FrozenPivotConfig(
    allow_long=False,
    allow_short=True,
    one_trade_per_session=True,
    separate_long_short_quota=False,
)


def _symbol_key(symbol: str) -> str:
    key = symbol.upper()
    if key in {"NIFTY", "NIFTY_50"}:
        return "NIFTY"
    if key in {"BANKNIFTY", "BANK_NIFTY"}:
        return "BANKNIFTY"
    raise KeyError(f"Unknown symbol for frozen pivot config: {symbol}")


def entry_window_minutes(symbol: str, side: Side) -> tuple[int, int]:
    key = _symbol_key(symbol)
    windows = LONG_ENTRY_WINDOWS_MINUTES if side == "LONG" else SHORT_ENTRY_WINDOWS_MINUTES
    return windows[key]


def max_risk_for(symbol: str, config: FrozenPivotConfig = DEFAULT_CONFIG) -> float:
    return float(config.max_risk_points[_symbol_key(symbol)])


def add_session_vwap_futures(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    typical = (x.high_fut + x.low_fut + x.close_fut) / 3.0
    cumulative_pv = (typical * x.volume_fut).groupby(x.session, sort=False).cumsum()
    cumulative_volume = x.volume_fut.groupby(x.session, sort=False).cumsum()
    x["session_vwap_fut"] = cumulative_pv / cumulative_volume.where(cumulative_volume > 0)
    return x


def add_session_typical_avg_spot(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    typical = (x["high"] + x["low"] + x["close"]) / 3.0
    x["session_typical_avg"] = typical.groupby(x["session"], sort=False).expanding().mean().reset_index(
        level=0, drop=True
    )
    return x


def apply_side_entry_time_filter(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    x = frame.copy()
    signal_minute = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    entry_minute = signal_minute + 5
    long_start, long_end = entry_window_minutes(symbol, "LONG")
    short_start, short_end = entry_window_minutes(symbol, "SHORT")
    x.loc[~(entry_minute.ge(long_start) & entry_minute.lt(long_end)), "long_signal"] = False
    x.loc[~(entry_minute.ge(short_start) & entry_minute.lt(short_end)), "short_signal"] = False
    return x


def apply_frozen_filters(
    frame: pd.DataFrame,
    symbol: str,
    config: FrozenPivotConfig = DEFAULT_CONFIG,
    *,
    execution_basis: str = "futures",
) -> pd.DataFrame:
    if execution_basis not in {"futures", "spot"}:
        raise ValueError("execution_basis must be 'futures' or 'spot'")

    x = frame.copy()
    if not config.allow_long:
        x["long_signal"] = False
    if not config.allow_short:
        x["short_signal"] = False

    if execution_basis == "futures":
        x = add_session_vwap_futures(x)
        if config.require_vwap_below_for_short:
            x["short_signal"] &= x.close_fut < x.session_vwap_fut
        if config.require_vwap_above_for_long:
            x["long_signal"] &= x.close_fut > x.session_vwap_fut
    else:
        x = add_session_typical_avg_spot(x)
        if config.require_vwap_below_for_short:
            x["short_signal"] &= x["close"] < x["session_typical_avg"]
        if config.require_vwap_above_for_long:
            x["long_signal"] &= x["close"] > x["session_typical_avg"]

    return apply_side_entry_time_filter(x, symbol)


def spot_paper_entry_window_ok(symbol: str, signal_ts: pd.Timestamp, side: Side) -> bool:
    start, end = entry_window_minutes(symbol, side)
    entry_minute = signal_ts.hour * 60 + signal_ts.minute + 5
    return start <= entry_minute < end


def side_quota_available(
    cfg: FrozenPivotConfig,
    side: Side,
    *,
    long_taken: bool,
    short_taken: bool,
    trade_taken: bool,
) -> bool:
    if side == "LONG":
        if not cfg.allow_long:
            return False
        if cfg.separate_long_short_quota:
            return not long_taken
        if cfg.one_trade_per_session:
            return not trade_taken
        return True
    if not cfg.allow_short:
        return False
    if cfg.separate_long_short_quota:
        return not short_taken
    if cfg.one_trade_per_session:
        return not trade_taken
    return True


def mark_side_taken(
    cfg: FrozenPivotConfig,
    side: Side,
    *,
    long_taken: bool,
    short_taken: bool,
    trade_taken: bool,
) -> tuple[bool, bool, bool]:
    if cfg.separate_long_short_quota:
        if side == "LONG":
            return True, short_taken, trade_taken
        return long_taken, True, trade_taken
    return long_taken, short_taken, True


def config_for_symbol(symbol: str, preset: str = CONFIG_VERSION) -> FrozenPivotConfig:
    """Return frozen rules for a symbol under a named preset.

    v5_short_only (default): both symbols short-only (no morning longs).
    v4_split: NIFTY v1 short-only; BANKNIFTY v3 hybrid.
    v3_hybrid: both symbols morning long + afternoon short.
    """
    key = _symbol_key(symbol)
    if preset in {V5_SHORT_ONLY_VERSION, "v5_short_only"}:
        return V1_SHORT_ONLY_CONFIG
    if preset in {V4_SPLIT_VERSION, "v4_split"}:
        return V1_SHORT_ONLY_CONFIG if key == "NIFTY" else V3_HYBRID_CONFIG
    if preset in {V3_HYBRID_VERSION, "v3_hybrid"}:
        return V3_HYBRID_CONFIG
    if preset in {"v1_short_only", "frozen_v1"}:
        return V1_SHORT_ONLY_CONFIG
    if preset == "v2":
        return V2_CONFIG
    raise ValueError(f"Unknown frozen pivot preset: {preset}")
