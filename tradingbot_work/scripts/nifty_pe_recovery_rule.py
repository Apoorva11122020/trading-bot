"""NIFTY PE recovery rule: fade a bounce after a deep morning selloff.

Trigger (once per session, NIFTY spot index):
  1. Session low was at least `drop_points` below the morning open.
  2. Price crosses up through `recovery_down_points` below the morning open
     (e.g. open - 200 seen, then cross to open - 70).

Action: BUY PE, fixed lots, stop-loss in option premium points (alert metadata).
Paper mode logs the signal; it does not place broker orders.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

DROP_POINTS: float = 200.0
RECOVERY_DOWN_POINTS: float = 70.0
PE_LOTS: int = 2
PE_STOP_PREMIUM_POINTS: float = 20.0
NIFTY_LOT_SIZE: int = 25


@dataclass
class PeRecoverySessionState:
    morning_open: float | None = None
    session_low: float | None = None
    saw_deep_drop: bool = False
    signal_fired: bool = False


@dataclass(frozen=True)
class PeRecoverySignal:
    morning_open: float
    session_low: float
    recovery_level: float
    spot_ref: float
    lots: int
    stop_premium_points: float
    lot_size: int


def recovery_level(morning_open: float, recovery_down_points: float = RECOVERY_DOWN_POINTS) -> float:
    return morning_open - recovery_down_points


def update_session_state(
    state: PeRecoverySessionState,
    *,
    open_price: float,
    low: float,
    drop_points: float = DROP_POINTS,
) -> PeRecoverySessionState:
    if state.morning_open is None:
        state.morning_open = float(open_price)
        state.session_low = float(low)
    else:
        state.session_low = min(float(state.session_low or low), float(low))
    if state.session_low <= state.morning_open - drop_points:
        state.saw_deep_drop = True
    return state


def evaluate_pe_recovery(
    state: PeRecoverySessionState,
    *,
    prev_close: float,
    close: float,
    drop_points: float = DROP_POINTS,
    recovery_down_points: float = RECOVERY_DOWN_POINTS,
    lots: int = PE_LOTS,
    stop_premium_points: float = PE_STOP_PREMIUM_POINTS,
) -> PeRecoverySignal | None:
    if state.signal_fired or not state.saw_deep_drop or state.morning_open is None or state.session_low is None:
        return None
    level = recovery_level(state.morning_open, recovery_down_points)
    crossed_up = prev_close < level <= close
    if not crossed_up:
        return None
    state.signal_fired = True
    return PeRecoverySignal(
        morning_open=state.morning_open,
        session_low=state.session_low,
        recovery_level=level,
        spot_ref=float(close),
        lots=lots,
        stop_premium_points=stop_premium_points,
        lot_size=NIFTY_LOT_SIZE,
    )


def format_alert(signal: PeRecoverySignal) -> str:
    return (
        f"BUY PE | NIFTY | lots={signal.lots} | spot_ref={signal.spot_ref:.2f} | "
        f"morning_open={signal.morning_open:.2f} | session_low={signal.session_low:.2f} | "
        f"recovery_level={signal.recovery_level:.2f} | "
        f"PE_SL_premium_pts={signal.stop_premium_points:.0f} | "
        f"lot_size={signal.lot_size} | resolving ATM PE via instrument master"
    )
