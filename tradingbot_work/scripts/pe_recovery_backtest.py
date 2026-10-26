"""Historical backtest helpers for the NIFTY PE recovery add-on."""
from __future__ import annotations

from datetime import date, datetime, time as daytime
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.nifty_pe_contract_resolver import resolve_atm_pe
from scripts.frozen_pivot_config import FLAT_SESSION_MINUTE
from scripts.nifty_pe_recovery_rule import (
    PE_LOTS,
    PE_STOP_PREMIUM_POINTS,
    PeRecoverySessionState,
    evaluate_pe_recovery,
    update_session_state,
)
from src.historical_data.models import HistoricalCandle
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.options.instrument_resolver import OptionInstrumentResolver
from src.options.models import OptionResolutionError
from src.options.strategy import round_to_tick

IST = ZoneInfo("Asia/Kolkata")


def _premium_at_or_after(candles: list[HistoricalCandle], entry_ts: datetime) -> float | None:
    for candle in candles:
        ts = candle.timestamp.astimezone(IST)
        if ts >= entry_ts:
            premium = float(candle.close)
            if premium > 0:
                return premium
    return None


def option_candles_for_day(
    client: UpstoxHistoricalClient,
    instrument_key: str,
    day: date,
) -> list[HistoricalCandle]:
    try:
        return client.candles(instrument_key, day, day, unit="minutes", interval=1)
    except Exception:
        return []


def simulate_pe_exit(
    candles: list[HistoricalCandle],
    *,
    entry_ts: datetime,
    entry_premium: float,
    stop_premium: float,
    flat_minute: int = 15 * 60 + 15,
) -> tuple[float, datetime, str] | None:
    if not candles:
        return None
    entry_ts = entry_ts.astimezone(IST)
    started = False
    last_ts = entry_ts
    last_close = entry_premium
    for candle in candles:
        ts = candle.timestamp.astimezone(IST)
        if ts.date() != entry_ts.date() or ts < entry_ts:
            continue
        started = True
        last_ts, last_close = ts, float(candle.close)
        minute = ts.hour * 60 + ts.minute
        if float(candle.low) <= stop_premium:
            return stop_premium, ts, "pe_stop (premium)"
        if minute >= flat_minute:
            return float(candle.close), ts, "pe_session_flat"
    if started:
        return last_close, last_ts, "pe_session_flat (last_bar)"
    return None


def backtest_pe_day(
    spot_frame: pd.DataFrame,
    day: date,
    client: UpstoxHistoricalClient,
    resolver: OptionInstrumentResolver,
) -> dict | None:
    """Return one PE paper trade dict for the session, or None."""
    day_rows = spot_frame.loc[spot_frame["session"] == day].sort_values("ts")
    if len(day_rows) < 2:
        return None

    pe_state = PeRecoverySessionState()
    morning_open = float(day_rows.iloc[0]["open"])
    session_low = float(day_rows["low"].min())

    for idx in range(1, len(day_rows)):
        prev_row = day_rows.iloc[idx - 1]
        row = day_rows.iloc[idx]
        update_session_state(
            pe_state,
            open_price=morning_open,
            low=min(session_low, float(day_rows.iloc[: idx + 1]["low"].min())),
        )
        session_low = float(pe_state.session_low or session_low)
        signal = evaluate_pe_recovery(
            pe_state,
            prev_close=float(prev_row["close"]),
            close=float(row["close"]),
        )
        if signal is None:
            continue

        entry_ts = pd.Timestamp(row["ts"]).to_pydatetime()
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=IST)
        else:
            entry_ts = entry_ts.astimezone(IST)

        try:
            contract, expiry, strike = resolve_atm_pe(
                signal.spot_ref, resolver, as_of=day,
            )
            opt_candles = option_candles_for_day(client, contract.instrument_key, day)
            raw_premium = _premium_at_or_after(opt_candles, entry_ts)
            if raw_premium is None:
                raise OptionResolutionError(
                    "OPTION_PREMIUM_UNAVAILABLE",
                    f"No option candles for {contract.instrument_key} on {day}",
                )
            tick = contract.tick_size or 0.05
            entry_premium = round_to_tick(raw_premium, tick, "nearest")
            stop_premium = round_to_tick(entry_premium - PE_STOP_PREMIUM_POINTS, tick, "down")
            if stop_premium <= 0 or stop_premium >= entry_premium:
                raise OptionResolutionError("OPTION_STOP_INVALID", "Invalid PE stop for historical entry")

            exit_info = simulate_pe_exit(
                opt_candles,
                entry_ts=entry_ts,
                entry_premium=entry_premium,
                stop_premium=stop_premium,
                flat_minute=FLAT_SESSION_MINUTE,
            )
            if exit_info is None:
                raise OptionResolutionError("OPTION_EXIT_UNAVAILABLE", "Could not simulate PE exit")

            exit_premium, exit_ts, exit_reason = exit_info
            quantity = int(PE_LOTS) * int(contract.lot_size)
            gross_inr = (exit_premium - entry_premium) * quantity
            return {
                "session_date": str(day),
                "entry_ts": entry_ts.isoformat(),
                "exit_ts": exit_ts.isoformat(),
                "symbol": "NIFTY_50",
                "side": "PE",
                "morning_open": round(signal.morning_open, 2),
                "session_low": round(signal.session_low, 2),
                "recovery_level": round(signal.recovery_level, 2),
                "spot_ref": round(signal.spot_ref, 2),
                "trading_symbol": contract.trading_symbol,
                "instrument_key": contract.instrument_key,
                "strike": strike,
                "expiry": expiry.isoformat(),
                "lots": PE_LOTS,
                "quantity": quantity,
                "entry_premium": entry_premium,
                "stop_premium": stop_premium,
                "exit_premium": round(exit_premium, 2),
                "exit_reason": exit_reason,
                "gross_inr": round(gross_inr, 2),
                "net_inr": round(gross_inr, 2),
                "win": gross_inr > 0,
            }
        except OptionResolutionError as exc:
            return {
                "session_date": str(day),
                "entry_ts": entry_ts.isoformat(),
                "exit_ts": "",
                "symbol": "NIFTY_50",
                "side": "PE",
                "morning_open": round(signal.morning_open, 2),
                "session_low": round(signal.session_low, 2),
                "recovery_level": round(signal.recovery_level, 2),
                "spot_ref": round(signal.spot_ref, 2),
                "trading_symbol": "",
                "instrument_key": "",
                "strike": "",
                "expiry": "",
                "lots": PE_LOTS,
                "quantity": 0,
                "entry_premium": "",
                "stop_premium": "",
                "exit_premium": "",
                "exit_reason": f"unresolved:{exc.code}",
                "gross_inr": 0.0,
                "net_inr": 0.0,
                "win": False,
            }
    return None


def scan_pe_signals(spot_frame: pd.DataFrame, day: date) -> bool:
    """True if the PE recovery rule would arm or fire on this session (spot only)."""
    day_rows = spot_frame.loc[spot_frame["session"] == day].sort_values("ts")
    if len(day_rows) < 2:
        return False
    pe_state = PeRecoverySessionState()
    morning_open = float(day_rows.iloc[0]["open"])
    for idx in range(1, len(day_rows)):
        prev_row = day_rows.iloc[idx - 1]
        row = day_rows.iloc[idx]
        update_session_state(
            pe_state,
            open_price=morning_open,
            low=float(day_rows.iloc[: idx + 1]["low"].min()),
        )
        if pe_state.saw_deep_drop:
            return True
        if evaluate_pe_recovery(
            pe_state,
            prev_close=float(prev_row["close"]),
            close=float(row["close"]),
        ):
            return True
    return pe_state.saw_deep_drop
