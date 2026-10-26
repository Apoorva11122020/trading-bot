"""Resolve ATM NIFTY PE from the Upstox instrument master and fetch live premium."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from scripts.nifty_pe_recovery_rule import PeRecoverySignal
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.options.instrument_resolver import OptionInstrumentResolver, UpstoxInstrumentMasterSource
from src.options.models import OptionContract, OptionResolutionError, OptionType
from src.options.selection import atm_strike, select_expiry
from src.options.strategy import round_to_tick


@dataclass(frozen=True)
class ResolvedPeOrder:
    signal: PeRecoverySignal
    contract: OptionContract
    expiry: date
    strike: float
    entry_premium: float
    stop_premium: float
    quantity: int

    @property
    def lots(self) -> int:
        return self.signal.lots


def build_nifty_pe_resolver() -> OptionInstrumentResolver:
    return OptionInstrumentResolver(UpstoxInstrumentMasterSource())


def resolve_atm_pe(
    spot: float,
    resolver: OptionInstrumentResolver,
    *,
    underlying: str = "NIFTY",
    expiry_mode: str = "nearest",
    as_of: date | None = None,
) -> tuple[OptionContract, date, float]:
    if as_of is not None:
        resolver.today = lambda: as_of
    expiries = resolver.expiries(underlying)
    expiry = select_expiry(expiries, expiry_mode)
    strikes = resolver.strikes(underlying, expiry, OptionType.PE)
    strike = atm_strike(strikes, float(spot))
    contract = resolver.resolve(underlying, expiry, strike, OptionType.PE)
    return contract, expiry, strike


def fetch_option_premium(client: UpstoxHistoricalClient, instrument_key: str) -> float:
    candles = client.intraday_candles(instrument_key, unit="minutes", interval=1)
    if not candles:
        raise OptionResolutionError("OPTION_PREMIUM_UNAVAILABLE", f"No intraday candles for {instrument_key}")
    premium = float(candles[-1].close)
    if premium <= 0:
        raise OptionResolutionError("OPTION_PREMIUM_INVALID", f"Invalid option premium for {instrument_key}")
    return premium


def build_pe_recovery_order(
    signal: PeRecoverySignal,
    resolver: OptionInstrumentResolver,
    client: UpstoxHistoricalClient,
    *,
    underlying: str = "NIFTY",
    expiry_mode: str = "nearest",
    as_of: date | None = None,
) -> ResolvedPeOrder:
    contract, expiry, strike = resolve_atm_pe(
        signal.spot_ref, resolver, underlying=underlying, expiry_mode=expiry_mode, as_of=as_of,
    )
    premium = fetch_option_premium(client, contract.instrument_key)
    tick = contract.tick_size or 0.05
    entry_premium = round_to_tick(premium, tick, "nearest")
    stop_premium = round_to_tick(entry_premium - signal.stop_premium_points, tick, "down")
    if stop_premium <= 0 or stop_premium >= entry_premium:
        raise OptionResolutionError(
            "OPTION_STOP_INVALID",
            f"PE stop {stop_premium} invalid for entry {entry_premium}",
        )
    quantity = int(signal.lots) * int(contract.lot_size)
    return ResolvedPeOrder(
        signal=signal,
        contract=contract,
        expiry=expiry,
        strike=strike,
        entry_premium=entry_premium,
        stop_premium=stop_premium,
        quantity=quantity,
    )


def format_pe_order_alert(order: ResolvedPeOrder) -> str:
    c = order.contract
    return (
        f"BUY PE | {c.underlying} | lots={order.lots} | qty={order.quantity} | "
        f"symbol={c.trading_symbol} | key={c.instrument_key} | "
        f"strike={order.strike:g} | expiry={order.expiry.isoformat()} | "
        f"entry_premium={order.entry_premium:.2f} | stop_premium={order.stop_premium:.2f} | "
        f"spot_ref={order.signal.spot_ref:.2f} | morning_open={order.signal.morning_open:.2f} | "
        f"session_low={order.signal.session_low:.2f} | recovery_level={order.signal.recovery_level:.2f}"
    )
