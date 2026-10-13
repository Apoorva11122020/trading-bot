"""Option signals on top of the existing StrategyEngine (no duplicate engine).

UNDERLYING mode (default): the equity entry rules run on the underlying;
BUY -> BUY CALL (CE), SELL -> BUY PUT (PE).
OPTION mode: the long entry rules run on the option's own premium candles
and a bullish premium setup buys that contract.

Stops come from premium ATR (default) or underlying ATR x delta; when the
required ATR is unavailable no signal is produced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable
from uuid import uuid4

from src.models.risk import AccountState
from src.models.signals import SignalSide

from .instrument_resolver import OptionInstrumentResolver
from .models import (
    OPTION_STRATEGY_VERSION,
    OptionContract,
    OptionResolutionError,
    OptionSignal,
    OptionType,
)
from .selection import select_expiry, select_strike

SIGNAL_SOURCES = ("UNDERLYING", "OPTION")
OPTION_TYPE_MODES = ("AUTO", "CE", "PE")


class OptionSignalError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def round_to_tick(price: float, tick: float | None, mode: str = "nearest") -> float:
    tick = float(tick or 0.05)
    units = price / tick
    if mode == "down":
        units = math.floor(units + 1e-9)
    elif mode == "up":
        units = math.ceil(units - 1e-9)
    else:
        units = round(units)
    return round(units * tick, 4)


def option_stop_loss(
    entry_premium: float,
    *,
    stop_mode: str,
    multiplier: float,
    premium_atr: float | None = None,
    underlying_atr: float | None = None,
    underlying_delta: float | None = None,
    tick_size: float | None = 0.05,
) -> float:
    """Stop premium for a BUY option position (rounded down to tick)."""
    mode = str(stop_mode).lower()
    if mode == "premium_atr":
        if premium_atr is None or not math.isfinite(premium_atr) or premium_atr <= 0:
            raise OptionSignalError("OPTION_PREMIUM_ATR_UNAVAILABLE", "Option premium ATR is unavailable")
        distance = premium_atr * multiplier
    elif mode == "underlying_atr":
        if underlying_atr is None or not math.isfinite(underlying_atr) or underlying_atr <= 0:
            raise OptionSignalError("UNDERLYING_ATR_UNAVAILABLE", "Underlying ATR is unavailable")
        if not underlying_delta or underlying_delta <= 0:
            raise OptionSignalError("UNDERLYING_DELTA_REQUIRED", "underlying_delta must be configured")
        distance = underlying_atr * multiplier * underlying_delta
    else:
        raise OptionSignalError("INVALID_STOP_MODE", f"Unknown option stop mode {stop_mode!r}")
    stop = round_to_tick(entry_premium - distance, tick_size, "down")
    if stop <= 0 or stop >= entry_premium:
        raise OptionSignalError(
            "OPTION_STOP_INVALID",
            f"Stop {stop:g} is not a valid premium below entry {entry_premium:g}")
    return stop


def option_target(entry_premium: float, stop_premium: float, target_r: float, tick_size: float | None = 0.05) -> float:
    risk = entry_premium - stop_premium
    if risk <= 0:
        raise OptionSignalError("OPTION_STOP_INVALID", "Stop must be below entry")
    return round_to_tick(entry_premium + risk * target_r, tick_size, "nearest")


def option_exit_config(config):
    """StrategyConfig copy whose exit section carries the option breakeven,
    target and trailing settings. Equity config is not modified."""
    opt = config.options.risk
    exit_cfg = config.exit
    return config.model_copy(update={
        "exit": exit_cfg.model_copy(update={
            "stop_loss": exit_cfg.stop_loss.model_copy(update={"multiplier": opt.stop_atr_multiplier}),
            "target": exit_cfg.target.model_copy(update={"risk_reward": opt.target_r}),
            "breakeven": exit_cfg.breakeven.model_copy(update={"activation_r": opt.breakeven_r}),
            "trailing_stop": exit_cfg.trailing_stop.model_copy(update={"atr_multiplier": opt.trailing_atr_multiplier}),
        })
    })


@dataclass
class OptionDecision:
    """Direction decided by the strategy (before contract resolution)."""

    source: str
    side: SignalSide | None
    option_type: OptionType | None
    conditions: dict[str, bool]
    timestamp: datetime | None
    spot: float | None
    underlying_atr: float | None
    reason: str

    @property
    def tradeable(self) -> bool:
        return self.side == SignalSide.BUY and self.option_type is not None


@dataclass
class ResolvedOption:
    expiry: date
    strike: float
    contract: OptionContract
    expiries: list[date] = field(default_factory=list)
    strikes: list[float] = field(default_factory=list)


def resolve_option_contract(
    resolver: OptionInstrumentResolver,
    *,
    underlying: str,
    option_type: OptionType | str,
    spot: float | None,
    expiry_mode: str = "nearest",
    expiry_date: str | date | None = None,
    strike_mode: str = "ATM",
    strike_offset: int = 0,
    exact_strike: float | None = None,
) -> ResolvedOption:
    kind = resolver.option_type(option_type)
    expiries = resolver.expiries(underlying)
    expiry = select_expiry(expiries, expiry_mode, expiry_date)
    strikes = resolver.strikes(underlying, expiry, kind)
    strike = select_strike(
        strikes, option_type=kind, mode=strike_mode, spot=spot, offset=strike_offset, exact=exact_strike)
    contract = resolver.resolve(underlying, expiry, strike, kind)
    return ResolvedOption(expiry, strike, contract, expiries, strikes)


def _conditions(values: dict) -> dict[str, bool]:
    return {name: bool(value) for name, value in (values or {}).items()}


class OptionSignalEngine:
    def __init__(self, config, strategy_engine, risk_manager, session_manager) -> None:
        self.config = config
        self.options = config.options
        self.strategy_engine = strategy_engine
        self.risk_manager = risk_manager
        self.session_manager = session_manager

    @property
    def strategy_version(self) -> str:
        return f"{self.config.name} | {OPTION_STRATEGY_VERSION}"

    def _last_row(self, candles):
        frame = self.strategy_engine.indicator_frame(candles)
        if frame.empty or len(frame) < 2:
            return None
        return frame.iloc[-1]

    @staticmethod
    def _ts(row) -> datetime | None:
        ts = row.get("timestamp") if hasattr(row, "get") else None
        return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts

    def underlying_decision(self, candles, option_type_mode: str = "AUTO") -> OptionDecision:
        mode = str(option_type_mode or "AUTO").upper()
        if mode not in OPTION_TYPE_MODES:
            raise OptionSignalError("OPTION_TYPE_NOT_FOUND", f"Option type must be AUTO, CE or PE, got {mode!r}")
        row = self._last_row(candles)
        if row is None:
            return OptionDecision("UNDERLYING", None, None, {}, None, None, None, "INSUFFICIENT_UNDERLYING_DATA")
        ts = self._ts(row)
        spot = float(row["close"])
        atr = float(row["atr"]) if row.get("atr") == row.get("atr") else None
        if ts is not None and not self.session_manager.is_entry_allowed(ts):
            return OptionDecision("UNDERLYING", None, None, {}, ts, spot, atr, "ENTRY_OUTSIDE_TRADING_SESSION")

        side, conditions = self.strategy_engine.entry_direction(row)
        if side is None:
            return OptionDecision("UNDERLYING", None, None, {}, ts, spot, atr, "ENTRY_CONDITIONS_NOT_MET")
        kind = OptionType.CE if side == SignalSide.BUY else OptionType.PE
        if mode in ("CE", "PE") and OptionType(mode) != kind:
            return OptionDecision(
                "UNDERLYING", None, None, _conditions(conditions), ts, spot, atr,
                f"UNDERLYING_{side.value}_SIGNAL_CONFLICTS_WITH_{mode}")
        return OptionDecision(
            "UNDERLYING", SignalSide.BUY, kind, _conditions(conditions), ts, spot, atr,
            "UNDERLYING_BULLISH_BUY_CALL" if kind == OptionType.CE else "UNDERLYING_BEARISH_BUY_PUT")

    def option_decision(self, premium_candles, option_type: str) -> OptionDecision:
        mode = str(option_type or "").upper()
        if mode not in ("CE", "PE"):
            raise OptionSignalError(
                "OPTION_TYPE_REQUIRED", "Option-candle signals need an explicit CE or PE option type")
        row = self._last_row(premium_candles)
        if row is None:
            return OptionDecision("OPTION", None, None, {}, None, None, None, "INSUFFICIENT_OPTION_DATA")
        ts = self._ts(row)
        if ts is not None and not self.session_manager.is_entry_allowed(ts):
            return OptionDecision("OPTION", None, None, {}, ts, None, None, "ENTRY_OUTSIDE_TRADING_SESSION")
        long_conditions = self.strategy_engine.entry_conditions(row)["long"]
        if long_conditions is None or not all(long_conditions.values()):
            return OptionDecision(
                "OPTION", None, None, _conditions(long_conditions or {}), ts, None, None,
                "OPTION_ENTRY_CONDITIONS_NOT_MET")
        return OptionDecision(
            "OPTION", SignalSide.BUY, OptionType(mode), _conditions(long_conditions), ts, None, None,
            "OPTION_PREMIUM_BREAKOUT_BUY")

    def build_signal(
        self,
        *,
        contract: OptionContract,
        decision: OptionDecision,
        premium_candles,
        account: AccountState,
        spot: float | None = None,
        details: dict[str, Any] | None = None,
        open_option_positions: int = 0,
    ) -> OptionSignal:
        if not decision.tradeable:
            raise OptionSignalError("NO_SIGNAL", decision.reason)
        if contract.option_type != decision.option_type:
            raise OptionSignalError("OPTION_TYPE_MISMATCH", "Resolved contract does not match the signal option type")
        risk_cfg = self.options.risk

        frame = self.strategy_engine.indicator_frame(premium_candles) if premium_candles is not None else None
        if frame is None or frame.empty:
            raise OptionSignalError("OPTION_PREMIUM_DATA_UNAVAILABLE", "Option premium candles are unavailable")
        last = frame.iloc[-1]
        entry = round_to_tick(float(last["close"]), contract.tick_size, "nearest")
        if entry <= 0:
            raise OptionSignalError("OPTION_PREMIUM_DATA_UNAVAILABLE", "Option premium must be positive")
        premium_atr = float(last["atr"]) if last.get("atr") == last.get("atr") else None

        stop = option_stop_loss(
            entry,
            stop_mode=risk_cfg.stop_mode,
            multiplier=risk_cfg.stop_atr_multiplier,
            premium_atr=premium_atr,
            underlying_atr=decision.underlying_atr,
            underlying_delta=risk_cfg.underlying_delta,
            tick_size=contract.tick_size,
        )
        target = option_target(entry, stop, risk_cfg.target_r, contract.tick_size)

        sizing = self.risk_manager.calculate_option_quantity(
            account=account,
            entry_premium=entry,
            stop_premium=stop,
            lot_size=contract.lot_size,
            freeze_quantity=contract.freeze_quantity,
            max_premium_exposure_pct=risk_cfg.max_premium_exposure_pct,
        )
        timestamp = decision.timestamp or datetime.now()
        check = self.risk_manager.validate_option_entry(
            account=account,
            entry_premium=entry,
            stop_premium=stop,
            target_premium=target,
            quantity=sizing.quantity,
            lot_size=contract.lot_size,
            session_allowed=self.session_manager.is_entry_allowed(timestamp),
            freeze_quantity=contract.freeze_quantity,
            max_premium_exposure_pct=risk_cfg.max_premium_exposure_pct,
            open_option_positions=open_option_positions,
            max_option_positions=risk_cfg.max_option_positions,
        )
        return OptionSignal(
            signal_id=str(uuid4()),
            timestamp=timestamp,
            market_type=contract.market_type,
            underlying=contract.underlying,
            side=SignalSide.BUY,
            option_type=contract.option_type,
            strike=contract.strike,
            expiry=contract.expiry,
            instrument_key=contract.instrument_key,
            trading_symbol=contract.trading_symbol,
            lot_size=contract.lot_size,
            lots=sizing.lots,
            quantity=sizing.quantity,
            entry_price=entry,
            stop_loss=stop,
            target=target,
            risk_per_unit=sizing.risk_per_unit,
            max_loss=sizing.max_loss,
            premium_exposure=sizing.premium_exposure,
            conditions=dict(decision.conditions),
            details={
                "signal_source": decision.source,
                "decision_reason": decision.reason,
                "spot": spot if spot is not None else decision.spot,
                "premium_atr": premium_atr,
                "underlying_atr": decision.underlying_atr,
                "stop_mode": risk_cfg.stop_mode,
                "target_r": risk_cfg.target_r,
                "sizing": sizing.model_dump(),
                "risk": check.model_dump(),
                **(details or {}),
            },
            reason="ALL_ENTRY_CONDITIONS_MET" if check.allowed else "ENTRY_BLOCKED_BY_RISK",
            risk_status="ALLOWED" if check.allowed else "BLOCKED",
            blocked=not check.allowed,
            block_reason=";".join(check.reasons) or None,
            strategy_version=self.strategy_version,
        )

    def generate(
        self,
        *,
        resolver: OptionInstrumentResolver,
        underlying: str,
        account: AccountState,
        underlying_candles=None,
        premium_candles_for: Callable[[OptionContract], Any] | None = None,
        signal_source: str = "UNDERLYING",
        option_type: str = "AUTO",
        expiry_mode: str = "nearest",
        expiry_date: str | date | None = None,
        strike_mode: str = "ATM",
        strike_offset: int = 0,
        exact_strike: float | None = None,
        spot: float | None = None,
    ) -> tuple[OptionDecision, ResolvedOption | None, OptionSignal | None]:
        """Whole flow: decision -> expiry -> strike -> contract -> signal."""
        source = str(signal_source).upper()
        if source not in SIGNAL_SOURCES:
            raise OptionSignalError("INVALID_SIGNAL_SOURCE", f"Signal source must be one of {SIGNAL_SOURCES}")
        premium_for = premium_candles_for or (lambda contract: None)

        if source == "UNDERLYING":
            decision = self.underlying_decision(underlying_candles, option_type)
            if not decision.tradeable:
                return decision, None, None
            reference = spot if spot is not None else decision.spot
            resolved = resolve_option_contract(
                resolver, underlying=underlying, option_type=decision.option_type, spot=reference,
                expiry_mode=expiry_mode, expiry_date=expiry_date, strike_mode=strike_mode,
                strike_offset=strike_offset, exact_strike=exact_strike)
            signal = self.build_signal(
                contract=resolved.contract, decision=decision, premium_candles=premium_for(resolved.contract),
                account=account, spot=reference)
            return decision, resolved, signal

        if str(option_type).upper() not in ("CE", "PE"):
            raise OptionSignalError(
                "OPTION_TYPE_REQUIRED", "Option-candle signals need an explicit CE or PE option type")
        if spot is None and strike_mode.upper() != "EXACT":
            raise OptionResolutionError("SPOT_REQUIRED", "A spot is required to pick a strike")
        resolved = resolve_option_contract(
            resolver, underlying=underlying, option_type=option_type, spot=spot, expiry_mode=expiry_mode,
            expiry_date=expiry_date, strike_mode=strike_mode, strike_offset=strike_offset, exact_strike=exact_strike)
        premium = premium_for(resolved.contract)
        decision = self.option_decision(premium, option_type)
        if not decision.tradeable:
            return decision, resolved, None
        signal = self.build_signal(
            contract=resolved.contract, decision=decision, premium_candles=premium, account=account, spot=spot)
        return decision, resolved, signal
