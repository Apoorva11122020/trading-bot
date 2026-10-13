"""Map S2 research signals to sized option entry signals (unchanged S2 logic upstream)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from src.backtest.research_strategies import ResearchSignal, generate_signals
from src.indicators.technical import atr as calc_atr
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.options.instrument_resolver import OptionInstrumentResolver
from src.options.models import OptionResolutionError, OptionType
from src.options.selection import atm_strike, select_expiry
from src.options.strategy import option_stop_loss, option_target, round_to_tick
from src.risk.risk_manager import RiskManager


@dataclass
class S2EntryResult:
    signal: Signal | None
    contract: dict[str, Any] | None
    sizing: dict[str, Any] | None
    risk: dict[str, Any] | None
    skip_reason: str | None = None


class S2SignalDetector:
    """Uses batch generate_signals on candle history; emits only the latest bar's signal."""

    def __init__(self, strategy_id: str, underlying: str, research_config):
        self.strategy_id = strategy_id.upper()
        self.underlying = underlying.upper()
        self.research_config = research_config
        self._keys: set[tuple[str, str]] = set()

    def on_candles(self, frame: pd.DataFrame) -> ResearchSignal | None:
        if frame is None or frame.empty:
            return None
        signals = generate_signals(self.strategy_id, frame, self.underlying, self.research_config)
        if not signals:
            return None
        latest_ts = pd.Timestamp(frame.iloc[-1]["timestamp"])
        for signal in reversed(signals):
            if pd.Timestamp(signal.timestamp) == latest_ts:
                key = (signal.timestamp.isoformat(), signal.direction)
                if key in self._keys:
                    return None
                self._keys.add(key)
                return signal
        return None


class S2OptionEntryBuilder:
    def __init__(
        self,
        strategy_config,
        research_config,
        risk_manager: RiskManager,
        resolver: OptionInstrumentResolver,
    ):
        self.strategy_config = strategy_config
        self.research_config = research_config
        self.risk_manager = risk_manager
        self.resolver = resolver

    def _resolve_contract(self, signal: ResearchSignal):
        self.resolver.today = lambda: signal.timestamp.date()
        option_type = OptionType.CE if signal.direction == "BUY" else OptionType.PE
        expiry = select_expiry(
            self.resolver.expiries(signal.underlying),
            self.research_config.option_mapping.get("expiry_mode", "nearest"),
        )
        strike = atm_strike(
            self.resolver.strikes(signal.underlying, expiry, option_type),
            signal.spot,
        )
        return self.resolver.resolve(signal.underlying, expiry, strike, option_type)

    def build(
        self,
        research_signal: ResearchSignal,
        option_candles: pd.DataFrame,
        account: AccountState,
        *,
        open_option_positions: int = 0,
        session_allowed: bool = True,
    ) -> S2EntryResult:
        try:
            contract = self._resolve_contract(research_signal)
        except OptionResolutionError as exc:
            return S2EntryResult(None, None, None, None, skip_reason=exc.code)

        if option_candles is None or option_candles.empty:
            return S2EntryResult(None, contract.as_dict(), None, None, skip_reason="OPTION_PREMIUM_DATA_UNAVAILABLE")

        prem = option_candles.sort_values("timestamp").reset_index(drop=True)
        last = prem.iloc[-1]
        entry = round_to_tick(float(last["close"]), contract.tick_size, "nearest")
        if entry <= 0:
            return S2EntryResult(None, contract.as_dict(), None, None, skip_reason="OPTION_PREMIUM_INVALID")

        prem = prem.copy()
        prem["atr"] = calc_atr(prem, self.strategy_config.indicators.atr_period)
        if prem.empty or pd.isna(prem.iloc[-1].get("atr")):
            return S2EntryResult(None, contract.as_dict(), None, None, skip_reason="INSUFFICIENT_ATR_HISTORY")

        premium_atr = float(prem.iloc[-1]["atr"])
        stop = option_stop_loss(
            entry,
            stop_mode="premium_atr",
            multiplier=self.strategy_config.options.risk.stop_atr_multiplier,
            premium_atr=premium_atr,
            tick_size=contract.tick_size,
        )
        target = option_target(entry, stop, self.strategy_config.options.risk.target_r, contract.tick_size)

        sizing = self.risk_manager.calculate_option_quantity(
            account,
            entry,
            stop,
            contract.lot_size,
            contract.freeze_quantity,
            self.research_config.risk.max_premium_exposure_pct,
            open_position_max_loss=0.0,
        )
        if sizing.quantity <= 0:
            return S2EntryResult(None, contract.as_dict(), sizing.model_dump(), None, skip_reason="QUANTITY_BELOW_ONE_LOT")

        check = self.risk_manager.validate_option_entry(
            account,
            entry,
            stop,
            target,
            sizing.quantity,
            contract.lot_size,
            session_allowed=session_allowed,
            freeze_quantity=contract.freeze_quantity,
            max_premium_exposure_pct=self.research_config.risk.max_premium_exposure_pct,
            open_option_positions=open_option_positions,
            max_option_positions=self.research_config.risk.max_option_positions,
        )
        if not check.allowed:
            return S2EntryResult(
                None,
                contract.as_dict(),
                sizing.model_dump(),
                check.model_dump(),
                skip_reason=";".join(check.reasons),
            )

        signal = Signal(
            signal_id=str(uuid4()),
            timestamp=research_signal.timestamp,
            symbol=contract.instrument_key,
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY,
            entry_price=entry,
            quantity=sizing.quantity,
            stop_loss=stop,
            target=target,
            reason=f"S2_{research_signal.direction}",
            confidence_metadata={k: v for k, v in research_signal.conditions.items()},
            risk_status="ALLOWED",
            blocked=False,
        )
        return S2EntryResult(signal, contract.as_dict(), sizing.model_dump(), check.model_dump())
