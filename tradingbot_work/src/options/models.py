"""Normalized option-trading models (NSE F&O, BUY-side CE/PE)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.signals import Signal, SignalSide, SignalType

NSE_FO = "NSE_FO"
OPTION_STRATEGY_VERSION = "options-v1"


class MarketType(str, Enum):
    EQUITY = "EQUITY"
    INDEX_OPTION = "INDEX_OPTION"
    STOCK_OPTION = "STOCK_OPTION"


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


class OptionResolutionError(ValueError):
    """A contract, expiry or strike could not be resolved from instrument data."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OptionContract:
    instrument_key: str
    trading_symbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: OptionType
    lot_size: int
    segment: str = NSE_FO
    exchange: str = "NSE"
    freeze_quantity: int | None = None
    underlying_key: str | None = None
    underlying_type: str | None = None
    tick_size: float | None = None
    weekly: bool | None = None

    def __post_init__(self) -> None:
        if self.segment != NSE_FO or not str(self.instrument_key).startswith(NSE_FO + "|"):
            raise OptionResolutionError("NOT_NSE_FO", f"{self.instrument_key} is not an NSE_FO instrument")
        if not isinstance(self.option_type, OptionType):
            object.__setattr__(self, "option_type", OptionType(str(self.option_type).upper()))
        if int(self.lot_size) <= 0:
            raise OptionResolutionError("INVALID_LOT_SIZE", f"{self.trading_symbol}: lot size must be > 0")
        if float(self.strike) <= 0:
            raise OptionResolutionError("INVALID_STRIKE", f"{self.trading_symbol}: strike must be > 0")
        if self.freeze_quantity is not None and int(self.freeze_quantity) < int(self.lot_size):
            raise OptionResolutionError(
                "INVALID_FREEZE_QUANTITY", f"{self.trading_symbol}: freeze quantity below one lot")

    @property
    def market_type(self) -> MarketType:
        return MarketType.INDEX_OPTION if (self.underlying_type or "").upper() == "INDEX" else MarketType.STOCK_OPTION

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expiry"] = self.expiry.isoformat()
        data["option_type"] = self.option_type.value
        data["market_type"] = self.market_type.value
        return data

    def order_metadata(self) -> dict[str, Any]:
        """Columns persisted with the order row."""
        return {
            "market_type": self.market_type.value,
            "underlying": self.underlying,
            "option_type": self.option_type.value,
            "strike": float(self.strike),
            "expiry": self.expiry.isoformat(),
            "lot_size": int(self.lot_size),
            "instrument_key": self.instrument_key,
        }


@dataclass
class OptionQuote:
    ltp: float | None = None
    volume: float | None = None
    oi: float | None = None


@dataclass
class OptionChainRow:
    strike: float
    ce: OptionContract | None = None
    pe: OptionContract | None = None
    ce_quote: OptionQuote | None = None
    pe_quote: OptionQuote | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strike": self.strike,
            "ce": self.ce.as_dict() if self.ce else None,
            "pe": self.pe.as_dict() if self.pe else None,
            "ce_quote": asdict(self.ce_quote) if self.ce_quote else None,
            "pe_quote": asdict(self.pe_quote) if self.pe_quote else None,
        }


@dataclass
class OptionChain:
    underlying: str
    expiry: date
    rows: list[OptionChainRow]
    lot_size: int | None
    market_data_available: bool
    market_data_message: str | None = None
    atm_strike: float | None = None
    spot: float | None = None
    spot_source: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "expiry": self.expiry.isoformat(),
            "lot_size": self.lot_size,
            "market_data_available": self.market_data_available,
            "market_data_message": self.market_data_message,
            "atm_strike": self.atm_strike,
            "spot": self.spot,
            "spot_source": self.spot_source,
            "source": self.source,
            "metadata": self.metadata,
            "rows": [row.as_dict() for row in self.rows],
        }


class OptionSignal(BaseModel):
    """A BUY CE / BUY PE decision on a resolved NSE_FO contract."""

    signal_id: str
    timestamp: datetime
    market_type: MarketType
    underlying: str
    side: SignalSide = SignalSide.BUY
    option_type: OptionType
    strike: float = Field(gt=0)
    expiry: date
    instrument_key: str
    trading_symbol: str
    lot_size: int = Field(gt=0)
    lots: int = Field(ge=0)
    quantity: int = Field(ge=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    target: float = Field(gt=0)
    risk_per_unit: float = Field(gt=0)
    max_loss: float = Field(ge=0)
    premium_exposure: float = Field(ge=0)
    conditions: dict[str, bool] = {}
    details: dict[str, Any] = {}
    reason: str
    risk_status: str
    blocked: bool = False
    block_reason: str | None = None
    strategy_version: str

    @property
    def action(self) -> str:
        return "BUY CALL" if self.option_type == OptionType.CE else "BUY PUT"

    def summary(self) -> dict[str, Any]:
        data = self.model_dump(mode="json")
        data["action"] = self.action
        return data

    def to_order_signal(self) -> Signal:
        """Standard Signal for the existing OrderManager: the symbol is the
        resolved NSE_FO instrument key and quantity is a lot multiple."""
        if self.side != SignalSide.BUY:
            raise ValueError("Only BUY option orders are supported")
        if self.quantity <= 0 or self.quantity % self.lot_size:
            raise ValueError("Option quantity must be a positive lot-size multiple")
        return Signal(
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            symbol=self.instrument_key,
            side=SignalSide.BUY,
            signal_type=SignalType.ENTRY,
            entry_price=self.entry_price,
            quantity=self.quantity,
            stop_loss=self.stop_loss,
            target=self.target,
            reason=self.reason,
            confidence_metadata=dict(self.conditions),
            risk_status=self.risk_status,
            blocked=self.blocked,
            block_reason=self.block_reason,
        )
