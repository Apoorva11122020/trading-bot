from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    signal_id: str | None
    symbol: str
    side: str
    order_type: str
    quantity: int
    status: str
    filled_quantity: int
    average_fill_price: float | None
    requested_price: float | None
    created_at: datetime
    updated_at: datetime
    reason: str | None = None


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    order_id: str
    signal_id: str | None
    symbol: str
    side: str
    quantity: int
    entry_price: float | None
    exit_price: float | None
    realized_pnl: float | None
    opened_at: datetime | None
    closed_at: datetime | None
    status: str


@dataclass(frozen=True)
class PositionRecord:
    symbol: str
    side: str
    quantity: int
    average_entry: float
    current_price: float | None
    unrealized_pnl: float | None
    updated_at: datetime


@dataclass(frozen=True)
class SystemEvent:
    event_id: str
    event_type: str
    severity: str
    message: str
    timestamp: datetime
    metadata_json: str = "{}"
