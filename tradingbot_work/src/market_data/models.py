from datetime import datetime
from enum import Enum
from pydantic import BaseModel, field_validator


class MarketDataStatus(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    STALE = "STALE"


class Tick(BaseModel):
    symbol: str
    timestamp: datetime
    ltp: float
    volume: float = 0.0
    last_quantity: float = 0.0

    @field_validator("timestamp")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @field_validator("ltp")
    @classmethod
    def positive_ltp(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ltp must be > 0")
        return value

    @field_validator("volume", "last_quantity")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("volume values must be >= 0")
        return value


class MarketDataHealth(BaseModel):
    status: MarketDataStatus = MarketDataStatus.DISCONNECTED
    last_tick_at: datetime | None = None
    reconnect_attempts: int = 0
    subscribed_symbols: list[str] = []
    last_error: str | None = None
