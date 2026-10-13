from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class SignalSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class SignalType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class Signal(BaseModel):
    signal_id: str
    timestamp: datetime
    symbol: str
    side: SignalSide
    signal_type: SignalType
    entry_price: float = Field(gt=0)
    quantity: int = Field(ge=0)
    stop_loss: float | None = None
    target: float | None = None
    trailing_stop: float | None = None
    reason: str
    confidence_metadata: dict[str, bool | float | str] = {}
    risk_status: str
    blocked: bool = False
    block_reason: str | None = None
