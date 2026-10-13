from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"

class OrderStatus(str, Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    MODIFIED = "MODIFIED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"

class BrokerOrder(BaseModel):
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int = Field(gt=0)
    order_type: OrderType
    price: float | None = None
    status: OrderStatus
    created_at: datetime
    filled_quantity: int = 0
    average_fill_price: float | None = None
    rejection_reason: str | None = None
    tag: str | None = None
