from datetime import datetime
from pydantic import BaseModel, Field


class Position(BaseModel):
    symbol: str
    side: str
    quantity: int = Field(gt=0)
    average_entry: float = Field(gt=0)
    current_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    target: float = Field(gt=0)
    entry_time: datetime
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
