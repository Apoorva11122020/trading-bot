from datetime import datetime
from pydantic import BaseModel, field_validator


class Candle(BaseModel):
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("timestamp")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value

    @field_validator("open", "high", "low", "close")
    @classmethod
    def positive_price(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("prices must be > 0")
        return value

    @field_validator("volume")
    @classmethod
    def non_negative_volume(cls, value: float) -> float:
        if value < 0:
            raise ValueError("volume must be >= 0")
        return value

    def model_post_init(self, __context) -> None:
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open and close")
