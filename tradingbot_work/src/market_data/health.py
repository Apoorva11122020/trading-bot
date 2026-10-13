from datetime import datetime, timedelta, timezone

from src.market_data.models import MarketDataHealth, MarketDataStatus, Tick


class MarketDataHealthMonitor:
    def __init__(self, stale_after_seconds: int = 15):
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be > 0")
        self.stale_after_seconds = stale_after_seconds
        self.health = MarketDataHealth()

    def connected(self, symbols: list[str] | None = None) -> None:
        self.health.status = MarketDataStatus.CONNECTED
        if symbols is not None:
            self.health.subscribed_symbols = list(symbols)
        self.health.last_error = None

    def disconnected(self, error: str | None = None) -> None:
        self.health.status = MarketDataStatus.DISCONNECTED
        self.health.last_error = error

    def reconnecting(self) -> None:
        self.health.status = MarketDataStatus.CONNECTING

    def record_tick(self, tick: Tick) -> None:
        self.health.last_tick_at = tick.timestamp
        if self.health.status != MarketDataStatus.CONNECTED:
            self.health.status = MarketDataStatus.CONNECTED

    def is_stale(self, now: datetime | None = None) -> bool:
        if self.health.last_tick_at is None:
            return True
        now = now or datetime.now(timezone.utc)
        age = (now - self.health.last_tick_at).total_seconds()
        stale = age > self.stale_after_seconds
        if stale:
            self.health.status = MarketDataStatus.STALE
        return stale

    def can_trade(self, now: datetime | None = None) -> bool:
        return self.health.status == MarketDataStatus.CONNECTED and not self.is_stale(now)
