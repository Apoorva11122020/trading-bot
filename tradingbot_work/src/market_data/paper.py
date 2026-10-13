from collections.abc import Callable

from src.market_data.interface import MarketDataFeed
from src.market_data.models import MarketDataHealth, MarketDataStatus, Tick
from src.market_data.health import MarketDataHealthMonitor


class PaperMarketDataFeed(MarketDataFeed):
    """Deterministic in-memory feed. It never contacts an exchange."""

    def __init__(self, stale_after_seconds: int = 15):
        self._monitor = MarketDataHealthMonitor(stale_after_seconds)
        self._handler: Callable[[Tick], None] | None = None
        self._symbols: list[str] = []

    def connect(self) -> None:
        self._monitor.connected(self._symbols)

    def disconnect(self) -> None:
        self._monitor.disconnected()

    def subscribe(self, symbols: list[str]) -> None:
        self._symbols = list(dict.fromkeys(symbols))
        if self._monitor.health.status == MarketDataStatus.CONNECTED:
            self._monitor.health.subscribed_symbols = self._symbols

    def unsubscribe(self, symbols: list[str]) -> None:
        remove = set(symbols)
        self._symbols = [s for s in self._symbols if s not in remove]
        self._monitor.health.subscribed_symbols = self._symbols

    def set_tick_handler(self, handler: Callable[[Tick], None]) -> None:
        self._handler = handler

    def publish(self, tick: Tick) -> None:
        if tick.symbol not in self._symbols:
            return
        self._monitor.record_tick(tick)
        if self._handler:
            self._handler(tick)

    def health(self) -> MarketDataHealth:
        return self._monitor.health.model_copy(deep=True)

    def can_trade(self, now=None) -> bool:
        return self._monitor.can_trade(now)

    def simulate_disconnect(self, error: str = "simulated disconnect") -> None:
        self._monitor.disconnected(error)
