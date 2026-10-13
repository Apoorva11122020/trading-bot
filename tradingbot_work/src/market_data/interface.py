from abc import ABC, abstractmethod
from collections.abc import Callable

from src.market_data.models import MarketDataHealth, Tick


class MarketDataFeed(ABC):
    """Broker-independent real-time market data contract."""

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, symbols: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def unsubscribe(self, symbols: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_tick_handler(self, handler: Callable[[Tick], None]) -> None:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> MarketDataHealth:
        raise NotImplementedError
