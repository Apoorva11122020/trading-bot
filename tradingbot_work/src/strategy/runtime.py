from collections import defaultdict, deque
from datetime import datetime

import pandas as pd

from src.market_data.models import MarketDataHealth, MarketDataStatus
from src.models.market import Candle
from src.models.risk import AccountState
from src.models.signals import Signal
from src.strategy.signal_generator import SignalGenerator


class StrategyRuntime:
    """Connects completed candles to signal generation.

    It is deliberately broker/order-manager agnostic. A caller supplies the
    current account state and market-data health; this runtime only decides
    whether a standardized signal should be generated.
    """

    def __init__(
        self,
        signal_generator: SignalGenerator,
        history_size: int = 250,
    ):
        if history_size < 2:
            raise ValueError("history_size must be >= 2")
        self.signal_generator = signal_generator
        self.history_size = history_size
        self._history: dict[str, deque[Candle]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._processed_candles: set[tuple[str, datetime]] = set()
        self._signal_keys: set[tuple[str, datetime, str, str]] = set()

    def history(self, symbol: str) -> pd.DataFrame:
        candles = list(self._history.get(symbol, ()))
        return pd.DataFrame([c.model_dump() for c in candles])

    def on_candle(
        self,
        candle: Candle,
        account: AccountState,
        market_data_health: MarketDataHealth,
    ) -> Signal | None:
        candle_key = (candle.symbol, candle.timestamp)
        if candle_key in self._processed_candles:
            return None

        # Only completed candles from a healthy feed may generate entries.
        if market_data_health.status != MarketDataStatus.CONNECTED:
            return None

        if market_data_health.last_tick_at is None:
            return None

        self._history[candle.symbol].append(candle)
        self._processed_candles.add(candle_key)

        data = self.history(candle.symbol)
        signal = self.signal_generator.generate_entry_signal(
            data,
            account,
            candle.timestamp,
        )
        if signal is None:
            return None

        signal_key = (
            signal.symbol,
            signal.timestamp,
            signal.side.value,
            signal.signal_type.value,
        )
        if signal_key in self._signal_keys:
            return None

        self._signal_keys.add(signal_key)
        return signal
