from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.market_data.candle_builder import CandleBuilder
from src.market_data.interface import MarketDataFeed
from src.market_data.models import Tick
from src.models.risk import AccountState
from src.models.signals import Signal
from src.strategy.runtime import StrategyRuntime
from src.execution.order_manager import OrderManager, ExecutionRecord


@dataclass
class PaperTradingStats:
    ticks_received: int = 0
    candles_completed: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0


class PaperTradingSession:
    """Connect market-data events to strategy and paper execution.

    This orchestration layer is deliberately broker-neutral and paper-only.
    It never enables live orders; a real broker adapter remains a separate concern.
    """

    def __init__(
        self,
        *,
        feed: MarketDataFeed,
        candle_builder: CandleBuilder,
        strategy_runtime: StrategyRuntime,
        order_manager: OrderManager,
        account_provider: Callable[[], AccountState],
        on_signal: Callable[[Signal], None] | None = None,
    ):
        self.feed = feed
        self.candle_builder = candle_builder
        self.strategy_runtime = strategy_runtime
        self.order_manager = order_manager
        self.account_provider = account_provider
        self.on_signal = on_signal
        self.stats = PaperTradingStats()
        self.last_execution: ExecutionRecord | None = None
        self.feed.set_tick_handler(self.on_tick)

    def subscribe(self, symbols: list[str]) -> None:
        self.feed.subscribe(symbols)

    def start(self) -> None:
        self.feed.connect()

    def stop(self) -> None:
        self.feed.disconnect()

    def on_tick(self, tick: Tick) -> None:
        self.stats.ticks_received += 1
        candle = self.candle_builder.update(tick)
        if candle is None:
            return
        self.stats.candles_completed += 1
        if not self.feed.can_trade(candle.timestamp):
            return

        signal = self.strategy_runtime.on_candle(
            candle,
            self.account_provider(),
            self.feed.health(),
        )
        if signal is None:
            return

        self.stats.signals_generated += 1
        if self.on_signal:
            self.on_signal(signal)

        if signal.blocked or signal.risk_status != "ALLOWED":
            return

        self.stats.orders_submitted += 1
        record = self.order_manager.submit_signal(signal, self.account_provider())
        self.last_execution = record
        if record.status.value == "FILLED":
            self.stats.orders_filled += 1
        elif record.status.value == "REJECTED":
            self.stats.orders_rejected += 1
