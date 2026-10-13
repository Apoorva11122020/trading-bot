from datetime import datetime
import pandas as pd

from src.models.risk import AccountState
from src.models.signals import Signal
from src.strategy.strategy_engine import StrategyEngine


class SignalGenerator:
    """Converts completed candle history into standardized trading signals.

    This layer intentionally accepts completed candles only. It does not place
    orders and it does not bypass the RiskManager used by StrategyEngine.
    """

    def __init__(self, strategy_engine: StrategyEngine):
        self.engine = strategy_engine

    def generate_entry_signal(
        self,
        candles: pd.DataFrame,
        account: AccountState,
        timestamp: datetime | None = None,
    ) -> Signal | None:
        return self.engine.evaluate_entry(candles, account, timestamp)

    def generate_exit_signal(
        self,
        position,
        current_price: float,
        atr_value: float,
        timestamp: datetime,
    ) -> Signal | None:
        return self.engine.evaluate_exit(
            position, current_price, atr_value, timestamp
        )
