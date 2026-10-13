from datetime import datetime, timedelta
from collections import defaultdict

from src.models.market import Candle
from src.market_data.models import Tick


class CandleBuilder:
    """Builds time-based candles from tick data.

    A candle is emitted when the next tick arrives in a later interval.
    The final open candle is intentionally not emitted until the interval closes.
    """

    def __init__(self, interval_minutes: int = 5):
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be > 0")
        self.interval_minutes = interval_minutes
        self._current: dict[str, dict] = {}

    def _bucket_start(self, timestamp: datetime) -> datetime:
        minute = (timestamp.minute // self.interval_minutes) * self.interval_minutes
        return timestamp.replace(minute=minute, second=0, microsecond=0)

    def update(self, tick: Tick) -> Candle | None:
        bucket = self._bucket_start(tick.timestamp)
        current = self._current.get(tick.symbol)

        if current is None:
            self._current[tick.symbol] = {
                "timestamp": bucket,
                "symbol": tick.symbol,
                "open": tick.ltp,
                "high": tick.ltp,
                "low": tick.ltp,
                "close": tick.ltp,
                "volume": tick.volume,
            }
            return None

        if bucket < current["timestamp"]:
            # Late/out-of-order tick: ignore rather than corrupting candles.
            return None

        if bucket == current["timestamp"]:
            current["high"] = max(current["high"], tick.ltp)
            current["low"] = min(current["low"], tick.ltp)
            current["close"] = tick.ltp
            current["volume"] = max(current["volume"], tick.volume)
            return None

        completed = Candle(**current)
        self._current[tick.symbol] = {
            "timestamp": bucket,
            "symbol": tick.symbol,
            "open": tick.ltp,
            "high": tick.ltp,
            "low": tick.ltp,
            "close": tick.ltp,
            "volume": tick.volume,
        }
        return completed

    def flush(self, symbol: str) -> Candle | None:
        current = self._current.pop(symbol, None)
        return Candle(**current) if current else None
