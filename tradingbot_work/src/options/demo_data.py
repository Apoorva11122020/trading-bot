"""DEMO DATA for the option Sandbox demo (no market-data entitlement).

The Upstox Sandbox exposes order APIs only, so the demo cannot read live
spot or premium quotes. These helpers produce clearly labelled demo candles:

* underlying: the equity demo pattern (src.demo.sandbox_demo) rescaled to a
  reference spot, so the actual StrategyEngine evaluates a real setup;
  a bearish variant mirrors it for the PUT path.
* premium: derived from the underlying path with a fixed delta. Used only to
  size and bracket the demo order; never presented as a market price.
"""
from __future__ import annotations

from src.demo.sandbox_demo import build_demo_candles
from src.models.market import Candle

from .models import OptionType

DEMO_DATA_LABEL = "DEMO DATA"
DEMO_SCALE = 0.002
DEMO_PREMIUM_FRACTION = 0.006
DEMO_DELTA = 0.5


def demo_underlying_candles(symbol: str, spot: float, bearish: bool = False, count: int = 55) -> list[Candle]:
    if spot <= 0:
        raise ValueError("spot must be > 0")
    base = build_demo_candles(symbol, count)
    k = spot * DEMO_SCALE
    sign = -1.0 if bearish else 1.0
    final_close = base[-1].close

    def tx(price: float) -> float:
        return spot + sign * k * (price - final_close)

    candles = []
    for c in base:
        high, low = (tx(c.high), tx(c.low)) if not bearish else (tx(c.low), tx(c.high))
        candles.append(Candle(
            timestamp=c.timestamp, symbol=symbol, open=tx(c.open), high=high, low=low,
            close=tx(c.close), volume=c.volume))
    return candles


def demo_premium_candles(
    instrument_key: str,
    underlying_candles: list[Candle],
    option_type: OptionType | str,
    spot: float,
    tick_size: float | None = 0.05,
    delta: float = DEMO_DELTA,
) -> list[Candle]:
    kind = option_type if isinstance(option_type, OptionType) else OptionType(str(option_type).upper())
    tick = float(tick_size or 0.05)
    base = max(tick * 20, spot * DEMO_PREMIUM_FRACTION)
    u0 = underlying_candles[-1].close
    sign = 1.0 if kind == OptionType.CE else -1.0

    def px(u: float) -> float:
        value = base + sign * delta * (u - u0)
        return max(tick, round(round(value / tick) * tick, 4))

    candles = []
    for c in underlying_candles:
        a, b = px(c.high), px(c.low)
        candles.append(Candle(
            timestamp=c.timestamp, symbol=instrument_key, open=px(c.open), high=max(a, b), low=min(a, b),
            close=px(c.close), volume=c.volume))
    return candles
