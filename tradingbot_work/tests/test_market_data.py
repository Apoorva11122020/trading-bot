from datetime import datetime, timedelta, timezone

from src.market_data.candle_builder import CandleBuilder
from src.market_data.health import MarketDataHealthMonitor
from src.market_data.models import MarketDataStatus, Tick
from src.market_data.paper import PaperMarketDataFeed


def tick(minute, price, volume=100):
    return Tick(
        symbol="NIFTY",
        timestamp=datetime(2026, 10, 1, 9, minute, tzinfo=timezone.utc),
        ltp=price,
        volume=volume,
    )


def test_tick_validation_and_candle_creation():
    builder = CandleBuilder(5)
    assert builder.update(tick(15, 100)) is None
    assert builder.update(tick(17, 102)) is None
    candle = builder.update(tick(20, 101, 200))
    assert candle is not None
    assert candle.open == 100
    assert candle.high == 102
    assert candle.low == 100
    assert candle.close == 102


def test_out_of_order_tick_does_not_corrupt_candle():
    builder = CandleBuilder(5)
    builder.update(tick(15, 100))
    builder.update(tick(20, 110))
    # 16:00 belongs to the previous bucket and must be ignored.
    late = Tick(
        symbol="NIFTY",
        timestamp=datetime(2026, 10, 1, 9, 16, tzinfo=timezone.utc),
        ltp=1,
        volume=1,
    )
    assert builder.update(late) is None
    candle = builder.flush("NIFTY")
    assert candle.close == 110


def test_health_becomes_stale():
    monitor = MarketDataHealthMonitor(stale_after_seconds=10)
    t = tick(15, 100)
    monitor.connected(["NIFTY"])
    monitor.record_tick(t)
    now = t.timestamp + timedelta(seconds=11)
    assert monitor.is_stale(now)
    assert monitor.health.status == MarketDataStatus.STALE
    assert not monitor.can_trade(now)


def test_paper_feed_delivers_ticks():
    feed = PaperMarketDataFeed()
    received = []
    feed.subscribe(["NIFTY"])
    feed.set_tick_handler(received.append)
    feed.connect()
    feed.publish(tick(15, 100))
    assert len(received) == 1
    assert received[0].ltp == 100
    assert feed.health().status == MarketDataStatus.CONNECTED
    assert feed.can_trade(now=datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc))


def test_unsubscribed_symbol_is_ignored():
    feed = PaperMarketDataFeed()
    received = []
    feed.subscribe(["NIFTY"])
    feed.set_tick_handler(received.append)
    feed.connect()
    feed.publish(Tick(
        symbol="BANKNIFTY",
        timestamp=datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc),
        ltp=100,
    ))
    assert received == []
