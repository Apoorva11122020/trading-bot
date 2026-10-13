from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.broker.paper import PaperBroker
from src.config.loader import load_strategy_config
from src.database import Database
from src.execution.order_manager import OrderManager
from src.market_data.candle_builder import CandleBuilder
from src.market_data.models import Tick
from src.market_data.paper import PaperMarketDataFeed
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.paper_trading import PaperTradingSession
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager
from src.strategy.runtime import StrategyRuntime


class StubGenerator:
    def generate_entry_signal(self, data, account, timestamp):
        if len(data) < 2:
            return None
        return Signal(
            signal_id=f"S-{timestamp.isoformat()}", timestamp=timestamp,
            symbol="DEMO", side=SignalSide.BUY, signal_type=SignalType.ENTRY,
            entry_price=101.0, quantity=495, stop_loss=99.99, target=103.02,
            reason="TEST", risk_status="ALLOWED",
        )


def make_session(tmp_path: Path):
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("DEMO", 101.0)
    db = Database(tmp_path / "paper.db")
    feed = PaperMarketDataFeed()
    cfg = load_strategy_config("config/strategy.yaml")
    om = OrderManager(broker, RiskManager(cfg.risk), SessionManager(cfg.session), database=db)
    session = PaperTradingSession(
        feed=feed,
        candle_builder=CandleBuilder(interval_minutes=5),
        strategy_runtime=StrategyRuntime(StubGenerator()),
        order_manager=om,
        account_provider=lambda: AccountState(equity=100000, open_positions=len(broker.get_positions())),
    )
    return session, feed, broker, db


def test_paper_session_runs_tick_to_order(tmp_path):
    session, feed, broker, db = make_session(tmp_path)
    session.subscribe(["DEMO"])
    session.start()
    base = datetime(2026, 10, 5, 9, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    for minute, price in [(30, 100.0), (31, 100.5), (34, 101.0), (35, 101.0), (36, 101.2), (39, 101.0), (40, 101.0)]:
        broker.set_ltp("DEMO", price)
        feed.publish(Tick(symbol="DEMO", timestamp=base.replace(minute=minute), ltp=price, volume=1000))
    session.stop()
    assert session.stats.ticks_received == 7
    assert session.stats.candles_completed == 2
    assert session.stats.signals_generated == 1
    assert session.stats.orders_filled == 1
    assert len(broker.get_positions()) == 1
    assert db.counts()["trades"] == 1
    db.close()


def test_unsubscribed_ticks_do_not_reach_strategy(tmp_path):
    session, feed, broker, db = make_session(tmp_path)
    session.subscribe(["OTHER"])
    session.start()
    ts = datetime(2026, 10, 5, 9, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    feed.publish(Tick(symbol="DEMO", timestamp=ts, ltp=101.0, volume=1000))
    assert session.stats.ticks_received == 0
    assert session.stats.signals_generated == 0
    db.close()
