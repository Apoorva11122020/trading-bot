"""Real-time NIFTY S2 paper trading."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.research_strategies import generate_signals
from src.broker.paper import PaperBroker
from src.config.research_loader import load_research_config
from src.database import Database
from src.market_data.candle_builder import CandleBuilder
from src.market_data.models import Tick
from src.market_data.paper import PaperMarketDataFeed
from src.models.risk import AccountState
from src.options.instrument_resolver import OptionInstrumentResolver, StaticContractSource
from src.paper_trading.config import load_paper_trading_settings, merged_strategy_config
from src.paper_trading.nifty_s2_runner import NiftyS2PaperRunner
from src.paper_trading.s2_entry import S2OptionEntryBuilder, S2SignalDetector
from src.risk.risk_manager import RiskManager
from src.safety.live_gate import assert_paper_trading_only

from src.models.market import Candle

from tests.options_support import NIFTY_EXPIRIES, contracts

IST = ZoneInfo("Asia/Kolkata")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
UNDERLYING_KEY = "NSE_INDEX|Nifty 50"
OPTION_KEY = next(
    c.instrument_key for c in contracts()
    if c.underlying == "NIFTY" and c.strike == 23000.0 and c.option_type.value == "CE"
)


def _ts(h, m, day=1):
    return datetime(2026, 10, day, h, m, tzinfo=IST)


def _orb_breakout_frame():
    rows = []
    price = 22900.0
    for i, (h, m) in enumerate([(9, 15), (9, 20), (9, 25), (9, 30), (9, 35)]):
        if i < 3:
            hi, lo, close = 22950.0, 22900.0, 22920.0 + i * 5
        else:
            hi, lo, close = 23010.0, 22980.0, 23000.0
        rows.append({
            "timestamp": _ts(h, m),
            "symbol": "NIFTY",
            "open": close - 2,
            "high": hi,
            "low": lo,
            "close": close,
            "volume": 1000,
        })
    return pd.DataFrame(rows)


def _option_premium_frame(n=20, start_price=120.0):
    rows = []
    t = _ts(9, 15)
    price = start_price
    for i in range(n):
        price += 1.0
        rows.append({
            "timestamp": t + timedelta(minutes=5 * i),
            "open": price,
            "high": price + 2,
            "low": price - 1,
            "close": price,
            "volume": 100,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def research_config():
    return load_research_config(ROOT / "config" / "strategy_research.yaml")


def test_candle_builder_completes_five_minute_bar():
    builder = CandleBuilder(5)
    sym = UNDERLYING_KEY
    assert builder.update(Tick(symbol=sym, timestamp=_ts(9, 15), ltp=22900.0)) is None
    assert builder.update(Tick(symbol=sym, timestamp=_ts(9, 17), ltp=22910.0)) is None
    completed = builder.update(Tick(symbol=sym, timestamp=_ts(9, 20), ltp=22920.0))
    assert completed is not None
    assert completed.open == 22900.0
    assert completed.close == 22910.0


def test_s2_signal_generation_matches_research(research_config):
    frame = _orb_breakout_frame()
    batch = generate_signals("S2", frame, "NIFTY", research_config)
    assert batch
    detector = S2SignalDetector("S2", "NIFTY", research_config)
    live = detector.on_candles(frame)
    assert live is not None
    assert live.direction == "BUY"
    assert detector.on_candles(frame) is None


def test_option_selection_buy_maps_to_ce(research_config):
    strategy_config, research, _ = merged_strategy_config()
    resolver = OptionInstrumentResolver(StaticContractSource(contracts()), today=lambda: _ts(9, 35).date())
    builder = S2OptionEntryBuilder(strategy_config, research, RiskManager(strategy_config.risk), resolver)
    frame = _orb_breakout_frame()
    rs = generate_signals("S2", frame, "NIFTY", research_config)[-1]
    contract = builder._resolve_contract(rs)
    assert contract.option_type.value == "CE"
    assert contract.underlying == "NIFTY"
    assert contract.expiry in NIFTY_EXPIRIES


def test_risk_rejection_when_daily_budget_exhausted(research_config):
    strategy_config, research, _ = merged_strategy_config()
    resolver = OptionInstrumentResolver(StaticContractSource(contracts()), today=lambda: _ts(9, 35).date())
    rm = RiskManager(strategy_config.risk)
    builder = S2OptionEntryBuilder(strategy_config, research, rm, resolver)
    frame = _orb_breakout_frame()
    rs = generate_signals("S2", frame, "NIFTY", research_config)[-1]
    account = AccountState(equity=1_000_000, daily_pnl=-3000)
    result = builder.build(rs, _option_premium_frame(), account)
    assert result.signal is None
    assert result.skip_reason


def test_duplicate_signal_suppressed(research_config):
    frame = _orb_breakout_frame()
    detector = S2SignalDetector("S2", "NIFTY", research_config)
    assert detector.on_candles(frame) is not None
    assert detector.on_candles(frame) is None


def test_stale_market_data_blocks_trading(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    feed = PaperMarketDataFeed(stale_after_seconds=1)
    db = Database(tmp_path / "paper.db")
    runner = NiftyS2PaperRunner(database=db, feed=feed)
    feed.connect()
    old = _ts(9, 15) - timedelta(minutes=30)
    feed.publish(Tick(symbol=UNDERLYING_KEY, timestamp=old, ltp=22900.0))
    assert not feed.can_trade(_ts(9, 35))
    db.close()


def test_websocket_disconnect_and_reconnect(tmp_path):
    feed = PaperMarketDataFeed()
    feed.connect()
    feed.subscribe([UNDERLYING_KEY])
    feed.publish(Tick(symbol=UNDERLYING_KEY, timestamp=_ts(9, 15), ltp=22900.0))
    feed.simulate_disconnect("test disconnect")
    assert feed.health().status.value == "DISCONNECTED"
    feed.connect()
    feed.publish(Tick(symbol=UNDERLYING_KEY, timestamp=_ts(9, 16), ltp=22901.0))
    assert feed.health().status.value == "CONNECTED"


def _prime_runner_market(runner, feed, at=None):
    from datetime import timezone

    at = at or _ts(10, 0)
    runner.engine.clock = lambda: at.astimezone(timezone.utc)
    runner.engine.risk.clock = runner.engine.clock
    feed.connect()
    feed.subscribe([UNDERLYING_KEY])
    feed.publish(Tick(symbol=UNDERLYING_KEY, timestamp=at.astimezone(timezone.utc), ltp=22900.0))


def test_paper_order_lifecycle(tmp_path, monkeypatch, research_config):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    feed = PaperMarketDataFeed()
    broker = PaperBroker(initial_cash=1_000_000)
    broker.connect()
    db = Database(tmp_path / "paper_lifecycle.db")
    settings = load_paper_trading_settings()
    runner = NiftyS2PaperRunner(database=db, feed=feed, settings=settings, broker=broker)
    runner.resolver = OptionInstrumentResolver(
        StaticContractSource(contracts()), today=lambda: _ts(9, 35).date(),
    )
    runner.entry_builder.resolver = runner.resolver
    runner.engine.start()
    _prime_runner_market(runner, feed, _ts(10, 0))

    for row in _orb_breakout_frame().to_dict("records"):
        candle = Candle(**row)
        runner._underlying_history.append(candle.model_dump())

    rs = runner.detector.on_candles(_orb_breakout_frame())
    assert rs is not None
    runner._active_option_key = OPTION_KEY
    for row in _option_premium_frame().to_dict("records"):
        runner._option_history.append({**row, "symbol": OPTION_KEY})
    broker.set_ltp(OPTION_KEY, 130.0)
    account = runner.engine.risk.account_state()
    result = runner.entry_builder.build(rs, runner._option_df(), account)
    assert result.signal is not None
    record = runner.engine.submit_entry(result.signal, instrument=result.contract)
    assert record.status.value == "FILLED"
    events = [e["event_type"] for e in db.get_events(limit=50)]
    assert "POSITION_ENTRY_SUBMITTED" in events
    db.close()


def test_restart_recovery_restores_managed_position(tmp_path, monkeypatch, research_config):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    feed = PaperMarketDataFeed()
    broker = PaperBroker(initial_cash=1_000_000)
    broker.connect()
    db = Database(tmp_path / "paper_recovery.db")
    settings = load_paper_trading_settings()
    runner = NiftyS2PaperRunner(database=db, feed=feed, settings=settings, broker=broker)
    runner.resolver = OptionInstrumentResolver(
        StaticContractSource(contracts()), today=lambda: _ts(9, 35).date(),
    )
    runner.entry_builder.resolver = runner.resolver
    runner.engine.start()
    _prime_runner_market(runner, feed, _ts(10, 0))
    broker.set_ltp(OPTION_KEY, 130.0)
    rs = generate_signals("S2", _orb_breakout_frame(), "NIFTY", research_config)[-1]
    runner._active_option_key = OPTION_KEY
    for row in _option_premium_frame().to_dict("records"):
        runner._option_history.append({**row, "symbol": OPTION_KEY})
    result = runner.entry_builder.build(rs, runner._option_df(), runner.engine.risk.account_state())
    runner.engine.submit_entry(result.signal, instrument=result.contract)

    runner2 = NiftyS2PaperRunner(database=db, feed=feed, settings=settings, broker=broker)
    runner2.engine.start()
    assert runner2.engine.lifecycle.open_symbols()
    db.close()


def test_eod_exit_submitted(tmp_path, monkeypatch, research_config):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    feed = PaperMarketDataFeed()
    broker = PaperBroker(initial_cash=1_000_000)
    broker.connect()
    db = Database(tmp_path / "paper_eod.db")
    settings = load_paper_trading_settings()
    runner = NiftyS2PaperRunner(database=db, feed=feed, settings=settings, broker=broker)
    runner.resolver = OptionInstrumentResolver(
        StaticContractSource(contracts()), today=lambda: _ts(9, 35).date(),
    )
    runner.entry_builder.resolver = runner.resolver
    runner.engine.start()
    _prime_runner_market(runner, feed, _ts(10, 0))
    broker.set_ltp(OPTION_KEY, 130.0)
    rs = generate_signals("S2", _orb_breakout_frame(), "NIFTY", research_config)[-1]
    for row in _option_premium_frame().to_dict("records"):
        runner._option_history.append({**row, "symbol": OPTION_KEY})
    result = runner.entry_builder.build(rs, runner._option_df(), runner.engine.risk.account_state())
    runner.engine.submit_entry(result.signal, instrument=result.contract)
    broker.set_ltp(OPTION_KEY, 125.0)
    exits = runner.engine.on_price(OPTION_KEY, 125.0, 5.0, _ts(15, 16))
    assert exits or runner.engine.lifecycle.eod_sweep(_ts(15, 16))
    db.close()


def test_daily_loss_lock_blocks_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    feed = PaperMarketDataFeed()
    broker = PaperBroker()
    broker.connect()
    db = Database(tmp_path / "paper_lock.db")
    runner = NiftyS2PaperRunner(database=db, feed=feed, broker=broker)
    runner.engine.start()
    now = _ts(10, 0)
    _prime_runner_market(runner, feed, now)
    db.save_trade({
        "trade_id": "t1", "order_id": "o1", "symbol": OPTION_KEY, "side": "BUY", "quantity": 65,
        "entry_price": 100, "exit_price": 50, "realized_pnl": -3001,
        "opened_at": now.isoformat(), "closed_at": now.isoformat(),
        "last_exit_at": now.isoformat(), "status": "CLOSED", "exit_quantity": 65,
    })
    reasons = runner.engine.risk.entry_block_reasons()
    assert "DAILY_LOSS_LOCKED" in reasons
    db.close()


def test_assert_paper_trading_only_refuses_live(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(Exception):
        assert_paper_trading_only()
