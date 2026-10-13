from datetime import datetime, timedelta, timezone

import pytest

from src.broker.paper import PaperBroker
from src.database import Database
from src.engine import TradingEngine
from src.market_data.models import MarketDataHealth, MarketDataStatus
from src.risk.controller import PersistentKillSwitch, RiskController
from src.session.session_manager import SessionManager

from trading_support import CFG, SYMBOL, entry, exit_signal, ts

NOW = ts(10, 0)


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def healthy():
    return MarketDataHealth(status=MarketDataStatus.CONNECTED, last_tick_at=NOW)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp(SYMBOL, 100.0)
    db = Database(tmp_path / "risk.db")
    health = {"value": healthy()}
    clock = Clock(NOW)
    eng = TradingEngine(CFG, broker, db, equity=100_000, market_health=lambda: health["value"],
                        clock=clock, start_poller=False)
    eng.start()
    eng.health = health
    eng.clock_ref = clock
    yield eng
    eng.stop()
    db.close()


def _close_trade(db, trade_id, pnl, when):
    db.save_trade({"trade_id": trade_id, "order_id": trade_id, "symbol": SYMBOL, "side": "BUY",
                   "quantity": 1, "entry_price": 100, "exit_price": 100 + pnl, "realized_pnl": pnl,
                   "opened_at": when.isoformat(), "closed_at": when.isoformat(),
                   "last_exit_at": when.isoformat(), "status": "CLOSED", "exit_quantity": 1})


def test_kill_switch_blocks_entries_but_allows_exits(engine):
    engine.submit_entry(entry("E1"))
    engine.risk.kill_switch.activate("TEST")
    with pytest.raises(RuntimeError, match="KILL_SWITCH_ACTIVE"):
        engine.order_manager.submit_signal(entry("E2", symbol="OTHER"), engine.risk.account_state())
    exit_record = engine.order_manager.submit_signal(exit_signal("X1", "SELL", 100), engine.risk.account_state())
    assert exit_record.status.value == "FILLED"
    assert engine.order_manager.expected_positions == {}


def test_kill_switch_persists_across_restart(engine, tmp_path):
    engine.risk.kill_switch.activate("OPERATOR")
    assert PersistentKillSwitch(engine.database).is_halted is True
    assert PersistentKillSwitch(engine.database).reason == "OPERATOR"
    PersistentKillSwitch(engine.database).resume("OK")
    assert engine.risk.kill_switch.is_halted is False


def test_daily_loss_lock_latches_for_the_ist_day(engine):
    _close_trade(engine.database, "L1", -3_001.0, NOW)
    assert "DAILY_LOSS_LOCKED" in engine.risk.entry_block_reasons()
    assert engine.database.get_events(event_type="DAILY_LOSS_LOCK")
    # A later profit the same day does not unlock new entries.
    _close_trade(engine.database, "W1", 5_000.0, NOW + timedelta(minutes=5))
    assert engine.risk.daily_loss_locked() is True
    # Next IST trading day starts unlocked.
    engine.clock_ref.value = ts(10, 0, day=2)
    assert engine.risk.daily_loss_locked() is False


def test_daily_pnl_is_computed_from_persisted_trades(engine):
    _close_trade(engine.database, "A", -300.0, NOW)
    _close_trade(engine.database, "B", 200.0, NOW)
    _close_trade(engine.database, "OLD", -900.0, NOW - timedelta(days=1))
    account = engine.risk.account_state()
    assert account.daily_pnl == pytest.approx(-100.0)


def test_stale_market_data_blocks_new_entries_only(engine):
    engine.submit_entry(entry("E1"))
    engine.health["value"] = MarketDataHealth(status=MarketDataStatus.CONNECTED,
                                              last_tick_at=NOW - timedelta(seconds=30))
    with pytest.raises(RuntimeError, match="STALE_MARKET_DATA"):
        engine.order_manager.submit_signal(entry("E2", symbol="OTHER"), engine.risk.account_state())
    record = engine.order_manager.submit_signal(exit_signal("X1", "SELL", 100), engine.risk.account_state())
    assert record.status.value == "FILLED"


def test_market_data_disconnect_blocks_entries(engine):
    engine.health["value"] = MarketDataHealth(status=MarketDataStatus.DISCONNECTED, last_tick_at=NOW)
    assert "STALE_MARKET_DATA" in engine.risk.entry_block_reasons()


def test_broker_disconnect_and_session_cutoff_block_entries(engine):
    engine.clock_ref.value = ts(14, 50)
    assert "OUTSIDE_ENTRY_SESSION" in engine.risk.entry_block_reasons()
    engine.clock_ref.value = NOW
    engine.broker.disconnect()
    assert "BROKER_DISCONNECTED" in engine.risk.entry_block_reasons()


def test_max_trades_per_day(engine):
    for i in range(CFG.risk.max_trades_per_day):
        _close_trade(engine.database, f"T{i}", 1.0, NOW)
    assert "MAX_TRADES_PER_DAY_REACHED" in engine.risk.entry_block_reasons()


def test_open_position_and_exposure_come_from_live_state(engine):
    engine.submit_entry(entry("E1"))
    account = engine.risk.account_state()
    assert account.open_positions == 1
    assert account.current_exposure == pytest.approx(10_000.0)
    status = engine.risk.status()
    assert status["open_positions"] == 1
    assert status["kill_switch_active"] is False


def test_engine_heartbeat_and_startup_recovery(engine):
    status = engine.database.get_state("runtime_status")
    assert status["broker_connected"] is True
    assert status["safety"]["live_trading_enabled"] is False
    assert engine.last_recovery.ok
    assert engine.database.get_events(event_type="RECOVERY_COMPLETED")


def test_engine_eod_sweep_runs_on_price(engine):
    engine.submit_entry(entry("E1"))
    submitted = engine.on_price(SYMBOL, 100.0, 1.0, ts(15, 16))
    assert submitted and submitted[0].reason == "EOD_FORCE_EXIT"
    assert engine.order_manager.expected_positions == {}


def test_live_mode_refuses_to_start(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    broker = PaperBroker()
    db = Database(tmp_path / "live.db")
    eng = TradingEngine(CFG, broker, db, equity=100_000, start_poller=False)
    with pytest.raises(Exception, match="LIVE_MODE_REQUIRES"):
        eng.start()
    assert broker.is_connected() is False
    db.close()


def test_session_manager_reads_configured_entry_window():
    session = SessionManager(CFG.session)
    assert session.config.entry_start.strftime("%H:%M") == CFG.session.allow_entries_from
    assert session.config.entry_end.strftime("%H:%M") == CFG.session.stop_new_entries
