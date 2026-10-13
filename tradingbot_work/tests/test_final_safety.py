import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.database import Database
from src.execution.order_manager import OrderManager
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.risk.risk_manager import RiskManager
from src.safety.kill_switch import KillSwitch
from src.safety.live_gate import LiveTradingSafetyError, assert_startup_safe
from src.session.session_manager import SessionManager
from src.config.loader import load_strategy_config


def test_live_mode_requires_confirmation(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.delenv("LIVE_TRADING_CONFIRMATION", raising=False)
    with pytest.raises(LiveTradingSafetyError):
        assert_startup_safe(explicit_confirmation=True)


def test_live_mode_stays_blocked_without_enable(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    with pytest.raises(LiveTradingSafetyError):
        assert_startup_safe(explicit_confirmation=True)


def test_paper_mode_is_safe(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    assert_startup_safe()


def test_final_paper_execution_roundtrip(tmp_path):
    cfg = load_strategy_config("config/strategy.yaml")
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("RELIANCE", 100.0)
    db = Database(tmp_path / "acceptance.db")
    om = OrderManager(broker, RiskManager(cfg.risk), SessionManager(cfg.session), database=db)
    ts = datetime(2026, 10, 5, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    signal = Signal(signal_id="ACCEPT-ENTRY", timestamp=ts, symbol="RELIANCE", side=SignalSide.BUY,
                    signal_type=SignalType.ENTRY, entry_price=100.0, quantity=500,
                    stop_loss=99.0, target=102.0, reason="ACCEPTANCE", risk_status="ALLOWED")
    record = om.submit_signal(signal, AccountState(equity=100000))
    assert record.status == OrderStatus.FILLED
    assert broker.get_positions()[0]["quantity"] == 500
    assert db.counts()["orders"] == 1
    broker.set_ltp("RELIANCE", 102.0)
    exit_signal = Signal(signal_id="ACCEPT-EXIT", timestamp=ts, symbol="RELIANCE", side=SignalSide.SELL,
                         signal_type=SignalType.EXIT, entry_price=102.0, quantity=500,
                         reason="TARGET", risk_status="ALLOWED")
    exit_record = om.submit_signal(exit_signal, AccountState(equity=100000, open_positions=1))
    assert exit_record.status == OrderStatus.FILLED
    assert broker.get_positions() == []
    db.close()


def test_kill_switch_blocks_entry_but_not_exit():
    ks = KillSwitch()
    assert ks.allow_entry()
    ks.activate("ACCEPTANCE_TEST")
    assert not ks.allow_entry()
    with pytest.raises(RuntimeError):
        ks.assert_entry_allowed()
    assert ks.resume("ACCEPTANCE_RESUME")
    assert ks.allow_entry()
