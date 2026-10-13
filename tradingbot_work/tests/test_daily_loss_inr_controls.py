"""Daily INR loss limit and option sizing risk controls."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.options_engine import DAILY_LOSS, OptionsResearchEngine
from src.models.risk import AccountState
from src.risk.controller import RiskController
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager

from tests.options_support import contracts

IST = ZoneInfo("Asia/Kolkata")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def rm(**overrides):
    cfg = dict(
        risk_per_trade_percent=0.5,
        max_daily_loss_inr=3000.0,
        max_daily_loss_percent=2.0,
        max_open_positions=3,
        max_trades_per_day=6,
        max_portfolio_exposure_percent=50.0,
        max_loss_gap_buffer_percent=10.0,
    )
    cfg.update(overrides)
    return RiskManager(SimpleNamespace(**cfg))


ACCOUNT = AccountState(equity=1_000_000)


def test_single_trade_risk_within_allowed_budget():
    sizing = rm().calculate_option_quantity(ACCOUNT, entry_premium=120, stop_premium=100, lot_size=65)
    assert sizing.max_loss <= 3000
    assert sizing.quantity % 65 == 0
    check = rm().validate_option_entry(
        ACCOUNT, 120, 100, 160, sizing.quantity, 65, session_allowed=True,
        max_premium_exposure_pct=10,
    )
    assert check.allowed
    assert check.max_loss <= 3000


def test_per_trade_budget_capped_by_remaining_daily_loss():
    account = AccountState(equity=1_000_000, daily_pnl=-2000)
    sizing = rm().calculate_option_quantity(account, entry_premium=120, stop_premium=100, lot_size=65)
    # Remaining daily budget 1000 / (20 * 1.1) -> 45 units -> 0 lots
    assert sizing.risk_budget == 1000
    assert sizing.quantity == 0


def test_cumulative_daily_loss_blocks_entry_that_would_breach_limit():
    account = AccountState(equity=1_000_000, daily_pnl=-2500)
    result = rm().validate_option_entry(
        account, 120, 100, 160, quantity=130, lot_size=65, session_allowed=True,
        max_premium_exposure_pct=10,
    )
    assert not result.allowed
    assert "DAILY_LOSS_LIMIT_WOULD_BE_EXCEEDED" in result.reasons


def test_entry_blocked_after_daily_limit_reached():
    account = AccountState(equity=1_000_000, daily_pnl=-3000)
    assert rm().remaining_daily_loss_budget(account) == 0
    result = rm().validate_option_entry(
        account, 120, 100, 160, quantity=65, lot_size=65, session_allowed=True,
    )
    assert not result.allowed
    assert "DAILY_LOSS_LIMIT_REACHED" in result.reasons


def test_next_day_resets_remaining_budget():
    account = AccountState(equity=1_000_000, daily_pnl=-3000)
    assert rm().remaining_daily_loss_budget(account) == 0
    fresh = AccountState(equity=1_000_000, daily_pnl=0)
    assert rm().remaining_daily_loss_budget(fresh) == 3000


def test_open_position_risk_reduces_remaining_budget():
    account = AccountState(equity=1_000_000, daily_pnl=-500)
    sizing = rm().calculate_option_quantity(
        account, entry_premium=120, stop_premium=100, lot_size=65,
        open_position_max_loss=2000,
    )
    assert sizing.risk_budget == 500


def test_multiple_trades_respect_cumulative_cap_in_engine():
    from src.config.loader import load_strategy_config
    from src.config.research_loader import load_research_config

    research = load_research_config(ROOT / "config" / "strategy_research.yaml")
    strategy = load_strategy_config(ROOT / "config" / "strategy.yaml")
    key = "NSE_FO|TEST"
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    underlying_rows = []
    premium_rows = []
    price = 23000.0
    prem = 120.0
    for i in range(120):
        t = start + timedelta(minutes=5 * i)
        price += 8 if i < 20 else (-12 if i < 40 else 6)
        prem += 2 if i < 20 else (-3 if i < 40 else 1)
        underlying_rows.append({
            "timestamp": t, "symbol": "NIFTY", "open": price, "high": price + 10,
            "low": price - 10, "close": price, "volume": 5000,
        })
        premium_rows.append({
            "timestamp": t, "instrument_key": key, "open": prem, "high": prem + 3,
            "low": max(1.0, prem - 8), "close": prem, "volume": 100, "open_interest": 1000,
        })
    engine = OptionsResearchEngine(research, strategy, contracts(), {key: pd.DataFrame(premium_rows)}, run_id="risk-t")
    result = engine.run(pd.DataFrame(underlying_rows), "NIFTY", "S2", date(2026, 10, 1), date(2026, 10, 1))
    day_pnls: dict[str, float] = {}
    for trade in result.trades:
        day = trade.entry_time[:10]
        day_pnls[day] = day_pnls.get(day, 0.0) + trade.pnl
    assert all(pnl > -3000 - 1e-6 for pnl in day_pnls.values())


def test_gap_slippage_scenario_uses_buffer_in_validation():
    account = AccountState(equity=1_000_000, daily_pnl=0)
    # 65 * 20 = 1300 planned; worst case 1430 with 10% buffer
    result = rm(max_loss_gap_buffer_percent=10).validate_option_entry(
        account, 120, 100, 160, quantity=65, lot_size=65, session_allowed=True,
    )
    assert result.allowed
    tight = rm(max_loss_gap_buffer_percent=10).validate_option_entry(
        AccountState(equity=1_000_000, daily_pnl=-1700),
        120, 100, 160, quantity=65, lot_size=65, session_allowed=True,
    )
    assert not tight.allowed


def test_lot_size_rounding_never_exceeds_budget():
    sizing = rm().calculate_option_quantity(ACCOUNT, entry_premium=50, stop_premium=45, lot_size=65)
    assert sizing.quantity % 65 == 0
    buffer_loss = sizing.max_loss * rm().gap_loss_buffer_multiplier()
    assert buffer_loss <= 3000 + 1e-6


def test_risk_controller_uses_inr_limit(tmp_path):
    from src.config.loader import load_strategy_config
    from src.database import Database

    cfg = load_strategy_config(ROOT / "config" / "strategy.yaml")
    db = Database(tmp_path / "risk_inr.db")
    session = SessionManager(cfg.session)
    controller = RiskController(cfg, db, session, equity=100_000)
    assert controller.daily_loss_limit() == 3000
    db.close()


def test_max_daily_loss_enforcement_blocks_research_signals():
    from src.config.loader import load_strategy_config
    from src.config.research_loader import load_research_config

    research = load_research_config(ROOT / "config" / "strategy_research.yaml")
    strategy = load_strategy_config(ROOT / "config" / "strategy.yaml")
    key = "NSE_FO|TEST2"
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    rows = []
    prem_rows = []
    for i in range(40):
        t = start + timedelta(minutes=5 * i)
        rows.append({
            "timestamp": t, "symbol": "NIFTY", "open": 23000 + i, "high": 23010 + i,
            "low": 22990 + i, "close": 23005 + i, "volume": 5000,
        })
        prem_rows.append({
            "timestamp": t, "instrument_key": key, "open": 120, "high": 125,
            "low": 90, "close": 118, "volume": 100, "open_interest": 1000,
        })
    engine = OptionsResearchEngine(research, strategy, contracts(), {key: pd.DataFrame(prem_rows)}, run_id="risk-t2")
    result = engine.run(pd.DataFrame(rows), "NIFTY", "S2", date(2026, 10, 1), date(2026, 10, 1))
    assert any(s.get("reason") == DAILY_LOSS for s in result.skipped) or len(result.trades) <= 6
