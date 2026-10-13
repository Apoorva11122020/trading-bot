"""Option risk: lot-size sizing, freeze quantity, premium exposure, SL/target,
breakeven/trailing configuration. Equity risk behaviour is unchanged."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.models.positions import Position
from src.models.risk import AccountState
from src.options.strategy import (
    OptionSignalError,
    option_exit_config,
    option_stop_loss,
    option_target,
    round_to_tick,
)
from src.position.monitor import PositionMonitor
from src.risk.risk_manager import RiskManager

from tests.options_support import strategy_stack

IST = ZoneInfo("Asia/Kolkata")


def manager(**overrides):
    cfg = dict(risk_per_trade_percent=0.5, max_daily_loss_inr=3000.0, max_daily_loss_percent=2.0,
               max_open_positions=3, max_trades_per_day=6, max_portfolio_exposure_percent=50.0,
               max_loss_gap_buffer_percent=10.0)
    cfg.update(overrides)
    return RiskManager(SimpleNamespace(**cfg))


ACCOUNT = AccountState(equity=1_000_000)


# --------------------------------------------------------------------------
# Lot-size quantity
# --------------------------------------------------------------------------

def test_quantity_rounds_down_to_whole_lots():
    # daily cap 3000 / (20 * 1.1) = 136 units -> 2 lots of 65 = 130
    sizing = manager().calculate_option_quantity(ACCOUNT, entry_premium=120, stop_premium=100, lot_size=65)
    assert sizing.raw_quantity == 136
    assert sizing.lots == 2 and sizing.quantity == 130
    assert sizing.quantity % 65 == 0
    assert sizing.risk_per_unit == 20 and sizing.max_loss == 130 * 20
    assert sizing.premium_exposure == 130 * 120
    assert sizing.limiting_factor == "RISK"


def test_quantity_is_zero_when_one_lot_exceeds_risk_budget():
    sizing = manager().calculate_option_quantity(
        AccountState(equity=100_000), entry_premium=120, stop_premium=100, lot_size=65)
    assert sizing.raw_quantity == 22 and sizing.lots == 0 and sizing.quantity == 0
    assert sizing.limiting_factor == "LOT_SIZE"


def test_freeze_quantity_caps_the_order():
    sizing = manager(risk_per_trade_percent=50).calculate_option_quantity(
        ACCOUNT, entry_premium=10, stop_premium=9, lot_size=65, freeze_quantity=1755)
    assert sizing.quantity == 1755 and sizing.lots == 27
    assert sizing.limiting_factor == "FREEZE_QUANTITY"


def test_premium_exposure_caps_the_order():
    # 10% of 1,000,000 = 100,000 premium -> 833 units -> 1 lot of 500
    sizing = manager(risk_per_trade_percent=5, max_daily_loss_inr=100_000).calculate_option_quantity(
        ACCOUNT, entry_premium=120, stop_premium=110, lot_size=500, max_premium_exposure_pct=10)
    assert sizing.max_quantity_by_premium == 833
    assert sizing.quantity == 500 and sizing.limiting_factor == "PREMIUM_EXPOSURE"


@pytest.mark.parametrize("entry, stop, lot", [(120, 120, 65), (120, 130, 65), (0, 0, 65), (120, 100, 0)])
def test_invalid_sizing_inputs_raise(entry, stop, lot):
    with pytest.raises(ValueError):
        manager().calculate_option_quantity(ACCOUNT, entry_premium=entry, stop_premium=stop, lot_size=lot)


# --------------------------------------------------------------------------
# Entry validation
# --------------------------------------------------------------------------

def _validate(rm=None, account=ACCOUNT, **overrides):
    args = dict(account=account, entry_premium=120, stop_premium=100, target_premium=160, quantity=130,
                lot_size=65, session_allowed=True, freeze_quantity=1755, max_premium_exposure_pct=10,
                open_option_positions=0, max_option_positions=2)
    args.update(overrides)
    return (rm or manager()).validate_option_entry(**args)


def test_valid_option_entry_is_allowed():
    result = _validate()
    assert result.allowed and result.reasons == []
    assert result.lots == 2 and result.max_loss == 2600 and result.premium_exposure == 15600
    assert result.daily_loss_limit == 3000


@pytest.mark.parametrize("overrides, reason", [
    ({"quantity": 200}, "QUANTITY_NOT_LOT_MULTIPLE"),
    ({"quantity": 0}, "QUANTITY_BELOW_ONE_LOT"),
    ({"quantity": 1820, "freeze_quantity": 1755}, "FREEZE_QUANTITY_EXCEEDED"),
    ({"quantity": 910, "max_premium_exposure_pct": 5}, "MAX_PREMIUM_EXPOSURE_REACHED"),
    ({"open_option_positions": 2}, "MAX_OPTION_POSITIONS_REACHED"),
    ({"stop_premium": 125}, "INVALID_OPTION_STOP"),
    ({"target_premium": 110}, "INVALID_OPTION_TARGET"),
    ({"session_allowed": False}, "ENTRY_OUTSIDE_TRADING_SESSION"),
])
def test_option_entry_blocks(overrides, reason):
    result = _validate(**overrides)
    assert not result.allowed
    assert reason in result.reasons


def test_daily_loss_impact_blocks_entries_that_could_breach_the_limit():
    account = AccountState(equity=1_000_000, daily_pnl=-2500)
    result = _validate(account=account)  # worst case 2600 * 1.1 = 2860 -> -5360 < -3000
    assert "DAILY_LOSS_LIMIT_WOULD_BE_EXCEEDED" in result.reasons
    assert result.daily_loss_after_stop == pytest.approx(-5360)


def test_account_level_equity_limits_still_apply():
    account = AccountState(equity=1_000_000, trades_today=6, open_positions=3)
    result = _validate(account=account)
    assert {"MAX_TRADES_PER_DAY_REACHED", "MAX_OPEN_POSITIONS_REACHED"} <= set(result.reasons)


def test_equity_validate_entry_is_unchanged():
    result = manager().validate_entry(
        account=AccountState(equity=100_000), entry_price=100, stop_loss=98,
        proposed_exposure=None, session_allowed=True)
    assert result.allowed and result.quantity == 250


# --------------------------------------------------------------------------
# Stop loss, target, tick rounding
# --------------------------------------------------------------------------

def test_premium_atr_stop():
    assert option_stop_loss(120, stop_mode="premium_atr", multiplier=1.5, premium_atr=10) == 105.0


def test_underlying_atr_stop_uses_delta():
    stop = option_stop_loss(120, stop_mode="underlying_atr", multiplier=1.5, underlying_atr=20,
                            underlying_delta=0.5)
    assert stop == 105.0


@pytest.mark.parametrize("kwargs, code", [
    ({"stop_mode": "premium_atr", "premium_atr": None}, "OPTION_PREMIUM_ATR_UNAVAILABLE"),
    ({"stop_mode": "premium_atr", "premium_atr": float("nan")}, "OPTION_PREMIUM_ATR_UNAVAILABLE"),
    ({"stop_mode": "premium_atr", "premium_atr": 0}, "OPTION_PREMIUM_ATR_UNAVAILABLE"),
    ({"stop_mode": "underlying_atr", "underlying_atr": None, "underlying_delta": 0.5},
     "UNDERLYING_ATR_UNAVAILABLE"),
    ({"stop_mode": "premium_atr", "premium_atr": 100}, "OPTION_STOP_INVALID"),
    ({"stop_mode": "fixed", "premium_atr": 5}, "INVALID_STOP_MODE"),
])
def test_stop_fails_safely(kwargs, code):
    with pytest.raises(OptionSignalError) as exc:
        option_stop_loss(120, multiplier=1.5, **kwargs)
    assert exc.value.code == code


def test_target_is_two_r_by_default():
    assert option_target(120, 100, 2.0) == 160.0
    with pytest.raises(OptionSignalError):
        option_target(100, 100, 2.0)


def test_tick_rounding_is_conservative_for_stops():
    assert round_to_tick(104.97, 0.05, "down") == 104.95
    assert round_to_tick(104.97, 0.05) == 104.95
    assert round_to_tick(104.98, 0.05) == 105.0


# --------------------------------------------------------------------------
# Breakeven and trailing (option multipliers, equity defaults unchanged)
# --------------------------------------------------------------------------

def test_option_exit_config_overrides_only_a_copy():
    config = strategy_stack()[0]
    equity_trailing = config.exit.trailing_stop.atr_multiplier
    equity_breakeven = config.exit.breakeven.activation_r
    option_cfg = option_exit_config(config)
    assert option_cfg.exit.breakeven.activation_r == config.options.risk.breakeven_r
    assert option_cfg.exit.trailing_stop.atr_multiplier == config.options.risk.trailing_atr_multiplier
    assert option_cfg.exit.target.risk_reward == config.options.risk.target_r
    assert config.exit.trailing_stop.atr_multiplier == equity_trailing
    assert config.exit.breakeven.activation_r == equity_breakeven


def test_breakeven_moves_option_stop_to_entry_at_one_r():
    config, _, session_manager, _, _ = strategy_stack()
    monitor = PositionMonitor(option_exit_config(config), session_manager)
    ts = datetime(2026, 10, 1, 11, 0, tzinfo=IST)
    position = Position(symbol="NSE_FO|1", side="LONG", quantity=195, average_entry=120.0, current_price=120.0,
                        stop_loss=100.0, target=160.0, entry_time=ts)
    monitor.register(position)
    assert monitor.evaluate(position, 130.0, 50.0, ts) is None
    assert monitor.state("NSE_FO|1").active_stop == 100.0
    assert monitor.evaluate(position, 140.0, 50.0, ts) is None  # +1R
    assert monitor.state("NSE_FO|1").active_stop == 120.0
    exit_signal = monitor.evaluate(position, 119.0, 50.0, ts)
    assert exit_signal is not None and exit_signal.reason == "BREAKEVEN_STOP"
