"""OptionSignal generation on top of the existing StrategyEngine."""
from __future__ import annotations

import pytest

from src.demo.sandbox_demo import build_demo_candles
from src.models.risk import AccountState
from src.models.signals import SignalSide, SignalType
from src.options.demo_data import demo_premium_candles, demo_underlying_candles
from src.options.models import OptionSignal, OptionType
from src.options.strategy import OptionSignalError, resolve_option_contract

from tests.options_support import NIFTY_EXPIRIES, make_resolver, strategy_stack

ACCOUNT = AccountState(equity=1_000_000)
SPOT = 23000.0


def _premium_for(candles, spot=SPOT):
    return lambda contract: demo_premium_candles(contract.instrument_key, candles, contract.option_type, spot,
                                                 contract.tick_size)


def test_bullish_underlying_signal_buys_a_call():
    *_, engine = strategy_stack()
    decision = engine.underlying_decision(demo_underlying_candles("NIFTY", SPOT), "AUTO")
    assert decision.tradeable and decision.side is SignalSide.BUY and decision.option_type is OptionType.CE
    assert decision.reason == "UNDERLYING_BULLISH_BUY_CALL"
    assert decision.spot == pytest.approx(SPOT)
    assert decision.conditions == {"ema_trend": True, "vwap": True, "rsi": True, "breakout": True, "volume": True}


def test_bearish_underlying_signal_buys_a_put_never_sells():
    *_, engine = strategy_stack()
    decision = engine.underlying_decision(demo_underlying_candles("NIFTY", SPOT, bearish=True), "AUTO")
    assert decision.side is SignalSide.BUY  # BUY PE, never naked selling
    assert decision.option_type is OptionType.PE
    assert decision.reason == "UNDERLYING_BEARISH_BUY_PUT"


def test_forced_option_type_conflicting_with_signal_does_not_trade():
    *_, engine = strategy_stack()
    decision = engine.underlying_decision(demo_underlying_candles("NIFTY", SPOT), "PE")
    assert not decision.tradeable
    assert decision.reason == "UNDERLYING_BUY_SIGNAL_CONFLICTS_WITH_PE"
    assert engine.underlying_decision(demo_underlying_candles("NIFTY", SPOT), "CE").tradeable


def test_no_setup_means_no_signal():
    *_, engine = strategy_stack()
    flat = demo_underlying_candles("NIFTY", SPOT)[:-1]  # without the breakout bar
    decision = engine.underlying_decision(flat, "AUTO")
    assert not decision.tradeable and decision.reason == "ENTRY_CONDITIONS_NOT_MET"
    assert engine.underlying_decision([], "AUTO").reason == "INSUFFICIENT_UNDERLYING_DATA"
    with pytest.raises(OptionSignalError):
        engine.underlying_decision(flat, "STRADDLE")


def test_option_candle_mode_requires_explicit_type_and_uses_premium_breakout():
    *_, engine = strategy_stack()
    underlying = demo_underlying_candles("NIFTY", SPOT)
    premium = demo_premium_candles("NSE_FO|1", underlying, "CE", SPOT, 0.05)
    with pytest.raises(OptionSignalError) as exc:
        engine.option_decision(premium, "AUTO")
    assert exc.value.code == "OPTION_TYPE_REQUIRED"
    decision = engine.option_decision(premium, "CE")
    assert decision.tradeable and decision.source == "OPTION" and decision.option_type is OptionType.CE
    falling = demo_premium_candles("NSE_FO|2", underlying, "PE", SPOT, 0.05)
    assert not engine.option_decision(falling, "PE").tradeable


def test_built_signal_is_lot_sized_with_validated_stop_and_target():
    *_, engine = strategy_stack()
    candles = demo_underlying_candles("NIFTY", SPOT)
    decision, resolved, signal = engine.generate(
        resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT, underlying_candles=candles,
        premium_candles_for=_premium_for(candles))
    assert isinstance(signal, OptionSignal)
    assert resolved.contract.trading_symbol == "NIFTY 23000 CE 06 OCT 26"
    assert signal.instrument_key == resolved.contract.instrument_key
    assert signal.market_type.value == "INDEX_OPTION" and signal.underlying == "NIFTY"
    assert signal.option_type is OptionType.CE and signal.action == "BUY CALL"
    assert signal.expiry == NIFTY_EXPIRIES[0] and signal.strike == 23000.0
    assert signal.quantity > 0 and signal.quantity % 65 == 0 and signal.lots == signal.quantity // 65
    assert 0 < signal.stop_loss < signal.entry_price < signal.target
    assert signal.target - signal.entry_price == pytest.approx(2 * (signal.entry_price - signal.stop_loss), abs=0.05)
    assert signal.risk_status == "ALLOWED" and not signal.blocked
    assert signal.details["stop_mode"] == "premium_atr" and signal.details["premium_atr"] > 0
    assert signal.strategy_version.endswith("options-v1")
    summary = signal.summary()
    assert summary["action"] == "BUY CALL" and summary["expiry"] == "2026-10-06"


def test_to_order_signal_targets_the_nse_fo_instrument_with_lot_quantity():
    *_, engine = strategy_stack()
    candles = demo_underlying_candles("NIFTY", SPOT, bearish=True)
    _, resolved, signal = engine.generate(
        resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT, underlying_candles=candles,
        premium_candles_for=_premium_for(candles))
    order = signal.to_order_signal()
    assert order.symbol == resolved.contract.instrument_key and order.symbol.startswith("NSE_FO|")
    assert order.side is SignalSide.BUY and order.signal_type is SignalType.ENTRY
    assert order.quantity == signal.quantity and order.stop_loss == signal.stop_loss
    with pytest.raises(ValueError):
        signal.model_copy(update={"quantity": 100}).to_order_signal()
    with pytest.raises(ValueError):
        signal.model_copy(update={"quantity": 0}).to_order_signal()


def test_stock_option_flow_uses_stock_lot_size():
    *_, engine = strategy_stack()
    spot = 1251.0
    candles = demo_underlying_candles("RELIANCE", spot, bearish=True)
    _, resolved, signal = engine.generate(
        resolver=make_resolver(), underlying="RELIANCE", account=ACCOUNT, underlying_candles=candles,
        premium_candles_for=_premium_for(candles, spot), strike_mode="OTM", strike_offset=1)
    assert resolved.contract.option_type is OptionType.PE and resolved.strike == 1240.0
    assert signal.market_type.value == "STOCK_OPTION" and signal.lot_size == 500
    assert signal.quantity % 500 == 0 and signal.quantity > 0


def test_missing_premium_history_fails_safely():
    *_, engine = strategy_stack()
    candles = demo_underlying_candles("NIFTY", SPOT)
    short = lambda contract: demo_premium_candles(contract.instrument_key, candles, contract.option_type,
                                                  SPOT, 0.05)[-3:]
    with pytest.raises(OptionSignalError) as exc:
        engine.generate(resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT,
                        underlying_candles=candles, premium_candles_for=short)
    assert exc.value.code == "OPTION_PREMIUM_ATR_UNAVAILABLE"
    with pytest.raises(OptionSignalError) as exc:
        engine.generate(resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT,
                        underlying_candles=candles, premium_candles_for=lambda c: None)
    assert exc.value.code == "OPTION_PREMIUM_DATA_UNAVAILABLE"


def test_small_account_produces_blocked_signal_not_fractional_lots():
    *_, engine = strategy_stack()
    candles = demo_underlying_candles("NIFTY", SPOT)
    _, _, signal = engine.generate(
        resolver=make_resolver(), underlying="NIFTY", account=AccountState(equity=50_000),
        underlying_candles=candles, premium_candles_for=_premium_for(candles))
    assert signal.blocked and signal.quantity == 0
    assert "QUANTITY_BELOW_ONE_LOT" in signal.block_reason
    with pytest.raises(ValueError):
        signal.to_order_signal()


def test_option_signal_source_mode_b():
    *_, engine = strategy_stack()
    candles = demo_underlying_candles("NIFTY", SPOT)
    decision, resolved, signal = engine.generate(
        resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT, signal_source="OPTION",
        option_type="CE", spot=SPOT, premium_candles_for=_premium_for(candles))
    assert decision.source == "OPTION" and signal.details["signal_source"] == "OPTION"
    with pytest.raises(OptionSignalError):
        engine.generate(resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT, signal_source="OPTION",
                        option_type="AUTO", spot=SPOT)
    with pytest.raises(OptionSignalError):
        engine.generate(resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT, signal_source="NEWS")


def test_equity_strategy_signal_is_unchanged_by_refactor():
    config, risk_manager, session_manager, strategy_engine, _ = strategy_stack()
    candles = build_demo_candles("RELIANCE")
    frame = strategy_engine.indicator_frame(candles)
    side, conditions = strategy_engine.entry_direction(frame.iloc[-1])
    assert side is SignalSide.BUY and all(conditions.values())
    signal = strategy_engine.evaluate_entry(frame, AccountState(equity=100_000), candles[-1].timestamp)
    assert signal is not None and signal.side is SignalSide.BUY and signal.symbol == "RELIANCE"
    assert signal.confidence_metadata == {
        "ema_trend": True, "vwap": True, "rsi": True, "breakout": True, "volume": True}


def test_contract_resolution_failure_propagates():
    *_, engine = strategy_stack()
    candles = demo_underlying_candles("NIFTY", SPOT)
    from src.options.models import OptionResolutionError
    with pytest.raises(OptionResolutionError) as exc:
        engine.generate(resolver=make_resolver(), underlying="NIFTY", account=ACCOUNT, underlying_candles=candles,
                        premium_candles_for=_premium_for(candles), strike_mode="EXACT", exact_strike=23025)
    assert exc.value.code == "STRIKE_NOT_FOUND"
    assert resolve_option_contract(make_resolver(), underlying="NIFTY", option_type="CE", spot=SPOT).strike == 23000
