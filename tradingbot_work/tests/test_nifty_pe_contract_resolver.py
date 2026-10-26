from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from scripts.nifty_pe_contract_resolver import (
    build_pe_recovery_order,
    format_pe_order_alert,
    resolve_atm_pe,
)
from scripts.nifty_pe_recovery_rule import PeRecoverySignal
from src.options.models import OptionResolutionError
from tests.options_support import TODAY, make_resolver


class FakeHistoricalClient:
    def __init__(self, premium: float = 85.5) -> None:
        self.premium = premium

    def intraday_candles(self, instrument_key: str, *, unit: str = "minutes", interval: int = 1):
        return [SimpleNamespace(close=self.premium, low=self.premium - 1.0, high=self.premium + 1.0)]


def _signal(spot: float = 23050.0) -> PeRecoverySignal:
    return PeRecoverySignal(
        morning_open=23200.0,
        session_low=22950.0,
        recovery_level=23130.0,
        spot_ref=spot,
        lots=2,
        stop_premium_points=20.0,
        lot_size=65,
    )


def test_resolve_atm_pe_nearest_expiry():
    resolver = make_resolver(TODAY)
    contract, expiry, strike = resolve_atm_pe(23050.0, resolver, as_of=TODAY)
    assert expiry == date(2026, 10, 6)
    assert strike == 23050.0
    assert contract.option_type.value == "PE"
    assert contract.lot_size == 65


def test_build_pe_recovery_order_rounds_premium_and_stop():
    resolver = make_resolver(TODAY)
    order = build_pe_recovery_order(_signal(), resolver, FakeHistoricalClient(85.55), as_of=TODAY)
    assert order.entry_premium == 85.55
    assert order.stop_premium == 65.55
    assert order.quantity == 130
    assert "NIFTY" in order.contract.trading_symbol
    alert = format_pe_order_alert(order)
    assert "entry_premium=85.55" in alert
    assert "stop_premium=65.55" in alert


def test_build_pe_recovery_order_rejects_missing_premium():
    resolver = make_resolver(TODAY)

    class EmptyClient:
        def intraday_candles(self, instrument_key: str, *, unit: str = "minutes", interval: int = 1):
            return []

    with pytest.raises(OptionResolutionError) as exc:
        build_pe_recovery_order(_signal(), resolver, EmptyClient(), as_of=TODAY)
    assert exc.value.code == "OPTION_PREMIUM_UNAVAILABLE"
