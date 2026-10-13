"""Expiry and strike selection (ATM / ITM / OTM / EXACT) on listed contracts."""
from __future__ import annotations

from datetime import date

import pytest

from src.options.models import OptionResolutionError, OptionType
from src.options.selection import atm_strike, select_expiry, select_strike
from src.options.strategy import resolve_option_contract

from tests.options_support import NIFTY_EXPIRIES, NIFTY_STRIKES, RELIANCE_EXPIRIES, RELIANCE_STRIKES, make_resolver

EXPIRIES = list(NIFTY_EXPIRIES)


def test_expiry_nearest_next_and_specific():
    assert select_expiry(EXPIRIES, "nearest") == date(2026, 10, 6)
    assert select_expiry(EXPIRIES, "next") == date(2026, 10, 13)
    assert select_expiry(EXPIRIES, "date", "2026-10-27") == date(2026, 10, 27)
    assert select_expiry(list(reversed(EXPIRIES)), "NEAREST") == date(2026, 10, 6)


@pytest.mark.parametrize("mode, value, code", [
    ("date", "2026-10-20", "EXPIRY_NOT_FOUND"),
    ("date", "27-10-2026", "INVALID_EXPIRY"),
    ("date", None, "EXPIRY_NOT_FOUND"),
    ("weekly", None, "INVALID_EXPIRY_MODE"),
])
def test_invalid_expiry_requests(mode, value, code):
    with pytest.raises(OptionResolutionError) as exc:
        select_expiry(EXPIRIES, mode, value)
    assert exc.value.code == code


def test_next_expiry_requires_two_listed_expiries():
    with pytest.raises(OptionResolutionError) as exc:
        select_expiry([date(2026, 10, 27)], "next")
    assert exc.value.code == "EXPIRY_NOT_FOUND"
    with pytest.raises(OptionResolutionError):
        select_expiry([], "nearest")


@pytest.mark.parametrize("spot, expected", [
    (23000.0, 23000.0), (23012.0, 23000.0), (23030.0, 23050.0), (23025.0, 23000.0),
    (20000.0, 22500.0), (24000.0, 23500.0),
])
def test_atm_is_nearest_listed_strike(spot, expected):
    assert atm_strike(NIFTY_STRIKES, spot) == expected


def test_atm_requires_positive_spot():
    with pytest.raises(OptionResolutionError) as exc:
        atm_strike(NIFTY_STRIKES, 0)
    assert exc.value.code == "SPOT_REQUIRED"


@pytest.mark.parametrize("mode, kind, offset, expected", [
    ("ATM", "CE", 0, 23000.0),
    ("ATM", "PE", 3, 23000.0),
    ("OTM", "CE", 2, 23100.0),   # call OTM = above ATM
    ("OTM", "PE", 2, 22900.0),   # put OTM = below ATM
    ("ITM", "CE", 1, 22950.0),   # call ITM = below ATM
    ("ITM", "PE", 1, 23050.0),   # put ITM = above ATM
    ("OTM", "CE", 0, 23000.0),
])
def test_strike_modes(mode, kind, offset, expected):
    assert select_strike(NIFTY_STRIKES, option_type=kind, mode=mode, spot=23010, offset=offset) == expected


def test_strike_offsets_follow_listed_strikes_with_uneven_steps():
    # RELIANCE strikes: ...1250, 1260, 1280, 1300 (1270 not listed)
    assert select_strike(RELIANCE_STRIKES, option_type=OptionType.CE, mode="OTM", spot=1258, offset=1) == 1280.0
    assert select_strike(RELIANCE_STRIKES, option_type="PE", mode="ITM", spot=1258, offset=2) == 1300.0


def test_exact_strike_must_be_listed():
    assert select_strike(NIFTY_STRIKES, option_type="CE", mode="EXACT", exact=22750) == 22750.0
    with pytest.raises(OptionResolutionError) as exc:
        select_strike(NIFTY_STRIKES, option_type="CE", mode="EXACT", exact=22775)
    assert exc.value.code == "STRIKE_NOT_FOUND"
    with pytest.raises(OptionResolutionError) as exc:
        select_strike(NIFTY_STRIKES, option_type="CE", mode="EXACT")
    assert exc.value.code == "STRIKE_NOT_FOUND"


@pytest.mark.parametrize("kwargs, code", [
    ({"mode": "OTM", "offset": 50}, "STRIKE_OUT_OF_RANGE"),
    ({"mode": "OTM", "offset": -1}, "INVALID_STRIKE_OFFSET"),
    ({"mode": "DEEP"}, "INVALID_STRIKE_MODE"),
])
def test_invalid_strike_requests(kwargs, code):
    with pytest.raises(OptionResolutionError) as exc:
        select_strike(NIFTY_STRIKES, option_type="CE", spot=23000, **kwargs)
    assert exc.value.code == code


def test_resolve_option_contract_end_to_end_index_and_stock():
    resolver = make_resolver()
    nifty = resolve_option_contract(resolver, underlying="NIFTY", option_type="PE", spot=23010,
                                    expiry_mode="next", strike_mode="OTM", strike_offset=1)
    assert nifty.expiry == NIFTY_EXPIRIES[1] and nifty.strike == 22950.0
    assert nifty.contract.trading_symbol == "NIFTY 22950 PE 13 OCT 26"

    reliance = resolve_option_contract(resolver, underlying="RELIANCE", option_type="CE", spot=1251,
                                       expiry_mode="date", expiry_date="2026-11-24", strike_mode="ATM")
    assert reliance.expiry == RELIANCE_EXPIRIES[1] and reliance.strike == 1250.0
    assert reliance.contract.lot_size == 500 and reliance.contract.market_type.value == "STOCK_OPTION"
