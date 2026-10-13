"""Option contract model, instrument-master source/cache, resolver and chain."""
from __future__ import annotations

import os
import time
from datetime import date

import pytest

from src.options.instrument_resolver import (
    OptionInstrumentResolver,
    StaticContractSource,
    UpstoxInstrumentMasterSource,
    parse_instrument_rows,
)
from src.options.models import MarketType, OptionContract, OptionQuote, OptionResolutionError, OptionType
from src.options.option_chain import MARKET_DATA_UNAVAILABLE, OptionChainService

from tests.options_support import (
    EXPIRED,
    NIFTY_EXPIRIES,
    NIFTY_STRIKES,
    RELIANCE_EXPIRIES,
    TODAY,
    gz_payload,
    instrument_rows,
    make_resolver,
)


# --------------------------------------------------------------------------
# Contract model
# --------------------------------------------------------------------------

def _contract(**overrides):
    data = dict(instrument_key="NSE_FO|1", trading_symbol="NIFTY 23000 CE 06 OCT 26", underlying="NIFTY",
                expiry=date(2026, 10, 6), strike=23000.0, option_type="CE", lot_size=65,
                freeze_quantity=1755, underlying_type="INDEX")
    data.update(overrides)
    return OptionContract(**data)


def test_contract_model_normalises_and_exposes_order_metadata():
    contract = _contract()
    assert contract.option_type is OptionType.CE
    assert contract.market_type is MarketType.INDEX_OPTION
    assert _contract(underlying_type="EQUITY").market_type is MarketType.STOCK_OPTION
    assert contract.order_metadata() == {
        "market_type": "INDEX_OPTION", "underlying": "NIFTY", "option_type": "CE", "strike": 23000.0,
        "expiry": "2026-10-06", "lot_size": 65, "instrument_key": "NSE_FO|1"}
    assert contract.as_dict()["expiry"] == "2026-10-06"


@pytest.mark.parametrize("overrides, code", [
    ({"instrument_key": "NSE_EQ|INE002A01018"}, "NOT_NSE_FO"),
    ({"segment": "NSE_EQ"}, "NOT_NSE_FO"),
    ({"lot_size": 0}, "INVALID_LOT_SIZE"),
    ({"strike": 0}, "INVALID_STRIKE"),
    ({"freeze_quantity": 10}, "INVALID_FREEZE_QUANTITY"),
])
def test_contract_model_rejects_invalid_contracts(overrides, code):
    with pytest.raises(OptionResolutionError) as exc:
        _contract(**overrides)
    assert exc.value.code == code


def test_contract_model_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        _contract(option_type="FUT")


# --------------------------------------------------------------------------
# Instrument master parsing
# --------------------------------------------------------------------------

def test_parse_keeps_only_valid_nse_fo_options():
    contracts, skipped = parse_instrument_rows(instrument_rows())
    assert skipped == 1  # the lot_size=0 row
    assert all(c.segment == "NSE_FO" and c.instrument_key.startswith("NSE_FO|") for c in contracts)
    assert {c.option_type for c in contracts} == {OptionType.CE, OptionType.PE}
    nifty = next(c for c in contracts if c.underlying == "NIFTY" and c.expiry == NIFTY_EXPIRIES[0])
    assert nifty.lot_size == 65 and nifty.freeze_quantity == 1755
    assert nifty.tick_size == pytest.approx(0.05)  # paise -> rupees
    assert nifty.underlying_key == "NSE_INDEX|Nifty 50"


def test_expiry_epoch_is_converted_to_ist_trading_date():
    contracts, _ = parse_instrument_rows(instrument_rows())
    assert {c.expiry for c in contracts if c.underlying == "NIFTY"} == set(NIFTY_EXPIRIES) | {EXPIRED}


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------

def test_resolver_lists_underlyings_by_market_type():
    resolver = make_resolver()
    index = {u["symbol"] for u in resolver.underlyings("INDEX_OPTION")}
    stock = {u["symbol"] for u in resolver.underlyings("STOCK_OPTION")}
    assert index == {"NIFTY", "BANKNIFTY"}
    assert stock == {"RELIANCE"}
    reliance = resolver.underlyings("STOCK_OPTION")[0]
    assert reliance["lot_size"] == 500 and reliance["underlying_key"] == "NSE_EQ|INE002A01018"


def test_resolver_excludes_expired_contracts():
    resolver = make_resolver()
    assert resolver.expiries("NIFTY") == list(NIFTY_EXPIRIES)
    assert EXPIRED not in resolver.expiries("nifty")


@pytest.mark.parametrize("kind", ["CE", "PE"])
def test_resolver_resolves_ce_and_pe(kind):
    resolver = make_resolver()
    contract = resolver.resolve("NIFTY", NIFTY_EXPIRIES[0], 23000, kind)
    assert contract.option_type.value == kind
    assert contract.strike == 23000.0 and contract.expiry == NIFTY_EXPIRIES[0]
    assert contract.trading_symbol == f"NIFTY 23000 {kind} 06 OCT 26"
    assert contract.segment == "NSE_FO" and contract.lot_size == 65


def test_resolver_accepts_iso_expiry_strings():
    contract = make_resolver().resolve("RELIANCE", "2026-10-27", 1250, "PE")
    assert contract.expiry == RELIANCE_EXPIRIES[0] and contract.market_type is MarketType.STOCK_OPTION


@pytest.mark.parametrize("args, code", [
    (("NIFTY", date(2026, 10, 7), 23000, "CE"), "EXPIRY_NOT_FOUND"),
    (("NIFTY", NIFTY_EXPIRIES[0], 23025, "CE"), "STRIKE_NOT_FOUND"),
    (("RELIANCE", RELIANCE_EXPIRIES[0], 1290, "PE"), "CONTRACT_NOT_FOUND"),
    (("NOSUCH", NIFTY_EXPIRIES[0], 23000, "CE"), "UNDERLYING_NOT_FOUND"),
    (("NIFTY", NIFTY_EXPIRIES[0], 23000, "XX"), "OPTION_TYPE_NOT_FOUND"),
    (("NIFTY", EXPIRED, 23000, "CE"), "EXPIRY_NOT_FOUND"),
])
def test_resolver_rejects_invalid_requests(args, code):
    with pytest.raises(OptionResolutionError) as exc:
        make_resolver().resolve(*args)
    assert exc.value.code == code


def test_resolver_strikes_are_sorted_and_unique():
    assert make_resolver().strikes("NIFTY", NIFTY_EXPIRIES[0], "CE") == NIFTY_STRIKES


# --------------------------------------------------------------------------
# Instrument master source (download, cache, safety)
# --------------------------------------------------------------------------

def test_source_downloads_without_credentials_and_caches(tmp_path):
    calls = []

    def fetcher(url):
        calls.append(url)
        return gz_payload()

    source = UpstoxInstrumentMasterSource(cache_dir=tmp_path, cache_hours=12, fetcher=fetcher)
    first = source.option_contracts()
    second = source.option_contracts()
    assert first is second and len(calls) == 1
    assert calls == ["https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"]
    assert source.cache_file.exists()
    assert source.metadata["source"] == "UPSTOX_INSTRUMENT_MASTER" and source.metadata["stale"] is False

    fresh = UpstoxInstrumentMasterSource(cache_dir=tmp_path, cache_hours=12,
                                         fetcher=lambda url: pytest.fail("cache should be used"))
    assert len(fresh.option_contracts()) == len(first)
    assert fresh.metadata["from_cache"] is True


def test_source_falls_back_to_stale_cache_when_download_fails(tmp_path):
    UpstoxInstrumentMasterSource(cache_dir=tmp_path, fetcher=lambda url: gz_payload()).option_contracts()
    cache = tmp_path / "NSE.json.gz"
    old = time.time() - 48 * 3600
    os.utime(cache, (old, old))

    def broken(url):
        raise ConnectionError("offline")

    source = UpstoxInstrumentMasterSource(cache_dir=tmp_path, cache_hours=12, fetcher=broken)
    assert source.option_contracts()
    assert source.metadata["stale"] is True


def test_source_without_data_fails_clearly(tmp_path):
    def broken(url):
        raise ConnectionError("offline")

    with pytest.raises(OptionResolutionError) as exc:
        UpstoxInstrumentMasterSource(cache_dir=tmp_path, fetcher=broken).option_contracts()
    assert exc.value.code == "INSTRUMENT_DATA_UNAVAILABLE"


@pytest.mark.parametrize("url", [
    "https://api.upstox.com/v2/option/contract",
    "http://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "https://evil.example.com/NSE.json.gz",
])
def test_source_refuses_other_hosts(url):
    with pytest.raises(OptionResolutionError) as exc:
        UpstoxInstrumentMasterSource(url=url)
    assert exc.value.code == "INSTRUMENT_SOURCE_REFUSED"


def test_corrupt_download_is_not_cached(tmp_path):
    def corrupt(url):
        return b"not json"

    with pytest.raises(OptionResolutionError):
        UpstoxInstrumentMasterSource(cache_dir=tmp_path, fetcher=corrupt).option_contracts()
    assert not (tmp_path / "NSE.json.gz").exists()


# --------------------------------------------------------------------------
# Option chain
# --------------------------------------------------------------------------

def test_chain_lists_real_contracts_without_fabricated_quotes():
    service = OptionChainService(make_resolver())
    chain = service.get_option_chain("NIFTY", NIFTY_EXPIRIES[0])
    assert chain.market_data_available is False
    assert chain.market_data_message == MARKET_DATA_UNAVAILABLE == "Option chain requires market-data credentials"
    assert [r.strike for r in chain.rows] == NIFTY_STRIKES
    assert all(r.ce and r.pe for r in chain.rows)
    assert all(r.ce_quote is None and r.pe_quote is None for r in chain.rows)
    assert chain.spot_source == "DEMO_REFERENCE_MEDIAN_STRIKE" and chain.atm_strike == 23000.0
    data = chain.as_dict()
    assert data["rows"][0]["ce_quote"] is None and data["lot_size"] == 65


def test_chain_uses_quotes_only_from_a_market_data_provider():
    class Provider:
        def quotes(self, keys):
            return {k: OptionQuote(ltp=10.0, volume=100, oi=1000) for k in keys}

        def spot(self, underlying_key):
            return 23110.0

    chain = OptionChainService(make_resolver(), Provider()).get_option_chain("NIFTY", NIFTY_EXPIRIES[0])
    assert chain.market_data_available is True and chain.market_data_message is None
    assert chain.spot == 23110.0 and chain.spot_source == "MARKET_DATA" and chain.atm_strike == 23100.0
    assert chain.rows[0].ce_quote.ltp == 10.0


def test_chain_reports_provider_failure_instead_of_values():
    class Broken:
        def quotes(self, keys):
            raise PermissionError("no entitlement")

        def spot(self, key):
            return None

    chain = OptionChainService(make_resolver(), Broken()).get_option_chain("NIFTY", NIFTY_EXPIRIES[0], spot=22990)
    assert chain.market_data_available is False
    assert chain.market_data_message.startswith(MARKET_DATA_UNAVAILABLE)
    assert chain.atm_strike == 23000.0 and chain.spot_source == "REQUEST"


def test_chain_filters():
    service = OptionChainService(make_resolver())
    contracts = service.get_contracts("RELIANCE", RELIANCE_EXPIRIES[0])
    assert {c.strike for c in service.filter_by_strike(contracts, 1250)} == {1250.0}
    window = service.filter_by_strike(contracts, min_strike=1240, max_strike=1260)
    assert {c.strike for c in window} == {1240.0, 1250.0, 1260.0}
    puts = service.filter_by_option_type(contracts, "PE")
    assert puts and all(c.option_type is OptionType.PE for c in puts)
    assert 1290.0 not in {c.strike for c in puts}
    assert service.get_expiries("RELIANCE") == list(RELIANCE_EXPIRIES)


def test_static_source_metadata():
    resolver = OptionInstrumentResolver(StaticContractSource([]), today=lambda: TODAY)
    assert resolver.metadata["source"] == "STATIC"
    assert resolver.underlyings() == []
