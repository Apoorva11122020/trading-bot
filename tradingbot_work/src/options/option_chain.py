"""Option chain built from listed contracts plus (optional) market quotes.

Quotes (LTP / volume / OI) are only shown when a market-data provider is
configured. Without one the chain lists the real contracts and reports
that market data is unavailable; quote values are never fabricated.
"""
from __future__ import annotations

from datetime import date
from statistics import median
from typing import Iterable, Protocol

from .instrument_resolver import OptionInstrumentResolver
from .models import OptionChain, OptionChainRow, OptionContract, OptionQuote, OptionType
from .selection import atm_strike

MARKET_DATA_UNAVAILABLE = "Option chain requires market-data credentials"
DEMO_REFERENCE_SPOT = "DEMO_REFERENCE_MEDIAN_STRIKE"


class QuoteProvider(Protocol):
    def quotes(self, instrument_keys: list[str]) -> dict[str, OptionQuote]: ...

    def spot(self, underlying_key: str) -> float | None: ...


def demo_reference_spot(strikes: Iterable[float]) -> float:
    """Median listed strike. A demo anchor only, never a market price."""
    values = sorted(set(strikes))
    if not values:
        raise ValueError("No strikes")
    return float(median(values))


class OptionChainService:
    def __init__(self, resolver: OptionInstrumentResolver, quote_provider: QuoteProvider | None = None) -> None:
        self.resolver = resolver
        self.quote_provider = quote_provider

    def get_expiries(self, underlying: str) -> list[date]:
        return self.resolver.expiries(underlying)

    def get_contracts(self, underlying: str, expiry: date) -> list[OptionContract]:
        return self.resolver.contracts(underlying, expiry)

    @staticmethod
    def filter_by_strike(
        contracts: Iterable[OptionContract],
        strike: float | None = None,
        min_strike: float | None = None,
        max_strike: float | None = None,
    ) -> list[OptionContract]:
        result = []
        for contract in contracts:
            if strike is not None and abs(contract.strike - float(strike)) >= 1e-6:
                continue
            if min_strike is not None and contract.strike < min_strike:
                continue
            if max_strike is not None and contract.strike > max_strike:
                continue
            result.append(contract)
        return result

    @staticmethod
    def filter_by_option_type(contracts: Iterable[OptionContract], option_type: OptionType | str) -> list[OptionContract]:
        kind = OptionInstrumentResolver.option_type(option_type)
        return [c for c in contracts if c.option_type == kind]

    def get_option_chain(self, underlying: str, expiry: date, spot: float | None = None) -> OptionChain:
        contracts = self.get_contracts(underlying, expiry)
        rows: dict[float, OptionChainRow] = {}
        for contract in contracts:
            row = rows.setdefault(contract.strike, OptionChainRow(strike=contract.strike))
            if contract.option_type == OptionType.CE:
                row.ce = contract
            else:
                row.pe = contract

        quotes: dict[str, OptionQuote] = {}
        available = False
        message = MARKET_DATA_UNAVAILABLE
        spot_source = "REQUEST" if spot else None
        if self.quote_provider is not None:
            try:
                quotes = self.quote_provider.quotes([c.instrument_key for c in contracts])
                available = True
                message = None
                if spot is None and contracts and contracts[0].underlying_key:
                    spot = self.quote_provider.spot(contracts[0].underlying_key)
                    spot_source = "MARKET_DATA" if spot else None
            except Exception as exc:
                quotes, available = {}, False
                message = f"{MARKET_DATA_UNAVAILABLE} ({type(exc).__name__})"

        for row in rows.values():
            if row.ce is not None:
                row.ce_quote = quotes.get(row.ce.instrument_key)
            if row.pe is not None:
                row.pe_quote = quotes.get(row.pe.instrument_key)

        ordered = [rows[k] for k in sorted(rows)]
        if spot is None and ordered:
            spot = demo_reference_spot(r.strike for r in ordered)
            spot_source = DEMO_REFERENCE_SPOT
        atm = atm_strike([r.strike for r in ordered], spot) if ordered and spot else None
        return OptionChain(
            underlying=self.resolver._norm(underlying),
            expiry=expiry,
            rows=ordered,
            lot_size=contracts[0].lot_size if contracts else None,
            market_data_available=available,
            market_data_message=message,
            atm_strike=atm,
            spot=spot,
            spot_source=spot_source,
            source=self.resolver.metadata.get("source"),
            metadata=self.resolver.metadata,
        )
