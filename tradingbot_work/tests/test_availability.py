"""Historical availability discovery."""
from __future__ import annotations

from datetime import date

from src.historical_data.availability import AvailabilityRange, discover_research_range


class FakeClient:
    def candles(self, instrument_key, from_date, to_date, *, unit="minutes", interval=5):
        start = date.fromisoformat(str(from_date)[:10])
        if "INDEX" in instrument_key and start.year >= 2022:
            return [1] * 10
        if "NSE_FO" in instrument_key and start >= date(2026, 9, 1):
            return [1] * 5
        return []


class FakeResolver:
    def __init__(self):
        self.today = lambda: date(2026, 10, 1)

    def expiries(self, underlying):
        return [date(2026, 10, 27)]

    def strikes(self, underlying, expiry, option_type):
        return [56000.0]

    def resolve(self, underlying, expiry, strike, option_type):
        from src.options.models import OptionContract, OptionType
        return OptionContract(
            instrument_key="NSE_FO|1", trading_symbol="TEST", underlying=underlying,
            expiry=expiry, strike=strike, option_type=OptionType.CE, lot_size=30,
        )


def test_discover_research_range_overlap():
    info = discover_research_range(
        FakeClient(), "NIFTY", "NSE_INDEX|Nifty 50", FakeResolver(),
        requested_from=date(2022, 1, 1), requested_to=date(2026, 10, 5),
    )
    assert info.index_available_from == date(2022, 1, 1)
    assert info.option_available_from == date(2026, 9, 1)
    assert info.research_from == date(2026, 9, 1)
    assert "Plus" in info.limitation
