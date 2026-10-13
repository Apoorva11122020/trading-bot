"""Option premium coverage workflow for strategy research."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.research_strategies import ResearchSignal
from src.historical_data.models import HistoricalCandle, HistoricalDataError
from src.historical_data.option_coverage import (
    extract_required_contracts,
    has_stored_option_data,
    resolve_signal_contract,
    signal_has_premium_coverage,
)
from src.historical_data.storage import ParquetCandleStore
from src.options.instrument_resolver import OptionInstrumentResolver, StaticContractSource

from tests.options_support import contracts, strategy_stack

IST = ZoneInfo("Asia/Kolkata")
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _signal(ts: datetime, direction: str = "BUY", spot: float = 23000.0) -> ResearchSignal:
    return ResearchSignal(ts, direction, "S2", "NIFTY", spot, {"orb_breakout": True})


def test_extract_required_contracts_unique():
    config, _, _, _, _ = strategy_stack()
    research = __import__("src.config.research_loader", fromlist=["load_research_config"]).load_research_config(
        ROOT / "config" / "strategy_research.yaml"
    )
    resolver = OptionInstrumentResolver(StaticContractSource(contracts()), today=lambda: date(2026, 10, 1))
    signals = [
        _signal(datetime(2026, 10, 1, 10, 0, tzinfo=IST), "BUY", 23000),
        _signal(datetime(2026, 10, 1, 11, 0, tzinfo=IST), "SELL", 22950),
    ]
    required, errors = extract_required_contracts(
        signals,
        lambda d: resolver,
        research,
        date(2026, 10, 1),
        date(2026, 10, 1),
    )
    assert not errors
    assert len(required) >= 1
    keys = {c.instrument_key for c in required}
    assert len(keys) == len(required)


def test_missing_contract_detection(tmp_path):
    store = ParquetCandleStore(tmp_path)
    from src.historical_data.option_coverage import RequiredOptionContract

    contract = RequiredOptionContract(
        instrument_key="NSE_FO|999",
        underlying="NIFTY",
        expiry=date(2026, 10, 6),
        strike=23000,
        option_type="CE",
        trading_symbol="TEST",
        lot_size=65,
        freeze_quantity=None,
        tick_size=0.05,
        from_date=date(2026, 10, 1),
        to_date=date(2026, 10, 1),
    )
    assert not has_stored_option_data(store, contract, contract.from_date, contract.to_date)


def test_coverage_calculation(tmp_path):
    from src.historical_data.models import candles_to_frame
    from src.historical_data.option_coverage import RequiredOptionContract

    store = ParquetCandleStore(tmp_path)
    contract = RequiredOptionContract(
        instrument_key="NSE_FO|40865",
        underlying="NIFTY",
        expiry=date(2026, 10, 6),
        strike=25000,
        option_type="CE",
        trading_symbol="NIFTY 25000 CE 06 OCT 26",
        lot_size=65,
        freeze_quantity=None,
        tick_size=0.05,
        from_date=date(2026, 10, 1),
        to_date=date(2026, 10, 1),
        signal_count=1,
        signal_times=(datetime(2026, 10, 1, 10, 0, tzinfo=IST).isoformat(),),
    )
    rows = []
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    for i in range(6):
        price = 100 + i
        rows.append(HistoricalCandle(
            start.replace(hour=9, minute=15 + 5 * i),
            contract.instrument_key, price, price + 1, price - 1, price, 10, 1,
        ))
    frame = candles_to_frame(rows, underlying="NIFTY", expiry=contract.expiry.isoformat(),
                             strike=contract.strike, option_type="CE")
    store.write("options", "NIFTY", contract.instrument_key, frame, expiry=contract.expiry)
    signal = _signal(datetime(2026, 10, 1, 9, 30, tzinfo=IST))
    ok, reason = signal_has_premium_coverage(store, contract, signal, entry_timing="next_open")
    assert ok, reason


def test_unavailable_contract_raises_on_verify(tmp_path):
    from src.historical_data.option_coverage import RequiredOptionContract, verify_option_premium

    store = ParquetCandleStore(tmp_path)
    contract = RequiredOptionContract(
        instrument_key="NSE_FO|missing",
        underlying="NIFTY",
        expiry=date(2026, 10, 6),
        strike=23000,
        option_type="CE",
        trading_symbol="X",
        lot_size=65,
        freeze_quantity=None,
        tick_size=0.05,
        from_date=date(2026, 10, 1),
        to_date=date(2026, 10, 1),
    )
    with pytest.raises(HistoricalDataError) as exc:
        verify_option_premium(store, contract, contract.from_date, contract.to_date)
    assert exc.value.code == "OPTION_DATA_MISSING"
