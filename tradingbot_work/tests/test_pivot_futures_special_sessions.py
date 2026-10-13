from __future__ import annotations

from datetime import date

import pandas as pd

from src.historical_data import nifty_futures as nf
from src.historical_data.contracts import EXPIRED, HistoricalContract


def test_muhurat_session_uses_exchange_scheduled_12_candle_grid():
    day = date(2025, 10, 21)
    slots = nf.session_slots(day)

    assert len(slots) == 12
    assert slots[0] == pd.Timestamp("2025-10-21 13:45:00", tz="Asia/Kolkata")
    assert slots[-1] == pd.Timestamp("2025-10-21 14:40:00", tz="Asia/Kolkata")
    assert nf.session_slot_count(day) == 12


def test_muhurat_futures_bars_are_not_reported_as_missing_intervals():
    day = date(2025, 10, 21)
    slots = nf.session_slots(day)
    contract = HistoricalContract(
        instrument_key="NSE_FO|NIFTY_FUT_TEST",
        underlying="NIFTY",
        expiry=date(2025, 10, 28),
        instrument_type="FUT",
        lot_size=65,
        source=EXPIRED,
        trading_symbol="NIFTY FUT TEST",
    )
    frame = pd.DataFrame({
        "timestamp": slots,
        "open": [100.0] * len(slots),
        "high": [101.0] * len(slots),
        "low": [99.0] * len(slots),
        "close": [100.0] * len(slots),
        "volume": [1000.0] * len(slots),
        "open_interest": [10000.0] * len(slots),
    })

    assert nf.session_rows(contract, frame, [day], "FRONT_MONTH_CONTRACT") == []

    selection = [{
        "session_date": day.isoformat(),
        "status": nf.SELECTED,
        "instrument_key": contract.instrument_key,
    }]
    rows, summary = nf.alignment(selection, frame, frame, [day])
    assert summary["passed"]
    assert summary["exact_matches"] == 12
    assert summary["missing_futures"] == 0
    assert rows[0]["mismatched"] == 0
