from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from scripts.pe_recovery_backtest import simulate_pe_exit

IST = ZoneInfo("Asia/Kolkata")


def _candle(h, m, close, low=None):
    ts = datetime(2026, 10, 21, h, m, tzinfo=IST)
    return SimpleNamespace(timestamp=ts, close=close, low=low if low is not None else close)


def test_simulate_pe_exit_hits_stop():
    entry_ts = datetime(2026, 10, 21, 12, 40, tzinfo=IST)
    candles = [_candle(12, 40, 85.0), _candle(12, 41, 80.0, low=64.0)]
    out = simulate_pe_exit(
        candles, entry_ts=entry_ts, entry_premium=85.0, stop_premium=65.0,
    )
    assert out is not None
    assert out[0] == 65.0
    assert out[2] == "pe_stop (premium)"
