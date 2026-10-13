from __future__ import annotations

import importlib.util
import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest import current_day_s2 as cd
from src.config.loader import load_strategy_config
from src.config.research_loader import load_research_config
from src.historical_data.models import HistoricalCandle
from src.historical_data.validation import expected_slots
from src.options.models import OptionResolutionError
from src.session.session_manager import SessionManager

ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 10, 5)
KEY = "NSE_INDEX|Nifty 50"
OPT = "NSE_FO|777"
RESEARCH = load_research_config(ROOT / "config" / "strategy_research.yaml")
STRATEGY = load_strategy_config(ROOT / "config" / "strategy.yaml")
PARAMS = cd.LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=14,
                        stop_atr_multiplier=1.5, target_r=2.0, slippage_bps=0.0)


def bars(day: date = DAY, key: str = KEY, base: float = 22500.0, breakout_at: int | None = 6,
         flat_last: int = 0) -> list[HistoricalCandle]:
    out = []
    slots = expected_slots(day)
    for i, ts in enumerate(slots):
        price = base + (i % 3)
        if breakout_at is not None and i >= breakout_at:
            price = base + 50 + i
        if i >= len(slots) - flat_last:
            out.append(HistoricalCandle(ts.to_pydatetime(), key, price, price, price, price, 0.0, 0.0))
        else:
            out.append(HistoricalCandle(ts.to_pydatetime(), key, price - 1, price + 2, price - 2, price, 0.0, 0.0))
    return out


def option_bars(day: date = DAY, *, skip: set[int] = frozenset()) -> list[HistoricalCandle]:
    return [HistoricalCandle(ts.to_pydatetime(), OPT, 100 + i, 104 + i, 97 + i, 101 + i, 500.0, 0.0)
            for i, ts in enumerate(expected_slots(day)) if i not in skip]


def contract():
    return SimpleNamespace(instrument_key=OPT, trading_symbol="NIFTY 22550 CE 06 OCT 26", strike=22550.0,
                           option_type=SimpleNamespace(value="CE"), expiry=date(2026, 10, 6), tick_size=0.05)


def signals_for(candles):
    clean, _ = cd.validate_intraday(candles, DAY)
    return cd.point_in_time_signals(clean, "NIFTY", RESEARCH, SessionManager(STRATEGY.session), "14:30")


# ------------------------------------------------------------ candle checks
def test_descending_input_sorted_and_reported_ist():
    clean, v = cd.validate_intraday(list(reversed(bars())), DAY)
    assert v.received_order == "descending" and v.non_ist_timestamps == 0
    assert clean["timestamp"].is_monotonic_increasing
    assert str(clean["timestamp"].dt.tz) == "Asia/Kolkata"
    assert (v.valid_candles, v.expected_candles, v.missing_candles) == (75, 75, [])
    assert v.first.endswith("09:15:00+05:30") and v.last.endswith("15:25:00+05:30")


def test_duplicates_detected_and_removed():
    data = bars()
    clean, v = cd.validate_intraday(data + data[:3], DAY)
    assert v.duplicates == 3 and len(clean) == 75


def test_flat_candles_kept_unaltered():
    data = bars(flat_last=2)
    clean, v = cd.validate_intraday(data, DAY)
    assert len(v.flat_candles) == 2 and len(clean) == 75
    last = clean.iloc[-1]
    assert last["open"] == last["high"] == last["low"] == last["close"] == data[-1].close


def test_missing_slots_and_other_day_rows_reported():
    data = bars()[:-3] + bars(day=date(2026, 10, 1))[:2]
    clean, v = cd.validate_intraday(data, DAY)
    assert len(v.missing_candles) == 3 and v.wrong_day_rows == 2 and len(clean) == 72


# ---------------------------------------------------------- signal generation
def test_signals_point_in_time_match_full_run_and_use_existing_s2():
    signals, same = signals_for(bars())
    assert same and signals
    assert all(s.strategy_id == "S2" and s.direction == "BUY" for s in signals)
    assert pd.Timestamp(signals[0].timestamp).strftime("%H:%M") == "09:45"


def test_future_candles_do_not_change_earlier_signals():
    full, _ = signals_for(bars())
    truncated = [c for c in bars() if c.timestamp.hour < 12]
    early, _ = signals_for(truncated)
    cutoff = pd.Timestamp("2026-10-05 12:00", tz=cd.IST)
    assert [s.timestamp for s in early] == [s.timestamp for s in full if pd.Timestamp(s.timestamp) < cutoff]


def test_no_signals_without_breakout():
    signals, same = signals_for(bars(breakout_at=None))
    assert signals == [] and same


# ------------------------------------------------------------ option levels
def _resolve_one(premium, sig=None):
    sig = sig or signals_for(bars())[0][0]
    return cd.resolve_signals([sig], lambda s: contract(), lambda c: premium, PARAMS)[0]


def test_levels_match_engine_formulas():
    premium = cd.premium_frame(None, option_bars())
    sig = signals_for(bars())[0][15]
    row = _resolve_one(premium, sig)
    assert row["status"] == "RESOLVED"
    entry_ts = pd.Timestamp(sig.timestamp) + pd.Timedelta(minutes=5)
    entry_row = premium[premium["timestamp"] == entry_ts].iloc[0]
    assert row["entry"] == pytest.approx(entry_row["open"])
    from src.indicators.technical import atr
    from src.options.strategy import option_stop_loss, option_target
    p_atr = atr(premium.loc[: entry_row.name], 14).iloc[-1]
    stop = option_stop_loss(row["entry"], stop_mode="premium_atr", multiplier=1.5, premium_atr=p_atr, tick_size=0.05)
    assert row["stop_loss"] == stop
    assert row["target"] == option_target(row["entry"], stop, 2.0, 0.05)
    assert row["trading_symbol"] == "NIFTY 22550 CE 06 OCT 26" and row["option_type"] == "CE"


def test_missing_exact_signal_or_entry_candle_is_unresolved():
    sig = signals_for(bars())[0][0]
    idx = list(expected_slots(DAY)).index(pd.Timestamp(sig.timestamp))
    no_signal_bar = cd.premium_frame(None, option_bars(skip={idx}))
    no_entry_bar = cd.premium_frame(None, option_bars(skip={idx + 1}))
    assert _resolve_one(no_signal_bar, sig)["reason"] == "NO_PREMIUM_CANDLE_AT_SIGNAL"
    row = _resolve_one(no_entry_bar, sig)
    assert row["status"] == "UNRESOLVED" and row["reason"] == "NO_PREMIUM_ENTRY_CANDLE"
    assert row["entry"] is None and row["stop_loss"] is None


def test_insufficient_atr_history_unresolved():
    sig = signals_for(bars())[0][0]
    idx = list(expected_slots(DAY)).index(pd.Timestamp(sig.timestamp))
    premium = cd.premium_frame(None, option_bars(skip=set(range(0, idx - 3))))
    row = _resolve_one(premium, sig)
    assert row["status"] == "UNRESOLVED" and row["reason"] == "INSUFFICIENT_ATR_HISTORY"
    assert row["stop_loss"] is None and row["target"] is None


def test_contract_resolution_failure_and_api_error_unresolved():
    sig = signals_for(bars())[0][0]

    def fail(_):
        raise OptionResolutionError("NO_EXPIRY", "none")
    rows = cd.resolve_signals([sig], fail, lambda c: None, PARAMS)
    assert rows[0]["reason"] == "CONTRACT_UNRESOLVED:NO_EXPIRY"
    rows = cd.resolve_signals([sig], lambda s: contract(), lambda c: "PREMIUM_API_ERROR:AUTH_FAILED", PARAMS)
    assert rows[0]["status"] == "UNRESOLVED" and rows[0]["reason"] == "PREMIUM_API_ERROR:AUTH_FAILED"


def test_premium_frame_merges_real_sources_without_inventing_rows():
    stored = pd.DataFrame([{"timestamp": pd.Timestamp("2026-10-01 15:25", tz=cd.IST), "open": 90.0, "high": 91.0,
                            "low": 89.0, "close": 90.5, "volume": 10.0}])
    merged = cd.premium_frame(stored, option_bars())
    assert len(merged) == 76 and merged["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------- reports
def _script():
    spec = importlib.util.spec_from_file_location("cd_script", ROOT / "scripts" / "run_current_day_s2_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, data):
        self.data, self.calls = data, []

    def intraday_candles(self, key, *, unit="minutes", interval=5):
        self.calls.append(key)
        return self.data.get(key, [])


def test_end_to_end_report_no_win_rate_and_no_parquet_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    script = _script()
    monkeypatch.setattr(script, "resolve_signal_contract", lambda *a, **k: contract())
    from src.historical_data.service import HistoricalDataService
    service = HistoricalDataService(store_root=tmp_path / "store")
    client = FakeClient({KEY: list(reversed(bars())), OPT: option_bars()})
    report = script.run(day=DAY, service=service, client=client, out_dir=tmp_path / "out")
    out = capsys.readouterr().out
    assert "CURRENT DAY S2 RESEARCH COMPLETE" in out
    s = report["summary"]
    assert s["candle_count"] == 75 and s["signal_count"] == s["buy_signals"] + s["sell_signals"] > 0
    assert s["resolved_option_signals"] + s["unresolved_signals"] == s["signal_count"]
    assert s["point_in_time_matches_full_run"] is True
    text = json.dumps(report).lower()
    for banned in ("win_rate", "profit_factor", "net_pnl", "pnl"):
        assert banned not in text
    assert not (tmp_path / "store").exists() or not list((tmp_path / "store").rglob("*.parquet"))
    stem = tmp_path / "out" / "NIFTY_S2_current_day_2026-10-05"
    for ext in ("json", "csv", "md"):
        assert stem.with_suffix(f".{ext}").exists()
    assert client.calls[0] == KEY and set(client.calls) == {KEY, OPT}


def test_wrong_day_data_stops_without_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    script = _script()
    from src.historical_data.service import HistoricalDataService
    with pytest.raises(SystemExit):
        script.run(day=DAY, service=HistoricalDataService(store_root=tmp_path / "s"),
                   client=FakeClient({KEY: bars(day=date(2026, 10, 1))}), out_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("path", ["src/backtest/current_day_s2.py", "scripts/run_current_day_s2_research.py"])
def test_no_order_or_write_paths(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    for banned in ("place_order", "OrderManager", "TradingEngine", "api-sandbox", "store.write", ".to_parquet"):
        assert banned not in source
