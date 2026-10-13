"""S2 forward paper validation: recording, outcomes, statistics, API and safety."""
from __future__ import annotations

import dataclasses
import inspect
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.backtest.current_day_s2 import LevelParams
from src.backtest.research_strategies import ResearchSignal
from src.database import Database
from src.options.models import OptionResolutionError
from src.paper_trading import forward_validation as fv
from src.paper_trading.forward_validation import (
    DATA_UNAVAILABLE,
    ForwardSignal,
    ForwardValidationTracker,
    accuracy_label,
    build_signals,
    daily_stats,
    forward_stats,
    resolve_outcome,
    sample_status,
)
from src.paper_trading.signal_tracker import AMBIGUOUS, INVALID, LOSS, PENDING, WIN

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 10, 6)
START = datetime(2026, 10, 6, 9, 15, tzinfo=IST)
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="15:00", atr_period=3,
                     stop_atr_multiplier=1.5, target_r=2.0, slippage_bps=0.0)
S2_CFG = {"range_start": "09:15", "range_end": "09:30", "atr_period": 14}
SQUARE_OFF = time(15, 15)

REQUIRED_FIELDS = (
    "timestamp", "direction", "underlying_price", "trading_symbol", "expiry", "dte", "entry_price", "premium_atr",
    "premium_atr_pct", "stop_loss", "target", "entry_time", "rsi", "ema20", "ema50", "ema_alignment",
    "breakout_distance", "breakout_distance_atr", "option_vwap", "option_vs_vwap", "market_regime",
    "signal_number_of_day", "same_direction_number",
)


def at(i: int) -> datetime:
    return START + timedelta(minutes=5 * i)


@dataclass(frozen=True)
class Contract:
    instrument_key: str
    option_type: str
    strike: float = 25000.0
    expiry: date = date(2026, 10, 7)
    lot_size: int = 65
    tick_size: float = 0.05

    @property
    def trading_symbol(self) -> str:
        return f"NIFTY {self.strike:g} {self.option_type} 07 OCT 26"


CE, PE = Contract("NSE_FO|CE1", "CE"), Contract("NSE_FO|PE1", "PE")


def underlying_frame(bars: int = 40) -> pd.DataFrame:
    rows = []
    for i in range(bars):
        price = 25000 + 4 * i
        rows.append({"timestamp": at(i), "open": price, "high": price + 6, "low": price - 6, "close": price + 2,
                     "volume": 0, "symbol": "NIFTY"})
    return pd.DataFrame(rows)


def premium(bars: int = 40, base: float = 100.0) -> pd.DataFrame:
    rows = []
    for i in range(bars):
        p = base + 0.5 * (i % 3)
        rows.append({"timestamp": at(i), "open": p, "high": p + 1.0, "low": p - 1.0, "close": p + 0.5,
                     "volume": 100 + i})
    return pd.DataFrame(rows)


def signal(i: int, direction: str = "BUY", spot: float | None = None) -> ResearchSignal:
    spot = spot if spot is not None else 25002 + 4 * i
    key = "above" if direction == "BUY" else "below"
    return ResearchSignal(timestamp=at(i), direction=direction, strategy_id="S2", underlying="NIFTY", spot=spot,
                          conditions={key: spot - 10 if direction == "BUY" else spot + 10})


def build(signals, prem=None, under=None, resolve=None, regime=lambda _ts: {"range_bound"}):
    prem = premium() if prem is None else prem
    return build_signals(
        signals, under if under is not None else underlying_frame(),
        resolve=resolve or (lambda s: CE if s.direction == "BUY" else PE),
        premium_for=lambda c: prem if not isinstance(prem, dict) else prem[c.instrument_key],
        params=PARAMS, s2_cfg=S2_CFG, regime_at=regime,
    )


def record(**over) -> ForwardSignal:
    base = dict(
        signal_id="sig-1", timestamp=at(10).isoformat(), session_date=DAY.isoformat(), strategy="S2",
        underlying="NIFTY", direction="BUY", underlying_price=25040.0, breakout_level=25030.0,
        instrument_key=CE.instrument_key, trading_symbol=CE.trading_symbol, option_type="CE", strike=25000.0,
        expiry="2026-10-07", dte=1, quantity=65, entry_time=at(11).isoformat(), entry_price=100.0,
        premium_atr=4.0, premium_atr_pct=4.0, stop_loss=94.0, target=112.0, rsi=60.0, ema20=25030.0,
        ema50=25010.0, ema_alignment="BULLISH (ALIGNED)", breakout_distance=10.0, breakout_distance_atr=0.8,
        option_vwap=99.0, option_vs_vwap="ABOVE", option_vs_vwap_pct=1.0, market_regime="none",
        signal_number_of_day=1, same_direction_number=1, premium_source="UPSTOX_V3_INTRADAY", data_reason=None,
    )
    base.update(over)
    return ForwardSignal(**base)


def bar(i: int, o: float, h: float, l: float, c: float) -> dict:
    return {"timestamp": pd.Timestamp(at(i)), "open": o, "high": h, "low": l, "close": c, "volume": 10}


def frame(*bars) -> pd.DataFrame:
    return pd.DataFrame(list(bars))


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "fv.db"))


# ------------------------------------------------------------------ sample status
@pytest.mark.parametrize("resolved,label", [
    (0, "INSUFFICIENT SAMPLE"), (29, "INSUFFICIENT SAMPLE"), (30, "EARLY SAMPLE"), (99, "EARLY SAMPLE"),
    (100, "PRELIMINARY"), (199, "PRELIMINARY"), (200, "MEANINGFUL SAMPLE"), (1000, "MEANINGFUL SAMPLE"),
])
def test_sample_status_thresholds(resolved, label):
    assert sample_status(resolved) == label


def test_accuracy_not_called_strategy_accuracy_below_100():
    assert "not strategy accuracy" in accuracy_label(99)
    assert accuracy_label(100) == "Strategy accuracy"


# ------------------------------------------------------------------ recording
def test_signal_snapshot_has_every_required_field():
    names = {f.name for f in dataclasses.fields(ForwardSignal)}
    assert set(REQUIRED_FIELDS) <= names
    rec = build([signal(6)])[0]
    assert rec.direction == "BUY" and rec.option_type == "CE" and rec.expiry == "2026-10-07" and rec.dte == 1
    assert rec.entry_time == at(7).isoformat() and rec.entry_price == premium().loc[7, "open"]
    assert rec.premium_atr and rec.premium_atr_pct == pytest.approx(rec.premium_atr / rec.entry_price * 100, abs=1e-3)
    assert rec.stop_loss < rec.entry_price < rec.target
    assert rec.option_vwap is not None and rec.option_vs_vwap in ("ABOVE", "BELOW", "AT")
    assert rec.market_regime == "range_bound" and rec.breakout_distance == pytest.approx(10.0)
    assert rec.initial_status == PENDING


def test_sell_maps_to_pe_and_all_repeated_signals_are_numbered():
    recs = build([signal(6), signal(8), signal(10, "SELL"), signal(12)])
    assert [r.option_type for r in recs] == ["CE", "CE", "PE", "CE"]
    assert [r.signal_number_of_day for r in recs] == [1, 2, 3, 4]
    assert [r.same_direction_number for r in recs] == [1, 2, 1, 3]
    assert len({r.signal_id for r in recs}) == 4


def test_features_and_levels_have_no_look_ahead():
    base = build([signal(10)])[0]
    under, prem = underlying_frame(), premium()
    later = under["timestamp"] > at(10)
    under.loc[later, ["open", "high", "low", "close"]] = 99999.0
    future = prem["timestamp"] > at(10)
    prem.loc[future, ["high", "close"]] = 500.0
    prem.loc[future, "low"] = 1.0
    prem.loc[future, "volume"] = 10**6
    again = build([signal(10)], prem=prem, under=under)[0]
    for name in ("rsi", "ema20", "ema50", "breakout_distance_atr", "option_vwap", "premium_atr", "stop_loss",
                 "target", "entry_price"):
        assert getattr(again, name) == getattr(base, name), name


def test_duplicate_signals_cannot_be_inserted_twice(db):
    tracker = ForwardValidationTracker(db)
    recs = build([signal(6), signal(8)])
    assert tracker.record_all(recs) == 2
    assert tracker.record_all(build([signal(6), signal(8)])) == 0
    assert len(db.get_forward_signals()) == 2
    row = recs[0].to_row()
    with pytest.raises(sqlite3.IntegrityError):
        db.connection.execute(
            "INSERT INTO forward_validation_signals (signal_id, timestamp, session_date, strategy, underlying, "
            "direction, record_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
            ("other-id", row["timestamp"], row["session_date"], row["strategy"], row["underlying"],
             row["direction"], "{}", row["created_at"]))


def test_records_are_immutable_and_cannot_be_deleted(db):
    tracker = ForwardValidationTracker(db)
    tracker.record(record())
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        db.connection.execute("UPDATE forward_validation_signals SET record_json='{}'")
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        db.connection.execute("DELETE FROM forward_validation_signals")


# ------------------------------------------------------------------ data unavailable
def test_missing_option_candles_marked_data_unavailable_without_outcome(db):
    recs = build([signal(6)], prem=pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]))
    rec = recs[0]
    assert rec.initial_status == DATA_UNAVAILABLE and rec.data_reason == "NO_PREMIUM_DATA"
    assert rec.entry_price is None and rec.stop_loss is None and rec.target is None
    tracker = ForwardValidationTracker(db)
    tracker.record(rec)
    stored = db.get_forward_signals()[0]
    assert stored["status"] == DATA_UNAVAILABLE and stored["pnl"] is None and stored["r_multiple"] is None
    assert tracker.resolve_pending(DAY.isoformat(), {CE.instrument_key: premium()}, True) == []


def test_premium_api_error_and_missing_entry_candle_are_data_unavailable():
    api = build_signals([signal(6)], underlying_frame(), resolve=lambda s: CE,
                        premium_for=lambda c: "PREMIUM_API_ERROR:HTTP_500", params=PARAMS, s2_cfg=S2_CFG,
                        regime_at=lambda _t: [])[0]
    assert api.initial_status == DATA_UNAVAILABLE
    short = build([signal(6)], prem=premium(bars=7))[0]
    assert short.data_reason == "NO_PREMIUM_ENTRY_CANDLE" and short.initial_status == DATA_UNAVAILABLE
    warmup = build([signal(1)])[0]
    assert warmup.data_reason == "INSUFFICIENT_ATR_HISTORY" and warmup.initial_status == DATA_UNAVAILABLE


def test_unresolved_contract_is_data_unavailable():
    def fail(_s):
        raise OptionResolutionError("NO_EXPIRY", "no expiry")
    rec = build([signal(6)], resolve=fail)[0]
    assert rec.instrument_key is None and rec.initial_status == DATA_UNAVAILABLE


def test_pending_without_exit_candles_becomes_data_unavailable_only_after_session(db):
    tracker = ForwardValidationTracker(db)
    tracker.record(record())
    assert tracker.resolve_pending(DAY.isoformat(), {}, session_complete=False) == []
    assert db.get_forward_signals()[0]["status"] == PENDING
    out = tracker.resolve_pending(DAY.isoformat(), {}, session_complete=True)
    assert out[0]["status"] == DATA_UNAVAILABLE and out[0]["reason"] == "NO_EXIT_EVIDENCE"
    assert db.get_forward_signals()[0]["pnl"] is None


# ------------------------------------------------------------------ outcomes
def test_target_stop_ambiguous_and_eod_outcomes():
    rec = record()
    win = resolve_outcome(rec, frame(bar(11, 100, 105, 99, 104), bar(12, 104, 113, 103, 110)), SQUARE_OFF, False)
    assert win["status"] == WIN and win["exit_price"] == 112.0 and win["r_multiple"] == 2.0
    assert win["pnl"] == pytest.approx(12 * 65) and win["reason"] == "TARGET"
    loss = resolve_outcome(rec, frame(bar(11, 100, 101, 93, 95)), SQUARE_OFF, False)
    assert loss["status"] == LOSS and loss["pnl"] == pytest.approx(-6 * 65) and loss["r_multiple"] == -1.0
    amb = resolve_outcome(rec, frame(bar(11, 100, 113, 93, 100)), SQUARE_OFF, False)
    assert amb["status"] == AMBIGUOUS and amb["pnl"] is None
    eod_rec = record(entry_time=datetime(2026, 10, 6, 15, 10, tzinfo=IST).isoformat())
    eod_bar = {"timestamp": pd.Timestamp(datetime(2026, 10, 6, 15, 15, tzinfo=IST)), "open": 103.0, "high": 104.0,
               "low": 102.0, "close": 103.0, "volume": 1}
    eod = resolve_outcome(eod_rec, frame(eod_bar), SQUARE_OFF, False)
    assert eod["status"] == WIN and eod["reason"] == "EOD_SQUARE_OFF" and eod["exit_price"] == 103.0


def test_candles_before_entry_are_ignored():
    rec = record()
    early = bar(10, 100, 200, 1, 100)
    assert resolve_outcome(rec, frame(early), SQUARE_OFF, False) is None


def test_outcome_recorded_once(db):
    tracker = ForwardValidationTracker(db)
    tracker.record(record())
    candles = {CE.instrument_key: frame(bar(11, 100, 101, 93, 95))}
    assert len(tracker.resolve_pending(DAY.isoformat(), candles, False)) == 1
    assert tracker.resolve_pending(DAY.isoformat(),
                                   {CE.instrument_key: frame(bar(11, 100, 120, 99, 115))}, False) == []
    row = db.get_forward_signals()[0]
    assert row["status"] == LOSS and row["exit_reason"] == "STOP_LOSS"


def test_invalid_levels_are_invalid_not_outcomes(db):
    rec = record(stop_loss=101.0)
    assert rec.initial_status == INVALID
    ForwardValidationTracker(db).record(rec)
    assert db.get_forward_signals()[0]["status"] == INVALID


# ------------------------------------------------------------------ statistics
def _seed(db):
    tracker = ForwardValidationTracker(db)
    specs = [  # (id, day, minute index, direction, dte, regime, candle)
        ("a", DAY, 10, "BUY", 1, "none", bar(11, 100, 113, 99, 110)),
        ("b", DAY, 20, "BUY", 1, "range_bound", bar(21, 100, 101, 93, 95)),
        ("c", DAY, 30, "SELL", 1, "none", bar(31, 100, 101, 93, 95)),
        ("d", date(2026, 10, 7), 10, "SELL", 0, "none", None),
    ]
    for sid, day, i, direction, dte, regime, candle in specs:
        ts = at(i) + timedelta(days=(day - DAY).days)
        rec = record(signal_id=sid, timestamp=ts.isoformat(), session_date=day.isoformat(), direction=direction,
                     dte=dte, market_regime=regime, entry_time=(ts + timedelta(minutes=5)).isoformat(),
                     instrument_key=f"K{sid}", option_type="CE" if direction == "BUY" else "PE",
                     data_reason=None if candle else "NO_PREMIUM_DATA")
        tracker.record(rec)
        if candle:
            tracker.resolve_pending(day.isoformat(), {f"K{sid}": frame(candle)}, False)
    return db.get_forward_signals()


def test_cumulative_and_breakdown_statistics(db):
    rows = _seed(db)
    stats = forward_stats(rows, DAY.isoformat())
    assert stats["signals"] == 4 and stats["resolved"] == 3 and stats["wins"] == 1 and stats["losses"] == 2
    assert stats["data_unavailable"] == 1
    assert stats["accuracy"] == pytest.approx(100 / 3, abs=1e-3)
    assert stats["net_pnl"] == pytest.approx((12 - 6 - 6) * 65)
    assert stats["profit_factor"] == pytest.approx(1.0)
    assert stats["max_drawdown"] == pytest.approx(12 * 65)
    assert stats["sample_status"] == "INSUFFICIENT SAMPLE" and not stats["accuracy_is_strategy_accuracy"]
    assert stats["days_tracked"] == 2 and stats["today"]["signals"] == 3
    assert stats["by_direction"]["BUY"]["accuracy"] == 50.0 and stats["by_direction"]["SELL"]["accuracy"] == 0.0
    assert stats["by_dte"]["1 DTE"]["resolved"] == 3 and stats["by_dte"]["0 DTE"]["data_unavailable"] == 1
    assert stats["by_regime"]["range_bound"]["losses"] == 1
    assert set(stats["by_time_of_day"]) >= {"10:00-11:00", "11:00-12:00"}
    assert stats["historical_comparable_1dte"] == "Historical comparable 1-DTE data: UNAVAILABLE"


def test_daily_stats_cumulative_pnl(db):
    days = daily_stats(_seed(db))
    assert [d["session_date"] for d in days] == ["2026-10-06", "2026-10-07"]
    assert days[0]["signals"] == 3 and days[0]["wins"] == 1 and days[0]["losses"] == 2
    assert days[0]["accuracy"] == pytest.approx(100 / 3, abs=1e-3)
    assert days[1]["data_unavailable"] == 1 and days[1]["accuracy"] is None
    assert days[1]["cumulative_pnl"] == days[0]["net_pnl"] == 0.0


def test_reports_written(db, tmp_path):
    paths = fv.write_reports(_seed(db), tmp_path / "out", DAY.isoformat(), ["note"])
    for path in paths.values():
        assert Path(path).exists()
    md = paths["md"].read_text(encoding="utf-8")
    assert "INSUFFICIENT SAMPLE" in md and "UNAVAILABLE" in md and "not strategy accuracy" in md
    header = paths["signals_csv"].read_text(encoding="utf-8").splitlines()[0]
    for col in ("premium_atr_pct", "ema_alignment", "option_vs_vwap", "same_direction_number", "pnl_1_lot"):
        assert col in header


# ------------------------------------------------------------------ isolation and safety
def test_forward_validation_does_not_touch_live_signal_benchmark(db):
    from src.paper_trading.signal_tracker import LiveSignalTracker, SignalRecord
    live = LiveSignalTracker(db)
    live.register(SignalRecord.create(timestamp=at(10), underlying="NIFTY", direction="BUY", active_from=at(11),
                                      instrument_key="K", entry_price=100.0, stop_loss=94.0, target=112.0,
                                      quantity=65))
    before = db.get_live_signals()
    _seed(db)
    assert db.get_live_signals() == before


def test_module_has_no_order_path():
    source = inspect.getsource(fv)
    for forbidden in ("place_order", "TradingEngine", "OrderManager", "create_broker", "submit_entry"):
        assert forbidden not in source
    assert ForwardValidationTracker(Database(":memory:")).safety["places_orders"] is False


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_KEY", "test-dashboard-key")
    monkeypatch.setenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", "secret-token-value")
    from src.api import app as api_module
    api_module.database = Database(str(tmp_path / "api.db"))
    return TestClient(api_module.app), api_module.database


def test_api_endpoints(api, monkeypatch):
    client, database = api
    monkeypatch.setenv("DASHBOARD_API_AUTH_REQUIRED", "true")
    _seed(database)
    headers = {"X-API-Key": "test-dashboard-key"}
    lst = client.get("/api/forward-validation", headers=headers)
    stats = client.get(f"/api/forward-validation/stats?date={DAY.isoformat()}", headers=headers)
    daily = client.get("/api/forward-validation/daily", headers=headers)
    assert lst.status_code == stats.status_code == daily.status_code == 200
    signals = lst.json()["signals"]
    assert len(signals) == 4 and "record_json" not in signals[0]
    assert {"outcome", "pnl_1_lot", "r_multiple", "exit_time", "exit_reason"} <= set(signals[0])
    body = stats.json()
    assert body["live_trading_enabled"] is False and body["observation_only"] is True
    assert body["sample_status"] == "INSUFFICIENT SAMPLE"
    assert body["historical_comparable_1dte"] == "Historical comparable 1-DTE data: UNAVAILABLE"
    assert body["today"]["signals"] == 3
    assert len(daily.json()["days"]) == 2
    one_day = client.get("/api/forward-validation?date=2026-10-07", headers=headers).json()
    assert len(one_day["signals"]) == 1
    for resp in (lst, stats, daily):
        assert "secret-token-value" not in resp.text and "ACCESS_TOKEN" not in resp.text
    assert client.post("/api/forward-validation", headers=headers).status_code == 405
    assert client.get("/api/forward-validation/stats?date=bad", headers=headers).status_code == 400
    assert client.get("/api/forward-validation/stats").status_code in (401, 403)
