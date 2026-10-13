"""Live S2 signal accuracy tracker (observation only)."""
from __future__ import annotations

import ast
import json
import sqlite3
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from src.database import Database
from src.paper_trading import signal_tracker as st
from src.paper_trading.signal_tracker import (
    AMBIGUOUS, INVALID, LOSS, PENDING, WIN, LiveSignalTracker, SignalRecord, compute_stats, write_daily_report,
)

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 10, 5)
KEY = "NSE_FO|TEST"


def at(hh, mm, day=DAY):
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=IST)


def record(hh=10, mm=0, direction="BUY", entry=100.0, stop=90.0, target=120.0, qty=75, **kw):
    return SignalRecord.create(
        timestamp=at(hh, mm), underlying="NIFTY", direction=direction, active_from=at(hh, mm) + timedelta(minutes=5),
        option_type="CE" if direction == "BUY" else "PE", strike=25000.0, expiry="2026-10-06",
        instrument_key=kw.pop("instrument_key", KEY), trading_symbol="NIFTY TEST", signal_price=25010.0,
        entry_price=entry, stop_loss=stop, target=target, quantity=qty, **kw,
    )


def candle(hh, mm, o, h, l, c):
    return {"timestamp": at(hh, mm), "open": o, "high": h, "low": l, "close": c}


@pytest.fixture()
def db():
    return Database(":memory:")


@pytest.fixture()
def tracker(db, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    return LiveSignalTracker(db, square_off="15:15")


def rows(db):
    return db.get_live_signals(DAY.isoformat())


def stat_rows(*statuses, pnl=None):
    out = []
    for i, s in enumerate(statuses):
        p = pnl[i] if pnl else (100.0 if s == WIN else -50.0 if s == LOSS else None)
        out.append({"signal_id": str(i), "timestamp": f"2026-10-05T04:{i:02d}:00+00:00", "status": s, "pnl": p,
                    "r_multiple": (1.0 if s == WIN else -1.0) if s in (WIN, LOSS) else None,
                    "exit_time": f"2026-10-05T05:{i:02d}:00+00:00", "resolved_at": None})
    return out


# 1-6 accuracy -------------------------------------------------------------

def test_win_increases_accuracy():
    base = compute_stats(stat_rows(WIN, LOSS))
    after = compute_stats(stat_rows(WIN, LOSS, WIN))
    assert after["accuracy"] > base["accuracy"]


def test_loss_decreases_accuracy():
    base = compute_stats(stat_rows(WIN, LOSS))
    after = compute_stats(stat_rows(WIN, LOSS, LOSS))
    assert after["accuracy"] < base["accuracy"]


def test_pending_does_not_affect_accuracy():
    base = compute_stats(stat_rows(WIN, LOSS, WIN))
    after = compute_stats(stat_rows(WIN, LOSS, WIN, PENDING, PENDING))
    assert after["accuracy"] == base["accuracy"] and after["resolved"] == 3 and after["pending"] == 2


def test_ambiguous_does_not_affect_accuracy():
    base = compute_stats(stat_rows(WIN, LOSS, WIN))
    after = compute_stats(stat_rows(WIN, LOSS, WIN, AMBIGUOUS, INVALID))
    assert after["accuracy"] == base["accuracy"] and after["ambiguous"] == 1 and after["invalid"] == 1


def test_18_wins_10_losses_is_64_2857():
    s = compute_stats(stat_rows(*([WIN] * 18 + [LOSS] * 10)))
    assert s["resolved"] == 28
    assert s["accuracy"] == pytest.approx(64.2857, abs=1e-4)
    assert s["accuracy_display"] == "64.29%"


def test_zero_resolved_is_na():
    s = compute_stats(stat_rows(PENDING, AMBIGUOUS))
    assert s["accuracy"] is None and s["accuracy_display"] == "N/A" and s["profit_factor"] is None
    assert compute_stats([])["accuracy_display"] == "N/A"


def test_risk_metrics():
    s = compute_stats(stat_rows(WIN, LOSS, LOSS, WIN, WIN, WIN, pnl=[100, -50, -70, 30, 40, 10]))
    assert s["gross_profit"] == 180 and s["gross_loss"] == 120 and s["net_pnl"] == 60
    assert s["profit_factor"] == pytest.approx(1.5)
    assert s["max_drawdown"] == 120
    assert s["max_consecutive_wins"] == 3 and s["max_consecutive_losses"] == 2
    assert s["average_r"] == pytest.approx(2 / 6, abs=1e-4)


# 7-8 outcome resolution ----------------------------------------------------

def test_target_hit_is_win(tracker, db):
    tracker.register(record())
    assert tracker.on_candle(KEY, candle(10, 5, 101, 110, 95, 108)) == []
    out = tracker.on_candle(KEY, candle(10, 10, 108, 121, 104, 119))
    assert out[0]["status"] == WIN and out[0]["exit_price"] == 120
    row = rows(db)[0]
    assert row["status"] == WIN and row["pnl"] == pytest.approx(20 * 75) and row["r_multiple"] == pytest.approx(2.0)
    assert row["reason"] == "TARGET" and row["resolved_at"]


def test_stop_hit_is_loss(tracker, db):
    tracker.register(record(direction="SELL"))
    out = tracker.on_candle(KEY, candle(10, 5, 99, 101, 89, 92))
    assert out[0]["status"] == LOSS
    row = rows(db)[0]
    assert row["exit_price"] == 90 and row["pnl"] == pytest.approx(-750) and row["r_multiple"] == pytest.approx(-1.0)


def test_gap_beyond_stop_fills_at_open(tracker, db):
    tracker.register(record())
    tracker.on_candle(KEY, candle(10, 5, 85, 125, 80, 120))
    row = rows(db)[0]
    assert row["status"] == LOSS and row["exit_price"] == 85 and row["reason"] == "STOP_LOSS_GAP"


def test_neither_level_stays_pending(tracker, db):
    tracker.register(record())
    tracker.on_candle(KEY, candle(10, 5, 101, 115, 92, 110))
    assert rows(db)[0]["status"] == PENDING
    assert compute_stats(rows(db))["accuracy_display"] == "N/A"


def test_invalid_levels_marked_invalid(tracker, db):
    tracker.register(record(stop=105.0))
    tracker.register(record(hh=11, entry=None, stop=None, target=None), invalid_reason="NO_PREMIUM_ENTRY_CANDLE")
    statuses = {r["status"] for r in rows(db)}
    assert statuses == {INVALID}
    assert {r["reason"] for r in rows(db)} == {"INVALID_LEVELS", "NO_PREMIUM_ENTRY_CANDLE"}


# 9 no look-ahead -----------------------------------------------------------

def test_no_look_ahead(tracker, db):
    tracker.register(record())
    # Signal-bar candle (before entry) touches the target: must be ignored.
    assert tracker.on_candle(KEY, candle(10, 0, 100, 130, 99, 125)) == []
    tracker.on_candle(KEY, candle(10, 10, 101, 110, 95, 105))
    # An older candle arriving after a newer one is ignored.
    assert tracker.on_candle(KEY, candle(10, 5, 101, 130, 99, 125)) == []
    assert rows(db)[0]["status"] == PENDING
    # Other instruments never affect the signal.
    assert tracker.on_candle("NSE_FO|OTHER", candle(10, 15, 101, 130, 80, 125)) == []
    assert rows(db)[0]["status"] == PENDING


# 10-11 duplicates and immutability -------------------------------------------

def test_duplicate_signal_does_not_duplicate_outcome(tracker, db):
    assert tracker.register(record()) is True
    assert tracker.register(record()) is False
    assert tracker.register(record(entry=101.0)) is False
    assert len(rows(db)) == 1
    tracker.on_candle(KEY, candle(10, 5, 101, 121, 99, 119))
    assert db.resolve_live_signal(rows(db)[0]["signal_id"], {"status": LOSS, "reason": "X"}) is False
    tracker2 = LiveSignalTracker(db, square_off="15:15")
    assert tracker2.register(record()) is False
    assert len(rows(db)) == 1 and rows(db)[0]["status"] == WIN


def test_record_immutable_after_creation(tracker, db):
    rec = record()
    with pytest.raises(FrozenInstanceError):
        rec.entry_price = 1.0
    tracker.register(rec)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.connection.execute("UPDATE live_signal_outcomes SET entry_price=1 WHERE signal_id=?", (rec.signal_id,))
    stored = rows(db)[0]
    assert stored["entry_price"] == 100.0
    assert SignalRecord.from_json(stored["record_json"]) == rec


def test_pending_reloaded_after_restart(db, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    LiveSignalTracker(db).register(record())
    restarted = LiveSignalTracker(db)
    assert restarted.on_candle(KEY, candle(10, 5, 101, 121, 99, 119))[0]["status"] == WIN


# 12-13 safety ----------------------------------------------------------------

FORBIDDEN = ("src.broker", "src.engine", "src.orders", "src.execution", "order_manager", "TradingEngine")


@pytest.mark.parametrize("path", ["src/paper_trading/signal_tracker.py", "scripts/run_live_signal_accuracy.py",
                                  "scripts/run_live_premium_signal_accuracy.py"])
def test_no_broker_order_is_called(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in imported if any(f in m for f in FORBIDDEN)]
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"place_order", "submit_entry", "submit_signal", "cancel_order", "modify_order"}


def test_tracker_does_not_touch_orders_or_trades(tracker, db):
    tracker.register(record())
    tracker.on_candle(KEY, candle(10, 5, 101, 121, 99, 119))
    counts = db.counts()
    assert counts["orders"] == 0 and counts["trades"] == 0 and counts["positions"] == 0


def test_live_trading_stays_disabled(db, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    t = LiveSignalTracker(db)
    assert t.safety == {"trading_mode": "PAPER", "live_trading_enabled": False, "paper_mode_ok": True,
                        "places_orders": False}
    import os
    assert os.environ["LIVE_TRADING_ENABLED"] == "false"
    # Even with the flag on by mistake the tracker only observes.
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    t2 = LiveSignalTracker(db)
    assert t2.safety["paper_mode_ok"] is False and t2.safety["places_orders"] is False
    t2.register(record())
    t2.on_candle(KEY, candle(10, 5, 101, 121, 99, 119))
    assert db.counts()["orders"] == 0


# 14-15 EOD and ambiguity -------------------------------------------------------

def test_eod_square_off_uses_actual_exit_price(tracker, db):
    tracker.register(record(hh=14, mm=0))
    tracker.register(record(hh=14, mm=30))
    tracker.on_candle(KEY, candle(15, 10, 104, 110, 95, 106))
    out = tracker.on_candle(KEY, candle(15, 15, 106, 125, 85, 107))
    by_time = {r["timestamp"]: r for r in rows(db)}
    assert len(out) == 2
    assert all(r["status"] == WIN and r["exit_price"] == 106 and r["reason"] == "EOD_SQUARE_OFF" for r in by_time.values())
    assert rows(db)[0]["pnl"] == pytest.approx(6 * 75)


def test_eod_losing_exit_and_no_evidence_stays_pending(tracker, db):
    tracker.register(record(hh=14, mm=0))
    tracker.register(record(hh=14, mm=30, instrument_key="NSE_FO|NO_DATA"))
    tracker.on_candle(KEY, candle(15, 15, 95, 97, 94, 96))
    assert tracker.finalize_session(DAY) == 1
    by_key = {r["instrument_key"]: r for r in rows(db)}
    assert by_key[KEY]["status"] == LOSS and by_key[KEY]["exit_price"] == 95
    assert by_key["NSE_FO|NO_DATA"]["status"] == PENDING
    assert by_key["NSE_FO|NO_DATA"]["reason"] == "SESSION_ENDED_WITHOUT_EXIT_EVIDENCE"


def test_same_candle_sl_and_target_is_ambiguous(tracker, db):
    tracker.register(record())
    out = tracker.on_candle(KEY, candle(10, 5, 100, 125, 85, 110))
    assert out[0]["status"] == AMBIGUOUS and out[0]["pnl"] is None
    s = compute_stats(rows(db))
    assert s["ambiguous"] == 1 and s["resolved"] == 0 and s["accuracy_display"] == "N/A"


# reports and API ------------------------------------------------------------

def test_daily_report_files(tracker, db, tmp_path):
    tracker.register(record())
    tracker.register(record(hh=11))
    tracker.on_candle(KEY, candle(10, 5, 101, 121, 99, 119))
    paths = write_daily_report(rows(db), DAY, tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert paths["json"].name == "2026-10-05.json" and paths["csv"].name == "2026-10-05.csv"
    for key in ("total", "resolved", "pending", "wins", "losses", "ambiguous", "accuracy", "profit_factor",
                "net_pnl", "max_drawdown"):
        assert key in data
    assert data["total"] == 2 and data["wins"] == 1 and data["pending"] == 1 and data["accuracy"] == 100.0
    assert "accuracy=100.0" in paths["csv"].read_text(encoding="utf-8").splitlines()[0]


@pytest.fixture()
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("DASHBOARD_API_KEY", "test-dashboard-key")
    monkeypatch.setenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", "secret-token-value")
    from src.api import app as api_module
    api_module.database = Database(str(tmp_path / "api.db"))
    return TestClient(api_module.app), api_module.database


def test_api_endpoints_read_only_and_no_secrets(api):
    client, database = api
    t = LiveSignalTracker(database)
    t.register(record())
    t.on_candle(KEY, candle(10, 5, 101, 121, 99, 119))
    headers = {"X-API-Key": "test-dashboard-key"}
    stats = client.get("/api/signals/live/stats?date=2026-10-05", headers=headers)
    lst = client.get("/api/signals/live?date=2026-10-05", headers=headers)
    assert stats.status_code == 200 and lst.status_code == 200
    body = stats.json()
    assert body["wins"] == 1 and body["accuracy"] == 100.0 and body["live_trading_enabled"] is False
    assert lst.json()["signals"][0]["status"] == WIN
    assert "record_json" not in lst.json()["signals"][0]
    for text in (stats.text, lst.text):
        assert "secret-token-value" not in text and "ACCESS_TOKEN" not in text
    empty = client.get("/api/signals/live/stats?date=2026-01-01", headers=headers).json()
    assert empty["accuracy"] is None and empty["accuracy_display"] == "N/A"
    assert client.post("/api/signals/live", headers=headers).status_code == 405
    assert client.get("/api/signals/live/stats?date=bad", headers=headers).status_code == 400


# premium-replay registration timing ------------------------------------------------

LIVE_DAY = date(2026, 10, 6)
PREMIUM_KEY = "NSE_FO|40710"


def _load_premium_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("premium_accuracy_under_test",
                                                  ROOT / "scripts" / "run_live_premium_signal_accuracy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _premium_candles(entry_low=40.0, include_entry=True):
    from src.historical_data.models import HistoricalCandle
    out = []
    ts = datetime(2026, 10, 6, 9, 15, tzinfo=IST)
    while ts <= datetime(2026, 10, 6, 11, 5, tzinfo=IST):
        out.append(HistoricalCandle(ts, PREMIUM_KEY, 50.0, 52.0, 48.0, 50.0, 0.0))
        ts += timedelta(minutes=5)
    if include_entry:
        out.append(HistoricalCandle(at(11, 10, LIVE_DAY), PREMIUM_KEY, 41.05, 42.0, 40.0, 41.0, 0.0))
        out.append(HistoricalCandle(at(11, 15, LIVE_DAY), PREMIUM_KEY, 41.0, 43.0, entry_low, 41.0, 0.0))
    return out


@pytest.fixture()
def premium_run(db, monkeypatch, tmp_path):
    """run_once with real option_levels/premium_frame/tracker; only market-data access is faked."""
    from types import SimpleNamespace

    import pandas as pd

    from src.backtest.research_strategies import ResearchSignal
    from src.config.loader import load_strategy_config
    from src.config.research_loader import load_research_config

    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    script = _load_premium_script()
    sig = ResearchSignal(at(11, 5, LIVE_DAY), "BUY", "S2", "NIFTY", 22682.95, {})
    contract = SimpleNamespace(instrument_key=PREMIUM_KEY, trading_symbol="NIFTY 22700 CE 06 OCT 26",
                               strike=22700.0, option_type="CE", expiry=LIVE_DAY, lot_size=65, tick_size=0.05)
    spot = pd.DataFrame([{"timestamp": pd.Timestamp(at(9, 15, LIVE_DAY))}])
    monkeypatch.setattr(script, "validate_intraday", lambda raw, day, interval: (spot, SimpleNamespace(received=1)))
    monkeypatch.setattr(script, "point_in_time_signals", lambda *a, **k: ([sig], True))
    monkeypatch.setattr(script, "resolve_signal_contract", lambda *a, **k: contract)
    service = SimpleNamespace(
        research_config=load_research_config(ROOT / "config" / "strategy_research.yaml"),
        strategy_config=load_strategy_config(ROOT / "config" / "strategy.yaml"),
        resolver_factory=lambda day: None,
        store=SimpleNamespace(read=lambda *a, **k: pd.DataFrame()),
    )
    tracker = LiveSignalTracker(db, square_off="15:15")

    def run(now, candles):
        client = SimpleNamespace(intraday_candles=lambda key, interval=5: candles if key == PREMIUM_KEY else [])
        return script.run_once(day=LIVE_DAY, database=db, tracker=tracker, service=service, client=client,
                               out_dir=tmp_path, now=now)

    return run


def test_incomplete_entry_candle_is_deferred_not_invalid(premium_run, db):
    candles = _premium_candles()
    result = premium_run(datetime(2026, 10, 6, 11, 13, 59, tzinfo=IST), candles)
    assert result["deferred_signals"] == 1 and result["new_signals"] == 0
    assert db.get_live_signals(LIVE_DAY.isoformat()) == []


def test_deferred_signal_registers_with_levels_once_entry_candle_completes(premium_run, db):
    candles = _premium_candles()
    premium_run(datetime(2026, 10, 6, 11, 13, 59, tzinfo=IST), candles)
    result = premium_run(datetime(2026, 10, 6, 11, 21, tzinfo=IST), candles)
    assert result["deferred_signals"] == 0 and result["new_signals"] == 1
    [row] = db.get_live_signals(LIVE_DAY.isoformat())
    assert row["status"] == PENDING and row["reason"] is None
    assert (row["entry_price"], row["stop_loss"], row["target"]) == (41.05, 35.05, 53.05)
    assert row["active_from"] == at(11, 10, LIVE_DAY).astimezone(ZoneInfo("UTC")).isoformat()


def test_missing_completed_entry_candle_stays_invalid(premium_run, db):
    result = premium_run(datetime(2026, 10, 6, 11, 30, tzinfo=IST), _premium_candles(include_entry=False))
    assert result["deferred_signals"] == 0 and result["new_signals"] == 1
    [row] = db.get_live_signals(LIVE_DAY.isoformat())
    assert row["status"] == INVALID and row["reason"] == "NO_PREMIUM_ENTRY_CANDLE"
    assert row["entry_price"] is None


def test_stop_loss_resolution_unchanged_after_deferral(premium_run, db):
    candles = _premium_candles(entry_low=35.0)
    premium_run(datetime(2026, 10, 6, 11, 13, 59, tzinfo=IST), candles)
    premium_run(datetime(2026, 10, 6, 11, 21, tzinfo=IST), candles)
    [row] = db.get_live_signals(LIVE_DAY.isoformat())
    assert row["status"] == LOSS and row["reason"] == "STOP_LOSS" and row["exit_price"] == 35.05
    assert row["exit_time"] == at(11, 15, LIVE_DAY).astimezone(ZoneInfo("UTC")).isoformat()
    assert row["pnl"] == pytest.approx(-6.0 * 65) and row["r_multiple"] == pytest.approx(-1.0)


def test_runner_hook_only_observes(monkeypatch):
    """The runner registers signals before submit; the tracker hook itself has no order path."""
    source = (ROOT / "src/paper_trading/nifty_s2_runner.py").read_text(encoding="utf-8")
    hook = source[source.index("def _track_signal"):source.index("def _manage_open_positions")]
    assert "submit_entry" not in hook and "place_order" not in hook and "broker" not in hook
    assert st.__doc__ and "never places" in st.__doc__
