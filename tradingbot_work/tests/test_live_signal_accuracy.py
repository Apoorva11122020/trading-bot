"""Realtime S2 signal-accuracy tracker (PAPER only). Synthetic ticks are test fixtures only."""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.research_strategies import generate_signals
from src.market_data.models import Tick
from src.market_data.paper import PaperMarketDataFeed
from src.safety.live_gate import LiveTradingSafetyError
from src.signal_accuracy.realtime import LiveSignalAccuracyRunner, TrackerSettings
from src.signal_accuracy.tracker import (
    AMBIGUOUS,
    LOSS,
    SIGNAL_GENERATED,
    STOP_HIT,
    TARGET_HIT,
    TIME_EXIT,
    TRACKING,
    UNRESOLVED,
    WIN,
    LevelConvention,
    OutcomeTracker,
    new_signal,
    pct,
    signal_id_for,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
IST = ZoneInfo("Asia/Kolkata")
KEY = "NSE_INDEX|Nifty 50"
SCRIPT = ROOT / "scripts" / "run_live_signal_accuracy.py"
CONV = LevelConvention(entry_timing="next_open", stop_atr_multiplier=1.5, target_r=2.0,
                       square_off=time(15, 15), interval_minutes=5)


def at(h, m, s=0):
    return datetime(2026, 10, 1, h, m, s, tzinfo=IST)


def signal(direction="BUY", h=10, m=0, close=22000.0, atr=10.0):
    candle = {"timestamp": at(h, m), "open": close - 5, "high": close + 5, "low": close - 10, "close": close}
    return new_signal(session_id="t", candle=candle, direction=direction, atr=atr, convention=CONV,
                      instrument="NIFTY", timeframe="5m", strategy="S2")


def tracked(direction="BUY", entry=22000.0):
    """Entry 22000, stop distance 15: BUY SL 21985 / TGT 22030, SELL SL 22015 / TGT 21970."""
    t = OutcomeTracker(CONV)
    ev = signal(direction)
    t.register(ev)
    t.on_tick(at(10, 5), entry)
    return t, ev


@pytest.fixture(autouse=True)
def paper_env(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def make_runner(tmp_path, *, session_id="t1", start=None, stale=600):
    feed = PaperMarketDataFeed()
    settings = replace(TrackerSettings.load(), stale_data_seconds=stale)
    clock = Clock(start or at(9, 10))
    lines: list[str] = []
    runner = LiveSignalAccuracyRunner(feed, settings=settings, out_dir=tmp_path, session_id=session_id,
                                      clock=clock, echo=lines.append)
    runner.start()
    return runner, feed, clock, lines


def tick(feed, clock, ts, price):
    clock.now = ts
    feed.publish(Tick(symbol=KEY, timestamp=ts, ltp=price))


def bar(feed, clock, h, m, o, hi, lo, c):
    base = at(h, m)
    for sec, price in ((0, o), (60, hi), (120, lo), (240, c)):
        tick(feed, clock, base + timedelta(seconds=sec), price)


INSIDE = [(9, 30), (9, 35), (9, 40), (9, 45), (9, 50), (9, 55), (10, 0), (10, 5), (10, 10), (10, 15), (10, 20)]


def drive_to_breakout(feed, clock):
    """Opening range 22900-22950, 11 inside candles, then a 10:25 close above the range (not yet completed)."""
    for m in (15, 20, 25):
        bar(feed, clock, 9, m, 22920, 22950, 22900, 22925)
    for h, m in INSIDE:
        bar(feed, clock, h, m, 22920, 22945, 22905, 22930)
    bar(feed, clock, 10, 25, 22930, 23010, 22925, 23000)


# 1-2 PAPER / LIVE_TRADING_ENABLED enforcement ------------------------------------

def test_refuses_to_start_unless_paper_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    with pytest.raises(LiveTradingSafetyError):
        LiveSignalAccuracyRunner(PaperMarketDataFeed(), out_dir=tmp_path)


def test_refuses_to_start_when_live_trading_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(LiveTradingSafetyError):
        LiveSignalAccuracyRunner(PaperMarketDataFeed(), out_dir=tmp_path)


def _load_script():
    spec = importlib.util.spec_from_file_location("run_live_signal_accuracy_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("env", [("TRADING_MODE", "LIVE"), ("LIVE_TRADING_ENABLED", "true")])
def test_script_fails_closed_before_connecting(monkeypatch, env, capsys):
    script = _load_script()
    monkeypatch.setattr(script, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(script, "create_feed", lambda **_: pytest.fail("feed must not be created"))
    monkeypatch.setenv(*env)
    assert script.main(["--seconds", "5"]) == 2
    assert "REFUSED TO START" in capsys.readouterr().out


def test_script_duration_options():
    script = _load_script()
    parser = script.build_parser()
    now = at(10, 0)
    assert script.deadline_for(parser.parse_args(["--seconds", "3600"]), now, "15:30") == at(11, 0)
    assert script.deadline_for(parser.parse_args(["--minutes", "60"]), now, "15:30") == at(11, 0)
    assert script.deadline_for(parser.parse_args(["--until", "14:30"]), now, "15:30") == at(14, 30)
    assert script.deadline_for(parser.parse_args([]), now, "15:30") == at(15, 30)
    with pytest.raises(ValueError):
        script.deadline_for(parser.parse_args(["--until", "09:00"]), now, "15:30")


# 3 completed-candle signal generation -------------------------------------------

def test_signal_only_after_candle_completes(tmp_path):
    runner, feed, clock, _ = make_runner(tmp_path)
    drive_to_breakout(feed, clock)
    assert runner.tracker.signals == {}
    assert runner.dq.completed_candles == 14

    tick(feed, clock, at(10, 30), 23002.0)
    assert runner.dq.completed_candles == 15
    [ev] = runner.tracker.signals.values()
    assert ev.direction == "BUY" and ev.strategy_name == "S2" and ev.timeframe == "5m"
    assert ev.timestamp_ist == at(10, 25).isoformat()
    assert (ev.candle_open, ev.candle_high, ev.candle_low, ev.candle_close) == (22930, 23010, 22925, 23000)
    assert ev.signal_price == 23000.0
    assert ev.status == TRACKING
    assert ev.entry_price == 23002.0 and ev.entry_time == at(10, 30).isoformat()
    assert ev.stop_price == pytest.approx(23002.0 - ev.stop_distance, abs=0.01)
    assert ev.target_price == pytest.approx(23002.0 + 2 * ev.stop_distance, abs=0.01)
    assert ev.outcome is None and ev.exit_price is None


def test_signal_matches_existing_s2_generator(tmp_path):
    runner, feed, clock, _ = make_runner(tmp_path)
    drive_to_breakout(feed, clock)
    tick(feed, clock, at(10, 30), 23002.0)
    frame = pd.DataFrame(list(runner.history))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST)
    expected = [(s.timestamp, s.direction) for s in generate_signals("S2", frame, "NIFTY", runner.research)]
    got = [(datetime.fromisoformat(e.timestamp_ist), e.direction) for e in runner.tracker.signals.values()]
    assert got == expected


def test_no_signal_without_complete_opening_range(tmp_path):
    runner, feed, clock, _ = make_runner(tmp_path, start=at(9, 22))
    drive_to_breakout(feed, clock)
    tick(feed, clock, at(10, 30), 23002.0)
    assert runner.tracker.signals == {}
    assert runner.dq.incomplete_candles >= 1
    assert runner.dq.opening_range_incomplete >= 1


def test_ticks_outside_session_are_ignored(tmp_path):
    runner, feed, clock, _ = make_runner(tmp_path, start=at(9, 0))
    tick(feed, clock, at(9, 5), 22900.0)
    tick(feed, clock, at(9, 10), 22910.0)
    assert runner.dq.outside_session_ticks == 2
    assert runner.dq.completed_candles == 0


# 4 duplicate prevention -----------------------------------------------------------

def test_duplicate_signal_is_not_registered_twice(tmp_path):
    t = OutcomeTracker(CONV)
    assert t.register(signal("BUY")) is True
    assert t.register(signal("BUY")) is False
    assert t.register(signal("SELL")) is True
    assert len(t.signals) == 2

    runner, feed, clock, _ = make_runner(tmp_path)
    drive_to_breakout(feed, clock)
    tick(feed, clock, at(10, 30), 23002.0)
    assert runner._evaluate(dict(runner.history[-1])) == "BUY_DUPLICATE"
    assert len(runner.tracker.signals) == 1


def test_signal_id_is_deterministic():
    a = signal_id_for("S2", "NIFTY", "5m", at(10, 25), "BUY")
    assert a == signal_id_for("s2", "NIFTY", "5m", at(10, 25), "buy")
    assert a != signal_id_for("S2", "NIFTY", "5m", at(10, 25), "SELL")
    assert a != signal_id_for("S2", "NIFTY", "5m", at(10, 30), "BUY")


# 5-8 target / stop -------------------------------------------------------------------

def test_buy_target_hit():
    t, ev = tracked("BUY")
    assert (ev.stop_price, ev.target_price, ev.risk_points) == (21985.0, 22030.0, 15.0)
    t.on_tick(at(10, 6), 22010.0)
    assert ev.status == TRACKING
    t.on_tick(at(10, 7), 22031.0)
    assert (ev.status, ev.outcome, ev.exit_price, ev.pnl_points) == (TARGET_HIT, WIN, 22030.0, 30.0)
    assert ev.outcome_timestamp == at(10, 7).isoformat()


def test_buy_stop_hit():
    t, ev = tracked("BUY")
    t.on_tick(at(10, 6), 21990.0)
    t.on_tick(at(10, 7), 21984.0)
    assert (ev.status, ev.outcome, ev.exit_price, ev.pnl_points) == (STOP_HIT, LOSS, 21984.0, -16.0)


def test_sell_target_hit():
    t, ev = tracked("SELL")
    assert (ev.stop_price, ev.target_price) == (22015.0, 21970.0)
    t.on_tick(at(10, 6), 21980.0)
    t.on_tick(at(10, 7), 21969.0)
    assert (ev.status, ev.outcome, ev.exit_price, ev.pnl_points) == (TARGET_HIT, WIN, 21970.0, 30.0)


def test_sell_stop_hit():
    t, ev = tracked("SELL")
    t.on_tick(at(10, 6), 22016.0)
    assert (ev.status, ev.outcome, ev.exit_price, ev.pnl_points) == (STOP_HIT, LOSS, 22016.0, -16.0)


def test_first_touch_decides_in_tick_order():
    t, ev = tracked("BUY")
    t.on_tick(at(10, 6), 21985.0)
    t.on_tick(at(10, 7), 22040.0)
    assert ev.status == STOP_HIT and ev.exit_price == 21985.0


# 9 time exit / 10 unresolved ------------------------------------------------------------

def test_time_exit_at_first_price_after_square_off():
    t, ev = tracked("BUY")
    t.on_tick(at(15, 14, 59), 22005.0)
    assert ev.status == TRACKING
    t.on_tick(at(15, 15), 22008.0)
    assert (ev.status, ev.outcome, ev.exit_price, ev.pnl_points, ev.reason) == (TIME_EXIT, WIN, 22008.0, 8.0,
                                                                                   "SQUARE_OFF")


def test_no_entry_before_square_off_is_unresolved():
    t = OutcomeTracker(CONV)
    ev = signal("BUY", h=15, m=10)
    t.register(ev)
    t.on_tick(at(15, 15), 22000.0)
    assert ev.status == UNRESOLVED and ev.entry_price is None


def test_unresolved_signal_excluded_from_accuracy():
    t, ev = tracked("BUY")
    t.on_tick(at(10, 6), 22010.0)
    t.finalize(at(11, 0))
    assert (ev.status, ev.outcome, ev.exit_price, ev.pnl_points) == (UNRESOLVED, UNRESOLVED, None, None)
    s = summarize([ev], session_date="2026-10-01", strategy="S2", instrument="NIFTY", timeframe="5m")
    assert s["unresolved_signals"] == 1 and s["resolved_signals"] == 0
    assert s["accuracy"] is None


def test_missing_atr_history_is_unresolved_not_guessed():
    t = OutcomeTracker(CONV)
    ev = signal("BUY", atr=None)
    t.register(ev)
    assert ev.status == UNRESOLVED and ev.reason == "INSUFFICIENT_ATR_HISTORY"
    assert ev.stop_price is None and ev.target_price is None


# 11 ambiguous ---------------------------------------------------------------------------

def test_stop_and_target_in_same_candle_is_ambiguous():
    t = OutcomeTracker(CONV)
    ev = signal("BUY")
    t.register(ev)
    t.on_bar({"timestamp": at(10, 5), "open": 22000.0, "high": 22035.0, "low": 21980.0, "close": 22000.0})
    assert (ev.status, ev.outcome, ev.exit_price, ev.pnl_points) == (AMBIGUOUS, AMBIGUOUS, None, None)
    s = summarize([ev], session_date="2026-10-01", strategy="S2", instrument="NIFTY", timeframe="5m")
    assert s["accuracy"] is None and s["ambiguous"] == 1
    assert s["accuracy_conservative"] == 0.0


def test_candle_path_single_touch_resolves():
    t = OutcomeTracker(CONV)
    ev = signal("SELL")
    t.register(ev)
    t.on_bar({"timestamp": at(10, 5), "open": 22000.0, "high": 22010.0, "low": 21965.0, "close": 21970.0})
    assert (ev.status, ev.exit_price, ev.pnl_points) == (TARGET_HIT, 21970.0, 30.0)


# 12-13 MFE / MAE --------------------------------------------------------------------------

def test_mfe_for_buy():
    t, ev = tracked("BUY")
    for minute, price in ((6, 22012.0), (7, 21995.0), (8, 22020.0), (9, 22031.0)):
        t.on_tick(at(10, minute), price)
    assert ev.status == TARGET_HIT
    assert ev.mfe_points == 31.0
    assert ev.mae_points == 5.0


def test_mae_for_sell():
    t, ev = tracked("SELL")
    for minute, price in ((6, 22010.0), (7, 21990.0), (8, 22016.0)):
        t.on_tick(at(10, minute), price)
    assert ev.status == STOP_HIT
    assert ev.mae_points == 16.0
    assert ev.mfe_points == 10.0


# 14-15 accuracy ------------------------------------------------------------------------------

def _finished(direction, status, outcome, pnl, minute):
    ev = signal(direction, h=10, m=minute)
    ev.status, ev.outcome, ev.pnl_points = status, outcome, pnl
    ev.outcome_timestamp = at(11, minute).isoformat()
    ev.mfe_points, ev.mae_points = (max(pnl, 0.0), max(-pnl, 0.0)) if pnl is not None else (None, None)
    return ev


def test_accuracy_formula():
    rows = [
        _finished("BUY", TARGET_HIT, WIN, 30.0, 0),
        _finished("BUY", STOP_HIT, LOSS, -16.0, 5),
        _finished("SELL", TARGET_HIT, WIN, 30.0, 10),
        _finished("SELL", TARGET_HIT, WIN, 30.0, 15),
        _finished("BUY", AMBIGUOUS, AMBIGUOUS, None, 20),
        _finished("SELL", UNRESOLVED, UNRESOLVED, None, 25),
    ]
    s = summarize(rows, session_date="2026-10-01", strategy="S2", instrument="NIFTY", timeframe="5m")
    assert (s["total_signals"], s["buy_signals"], s["sell_signals"]) == (6, 3, 3)
    assert (s["wins"], s["losses"], s["ambiguous"], s["unresolved_signals"]) == (3, 1, 1, 1)
    assert s["resolved_signals"] == 5
    assert s["accuracy"] == 75.0
    assert s["accuracy_conservative"] == 60.0
    assert s["buy_accuracy"] == 50.0 and s["sell_accuracy"] == 100.0
    assert s["total_points"] == 74.0
    assert s["average_winning_points"] == 30.0 and s["average_losing_points"] == -16.0
    assert s["profit_factor"] == 5.625
    assert s["max_drawdown_points"] == 16.0


def test_zero_signals_accuracy_is_na():
    s = summarize([], session_date="2026-10-01", strategy="S2", instrument="NIFTY", timeframe="5m")
    assert s["total_signals"] == 0
    assert s["accuracy"] is None and s["buy_accuracy"] is None and s["sell_accuracy"] is None
    assert pct(s["accuracy"]) == "N/A"
    assert s["profit_factor"] is None


# 16 no future leakage ----------------------------------------------------------------------------

def test_no_future_leakage(tmp_path):
    t = OutcomeTracker(CONV)
    ev = signal("BUY")
    t.register(ev)
    t.on_tick(at(10, 4, 59), 99999.0)
    assert ev.status == SIGNAL_GENERATED and ev.entry_price is None

    snapshots = []
    for i, later in enumerate((23100.0, 22800.0)):
        runner, feed, clock, _ = make_runner(tmp_path / str(i))
        drive_to_breakout(feed, clock)
        tick(feed, clock, at(10, 30), 23002.0)
        [sig] = runner.tracker.signals.values()
        snapshot = {k: getattr(sig, k) for k in ("signal_id", "timestamp_ist", "signal_price", "atr",
                                                  "stop_distance", "entry_price", "stop_price", "target_price")}
        tick(feed, clock, at(10, 31), later)
        snapshots.append(snapshot)
        assert {k: getattr(sig, k) for k in snapshot} == snapshot
    assert snapshots[0] == snapshots[1]


# 17 no order placement ------------------------------------------------------------------------------

def test_tracker_never_places_orders(tmp_path, monkeypatch):
    def _fail(*_a, **_k):
        raise AssertionError("place_order must never be called")

    monkeypatch.setattr("src.broker.paper.PaperBroker.place_order", _fail)
    runner, feed, clock, _ = make_runner(tmp_path)
    drive_to_breakout(feed, clock)
    tick(feed, clock, at(10, 30), 23002.0)
    tick(feed, clock, at(10, 31), 23500.0)
    summary, paths = runner.finish()
    assert runner.dq.orders_placed == 0
    assert json.loads(paths["summary_json"].read_text())["safety"]["orders_placed"] == 0


FORBIDDEN = ("src.broker", "src.engine", "src.orders", "src.execution", "src.paper_trading", "order_manager",
             "TradingEngine")


@pytest.mark.parametrize("path", ["src/signal_accuracy/tracker.py", "src/signal_accuracy/realtime.py",
                                  "scripts/run_live_signal_accuracy.py"])
def test_no_order_code_imported_or_called(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in imported if any(f in m for f in FORBIDDEN)]
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"place_order", "submit_entry", "submit_signal", "cancel_order", "modify_order"}


def test_order_modules_not_loaded_transitively():
    code = (
        "import sys, importlib.util\n"
        "import src.signal_accuracy.realtime, src.signal_accuracy.tracker\n"
        f"spec = importlib.util.spec_from_file_location('tracker_script', r'{SCRIPT}')\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "bad = sorted(m for m in sys.modules if m.startswith(('src.broker', 'src.execution', 'src.engine', "
        "'src.paper_trading')))\n"
        "print(bad); sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


# 18 restart / deduplication -----------------------------------------------------------------------------

def test_restart_preserves_records_without_duplicates(tmp_path):
    first, feed, clock, _ = make_runner(tmp_path, session_id="20261001")
    drive_to_breakout(feed, clock)
    tick(feed, clock, at(10, 30), 23002.0)
    [original] = first.tracker.signals.values()
    assert original.status == TRACKING

    second, feed2, clock2, lines = make_runner(tmp_path, session_id="20261001", start=at(10, 31))
    [restored] = second.tracker.signals.values()
    assert restored.signal_id == original.signal_id
    assert restored.status == UNRESOLVED and restored.reason == "DATA_GAP:PROCESS_RESTART"
    assert second.dq.restored_signals == 1 and len(second.history) == 15
    assert any(line.startswith("[RESTART]") for line in lines)
    assert second._evaluate(dict(second.history[-1])) == "BUY_DUPLICATE"

    _, paths = second.finish()
    session = json.loads(paths["session_json"].read_text())
    assert [s["signal_id"] for s in session["signals"]] == [original.signal_id]
    csv_rows = paths["signals_csv"].read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_rows) == 2
    kinds = [json.loads(line)["type"] for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()]
    assert kinds.count("SIGNAL") == 1


# 19-20 stale data / disconnect ------------------------------------------------------------------------------

def _runner_with_open_signal(tmp_path, stale):
    runner, feed, clock, _ = make_runner(tmp_path, start=at(10, 4), stale=stale)
    ev = signal("BUY")
    runner.tracker.register(ev)
    tick(feed, clock, at(10, 5), 22000.0)
    assert ev.status == TRACKING
    return runner, feed, clock, ev


def test_stale_feed_marks_open_signals_unresolved(tmp_path):
    runner, feed, clock, ev = _runner_with_open_signal(tmp_path, stale=15)
    tick(feed, clock, at(10, 5, 5), 22005.0)
    runner.poll(at(10, 5, 25))
    assert runner.dq.stale_events == 1 and runner.dq.data_gaps == 1
    assert ev.status == UNRESOLVED and ev.reason == "DATA_GAP:STALE_FEED"
    tick(feed, clock, at(10, 5, 30), 22100.0)
    assert ev.status == UNRESOLVED and ev.exit_price is None
    tick(feed, clock, at(10, 10), 22100.0)
    assert runner.dq.incomplete_candles == 1 and runner.dq.completed_candles == 0


def test_tick_gap_is_a_data_gap(tmp_path):
    runner, feed, clock, ev = _runner_with_open_signal(tmp_path, stale=15)
    tick(feed, clock, at(10, 5, 40), 21900.0)
    assert runner.dq.data_gaps == 1
    assert ev.status == UNRESOLVED and ev.reason == "DATA_GAP:TICK_GAP"


def test_disconnect_marks_open_signals_unresolved(tmp_path):
    runner, feed, clock, ev = _runner_with_open_signal(tmp_path, stale=600)
    clock.now = at(10, 5, 10)
    feed.disconnect()
    runner.poll()
    assert runner.dq.disconnect_events == 1
    assert ev.status == UNRESOLVED and ev.reason == "DATA_GAP:DISCONNECTED"
    feed.connect()
    tick(feed, clock, at(10, 5, 20), 22100.0)
    assert ev.status == UNRESOLVED and ev.exit_price is None


# reports -------------------------------------------------------------------------------------------------------

def test_reports_written_with_na_accuracy_when_no_signals(tmp_path):
    runner, feed, clock, _ = make_runner(tmp_path, session_id="empty")
    for m in (15, 20):
        bar(feed, clock, 9, m, 22920, 22950, 22900, 22925)
    summary, paths = runner.finish()
    for key in ("jsonl", "session_json", "signals_csv", "summary_json", "summary_md"):
        assert paths[key].exists()
    assert paths["jsonl"].name == "session_empty.jsonl"
    data = json.loads(paths["summary_json"].read_text())
    assert data["session"]["total_signals"] == 0 and data["session"]["accuracy"] is None
    assert data["data_quality"]["completed_candles"] == 1
    assert data["safety"]["trading_mode"] == "PAPER" and data["safety"]["live_trading_enabled"] is False
    assert "| Accuracy | N/A |" in paths["summary_md"].read_text(encoding="utf-8")
