"""Live-day vs historical S2 analysis helpers (analysis only)."""
from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest import live_vs_historical as lvh
from src.backtest.current_day_s2 import LevelParams
from src.backtest.research_strategies import ResearchSignal

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 10, 5)
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=3,
                     stop_atr_multiplier=1.5, target_r=2.0, slippage_bps=0.0)
CONTRACT = SimpleNamespace(instrument_key="NSE_FO|X", trading_symbol="NIFTY 22600 CE 06 OCT 26", tick_size=0.05,
                           expiry=date(2026, 10, 6))


def at(hh, mm):
    return pd.Timestamp(datetime(2026, 10, 5, hh, mm, tzinfo=IST))


def premium(bars):
    """bars: list of (hh, mm, o, h, l, c); unit-volume test fixture."""
    return pd.DataFrame([{"timestamp": at(h, m), "open": o, "high": hi, "low": lo, "close": c, "volume": 1.0}
                         for h, m, o, hi, lo, c in bars])


def signal(hh, mm, direction="BUY"):
    return ResearchSignal(timestamp=at(hh, mm).to_pydatetime(), direction=direction, strategy_id="S2",
                          underlying="NIFTY", spot=22600.0, conditions={"above": 22590.0})


WARMUP = [(9, 15, 100, 102, 98, 100), (9, 20, 100, 102, 98, 100), (9, 25, 100, 102, 98, 100),
          (9, 30, 100, 102, 98, 100)]


def outcome(bars, hh=9, mm=30):
    rows = lvh.signal_outcomes([signal(hh, mm)], lambda s: CONTRACT, lambda c: premium(WARMUP + bars), PARAMS,
                               "15:15", "test")
    return rows[0]


# ------------------------------------------------------------------ statistics
def test_binomial_and_wilson():
    assert lvh.binom_cdf(0, 4, 0.25) == pytest.approx(0.75 ** 4)
    assert lvh.binom_cdf(4, 4, 0.3) == pytest.approx(1.0)
    lo, hi = lvh.wilson(9, 40)
    assert lo < 0.225 < hi and lo == pytest.approx(0.123, abs=0.002) and hi == pytest.approx(0.375, abs=0.002)
    assert lvh.wilson(0, 0) is None


def test_mann_whitney_detects_shift_and_needs_sample():
    a, b = [float(x) for x in range(10)], [float(x) + 20 for x in range(10)]
    assert lvh.mann_whitney(a, b)["p_value"] < 0.001
    assert lvh.mann_whitney(a, a)["p_value"] == pytest.approx(1.0)
    assert lvh.mann_whitney([1, 2], b) is None


def test_percentile_and_significance_labels():
    assert lvh.percentile_rank(5, [1, 2, 3, 4, 6, 7, 8, 9]) == 50.0
    assert lvh.percentile_rank(None, [1]) is None
    assert lvh.significance(None, 50) == "NOT_TESTABLE"
    assert lvh.significance(0.001, 40) == "STATISTICALLY_MEANINGFUL"
    assert lvh.significance(0.3, 40) == "LIKELY_NOISE"
    assert lvh.significance(0.3, 3).startswith("LIKELY_NOISE_OR_UNTESTABLE")


def test_day_level_empirical_test():
    p, text = lvh._day_level(0.25, [0.1, 0.2, 0.3, 0.5])
    assert p == pytest.approx(0.5) and "2/4" in text
    assert lvh._day_level(None, [0.1]) == (None, "n/a")


# ------------------------------------------------------------ signal outcomes
def test_signal_outcome_target_matches_tracker_rules():
    row = outcome([(9, 35, 100, 101, 99, 100), (9, 40, 100, 140, 99, 130)])
    assert row["status"] == lvh.WIN and row["exit_reason"] == "TARGET"
    assert row["exit_price"] == pytest.approx(row["target"])
    assert row["r_multiple"] == pytest.approx(2.0, abs=0.05)
    assert row["holding_minutes"] == 5 and row["dte"] == 1


def test_signal_outcome_stop_same_candle_and_pending():
    loss = outcome([(9, 35, 100, 101, 80, 85)])
    assert loss["status"] == lvh.LOSS and loss["exit_reason"] == "STOP_LOSS"
    amb = outcome([(9, 35, 100, 101, 99, 100), (9, 40, 100, 140, 80, 100)])
    assert amb["status"] == lvh.AMBIGUOUS and amb["pnl"] is None
    pending = outcome([(9, 35, 100, 101, 99, 100)])
    assert pending["status"] == lvh.PENDING and pending["reason"] == "NO_EXIT_EVIDENCE"


def test_point_in_time_levels_exclude_entry_candle():
    # A wide entry candle must not widen the stop: levels use ATR through the signal candle only.
    row = outcome([(9, 35, 100, 101, 80, 100), (9, 40, 100, 101, 99, 100)])
    assert row["premium_atr"] == pytest.approx(row["pit_premium_atr"])
    assert row["stop_loss"] == pytest.approx(row["pit_stop_loss"])
    assert row["status"] == row["pit_status"] == lvh.LOSS


def test_signal_outcome_ignores_candles_before_entry():
    # The signal bar itself (09:30) spikes to 140 but entry is the 09:35 open: no look-ahead win.
    rows = lvh.signal_outcomes([signal(9, 30)], lambda s: CONTRACT,
                               lambda c: premium(WARMUP[:-1] + [(9, 30, 100, 140, 99, 100), (9, 35, 100, 101, 99, 100)]),
                               PARAMS, "15:15", "test")
    assert rows[0]["status"] == lvh.PENDING


def test_signal_outcome_unresolved_without_entry_candle():
    row = lvh.signal_outcomes([signal(9, 30)], lambda s: CONTRACT, lambda c: premium(WARMUP), PARAMS,
                              "15:15", "test")[0]
    assert row["status"] == lvh.UNRESOLVED and row["reason"] == "NO_PREMIUM_ENTRY_CANDLE"


# -------------------------------------------------------- clusters / metrics
def _row(i, status, exit_time="2026-10-05T10:00:00+05:30", key="K", pnl=None, direction="SELL"):
    return {"signal_time": f"2026-10-05T09:{30 + i:02d}:00+05:30", "session_date": "2026-10-05", "status": status,
            "exit_time": exit_time, "instrument_key": key, "direction": direction,
            "pnl": pnl if pnl is not None else (100.0 if status == lvh.WIN else -50.0),
            "r_multiple": 2.0 if status == lvh.WIN else -1.0, "holding_minutes": 10.0}


def test_clusters_group_shared_exit_events():
    rows = [_row(0, lvh.LOSS), _row(1, lvh.LOSS), _row(2, lvh.LOSS, key="OTHER"),
            _row(3, lvh.WIN, exit_time="2026-10-05T11:00:00+05:30"), _row(4, lvh.PENDING)]
    assert lvh.assign_clusters(rows) == 3
    assert rows[0]["outcome_cluster"] == rows[1]["outcome_cluster"] != rows[2]["outcome_cluster"]
    assert "outcome_cluster" not in rows[4]


def test_group_metrics_accuracy_and_exclusions():
    rows = [_row(i, lvh.WIN) for i in range(18)] + [_row(20 + i, lvh.LOSS) for i in range(10)]
    rows += [_row(1, lvh.AMBIGUOUS), _row(2, lvh.PENDING), _row(3, lvh.UNRESOLVED)]
    m = lvh.group_metrics(rows)
    assert m["resolved"] == 28 and m["win_rate"] == pytest.approx(18 / 28)
    assert m["ambiguous"] == 1 and m["pending"] == 1 and m["unresolved_levels"] == 1
    assert m["profit_factor"] == pytest.approx(1800 / 500)
    assert lvh.group_metrics([])["win_rate"] is None


def test_day_profile_efficiency_and_orb_crossings():
    ts = [at(9, 15) + timedelta(minutes=5 * i) for i in range(8)]
    closes = [100, 101, 100, 105, 99, 106, 98, 100]
    frame = pd.DataFrame({"timestamp": ts, "open": [100] + closes[:-1], "high": [c + 1 for c in closes],
                          "low": [c - 1 for c in closes], "close": closes, "atr14": 2.0})
    frame["day"] = frame["timestamp"].dt.date
    profile = lvh.day_profiles(frame, [], "09:30")[0]
    assert profile["orb_crossings"] >= 4
    assert profile["efficiency_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert profile["first_break"] == "BUY" and profile["returned_inside_after_first_break"] is True


# ------------------------------------------------------------------- safety
FORBIDDEN = ("src.broker", "src.engine", "order_manager", "TradingEngine")


@pytest.mark.parametrize("path", ["src/backtest/live_vs_historical.py", "scripts/analyze_s2_live_vs_historical.py"])
def test_analysis_has_no_order_path(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in imported if any(f in m for f in FORBIDDEN)]
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"place_order", "submit_entry", "submit_signal", "cancel_order", "modify_order",
                        "write", "append_only_write", "save_live_signal"}
