from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest import losing_trade_analysis as lta
from src.historical_data.validation import expected_slots

ROOT = Path(__file__).resolve().parents[1]
S2_CFG = {"range_start": "09:15", "range_end": "09:30", "atr_period": 14}
KEY = "NSE_FO|1"
DAYS = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]


def underlying(volume: float = 0.0) -> pd.DataFrame:
    rows, price = [], 25000.0
    for day in DAYS:
        for i, ts in enumerate(expected_slots(day)):
            price += 3.0 if i % 4 else -5.0
            rows.append({"timestamp": ts, "open": price - 1, "high": price + 4, "low": price - 4,
                         "close": price, "volume": volume, "open_interest": 0.0})
    return pd.DataFrame(rows)


def premium(stop_first: bool = False) -> pd.DataFrame:
    rows = []
    for day in DAYS:
        for i, ts in enumerate(expected_slots(day)):
            base = 200.0 + i * 0.2
            rows.append({"timestamp": ts, "instrument_key": KEY, "open": base, "high": base + 2, "low": base - 2,
                         "close": base, "volume": 1000.0 + i, "open_interest": 0.0, "strike": 25000.0,
                         "option_type": "CE", "expiry": "2026-09-08", "underlying": "NIFTY"})
    frame = pd.DataFrame(rows)
    if stop_first:
        idx = frame.index[frame["timestamp"] == pd.Timestamp("2026-09-03 11:05", tz=lta.IST)][0]
        frame.loc[idx, "low"] = 150.0
    return frame


def trade(signal: str, entry: str, exit_: str, pnl: float, *, key: str = KEY, entry_price: float = 210.0,
          stop: float = 200.0, target: float = 230.0, exit_reason: str = "TARGET") -> dict:
    return {"strategy_id": "S2", "underlying": "NIFTY", "direction": "BUY", "option_type": "CE",
            "instrument_key": key, "trading_symbol": "NIFTY 25000 CE 08 SEP 26", "quantity": 65,
            "entry_price": entry_price, "exit_price": entry_price + pnl / 65, "stop_loss": stop, "target": target,
            "entry_time": f"{entry}+05:30", "exit_time": f"{exit_}+05:30", "exit_reason": exit_reason,
            "gross_pnl": pnl, "costs": 0.0, "pnl": pnl, "r_multiple": pnl / (65 * (entry_price - stop)),
            "signal_time": f"{signal}+05:30", "conditions": {"orb_breakout": True, "above": 25010.0}}


def sample_trades() -> list[dict]:
    return [
        trade("2026-09-01T10:00:00", "2026-09-01T10:05:00", "2026-09-01T10:30:00", 1300.0),
        trade("2026-09-03T11:00:00", "2026-09-03T11:05:00", "2026-09-03T11:20:00", -650.0, exit_reason="STOP_LOSS"),
        trade("2026-09-03T13:00:00", "2026-09-03T13:05:00", "2026-09-03T13:30:00", 0.0, exit_reason="SQUARE_OFF"),
    ]


def run(trades=None, candles=None, prem=None):
    trades = sample_trades() if trades is None else trades
    prem = {KEY: premium()} if prem is None else prem
    return lta.analyze(trades, underlying() if candles is None else candles, prem, S2_CFG, 1_000_000.0)


# --------------------------------------------------------- win / loss classes
def test_win_loss_classification():
    assert lta.classify_outcome(0.01) == lta.WIN
    assert lta.classify_outcome(-0.01) == lta.LOSS
    assert lta.classify_outcome(0.0) == lta.BREAKEVEN
    result = run()
    assert [r["outcome"] for r in result["trades"]] == [lta.WIN, lta.LOSS, lta.BREAKEVEN]
    o = result["overall"]
    assert (o["wins"], o["losses"], o["breakeven"]) == (1, 1, 1)
    assert o["profit_factor"] == pytest.approx(2.0)
    assert o["net_pnl"] == pytest.approx(650.0)


# ------------------------------------------------------ all trades included
def test_all_resolved_trades_included_and_unresolved_excluded():
    script = _load_script()
    raw = {"trades": sample_trades() + [dict(sample_trades()[0], exit_reason="DATA_INVALID"),
                                        dict(sample_trades()[0], exit_reason="SKIPPED")]}
    resolved = script.resolved_trades(raw)
    assert len(resolved) == 3
    result = run(resolved)
    assert len(result["trades"]) == 3 and result["overall"]["trades"] == 3
    for factor in result["factors"]:
        if factor["factor_id"] not in ("day_regime", "pit_regime"):
            assert sum(b["trades"] for b in factor["buckets"]) == 3, factor["factor_id"]


# ----------------------------------------------------- no fabricated values
def test_zero_index_volume_reported_unavailable_not_fabricated():
    result = run()
    for row in result["trades"]:
        assert row["underlying_volume"] is None
        assert row["underlying_volume_avg20"] is None
        assert row["underlying_vwap"] is None and row["vwap_relationship"] is None
    vwap = next(f for f in result["factors"] if f["factor_id"] == "underlying_vwap")
    assert [b["bucket"] for b in vwap["buckets"]] == [lta.UNAVAILABLE]
    assert result["fields_with_unavailable_values"]["underlying_volume"] == 3


def test_real_underlying_volume_is_used_when_present():
    rows = lta.build_trade_rows(sample_trades(), underlying(volume=500.0), {KEY: premium()}, S2_CFG)
    assert rows[0]["underlying_volume"] == 500.0
    assert rows[0]["underlying_vwap"] is not None


def test_insufficient_history_leaves_indicators_and_gap_unavailable():
    first = run()["trades"][0]
    assert first["ema50"] is None and first["ema_trend_aligned"] is None
    assert first["gap_pct"] is None and first["gap_condition"] is None
    third = run()["trades"][2]
    assert third["ema50"] is not None and third["gap_condition"] is not None


# ------------------------------------------------------ missing-data handling
def test_missing_option_premium_handled_as_unavailable():
    trades = [trade("2026-09-01T10:00:00", "2026-09-01T10:05:00", "2026-09-01T10:30:00", -100.0,
                    key="NSE_FO|404", exit_reason="STOP_LOSS")]
    row = run(trades, prem={})["trades"][0]
    for field_name in ("mfe_r", "mae_r", "initial_favorable_move", "sl_vs_target_first", "sl_hit_before_target",
                       "option_volume", "option_vwap", "entry_premium_open"):
        assert row[field_name] is None, field_name
    assert row["strike"] == 25000.0
    assert row["pnl"] == -100.0 and row["outcome"] == lta.LOSS


def test_signal_bar_missing_from_underlying_keeps_values_none():
    trades = [trade("2026-09-05T10:00:00", "2026-09-05T10:05:00", "2026-09-05T10:30:00", 50.0)]
    row = run(trades)["trades"][0]
    assert row["rsi14"] is None and row["atr14"] is None and row["underlying_at_signal"] is None
    assert row["distance_from_breakout"] is None


# ----------------------------------------------------- option path behaviour
def test_stop_touched_before_target_detected():
    rows = lta.build_trade_rows(sample_trades(), underlying(), {KEY: premium(stop_first=True)}, S2_CFG)
    loser = rows[1]
    assert loser["sl_vs_target_first"] == "SL_FIRST" and loser["sl_hit_before_target"] is True
    assert loser["mae_r"] == pytest.approx((210.0 - 150.0) / 10.0)


# ----------------------------------------------------- hypothetical filters
def test_hypothetical_filters_are_consistent_and_flag_small_samples():
    result = run()
    assert result["hypothetical_filters"], "expected at least one hypothetical filter"
    for c in result["hypothetical_filters"]:
        assert c["trades_removed"] + c["trades_remaining"] == 3
        assert c["wins_removed"] + c["losses_removed"] <= c["trades_removed"]
        assert c["assessment"] == "TOO_FEW_TRADES"
        assert c["original_net_pnl"] == pytest.approx(650.0)
    assert result["strongest_candidate"] is None


def test_non_monotonic_detection():
    table = [{"bucket": "a", "net_pnl": 10}, {"bucket": "b", "net_pnl": -5}, {"bucket": "c", "net_pnl": 3}]
    assert lta.is_non_monotonic(table, "b")
    assert not lta.is_non_monotonic(table, "a")


# ------------------------------------------------- no modification of S2 logic
PROTECTED = ["src/backtest/research_strategies.py", "src/backtest/options_engine.py",
             "src/backtest/regime_filter.py", "config/strategy.yaml", "config/strategy_research.yaml"]


def _hashes() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PROTECTED}


def test_analysis_does_not_modify_strategy_config_or_inputs(tmp_path):
    before = _hashes()
    trades, candles, prem = sample_trades(), underlying(), {KEY: premium()}
    trades_copy, candles_copy, prem_copy = copy.deepcopy(trades), candles.copy(), prem[KEY].copy()
    result = lta.analyze(trades, candles, prem, S2_CFG, 1_000_000.0)
    lta.write_reports({"source_run_id": "x", "replay_run_id": "y", "from_date": "2026-09-01",
                       "to_date": "2026-09-03", "mode": "test", "notes": [], "source_row_summary": "-",
                       "conclusions": [], "analysis": result}, tmp_path)
    assert _hashes() == before
    assert trades == trades_copy
    pd.testing.assert_frame_equal(candles, candles_copy)
    pd.testing.assert_frame_equal(prem[KEY], prem_copy)


@pytest.mark.parametrize("path", ["src/backtest/losing_trade_analysis.py", "scripts/analyze_s2_losing_trades.py"])
def test_no_order_or_broker_code(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    for name in ("place_order", "OrderManager", "TradingEngine", "api-sandbox", "signal_filter=",
                 "strategies[strategy_id] =", "write_text(\"config"):
        assert name not in source


# ----------------------------------------------------------------- reports
def test_reports_written_with_unavailable_markers(tmp_path):
    result = run()
    paths = lta.write_reports({"source_run_id": "x", "replay_run_id": "y", "from_date": "2026-09-01",
                               "to_date": "2026-09-03", "mode": "test", "notes": ["n"], "source_row_summary": "-",
                               "conclusions": ["c"], "analysis": result}, tmp_path)
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["analysis"]["overall"]["trades"] == 3
    with paths["csv"].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3 and rows[0]["underlying_volume"] == lta.UNAVAILABLE
    md = paths["md"].read_text(encoding="utf-8")
    for heading in ("## 1. Overall S2 summary", "## 2. Winning vs losing comparison", "## 3. Losing-trade breakdown",
                    "## 4. Worst losing setups", "## 5. Best winning setups", "## 6. Time-of-day analysis",
                    "## 7. Market-regime analysis", "## 8. Indicator-condition analysis",
                    "## 9. Option behavior analysis", "## 10. Potential improvement candidates"):
        assert heading in md


def test_infinite_profit_factor_serialised_as_string(tmp_path):
    winners = [trade("2026-09-01T10:00:00", "2026-09-01T10:05:00", "2026-09-01T10:30:00", 500.0)]
    result = run(winners)
    assert result["overall"]["profit_factor"] == float("inf")
    paths = lta.write_reports({"source_run_id": "x", "replay_run_id": "y", "from_date": "a", "to_date": "b",
                               "mode": "t", "notes": [], "source_row_summary": "-", "conclusions": [],
                               "analysis": result}, tmp_path)
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["analysis"]["overall"]["profit_factor"] == "inf"


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_s2", ROOT / "scripts" / "analyze_s2_losing_trades.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conclusions_without_candidate_say_so():
    lines = _load_script().build_conclusions(run(), (25000.0, 24000.0))
    assert any("No point-in-time filter" in line for line in lines)
    assert any("-4.0%" in line for line in lines)
