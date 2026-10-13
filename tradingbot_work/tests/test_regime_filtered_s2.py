"""Regime-filtered S2: filter behaviour, no look-ahead, unchanged S2/options/risk, paper-only."""
from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.options_engine import REGIME_FILTERED, OptionsResearchEngine
from src.backtest.regime_filter import (
    AVOID_REGIMES,
    PointInTimeRegimeClassifier,
    RegimeSignalFilter,
    validate_hypothesis,
)
from src.backtest.regime_filtered import build_comparison, filtered_coverage, trade_diff
from src.backtest.research_strategies import ResearchSignal, generate_signals
from src.config.loader import load_strategy_config
from src.config.research_loader import load_research_config
from src.safety.live_gate import LiveTradingSafetyError
from tests.options_support import contracts

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]


def _day_bars(day: int, base: float, width: float, n: int = 30) -> list[dict]:
    start = datetime(2026, 9, day, 9, 15, tzinfo=IST)
    return [{
        "timestamp": start + timedelta(minutes=5 * i), "symbol": "NIFTY",
        "open": base, "high": base + width, "low": base - width, "close": base, "volume": 1000.0,
    } for i in range(n)]


def _wide_then_narrow() -> pd.DataFrame:
    rows: list[dict] = []
    for day in range(1, 7):
        rows += _day_bars(day, 23000.0, 60.0)
    rows += _day_bars(7, 23000.0, 2.0)
    return pd.DataFrame(rows)


def _signal(ts: datetime, direction: str = "BUY", spot: float = 23000.0) -> ResearchSignal:
    return ResearchSignal(ts, direction, "S2", "NIFTY", spot, {"orb_breakout": True})


@pytest.fixture
def configs():
    return (load_research_config(ROOT / "config" / "strategy_research.yaml"),
            load_strategy_config(ROOT / "config" / "strategy.yaml"))


def _engine_fixture():
    key = "NSE_FO|TEST"
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    under, prem_rows = [], []
    price, prem = 23000.0, 120.0
    for i in range(120):
        t = start + timedelta(minutes=5 * i)
        price += 8 if i < 20 else (-12 if i < 40 else 6)
        prem += 2 if i < 20 else (-3 if i < 40 else 1)
        under.append({"timestamp": t, "symbol": "NIFTY", "open": price, "high": price + 10,
                      "low": price - 10, "close": price, "volume": 5000})
        prem_rows.append({"timestamp": t, "instrument_key": key, "open": prem, "high": prem + 3,
                          "low": max(1.0, prem - 8), "close": prem, "volume": 100, "open_interest": 1000})
    return pd.DataFrame(under), {key: pd.DataFrame(prem_rows)}


def test_regime_filter_blocks_avoided_regimes_only():
    frame = pd.concat([_wide_then_narrow(), pd.DataFrame(_day_bars(8, 23000.0, 120.0))], ignore_index=True)
    rfilter = RegimeSignalFilter(PointInTimeRegimeClassifier(frame))
    narrow = rfilter(_signal(datetime(2026, 9, 7, 11, 0, tzinfo=IST)))
    wide = rfilter(_signal(datetime(2026, 9, 8, 11, 0, tzinfo=IST)))
    assert {"range_bound", "low_volatility"} <= set(narrow)
    assert wide == []
    assert set(narrow) <= set(AVOID_REGIMES)
    assert len(rfilter.decisions) == 2


def test_upward_spike_is_blocked():
    rows = []
    for day in range(1, 7):
        rows += _day_bars(day, 23000.0, 30.0)
    spike_day = _day_bars(7, 23000.0, 30.0)
    spike_day[5] = {**spike_day[5], "close": 23200.0, "high": 23210.0}
    rows += spike_day
    classifier = PointInTimeRegimeClassifier(pd.DataFrame(rows))
    before = classifier.regimes_at(spike_day[4]["timestamp"])
    after = classifier.regimes_at(spike_day[5]["timestamp"])
    assert "sudden_upward_spike" not in before
    assert "sudden_upward_spike" in after


def test_unknown_regime_rejected():
    with pytest.raises(ValueError):
        RegimeSignalFilter(PointInTimeRegimeClassifier(_wide_then_narrow()), avoid=("not_a_regime",))


def test_no_look_ahead_future_bars_do_not_change_decision():
    frame = _wide_then_narrow()
    ts = datetime(2026, 9, 7, 11, 0, tzinfo=IST)
    baseline = PointInTimeRegimeClassifier(frame)
    tags = baseline.regimes_at(ts)
    features = baseline.features_at(ts)

    mutated = frame.copy()
    later = pd.to_datetime(mutated["timestamp"]) > pd.Timestamp(ts)
    mutated.loc[later, "high"] = 99999.0
    mutated.loc[later, "low"] = 1.0
    future_day = pd.DataFrame(_day_bars(8, 30000.0, 500.0))
    mutated = pd.concat([mutated, future_day], ignore_index=True)

    shifted = PointInTimeRegimeClassifier(mutated)
    assert shifted.regimes_at(ts) == tags
    assert shifted.features_at(ts) == features


def test_insufficient_history_does_not_flag_range_regimes():
    frame = pd.DataFrame(_day_bars(1, 23000.0, 2.0) + _day_bars(2, 23000.0, 2.0))
    tags = PointInTimeRegimeClassifier(frame).regimes_at(datetime(2026, 9, 2, 11, 0, tzinfo=IST))
    assert "range_bound" not in tags and "low_volatility" not in tags


def test_daily_loss_limit_kept_at_3000(configs):
    research, strategy = configs
    under, premium = _engine_fixture()
    allow_all = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t",
                                      signal_filter=lambda _s: [])
    assert allow_all.max_daily_loss == 3000
    result = allow_all.run(under, "NIFTY", "S2", date(2026, 10, 1), date(2026, 10, 1))
    day_pnl: dict[str, float] = {}
    for trade in result.trades:
        day_pnl[trade["entry_time"][:10]] = day_pnl.get(trade["entry_time"][:10], 0.0) + trade["pnl"]
    assert all(pnl > -3000 - 1e-6 for pnl in day_pnl.values())


def test_unchanged_s2_signals_and_results_without_filter(configs):
    research, strategy = configs
    under, premium = _engine_fixture()
    base = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t")
    none_filter = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t", signal_filter=None)
    allow_all = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t",
                                      signal_filter=lambda _s: [])
    runs = [e.run(under, "NIFTY", "S2", date(2026, 10, 1), date(2026, 10, 1)).as_dict()
            for e in (base, none_filter, allow_all)]
    assert runs[0] == runs[1] == runs[2]
    expected = [s.as_dict() for s in generate_signals("S2", under.sort_values("timestamp").reset_index(drop=True),
                                                      "NIFTY", research)]
    assert {s["timestamp"] for s in runs[0]["signals"]} <= {s["timestamp"] for s in expected}


def test_blocking_filter_records_skips_and_keeps_signal_count(configs):
    research, strategy = configs
    under, premium = _engine_fixture()
    original = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t").run(
        under, "NIFTY", "S2", date(2026, 10, 1), date(2026, 10, 1)).as_dict()
    blocked = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t",
                                    signal_filter=lambda _s: ["range_bound"]).run(
        under, "NIFTY", "S2", date(2026, 10, 1), date(2026, 10, 1)).as_dict()
    assert blocked["metrics"]["total_signals"] == original["metrics"]["total_signals"]
    assert blocked["trades"] == []
    assert any(s["reason"] == REGIME_FILTERED for s in blocked["skipped"])
    diff = trade_diff(original, blocked)
    assert len(diff["removed"]) == len(original["trades"])
    assert diff["added"] == []


def test_option_resolution_unchanged(configs):
    research, strategy = configs
    _, premium = _engine_fixture()
    signal = _signal(datetime(2026, 10, 1, 10, 0, tzinfo=IST), "SELL")
    plain = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t")
    filtered = OptionsResearchEngine(research, strategy, contracts(), premium, run_id="t",
                                     signal_filter=lambda _s: [])
    a, b = plain._resolve_contract(signal), filtered._resolve_contract(signal)
    assert a.instrument_key == b.instrument_key
    assert a.option_type.value == "PE"


def test_paper_only_refuses_live(monkeypatch):
    spec = importlib.util.spec_from_file_location("rfb", ROOT / "scripts" / "run_regime_filtered_backtest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(LiveTradingSafetyError):
        module.run_regime_filtered(from_date=date(2026, 9, 1), to_date=date(2026, 10, 5))


def test_regime_modules_have_no_order_calls():
    for rel in ("src/backtest/regime_filter.py", "src/backtest/regime_filtered.py",
                "scripts/run_regime_filtered_backtest.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "place_order" not in text and "OrderManager" not in text and "create_broker" not in text


def test_validate_hypothesis_requires_known_regimes():
    report = {"regimes": [{"regime_id": rid} for rid in ("range_bound", "gap_up")]}
    assert set(validate_hypothesis(report, ["range_bound"], ["gap_up"])) == {"range_bound"}
    with pytest.raises(ValueError):
        validate_hypothesis(report, ["low_volatility"], [])


def test_filtered_coverage_excludes_blocked_signals():
    coverage = {"total_signals": 4, "signals_missing_coverage": 2, "signals_with_complete_coverage": 2,
                "coverage_percent": 50.0, "unavailable": [{"signal_time": "a"}, {"signal_time": "b"}]}
    out = filtered_coverage(coverage, {"a"})
    assert out["total_signals"] == 3
    assert out["signals_missing_coverage"] == 1
    assert out["coverage_percent"] == pytest.approx(66.67)


def _summary(**overrides):
    base = {"trading_days": 60, "resolved_trades": 60, "win_rate": 0.5, "profit_factor": 1.5,
            "net_pnl": 10000.0, "max_drawdown": 5000.0, "daily_pnl": []}
    return {**base, **overrides}


def test_verdict_needs_more_data_for_short_window():
    comp = build_comparison(original=_summary(trading_days=22), filtered=_summary(trading_days=22, profit_factor=3.0),
                            diff={"removed": [], "added": [], "removed_directly_by_filter": []},
                            filtered_regimes={}, meta={})
    assert comp["verdict"] == "NEEDS MORE DATA"


def test_verdict_keeps_original_when_filter_does_not_help():
    comp = build_comparison(original=_summary(), filtered=_summary(profit_factor=1.2, net_pnl=8000.0),
                            diff={"removed": [], "added": [], "removed_directly_by_filter": []},
                            filtered_regimes={}, meta={})
    assert comp["verdict"] == "KEEP ORIGINAL"
    assert comp["assessment"]["profit_factor_improved"] is False


def test_verdict_keeps_filtered_only_when_metrics_prove_it():
    comp = build_comparison(original=_summary(),
                            filtered=_summary(resolved_trades=50, profit_factor=2.0, net_pnl=12000.0,
                                              max_drawdown=4000.0),
                            diff={"removed": [], "added": [], "removed_directly_by_filter": []},
                            filtered_regimes={}, meta={})
    assert comp["verdict"] == "KEEP FILTERED"
