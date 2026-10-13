from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_s2_diagnostic as script
from src.backtest import s2_direction_research as dr
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s4_six_month_lab as s4
from src.safety import live_gate
from tests.test_s4_six_month_research import make_store

ROOT = Path(__file__).resolve().parents[1]
S2_FILES = ["src/backtest/research_strategies.py", "src/backtest/s2_1dte_improvement.py",
            "src/backtest/s2_direction_research.py", "src/backtest/current_day_s2.py", "src/options/strategy.py",
            "src/options/selection.py", "config/strategy.yaml", "config/strategy_research.yaml"]


def _hashes() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in S2_FILES if (ROOT / f).is_file()}


def _boom(*_a, **_k):
    raise AssertionError("diagnostic must read only the 6-month store")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before, baseline = _hashes(), repr(dr.BASELINE)
    base = tmp_path_factory.mktemp("diag")
    make_store(base / "upstox_6m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=base / "upstox_6m", out_dir=base / "out", check_sample=6)
    store = s4.SixMonthStore(base / "upstox_6m")
    return SimpleNamespace(a=a, base=base, before=before, baseline=baseline, store=store, spot=store.spot())


def _decided(a):
    return dg.decided(a["rows"])


# ------------------------------------------------------------------ no S2 mutation / safety
def test_no_s2_mutation_and_no_order_path(env):
    assert _hashes() == env.before and repr(dr.BASELINE) == env.baseline
    a = env.a
    assert a["s2_changed"] is False and a["orders"] == 0 and a["paper"] is True and a["live_trading_enabled"] is False
    order_call = "place" + "_order"
    for mod in (dg, script):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert order_call not in src
        imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
        assert not any("OrderManager" in ln or "order_manager" in ln for ln in imports)


def test_paper_mode_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))


# ------------------------------------------------------------------ chronology / leakage
def test_chronological_correctness(env):
    ck = env.a["checks"]["chronology"]
    assert ck["passed"] and ck["ordered"] and not ck["entry_not_next_bar"] and not ck["exit_before_entry"]
    ts = [dra._ts(r["signal_time"]) for r in env.a["rows"]]
    assert ts == sorted(ts)


def test_no_future_feature_leakage(env):
    tc = env.a["checks"]["truncation"]
    assert tc["checked"] > 0 and tc["passed"], tc["mismatches"]


def test_truncation_check_catches_a_leaky_feature(env, monkeypatch):
    real = dg.feature_bars
    monkeypatch.setattr(dg, "feature_bars", lambda spot: real(spot).shift(-1))  # reads the NEXT candle
    rows = [r for r in env.a["rows"] if r.get("instrument_key")]
    raw = [{k: v for k, v in r.items() if k not in ("features", "month")} for r in rows]
    leaky = dg.annotate(raw, dg.feature_bars(env.spot), {})
    assert not dg.truncation_check(env.spot, leaky, {}, sample=5)["passed"]


# ------------------------------------------------------------------ month assignment
def test_month_assignment_and_separate_monthly_replays(env):
    mc = env.a["checks"]["monthly_consistency"]
    assert mc["passed"]
    assert set(mc["months"]) == {m for m, _ in dg.MONTHS}
    for r in env.a["rows"]:
        assert r["month"] == str(r["session_date"])[:7] == dra._ts(r["signal_time"]).strftime("%Y-%m")
    by = {(r["month"], r["scope"]): r for r in env.a["monthly"]}
    for s in dg.SCOPES:
        assert sum(by[(m, s)]["trades"] for m, _ in dg.MONTHS) == env.a["overall"][s]["trades"]
        assert sum(by[(m, s)]["wins"] for m, _ in dg.MONTHS) == env.a["overall"][s]["wins"]


def test_daily_table_sums(env):
    d = env.a["daily"]
    assert sum(x["combined_trades"] for x in d) == env.a["overall"]["COMBINED"]["trades"]
    assert sum(x["buy_trades"] + x["sell_trades"] for x in d) == env.a["overall"]["COMBINED"]["trades"]
    s = env.a["daily_summary"]
    assert s["profitable_days"] + s["losing_days"] + s["flat_days"] == s["traded_days"]


# ------------------------------------------------------------------ regimes
def test_regime_classification_rules():
    up = {"close": 105, "ema20": 103, "ema50": 100, "ema20_slope_atr": 0.2, "atr_pctile": 0.9}
    assert dg.classify_regime(up) == ("TREND_UP", "HIGH_VOLATILITY")
    down = {"close": 95, "ema20": 97, "ema50": 100, "ema20_slope_atr": -0.2, "atr_pctile": 0.1}
    assert dg.classify_regime(down) == ("TREND_DOWN", "LOW_VOLATILITY")
    rng = {"close": 101, "ema20": 99, "ema50": 100, "ema20_slope_atr": 0.1, "atr_pctile": 0.5}
    assert dg.classify_regime(rng) == ("RANGE", "NORMAL_VOLATILITY")
    assert dg.classify_regime({"close": 1}) == (dg.UNAVAILABLE, dg.UNAVAILABLE)


def test_regime_assignment_matches_signal_candle(env):
    bars = dg.feature_bars(env.spot)
    for r in env.a["rows"][:40]:
        b = dg.bar_values(bars, dra._ts(r["signal_time"]))
        assert dg.classify_regime(b) == (r["features"]["trend_regime"], r["features"]["vol_regime"])
    trades = {(x["regime"], x["scope"]): x["trades"] for x in env.a["regimes"]}
    total = env.a["overall"]["COMBINED"]["trades"]
    trend = sum(trades.get((g, "COMBINED"), 0) for g in dg.TREND_REGIMES)
    vol = sum(trades.get((g, "COMBINED"), 0) for g in dg.VOL_REGIMES)
    unavailable_trend = sum(1 for r in _decided(env.a) if r["features"]["trend_regime"] == dg.UNAVAILABLE)
    unavailable_vol = sum(1 for r in _decided(env.a) if r["features"]["vol_regime"] == dg.UNAVAILABLE)
    assert trend + unavailable_trend == total and vol + unavailable_vol == total


# ------------------------------------------------------------------ BUY / SELL separation + options
def test_buy_sell_separation(env):
    assert env.a["checks"]["buy_sell_separation"]
    for r in env.a["rows"]:
        if r.get("option_type"):
            assert r["option_type"] == ("CE" if r["direction"] == "BUY" else "PE")
    o = env.a["overall"]
    assert o["BUY"]["trades"] + o["SELL"]["trades"] == o["COMBINED"]["trades"]
    assert {x["scope"] for x in env.a["features"]} <= {"BUY", "SELL"}
    assert {x["side"] for x in env.a["options"]} <= {"BUY / CALL", "SELL / PUT"}
    assert {x["direction"] for x in env.a["patterns"]} <= {"BUY", "SELL"}


def test_option_metadata(env):
    rows = [r for r in env.a["rows"] if r.get("instrument_key")]
    assert rows
    for r in rows:
        sd = date.fromisoformat(str(r["session_date"])[:10])
        assert r["dte"] == (r["expiry_date"] - sd).days >= 0
        assert r["features"]["dte_bucket"] == dg.dte_bucket(r["dte"])
        assert r["features"]["moneyness"] in ("ATM", "SLIGHT_ITM", "SLIGHT_OTM")
        assert abs(float(r["strike"]) - float(r["spot"])) <= 25.0 + 1e-6
    assert dg.moneyness("CE", 24000, 24030) == "SLIGHT_ITM"
    assert dg.moneyness("PE", 24000, 24030) == "SLIGHT_OTM"
    assert dg.moneyness("CE", 24000, 24005) == "ATM"
    assert dg.moneyness(None, 24000, 24005) == dg.UNAVAILABLE
    small = [x for x in env.a["options"] if not x["sample_ok"]]
    assert all(x["interpretation"].startswith("REJECTED") for x in small)


# ------------------------------------------------------------------ buckets / stability
def test_buckets_and_windows():
    assert dg.bucket(35, (40, 50, 60, 70)) == "<40"
    assert dg.bucket(40, (40, 50, 60, 70)) == "40..50"
    assert dg.bucket(70, (40, 50, 60, 70)) == ">=70"
    assert dg.bucket(None, (1,)) == dg.UNAVAILABLE
    assert [dg.dte_bucket(x) for x in (0, 1, 3, 6, 20, None)] == ["0", "1", "2-3", "4-7", ">7", dg.UNAVAILABLE]
    assert dg.time_window("2026-09-01T09:30:00+05:30") == "09:15-10:00"
    assert dg.time_window("2026-09-01T14:25:00+05:30") == "14:00-14:30"
    assert dg.time_window("2026-09-01T14:30:00+05:30") == "OUTSIDE_WINDOWS"


def test_stability_status_rules():
    base = {"trades": 120, "net_pnl": 5000.0, "profit_factor": 1.5, "months_active": 5, "months_profitable": 4,
            "pnl_without_best_day": 2000.0, "regimes_profitable": 2, "pnl_without_best_regime": 500.0}
    assert dg.status_of(base) == "STABLE"
    assert dg.status_of(dict(base, trades=20)) == "WEAK_SAMPLE"
    assert dg.status_of(dict(base, months_active=2)) == "WEAK_SAMPLE"
    assert dg.status_of(dict(base, pnl_without_best_day=-1.0)) == "UNSTABLE"
    assert dg.status_of(dict(base, regimes_profitable=1)) == "PROMISING"
    assert dg.status_of(dict(base, net_pnl=-10.0)) == "UNSTABLE"
    assert [dg.sample_grade(n) for n in (150, 70, 35, 5)] == ["A", "B", "C", "D"]


def test_patterns_are_causal_and_never_called_good_on_tiny_samples(env):
    for p in env.a["patterns"]:
        assert p["uses_future_information"] is False
        if p["trades"] < dg.MIN_SAMPLE or p["months_active"] < 3:
            assert p["status"] == "WEAK_SAMPLE"
    for d in dg.DIRECTIONS:
        for line in dg.strongest(env.a, d):
            assert "[STABLE]" in line or "[PROMISING]" in line


def test_win_loss_uses_outcome_only_for_grouping(env):
    for r in env.a["rows"]:
        assert not ({"outcome", "pnl", "exit_time", "exit_price", "r_multiple"} & set(r["features"]))
    wl = env.a["win_loss"]
    assert {x["direction"] for x in wl} == {"BUY", "SELL"}
    for d in dg.DIRECTIONS:
        dec = [r for r in _decided(env.a) if r["direction"] == d]
        rsi = next(x for x in wl if x["direction"] == d and x["feature"] == "rsi14")
        wins = [r for r in dec if r["outcome"] == "WIN" and r["features"]["rsi14"] is not None]
        assert rsi["win_n"] == len(wins)


# ------------------------------------------------------------------ reports / output
def test_reports_and_final_output(env):
    out_dir = env.base / "out"
    for n in dg.REPORT_FILES:
        assert (out_dir / n).is_file(), n
    md = (out_dir / "s2_diagnostic.md").read_text(encoding="utf-8")
    for s in ("Executive Summary", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6", "Phase 7",
              "Phase 8/9", "Phase 10"):
        assert s in md
    text = dg.summary_text(env.a)
    for label in ("S2 MONTHLY DIAGNOSTIC", "APRIL:", "MAY:", "JUNE:", "JULY:", "AUGUST:", "SEPTEMBER:",
                  "BEST REGIME:", "BEST TIME WINDOW:", "BEST REPEATABLE FEATURES:", "DIAGNOSTIC COMPLETE",
                  "S2 CHANGED:\nNO", "ORDERS:\n0", "PAPER:\nYES"):
        assert label in text, label
    for n in dg.REPORT_FILES:
        t = (out_dir / n).read_text(encoding="utf-8").lower()
        assert "access_token" not in t and "bearer " not in t
