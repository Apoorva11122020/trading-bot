from __future__ import annotations

import json
import math
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import buy_strategy_lab as lab
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.safety.live_gate import LiveTradingSafetyError
from tests.test_shadow_historical import (LISTING, PARAMS, RC, _ExpiredApiService, _patch_expired, _premium_for,
                                          _resolve, _signal_fn, _spot_frame)
from tests.test_shadow_validation import DAY, _at

NAMES = ["BUY_BASELINE", "BUY_EMA_CONFIRM", "BUY_CANDLE_CONFIRM", "BUY_VOLUME_CONFIRM", "BUY_RSI_CONFIRM",
         "BUY_BREAKOUT_CONFIRM", "BUY_EMA_CANDLE", "BUY_EMA_VOLUME", "BUY_EMA_RSI", "BUY_EMA_CANDLE_VOLUME",
         "BUY_EMA_CANDLE_RSI", "BUY_STRONG_CONFIRMATION"]
TEST_SPLITS = {"TRAIN": ("2026-09-30", "2026-09-30"), "VALIDATION": ("2026-10-01", "2026-10-01"),
               "OOS": ("2026-10-05", "2026-10-05")}
BAR = {"open": 100.0, "close": 105.0, "ema20": 103.0, "ema50": 101.0, "prev_open": 99.0, "prev_close": 100.0,
       "volume": 1300.0, "vol_avg": 1000.0, "rsi14": 56.0, "prev_orb_high": 99.5}
CTX = {"direction": "BUY", "breakout_level": 102.0, "break_atr": 0.30}


def _bar(**kw):
    return pd.Series({**BAR, **kw})


def _replay_full(**kw):
    check = sh.ExpiryCheck(LISTING, frozenset(), "test official list")
    return sh.replay(spot=_spot_frame(), signal_fn=_signal_fn, resolve=sh.strict_resolver(_resolve, check),
                     premium_for=_premium_for(), params=PARAMS, square_off="15:15", range_start="09:15",
                     range_end="09:30", days=[DAY], mutation_cut=None, keep_pipeline=True, **kw)


# --------------------------------------------------------------------------- candidates
def test_all_12_predeclared_candidates_exist():
    assert [c.name for c in lab.CANDIDATES] == NAMES
    assert len({c.name for c in lab.CANDIDATES}) == 12
    by = {c.name: c.checks for c in lab.CANDIDATES}
    assert by["BUY_BASELINE"] == ()
    assert by["BUY_EMA_CANDLE_VOLUME"] == ("ema", "candle", "volume")
    assert by["BUY_EMA_CANDLE_RSI"] == ("ema", "candle", "rsi")
    assert by["BUY_STRONG_CONFIRMATION"] == ("strong",)


# --------------------------------------------------------------------------- confirmations
def test_ema_confirmation():
    assert lab.check("ema", CTX, _bar()) is True
    assert lab.check("ema", CTX, _bar(close=102.0)) is False
    assert lab.check("ema", CTX, _bar(ema20=100.0, close=101.0)) is False
    assert lab.check("ema", CTX, _bar(ema50=float("nan"))) is None


def test_candle_confirmation():
    assert lab.check("candle", CTX, _bar()) is True
    assert lab.check("candle", CTX, _bar(prev_close=98.0)) is False
    assert lab.check("candle", CTX, _bar(close=99.0)) is False
    assert lab.check("candle", CTX, _bar(prev_open=float("nan"))) is None


def test_volume_confirmation_and_missing_volume_is_unavailable():
    assert lab.check("volume", CTX, _bar(volume=1200.0)) is True
    assert lab.check("volume", CTX, _bar(volume=1199.0)) is False
    assert lab.check("volume", CTX, _bar(volume=0.0)) is None
    assert lab.check("volume", CTX, _bar(vol_avg=float("nan"))) is None


def test_rsi_confirmation():
    assert lab.check("rsi", CTX, _bar(rsi14=55.0)) is True
    assert lab.check("rsi", CTX, _bar(rsi14=54.9)) is False
    assert lab.check("rsi", CTX, _bar(rsi14=float("nan"))) is None


def test_breakout_and_strong_confirmation():
    assert lab.check("breakout", CTX, _bar()) is True
    assert lab.check("breakout", dict(CTX, break_atr=0.51), _bar()) is False
    assert lab.check("breakout", dict(CTX, breakout_level=106.0), _bar()) is False
    assert lab.check("breakout", dict(CTX, break_atr=None), _bar()) is None
    assert lab.check("strong", CTX, _bar()) is True
    assert lab.check("strong", CTX, _bar(prev_close=99.0)) is False
    assert lab.check("strong", CTX, _bar(close=99.5, open=100.0)) is False


def _conf(ts, **kw):
    return pd.DataFrame([{**BAR, **kw}], index=[ts])


def test_combined_confirmations_and_buy_only_isolation():
    ts = _at("10:00")
    buy = {"direction": "BUY", "signal_time": ts, "breakout_level": 102.0, "break_atr": 0.3}
    sell = dict(buy, direction="SELL")
    cand = {c.name: c for c in lab.CANDIDATES}
    ok = _conf(ts)
    assert lab.buy_filter([buy, sell], ok, cand["BUY_EMA_CANDLE_RSI"]) == [None, dr.DIRECTION_SKIP]
    weak = _conf(ts, rsi14=40.0, prev_close=98.0)
    assert lab.buy_filter([buy], weak, cand["BUY_EMA_CANDLE_RSI"]) == ["CONFIRMATION_FAILED:candle,rsi"]
    assert lab.buy_filter([buy], weak, cand["BUY_EMA_CONFIRM"]) == [None]
    no_vol = _conf(ts, volume=0.0)
    assert lab.buy_filter([buy], no_vol, cand["BUY_EMA_CANDLE_VOLUME"]) == [f"{dr.CONFIRM_UNAVAILABLE}:volume"]
    assert lab.buy_filter([buy], _conf(_at("11:00")), cand["BUY_EMA_CONFIRM"]) == [
        f"{dr.CONFIRM_UNAVAILABLE}:SIGNAL_BAR_MISSING"]
    for c in lab.CANDIDATES:
        assert lab.buy_filter([sell], ok, c) == [dr.DIRECTION_SKIP]


# --------------------------------------------------------------------------- causal features
def _trend_spot(n=130):
    start = pd.Timestamp(datetime.combine(date(2026, 9, 29), time(9, 15)), tz=sh.IST)
    rows, price = [], 25000.0
    rng = np.random.default_rng(3)
    for i in range(n):
        day_off, slot = divmod(i, 75)
        ts = start + pd.Timedelta(days=day_off) + pd.Timedelta(minutes=5 * slot)
        o = price
        price = price + float(rng.normal(0, 8))
        rows.append({"timestamp": ts, "open": o, "high": max(o, price) + 2, "low": min(o, price) - 2, "close": price,
                     "volume": float(1000 + 50 * (i % 7))})
    return pd.DataFrame(rows)


def test_features_use_only_data_available_at_each_bar():
    spot = _trend_spot()
    full = lab.buy_features(spot, imp.underlying_features(spot))
    k = 110
    part_spot = spot.iloc[:k + 1]
    part = lab.buy_features(part_spot, imp.underlying_features(part_spot))
    cols = ["ema20", "ema50", "rsi14", "vol_avg", "prev_close", "prev_open", "prev_orb_high"]
    pd.testing.assert_frame_equal(full[cols].iloc[:k + 1], part[cols], check_freq=False)
    bumped = spot.copy()
    bumped.loc[k, "volume"] = 10_000_000.0
    b = lab.buy_features(bumped, imp.underlying_features(bumped))
    assert b["vol_avg"].iloc[k] == full["vol_avg"].iloc[k]
    assert b["vol_avg"].iloc[k + 1] != full["vol_avg"].iloc[k + 1]


# --------------------------------------------------------------------------- lookahead mutation
def test_mutation_lookahead_passes_for_all_candidates_on_real_pipeline():
    p = _replay_full()["pipeline"]
    r = lab.buy_mutation_test(p["pipeline"], p["spot"], p["premium_for"], lab.CANDIDATES, [_at("12:00")])
    assert set(r) == set(NAMES)
    assert all(v["passed"] and v["cuts"] == 1 for v in r.values())


def test_mutation_lookahead_detects_future_data_usage(monkeypatch):
    p = _replay_full()["pipeline"]
    real = lab.buy_features
    original_last = float(p["spot"]["close"].iloc[-1])

    def leaky(spot, features):
        f = real(spot, features)
        future = float(spot["close"].iloc[-1])
        f["ema20"] = f["close"] - abs(future - original_last)
        f["ema50"] = 0.0
        return f

    monkeypatch.setattr(lab, "buy_features", leaky)
    r = lab.buy_mutation_test(p["pipeline"], p["spot"], p["premium_for"], lab.CANDIDATES, [_at("12:00")])
    assert r["BUY_BASELINE"]["passed"] is True
    assert r["BUY_EMA_CONFIRM"]["passed"] is False and r["BUY_EMA_CONFIRM"]["failures"]


# --------------------------------------------------------------------------- outcomes / metrics
def test_baseline_candidate_reproduces_original_buy_rows_and_outcome():
    res = _replay_full()
    p = res["pipeline"]
    conf = lab.buy_features(p["spot"], p["features"])
    rows = lab.run_candidate(p["contexts"], p["features"], conf, lab.CANDIDATES[0], square_off="15:15")
    original = [r for r in res["rows"][lab.ORIGINAL] if r["direction"] == "BUY"]
    key = lambda rs: [(r["signal_time"], r["state"], r["outcome"], r["pnl"]) for r in rs]  # noqa: E731
    assert key(rows) == key(original)
    assert [(r["outcome"], r["pnl"]) for r in rows] == [(imp.WIN, 390.0)]
    assert all(r["direction"] == "BUY" for c in lab.CANDIDATES
               for r in lab.run_candidate(p["contexts"], p["features"], conf, c, square_off="15:15"))


def _r(day, pnl, i=0, state=sh.EVALUATED, dte="1 DTE", regimes=()):
    ev = state == sh.EVALUATED
    return {"session_date": day, "signal_time": f"{day}T10:{i:02d}:00+05:30", "direction": "BUY", "state": state,
            "outcome": (imp.WIN if pnl > 0 else imp.LOSS) if ev else None, "pnl": pnl if ev else None,
            "reason": None if ev else "option_candle_missing", "dte_bucket": dte, "regimes": list(regimes),
            "time_bucket": None, "skip_reason": "CONFIRMATION_FAILED:ema" if state == sh.SKIPPED else None}


def test_accuracy_pf_pnl_drawdown_dte_and_regime_aggregation():
    days = ["2026-09-25", "2026-09-28"]
    rows = [_r(days[0], 300.0, 0, dte="0 DTE", regimes=["strong_bullish_trend"]), _r(days[0], -100.0, 1),
            _r(days[1], -150.0, 0, regimes=["range_bound"]), _r(days[1], 0, 1, state=sh.DATA_UNAVAILABLE),
            _r(days[1], 0, 2, state=sh.SKIPPED)]
    m = so.split_metrics(rows, days)
    assert (m["signals"], m["evaluated"], m["wins"], m["losses"], m["data_unavailable"], m["skipped"]) == (5, 3, 1, 2,
                                                                                                          1, 1)
    assert m["win_rate"] == pytest.approx(1 / 3) and m["profit_factor"] == pytest.approx(1.2)
    assert m["net_pnl"] == 50 and m["max_drawdown"] == 250
    assert (m["profitable_days"], m["losing_days"]) == (1, 1)
    assert m["dte"]["0 DTE"]["net_pnl"] == 300 and m["dte"]["1 DTE"]["evaluated"] == 2
    assert set(m["regime"]) == {"strong_bullish_trend", "range_bound"}
    assert lab.skip_reasons(rows) == {"CONFIRMATION_FAILED:ema": 1}
    assert lab._without_day(rows, days[0])["net_pnl"] == -150
    assert lab._without_day(rows, days[1])["net_pnl"] == 200


# --------------------------------------------------------------------------- status / ranking
def _m(evaluated=40, wr=0.5, pf=1.5, pnl=1000.0, dd=300.0, cov=1.0):
    return {"evaluated": evaluated, "win_rate": wr, "profit_factor": pf, "net_pnl": pnl, "max_drawdown": dd,
            "data_coverage": cov}


def test_promising_requires_every_rule_and_never_robust():
    base_train = _m(wr=0.26)
    good = lab.candidate_status(_m(wr=0.40), _m(wr=0.46), _m(), base_train, True)
    assert good["status"] == lab.PROMISING and good["failed_checks"] == []
    cases = {"validation_wr": dict(val=_m(wr=0.44)), "oos_wr": dict(oos=_m(wr=0.44)),
             "oos_pf": dict(oos=_m(pf=1.10)), "oos_pnl": dict(oos=_m(pnl=-1.0)),
             "lookahead": dict(la=False), "no_major_data_gaps": dict(oos=_m(cov=0.80)),
             "train_wr_improves_or_reasonable": dict(train=_m(wr=0.20))}
    for check_name, kw in cases.items():
        st = lab.candidate_status(kw.get("train", _m(wr=0.40)), kw.get("val", _m()), kw.get("oos", _m()), base_train,
                                  kw.get("la", True))
        assert st["status"] == lab.NOT_PROMISING and check_name in st["failed_checks"]
    for st in (good, lab.candidate_status(_m(), _m(), _m(evaluated=5), base_train, True)):
        assert st["status"] != "ROBUST"


def test_insufficient_data_below_30_oos_trades_and_ranked_last():
    st = lab.candidate_status(_m(), _m(), _m(evaluated=29, pf=9.0), _m(), True)
    assert st["status"] == lab.INSUFFICIENT_DATA
    oos = {"TINY": _m(evaluated=5, pf=math.inf, pnl=5000.0), "A": _m(pf=1.5), "B": _m(pf=2.0),
           "C": _m(pf=1.5, pnl=2000.0), "D": _m(pf=1.5, dd=100.0)}
    assert lab.rank_by_oos(oos) == ["B", "C", "D", "A", "TINY"]


def test_comparison_reports_percentage_point_change():
    c = lab.comparison(_m(evaluated=254, wr=0.256, pf=0.58, pnl=-84782.75, dd=84782.75),
                       _m(evaluated=120, wr=0.482, pf=1.3, pnl=12000.0, dd=9000.0))
    assert c["wr_change_pp"] == pytest.approx(22.6)
    assert c["pf_change"] == pytest.approx(0.72) and c["pnl_change"] == pytest.approx(96782.75)
    assert c["dd_improved"] is True and c["trade_count_change"] == -134


# --------------------------------------------------------------------------- splits
def test_chronological_split_and_no_overlap():
    s = imp.split_dates(imp.DEFAULT_SPLITS)
    assert s["TRAIN"] == (date(2026, 9, 1), date(2026, 9, 17)) and s["OOS"] == (date(2026, 9, 25), date(2026, 10, 1))
    by, unassigned = so.assign_days(["2026-10-05", "2026-09-25", "2026-09-18", "2026-09-01"], s)
    assert by == {"TRAIN": ["2026-09-01"], "VALIDATION": ["2026-09-18"], "OOS": ["2026-09-25"]}
    assert unassigned == ["2026-10-05"]


# --------------------------------------------------------------------------- runner
def _lab_runner(monkeypatch, service):
    _patch_expired(monkeypatch, service)
    import scripts.run_buy_strategy_lab as runner
    return runner


def test_runner_end_to_end_with_real_expired_option_data(monkeypatch, tmp_path):
    service = _ExpiredApiService(_spot_frame())
    runner = _lab_runner(monkeypatch, service)
    other = tmp_path / "strategy_direction_20_sessions"
    other.mkdir()
    (other / "summary.json").write_text("keep")
    out = tmp_path / "buy_strategy_lab"
    a = runner.run(out_dir=out, splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert (other / "summary.json").read_text() == "keep"
    assert a["baseline_matches_original_s2_buy"] is True
    base = a["metrics"]["OOS"]["BUY_BASELINE"]
    assert (base["signals"], base["evaluated"], base["wins"], base["net_pnl"]) == (1, 1, 1, 390.0)
    assert base["dte"]["0 DTE"]["evaluated"] == 1
    vol = a["metrics"]["OOS"]["BUY_VOLUME_CONFIRM"]
    assert vol["evaluated"] == 0 and a["skip_reasons"]["BUY_VOLUME_CONFIRM"] == {f"{dr.CONFIRM_UNAVAILABLE}:volume": 1}
    assert all(r["passed"] for r in a["lookahead"].values())
    assert {s["status"] for s in a["status"].values()} == {lab.INSUFFICIENT_DATA}
    assert a["best_candidate"] is None and a["splits"]["OOS"]["days"] == ["2026-10-05"]
    assert {c[3] for c in service.downloader.calls} == {"EXPIRED"}
    assert {p.name for p in out.iterdir()} == {"summary.md", "summary.json", "summary.csv", "daily.csv", "dte.csv",
                                              "regime.csv"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["trading_mode"] == "PAPER" and summary["live_trading_enabled"] is False
    assert "_rows" not in summary and "ROBUST" not in {s["status"] for s in summary["status"].values()}


@pytest.mark.parametrize("env", [{"LIVE_TRADING_ENABLED": "true"}, {"TRADING_MODE": "LIVE"}])
def test_runner_refuses_non_paper_mode(monkeypatch, tmp_path, env):
    runner = _lab_runner(monkeypatch, _ExpiredApiService(_spot_frame()))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(LiveTradingSafetyError):
        runner.run(out_dir=tmp_path / "out", splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert not (tmp_path / "out").exists()


def test_runner_blocks_broker_order_calls(monkeypatch, tmp_path):
    from src.execution.order_manager import OrderManager
    import scripts.run_shadow_historical_validation as hv

    runner = _lab_runner(monkeypatch, _ExpiredApiService(_spot_frame()))
    attempts = []

    def signal_fn(frame):
        try:
            OrderManager.submit_signal(None)
        except LiveTradingSafetyError as exc:
            attempts.append(str(exc))
        return _signal_fn(frame)

    monkeypatch.setattr(hv, "s2_components", lambda svc, name: (
        RC, PARAMS, None, {"range_start": "09:15", "range_end": "09:30"}, signal_fn, None))
    runner.run(out_dir=tmp_path / "out", splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert attempts and all("SHADOW_VALIDATION_ORDER_BLOCKED" in a for a in attempts)


def test_lab_modules_have_no_order_imports():
    for path in (Path(lab.__file__), Path("scripts/run_buy_strategy_lab.py")):
        src = path.read_text(encoding="utf-8")
        for forbidden in ("from src.broker", "from src.execution", "from src.engine", "place_order(",
                          "submit_signal("):
            assert forbidden not in src
