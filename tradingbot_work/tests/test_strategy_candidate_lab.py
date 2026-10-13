from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import strategy_candidate_lab as lab
from src.backtest.current_day_s2 import LevelParams
from src.options.models import OptionResolutionError

IST = lab.IST
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=14, stop_atr_multiplier=1.5,
                     target_r=2.0, slippage_bps=0.0)


# ------------------------------------------------------------------ fixtures
def _days(n: int, start: date = date(2026, 9, 1)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _spot(days: list[date], seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], 24000.0
    for d in days:
        drift = rng.choice([-6.0, 6.0])
        for i in range(75):
            ts = pd.Timestamp(datetime.combine(d, time(9, 15)), tz=IST) + pd.Timedelta(minutes=5 * i)
            o = price
            c = o + drift + rng.normal(0, 12)
            h, l = max(o, c) + abs(rng.normal(0, 5)), min(o, c) - abs(rng.normal(0, 5))
            rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 0.0})
            price = c
    return pd.DataFrame(rows)


def _futures(spot: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = spot.copy()
    for col in ("open", "high", "low", "close"):
        f[col] = f[col] + 40.0
    f["volume"] = rng.integers(1_000, 50_000, len(f)).astype(float)
    return f


def _contract(direction: str, expiry: date = date(2026, 10, 6)):
    kind = "CE" if direction == "BUY" else "PE"
    return SimpleNamespace(instrument_key=f"NSE_FO|{kind}", trading_symbol=f"NIFTY 24000 {kind}", expiry=expiry,
                           strike=24000.0, option_type=SimpleNamespace(value=kind), tick_size=0.05, lot_size=65)


def _premium(spot: pd.DataFrame, direction: str, invert_days: set[date] = frozenset()) -> pd.DataFrame:
    """Option premium that moves with the index (CE) or against it (PE); inverted on chosen days."""
    sign = 1.0 if direction == "BUY" else -1.0
    base = spot["close"].iloc[0]
    out = spot[["timestamp"]].copy()
    flip = np.where(spot["timestamp"].dt.date.isin(invert_days), -1.0, 1.0)
    for col in ("open", "high", "low", "close"):
        out[col] = 200.0 + sign * flip * (spot[col] - base) * 0.5
    out["high"], out["low"] = out[["open", "high", "low", "close"]].max(axis=1), out[["open", "high", "low",
                                                                                     "close"]].min(axis=1)
    out["volume"] = 1000.0
    return out.reset_index(drop=True)


def _dataset(spot, futures=None, invert_days=frozenset(), days=None):
    prem = {d: _premium(spot, d, invert_days) for d in ("BUY", "SELL")}
    return lab.Dataset(spot=spot, futures=futures, resolve=lambda s: _contract(s.direction),
                       premium_for=lambda c: prem["BUY" if c.option_type.value == "CE" else "SELL"],
                       days=days or sorted(set(spot["timestamp"].dt.date)))


def _sig(ts: str, direction: str = "BUY", cid: str = "X", spot: float = 24000.0):
    return lab.CandidateSignal(pd.Timestamp(ts, tz=IST).to_pydatetime(), direction, "NIFTY", spot, cid, "ORB_EMA",
                               {"ema20": 1.0, "ema50": 0.0, "atr14": 1.0})


def _trade(day: str, outcome: str, pnl: float, r: float, direction="BUY", bucket="10:00-11:00", regime="RANGE",
           dte=5):
    return {"status": lab.TRADED, "outcome": outcome, "pnl": pnl, "r_multiple": r, "direction": direction,
            "signal_time": f"{day}T10:00:00+05:30", "session_date": day, "time_bucket": bucket, "regime": regime,
            "dte": dte, "reason": None}


# ------------------------------------------------------- look-ahead prevention
def test_features_do_not_change_when_future_bars_change():
    spot = _spot(_days(3))
    fut = _futures(spot)
    full = lab.build_features(spot, fut)
    cut = 120
    poisoned = spot.copy()
    poisoned.loc[cut + 1:, ["open", "high", "low", "close"]] *= 1.10
    pf = fut.copy()
    pf.loc[cut + 1:, "volume"] *= 50
    pf.loc[cut + 1:, ["open", "high", "low", "close"]] *= 0.9
    again = lab.build_features(poisoned, pf)
    cols = ["ema20", "ema50", "rsi14", "atr14", "orb_high", "orb_low", "fut_vwap", "prior_high", "prior_low"]
    pd.testing.assert_frame_equal(full.loc[:cut, cols], again.loc[:cut, cols])


def test_orb_levels_unknown_until_range_complete():
    f = lab.build_features(_spot(_days(2)), orb_minutes=30)
    t = f["timestamp"].dt.time
    assert f.loc[t < time(9, 45), "orb_high"].isna().all()
    assert f.loc[t >= time(9, 45), "orb_high"].notna().all()


def test_every_grid_candidate_passes_truncation_audit():
    spot = _spot(_days(4))
    fut = _futures(spot)
    cuts = lab.audit_cut_points(spot, _days(4), ("10:30", "12:00", "14:00"))
    for spec in lab.candidate_grid():
        assert lab.lookahead_audit(spot, fut, spec, cuts)["passed"], spec.candidate_id


def test_audit_detects_a_leaky_feature(monkeypatch):
    spot = _spot(_days(3))
    fut = _futures(spot)
    real = lab.build_features

    def leaky(s, f=None, **kw):
        out = real(s, f, **kw)
        out["close"] = out["close"].shift(-3).fillna(out["close"])
        return out

    monkeypatch.setattr(lab, "build_features", leaky)
    spec = lab.CandidateSpec("BREAKOUT_ATR", (("range_atr", 0.5),), 1.0)
    cuts = lab.audit_cut_points(spot, _days(3), ("10:30", "11:30", "12:30", "13:30"))
    assert not lab.lookahead_audit(spot, fut, spec, cuts)["passed"]


def test_signals_only_inside_entry_window():
    spot = _spot(_days(3))
    feats = lab.build_features(spot, _futures(spot))
    for spec in lab.candidate_grid():
        for s in lab.generate_signals(feats, spec, stop_new_entries="14:30"):
            assert time(9, 30) <= s.timestamp.time() < time(14, 30)


# ------------------------------------------------------- chronological split
def test_chronological_split_is_ordered_disjoint_and_complete():
    days = _days(22)
    split = lab.chronological_split(reversed(days))
    assert [len(split[k]) for k in lab.SPLITS] == [12, 5, 5]
    assert max(split["TRAIN"]) < min(split["VALIDATION"])
    assert max(split["VALIDATION"]) < min(split["OOS"])
    assert sorted(split["TRAIN"] + split["VALIDATION"] + split["OOS"]) == days
    assert lab.split_of(days[0], split) == "TRAIN" and lab.split_of(days[-1].isoformat(), split) == "OOS"
    assert lab.split_of(date(2030, 1, 1), split) is None


def test_split_requires_three_sessions():
    with pytest.raises(ValueError):
        lab.chronological_split(_days(2))


def test_trades_never_cross_split_boundaries():
    spot = _spot(_days(9))
    report = lab.run_lab(_dataset(spot, _futures(spot)), params=PARAMS, square_off="15:15", stop_new_entries="14:30",
                         specs=lab.candidate_grid()[:4], audit_times=("11:00",))
    for row in report["trades"]:
        if row["exit_time"]:
            assert pd.Timestamp(row["exit_time"]).tz_convert(IST).date().isoformat() == row["session_date"]
        assert row["split"] == lab.split_of(row["session_date"], {k: [date.fromisoformat(d) for d in v]
                                                                  for k, v in report["split_days"].items()})


# --------------------------------------------------------- parameter isolation
def test_selection_reads_train_metrics_only():
    train = {"A": {"resolved_trades": 30, "win_rate": 0.62, "profit_factor": 1.4},
             "B": {"resolved_trades": 30, "win_rate": 0.58, "profit_factor": 3.0},
             "C": {"resolved_trades": 5, "win_rate": 0.90, "profit_factor": 9.0}}
    assert lab.select_by_train(train, min_trades=20) == "A"
    assert lab.select_by_train({"C": train["C"]}, min_trades=20) == "C"


def test_changing_oos_outcomes_does_not_change_selection_or_train_metrics():
    days = _days(9)
    spot = _spot(days)
    fut = _futures(spot)
    oos_days = set(lab.chronological_split(days)["OOS"])
    specs = [s for s in lab.candidate_grid() if s.family in ("BREAKOUT_ATR", "ORB_VWAP")]
    kw = dict(params=PARAMS, square_off="15:15", stop_new_entries="14:30", specs=specs, audit_times=("11:00",))
    a = lab.run_lab(_dataset(spot, fut), **kw)
    b = lab.run_lab(_dataset(spot, fut, invert_days=oos_days), **kw)
    sel = lambda r: sorted(c["candidate_id"] for c in r["candidates"] if c["selected_for_family"])
    assert sel(a) == sel(b)
    for ca, cb in zip(a["candidates"], b["candidates"]):
        assert ca["metrics"]["TRAIN"] == cb["metrics"]["TRAIN"]
    assert any(ca["metrics"]["OOS"] != cb["metrics"]["OOS"] for ca, cb in zip(a["candidates"], b["candidates"]))


def test_current_day_does_not_affect_status_or_selection():
    days = _days(10)
    spot = _spot(days)
    fut = _futures(spot)
    hist_days, cur_day = days[:-1], days[-1]
    hist_spot = spot[spot["timestamp"].dt.date <= hist_days[-1]].reset_index(drop=True)
    hist_fut = fut[fut["timestamp"].dt.date <= hist_days[-1]].reset_index(drop=True)
    specs = lab.candidate_grid()[:8]
    kw = dict(params=PARAMS, square_off="15:15", stop_new_entries="14:30", specs=specs, audit_times=("11:00",))
    without = lab.run_lab(_dataset(hist_spot, hist_fut), **kw)
    with_cur = lab.run_lab(_dataset(hist_spot, hist_fut), current=_dataset(spot, fut, days=[cur_day]),
                           current_day=cur_day, **kw)
    for a, b in zip(without["candidates"], with_cur["candidates"]):
        assert (a["status"], a["selected_for_family"], a["metrics"]) == (b["status"], b["selected_for_family"],
                                                                        b["metrics"])
    assert all(r["split"] == "CURRENT_DAY" for r in with_cur["trades"] if r["session_date"] == cur_day.isoformat())


def test_grid_is_small_and_predeclared():
    grid = lab.candidate_grid()
    assert len(grid) == 32
    assert {s.family for s in grid} == set(lab.FAMILIES)
    assert len({s.candidate_id for s in grid}) == 32


# ------------------------------------------------------ no future data leakage
def test_stop_uses_only_premium_up_to_signal_and_entry_is_next_bar():
    spot = _spot(_days(2))
    prem = _premium(spot, "BUY")
    sig_idx = 100
    sig = _sig(str(spot["timestamp"].iloc[sig_idx].tz_convert(IST).tz_localize(None)))
    wild = prem.copy()
    wild.loc[sig_idx + 1:, "high"] += 500
    wild.loc[sig_idx + 1:, "low"] = 0.05
    common = dict(resolve=lambda s: _contract("BUY"), params=PARAMS, square_off="15:15", max_trades_per_day=6)
    a = lab.simulate([sig], premium_for=lambda c: prem, **common)[0]
    b = lab.simulate([sig], premium_for=lambda c: wild, **common)[0]
    assert a["stop_loss"] == b["stop_loss"] and a["target"] == b["target"]
    assert pd.Timestamp(a["entry_time"]) == pd.Timestamp(a["signal_time"]) + pd.Timedelta(minutes=5)
    assert lab.entry_after_signal([a, b])


def test_outcome_ignores_next_session_candles():
    days = _days(2)
    spot = _spot(days)
    prem = _premium(spot, "BUY")
    day1 = prem["timestamp"].dt.date == days[0]
    prem.loc[day1 & (prem["timestamp"].dt.time > time(14, 0)), ["open", "high", "low", "close"]] = None
    prem = prem.dropna().reset_index(drop=True)
    sig = _sig(f"{days[0]} 13:50")
    row = lab.simulate([sig], resolve=lambda s: _contract("BUY"), premium_for=lambda c: prem, params=PARAMS,
                       square_off="15:15", max_trades_per_day=6)[0]
    if row["exit_time"]:
        assert pd.Timestamp(row["exit_time"]).tz_convert(IST).date() == days[0]
    else:
        assert row["status"] == lab.UNRESOLVED and row["reason"] == "NO_EXIT_EVIDENCE"


# --------------------------------------------------------- missing option data
@pytest.mark.parametrize("premium,reason", [("NO_PREMIUM_DATA", "NO_PREMIUM_DATA"),
                                            (pd.DataFrame(), "NO_PREMIUM_DATA"),
                                            ("PREMIUM_API_ERROR:X", "PREMIUM_API_ERROR:X")])
def test_missing_premium_is_unresolved_not_traded(premium, reason):
    row = lab.simulate([_sig("2026-09-01 10:00")], resolve=lambda s: _contract("BUY"), premium_for=lambda c: premium,
                       params=PARAMS, square_off="15:15", max_trades_per_day=6)[0]
    assert row["status"] == lab.UNRESOLVED and row["reason"] == reason and row["pnl"] is None
    m = lab.compute_metrics([row])
    assert m["resolved_trades"] == 0 and m["unresolved"] == 1 and m["win_rate"] is None


def test_missing_entry_candle_is_not_filled_in():
    spot = _spot(_days(1))
    prem = _premium(spot, "BUY")
    sig_ts = prem["timestamp"].iloc[40]
    prem = prem[prem["timestamp"] != sig_ts + pd.Timedelta(minutes=5)].reset_index(drop=True)
    row = lab.simulate([_sig(str(sig_ts.tz_convert(IST).tz_localize(None)))], resolve=lambda s: _contract("BUY"),
                       premium_for=lambda c: prem, params=PARAMS, square_off="15:15", max_trades_per_day=6)[0]
    assert row["status"] == lab.UNRESOLVED and row["reason"] == "NO_PREMIUM_ENTRY_CANDLE"


def test_unresolvable_contract_is_reported():
    def resolve(_):
        raise OptionResolutionError("NO_EXPIRY", "none")
    row = lab.simulate([_sig("2026-09-01 10:00")], resolve=resolve, premium_for=lambda c: None, params=PARAMS,
                       square_off="15:15", max_trades_per_day=6)[0]
    assert row["status"] == lab.UNRESOLVED and row["reason"] == "CONTRACT_UNRESOLVED:NO_EXPIRY"


def test_historical_1dte_flag_reported_when_unavailable():
    spot = _spot(_days(4))
    r = lab.run_lab(_dataset(spot), params=PARAMS, square_off="15:15", stop_new_entries="14:30",
                    specs=lab.candidate_grid()[:1], audit_times=("11:00",))
    assert lab.HISTORICAL_1DTE_UNAVAILABLE in r["data_flags"]
    r2 = lab.run_lab(_dataset(spot), params=PARAMS, square_off="15:15", stop_new_entries="14:30",
                     specs=lab.candidate_grid()[:1], audit_times=("11:00",), historical_1dte_available=True)
    assert r2["data_flags"] == []


def test_vwap_rules_do_not_fire_without_real_futures_volume():
    spot = _spot(_days(3))
    feats = lab.build_features(spot, None)
    assert feats["fut_vwap"].isna().all()
    for spec in lab.candidate_grid():
        if spec.family in ("ORB_VWAP", "TREND_RSI_VWAP", "EMA_VWAP_PULLBACK"):
            assert lab.generate_signals(feats, spec) == []


# ------------------------------------------------------------ duplicate signals
def test_duplicate_signals_are_collapsed():
    s = _sig("2026-09-01 10:00")
    assert len(lab.dedupe_signals([s, s, _sig("2026-09-01 10:00")])) == 1
    assert len(lab.dedupe_signals([s, _sig("2026-09-01 10:00", "SELL")])) == 2


def test_generator_emits_at_most_one_signal_per_bar():
    spot = _spot(_days(4))
    feats = lab.build_features(spot, _futures(spot))
    for spec in lab.candidate_grid():
        sigs = lab.generate_signals(feats, spec)
        stamps = [s.timestamp for s in sigs]
        assert len(stamps) == len(set(stamps)), spec.candidate_id


def test_overlapping_signals_are_kept_as_skipped_not_deleted():
    spot = _spot(_days(1))
    prem = _premium(spot, "BUY")
    t0 = prem["timestamp"].iloc[30].tz_convert(IST).tz_localize(None)
    sigs = [_sig(str(t0)), _sig(str(t0)), _sig(str(t0 + pd.Timedelta(minutes=5)))]
    rows = lab.simulate(sigs, resolve=lambda s: _contract("BUY"), premium_for=lambda c: prem, params=PARAMS,
                        square_off="15:15", max_trades_per_day=6)
    assert len(rows) == 2
    assert rows[0]["status"] == lab.TRADED
    assert rows[1]["status"] in (lab.SKIPPED_OPEN, lab.TRADED)
    if pd.Timestamp(rows[0]["exit_time"]) >= pd.Timestamp(rows[1]["signal_time"]):
        assert rows[1]["status"] == lab.SKIPPED_OPEN


def test_first_breakout_only_caps_one_trade_per_day():
    spot = _spot(_days(5))
    feats = lab.build_features(spot, _futures(spot))
    spec = lab.CandidateSpec("FIRST_BREAKOUT_ONLY", (("orb_minutes", 15), ("buffer_atr", 0.0)), 1.0, 1)
    per_day = pd.Series([s.timestamp.date() for s in lab.generate_signals(feats, spec)]).value_counts()
    assert (per_day <= 1).all()


# ---------------------------------------------------------- metrics calculation
def test_metrics_calculation():
    rows = [_trade("2026-09-01", "WIN", 200, 1.0), _trade("2026-09-01", "LOSS", -100, -1.0, "SELL"),
            _trade("2026-09-02", "LOSS", -100, -1.0, "SELL", "11:00-12:00"), _trade("2026-09-02", "SCRATCH", 0, 0.0),
            _trade("2026-09-03", "WIN", 300, 1.5, regime="TREND_UP"),
            {"status": lab.UNRESOLVED, "reason": "NO_PREMIUM_DATA"}, {"status": lab.SKIPPED_OPEN, "reason": None}]
    m = lab.compute_metrics(rows)
    assert m["total_signals"] == 7 and m["resolved_trades"] == 5
    assert (m["wins"], m["losses"], m["scratches"]) == (2, 2, 1)
    assert m["win_rate"] == pytest.approx(0.4)
    assert m["profit_factor"] == pytest.approx(500 / 200)
    assert m["net_pnl"] == 300
    assert m["max_drawdown"] == 200
    assert m["average_r"] == pytest.approx(0.1)
    assert m["max_consecutive_losses"] == 3
    assert m["unresolved"] == 1 and m["skipped_position_open"] == 1
    assert m["by_direction"]["SELL"]["trades"] == 2 and m["by_direction"]["SELL"]["wins"] == 0
    assert m["by_direction"]["BUY"]["win_rate"] == pytest.approx(2 / 3)
    assert m["by_time_of_day"]["11:00-12:00"]["trades"] == 1
    assert m["by_regime"]["TREND_UP"]["wins"] == 1
    assert m["trading_days"] == 3


def test_profit_factor_without_losses_is_infinite_and_serialisable(tmp_path):
    m = lab.compute_metrics([_trade("2026-09-01", "WIN", 100, 1.0)])
    assert m["profit_factor"] == math.inf
    assert json.loads(json.dumps(lab._jsonable(m)))["profit_factor"] == "inf"


def test_ambiguous_candle_counts_as_loss():
    spot = _spot(_days(1))
    prem = _premium(spot, "BUY")
    i = 40
    sig = _sig(str(prem["timestamp"].iloc[i].tz_convert(IST).tz_localize(None)))
    prem.loc[i + 1, "high"] = prem.loc[i + 1, "open"] + 1_000
    prem.loc[i + 1, "low"] = 0.05
    row = lab.simulate([sig], resolve=lambda s: _contract("BUY"), premium_for=lambda c: prem, params=PARAMS,
                       square_off="15:15", max_trades_per_day=6)[0]
    assert row["outcome"] == "LOSS" and row["r_multiple"] == pytest.approx(-1.0)
    assert row["exit_reason"] == "SL_AND_TARGET_SAME_CANDLE_COUNTED_AS_LOSS"


def _m(n, wr, pf=1.5, pnl=100.0):
    return {"resolved_trades": n, "win_rate": wr, "profit_factor": pf, "net_pnl": pnl}


@pytest.mark.parametrize("train,val,oos,ok,expected", [
    (_m(30, .65), _m(10, .60), _m(10, .60), True, "ROBUST"),
    (_m(30, .65), _m(10, .60), _m(10, .60), False, "REJECTED"),
    (_m(30, .65), _m(10, .40), _m(10, .60), True, "OVERFIT"),
    (_m(30, .65), _m(10, .60), _m(10, .60, pf=0.9, pnl=-5), True, "OVERFIT"),
    (_m(10, .65), _m(10, .60), _m(10, .60), True, "INSUFFICIENT_DATA"),
    (_m(30, .65), _m(10, .60), _m(5, .60), True, "INSUFFICIENT_DATA"),
    (_m(30, .55), _m(10, .52), _m(10, .51), True, "PROMISING"),
    (_m(30, .45), _m(10, .40), _m(10, .30, pf=.5, pnl=-50), True, "REJECTED"),
])
def test_classification(train, val, oos, ok, expected):
    assert lab.classify(train, val, oos, ok)[0] == expected


def test_reports_written(tmp_path):
    spot = _spot(_days(6))
    r = lab.run_lab(_dataset(spot, _futures(spot)), params=PARAMS, square_off="15:15", stop_new_entries="14:30",
                    specs=lab.candidate_grid()[:4], audit_times=("11:00",))
    paths = lab.write_outputs(r, tmp_path)
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    body = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "trades" not in body and body["ranked"] and body["best_candidate"]
    md = paths["md"].read_text(encoding="utf-8")
    assert "| Candidate | Train WR" in md and "Current Day WR" in md
    trades = pd.read_csv(paths["trades"])
    assert len(trades) == len(r["trades"])
