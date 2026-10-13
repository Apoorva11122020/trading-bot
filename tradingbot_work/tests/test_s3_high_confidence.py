from __future__ import annotations

import csv
import inspect
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import accuracy_70_lab as acc
from src.backtest import research_strategies
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s3_high_confidence_lab as s3lab
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.backtest.current_day_s2 import LevelParams
from src.backtest.live_vs_historical import time_bucket
from src.safety import live_gate
from src.strategy import s3_high_confidence as s3

IST = imp.IST
ROOT = Path(__file__).resolve().parents[1]
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=14, stop_atr_multiplier=1.5,
                     target_r=2.0, slippage_bps=0.0)
TODAY = date(2026, 10, 6)


def _days(n, start=date(2026, 9, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _days(20)
SPLIT_DAYS = {"TRAIN": DAYS[:10], "VALIDATION": DAYS[10:15], "OOS": DAYS[15:]}
SPLITS = imp.split_dates({"TRAIN": (DAYS[0], DAYS[9]), "VALIDATION": (DAYS[10], DAYS[14]),
                          "OOS": (DAYS[15], DAYS[19])})


def _ts(day: str, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(date.fromisoformat(day), time.fromisoformat(hhmm)), tz=IST)


def _spot(days=DAYS, seed=5):
    rng = np.random.default_rng(seed)
    rows, price = [], 24000.0
    for d in days:
        drift = rng.choice([-6.0, 0.0, 6.0])
        for i in range(75):
            ts = _ts(d, "09:15") + pd.Timedelta(minutes=5 * i)
            o = price
            c = o + drift + rng.normal(0, 10)
            rows.append({"timestamp": ts, "open": o, "high": max(o, c) + abs(rng.normal(0, 4)),
                         "low": min(o, c) - abs(rng.normal(0, 4)), "close": c, "volume": 0.0})
            price = c
    return pd.DataFrame(rows)


def _contract(direction):
    kind = "CE" if direction == "BUY" else "PE"
    return SimpleNamespace(instrument_key=f"NSE_FO|{kind}", trading_symbol=f"NIFTY 24000 {kind} 06 OCT 26",
                           expiry=date(2026, 10, 6), strike=24000.0, option_type=SimpleNamespace(value=kind),
                           tick_size=0.05, lot_size=65)


def _premium(spot, kind):
    sign = 1.0 if kind == "CE" else -1.0
    base = spot["close"].iloc[0]
    out = spot[["timestamp"]].copy()
    for col in ("open", "high", "low", "close"):
        out[col] = 600.0 + sign * (spot[col] - base) * 0.5
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = 1000.0 + (np.arange(len(out)) % 7) * 300.0
    return out.reset_index(drop=True)


def _premium_for(spot):
    prem = {k: _premium(spot, k) for k in ("CE", "PE")}
    return lambda c: prem[c.option_type.value]


def _resolve(sig):
    return _contract(sig.direction)


def _row(day, hhmm, direction, win, pnl=None):
    ts = _ts(day, hhmm)
    pnl = (650.0 if win else -325.0) if pnl is None else pnl
    return {"session_date": day, "signal_time": ts.isoformat(), "direction": direction, "decision": imp.ACCEPTED,
            "skip_reason": None, "state": sh.EVALUATED, "outcome": imp.WIN if win else imp.LOSS, "pnl": pnl,
            "r_multiple": 2.0 if win else -1.0, "time_bucket": time_bucket(ts), "dte": 5,
            "dte_bucket": sh.dte_bucket(5), "regimes": [], "contract": "NIFTY CE", "break_atr": 0.3,
            "premium_atr": 2.0, "reason": None, "reason_raw": None, "strategy": lab.BASELINE}


BASE_ROWS = [_row(d, hh, dr, (i + j) % 2 == 0) for i, d in enumerate(DAYS)
             for j, (hh, dr) in enumerate((("10:00", "BUY"), ("11:30", "SELL")))]


@pytest.fixture(scope="module")
def built():
    spot = _spot()
    base = [dict(r) for r in BASE_ROWS]
    a = s3lab.build_s3(spot=spot, split_days=SPLIT_DAYS, splits=SPLITS, resolve=_resolve,
                       premium_for=_premium_for(spot), params=PARAMS, square_off="15:15", today=TODAY,
                       base_rows=base, run_mutation=False, lookahead_days=DAYS[-2:])
    return spot, a, base


def _status(a):
    return {x["id"]: x["status"] for x in a["integrity"]}


def _points(**kw):
    p = {k: 0 for k in s3.UNDERLYING_COMPONENTS}
    p.update(trend=20, vwap=15, ema=15, momentum=10, price_action=0, breakout=0)
    p.update(available=True, tradeable_regime=True, regime="TREND_UP")
    p.update(kw)
    p["underlying"] = sum(p[k] for k in s3.UNDERLYING_COMPONENTS)
    return p


OPT_FULL = {"premium_mom3": 0.02, "premium_body": 0.6, "premium_bull": True, "premium_above_vwap": True,
            "premium_vol_ratio": 1.8}


# ------------------------------------------------------------------ score / weights / tiers
def test_weights_are_the_documented_starting_weights():
    assert s3.WEIGHTS == {"trend": 20, "vwap": 15, "ema": 15, "momentum": 10, "volume": 10, "price_action": 10,
                          "breakout": 10, "option_premium": 10}
    assert sum(s3.WEIGHTS.values()) == 100
    assert sum(s3.WEIGHTS[k] for k in s3.UNDERLYING_COMPONENTS) == 80
    assert s3.THRESHOLDS == (60, 65, 70, 75, 80, 85, 90)


def test_score_is_sum_of_components_and_capped_at_100(built):
    spot, _, _ = built
    f = s3.s3_bars(spot)
    for d in s3.DIRECTIONS:
        p = s3.underlying_points(f, s3.PROFILES[d])
        for k in s3.UNDERLYING_COMPONENTS:
            assert p[k].between(0, s3.WEIGHTS[k]).all(), (d, k)
        assert (p["underlying"] == p[list(s3.UNDERLYING_COMPONENTS)].sum(axis=1)).all()
    dec = s3.decide("BUY", _points(price_action=10, breakout=10), OPT_FULL, 60)
    assert dec.decision == "BUY" and dec.score == 100 and sum(dec.components.values()) == 100


def test_threshold_gate_and_tiers():
    pts = _points()
    assert s3.decide("BUY", pts, OPT_FULL, 80).decision == "BUY"
    low = s3.decide("BUY", pts, OPT_FULL, 85)
    assert low.decision == s3.NO_SIGNAL and low.reason == "SCORE_BELOW_THRESHOLD" and low.score == 80
    assert [s3.score_tier(v) for v in (59, 60, 69, 70, 79, 80, 89, 90, 100)] == [
        None, "S3-A", "S3-A", "S3-B", "S3-B", "S3-C", "S3-C", "S3-D", "S3-D"]


def test_threshold_signals_edge_trigger_and_conflict_is_no_signal():
    spot = _spot(DAYS[:2])
    f = s3.s3_bars(spot)
    iso = [t.isoformat() for t in f["timestamp"]]
    mask = pd.Series(True, index=f.index)
    totals = {"BUY": {iso[100]: 70, iso[101]: 75, iso[103]: 90, iso[110]: 80},
              "SELL": {iso[110]: 85, iso[120]: 60}}
    keys, conflicts = s3lab.threshold_signals(f, totals, 70, mask)
    assert keys["BUY"] == [(iso[100], "BUY"), (iso[103], "BUY")]
    assert keys["SELL"] == [] and conflicts == 2
    keys80, _ = s3lab.threshold_signals(f, totals, 80, mask)
    assert keys80["BUY"] == [(iso[103], "BUY")]
    assert s3.resolve_conflict(s3.decide("BUY", _points(), OPT_FULL, 60),
                               s3.decide("SELL", _points(regime="TREND_DOWN"), OPT_FULL, 60)) == s3.NO_SIGNAL


# ------------------------------------------------------------------ option premium confirmation / missing data
def test_missing_premium_is_no_signal_never_fabricated():
    for opt in (None, dict(OPT_FULL, premium_mom3=None), dict(OPT_FULL, premium_body=None)):
        d = s3.decide("BUY", _points(), opt, 60)
        assert d.decision == s3.NO_SIGNAL and d.reason == "PREMIUM_DATA_UNAVAILABLE"
    no_vol = s3.option_points(dict(OPT_FULL, premium_vol_ratio=None), s3.BUY_PROFILE)
    assert no_vol["volume"] == 0 and no_vol["volume_available"] is False


def test_buy_and_sell_premium_rules_are_asymmetric():
    two = dict(OPT_FULL, premium_above_vwap=False)
    assert s3.option_points(two, s3.BUY_PROFILE)["option_premium"] == 5
    assert s3.option_points(two, s3.SELL_PROFILE)["option_premium"] == 10
    assert s3.BUY_PROFILE != s3.SELL_PROFILE
    assert s3.BUY_PROFILE.max_streak == 3 and s3.SELL_PROFILE.max_streak is None
    assert s3.BUY_PROFILE.max_ext_atr < s3.SELL_PROFILE.max_ext_atr


def test_option_features_ignore_future_premium(built):
    _, a, _ = built
    assert a["lookahead"]["option"]["passed"] and a["lookahead"]["option"]["checked"] > 0


def test_premium_confirmation_mode_reads_no_future_premium():
    spot = _spot(DAYS[:3])
    f = s3.s3_bars(spot)
    prem = _premium(spot, "CE")
    keys = [(t.isoformat(), "BUY") for t in f["timestamp"].iloc[60:200:7]]
    mask = acc.allowed_mask(f)
    ctx = {k: {"data_status": "OK", "premium": prem} for k in keys}
    base = s3lab.confirmations(f, keys, "E_PREMIUM_CONFIRMATION", mask, ctx)
    for setup_iso, _ in keys:
        nxt = pd.Timestamp(setup_iso) + pd.Timedelta(minutes=5)
        mut = imp.mutate_after(prem, nxt, 11)
        one = s3lab.confirmations(f, [(setup_iso, "BUY")], "E_PREMIUM_CONFIRMATION", mask,
                                  {(setup_iso, "BUY"): {"data_status": "OK", "premium": mut}})
        assert one == {k: v for k, v in base.items() if v[0] == setup_iso}
    missing = s3lab.confirmations(f, keys, "E_PREMIUM_CONFIRMATION", mask, {})
    assert missing == {}


# ------------------------------------------------------------------ regime / completed candles
def _regime_frame(**over):
    base = {"close": 101.0, "ema20": 100.0, "ema50": 99.0, "ema20_slope_atr": 0.3, "dist_ema20_atr": 0.6,
            "atr_pctile": 0.5, "flips12": 3.0, "gap_so_far": False}
    rows = [dict(base, **o) for o in over.get("rows", [{}])]
    return pd.DataFrame(rows)


def test_regime_classification():
    f = _regime_frame(rows=[
        {}, {"close": 99.0, "ema20": 100.0, "ema50": 101.0, "ema20_slope_atr": -0.3},
        {"flips12": 9.0}, {"ema20_slope_atr": 0.01, "dist_ema20_atr": 0.1},
        {"ema20_slope_atr": 0.1, "atr_pctile": 0.9}, {"ema20_slope_atr": 0.1, "atr_pctile": 0.1},
        {"ema20_slope_atr": 0.1}, {"rsi": 1, "ema50": np.nan}, {"gap_so_far": True}])
    reg, vol = s3.classify_regimes(f)
    assert list(reg) == ["TREND_UP", "TREND_DOWN", "RANGE_CHOP", "RANGE_CHOP", "HIGH_VOLATILITY", "LOW_VOLATILITY",
                         "UNCERTAIN", "UNCERTAIN", "UNCERTAIN"]
    assert vol.iloc[4] == "HIGH_VOLATILITY" and vol.iloc[7] == "UNKNOWN"
    for r in s3.NO_TRADE_REGIMES:
        d = s3.decide("BUY", _points(tradeable_regime=False, regime=r), OPT_FULL, 60)
        assert d.decision == s3.NO_SIGNAL and d.reason == f"REGIME_{r}"


def test_missing_candle_marks_rest_of_session_uncertain():
    spot = _spot(DAYS[:7])
    hole = _ts(DAYS[6], "11:00")
    f = s3.s3_bars(spot[spot["timestamp"] != hole].reset_index(drop=True))
    day = f[f["day"] == date.fromisoformat(DAYS[6])]
    after = day[day["timestamp"] > hole]
    before = day[day["timestamp"] < hole]
    assert after["gap_so_far"].all() and (after["regime"] == "UNCERTAIN").all()
    assert not before["gap_so_far"].any()
    p = s3.underlying_points(f, s3.BUY_PROFILE)
    assert not p.loc[after.index, "available"].any()
    late = s3.s3_bars(spot[spot["timestamp"] != _ts(DAYS[6], "09:15")].reset_index(drop=True))
    assert late[late["day"] == date.fromisoformat(DAYS[6])]["gap_so_far"].all()
    assert s3.decide("BUY", _points(available=False), OPT_FULL, 60).reason == "DATA_UNAVAILABLE"


def test_scores_regime_and_setups_unchanged_under_truncation(built):
    _, a, _ = built
    t = a["lookahead"]["truncation"]
    assert t["passed"] and t["cuts"] == 4
    assert _status(a)["L1"] == acc.PASS


def test_truncation_test_catches_a_leaky_feature(monkeypatch):
    spot = _spot(DAYS[:8])
    orig = s3.s3_bars

    def leaky(frame, interval=5):
        f = orig(frame, interval)
        f["rsi14"] = f["rsi14"].shift(-1)
        return f

    monkeypatch.setattr(s3, "s3_bars", leaky)
    res = s3lab.truncation_test(spot, DAYS[6:8], cut_times=("11:00", "12:00"))
    assert not res["passed"] and res["failures"]


def test_pipeline_mutation_test_passes_for_s3_setups():
    spot = _spot(DAYS[:8])
    res = s3lab.pipeline_mutation_test(spot, acc.default_allowed, _resolve, _premium_for(spot), PARAMS, "NIFTY",
                                       DAYS[6:8])
    assert res["passed"] and res["cuts"] == 2


def test_point_in_time_contracts_next_candle_entries_and_aligned_option_candles(built):
    _, a, _ = built
    st = _status(a)
    for cid in ("C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3"):
        assert st[cid] == acc.PASS, cid
    for rows in a["_rows"].values():
        for r in rows:
            assert pd.Timestamp(r["s3_setup_time"]) <= pd.Timestamp(r["signal_time"])


# ------------------------------------------------------------------ entry modes
def test_confirmation_modes_are_causal_and_same_session():
    spot = _spot(DAYS[:4])
    f = s3.s3_bars(spot)
    mask = acc.allowed_mask(f)
    keys = [(t.isoformat(), d) for t in f.loc[mask, "timestamp"].iloc[::5] for d in s3.DIRECTIONS]
    for mode in s3lab.MODES:
        mp = s3lab.confirmations(f, keys, mode, mask, {})
        for k2, k1 in mp.items():
            t1, t2 = pd.Timestamp(k1[0]), pd.Timestamp(k2[0])
            assert k1[1] == k2[1] and t1.date() == t2.date()
            gap = (t2 - t1) / pd.Timedelta(minutes=5)
            assert gap == 0 if mode == "A_NEXT_BAR" else 1 <= gap <= (3 if mode == "D_BREAKOUT_RETEST" else 1)
    j = f.index[(f["timestamp"] == _ts(DAYS[1], "10:00"))][0]
    up = f.loc[j + 1, "close"] > f.loc[j, "high"]
    got = s3lab.confirmations(f, [(f.loc[j, "timestamp"].isoformat(), "BUY")], "C_BREAKOUT_CONTINUATION", mask, {})
    assert bool(got) == bool(up)


# ------------------------------------------------------------------ BUY / SELL isolation
def test_buy_and_sell_candidates_are_isolated(built):
    _, a, _ = built
    assert len(a["scores"]) == 70
    for n, rows in a["_rows"].items():
        d = a["scores"][n]["direction"]
        assert n.startswith(f"S3_{d}_") and all(r["direction"] == d for r in rows)
    st = _status(a)
    assert st["I1"] == st["I2"] == st["I3"] == st["I4"] == acc.PASS


def test_buy_metrics_unaffected_by_sell_rows(built):
    _, a, _ = built
    periods = {**SPLIT_DAYS, "ALL": DAYS}
    folds = lab.walk_forward_folds(DAYS, 8, 3, 3)
    n = s3lab.cid("BUY", 60, "A_NEXT_BAR")
    before = acc.score_rows(a["_rows"][n], periods, folds, {}, acc.Criteria())
    poisoned = {k: [dict(r, pnl=-1e6) for r in v] for k, v in a["_rows"].items() if k.startswith("S3_SELL_")}
    assert poisoned
    after = acc.score_rows(a["_rows"][n], periods, folds, {}, acc.Criteria())
    assert before["ALL"] == after["ALL"]


# ------------------------------------------------------------------ splits / selection / walk-forward
def test_chronological_split_and_current_day_excluded(built):
    _, a, _ = built
    st = _status(a)
    assert st["S1"] == st["S2"] == acc.PASS
    bad = acc.split_checks({"TRAIN": [DAYS[5]], "VALIDATION": [DAYS[2]], "OOS": [DAYS[9]]}, SPLITS, TODAY)
    assert bad[0]["status"] == acc.FAIL


def test_finalists_and_best_tier_never_read_oos(built):
    _, a, _ = built
    fake = {n: dict(s, OOS=dict(s["OOS"], win_rate=1.0, profit_factor=99.0, net_pnl=1e9))
            for n, s in a["scores"].items()}
    for d in s3.DIRECTIONS:
        assert acc.pick(fake, [n for n in fake if fake[n]["direction"] == d]) == a["finalists"][d]
    tiers = [dict(t, accuracy=1.0, trades=999) if t["period"] == "OOS" else t for t in a["tiers"]]
    bt, bt2 = s3lab.best_tier(a["tiers"]), s3lab.best_tier(tiers)
    assert (bt and (bt["direction"], bt["tier"])) == (bt2 and (bt2["direction"], bt2["tier"]))


def test_walk_forward_ordering(built):
    _, a, _ = built
    folds = lab.walk_forward_folds(DAYS, 8, 3, 3)
    for f in folds:
        assert max(f["train"]) < min(f["validation"]) and max(f["validation"]) < min(f["test"])
    tests = [d for f in folds for d in f["test"]]
    assert tests == sorted(tests) and len(tests) == len(set(tests))
    for d in s3.DIRECTIONS:
        assert len(a["wf_procedure"][d]["folds"]) == len(folds)


# ------------------------------------------------------------------ day stability / status / recommendation
def test_day_stability_flags_single_day_dependence():
    rows = [_row(DAYS[0], "10:00", "BUY", True, pnl=5000.0), _row(DAYS[1], "10:00", "BUY", False),
            _row(DAYS[2], "10:00", "BUY", True, pnl=200.0)]
    p = s3lab.day_stability(rows, DAYS[:3])
    assert p["best_day"] == DAYS[0] and p["worst_day"] == DAYS[1] and p["active_days"] == 3
    assert p["worst_day_loss"] == -325.0 and p["median_day_pnl"] == 200.0
    assert p["best_day_share_of_net"] > acc.Criteria().max_all_best_day_share


def _m(n, wr, pf, pnl, dd=1000.0, best_share=0.3):
    w = round(n * wr)
    return {"evaluated": n, "wins": w, "losses": n - w, "win_rate": wr, "profit_factor": pf, "net_pnl": pnl,
            "max_drawdown": dd, "best_day_share": best_share, "pnl_excl_best_day": pnl * 0.5,
            "pnl_excl_largest_win": pnl * 0.7, "max_consecutive_losses": 3}


def _s(oos, la=True):
    c = acc.Criteria()
    s = {"TRAIN": _m(60, 0.7, 2.0, 5000.0), "VALIDATION": _m(30, 0.72, 2.2, 4000.0), "OOS": oos,
         "ALL": _m(150, 0.7, 2.0, 15000.0), "walk_forward": {"wf_profitable_share": 0.8, "wf_pf": 1.6},
         "day_all": {"profitable_day_share": 0.7, "best_day_share_of_net": 0.2}, "regime_share": 0.8,
         "n_conditions": 1, "lookahead_passed": la, "data_status": "OK"}
    s["tier"] = acc.sample_tier(oos["evaluated"], c)
    s["gates"] = acc.gates(s, c)
    st, _ = acc.candidate_status(s, c)
    s["status"] = s3lab.s3_status(st)
    return s


def test_sample_categories_and_status_mapping():
    c = acc.Criteria()
    assert [acc.sample_tier(n, c) for n in (60, 30, 15, 5)] == ["A", "B", "C", "D"]
    assert _s(_m(5, 1.0, 50.0, 3000.0))["status"] == s3lab.INSUFFICIENT_DATA
    assert _s(_m(40, 0.75, 2.4, 9000.0, dd=2000.0))["status"] == s3lab.ROBUST_70
    assert _s(_m(20, 0.75, 2.5, 6000.0))["status"] == s3lab.PROMISING
    assert _s(_m(40, 0.75, 2.4, 9000.0, best_share=0.8))["status"] == s3lab.REJECTED
    assert _s(_m(40, 0.75, 2.4, 9000.0), la=False)["status"] == s3lab.REJECTED


def test_recommendation_logic():
    c = acc.Criteria()
    strong, small = _s(_m(40, 0.75, 2.4, 9000.0, dd=2000.0)), _s(_m(20, 0.75, 2.5, 6000.0))
    tiny, losing = _s(_m(5, 1.0, 50.0, 3000.0)), _s(_m(40, 0.3, 0.6, -4000.0))
    below = _s(_m(40, 0.55, 1.6, 5000.0, dd=2000.0))
    sc = {"S": strong, "P": small, "T": tiny, "L": losing, "B": below}
    assert s3lab.recommend({"BUY": "L", "SELL": "S"}, sc, True, {}, c) == s3lab.ROBUST_70
    assert s3lab.recommend({"BUY": "S", "SELL": "S"}, sc, False, {}, c) == s3lab.REJECTED
    assert s3lab.recommend({"BUY": "P", "SELL": None}, sc, True, {}, c) == s3lab.PROMISING
    assert s3lab.recommend({"BUY": "T", "SELL": None}, sc, True, {}, c) == s3lab.INSUFFICIENT_DATA
    assert s3lab.recommend({"BUY": None, "SELL": None}, sc, True, {"BUY": 3, "SELL": 4}, c) == s3lab.INSUFFICIENT_DATA
    assert s3lab.recommend({"BUY": None, "SELL": None}, sc, True, {"BUY": 30}, c) == s3lab.REJECTED
    assert s3lab.recommend({"BUY": "L", "SELL": None}, sc, True, {}, c) == s3lab.REJECTED


# ------------------------------------------------------------------ safety / reports
def test_s2_production_untouched_and_s3_not_registered(built):
    _, a, base = built
    assert base == BASE_ROWS
    assert a["production_s2_changed"] is False and a["live_orders"] == 0
    assert s3.STRATEGY_ID not in research_strategies.STRATEGY_IDS
    for name in ("runtime.py", "signal_generator.py", "strategy_engine.py"):
        p = ROOT / "src" / "strategy" / name
        if p.is_file():
            assert "s3_high_confidence" not in p.read_text(encoding="utf-8")
    assert "s3_high_confidence" not in inspect.getsource(research_strategies)


def test_paper_mode_and_no_broker_orders(built):
    _, a, _ = built
    assert a["trading_mode"] == "PAPER" and a["live_trading_enabled"] is False
    assert live_gate.trading_mode() == "PAPER" and live_gate.live_trading_enabled() is False
    for mod in (s3, s3lab):
        src = inspect.getsource(mod)
        assert "place_order" not in src and "OrderManager" not in src
    from scripts import run_s3_high_confidence_lab as script
    s = inspect.getsource(script)
    assert "place_order" not in s and "order_placement_blocked" in s and "assert_paper_trading_only" in s


def test_reports_written_with_banner_and_sections(built, tmp_path):
    _, a, _ = built
    paths = s3lab.write_reports(a, tmp_path)
    for n in s3lab.REPORT_FILES:
        assert paths[n].is_file() and paths[n].stat().st_size > 0
    md = paths["s3_high_confidence.md"].read_text(encoding="utf-8")
    assert s3.BANNER in md
    for i in range(1, 13):
        assert f"## {i}. " in md
    if not a["target_achieved"]:
        assert s3lab.NOT_ACHIEVED_TEXT in md
    assert a["recommendation"] in md
    with paths["s3_candidates.csv"].open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 70
    with paths["s3_threshold_comparison.csv"].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["threshold"] for r in rows} == {str(t) for t in s3.THRESHOLDS}
    assert {r["scope"] for r in rows} == {"BUY", "SELL", "COMBINED"}
    if a["recommendation"] in (s3lab.ROBUST_70, s3lab.PROMISING):
        assert a["forward_validation"] == s3lab.FORWARD_REQUIRED
