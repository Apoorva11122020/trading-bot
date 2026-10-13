from __future__ import annotations

import json
import math
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import s2_1dte_improvement as imp
from src.backtest.current_day_s2 import LevelParams
from src.backtest.research_strategies import generate_signals
from src.safety.live_gate import LiveTradingSafetyError

IST = imp.IST
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=14, stop_atr_multiplier=1.5,
                     target_r=2.0, slippage_bps=0.0)
CFG = SimpleNamespace(strategies={"S2": {"enabled": True, "range_start": "09:15", "range_end": "09:30",
                                         "atr_period": 14}})


# ----------------------------------------------------------------- fixtures
def _days(n, start=date(2026, 9, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _spot(days, seed=5):
    rng = np.random.default_rng(seed)
    rows, price = [], 24000.0
    for d in days:
        drift = rng.choice([-5.0, 5.0])
        for i in range(75):
            ts = pd.Timestamp(datetime.combine(d, time(9, 15)), tz=IST) + pd.Timedelta(minutes=5 * i)
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
        out[col] = 200.0 + sign * (spot[col] - base) * 0.5
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = 1000.0
    return out.reset_index(drop=True)


def _signal_fn(frame):
    return [s for s in generate_signals("S2", frame, "NIFTY", CFG) if s.timestamp.time() < time(14, 30)]


def _premium_for(spot):
    prem = {k: _premium(spot, k) for k in ("CE", "PE")}
    return lambda c: prem[c.option_type.value]


def _splits(days):
    return {"TRAIN": (days[0].isoformat(), days[3].isoformat()),
            "VALIDATION": (days[4].isoformat(), days[5].isoformat()),
            "OOS": (days[6].isoformat(), days[7].isoformat())}


def _run(days=None, with_current=True):
    days = days or _days(9)
    spot = _spot(days)
    hist = spot[spot["timestamp"].dt.date <= days[7]].reset_index(drop=True)
    pf = _premium_for(spot)
    return imp.run_research(hist_spot=hist, signal_fn=_signal_fn, resolve=lambda s: _contract(s.direction),
                            hist_premium_for=pf, params=PARAMS, current_spot=spot if with_current else None,
                            current_premium_for=pf if with_current else None,
                            current_day=days[8] if with_current else None, splits=_splits(days),
                            current_cut_every_min=60)


_CACHE: dict = {}


def _run_cached(with_current=True):
    if with_current not in _CACHE:
        _CACHE[with_current] = _run(with_current=with_current)
    return _CACHE[with_current]


def _feats(rows):
    """Hand-built bar closes with a fixed opening range 100..110."""
    f = pd.DataFrame(rows, columns=["t", "close"])
    f["timestamp"] = [pd.Timestamp(f"2026-09-01 {t}", tz=IST) for t in f["t"]]
    f["orb_high"], f["orb_low"] = 110.0, 100.0
    return f


def _ctx(t, direction="BUY", day="2026-09-01", pct=0.1, a=0.3):
    ts = pd.Timestamp(f"{day} {t}", tz=IST)
    return {"signal_time": ts, "session_date": ts.date(), "direction": direction, "break_pct": pct, "break_atr": a}


# --------------------------------------------------------------- look-ahead
def test_all_predeclared_variants_survive_future_mutation():
    days = _days(4)
    spot = _spot(days)
    pipe = imp._pipeline_factory(_signal_fn, lambda s: _contract(s.direction), PARAMS, "09:15", "09:30")
    cuts = [pd.Timestamp(datetime.combine(d, time(h, m)), tz=IST) for d in days for h, m in ((10, 30), (13, 0))]
    res = imp.mutation_test(pipe, spot, _premium_for(spot), imp.predeclared_variants(), cuts)
    assert all(r["passed"] and r["cuts"] == len(cuts) for r in res.values())


def test_mutation_test_detects_future_leak(monkeypatch):
    days = _days(3)
    spot = _spot(days)
    real = imp.underlying_features

    def leaky(*a, **k):
        f = real(*a, **k)
        f["atr14"] = f["atr14"].shift(-2).fillna(f["atr14"])
        return f

    monkeypatch.setattr(imp, "underlying_features", leaky)
    pipe = imp._pipeline_factory(_signal_fn, lambda s: _contract(s.direction), PARAMS, "09:15", "09:30")
    cuts = [pd.Timestamp(datetime.combine(d, time(12, 0)), tz=IST) for d in days]
    res = imp.mutation_test(pipe, spot, _premium_for(spot), [imp.Variant("ATR", "ENTRY", max_break_atr=0.5)], cuts)
    assert not res["ATR"]["passed"]


def test_premium_atr_ignores_candles_after_signal_and_entry_follows_signal():
    days = _days(2)
    spot = _spot(days)
    feats = imp.underlying_features(spot)
    sigs = [s for s in _signal_fn(spot) if s.timestamp.date() == days[1]]
    prem = _premium(spot, "CE")
    wild = prem.copy()
    cut = pd.Timestamp(sigs[0].timestamp)
    after = wild["timestamp"] > cut
    wild.loc[after, "high"] += 400
    wild.loc[after, "low"] = 0.05
    resolve = lambda s: _contract("BUY")
    a = imp.build_contexts(sigs[:1], feats, resolve, lambda c: prem, PARAMS)[0]
    b = imp.build_contexts(sigs[:1], feats, resolve, lambda c: wild, PARAMS)[0]
    assert a["premium_atr"] is not None and a["premium_atr"] == b["premium_atr"]
    assert a["entry_time"] == a["signal_time"] + pd.Timedelta(minutes=5)


def test_orb_levels_hidden_until_range_complete():
    f = imp.underlying_features(_spot(_days(1)))
    assert f.loc[f["timestamp"].dt.time < time(9, 30), "orb_high"].isna().all()
    assert f.loc[f["timestamp"].dt.time >= time(9, 30), "orb_high"].notna().all()


# ------------------------------------------------------- repeated filtering
def test_first_signal_only_blocks_until_opposite_break():
    feats = _feats([("10:00", 111), ("10:05", 112), ("10:10", 105), ("10:15", 111), ("10:20", 99), ("10:25", 112)])
    ctxs = [_ctx("10:00"), _ctx("10:05"), _ctx("10:15"), _ctx("10:20", "SELL"), _ctx("10:25")]
    v = imp.Variant("F", "REPEAT", repeat="FIRST_SIGNAL_ONLY")
    assert imp.repeat_filter(ctxs, feats, v) == [None, "FIRST_SIGNAL_ONLY", "FIRST_SIGNAL_ONLY", None, None]


def test_breakout_rearm_requires_close_back_inside_range():
    feats = _feats([("10:00", 111), ("10:05", 112), ("10:10", 105), ("10:15", 111)])
    ctxs = [_ctx("10:00"), _ctx("10:05"), _ctx("10:15")]
    v = imp.Variant("R", "REPEAT", repeat="BREAKOUT_REARM")
    assert imp.repeat_filter(ctxs, feats, v) == [None, "BREAKOUT_NOT_REARMED", None]


def test_repeat_filters_reset_each_day():
    feats = pd.concat([_feats([("10:00", 111)]), _feats([("10:00", 111)]).assign(
        timestamp=[pd.Timestamp("2026-09-02 10:00", tz=IST)])], ignore_index=True)
    ctxs = [_ctx("10:00"), _ctx("10:00", day="2026-09-02")]
    for v in (imp.Variant("F", "REPEAT", repeat="FIRST_SIGNAL_ONLY"), imp.Variant("C", "REPEAT", repeat="COOLDOWN",
                                                                                 cooldown_min=30)):
        assert imp.repeat_filter(ctxs, feats, v) == [None, None]


def test_baseline_keeps_every_signal():
    ctxs = [_ctx("10:00"), _ctx("10:05"), _ctx("10:10")]
    assert imp.repeat_filter(ctxs, _feats([("10:00", 111)]), imp.BASELINE) == [None, None, None]


# ------------------------------------------------------------------ cooldown
@pytest.mark.parametrize("minutes,expected", [
    (10, [None, "COOLDOWN_10M", None, "COOLDOWN_10M", None]),
    (15, [None, "COOLDOWN_15M", "COOLDOWN_15M", None, "COOLDOWN_15M"]),
    (20, [None, "COOLDOWN_20M", "COOLDOWN_20M", "COOLDOWN_20M", None]),
    (30, [None, "COOLDOWN_30M", "COOLDOWN_30M", "COOLDOWN_30M", "COOLDOWN_30M"]),
])
def test_cooldown_windows(minutes, expected):
    ctxs = [_ctx(f"10:{m:02d}") for m in (0, 5, 10, 15, 20)]
    v = imp.Variant("C", "REPEAT", repeat="COOLDOWN", cooldown_min=minutes)
    assert imp.repeat_filter(ctxs, _feats([("10:00", 111)]), v) == expected


def test_cooldown_is_per_direction():
    ctxs = [_ctx("10:00"), _ctx("10:05", "SELL"), _ctx("10:10")]
    v = imp.Variant("C", "REPEAT", repeat="COOLDOWN", cooldown_min=15)
    assert imp.repeat_filter(ctxs, _feats([("10:00", 111)]), v) == [None, None, "COOLDOWN_15M"]


# --------------------------------------------------- breakout distance filters
@pytest.mark.parametrize("limit,pct,expected", [(0.10, 0.09, None), (0.10, 0.10, None), (0.10, 0.11, "BREAK_PCT>0.10"),
                                                (0.30, 0.31, "BREAK_PCT>0.30"), (0.20, None, "BREAK_DISTANCE_UNKNOWN")])
def test_breakout_pct_filter(limit, pct, expected):
    assert imp.entry_filter(_ctx("10:00", pct=pct), imp.Variant("E", "ENTRY", max_break_pct=limit)) == expected


@pytest.mark.parametrize("limit,a,expected", [(0.25, 0.2, None), (0.50, 0.51, "BREAK_ATR>0.50"),
                                              (1.00, 1.0, None), (0.75, None, "BREAK_ATR_UNKNOWN")])
def test_breakout_atr_filter(limit, a, expected):
    assert imp.entry_filter(_ctx("10:00", a=a), imp.Variant("E", "ENTRY", max_break_atr=limit)) == expected


def test_breakout_distance_uses_signal_bar_close_and_orb_level():
    days = _days(2)
    spot = _spot(days)
    feats = imp.underlying_features(spot)
    sig = next(s for s in _signal_fn(spot) if s.timestamp.date() == days[1])
    ctx = imp.build_contexts([sig], feats, lambda s: _contract(s.direction), _premium_for(spot), PARAMS)[0]
    level = sig.conditions.get("above", sig.conditions.get("below"))
    sign = 1 if sig.direction == "BUY" else -1
    assert ctx["break_pct"] == pytest.approx(sign * (sig.spot - level) / sig.spot * 100)
    bar = feats.set_index("timestamp").loc[pd.Timestamp(sig.timestamp)]
    assert ctx["break_atr"] == pytest.approx(sign * (sig.spot - level) / bar["atr14"])
    assert ctx["break_pct"] > 0


# --------------------------------------------------------- stop/target variants
@pytest.mark.parametrize("k", [1.25, 1.5, 1.75, 2.0])
def test_atr_stop_variants(k):
    ctx = {"entry": 100.0, "premium_atr": 8.0, "tick": 0.05}
    stop, _ = imp.exit_levels(ctx, k, 2.0)
    assert stop == pytest.approx(round((100 - k * 8) / 0.05) * 0.05, abs=0.051)


@pytest.mark.parametrize("r", [1.0, 1.5, 2.0])
def test_target_variants(r):
    ctx = {"entry": 100.0, "premium_atr": 8.0, "tick": 0.05}
    stop, target = imp.exit_levels(ctx, 1.5, r)
    assert target == pytest.approx(100 + r * (100 - stop), abs=0.051)


def test_grid_is_predeclared_and_exact():
    v = imp.predeclared_variants()
    names = [x.name for x in v]
    assert len(v) == 27 and len(set(names)) == 27 and names[0] == "BASELINE_S2"
    assert {x.cooldown_min for x in v if x.repeat == "COOLDOWN"} == {10, 15, 20, 30}
    assert sorted(x.max_break_pct for x in v if x.max_break_pct) == [0.10, 0.15, 0.20, 0.25, 0.30]
    assert sorted(x.max_break_atr for x in v if x.max_break_atr) == [0.25, 0.50, 0.75, 1.00]
    exits = {(x.stop_atr, x.target_r) for x in v if x.group == "EXIT"} | {(1.5, 2.0)}
    assert exits == {(s, t) for s in (1.25, 1.5, 1.75, 2.0) for t in (1.0, 1.5, 2.0)}
    assert all(x.stop_atr == 1.5 and x.target_r == 2.0 for x in v if x.group in ("BASELINE", "REPEAT", "ENTRY"))


# ----------------------------------------------------------- chronological split
def test_split_dates_chronological_and_disjoint():
    sp = imp.split_dates()
    assert sp["TRAIN"][1] < sp["VALIDATION"][0] and sp["VALIDATION"][1] < sp["OOS"][0]
    assert imp.split_of("2026-09-17", sp) == "TRAIN" and imp.split_of("2026-09-18", sp) == "VALIDATION"
    assert imp.split_of("2026-10-01", sp) == "OOS" and imp.split_of("2026-10-05", sp) is None


def test_overlapping_splits_rejected():
    with pytest.raises(ValueError):
        imp.split_dates({"TRAIN": ("2026-09-01", "2026-09-20"), "VALIDATION": ("2026-09-18", "2026-09-24"),
                         "OOS": ("2026-09-25", "2026-10-01")})


def test_current_day_is_separate_from_historical_splits():
    rep = _run_cached()
    for r in rep["results"]:
        assert r["metrics"]["CURRENT_DAY"]["s2_signals"] > 0
        hist_n = sum(r["metrics"][s]["s2_signals"] for s in imp.SPLITS)
        assert hist_n == r["metrics"]["HISTORICAL_ALL"]["s2_signals"]
    assert rep["labels"] == {"historical": imp.NON_COMPARABLE, "current_day": imp.COMPARABLE}


# --------------------------------------------------------- parameter isolation
def test_combined_candidates_use_train_only_and_respect_cap():
    single = imp.predeclared_variants()
    base = {"evaluated": 50, "win_rate": 0.30, "profit_factor": 0.8}
    train = {v.name: dict(base) for v in single}
    for n in ("REPEAT_COOLDOWN_15M", "REPEAT_BREAKOUT_REARM", "ENTRY_MAX_BREAK_0.20PCT", "EXIT_STOP_1.50ATR_T1R",
              "EXIT_STOP_2.00ATR_T1R", "EXIT_STOP_1.75ATR_T1.5R", "EXIT_STOP_1.25ATR_T1R"):
        train[n] = {"evaluated": 40, "win_rate": 0.45, "profit_factor": 1.2}
    train["ENTRY_MAX_BREAK_0.10PCT"] = {"evaluated": 5, "win_rate": 0.9, "profit_factor": 5}
    combos, chosen = imp.combined_candidates(single, train)
    assert len(chosen["EXIT"]) == 3 and chosen["ENTRY"] == ["ENTRY_MAX_BREAK_0.20PCT"]
    assert 0 < len(combos) <= 15
    assert all(sum(a != b for a, b in zip(c.dims, imp.BASELINE.dims)) >= 2 for c in combos)
    assert len({c.dims for c in combos}) == len(combos)


def test_no_combinations_without_train_evidence():
    single = imp.predeclared_variants()
    train = {v.name: {"evaluated": 50, "win_rate": 0.30, "profit_factor": 0.8} for v in single}
    assert imp.combined_candidates(single, train)[0] == []


def _m(n, wr, pf=1.5):
    return {"evaluated": n, "win_rate": wr, "profit_factor": pf}


@pytest.mark.parametrize("tr,va,oo,cur,ok,expected", [
    (_m(30, .60), _m(10, .58), _m(10, .56), None, True, "ROBUST"),
    (_m(30, .60), _m(10, .58), _m(10, .56), None, False, "LOOKAHEAD_FAIL"),
    (_m(30, .70), _m(10, .40), _m(10, .56), None, True, "OVERFIT"),
    (_m(30, .45), _m(10, .45), _m(10, .52), None, True, "PROMISING"),
    (_m(30, .60), _m(4, .60), _m(10, .60), None, True, "PROMISING"),
    (_m(30, .30), _m(10, .30), _m(10, .30, .7), _m(10, .50), True, "CURRENT_DAY_ONLY"),
    (_m(30, .30), _m(10, .30), _m(10, .30, .7), _m(10, .22), True, "NOT_ROBUST"),
])
def test_classification(tr, va, oo, cur, ok, expected):
    m = {"TRAIN": tr, "VALIDATION": va, "OOS": oo}
    assert imp.classify(m, cur, 0.20, ok)[0] == expected


# ------------------------------------------------------------------ metrics
def _row(outcome, pnl, r, direction="BUY", t="10:00", stop=90.0):
    return {"decision": imp.ACCEPTED, "outcome": outcome, "pnl": pnl, "r_multiple": r, "direction": direction,
            "signal_time": f"2026-09-01T{t}:00+05:30", "session_date": "2026-09-01", "time_bucket": "10:00-11:00",
            "entry": 100.0, "stop_loss": stop, "skip_reason": None}


def test_metrics_count_ambiguous_as_loss_and_report_tracker_rate():
    rows = [_row("WIN", 1300.0, 2.0), _row("LOSS", -650.0, -1.0, "SELL"), _row("AMBIGUOUS", None, None),
            _row(imp.UNAVAILABLE, None, None), {**_row(None, None, None), "decision": "SKIPPED",
                                                "skip_reason": "COOLDOWN_10M"}]
    m = imp.compute_metrics(rows)
    assert (m["s2_signals"], m["accepted_signals"], m["skipped_signals"]) == (5, 4, 1)
    assert (m["evaluated"], m["wins"], m["losses"], m["ambiguous"], m["pending_or_unavailable"]) == (3, 1, 1, 1, 1)
    assert m["win_rate"] == pytest.approx(1 / 3) and m["tracker_win_rate"] == pytest.approx(0.5)
    assert m["net_pnl"] == pytest.approx(1300 - 650 - 650)
    assert m["profit_factor"] == pytest.approx(1300 / 1300)
    assert m["BUY"]["evaluated"] == 2 and m["SELL"]["wins"] == 0
    assert m["skip_reasons"] == {"COOLDOWN_10M": 1}


def test_repeated_signal_ordinals_and_gaps():
    rows = [_row("WIN", 1, 1, "SELL", "10:00"), _row("LOSS", -1, -1, "SELL", "10:05"),
            _row("LOSS", -1, -1, "SELL", "10:20"), _row("WIN", 1, 1, "BUY", "10:30")]
    seq = imp.annotate_sequence(rows)
    assert [s["ordinal"] for s in seq] == ["FIRST", "SECOND", "THIRD_PLUS", "FIRST"]
    assert [s["previous_relation"] for s in seq] == [None, "SAME_DIRECTION", "SAME_DIRECTION", "OPPOSITE_DIRECTION"]
    assert [s["minutes_since_same_direction"] for s in seq] == [None, 5.0, 15.0, None]
    ra = imp.repeated_signal_analysis(rows)
    assert ra["by_ordinal"]["FIRST"]["evaluated"] == 2 and ra["by_ordinal"]["FIRST"]["wins"] == 2
    assert ra["within_minutes_of_previous_same_direction"]["<= 5 min"]["evaluated"] == 1
    assert ra["within_minutes_of_previous_same_direction"]["<= 15 min"]["evaluated"] == 2


# ------------------------------------------------- determinism, baseline, paper
def test_results_are_deterministic():
    a, b = _run_cached(), _run()
    strip = lambda r: json.dumps(imp._jsonable({k: v for k, v in r.items() if k != "generated_at"}), sort_keys=True)
    assert strip(a) == strip(b)


def test_baseline_variant_is_unchanged_s2():
    days = _days(3)
    spot = _spot(days)
    sigs = generate_signals("S2", spot, "NIFTY", CFG)
    feats = imp.underlying_features(spot)
    ctxs = imp.build_contexts(sigs, feats, lambda s: _contract(s.direction), _premium_for(spot), PARAMS)
    rows = imp.run_variant(ctxs, feats, imp.BASELINE, square_off="15:15")
    assert [(r["signal_time"], r["direction"]) for r in rows] == \
        [(pd.Timestamp(s.timestamp).isoformat(), s.direction) for s in sigs]
    assert all(r["decision"] == imp.ACCEPTED for r in rows)
    assert generate_signals("S2", spot, "NIFTY", CFG) == sigs
    assert CFG.strategies["S2"] == {"enabled": True, "range_start": "09:15", "range_end": "09:30", "atr_period": 14}


def test_baseline_uses_configured_s2_exit():
    assert (imp.BASELINE.stop_atr, imp.BASELINE.target_r, imp.BASELINE.repeat) == (1.5, 2.0, "NONE")
    rep = _run_cached(with_current=False)
    assert rep["results"][0]["name"] == "BASELINE_S2" and rep["results"][0]["status"] == "BASELINE"


def test_runner_refuses_non_paper_mode(monkeypatch):
    import scripts.run_s2_1dte_improvement as runner
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(LiveTradingSafetyError):
        runner.run(day=date(2026, 10, 5), hist_from=date(2026, 9, 1), hist_to=date(2026, 10, 1))
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    with pytest.raises(LiveTradingSafetyError):
        runner.run(day=date(2026, 10, 5), hist_from=date(2026, 9, 1), hist_to=date(2026, 10, 1))


def test_reports_written(tmp_path):
    rep = _run_cached()
    paths = imp.write_reports(rep, tmp_path)
    assert {p.name for p in paths.values()} == {"s2_1dte_improvement_baseline.json",
                                               "s2_1dte_improvement_candidates.csv", "s2_1dte_improvement_report.md"}
    md = paths["md"].read_text(encoding="utf-8")
    for heading in ("1. Executive summary", "2. Baseline", "3. Repeated signal analysis", "4. Entry filter",
                    "5. Exit analysis", "6. Combined candidates", "7. Train / validation / OOS",
                    "8. Current 1-DTE result", "9. Look-ahead test", "10. Best candidate",
                    "11. Improvement over baseline", "12. Risks and limitations", "13. Final recommendation"):
        assert heading in md
    assert imp.NON_COMPARABLE in md and imp.COMPARABLE in md
    csv = pd.read_csv(paths["csv"])
    assert len(csv) == len(rep["results"]) and "oos_buy_accuracy" in csv.columns
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["summary"]["final_block"].startswith("BASELINE:")
