from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest.current_day_s2 import LevelParams
from src.backtest.research_strategies import generate_signals
from src.options.models import OptionResolutionError
from src.safety.live_gate import LiveTradingSafetyError

IST = imp.IST
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=14, stop_atr_multiplier=1.5,
                     target_r=2.0, slippage_bps=0.0)
CFG = SimpleNamespace(strategies={"S2": {"enabled": True, "range_start": "09:15", "range_end": "09:30",
                                         "atr_period": 14}})
CANDS = {c.name: c for c in dr.predeclared_candidates()}


def _days(n, start=date(2026, 9, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _spot(days, seed=5):
    """Alternating up/down sessions so every split contains both BUY and SELL signals."""
    rng = np.random.default_rng(seed)
    rows, price = [], 24000.0
    for k, d in enumerate(days):
        drift = 5.0 if k % 2 == 0 else -5.0
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


def _resolve(s):
    return _contract(s.direction)


def _splits(days):
    return {"TRAIN": (days[0].isoformat(), days[3].isoformat()),
            "VALIDATION": (days[4].isoformat(), days[5].isoformat()),
            "OOS": (days[6].isoformat(), days[7].isoformat())}


def _run(premium_for=None, resolve=_resolve):
    days = _days(9)
    spot = _spot(days)
    hist = spot[spot["timestamp"].dt.date <= days[7]].reset_index(drop=True)
    pf = premium_for or _premium_for(spot)
    return dr.run_direction_research(hist_spot=hist, signal_fn=_signal_fn, resolve=resolve, hist_premium_for=pf,
                                     params=PARAMS, current_spot=spot, current_premium_for=pf, current_day=days[8],
                                     splits=_splits(days), current_cut_every_min=60)


_CACHE: dict = {}


def _run_cached():
    if "rep" not in _CACHE:
        _CACHE["rep"] = _run()
    return _CACHE["rep"]


def _contexts(days=None, premium_for=None, resolve=_resolve):
    days = days or _days(3)
    spot = _spot(days)
    feats = imp.underlying_features(spot)
    ctxs = imp.build_contexts(_signal_fn(spot), feats, resolve, premium_for or _premium_for(spot), PARAMS)
    return spot, feats, ctxs


def _res(rep):
    return {r["name"]: r for r in rep["results"]}


# ----------------------------------------------------------------- filtering
def test_candidates_are_predeclared():
    assert list(CANDS) == ["BASELINE_S2_BUY_SELL", "SELL_ONLY", "BUY_ONLY", "BUY_CONFIRMED", "SELL_CONFIRMED"]
    assert CANDS["BUY_CONFIRMED"].directions == ("BUY",) and CANDS["BUY_CONFIRMED"].confirm
    assert CANDS["SELL_CONFIRMED"].directions == ("SELL",) and CANDS["SELL_CONFIRMED"].confirm


@pytest.mark.parametrize("name,keep,drop", [("BUY_ONLY", "BUY", "SELL"), ("SELL_ONLY", "SELL", "BUY")])
def test_single_direction_filtering(name, keep, drop):
    _, feats, ctxs = _contexts()
    assert {c["direction"] for c in ctxs} == {"BUY", "SELL"}
    base = imp.run_variant(ctxs, feats, imp.BASELINE, square_off="15:15")
    rows = dr.run_candidate(ctxs, feats, CANDS[name], square_off="15:15")
    assert len(rows) == len(base)
    for b, r in zip(base, rows):
        if r["direction"] == keep:
            assert r["decision"] == imp.ACCEPTED and r["outcome"] == b["outcome"] and r["pnl"] == b["pnl"]
        else:
            assert r["decision"] == "SKIPPED" and r["skip_reason"] == dr.DIRECTION_SKIP and r["outcome"] is None
    m = dr.candidate_metrics(rows)
    assert m[drop]["signals"] == 0 and m[keep]["signals"] == m["accepted_signals"] > 0


def test_buy_only_and_sell_only_partition_baseline():
    rep = _run_cached()
    res = _res(rep)
    for s in ("TRAIN", "VALIDATION", "OOS", "CURRENT_DAY"):
        b, buy, sell = (res[n]["metrics"][s] for n in ("BASELINE_S2_BUY_SELL", "BUY_ONLY", "SELL_ONLY"))
        assert buy["evaluated"] + sell["evaluated"] == b["evaluated"]
        assert buy["wins"] + sell["wins"] == b["wins"]
        assert buy["net_pnl"] + sell["net_pnl"] == pytest.approx(b["net_pnl"])
        assert buy["evaluated"] == b["BUY"]["evaluated"] and sell["evaluated"] == b["SELL"]["evaluated"]


def test_confirmed_candidates_are_subsets_of_direction_only():
    _, feats, ctxs = _contexts()
    for d in ("BUY", "SELL"):
        plain = dr.run_candidate(ctxs, feats, CANDS[f"{d}_ONLY"], square_off="15:15")
        conf = dr.run_candidate(ctxs, feats, CANDS[f"{d}_CONFIRMED"], square_off="15:15")
        acc_plain = {r["signal_time"] for r in plain if r["decision"] == imp.ACCEPTED}
        acc_conf = {r["signal_time"] for r in conf if r["decision"] == imp.ACCEPTED}
        assert acc_conf <= acc_plain and acc_conf != acc_plain
        assert all(r["direction"] == d for r in conf if r["decision"] == imp.ACCEPTED)


def _conf_frame(prev_close, close, open_, ema_level, n=30):
    ts = [pd.Timestamp("2026-09-01 09:15", tz=IST) + pd.Timedelta(minutes=5 * i) for i in range(n)]
    f = pd.DataFrame({"timestamp": ts, "open": 105.0, "close": 105.0, "high": 106.0, "low": 104.0})
    f["day"] = f["timestamp"].dt.date
    f["orb_high"], f["orb_low"] = 110.0, 100.0
    f.loc[n - 2, "close"] = prev_close
    f.loc[n - 1, ["open", "close"]] = open_, close
    conf = dr.confirmation_features(f)
    conf["ema20"] = ema_level
    return conf, ts[-1]


@pytest.mark.parametrize("prev_close,close,open_,ema,ok,failed", [
    (111.0, 113.0, 111.5, 108.0, True, []),
    (109.0, 113.0, 111.5, 108.0, False, ["two_close"]),
    (111.0, 113.0, 113.5, 108.0, False, ["body"]),
    (111.0, 113.0, 111.5, 114.0, False, ["ema"]),
])
def test_buy_confirmation_rules(prev_close, close, open_, ema, ok, failed):
    conf, ts = _conf_frame(prev_close, close, open_, ema)
    passed, detail = dr.confirmation({"signal_time": ts, "direction": "BUY"}, conf)
    assert passed is ok and [k for k, v in detail.items() if not v] == failed


def test_sell_confirmation_rules():
    conf, ts = _conf_frame(99.0, 97.0, 98.5, 101.0)
    assert dr.confirmation({"signal_time": ts, "direction": "SELL"}, conf)[0] is True
    conf, ts = _conf_frame(101.0, 97.0, 98.5, 101.0)
    assert dr.confirmation({"signal_time": ts, "direction": "SELL"}, conf) == \
        (False, {"two_close": False, "body": True, "ema": True})


def test_confirmation_needs_history_and_reports_unavailable():
    conf, ts = _conf_frame(111.0, 113.0, 111.5, 108.0, n=10)
    assert dr.confirmation({"signal_time": ts, "direction": "BUY"}, conf) == (None, {"reason": "INSUFFICIENT_HISTORY"})
    _, feats, ctxs = _contexts(days=_days(1))
    skips = dr.direction_filter(ctxs, feats, CANDS["BUY_CONFIRMED"])
    first_buy = next(i for i, c in enumerate(ctxs) if c["direction"] == "BUY")
    assert skips[first_buy].startswith(dr.CONFIRM_UNAVAILABLE)


# ---------------------------------------------------------------- look-ahead
def test_no_lookahead_mutation_passes_for_all_candidates():
    days = _days(4)
    spot = _spot(days)
    pipe = imp._pipeline_factory(_signal_fn, _resolve, PARAMS, "09:15", "09:30")
    cuts = [pd.Timestamp(datetime.combine(d, time(h, m)), tz=IST) for d in days for h, m in ((10, 30), (13, 0))]
    res = dr.mutation_test(pipe, spot, _premium_for(spot), dr.predeclared_candidates(), cuts)
    assert all(r["passed"] and r["cuts"] == len(cuts) for r in res.values())


def test_mutation_test_detects_future_leak_in_confirmation(monkeypatch):
    days = _days(3)
    spot = _spot(days)
    real = dr.confirmation_features

    def leaky(features):
        f = real(features)
        f["ema20"] = f["close"].shift(-3).fillna(f["close"])
        return f

    monkeypatch.setattr(dr, "confirmation_features", leaky)
    pipe = imp._pipeline_factory(_signal_fn, _resolve, PARAMS, "09:15", "09:30")
    cuts = [pd.Timestamp(datetime.combine(d, time(h, m)), tz=IST) for d in days for h, m in ((10, 30), (12, 0))]
    res = dr.mutation_test(pipe, spot, _premium_for(spot), [CANDS["BUY_CONFIRMED"], CANDS["SELL_CONFIRMED"]], cuts)
    assert not all(r["passed"] for r in res.values())


def test_confirmation_ignores_future_candles():
    _, feats, ctxs = _contexts()
    cut = ctxs[len(ctxs) // 2]["signal_time"]
    wild = feats.copy()
    after = wild["timestamp"] > cut
    wild.loc[after, ["open", "close"]] = wild.loc[after, ["close", "open"]].to_numpy() * 1.1
    for name in ("BUY_CONFIRMED", "SELL_CONFIRMED"):
        a = dr.direction_filter(ctxs, feats, CANDS[name])
        b = dr.direction_filter(ctxs, wild, CANDS[name])
        early = [i for i, c in enumerate(ctxs) if c["signal_time"] <= cut]
        assert [a[i] for i in early] == [b[i] for i in early]


def test_full_run_lookahead_passes():
    rep = _run_cached()
    assert rep["lookahead"]["passed"] == rep["lookahead"]["candidates"] == 5 and not rep["lookahead"]["failed"]


# ----------------------------------------------------------- baseline S2
def test_baseline_is_unchanged_s2():
    spot, feats, ctxs = _contexts()
    sigs = _signal_fn(spot)
    rows = dr.run_candidate(ctxs, feats, dr.BASELINE, square_off="15:15")
    ref = imp.run_variant(ctxs, feats, imp.BASELINE, square_off="15:15")
    assert [(r["signal_time"], r["direction"]) for r in rows] == \
        [(pd.Timestamp(s.timestamp).isoformat(), s.direction) for s in sigs]
    assert all(r["decision"] == imp.ACCEPTED for r in rows)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "variant"} for r in rs]
    assert strip(rows) == strip(ref)
    assert dr.candidate_metrics(rows)["win_rate"] == imp.compute_metrics(ref)["win_rate"]
    assert _signal_fn(spot) == sigs
    assert CFG.strategies["S2"] == {"enabled": True, "range_start": "09:15", "range_end": "09:30", "atr_period": 14}
    assert (imp.BASELINE.stop_atr, imp.BASELINE.target_r, imp.BASELINE.repeat) == (1.5, 2.0, "NONE")


def test_baseline_status_and_metrics_match_improvement_module():
    days = _days(9)
    spot = _spot(days)
    hist = spot[spot["timestamp"].dt.date <= days[7]].reset_index(drop=True)
    pf = _premium_for(spot)
    ref = imp.run_research(hist_spot=hist, signal_fn=_signal_fn, resolve=_resolve, hist_premium_for=pf, params=PARAMS,
                           current_spot=spot, current_premium_for=pf, current_day=days[8], splits=_splits(days),
                           current_cut_every_min=60)
    base = _res(_run_cached())["BASELINE_S2_BUY_SELL"]
    assert base["status"] == "BASELINE"
    for s in ("TRAIN", "VALIDATION", "OOS", "CURRENT_DAY"):
        a = {k: v for k, v in base["metrics"][s].items() if k != "unavailable_reasons"}
        assert json.dumps(imp._jsonable(a), sort_keys=True) == \
            json.dumps(imp._jsonable(ref["results"][0]["metrics"][s]), sort_keys=True)


# ------------------------------------------------------------- missing data
def test_missing_premium_is_unavailable_not_synthesised():
    spot = _spot(_days(9))
    real = _premium_for(spot)
    pf = lambda c: "NO_PREMIUM_DATA" if c.option_type.value == "CE" else real(c)
    res = _res(_run(premium_for=pf))
    for s in ("TRAIN", "VALIDATION", "OOS", "CURRENT_DAY"):
        buy = res["BUY_ONLY"]["metrics"][s]
        assert buy["evaluated"] == 0 and buy["wins"] == buy["losses"] == 0 and buy["net_pnl"] == 0
        assert buy["win_rate"] is None and buy["profit_factor"] is None
        assert buy["pending_or_unavailable"] == buy["accepted_signals"] > 0
        assert buy["unavailable_reasons"] == {"NO_PREMIUM_DATA": buy["accepted_signals"]}
        sell, ref = res["SELL_ONLY"]["metrics"][s], _res(_run_cached())["SELL_ONLY"]["metrics"][s]
        assert sell["evaluated"] == ref["evaluated"] and sell["net_pnl"] == ref["net_pnl"]


def test_unresolved_contract_is_unavailable():
    def resolve(s):
        if s.direction == "SELL":
            raise OptionResolutionError("NO_STRIKE", "no strike")
        return _contract(s.direction)

    _, feats, ctxs = _contexts(resolve=resolve)
    rows = dr.run_candidate(ctxs, feats, CANDS["SELL_ONLY"], square_off="15:15")
    m = dr.candidate_metrics(rows)
    assert m["evaluated"] == 0 and m["accepted_signals"] > 0
    assert m["unavailable_reasons"] == {"CONTRACT_UNRESOLVED:NO_STRIKE": m["accepted_signals"]}


# ------------------------------------------------- determinism, paper, reports
def test_results_are_deterministic():
    strip = lambda r: json.dumps(imp._jsonable({k: v for k, v in r.items() if k != "generated_at"}), sort_keys=True)
    assert strip(_run_cached()) == strip(_run())


def test_runner_refuses_non_paper_mode(monkeypatch):
    import scripts.run_s2_direction_research as runner
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(LiveTradingSafetyError):
        runner.run(day=date(2026, 10, 5), hist_from=date(2026, 9, 1), hist_to=date(2026, 10, 1))
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    with pytest.raises(LiveTradingSafetyError):
        runner.run(day=date(2026, 10, 5), hist_from=date(2026, 9, 1), hist_to=date(2026, 10, 1))


def test_reports_written(tmp_path):
    rep = _run_cached()
    paths = dr.write_reports(rep, tmp_path)
    assert {p.name for p in paths.values()} == {"s2_direction_research.json", "s2_direction_candidates.csv",
                                               "s2_direction_summary.md"}
    csv = pd.read_csv(paths["csv"])
    assert list(csv["candidate"]) == list(CANDS)
    for col in ("train_win_rate", "validation_win_rate", "oos_win_rate", "current_day_win_rate", "oos_profit_factor",
                "historical_all_net_pnl", "historical_all_max_drawdown", "oos_average_r", "train_buy_wins",
                "oos_sell_evaluated", "current_day_unavailable_reasons"):
        assert col in csv.columns
    md = paths["md"].read_text(encoding="utf-8")
    for name in CANDS:
        assert f"## {name}" in md
    assert imp.NON_COMPARABLE in md and imp.COMPARABLE in md and "Look-ahead test" in md
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert [r["name"] for r in data["results"]] == list(CANDS) and data["summary"]["recommendation"]
