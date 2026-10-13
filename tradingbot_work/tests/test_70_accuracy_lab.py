from __future__ import annotations

import csv
import inspect
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest import accuracy_70_lab as acc
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.backtest.current_day_s2 import LevelParams
from src.backtest.live_vs_historical import time_bucket
from src.backtest.research_strategies import ResearchSignal
from src.indicators.technical import atr
from src.safety import live_gate

IST = imp.IST
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


@pytest.fixture(scope="module")
def built():
    spot = _spot()
    a = acc.build_lab70(spot=spot, split_days=SPLIT_DAYS, splits=SPLITS, resolve=_resolve,
                        premium_for=_premium_for(spot), params=PARAMS, square_off="15:15", today=TODAY,
                        run_mutation=False, lookahead_days=DAYS[-2:])
    return spot, a


def _status(a):
    return {x["id"]: x["status"] for x in a["integrity"]}


# ------------------------------------------------------------------ Phase 13: direction isolation / contamination
def _row(day, hhmm, direction, win, pnl=None):
    ts = _ts(day, hhmm)
    pnl = (650.0 if win else -325.0) if pnl is None else pnl
    return {"session_date": day, "signal_time": ts.isoformat(), "direction": direction, "decision": imp.ACCEPTED,
            "skip_reason": None, "state": sh.EVALUATED, "outcome": imp.WIN if win else imp.LOSS, "pnl": pnl,
            "r_multiple": 2.0 if win else -1.0, "time_bucket": time_bucket(ts), "dte": 5,
            "dte_bucket": sh.dte_bucket(5), "regimes": [], "contract": "NIFTY CE", "break_atr": 0.3,
            "premium_atr": 2.0, "reason": None, "reason_raw": None, "strategy": lab.BASELINE}


def _fact(direction, minute):
    up = direction == "BUY"
    return {"direction": direction, "minute": minute, "open": 100.0 if up else 101.0, "close": 101.0 if up else 99.0,
            "ema20": 100.5 if up else 99.5, "ema50": 100.0, "rsi14": 60.0 if up else 40.0, "vwap": None,
            "vol_ratio": None, "prev_open": 99.5 if up else 100.5, "prev_close": 100.2 if up else 100.0,
            "prev_orb_high": 100.1, "prev_orb_low": 100.2, "atr14": 10.0, "break_atr": 0.3, "atr_pct": 0.05,
            "vwap_dist_atr": None, "intraday_move": 0.002, "sign_changes": 3.0, "trending": False, "choppy": False,
            "pit_tags": [], "range_bound": False, "prev_day_labels": [], "has_prev_day": True,
            "premium_atr_pct": 5.0, "dte": 5, "dte_bucket": "5-7 DTE"}


def _legacy_dataset():
    rows, facts = [], {}
    for i, d in enumerate(DAYS[:12]):
        for hhmm, direction, win in [("10:00", "BUY", i % 2 == 0), ("11:00", "BUY", i % 3 == 0),
                                     ("10:30", "SELL", i % 3 != 0), ("13:45", "SELL", True)]:
            r = _row(d, hhmm, direction, win)
            rows.append(r)
            ts = _ts(d, hhmm)
            facts[lab.row_key(r)] = _fact(direction, ts.hour * 60 + ts.minute)
    return rows, facts


def test_legacy_duplicate_buy_metrics_are_explained_by_unavailable_volume_vwap():
    rows, facts = _legacy_dataset()
    au = acc.legacy_duplicate_audit(rows, facts, DAYS[:6])
    t = {x["candidate"]: x for x in au["table"]}
    for name in ("BUY_VOLUME_CONFIRM", "BUY_VWAP_CONFIRM", "BUY_VWAP_RSI", "BUY_EMA_VOLUME"):
        assert name in au["duplicates_of_sell_only"]
        assert t[name]["buy_evaluated"] == 0
        assert "CONFIRMATION_DATA_UNAVAILABLE" in t[name]["explanation"]
    assert t["BUY_EMA_CONFIRM"]["duplicates_sell_only"] is False and t["BUY_EMA_CONFIRM"]["buy_evaluated"] > 0
    assert "SELL_VOLUME_CONFIRM" in au["duplicates_of_buy_only"]
    assert len({x["sell_pnl"] for x in au["table"] if x["sell_checks"] == "-"}) == 1
    assert all(c["status"] == acc.PASS for c in au["checks"]), au["checks"]


def test_legacy_audit_detects_cross_direction_leak_and_shared_mutation(monkeypatch):
    rows, facts = _legacy_dataset()
    orig = lab.apply_candidate

    def leaky(base_rows, f, c, p):
        out = orig(base_rows, f, c, p)
        if c.name == "BUY_EMA_CONFIRM":
            out = [dict(r, decision="SKIPPED", state=sh.SKIPPED, outcome=None, pnl=None)
                   if r["direction"] == "SELL" else r for r in out]
            base_rows[0]["pnl"] = 1.0
        return out

    monkeypatch.setattr(lab, "apply_candidate", leaky)
    st = {c["id"]: c["status"] for c in acc.legacy_duplicate_audit(rows, facts, DAYS[:6])["checks"]}
    assert st["P13-2"] == acc.FAIL and st["P13-5"] == acc.FAIL


def test_every_candidate_holds_only_its_own_direction(built):
    _, a = built
    for n, rows in a["_rows"].items():
        d = a["scores"][n]["direction"]
        assert n.startswith(d + "_")
        assert all(r["direction"] == d for r in rows)
    st = _status(a)
    assert st["I1"] == st["I2"] == st["I3"] == st["I4"] == acc.PASS


def test_candidate_rows_refuse_foreign_direction_and_are_fresh_copies():
    with pytest.raises(ValueError):
        acc.candidate_rows(acc.Strategy("BUY_X", "F", "BUY", ("ema_stack",)),
                           [(_ts(DAYS[0], "10:00").isoformat(), "SELL")], {}, {})
    k = (_ts(DAYS[0], "10:00").isoformat(), "BUY")
    shared = {k: _row(DAYS[0], "10:00", "BUY", True)}
    a = acc.candidate_rows(acc.Strategy("BUY_A", "F", "BUY", ("ema_stack",)), [k], shared, {})
    b = acc.candidate_rows(acc.Strategy("BUY_B", "F", "BUY", ("ema_side",)), [k], shared, {})
    a[0]["pnl"] = -1.0
    assert b[0]["pnl"] == 650.0 and shared[k]["pnl"] == 650.0 and b[0]["strategy"] == "BUY_B"


def test_buy_metrics_do_not_change_when_sell_rows_change(built):
    _, a = built
    periods = {**SPLIT_DAYS, "ALL": DAYS}
    folds = lab.walk_forward_folds(DAYS, 8, 3, 3)
    buy = next(n for n in a["_rows"] if n.startswith("BUY_") and a["scores"][n]["ALL"]["evaluated"])
    before = acc.score_rows(a["_rows"][buy], periods, folds, {}, acc.Criteria())
    for n in a["_rows"]:
        if n.startswith("SELL_"):
            for r in a["_rows"][n]:
                r["pnl_shadow"] = "poisoned"
    after = acc.score_rows(a["_rows"][buy], periods, folds, {}, acc.Criteria())
    assert before["ALL"] == after["ALL"] and before["OOS"] == after["OOS"]
    assert a["scores"][buy]["ALL"]["evaluated"] == before["ALL"]["evaluated"]


def test_predeclared_candidates_unique_and_both_directions():
    s = acc.predeclared_strategies()
    ids = [x.cid for x in s]
    assert len(ids) == len(set(ids))
    assert sum(x.direction == "BUY" for x in s) == sum(x.direction == "SELL" for x in s) == len(s) // 2
    assert {x.family for x in s} == set(acc.FAMILIES)


# ------------------------------------------------------------------ lookahead
def test_feature_signal_and_option_lookahead_pass(built):
    _, a = built
    assert a["lookahead"]["features"]["passed"] and a["lookahead"]["option"]["passed"]
    assert a["lookahead"]["passed"]


def test_truncation_test_catches_a_leaky_condition(monkeypatch):
    spot = _spot(DAYS[:8])
    monkeypatch.setitem(acc.CONDITIONS, "ema_stack",
                        lambda f, up, p: (f["close"].shift(-1) > f["close"], pd.Series(True, index=f.index)))
    cuts = [f"{h:02d}:{m:02d}" for h in range(10, 14) for m in range(0, 60, 15)]
    res = acc.feature_truncation_test(spot, [acc.Strategy("BUY_LEAK", "X", "BUY", ("ema_stack",))], DAYS[6:8],
                                      cut_times=cuts)
    assert not res["passed"] and res["signal_failures"]


def test_pipeline_mutation_test_passes_for_family_signals():
    spot = _spot(DAYS[:8])
    bases = [s for s in acc.predeclared_strategies()
             if s.cid.startswith(("BUY_TREND_MOMENTUM", "SELL_ORB_FIRST_BREAK", "BUY_BREAKOUT_NBAR"))]
    res = acc.pipeline_mutation_test(spot, bases, acc.default_allowed, _resolve, _premium_for(spot), PARAMS, "NIFTY",
                                     DAYS[6:8])
    assert res["passed"] and res["cuts"] == 2


def test_option_features_use_only_premium_candles_up_to_signal():
    spot = _spot(DAYS[:8])
    prem = _premium(spot, "CE")
    sig = _ts(DAYS[7], "11:00")
    trunc = prem[prem["timestamp"] <= sig].reset_index(drop=True)
    ctx = {"data_status": "OK", "premium": prem, "signal_time": sig, "direction": "BUY", "dte": 3,
           "premium_atr": float(atr(trunc, 14).iloc[-1])}
    base = acc.option_features(ctx)
    assert base == acc.option_features(ctx, imp.mutate_after(prem, sig, 3))
    changed = prem.copy()
    changed.loc[changed["timestamp"] == sig, "close"] += 25.0
    assert acc.option_features(ctx, changed) != base
    assert acc.option_feature_mutation_test([ctx])["passed"]
    assert not acc.option_feature_mutation_test([dict(ctx, premium_atr=ctx["premium_atr"] * 1.3)])["passed"]


# ------------------------------------------------------------------ splits / completed candles / alignment
def test_chronological_splits_and_current_day_excluded(built):
    _, a = built
    st = _status(a)
    assert st["S1"] == st["S2"] == acc.PASS
    bad = acc.split_checks({"TRAIN": [DAYS[5]], "VALIDATION": [DAYS[2]], "OOS": [DAYS[9]]}, SPLITS, TODAY)
    assert bad[0]["status"] == acc.FAIL
    cur = acc.split_checks({"TRAIN": [DAYS[0]], "VALIDATION": [DAYS[1]], "OOS": [TODAY.isoformat()]}, SPLITS, TODAY)
    assert cur[1]["status"] == acc.FAIL


def test_finalist_selection_never_reads_oos(built):
    _, a = built
    fake = {n: dict(s, OOS=dict(s["OOS"], win_rate=1.0, profit_factor=99.0, net_pnl=1e9))
            for n, s in a["scores"].items()}
    for d in acc.DIRECTIONS:
        assert acc.pick(fake, [n for n in fake if fake[n]["direction"] == d]) == a["finalists"][d]


def test_completed_candle_and_session_checks():
    spot = _spot(DAYS[:3])
    assert all(c["status"] == acc.PASS for c in acc.data_checks(spot, DAYS[:3]))
    assert acc.data_checks(spot.iloc[:-10], DAYS[:3])[1]["status"] == acc.FAIL
    assert acc.data_checks(spot.drop(index=100).reset_index(drop=True), DAYS[:3])[2]["status"] == acc.FAIL


def test_entries_are_next_candle_and_option_candles_aligned(built):
    _, a = built
    st = _status(a)
    for cid in ("C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3"):
        assert st[cid] == acc.PASS, cid
    assert a["integrity_ok"]


def test_missing_option_candle_at_signal_is_data_unavailable():
    spot = _spot(DAYS[:8])
    sig_ts = _ts(DAYS[7], "11:00")
    holes = _premium(spot, "CE")
    holes = holes[holes["timestamp"] != sig_ts].reset_index(drop=True)
    feats = imp.underlying_features(spot)
    sig = ResearchSignal(sig_ts.to_pydatetime(), "BUY", "R70", "NIFTY", 24000.0, {})
    ctx = imp.build_contexts([sig], feats, _resolve, lambda c: holes, PARAMS)[0]
    assert ctx["data_status"] != "OK" and "NO_PREMIUM_CANDLE_AT_SIGNAL" in ctx["reason"]
    row = sh.classify_row(imp.run_variant([ctx], feats, imp.BASELINE, square_off="15:15")[0], {})
    assert row["state"] == sh.DATA_UNAVAILABLE


def test_signals_are_edge_triggered_inside_the_entry_window():
    spot = _spot(DAYS[:7])
    f = acc.family_bars(spot)
    m = acc.allowed_mask(f)
    s = acc.Strategy("BUY_W", "F", "BUY", ("window",), (("win_from", 1000.0), ("win_to", 1100.0)))
    keys, avail = acc.strategy_signals(f, s, m)
    assert len(keys) == 7 and all(k[0][11:16] == "10:00" for k in keys) and avail == 1.0
    late = acc.Strategy("BUY_L", "F", "BUY", ("window",), (("win_from", 1430.0), ("win_to", 1500.0)))
    assert acc.strategy_signals(f, late, m)[0] == []


# ------------------------------------------------------------------ data availability
def test_volume_and_vwap_candidates_are_data_unavailable_and_never_selected(built):
    _, a = built
    names = ("BUY_BREAKOUT_VOLUME[n12_vol_mult1.5]", "SELL_BREAKOUT_VWAP[n12]", "BUY_VWAP_REVERSION[k1]",
             "SELL_VOLUME_PERCENTILE_TREND[vol_pct0.7]")
    for n in names:
        s = a["scores"][n]
        assert s["status"] == acc.DATA_UNAVAILABLE and s["selection_score"] is None and s["final_score"] == 0.0
        assert f"{n}+PREM_MOMENTUM" not in a["scores"]
    assert not set(names) & set(a["finalists"].values())
    assert not set(names) & set(a["family_finalists"].values())


# ------------------------------------------------------------------ gates / tiers / scoring
def _m(n, wr, pf, pnl, dd=1000.0, best_share=0.3, streak=3):
    w = round(n * wr)
    return {"evaluated": n, "wins": w, "losses": n - w, "win_rate": wr, "profit_factor": pf, "net_pnl": pnl,
            "max_drawdown": dd, "best_day_share": best_share, "pnl_excl_best_day": pnl * 0.5,
            "pnl_excl_largest_win": pnl * 0.7, "max_consecutive_losses": streak}


def _s(oos, val=None, wf_share=0.8, day_share=0.7, best_all=0.2, regime=0.8, ncond=2, la=True, data="OK"):
    c = acc.Criteria()
    s = {"TRAIN": _m(60, 0.7, 2.0, 5000.0), "VALIDATION": val or _m(30, 0.72, 2.2, 4000.0), "OOS": oos,
         "ALL": _m(150, 0.7, 2.0, 15000.0), "walk_forward": {"wf_profitable_share": wf_share, "wf_pf": 1.6},
         "day_all": {"profitable_day_share": day_share, "best_day_share_of_net": best_all},
         "regime_share": regime, "n_conditions": ncond, "lookahead_passed": la, "data_status": data}
    s["tier"] = acc.sample_tier(oos["evaluated"], c)
    s["gates"] = acc.gates(s, c)
    s["status"], s["failed"] = acc.candidate_status(s, c)
    s["final_score"] = acc.final_score(s, c)
    return s


def test_sample_tiers_and_minimum_sample():
    c = acc.Criteria()
    assert [acc.sample_tier(n, c) for n in (60, 30, 15, 14)] == ["A", "B", "C", "D"]
    assert _s(_m(10, 0.8, 3.0, 5000.0))["status"] == acc.INSUFFICIENT_SAMPLE


def test_candidate_scoring_formula_and_status():
    strong = _s(_m(40, 0.75, 2.4, 9000.0, dd=2000.0))
    assert strong["status"] == acc.ALL_GATES_PASS
    assert strong["final_score"] == pytest.approx(92.25)
    below = _s(_m(40, 0.6, 2.0, 8000.0, dd=2000.0))
    assert below["status"] == acc.QUALITY_PASS_BELOW_70 and below["failed"] == ["oos_accuracy_70"]
    small = _s(_m(20, 0.75, 2.5, 6000.0))
    assert small["status"] == acc.PROMISING_SMALL_SAMPLE and small["final_score"] <= 40.0
    one_day = _s(_m(40, 0.75, 2.4, 9000.0, best_share=0.8))
    assert one_day["status"] == acc.REJECTED and "no_single_day_dependency" in one_day["failed"]
    assert _s(_m(40, 0.75, 2.4, 9000.0), data=acc.DATA_UNAVAILABLE)["final_score"] == 0.0
    assert _s(_m(40, 0.75, 2.4, 9000.0), la=False)["status"] == acc.LOOKAHEAD_FAIL
    perfect = _s(_m(40, 0.98, 50.0, 9000.0))
    assert "not_suspicious" in perfect["failed"]


def test_recommendation_logic():
    strong, small = _s(_m(40, 0.75, 2.4, 9000.0, dd=2000.0)), _s(_m(20, 0.75, 2.5, 6000.0))
    below, losing = _s(_m(40, 0.6, 2.0, 8000.0, dd=2000.0)), _s(_m(40, 0.3, 0.6, -4000.0))
    sc = {"S": strong, "P": small, "B": below, "L": losing}
    assert acc.recommend({"BUY": "L", "SELL": "S"}, sc, True) == acc.ACCEPT
    assert acc.recommend({"BUY": "P", "SELL": None}, sc, True) == acc.PROMISING
    assert acc.recommend({"BUY": "B", "SELL": "L"}, sc, True) == acc.KEEP
    assert acc.recommend({"BUY": "L", "SELL": None}, sc, True) == acc.REJECT
    assert acc.recommend({"BUY": "S", "SELL": "S"}, sc, False) == acc.REJECT


# ------------------------------------------------------------------ walk-forward / day / regime
def test_walk_forward_ordering_and_selection_uses_only_prior_days(built):
    _, a = built
    folds = lab.walk_forward_folds(DAYS, 8, 3, 3)
    for f in folds:
        assert max(f["train"]) < min(f["validation"]) and max(f["validation"]) < min(f["test"])
    tests = [d for f in folds for d in f["test"]]
    assert tests == sorted(tests) and len(tests) == len(set(tests))
    names = [n for n in a["_rows"] if n.startswith("BUY_") and a["scores"][n]["data_status"] == "OK"]
    ncond = {n: a["scores"][n]["n_conditions"] for n in names}
    p1 = acc.wf_procedure(names, a["_rows"], ncond, folds, acc.Criteria())
    last = set(folds[-1]["test"])
    flipped = {n: [dict(r, outcome=imp.LOSS, pnl=-abs(r["pnl"])) if r["session_date"] in last and r["pnl"] else r
                   for r in a["_rows"][n]] for n in names}
    p2 = acc.wf_procedure(names, flipped, ncond, folds, acc.Criteria())
    assert [f["selected"] for f in p1["folds"]] == [f["selected"] for f in p2["folds"]]


def test_day_profile_flags_single_day_dominance():
    rows = [_row(DAYS[0], "10:00", "BUY", True, pnl=5000.0), _row(DAYS[1], "10:00", "BUY", False),
            _row(DAYS[2], "10:00", "BUY", True, pnl=200.0)]
    p = acc.day_profile(rows)
    assert p["best_day"] == DAYS[0] and p["worst_day"] == DAYS[1] and p["median_day_pnl"] == 200.0
    assert p["profitable_day_share"] == pytest.approx(2 / 3)
    assert p["best_day_share_of_net"] > 1.0
    m = lab.perf(rows)
    assert m["best_day_share"] > acc.Criteria().max_oos_best_day_share


def test_regime_labels_are_causal_inputs():
    bar = pd.Series({"intraday_move": 0.007, "sign_changes": 7, "atr_pctile": 0.8})
    assert acc.regime_labels(bar, {"premium_vol_ratio": 0.5}) == ["trending", "choppy", "high_volatility",
                                                                   "weak_option_volume"]
    calm = pd.Series({"intraday_move": -0.001, "sign_changes": 2, "atr_pctile": 0.2})
    assert acc.regime_labels(calm, None) == ["sideways", "non_choppy", "low_volatility"]


# ------------------------------------------------------------------ safety / reports
def test_paper_mode_and_production_untouched(built):
    _, a = built
    assert a["trading_mode"] == "PAPER" and a["live_trading_enabled"] is False
    assert a["production_s2_changed"] is False and a["live_orders"] == 0
    assert live_gate.trading_mode() == "PAPER" and live_gate.live_trading_enabled() is False
    src = inspect.getsource(acc)
    assert "place_order" not in src and "OrderManager" not in src
    from scripts import run_70_accuracy_lab as script
    s = inspect.getsource(script)
    assert "place_order" not in s and "order_placement_blocked" in s and "assert_paper_trading_only" in s


def test_reports_have_all_files_and_15_sections(built, tmp_path):
    _, a = built
    paths = acc.write_reports(a, tmp_path)
    for n in acc.REPORT_FILES:
        assert paths[n].is_file() and paths[n].stat().st_size > 0
    md = paths["70_accuracy_lab.md"].read_text(encoding="utf-8")
    for i in range(1, 16):
        assert f"## {i}. " in md
    if not a["target_achieved"]:
        assert acc.NOT_ACHIEVED_TEXT in md
    assert a["recommendation"] in md
    with paths["70_accuracy_candidates.csv"].open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == len(a["scores"])
    audit = paths["70_accuracy_integrity_audit.md"].read_text(encoding="utf-8")
    assert "| C2 |" in audit and "| L1 |" in audit
