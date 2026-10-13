from __future__ import annotations

import csv
import inspect
import shutil
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_s4_six_month_research as script
from src.backtest import accuracy_70_lab as acc
from src.backtest import research_strategies
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import HistoricalContract
from src.historical_data.option_coverage import entry_allowed
from src.options.models import OptionResolutionError
from src.research import s4_six_month_lab as s4
from src.safety import live_gate
from src.strategy import s3_high_confidence as s3

IST = imp.IST
ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 10, 6)
EXPIRIES = [date(2026, 4, 28), date(2026, 5, 26), date(2026, 6, 30), date(2026, 7, 28), date(2026, 8, 25),
            date(2026, 9, 29)]


def _days(per_month: int = 3) -> list[date]:
    out = []
    for m in range(4, 10):
        d, got = date(2026, m, 1), 0
        while got < per_month:
            if d.weekday() < 5:
                out.append(d)
                got += 1
            d += timedelta(days=1)
    return out


DAYS = _days()


def _spot(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in DAYS:
        price = 24000.0 + rng.normal(0, 20)
        drift = rng.choice([-3.0, 3.0])
        for i in range(75):
            ts = pd.Timestamp(datetime.combine(d, time(9, 15)), tz=IST) + pd.Timedelta(minutes=5 * i)
            o = price
            c = o + drift + rng.normal(0, 5)
            rows.append({"timestamp": ts, "instrument_key": "NSE_INDEX|Nifty 50", "open": o,
                         "high": max(o, c) + abs(rng.normal(0, 2)), "low": min(o, c) - abs(rng.normal(0, 2)),
                         "close": c, "volume": 0.0, "open_interest": 0.0})
            price = c
    return pd.DataFrame(rows)


def _chain(expiry: date, kind: str) -> list[HistoricalContract]:
    return [HistoricalContract(instrument_key=f"NSE_FO|{expiry:%m%d}{kind}{k}", underlying="NIFTY", expiry=expiry,
                               instrument_type=kind, lot_size=65, source="EXPIRED", strike=float(k),
                               trading_symbol=f"NIFTY {k} {kind} {expiry:%d %b %y}".upper())
            for k in range(23000, 25001, 50)]


def _premium(spot_rows: pd.DataFrame, c: HistoricalContract, scale: float = 1.0) -> pd.DataFrame:
    sign = 1.0 if c.instrument_type == "CE" else -1.0
    out = spot_rows[["timestamp"]].copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        out[col] = np.maximum(1.0, 150.0 + sign * (spot_rows[col].to_numpy() - c.strike) * 0.5) * scale
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = 1000.0 + (np.arange(len(out)) % 7) * 300.0
    out["open_interest"] = 50000.0
    out["instrument_key"] = c.instrument_key
    out["underlying"], out["expiry"], out["strike"], out["option_type"] = (
        "NIFTY", c.expiry.isoformat(), float(c.strike), c.instrument_type)
    return out


def make_store(root: Path, mutate_september: bool = False) -> dict:
    lay = u6.Layout(root)
    spot = _spot()
    disc = u6.Discovery(calendar=EXPIRIES, chains={(e, k): _chain(e, k) for e in EXPIRIES for k in ("CE", "PE")},
                        calendar_from=date(2026, 4, 1))
    reqs, unresolved = u6.plan_requirements(spot, disc)
    u6.save_plan(lay, date(2026, 4, 1), date(2026, 9, 30), reqs, unresolved, disc)
    written = spot.copy()
    if mutate_september:
        sep = written["timestamp"].dt.month == 9
        for col in ("open", "high", "low", "close"):
            written.loc[sep, col] = written.loc[sep, col] * 1.004
    for m in sorted({d.strftime("%Y-%m") for d in DAYS}):
        u6.write_parquet(lay.underlying_file(m), written[written["timestamp"].dt.strftime("%Y-%m") == m])
    by_key = {r.contract.instrument_key: r.contract for r in reqs}
    candles = 0
    for item in u6.option_items(reqs, lay):
        d = spot["timestamp"].dt.date
        rows = spot[(d >= item.start) & (d <= item.end)]
        scale = 1.3 if mutate_september and item.month == "2026-09" else 1.0
        u6.write_parquet(item.path, _premium(rows, by_key[item.instrument_key], scale))
        candles += len(rows)
    return {"root": root, "candles": candles, "requirements": len(reqs)}


def _allowed(session, params):
    return lambda ts: entry_allowed(ts.to_pydatetime(), session, params.stop_new_entries)


def _boom(*_a, **_k):
    raise AssertionError("legacy data store must never be used by S4")


def build(root: Path):
    rc, params, session, s2cfg, signal_fn = script.load_s2()
    real_pick = acc.pick
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        # synthetic sample is tiny: fall back to the first pre-declared name so the holdout path is exercised
        mp.setattr(acc, "pick", lambda scores, names: real_pick(scores, names) or sorted(names)[0])
        store = s4.SixMonthStore(root)
        spot = store.spot()
        di = s4.data_integrity(store, spot)
        a = s4.build_s4(spot=spot, signal_fn=signal_fn, resolve=s4.make_resolve(store, rc),
                        premium_for_factory=store.premium_for, params=params, square_off=rc.session.square_off,
                        range_start=s2cfg["range_start"], range_end=s2cfg["range_end"],
                        allowed=_allowed(session, params), today=TODAY, rules=lab.s2_rules(rc, params, s2cfg),
                        run_mutation=False, lookahead_days=[DAYS[7].isoformat()], cut_times=("12:00",),
                        integrity_data=di, store=store)
    return store, spot, di, a


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    info = make_store(tmp_path_factory.mktemp("s4") / "upstox_6m")
    store, spot, di, a = build(info["root"])
    return SimpleNamespace(info=info, store=store, spot=spot, di=di, a=a)


@pytest.fixture(scope="module")
def built_mutated_september(tmp_path_factory):
    info = make_store(tmp_path_factory.mktemp("s4m") / "upstox_6m", mutate_september=True)
    return build(info["root"])[3]


def _status(a):
    return {x["id"]: x["status"] for x in a["integrity"]}


# ------------------------------------------------------------------ dataset
def test_dataset_loading_and_six_month_coverage(built):
    s = built.di["stats"]
    assert built.di["ok"], [c for c in built.di["checks"] if c["status"] == acc.FAIL]
    assert s["trading_days"] == len(DAYS) and s["underlying_candles"] == 75 * len(DAYS)
    assert s["first_day"] == DAYS[0].isoformat() and s["last_day"] == DAYS[-1].isoformat()
    assert s["option_candles"] == built.info["candles"] and s["requirements"] == built.info["requirements"]
    assert {m for m in (d.strftime("%Y-%m") for d in DAYS)} == {f"2026-0{m}" for m in range(4, 10)}
    a = built.a
    assert a["dataset"]["trading_days"] == len(DAYS) and _status(a)["A1"] == acc.PASS


def test_only_the_six_month_store_is_read(built, tmp_path):
    assert set(built.store.reads) <= {s4.SOURCE, "missing"} and built.store.reads[s4.SOURCE] > 0
    assert _status(built.a)["A2"] == acc.PASS
    assert s4.module_safety()["no_legacy_store"]
    with pytest.raises(FileNotFoundError):
        s4.SixMonthStore(tmp_path / "missing_dataset")


def test_data_integrity_detects_corrupt_option_file(built, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(built.info["root"], root)
    store = s4.SixMonthStore(root)
    p = store.layout.option_files()[0]
    f = u6.read_parquet(p)
    pd.concat([f, f.iloc[:3]], ignore_index=True).to_parquet(p, index=False)
    di = s4.data_integrity(store, store.spot())
    st = {c["id"]: c["status"] for c in di["checks"]}
    assert not di["ok"] and st["P1-07"] == acc.FAIL


# ------------------------------------------------------------------ splits / September isolation
def test_chronological_splits_and_september_isolation(built):
    sp = built.a["splits"]
    assert sp["TRAIN"]["days"] and all("2026-04-01" <= d <= "2026-06-30" for d in sp["TRAIN"]["days"])
    assert all(d[:7] == "2026-07" for d in sp["VALIDATION"]["days"]) and sp["VALIDATION"]["days"]
    assert all(d[:7] == "2026-08" for d in sp["OOS"]["days"]) and sp["OOS"]["days"]
    assert all(d[:7] == "2026-09" for d in sp["HOLDOUT"]["days"]) and sp["HOLDOUT"]["days"]
    st = _status(built.a)
    assert st["T1"] == st["T2"] == st["T3"] == st["A3"] == acc.PASS
    assert all(r["session_date"] <= "2026-08-31" for rs in built.a["_rows"].values() for r in rs)
    assert all(r["session_date"] >= "2026-09-01" for rs in built.a["_holdout_rows"].values() for r in rs)
    assert built.a["september_used_for_optimization"] is False


def test_september_data_cannot_change_selection(built, built_mutated_september):
    a, b = built.a, built_mutated_september
    assert a["finalists"] == b["finalists"]
    assert {d: [p["selected"] for p in v] for d, v in a["wf_procedure"].items()} == \
           {d: [p["selected"] for p in v] for d, v in b["wf_procedure"].items()}
    for n in a["scores"]:
        for p in (*s4.SELECTION_SPLITS, "ALL"):
            assert acc._fp(a["scores"][n][p]) == acc._fp(b["scores"][n][p]), (n, p)
    for k in ("BUY", "SELL", "COMBINED"):
        assert acc._fp(a["s2"][k]["OOS"]) == acc._fp(b["s2"][k]["OOS"])


def test_holdout_stage_uses_only_frozen_candidates(built):
    a = built.a
    frozen = {n for n in a["finalists"].values() if n} | {p["selected"] for v in a["w5_pick"].values() for p in v
                                                          if p["selected"]}
    assert set(a["_holdout_rows"]) == frozen
    for d, f in a["final"].items():
        if d in s4.DIRECTIONS:
            assert f["cid"] == a["finalists"][d]


# ------------------------------------------------------------------ lookahead / point-in-time
def test_no_lookahead_filters_scores_and_option_features(built):
    la = built.a["lookahead"]
    assert la["truncation"]["passed"] and la["option"]["passed"] and la["passed"]
    st = _status(built.a)
    assert st["L2"] == st["L3"] == acc.PASS and st["L1"] == acc.INFO


def test_truncation_test_catches_a_leaky_feature(built, monkeypatch):
    real = s3.s3_bars

    def leaky(frame, interval=5):
        f = real(frame, interval)
        f["ema20"] = f["close"].shift(-1)
        return f

    monkeypatch.setattr(s3, "s3_bars", leaky)
    assert not s4.truncation_test(built.spot, [DAYS[8].isoformat()], ("12:00",))["passed"]


def test_exact_s2_pipeline_mutation_test_passes(built):
    rc, params, session, s2cfg, signal_fn = script.load_s2()
    day = [d.isoformat() for d in DAYS if d.month == 5][1]
    cand = [c for c in s4.predeclared_candidates() if c.cid == "S4_BUY_A_A_NEXT_BAR"]
    st = s4.run_stage(spot=built.spot[built.spot["timestamp"].dt.date <= s4.SELECTION_END].reset_index(drop=True),
                      days=[day], signal_fn=signal_fn, resolve=s4.make_resolve(built.store, rc),
                      premium_for=built.store.premium_for(s4.SELECTION_END), params=params, square_off="15:15",
                      range_start="09:15", range_end="09:30", underlying="NIFTY", allowed=_allowed(session, params),
                      candidates=cand)
    assert st["base_contexts"]
    assert s4.baseline_mutation_test(st, [day])["passed"]


def test_completed_candles_next_bar_entry_and_point_in_time_contracts(built):
    st = _status(built.a)
    assert st["C1"] == st["C2"] == st["C3"] == st["C4"] == acc.PASS
    for rs in built.a["_rows"].values():
        for r in rs:
            if r["state"] == sh.EVALUATED:
                assert pd.Timestamp(r["entry_time"]) == pd.Timestamp(r["signal_time"]) + pd.Timedelta(minutes=5)


def test_resolver_is_point_in_time_and_atm_only(built):
    store = built.store
    res = s4.SixMonthResolver(store)
    d = DAYS[4]
    res.today = lambda: d
    assert res.expiries("NIFTY")[0] == next(e for e in EXPIRIES if e >= d)
    assert all(e >= d for e in res.expiries("NIFTY"))
    strikes = res.strikes("NIFTY", res.expiries("NIFTY")[0], "CE")
    non_atm = [k for k, c in store.chain[(d, res.expiries("NIFTY")[0], "CE")].items()
               if c.instrument_key not in store.atm[d]]
    assert strikes and non_atm
    with pytest.raises(OptionResolutionError) as e:
        res.resolve("NIFTY", res.expiries("NIFTY")[0], non_atm[0], "CE")
    assert e.value.code == "OUTSIDE_PLANNED_STRIKES"
    key = next(iter(store.atm[d]))
    prem = store.premium_for(date(2026, 4, 2))(SimpleNamespace(instrument_key=key))
    assert isinstance(prem, str) or prem["timestamp"].dt.date.max() <= date(2026, 4, 2)
    assert store.premium_for()(SimpleNamespace(instrument_key="NSE_FO|UNKNOWN")) == s4.MISSING_PREMIUM


def test_missing_premium_is_no_signal():
    assert s4.premium_confirms(None) is None
    assert s4.premium_confirms({"premium_mom3": None, "premium_bull": True}) is None
    assert s4.premium_confirms({"premium_mom3": 0.01, "premium_bull": True}) is True
    f = pd.DataFrame({"timestamp": [pd.Timestamp("2026-05-04 10:00", tz=IST)]})
    cand = s4.Candidate("X", "F", "BUY", "A_NEXT_BAR")
    key = (f["timestamp"].iat[0].isoformat(), "BUY")
    ns = Counter()
    assert s4.setup_keys(cand, {"BUY": [key]}, {}, {key[0]: 0}, {}, {}, ns) == []
    assert ns["X:PREMIUM_DATA_UNAVAILABLE"] == 1


# ------------------------------------------------------------------ isolation
def test_buy_sell_and_candidate_isolation(built):
    a = built.a
    for n, rs in a["_rows"].items():
        assert all(r["direction"] == a["scores"][n]["direction"] and r["strategy"] == n for r in rs)
    st = _status(a)
    for cid in ("I1", "I2", "I3", "I4", "I5", "I6"):
        assert st[cid] == acc.PASS, cid
    assert len(a["scores"]) == 94 and len({c.cid for c in s4.predeclared_candidates()}) == 94


def test_family_a_next_bar_is_exact_s2(built):
    a = built.a
    for d in s4.DIRECTIONS:
        assert acc._sig(a["_rows"][f"S4_{d}_A_A_NEXT_BAR"]) == acc._sig(
            [r for r in a["_base_rows"] if r["direction"] == d])


# ------------------------------------------------------------------ score / regime
def test_score_weights_and_threshold_monotonic(built):
    assert s3.WEIGHTS == {"trend": 20, "vwap": 15, "ema": 15, "momentum": 10, "volume": 10, "price_action": 10,
                          "breakout": 10, "option_premium": 10}
    assert sum(s3.WEIGHTS.values()) == 100 and list(s3.THRESHOLDS) == [60, 65, 70, 75, 80, 85, 90]
    rows = built.a["_rows"]
    for d in s4.DIRECTIONS:
        prev = None
        for t in s3.THRESHOLDS:
            keys = {acc.row_key(r) for r in rows[f"S4_{d}_I_T{t}_A_NEXT_BAR"]}
            assert all((r.get("s4_score") or 0) >= t for r in rows[f"S4_{d}_I_T{t}_A_NEXT_BAR"])
            assert prev is None or keys <= prev
            prev = keys


def test_regimes_are_causal_labels_and_regime_filter(built):
    bars = s3.s3_bars(built.spot)
    assert set(bars["regime"]) <= set(s4.PRIMARY_REGIMES)
    m = s4.family_masks(bars)
    for d in s4.DIRECTIONS:
        bad = bars.loc[m[d]["G"], "regime"]
        assert not bad.isin(s3.NO_TRADE_REGIMES).any()
        assert not bad.eq("TREND_DOWN" if d == "BUY" else "TREND_UP").any()
    assert set(built.a["s2_by_regime"]) == set(s4.DIRECTIONS)


# ------------------------------------------------------------------ walk-forward / holdout
def test_walk_forward_ordering_and_holdout_isolation(built):
    days = [d.isoformat() for d in DAYS]
    ws = s4.wf_windows(days, [w[0] for w in s4.WF_WINDOWS])
    assert [w["window"] for w in ws] == ["W1", "W2", "W3", "W4", "W5"]
    for w in ws:
        assert max(w["train"]) < min(w["test"])
        tr, va = s4.inner_split(w)
        assert tr and va and max(tr) < min(va) and max(va) < min(w["test"])
    assert all(d[:7] != "2026-09" for w in s4.wf_windows(days, s4.CANDIDATE_WF) for d in w["test"] + w["train"])
    assert all(d[:7] == "2026-09" for d in ws[-1]["test"])
    proc = built.a["windows"]["SELECTION_PROCEDURE"]
    assert [r["window"] for r in proc] == ["W1", "W2", "W3", "W4", "W5"]


def test_holdout_verdict_final_status_and_recommendation():
    hc = s4.HoldoutCriteria()
    m = lambda n, wr, pf, pnl, share=0.3: {"evaluated": n, "win_rate": wr, "profit_factor": pf,  # noqa: E731
                                           "net_pnl": pnl, "best_day_share": share}
    assert s4.holdout_verdict(m(5, 0.9, 3.0, 100), hc) == s4.HOLDOUT_INSUFFICIENT
    assert s4.holdout_verdict(m(20, 0.7, 2.0, 100), hc) == s4.HOLDOUT_PASS
    assert s4.holdout_verdict(m(20, 0.5, 2.0, 100), hc) == s4.HOLDOUT_FAIL
    assert s4.holdout_verdict(m(20, 0.7, 2.0, 100, 0.8), hc) == s4.HOLDOUT_FAIL
    ok = {k: True for k in s4.ROBUST_CRITERIA}
    assert len(s4.ROBUST_CRITERIA) == 13
    assert s4.final_status(s4.PASS_PENDING, s4.HOLDOUT_PASS, ok) == s4.ROBUST_70
    assert s4.final_status(s4.PASS_PENDING, s4.HOLDOUT_FAIL, ok) == s4.FAILED_HOLDOUT
    for k in s4.ROBUST_CRITERIA:
        assert s4.final_status(s4.PASS_PENDING, s4.HOLDOUT_PASS, dict(ok, **{k: False})) != s4.ROBUST_70
    assert s4.final_status(s4.PROMISING, s4.HOLDOUT_PASS, dict(ok, oos_accuracy_70=False)) == s4.READY
    assert s4.final_status(s4.PROMISING, s4.HOLDOUT_INSUFFICIENT, dict(ok, september_acceptable=False)) == s4.PROMISING
    assert s4.final_status(s4.INSUFFICIENT_DATA, s4.HOLDOUT_PASS, dict(ok, oos_sample_sufficient=False)) == \
        s4.INSUFFICIENT_DATA
    c = acc.Criteria()
    assert s4.recommend([s4.ROBUST_70], False, {}, c) == s4.REJECTED
    assert s4.recommend([s4.READY, s4.REJECTED], True, {}, c) == s4.READY
    assert s4.recommend([s4.FAILED_HOLDOUT], True, {}, c) == s4.KEEP
    assert s4.recommend([], True, {"BUY": 3, "SELL": 2}, c) == s4.INSUFFICIENT_DATA
    assert s4.recommend([s4.INSUFFICIENT_DATA], True, {}, c) == s4.INSUFFICIENT_DATA


def test_insufficient_sample_never_counts_as_70(built):
    a = built.a
    for n, s in a["scores"].items():
        if s["tier"] in ("C", "D"):
            assert s["status"] in (s4.INSUFFICIENT_DATA, s4.REJECTED, s4.PROMISING)
            assert s["status"] != s4.PASS_PENDING
    if a["target_achieved"]:
        assert any(a["scores"][f["cid"]]["tier"] in ("A", "B") for d, f in a["final"].items() if d in s4.DIRECTIONS)


# ------------------------------------------------------------------ safety
def test_no_s2_mutation_and_s4_not_registered(built):
    assert built.a["production_s2_changed"] is False and _status(built.a)["I6"] == acc.PASS
    assert "s4_six_month" not in inspect.getsource(research_strategies)
    for name in ("runtime.py", "signal_generator.py", "strategy_engine.py"):
        p = ROOT / "src" / "strategy" / name
        if p.is_file():
            assert "s4_six_month" not in p.read_text(encoding="utf-8")


def test_paper_mode_and_no_orders(built):
    a = built.a
    assert a["trading_mode"] == "PAPER" and a["live_trading_enabled"] is False and a["live_orders"] == 0
    assert live_gate.trading_mode() == "PAPER" and live_gate.live_trading_enabled() is False
    assert "place_order" not in inspect.getsource(s4) and "OrderManager" not in inspect.getsource(s4)
    src = inspect.getsource(script)
    assert "place_order" not in src and "order_placement_blocked" in src and "assert_paper_trading_only" in src
    assert _status(a)["Z1"] == acc.PASS


# ------------------------------------------------------------------ reports
def test_reports_and_final_summary(built, tmp_path, capsys):
    a = built.a
    paths = s4.write_reports(a, tmp_path)
    for n in s4.REPORT_FILES:
        assert paths[n].is_file() and paths[n].stat().st_size > 0
    md = paths["s4_six_month_research.md"].read_text(encoding="utf-8")
    assert s4.BANNER in md and a["recommendation"] in md
    for i in range(1, 17):
        assert f"## {i}. " in md
    with paths["s4_candidates.csv"].open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 94
    with paths["s4_baseline.csv"].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["scope"] for r in rows} == {"BUY", "SELL", "COMBINED"}
    assert {"TRAIN", "VALIDATION", "OOS", "HOLDOUT"} <= {r["period"] for r in rows}
    script.print_summary(a, paths)
    out = capsys.readouterr().out
    for line in ("S4 SIX-MONTH RESEARCH", "Dataset: ", "Trading days: ", "Option candles: ", "S2 BASELINE:",
                 "BEST S4:", "FINAL SEPTEMBER HOLDOUT:", "70% OOS TARGET: ", "ROBUST_70: ", "LOOKAHEAD: ",
                 "INTEGRITY: ", "SEPTEMBER USED FOR OPTIMIZATION: NO", "PRODUCTION S2 CHANGED: NO", "LIVE ORDERS: 0",
                 "FINAL RECOMMENDATION: "):
        assert line in out
