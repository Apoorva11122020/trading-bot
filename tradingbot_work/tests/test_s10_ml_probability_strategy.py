from __future__ import annotations

import inspect
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="S10 needs scikit-learn (pip install scikit-learn)")

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s10_ml_probability_strategy as script
from src.historical_data import nifty_futures as nf
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.options.selection import atm_strike
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s10_ml_probability as s10
from tests.test_s4_six_month_research import _chain, _premium, make_store
from tests.test_s9_futures_data import EXPIRED, add_futures, bars, fut

IST = sd.IST
STEP = sd.STEP
TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S10"
OCT_DAYS = [date(2026, 10, 1), date(2026, 10, 5), date(2026, 10, 6)]
OCT_EXPIRY = date(2026, 10, 27)
DECISIONS = {s10.PROMISING, s10.REJECTED, s10.INSUFF_DECISION, s10.UNSTABLE}


def _boom(*_a, **_k):
    raise AssertionError("S10 must read only the 6-month / fresh October stores")


def at(day: date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time.fromisoformat(hhmm)), tz=IST)


# ------------------------------------------------------------------ synthetic fresh October store
def make_fresh_store(root: Path) -> None:
    lay = u6.Layout(root)
    spot = bars(OCT_DAYS, base=24100.0, seed=17, volume=False)
    spot["volume"], spot["open_interest"] = 0.0, 0.0
    spot["instrument_key"] = "NSE_INDEX|Nifty 50"
    disc = u6.Discovery(calendar=[OCT_EXPIRY], chains={(OCT_EXPIRY, k): _chain(OCT_EXPIRY, k) for k in ("CE", "PE")},
                        calendar_from=OCT_DAYS[0])
    reqs, unresolved = u6.plan_requirements(spot, disc)
    u6.save_plan(lay, OCT_DAYS[0], OCT_DAYS[-1], reqs, unresolved, disc)
    u6.write_parquet(lay.underlying_file("2026-10"), spot)
    by_key = {r.contract.instrument_key: r.contract for r in reqs}
    for item in u6.option_items(reqs, lay):
        d = spot["timestamp"].dt.date
        u6.write_parquet(item.path, _premium(spot[(d >= item.start) & (d <= item.end)], by_key[item.instrument_key]))
    flay = nf.FuturesLayout(root)
    c = fut(OCT_EXPIRY)
    lo, hi = nf.contract_window(c, OCT_DAYS[0], OCT_DAYS[-1])
    dd = spot["timestamp"].dt.date
    sub = spot[(dd >= lo) & (dd <= hi)][["timestamp", "open", "high", "low", "close"]].copy()
    for col in ("open", "high", "low", "close"):
        sub[col] = sub[col] + 30.0
    sub["volume"] = np.random.default_rng(3).integers(500, 2000, len(sub)).astype(float)
    sub["open_interest"] = 1e5
    u6.write_parquet(flay.contract_file(c), nf._attach(sub, c))
    u6._write_csv(flay.contracts, nf.CONTRACT_COLUMNS,
                  [{"instrument_key": c.instrument_key, "trading_symbol": c.trading_symbol,
                    "exchange_segment": "NSE_FO", "instrument_type": "FUT", "underlying": "NIFTY",
                    "expiry": OCT_EXPIRY.isoformat(), "lot_size": 65, "tick_size": 0.1, "source": EXPIRED,
                    "candles": len(sub), "status": u6.SUCCESS}])
    u6._write_csv(flay.invalid, nf.INVALID_COLUMNS, [])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Full S10 pipeline on the synthetic stores. The stage gate is forced to PASS so that every stage (OOS,
    September, fresh October) executes and the chronology / isolation can be verified; the real gates are
    unit-tested separately."""
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s10")
    six, fresh, out = base / "upstox_6m", base / "fresh_oct", base / "out"
    make_store(six)
    add_futures(six)
    make_fresh_store(fresh)
    calls: list[dict] = []
    real_load = sd.load_market

    def spy(roots, underlying="NIFTY", max_day=None):
        calls.append({"roots": [Path(r).as_posix() for r in roots], "max_day": max_day,
                      "frozen": (out / "models" / "model_metadata.json").is_file()})
        return real_load(roots, underlying, max_day)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(sd, "load_market", spy)
        mp.setattr(s10, "gate", lambda m, n: s10.PASS)
        a = script.run(data_root=six, fresh_root=fresh, out_dir=out, mutation_days=2, check_sample=5,
                       secret_values=[TOKEN])
    return SimpleNamespace(a=a, base=base, six=six, fresh=fresh, out=out, calls=calls, before=before)


# ------------------------------------------------------------------ toy dataset for model-layer unit tests
def toy(n_days: int = 20, per_day: int = 30, seed: int = 0, start: date = date(2026, 4, 1)):
    rng = np.random.default_rng(seed)
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rows, grid = [], []
    for d in days:
        grid += [at(d, "09:15") + i * STEP for i in range(75)]
        for i in range(per_day):
            ts = at(d, "09:45") + i * STEP
            for side in sd.SIDES:
                x1 = rng.normal()
                x2 = rng.normal() if rng.random() > 0.1 else np.nan
                y = float(rng.random() < 1 / (1 + np.exp(-1.5 * x1)))
                rows.append({"signal_time": ts, "session_date": d.isoformat(), "side": side,
                             "outcome": sd.WIN if y else sd.LOSS, "y": y, "event_end": ts + 7 * STEP,
                             "f1": x1, "f2": x2, "f_empty": np.nan})
    data = pd.DataFrame(rows).sort_values(["signal_time", "side"]).reset_index(drop=True)
    return data, np.sort(pd.DatetimeIndex(grid).asi8), [x.isoformat() for x in days]


COLS = ["f1", "f2", "f_empty"]
LR_BOTH = {s: sm.LR for s in sd.SIDES}


def series(rows, start: pd.Timestamp, gap_after: int | None = None) -> sd.OptionSeries:
    ts = [start + i * STEP for i in range(len(rows))]
    if gap_after is not None:
        ts = ts[:gap_after + 1] + [t + STEP for t in ts[gap_after + 1:]]
    t = pd.DatetimeIndex(ts)
    o, h, l, c = (np.array(v, dtype=float) for v in zip(*rows))
    return sd.OptionSeries(pd.DataFrame(index=t), t, o, h, l, c)


D0 = date(2026, 4, 6)
SIG = at(D0, "10:00")
FLAT = (100.0, 101.0, 99.0, 100.0)


# 1 ------------------------------------------------------------------ point-in-time contract selection
def test_point_in_time_selection(env):
    a, m = env.a, env.a["_market"]
    data = a["_data"]
    spot = m.spot_close()
    rows = data.iloc[::max(1, len(data) // 40)].to_dict("records")
    assert rows
    for r in rows:
        d = date.fromisoformat(r["session_date"])
        exp = sd.nearest_expiry(m.calendar, d)
        chain = m.chains[(d, exp, r["side"])]
        assert r["spot"] == spot[r["signal_time"]]
        assert float(atm_strike(sorted(chain), spot[r["signal_time"]])) == r["strike"]
        assert chain[r["strike"]].instrument_key == r["instrument_key"]
    assert a["lookahead"]["contract_selection_causality"]["passed"]
    assert sd.nearest_expiry([date(2026, 4, 28), date(2026, 5, 26)], date(2026, 4, 28)) == date(2026, 4, 28)
    assert sd.nearest_expiry([date(2026, 4, 28), date(2026, 5, 26)], date(2026, 4, 29)) == date(2026, 5, 26)


# 2 ------------------------------------------------------------------ feature causality (full pipeline)
def test_feature_causality(env):
    la = env.a["lookahead"]
    for k in ("future_candle_scrambling", "feature_truncation"):
        assert la[k]["passed"] and (la[k].get("rows_checked") or la[k].get("checked")), (k, la[k])


# 3 ------------------------------------------------------------------ rolling-window causality
def test_rolling_causality():
    days = [date(2026, 4, 6), date(2026, 4, 7)]
    frame = bars(days, seed=5)
    cut = at(days[1], "12:00")
    full = sd.futures_feature_frame(frame)
    for kw in ({"price": True, "volume": True}, {"drop": True}):
        part = sd.futures_feature_frame(sd.mutate_frame(frame, cut, seed=9, **kw))
        cols = [*sd.FUT_FEATURES, *sd.TIME_FEATURES]
        a, b = full.loc[full.index <= cut, cols], part.loc[part.index <= cut, cols]
        assert len(a) == len(b) and np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), kw
        later = full.index > cut
        if "price" in kw:
            assert not np.allclose(full.loc[later, "fut_ret_1"], part.loc[later, "fut_ret_1"], equal_nan=True)


# 4 ------------------------------------------------------------------ option feature causality
def test_option_causality(env):
    days = [date(2026, 4, 6), date(2026, 4, 7)]
    prem = bars(days, base=150.0, seed=8)
    cut = at(days[1], "11:00")
    full = sd.option_series(prem)
    for kw in ({"price": True, "volume": True, "oi": True}, {"drop": True}):
        part = sd.option_series(sd.mutate_frame(prem, cut, seed=4, **kw))
        a, b = full.feat[full.feat.index <= cut], part.feat[part.feat.index <= cut]
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), kw
    for k in ("future_option_candle_mutation", "future_option_price_mutation", "future_volume_mutation",
              "future_oi_mutation"):
        assert env.a["lookahead"][k]["passed"], (k, env.a["lookahead"][k])


# 5 ------------------------------------------------------------------ labels
def test_labels_win_and_levels():
    s = series([FLAT, (100.0, 103.0, 99.0, 102.0), (102.0, 107.0, 101.0, 106.0), FLAT], SIG)
    r = sd.label_trade(s, SIG, 2.0, sd.HORIZON)
    assert (r["entry"], r["risk"], r["stop_loss"], r["target"]) == (100.0, 3.0, 97.0, 106.0)
    assert r["outcome"] == sd.WIN and r["exit_reason"] == "TARGET" and r["exit_price"] == 106.0
    assert r["r_multiple"] == 2.0 and r["pnl"] == round(6.0 * sd.LOT, 2)
    assert r["exit_time"] == SIG + 3 * STEP and r["event_end"] == SIG + STEP + sd.HORIZON * STEP
    assert sd.label_trade(s, SIG, 0.0, sd.HORIZON)["outcome"] == sd.INVALID_LEVELS
    assert sd.label_trade(s, SIG, 80.0, sd.HORIZON)["outcome"] == sd.INVALID_LEVELS  # stop <= 0


# 6 ------------------------------------------------------------------ target / stop order
def test_target_stop_order():
    s = series([FLAT, (100.0, 102.0, 98.0, 101.0), (101.0, 102.0, 96.0, 97.0), (97.0, 110.0, 97.0, 108.0)], SIG)
    r = sd.label_trade(s, SIG, 2.0, sd.HORIZON)
    assert r["outcome"] == sd.LOSS and r["exit_reason"] == "STOP" and r["r_multiple"] == -1.0
    gap = series([FLAT, (100.0, 101.0, 99.0, 100.0), (95.0, 96.0, 94.0, 95.0)], SIG)
    g = sd.label_trade(gap, SIG, 2.0, sd.HORIZON)
    assert g["outcome"] == sd.LOSS and g["exit_reason"] == "STOP_GAP" and g["exit_price"] == 95.0
    up = series([FLAT, (100.0, 101.0, 99.0, 100.0), (108.0, 109.0, 107.0, 108.0)], SIG)
    u = sd.label_trade(up, SIG, 2.0, sd.HORIZON)
    assert u["outcome"] == sd.WIN and u["exit_reason"] == "TARGET_GAP" and u["exit_price"] == 108.0


# 7 ------------------------------------------------------------------ ambiguous
def test_ambiguous_candle():
    s = series([FLAT, (100.0, 107.0, 96.0, 100.0), (100.0, 120.0, 100.0, 119.0)], SIG)
    r = sd.label_trade(s, SIG, 2.0, sd.HORIZON)
    assert r["outcome"] == sd.AMBIGUOUS and r["r_multiple"] == -1.0 and r["exit_price"] == 97.0
    assert r["outcome"] not in sd.BINARY


# 8 ------------------------------------------------------------------ timeout (and no peeking past the horizon)
def test_timeout_at_horizon_close():
    rows = [FLAT] + [(100.0, 101.0, 99.0, 100.0 + 0.1 * i) for i in range(1, 7)] + [(100.0, 120.0, 99.0, 119.0)]
    s = series(rows, SIG)
    r = sd.label_trade(s, SIG, 2.0, sd.HORIZON)
    assert r["outcome"] == sd.TIMEOUT and r["exit_reason"] == "TIMEOUT_HORIZON_CLOSE"
    assert r["exit_price"] == pytest.approx(100.6) and r["horizon_complete"] and r["horizon_candles"] == 6
    assert r["exit_time"] == SIG + STEP + 6 * STEP
    assert sd.label_trade(s, SIG, 2.0, 7)["outcome"] == sd.WIN


# 9 ------------------------------------------------------------------ next-bar entry
def test_next_bar_entry(env):
    s = series([(90.0, 91.0, 89.0, 90.0), (100.5, 101.0, 99.0, 100.0), FLAT], SIG)
    assert sd.label_trade(s, SIG, 2.0, sd.HORIZON)["entry"] == 100.5
    missing = series([FLAT, FLAT, FLAT], SIG, gap_after=0)
    assert sd.label_trade(missing, SIG, 2.0, sd.HORIZON)["outcome"] == sd.NO_ENTRY
    data = env.a["_data"]
    assert (pd.DatetimeIndex(data["entry_time"]) == pd.DatetimeIndex(data["signal_time"]) + STEP).all()
    for r in env.a["results"].values():
        assert all(t["entry_time"] == t["signal_time"] + STEP for t in r["trades"])


# 10 ------------------------------------------------------------------ scaler fit on TRAIN (model-fit fold) only
def test_scaler_train_only():
    data, grid, days = toy()
    train, test = days[:15], days[15:]
    m = sm.fit_model(data, COLS, train, LR_BOTH, grid, test_days=test)
    X = data.loc[m.pre_fit_ids, COLS].to_numpy(dtype=float)
    sc = m.preprocessor.named_steps["scaler"]
    assert int(sc.n_samples_seen_) == len(X)
    assert np.allclose(sc.mean_, m.preprocessor.named_steps["imputer"].transform(X).mean(axis=0))
    assert set(data.loc[m.pre_fit_ids, "session_date"]) <= set(m.fit_days)
    shifted = data.copy()
    later = shifted["session_date"] >= test[0]
    shifted.loc[later, ["f1", "f2"]] = shifted.loc[later, ["f1", "f2"]] * 100 + 50
    m2 = sm.fit_model(shifted, COLS, train, LR_BOTH, grid, test_days=test)
    assert np.allclose(m2.preprocessor.named_steps["scaler"].mean_, sc.mean_)
    assert np.allclose(m2.preprocessor.named_steps["scaler"].scale_, sc.scale_)


# 11 ------------------------------------------------------------------ imputer fit on TRAIN only
def test_imputer_train_only():
    data, grid, days = toy()
    train, test = days[:15], days[15:]
    m = sm.fit_model(data, COLS, train, LR_BOTH, grid, test_days=test)
    imp = m.preprocessor.named_steps["imputer"]
    X = data.loc[m.pre_fit_ids, COLS].to_numpy(dtype=float)
    assert np.isclose(imp.statistics_[0], np.nanmedian(X[:, 0]))
    assert np.isclose(imp.statistics_[1], np.nanmedian(X[:, 1]))
    assert imp.statistics_[2] == 0.0 and m.preprocessor.transform(X).shape[1] == len(COLS)
    changed = data.copy()
    changed.loc[changed["session_date"] >= test[0], "f2"] = 1e6
    m2 = sm.fit_model(changed, COLS, train, LR_BOTH, grid, test_days=test)
    assert np.allclose(m2.preprocessor.named_steps["imputer"].statistics_, imp.statistics_)


# 12 ------------------------------------------------------------------ calibration without leakage
def test_calibration_no_leakage(env):
    data, grid, days = toy()
    train, test = days[:15], days[15:]
    m = sm.fit_model(data, COLS, train, LR_BOTH, grid, test_days=test)
    assert m.fit_days and m.calib_days and max(m.fit_days) < min(m.calib_days) and max(m.calib_days) <= train[-1]
    for s in sd.SIDES:
        side = m.sides[s]
        assert side.status == sm.FITTED and side.calibration == sm.PLATT
        assert not set(side.calib_ids) & set(side.fit_ids) and not set(side.calib_ids) & set(m.pre_fit_ids)
        assert set(data.loc[side.calib_ids, "session_date"]) <= set(m.calib_days)
        p = m.predict(s, data[data["side"] == s])
        assert np.all((p > 0) & (p < 1))
    flipped = data.copy()
    later = flipped["session_date"] >= test[0]
    flipped.loc[later, "y"] = 1.0 - flipped.loc[later, "y"]
    flipped.loc[later, "outcome"] = np.where(flipped.loc[later, "y"] == 1.0, sd.WIN, sd.LOSS)
    assert sm.fingerprint(sm.fit_model(flipped, COLS, train, LR_BOTH, grid, test_days=test)) == sm.fingerprint(m)
    few, _, _ = toy(n_days=4, per_day=10)
    tiny = sm.fit_model(few, COLS, sorted(set(few["session_date"])), LR_BOTH, grid)
    assert all(np.isnan(tiny.predict(s, few[few["side"] == s])).all() for s in sd.SIDES)
    assert env.a["leakage"]["probability_calibration_leakage"]["passed"]


# 13 ------------------------------------------------------------------ threshold isolation
def test_threshold_isolation(env):
    a = env.a
    lo, hi = sd.INNER_VALIDATION
    assert a["threshold"] in sm.THRESHOLDS
    assert all(lo.isoformat() <= d <= hi.isoformat() for d in a["inner"]["threshold_input_days"])
    assert a["leakage"]["threshold_leakage"]["passed"]
    rows = [{"threshold": 0.55, "resolved": 40, "expectancy_r": 0.10},
            {"threshold": 0.60, "resolved": 25, "expectancy_r": 0.30},
            {"threshold": 0.65, "resolved": 5, "expectancy_r": 0.90}]
    assert sm.choose_threshold(rows)[0] == 0.60
    assert sm.choose_threshold([{**r, "resolved": 3} for r in rows])[0] == 0.55
    tie = [{"threshold": 0.55, "resolved": 30, "expectancy_r": 0.2}, {"threshold": 0.6, "resolved": 30,
                                                                     "expectancy_r": 0.2}]
    assert sm.choose_threshold(tie)[0] == 0.55
    assert all(not r["used_for_selection"] for r in a["robustness"])


# 14 ------------------------------------------------------------------ purged split with embargo
def test_purged_split():
    d1 = date(2026, 4, 6)
    grid = np.sort(pd.DatetimeIndex([at(d1, "09:15") + i * STEP for i in range(75)]).asi8)
    test = pd.DataFrame({"signal_time": [at(d1, "12:00"), at(d1, "12:30")]})
    test["event_end"] = test["signal_time"] + 7 * STEP
    sig = [at(d1, "10:00"), at(d1, "11:40"), at(d1, "13:10"), at(d1, "14:00")]
    train = pd.DataFrame({"signal_time": sig, "event_end": [t + 7 * STEP for t in sig]})
    keep, info = sm.purge_mask(train, test, grid, embargo_bars=6)
    assert keep.tolist() == [True, False, False, True]
    assert info == {"train": 4, "purged": 1, "embargoed": 1, "kept": 2}
    assert sm.EMBARGO_BARS >= 6
    data, g2, days = toy()
    m = sm.fit_model(data, COLS, days[:15], LR_BOTH, g2, test_days=days[15:])
    test2 = data[data["session_date"].isin(days[15:])]
    assert sm.purge_violations(data.loc[m.pre_fit_ids], test2, g2) == 0
    cal = data[data["session_date"].isin(m.calib_days)]
    assert sm.purge_violations(data.loc[m.pre_fit_ids], cal, g2) == 0


# 15 ------------------------------------------------------------------ no TRAIN / OOS contamination
def test_no_train_oos_contamination(env):
    a = env.a
    lk = a["leakage"]
    assert lk["train_oos_contamination"]["passed"] and lk["purging_correctness"]["passed"]
    assert lk["scaler_leakage"]["passed"] and lk["imputer_leakage"]["passed"]
    hi = sd.SPLITS["TRAIN"][1].isoformat()
    meta = a["frozen"]["metadata"]
    assert all(d <= hi for d in meta["fit_days"] + meta["calibration_days"])
    assert a["stages"].index("freeze") < a["stages"].index("validation_july") < a["stages"].index("oos_august")


# 16 ------------------------------------------------------------------ no holdout contamination
def test_no_holdout_contamination(env):
    a = env.a
    hc = a["leakage"]["holdout_contamination"]
    assert hc["passed"] and hc["fingerprint_unchanged"] and hc["fresh_loaded_after_freeze"]
    assert pd.Timestamp(hc["pre_freeze_max_read"]).date() <= sd.PRE_FREEZE_END
    assert a["stages"].index("oos_august") < a["stages"].index("legacy_september") < a["stages"].index(
        "fresh_october")
    assert a["labels"]["LEGACY_HOLDOUT"] == sd.LEGACY_LABEL
    md = (env.out / "s10_report.md").read_text(encoding="utf-8")
    assert sd.LEGACY_LABEL in md and "never as a clean holdout" in md


# 17 ------------------------------------------------------------------ deterministic pipeline
def test_deterministic_pipeline(env):
    assert env.a["lookahead"]["deterministic_replay"]["passed"]
    assert env.a["lookahead"]["deterministic_replay"]["fingerprint_identical"]
    data, grid, days = toy()
    for fams in (LR_BOTH, {s: sm.HGB for s in sd.SIDES}):
        m1 = sm.fit_model(data, COLS, days[:15], fams, grid, test_days=days[15:])
        m2 = sm.fit_model(data, COLS, days[:15], fams, grid, test_days=days[15:])
        assert sm.fingerprint(m1, 0.6) == sm.fingerprint(m2, 0.6)


def _wide(entries: list[tuple[pd.Timestamp, float, float, str, int]]) -> pd.DataFrame:
    rows = []
    for ts, p_ce, p_pe, outcome, minutes in entries:
        r = {"p_CE": p_ce, "p_PE": p_pe, "session_date": ts.date().isoformat()}
        for side in sd.SIDES:
            r.update({f"{k}_{side}": None for k in sm.LABEL_FIELDS})
            r.update({f"instrument_key_{side}": f"K{side}", f"entry_time_{side}": ts + STEP, f"entry_{side}": 100.0,
                      f"outcome_{side}": outcome, f"exit_time_{side}": ts + STEP + pd.Timedelta(minutes=minutes),
                      f"r_multiple_{side}": 2.0 if outcome == sd.WIN else -1.0,
                      f"pnl_{side}": 100.0 if outcome == sd.WIN else -50.0, f"holding_minutes_{side}": minutes})
        rows.append(r)
    return pd.DataFrame(rows, index=pd.DatetimeIndex([e[0] for e in entries]))


# 18 ------------------------------------------------------------------ no duplicate / overlapping trades
def test_no_duplicate_trades(env):
    t0 = at(D0, "10:00")
    w = _wide([(t0, 0.70, 0.30, sd.WIN, 20), (t0 + STEP, 0.80, 0.10, sd.WIN, 10),
               (t0 + 4 * STEP, 0.70, 0.20, sd.LOSS, 10), (t0 + 8 * STEP, 0.60, 0.58, sd.WIN, 10),
               (t0 + 9 * STEP, 0.50, 0.20, sd.WIN, 10)])
    trades, dec, opp = sm.simulate(w, 0.55)
    assert [t["signal_time"] for t in trades] == [t0, t0 + 4 * STEP] and opp == 5
    assert dec["ACTIVE_TRADE_OPEN"] == 1 and dec["MARGIN_TOO_SMALL"] == 1 and dec["BELOW_THRESHOLD"] == 1
    assert s10.duplicate_trade_check({"X": trades})["passed"]
    assert env.a["lookahead"]["duplicate_trades"]["passed"]
    assert sm.decide(0.70, 0.20, 0.60) == ("CE", None, 0.70, pytest.approx(1.1))
    assert sm.decide(0.20, 0.66, 0.60)[0] == "PE"
    assert sm.decide(0.62, 0.60, 0.60)[1] == "MARGIN_TOO_SMALL"
    assert sm.decide(float("nan"), 0.9, 0.60)[1] == "PROBABILITY_UNAVAILABLE"
    assert sm.decide(0.32, 0.10, 0.30)[1] == "EV_NOT_POSITIVE"


# 19 ------------------------------------------------------------------ expiry correctness
def test_expiry_correctness(env):
    a, m = env.a, env.a["_market"]
    data = a["_data"]
    for d, exp in set(zip(data["session_date"], data["expiry"])):
        assert exp >= d and date.fromisoformat(exp) == sd.nearest_expiry(m.calendar, date.fromisoformat(d))
    for r in a["results"].values():
        for t in r["trades"]:
            if t["outcome"] in sd.EXECUTED:
                assert pd.Timestamp(t["exit_time"]).date() <= date.fromisoformat(t["expiry"])
                assert pd.Timestamp(t["exit_time"]).time() <= sd.LAST_EXIT_END


# 20 ------------------------------------------------------------------ October isolation
def test_october_isolation(env):
    a, calls = env.a, env.calls
    first = calls[0]
    assert first["roots"] == [env.six.as_posix()] and first["max_day"] == sd.PRE_FREEZE_END and not first["frozen"]
    fresh_calls = [c for c in calls if env.fresh.as_posix() in c["roots"]]
    assert len(fresh_calls) == 1 and fresh_calls[0]["frozen"]
    pre = a["_market"]
    assert pre.spot["timestamp"].dt.date.max() <= sd.PRE_FREEZE_END and max(pre.sessions) <= sd.PRE_FREEZE_END
    assert a["_data"]["session_date"].max() <= sd.PRE_FREEZE_END.isoformat()
    assert a["fresh_quality"]["usable"] and a["fresh"]["loaded"] and a["fresh"]["rows"] > 0
    assert a["fresh"]["period"][0] >= sd.SPLITS["FRESH_HOLDOUT"][0].isoformat()
    assert a["stages"][-1] == "fresh_october" and a["stages"].count("fresh_october") == 1
    assert "FRESH_HOLDOUT" in a["results"]


def test_october_not_touched_when_oos_fails(env, tmp_path):
    def loader():
        raise AssertionError("October must not be loaded when the candidate did not pass")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s10, "gate", lambda m, n: s10.FAIL)
        b = s10.build_study(market=env.a["_market"], availability=env.a["availability"],
                            data_checks={"passed": True}, fresh_quality={"usable": True}, fresh_loader=loader,
                            benchmarks=lambda: {}, freeze_dir=tmp_path, mutation_days=1, check_sample=2)
    assert b["labels"]["OOS"] == s10.NOT_RUN and b["fresh_status"] == s10.NOT_RUN
    assert "fresh_october" not in b["stages"] and "oos_august" not in b["stages"]
    assert not {"OOS", "LEGACY_HOLDOUT", "FRESH_HOLDOUT"} & set(b["results"])
    assert b["decision"] in (s10.REJECTED, s10.INSUFF_DECISION)


# ------------------------------------------------------------------ reports, artifacts, terminal, safety
def test_reports_artifacts_and_safety(env):
    a = env.a
    for n in s10.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        t = p.read_text(encoding="utf-8")
        assert TOKEN not in t and "bearer " not in t.lower() and "AI predicts" not in t
    for n in ("ce_model.pkl", "pe_model.pkl", "preprocessor.pkl", "calibrator_ce.pkl", "calibrator_pe.pkl",
              "feature_columns.json", "thresholds.json", "model_metadata.json"):
        assert (env.out / "models" / n).is_file(), n
    cols = json.loads((env.out / "models" / "feature_columns.json").read_text(encoding="utf-8"))
    assert cols == a["feature_columns"] and not any("iv" == c or "spread" in c for c in cols)
    assert a["production_unchanged"] is True and a["live_orders"] == 0 and a["trading_mode"] == "PAPER"
    assert a["live_trading_enabled"] is False and a["cost_model"] == "COST_MODEL_UNAVAILABLE"
    assert a["token_leak"]["passed"] and clean_script.production_hashes() == env.before
    assert a["decision"] in DECISIONS and a["data_ok"]


def test_terminal_format(env):
    text = s10.terminal_text(env.a)
    lines = text.splitlines()
    assert lines[0] == "S10 ML PROBABILITY STRATEGY COMPLETE"
    for needle in ("DATA:", "historical rows:", "option rows:", "fresh October rows:", "MODEL:", "CE MODEL:",
                   "PE MODEL:", "TRAIN:", "VALIDATION JULY:", "OOS AUGUST:", "SEPTEMBER:", sd.LEGACY_LABEL,
                   "FRESH OCTOBER HOLDOUT:", "Brier:", "Log Loss:", "ROC AUC:", "PR AUC:", "Calibration:",
                   "S2 vs S9 vs S10:", "OOS accuracy:", "OOS PF:", "OOS expectancy:", "OOS P&L:", "OOS DD:",
                   "FINAL:", "70% TARGET:", "LOOKAHEAD:", "DATA INTEGRITY:", "MODEL LEAKAGE:", "FRESH HOLDOUT:",
                   "PRODUCTION CHANGED:", "LIVE ORDERS:", "FORWARD PAPER:", "FINAL DECISION:"):
        assert needle in lines or any(l.startswith(needle) for l in lines), needle
    for scope in ("CE:", "PE:", "COMBINED:"):
        assert sum(l.startswith(scope) for l in lines) == 5
    i = lines.index("PRODUCTION CHANGED:")
    assert lines[i + 1] == "NO" and lines[lines.index("LIVE ORDERS:") + 1] == "0"
    assert lines[lines.index("FINAL DECISION:") + 1] in DECISIONS
    assert "ML model estimates the historical probability" in text and "AI predicts" not in text


def test_not_run_without_required_data(tmp_path):
    make_store(tmp_path / "six")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=tmp_path / "six", fresh_root=tmp_path / "none", out_dir=tmp_path / "out",
                       secret_values=[TOKEN], with_benchmarks=False)
    assert a["ran"] is False and a["decision"] == s10.INSUFF_DECISION and a["forward_paper"] == "NOT SAFE"
    assert not a["availability"]["required"]["futures_ohlc"] and not a["fresh_quality"]["usable"]
    text = s10.terminal_text(a)
    assert "NOT_RUN" in text and text.splitlines()[0] == s10.HEADING
    assert (tmp_path / "out" / "summary.json").is_file()


def test_no_random_split_or_resampling_in_s10_code():
    for mod in (sd, sm, s10, script):
        src = inspect.getsource(mod)
        for bad in ("train_test_split", "shuffle=True", "SMOTE", "imblearn", "RandomOverSampler", "KFold(",
                    "place_order"):
            assert bad not in src, (mod.__name__, bad)
