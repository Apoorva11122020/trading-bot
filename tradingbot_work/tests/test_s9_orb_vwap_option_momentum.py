from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s9_orb_vwap_option_momentum as script
from src.backtest import shadow_historical as sh
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.options.strategy import option_stop_loss, option_target
from src.research import s9_orb_vwap_option_momentum as s9
from src.safety import live_gate
from tests.test_s4_six_month_research import make_store

IST = s9.IST
STEP = pd.Timedelta(minutes=5)
DAY = date(2026, 4, 6)
CE = SimpleNamespace(instrument_key="NSE_FO|CE100", trading_symbol="NIFTY 100 CE", strike=100.0, expiry=DAY,
                     tick_size=0.05)
PE = SimpleNamespace(instrument_key="NSE_FO|PE100", trading_symbol="NIFTY 100 PE", strike=100.0, expiry=DAY,
                     tick_size=0.05)


def _boom(*_a, **_k):
    raise AssertionError("S9 must read only the 6-month store")


def at(hhmm: str, day: date = DAY) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time.fromisoformat(hhmm)), tz=IST)


def pos(hhmm: str) -> int:
    t = time.fromisoformat(hhmm)
    return (t.hour * 60 + t.minute - 555) // 5


def frame(o, h, l, c, v, day: date = DAY) -> pd.DataFrame:
    start = at("09:15", day)
    n = 75
    return pd.DataFrame({"timestamp": [start + i * STEP for i in range(n)], "open": float(o), "high": float(h),
                         "low": float(l), "close": float(c), "volume": float(v)})


def put(df: pd.DataFrame, hhmm: str, o, h, l, c, v=None) -> pd.DataFrame:
    i = pos(hhmm)
    df.loc[i, ["open", "high", "low", "close"]] = [o, h, l, c]
    if v is not None:
        df.loc[i, "volume"] = v
    return df


def flat_spot() -> pd.DataFrame:
    df = frame(100.0, 100.2, 99.8, 100.0, 0.0)
    for hhmm in ("09:15", "09:20", "09:25"):
        put(df, hhmm, 100.0, 100.5, 99.5, 100.0)
    return df


BUY_BAR = (100.4, 100.85, 100.35, 100.8)


def ce_premium() -> pd.DataFrame:
    df = frame(100.0, 101.0, 99.0, 100.0, 1000.0)
    put(df, "11:15", 100.0, 103.5, 99.8, 103.0, 2000.0)
    for i in range(pos("11:20"), 75):
        df.loc[i, ["open", "high", "low", "close"]] = [103.0, 104.0, 102.0, 103.0]
    return df


def leg(prem: pd.DataFrame, hhmm: str, side: str = "BUY", cfg: s9.Config = s9.BASE, contract=CE) -> dict:
    p, t, _ = s9.premium_frame(contract, lambda _c: prem, {})
    return s9.option_leg(p, t, at(hhmm), side, cfg, 0.05)


# ------------------------------------------------------------------ NIFTY features
def test_opening_range_uses_only_0915_to_0925_candles():
    spot = flat_spot()
    f = s9.nifty_features(spot)
    assert f.loc[at("09:30"), "or_high"] == 100.5 and f.loc[at("09:30"), "or_low"] == 99.5
    assert not f.loc[at("09:25"), "or_complete"] and math.isnan(f.loc[at("09:25"), "or_high"])
    later = put(flat_spot(), "09:30", 100.0, 200.0, 1.0, 100.0)
    assert s9.nifty_features(later).loc[at("10:00"), "or_high"] == 100.5
    inside = put(flat_spot(), "09:20", 100.0, 101.0, 99.0, 100.0)
    assert s9.nifty_features(inside).loc[at("10:00"), "or_high"] == 101.0
    missing = flat_spot().drop(index=pos("09:20")).reset_index(drop=True)
    g = s9.nifty_features(missing)
    assert not g.loc[at("10:00"), "or_complete"]
    ok, why = s9.underlying_check({**g.loc[at("10:00")].to_dict(), "timestamp": at("10:00")}, "BUY", s9.BASE)
    assert not ok and why == "OR_INCOMPLETE"


def test_vwap_twap_and_volume_ratio_are_causal():
    rng = np.random.default_rng(3)
    spot = flat_spot()
    spot["close"] = 100 + rng.normal(0, 0.3, 75).cumsum()
    spot["high"], spot["low"] = spot["close"] + 0.4, spot["close"] - 0.4
    spot["volume"] = rng.integers(100, 1000, 75).astype(float)
    full = s9.nifty_features(spot)
    for hhmm in ("09:40", "11:00", "13:55"):
        ts = at(hhmm)
        trunc = s9.nifty_features(spot.iloc[:pos(hhmm) + 1])
        for c in s9.FEATURE_COLS:
            a, b = full.loc[ts, c], trunc.loc[ts, c]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), (hhmm, c)
    v = spot["volume"].to_numpy()
    assert full["volume_ratio_index"].iloc[20] == pytest.approx(v[20] / v[:20].mean())
    assert math.isnan(full["volume_ratio_index"].iloc[19])
    tp = (spot["high"] + spot["low"] + spot["close"]) / 3
    assert full["twap"].iloc[10] == pytest.approx(tp.iloc[:11].mean())
    assert full["vwap"].iloc[10] == pytest.approx((tp.iloc[:11] * v[:11]).sum() / v[:11].sum())
    zero = s9.nifty_features(flat_spot())
    assert zero["vwap"].isna().all() and zero["volume_ratio_index"].isna().all() and zero["twap"].notna().all()


def _row(o, h, l, c, **kw) -> dict:
    base = {"timestamp": at("11:15"), "open": o, "high": h, "low": l, "close": c, "or_complete": True,
            "or_high": 100.5, "or_low": 99.5, "twap": 100.0, "vwap": float("nan"), "volume_ratio_index": float("nan")}
    return {**base, **kw}


def test_underlying_signal_rules_buy_and_sell():
    assert s9.underlying_check(_row(*BUY_BAR), "BUY", s9.BASE) == (True, None)
    assert s9.underlying_check(_row(*BUY_BAR), "SELL", s9.BASE) == (False, "NO_BREAKOUT")
    small = 100.5 * 1.0004
    assert s9.underlying_check(_row(100.3, small + 0.01, 100.25, small), "BUY", s9.BASE)[1] == "BREAKOUT_TOO_SMALL"
    assert s9.underlying_check(_row(100.4, 100.85, 100.35, 100.8), "BUY", s9.Config("B", breakout_pct=0.10)) \
        == (True, None)
    assert s9.underlying_check(_row(100.7, 101.6, 100.2, 100.9), "BUY", s9.BASE)[1] == "WEAK_BODY"
    assert s9.underlying_check(_row(100.2, 101.6, 100.0, 101.0), "BUY", s9.BASE)[1] == "CLOSE_NOT_AT_EXTREME"
    assert s9.underlying_check(_row(100.9, 101.0, 100.6, 100.8), "BUY", s9.BASE)[1] == "WRONG_CANDLE_DIRECTION"
    assert s9.underlying_check(_row(*BUY_BAR, twap=101.0), "BUY", s9.BASE)[1] == "WRONG_SIDE_OF_VWAP"
    assert s9.underlying_check(_row(100.8, 100.8, 100.8, 100.8), "BUY", s9.BASE)[1] in ("WRONG_CANDLE_DIRECTION",
                                                                                         "ZERO_RANGE")
    sell = _row(99.6, 99.65, 99.15, 99.2)
    assert s9.underlying_check(sell, "SELL", s9.BASE) == (True, None)
    assert s9.underlying_check(sell, "BUY", s9.BASE) == (False, "NO_BREAKOUT")
    assert s9.underlying_check(_row(*BUY_BAR), "BUY", s9.EXACT_BASE)[1] == "VWAP_UNAVAILABLE"
    exact = _row(*BUY_BAR, vwap=100.0, volume_ratio_index=1.1)
    assert s9.underlying_check(exact, "BUY", s9.EXACT_BASE)[1] == "LOW_VOLUME"
    assert s9.underlying_check(dict(exact, volume_ratio_index=float("nan")), "BUY", s9.EXACT_BASE)[1] \
        == "VOLUME_UNAVAILABLE"
    assert s9.underlying_check(dict(exact, volume_ratio_index=1.3), "BUY", s9.EXACT_BASE) == (True, None)


def test_entry_window_0930_to_1400():
    assert s9.in_window(at("09:30")) and s9.in_window(at("13:55"))
    assert not s9.in_window(at("09:25")) and not s9.in_window(at("14:00"))


# ------------------------------------------------------------------ option leg
def test_option_confirmation_entry_atr_and_levels():
    prem = ce_premium()
    out = leg(prem, "11:15")
    assert out["fired"] and out["outcome"] == s9.READY
    assert out["entry"] == 103.0 and out["entry_time"] == at("11:20").isoformat()
    assert out["option_volume_ratio"] == pytest.approx(2.0)
    a = s9.atr(prem.iloc[:pos("11:15") + 1], 14).iloc[-1]
    assert out["atr"] == pytest.approx(a)
    stop = option_stop_loss(103.0, stop_mode="premium_atr", multiplier=1.5, premium_atr=a, tick_size=0.05)
    assert out["stop_loss"] == stop and out["target"] == option_target(103.0, stop, 2.0, 0.05)
    warped = prem.copy()
    later = pos("11:20") + 1
    warped.loc[later:, ["open", "high", "low", "close", "volume"]] *= 3.0
    again = leg(warped, "11:15")
    assert {k: again[k] for k in ("entry", "atr", "stop_loss", "target", "option_volume_ratio")} == \
        {k: out[k] for k in ("entry", "atr", "stop_loss", "target", "option_volume_ratio")}


def test_option_leg_missing_data_and_weak_volume():
    prem = ce_premium()
    assert leg(prem.drop(index=pos("11:15")), "11:15")["reason"] == "OPTION_SIGNAL_CANDLE_MISSING"
    gone = leg(prem.drop(index=pos("11:20")), "11:15")
    assert gone["fired"] and gone["outcome"] == s9.UNRESOLVED and gone["reason"] == "NO_OPTION_ENTRY_CANDLE"
    early = put(ce_premium(), "10:00", 100.0, 103.5, 99.8, 103.0, 5000.0)
    assert leg(early, "10:00")["reason"] == "OPTION_VOLUME_HISTORY_UNAVAILABLE"
    low = put(ce_premium(), "11:15", 100.0, 103.5, 99.8, 103.0, 1100.0)
    assert leg(low, "11:15")["reason"] == "LOW_VOLUME"
    assert leg(low, "11:15", cfg=s9.EXACT_BASE)["fired"]
    assert leg(ce_premium(), "11:15", cfg=s9.Config("D", volume_ratio=1.5))["fired"]
    assert leg(ce_premium(), "11:15", cfg=s9.Config("D", volume_ratio=2.5))["reason"] == "LOW_VOLUME"


def test_buy_ce_and_literal_sell_pe_confirmation():
    flat = put(ce_premium(), "11:15", 100.0, 100.5, 99.0, 99.5, 2000.0)
    assert leg(flat, "11:15")["reason"] == "OPTION_NOT_CONFIRMED"
    pe_down = put(frame(100.0, 101.0, 99.0, 100.0, 1000.0), "11:15", 100.0, 100.2, 96.5, 97.0, 2000.0)
    assert leg(pe_down, "11:15", "SELL", contract=PE)["fired"]
    pe_up = put(frame(100.0, 101.0, 99.0, 100.0, 1000.0), "11:15", 100.0, 103.5, 99.8, 103.0, 2000.0)
    assert leg(pe_up, "11:15", "SELL", contract=PE)["reason"] == "OPTION_NOT_CONFIRMED"


# ------------------------------------------------------------------ exits
def _sim(prem: pd.DataFrame) -> dict:
    p, t, _ = s9.premium_frame(CE, lambda _c: prem, {})
    return s9.simulate(p, t, at("11:20"), 103.0, 100.0, 109.0)


def test_exit_target_stop_ambiguous_gap_and_1430():
    win = _sim(put(ce_premium(), "11:40", 103.0, 110.0, 102.0, 108.0))
    assert win["outcome"] == s9.WIN and win["exit_price"] == 109.0 and win["r_multiple"] == 2.0
    assert win["exit_time"] == at("11:45").isoformat()
    loss = _sim(put(ce_premium(), "11:40", 103.0, 104.0, 99.5, 101.0))
    assert loss["outcome"] == s9.LOSS and loss["exit_price"] == 100.0 and loss["r_multiple"] == -1.0
    amb = _sim(put(ce_premium(), "11:40", 103.0, 110.0, 99.0, 104.0))
    assert amb["outcome"] == s9.AMBIGUOUS and amb["r_multiple"] == -1.0 and amb["pnl"] == -3.0 * s9.LOT
    gap = _sim(put(ce_premium(), "11:40", 98.0, 98.5, 97.0, 98.0))
    assert gap["outcome"] == s9.LOSS and gap["exit_reason"] == "STOP_LOSS_GAP" and gap["exit_price"] == 98.0
    eod = put(put(ce_premium(), "14:25", 103.0, 104.0, 102.0, 104.0), "14:30", 103.0, 200.0, 1.0, 150.0)
    out = _sim(eod)
    assert out["exit_reason"] == "EOD_EXIT_1430" and out["exit_price"] == 104.0
    assert out["exit_time"] == at("14:30").isoformat() and out["outcome"] == s9.WIN
    scratch = _sim(ce_premium())
    assert scratch["outcome"] == s9.SCRATCH and scratch["r_multiple"] == 0.0


# ------------------------------------------------------------------ chronological replay
def _replay_inputs():
    spot = put(flat_spot(), "11:15", *BUY_BAR)
    put(spot, "11:20", 100.8, 101.25, 100.75, 101.2)
    put(spot, "11:45", *BUY_BAR)
    prem = put(ce_premium(), "11:45", 103.0, 106.5, 102.8, 106.0, 3000.0)
    return spot, prem


def test_replay_one_active_trade_and_no_repeated_breakout():
    spot, prem = _replay_inputs()
    recs, funnel, _ = s9.replay(s9.nifty_features(spot), [DAY], lambda sig: CE if sig.direction == "BUY" else PE,
                                lambda c: prem if c is CE else None, s9.BASE)
    assert [r["signal_time"] for r in recs] == [at("11:15").isoformat(), at("11:45").isoformat()]
    first, second = recs
    assert first["outcome"] == s9.SCRATCH and first["state"] == sh.EVALUATED
    assert first["entry_time"] == at("11:20").isoformat() and first["exit_time"] == at("14:30").isoformat()
    assert first["contract"] == "NIFTY 100 CE" and first["vwap"] == pytest.approx(100.0, abs=0.2)
    assert second["fired"] and second["outcome"] == s9.NO_SIGNAL and second["reason"] == "ACTIVE_TRADE_OPEN"
    assert funnel["BUY:REPEATED_BREAKOUT"] == 1
    assert s9.chronology_check(recs)["passed"]


def test_missing_contract_is_no_signal_never_substituted():
    spot, prem = _replay_inputs()

    def resolve(_sig):
        raise s9.OptionResolutionError("CONTRACT_NOT_FOUND", "missing")
    recs, _, _ = s9.replay(s9.nifty_features(spot), [DAY], resolve, lambda c: prem, s9.BASE)
    assert recs and all(r["outcome"] == s9.NO_SIGNAL and r["reason"].startswith("CONTRACT_UNAVAILABLE")
                        and r["instrument_key"] is None for r in recs)
    recs, _, _ = s9.replay(s9.nifty_features(spot), [DAY], lambda sig: CE, lambda c: None, s9.BASE)
    assert all(r["reason"] == "OPTION_DATA_MISSING" and not r["fired"] for r in recs)


# ------------------------------------------------------------------ metrics, labels, selection
def mk(day: date, outcome: str, r: float, side: str = "BUY", hhmm: str = "10:00") -> dict:
    ts = at(hhmm, day)
    executed = outcome in s9.EXECUTED
    return {"signal_time": ts.isoformat(), "session_date": day.isoformat(), "direction": side, "outcome": outcome,
            "r_multiple": r if executed else None, "pnl": r * 100.0 if executed else None, "fired": True,
            "reason": None, "holding_minutes": 30.0 if executed else None,
            "entry_time": (ts + STEP).isoformat(), "state": sh.EVALUATED if executed else None}


def test_accuracy_excludes_ambiguous_unresolved_and_scratch():
    d = DAY
    m = s9.metrics([mk(d, s9.WIN, 2.0), mk(d, s9.LOSS, -1.0, hhmm="10:30"), mk(d, s9.AMBIGUOUS, -1.0, hhmm="11:00"),
                    mk(d, s9.SCRATCH, 0.0, hhmm="11:30"), mk(d, s9.UNRESOLVED, 0.0, hhmm="12:00")])
    assert (m["wins"], m["losses"], m["resolved"], m["accuracy"]) == (1, 1, 2, 0.5)
    assert m["executable_trades"] == 4 and m["total_signals"] == 5 and m["unresolved"] == 1
    assert m["expectancy_r"] == 0.0 and m["max_losing_streak"] == 2 and "1 wins / 2 resolved" in m["accuracy_text"]
    assert m["profit_factor"] == pytest.approx(1.0) and m["trade_retention"] == 0.8


def weekdays(lo: date, hi: date) -> list[date]:
    out, d = [], lo
    while d <= hi:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def good_records(lo=date(2026, 4, 1), hi=date(2026, 9, 30)) -> list[dict]:
    return [mk(d, s9.WIN if i % 2 == 0 else s9.LOSS, 2.0 if i % 2 == 0 else -1.0) for i, d in
            enumerate(weekdays(lo, hi))]


def test_status_labels():
    m = s9.metrics(good_records(date(2026, 4, 1), date(2026, 6, 30)))
    assert s9.train_label(m)["label"] == s9.ELIGIBLE
    assert s9.train_label(s9.metrics(good_records(date(2026, 4, 1), date(2026, 4, 20))))["label"] == s9.INSUFFICIENT
    bad = s9.metrics([mk(d, s9.LOSS, -1.0) for d in weekdays(date(2026, 4, 1), date(2026, 6, 30))])
    assert s9.train_label(bad)["label"] == s9.REJECTED and s9.validation_label(bad) == s9.REJECTED
    assert s9.validation_label(s9.metrics(good_records(date(2026, 7, 1), date(2026, 7, 31)))) == s9.PASS
    assert s9.oos_label(s9.metrics(good_records(date(2026, 8, 1), date(2026, 8, 31)))) == s9.PROMISING
    assert s9.oos_label(s9.metrics(good_records(date(2026, 8, 1), date(2026, 8, 10)))) == s9.INSUFFICIENT
    el = {"label": s9.ELIGIBLE}
    assert s9.s9_status(el, s9.PASS, s9.PROMISING, True) == s9.PROMISING
    assert s9.s9_status(el, s9.PASS, s9.PROMISING, False) == s9.REJECTED
    assert s9.s9_status(el, s9.DEGRADED, s9.PROMISING, True) == s9.REJECTED
    assert s9.s9_status({"label": s9.INSUFFICIENT}, s9.PASS, s9.PROMISING, True) == s9.INSUFFICIENT
    assert s9.s9_status(el, s9.PASS, s9.INSUFFICIENT, True) == s9.INSUFFICIENT


def test_selection_uses_only_train_and_validation():
    by = {"BASE": good_records(), "B": [dict(r, outcome=s9.LOSS, r_multiple=-1.0, pnl=-100.0) for r in good_records()],
          "D": good_records(), "F": good_records()[:10]}
    with pytest.raises(ValueError):
        s9.select(by)
    sel = s9.select(s9.pre(by))
    assert sel["base_passes_train_and_validation"] and sel["validated_variant"] == "D"
    assert sel["train"]["B"]["label"] == s9.REJECTED and sel["train"]["F"]["label"] == s9.INSUFFICIENT
    iso = s9.isolation(by, sel)
    assert iso["passed"], iso["checks"]


def test_variants_predefined_and_limited():
    assert len(s9.VARIANTS) == 6 and set(s9.VARIANTS) == set("ABCDEF")
    assert all(s9.VARIANTS[k][1] is s9.BASE for k in "ACE")
    assert [c.name for c in s9.DISTINCT] == ["BASE", "B", "D", "F"]
    assert s9.BASE.mode == s9.PROXY and s9.EXACT_BASE.mode == s9.EXACT


# ------------------------------------------------------------------ end-to-end on the synthetic 6-month store
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s9")
    make_store(base / "upstox_6m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=base / "upstox_6m", out_dir=base / "out", mutation_days=2, check_sample=5)
    return SimpleNamespace(a=a, base=base, before=before)


def test_all_anti_lookahead_checks_pass(env):
    for k, v in env.a["lookahead"].items():
        assert v["passed"], (k, v)
    assert env.a["lookahead_ok"] and env.a["isolation"]["passed"], env.a["isolation"]["checks"]


def test_chronology_entries_exits_and_contracts(env):
    for recs in env.a["variant_records"].values():
        ts = [s9.dra._ts(r["signal_time"]) for r in recs]
        assert ts == sorted(ts)
        for r in recs:
            if r["outcome"] in s9.EXECUTED:
                et, xt = s9.dra._ts(r["entry_time"]), s9.dra._ts(r["exit_time"])
                assert et == s9.dra._ts(r["signal_time"]) + STEP and et.time() <= s9.LAST_ENTRY
                assert et < xt and xt.time() <= s9.SQUARE_OFF
                assert r["option_type"] == ("CE" if r["direction"] == "BUY" else "PE")
    dc = env.a["data_checks"]
    assert all(d["passed"] for d in dc["duplicates"].values()) and dc["timestamps"]["passed"]
    assert dc["contract_validity"]["passed"] and env.a["data_ok"]


def test_exact_mode_reports_index_volume_limitation(env):
    assert env.a["exact_signals"] == 0
    assert all(k.split(":")[1] in ("OR_INCOMPLETE", "NO_BREAKOUT", "VWAP_UNAVAILABLE") for k in env.a["exact_funnel"])


def test_no_production_change_and_paper_only(env):
    assert clean_script.production_hashes() == env.before
    assert env.a["production_unchanged"] is True and env.a["live_orders"] == 0
    assert env.a["trading_mode"] == "PAPER" and env.a["live_trading_enabled"] is False


def test_reports_and_terminal_output(env):
    out = env.base / "out"
    for n in s9.REPORT_FILES:
        assert (out / n).is_file(), n
    json.loads((out / "summary.json").read_text(encoding="utf-8"))
    json.loads((out / "data_quality.json").read_text(encoding="utf-8"))
    with (out / "combined_results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["label"] == s9.HOLDOUT_LABEL for r in rows if r["split"] == "FINAL HOLDOUT SEPTEMBER")
    with (out / "variant_results.csv").open(encoding="utf-8") as fh:
        assert set("ABCDEF") <= {r["variant"] for r in csv.DictReader(fh)}
    text = s9.terminal_text(env.a)
    for label in ("S9 ORB + VWAP + OPTION MOMENTUM COMPLETE", "BASE S9", "TRAIN:", "VALIDATION:", "OOS AUGUST:",
                  "FINAL HOLDOUT SEPTEMBER:", "BUY: Trades", "SELL: Trades", "COMBINED: Trades",
                  "DIRECT S2 vs S9:", "S2 OOS:", "S9 OOS:", "S9 STATUS:", "70% TARGET:", "LOOKAHEAD:\nPASS",
                  "DATA INTEGRITY:\nPASS", "PRODUCTION CHANGED:\nNO", "LIVE ORDERS:\n0",
                  "FORWARD PAPER VALIDATION:", "not live accuracy"):
        assert label in text, label
    assert env.a["status"] in (s9.PROMISING, s9.REJECTED, s9.INSUFFICIENT)
    for n in s9.REPORT_FILES:
        t = (out / n).read_text(encoding="utf-8").lower()
        assert "access_token" not in t and "bearer " not in t


def test_paper_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))
