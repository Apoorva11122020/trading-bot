from __future__ import annotations

import csv
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s8_architecture_audit as script
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.research import s8_architecture_audit as s8
from src.safety import live_gate
from tests.test_s4_six_month_research import make_store

IST = s8.IST
STEP = pd.Timedelta(minutes=5)
DAY = date(2026, 4, 6)
REGIMES = ("TREND_UP", "RANGE", "CHOPPY")


def _boom(*_a, **_k):
    raise AssertionError("S8 must read only the 6-month store")


def day_index(day: date = DAY) -> pd.DatetimeIndex:
    start = pd.Timestamp(datetime.combine(day, time(9, 15)), tz=IST)
    return pd.DatetimeIndex([start + i * STEP for i in range(75)])


def bars_from(close: np.ndarray, spread: float = 0.5, atr: float = 2.0) -> pd.DataFrame:
    idx = day_index()
    return pd.DataFrame({"open": close, "high": close + spread, "low": close - spread, "close": close,
                         "atr14": atr}, index=idx)


def frame_from(close, high=None, low=None, open_=None, **cols) -> pd.DataFrame:
    idx = day_index()
    close = np.asarray(close, float)
    f = pd.DataFrame({"open": close if open_ is None else open_, "high": close if high is None else high,
                      "low": close if low is None else low, "close": close}, index=idx)
    for c in ("ema20", "ema50", "atr14", "ema20_slope_atr", "rsi14", "ref", "hh6", "ll6", "orb_high", "orb_low",
              "orb30_high", "orb30_low", "pdh", "pdl"):
        f[c] = cols.get(c, np.nan)
    return f


def row(day: date, t: str, direction: str, win: bool, regime: str = "TREND_UP", window: str = "10:00-11:00") -> dict:
    st = f"{day.isoformat()}T{t}:00+05:30"
    return {"session_date": day.isoformat(), "signal_time": st, "direction": direction, "state": sh.EVALUATED,
            "decision": "ACCEPTED", "outcome": "WIN" if win else "LOSS", "pnl": 200.0 if win else -100.0,
            "r_multiple": 2.0 if win else -1.0, "entry_time": (pd.Timestamp(st) + STEP).isoformat(),
            "instrument_key": f"K{day}{t}{direction}", "breakout_level": 100.0,
            "features": {"time_window": window, "vol_regime": "NORMAL_VOLATILITY"}, "s6": {"market_regime": regime}}


def weekdays(lo: date = date(2026, 4, 1), hi: date = date(2026, 9, 30)):
    d = lo
    while d <= hi:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def alt_synth() -> dict[str, list[dict]]:
    """S8-TC1 BUY wins 2 of 3 in every month and regime; S8-VP1 SELL always loses."""
    good, bad, j = [], [], 0
    for d in weekdays():
        for t in ("10:00", "11:30"):
            good.append(row(d, t, "BUY", j % 3 != 0, REGIMES[(j // 3) % 3]))
            bad.append(row(d, t, "SELL", False, REGIMES[j % 3]))
            j += 1
    return {"S8-TC1": good, "S8-VP1": bad}


# ------------------------------------------------------------------ Part A / B
def test_underlying_forward_returns_mfe_mae():
    c = np.full(75, 100.0)
    c[11], c[12], c[13], c[15] = 101.0, 102.0, 99.0, 103.0
    bx = s8.BarIndex(bars_from(c))
    st = day_index()[10].isoformat()
    buy = s8.underlying_item({"signal_time": st, "direction": "BUY", "session_date": DAY.isoformat()}, bx)
    assert (buy["ret1"], buy["ret2"], buy["ret3"], buy["ret5"]) == (1.0, 2.0, -1.0, 3.0)
    assert buy["mfe"] == pytest.approx(3.5) and buy["mae"] == pytest.approx(1.5)
    assert buy["ret5_atr"] == pytest.approx(1.5)
    sell = s8.underlying_item({"signal_time": st, "direction": "SELL", "session_date": DAY.isoformat()}, bx)
    assert sell["ret5"] == -3.0 and sell["mfe"] == pytest.approx(1.5) and sell["mae"] == pytest.approx(3.5)
    last = s8.underlying_item({"signal_time": day_index()[74].isoformat(), "direction": "BUY",
                               "session_date": DAY.isoformat()}, bx)
    assert last["ret1"] is None and last["mfe"] is None


def _und_items(pattern_even, pattern_odd, direction="BUY"):
    items, k = [], 0
    for split, days in (("TRAIN", list(weekdays(date(2026, 4, 1), date(2026, 6, 30)))),
                        ("VALIDATION", list(weekdays(date(2026, 7, 1), date(2026, 7, 31))))):
        for n, d in enumerate(days):
            for v in (pattern_even if n % 2 == 0 else pattern_odd):
                it = {"key": (str(k), direction), "direction": direction, "split": split, "day": d.isoformat(),
                      "mfe": abs(v) + 1, "mae": 1.0, "mfe_atr": abs(v) + 1, "mae_atr": 1.0}
                for h in range(1, 6):
                    it[f"ret{h}"], it[f"ret{h}_atr"] = float(v), float(v)
                items.append(it)
                k += 1
    return items


def test_underlying_verdict_edge_and_no_edge():
    edge = s8.underlying_verdict(_und_items((2, 2, -1), (3, 2, -1)), {})
    assert edge["verdict"] == s8.UNDERLYING_EDGE and all(edge["checks"].values())
    none = s8.underlying_verdict(_und_items((1, -1), (-1, 1)), {})
    assert none["verdict"] == s8.NO_EDGE
    beaten = s8.underlying_verdict(_und_items((2, 2, -1), (3, 2, -1)),
                                   {"TRAIN": {"BUY": 0.70}, "VALIDATION": {"BUY": 0.70}})
    assert beaten["verdict"] != s8.UNDERLYING_EDGE


def test_option_translation_loss_when_premium_does_not_follow():
    items = []
    for n, d in enumerate(weekdays(date(2026, 4, 1), date(2026, 7, 31))):
        it = {"key": (str(n), "BUY"), "direction": "BUY", "split": "TRAIN" if d.month < 7 else "VALIDATION",
              "day": d.isoformat(), "opt_mfe_r": 0.2, "opt_mae_r": 1.0, "lag": None, "premium_per_point": None}
        for k in range(1, 6):
            it[f"und{k}"], it[f"opt{k}"], it[f"opt{k}_r"] = 5.0, -2.0, -0.2
        items.append(it)
    t = s8.translation_verdict(items, s8.UNDERLYING_EDGE)
    assert t["label"] == s8.T_LOSS and t["option_translation_loss"] and t["underlying_edge"]
    assert t["train_validation"]["option_response_rate"] == 0.0


# ------------------------------------------------------------------ Part C - entry methods (causal decision candle)
def test_entry_methods_decision_candle():
    c = np.full(75, 105.0)
    bars = bars_from(c, spread=1.0)
    ts = day_index()
    r = {"signal_time": ts[10].isoformat(), "direction": "BUY", "breakout_level": 100.0}
    late = time(15, 20)
    assert s8.entry_ref("NEXT_CONFIRM", r, s8.BarIndex(bars), late) == (ts[11], None)
    assert s8.entry_ref("BREAKOUT_CONTINUATION", r, s8.BarIndex(bars), late) == (None, "NO_CONFIRMATION")
    assert s8.entry_ref("PULLBACK", r, s8.BarIndex(bars), late) == (None, "NO_PULLBACK")
    b2 = bars.copy()
    b2.iloc[11, b2.columns.get_loc("close")] = 107.0
    b2.iloc[12, b2.columns.get_loc("low")] = 99.5
    b2.iloc[12, b2.columns.get_loc("close")] = 101.0
    assert s8.entry_ref("BREAKOUT_CONTINUATION", r, s8.BarIndex(b2), late) == (ts[11], None)
    assert s8.entry_ref("PULLBACK", r, s8.BarIndex(b2), late) == (ts[12], None)
    assert s8.entry_ref("NEXT_CONFIRM", r, s8.BarIndex(bars), time(10, 10))[1] == "ENTRY_AFTER_CUTOFF"
    end = dict(r, signal_time=ts[74].isoformat())
    assert s8.entry_ref("NEXT_CONFIRM", end, s8.BarIndex(bars), late)[1] == "SESSION_END"
    sell = {"signal_time": ts[10].isoformat(), "direction": "SELL", "breakout_level": 110.0}
    assert s8.entry_ref("NEXT_CONFIRM", sell, s8.BarIndex(bars), late) == (ts[11], None)


def test_duplicate_entries_are_one_trade():
    a = row(DAY, "10:00", "BUY", True)
    b = dict(row(DAY, "10:05", "BUY", True), instrument_key=a["instrument_key"], entry_time=a["entry_time"])
    out = s8.dedupe([b, a])
    assert [x["state"] for x in out] == [sh.EVALUATED, sh.SKIPPED] and out[1]["skip_reason"] == "DUPLICATE_ENTRY"


def test_exit_engine_check_and_choice_ignores_oos():
    base = [row(d, "10:00", "BUY", i % 3 == 0) for i, d in enumerate(weekdays())]
    assert s8.exit_engine_check(base, [dict(r) for r in base])["passed"]
    assert not s8.exit_engine_check(base, [dict(r, pnl=1.0) for r in base])["passed"]
    better_later = [dict(r, outcome="WIN", pnl=200.0, r_multiple=2.0) if r["session_date"] > "2026-07-31" else r
                    for r in base]
    tables = s8.variant_tables({"NEXT_OPEN": base, "PULLBACK": better_later})
    assert s8.best_variant(tables, "NEXT_OPEN", "BUY")["best"] == "NEXT_OPEN"
    better = [dict(r, outcome="WIN", pnl=200.0, r_multiple=2.0) for r in base]
    tables = s8.variant_tables({"NEXT_OPEN": base, "PULLBACK": better})
    assert s8.best_variant(tables, "NEXT_OPEN", "BUY")["best"] == "PULLBACK"


# ------------------------------------------------------------------ Part E - alternative signal rules
def _br_frame(sell: bool = False, fail: bool = False) -> pd.DataFrame:
    c = np.full(75, 101.2)
    h, lo = c + 0.2, c - 0.2
    c[:4], h[:4], lo[:4] = 99.0, 99.2, 98.8
    c[4], h[4], lo[4] = 101.0, 101.1, 100.4
    c[5], h[5], lo[5] = 100.5, 100.8, 99.8
    c[6], h[6], lo[6] = (99.5, 100.3, 99.4) if fail else (100.7, 100.75, 100.2)
    level = np.where(np.arange(75) >= 3, 100.0, np.nan)
    if not sell:
        return frame_from(c, h, lo, orb_high=level)
    return frame_from(200 - c, 200 - lo, 200 - h, orb_low=level)


def test_breakout_retest_rule_mirrored_and_reset():
    spec = next(s for s in s8.SPECS if s.sid == "S8-BR1")
    ts = day_index()
    first, cutoff = time(9, 30), time(15, 0)
    assert [x[0] for x in s8.generate(_br_frame(), spec, "BUY", first, cutoff)] == [ts[7]]
    assert [x[0] for x in s8.generate(_br_frame(sell=True), spec, "SELL", first, cutoff)] == [ts[7]]
    assert s8.generate(_br_frame(), spec, "SELL", first, cutoff) == []
    assert s8.generate(_br_frame(fail=True), spec, "BUY", first, cutoff) == []


def test_alternative_signal_uses_only_completed_candles():
    spec = next(s for s in s8.SPECS if s.sid == "S8-BR1")
    f = _br_frame()
    first, cutoff = time(9, 30), time(15, 0)
    full = s8.generate(f, spec, "BUY", first, cutoff)
    assert s8.generate(f.iloc[:8], spec, "BUY", first, cutoff) == full
    assert s8.generate(f.iloc[:7], spec, "BUY", first, cutoff) == []
    warped = f.copy()
    warped.iloc[8:, :4] = warped.iloc[8:, :4] * 3.0
    assert s8.generate(warped, spec, "BUY", first, cutoff)[:1] == full


def test_cooldown_and_daily_cap():
    c = 100.0 + np.arange(75)
    f = frame_from(c, c + 0.1, c - 0.6, c - 0.5, rsi14=20.0)
    spec = next(s for s in s8.SPECS if s.sid == "S8-MR2")
    got = [x[0] for x in s8.generate(f, spec, "BUY", time(9, 30), time(15, 0))]
    ts = day_index()
    assert got == [ts[4], ts[4 + s8.COOLDOWN_BARS], ts[4 + 2 * s8.COOLDOWN_BARS]]
    assert len(got) == s8.MAX_SIGNALS_PER_DAY


def test_families_predeclared_limited_and_sides_separate():
    assert len(s8.FAMILIES) <= s8.MAX_FAMILIES
    assert all(sum(s.family == f for s in s8.SPECS) <= s8.MAX_VARIANTS for f in s8.FAMILIES)
    assert all(s.rationale and s.rule for s in s8.SPECS)
    for d in s8.DIRECTIONS:
        pool = [c for c in s8.ALTS if c.direction == d]
        assert len(pool) + len(s8.CONFIRMS) <= s8.MAX_CANDIDATES_PER_SIDE
    assert len({c.cid for c in s8.ALTS}) == len(s8.ALTS)


def test_premium_confirmation_missing_data_is_no_signal():
    rows = {"S8-TC1": [row(DAY, "10:00", "BUY", True)]}
    c = s8.Alt("S8-TC1-B+PREM_MOMENTUM", "S8-TC1", "BUY", "PREM_MOMENTUM")
    assert s8.cand_rows(c, rows) == []
    rows["S8-TC1"][0]["opt_confirm"] = True
    assert len(s8.cand_rows(c, rows)) == 1


# ------------------------------------------------------------------ Part H - clean selection
@pytest.fixture(scope="module")
def synth():
    rows = alt_synth()
    return SimpleNamespace(rows=rows, sel=s8.select(s8.pre_rows(rows)))


def test_selection_train_ranked_july_validated(synth):
    assert s8.ids(synth.sel) == {"BUY": "S8-TC1-B", "SELL": None}
    assert synth.sel["validation"]["S8-TC1-B"]["verdict"] == s8.VALIDATED
    assert not synth.sel["train"]["S8-VP1-S"]["eligible"]
    for t in synth.sel["train"].values():
        if t["metrics"]["total_trades"] < s8.MIN_TRAIN_TRADES:
            assert not t["eligible"]


def test_oos_and_holdout_isolation_and_determinism(synth):
    base = [row(d, "10:00", "BUY", i % 3 == 0) for i, d in enumerate(weekdays())]
    iso = s8.isolation(synth.rows, synth.sel, {"NEXT_OPEN": base}, {"CURRENT_S2": base})
    assert iso["passed"], iso["checks"]
    with pytest.raises(ValueError):
        s8.select(synth.rows)
    worse = {k: [dict(r, outcome="LOSS", pnl=-100.0, r_multiple=-1.0) if r["session_date"] > "2026-07-31" else r
                 for r in v] for k, v in synth.rows.items()}
    assert s8.ids(s8.select(s8.pre_rows(worse))) == s8.ids(synth.sel)
    assert s8.fingerprint(s8.select(s8.pre_rows(synth.rows))) == s8.fingerprint(synth.sel)


def test_one_month_profit_fails_quality(synth):
    only_may = {k: [dict(r, outcome="LOSS", pnl=-100.0, r_multiple=-1.0)
                    if r["session_date"][:7] in ("2026-04", "2026-06") else r for r in v]
                for k, v in synth.rows.items()}
    c = next(c for c in s8.ALTS if c.cid == "S8-TC1-B")
    tr = s8.train_eval(c, only_may)
    assert not tr["eligible"] and "monthly_stability" in tr["failed"]


# ------------------------------------------------------------------ Part L - decision rules
def test_final_decision_rules():
    bad = {"profit_factor": 0.8, "average_r": -0.1}
    s2 = {k: {d: bad for d in s8.DIRECTIONS} for k in s8.PRE_OOS}
    no_exit = {d: {"improved": False, "positive_on_train_and_validation": False} for d in s8.SCOPES}
    empty = {k: None for k in s8.SCOPES}
    und = {"BUY": {"verdict": s8.UNDERLYING_EDGE}, "SELL": {"verdict": s8.NO_EDGE}}
    trans = {"BUY": {"label": s8.T_LOSS}, "SELL": {"label": s8.T_NONE}}
    assert s8.decide(und, trans, no_exit, s2, empty, True)["outcomes"] == [s8.D_OPTION, s8.D_NONE]
    und_none = {d: {"verdict": s8.NO_EDGE} for d in s8.DIRECTIONS}
    trans_none = {d: {"label": s8.T_NONE} for d in s8.DIRECTIONS}
    assert s8.decide(und_none, trans_none, no_exit, s2, empty, True)["s2"] == s8.D_NO_EDGE
    ex = dict(no_exit, SELL={"improved": True, "positive_on_train_and_validation": True})
    assert s8.decide(und_none, trans_none, ex, s2, empty, True)["s2"] == s8.D_EXIT
    good = {k: {"BUY": bad, "SELL": {"profit_factor": 1.3, "average_r": 0.2}} for k in s8.PRE_OOS}
    assert s8.decide(und_none, trans_none, no_exit, good, empty, True)["s2"] == s8.D_S2_EDGE
    prom = dict(empty, SELL={"status": s8.PROMISING})
    assert s8.decide(und_none, trans_none, no_exit, s2, prom, True)["alternative"] == s8.D_ALT
    assert s8.decide(und_none, trans_none, no_exit, s2, prom, False)["alternative"] == s8.D_NONE


# ------------------------------------------------------------------ end-to-end on the synthetic 6-month store
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before, baseline = clean_script.production_hashes(), repr(dr.BASELINE)
    base = tmp_path_factory.mktemp("s8")
    make_store(base / "upstox_6m")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        a = script.run(data_root=base / "upstox_6m", out_dir=base / "out", mutation_days=2, check_sample=5,
                       force_alternatives=True)
    return SimpleNamespace(a=a, base=base, before=before, baseline=baseline)


def test_all_anti_lookahead_checks_pass(env):
    for k, v in env.a["lookahead"].items():
        assert v["passed"], (k, v)
    assert env.a["lookahead_ok"] and env.a["alternatives_run"]


def test_chronological_replay_and_next_candle_entry(env):
    for rows in [env.a["rows"], *env.a["alt_rows"].values()]:
        ts = [s8.dra._ts(r["signal_time"]) for r in rows]
        assert ts == sorted(ts)
        for r in rows:
            if r.get("state") == sh.EVALUATED:
                assert s8.dra._ts(r["entry_time"]) == s8.dra._ts(r["signal_time"]) + STEP


def test_isolation_duplicates_and_engine(env):
    assert env.a["isolation"]["passed"], env.a["isolation"]["checks"]
    dc = env.a["data_checks"]
    assert dc["duplicates"]["passed"] and dc["timestamps"]["passed"] and dc["contract_validity"]["passed"]
    assert dc["exit_engine_reproduces_s2"]["passed"]
    assert all(x["passed"] for x in dc["entry_duplicates"].values())
    assert all(x["passed"] for x in dc["alternatives"]["duplicates"].values())
    assert env.a["data_ok"]


def test_no_strategy_or_production_change(env):
    assert clean_script.production_hashes() == env.before and repr(dr.BASELINE) == env.baseline
    assert env.a["production_unchanged"] is True and env.a["live_orders"] == 0
    assert env.a["trading_mode"] == "PAPER" and env.a["live_trading_enabled"] is False


def test_reports_and_terminal_output(env):
    out = env.base / "out"
    for n in s8.REPORT_FILES:
        assert (out / n).is_file(), n
    json.loads((out / "summary.json").read_text(encoding="utf-8"))
    with (out / "holdout_results.csv").open(encoding="utf-8") as fh:
        hold = list(csv.DictReader(fh))
    assert all(r["label"] == s8.HOLDOUT_LABEL for r in hold if r["system"] == "BEST_ALTERNATIVE" and r.get("label"))
    with (out / "entry_audit.csv").open(encoding="utf-8") as fh:
        assert {r["entry_method"] for r in csv.DictReader(fh)} == set(s8.ENTRY_METHODS)
    text = s8.terminal_text(env.a)
    for label in ("S8 COMPLETE", "CURRENT S2 UNDERLYING EDGE:", "OPTION TRANSLATION:", "ENTRY AUDIT:", "EXIT AUDIT:",
                  "best:", "reason:", "BEST ALTERNATIVE:", "TRAIN:", "VALIDATION:", "OOS:", "HOLDOUT:",
                  "BEST BUY:", "BEST SELL:", "COMBINED:", "70% TARGET:", "LOOKAHEAD:\nPASS", "DATA INTEGRITY:\nPASS",
                  "PRODUCTION CHANGED:\nNO", "LIVE ORDERS:\n0", "FORWARD PAPER VALIDATION:", "FINAL DECISION:",
                  "not live accuracy"):
        assert label in text, label
    for n in s8.REPORT_FILES:
        t = (out / n).read_text(encoding="utf-8").lower()
        assert "access_token" not in t and "bearer " not in t


def test_paper_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))
