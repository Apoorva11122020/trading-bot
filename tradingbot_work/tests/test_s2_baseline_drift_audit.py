from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_s2_baseline_drift_audit as script
from scripts import run_s4_six_month_research as s4script
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.indicators.technical import atr
from src.options.strategy import round_to_tick
from src.research import s2_baseline_drift_audit as dra
from src.research import s4_six_month_lab as s4
from src.safety import live_gate
from tests.test_s4_six_month_research import EXPIRIES, make_store

ROOT = Path(__file__).resolve().parents[1]
S2_FILES = ["src/backtest/research_strategies.py", "src/backtest/s2_1dte_improvement.py",
            "src/backtest/s2_direction_research.py", "src/backtest/current_day_s2.py", "src/options/strategy.py",
            "src/options/selection.py", "config/strategy.yaml", "config/strategy_research.yaml"]
STEP = pd.Timedelta(minutes=5)


def _hashes() -> dict[str, str]:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in S2_FILES if (ROOT / f).is_file()}


def _boom(*_a, **_k):
    raise AssertionError("audit must not open the legacy store / historical service here")


class FakeLegacy:
    """Legacy-store stand-in: same candles as the 6-month store (so only the tick size differs)."""

    def __init__(self, store, spot):
        self.store, self._spot = store, spot

    def spot(self):
        return self._spot

    def premium(self, key, expiry):
        return self.store.premium(key)

    def option_files(self):
        return []


def _old_rows(new, store, params, square_off):
    """Old-pipeline imitation: identical signals/contracts/candles, entry/SL/target rounded with tick 5.0."""
    base = [dict(r) for r in new["rows"] if r.get("instrument_key")]
    redo = dra.reevaluate(base, new["features"], lambda r: store.premium(r["instrument_key"]), lambda r: 5.0,
                          params, square_off)
    out = []
    for r in base:
        x = redo.get(dra.key_of(r))
        if x is None:
            continue
        row = dict(x, **{k: r[k] for k in ("contract", "instrument_key", "strike", "option_type", "expiry_date")})
        row["tick_inferred"] = dra.inferred_tick(row)
        out.append(row)
    return out


def _listed(store):
    return lambda sd, e, k: sorted(store.chain.get((sd, e, k), {}))


def _exists(store):
    def fn(r):
        sd = date.fromisoformat(str(r["session_date"])[:10])
        chain = store.chain.get((sd, r.get("expiry_date"), r.get("option_type")), {})
        return any(c.instrument_key == r.get("instrument_key") for c in chain.values())
    return fn


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = _hashes()
    baseline_repr = repr(dr.BASELINE)
    root = tmp_path_factory.mktemp("drift") / "upstox_6m"
    make_store(root)
    rc, params, session, s2cfg, signal_fn = s4script.load_s2()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        store = s4.SixMonthStore(root)
        spot = store.spot()
        days = [d.isoformat() for d in sh.complete_sessions(spot, store.start, store.end)[0]]
        kw = dict(params=params, square_off=rc.session.square_off, range_start=s2cfg["range_start"],
                  range_end=s2cfg["range_end"])
        new = dra.replay(spot=spot, days=days, signal_fn=signal_fn, resolve=s4.make_resolve(store, rc),
                         premium_for=store.premium_for(None), **kw)
        old = _old_rows(new, store, params, rc.session.square_off)
        a = dra.build_audit(old_rows=old, new=new, new_spot=spot, legacy=FakeLegacy(store, spot),
                            six_premium=store.premium, calendar=store.calendar, listed_strikes=_listed(store),
                            new_exists=_exists(store), implementation=dra.s2_implementation(rc, params, s2cfg),
                            old_spot=spot, legacy_profile=dra.frame_profile([]),
                            six_profile=dra.frame_profile(pd.read_parquet(p) for p in store.layout.option_files()),
                            **kw)
    return SimpleNamespace(root=root, rc=rc, params=params, s2cfg=s2cfg, signal_fn=signal_fn, store=store, spot=spot,
                           days=days, new=new, old=old, a=a, before=before, baseline_repr=baseline_repr, kw=kw)


# ------------------------------------------------------------------ no S2 mutation / safety
def test_no_s2_mutation_and_no_order_path(env):
    assert _hashes() == env.before
    assert repr(dr.BASELINE) == env.baseline_repr
    assert env.a["s2_changed"] is False and env.a["orders"] == 0 and env.a["live_trading_enabled"] is False
    order_call = "place" + "_order"
    for mod in (dra, script):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert order_call not in src
        imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
        assert not any("OrderManager" in ln or "order_manager" in ln for ln in imports)


def test_paper_mode_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        live_gate.assert_paper_trading_only()
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))


# ------------------------------------------------------------------ candle alignment / entry timing / no future data
def test_candle_alignment_passes_on_exact_s2(env):
    al = env.a["alignment"]
    assert al["checked"] > 0 and al["passed"] and not al["problems"]


def test_entry_is_next_candle_open(env):
    ok = [c for c in env.new["contexts"] if c.get("data_status") == "OK"]
    assert ok
    for c in ok:
        assert c["entry_time"] == c["signal_time"] + STEP
        t = dra._ist_series(c["premium"]["timestamp"])
        o = float(c["premium"].loc[(t == c["entry_time"]).to_numpy(), "open"].iat[0])
        assert c["entry"] == pytest.approx(round_to_tick(o, c.get("tick"), "nearest"))


def _ok_ctx(env):
    return dict(next(c for c in env.new["contexts"] if c.get("data_status") == "OK"))


def test_alignment_flags_shifted_lookahead_duplicate_and_future_atr(env):
    p = env.params
    shifted = _ok_ctx(env)
    shifted["entry_time"] = shifted["signal_time"] + 2 * STEP
    same = _ok_ctx(env)
    same["entry_time"] = same["signal_time"]
    dup = _ok_ctx(env)
    dup["premium"] = pd.concat([dup["premium"], dup["premium"].iloc[[0]]], ignore_index=True)
    fut = _ok_ctx(env)
    fut["premium_atr"] = float(atr(fut["premium"].reset_index(drop=True), p.atr_period).iloc[-1]) + 1.0
    probs = {k: dra.alignment_checks([c], p)["problems"] for k, c in
             (("shifted", shifted), ("same", same), ("dup", dup), ("fut", fut))}
    assert "SHIFTED_ENTRY_CANDLE" in probs["shifted"]
    assert "SAME_CANDLE_LOOKAHEAD" in probs["same"]
    assert "DUPLICATE_CANDLE" in probs["dup"]
    assert "ATR_USES_FUTURE_OR_WRONG_CANDLES" in probs["fut"]


def test_no_future_data_in_entry_atr_sl_target(env):
    """Tripling every premium candle after the entry candle must not change entry, ATR, SL or target."""
    rows = [r for r in env.new["rows"] if r.get("state") == sh.EVALUATED]

    def warped(r):
        f = env.store.premium(r["instrument_key"]).copy()
        later = dra._ist_series(f["timestamp"]) > dra._ts(r["entry_time"])
        for col in ("open", "high", "low", "close"):
            f.loc[later.to_numpy(), col] *= 3.0
        return f

    assert rows
    for r in rows:
        x = dra.reevaluate([r], env.new["features"], warped, lambda _r: 0.05, env.params,
                           env.kw["square_off"])[dra.key_of(r)]
        for k in ("entry", "premium_atr", "stop_loss", "target"):
            assert x[k] == pytest.approx(r[k]), k


def test_reevaluate_reproduces_exact_s2(env):
    rows = [r for r in env.new["rows"] if r.get("instrument_key")]
    redo = dra.reevaluate(rows, env.new["features"], lambda r: env.store.premium(r["instrument_key"]),
                          lambda r: 0.05, env.params, env.kw["square_off"])
    for r in rows:
        x = redo[dra.key_of(r)]
        assert (x.get("outcome"), x.get("pnl"), x.get("entry")) == (r.get("outcome"), r.get("pnl"), r.get("entry"))


# ------------------------------------------------------------------ contract validity / option selection
def test_contracts_valid_on_exact_s2(env):
    new = [c for c in env.a["contracts"] if c["dataset"] == "NEW_6M"]
    assert new and all(c["valid"] for c in new)


def test_contract_check_flags_wrong_expiry_type_strike(env):
    r = dict(next(x for x in env.new["rows"] if x.get("instrument_key")))
    listed, exists, cal = _listed(env.store), _exists(env.store), env.store.calendar
    assert dra.contract_check(r, cal, listed, exists)["valid"]
    later = next(e for e in EXPIRIES if e > r["expiry_date"]) if r["expiry_date"] < EXPIRIES[-1] else None
    if later:
        bad = dict(r, expiry_date=later)
        c = dra.contract_check(bad, cal, listed, exists)
        assert not c["nearest_listed_expiry"] and not c["valid"]
    wrong_type = dict(r, option_type="PE" if r["option_type"] == "CE" else "CE")
    c = dra.contract_check(wrong_type, cal, listed, exists)
    assert not c["option_type_matches_direction"] and c["impossible"]
    far = dict(r, strike=float(r["strike"]) + 500.0)
    assert not dra.contract_check(far, cal, listed, exists)["atm_strike"]
    expired = dict(r, expiry_date=date.fromisoformat(str(r["session_date"])[:10]) - timedelta(days=1))
    assert dra.contract_check(expired, cal, listed, exists)["impossible"]


def test_option_selection_drift_detects_every_mismatch(env):
    r = next(x for x in env.new["rows"] if x.get("instrument_key"))
    pair = {"key": dra.key_of(r), "status": "BOTH", "old": dict(r), "new": dict(r)}
    assert not dra.option_selection_rows([pair])[0]["mismatch"]
    changed = dict(pair, new=dict(r, strike=float(r["strike"]) + 50, instrument_key="NSE_FO|X|01-01-2027"))
    row = dra.option_selection_rows([changed])[0]
    assert row["mismatch"] and "STRIKE" in row["mismatches"] and "INSTRUMENT_KEY" in row["mismatches"]
    sel = [x for x in env.a["option_selection"] if x["status"] == "BOTH"]
    assert sel and not any(x["mismatch"] for x in sel)


# ------------------------------------------------------------------ signal comparison / drift attribution
def test_signal_matching_statuses():
    def row(t, d):
        return {"signal_time": f"2026-09-25T{t}:00+05:30", "direction": d, "session_date": "2026-09-25"}
    pairs = dra.match_signals([row("09:30", "BUY"), row("10:00", "SELL")], [row("09:30", "BUY"), row("11:00", "BUY")],
                              {"2026-09-25"})
    assert sorted(p["status"] for p in pairs) == ["BOTH", "NEW_ONLY", "OLD_ONLY"]
    assert dra.match_signals([row("09:30", "BUY")], [], {"2026-09-26"}) == []


def test_old_tick5_signals_are_matched_and_attributed(env):
    a = env.a
    assert a["pairs_summary"].get("BOTH") == len(env.old)
    assert not a["pairs_summary"].get("OLD_ONLY") and not a["pairs_summary"].get("NEW_ONLY")
    assert a["evidence"]["old_tick5_share"] > 0.5
    evaluated = [r for r in env.old if r.get("state") == sh.EVALUATED]
    assert evaluated and all(r["tick_inferred"] == 5.0 for r in evaluated)
    for r in a["comparison"]:
        assert r["outcome_diff_reason"] != "UNEXPLAINED"
        if r["same_outcome"] is False:
            assert r["outcome_diff_reason"]
    for r in a["sl_target"]:
        if not (r["old_sl"] == r["new_sl"] and r["old_target"] == r["new_target"]):
            assert "CONFIGURATION_TICK" in r["difference_sources"] or "ENTRY_PRICE" in r["difference_sources"]
    sm = {s["scenario"]: s for s in a["scenarios"]}
    assert sm["LEGACY_PREMIUM_OLD_TICK"]["reproduction_share"] == pytest.approx(1.0)
    assert sm["SIXM_PREMIUM_OLD_TICK"]["reproduction_share"] == pytest.approx(1.0)
    assert "D" not in a["classes"] and "B" not in a["classes"]
    assert a["recommendation"] == dra.CORRECT


def test_entry_drift_flags():
    p = {"key": ("t", "BUY"), "status": "BOTH",
         "old": {"entry": 100.0, "instrument_key": "k"}, "new": {"entry": 112.0, "instrument_key": "k"}}
    r = dra.entry_drift_rows([p])[0]
    assert (r["gt_5pct"], r["gt_10pct"], r["gt_20pct"]) == (True, True, False)
    assert r["abs_diff"] == pytest.approx(12.0) and r["pct_diff"] == pytest.approx(0.12)


def test_helpers_parse_symbol_tick_and_gap(env):
    m = dra.parse_symbol("NIFTY 23150 CE 29 SEP 26")
    assert m == {"strike": 23150.0, "option_type": "CE", "expiry": date(2026, 9, 29)}
    assert dra.parse_symbol("garbage") == {"strike": None, "option_type": None, "expiry": None}
    assert dra.key_expiry("NSE_FO|73908|29-09-2026") == date(2026, 9, 29)
    assert dra.inferred_tick({"entry": 150.0, "stop_loss": 130.0, "target": 190.0}) == 5.0
    assert dra.inferred_tick({"entry": 150.35, "stop_loss": 130.0, "target": 190.0}) == 0.05
    r = next(x for x in env.new["rows"] if x.get("state") == sh.EVALUATED)
    prem = env.store.premium(r["instrument_key"])
    assert dra.gap_check(prem, r, env.kw["square_off"])["gap"] is False
    t = dra._ist_series(prem["timestamp"])
    holed = prem[(t != dra._ts(r["entry_time"]) + STEP).to_numpy()]
    if r.get("exit_time") and dra._ts(r["exit_time"]) > dra._ts(r["entry_time"]):
        g = dra.gap_check(holed, r, env.kw["square_off"])
        assert g["gap"] and g["missing_after_entry"] >= 1
    assert dra.gap_check(None, r, env.kw["square_off"])["gap"] is True


def test_classify_and_recommend():
    ev = {"underlying_share_diff": 0.0, "option_share_diff": 0.0, "contract_mismatch_share": 0.0,
          "entry_gt5_share": 0.0, "alignment_pass": True, "sl_target_diff_share": 0.0, "gap_losses": 0,
          "old_tick5_share": 0.0, "tick_effect_pp": None, "new_contracts_valid": True,
          "unexplained_outcome_diffs": 0, "old_reproduction_share": 1.0}
    assert dra.classify(ev)[0] == ["H"] and dra.recommend(ev) == dra.CORRECT
    assert dra.recommend(dict(ev, alignment_pass=False)) == dra.FIX
    assert dra.recommend(dict(ev, new_contracts_valid=False)) == dra.FIX
    assert dra.recommend(dict(ev, old_reproduction_share=0.5)) == dra.FURTHER
    assert dra.recommend(dict(ev, unexplained_outcome_diffs=2)) == dra.FURTHER
    cls = dra.classify(dict(ev, old_tick5_share=0.9, tick_effect_pp=-0.08, gap_losses=3, entry_gt5_share=0.2))[0]
    assert {"C", "F", "G"} <= set(cls) and "H" not in cls


# ------------------------------------------------------------------ monthly aggregation / counts
def test_monthly_aggregation(env):
    m = dra.monthly_rows(env.new["rows"], "NEW_6M")
    months = {r["month"] for r in m}
    assert months == {str(r["session_date"])[:7] for r in env.new["rows"]}
    total = dra.metrics(env.new["rows"])
    comb = [r for r in m if r["scope"] == "COMBINED"]
    assert sum(r["trades"] for r in comb) == total["trades"]
    assert sum(r["wins"] for r in comb) == total["wins"]
    for mo in months:
        by = {r["scope"]: r for r in m if r["month"] == mo}
        assert by["BUY"]["trades"] + by["SELL"]["trades"] == by["COMBINED"]["trades"]
        assert by["BUY"]["wins"] + by["SELL"]["wins"] == by["COMBINED"]["wins"]
    c = dra.count_rows(env.new["rows"], "X")
    assert sum(r["signals"] for r in c) == len(env.new["rows"])


# ------------------------------------------------------------------ reports + script output
def test_reports_written_with_all_sections(env, tmp_path):
    paths = dra.write_reports(env.a, tmp_path)
    assert set(paths) == set(dra.REPORT_FILES) and all(p.is_file() for p in paths.values())
    md = paths["s2_baseline_drift_audit.md"].read_text(encoding="utf-8")
    for s in ("Executive Summary", "Old vs New Results", "Signal Count Comparison", "Option Selection Drift",
              "Entry Price Drift", "Candle Alignment", "SL/Target Drift", "Outcome Drift", "Monthly Results",
              "Data Gaps", "Contract Validation", "Root Cause", "Severity", "Recommended Next Step"):
        assert f"## {s}" in md, s
    with paths["s2_baseline_drift_signal_comparison.csv"].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(env.a["comparison"]) and {"old_entry", "new_entry", "old_outcome"} <= set(rows[0])
    for text in (md, *(p.read_text(encoding="utf-8") for p in paths.values())):
        assert "access_token" not in text.lower() and "bearer " not in text.lower()


def _dump_old(rows, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    extra = ("old_split", "strike", "option_type", "expiry_date", "tick_inferred")
    clean = [{k: v for k, v in r.items() if k not in extra} for r in rows]
    for name, (lo, hi) in (("train.json", ("0000", "2026-08-31")), ("validation.json", ("2026-09-01", "2026-09-02")),
                           ("oos.json", ("2026-09-03", "9999"))):
        part = [r for r in clean if lo <= str(r["session_date"])[:10] <= hi]
        (d / name).write_text(json.dumps({"signals": {dra.S2_NAME: part}}, default=str), encoding="utf-8")


def test_script_end_to_end_prints_required_summary(env, tmp_path, monkeypatch, capsys):
    _dump_old(env.old, tmp_path / "old")
    monkeypatch.setattr(script, "_legacy_root", lambda: tmp_path / "no_legacy")
    monkeypatch.setattr(storage.ParquetCandleStore, "__init__", _boom)
    a = script.run(data_root=env.root, old_dir=tmp_path / "old", out_dir=tmp_path / "out")
    out = capsys.readouterr().out
    for label in ("S2 BASELINE DRIFT AUDIT", "OLD", "NEW 6-MONTH", "BUY:", "SELL:", "COMBINED:",
                  "Signal count drift:", "Option selection drift:", "Entry price drift:", "Candle alignment: PASS",
                  "Contract validity: PASS", "Data gaps:", "Primary root cause:", "S2 changed: NO", "Orders: 0",
                  "Final recommendation:"):
        assert label in out, label
    assert a["recommendation"] == dra.FURTHER  # legacy store absent -> old reproduction cannot be verified
    assert (tmp_path / "out" / "s2_baseline_drift_audit.md").is_file()
    assert _hashes() == env.before
