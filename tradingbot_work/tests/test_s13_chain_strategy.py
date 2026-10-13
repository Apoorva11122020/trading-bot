from __future__ import annotations

import inspect
import json
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="S13.2 reuses the S12.2 bounded loader, which imports scikit-learn")

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s13_chain_strategy as script
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.research import s4_six_month_lab as s4
from src.research import s10_ml_data as sd
from src.research import s12_2_constrained_ml as s122
from src.research import s13_chain_strategy as s13
from tests.test_s4_six_month_research import make_store
from tests.test_s9_futures_data import add_futures
from tests.test_s12_2_constrained_ml import _s12_reports

STEP = sd.STEP
IST = sd.IST
TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-S13-2"
DECISIONS = {s13.PROMISING, s13.REJECTED, s13.INSUFF}
SYNTH = {"CLEAN_START": date(2026, 4, 1), "TRAIN_END": date(2026, 6, 30),
         "VALIDATION": (date(2026, 7, 1), date(2026, 7, 31)), "OOS": (date(2026, 8, 1), date(2026, 9, 30)),
         "MAX_LOAD": date(2026, 9, 30)}
T0 = pd.Timestamp(datetime.combine(date(2026, 5, 4), time(9, 15)), tz=IST)


def _boom(*_a, **_k):
    raise AssertionError("S13.2 must read only the stored research roots")


def _s131_summary(p: Path, lo: date, hi: date, *, ok: bool = True, atm: float = 1.0, pm1: float = 1.0) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    per = {"ran": True, "range": [lo.isoformat(), hi.isoformat()], "data_quality": {"passed": ok},
           "point_in_time": {"passed": True}, "reconstruction": {"passed": True},
           "chain": {"atm_ce": atm, "atm_pe": atm, "band_pm1_ce": pm1, "band_pm1_pe": pm1, "band_pm2_ce": 0.8,
                     "band_pm2_pe": 0.8, "band_pm3_ce": 0.33, "band_pm3_pe": 0.33}}
    p.write_text(json.dumps({"periods": {"CLEAN_USABLE": per}}), encoding="utf-8")
    return p


def _comparison_files(d: Path) -> dict[str, Path]:
    d.mkdir(parents=True, exist_ok=True)
    p = {"S2": d / "s2.csv", "S9": d / "s9.csv"}
    p["S2"].write_text("month,scope,total_trades,wins,losses,gross_profit,gross_loss,net_pnl\n"
                       "2026-04,COMBINED,99,50,49,9999,-1,9998\n"
                       "2026-05,COMBINED,10,6,4,300,-100,200\n"
                       "2026-06,COMBINED,10,4,6,200,-300,-100\n", encoding="utf-8")
    p["S9"].write_text("month,scope,trades,wins,losses,accuracy,profit_factor,net_pnl,max_drawdown\n"
                       "MAY,COMBINED,8,3,5,0.375,0.7,-40,60\nJUNE,COMBINED,6,3,3,0.5,1.1,10,20\n", encoding="utf-8")
    return p


def _patch(mp) -> None:
    for k, v in SYNTH.items():
        mp.setattr(s122, k, v)
        mp.setattr(s13, k, v)


# ------------------------------------------------------------------ synthetic contract helpers
def _frame(closes, vols=None, ois=None, start=T0) -> pd.DataFrame:
    n = len(closes)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"timestamp": [start + i * STEP for i in range(n)], "open": c, "high": c + 1.0,
                         "low": c - 1.0, "close": c,
                         "volume": np.asarray(vols if vols is not None else [1000.0] * n, dtype=float),
                         "open_interest": np.asarray(ois if ois is not None else [50000.0] * n, dtype=float)})


def _band(ce=None, pe=None, **over) -> dict:
    """L[(kind, k)] with 30 candles each; ``over`` replaces one contract's frame, e.g. CE0=frame."""
    base_ce = ce if ce is not None else _frame(np.linspace(100, 115, 30))
    base_pe = pe if pe is not None else _frame(np.linspace(100, 90, 30))
    L = {}
    for kind, f in (("CE", base_ce), ("PE", base_pe)):
        for k in (-1, 0, 1):
            L[(kind, k)] = s13.Contract(over.get(f"{kind}{k}", f))
    return L


TS = T0 + 20 * STEP


# ------------------------------------------------------------------ numeric helpers / zero denominators
def test_ratio_imb_ret_undefined_is_nan():
    assert s13.ratio(2.0, 4.0) == 0.5 and np.isnan(s13.ratio(1.0, 0.0)) and np.isnan(s13.ratio(np.nan, 1.0))
    assert s13.imb(3.0, 1.0) == 0.5 and np.isnan(s13.imb(0.0, 0.0)) and np.isnan(s13.imb(np.nan, 1.0))
    assert s13.ret(110.0, 100.0) == pytest.approx(0.1) and np.isnan(s13.ret(1.0, 0.0)) and np.isnan(s13.ret(1.0, np.nan))


# ------------------------------------------------------------------ point-in-time chain features
def test_chain_features_ignore_future_candles():
    a = s13._chain_features(_band(), TS)
    ce = _frame(np.linspace(100, 115, 30))
    ce.loc[ce["timestamp"] > TS, ["open", "high", "low", "close"]] *= 3.0
    ce.loc[ce["timestamp"] > TS, ["volume", "open_interest"]] *= 7.0
    b = s13._chain_features(_band(ce=ce), TS)
    assert a.keys() == b.keys()
    assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == pytest.approx(b[k]) for k in a)


def test_oi_change_uses_previous_completed_candle():
    ois = [50000.0] * 30
    ois[20], ois[19], ois[18] = 52000.0, 51000.0, 10.0
    f = s13._chain_features(_band(ce=_frame(np.linspace(100, 115, 30), ois=ois)), TS)
    assert f["ce_oi_chg"] == pytest.approx(1000.0) and f["pe_oi_chg"] == pytest.approx(0.0)
    assert f["oi_chg_imb"] == pytest.approx(1.0)
    gap = _frame(np.linspace(100, 115, 30), ois=ois).drop(index=19)
    g = s13._chain_features(_band(ce=gap), TS)
    assert np.isnan(g["ce_oi_chg"]) and np.isnan(g["oi_chg_imb"])


def test_volume_imbalance_uses_candle_t_only():
    vols = [1000.0] * 30
    vols[20], vols[21] = 3000.0, 1e9
    f = s13._chain_features(_band(ce=_frame(np.linspace(100, 115, 30), vols=vols)), TS)
    assert f["ce_vol"] == 3000.0 and f["pe_vol"] == 1000.0 and f["total_vol"] == 4000.0
    assert f["vol_imb"] == pytest.approx(0.5) and f["vol_ratio"] == pytest.approx(3.0)
    neg = list(vols)
    neg[20] = -5.0
    g = s13._chain_features(_band(ce=_frame(np.linspace(100, 115, 30), vols=neg)), TS)
    assert np.isnan(g["ce_vol"]) and np.isnan(g["vol_imb"])


def test_pcr_uses_atm_pm1_band_and_zero_denominator_unavailable():
    f = s13._chain_features(_band(), TS)
    assert f["pcr_oi"] == pytest.approx(1.0) and f["pcr_vol"] == pytest.approx(1.0)
    pe_hi = _frame(np.linspace(100, 90, 30), ois=[100000.0] * 30)
    g = s13._chain_features(_band(**{"PE-1": pe_hi}), TS)
    assert g["pcr_oi"] == pytest.approx((100000 + 50000 + 50000) / 150000.0)
    zero = _frame(np.linspace(100, 115, 30), vols=[0.0] * 30, ois=[0.0] * 30)
    z = s13._chain_features(_band(**{"CE-1": zero, "CE0": zero, "CE1": zero}), TS)
    assert np.isnan(z["pcr_oi"]) and np.isnan(z["pcr_vol"]) and np.isnan(z["call_oi_conc"])
    src = inspect.getsource(s13._chain_features)
    assert "(-1, 0, 1)" in src and "(-2" not in src and "(-3" not in src and "(-5" not in src


def test_straddle_uses_completed_candles_only():
    f = s13._chain_features(_band(), TS)
    ce, pe = np.linspace(100, 115, 30), np.linspace(100, 90, 30)
    assert f["straddle"] == pytest.approx(ce[20] + pe[20])
    assert f["straddle_ret1"] == pytest.approx((ce[20] + pe[20]) / (ce[19] + pe[19]) - 1)
    assert f["straddle_mom"] == pytest.approx((ce[20] + pe[20]) / (ce[17] + pe[17]) - 1)
    assert f["straddle_expanding"] == float(f["straddle_mom"] > 0)
    gap = _frame(ce).drop(index=17)
    assert np.isnan(s13._chain_features(_band(ce=gap), TS)["straddle_mom"])


def test_momentum_and_breadth():
    f = s13._chain_features(_band(), TS)
    assert f["ce_mom"] > 0 > f["pe_mom"] and f["mom_diff"] > 0 and f["strength_diff"] > 0
    assert f["bull_breadth"] == 1.0 and f["bear_breadth"] == 0.0 and f["call_breadth"] == 1.0
    assert f["put_breadth"] == 0.0 and f["call_oi_conc"] == pytest.approx(1 / 3)


# ------------------------------------------------------------------ signals: never both sides
def _panel(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {c: [np.nan] * n for c in s13.FEATURES}
    base.update(cols)
    base.setdefault("eligible", [True] * n)
    return pd.DataFrame(base)


def test_signals_conflict_is_no_trade():
    P = _panel(ce_mom=[1.0, 1.0, 0.0], pe_mom=[1.0, 0.0, 1.0], pcr_oi=[1.0, 1.0, 1.0], pcr_vol=[1.0, 1.0, 1.0])
    side, cnt = s13.signals(P, "D_PCR_CONFIRMATION")
    assert side.tolist() == [None, "CE", "PE"] and cnt["CONFLICT_NO_TRADE"] == 1
    P2 = _panel(ce_mom=[1.0], pe_mom=[0.0], pcr_oi=[1.0], pcr_vol=[1.0], eligible=[False])
    assert s13.signals(P2, "D_PCR_CONFIRMATION")[0].tolist() == [None]
    s, c = s13.signals(pd.DataFrame(), "A_CHAIN_PRICE_MOMENTUM")
    assert s.empty and not c


def test_signals_missing_feature_never_fires():
    P = _panel(ce_mom=[1.0], pe_mom=[np.nan], mom_diff=[np.nan], strength_diff=[np.nan])
    for fam in s13.FAMILIES:
        assert s13.signals(P, fam)[0].tolist() == [None], fam


def test_six_predeclared_families_and_fixed_rules():
    assert list(s13.FAMILIES) == ["A_CHAIN_PRICE_MOMENTUM", "B_CHAIN_VOLUME", "C_CHAIN_OI", "D_PCR_CONFIRMATION",
                                  "E_STRADDLE_CONTEXT", "F_CHAIN_PLUS_FUTURES"]
    assert len(s13.candidates()) == 12 and s13.RULES == {"MOM_MIN": 0.5, "VOL_IMB_MIN": 0.10,
                                                         "OI_CHG_IMB_MIN": 0.20, "PCR_NEUTRAL": 1.0}
    assert s13.TARGET_R == 1.5 and s13.HORIZON == 6 and sd.STOP_ATR == 1.5


def test_real_window():
    assert s13.CLEAN_START == date(2025, 9, 26) and s13.TRAIN_END == date(2026, 3, 31)
    assert s13.VALIDATION == (date(2026, 4, 1), date(2026, 4, 30))
    assert s13.OOS == (date(2026, 5, 1), date(2026, 6, 30)) and s13.MAX_LOAD == date(2026, 6, 30)


# ------------------------------------------------------------------ selection / status / decision
def _m(resolved, pf, ex, dd=1.0):
    return {"resolved": resolved, "profit_factor": pf, "expectancy_r": ex, "max_drawdown_r": dd}


def test_selection_and_status_rules():
    assert s13.train_qualified(_m(30, 1.2, 0.1)) and not s13.train_qualified(_m(29, 2.0, 0.5))
    assert not s13.train_qualified(_m(40, 0.9, 0.1)) and not s13.train_qualified(_m(40, 1.2, -0.1))
    assert not s13.train_qualified(_m(40, 1.2, 0.1, dd=11.0))
    assert s13.stage_status(_m(29, 2.0, 0.5), None) == s13.INSUFF
    assert s13.stage_status(_m(30, 0.9, -0.1), None) == s13.REJECTED
    assert s13.stage_status(_m(30, 1.1, 0.04), _m(60, 1.5, 0.2)) == s13.DEGRADED
    assert s13.stage_status(_m(30, 1.3, 0.15), _m(60, 1.5, 0.2)) == s13.PASS
    assert s13.sample_label(29) == "INSUFFICIENT_DATA" and s13.sample_label(30) == "LOW_SAMPLE"
    assert s13.sample_label(50) == "MODERATE_SAMPLE" and s13.sample_label(100) == "STRONGER_SAMPLE"


def test_final_decision_rules():
    q = ["A_CHAIN_PRICE_MOMENTUM__BUY_CE"]
    kw = dict(checks_ok=True, qualified=q, april_pass=q, april_status={q[0]: s13.PASS})
    assert s13.final_decision(**kw, oos_m=_m(40, 1.2, 0.1))[0] == s13.PROMISING
    assert s13.final_decision(**kw, oos_m=_m(40, 0.9, -0.1))[0] == s13.REJECTED
    assert s13.final_decision(**kw, oos_m=_m(10, 2.0, 0.5))[0] == s13.INSUFF
    assert s13.final_decision(**{**kw, "checks_ok": False}, oos_m=_m(40, 1.2, 0.1))[0] == s13.REJECTED
    assert s13.final_decision(**{**kw, "qualified": []}, oos_m=None)[0] == s13.REJECTED
    assert s13.final_decision(**{**kw, "april_pass": [], "april_status": {q[0]: s13.INSUFF}}, oos_m=None)[0] \
        == s13.INSUFF
    assert s13.final_decision(**{**kw, "april_pass": [], "april_status": {q[0]: s13.DEGRADED}}, oos_m=None)[0] \
        == s13.REJECTED


def test_s131_gate_rules(tmp_path):
    lo, hi = s13.CLEAN_START, s13.MAX_LOAD
    load = lambda p: json.loads(p.read_text())  # noqa: E731
    assert s13.s131_gate(load(_s131_summary(tmp_path / "ok.json", lo, hi)), lo, hi)["passed"]
    assert not s13.s131_gate(load(_s131_summary(tmp_path / "dq.json", lo, hi, ok=False)), lo, hi)["passed"]
    assert not s13.s131_gate(load(_s131_summary(tmp_path / "pm1.json", lo, hi, pm1=0.9)), lo, hi)["passed"]
    assert not s13.s131_gate(load(_s131_summary(tmp_path / "rng.json", lo, date(2026, 9, 30))), lo, hi)["passed"]
    assert not s13.s131_gate(None)["passed"]


# ------------------------------------------------------------------ synthetic end-to-end pipeline
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Full S13.2 pipeline on the synthetic store (Apr..Sep 2026, 3 sessions / month) with the window scaled to it:
    TRAIN Apr..Jun, VALIDATION July, OOS Aug..Sep. Selection / April status are forced open so the frozen OOS path
    and the chronology can be verified (the real rules are unit-tested above)."""
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s132")
    six, out = base / "upstox_6m", base / "out"
    make_store(six)
    add_futures(six)
    cal = [d for d in s4.SixMonthStore(six).calendar if SYNTH["CLEAN_START"] <= d <= SYNTH["MAX_LOAD"]]
    _s12_reports(base / "s12", cal, SYNTH["CLEAN_START"])
    s131 = _s131_summary(base / "s131" / "summary.json", SYNTH["CLEAN_START"], SYNTH["MAX_LOAD"])
    cmp = _comparison_files(base / "cmp")
    opened: list[str] = []
    real = u6.read_parquet

    def spy(p):
        opened.append(Path(p).as_posix())
        return real(p)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        mp.setattr(u6, "read_parquet", spy)
        _patch(mp)
        mp.setattr(s13, "train_qualified", lambda m: m["trades"] > 0)
        mp.setattr(s13, "stage_status", lambda m, t: s13.PASS if m and m["trades"] else s13.INSUFF)
        a = script.run(expanded_root=base / "upstox_expanded", data_root=six, out_dir=out, s12_dir=base / "s12",
                       s131_summary=s131, comparison_sources=cmp, mutation_days=2, check_sample=5,
                       secret_values=[TOKEN])
    return SimpleNamespace(a=a, base=base, six=six, out=out, before=before, opened=opened, cmp=cmp)


def test_pipeline_ran_with_data_quality_gate(env):
    a = env.a
    assert a["ran"], a.get("reason")
    dq = a["data_quality"]
    assert dq["passed"] and dq["s12"]["passed"] and dq["s13_1"]["passed"] and "off_grid_quarantine" in dq
    assert a["funnels"]["TRAIN"].get("ELIGIBLE", 0) > 0 and a["strike_step"] == 50.0


def test_all_lookahead_checks_pass(env):
    la = env.a["lookahead"]
    for k in ("chain_point_in_time_mutation", "feature_truncation", "contract_selection_causal", "next_candle_entry",
              "duplicate_trade_prevention", "deterministic_replay", "max_loaded_timestamp", "selection_isolation"):
        assert la[k]["passed"], (k, la[k])
    assert env.a["lookahead_ok"] and la["feature_truncation"]["checked"] > 0


def test_futures_features_causal(env):
    P = env.a["_panels"]["TRAIN"]
    t = P[P["eligible"]]["signal_time"].iloc[len(P[P["eligible"]]) // 2]
    view = s122.phase_view(env.a["_market"], SYNTH["TRAIN_END"])
    mm = sd.mutated_market(view, t, spot=False, options=False, seed=3, price=True, volume=True)
    Q, _ = s13.build_panel(mm, [t.date()], env.a["strike_step"])
    a, b = P[P["signal_time"] == t], Q[Q["signal_time"] == t]
    assert s13._feat_equal(a, b, s13.FUT_COLS)


def test_contracts_never_from_the_future(env):
    P = env.a["_panels"]["TRAIN"]
    E = P[P["eligible"]]
    assert (E["expiry"] >= E["session_date"]).all()
    assert (E["CE_strike"] == E["atm"]).all() and (E["PE_strike"] == E["atm"]).all()
    for split in ("train", "validation", "oos"):
        for r in env.a[split].values():
            for t in r["trades"]:
                assert t["expiry"] >= t["session_date"] and t["entry_time"] == t["signal_time"] + STEP


def test_no_duplicate_or_overlapping_trades(env):
    for split in ("train", "validation", "oos"):
        for r in env.a[split].values():
            ts = [t["signal_time"] for t in r["trades"]]
            assert len(ts) == len(set(ts))
            ex = [t for t in r["trades"] if t["outcome"] in sd.EXECUTED]
            for x, y in zip(ex, ex[1:]):
                assert y["entry_time"] >= x["exit_time"]
    c = env.a["combined_oos_trades"]
    assert all(t["side"] in ("CE", "PE") for t in c)


def test_labels_use_fixed_exit(env):
    ok = [t for r in env.a["train"].values() for t in r["trades"] if t["outcome"] in sd.EXECUTED]
    assert ok
    for t in ok:
        assert t["risk"] == pytest.approx(1.5 * t["atr"])
        assert t["stop_loss"] == pytest.approx(t["entry"] - t["risk"])
        assert t["target"] == pytest.approx(t["entry"] + 1.5 * t["risk"])


def test_deterministic_replay(env):
    d = env.a["lookahead"]["deterministic_replay"]
    assert d["passed"] and d["features_identical"] and d["labels_identical"] and d["trades_identical"]


def test_chronology_and_frozen_oos(env):
    st = env.a["stages"]
    assert st.index("train_selection") < st.index("freeze") < st.index("validation_april") < st.index("oos_panel") \
        < st.index("oos_may_june")
    assert env.a["_panels"]["TRAIN"]["session_date"].max() <= SYNTH["TRAIN_END"].isoformat()
    assert set(env.a["oos"]) == set(env.a["april_pass"]) and env.a["april_pass"]
    v = env.a["_panels"]["VALIDATION"]["session_date"]
    assert v.min() >= SYNTH["VALIDATION"][0].isoformat() and v.max() <= SYNTH["VALIDATION"][1].isoformat()
    for s in s13.SIDES:
        n = env.a["picks_oos"][s]
        assert n is None or (n in env.a["april_pass"] and n == s13.best(env.a["april_pass"], env.a["train"], s))


def test_oos_not_run_when_april_fails(env):
    with pytest.MonkeyPatch.context() as mp:
        _patch(mp)
        mp.setattr(s13, "train_qualified", lambda m: m["trades"] > 0)
        mp.setattr(s13, "stage_status", lambda m, t: s13.REJECTED)
        b = s13.build_study(market=env.a["_market"], read_log=[], data_quality={"passed": True}, mutation_days=0,
                            check_sample=2)
    assert "oos_panel" not in b["stages"] and b["oos"] == {} and b["combined_oos"] is None
    assert set(b["oos_status"].values()) == {s13.NOT_RUN} and b["decision"] == s13.REJECTED
    assert b["forward_paper"] == "NOT SAFE" and "NOT_RUN" in s13.terminal_text(b)


def test_data_quality_failure_stops_before_research(env):
    b = s13.build_study(market=env.a["_market"], read_log=[], data_quality={"passed": False, "reason": "synthetic"})
    assert b["ran"] is False and b["decision"] == s13.INSUFF and b["forward_paper"] == "NOT SAFE"
    assert "S13.2 NOT RUN" in s13.terminal_text(b)


def test_bounded_loading(env):
    la = env.a["lookahead"]["max_loaded_timestamp"]
    assert la["passed"] and la["max_loaded_date"] <= SYNTH["MAX_LOAD"].isoformat()
    assert env.opened and all("/2026-10" not in p for p in env.opened)
    assert env.a["october"] == "NOT LOADED" and env.a["july_plus"] == "NOT LOADED"


def test_comparison_from_saved_reports(env):
    rows = {r["system"]: r for r in env.a["comparison"]}
    s2 = rows["S2"]
    assert s2["trades"] == 20 and s2["resolved"] == 20 and s2["accuracy"] == pytest.approx(0.5)
    assert s2["profit_factor"] == pytest.approx(500 / 400) and s2["net_pnl"] == pytest.approx(100)
    s9 = rows["S9_FUTURES_VOLUME"]
    assert s9["trades"] == 14 and s9["accuracy"] == pytest.approx(6 / 14) and s9["profit_factor"] is None
    assert rows[s13.STRATEGY]["status"] in ("AVAILABLE", s13.NOT_RUN)
    assert all(r["status"] == "NOT_AVAILABLE" for r in s13.oos_comparison({}, None)[:2])


def test_reports_and_token_leakage(env):
    for n in s13.REPORT_FILES:
        p = env.out / n
        assert p.is_file(), n
        t = p.read_text(encoding="utf-8")
        assert TOKEN not in t and "bearer " not in t.lower() and "AI predicts" not in t
    assert env.a["token_leak"]["passed"]
    ok, leaks = u6.token_leak_check([env.out], [TOKEN])
    assert ok and not leaks
    tr = pd.read_csv(env.out / "train_results.csv")
    assert len(tr) == 18 and set(tr["family"]) == set(s13.FAMILIES)
    cov = pd.read_csv(env.out / "coverage.csv")
    assert {"TRAIN", "VALIDATION"} <= set(cov["split"]) and "atm_pm2_listed_share_descriptive" in cov.columns


def test_production_unchanged_and_paper(env):
    a = env.a
    assert a["production_unchanged"] is True and clean_script.production_hashes() == env.before
    assert a["live_orders"] == 0 and a["trading_mode"] == "PAPER" and a["live_trading_enabled"] is False
    assert a["ml_trained"] is False and a["decision"] in DECISIONS
    assert a["forward_paper"] == ("SAFE" if a["decision"] == s13.PROMISING else "NOT SAFE")


def test_terminal_format(env):
    text = s13.terminal_text(env.a)
    lines = text.splitlines()
    assert lines[0] == s13.HEADING
    order = ["TRAIN:", "Best CE:", "Best PE:", "VALIDATION APRIL:", "Best CE:", "Best PE:", "OOS MAY-JUNE:",
             "Best CE:", "Best PE:", "COMBINED:", "S2 vs S9 vs S13 OOS:", "S2:", "S9:", "S13:", "70%:", "LOOKAHEAD:",
             "DATA QUALITY:", "OCTOBER LOADED:", "PRODUCTION CHANGED:", "LIVE ORDERS:", "FORWARD PAPER:", "FINAL:"]
    pos = -1
    for needle in order:
        pos = next(i for i, l in enumerate(lines) if i > pos and l.startswith(needle))
    assert lines[lines.index("70%:") + 1] in ("ACHIEVED", "NOT ACHIEVED", s13.INSUFF)
    assert lines[lines.index("OCTOBER LOADED:") + 1] == "NO"
    assert lines[lines.index("PRODUCTION CHANGED:") + 1] == "NO" and lines[lines.index("LIVE ORDERS:") + 1] == "0"
    assert lines[lines.index("FORWARD PAPER:") + 1] in ("SAFE", "NOT SAFE")
    assert lines[lines.index("FINAL:") + 1] in DECISIONS
    assert "AI predicts" not in text and "live accuracy" in text


def test_no_ml_training_or_order_code():
    for mod in (s13, script):
        src = inspect.getsource(mod)
        for bad in ("LogisticRegression", "XGB", "xgboost", "LGBM", "lightgbm", "MLPClassifier", "torch",
                    "tensorflow", "keras", ".fit(", "predict_proba", "GridSearch", "optuna", "SMOTE", "place_order"):
            assert bad not in src, (mod.__name__, bad)
    src = inspect.getsource(s13.build_panel)
    assert "(-1, 0, 1)" in src and "(-2, 2)" in src and "(-3" not in src and "(-5" not in src \
        and "3 * step" not in src and "5 * step" not in src
