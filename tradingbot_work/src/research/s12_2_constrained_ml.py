"""S12.2 - constrained 9.1-month ML research (PAPER | RESEARCH ONLY | NO ORDERS).

S12 / S12.1 showed the clean option history is 2025-09-26 -> 2026-06-30 (about 9.1 months, NOT 12+). This study asks
only whether that clean segment carries a weak but repeatable signal worth carrying into future fresh-paper
validation. It is NOT a 12-month or 18-month validation and makes no accuracy or profit claim.

Reused unchanged from S10 / S11: L1 label (entry = next option candle open, R = 1.5 x option ATR14, STOP = entry - R,
TARGET = entry + 1.5R, 6 option candles, WIN / LOSS / TIMEOUT / AMBIGUOUS), the frozen S10 feature family, separate
CE / PE models (Logistic Regression or HistGradientBoosting only, fixed parameters), median imputer + scaler and
chronological Platt calibration fit inside each training fold only, purge + 6-bar embargo, thresholds {0.55, 0.60,
0.65}, margin >= 0.05, EV = 1.5 * P - (1 - P) > 0.

Chronology (never reversed):
  1. data-quality gate (S12 verdicts for the clean window) - stop if it fails
  2. TRAIN 2025-09-26 .. 2026-03-31 built from a view truncated at 2026-03-31; three inner folds; family and ONE
     threshold chosen on inner folds only
  3. April 2026 rows built (view truncated at 2026-04-30); FREEZE (final fit on the whole TRAIN); April VALIDATION
  4. May-June 2026 rows built and evaluated ONLY if April passed (frozen, no modification)
Files that could hold candles after 2026-06-30 are never opened (July / August / September / October not loaded);
candles before 2025-09-26 are not used. Historical results are not live accuracy; no profit is guaranteed.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import sklearn

from src.backtest import shadow_historical as sh
from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.research import s10_ml_data as sd
from src.research import s10_ml_model as sm
from src.research import s10_ml_probability as s10
from src.research import s11_history as hx
from src.research import s11_ml_expanded as s11
from src.research import s4_six_month_lab as s4
from src.research import s6_clean_strategy_lab as s6

HEADING = "S12.2 CONSTRAINED ML RESEARCH COMPLETE"
STRATEGY = "S12_2_CONSTRAINED_ML"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | 9.1-month constrained historical experiment (not live accuracy)"
IST = sd.IST
DISCLAIMER = ("Constrained 9.1-month research experiment on the clean option history (2025-09-26 -> 2026-06-30). Not a "
              "12-month or 18-month validation. Historical results are not live accuracy; no guaranteed profit.")
TARGET_R, HORIZON = s11.TARGET_R, sd.HORIZON
LABEL = s11.LABEL

CLEAN_START = date(2025, 9, 26)
TRAIN_END = date(2026, 3, 31)
VALIDATION = (date(2026, 4, 1), date(2026, 4, 30))
OOS = (date(2026, 5, 1), date(2026, 6, 30))
MAX_LOAD = date(2026, 6, 30)
CLEAN_MONTHS = 9.1
FOLDS = (("FOLD_1", date(2025, 12, 31), date(2026, 1, 1), date(2026, 1, 31)),
         ("FOLD_2", date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 28)),
         ("FOLD_3", date(2026, 2, 28), date(2026, 3, 1), date(2026, 3, 31)))
FAMILIES = (sm.LR, sm.HGB)

# Pre-declared rules (fixed before any S12.2 model was fitted).
MIN_FOLD_TRAIN_SESSIONS, MIN_FOLD_VAL_SESSIONS = 50, 15
MIN_VAL_RESOLVED = MIN_OOS_RESOLVED = 30
MAX_DD_R, MIN_COVERAGE = 10.0, 0.005
MAX_AUC_DEGRADATION = 0.10
MIN_S12_ATM_COVERAGE = 0.90
TOP_FEATURES = 10
PASS, FAIL, INSUFFICIENT, NOT_RUN, RUN = s11.PASS, s11.FAIL, s11.INSUFFICIENT, s11.NOT_RUN, s11.RUN
STABLE = s11.STABLE
PROMISING, REJECTED, INSUFF, UNSTABLE = "PROMISING", "REJECTED", "INSUFFICIENT_DATA", "MODEL_UNSTABLE"
TARGET_70_TEXT = "NOT A SUCCESS CRITERION"
SCOPES = ("CE", "PE", "COMBINED")
SPLIT_NAMES = {"TRAIN": "TRAIN (in-sample)", "VALIDATION": "APRIL 2026 VALIDATION", "OOS": "MAY-JUNE 2026 OOS"}


# ------------------------------------------------------------------ bounded loading (never opens July+ files)
def _month_in(name: str, lo: date, hi: date) -> bool:
    return hx.month_label(lo) <= name <= hx.month_label(hi)


def _between(f: pd.DataFrame | None, lo: date, hi: date) -> pd.DataFrame | None:
    if f is None or f.empty:
        return None
    d = f["timestamp"].dt.date
    return sd._clean(f[(d >= lo) & (d <= hi)])


def load_clean_market(roots: Sequence[str | Path], underlying: str = "NIFTY", lo: date = CLEAN_START,
                      hi: date = MAX_LOAD) -> tuple[sd.MarketData, list[dict]]:
    """Market view of [lo, hi]. Only files that cannot hold candles after ``hi`` are opened: underlying / option month
    folders <= hi and futures contracts expiring <= hi. Every file read is logged with its raw min / max timestamp."""
    hx.guard_no_october(hi)
    if hi > MAX_LOAD:
        raise SystemExit(f"Refusing {hi}: S12.2 never loads candles after {MAX_LOAD}")
    log: list[dict] = []

    def rd(p: Path) -> pd.DataFrame:
        f = u6.read_parquet(p)
        log.append({"file": Path(p).as_posix(), "rows": int(len(f)),
                    "min": f["timestamp"].min() if len(f) else None, "max": f["timestamp"].max() if len(f) else None})
        return f

    stores = [s4.SixMonthStore(r, underlying) for r in roots]
    spots, sessions = [], set()
    chains: dict = {}
    atm: dict[date, set[str]] = defaultdict(set)
    calendar: set[date] = set()
    files: dict[str, set[Path]] = defaultdict(set)
    for s in stores:
        parts = [rd(p) for p in s.layout.underlying_files() if _month_in(p.stem, lo, hi)]
        parts = [p for p in parts if len(p)]
        sp = _between(pd.concat(parts, ignore_index=True), lo, hi) if parts else None
        if sp is not None:
            spots.append(sp)
            a, b = max(s.start, lo), min(s.end, hi)
            if a <= b:
                sessions |= set(sh.complete_sessions(sp, a, b)[0])
        for r in s.requirements:
            if lo <= r.trading_date <= hi:
                c = r.contract
                chains.setdefault((r.trading_date, c.expiry, c.instrument_type), {})[float(c.strike)] = c
                if r.role == u6.ATM:
                    atm[r.trading_date].add(c.instrument_key)
                files[c.instrument_key].add(s.layout.option_file(u6.month_label(r.trading_date), c))
        calendar |= set(s.calendar)
    spot = sd._clean(pd.concat(spots, ignore_index=True)) if spots else None
    if spot is None:
        spot = pd.DataFrame(columns=["timestamp", "close"])
    by_expiry: dict[date, tuple] = {}
    for r in roots:
        fl = nf.FuturesLayout(r, underlying)
        for c in nf.contract_rows_from_csv(fl):
            got = []
            if lo <= c.expiry <= hi:
                p = fl.contract_file(c)
                if p.is_file():
                    f = rd(p)
                    if len(f):
                        got = [f]
            prev = by_expiry.get(c.expiry)
            by_expiry[c.expiry] = (c, (prev[1] if prev else []) + got)
    contracts = sorted((c for c, _ in by_expiry.values()), key=lambda c: (c.expiry, c.instrument_key))
    frames = {c.instrument_key: _between(pd.concat(fs, ignore_index=True), lo, hi) for c, fs in by_expiry.values() if fs}
    frames = {k: v.reset_index(drop=True) for k, v in frames.items() if v is not None}
    cache: dict[str, pd.DataFrame | None] = {}

    def premium_fn(key: str) -> pd.DataFrame | None:
        if key not in cache:
            parts = [rd(p) for p in sorted(files.get(key, ())) if p.is_file()]
            parts = [p for p in parts if len(p)]
            cache[key] = _between(pd.concat(parts, ignore_index=True), lo, hi) if parts else None
        return cache[key]

    market = sd.MarketData(spot=spot, contracts=contracts, frames=frames, calendar=sorted(calendar), chains=chains,
                           atm=dict(atm), premium_fn=premium_fn, sessions=sorted(sessions),
                           roots=[Path(r).as_posix() for r in roots])
    return market, log


def read_summary(log: Sequence[dict]) -> dict[str, Any]:
    mx = [r["max"] for r in log if r["max"] is not None]
    mn = [r["min"] for r in log if r["min"] is not None]
    return {"files_opened": len(log), "rows_read": int(sum(r["rows"] for r in log)),
            "min_raw_timestamp": min(mn).isoformat() if mn else None,
            "max_raw_timestamp": max(mx).isoformat() if mx else None,
            "files_after_max_load": sorted({r["file"] for r in log if r["max"] is not None
                                            and r["max"].date() > MAX_LOAD})}


def phase_view(market: sd.MarketData, cutoff: date) -> sd.MarketData:
    """Same market truncated at ``cutoff`` (no later candle visible) - used so selection never sees April+ and April
    never sees May+."""
    base = market.premium_fn
    cache: dict[str, pd.DataFrame | None] = {}

    def premium_fn(key: str) -> pd.DataFrame | None:
        if key not in cache:
            p = base(key)
            cache[key] = None if p is None or p.empty else sd._clean(p[p["timestamp"].dt.date <= cutoff])
        return cache[key]

    spot = market.spot[market.spot["timestamp"].dt.date <= cutoff].reset_index(drop=True)
    frames = {k: v[v["timestamp"].dt.date <= cutoff].reset_index(drop=True) for k, v in market.frames.items()}
    return replace(market, spot=spot, frames={k: v for k, v in frames.items() if len(v)}, premium_fn=premium_fn,
                   sessions=[d for d in market.sessions if d <= cutoff], reads={"max_ts": None, "premium_calls": 0})


# ------------------------------------------------------------------ data-quality gate (S12 verdicts)
def _json(p: Path | None) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p is not None and p.is_file() else None


def s12_gate(summary: dict | None, manifest: dict | None, lo: date = CLEAN_START, hi: date = MAX_LOAD) -> dict:
    if not summary or not manifest:
        return {"passed": False, "checks": {}, "reason": "S12 summary / clean_history_manifest missing "
                                                          "(run scripts/run_s12_data_quality.py first)"}
    n, f, o = summary.get("nifty") or {}, summary.get("futures") or {}, summary.get("options") or {}
    cond = (summary.get("gate") or {}).get("conditions") or {}
    cov = summary.get("coverage_clean_history") or {}
    run = (summary.get("clean_history") or {}).get("ending_train_end") or {}
    win = [s for s in manifest.get("sessions") or [] if lo.isoformat() <= s["date"] <= hi.isoformat()]
    checks = {
        "nifty_pass": n.get("status") == PASS,
        "futures_pass": f.get("status") == PASS,
        "option_pass_or_quarantine_on_grid_only": o.get("status") in (PASS, "PASS_WITH_QUARANTINE")
        and not o.get("blocking_reasons"),
        "no_invalid_rows": not (o.get("invalid_rows") or 0) and not (n.get("invalid_ohlc") or 0)
        and not (f.get("invalid_ohlc") or 0),
        "causal_contract_selection_pass": cond.get("causal_contract_selection_pass") is True,
        "sufficient_atm_coverage": (cov.get("options_ce") or 0) >= MIN_S12_ATM_COVERAGE
        and (cov.get("options_pe") or 0) >= MIN_S12_ATM_COVERAGE,
        "no_unresolved_timestamp_corruption": cond.get("no_unresolved_timestamp_corruption") is True,
        "no_future_leakage": cond.get("no_material_leakage") is True,
        "raw_data_unchanged": summary.get("raw_unchanged") is True,
        "clean_run_starts_at_window_start": run.get("start") == lo.isoformat(),
        "every_window_session_clean": bool(win) and all(s["clean"] for s in win),
    }
    return {"passed": all(checks.values()), "checks": checks, "window_sessions": len(win),
            "clean_run": run, "s12_clean_history_sufficient_12m": cond.get("clean_history_sufficient"),
            "note": "S12 12-month gate is NOT met (9.1 months); this study is explicitly a constrained experiment",
            "off_grid_option_rows": "UNUSABLE (dropped at read time, never shifted)"}


# ------------------------------------------------------------------ helpers
def _iso(data: pd.DataFrame, lo, hi) -> list[str]:
    return s11._iso_days(data, lo, hi)


def _rows(data: pd.DataFrame, days: Sequence[str]) -> pd.DataFrame:
    return s11._rows(data, days)


def gate(m: dict | None, min_resolved: int) -> str:
    if not m or m["resolved"] < min_resolved:
        return INSUFFICIENT
    pf, ex = m["profit_factor"], m["expectancy_r"]
    return PASS if ex is not None and ex > 0 and pf is not None and pf > 1 else FAIL


def run_folds(data: pd.DataFrame, cols: Sequence[str], grid: np.ndarray, start: str,
              say: Callable[[str], None]) -> list[dict]:
    out = []
    for name, tr_end, va_lo, va_hi in FOLDS:
        trd, vad = _iso(data, start, tr_end), _iso(data, va_lo, va_hi)
        base = {"fold": name, "train": [start, tr_end.isoformat()], "validation": [va_lo.isoformat(), va_hi.isoformat()],
                "train_days": len(trd), "validation_days": len(vad)}
        if len(trd) < MIN_FOLD_TRAIN_SESSIONS or len(vad) < MIN_FOLD_VAL_SESSIONS:
            out.append({**base, "status": NOT_RUN, "reason": f"needs >= {MIN_FOLD_TRAIN_SESSIONS} train and >= "
                                                             f"{MIN_FOLD_VAL_SESSIONS} validation sessions"})
            continue
        say(f"  {name}: train {start}..{tr_end} ({len(trd)} days), validate {va_lo}..{va_hi} ({len(vad)} days)")
        val = _rows(data, vad)
        fits = {}
        for fam in FAMILIES:
            m = sm.fit_model(data, cols, trd, {s: fam for s in sd.SIDES}, grid, test_days=vad)
            models = {s: m for s in sd.SIDES}
            fits[fam] = {"model": m, "metrics": sm.evaluate_models(val, models, sm.THRESHOLDS[0]),
                         "trading": {t: s11.evaluate(val, models, t)["trading"] for t in sm.THRESHOLDS},
                         "importance": {s: s11.importance(m, s, val) for s in sd.SIDES}}
        out.append({**base, "status": RUN, "_fits": fits, "_train_days": trd, "_val_days": vad})
    return out


def side_verdict(*, checks_ok: bool, usable: bool, stability: str, inner_positive: int, folds_run: int,
                 val: dict | None, oos: dict | None, oos_ran: bool, degraded: bool) -> tuple[str, str]:
    if not checks_ok:
        return REJECTED, "lookahead / leakage / data-integrity check failed"
    if not usable:
        return INSUFF, "side model not fitted / calibrated (too few TRAIN samples)"
    if stability != STABLE:
        return UNSTABLE, f"inner folds {stability}"
    if folds_run and inner_positive < math.ceil(folds_run / 2):
        return UNSTABLE, f"only {inner_positive}/{folds_run} inner folds with positive expectancy"
    vg = gate(val, MIN_VAL_RESOLVED)
    if vg == FAIL:
        return REJECTED, "April validation PF <= 1 or expectancy <= 0"
    if vg == INSUFFICIENT:
        return INSUFF, f"fewer than {MIN_VAL_RESOLVED} resolved April trades"
    if not oos_ran:
        return INSUFF, "May-June OOS not run (combined April validation not acceptable)"
    og = gate(oos, MIN_OOS_RESOLVED)
    if og == FAIL:
        return REJECTED, "May-June OOS PF <= 1 or expectancy <= 0"
    if og == INSUFFICIENT:
        return INSUFF, f"fewer than {MIN_OOS_RESOLVED} resolved May-June trades"
    if degraded:
        return REJECTED, f"inner-fold -> April AUC degradation > {MAX_AUC_DEGRADATION}"
    if oos["max_drawdown_r"] > MAX_DD_R:
        return REJECTED, f"May-June drawdown > {MAX_DD_R}R"
    return PROMISING, "stable inner folds, April and May-June passed with >= 30 resolved trades"


def final_decision(*, checks_ok: bool, any_usable: bool, val_label: str, oos_label: str, inner_trading_stable: bool,
                   sides: dict[str, str], oos: dict | None) -> tuple[str, str]:
    if not checks_ok:
        return REJECTED, "lookahead / leakage / data-integrity check failed"
    if not any_usable:
        return INSUFF, "not enough TRAIN samples to fit and calibrate a side model"
    if val_label == FAIL:
        return REJECTED, "April validation failed (PF <= 1 or expectancy <= 0); not rescued by tuning"
    if oos_label == FAIL:
        return REJECTED, "May-June OOS failed (PF <= 1 or expectancy <= 0)"
    if not inner_trading_stable or all(v == UNSTABLE for v in sides.values()):
        return UNSTABLE, "inner folds unstable"
    if val_label == INSUFFICIENT:
        return INSUFF, f"fewer than {MIN_VAL_RESOLVED} resolved April trades - OOS not run"
    if oos_label in (INSUFFICIENT, NOT_RUN):
        return INSUFF, f"fewer than {MIN_OOS_RESOLVED} resolved May-June trades"
    if oos["max_drawdown_r"] > MAX_DD_R or (oos["coverage"] or 0) < MIN_COVERAGE:
        return REJECTED, "May-June drawdown too large or coverage too low"
    if any(v == PROMISING for v in sides.values()):
        good = [s for s, v in sides.items() if v == PROMISING]
        return PROMISING, (f"{'/'.join(good)} promising on a 9.1-month constrained sample - candidate for fresh-paper "
                           "validation only")
    if any(v == UNSTABLE for v in sides.values()):
        return UNSTABLE, "combined passed but no side individually stable and promising"
    return INSUFF, "combined passed but no side individually reached the sample / quality bar"


# ------------------------------------------------------------------ checks
def leakage_checks(data: pd.DataFrame, *, final: sm.S10Model, fits: Sequence[tuple], start: str,
                   thr_rows: Sequence[dict], threshold: float, grid: np.ndarray, oos_days: Sequence[str],
                   stages: Sequence[str]) -> dict[str, dict]:
    """fits: (name, model, train_end_iso, test_days)."""
    te = TRAIN_END.isoformat()
    contamination = {n: s11.ids_ok(data, m, start, e) for n, m, e, _ in fits}
    scaler = {n: s11.preprocessor_check(data, m) for n, m, _, _ in fits}
    calib = {n: s11.calibration_check(data, m, e) for n, m, e, _ in fits}
    purge_bad = []
    for n, m, _, test_days in fits:
        test = _rows(data, test_days)
        for label, idx in (("fit", m.pre_fit_ids), *((f"calib_{s}", m.sides[s].calib_ids) for s in sd.SIDES)):
            if idx and sm.purge_violations(data.loc[list(idx)], test, grid):
                purge_bad.append(f"{n}|{label}")
    input_days = thr_rows[0]["input_days"] if thr_rows else []
    redo, _ = s11.choose_threshold(thr_rows)
    oos_ids = set(data.index[data["session_date"].isin(set(oos_days))])
    used: set = set()
    for _, m, _, _ in fits:
        used |= set(m.pre_fit_ids)
        for s in sd.SIDES:
            used |= set(m.sides[s].fit_ids) | set(m.sides[s].calib_ids)
    st = list(stages)
    order_ok = "oos_dataset" not in st or (st.index("freeze") < st.index("validation_april") < st.index("oos_dataset"))
    return {
        "train_oos_contamination": {"checks": contamination, "passed": all(contamination.values())},
        "scaler_leakage": {"checks": scaler, "note": "imputer medians and scaler means recomputed from the fit rows",
                           "passed": all(scaler.values())},
        "calibration_leakage": {"checks": calib, "passed": all(calib.values())},
        "threshold_leakage": {"input_days": f"{input_days[0] if input_days else '-'}..{input_days[-1] if input_days else '-'}",
                              "recomputed": redo, "frozen": threshold,
                              "passed": bool(input_days) and all(d <= te for d in input_days) and redo == threshold},
        "purge_embargo_correctness": {"fits_checked": len(fits), "embargo_bars": sm.EMBARGO_BARS,
                                      "label_horizon_bars": HORIZON, "violations": purge_bad,
                                      "passed": not purge_bad and sm.EMBARGO_BARS >= HORIZON},
        "oos_untouched_during_selection": {"oos_rows_in_any_fit": len(oos_ids & used), "stage_order_ok": order_ok,
                                           "passed": not (oos_ids & used) and order_ok},
    }


def max_loaded_check(market: sd.MarketData, log: Sequence[dict]) -> dict[str, Any]:
    r = read_summary(log)
    spot_max = market.spot["timestamp"].max() if len(market.spot) else None
    fut_max = max((f["timestamp"].max() for f in market.frames.values() if len(f)), default=None)
    mx = [t for t in (r["max_raw_timestamp"] and pd.Timestamp(r["max_raw_timestamp"]), spot_max, fut_max) if t]
    top = max(mx) if mx else None
    roots_ok = not any(w in Path(x).name.lower() for x in market.roots for w in ("fresh", "oct"))
    ok = bool(roots_ok and top is not None and top.date() <= MAX_LOAD and not r["files_after_max_load"]
              and all(d <= MAX_LOAD for d in market.sessions))
    return {**r, "max_loaded_timestamp": top.isoformat() if top is not None else None,
            "max_loaded_date": top.date().isoformat() if top is not None else None, "limit": MAX_LOAD.isoformat(),
            "roots": market.roots, "passed": ok}


# ------------------------------------------------------------------ the study
def not_run(base: dict, reason: str, decision: str = INSUFF) -> dict[str, Any]:
    return {**base, "ran": False, "reason": reason, "decision": decision, "decision_reason": reason,
            "forward_paper": "NOT SAFE", "target_70": f"{TARGET_70_TEXT} / {INSUFFICIENT}", "lookahead_ok": None,
            "leakage_ok": None, "folds": [], "results": {}, "side_verdicts": {},
            "labels": {"VALIDATION": NOT_RUN, "OOS": NOT_RUN}, "stages": base.get("stages", [])}


def _base(availability: dict, data_quality: dict, flags: dict, cols: list[str], read_log: Sequence[dict]) -> dict:
    return {"heading": HEADING, "strategy": STRATEGY, "banner": BANNER, "disclaimer": DISCLAIMER, "label": LABEL,
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"), "flags": flags, "feature_columns": cols,
            "availability": availability, "data_quality": data_quality, "october": "NOT LOADED", "july_plus": "NOT LOADED",
            "production_unchanged": None, "live_orders": 0, "trading_mode": "PAPER", "live_trading_enabled": False,
            "clean_history": {"range": [CLEAN_START.isoformat(), MAX_LOAD.isoformat()], "months": CLEAN_MONTHS,
                              "not_12_month_validation": True},
            "splits": {"TRAIN": [CLEAN_START.isoformat(), TRAIN_END.isoformat()],
                       "VALIDATION": [x.isoformat() for x in VALIDATION], "OOS": [x.isoformat() for x in OOS],
                       "NOT_LOADED": "2026-07-01 onward (July / August / September contaminated; October fresh)"},
            "folds_plan": [{"fold": n, "train": [CLEAN_START.isoformat(), a.isoformat()],
                            "validation": [b.isoformat(), c.isoformat()]} for n, a, b, c in FOLDS],
            "read_summary": read_summary(read_log)}


def build_study(*, market: sd.MarketData, read_log: list[dict], availability: dict, data_quality: dict,
                freeze_dir: Path, comparison_sources: dict[str, Path] | None = None, mutation_days: int = 6,
                check_sample: int = 20, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    stages: list[str] = []
    flags = {"volume": bool(availability.get("use_option_volume")), "oi": bool(availability.get("use_option_oi"))}
    cols = sd.feature_columns(flags["volume"], flags["oi"])
    base = _base(availability, data_quality, flags, cols, read_log)
    base["stages"] = stages
    if not data_quality.get("passed"):
        return not_run(base, f"DATA_QUALITY_GATE_FAILED: {data_quality.get('reason') or data_quality.get('checks')}")
    if not availability.get("required_ok"):
        missing = [k for k, v in (availability.get("required") or {}).items() if not v]
        return not_run(base, f"required data unavailable: {', '.join(missing)}")

    say("building the TRAIN dataset from a view truncated at 2026-03-31 (April+ invisible) ...")
    stages.append("train_dataset")
    sel = phase_view(market, TRAIN_END)
    built = sd.build_samples(sel, sd.days_in(sel.sessions, CLEAN_START, TRAIN_END), volume=flags["volume"],
                             oi=flags["oi"], caches={"fut": {}, "opt": {}}, target_r=TARGET_R)
    start = CLEAN_START.isoformat()
    train = built.data[(built.data["session_date"] >= start)
                       & (built.data["session_date"] <= TRAIN_END.isoformat())].reset_index(drop=True)
    n_train = len(train)
    base.update(build_funnel=dict(built.funnel))
    if train.empty:
        return not_run(base, "no eligible TRAIN samples")
    sel_grid = sel.grid()

    say("three inner walk-forward folds inside TRAIN (April+ never touched) ...")
    stages.append("inner_folds")
    folds = run_folds(train, cols, sel_grid, start, say)
    run = [f for f in folds if f["status"] == RUN]
    stab0 = {(fam, s): s11.family_stability(folds, fam, s, sm.THRESHOLDS[0]) for fam in FAMILIES for s in sd.SIDES}
    chosen = {s: s11.choose_family([stab0[(fam, s)] for fam in FAMILIES]) for s in sd.SIDES}
    fams = {s: chosen[s][0] for s in sd.SIDES}
    stages.append("threshold_selection")
    thr_rows, thr_folds = s11.threshold_search(train, folds, fams) if run else ([], {})
    thr, thr_why = s11.choose_threshold(thr_rows) if thr_rows else (min(sm.THRESHOLDS), "NO_INNER_FOLDS")
    stability = [s11.family_stability(folds, fam, s, thr) for fam in FAMILIES for s in sd.SIDES]
    chosen_stab = {s: next(r for r in stability if r["family"] == fams[s] and r["side"] == s) for s in sd.SIDES}
    fold_eval = {r["fold"]: r["ev"] for r in thr_folds.get(thr, [])}
    pos_folds = sum(1 for e in fold_eval.values() if (e["trading"]["COMBINED"]["expectancy_r"] or 0) > 0)
    side_pos = {s: sum(1 for e in fold_eval.values() if (e["trading"][s]["expectancy_r"] or 0) > 0) for s in sd.SIDES}
    inner_trading_stable = len(run) >= s11.MIN_STABILITY_FOLDS and pos_folds >= math.ceil(len(run) / 2)
    selection_max_read = sel.reads["max_ts"]

    say("building April rows (view truncated at 2026-04-30; May-June invisible) ...")
    stages.append("validation_dataset")
    vview = phase_view(market, VALIDATION[1])
    vbuilt = sd.build_samples(vview, sd.days_in(vview.sessions, *VALIDATION), volume=flags["volume"], oi=flags["oi"],
                              caches={"fut": {}, "opt": {}}, target_r=TARGET_R)
    data = pd.concat([train, vbuilt.data], ignore_index=True)
    train_days, val_days = _iso(data, start, TRAIN_END), _iso(data, *VALIDATION)
    vgrid = vview.grid()

    say("freezing (final fit on the whole TRAIN) ...")
    stages.append("freeze")
    final = sm.fit_model(data, cols, train_days, fams, vgrid, test_days=val_days)
    frozen_fp = sm.fingerprint(final, thr)
    models = {s: final for s in sd.SIDES}
    meta = {"strategy": STRATEGY, "research_only": True, "label": LABEL, "families": fams,
            "family_reason": {s: chosen[s][1] for s in sd.SIDES}, "threshold": thr, "threshold_reason": thr_why,
            "margin": sm.MARGIN, "ev_gate": f"EV = {TARGET_R} * P - (1 - P) > 0", "embargo_bars": sm.EMBARGO_BARS,
            "calibration": {s: final.sides[s].calibration for s in sd.SIDES},
            "status": {s: final.sides[s].status for s in sd.SIDES}, "train": [start, TRAIN_END.isoformat()],
            "fit_days": [final.fit_days[0], final.fit_days[-1]] if final.fit_days else [],
            "calibration_days": [final.calib_days[0], final.calib_days[-1]] if final.calib_days else [],
            "purge": final.purge, "fingerprint": frozen_fp, "sklearn_version": sklearn.__version__,
            "frozen_at": datetime.now(IST).isoformat(timespec="seconds"),
            "note": "research artifact only - never loaded by production execution"}
    artifacts = s11.save_artifacts(final, thr, meta, freeze_dir)

    results: dict[str, dict] = {"TRAIN": s11.evaluate(_rows(data, train_days), models, thr)}
    say("April 2026 VALIDATION (frozen) ...")
    stages.append("validation_april")
    results["VALIDATION"] = s11.evaluate(_rows(data, val_days), models, thr)
    val_label = gate(results["VALIDATION"]["trading"]["COMBINED"], MIN_VAL_RESOLVED)
    oos_ran = val_label == PASS
    oos_label, oos_reason, oos_days, obuilt = NOT_RUN, None, [], None
    if oos_ran:
        say("May-June 2026 OOS (frozen, no modification) ...")
        stages.append("oos_dataset")
        obuilt = sd.build_samples(market, sd.days_in(market.sessions, *OOS), volume=flags["volume"], oi=flags["oi"],
                                  caches={"fut": {}, "opt": {}}, target_r=TARGET_R)
        data = pd.concat([data, obuilt.data], ignore_index=True)
        oos_days = _iso(data, *OOS)
        stages.append("oos_may_june")
        results["OOS"] = s11.evaluate(_rows(data, oos_days), models, thr)
        oos_label = gate(results["OOS"]["trading"]["COMBINED"], MIN_OOS_RESOLVED)
    else:
        oos_reason = f"April validation {val_label} - OOS NOT_RUN"
    fp_after = sm.fingerprint(final, thr)

    say("feature importance (reported only; nothing removed) ...")
    imp_rows = []
    for f in run:
        for fam, fit in f["_fits"].items():
            for s in sd.SIDES:
                method, imp = fit["importance"][s]
                for rank, c in enumerate(sm.top_k(imp, TOP_FEATURES), 1):
                    imp_rows.append({"scope": f["fold"], "family": fam, "side": s, "method": method, "rank": rank,
                                     "feature": c, "value": imp[c], "chosen_family": fam == fams[s]})
    for s in sd.SIDES:
        method, imp = s11.importance(final, s, _rows(data, train_days))
        for rank, c in enumerate(sm.top_k(imp, TOP_FEATURES), 1):
            imp_rows.append({"scope": "FINAL_TRAIN", "family": fams[s], "side": s, "method": method, "rank": rank,
                             "feature": c, "value": imp[c], "chosen_family": True})

    say("anti-lookahead / leakage checks ...")
    mdays = s4.sample_days(train_days, mutation_days) if mutation_days else []
    src_all = {"spot": True, "futures": True, "options": True}
    src_opt = {"spot": False, "futures": False, "options": True}
    tr = _rows(data, train_days)
    lookahead = {
        "future_candle_scrambling": s10.mutation_check(sel, data, cols, mdays, flags, final, seed=11, sources=src_all,
                                                       price=True, volume=True, oi=True),
        "feature_truncation": s10.truncation_check(sel, tr, cols, flags, check_sample),
        "future_option_mutation": s10.mutation_check(sel, data, cols, mdays, flags, final, seed=31, sources=src_opt,
                                                     price=True, volume=True),
        "rolling_feature_causality": s11.rolling_causality_check(sel, tr, check_sample),
        "contract_selection_causality": {
            "options": s10.contract_causality_check(sel, tr, check_sample),
            "futures": s11.futures_causality_check(sel, tr, [date.fromisoformat(d) for d in train_days])},
    }
    cs = lookahead["contract_selection_causality"]
    cs["passed"] = bool(cs["options"]["passed"] and cs["futures"]["passed"])
    fits = [(f"{f['fold']}_{fam}", fit["model"], f["train"][1], f["_val_days"])
            for f in run for fam, fit in f["_fits"].items()]
    fits.append(("FINAL", final, TRAIN_END.isoformat(), val_days))
    leakage = leakage_checks(data, final=final, fits=fits, start=start, thr_rows=thr_rows, threshold=thr,
                             grid=vgrid, oos_days=oos_days, stages=stages)
    ends = pd.DatetimeIndex(train["event_end"])
    sel_ok = bool(len(train)) and train["session_date"].max() <= TRAIN_END.isoformat() \
        and ends.max().date() <= TRAIN_END
    tc = leakage["train_oos_contamination"]
    tc.update(selection_rows_and_labels_end_by_train_end=sel_ok, fingerprint_unchanged=fp_after == frozen_fp,
              selection_max_option_read=selection_max_read.isoformat() if selection_max_read is not None else None)
    tc["passed"] = bool(tc["passed"] and sel_ok and fp_after == frozen_fp
                        and (selection_max_read is None or selection_max_read.date() <= TRAIN_END))
    again = sm.fit_model(data, cols, train_days, fams, vgrid, test_days=val_days)
    rerun = s11.evaluate(_rows(data, val_days), {s: again for s in sd.SIDES}, thr)
    w0, w1 = results["VALIDATION"]["wide"], rerun["wide"]
    same_p = all(np.array_equal(w0[f"p_{s}"].to_numpy(), w1[f"p_{s}"].to_numpy(), equal_nan=True) for s in sd.SIDES)
    key = lambda ts: [json.dumps(t, sort_keys=True, default=str) for t in ts]  # noqa: E731
    same_t = key(results["VALIDATION"]["trades"]) == key(rerun["trades"])
    same_fp = sm.fingerprint(again, thr) == frozen_fp
    lookahead["deterministic_replay"] = {"probabilities_identical": same_p, "trades_identical": same_t,
                                         "fingerprint_identical": same_fp, "passed": bool(same_p and same_t and same_fp)}
    lookahead["duplicate_trade_prevention"] = s10.duplicate_trade_check(
        {**{k: v["trades"] for k, v in results.items()}, **{f"FOLD_{k}": v["trades"] for k, v in fold_eval.items()}})
    lookahead["max_loaded_timestamp"] = max_loaded_check(market, read_log)
    lookahead_ok = all(v["passed"] for v in lookahead.values())
    leakage_ok = all(v["passed"] for v in leakage.values())
    ds = s10.dataset_checks(data)
    cv = s11.contract_validity(market, data)
    data_ok = bool(data_quality.get("passed")) and ds["passed"] and cv["passed"] and bool(availability["required_ok"])
    checks_ok = bool(lookahead_ok and leakage_ok and data_ok)

    trading = {k: v["trading"] for k, v in results.items()}
    val_auc = {s: results["VALIDATION"]["model"][s]["roc_auc"] for s in sd.SIDES}
    degradation = {s: (chosen_stab[s]["auc_mean"] - val_auc[s]) if chosen_stab[s]["auc_mean"] is not None
                   and val_auc[s] is not None else None for s in sd.SIDES}
    side_verdicts = {}
    for s in sd.SIDES:
        v, why = side_verdict(checks_ok=checks_ok, usable=s11.usable(final, s), stability=chosen_stab[s]["status"],
                              inner_positive=side_pos[s], folds_run=len(run), val=trading["VALIDATION"][s],
                              oos=trading.get("OOS", {}).get(s), oos_ran=oos_ran,
                              degraded=degradation[s] is not None and degradation[s] > MAX_AUC_DEGRADATION)
        side_verdicts[s] = {"verdict": v, "reason": why, "family": fams[s], "stability": chosen_stab[s]["status"],
                            "inner_positive_folds": side_pos[s], "auc_degradation": degradation[s]}
    oos_c = trading.get("OOS", {}).get("COMBINED")
    decision, why = final_decision(checks_ok=checks_ok, any_usable=any(s11.usable(final, s) for s in sd.SIDES),
                                   val_label=val_label, oos_label=oos_label, inner_trading_stable=inner_trading_stable,
                                   sides={s: v["verdict"] for s, v in side_verdicts.items()}, oos=oos_c)
    forward = "SAFE" if decision == PROMISING and checks_ok else "NOT SAFE"
    if oos_c is None or oos_c["resolved"] < MIN_OOS_RESOLVED:
        target = f"{TARGET_70_TEXT} / {INSUFFICIENT}"
    else:
        target = f"{TARGET_70_TEXT} (May-June win rate {oos_c['accuracy']:.1%}, {oos_c['resolved']} resolved)"

    comparison = comparison_rows(results, comparison_sources or {})
    return {
        **base, "ran": True, "stages": stages,
        "train": {"range": [start, TRAIN_END.isoformat()], "trading_days": len(train_days)},
        "dataset": {"train_rows": n_train, "april_rows": int(len(vbuilt.data)),
                    "oos_rows": int(len(obuilt.data)) if obuilt is not None else 0, "rows": int(len(data)),
                    "train_timestamps": int(train["signal_time"].nunique()),
                    "label_rows": int(data["outcome"].isin(sd.BINARY).sum()), "labels": sd.label_counts(data),
                    "class_balance": s11.class_balance(data, train_days),
                    "build_funnel_train": dict(built.funnel), "build_funnel_april": dict(vbuilt.funnel),
                    "build_funnel_oos": dict(obuilt.funnel) if obuilt is not None else None,
                    "option_contracts": sorted(built.option_contracts | vbuilt.option_contracts
                                               | (obuilt.option_contracts if obuilt is not None else set())),
                    "futures_contracts": sorted(built.futures_contracts | vbuilt.futures_contracts
                                                | (obuilt.futures_contracts if obuilt is not None else set())),
                    "sessions": {"TRAIN": len(train_days), "VALIDATION": len(val_days), "OOS": len(oos_days)},
                    "hash": sd.dataset_hash(data, cols)},
        "folds": [s11.fold_summary(f, fams, thr, fold_eval) for f in folds], "stability": stability,
        "chosen_stability": chosen_stab, "inner_positive_folds": pos_folds, "side_inner_positive_folds": side_pos,
        "inner_trading_stable": inner_trading_stable, "families": fams,
        "family_reason": {s: chosen[s][1] for s in sd.SIDES}, "threshold": thr, "threshold_reason": thr_why,
        "threshold_rows": [{k: v for k, v in r.items() if k != "input_days"} for r in thr_rows],
        "frozen": {"fingerprint": frozen_fp, "artifacts": artifacts, "metadata": meta},
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "wide"} for k, v in results.items()},
        "labels": {"VALIDATION": val_label, "OOS": oos_label}, "not_run_reasons": {"OOS": oos_reason},
        "degradation": degradation, "importance": imp_rows, "calibration": calibration_rows(fold_eval, results),
        "comparison": comparison, "lookahead": lookahead, "lookahead_ok": lookahead_ok, "leakage": leakage,
        "leakage_ok": leakage_ok, "dataset_checks": ds, "contract_validity": cv, "data_ok": data_ok,
        "side_verdicts": side_verdicts, "target_70": target, "decision": decision, "decision_reason": why,
        "forward_paper": forward, "_data": data, "_market": market,
    }


def calibration_rows(fold_eval: dict, results: dict) -> list[dict]:
    out = []
    scopes = [(k, v["model"]) for k, v in fold_eval.items()]
    scopes += [(SPLIT_NAMES[k], results[k]["model"]) for k in ("VALIDATION", "OOS") if k in results]
    for name, m in scopes:
        for s in SCOPES:
            r = m[s]
            out.append({"scope": name, "side": s, "bin": "SUMMARY", "samples": r["samples"], "brier": r["brier"],
                        "log_loss": r["log_loss"], "ece": r["ece"], "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"]})
            for b in r["calibration"]:
                out.append({"scope": name, "side": s, "bin": b["bin"], "lo": b["lo"], "hi": b["hi"],
                            "samples": b["count"], "mean_predicted": b["mean_p"], "actual_win_rate": b["win_rate"]})
    return out


# ------------------------------------------------------------------ descriptive comparison (saved reports only)
MONTHS = (("2026-04", "APRIL"), ("2026-05", "MAY"), ("2026-06", "JUNE"))
CMP_KEYS = ("trades", "resolved", "wins", "losses", "accuracy", "profit_factor", "expectancy_r", "net_pnl",
            "max_drawdown")


def _read_csv(p: Path | None) -> list[dict]:
    if p is None or not Path(p).is_file():
        return []
    with Path(p).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) or x == math.inf else None


def _resolved(r: dict) -> float | None:
    w, l = _num(r.get("wins")), _num(r.get("losses"))
    return None if w is None or l is None else w + l


def _cmp(period: str, system: str, relation: str, source: str, vals: dict | None, note: str = "") -> dict[str, Any]:
    vals = vals or {}
    return {"period": period, "system": system, "relation_to_period": relation, "source": source,
            **{k: vals.get(k) for k in CMP_KEYS}, "status": "AVAILABLE" if vals else "NOT_AVAILABLE", "note": note}


def _trade_vals(trades: Sequence[dict] | None) -> dict[str, Any] | None:
    if trades is None:
        return None
    m = sm.trade_metrics(trades, 0)
    return {k: m.get(k) for k in CMP_KEYS}


def comparison_rows(results: dict, sources: dict[str, Path]) -> list[dict]:
    """Descriptive only: S12.2 (this run) next to previously saved S2 / S9 / S10 / S11 monthly results for the same
    calendar months. No market data is loaded and nothing here influences any S12.2 choice."""
    out = []
    mine = {k: [t for t in v["trades"]] for k, v in results.items() if k in ("VALIDATION", "OOS")}
    s2 = {r["month"]: r for r in _read_csv(sources.get("S2")) if r.get("scope") == "COMBINED"}
    s9 = {r["month"]: r for r in _read_csv(sources.get("S9")) if r.get("scope") == "COMBINED"}
    s10r = {r["month"]: r for r in _read_csv(sources.get("S10")) if r.get("scope") == "COMBINED"}
    s11t = _read_csv(sources.get("S11"))
    s11_ok = sources.get("S11") is not None and Path(sources["S11"]).is_file()
    for iso, name in MONTHS:
        split = "VALIDATION" if iso == "2026-04" else "OOS"
        tr = [t for t in mine.get(split, []) if str(t["session_date"])[:7] == iso]
        out.append(_cmp(name, STRATEGY, "APRIL VALIDATION (frozen)" if split == "VALIDATION" else "MAY-JUNE OOS (frozen)",
                        "this run", _trade_vals(tr) if split in mine else None,
                        "" if split in mine else "OOS NOT_RUN"))
        r = s2.get(iso)
        out.append(_cmp(name, "S2", "frozen rules, period previously researched (not pristine)", "clean_historical_backtest",
                        {"trades": _num(r.get("total_trades")), "resolved": _resolved(r),
                         "wins": _num(r.get("wins")), "losses": _num(r.get("losses")),
                         "accuracy": _num(r.get("win_rate")), "profit_factor": _num(r.get("profit_factor")),
                         "expectancy_r": _num(r.get("average_r")), "net_pnl": _num(r.get("net_pnl")),
                         "max_drawdown": _num(r.get("max_drawdown"))} if r else None, "different label / exits"))
        r = s9.get(name)
        out.append(_cmp(name, "S9_FUTURES_VOLUME", "REJECTED rules, period previously researched", "s9_futures_volume",
                        {"trades": _num(r.get("trades")), "resolved": _resolved(r), "wins": _num(r.get("wins")),
                         "losses": _num(r.get("losses")), "accuracy": _num(r.get("accuracy")),
                         "profit_factor": _num(r.get("profit_factor")), "net_pnl": _num(r.get("net_pnl")),
                         "max_drawdown": _num(r.get("max_drawdown"))} if r else None, "different label / exits"))
        r = s10r.get(name)
        out.append(_cmp(name, "S10_SHORT_HISTORY", "IN-SAMPLE (S10 was trained on Apr-Jun 2026)", "s10_ml",
                        {k: _num(r.get(k)) for k in CMP_KEYS} if r else None, "2R label; in-sample - not comparable"))
        t11 = [{"signal_time": pd.Timestamp(t["signal_time"]), "side": t.get("side"), "outcome": t.get("outcome"),
                "pnl": _num(t.get("pnl")) or 0.0, "r_multiple": _num(t.get("r_multiple")) or 0.0,
                "holding_minutes": _num(t.get("holding_minutes"))}
               for t in s11t if str(t.get("session_date", ""))[:7] == iso and t.get("signal_time")]
        out.append(_cmp(name, "S11_EXPANDED", "IN-SAMPLE (S11 TRAIN ended 2026-06-30)", "s11_ml_expanded",
                        _trade_vals(t11) if s11_ok else None, "in-sample - not comparable"))
    return out


# ------------------------------------------------------------------ manifest / reports
def manifest(a: dict, market: sd.MarketData, store_info: dict) -> dict[str, Any]:
    ds = a.get("dataset", {})
    used = ds.get("option_contracts", [])
    fut = [{"instrument_key": c.instrument_key, "symbol": c.trading_symbol, "expiry": c.expiry.isoformat(),
            "first": str(market.frames[c.instrument_key]["timestamp"].min()) if c.instrument_key in market.frames else None,
            "last": str(market.frames[c.instrument_key]["timestamp"].max()) if c.instrument_key in market.frames else None}
           for c in market.contracts if c.instrument_key in market.frames]
    return {"banner": BANNER, "clean_history": a.get("clean_history"), "splits": a.get("splits"),
            "folds": a.get("folds_plan"), "october": "NOT LOADED", "july_plus": "NOT LOADED",
            "sessions_loaded": len(market.sessions),
            "first_session": market.sessions[0].isoformat() if market.sessions else None,
            "last_session": market.sessions[-1].isoformat() if market.sessions else None,
            "sessions_by_split": ds.get("sessions"), "roots": market.roots, "read_summary": a.get("read_summary"),
            "max_loaded": (a.get("lookahead") or {}).get("max_loaded_timestamp"),
            "futures_contracts": fut, "futures_candles": int(sum(len(f) for f in market.frames.values())),
            "option_contracts_used": len(used), "feature_rows": ds.get("rows"), "label_rows": ds.get("label_rows"),
            "rows_by_split": {"TRAIN": ds.get("train_rows"), "VALIDATION": ds.get("april_rows"), "OOS": ds.get("oos_rows")},
            "dataset_hash": ds.get("hash"), "feature_columns": a.get("feature_columns"), "stores": store_info}


REPORT_FILES = ("summary.json", "constrained_report.md", "dataset_manifest.json", "data_quality.json",
                "fold_results.csv", "model_results.csv", "calibration.csv", "threshold_results.csv",
                "coverage_funnel.csv", "april_validation.csv", "may_june_oos.csv", "feature_importance.csv",
                "trade_log.csv", "comparison_s2_s9_s10_s11_s12_2.csv")


def split_rows(a: dict, split: str) -> list[dict]:
    r = a.get("results", {}).get(split)
    if not r:
        return [{"split": SPLIT_NAMES[split], "status": NOT_RUN,
                 "reason": (a.get("not_run_reasons") or {}).get(split) or a.get("reason")}]
    return [{"split": SPLIT_NAMES[split], "label": a["labels"].get(split), "scope": sc,
             "verdict": (a.get("side_verdicts", {}).get(sc) or {}).get("verdict", a.get("decision")),
             **{k: r["model"][sc][k] for k in s10.METRIC_KEYS}, **{k: r["trading"][sc][k] for k in s10.TRADE_KEYS}}
            for sc in SCOPES]


def funnel_rows(a: dict) -> list[dict]:
    ds = a.get("dataset", {})
    out = [{"scope": f"DATASET_BUILD_{k}", **(ds.get(f"build_funnel_{k.lower()}") or {})}
           for k in ("TRAIN", "APRIL", "OOS") if ds.get(f"build_funnel_{k.lower()}")]
    for f in a.get("folds", []):
        if f.get("chosen"):
            out.append({"scope": f["fold"], "threshold": a["threshold"], **f["chosen"]["funnel"]})
    for k, v in a.get("results", {}).items():
        out.append({"scope": SPLIT_NAMES[k], "threshold": a["threshold"], **v["funnel"]})
    return out


def trade_rows(a: dict) -> list[dict]:
    out = []
    for k, v in a.get("results", {}).items():
        for t in v["trades"]:
            out.append({c: (SPLIT_NAMES[k] if c == "split" else t.get(c)) for c in s11.TRADE_COLS})
    return out


def write_reports(a: dict, out_dir: Path, manifest_obj: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    js = lambda path, obj: path.write_text(json.dumps(s6._jsonable(obj), indent=2, default=str),  # noqa: E731
                                           encoding="utf-8")
    js(p["summary.json"], s11._strip(a))
    js(p["dataset_manifest.json"], manifest_obj)
    js(p["data_quality.json"], {"s12_gate": a.get("data_quality"), "dataset_checks": a.get("dataset_checks"),
                                "contract_validity": a.get("contract_validity"), "availability": a.get("availability"),
                                "max_loaded": (a.get("lookahead") or {}).get("max_loaded_timestamp"),
                                "data_ok": a.get("data_ok")})
    s6._csv(p["fold_results.csv"], s11.fold_rows(a) if a.get("ran") else [])
    s6._csv(p["model_results.csv"], a.get("stability", []))
    s6._csv(p["calibration.csv"], a.get("calibration", []))
    s6._csv(p["threshold_results.csv"], [{**r, "chosen": r["threshold"] == a.get("threshold")}
                                         for r in a.get("threshold_rows", [])])
    s6._csv(p["coverage_funnel.csv"], funnel_rows(a))
    s6._csv(p["april_validation.csv"], split_rows(a, "VALIDATION"))
    s6._csv(p["may_june_oos.csv"], split_rows(a, "OOS"))
    s6._csv(p["feature_importance.csv"], a.get("importance", []))
    s6._csv(p["trade_log.csv"], trade_rows(a))
    s6._csv(p["comparison_s2_s9_s10_s11_s12_2.csv"], a.get("comparison", []))
    p["constrained_report.md"].write_text(markdown(a), encoding="utf-8")
    return p


# ------------------------------------------------------------------ text
def _ok(v) -> str:
    return "N/A (not run)" if v is None else ("PASS" if v else "FAIL")


def terminal_text(a: dict) -> str:
    out = [HEADING, f"({DISCLAIMER})", "", "CLEAN HISTORY:", f"{CLEAN_START} -> {MAX_LOAD}", f"{CLEAN_MONTHS} months",
           "", "TRAIN:", f"{CLEAN_START} -> {TRAIN_END}", "", "VALIDATION:", "April 2026", "", "OOS:", "May-June 2026",
           "", "JULY+:", "NOT LOADED", "", "OCTOBER:", "NOT LOADED", ""]
    if not a.get("ran"):
        out += [f"S12.2 NOT RUN: {a.get('reason')}", ""]
    else:
        out.append("INNER FOLDS:")
        for i, f in enumerate(a["folds"], 1):
            out += [f"Fold {i}:", f"CE: {s11.fold_line(f, 'CE')}", f"PE: {s11.fold_line(f, 'PE')}", ""]
        out.append("MODEL STABILITY:")
        for s in sd.SIDES:
            c = a["chosen_stability"][s]
            out.append(f"{s}: {c['family']} {c['status']} (AUC mean {s11._f(c['auc_mean'], 3)} +/- "
                       f"{s11._f(c['auc_std'], 3)}, good folds {c['good_folds']}/{c['folds_run']}, positive-expectancy "
                       f"folds {a['side_inner_positive_folds'][s]}/{c['folds_run']}; {c['reason']})")
        out.append(f"threshold {a['threshold']:.2f} ({a['threshold_reason']}); margin {sm.MARGIN}; EV > 0")
        for split, title in (("VALIDATION", "VALIDATION"), ("OOS", "OOS MAY-JUNE")):
            r = a["results"].get(split)
            out += ["", f"{title}:" + ("" if r else f" NOT_RUN ({a['not_run_reasons'].get(split)})")]
            if r:
                out += [f"{sc}: {s11.tline(r['trading'][sc])}" for sc in SCOPES]
        out += ["", "SIDE VERDICTS:"] + [f"{s}: {v['verdict']} ({v['reason']})" for s, v in a["side_verdicts"].items()]
    out += ["", "70% TARGET:", a.get("target_70", f"{TARGET_70_TEXT} / {INSUFFICIENT}"), "",
            "LOOKAHEAD:", _ok(a.get("lookahead_ok")), "", "MODEL LEAKAGE:", _ok(a.get("leakage_ok")), "",
            "DATA INTEGRITY:", _ok(a.get("data_ok") if a.get("ran") else (a.get("data_quality") or {}).get("passed")),
            "", "MAX LOADED DATE:",
            ((a.get("lookahead") or {}).get("max_loaded_timestamp") or {}).get("max_loaded_date")
            or (a.get("read_summary") or {}).get("max_raw_timestamp", "N/A") or "N/A",
            "", "OCTOBER LOADED:", "NO", "", "PRODUCTION CHANGED:",
            "NO" if a.get("production_unchanged") is not False else "YES", "", "LIVE ORDERS:", "0", "",
            "FORWARD PAPER:", a.get("forward_paper", "NOT SAFE"), "", "FINAL:", a.get("decision", INSUFF),
            f"({a.get('decision_reason')})", "",
            "9.1-month constrained research experiment - not a 12-month or 18-month validation. Historical results "
            "are not live accuracy; no guaranteed profitability."]
    return "\n".join(out)


def markdown(a: dict) -> str:
    md = ["# S12.2 constrained 9.1-month ML research", "", f"**{BANNER}**", "", f"> {DISCLAIMER}", "",
          f"## Final: **{a.get('decision')}** - {a.get('decision_reason')}", "",
          f"Forward PAPER: **{a.get('forward_paper')}** | 70% target: {a.get('target_70')} | Lookahead "
          f"{_ok(a.get('lookahead_ok'))} | Leakage {_ok(a.get('leakage_ok'))} | Data integrity {_ok(a.get('data_ok'))} "
          "| July+ / October: NOT LOADED | Production changed: NO | Live orders: 0", "",
          "## Clean window and splits", "",
          f"- Clean history {CLEAN_START} -> {MAX_LOAD} ({CLEAN_MONTHS} months; the S12 12-month gate is NOT met - "
          "this is a constrained experiment).",
          f"- TRAIN {CLEAN_START} -> {TRAIN_END}; VALIDATION April 2026; OOS May-June 2026 (only if April passed).",
          f"- Inner folds: " + "; ".join(f"{n} train ..{a_} / validate {b}..{c}" for n, a_, b, c in FOLDS),
          f"- Data-quality gate (S12): {_ok((a.get('data_quality') or {}).get('passed'))} "
          f"{(a.get('data_quality') or {}).get('checks')}", "", "## Label (fixed)", ""]
    md += [f"- {k}: {v}" for k, v in LABEL.items()]
    if not a.get("ran"):
        md += ["", f"**Not run:** {a.get('reason')}"]
        return "\n".join(md) + "\n"
    md += ["", "## Inner folds (TRAIN only)", "", "| Fold | Train | Validation | Status | CE | PE |",
           "|---|---|---|---|---|---|"]
    md += [f"| {f['fold']} | {'..'.join(f['train'])} | {'..'.join(f['validation'])} | {f['status']} | "
           f"{s11.fold_line(f, 'CE')} | {s11.fold_line(f, 'PE')} |" for f in a["folds"]]
    md += ["", "## Model stability", "",
           "| Family | Side | Folds | Good | AUC mean | AUC std | Brier | Log loss | ECE | PF mean | Exp mean | Status |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['family']} | {r['side']} | {r['folds_run']} | {r['good_folds']} | {s11._f(r['auc_mean'], 3)} | "
           f"{s11._f(r['auc_std'], 3)} | {s11._f(r['brier_mean'])} | {s11._f(r['log_loss_mean'])} | "
           f"{s11._f(r['ece_mean'], 3)} | {s11._f(r['pf_mean'], 2)} | {s11._f(r['expectancy_mean'], 3)} | {r['status']} |"
           for r in a["stability"]]
    md += ["", f"Chosen families {a['families']} ({a['family_reason']}); threshold {a['threshold']:.2f} "
           f"({a['threshold_reason']}).", "", "## Results (frozen)", ""]
    for split in ("TRAIN", "VALIDATION", "OOS"):
        r = a["results"].get(split)
        md.append(f"**{SPLIT_NAMES[split]}** ({a['labels'].get(split, 'IN_SAMPLE')})"
                  + ("" if r else f": NOT_RUN - {a['not_run_reasons'].get(split)}"))
        if r:
            md += [f"- {sc}: {s11.tline(r['trading'][sc])} | AUC {s11._f(r['model'][sc]['roc_auc'])} | Brier "
                   f"{s11._f(r['model'][sc]['brier'])}" for sc in SCOPES]
        md.append("")
    md += ["## Side verdicts", ""] + [f"- {s}: **{v['verdict']}** - {v['reason']}" for s, v in a["side_verdicts"].items()]
    md += ["", "## Descriptive comparison (saved reports; not used for any choice)", "",
           "| Month | System | Relation | Trades | Win rate | PF | Expectancy | P&L |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['period']} | {r['system']} | {r['relation_to_period']} | {r['trades']} | {s11._pct(r['accuracy'])} | "
           f"{s11._f(r['profit_factor'], 2)} | {s11._f(r['expectancy_r'], 3)} | {r['net_pnl']} |" for r in a["comparison"]]
    md += ["", "## Checks", ""]
    md += [f"- lookahead `{k}`: {'PASS' if v['passed'] else 'FAIL'}" for k, v in a["lookahead"].items()]
    md += [f"- leakage `{k}`: {'PASS' if v['passed'] else 'FAIL'}" for k, v in a["leakage"].items()]
    md += ["", "Nothing was tuned on April, May, June, July, August or September. Files that could hold candles after "
           "2026-06-30 were never opened. October was not loaded. Production unchanged; live orders 0."]
    return "\n".join(md) + "\n"
