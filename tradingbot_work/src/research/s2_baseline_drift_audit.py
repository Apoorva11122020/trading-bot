"""S2 BASELINE DRIFT AUDIT (PAPER | RESEARCH ONLY | NO ORDERS; S2 and production code untouched).

Why did the previous limited-data S2 OOS (2026-09-25..10-01, legacy store) and the new 6-month S2 replay
(OOS 2026-08-01..08-31, data/historical/upstox_6m) disagree so much?

OLD  = the per-signal rows saved by the previous replay (reports/shadow_strategy_oos/{train,validation,oos}.json).
NEW  = the exact S2 pipeline replayed on the 6-month store (same signal, contract, entry, ATR, SL/target, exit code).
Signals on overlapping sessions are compared one by one. Diagnostic re-evaluations of the OLD signals (same signal,
same contract, exact S2 evaluation code) swap only the premium source (legacy vs 6-month) and the contract tick size
to attribute differences. Nothing is optimized, no strategy parameter is changed and no outcome is reconciled.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.backtest.research_strategies import ResearchSignal
from src.indicators.technical import atr
from src.options.models import OptionType
from src.options.selection import atm_strike
from src.options.strategy import round_to_tick

HEADING = "S2 BASELINE DRIFT AUDIT"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS"
DIRECTIONS = ("BUY", "SELL")
SCOPES = ("BUY", "SELL", "COMBINED")
S2_NAME = dr.BASELINE.name
IST = imp.IST
INTERVAL = 5
OLD_FILES = {"TRAIN": "train.json", "VALIDATION": "validation.json", "OOS": "oos.json"}
OLD_SPLITS = {"TRAIN": ("2026-09-01", "2026-09-17"), "VALIDATION": ("2026-09-18", "2026-09-24"),
              "OOS": ("2026-09-25", "2026-10-01")}
NEW_SPLITS = {"TRAIN": ("2026-04-01", "2026-06-30"), "VALIDATION": ("2026-07-01", "2026-07-31"),
              "OOS": ("2026-08-01", "2026-08-31"), "HOLDOUT": ("2026-09-01", "2026-09-30")}
OLD_TO_DATE = date(2026, 10, 1)
DRIFT_FLAGS = (0.05, 0.10, 0.20)
FIVE = 5.0
SCENARIOS = ("LEGACY_PREMIUM_OLD_TICK", "LEGACY_PREMIUM_TICK_0_05", "SIXM_PREMIUM_OLD_TICK", "SIXM_PREMIUM_TICK_0_05")
CLASSES = {"A": "DATA DIFFERENCE", "B": "OPTION CONTRACT DIFFERENCE", "C": "ENTRY DIFFERENCE",
           "D": "CANDLE ALIGNMENT DIFFERENCE", "E": "EXIT CALCULATION DIFFERENCE", "F": "DATA GAP",
           "G": "STRATEGY IMPLEMENTATION DIFFERENCE", "H": "NO MATERIAL DIFFERENCE"}
FIX, CORRECT, FURTHER = "FIX DATA/REPLAY PIPELINE", "BASELINE IS CORRECT", "FURTHER INVESTIGATION REQUIRED"
PASS, FAIL, INFO = "PASS", "FAIL", "INFO"


# ------------------------------------------------------------------ helpers
def _ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)


def _ist_series(values: pd.Series) -> pd.Series:
    t = pd.to_datetime(values)
    return t.dt.tz_localize(IST) if t.dt.tz is None else t.dt.tz_convert(IST)


def key_of(r: dict) -> tuple[str, str]:
    return (_ts(r["signal_time"]).isoformat(), r["direction"])


def parse_symbol(sym: str | None) -> dict[str, Any]:
    """'NIFTY 23150 CE 29 SEP 26' -> strike, option type, expiry. Unparseable -> Nones (never guessed)."""
    m = re.match(r"^\s*([A-Z]+)\s+(\d+(?:\.\d+)?)\s+(CE|PE)\s+(\d{1,2}\s+[A-Z]{3}\s+\d{2})\s*$", str(sym or "").upper())
    if not m:
        return {"strike": None, "option_type": None, "expiry": None}
    return {"strike": float(m.group(2)), "option_type": m.group(3),
            "expiry": datetime.strptime(m.group(4).title(), "%d %b %y").date()}


def key_expiry(key: str | None) -> date | None:
    m = re.search(r"\|(\d{2})-(\d{2})-(\d{4})$", str(key or ""))
    return date(int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else None


def on_grid(v, step: float) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(v / step - round(v / step)) < 1e-6


def inferred_tick(r: dict) -> float:
    """Tick the row's entry/SL/target were rounded with: all three on a Rs 5 grid -> 5.0, else 0.05."""
    vals = [r.get("entry"), r.get("stop_loss"), r.get("target")]
    return FIVE if all(v is not None and on_grid(float(v), FIVE) for v in vals) else 0.05


def _pct(v) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def _num(v) -> str:
    return "N/A" if v is None else (f"{v:,.2f}" if isinstance(v, float) else str(v))


def metrics(rows: Sequence[dict]) -> dict[str, Any]:
    m = lab.perf(rows)
    return {"trades": m["evaluated"], "wins": m["wins"], "losses": m["losses"], "accuracy": m["win_rate"],
            "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"], "signals": len(rows)}


def scoped(rows: Sequence[dict]) -> dict[str, dict]:
    return {"BUY": metrics([r for r in rows if r["direction"] == "BUY"]),
            "SELL": metrics([r for r in rows if r["direction"] == "SELL"]), "COMBINED": metrics(list(rows))}


def in_range(rows: Iterable[dict], lo: str, hi: str) -> list[dict]:
    return [r for r in rows if lo <= str(r["session_date"])[:10] <= hi]


def _check(cid: str, name: str, ok: bool | None, detail: str) -> dict[str, str]:
    return {"id": cid, "check": name, "status": INFO if ok is None else (PASS if ok else FAIL), "detail": detail}


# ------------------------------------------------------------------ 1. exact S2 implementation (read from code/config)
def s2_implementation(rc, params, s2cfg: dict) -> dict[str, str]:
    rules = lab.s2_rules(rc, params, s2cfg)
    return {
        "signal_conditions": rules["signal"],
        "indicator_calculations": (f"ORB high/low = max(high)/min(low) of the 5m candles {s2cfg['range_start']}-"
                                   f"{s2cfg['range_end']} (research_strategies.signals_s2). No other indicator gates "
                                   "the signal. Underlying ATR14 is computed (underlying_features) but only feeds the "
                                   "break_atr diagnostic, never SL/target."),
        "candle_completion": ("A signal is stamped with the 5m candle's start label (e.g. 09:30 = 09:30-09:35) and uses "
                              "that candle's close, so it is known at the candle's end."),
        "entry_timing": rules["entry_timing"] + " (current_day_s2.option_levels: entry candle = signal label + 5 min).",
        "option_selection": rules["option_selection"] + " (option_coverage.resolve_signal_contract).",
        "strike_selection": "selection.atm_strike: listed strike nearest to the signal close; ties -> lower strike.",
        "expiry_selection": "selection.select_expiry(expiries >= signal date, 'nearest').",
        "premium_entry": ("open of the option candle at signal + 5 min, rounded with round_to_tick(open, "
                          "contract.tick_size, 'nearest'); tick None -> 0.05."),
        "atr": (f"premium ATR{params.atr_period} = Wilder EWM (alpha 1/{params.atr_period}, adjust=False, min_periods "
                f"{params.atr_period}) of the true range over EVERY stored premium candle of the contract up to and "
                "including the signal candle (history-length dependent)."),
        "sl": rules["stop_loss"] + " Rounded DOWN to contract.tick_size (options.strategy.option_stop_loss).",
        "target": rules["target"] + " Rounded to the nearest contract.tick_size (options.strategy.option_target).",
        "eod": rules["square_off"],
        "outcome_logic": rules["outcome"] + " No exit evidence -> PENDING; missing data -> DATA_UNAVAILABLE.",
        "costs": rules["costs"],
    }


# ------------------------------------------------------------------ OLD rows (saved by the previous replay)
def load_old_rows(old_dir: Path) -> list[dict]:
    out = []
    for split, name in OLD_FILES.items():
        p = Path(old_dir) / name
        if not p.is_file():
            raise FileNotFoundError(f"previous replay output missing: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in data["signals"][S2_NAME]:
            meta = parse_symbol(r.get("contract"))
            out.append(dict(r, old_split=split, strike=meta["strike"], option_type=meta["option_type"],
                            expiry_date=meta["expiry"] or key_expiry(r.get("instrument_key")),
                            tick_inferred=inferred_tick(r)))
    return out


# ------------------------------------------------------------------ NEW rows (exact S2 on the 6-month store)
def replay(*, spot: pd.DataFrame, days: Sequence[str], signal_fn: Callable, resolve: Callable, premium_for: Callable,
           params, square_off: str, range_start: str, range_end: str) -> dict[str, Any]:
    """Exactly what shadow_historical.replay does for the S2 baseline (pipeline + run_candidate + classify_row)."""
    pipeline = imp._pipeline_factory(signal_fn, resolve, params, range_start, range_end)
    contexts, feats = pipeline(spot, premium_for, {date.fromisoformat(d) for d in days})
    raw = dr.run_candidate(contexts, feats, dr.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE,
                           cache={}) if contexts else []
    ctx_by_key = {(c["signal_time"].isoformat(), c["direction"]): c for c in contexts}
    rows, seen = [], set()
    for r in raw:
        k = (r["signal_time"], r["direction"])
        if k in seen:
            continue
        seen.add(k)
        row = dict(sh.classify_row(r, {}), strategy=S2_NAME)
        ctx = ctx_by_key.get(key_of(row), {})
        meta = parse_symbol(row.get("contract"))
        row.update(strike=meta["strike"], option_type=meta["option_type"],
                   expiry_date=ctx.get("expiry") or meta["expiry"], tick_inferred=ctx.get("tick"))
        rows.append(row)
    return {"rows": rows, "contexts": contexts, "ctx_by_key": ctx_by_key, "features": feats, "pipeline": pipeline}


# ------------------------------------------------------------------ legacy store (read-only, offline)
class LegacyData:
    """Read-only view of the previous dataset (data/historical parquet store). Never downloads."""

    def __init__(self, store, underlying: str = "NIFTY", to_date: date = OLD_TO_DATE) -> None:
        self.store, self.underlying, self.to_date = store, underlying, to_date
        self._cache: dict[str, pd.DataFrame | None] = {}

    def spot(self) -> pd.DataFrame:
        f = self.store.read("underlying", self.underlying)
        if f.empty:
            return f
        f = f.copy()
        f["timestamp"] = _ist_series(f["timestamp"])
        return f.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)

    def premium(self, key: str, expiry: date | None) -> pd.DataFrame | None:
        if key not in self._cache:
            f = self.store.read("options", self.underlying, key, expiry=expiry) if expiry else pd.DataFrame()
            if not f.empty:
                f = f.copy()
                f["timestamp"] = _ist_series(f["timestamp"])
                f = (f[f["timestamp"].dt.date <= self.to_date].drop_duplicates("timestamp", keep="last")
                     .sort_values("timestamp").reset_index(drop=True))
            self._cache[key] = None if f.empty else f
        return self._cache[key]

    def option_files(self) -> list[Path]:
        return self.store.files("options", self.underlying)


# ------------------------------------------------------------------ 2. dataset comparison
def frame_profile(frames: Iterable[pd.DataFrame], days: set[str] | None = None) -> dict[str, Any]:
    n = files = 0
    keys, expiries, strikes, kinds = set(), set(), set(), Counter()
    vol0 = oi0 = dup = off_grid = missing = 0
    tz = set()
    for f in frames:
        if f is None or f.empty:
            continue
        t = pd.to_datetime(f["timestamp"])
        tz.add(str(t.dt.tz))
        t = _ist_series(f["timestamp"])
        sel = t.dt.date.map(lambda d: d.isoformat()).isin(days) if days is not None else pd.Series(True, index=f.index)
        f, t = f[sel.to_numpy()], t[sel.to_numpy()]
        if f.empty:
            continue
        files += 1
        n += len(f)
        if "instrument_key" in f:
            keys |= set(f["instrument_key"].astype(str))
        if "expiry" in f:
            expiries |= set(f["expiry"].astype(str))
        if "strike" in f:
            strikes |= set(f["strike"].astype(float))
        if "option_type" in f:
            kinds.update(f["option_type"].astype(str))
        vol0 += int((f.get("volume", pd.Series(0, index=f.index)).fillna(0) == 0).sum())
        oi0 += int((f.get("open_interest", pd.Series(0, index=f.index)).fillna(0) == 0).sum())
        dup += int(t.duplicated().sum())
        off_grid += int(((t.dt.minute % INTERVAL != 0) | (t.dt.second != 0)).sum())
        per_day = t.groupby(t.dt.date).agg(["min", "max", "count"])
        for _, row in per_day.iterrows():
            span = int((row["max"] - row["min"]) / pd.Timedelta(minutes=INTERVAL)) + 1
            missing += max(0, span - int(row["count"]))
    return {"files": files, "candles": n, "instrument_keys": len(keys), "expiries": sorted(expiries),
            "strikes": len(strikes), "ce": kinds.get("CE", 0), "pe": kinds.get("PE", 0), "timezones": sorted(tz),
            "zero_volume_share": vol0 / n if n else None, "zero_oi_share": oi0 / n if n else None,
            "duplicate_timestamps": dup, "off_grid": off_grid, "missing_inside_span": missing}


def candle_diff(a: pd.DataFrame, b: pd.DataFrame, lags: Sequence[int] = (-1, 0, 1)) -> dict[str, Any]:
    """Join two candle frames on timestamp. Reports OHLC differences and which time shift aligns best."""
    if a is None or b is None or a.empty or b.empty:
        return {"matched": 0, "only_a": 0 if a is None else len(a), "only_b": 0 if b is None else len(b),
                "close_mae": None, "close_max": None, "share_close_diff_gt_0_5pct": None, "best_lag": None,
                "lag_mae": {}}
    x = a[["timestamp", "open", "high", "low", "close"]].copy()
    y = b[["timestamp", "open", "high", "low", "close"]].copy()
    x["timestamp"], y["timestamp"] = _ist_series(x["timestamp"]), _ist_series(y["timestamp"])
    j = x.merge(y, on="timestamp", suffixes=("_a", "_b"))
    out = {"matched": len(j), "only_a": len(x) - len(j), "only_b": len(y) - len(j)}
    if j.empty:
        return {**out, "close_mae": None, "close_max": None, "share_close_diff_gt_0_5pct": None, "best_lag": None,
                "lag_mae": {}}
    d = (j["close_a"] - j["close_b"]).abs()
    rel = d / j["close_b"].abs().where(j["close_b"] != 0)
    ohlc = max(float((j[f"{c}_a"] - j[f"{c}_b"]).abs().max()) for c in ("open", "high", "low", "close"))
    lag_mae = {}
    for lag in lags:
        z = y.copy()
        z["timestamp"] = z["timestamp"] + pd.Timedelta(minutes=INTERVAL * lag)
        jj = x.merge(z, on="timestamp", suffixes=("_a", "_b"))
        lag_mae[lag] = float((jj["close_a"] - jj["close_b"]).abs().mean()) if len(jj) else None
    valid = {k: v for k, v in lag_mae.items() if v is not None}
    return {**out, "close_mae": float(d.mean()), "close_max": float(d.max()), "ohlc_max": ohlc,
            "share_close_diff_gt_0_5pct": float((rel > 0.005).mean()),
            "best_lag": min(valid, key=lambda k: (valid[k], abs(k))) if valid else None, "lag_mae": lag_mae}


# ------------------------------------------------------------------ 3/4/5/7/8. signal comparison
def match_signals(old_rows: Sequence[dict], new_rows: Sequence[dict], days: set[str]) -> list[dict]:
    old = {key_of(r): r for r in old_rows if str(r["session_date"])[:10] in days}
    new = {key_of(r): r for r in new_rows if str(r["session_date"])[:10] in days}
    out = []
    for k in sorted(set(old) | set(new)):
        out.append({"key": k, "status": "BOTH" if k in old and k in new else ("OLD_ONLY" if k in old else "NEW_ONLY"),
                    "old": old.get(k), "new": new.get(k)})
    return out


def _f(r: dict | None, k: str):
    return None if r is None else r.get(k)


def _diff(a, b) -> float | None:
    return None if a is None or b is None else float(b) - float(a)


def outcome_reason(o: dict, n: dict, prem_same: bool | None) -> str:
    """First concrete difference that explains a changed outcome for an identical signal (never reconciled)."""
    if o.get("state") != sh.EVALUATED or n.get("state") != sh.EVALUATED:
        return f"STATE {o.get('state')} -> {n.get('state')} ({o.get('reason') or ''} -> {n.get('reason') or ''})"
    if o.get("instrument_key") != n.get("instrument_key"):
        return "DIFFERENT_CONTRACT"
    parts = []
    if not _close(o.get("entry"), n.get("entry")):
        parts.append(f"ENTRY {o.get('entry')} -> {n.get('entry')}")
    if not _close(o.get("premium_atr"), n.get("premium_atr"), 1e-6):
        parts.append(f"PREMIUM_ATR {_r(o.get('premium_atr'))} -> {_r(n.get('premium_atr'))}")
    if not _close(o.get("stop_loss"), n.get("stop_loss")):
        parts.append(f"SL {o.get('stop_loss')} -> {n.get('stop_loss')}")
    if not _close(o.get("target"), n.get("target")):
        parts.append(f"TARGET {o.get('target')} -> {n.get('target')}")
    if prem_same is False:
        parts.append("PREMIUM_CANDLES_AFTER_ENTRY_DIFFER")
    return "; ".join(parts) or "UNEXPLAINED"


def _close(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=1e-9)


def _r(v, n: int = 3):
    return None if v is None else round(float(v), n)


def comparison_rows(pairs: Sequence[dict], prem_same: dict | None = None) -> list[dict]:
    out = []
    for p in pairs:
        o, n = p["old"], p["new"]
        same_contract = None if o is None or n is None else o.get("instrument_key") == n.get("instrument_key")
        same_outcome = None if o is None or n is None else o.get("outcome") == n.get("outcome")
        reason = None
        if p["status"] == "BOTH" and not same_outcome:
            reason = outcome_reason(o, n, (prem_same or {}).get(p["key"]))
        out.append({
            "timestamp": p["key"][0], "direction": p["key"][1], "status": p["status"],
            "old_spot": _f(o, "spot"), "new_spot": _f(n, "spot"), "spot_diff": _diff(_f(o, "spot"), _f(n, "spot")),
            "old_contract": _f(o, "contract"), "new_contract": _f(n, "contract"),
            "old_strike": _f(o, "strike"), "new_strike": _f(n, "strike"),
            "old_expiry": str(_f(o, "expiry_date") or ""), "new_expiry": str(_f(n, "expiry_date") or ""),
            "old_entry": _f(o, "entry"), "new_entry": _f(n, "entry"),
            "old_premium_atr": _r(_f(o, "premium_atr"), 6), "new_premium_atr": _r(_f(n, "premium_atr"), 6),
            "old_sl": _f(o, "stop_loss"), "new_sl": _f(n, "stop_loss"),
            "old_target": _f(o, "target"), "new_target": _f(n, "target"),
            "old_exit_time": _f(o, "exit_time"), "new_exit_time": _f(n, "exit_time"),
            "old_exit_price": _f(o, "exit_price"), "new_exit_price": _f(n, "exit_price"),
            "old_exit_reason": _f(o, "exit_reason"), "new_exit_reason": _f(n, "exit_reason"),
            "old_outcome": _f(o, "outcome") or _f(o, "state"), "new_outcome": _f(n, "outcome") or _f(n, "state"),
            "old_pnl": _f(o, "pnl"), "new_pnl": _f(n, "pnl"),
            "same_contract": same_contract, "same_outcome": same_outcome, "outcome_diff_reason": reason})
    return out


def option_selection_rows(pairs: Sequence[dict]) -> list[dict]:
    out = []
    for p in pairs:
        o, n = p["old"], p["new"]
        row = {"timestamp": p["key"][0], "direction": p["key"][1], "status": p["status"]}
        for tag, r in (("old", o), ("new", n)):
            exp = _f(r, "expiry_date")
            sd = None if r is None else date.fromisoformat(str(r["session_date"])[:10])
            row.update({f"{tag}_instrument_key": _f(r, "instrument_key"), f"{tag}_contract": _f(r, "contract"),
                        f"{tag}_expiry": str(exp or ""), f"{tag}_strike": _f(r, "strike"),
                        f"{tag}_option_type": _f(r, "option_type"),
                        f"{tag}_dte": None if exp is None or sd is None else (exp - sd).days,
                        f"{tag}_atm_distance": (None if r is None or r.get("strike") is None or r.get("spot") is None
                                                else round(abs(float(r["strike"]) - float(r["spot"])), 4))})
        mism = []
        if p["status"] != "BOTH":
            mism.append(f"SIGNAL_{p['status']}")
        else:
            for fld in ("expiry", "strike", "option_type", "dte", "instrument_key"):
                if str(row[f"old_{fld}"]) != str(row[f"new_{fld}"]):
                    mism.append(fld.upper())
        row["mismatches"] = ",".join(mism)
        row["mismatch"] = bool(mism)
        out.append(row)
    return out


def entry_drift_rows(pairs: Sequence[dict]) -> list[dict]:
    out = []
    for p in pairs:
        o, n = p["old"], p["new"]
        if p["status"] != "BOTH" or o.get("entry") is None or n.get("entry") is None:
            continue
        a, b = float(o["entry"]), float(n["entry"])
        pct = abs(b - a) / a if a else None
        out.append({"timestamp": p["key"][0], "direction": p["key"][1],
                    "same_contract": o.get("instrument_key") == n.get("instrument_key"),
                    "old_contract": o.get("contract"), "new_contract": n.get("contract"), "old_entry": a,
                    "new_entry": b, "abs_diff": round(abs(b - a), 4), "pct_diff": None if pct is None else round(pct, 6),
                    "old_tick_inferred": o.get("tick_inferred"),
                    **{f"gt_{int(t * 100)}pct": bool(pct is not None and pct > t) for t in DRIFT_FLAGS}})
    return out


def sl_target_rows(pairs: Sequence[dict]) -> list[dict]:
    out = []
    for p in pairs:
        o, n = p["old"], p["new"]
        if p["status"] != "BOTH" or o.get("stop_loss") is None or n.get("stop_loss") is None:
            continue
        src = []
        if o.get("instrument_key") != n.get("instrument_key"):
            src.append("CONTRACT")
        if not _close(o.get("entry"), n.get("entry")):
            src.append("ENTRY_PRICE")
        if not _close(o.get("premium_atr"), n.get("premium_atr"), 1e-6):
            src.append("OPTION_PREMIUM_ATR")
        if o.get("tick_inferred") != (n.get("tick_inferred") or 0.05):
            src.append(f"CONFIGURATION_TICK {o.get('tick_inferred')} vs {n.get('tick_inferred') or 0.05}")
        ro = float(o["entry"]) - float(o["stop_loss"])
        rn = float(n["entry"]) - float(n["stop_loss"])
        out.append({"timestamp": p["key"][0], "direction": p["key"][1], "old_atr": _r(o.get("premium_atr"), 6),
                    "new_atr": _r(n.get("premium_atr"), 6), "old_sl": o["stop_loss"], "new_sl": n["stop_loss"],
                    "old_target": o["target"], "new_target": n["target"], "old_r": round(ro, 4), "new_r": round(rn, 4),
                    "underlying_atr_used": False, "difference_sources": ",".join(src) or "NONE"})
    return out


# ------------------------------------------------------------------ diagnostic re-evaluation (attribution only)
def _signal(r: dict) -> ResearchSignal:
    lvl = r.get("breakout_level")
    cond = {} if lvl is None else ({"above": float(lvl)} if r["direction"] == "BUY" else {"below": float(lvl)})
    return ResearchSignal(_ts(r["signal_time"]).to_pydatetime(), r["direction"], "S2", "NIFTY", float(r["spot"]), cond)


def _contract(r: dict, tick: float | None):
    return SimpleNamespace(instrument_key=r["instrument_key"], trading_symbol=r.get("contract"),
                           expiry=r["expiry_date"], strike=r.get("strike"),
                           option_type=OptionType(r["option_type"]), tick_size=tick, lot_size=sh.LOT_SIZE,
                           underlying="NIFTY")


def reevaluate(old_rows: Sequence[dict], features: pd.DataFrame, premium_of: Callable[[dict], Any],
               tick_of: Callable[[dict], float | None], params, square_off: str) -> dict[tuple, dict]:
    """Same signal + same contract through the exact S2 evaluation code; only premium source / tick differ."""
    usable = [r for r in old_rows if r.get("instrument_key") and r.get("expiry_date") and r.get("option_type")]
    by_key = {key_of(r): r for r in usable}

    def resolve(sig):
        r = by_key[(_ts(sig.timestamp).isoformat(), sig.direction)]
        return _contract(r, tick_of(r))

    def premium_for(c):
        r = next(x for x in usable if x["instrument_key"] == c.instrument_key)
        return premium_of(r)

    ctxs = imp.build_contexts([_signal(r) for r in usable], features, resolve, premium_for, params)
    raw = imp.run_variant(ctxs, features, imp.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE, cache={})
    return {key_of(x): sh.classify_row(x, {}) for x in raw}


def scenario_summary(name: str, rows: dict[tuple, dict], old_by_key: dict, keys: Sequence[tuple]) -> dict[str, Any]:
    rs = [rows[k] for k in keys if k in rows]
    same = sum(1 for k in keys if k in rows and rows[k].get("outcome") == old_by_key[k].get("outcome")
               and _close(rows[k].get("pnl"), old_by_key[k].get("pnl"), 1e-6))
    return {"scenario": name, "signals": len(keys), "reproduces_old_outcome_and_pnl": same,
            "reproduction_share": same / len(keys) if keys else None, **{s: scoped(rs)[s] for s in SCOPES}}


# ------------------------------------------------------------------ 6. candle alignment
def alignment_checks(contexts: Sequence[dict], params) -> dict[str, Any]:
    ok = [c for c in contexts if c.get("data_status") == "OK"]
    step = pd.Timedelta(minutes=INTERVAL)
    bad = Counter()
    examples: list[str] = []
    for c in ok:
        prem = c["premium"]
        t = _ist_series(prem["timestamp"])
        tag = f"{c['signal_time'].isoformat()}|{c['direction']}"
        problems = []
        if str(t.dt.tz) not in ("Asia/Kolkata", "IST"):
            problems.append("TIMEZONE")
        if t.duplicated().any():
            problems.append("DUPLICATE_CANDLE")
        if ((t.dt.minute % INTERVAL != 0) | (t.dt.second != 0)).any():
            problems.append("OFF_GRID_CANDLE")
        if not (t == c["signal_time"]).any():
            problems.append("MISSING_SIGNAL_CANDLE")
        if c["entry_time"] != c["signal_time"] + step:
            problems.append("SHIFTED_ENTRY_CANDLE" if c["entry_time"] > c["signal_time"] else "SAME_CANDLE_LOOKAHEAD")
        er = prem[t == c["entry_time"]]
        if er.empty:
            problems.append("MISSING_ENTRY_CANDLE")
        elif not math.isclose(round_to_tick(float(er["open"].iat[0]), c.get("tick"), "nearest"), c["entry"],
                              abs_tol=1e-9):
            problems.append("WRONG_ENTRY_CANDLE")
        hist = prem[t <= c["signal_time"]].reset_index(drop=True)
        patr = float(atr(hist, params.atr_period).iloc[-1]) if len(hist) else float("nan")
        if not math.isclose(patr, c["premium_atr"], rel_tol=1e-9, abs_tol=1e-9):
            problems.append("ATR_USES_FUTURE_OR_WRONG_CANDLES")
        for p in problems:
            bad[p] += 1
        if problems and len(examples) < 10:
            examples.append(f"{tag}: {','.join(problems)}")
    return {"checked": len(ok), "problems": dict(bad), "examples": examples, "passed": bool(ok) and not bad}


# ------------------------------------------------------------------ 11. data gaps
def gap_check(prem: pd.DataFrame | None, r: dict, square_off: str) -> dict[str, Any]:
    if prem is None or isinstance(prem, str) or prem.empty or not r.get("entry_time"):
        return {"entry_candle": False, "missing_after_entry": None, "evaluable": False, "gap": True}
    t = _ist_series(prem["timestamp"])
    entry = _ts(r["entry_time"])
    end = _ts(r["exit_time"]) if r.get("exit_time") else pd.Timestamp(
        datetime.combine(entry.date(), datetime.strptime(square_off, "%H:%M").time()), tz=IST)
    want = pd.date_range(entry, end, freq=f"{INTERVAL}min")
    have = set(t[(t >= entry) & (t <= end)])
    missing = [x for x in want if x not in have]
    has_entry = entry in have
    return {"entry_candle": has_entry, "missing_after_entry": len(missing), "evaluable": has_entry and not missing,
            "gap": (not has_entry) or bool(missing)}


def gap_rows(rows: Sequence[dict], premium_of: Callable[[dict], Any], dataset: str, square_off: str) -> list[dict]:
    out = []
    for r in rows:
        if r.get("outcome") not in (imp.LOSS, imp.AMBIGUOUS):
            continue
        g = gap_check(premium_of(r), r, square_off)
        out.append({"dataset": dataset, "timestamp": key_of(r)[0], "direction": r["direction"],
                    "contract": r.get("contract"), "outcome": r["outcome"], "exit_reason": r.get("exit_reason"),
                    **g, "loss_caused_by_data_gap": g["gap"]})
    return out


# ------------------------------------------------------------------ 12. contract history
def contract_check(r: dict, calendar: Sequence[date], listed: Callable[[date, date, str], list[float]] | None,
                   exists: Callable[[dict], bool]) -> dict[str, Any]:
    sd = date.fromisoformat(str(r["session_date"])[:10])
    exp, strike, kind, spot = r.get("expiry_date"), r.get("strike"), r.get("option_type"), r.get("spot")
    nearest = next((e for e in sorted(calendar) if e >= sd), None)
    strikes = listed(sd, exp, kind) if listed and exp and kind else []
    want_strike = atm_strike(strikes, float(spot)) if strikes and spot else None
    k_exp = key_expiry(r.get("instrument_key"))
    checks = {
        "contract_existed": bool(exists(r)),
        "expiry_not_before_signal": exp is not None and exp >= sd,
        "nearest_listed_expiry": exp is not None and nearest is not None and exp == nearest,
        "option_type_matches_direction": kind == ("CE" if r["direction"] == "BUY" else "PE"),
        "atm_strike": (want_strike is not None and strike is not None and abs(want_strike - float(strike)) < 1e-6)
        if strikes else (strike is not None and spot is not None and abs(float(strike) - float(spot)) <= 25.0 + 1e-6),
        "instrument_key_consistent": k_exp is None or k_exp == exp,
    }
    return {"timestamp": key_of(r)[0], "direction": r["direction"], "contract": r.get("contract"),
            "instrument_key": r.get("instrument_key"), "expiry": str(exp or ""), "strike": strike, "spot": spot,
            **checks, "valid": all(checks.values()),
            "impossible": not (checks["expiry_not_before_signal"] and checks["option_type_matches_direction"]
                               and checks["contract_existed"])}


# ------------------------------------------------------------------ 9/10. monthly + counts
def monthly_rows(rows: Sequence[dict], dataset: str) -> list[dict]:
    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_month[str(r["session_date"])[:7]].append(r)
    out = []
    for m in sorted(by_month):
        for s, v in scoped(by_month[m]).items():
            out.append({"dataset": dataset, "month": m, "scope": s, **v})
    return out


def count_rows(rows: Sequence[dict], dataset: str) -> list[dict]:
    c = Counter((str(r["session_date"])[:7], r["direction"], str(r.get("expiry_date") or "UNRESOLVED"),
                 r.get("option_type") or "UNRESOLVED") for r in rows)
    return [{"dataset": dataset, "month": k[0], "direction": k[1], "expiry": k[2], "option_type": k[3], "signals": v}
            for k, v in sorted(c.items())]


def orb_levels(spot: pd.DataFrame, start: str, end: str) -> dict[str, tuple[float, float]]:
    if spot is None or spot.empty:
        return {}
    t = _ist_series(spot["timestamp"])
    hhmm = t.dt.strftime("%H:%M")
    rng = spot[(hhmm >= start) & (hhmm < end)]
    g = rng.groupby(t[rng.index].dt.date)
    return {d.isoformat(): (float(v["high"].max()), float(v["low"].min())) for d, v in g}


def count_reasons(pairs: Sequence[dict], old_spot: pd.DataFrame, new_spot: pd.DataFrame, start: str,
                  end: str) -> list[dict]:
    """Why a signal exists in only one dataset: compare the underlying candle and ORB levels at that timestamp."""
    oo, no = orb_levels(old_spot, start, end), orb_levels(new_spot, start, end)
    oc = {} if old_spot is None or old_spot.empty else dict(zip(_ist_series(old_spot["timestamp"]), old_spot["close"]))
    nc = dict(zip(_ist_series(new_spot["timestamp"]), new_spot["close"]))
    out = []
    for p in pairs:
        if p["status"] == "BOTH":
            continue
        t = _ts(p["key"][0])
        d = t.date().isoformat()
        a, b = oc.get(t), nc.get(t)
        if a is None or b is None:
            why = "UNDERLYING_CANDLE_MISSING_IN_" + ("OLD" if a is None else "NEW")
        elif oo.get(d) != no.get(d):
            why = "ORB_LEVELS_DIFFER"
        elif not math.isclose(float(a), float(b), abs_tol=1e-6):
            why = "UNDERLYING_CLOSE_DIFFERS"
        else:
            why = "SAME_UNDERLYING_INPUTS (filter/entry-window difference)"
        out.append({"timestamp": p["key"][0], "direction": p["key"][1], "status": p["status"], "old_close": a,
                    "new_close": b, "old_orb": oo.get(d), "new_orb": no.get(d), "reason": why})
    return out


# ------------------------------------------------------------------ 13. classification
def classify(ev: dict) -> tuple[list[str], dict[str, str]]:
    found, why = [], {}
    if ev["underlying_share_diff"] > 0.01 or ev["option_share_diff"] > 0.05:
        found.append("A")
        why["A"] = (f"underlying candles with close diff > 0.5%: {ev['underlying_share_diff']:.1%}; option candles: "
                    f"{ev['option_share_diff']:.1%}")
    if ev["contract_mismatch_share"] > 0.0:
        found.append("B")
        why["B"] = f"{ev['contract_mismatch_share']:.1%} of matched signals selected a different contract"
    if ev["entry_gt5_share"] > 0.05:
        found.append("C")
        why["C"] = f"{ev['entry_gt5_share']:.1%} of matched signals have entry drift > 5%"
    if not ev["alignment_pass"]:
        found.append("D")
        why["D"] = "candle alignment check failed"
    if ev["sl_target_diff_share"] > 0.05:
        found.append("E")
        why["E"] = f"{ev['sl_target_diff_share']:.1%} of matched signals have different SL/target"
    if ev["gap_losses"] > 0:
        found.append("F")
        why["F"] = f"{ev['gap_losses']} losing signals lack candles needed for normal SL/target evaluation"
    if ev["old_tick5_share"] > 0.5 and ev["tick_effect_pp"] is not None and abs(ev["tick_effect_pp"]) >= 0.02:
        found.append("G")
        why["G"] = (f"{ev['old_tick5_share']:.0%} of old signals have entry/SL/target on a Rs 5 grid (tick_size 5.0 "
                    f"applied to expired contracts); same data with tick 0.05 moves old accuracy by "
                    f"{ev['tick_effect_pp'] * 100:+.1f} pp")
    if not found:
        found.append("H")
        why["H"] = "no material signal-level difference on overlapping sessions"
    return found, why


def recommend(ev: dict) -> str:
    if not ev["alignment_pass"] or not ev["new_contracts_valid"]:
        return FIX
    if ev["unexplained_outcome_diffs"] > 0 or ev["old_reproduction_share"] is None or ev["old_reproduction_share"] < 0.9:
        return FURTHER
    return CORRECT


# ------------------------------------------------------------------ study
def build_audit(*, old_rows: Sequence[dict], new: dict[str, Any], new_spot: pd.DataFrame, legacy: LegacyData | None,
                six_premium: Callable[[str], pd.DataFrame | None], calendar: Sequence[date],
                listed_strikes: Callable[[date, date, str], list[float]], new_exists: Callable[[dict], bool],
                params, square_off: str, range_start: str, range_end: str, implementation: dict,
                old_spot: pd.DataFrame | None = None, legacy_profile: dict | None = None,
                six_profile: dict | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    new_rows = new["rows"]
    old_days = {str(r["session_date"])[:10] for r in old_rows}
    new_days = {str(r["session_date"])[:10] for r in new_rows}
    overlap = sorted(old_days & new_days)
    ov = set(overlap)
    say(f"matching signals on {len(overlap)} overlapping sessions ...")
    pairs = match_signals(old_rows, new_rows, ov)
    both = [p for p in pairs if p["status"] == "BOTH"]

    def six_of(r):
        return six_premium(r.get("instrument_key"))

    def legacy_of(r):
        return legacy.premium(r["instrument_key"], r.get("expiry_date")) if legacy else None

    say("comparing option candles of every old-selected contract (legacy vs 6-month) ...")
    opt_cmp, prem_same = {}, {}
    for key in sorted({r["instrument_key"] for r in old_rows if r.get("instrument_key")}):
        r = next(x for x in old_rows if x.get("instrument_key") == key)
        opt_cmp[key] = candle_diff(legacy_of(r), six_of(r))
    for p in both:
        o, n = p["old"], p["new"]
        if o.get("instrument_key") == n.get("instrument_key") and o.get("entry_time"):
            a, b = legacy_of(o), six_of(n)
            if a is None or b is None:
                prem_same[p["key"]] = None
                continue
            day = _ts(o["entry_time"]).date()
            sa = a[(a["timestamp"] >= _ts(o["entry_time"])) & (a["timestamp"].dt.date == day)]
            sb = b[(b["timestamp"] >= _ts(o["entry_time"])) & (b["timestamp"].dt.date == day)]
            cols = ["timestamp", "open", "high", "low", "close"]
            prem_same[p["key"]] = sa[cols].reset_index(drop=True).equals(sb[cols].reset_index(drop=True))
    und_cmp = candle_diff(old_spot, new_spot) if old_spot is not None else candle_diff(None, None)
    und_cmp_overlap = candle_diff(
        None if old_spot is None else old_spot[_ist_series(old_spot["timestamp"]).dt.date.map(
            lambda d: d.isoformat()).isin(ov).to_numpy()],
        new_spot[_ist_series(new_spot["timestamp"]).dt.date.map(lambda d: d.isoformat()).isin(ov).to_numpy()])

    comp = comparison_rows(pairs, prem_same)
    sel = option_selection_rows(pairs)
    entry = entry_drift_rows(pairs)
    slt = sl_target_rows(pairs)

    say("diagnostic re-evaluation of the old signals (premium source x tick size) ...")
    old_ov = [r for r in old_rows if str(r["session_date"])[:10] in ov]
    old_by_key = {key_of(r): r for r in old_ov}
    feats = new["features"]
    scen = {}
    if legacy is not None:
        scen["LEGACY_PREMIUM_OLD_TICK"] = reevaluate(old_ov, feats, legacy_of, lambda r: r["tick_inferred"], params,
                                                     square_off)
        scen["LEGACY_PREMIUM_TICK_0_05"] = reevaluate(old_ov, feats, legacy_of, lambda r: 0.05, params, square_off)
    scen["SIXM_PREMIUM_OLD_TICK"] = reevaluate(old_ov, feats, six_of, lambda r: r["tick_inferred"], params, square_off)
    scen["SIXM_PREMIUM_TICK_0_05"] = reevaluate(old_ov, feats, six_of, lambda r: 0.05, params, square_off)
    keys = sorted(old_by_key)
    scen_sum = [scenario_summary(n, v, old_by_key, keys) for n, v in scen.items()]

    say("candle alignment, data gaps and contract history ...")
    align = alignment_checks(new["contexts"], params)
    step = pd.Timedelta(minutes=INTERVAL)
    old_timed = [r for r in old_rows if r.get("entry_time")]
    old_next = sum(_ts(r["entry_time"]) == _ts(r["signal_time"]) + step for r in old_timed)
    align["old_next_bar_entries"] = f"{old_next}/{len(old_timed)}"
    align["passed"] = align["passed"] and old_next == len(old_timed)
    new_ctx = new["ctx_by_key"]
    gaps = gap_rows(new_rows, lambda r: (new_ctx.get(key_of(r)) or {}).get("premium"), "NEW_6M", square_off)
    gaps += gap_rows(old_ov, legacy_of, "OLD_LEGACY", square_off) if legacy is not None else []

    def old_exists(r):
        return legacy_of(r) is not None or six_of(r) is not None

    contracts = [dict(contract_check(r, calendar, listed_strikes, new_exists), dataset="NEW_6M")
                 for r in new_rows if r.get("instrument_key")]
    contracts += [dict(contract_check(r, calendar, listed_strikes, old_exists), dataset="OLD_LEGACY")
                  for r in old_ov if r.get("instrument_key")]
    new_contract_ok = all(c["valid"] for c in contracts if c["dataset"] == "NEW_6M")

    monthly = monthly_rows(new_rows, "NEW_6M") + monthly_rows(old_rows, "OLD_LEGACY")
    counts = count_rows([r for r in new_rows if str(r["session_date"])[:10] in ov], "NEW_6M_OVERLAP")
    counts += count_rows(old_ov, "OLD_OVERLAP")
    counts += count_rows(new_rows, "NEW_6M_ALL") + count_rows(old_rows, "OLD_ALL")
    reasons = count_reasons(pairs, old_spot, new_spot, range_start, range_end)

    headline = {
        "old_oos": scoped(in_range(old_rows, *OLD_SPLITS["OOS"])),
        "new_oos": scoped(in_range(new_rows, *NEW_SPLITS["OOS"])),
        "old_overlap": scoped(old_ov),
        "new_overlap": scoped([r for r in new_rows if str(r["session_date"])[:10] in ov]),
        "new_same_window_as_old_oos": scoped(in_range(new_rows, *OLD_SPLITS["OOS"])),
        "old_splits": {s: scoped(in_range(old_rows, *v)) for s, v in OLD_SPLITS.items()},
        "new_splits": {s: scoped(in_range(new_rows, *v)) for s, v in NEW_SPLITS.items()},
    }
    n_both = len(both)
    tick_effect = None
    sm = {s["scenario"]: s for s in scen_sum}
    if "LEGACY_PREMIUM_OLD_TICK" in sm and "LEGACY_PREMIUM_TICK_0_05" in sm:
        a = sm["LEGACY_PREMIUM_OLD_TICK"]["COMBINED"]["accuracy"]
        b = sm["LEGACY_PREMIUM_TICK_0_05"]["COMBINED"]["accuracy"]
        tick_effect = None if a is None or b is None else b - a
    opt_shares = [v["share_close_diff_gt_0_5pct"] for v in opt_cmp.values() if v["share_close_diff_gt_0_5pct"] is not None]
    unexplained = sum(1 for r in comp if r["outcome_diff_reason"] == "UNEXPLAINED")
    ev = {
        "underlying_share_diff": und_cmp_overlap["share_close_diff_gt_0_5pct"] or 0.0,
        "option_share_diff": (sum(opt_shares) / len(opt_shares)) if opt_shares else 0.0,
        "contract_mismatch_share": (sum(1 for r in sel if r["status"] == "BOTH" and r["mismatch"]) / n_both)
        if n_both else 0.0,
        "entry_gt5_share": (sum(1 for r in entry if r["gt_5pct"]) / len(entry)) if entry else 0.0,
        "alignment_pass": align["passed"],
        "sl_target_diff_share": (sum(1 for r in slt if not (_close(r["old_sl"], r["new_sl"])
                                                            and _close(r["old_target"], r["new_target"]))) / len(slt))
        if slt else 0.0,
        "gap_losses": sum(1 for g in gaps if g["dataset"] == "NEW_6M" and g["loss_caused_by_data_gap"]),
        "old_tick5_share": (sum(r["tick_inferred"] == FIVE for r in old_rows if r.get("entry") is not None)
                            / max(1, sum(r.get("entry") is not None for r in old_rows))),
        "tick_effect_pp": tick_effect,
        "new_contracts_valid": new_contract_ok,
        "unexplained_outcome_diffs": unexplained,
        "old_reproduction_share": sm.get("LEGACY_PREMIUM_OLD_TICK", {}).get("reproduction_share"),
    }
    classes, why = classify(ev)
    rec = recommend(ev)
    acc_gap = None
    a, b = headline["old_overlap"]["COMBINED"]["accuracy"], headline["new_overlap"]["COMBINED"]["accuracy"]
    if a is not None and b is not None:
        acc_gap = b - a
    severity = ("HIGH" if not align["passed"] or not new_contract_ok or (acc_gap is not None and abs(acc_gap) >= 0.10)
                else "MEDIUM" if acc_gap is not None and abs(acc_gap) >= 0.05 else "LOW")
    period_mismatch = OLD_SPLITS["OOS"] != NEW_SPLITS["OOS"]
    primary = root_cause_text(ev, classes, headline, period_mismatch, sm)
    return {
        "heading": HEADING, "banner": BANNER, "s2_changed": False, "orders": 0, "trading_mode": "PAPER",
        "live_trading_enabled": False, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "implementation": implementation, "overlap_days": overlap, "old_days": sorted(old_days),
        "new_days": sorted(new_days), "headline": headline, "period_mismatch": period_mismatch,
        "datasets": {"legacy": legacy_profile, "six_month": six_profile, "underlying_all": und_cmp,
                     "underlying_overlap": und_cmp_overlap, "option_contracts": opt_cmp},
        "pairs_summary": dict(Counter(p["status"] for p in pairs)), "comparison": comp, "option_selection": sel,
        "entry_drift": entry, "sl_target": slt, "scenarios": scen_sum, "alignment": align, "gaps": gaps,
        "contracts": contracts, "monthly": monthly, "counts": counts, "count_reasons": reasons,
        "evidence": ev, "classes": classes, "class_reasons": why, "severity": severity,
        "primary_root_cause": primary, "recommendation": rec,
    }


def root_cause_text(ev: dict, classes: Sequence[str], headline: dict, period_mismatch: bool, sm: dict) -> str:
    parts = []
    if period_mismatch:
        no, nw = headline["old_oos"]["COMBINED"]["accuracy"], headline["new_same_window_as_old_oos"]["COMBINED"]["accuracy"]
        parts.append(f"COMPARISON PERIOD MISMATCH: old OOS = {OLD_SPLITS['OOS'][0]}..{OLD_SPLITS['OOS'][1]}, new OOS = "
                     f"{NEW_SPLITS['OOS'][0]}..{NEW_SPLITS['OOS'][1]}; on the old OOS window the new data gives "
                     f"{_pct(nw)} combined vs old {_pct(no)}")
    for c in classes:
        if c != "H":
            parts.append(f"{c} {CLASSES[c]}")
    return "; ".join(parts) or "NO MATERIAL DIFFERENCE"


# ------------------------------------------------------------------ reports
def _csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, default=str) if isinstance(v, (list, dict, tuple)) else v)
                        for k, v in r.items()})


def _m(m: dict) -> str:
    pf = m["profit_factor"]
    pfs = "N/A" if pf is None else ("INF" if math.isinf(pf) else f"{pf:.2f}")
    return f"{m['trades']} trades / {m['wins']} W / {m['losses']} L / {_pct(m['accuracy'])} / PF {pfs} / Rs {m['net_pnl']:,.2f}"


REPORT_FILES = ("s2_baseline_drift_audit.md", "s2_baseline_drift_signal_comparison.csv", "s2_option_selection_drift.csv",
                "s2_entry_price_drift.csv", "s2_monthly_performance.csv", "s2_sl_target_drift.csv",
                "s2_drift_scenarios.csv", "s2_data_gap_audit.csv", "s2_contract_validation.csv",
                "s2_signal_count_audit.csv")


def markdown(a: dict) -> str:
    h, ev = a["headline"], a["evidence"]
    sel = [r for r in a["option_selection"] if r["status"] == "BOTH"]
    md = [f"# {HEADING}", "", f"**{BANNER}** | S2 changed: NO | Orders: 0 | LIVE_TRADING_ENABLED=false | "
          f"generated {a['generated_at']}", "",
          "## Executive Summary", "",
          f"- Primary root cause: **{a['primary_root_cause']}**",
          f"- Classification: {', '.join(f'{c} {CLASSES[c]}' for c in a['classes'])}",
          f"- Severity: **{a['severity']}** | Recommendation: **{a['recommendation']}**",
          f"- Overlapping sessions compared signal-by-signal: {len(a['overlap_days'])} "
          f"({a['overlap_days'][0] if a['overlap_days'] else '-'}..{a['overlap_days'][-1] if a['overlap_days'] else '-'})",
          "", "## 1. Exact S2 implementation (read from code and config)", ""]
    md += [f"- **{k}**: {v}" for k, v in a["implementation"].items()]
    md += ["", "## Old vs New Results", "", "| Basis | BUY | SELL | COMBINED |", "|---|---|---|---|"]
    rows = [("OLD OOS (legacy, 09-25..10-01)", h["old_oos"]), ("NEW OOS (6-month, 08-01..08-31)", h["new_oos"]),
            ("NEW on the OLD OOS window (09-25..09-30)", h["new_same_window_as_old_oos"]),
            ("OLD on overlapping sessions", h["old_overlap"]), ("NEW on overlapping sessions", h["new_overlap"])]
    rows += [(f"OLD {s}", v) for s, v in h["old_splits"].items()] + [(f"NEW {s}", v) for s, v in h["new_splits"].items()]
    md += [f"| {n} | {_m(v['BUY'])} | {_m(v['SELL'])} | {_m(v['COMBINED'])} |" for n, v in rows]
    md += ["", f"The two headline OOS figures cover different months (period mismatch: {a['period_mismatch']})."]
    md += ["", "## 2. Dataset comparison", "", "| Item | OLD legacy store | NEW upstox_6m |", "|---|---|---|"]
    lp, sp = a["datasets"]["legacy"] or {}, a["datasets"]["six_month"] or {}
    for k in ("files", "candles", "instrument_keys", "strikes", "ce", "pe", "timezones", "zero_volume_share",
              "zero_oi_share", "duplicate_timestamps", "off_grid", "missing_inside_span"):
        md.append(f"| option {k} | {lp.get(k, '-')} | {sp.get(k, '-')} |")
    md.append(f"| option expiries | {', '.join(lp.get('expiries', [])[:12]) or '-'} | "
              f"{', '.join(sp.get('expiries', [])[:12]) or '-'} |")
    for name in ("underlying_all", "underlying_overlap"):
        u = a["datasets"][name]
        md.append(f"| NIFTY {name.replace('_', ' ')}: matched / only-old / only-new | {u['matched']} / {u['only_a']} / "
                  f"{u['only_b']} | close MAE {_num(u.get('close_mae'))}, max {_num(u.get('close_max'))}, best lag "
                  f"{u.get('best_lag')} |")
    oc = a["datasets"]["option_contracts"]
    tot = {k: sum((v.get(k) or 0) for v in oc.values()) for k in ("matched", "only_a", "only_b")}
    lags = Counter(v.get("best_lag") for v in oc.values() if v.get("best_lag") is not None)
    md += ["", f"Option candles of the {len(oc)} old-selected contracts: matched {tot['matched']}, only legacy "
           f"{tot['only_a']}, only 6-month {tot['only_b']}; best alignment lag per contract {dict(lags)}; mean share of "
           f"candles with close diff > 0.5%: {_pct(ev['option_share_diff'])}."]
    md += ["", "## Signal Count Comparison", "", f"Pairs on overlapping sessions: {a['pairs_summary']}", "",
           "| Reason a signal exists in only one dataset | Count |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in Counter(r["reason"] for r in a["count_reasons"]).most_common()]
    cm = defaultdict(lambda: Counter())
    for r in a["counts"]:
        cm[r["dataset"]][(r["month"], r["direction"])] += r["signals"]
    md += ["", "| Dataset | Month | BUY | SELL |", "|---|---|---|---|"]
    for ds, c in cm.items():
        for m in sorted({k[0] for k in c}):
            md.append(f"| {ds} | {m} | {c[(m, 'BUY')]} | {c[(m, 'SELL')]} |")
    mism = Counter(x for r in sel for x in r["mismatches"].split(",") if x)
    md += ["", "## Option Selection Drift", "",
           f"Matched signals: {len(sel)}; with any mismatch: {sum(r['mismatch'] for r in sel)}; by field: "
           f"{dict(mism) or 'none'} (full list in s2_option_selection_drift.csv)."]
    e = a["entry_drift"]
    md += ["", "## Entry Price Drift", "",
           f"Matched signals with both entries: {len(e)}; same contract: {sum(r['same_contract'] for r in e)}; "
           + "; ".join(f"> {int(t * 100)}%: {sum(r[f'gt_{int(t * 100)}pct'] for r in e)}" for t in DRIFT_FLAGS)
           + f"; median abs diff {_num(float(pd.Series([r['abs_diff'] for r in e]).median()) if e else None)}.",
           f"Old rows whose entry/SL/target all lie on a Rs 5 grid: {_pct(ev['old_tick5_share'])} (new replay rounds "
           "to the 0.05 default because 6-month contracts carry no tick size)."]
    al = a["alignment"]
    md += ["", "## Candle Alignment", "", f"**{'PASS' if al['passed'] else 'FAIL'}** - {al['checked']} new signals "
           f"checked (signal candle present, entry = signal + 5 min open, ATR only from candles <= signal, no "
           f"duplicate / off-grid / shifted candle). Problems: {al['problems'] or 'none'} {al['examples'][:5]}. "
           f"Old rows with entry exactly at signal + 5 min: {al.get('old_next_bar_entries', '-')}.",
           f"Underlying best lag (old vs new, 5-minute steps): {a['datasets']['underlying_overlap'].get('best_lag')}; "
           f"option best lags: {dict(lags)}."]
    src = Counter(x for r in a["sl_target"] for x in r["difference_sources"].split(",") if x and x != "NONE")
    md += ["", "## SL/Target Drift", "", f"Matched signals: {len(a['sl_target'])}; SL or target different: "
           f"{_pct(ev['sl_target_diff_share'])}; sources: {dict(src) or 'none'}. Underlying ATR is not used by S2 "
           "exits."]
    oc2 = Counter((r["old_outcome"], r["new_outcome"]) for r in a["comparison"] if r["status"] == "BOTH")
    rs = Counter((r["outcome_diff_reason"] or "").split(" ")[0].split(";")[0] for r in a["comparison"]
                 if r["status"] == "BOTH" and not r["same_outcome"])
    md += ["", "## Outcome Drift", "", "| OLD outcome | NEW outcome | Signals |", "|---|---|---|"]
    md += [f"| {k[0]} | {k[1]} | {v} |" for k, v in sorted(oc2.items(), key=lambda kv: -kv[1])]
    md += ["", f"First explaining difference for changed outcomes: {dict(rs) or 'none'}; UNEXPLAINED: "
           f"{ev['unexplained_outcome_diffs']} (every row in s2_baseline_drift_signal_comparison.csv).", "",
           "Diagnostic re-evaluation of the OLD signals (same signal, same contract, exact S2 evaluation code):", "",
           "| Scenario | Reproduces old outcome+P&L | BUY | SELL | COMBINED |", "|---|---|---|---|---|"]
    for s in a["scenarios"]:
        md.append(f"| {s['scenario']} | {s['reproduces_old_outcome_and_pnl']}/{s['signals']} "
                  f"({_pct(s['reproduction_share'])}) | {_m(s['BUY'])} | {_m(s['SELL'])} | {_m(s['COMBINED'])} |")
    md += ["", "## Monthly Results", "", "| Dataset | Month | BUY | SELL | COMBINED |", "|---|---|---|---|---|"]
    mm = defaultdict(dict)
    for r in a["monthly"]:
        mm[(r["dataset"], r["month"])][r["scope"]] = r
    for (ds, m), v in sorted(mm.items()):
        md.append(f"| {ds} | {m} | {_m(v['BUY'])} | {_m(v['SELL'])} | {_m(v['COMBINED'])} |")
    g = Counter((x["dataset"], x["loss_caused_by_data_gap"]) for x in a["gaps"])
    md += ["", "## Data Gaps", "", "| Dataset | Losing/ambiguous signals | Caused by data gap |", "|---|---|---|"]
    for ds in sorted({x["dataset"] for x in a["gaps"]}):
        md.append(f"| {ds} | {g[(ds, False)] + g[(ds, True)]} | {g[(ds, True)]} |")
    md.append("Outcomes are not changed; gaps are only identified.")
    cc = Counter((c["dataset"], c["valid"]) for c in a["contracts"])
    fails = Counter((c["dataset"], k) for c in a["contracts"] for k in
                    ("contract_existed", "expiry_not_before_signal", "nearest_listed_expiry",
                     "option_type_matches_direction", "atm_strike", "instrument_key_consistent") if not c[k])
    md += ["", "## Contract Validation", "", "| Dataset | Valid | Invalid | Impossible |", "|---|---|---|---|"]
    for ds in sorted({c["dataset"] for c in a["contracts"]}):
        md.append(f"| {ds} | {cc[(ds, True)]} | {cc[(ds, False)]} | "
                  f"{sum(1 for c in a['contracts'] if c['dataset'] == ds and c['impossible'])} |")
    md += [f"Failed checks: {dict(fails) or 'none'}"]
    md += ["", "## Root Cause", ""]
    md += [f"- **{c} {CLASSES[c]}**: {a['class_reasons'][c]}" for c in a["classes"]]
    if a["period_mismatch"]:
        md.append("- **COMPARISON PERIOD MISMATCH**: " + a["primary_root_cause"].split(";")[0])
    md += ["", "## Severity", "", f"**{a['severity']}**", "", "## Recommended Next Step", "",
           f"**{a['recommendation']}**", "",
           "No strategy parameter was changed and nothing was optimized. S2 changed: NO. Orders: 0."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    p["s2_baseline_drift_audit.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["s2_baseline_drift_signal_comparison.csv"], a["comparison"])
    _csv(p["s2_option_selection_drift.csv"], a["option_selection"])
    _csv(p["s2_entry_price_drift.csv"], a["entry_drift"])
    _csv(p["s2_monthly_performance.csv"], a["monthly"])
    _csv(p["s2_sl_target_drift.csv"], a["sl_target"])
    _csv(p["s2_drift_scenarios.csv"], [{"scenario": s["scenario"], "scope": sc, "signals": s["signals"],
                                         "reproduces_old": s["reproduces_old_outcome_and_pnl"], **s[sc]}
                                        for s in a["scenarios"] for sc in SCOPES])
    _csv(p["s2_data_gap_audit.csv"], a["gaps"])
    _csv(p["s2_contract_validation.csv"], a["contracts"])
    _csv(p["s2_signal_count_audit.csv"], a["counts"] + [dict(r, dataset="ONLY_ONE_DATASET") for r in a["count_reasons"]])
    return p
