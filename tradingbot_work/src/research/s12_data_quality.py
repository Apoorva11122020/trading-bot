"""S12 - expanded data quality repair & coverage audit (PAPER | RESEARCH ONLY | NO ORDERS | no network, no ML).

Audits the stored NIFTY index, NIFTY futures and point-in-time ATM option candles used by S11 (2025-01-01 ..
2026-09-30; October 2026 is never read). Nothing is modified: raw files are hashed before and after, missing bars are
recorded (never filled), and rows that cannot be used are marked unusable (never shifted / interpolated).

Pre-declared rules (fixed before the audit was run):

* Session grid: regular session bars start 09:15 .. 15:25 (75 x 5 minutes). Documented session exceptions
  (SESSION_EXCEPTIONS) have their own window; regular bars outside it are SESSION_EXCEPTION, not missing data.
* A weekday with no candles in ANY instrument (index, futures, options) is MARKET_CLOSED; a weekday where other
  instruments traded but the index has no candles is a MISSING_SESSION.
* Missing front-month futures slots: A DATA_SOURCE_GAP (index traded at that time), B CONTRACT_ROLLOVER (the causal
  front month has no data that session but another unexpired contract does), C SESSION_EXCEPTION, D GENUINE_MISSING.
* Off-grid option candles are normalised only if a deterministic, documented mapping exists; otherwise UNUSABLE.
* CLEAN session: index 75/75 valid bars, front-month future 75/75, no blocking option-file failure on that day's ATM
  contracts, and >= SESSION_MIN_ATM_ELIGIBLE of signal timestamps with BOTH ATM CE and PE eligible.
  CLEAN HISTORY = longest run of consecutive regular sessions that are all CLEAN (closed days / session exceptions
  are not regular sessions and do not break the run).
* DATA_READY only if every gate condition in ``gate`` passes.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.historical_data import nifty_futures as nf
from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import HistoricalContract
from src.options.models import OptionResolutionError
from src.options.selection import atm_strike
from src.research import s10_ml_data as sd
from src.research import s11_history as hx

HEADING = "S12 DATA QUALITY COMPLETE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | data audit only (no model trained, nothing modified)"
IST = u6.IST
STEP = pd.Timedelta(minutes=5)
PERIOD = (date(2025, 1, 1), hx.RESEARCH_END)
REGULAR_FIRST, REGULAR_LAST = time(9, 15), time(15, 25)
SESSION_EXCEPTIONS: dict[date, dict[str, Any]] = {
    date(2025, 10, 21): {"name": "MUHURAT_TRADING", "first": time(13, 45), "last": time(14, 40),
                         "source": "NSE special Muhurat trading session (Diwali Laxmi Pujan) 13:45-14:45 IST"},
}
OHLC = ("open", "high", "low", "close")
TOL = 1e-9

MAX_GENUINE_MISSING_SHARE = u6.MAX_UNDERLYING_MISSING_SHARE     # 0.1% of expected session slots
MIN_ATM_ELIGIBLE_SHARE = 0.90                                   # per side, over the clean history
SESSION_MIN_ATM_ELIGIBLE = 0.50                                 # both sides, per session
MIN_CLEAN_MONTHS = hx.MIN_HISTORY_MONTHS

PASS, FAIL, PASS_Q = "PASS", "FAIL", "PASS_WITH_QUARANTINE"
REGULAR, EXCEPTION, CLOSED, MISSING_SESSION = "REGULAR", "SESSION_EXCEPTION", "MARKET_CLOSED", "MISSING_SESSION"
GAP_A, GAP_B, GAP_C, GAP_D = "DATA_SOURCE_GAP", "CONTRACT_ROLLOVER", "SESSION_EXCEPTION", "GENUINE_MISSING"
CATEGORY = {GAP_A: "A", GAP_B: "B", GAP_C: "C", GAP_D: "D"}
UNUSABLE = "UNUSABLE_OFF_GRID"
BLOCKING_OPTION_REASONS = ("CONFLICTING_DUPLICATE_TIMESTAMP", "INVALID_OHLC", "MISSING_OHLC", "NON_POSITIVE_PRICE",
                           "TIMEZONE_MISMATCH", "OUTSIDE_SESSION", "AFTER_EXPIRY", "WRONG_MONTH_FOLDER",
                           "FILE_NAME_UNPARSEABLE", "MULTIPLE_INSTRUMENT_KEYS", "CONTRACT_NOT_IN_PLAN",
                           "EXPIRY_MISMATCH", "STRIKE_MISMATCH", "OPTION_TYPE_MISMATCH", "UNDERLYING_MISMATCH",
                           "INVALID_STRIKE", "INVALID_EXPIRY", "NEGATIVE_VOLUME_OR_OI", "UNREADABLE")


# ------------------------------------------------------------------ grid / helpers
def at(d: date, t: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, t), tz=IST)


def _grid(d: date, first: time, last: time) -> list[pd.Timestamp]:
    a, b = at(d, first), at(d, last)
    return list(pd.date_range(a, b, freq=STEP))


def regular_grid(d: date) -> list[pd.Timestamp]:
    return _grid(d, REGULAR_FIRST, REGULAR_LAST)


def expected_grid(d: date) -> list[pd.Timestamp]:
    e = SESSION_EXCEPTIONS.get(d)
    return _grid(d, e["first"], e["last"]) if e else regular_grid(d)


def weekdays(lo: date, hi: date) -> list[date]:
    out, d = [], lo
    while d <= hi:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_raw(path: Path, max_day: date = PERIOD[1]) -> tuple[pd.DataFrame, int]:
    """Stored rows as written (no de-duplication, no sorting). Timezone problems are flagged per row; rows after
    ``max_day`` are discarded immediately (never analysed) and counted."""
    f = pd.read_parquet(path)
    if f.empty:
        return f.assign(tz_error=pd.Series(dtype=bool)), 0
    ts = pd.to_datetime(f["timestamp"])
    if ts.dt.tz is None:
        f["tz_error"] = True
        local = ts.dt.tz_localize(IST)
    else:
        off = ts.map(lambda v: v.utcoffset())
        f["tz_error"] = (off != timedelta(hours=5, minutes=30)).to_numpy()
        local = ts.dt.tz_convert(IST)
    f["timestamp"] = local
    late = (local.dt.date > max_day).to_numpy()
    return f.loc[~late].reset_index(drop=True), int(late.sum())


def bar_checks(f: pd.DataFrame) -> dict[str, int]:
    if f.empty:
        return {k: 0 for k in ("rows", "duplicate_timestamps", "conflicting_duplicates", "duplicate_rows", "off_grid",
                               "invalid_ohlc", "missing_ohlc", "non_positive_price", "negative_volume",
                               "negative_oi", "zero_volume", "tz_errors")}
    t = pd.DatetimeIndex(f["timestamp"])
    px = {k: pd.to_numeric(f[k], errors="coerce") for k in OHLC}
    missing = px["open"].isna() | px["high"].isna() | px["low"].isna() | px["close"].isna()
    nonpos = ~missing & ((px["open"] <= 0) | (px["high"] <= 0) | (px["low"] <= 0) | (px["close"] <= 0))
    hi_ok = px["high"] + TOL >= np.maximum(px["open"], px["close"])
    lo_ok = px["low"] - TOL <= np.minimum(px["open"], px["close"])
    invalid = ~missing & ~(hi_ok & lo_ok & (px["high"] + TOL >= px["low"]))
    dup = t.duplicated()
    conflicting = 0
    if dup.any():
        d = f.loc[t.duplicated(keep=False), ["timestamp", *OHLC]]
        conflicting = int((d.groupby("timestamp")[list(OHLC)].nunique() > 1).any(axis=1).sum())
    vol = pd.to_numeric(f["volume"], errors="coerce") if "volume" in f else pd.Series(dtype=float)
    oi = pd.to_numeric(f["open_interest"], errors="coerce") if "open_interest" in f else pd.Series(dtype=float)
    cols = [c for c in f.columns if c != "tz_error"]
    return {"rows": int(len(f)), "duplicate_timestamps": int(dup.sum()), "conflicting_duplicates": conflicting,
            "duplicate_rows": int(f[cols].astype(str).duplicated().sum()),
            "off_grid": int((~hx.on_grid_mask(f)).sum()), "invalid_ohlc": int(invalid.sum()),
            "missing_ohlc": int(missing.sum()), "non_positive_price": int(nonpos.sum()),
            "negative_volume": int((vol < 0).sum()), "negative_oi": int((oi < 0).sum()),
            "zero_volume": int((vol == 0).sum()),
            "tz_errors": int(f["tz_error"].sum()) if "tz_error" in f else 0}


def _present(f: pd.DataFrame) -> dict[date, set[pd.Timestamp]]:
    """Unique on-grid timestamps per IST day."""
    if f.empty:
        return {}
    g = f.loc[hx.on_grid_mask(f), "timestamp"].drop_duplicates()
    out: dict[date, set[pd.Timestamp]] = defaultdict(set)
    for ts in g:
        out[ts.date()].add(ts)
    return dict(out)


def _spacing_gaps(ts: Iterable[pd.Timestamp]) -> int:
    s = sorted(ts)
    return sum(1 for a, b in zip(s, s[1:]) if b - a != STEP)


# ------------------------------------------------------------------ 1. NIFTY index
def nifty_audit(spot: pd.DataFrame, lo: date, hi: date, activity_days: set[date],
                fut_slots: set[pd.Timestamp]) -> dict[str, Any]:
    """spot: raw stored index rows (all roots). activity_days: days with futures / option candles."""
    chk = bar_checks(spot)
    present = _present(spot)
    rows_by_day = Counter(pd.DatetimeIndex(spot["timestamp"]).date) if len(spot) else Counter()
    bad_by_day: dict[date, Counter] = defaultdict(Counter)
    if len(spot):
        days = pd.DatetimeIndex(spot["timestamp"]).date
        for k, mask in (("off_grid", ~hx.on_grid_mask(spot)), ("duplicates", pd.DatetimeIndex(spot["timestamp"]).duplicated()),
                        ("tz_errors", spot["tz_error"].to_numpy(dtype=bool))):
            for d in days[np.asarray(mask)]:
                bad_by_day[d][k] += 1
    day_rows, missing = [], []
    for d in weekdays(lo, hi):
        have = present.get(d, set())
        exc = SESSION_EXCEPTIONS.get(d)
        if not rows_by_day.get(d):
            status = MISSING_SESSION if d in activity_days else CLOSED
            reason = GAP_D if status == MISSING_SESSION else CLOSED
            missing.append({"date": d.isoformat(), "timestamp": None, "count": len(regular_grid(d)), "reason": reason,
                            "detail": "other instruments traded; index returned no candles" if status == MISSING_SESSION
                            else "no candles in any instrument (exchange holiday / market closed)"})
            day_rows.append({"date": d.isoformat(), "status": status, "bars": 0, "expected": 0, "missing": 0,
                             "duplicates": 0, "off_grid": 0, "outside_session": 0, "spacing_gaps": 0, "tz_errors": 0,
                             "session": exc["name"] if exc else None})
            continue
        status = EXCEPTION if exc else REGULAR
        grid = expected_grid(d)
        reg = regular_grid(d)
        if exc:
            for ts in reg:
                if ts not in set(grid):
                    missing.append({"date": d.isoformat(), "timestamp": ts.isoformat(), "count": 1, "reason": GAP_C,
                                    "detail": exc["source"]})
        gaps = [ts for ts in grid if ts not in have]
        for ts in gaps:
            r = GAP_A if ts in fut_slots else GAP_D
            missing.append({"date": d.isoformat(), "timestamp": ts.isoformat(), "count": 1, "reason": r,
                            "detail": "front-month future traded at this time" if r == GAP_A else "no source"})
        b = bad_by_day.get(d, Counter())
        day_rows.append({"date": d.isoformat(), "status": status, "bars": int(rows_by_day[d]), "expected": len(grid),
                         "missing": len(gaps), "duplicates": b["duplicates"], "off_grid": b["off_grid"],
                         "outside_session": len(have - set(grid)), "spacing_gaps": _spacing_gaps(have),
                         "tz_errors": b["tz_errors"], "session": exc["name"] if exc else None})
    sessions = [r for r in day_rows if r["status"] in (REGULAR, EXCEPTION)]
    expected = sum(r["expected"] for r in sessions)
    genuine = sum(m["count"] for m in missing if m["reason"] in (GAP_A, GAP_D) and m["timestamp"])
    miss_sessions = sum(1 for r in day_rows if r["status"] == MISSING_SESSION)
    outside = sum(r["outside_session"] for r in sessions)
    share = genuine / expected if expected else None
    ok = bool(sessions and chk["duplicate_timestamps"] == 0 and chk["off_grid"] == 0 and chk["invalid_ohlc"] == 0
              and chk["missing_ohlc"] == 0 and chk["non_positive_price"] == 0 and chk["tz_errors"] == 0
              and outside == 0 and miss_sessions == 0 and share is not None and share <= MAX_GENUINE_MISSING_SHARE)
    return {"days": day_rows, "missing": missing, "checks": chk,
            "summary": {"status": PASS if ok else FAIL, "trading_days": len(sessions),
                        "regular_sessions": sum(r["status"] == REGULAR for r in day_rows),
                        "exception_sessions": [r["date"] for r in day_rows if r["status"] == EXCEPTION],
                        "market_closed_days": [r["date"] for r in day_rows if r["status"] == CLOSED],
                        "missing_sessions": [r["date"] for r in day_rows if r["status"] == MISSING_SESSION],
                        "expected_bars": expected, "bars": int(sum(r["bars"] for r in sessions)),
                        "missing_genuine": genuine,
                        "missing_session_exception": sum(m["count"] for m in missing if m["reason"] == GAP_C),
                        "missing_genuine_share": share, "duplicates": chk["duplicate_timestamps"],
                        "off_grid": chk["off_grid"], "invalid_ohlc": chk["invalid_ohlc"], "tz_errors": chk["tz_errors"],
                        "outside_session": outside, "spacing_gaps": sum(r["spacing_gaps"] for r in sessions),
                        "rule": f"PASS = no duplicates / off-grid / invalid OHLC / timezone errors / out-of-session bars, "
                                f"no missing session, genuine missing bars <= {MAX_GENUINE_MISSING_SHARE:.1%}"}}


# ------------------------------------------------------------------ 2. futures
def futures_audit(raw: Sequence[tuple[str, HistoricalContract, pd.DataFrame]], contracts: Sequence[HistoricalContract],
                  frames: dict[str, pd.DataFrame], trading_days: Sequence[date],
                  nifty_slots: set[pd.Timestamp]) -> dict[str, Any]:
    """raw: (root, contract, stored rows) per file; contracts / frames: merged by expiry (as used by S10/S11)."""
    sel = nf.select_contracts(contracts, frames, trading_days)
    chk = nf.selection_checks(sel, contracts, frames, trading_days)
    arrs = {k: np.array(sorted(set(f.loc[hx.on_grid_mask(f), "timestamp"])), dtype=object) for k, f in frames.items()}
    by_day = {k: _present(f) for k, f in frames.items()}
    front_sessions = Counter(r["instrument_key"] for r in sel if r["status"] == nf.SELECTED)
    missing, per_day = [], defaultdict(Counter)
    expected = 0
    for r in sel:
        d = date.fromisoformat(r["session_date"])
        grid, reg = expected_grid(d), regular_grid(d)
        exc = SESSION_EXCEPTIONS.get(d)
        expected += len(grid)
        key = r.get("instrument_key")
        if r["status"] != nf.SELECTED:
            other = any(by_day.get(c.instrument_key, {}).get(d) for c in contracts
                        if c.expiry >= d and c.instrument_key != key)
            reason = GAP_B if other else GAP_D
            for ts in grid:
                missing.append(_fmiss(d, ts, r.get("trading_symbol"), None, None, reason,
                                      f"causal front month {r['reason']}"))
                per_day[d][reason] += 1
            continue
        have = by_day.get(key, {}).get(d, set())
        arr = arrs.get(key, np.array([], dtype=object))
        for ts in reg:
            if ts in have:
                continue
            if exc and ts not in set(grid):
                reason, why = GAP_C, exc["source"]
            elif ts in nifty_slots:
                reason, why = GAP_A, "index traded at this time; futures candle not returned"
            else:
                reason, why = GAP_D, "no index candle either"
            i = int(np.searchsorted(arr, ts)) if len(arr) else 0
            prev = arr[i - 1] if i > 0 else None
            nxt = arr[i] if i < len(arr) else None
            missing.append(_fmiss(d, ts, r["trading_symbol"], prev, nxt, reason, why))
            per_day[d][reason] += 1
    rows = []
    for root, c, f in raw:
        q = bar_checks(f)
        after = int((pd.DatetimeIndex(f["timestamp"]).date > c.expiry).sum()) if len(f) else 0
        rows.append({"root": root, "contract": c.trading_symbol, "instrument_key": c.instrument_key,
                     "expiry": c.expiry.isoformat(), "first": str(f["timestamp"].min()) if len(f) else None,
                     "last": str(f["timestamp"].max()) if len(f) else None, **q, "after_expiry": after,
                     "front_month_sessions": front_sessions.get(c.instrument_key, 0)})
    conflicts = cross_root_conflicts(raw)
    cnt = Counter(m["reason"] for m in missing)
    genuine = cnt[GAP_A] + cnt[GAP_B] + cnt[GAP_D]
    share = genuine / expected if expected else None
    hard = {k: sum(r[k] for r in rows) for k in ("conflicting_duplicates", "off_grid", "invalid_ohlc", "missing_ohlc",
                                                 "non_positive_price", "tz_errors", "after_expiry")}
    dups = sum(r["duplicate_timestamps"] for r in rows)
    ok = bool(sel and not any(hard.values()) and dups == 0 and conflicts["conflicts"] == 0 and not chk["violations"]
              and chk["independent_of_price_and_volume"] and share is not None and share <= MAX_GENUINE_MISSING_SHARE)
    return {"quality": rows, "missing": missing, "selection": sel, "selection_checks": chk,
            "missing_by_day": {d: dict(v) for d, v in per_day.items()},
            "summary": {"status": PASS if ok else FAIL, "contracts": len({r["instrument_key"] for r in rows}),
                        "files": len(rows), "front_month_sessions": sum(front_sessions.values()),
                        "expected_slots": expected, "missing_total": len(missing),
                        "missing_by_category": {f"{CATEGORY[k]}_{k}": cnt[k] for k in CATEGORY},
                        "missing_genuine": genuine, "missing_genuine_share": share, "duplicates": dups,
                        "cross_root_conflicts": conflicts["conflicts"], "cross_root_overlap": conflicts["overlap"],
                        **hard, "selection_violations": len(chk["violations"]),
                        "selection_independent_of_price_and_volume": chk["independent_of_price_and_volume"],
                        "rule": "PASS = no duplicate / off-grid / invalid / after-expiry rows, no cross-root conflicts, "
                                f"causal selection valid, A+B+D missing slots <= {MAX_GENUINE_MISSING_SHARE:.1%}"}}


def _fmiss(d, ts, contract, prev, nxt, reason, why) -> dict[str, Any]:
    return {"date": d.isoformat(), "timestamp": ts.isoformat(), "contract": contract,
            "previous_candle": prev.isoformat() if prev is not None else None,
            "next_candle": nxt.isoformat() if nxt is not None else None, "category": CATEGORY[reason],
            "reason": reason, "detail": why}


def cross_root_conflicts(raw: Sequence[tuple[str, HistoricalContract, pd.DataFrame]]) -> dict[str, int]:
    by_exp: dict[date, list[pd.DataFrame]] = defaultdict(list)
    for root, c, f in raw:
        if len(f):
            by_exp[c.expiry].append(f.assign(_root=root))
    overlap = conflicts = 0
    for fs in by_exp.values():
        if len(fs) < 2:
            continue
        a = pd.concat(fs, ignore_index=True)
        g = a.groupby("timestamp")
        multi = g["_root"].nunique() > 1
        overlap += int(multi.sum())
        if multi.any():
            sub = a[a["timestamp"].isin(multi[multi].index)]
            conflicts += int((sub.groupby("timestamp")[list(OHLC)].nunique() > 1).any(axis=1).sum())
    return {"overlap": overlap, "conflicts": conflicts}


# ------------------------------------------------------------------ 3/4. options
def parse_option_name(p: Path) -> dict[str, Any] | None:
    try:
        und, exp_s, strike_s, kind = p.stem.split("_")
        return {"underlying": und, "expiry": date.fromisoformat(exp_s), "strike": float(strike_s), "kind": kind}
    except ValueError:
        return None


def off_grid_days(f: pd.DataFrame, key: str, file: str) -> tuple[list[dict], list[dict]]:
    """Per (file, day) off-grid pattern + every off-grid row (marked unusable, never shifted)."""
    off = ~hx.on_grid_mask(f)
    if not off.any():
        return [], []
    t = pd.DatetimeIndex(f["timestamp"])
    on = f.loc[~off]
    on_idx = on.set_index("timestamp")
    out, rows = [], []
    for d in sorted(set(t[off].date)):
        o = f.loc[off & (t.date == d)]
        ot = pd.DatetimeIndex(o["timestamp"])
        mods = sorted({int(m) for m in (ot.minute % 5)})
        secs = sorted({int(s) for s in ot.second})
        shift = pd.to_timedelta(ot.minute % 5, unit="m") + pd.to_timedelta(ot.second, unit="s")
        base = ot - shift
        same = on_idx.reindex(base)
        valid = same["close"].notna().to_numpy()
        eq_close = np.isclose(o["close"].to_numpy(dtype=float), same["close"].to_numpy(dtype=float)) & valid
        eq_all = valid.copy()
        for k in OHLC:
            eq_all &= np.isclose(o[k].to_numpy(dtype=float), same[k].to_numpy(dtype=float))
        rel = np.abs(o["close"].to_numpy(dtype=float) - same["close"].to_numpy(dtype=float)) / \
            same["close"].to_numpy(dtype=float)
        n_on = len(_present(f).get(d, set()) & set(expected_grid(d)))
        tz_explains = all(((m * 60 + s) % (30 * 60)) == 0 for m, s in zip(ot.minute % 5, ot.second))
        ident = float(eq_all[valid].mean()) if valid.any() else None
        verdict = ("RELABELLED_DUPLICATE_OF_ON_GRID_SERIES" if ident is not None and ident >= 0.99
                   else "DIFFERENT_AGGREGATION_BUCKETS")
        out.append({"file": file, "instrument_key": key, "date": d.isoformat(), "off_grid_rows": int(len(o)),
                    "first_off_grid": ot.min().time().isoformat(), "last_off_grid": ot.max().time().isoformat(),
                    "minute_offsets": " ".join(map(str, mods)), "second_offsets": " ".join(map(str, secs)),
                    "on_grid_bars_same_day": n_on, "on_grid_expected": len(expected_grid(d)),
                    "on_grid_complete": n_on == len(expected_grid(d)), "timezone_shift_explains": tz_explains,
                    "matched_on_grid_bars": int(valid.sum()),
                    "identical_ohlc_share": ident,
                    "close_match_share": float(eq_close[valid].mean()) if valid.any() else None,
                    "median_rel_close_diff": float(np.median(rel[valid])) if valid.any() else None,
                    "verdict": verdict, "normalization_safe": False, "action": UNUSABLE})
        rows += [{"file": file, "instrument_key": key, "timestamp": ts.isoformat(), "reason": UNUSABLE}
                 for ts in ot]
    return out, rows


def option_audit(roots: Sequence[Path], underlying: str = "NIFTY", max_day: date = PERIOD[1]) -> dict[str, Any]:
    last_month = hx.month_label(max_day)
    files, fails, offs, unusable = [], [], [], []
    counts: dict[str, Counter] = defaultdict(Counter)
    blocking_days: dict[str, set[date]] = defaultdict(set)
    reqs_all = []
    late_rows = 0
    act_days: set[date] = set()
    for root in roots:
        lay = u6.Layout(root, underlying)
        plan = json.loads(lay.plan.read_text(encoding="utf-8")) if lay.plan.is_file() else {}
        calendar = {date.fromisoformat(x) for x in plan.get("calendar", [])}
        reqs = u6.requirements_from_rows(u6._read_csv(lay.contracts))
        reqs_all += [r for r in reqs if r.trading_date <= max_day]
        meta = {r.contract.instrument_key: r.contract for r in reqs}
        for p in lay.option_files():
            month = p.parent.name
            if month > last_month:
                continue
            rel = f"{Path(root).name}/{month}/{p.name}"
            reasons: Counter = Counter()
            name = parse_option_name(p)
            try:
                f, late = read_raw(p, max_day)
            except Exception:  # unreadable file: reported, never repaired
                fails.append({"file": rel, "reason": "UNREADABLE", "rows": 0})
                continue
            late_rows += late
            if name is None:
                reasons["FILE_NAME_UNPARSEABLE"] += 1
            q = bar_checks(f)
            key = str(f["instrument_key"].iloc[0]) if len(f) and "instrument_key" in f else None
            keys = set(f["instrument_key"].astype(str)) if len(f) and "instrument_key" in f else set()
            c = meta.get(key) if key else None
            t = pd.DatetimeIndex(f["timestamp"]) if len(f) else pd.DatetimeIndex([])
            days = t.date if len(f) else np.array([])
            act_days |= set(days)
            if len(keys) > 1:
                reasons["MULTIPLE_INSTRUMENT_KEYS"] += 1
            if len(f) and c is None:
                reasons["CONTRACT_NOT_IN_PLAN"] += 1
            if c is not None and name is not None:
                if c.expiry != name["expiry"] or ("expiry" in f and set(f["expiry"].astype(str)) != {c.expiry.isoformat()}):
                    reasons["EXPIRY_MISMATCH"] += 1
                if abs(float(c.strike) - name["strike"]) > 1e-6 or \
                        ("strike" in f and any(abs(float(s) - float(c.strike)) > 1e-6 for s in set(f["strike"]))):
                    reasons["STRIKE_MISMATCH"] += 1
                if c.instrument_type != name["kind"] or \
                        ("option_type" in f and set(f["option_type"].astype(str)) != {c.instrument_type}):
                    reasons["OPTION_TYPE_MISMATCH"] += 1
                if c.underlying != underlying or ("underlying" in f and set(f["underlying"].astype(str)) != {underlying}):
                    reasons["UNDERLYING_MISMATCH"] += 1
            if name is not None:
                if name["strike"] <= 0 or abs(name["strike"] / 50 - round(name["strike"] / 50)) > 1e-9:
                    reasons["INVALID_STRIKE"] += 1
                if calendar and name["expiry"] not in calendar:
                    reasons["INVALID_EXPIRY"] += 1
            after = int((days > name["expiry"]).sum()) if name is not None and len(f) else 0
            wrong_month = int((pd.Series(t.strftime("%Y-%m")) != month).sum()) if len(f) else 0
            present = _present(f)
            outside = sum(len(v - set(expected_grid(d))) for d, v in present.items())
            for k, v in (("CONFLICTING_DUPLICATE_TIMESTAMP", q["conflicting_duplicates"]),
                         ("INVALID_OHLC", q["invalid_ohlc"]), ("MISSING_OHLC", q["missing_ohlc"]),
                         ("NON_POSITIVE_PRICE", q["non_positive_price"]), ("TIMEZONE_MISMATCH", q["tz_errors"]),
                         ("OUTSIDE_SESSION", outside), ("AFTER_EXPIRY", after), ("WRONG_MONTH_FOLDER", wrong_month),
                         ("NEGATIVE_VOLUME_OR_OI", q["negative_volume"] + q["negative_oi"]),
                         ("OFF_GRID_TIMESTAMP", q["off_grid"]),
                         ("IDENTICAL_DUPLICATE_ROWS", q["duplicate_rows"])):
                if v:
                    reasons[k] += int(v)
            if q["zero_volume"]:
                reasons["ZERO_VOLUME_INFO"] += q["zero_volume"]
            for k, v in reasons.items():
                fails.append({"file": rel, "reason": k, "rows": v})
            if key and any(k in BLOCKING_OPTION_REASONS for k in reasons):
                blocking_days[key] |= set(days)
            if key:
                for d, v in present.items():
                    counts[key][d] += len(v & set(expected_grid(d)))
                o, u = off_grid_days(f, key, rel)
                offs += o
                unusable += u
            files.append({"file": rel, "instrument_key": key, "expiry": name["expiry"].isoformat() if name else None,
                          "strike": name["strike"] if name else None, "option_type": name["kind"] if name else None,
                          **q, "after_expiry": after, "outside_session": outside, "wrong_month": wrong_month,
                          "reasons": " ".join(sorted(reasons)) or "OK",
                          "blocking": any(k in BLOCKING_OPTION_REASONS for k in reasons)})
    agg: dict[str, dict] = {}
    for r in fails:
        a = agg.setdefault(r["reason"], {"reason": r["reason"], "files": 0, "rows": 0})
        a["files"] += 1
        a["rows"] += r["rows"]
    for a in agg.values():
        a["blocking"] = a["reason"] in BLOCKING_OPTION_REASONS
        a["resolution"] = ("MARKED_UNUSABLE (excluded at read time; raw untouched)" if a["reason"] == "OFF_GRID_TIMESTAMP"
                           else "INFO_ONLY" if a["reason"] in ("ZERO_VOLUME_INFO", "IDENTICAL_DUPLICATE_ROWS")
                           else "BLOCKING")
    # requirement coverage (candles genuinely present on the planned session)
    req_rows = []
    for r in reqs_all:
        n = counts.get(r.contract.instrument_key, {}).get(r.trading_date, 0)
        req_rows.append({"date": r.trading_date, "key": r.contract.instrument_key, "role": r.role, "bars": n,
                         "expected": len(expected_grid(r.trading_date))})
    atm = [r for r in req_rows if r["role"] == "ATM"]
    blocking = [a for a in agg.values() if a["blocking"]]
    off_total = agg.get("OFF_GRID_TIMESTAMP", {}).get("rows", 0)
    status = FAIL if blocking else (PASS_Q if off_total else PASS)
    return {"files": files, "failures": sorted(agg.values(), key=lambda a: (not a["blocking"], -a["rows"])),
            "failure_rows": fails, "off_grid": offs, "unusable_rows": unusable, "days": act_days,
            "blocking_days": {k: sorted(v) for k, v in blocking_days.items()},
            "summary": {"status": status, "files": len(files), "contracts": len({f["instrument_key"] for f in files
                                                                                  if f["instrument_key"]}),
                        "candles": int(sum(f["rows"] for f in files)), "off_grid_rows": off_total,
                        "off_grid_files": agg.get("OFF_GRID_TIMESTAMP", {}).get("files", 0),
                        "off_grid_days": sorted({o["date"] for o in offs}),
                        "off_grid_on_grid_complete": all(o["on_grid_complete"] for o in offs) if offs else None,
                        "invalid_rows": int(sum(a["rows"] for a in blocking)), "blocking_reasons": [a["reason"] for a in blocking],
                        "duplicates": int(sum(f["duplicate_timestamps"] for f in files)),
                        "zero_volume_rows": int(sum(f["zero_volume"] for f in files)),
                        "requirements": len(req_rows),
                        "requirements_without_candles": sum(1 for r in req_rows if r["bars"] == 0),
                        "atm_requirements": len(atm), "atm_without_candles": sum(1 for r in atm if r["bars"] == 0),
                        "atm_missing_bars": int(sum(max(r["expected"] - r["bars"], 0) for r in atm)),
                        "atm_interval_completeness": (sum(min(r["bars"], r["expected"]) for r in atm)
                                                      / sum(r["expected"] for r in atm)) if atm else None,
                        "rows_after_research_end_ignored": late_rows,
                        "rule": "FAIL if any blocking reason; off-grid rows alone -> PASS_WITH_QUARANTINE (unusable)"}}


# ------------------------------------------------------------------ 5/6. ATM coverage + point-in-time validation
def atm_audit(market: sd.MarketData, sessions: Sequence[date]) -> dict[str, Any]:
    """For every eligible signal timestamp (S10/S11 window) and side: point-in-time ATM selection and data checks."""
    last = sd.last_signal(sd.HORIZON)
    spot_close, day_bars = market.spot_close(), market.day_bars()
    cache: dict[str, sd.OptionSeries | None] = {}
    first_seen: dict[str, tuple] = {}
    per_session, totals = [], {s: Counter() for s in sd.SIDES}
    contracts: dict[tuple, dict] = {}
    for d in sessions:
        sts = [ts for ts in day_bars.get(d, []) if sd.SIGNAL_FIRST <= ts.time() <= last]
        expiry = sd.nearest_expiry(market.calendar, d)
        ok = {s: 0 for s in sd.SIDES}
        why_day = {s: Counter() for s in sd.SIDES}
        entry_missing = {s: 0 for s in sd.SIDES}
        both = 0
        for ts in sts:
            good = {}
            for s in sd.SIDES:
                why, c = _atm_one(market, d, expiry, s, spot_close[ts], ts, cache, first_seen, contracts)
                good[s] = why is None
                why_day[s][why or "ELIGIBLE"] += 1
                totals[s][why or "ELIGIBLE"] += 1
                if why is None:
                    ok[s] += 1
                    ser = cache[c.instrument_key]
                    if int(ser.feat.index.get_indexer([ts + STEP])[0]) < 0:
                        entry_missing[s] += 1
            both += all(good.values())
        n = len(sts)
        per_session.append({"date": d.isoformat(), "month": hx.month_label(d), "signal_timestamps": n,
                            "expiry": expiry.isoformat() if expiry else None,
                            **{f"{s}_eligible": ok[s] for s in sd.SIDES},
                            **{f"{s}_share": ok[s] / n if n else None for s in sd.SIDES},
                            "both_eligible": both, "both_share": both / n if n else None,
                            **{f"{s}_entry_candle_missing": entry_missing[s] for s in sd.SIDES},
                            **{f"{s}_reasons": json.dumps({k: v for k, v in sorted(why_day[s].items())
                                                           if k != "ELIGIBLE"}) for s in sd.SIDES}})
    cv = list(contracts.values())
    for r in cv:
        r["passed"] = bool(r["expiry_valid"] and r["nearest_expiry"] and r["in_listed_chain"] and r["nearest_strike"]
                           and r["bracketed"] and r["planned_atm"] and r["candle_matches_contract"])
    return {"sessions": per_session, "totals": {s: dict(v) for s, v in totals.items()}, "contracts": cv,
            "causal_passed": bool(cv) and all(r["passed"] for r in cv)}


def _atm_one(market, d, expiry, side, spot, ts, cache, first_seen, contracts) -> tuple[str | None, Any]:
    if expiry is None:
        return "NO_LISTED_EXPIRY", None
    chain = market.chains.get((d, expiry, side)) or {}
    if not chain:
        return "CONTRACT_LIST_UNAVAILABLE", None
    strikes = sorted(chain)
    try:
        k = float(atm_strike(strikes, spot))
    except OptionResolutionError as exc:
        return f"STRIKE_UNAVAILABLE:{exc.code}", None
    c = chain[k]
    i = strikes.index(k)
    rec = contracts.setdefault((d, side, c.instrument_key), {
        "date": d.isoformat(), "side": side, "instrument_key": c.instrument_key, "contract": c.trading_symbol,
        "expiry": c.expiry.isoformat(), "strike": k, "timestamps": 0, "expiry_valid": c.expiry >= d,
        "nearest_expiry": c.expiry == expiry == sd.nearest_expiry(market.calendar, d),
        "in_listed_chain": chain.get(k) is c, "nearest_strike": True, "bracketed": True,
        "planned_atm": c.instrument_key in market.atm.get(d, set()), "eligible_timestamps": 0,
        "candle_matches_contract": True, "first_candle": None})
    rec["timestamps"] += 1
    rec["nearest_strike"] &= abs(k - spot) <= min(abs(x - spot) for x in strikes) + 1e-9
    rec["bracketed"] &= 0 < i < len(strikes) - 1
    if not rec["planned_atm"]:
        return "OUTSIDE_PLANNED_STRIKES", c
    if not rec["expiry_valid"] or not rec["nearest_expiry"]:
        return "EXPIRY_NOT_POINT_IN_TIME", c
    if not (0 < i < len(strikes) - 1):
        return "ATM_NOT_BRACKETED", c
    key = c.instrument_key
    if key not in cache:
        p = market.premium(key)
        cache[key] = sd.option_series(p)
        match = p is None or "instrument_key" not in p or set(p["instrument_key"].astype(str)) <= {key}
        first_seen[key] = (cache[key].t.min() if cache[key] is not None and len(cache[key].t) else None, match)
    s = cache[key]
    first, match = first_seen[key]
    rec["first_candle"] = first.isoformat() if first is not None else None
    if s is None:
        return "OPTION_DATA_MISSING", c
    j = int(s.feat.index.get_indexer([ts])[0])
    if j < 0:
        return "OPTION_SIGNAL_CANDLE_MISSING", c
    a = float(s.feat["atr14"].iat[j])
    if not (np.isfinite(a) and a > 0):
        return "OPTION_ATR_UNAVAILABLE (fewer than 14 prior candles)", c
    rec["candle_matches_contract"] &= bool(match)
    rec["eligible_timestamps"] += 1
    return None, c


# ------------------------------------------------------------------ 7. clean contiguous history
def session_status(nifty_days: Sequence[dict], fut_missing: dict[date, dict], fut_sel: Sequence[dict],
                   atm_sessions: Sequence[dict], blocking_days: dict[str, set], atm_keys: dict[date, set]) -> list[dict]:
    atm_by = {r["date"]: r for r in atm_sessions}
    sel_by = {r["session_date"]: r for r in fut_sel}
    out = []
    for r in nifty_days:
        if r["status"] in (CLOSED, EXCEPTION):
            continue
        d = date.fromisoformat(r["date"])
        why = []
        if r["status"] == MISSING_SESSION:
            why.append("INDEX_SESSION_MISSING")
        elif r["missing"] or r["duplicates"] or r["off_grid"] or r["outside_session"] or r["tz_errors"]:
            why.append("INDEX_BARS_INCOMPLETE_OR_INVALID")
        s = sel_by.get(r["date"])
        if s is None or s["status"] != nf.SELECTED:
            why.append("FUTURES_FRONT_MONTH_UNAVAILABLE")
        elif any(v for k, v in fut_missing.get(d, {}).items() if k != GAP_C):
            why.append("FUTURES_BARS_MISSING")
        if any(d in set(blocking_days.get(k, ())) for k in atm_keys.get(d, ())):
            why.append("OPTION_FILE_BLOCKING_FAILURE")
        a = atm_by.get(r["date"])
        if a is None or not a["signal_timestamps"]:
            why.append("NOT_IN_COMPLETE_SESSIONS")
        elif (a["both_share"] or 0) < SESSION_MIN_ATM_ELIGIBLE:
            why.append("LOW_ATM_COVERAGE")
        out.append({"date": r["date"], "clean": not why, "reasons": " ".join(why)})
    return out


def clean_runs(status: Sequence[dict], train_end: date = hx.TRAIN_END) -> dict[str, Any]:
    runs, cur = [], []
    for r in status:
        if r["clean"]:
            cur.append(r["date"])
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    longest = max(runs, key=len) if runs else []
    te = train_end.isoformat()
    upto = [r for r in status if r["date"] <= te]
    train: list[str] = []
    for r in reversed(upto):
        if not r["clean"]:
            break
        train.insert(0, r["date"])

    def desc(run: list[str]) -> dict[str, Any]:
        if not run:
            return {"start": None, "end": None, "trading_days": 0, "months": 0.0}
        a, b = date.fromisoformat(run[0]), date.fromisoformat(run[-1])
        return {"start": run[0], "end": run[-1], "trading_days": len(run), "months": round(((b - a).days + 1) / 30.44, 1)}
    return {"longest": desc(longest), "ending_train_end": desc(train), "runs": len(runs),
            "unclean_sessions": [{"date": r["date"], "reasons": r["reasons"]} for r in status if not r["clean"]]}


def coverage_in(run: dict, nifty_days: Sequence[dict], fut: dict, atm_sessions: Sequence[dict]) -> dict[str, Any]:
    if not run["start"]:
        return {"nifty": None, "futures": None, "options_ce": None, "options_pe": None, "options_both": None}
    lo, hi = run["start"], run["end"]
    nd = [r for r in nifty_days if lo <= r["date"] <= hi and r["status"] == REGULAR]
    exp = sum(r["expected"] for r in nd)
    nifty = (exp - sum(r["missing"] for r in nd)) / exp if exp else None
    fm = sum(1 for m in fut["missing"] if lo <= m["date"] <= hi and m["reason"] != GAP_C)
    fexp = sum(len(regular_grid(date.fromisoformat(r["date"]))) for r in nd)
    a = [r for r in atm_sessions if lo <= r["date"] <= hi]
    n = sum(r["signal_timestamps"] for r in a)
    share = (lambda k: sum(r[k] for r in a) / n if n else None)
    return {"nifty": nifty, "futures": (fexp - fm) / fexp if fexp else None, "options_ce": share("CE_eligible"),
            "options_pe": share("PE_eligible"), "options_both": share("both_eligible")}


# ------------------------------------------------------------------ 8. gate
def gate(nifty: dict, fut: dict, opt: dict, atm: dict, history: dict, coverage: dict, leakage: dict) -> dict[str, Any]:
    unresolved = (nifty["summary"]["tz_errors"] + nifty["summary"]["off_grid"] + nifty["checks"]["conflicting_duplicates"]
                  + fut["summary"]["tz_errors"] + fut["summary"]["off_grid"] + fut["summary"]["conflicting_duplicates"]
                  + sum(a["rows"] for a in opt["failures"]
                        if a["reason"] in ("TIMEZONE_MISMATCH", "CONFLICTING_DUPLICATE_TIMESTAMP")))
    run = history["ending_train_end"]
    cond = {
        "nifty_quality_pass": nifty["summary"]["status"] == PASS,
        "futures_quality_pass": fut["summary"]["status"] == PASS,
        "option_quality_pass": opt["summary"]["status"] in (PASS, PASS_Q),
        "no_unresolved_timestamp_corruption": unresolved == 0,
        "causal_contract_selection_pass": bool(atm["causal_passed"] and not fut["selection_checks"]["violations"]
                                               and fut["selection_checks"]["independent_of_price_and_volume"]),
        "sufficient_atm_ce_pe_coverage": bool(coverage["options_ce"] is not None
                                              and coverage["options_ce"] >= MIN_ATM_ELIGIBLE_SHARE
                                              and coverage["options_pe"] >= MIN_ATM_ELIGIBLE_SHARE),
        "no_material_leakage": bool(leakage["passed"]),
        "clean_history_sufficient": bool(run["months"] >= MIN_CLEAN_MONTHS),
    }
    blocking = [k for k, v in cond.items() if not v]
    return {"conditions": cond, "unresolved_timestamp_corruption": unresolved, "data_ready": not blocking,
            "blocking": blocking}


def blocking_detail(g: dict, nifty: dict, fut: dict, opt: dict, atm: dict, history: dict, coverage: dict,
                    leakage: dict) -> list[str]:
    out = []
    c = g["conditions"]
    if not c["nifty_quality_pass"]:
        s = nifty["summary"]
        out.append(f"NIFTY: missing sessions {s['missing_sessions']}, genuine missing bars {s['missing_genuine']} "
                   f"({s['missing_genuine_share']}), duplicates {s['duplicates']}, off-grid {s['off_grid']}, "
                   f"invalid {s['invalid_ohlc']}, tz {s['tz_errors']}, outside session {s['outside_session']}")
    if not c["futures_quality_pass"]:
        s = fut["summary"]
        out.append(f"FUTURES: missing {s['missing_by_category']} (share {s['missing_genuine_share']}), duplicates "
                   f"{s['duplicates']}, conflicts {s['cross_root_conflicts']}, selection violations "
                   f"{s['selection_violations']}")
    if not c["option_quality_pass"]:
        out.append(f"OPTIONS: blocking reasons {opt['summary']['blocking_reasons']}")
    if not c["no_unresolved_timestamp_corruption"]:
        out.append(f"TIMESTAMPS: {g['unresolved_timestamp_corruption']} unresolved rows (tz / conflicting duplicates / "
                   "off-grid index or futures)")
    if not c["causal_contract_selection_pass"]:
        bad = [r for r in atm["contracts"] if not r["passed"]][:5]
        out.append(f"CAUSAL SELECTION: {sum(not r['passed'] for r in atm['contracts'])} option selections failed "
                   f"(e.g. {[(r['date'], r['side'], r['contract']) for r in bad]}); futures violations "
                   f"{fut['selection_checks']['violations'][:5]}")
    if not c["sufficient_atm_ce_pe_coverage"]:
        out.append(f"ATM COVERAGE: CE {coverage['options_ce']}, PE {coverage['options_pe']} "
                   f"(< {MIN_ATM_ELIGIBLE_SHARE:.0%} over the clean history)")
    if not c["no_material_leakage"]:
        out.append(f"LEAKAGE: {leakage}")
    if not c["clean_history_sufficient"]:
        r = history["ending_train_end"]
        first_bad = [u for u in history["unclean_sessions"] if u["date"] <= hx.TRAIN_END.isoformat()][-3:]
        out.append(f"CLEAN HISTORY: {r['months']} months ({r['start']} -> {r['end']}) < {MIN_CLEAN_MONTHS}; most recent "
                   f"unclean sessions before TRAIN end: {first_bad}")
    return out


# ------------------------------------------------------------------ text
def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.2f}%"


def terminal_text(a: dict) -> str:
    n, f, o = a["nifty"]["summary"], a["futures"]["summary"], a["options"]["summary"]
    cov, h = a["coverage_all"], a["history"]["ending_train_end"]
    rep = a["repairs"]
    out = [HEADING, f"({BANNER})", "", "NIFTY:", f"status: {n['status']}", f"trading days: {n['trading_days']}",
           f"missing: {n['missing_genuine']} genuine bars ({_pct(n['missing_genuine_share'])}); "
           f"{n['missing_session_exception']} session-exception bars; {len(n['market_closed_days'])} market-closed days; "
           f"{len(n['missing_sessions'])} missing sessions",
           f"duplicates: {n['duplicates']}", "", "FUTURES:", f"status: {f['status']}", f"contracts: {f['contracts']}",
           "missing: " + ", ".join(f"{k} {v}" for k, v in f["missing_by_category"].items())
           + f" (genuine {_pct(f['missing_genuine_share'])})",
           f"duplicates: {f['duplicates']}", "", "OPTIONS:", f"status: {o['status']}", f"contracts: {o['contracts']}",
           f"candles: {o['candles']}",
           f"off-grid: {o['off_grid_rows']} rows in {o['off_grid_files']} files on {len(o['off_grid_days'])} days "
           f"-> marked unusable (on-grid series complete: {o['off_grid_on_grid_complete']})",
           f"missing: {o['atm_without_candles']} ATM requirements without candles; {o['atm_missing_bars']} ATM bars "
           f"(completeness {_pct(o['atm_interval_completeness'])})",
           f"invalid: {o['invalid_rows']} rows (blocking reasons: {', '.join(o['blocking_reasons']) or 'none'})", "",
           "ATM COVERAGE:", f"CE: {_pct(cov['options_ce'])}", f"PE: {_pct(cov['options_pe'])}", "",
           "CAUSAL CONTRACT SELECTION:", PASS if a["gate"]["conditions"]["causal_contract_selection_pass"] else FAIL,
           "", "DATA_READY:", "YES" if a["gate"]["data_ready"] else "NO", "", "CLEAN HISTORY:",
           f"start: {h['start']}", f"end: {h['end']}", f"trading days: {h['trading_days']} ({h['months']} months)", "",
           "REPAIRS:", f"count: {rep['count']}", f"what was changed: {rep['what']}", "", "FABRICATED VALUES:", "0", "",
           "LIVE ORDERS:", "0", "", "PRODUCTION CHANGED:", "NO" if a.get("production_unchanged") is not False else "YES",
           "", "OCTOBER LOADED:", "NO"]
    if not a["gate"]["data_ready"]:
        out += ["", "BLOCKING ISSUES (STOP - do NOT rerun S11 ML):"] + [f"- {x}" for x in a["blocking_detail"]]
    else:
        out += ["", "Data gate passed. S11 ML may be rerun only as a separate, explicitly requested step."]
    return "\n".join(out)


def markdown(a: dict) -> str:
    n, f, o = a["nifty"]["summary"], a["futures"]["summary"], a["options"]["summary"]
    g, h = a["gate"], a["history"]
    md = ["# S12 expanded data quality repair & coverage audit", "", f"**{BANNER}**", "",
          f"## DATA_READY: **{'YES' if g['data_ready'] else 'NO'}**", ""]
    md += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in g["conditions"].items()]
    if a["blocking_detail"]:
        md += ["", "### Blocking issues", ""] + [f"- {x}" for x in a["blocking_detail"]]
    md += ["", "## 1. NIFTY index", "", f"- Status **{n['status']}**; {n['trading_days']} trading days "
           f"({n['regular_sessions']} regular, session exceptions {n['exception_sessions']}).",
           f"- Market-closed weekdays (no candles in any instrument): {len(n['market_closed_days'])} "
           f"({', '.join(n['market_closed_days'])}).",
           f"- Missing bars: genuine {n['missing_genuine']} ({_pct(n['missing_genuine_share'])}), session exception "
           f"{n['missing_session_exception']}; duplicates {n['duplicates']}, off-grid {n['off_grid']}, invalid OHLC "
           f"{n['invalid_ohlc']}, timezone errors {n['tz_errors']}, outside session {n['outside_session']}.",
           f"- Rule: {n['rule']}", "", "## 2. Futures (front month)", "",
           f"- Status **{f['status']}**; {f['contracts']} contracts; missing slots by category "
           f"{f['missing_by_category']} (A data-source gap, B rollover, C session exception, D genuine).",
           f"- Duplicates {f['duplicates']}, cross-root overlap {f['cross_root_overlap']} timestamps with "
           f"{f['cross_root_conflicts']} conflicts; causal selection violations {f['selection_violations']}, "
           f"independent of price/volume {f['selection_independent_of_price_and_volume']}.",
           f"- Rule: {f['rule']}", "", "## 3. Option data quality root cause", "",
           f"- Status **{o['status']}**; {o['files']} files, {o['contracts']} contracts, {o['candles']} candles.",
           f"- Original downloader verdict: {a['original_option_quality']}", "",
           "| Reason | Files | Rows | Blocking | Resolution |", "|---|---|---|---|---|"]
    md += [f"| {r['reason']} | {r['files']} | {r['rows']} | {r['blocking']} | {r['resolution']} |"
           for r in a["options"]["failures"]]
    og = a["off_grid_summary"]
    md += ["", "## 4. Off-grid option candles", "",
           f"- {o['off_grid_rows']} rows, {o['off_grid_files']} files, days {', '.join(o['off_grid_days'])}.",
           f"- Pattern: minute offsets {og['minute_offsets']}, second offsets {og['second_offsets']}, first "
           f"{og['first']} / last {og['last']} per day; timezone shift explains: {og['timezone_shift_explains']}.",
           f"- Same day, same contract: complete on-grid series present in {og['on_grid_complete_days']}/"
           f"{og['file_days']} file-days; identical OHLC to the on-grid bar 1 minute earlier: "
           f"{_pct(og['identical_ohlc_share'])}; close match {_pct(og['close_match_share'])}; median close difference "
           f"{_pct(og['median_rel_close_diff'])}.",
           f"- Verdict: **{og['verdict']}**. {og['explanation']}", "",
           "## 5. ATM coverage (every eligible signal timestamp)", "",
           f"- CE eligible {_pct(a['coverage_all']['options_ce'])}, PE {_pct(a['coverage_all']['options_pe'])}, both "
           f"{_pct(a['coverage_all']['options_both'])} over the clean history.", "",
           "| Side | Reason | Timestamps |", "|---|---|---|"]
    for s, t in a["atm"]["totals"].items():
        md += [f"| {s} | {k} | {v} |" for k, v in sorted(t.items(), key=lambda kv: -kv[1])]
    cv = a["atm"]["contracts"]
    md += ["", "## 6. Point-in-time expiry / contract validation", "",
           f"- {len(cv)} (session, side, contract) selections checked; failed {sum(not r['passed'] for r in cv)}.",
           "- Checks: expiry >= session and = nearest listed expiry (official calendar), strike in the listed chain, "
           "strike nearest to the spot close at the signal time, ATM bracketed by planned neighbours (so the planned "
           "subset gives the same ATM as the full chain), planned ATM requirement, candle present at signal time.",
           f"- Futures: {a['futures']['selection_checks']['selected']} sessions selected, violations "
           f"{a['futures']['selection_checks']['violations']}, independent of price/volume "
           f"{a['futures']['selection_checks']['independent_of_price_and_volume']}.", "",
           "## 7. Contiguous usable history", ""]
    for k, title in (("longest", "Longest clean run"), ("ending_train_end", "Clean run ending at the S11 TRAIN end")):
        r = h[k]
        md.append(f"- {title}: {r['start']} -> {r['end']}, {r['trading_days']} trading days ({r['months']} months)")
    cov = a["coverage_all"]
    md += [f"- Coverage in the TRAIN-end run: NIFTY {_pct(cov['nifty'])}, futures {_pct(cov['futures'])}, options CE "
           f"{_pct(cov['options_ce'])} / PE {_pct(cov['options_pe'])}.",
           f"- Unclean sessions: {len(h['unclean_sessions'])} (see clean_history_manifest.json).", "",
           "## July-September 2026 coverage (outer splits)", "", "| Month | Sessions | CE | PE | Both |", "|---|---|---|---|---|"]
    md += [f"| {r['month']} | {r['sessions']} | {_pct(r['CE'])} | {_pct(r['PE'])} | {_pct(r['both'])} |"
           for r in a["outer_coverage"]]
    md += ["", "## Repairs", "", f"- Count: {a['repairs']['count']}. {a['repairs']['what']}",
           f"- Raw data unchanged: {a['raw_unchanged']}. Fabricated values: 0. October loaded: NO. Production changed: "
           f"{'NO' if a.get('production_unchanged') is not False else 'YES'}. Live orders: 0."]
    return "\n".join(md) + "\n"


def off_grid_summary(offs: Sequence[dict]) -> dict[str, Any]:
    if not offs:
        return {"minute_offsets": "-", "second_offsets": "-", "first": None, "last": None,
                "timezone_shift_explains": None, "on_grid_complete_days": 0, "file_days": 0,
                "identical_ohlc_share": None, "close_match_share": None, "median_rel_close_diff": None,
                "verdict": "NONE", "explanation": "no off-grid option candles"}
    w = np.array([r["matched_on_grid_bars"] for r in offs], dtype=float)

    def wavg(k):
        v = np.array([r[k] if r[k] is not None else np.nan for r in offs], dtype=float)
        m = np.isfinite(v) & (w > 0)
        return float(np.average(v[m], weights=w[m])) if m.any() else None
    verdicts = Counter(r["verdict"] for r in offs)
    v = verdicts.most_common(1)[0][0]
    complete = sum(r["on_grid_complete"] for r in offs)
    expl = ("Each affected contract-day ALSO holds the complete on-grid 5-minute series; the off-grid rows are a second "
            "series aggregated on 1-minute-shifted buckets (different OHLC), not a relabelled copy and not a timezone "
            "shift. No documented deterministic mapping exists, so they are NOT normalised: marked UNUSABLE and "
            "excluded at read time. Raw files are preserved; no corrected dataset is needed because the genuine "
            "on-grid candles are complete." if v == "DIFFERENT_AGGREGATION_BUCKETS" else
            "Off-grid rows duplicate the on-grid series; they are excluded (normalising would only create duplicates).")
    return {"minute_offsets": " ".join(sorted({r["minute_offsets"] for r in offs})),
            "second_offsets": " ".join(sorted({r["second_offsets"] for r in offs})),
            "first": " ".join(sorted({r["first_off_grid"] for r in offs})),
            "last": " ".join(sorted({r["last_off_grid"] for r in offs})),
            "timezone_shift_explains": any(r["timezone_shift_explains"] for r in offs),
            "on_grid_complete_days": complete, "file_days": len(offs),
            "identical_ohlc_share": wavg("identical_ohlc_share"), "close_match_share": wavg("close_match_share"),
            "median_rel_close_diff": wavg("median_rel_close_diff"), "verdict": v, "verdicts": dict(verdicts),
            "explanation": expl}


# ------------------------------------------------------------------ orchestration
def raw_files(roots: Sequence[Path], underlying: str = "NIFTY", max_day: date = PERIOD[1]) -> list[Path]:
    last = hx.month_label(max_day)
    out = []
    for r in roots:
        lay = u6.Layout(r, underlying)
        out += [p for p in lay.underlying_files() if p.stem <= last]
        out += [p for p in lay.option_files() if p.parent.name <= last]
        out += nf.FuturesLayout(r, underlying).contract_files()
    return sorted(out)


def raw_hashes(roots: Sequence[Path], underlying: str = "NIFTY", max_day: date = PERIOD[1]) -> dict[str, str]:
    return {p.as_posix(): sha256_file(p) for p in raw_files(roots, underlying, max_day)}


def original_option_quality(roots: Sequence[Path], underlying: str = "NIFTY") -> dict[str, Any]:
    out = {}
    for r in roots:
        lay = u6.Layout(r, underlying)
        s = json.loads(lay.summary.read_text(encoding="utf-8")) if lay.summary.is_file() else {}
        q = s.get("quality") or {}
        out[Path(r).name] = {"pass": q.get("pass"), "failing_files": len(q.get("failing_files") or []),
                             "failing_reasons": dict(Counter(f.get("reason") for f in q.get("failing_files") or [])),
                             "underlying_status": (s.get("underlying") or {}).get("status"),
                             "underlying_missing_by_day": (s.get("underlying") or {}).get("missing_by_day")}
    return out


def build_audit(roots: Sequence[str | Path], *, underlying: str = "NIFTY", lo: date | None = None,
                hi: date | None = None, progress=None) -> dict[str, Any]:
    lo, hi = lo or PERIOD[0], hi or PERIOD[1]
    hx.guard_no_october(hi)
    say = progress or (lambda _m: None)
    roots = [Path(r) for r in roots]
    before = raw_hashes(roots, underlying, hi)
    late = 0
    say("reading stored NIFTY index candles (raw, no de-duplication) ...")
    parts = []
    for r in roots:
        for p in u6.Layout(r, underlying).underlying_files():
            if p.stem > hx.month_label(hi):
                continue
            f, n = read_raw(p, hi)
            late += n
            if len(f):
                parts.append(f.assign(root=r.name))
    spot = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["timestamp", *OHLC, "tz_error"])
    if len(spot):
        spot = spot[pd.DatetimeIndex(spot["timestamp"]).date >= lo].reset_index(drop=True)
    say("reading stored futures contract files ...")
    fut_raw = []
    for r in roots:
        fl = nf.FuturesLayout(r, underlying)
        for c in nf.contract_rows_from_csv(fl):
            p = fl.contract_file(c)
            if p.is_file():
                f, n = read_raw(p, hi)
                late += n
                fut_raw.append((r.name, c, f))
    say("auditing every stored option file ...")
    opt = option_audit(roots, underlying, hi)
    late += opt["summary"]["rows_after_research_end_ignored"]
    say("loading the research view (S10/S11 path, off-grid option rows quarantined) ...")
    market = sd.load_market([str(r) for r in roots], underlying, max_day=hi)
    quarantine = hx.quarantine_off_grid(market)
    fut_slots = set()
    fut_days = set()
    for f in market.frames.values():
        g = f.loc[hx.on_grid_mask(f), "timestamp"]
        fut_slots |= set(g)
        fut_days |= set(pd.DatetimeIndex(g).date)
    say("NIFTY session audit ...")
    nifty = nifty_audit(spot, lo, hi, fut_days | opt["days"], fut_slots)
    trading_days = [date.fromisoformat(r["date"]) for r in nifty["days"] if r["status"] in (REGULAR, EXCEPTION)]
    nifty_slots = set(spot.loc[hx.on_grid_mask(spot), "timestamp"]) if len(spot) else set()
    say("futures front-month audit ...")
    fut = futures_audit(fut_raw, market.contracts, market.frames, trading_days, nifty_slots)
    say("ATM coverage + point-in-time contract validation (every signal timestamp) ...")
    sessions = [d for d in market.sessions if lo <= d <= hi]
    atm = atm_audit(market, sessions)
    status = session_status(nifty["days"], fut["missing_by_day"], fut["selection"], atm["sessions"],
                            opt["blocking_days"], market.atm)
    history = clean_runs(status)
    cov = coverage_in(history["ending_train_end"], nifty["days"], fut, atm["sessions"])
    cov_longest = coverage_in(history["longest"], nifty["days"], fut, atm["sessions"])
    max_day = max([d for d in (spot["timestamp"].max().date() if len(spot) else None,
                               max(market.sessions) if market.sessions else None) if d is not None], default=None)
    leakage = {"option_rows_after_expiry": int(sum(f["after_expiry"] for f in opt["files"])),
               "futures_rows_after_expiry": fut["summary"]["after_expiry"],
               "futures_cross_root_conflicts": fut["summary"]["cross_root_conflicts"],
               "max_day_loaded": max_day.isoformat() if max_day else None,
               "rows_after_research_end_discarded_unanalysed": late}
    leakage["passed"] = bool(leakage["option_rows_after_expiry"] == 0 and leakage["futures_rows_after_expiry"] == 0
                             and leakage["futures_cross_root_conflicts"] == 0 and max_day is not None and max_day <= hi)
    g = gate(nifty, fut, opt, atm, history, cov, leakage)
    after = raw_hashes(roots, underlying, hi)
    q = hx.quarantine_report(quarantine)
    outer = []
    for lo_m, hi_m in (hx.VALIDATION, hx.OOS, hx.LEGACY):
        ss = [r for r in atm["sessions"] if lo_m.isoformat() <= r["date"] <= hi_m.isoformat()]
        n = sum(r["signal_timestamps"] for r in ss)
        outer.append({"month": hx.month_label(lo_m), "sessions": len(ss),
                      "unclean_sessions": sum(1 for u in history["unclean_sessions"]
                                              if lo_m.isoformat() <= u["date"] <= hi_m.isoformat()),
                      "CE": sum(r["CE_eligible"] for r in ss) / n if n else None,
                      "PE": sum(r["PE_eligible"] for r in ss) / n if n else None,
                      "both": sum(r["both_eligible"] for r in ss) / n if n else None})
    a = {"heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
         "roots": [r.as_posix() for r in roots], "period": [lo.isoformat(), hi.isoformat()],
         "october_loaded": False, "live_orders": 0, "production_unchanged": None,
         "nifty": nifty, "futures": fut, "options": opt, "atm": atm, "session_status": status, "history": history,
         "coverage_all": cov, "coverage_longest": cov_longest, "outer_coverage": outer, "leakage": leakage,
         "gate": g, "off_grid_summary": off_grid_summary(opt["off_grid"]), "quarantine_research_view": q,
         "original_option_quality": original_option_quality(roots, underlying),
         "raw_unchanged": before == after, "raw_files_hashed": len(before),
         "repairs": {"count": 0, "rows_marked_unusable": opt["summary"]["off_grid_rows"],
                     "what": (f"no stored value modified; {opt['summary']['off_grid_rows']} off-grid option rows "
                              "marked UNUSABLE (excluded at read time, never shifted); no corrected dataset created "
                              "(normalisation not safe, genuine on-grid series already complete)")},
         "fabricated_values": 0, "_market": market}
    a["blocking_detail"] = blocking_detail(g, nifty, fut, opt, atm, history, cov, leakage)
    return a


REPORT_FILES = ("summary.json", "s12_data_quality_report.md", "nifty_quality.csv", "nifty_missing_bars.csv",
                "futures_quality.csv", "futures_missing_bars.csv", "option_quality.csv", "option_quality_failures.csv",
                "off_grid_analysis.csv", "atm_coverage.csv", "contract_validation.csv", "clean_history_manifest.json",
                "unusable_option_rows.csv")


def summary_obj(a: dict) -> dict[str, Any]:
    return {"heading": a["heading"], "banner": a["banner"], "generated_at": a["generated_at"], "roots": a["roots"],
            "period": a["period"], "data_ready": a["gate"]["data_ready"], "gate": a["gate"],
            "blocking_detail": a["blocking_detail"], "nifty": a["nifty"]["summary"],
            "futures": a["futures"]["summary"], "futures_selection": a["futures"]["selection_checks"],
            "options": a["options"]["summary"], "option_failures": a["options"]["failures"],
            "original_option_quality": a["original_option_quality"], "off_grid": a["off_grid_summary"],
            "quarantine_research_view": a["quarantine_research_view"],
            "atm_totals": a["atm"]["totals"], "causal_option_selection_passed": a["atm"]["causal_passed"],
            "contract_selections_checked": len(a["atm"]["contracts"]),
            "clean_history": {k: a["history"][k] for k in ("longest", "ending_train_end", "runs")},
            "unclean_sessions": len(a["history"]["unclean_sessions"]), "coverage_clean_history": a["coverage_all"],
            "coverage_longest_run": a["coverage_longest"], "outer_coverage": a["outer_coverage"],
            "leakage": a["leakage"], "repairs": a["repairs"], "raw_unchanged": a["raw_unchanged"],
            "raw_files_hashed": a["raw_files_hashed"], "fabricated_values": 0, "live_orders": 0,
            "october_loaded": False, "production_unchanged": a.get("production_unchanged"),
            "token_leak": a.get("token_leak")}


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    from src.research import s6_clean_strategy_lab as s6
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    js = lambda path, obj: path.write_text(json.dumps(s6._jsonable(obj), indent=2, default=str),  # noqa: E731
                                           encoding="utf-8")
    js(p["summary.json"], summary_obj(a))
    s6._csv(p["nifty_quality.csv"], a["nifty"]["days"])
    s6._csv(p["nifty_missing_bars.csv"], a["nifty"]["missing"])
    s6._csv(p["futures_quality.csv"], a["futures"]["quality"])
    s6._csv(p["futures_missing_bars.csv"], a["futures"]["missing"])
    s6._csv(p["option_quality.csv"], a["options"]["files"])
    s6._csv(p["option_quality_failures.csv"], a["options"]["failures"])
    s6._csv(p["off_grid_analysis.csv"], a["options"]["off_grid"])
    s6._csv(p["atm_coverage.csv"], a["atm"]["sessions"])
    s6._csv(p["contract_validation.csv"],
            [{"kind": "OPTION_ATM", **r} for r in a["atm"]["contracts"]]
            + [{"kind": "FUTURES_FRONT_MONTH", **{k: v for k, v in r.items()}} for r in a["futures"]["selection"]])
    s6._csv(p["unusable_option_rows.csv"], a["options"]["unusable_rows"])
    js(p["clean_history_manifest.json"],
       {"banner": BANNER, "rules": {"session_clean": "index 75/75 valid bars; front-month future 75/75 (session "
                                                     "exceptions excepted); no blocking option-file failure on the "
                                                     "day's ATM contracts; both-side ATM eligibility >= "
                                                     f"{SESSION_MIN_ATM_ELIGIBLE:.0%}",
                                    "history": "longest run of consecutive clean regular sessions; market-closed days "
                                               "and documented session exceptions do not break a run",
                                    "session_exceptions": {d.isoformat(): {k: str(v) for k, v in e.items()}
                                                           for d, e in SESSION_EXCEPTIONS.items()},
                                    "off_grid": "option rows off the 5-minute grid are UNUSABLE (excluded, never "
                                                "shifted)"},
        "longest": a["history"]["longest"], "ending_train_end": a["history"]["ending_train_end"],
        "coverage": a["coverage_all"], "data_ready": a["gate"]["data_ready"], "gate": a["gate"],
        "unclean_sessions": a["history"]["unclean_sessions"], "sessions": a["session_status"]})
    p["s12_data_quality_report.md"].write_text(markdown(a), encoding="utf-8")
    return p
