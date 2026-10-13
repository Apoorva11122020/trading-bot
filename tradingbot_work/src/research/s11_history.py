"""S11 - expanded-history availability rules (PAPER | RESEARCH ONLY | NO ORDERS | no network, no sklearn).

Pre-declared rules (fixed before any expanded data was downloaded):

* Expiry calendar: a session is point-in-time resolvable only if its nearest listed expiry is genuinely the nearest
  one. Sessions on/before the earliest expired expiry Upstox returns, and sessions inside a gap of more than
  MAX_EXPIRY_GAP_DAYS between consecutive listed expiries, are NOT resolvable (a later expiry would be picked with
  hindsight) - they are recorded as missing, never filled.
* History: per calendar month of the requested range, a month is CLEAN when it has >= MIN_MONTH_SESSIONS complete
  sessions and >= MIN_MONTH_ELIGIBLE_SHARE of candidate prediction timestamps have genuine futures + ATM CE/PE data.
  The usable history is the longest run of consecutive CLEAN months ending with the TRAIN end month; fewer than
  MIN_HISTORY_MONTHS -> INSUFFICIENT_HISTORICAL_DATA (no strategy promotion).
* October 2026 is never requested or loaded.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REQUESTED_START = date(2025, 1, 1)
EXPANDED_END = date(2026, 3, 31)        # 2026-04..09 is reused from data/historical/upstox_6m
TRAIN_END = date(2026, 6, 30)
VALIDATION = (date(2026, 7, 1), date(2026, 7, 31))
OOS = (date(2026, 8, 1), date(2026, 8, 31))
LEGACY = (date(2026, 9, 1), date(2026, 9, 30))
RESEARCH_END = LEGACY[1]
OCTOBER_START = date(2026, 10, 1)
LEGACY_LABEL = "LEGACY_HOLDOUT_CONTAMINATED"

MAX_EXPIRY_GAP_DAYS = 10
MIN_HISTORY_MONTHS = 12
MIN_MONTH_SESSIONS = 15
MIN_MONTH_ELIGIBLE_SHARE = 0.80
CLEAN, NO_DATA, FEW_SESSIONS, LOW_ELIGIBILITY, OUTSIDE_RUN = (
    "CLEAN", "NO_DATA", "INSUFFICIENT_SESSIONS", "LOW_DATA_ELIGIBILITY", "CLEAN_BUT_NOT_CONTIGUOUS")
HISTORY_OK, HISTORY_INSUFFICIENT = "OK", "INSUFFICIENT_HISTORICAL_DATA"


def guard_no_october(*days: date | None) -> None:
    """Hard stop if any requested / loaded day reaches the protected October holdout."""
    for d in days:
        if d is not None and d >= OCTOBER_START:
            raise SystemExit(f"Refusing {d}: October 2026 is the protected FRESH_UNSEEN_HOLDOUT (never loaded in S11)")


def on_grid_mask(frame: pd.DataFrame) -> np.ndarray:
    t = pd.DatetimeIndex(frame["timestamp"])
    return np.asarray((t.minute % 5 == 0) & (t.second == 0) & (t.nanosecond == 0) & (t.microsecond == 0))


def quarantine_off_grid(market) -> dict[str, Any]:
    """Option candles whose timestamps are not on the 5-minute session grid are DROPPED at read time (never shifted
    or filled); affected sessions simply lose those candles and fall out as ineligible. Stats are reported."""
    base_fn, cache = market.premium_fn, {}
    stats: dict[str, Any] = {"contracts": set(), "rows": 0, "days": set()}

    def premium_fn(key: str) -> pd.DataFrame | None:
        if key not in cache:
            p = base_fn(key)
            if p is not None and not p.empty:
                ok = on_grid_mask(p)
                if not ok.all():
                    stats["contracts"].add(key)
                    stats["rows"] += int((~ok).sum())
                    stats["days"] |= set(pd.DatetimeIndex(p.loc[~ok, "timestamp"]).date)
                    p = p.loc[ok].reset_index(drop=True)
            cache[key] = p
        return cache[key]

    market.premium_fn = premium_fn
    return stats


def quarantine_report(stats: dict[str, Any]) -> dict[str, Any]:
    return {"rule": "option candles off the 5-minute grid are dropped (never shifted / filled)",
            "contracts_affected": len(stats["contracts"]), "rows_dropped": stats["rows"],
            "days": sorted(d.isoformat() for d in stats["days"])}


def month_label(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def months(start: date, end: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def expiry_coverage(expiries: Iterable[date], start: date, end: date) -> dict[str, Any]:
    """Which part of [start, end] has a complete, point-in-time expiry calendar (pure function of the listing)."""
    ex = sorted(set(expiries))
    base: dict[str, Any] = {"requested": [start.isoformat(), end.isoformat()], "expiries_listed": len(ex),
                            "earliest_expiry": ex[0].isoformat() if ex else None,
                            "latest_expiry": ex[-1].isoformat() if ex else None, "gaps": [], "missing_periods": []}
    if not ex:
        return {**base, "resolvable_start": None,
                "missing_periods": [{"from": start.isoformat(), "to": end.isoformat(),
                                     "reason": "NO_EXPIRED_EXPIRIES_LISTED"}]}
    first, missing = start, []
    if ex[0] >= start:
        first = _next_weekday(ex[0])
        missing.append({"from": start.isoformat(), "to": ex[0].isoformat(),
                        "reason": "BEFORE_EARLIEST_LISTED_EXPIRY (nearest expiry not provable)"})
    gaps = []
    for a, b in zip(ex, ex[1:]):
        if (b - a).days > MAX_EXPIRY_GAP_DAYS and b >= start and a <= end:
            gaps.append({"after": a.isoformat(), "before": b.isoformat(), "days": (b - a).days})
            missing.append({"from": _next_weekday(a).isoformat(), "to": b.isoformat(),
                            "reason": f"EXPIRY_CALENDAR_GAP > {MAX_EXPIRY_GAP_DAYS} days"})
            first = max(first, _next_weekday(b))
    return {**base, "gaps": gaps, "missing_periods": missing,
            "resolvable_start": first.isoformat() if first <= end else None}


def estimated_months(start: date | None, end: date = TRAIN_END) -> float:
    return round(((end - start).days + 1) / 30.44, 1) if start else 0.0


def month_rows(sessions: Sequence[date], candidates: dict[date, int], eligible: dict[date, int],
               start: date = REQUESTED_START, end: date = TRAIN_END) -> list[dict]:
    """One row per requested month: complete sessions, candidate / eligible prediction timestamps, status."""
    by_month: dict[str, list[date]] = {}
    for d in sessions:
        if start <= d <= end:
            by_month.setdefault(month_label(d), []).append(d)
    out = []
    for m in months(start, end):
        ds = sorted(by_month.get(m, []))
        cand = sum(candidates.get(d, 0) for d in ds)
        elig = sum(eligible.get(d, 0) for d in ds)
        share = elig / cand if cand else None
        if not ds:
            st = NO_DATA
        elif len(ds) < MIN_MONTH_SESSIONS:
            st = FEW_SESSIONS
        elif share is None or share < MIN_MONTH_ELIGIBLE_SHARE:
            st = LOW_ELIGIBILITY
        else:
            st = CLEAN
        out.append({"month": m, "sessions": len(ds), "first_session": ds[0].isoformat() if ds else None,
                    "last_session": ds[-1].isoformat() if ds else None, "candidate_timestamps": cand,
                    "eligible_timestamps": elig, "eligible_share": share, "status": st})
    return out


def history_range(rows: Sequence[dict], requested_start: date = REQUESTED_START,
                  end: date = TRAIN_END) -> dict[str, Any]:
    """Longest run of consecutive CLEAN months ending with the TRAIN end month."""
    run: list[dict] = []
    for r in reversed(list(rows)):
        if r["status"] != CLEAN:
            break
        run.insert(0, r)
    in_run = {r["month"] for r in run}
    missing = [{"month": r["month"], "status": r["status"] if r["status"] != CLEAN else OUTSIDE_RUN,
                "sessions": r["sessions"], "eligible_share": r["eligible_share"]}
               for r in rows if r["month"] not in in_run]
    n = len(run)
    return {"requested_start": requested_start.isoformat(), "requested_end": end.isoformat(),
            "actual_start": run[0]["first_session"] if run else None,
            "actual_end": run[-1]["last_session"] if run else None,
            "months": n, "trading_days": sum(r["sessions"] for r in run), "missing_periods": missing,
            "min_months": MIN_HISTORY_MONTHS,
            "status": HISTORY_OK if n >= MIN_HISTORY_MONTHS else HISTORY_INSUFFICIENT,
            "rule": f"consecutive CLEAN months ending {month_label(end)}; CLEAN = >= {MIN_MONTH_SESSIONS} complete "
                    f"sessions and >= {MIN_MONTH_ELIGIBLE_SHARE:.0%} of candidate timestamps with genuine data"}
