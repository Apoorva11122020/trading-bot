"""NIFTY futures 5-minute OHLCV for the 6-month research window (PAPER | DATA ACQUISITION ONLY | NO ORDERS).

Sources (read-only GET requests through the existing ``UpstoxHistoricalClient``; token from the environment only):
  live FUT contracts     official Upstox instrument master (``ContractCatalog.active_contracts``)
  expired FUT contracts  GET /v2/expired-instruments/expiries + /v2/expired-instruments/future/contract
  candles                V3 historical candles (live) / expired-instrument historical candles, monthly chunks

Every contract is stored separately (``futures/NIFTY_FUT_<EXPIRY>.parquet``); nothing is stitched, price-adjusted,
interpolated, forward-filled or synthesised. Invalid raw rows are never repaired: they are excluded from the stored
file and written to ``futures_invalid_rows.csv`` with the reason.

Causal front-month rule (per session): among discovered NIFTY FUT contracts (instrument metadata: expiry is public
information), take the nearest expiry that is not expired on the session date. If that contract has not started
trading or has no candle for the session / timestamp, the session / timestamp is DATA_UNAVAILABLE - a later contract
is never substituted. Price or volume never influences the selection.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import upstox_6m as u6
from .contracts import ACTIVE, EXPIRED, HistoricalContract, contract_from_row
from .downloader import FATAL_CODES
from .models import HistoricalDataError, candles_to_frame
from .validation import expected_slots

IST = u6.IST
BANNER = "PAPER | DATA ACQUISITION ONLY | NO ORDERS"
INTERVAL = 5
SESSION_START, SESSION_END, S9_END = time(9, 15), time(15, 30), time(14, 30)
SLOTS, S9_SLOTS = 75, 63
CONTRACT_LOOKBACK_DAYS = 100          # a NIFTY monthly future trades ~3 months before expiry
CONTRACTS_AFTER_END = 2               # first two expiries after the window (front month + next for rollover)
SEGMENT = "NSE_FO"

# NSE Diwali Muhurat on 2025-10-21 ran 13:45–14:45 IST (12 five-minute candles).
# This is a valid special session, not a truncated 75-bar regular session.
SPECIAL_SESSIONS = {date(2025, 10, 21): (time(13, 45), time(14, 45))}


def session_slots(day: date) -> list[pd.Timestamp]:
    """Expected candle starts for a known exchange session, including special sessions."""
    special = SPECIAL_SESSIONS.get(day)
    if special is not None:
        return expected_slots(day, INTERVAL, start=special[0], end=special[1])
    return expected_slots(day, INTERVAL)


def session_slot_count(day: date) -> int:
    return len(session_slots(day))

# Pre-declared data-quality thresholds (fixed before any futures data was downloaded).
MAX_ZERO_VOLUME_SHARE = 0.02          # front-month candles with zero volume
VOLUME_UNAVAILABLE_SHARE = 0.50       # >= this share of zero volume -> FUTURES_VOLUME_UNAVAILABLE
MAX_FRONT_MISSING_SHARE = 0.005       # front-month 09:15-15:30 slots missing / expected on trading days
MAX_UNAVAILABLE_SESSIONS = 0.02       # sessions without a causal front-month contract
MAX_BASIS_SHARE = 0.02                # |futures close - spot close| / spot above this is flagged as mismatched
DISCONTINUITY_SHARE = 0.005           # rollover price jump beyond the spot move, flagged (reported only)
VWAP_TOL = 1e-6

SELECTED, DATA_UNAVAILABLE = "SELECTED", "DATA_UNAVAILABLE"
FUTURES_VOLUME_UNAVAILABLE = "FUTURES_VOLUME_UNAVAILABLE"

CONTRACT_COLUMNS = ["instrument_key", "trading_symbol", "exchange_segment", "instrument_type", "underlying", "expiry",
                    "lot_size", "strike", "tick_size", "source", "window_from", "window_to", "first_available_date",
                    "last_available_date", "trading_days", "candles", "invalid_rows", "status", "detail"]
INVALID_COLUMNS = ["instrument_key", "trading_symbol", "expiry", "timestamp", "open", "high", "low", "close", "volume",
                   "open_interest", "reason"]
MISSING_COLUMNS = ["scope", "instrument_key", "trading_symbol", "expiry", "trading_date", "expected", "candles",
                   "missing", "missing_s9_window", "duplicates", "unexpected", "reason", "missing_times"]
SELECTION_COLUMNS = ["session_date", "status", "reason", "instrument_key", "trading_symbol", "expiry", "dte",
                     "first_available_date", "candles", "s9_window_candles", "candidates", "next_contract"]
ROLLOVER_COLUMNS = ["roll_date", "old_contract", "old_expiry", "new_contract", "new_expiry", "old_last_close",
                    "new_first_open", "price_gap", "spot_prev_close", "spot_open", "spot_gap", "discontinuity",
                    "discontinuity_share", "flagged", "basis_before", "basis_after", "basis_jump",
                    "old_volume_expiry_day", "new_volume_expiry_day", "new_to_old_volume_ratio",
                    "old_missing_around_roll", "new_missing_around_roll"]
ALIGN_COLUMNS = ["trading_date", "instrument_key", "spot_bars", "futures_bars", "exact_matches", "missing_futures",
                 "missing_spot", "mismatched", "max_basis_share"]
VWAP_COLUMNS = ["timestamp", "instrument_key", "futures_close", "futures_volume", "cumulative_volume", "session_vwap"]


# ------------------------------------------------------------------ layout
class FuturesLayout:
    def __init__(self, root: str | Path = u6.DEFAULT_ROOT, underlying: str = "NIFTY") -> None:
        self.root = Path(root)
        self.underlying = underlying.upper()
        self.dir = self.root / "futures"
        self.contracts = self.dir / "contracts.csv"
        self.manifest = self.dir / "download_manifest.csv"
        self.discovery = self.dir / "discovery.json"
        self.invalid = self.dir / "futures_invalid_rows.csv"
        self.missing = self.dir / "futures_missing_data.csv"
        self.selection = self.dir / "futures_contract_selection.csv"
        self.rollover = self.dir / "futures_rollover.csv"
        self.alignment = self.dir / "futures_spot_alignment.csv"
        self.vwap = self.dir / "futures_vwap_diagnostic.csv"
        self.quality = self.dir / "data_quality.json"

    def contract_file(self, c: HistoricalContract) -> Path:
        return self.dir / f"{self.underlying}_FUT_{c.expiry.isoformat()}.parquet"

    def contract_files(self) -> list[Path]:
        return sorted(self.dir.glob(f"{self.underlying}_FUT_*.parquet"))


# ------------------------------------------------------------------ 1. discovery
def genuine_future(c: HistoricalContract, underlying: str, segment: str | None = None) -> str | None:
    """None when ``c`` is a genuine exchange-listed NIFTY future; otherwise the rejection reason."""
    if c.instrument_type != "FUT":
        return "NOT_FUTURE"
    if c.underlying != underlying.upper():
        return "WRONG_UNDERLYING"
    if c.strike not in (None, 0, 0.0):
        return "HAS_STRIKE"
    if segment not in (None, "", SEGMENT) or not c.instrument_key.startswith(SEGMENT + "|"):
        return "WRONG_SEGMENT"
    if c.lot_size <= 0:
        return "NO_LOT_SIZE"
    return None


@dataclass
class FuturesDiscovery:
    contracts: list[HistoricalContract]
    rejected: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    expiries_checked: int = 0


def discover_futures(catalog, underlying: str, start: date, end: date) -> FuturesDiscovery:
    """Instrument master (live) + expired-instrument API (past). Expiries come from metadata, never assumed."""
    name = underlying.upper()
    found: dict[str, HistoricalContract] = {}
    rejected: list[dict] = []
    issues: list[dict] = []
    for c in catalog.active_contracts():
        if c.underlying == name and c.instrument_type == "FUT":
            why = genuine_future(c, name)
            if why:
                rejected.append({"instrument_key": c.instrument_key, "source": ACTIVE, "reason": why})
            elif c.expiry >= start:
                found[c.instrument_key] = c
    try:
        expiries = [e for e in catalog.expired_expiries(name) if e >= start]
    except HistoricalDataError as exc:
        issues.append({"scope": "EXPIRED_EXPIRIES", **exc.as_dict()})
        if exc.code in FATAL_CODES:
            raise
        expiries = []
    key = catalog.underlying_key(name)
    for e in sorted(expiries):
        try:
            rows = catalog.client.expired_future_contracts(key, e)
        except HistoricalDataError as exc:
            issues.append({"scope": "EXPIRED_FUTURE_CONTRACTS", "expiry": e.isoformat(), **exc.as_dict()})
            if exc.code in FATAL_CODES:
                raise
            continue
        for row in rows or []:
            try:
                c = contract_from_row(row, EXPIRED, tick_in_paise=False)
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append({"instrument_key": str(row.get("instrument_key")), "source": EXPIRED,
                                 "reason": f"MALFORMED:{type(exc).__name__}"})
                continue
            why = genuine_future(c, name, row.get("segment"))
            if why:
                rejected.append({"instrument_key": c.instrument_key, "source": EXPIRED, "reason": why})
            elif c.instrument_key not in found:
                found[c.instrument_key] = c
    by_expiry = Counter(c.expiry for c in found.values())
    issues += [{"scope": "DUPLICATE_EXPIRY", "expiry": e.isoformat(), "code": "AMBIGUOUS_FRONT_MONTH",
                "message": f"{n} NIFTY FUT contracts share expiry {e}"} for e, n in by_expiry.items() if n > 1]
    after = sorted({c.expiry for c in found.values() if c.expiry > end})[:CONTRACTS_AFTER_END]
    keep = [c for c in found.values() if start <= c.expiry <= end or c.expiry in after]
    return FuturesDiscovery(sorted(keep, key=lambda c: (c.expiry, c.instrument_key)), rejected, issues,
                            len(expiries))


def contract_window(c: HistoricalContract, start: date, end: date) -> tuple[date, date] | None:
    lo, hi = max(start, c.expiry - timedelta(days=CONTRACT_LOOKBACK_DAYS)), min(end, c.expiry)
    return (lo, hi) if lo <= hi else None


# ------------------------------------------------------------------ 3. candle validation (never repairs)
def validate_futures(frame: pd.DataFrame, *, expiry: date | None = None,
                     interval: int = INTERVAL) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(clean, invalid_rows_with_reason, info). Invalid rows are excluded and reported, never modified."""
    info: dict[str, Any] = {"raw_rows": int(len(frame)), "raw_sorted": True, "issues": {}}
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=INVALID_COLUMNS), info
    f = frame.copy().reset_index(drop=True)
    ts = pd.to_datetime(f["timestamp"])
    reasons: list[list[str]] = [[] for _ in range(len(f))]

    def flag(mask, code: str) -> None:
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            reasons[i].append(code)

    if ts.dt.tz is None:
        flag(np.ones(len(f), bool), "TIMEZONE_NAIVE")
        local = ts
    else:
        local = ts.dt.tz_convert(IST)
        f["timestamp"] = local
    info["raw_sorted"] = bool(local.is_monotonic_increasing)
    px = f[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").astype(float)
    o, h, lo_, c = (px[k].to_numpy() for k in ("open", "high", "low", "close"))
    finite = np.isfinite(px.to_numpy()).all(axis=1)
    vol = pd.to_numeric(f["volume"], errors="coerce").astype(float).to_numpy() if "volume" in f \
        else np.full(len(f), np.nan)
    with np.errstate(invalid="ignore"):
        flag(~finite, "NON_FINITE_PRICE")
        flag(finite & (px.to_numpy() <= 0).any(axis=1), "NON_POSITIVE_PRICE")
        flag(finite & (h < np.maximum(o, c)), "HIGH_BELOW_OPEN_OR_CLOSE")
        flag(finite & (lo_ > np.minimum(o, c)), "LOW_ABOVE_OPEN_OR_CLOSE")
        flag(finite & (h < lo_), "HIGH_BELOW_LOW")
        flag(~np.isfinite(vol), "NON_FINITE_VOLUME")
        flag(np.isfinite(vol) & (vol < 0), "NEGATIVE_VOLUME")
        if "open_interest" in f:
            oi = pd.to_numeric(f["open_interest"], errors="coerce").astype(float).to_numpy()
            flag(np.isfinite(oi) & (oi < 0), "NEGATIVE_OPEN_INTEREST")
    if local.dt.tz is not None:
        t = local.dt
        flag(((t.minute % interval) != 0) | (t.second != 0) | (t.microsecond != 0), "OFF_GRID")
        flag((t.time < SESSION_START) | (t.time >= SESSION_END), "OUTSIDE_SESSION")
        flag(t.weekday >= 5, "WEEKEND")
        if expiry is not None:
            flag(t.date > expiry, "AFTER_EXPIRY")
    ok = np.array([not r for r in reasons])
    if ok.any():
        sub = f[ok]
        cols = [k for k in ("open", "high", "low", "close", "volume", "open_interest") if k in f]
        for _, g in sub.groupby(sub["timestamp"]):
            if len(g) < 2:
                continue
            same = len(g[cols].astype(str).drop_duplicates()) == 1
            for i in (g.index[:-1] if same else g.index):
                reasons[i].append("DUPLICATE_EXACT" if same else "DUPLICATE_CONFLICT")
    bad = np.array([bool(r) for r in reasons])
    invalid = f[bad].copy()
    invalid["reason"] = [";".join(reasons[i]) for i in np.flatnonzero(bad)]
    invalid["timestamp"] = invalid["timestamp"].astype(str)
    clean = f[~bad].sort_values("timestamp").reset_index(drop=True)
    info["issues"] = dict(Counter(code for r in reasons for code in r))
    info.update(valid_rows=int(len(clean)), invalid_rows=int(bad.sum()))
    return clean, invalid.reindex(columns=INVALID_COLUMNS), info


def file_checks(frame: pd.DataFrame, expiry: date | None = None) -> dict[str, Any]:
    """Checks on a STORED contract file (expects zero problems; nothing is modified)."""
    if frame.empty:
        return {"rows": 0, "ok": False, "empty": True}
    ts = pd.to_datetime(frame["timestamp"])
    aware = ts.dt.tz is not None
    local = ts.dt.tz_convert(IST) if aware else ts
    ist = aware and all(v.utcoffset() == timedelta(hours=5, minutes=30) for v in local.iloc[[0, -1]])
    _, invalid, info = validate_futures(frame, expiry=expiry)
    q = {"rows": int(len(frame)), "tz_aware": bool(aware), "ist": bool(ist),
         "sorted": bool(local.is_monotonic_increasing), "duplicate_timestamps": int(local.duplicated().sum()),
         "invalid_rows": int(len(invalid)), "issues": info["issues"]}
    q["ok"] = bool(aware and ist and q["sorted"] and q["duplicate_timestamps"] == 0 and q["invalid_rows"] == 0)
    return q


# ------------------------------------------------------------------ 2. download (monthly chunks, resumable)
def _attach(frame: pd.DataFrame, c: HistoricalContract) -> pd.DataFrame:
    out = frame.copy()
    out["instrument_key"] = c.instrument_key
    out["trading_symbol"] = c.trading_symbol
    out["expiry"] = c.expiry.isoformat()
    return out


def download_contract(client, c: HistoricalContract, layout: FuturesLayout, manifest: u6.Manifest, start: date,
                      end: date, *, resume: bool = False,
                      now: Callable[[], datetime] = lambda: datetime.now(IST)) -> dict[str, Any]:
    win = contract_window(c, start, end)
    path = layout.contract_file(c)
    if win is None:
        return {"status": "OUT_OF_WINDOW", "invalid": pd.DataFrame(columns=INVALID_COLUMNS), "window": None}
    months = u6.month_windows(*win)
    prev = [manifest.get(c.instrument_key, m) for m, _, _ in months]
    if path.is_file() and all(p and p["status"] in (u6.SUCCESS, u6.EMPTY) for p in prev):
        q = file_checks(u6.read_parquet(path), c.expiry)
        if q["ok"]:
            return {"status": "SKIPPED_VALID", "invalid": None, "window": win}
    if resume and not path.is_file() and prev and all(p and p["status"] in (u6.EMPTY, u6.UNAVAILABLE) for p in prev):
        return {"status": "SKIPPED_TERMINAL", "invalid": None, "window": win}
    parts, failed = [], []
    for m, a, b in months:
        item = u6.DownloadItem(instrument_key=c.instrument_key, instrument_type="FUT", source=c.source, month=m,
                               start=a, end=b, path=path, underlying=c.underlying, expiry=c.expiry,
                               trading_symbol=c.trading_symbol)
        try:
            candles = (client.expired_candles(c.instrument_key, a, b, interval=INTERVAL) if c.source == EXPIRED
                       else client.candles(c.instrument_key, a, b, interval=INTERVAL))
        except HistoricalDataError as exc:
            status = u6.UNAVAILABLE if exc.code in u6.UNAVAILABLE_CODES else u6.FAILED
            manifest.upsert(u6._row(item, status, http_status=exc.status or "", error_code=exc.code,
                                    detail=exc.message, now=now()))
            if exc.code in FATAL_CODES:
                raise
            failed.append(f"{m}:{exc.code}")
            continue
        f = candles_to_frame(candles)
        if not f.empty:
            d = f["timestamp"].dt.date
            f = f[(d >= a) & (d <= b)]
        parts.append(f)
        manifest.upsert(u6._row(item, u6.SUCCESS if len(f) else u6.EMPTY, candle_count=len(f), http_status=200,
                                raw={"raw_rows": len(f)}, now=now()))
    raw = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) \
        else pd.DataFrame()
    if raw.empty:
        return {"status": u6.FAILED if failed else u6.EMPTY, "invalid": pd.DataFrame(columns=INVALID_COLUMNS),
                "window": win, "failed_months": failed}
    clean, invalid, info = validate_futures(_attach(raw, c), expiry=c.expiry)
    if not clean.empty:
        u6.write_parquet(path, clean)
    return {"status": u6.FAILED if failed else u6.SUCCESS, "invalid": invalid, "window": win, "info": info,
            "failed_months": failed}


# ------------------------------------------------------------------ loading
def contract_rows_from_csv(layout: FuturesLayout) -> list[HistoricalContract]:
    out = []
    for r in u6._read_csv(layout.contracts):
        out.append(HistoricalContract(instrument_key=r["instrument_key"], underlying=r["underlying"],
                                      expiry=date.fromisoformat(r["expiry"]), instrument_type=r["instrument_type"],
                                      lot_size=int(float(r["lot_size"] or 0)), source=r["source"], strike=None,
                                      trading_symbol=r["trading_symbol"],
                                      tick_size=float(r["tick_size"]) if r.get("tick_size") else None))
    return sorted(out, key=lambda c: (c.expiry, c.instrument_key))


def load_futures(root: str | Path = u6.DEFAULT_ROOT, underlying: str = "NIFTY"
                 ) -> tuple[list[HistoricalContract], dict[str, pd.DataFrame]]:
    """Discovered contracts and their STORED (validated) candle frames, keyed by instrument_key. Read-only."""
    layout = FuturesLayout(root, underlying)
    contracts = contract_rows_from_csv(layout)
    frames: dict[str, pd.DataFrame] = {}
    for c in contracts:
        p = layout.contract_file(c)
        if p.is_file():
            f = u6.read_parquet(p)
            if not f.empty:
                frames[c.instrument_key] = f.sort_values("timestamp").reset_index(drop=True)
    return contracts, frames


# ------------------------------------------------------------------ 4. session validation
def _days_of(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST).dt.date


def session_rows(c: HistoricalContract, frame: pd.DataFrame | None, trading_days: Sequence[date],
                 scope: str) -> list[dict]:
    """One row per trading day of the contract's active span with any gap (missing / duplicate / unexpected)."""
    rows = []
    if frame is None or frame.empty:
        return rows
    ts = pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST)
    first = ts.min().date()
    by_day = {d: g for d, g in ts.groupby(ts.dt.date)}
    for d in trading_days:
        if d < first or d > c.expiry:
            continue
        got = by_day.get(d, pd.Series([], dtype=ts.dtype))
        slots = session_slots(d)
        have = set(got)
        miss = [s for s in slots if s not in have]
        miss_s9 = [s for s in miss if s.time() < S9_END]
        dup = int(got.duplicated().sum())
        unexpected = int(sum(1 for t in have if t not in set(slots)))
        if miss or dup or unexpected:
            rows.append({"scope": scope, "instrument_key": c.instrument_key, "trading_symbol": c.trading_symbol,
                         "expiry": c.expiry.isoformat(), "trading_date": d.isoformat(), "expected": len(slots),
                         "candles": int(len(got)), "missing": len(miss), "missing_s9_window": len(miss_s9),
                         "duplicates": dup, "unexpected": unexpected,
                         "reason": "NO_CANDLES_ON_SESSION" if not len(got) else "MISSING_INTERVALS" if miss
                         else "DUPLICATE_OR_UNEXPECTED",
                         "missing_times": " ".join(s.strftime("%H:%M") for s in miss[:12])})
    return rows


# ------------------------------------------------------------------ 5. volume validation
def volume_stats(vol: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(vol, errors="coerce").astype(float)
    fin = v[np.isfinite(v)]
    n = int(len(v))
    out: dict[str, Any] = {"candles": n, "non_finite": int(n - len(fin)), "total": float(fin.sum()) if n else 0.0,
                           "zero": int((fin == 0).sum()), "zero_share": float((fin == 0).sum() / n) if n else None}
    if len(fin):
        out.update(median=float(fin.median()), mean=float(fin.mean()), min=float(fin.min()), max=float(fin.max()),
                   percentiles={f"p{p}": float(np.percentile(fin, p)) for p in (1, 5, 25, 50, 75, 95, 99)})
    return out


def volume_verdict(stats: dict) -> str:
    if not stats["candles"] or stats["total"] <= 0 or stats["non_finite"] or \
            (stats["zero_share"] or 0) >= VOLUME_UNAVAILABLE_SHARE:
        return FUTURES_VOLUME_UNAVAILABLE
    return "OK" if stats["zero_share"] <= MAX_ZERO_VOLUME_SHARE else "DEGRADED"


# ------------------------------------------------------------------ 6. session VWAP (causal, resets daily)
def session_vwap(frame: pd.DataFrame) -> pd.DataFrame:
    """timestamp-sorted copy with typical_price, cumulative_volume and session_vwap (NaN until volume > 0).
    Each value uses only candles of the same session up to and including that candle."""
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"]).dt.tz_convert(IST)
    f = f.sort_values("timestamp").reset_index(drop=True)
    day = f["timestamp"].dt.date
    tp = (f["high"] + f["low"] + f["close"]) / 3.0
    v = f["volume"].astype(float)
    f["typical_price"] = tp
    f["cumulative_volume"] = v.groupby(day).cumsum()
    pv = (tp * v).groupby(day).cumsum()
    f["session_vwap"] = (pv / f["cumulative_volume"]).where(f["cumulative_volume"] > 0)
    return f


def vwap_checks(frame: pd.DataFrame, sample: int = 25) -> dict[str, Any]:
    if frame.empty:
        return {"checked": 0, "passed": False, "reason": "NO_DATA"}
    f = session_vwap(frame)
    day = f["timestamp"].dt.date
    lo_run = f["low"].groupby(day).cummin()
    hi_run = f["high"].groupby(day).cummax()
    have = f["session_vwap"].notna()
    bounded = ((f["session_vwap"] >= lo_run - VWAP_TOL) & (f["session_vwap"] <= hi_run + VWAP_TOL)) | ~have
    nan_after_volume = int((~have & (f["cumulative_volume"] > 0)).sum())
    first = f.groupby(day).head(1)
    first_ok = first[first["volume"] > 0]
    reset_ok = bool(np.allclose(first_ok["session_vwap"], first_ok["typical_price"], rtol=0, atol=1e-9))
    cum_ok = bool((f["volume"].astype(float) >= 0).all())
    idx = np.linspace(0, len(f) - 1, num=min(sample, len(f))).astype(int)
    trunc_bad = 0
    for i in idx:
        t = f["timestamp"].iat[i]
        again = session_vwap(frame[pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST) <= t])
        a, b = f["session_vwap"].iat[i], again["session_vwap"].iat[-1]
        if not ((pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and math.isclose(a, b,
                                                                                                rel_tol=1e-12))):
            trunc_bad += 1
    out = {"checked": int(len(f)), "chronological": bool(f["timestamp"].is_monotonic_increasing),
           "within_session_range": bool(bounded.all()), "out_of_range": int((~bounded).sum()),
           "undefined_after_volume": nan_after_volume, "resets_each_session": reset_ok,
           "cumulative_volume_non_decreasing": cum_ok, "truncation_samples": int(len(idx)),
           "truncation_mismatches": trunc_bad, "defined_share": float(have.mean())}
    out["passed"] = bool(out["chronological"] and out["within_session_range"] and not nan_after_volume
                         and reset_ok and cum_ok and not trunc_bad
                         and out["defined_share"] >= 1.0 - MAX_ZERO_VOLUME_SHARE)
    return out


# ------------------------------------------------------------------ 7. causal front-month selection
def select_contracts(contracts: Sequence[HistoricalContract], frames: dict[str, pd.DataFrame],
                     sessions: Sequence[date]) -> list[dict]:
    """Per session: nearest unexpired expiry from metadata; never a later contract when that one lacks data."""
    first_seen = {k: _days_of(f).min() for k, f in frames.items() if not f.empty}
    by_day = {k: _days_of(f).value_counts().to_dict() for k, f in frames.items() if not f.empty}
    s9_by_day = {}
    for k, f in frames.items():
        if f.empty:
            continue
        t = pd.to_datetime(f["timestamp"]).dt.tz_convert(IST)
        s9_by_day[k] = t[t.dt.time < S9_END].dt.date.value_counts().to_dict()
    out = []
    for d in sessions:
        cands = sorted((c for c in contracts if c.expiry >= d), key=lambda c: (c.expiry, c.instrument_key))
        row: dict[str, Any] = {"session_date": d.isoformat(), "status": DATA_UNAVAILABLE, "reason": None,
                               "candidates": " ".join(c.instrument_key for c in cands[:3])}
        if not cands:
            out.append(dict(row, reason="NO_UNEXPIRED_CONTRACT"))
            continue
        front = [c for c in cands if c.expiry == cands[0].expiry]
        nxt = next((c for c in cands if c.expiry > cands[0].expiry), None)
        c = front[0]
        row.update(instrument_key=c.instrument_key, trading_symbol=c.trading_symbol, expiry=c.expiry.isoformat(),
                   dte=(c.expiry - d).days, next_contract=nxt.instrument_key if nxt else None,
                   first_available_date=first_seen[c.instrument_key].isoformat() if c.instrument_key in first_seen
                   else None, candles=int(by_day.get(c.instrument_key, {}).get(d, 0)),
                   s9_window_candles=int(s9_by_day.get(c.instrument_key, {}).get(d, 0)))
        if len(front) > 1:
            row["reason"] = "AMBIGUOUS_FRONT_MONTH"
        elif c.instrument_key not in frames:
            row["reason"] = "CONTRACT_DATA_MISSING"
        elif first_seen[c.instrument_key] > d:
            row["reason"] = "NOT_YET_TRADING"
        elif not row["candles"]:
            row["reason"] = "NO_CANDLES_ON_SESSION"
        else:
            row["status"] = SELECTED
        out.append(row)
    return out


def selection_checks(selection: Sequence[dict], contracts: Sequence[HistoricalContract],
                     frames: dict[str, pd.DataFrame], sessions: Sequence[date], seed: int = 5) -> dict[str, Any]:
    exp = {c.instrument_key: c.expiry for c in contracts}
    bad = []
    for r in selection:
        if r["status"] != SELECTED:
            continue
        d = date.fromisoformat(r["session_date"])
        e = exp.get(r["instrument_key"])
        nearest = min((c.expiry for c in contracts if c.expiry >= d), default=None)
        if e is None or e < d:
            bad.append(f"{d}:EXPIRED_OR_UNKNOWN")
        elif e != nearest:
            bad.append(f"{d}:NOT_NEAREST_EXPIRY")
        if r["first_available_date"] is None or date.fromisoformat(r["first_available_date"]) > d:
            bad.append(f"{d}:NOT_STARTED")
        if not r["candles"]:
            bad.append(f"{d}:NO_CANDLE")
    rng = np.random.default_rng(seed)
    warped = {}
    for k, f in frames.items():
        g = f.copy()
        for col in ("open", "high", "low", "close", "volume"):
            g[col] = g[col].astype(float) * rng.uniform(0.2, 5.0, len(g))
        warped[k] = g
    key = lambda rows: [(r["session_date"], r["status"], r.get("instrument_key")) for r in rows]  # noqa: E731
    invariant = key(select_contracts(contracts, warped, sessions)) == key(selection)
    unavailable = [r for r in selection if r["status"] != SELECTED]
    share = len(unavailable) / len(selection) if selection else 1.0
    out = {"sessions": len(selection), "selected": len(selection) - len(unavailable),
           "data_unavailable": len(unavailable), "unavailable_share": share,
           "unavailable_reasons": dict(Counter(r["reason"] for r in unavailable)), "violations": bad[:20],
           "independent_of_price_and_volume": invariant}
    out["passed"] = bool(selection and not bad and invariant and share <= MAX_UNAVAILABLE_SESSIONS)
    return out


def front_month_bars(selection: Sequence[dict], frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Selected contract's own candles for each selected session (contracts kept separate, no price adjustment)."""
    parts = []
    for r in selection:
        if r["status"] != SELECTED:
            continue
        f = frames[r["instrument_key"]]
        d = date.fromisoformat(r["session_date"])
        g = f[_days_of(f) == d].copy()
        g["instrument_key"] = r["instrument_key"]
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "instrument_key"])
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_convert(IST)
    return out.sort_values("timestamp").reset_index(drop=True)


# ------------------------------------------------------------------ 8. rollover analysis (reported, no stitching)
def _daily_volume(frame: pd.DataFrame | None, d: date) -> float | None:
    if frame is None or frame.empty:
        return None
    sub = frame[_days_of(frame) == d]
    return float(sub["volume"].sum()) if len(sub) else None


def rollover_analysis(contracts: Sequence[HistoricalContract], frames: dict[str, pd.DataFrame],
                      selection: Sequence[dict], spot: pd.DataFrame, trading_days: Sequence[date]
                      ) -> tuple[list[dict], dict]:
    by_key = {c.instrument_key: c for c in contracts}
    st = pd.to_datetime(spot["timestamp"]).dt.tz_convert(IST) if not spot.empty else pd.Series([], dtype=object)
    spot_day = {d: g for d, g in spot.assign(_t=st).groupby(st.dt.date)} if not spot.empty else {}
    active = Counter()
    for d in trading_days:
        active[sum(1 for f in frames.values() if (_days_of(f) == d).any())] += 1
    sel = [r for r in selection if r["status"] == SELECTED]
    rows = []
    for a, b in zip(sel, sel[1:]):
        if a["instrument_key"] == b["instrument_key"]:
            continue
        da, db = date.fromisoformat(a["session_date"]), date.fromisoformat(b["session_date"])
        fo, fn = frames[a["instrument_key"]], frames[b["instrument_key"]]
        old_day = fo[_days_of(fo) == da]
        new_day = fn[_days_of(fn) == db]
        old_close, new_open = float(old_day["close"].iloc[-1]), float(new_day["open"].iloc[0])
        sa, sb = spot_day.get(da), spot_day.get(db)
        s_close = float(sa["close"].iloc[-1]) if sa is not None and len(sa) else None
        s_open = float(sb["open"].iloc[0]) if sb is not None and len(sb) else None
        gap = new_open - old_close
        sgap = s_open - s_close if s_open is not None and s_close is not None else None
        disc = gap - sgap if sgap is not None else None
        near = [d for d in trading_days if abs((d - da).days) <= 4]
        miss = lambda f: sum(max(0, session_slot_count(d) - int((_days_of(f) == d).sum())) for d in near  # noqa: E731
                             if (_days_of(f) == d).any())
        vo, vn = _daily_volume(fo, da), _daily_volume(fn, da)
        rows.append({
            "roll_date": db.isoformat(), "old_contract": a["trading_symbol"], "old_expiry": a["expiry"],
            "new_contract": b["trading_symbol"], "new_expiry": b["expiry"], "old_last_close": old_close,
            "new_first_open": new_open, "price_gap": gap, "spot_prev_close": s_close, "spot_open": s_open,
            "spot_gap": sgap, "discontinuity": disc,
            "discontinuity_share": abs(disc) / s_open if disc is not None and s_open else None,
            "flagged": bool(disc is not None and s_open and abs(disc) / s_open > DISCONTINUITY_SHARE),
            "basis_before": old_close - s_close if s_close is not None else None,
            "basis_after": new_open - s_open if s_open is not None else None,
            "basis_jump": (new_open - s_open) - (old_close - s_close) if s_open is not None and s_close is not None
            else None,
            "old_volume_expiry_day": vo, "new_volume_expiry_day": vn,
            "new_to_old_volume_ratio": vn / vo if vo and vn is not None else None,
            "old_missing_around_roll": miss(fo), "new_missing_around_roll": miss(fn)})
    summary = {"rollovers": len(rows), "rollover_dates": [r["roll_date"] for r in rows],
               "flagged_discontinuities": sum(r["flagged"] for r in rows),
               "max_discontinuity_share": max((r["discontinuity_share"] or 0 for r in rows), default=0.0),
               "active_contracts_per_day": {str(k): v for k, v in sorted(active.items())},
               "contracts": [by_key[k].trading_symbol for k in frames if k in by_key],
               "method": "contracts kept separate; no price stitching or back-adjustment"}
    return rows, summary


# ------------------------------------------------------------------ 9. spot / futures alignment
def alignment(selection: Sequence[dict], bars: pd.DataFrame, spot: pd.DataFrame,
              trading_days: Sequence[date]) -> tuple[list[dict], dict]:
    st = pd.to_datetime(spot["timestamp"]).dt.tz_convert(IST)
    spot_close = pd.Series(spot["close"].astype(float).to_numpy(), index=st)
    spot_close = spot_close[~spot_close.index.duplicated(keep="last")]
    fb = bars.copy()
    sel = {r["session_date"]: r for r in selection}
    rows = []
    tot = Counter()
    for d in trading_days:
        r = sel.get(d.isoformat())
        s_ts = set(spot_close.index[spot_close.index.date == d])
        f_day = fb[fb["timestamp"].dt.date == d] if not fb.empty else fb
        f_ts = set(f_day["timestamp"])
        grid = set(session_slots(d))
        exact = s_ts & f_ts
        mism = len([t for t in f_ts if t not in grid])
        basis = 0.0
        if exact:
            fc = f_day.set_index("timestamp")["close"].astype(float)
            m = sorted(exact)
            basis = float((abs(fc.loc[m].to_numpy() - spot_close.loc[m].to_numpy()) / spot_close.loc[m].to_numpy())
                          .max())
        mism += int(basis > MAX_BASIS_SHARE)
        row = {"trading_date": d.isoformat(), "instrument_key": r.get("instrument_key") if r else None,
               "spot_bars": len(s_ts), "futures_bars": len(f_ts), "exact_matches": len(exact),
               "missing_futures": len(s_ts - f_ts), "missing_spot": len(f_ts - s_ts), "mismatched": mism,
               "max_basis_share": basis}
        rows.append(row)
        for k in ("spot_bars", "futures_bars", "exact_matches", "missing_futures", "missing_spot", "mismatched"):
            tot[k] += row[k]
    selected_days = {r["session_date"] for r in selection if r["status"] == SELECTED}
    expected_sel = sum(r["spot_bars"] for r in rows if r["trading_date"] in selected_days)
    missing_sel = sum(r["missing_futures"] for r in rows if r["trading_date"] in selected_days)
    summary = {**dict(tot), "selected_sessions": len(selected_days),
               "missing_futures_share_on_selected": missing_sel / expected_sel if expected_sel else None,
               "max_basis_share": max((r["max_basis_share"] for r in rows), default=0.0),
               "forward_fill": False}
    summary["passed"] = bool(expected_sel and tot["mismatched"] == 0 and summary["missing_futures_share_on_selected"]
                             <= MAX_FRONT_MISSING_SHARE)
    return rows, summary


# ------------------------------------------------------------------ 15. full validation of the stored data
def trading_days_of(spot: pd.DataFrame, start: date, end: date) -> list[date]:
    if spot.empty:
        return []
    d = pd.to_datetime(spot["timestamp"]).dt.tz_convert(IST).dt.date
    return sorted(x for x in set(d) if start <= x <= end)


def validate_store(*, root: str | Path, start: date, end: date, spot: pd.DataFrame, secrets: Iterable[str] = (),
                   extra_scan: Sequence[Path] = (), download: dict | None = None, write: bool = True,
                   underlying: str = "NIFTY") -> dict[str, Any]:
    """Offline validation of everything stored under ``futures/``. Writes the CSV / JSON diagnostics."""
    layout = FuturesLayout(root, underlying)
    contracts, frames = load_futures(root, underlying)
    days = trading_days_of(spot, start, end)
    selection = select_contracts(contracts, frames, days)
    sel_chk = selection_checks(selection, contracts, frames, days)
    bars = front_month_bars(selection, frames)
    files = {c.instrument_key: file_checks(frames[c.instrument_key], c.expiry) for c in contracts
             if c.instrument_key in frames}
    sel_keys = {r["instrument_key"] for r in selection if r["status"] == SELECTED}
    missing_rows: list[dict] = []
    for c in contracts:
        missing_rows += session_rows(c, frames.get(c.instrument_key), days,
                                     "FRONT_MONTH_CONTRACT" if c.instrument_key in sel_keys else "OTHER_CONTRACT")
    for r in selection:
        if r["status"] != SELECTED:
            missing_rows.append({"scope": "SESSION_NO_CAUSAL_CONTRACT", "instrument_key": r.get("instrument_key"),
                                 "trading_symbol": r.get("trading_symbol"), "expiry": r.get("expiry"),
                                 "trading_date": r["session_date"], "expected": session_slot_count(date.fromisoformat(r["session_date"])),
                                 "candles": r.get("candles", 0),
                                 "missing": session_slot_count(date.fromisoformat(r["session_date"])),
                                 "missing_s9_window": sum(slot.time() < S9_END for slot in session_slots(date.fromisoformat(r["session_date"]))),
                                 "reason": r["reason"]})
    vol_front = volume_stats(bars["volume"]) if not bars.empty else volume_stats(pd.Series([], dtype=float))
    s9_bars = bars[bars["timestamp"].dt.time < S9_END] if not bars.empty else bars
    vol_s9 = volume_stats(s9_bars["volume"]) if not bars.empty else vol_front
    vol_contracts = {c.trading_symbol: volume_stats(frames[c.instrument_key]["volume"]) for c in contracts
                     if c.instrument_key in frames}
    verdict = volume_verdict(vol_front)
    vwap_parts = [session_vwap(frames[r["instrument_key"]][_days_of(frames[r["instrument_key"]])
                                                           == date.fromisoformat(r["session_date"])])
                  for r in selection if r["status"] == SELECTED]
    vwap_frame = pd.concat(vwap_parts, ignore_index=True) if vwap_parts else pd.DataFrame()
    vchk = vwap_checks(bars) if not bars.empty else {"checked": 0, "passed": False, "reason": "NO_FRONT_MONTH_DATA"}
    roll_rows, roll = rollover_analysis(contracts, frames, selection, spot, days)
    align_rows, align = alignment(selection, bars, spot, days)
    invalid = u6._read_csv(layout.invalid)
    leak_ok, leaks = u6.token_leak_check([layout.dir, *extra_scan], secrets)
    discovery = json.loads(layout.discovery.read_text(encoding="utf-8")) if layout.discovery.is_file() else {}
    front_missing = sum(max(0, session_slot_count(date.fromisoformat(r["session_date"])) - int(r["candles"]))
                        for r in selection if r["status"] == SELECTED)
    dups = sum(int(r.get("duplicates") or 0) for r in missing_rows) + sum(q.get("duplicate_timestamps", 0)
                                                                         for q in files.values())
    stored_ok = bool(files) and all(q["ok"] for q in files.values())
    status_rows = u6._read_csv(layout.contracts)
    failed = [r["trading_symbol"] for r in status_rows if r.get("status") in (u6.FAILED, u6.UNAVAILABLE)
              and r["instrument_key"] in {x.get("instrument_key") for x in selection}]
    q = {
        "banner": BANNER, "period": {"from": start.isoformat(), "to": end.isoformat()},
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "source": "Upstox instrument master (live FUT) + expired-instrument API (expired FUT); real 5-minute candles",
        "contracts_discovered": len(contracts), "contracts_downloaded": len(frames),
        "contracts_failed": failed, "contracts": [{"instrument_key": c.instrument_key,
                                                    "trading_symbol": c.trading_symbol,
                                                    "expiry": c.expiry.isoformat(), "lot_size": c.lot_size,
                                                    "source": c.source} for c in contracts],
        "rejected_instruments": discovery.get("rejected", []), "discovery_issues": discovery.get("issues", []),
        "total_candles": int(sum(len(f) for f in frames.values())), "front_month_candles": int(len(bars)),
        "trading_days": len(days), "missing_candles": {"front_month": int(front_missing),
                                                       "all_contracts": int(sum(r["missing"] for r in missing_rows
                                                                                if "CONTRACT" in r["scope"]
                                                                                and r["scope"] != "SESSION_NO_CAUSAL_CONTRACT"))},
        "duplicate_candles": int(dups), "invalid_candles": len(invalid),
        "invalid_reasons": dict(Counter(r["reason"] for r in invalid)), "stored_files": files,
        "stored_files_ok": stored_ok,
        "zero_volume_candles": {"front_month": vol_front["zero"], "front_month_share": vol_front["zero_share"]},
        "volume": {"front_month": vol_front, "front_month_s9_window": vol_s9, "per_contract": vol_contracts,
                   "verdict": verdict, "substitution": "none - futures volume only"},
        "vwap": vchk, "contract_selection": sel_chk, "rollover": roll, "timestamp_alignment": align,
        "token_leak": {"passed": leak_ok, "files": leaks}, "download": download or {},
        "thresholds": {"max_zero_volume_share": MAX_ZERO_VOLUME_SHARE,
                       "volume_unavailable_share": VOLUME_UNAVAILABLE_SHARE,
                       "max_front_missing_share": MAX_FRONT_MISSING_SHARE,
                       "max_unavailable_sessions": MAX_UNAVAILABLE_SESSIONS, "max_basis_share": MAX_BASIS_SHARE},
    }
    q["pass"] = bool(contracts and frames and not failed and stored_ok and verdict == "OK" and vchk["passed"]
                     and sel_chk["passed"] and align["passed"] and leak_ok)
    q["status"] = "PASS" if q["pass"] else (FUTURES_VOLUME_UNAVAILABLE if verdict == FUTURES_VOLUME_UNAVAILABLE
                                            else "FAIL")
    if write:
        layout.dir.mkdir(parents=True, exist_ok=True)
        u6._write_csv(layout.missing, MISSING_COLUMNS, missing_rows)
        u6._write_csv(layout.selection, SELECTION_COLUMNS, selection)
        u6._write_csv(layout.rollover, ROLLOVER_COLUMNS, roll_rows)
        u6._write_csv(layout.alignment, ALIGN_COLUMNS, align_rows)
        diag = vwap_frame.rename(columns={"close": "futures_close", "volume": "futures_volume"}) \
            if not vwap_frame.empty else pd.DataFrame(columns=VWAP_COLUMNS)
        if not diag.empty:
            diag["timestamp"] = diag["timestamp"].astype(str)
        u6._write_csv(layout.vwap, VWAP_COLUMNS, diag.to_dict("records"))
        layout.quality.write_text(json.dumps(q, indent=2, default=str), encoding="utf-8")
        leak_ok2, leaks2 = u6.token_leak_check([layout.dir, *extra_scan], secrets)
        if not leak_ok2:
            q["token_leak"] = {"passed": False, "files": leaks2}
            q["pass"], q["status"] = False, "FAIL"
            layout.quality.write_text(json.dumps(q, indent=2, default=str), encoding="utf-8")
    q["_selection"], q["_bars"] = selection, bars
    return q


# ------------------------------------------------------------------ orchestration (network only here)
def run_futures_download(*, client, catalog, root: str | Path, start: date, end: date, spot: pd.DataFrame,
                         secrets: Iterable[str] = (), resume: bool = False, validate_only: bool = False,
                         underlying: str = "NIFTY", progress: Callable[[str], None] = lambda m: None) -> dict:
    layout = FuturesLayout(root, underlying)
    layout.dir.mkdir(parents=True, exist_ok=True)
    download: dict[str, Any] = {}
    if not validate_only:
        progress("discovering NIFTY FUT contracts (instrument master + expired-instrument API) ...")
        disc = discover_futures(catalog, underlying, start, end)
        layout.discovery.write_text(json.dumps({"expiries_checked": disc.expiries_checked,
                                                "rejected": disc.rejected, "issues": disc.issues}, indent=2,
                                               default=str), encoding="utf-8")
        progress(f"{len(disc.contracts)} genuine NIFTY FUT contracts: "
                 + ", ".join(f"{c.trading_symbol} ({c.expiry})" for c in disc.contracts))
        manifest = u6.Manifest(layout.manifest)
        rows, invalid_parts, actions = [], [], Counter()
        for c in disc.contracts:
            res = download_contract(client, c, layout, manifest, start, end, resume=resume)
            actions[res["status"]] += 1
            if res.get("invalid") is not None and len(res["invalid"]):
                invalid_parts.append(res["invalid"])
            p = layout.contract_file(c)
            f = u6.read_parquet(p) if p.is_file() else pd.DataFrame()
            d = _days_of(f) if not f.empty else pd.Series([], dtype=object)
            rows.append({"instrument_key": c.instrument_key, "trading_symbol": c.trading_symbol,
                         "exchange_segment": SEGMENT, "instrument_type": c.instrument_type,
                         "underlying": c.underlying, "expiry": c.expiry.isoformat(), "lot_size": c.lot_size,
                         "strike": "", "tick_size": c.tick_size, "source": c.source,
                         "window_from": res["window"][0].isoformat() if res["window"] else "",
                         "window_to": res["window"][1].isoformat() if res["window"] else "",
                         "first_available_date": d.min().isoformat() if len(d) else "",
                         "last_available_date": d.max().isoformat() if len(d) else "",
                         "trading_days": int(d.nunique()) if len(d) else 0, "candles": int(len(f)),
                         "invalid_rows": int(len(res["invalid"])) if res.get("invalid") is not None else "",
                         "status": res["status"], "detail": " ".join(res.get("failed_months", []))})
            progress(f"{c.trading_symbol}: {res['status']} ({len(f)} candles)")
        u6._write_csv(layout.contracts, CONTRACT_COLUMNS, rows)
        redone = {r["instrument_key"] for r in rows if r["status"] not in ("SKIPPED_VALID", "SKIPPED_TERMINAL")}
        kept = [r for r in u6._read_csv(layout.invalid) if r["instrument_key"] not in redone]
        new = [r for p in invalid_parts for r in p.to_dict("records")]
        u6._write_csv(layout.invalid, INVALID_COLUMNS, kept + new)
        download = {"actions": dict(actions), "contracts": len(disc.contracts), "issues": disc.issues,
                    "requests": getattr(client, "requests_made", None)}
    elif not layout.contracts.is_file():
        raise SystemExit(f"{layout.contracts} missing - run without --validate first")
    if not layout.invalid.is_file():
        u6._write_csv(layout.invalid, INVALID_COLUMNS, [])
    return validate_store(root=root, start=start, end=end, spot=spot, secrets=secrets, download=download,
                          underlying=underlying)


def data_text(q: dict) -> list[str]:
    return ["FUTURES DATA:", f"contracts: {q['contracts_downloaded']} downloaded / {q['contracts_discovered']} "
            f"discovered (failed: {len(q['contracts_failed'])})", f"candles: {q['total_candles']} "
            f"(front-month {q['front_month_candles']})", f"trading days: {q['trading_days']}",
            f"missing: {q['missing_candles']['front_month']} front-month 5m slots",
            f"duplicates: {q['duplicate_candles']}",
            f"zero volume: {q['zero_volume_candles']['front_month']} front-month candles "
            f"({(q['zero_volume_candles']['front_month_share'] or 0) * 100:.2f}%)",
            f"data quality: {q['status']}", "", "VWAP:", "PASS" if q["vwap"].get("passed") else "FAIL", "",
            "CAUSAL CONTRACT SELECTION:", "PASS" if q["contract_selection"]["passed"] else "FAIL", "",
            "SPOT/FUTURES ALIGNMENT:", "PASS" if q["timestamp_alignment"]["passed"] else "FAIL", "",
            "TOKEN LEAK:", "PASS" if q["token_leak"]["passed"] else "FAIL"]
