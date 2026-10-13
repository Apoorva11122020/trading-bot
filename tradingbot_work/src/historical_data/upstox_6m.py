"""Six-month Upstox NIFTY + NIFTY option 5-minute dataset (PAPER | DATA ACQUISITION ONLY | NO ORDERS).

Read-only Upstox requests through the existing ``UpstoxHistoricalClient`` (bounded exponential backoff, Retry-After,
throttling; token read from the environment and only ever sent in the Authorization header):
  underlying     GET /v3/historical-candle/{key}/minutes/5/{to}/{from}            (monthly chunks)
  live options   GET /v3/historical-candle/{key}/minutes/5/{to}/{from}
  expired        GET /v2/expired-instruments/expiries | /option/contract | /historical-candle/{expired_key}/5minute/...

Option selection mirrors the research/S2 rules (``option_coverage.resolve_signal_contract``): nearest listed expiry
on the session date (``select_expiry``), BUY -> CE / SELL -> PE, ATM = listed strike nearest the signal close
(``atm_strike``). A signal can fire on any completed candle, so for every session we keep every strike that was ATM
for some candle close of that session, plus the nearest ITM and nearest OTM listed strike beyond that range, for
both CE and PE. Expiries and strikes come only from the official listings (expired-instrument API for past expiries,
instrument master for live ones); today's contracts are never used for past sessions.

Nothing is interpolated, estimated or synthesised. Missing candles, unavailable contracts and unresolved sessions
are recorded explicitly.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.options.selection import atm_strike, select_expiry

from .contracts import ACTIVE, EXPIRED, HistoricalContract
from .downloader import FATAL_CODES, month_chunks
from .models import HistoricalCandle, HistoricalDataError, candles_to_frame
from .validation import expected_slots, validate_candles

IST = ZoneInfo("Asia/Kolkata")
BANNER = "PAPER | DATA ACQUISITION ONLY | NO ORDERS"
DEFAULT_ROOT = Path("data/historical/upstox_6m")
DEFAULT_FROM, DEFAULT_TO = date(2026, 4, 1), date(2026, 9, 30)
INTERVAL = 5
SLOTS_PER_SESSION = 75
SUCCESS, EMPTY, FAILED, UNAVAILABLE = "SUCCESS", "EMPTY", "FAILED", "UNAVAILABLE"
STATUSES = (SUCCESS, EMPTY, FAILED, UNAVAILABLE)
UNAVAILABLE_CODES = frozenset({"NOT_FOUND", "PLAN_NOT_SUPPORTED", "PERMISSION_DENIED"})
ATM, ITM, OTM = "ATM", "ITM", "OTM"

# Coverage / quality thresholds (fixed, documented in the reports).
MAX_UNDERLYING_MISSING_SHARE = 0.001   # missing 5m intervals / expected intervals on NIFTY trading days
MIN_REQUIREMENT_COVERAGE = 0.95        # (session, contract) requirements with >= 1 real candle that session
MIN_ATM_COVERAGE = 0.98                # same, ATM requirements only
MIN_ATM_INTERVAL_COMPLETENESS = 0.90   # ATM candles present / 75 per required session
MAX_RAW_REJECT_SHARE = 0.001           # rows Upstox returned that failed OHLC/price checks (never stored)
INSUFFICIENT_TEXT = "Underlying data available but option-premium coverage insufficient."

MANIFEST_COLUMNS = ["instrument_key", "instrument_type", "expiry", "strike", "option_type", "month", "from_date",
                    "to_date", "candle_count", "status", "http_status", "download_timestamp", "file_path", "source",
                    "trading_symbol", "error_code", "raw_rows", "duplicates_removed", "rejected_rows", "detail"]
CONTRACT_COLUMNS = ["trading_date", "underlying", "instrument_key", "trading_symbol", "expiry", "strike",
                    "option_type", "role", "dte", "source", "lot_size"]
MISSING_COLUMNS = ["scope", "instrument_key", "trading_symbol", "trading_date", "month", "reason", "count", "detail"]


# ------------------------------------------------------------------ layout / io
def month_label(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def month_windows(start: date, end: date) -> list[tuple[str, date, date]]:
    """Calendar-month request windows (Upstox serves 1-15 minute candles at most one month per request)."""
    return [(month_label(a), a, b) for a, b in month_chunks(start, end)]


def _strike_tag(strike: float) -> str:
    return f"{float(strike):g}"


class Layout:
    def __init__(self, root: str | Path = DEFAULT_ROOT, underlying: str = "NIFTY") -> None:
        self.root = Path(root)
        self.underlying = underlying.upper()
        self.metadata = self.root / "metadata"
        self.manifest = self.metadata / "download_manifest.csv"
        self.contracts = self.metadata / "contracts.csv"
        self.missing = self.metadata / "missing_data.csv"
        self.plan = self.metadata / "plan.json"
        self.summary = self.metadata / "summary.json"

    def underlying_file(self, month: str) -> Path:
        return self.root / "underlying" / self.underlying.lower() / f"{month}.parquet"

    def option_file(self, month: str, c: HistoricalContract) -> Path:
        name = f"{c.underlying}_{c.expiry.isoformat()}_{_strike_tag(c.strike)}_{c.instrument_type}.parquet"
        return self.root / "options" / month / name

    def underlying_files(self) -> list[Path]:
        return sorted((self.root / "underlying" / self.underlying.lower()).glob("*.parquet"))

    def option_files(self) -> list[Path]:
        return sorted((self.root / "options").glob("*/*.parquet"))


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if "expiry" in out:
        out["expiry"] = out["expiry"].astype(str)
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not frame.empty:
        ts = pd.to_datetime(frame["timestamp"])
        frame["timestamp"] = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts
    return frame


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    tmp.replace(path)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ quality
def quality_check(frame: pd.DataFrame, *, interval: int = INTERVAL, expiry: date | None = None,
                  option_type: str | None = None, start: date | None = None, end: date | None = None) -> dict:
    """Timestamp / timezone / ordering / duplicate / OHLC / volume / OI / spacing checks. Never modifies data."""
    q: dict[str, Any] = {"rows": int(len(frame))}
    if frame.empty:
        q.update(tz_ist=True, sorted=True, duplicate_timestamps=0, duplicate_rows=0, on_grid=True, rejected_rows=0,
                 invalid_ohlc=0, non_positive_price=0, missing_price=0, negative_volume=0, negative_oi=0,
                 outside_session=0, after_expiry=0, out_of_range=0, trading_days=0, expected_candles=0,
                 missing_intervals=0, missing_by_day={}, spacing_gaps=0, ok=True)
        return q
    ts = pd.to_datetime(frame["timestamp"])
    tz_ist = ts.dt.tz is not None and all(v.utcoffset() == timedelta(hours=5, minutes=30) for v in ts)
    local = ts.dt.tz_convert(IST) if ts.dt.tz is not None else ts.dt.tz_localize(IST)
    clean, rep = validate_candles(frame, interval=interval, expiry=expiry, option_type=option_type)
    days = local.dt.date
    out_of_range = int(((days < start) if start else pd.Series(False, index=frame.index)).sum()
                       + ((days > end) if end else pd.Series(False, index=frame.index)).sum())
    gaps = 0
    if not clean.empty:
        c = clean["timestamp"]
        d = c.groupby(c.dt.date).diff().dropna()
        gaps = int((d != pd.Timedelta(minutes=interval)).sum())
    q.update(
        tz_ist=bool(tz_ist), sorted=bool(local.is_monotonic_increasing),
        duplicate_timestamps=int(local.duplicated().sum()),
        duplicate_rows=int(frame.astype(str).duplicated().sum()),
        on_grid=bool(((local.dt.minute % interval == 0) & (local.dt.second == 0)).all()),
        rejected_rows=rep.rejected_rows, invalid_ohlc=rep.issues.get("INVALID_OHLC", 0),
        non_positive_price=rep.issues.get("NON_POSITIVE_PRICE", 0), missing_price=rep.issues.get("MISSING_PRICE", 0),
        negative_volume=rep.issues.get("NEGATIVE_VOLUME", 0), negative_oi=rep.issues.get("NEGATIVE_OPEN_INTEREST", 0),
        outside_session=rep.outside_session_rows, after_expiry=rep.issues.get("AFTER_EXPIRY", 0),
        out_of_range=out_of_range, trading_days=rep.trading_days, expected_candles=rep.expected_candles,
        missing_intervals=rep.missing_candles, missing_by_day=dict(rep.missing_by_day), spacing_gaps=gaps)
    q["ok"] = bool(q["tz_ist"] and q["sorted"] and q["duplicate_timestamps"] == 0 and q["duplicate_rows"] == 0
                   and q["on_grid"] and q["rejected_rows"] == 0 and q["outside_session"] == 0
                   and q["out_of_range"] == 0)
    return q


def prepare_candles(candles: Sequence[HistoricalCandle], start: date, end: date, *, extra: dict | None = None,
                    expiry: date | None = None, option_type: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Raw Upstox candles -> clean frame to store (sorted, de-duplicated, invalid rows excluded and counted)."""
    frame = candles_to_frame(candles, **(extra or {}))
    raw_rows = len(frame)
    out_of_window = 0
    if not frame.empty:
        d = frame["timestamp"].dt.date
        keep = (d >= start) & (d <= end)
        out_of_window = int((~keep).sum())
        frame = frame[keep].reset_index(drop=True)
    clean, rep = validate_candles(frame, interval=INTERVAL, expiry=expiry, option_type=option_type)
    return clean, {"raw_rows": raw_rows, "out_of_window": out_of_window, "duplicates_removed": rep.duplicates_removed,
                   "rejected_rows": rep.rejected_rows, "outside_session": rep.outside_session_rows,
                   "issues": dict(rep.issues)}


# ------------------------------------------------------------------ manifest
class Manifest:
    """One row per (instrument_key, month); rewritten atomically after every item so a run can resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: dict[tuple[str, str], dict] = {(r["instrument_key"], r["month"]): r for r in _read_csv(path)}

    def get(self, instrument_key: str, month: str) -> dict | None:
        return self._rows.get((instrument_key, month))

    def upsert(self, row: dict) -> None:
        self._rows[(row["instrument_key"], row["month"])] = {k: row.get(k, "") for k in MANIFEST_COLUMNS}
        self.save()

    def save(self) -> None:
        _write_csv(self.path, MANIFEST_COLUMNS, sorted(self._rows.values(),
                                                       key=lambda r: (r["month"], r["instrument_type"], r["expiry"],
                                                                      str(r["strike"]), r["instrument_key"])))

    def rows(self) -> list[dict]:
        return list(self._rows.values())


@dataclass(frozen=True)
class DownloadItem:
    instrument_key: str
    instrument_type: str        # INDEX | CE | PE
    source: str                 # ACTIVE (V3) | EXPIRED (expired-instrument endpoint)
    month: str
    start: date
    end: date
    path: Path
    underlying: str
    expiry: date | None = None
    strike: float | None = None
    trading_symbol: str = ""

    @property
    def option_type(self) -> str | None:
        return self.instrument_type if self.instrument_type in ("CE", "PE") else None


def _row(item: DownloadItem, status: str, *, candle_count: int = 0, http_status: Any = "", error_code: str = "",
         detail: str = "", raw: dict | None = None, now: datetime | None = None) -> dict:
    raw = raw or {}
    return {"instrument_key": item.instrument_key, "instrument_type": item.instrument_type,
            "expiry": item.expiry.isoformat() if item.expiry else "", "strike": "" if item.strike is None else item.strike,
            "option_type": item.option_type or "", "month": item.month, "from_date": item.start.isoformat(),
            "to_date": item.end.isoformat(), "candle_count": candle_count, "status": status,
            "http_status": http_status, "download_timestamp": (now or datetime.now(IST)).isoformat(timespec="seconds"),
            "file_path": item.path.as_posix(), "source": item.source, "trading_symbol": item.trading_symbol,
            "error_code": error_code, "raw_rows": raw.get("raw_rows", ""),
            "duplicates_removed": raw.get("duplicates_removed", ""), "rejected_rows": raw.get("rejected_rows", ""),
            "detail": detail[:300]}


def stored_valid(item: DownloadItem) -> tuple[bool, dict | None]:
    if not item.path.is_file():
        return False, None
    try:
        frame = read_parquet(item.path)
    except Exception:  # unreadable / partial file -> re-download
        return False, None
    if frame.empty:
        return False, None
    q = quality_check(frame, expiry=item.expiry, option_type=item.option_type,
                      start=date.fromisoformat(item.month + "-01"), end=_month_end(item.month))
    return q["ok"], q


def _month_end(month: str) -> date:
    first = date.fromisoformat(month + "-01")
    return month_chunks(first, first + timedelta(days=31))[0][1]


def _covers(prev: dict | None, item: DownloadItem) -> bool:
    return prev is not None and prev.get("from_date", "9") <= item.start.isoformat() \
        and prev.get("to_date", "0") >= item.end.isoformat()


def fetch_item(client, item: DownloadItem, manifest: Manifest, *, resume: bool = False,
               now: Callable[[], datetime] = lambda: datetime.now(IST)) -> tuple[dict, str]:
    """Download one (instrument, month) window unless a valid stored file already covers it. Returns (row, action)."""
    prev = manifest.get(item.instrument_key, item.month)
    valid, q = stored_valid(item)
    if valid and (prev is None or (prev.get("status") == SUCCESS and _covers(prev, item))):
        if prev is None:
            prev = _row(item, SUCCESS, candle_count=q["rows"], detail="existing valid file (manifest rebuilt)",
                        now=now())
            manifest.upsert(prev)
        return prev, "SKIPPED_VALID"
    if resume and prev is not None and prev.get("status") in (EMPTY, UNAVAILABLE) and _covers(prev, item):
        return prev, "SKIPPED_TERMINAL"
    try:
        if item.source == EXPIRED:
            candles = client.expired_candles(item.instrument_key, item.start, item.end, interval=INTERVAL)
        else:
            candles = client.candles(item.instrument_key, item.start, item.end, interval=INTERVAL)
    except HistoricalDataError as exc:
        status = UNAVAILABLE if exc.code in UNAVAILABLE_CODES else FAILED
        row = _row(item, status, http_status=exc.status or "", error_code=exc.code, detail=exc.message, now=now())
        manifest.upsert(row)
        if exc.code in FATAL_CODES:
            raise
        return row, status
    extra = {}
    if item.option_type:
        extra = {"underlying": item.underlying, "expiry": item.expiry.isoformat(), "strike": item.strike,
                 "option_type": item.option_type}
    clean, raw = prepare_candles(candles, item.start, item.end, extra=extra, expiry=item.expiry,
                                 option_type=item.option_type)
    if clean.empty:
        row = _row(item, EMPTY, http_status=200, raw=raw, detail="Upstox returned no valid candles in the window",
                   now=now())
        manifest.upsert(row)
        return row, EMPTY
    if valid:
        old = read_parquet(item.path)
        clean = (pd.concat([old, clean], ignore_index=True).drop_duplicates("timestamp", keep="last")
                 .sort_values("timestamp").reset_index(drop=True))
    write_parquet(item.path, clean)
    row = _row(item, SUCCESS, candle_count=len(clean), http_status=200, raw=raw, now=now(),
               detail=f"issues {raw['issues']}" if raw["issues"] else "")
    manifest.upsert(row)
    return row, SUCCESS


# ------------------------------------------------------------------ contract discovery / point-in-time plan
@dataclass
class Discovery:
    calendar: list[date]
    chains: dict[tuple[date, str], list[HistoricalContract]]
    calendar_from: date
    issues: list[dict] = field(default_factory=list)


def discover_contracts(catalog, underlying: str, start: date, end: date) -> Discovery:
    """Official expiry calendar + CE/PE chains: expired-instrument API for past expiries, instrument master for live."""
    name = underlying.upper()
    issues: list[dict] = []
    today = catalog.today()
    active = [c for c in catalog.active_contracts() if c.underlying == name and c.is_option and c.expiry >= start]
    try:
        expired_exp = [e for e in catalog.expired_expiries(name) if e >= start]
        expired_ok = True
    except HistoricalDataError as exc:
        issues.append({"scope": "EXPIRED_EXPIRIES", **exc.as_dict()})
        expired_exp, expired_ok = [], False
    wanted, after = [], False
    for e in sorted(expired_exp):
        if e <= end:
            wanted.append(e)
        elif not after:
            wanted.append(e)
            after = True
    chains: dict[tuple[date, str], list[HistoricalContract]] = defaultdict(list)
    for e in wanted:
        try:
            for c in catalog.expired_option_contracts(name, e):
                chains[(c.expiry, c.instrument_type)].append(c)
        except HistoricalDataError as exc:
            issues.append({"scope": "EXPIRED_CONTRACTS", "expiry": e.isoformat(), **exc.as_dict()})
    for c in active:
        if not any(x.instrument_key == c.instrument_key for x in chains.get((c.expiry, c.instrument_type), [])):
            chains[(c.expiry, c.instrument_type)].append(c)
    calendar = sorted(set(wanted) | {c.expiry for c in active})
    calendar_from = start if expired_ok else max(start, today)
    return Discovery(calendar=calendar, chains=dict(chains), calendar_from=calendar_from, issues=issues)


@dataclass(frozen=True)
class Requirement:
    trading_date: date
    contract: HistoricalContract
    role: str
    dte: int


def plan_requirements(spot: pd.DataFrame, disc: Discovery, *, expiry_mode: str = "nearest",
                      offset: int = 1) -> tuple[list[Requirement], list[dict]]:
    """Per session: nearest listed expiry; every strike ATM for some close that session, +/- ``offset`` listed strikes."""
    reqs: list[Requirement] = []
    unresolved: list[dict] = []
    if spot.empty:
        return reqs, unresolved
    ts = pd.to_datetime(spot["timestamp"])
    rng = spot.groupby(ts.dt.tz_convert(IST).dt.date)["close"].agg(["min", "max"])
    for d, (lo, hi) in rng.iterrows():
        if d < disc.calendar_from:
            unresolved.append({"trading_date": d, "reason": "EXPIRY_CALENDAR_INCOMPLETE",
                               "detail": "expired-instrument expiries unavailable; past expiry cannot be resolved"})
            continue
        future = [e for e in disc.calendar if e >= d]
        if not future:
            unresolved.append({"trading_date": d, "reason": "NO_LISTED_EXPIRY", "detail": ""})
            continue
        expiry = select_expiry(future, expiry_mode)
        for kind in ("CE", "PE"):
            chain = disc.chains.get((expiry, kind)) or []
            if not chain:
                unresolved.append({"trading_date": d, "reason": "CONTRACT_LIST_UNAVAILABLE",
                                   "detail": f"{kind} chain for expiry {expiry} not available"})
                continue
            by_strike = {float(c.strike): c for c in chain}
            strikes = sorted(by_strike)
            a = strikes.index(atm_strike(strikes, float(lo)))
            b = strikes.index(atm_strike(strikes, float(hi)))
            i0, i1 = a - offset, b + offset
            if i0 < 0 or i1 >= len(strikes):
                unresolved.append({"trading_date": d, "reason": "STRIKE_OUT_OF_RANGE",
                                   "detail": f"{kind} {expiry}: ITM/OTM neighbour beyond listed strikes"})
            for i in range(max(0, i0), min(len(strikes) - 1, i1) + 1):
                if a <= i <= b:
                    role = ATM
                else:
                    below = i < a
                    role = (ITM if below else OTM) if kind == "CE" else (OTM if below else ITM)
                reqs.append(Requirement(d, by_strike[strikes[i]], role, (expiry - d).days))
    return reqs, unresolved


def option_items(reqs: Sequence[Requirement], layout: Layout) -> list[DownloadItem]:
    by_key: dict[str, tuple[HistoricalContract, set[date]]] = {}
    for r in reqs:
        by_key.setdefault(r.contract.instrument_key, (r.contract, set()))[1].add(r.trading_date)
    items = []
    for key, (c, days) in by_key.items():
        months: dict[str, list[date]] = defaultdict(list)
        for d in days:
            months[month_label(d)].append(d)
        for m, ds in sorted(months.items()):
            items.append(DownloadItem(instrument_key=key, instrument_type=c.instrument_type, source=c.source, month=m,
                                      start=min(ds), end=max(ds), path=layout.option_file(m, c),
                                      underlying=c.underlying, expiry=c.expiry, strike=float(c.strike),
                                      trading_symbol=c.trading_symbol))
    return sorted(items, key=lambda i: (i.month, i.expiry, i.strike, i.instrument_type))


def underlying_items(key: str, start: date, end: date, layout: Layout) -> list[DownloadItem]:
    return [DownloadItem(instrument_key=key, instrument_type="INDEX", source=ACTIVE, month=m, start=a, end=b,
                         path=layout.underlying_file(m), underlying=layout.underlying)
            for m, a, b in month_windows(start, end)]


def requirement_rows(reqs: Sequence[Requirement]) -> list[dict]:
    return [{"trading_date": r.trading_date.isoformat(), "underlying": r.contract.underlying,
             "instrument_key": r.contract.instrument_key, "trading_symbol": r.contract.trading_symbol,
             "expiry": r.contract.expiry.isoformat(), "strike": float(r.contract.strike),
             "option_type": r.contract.instrument_type, "role": r.role, "dte": r.dte, "source": r.contract.source,
             "lot_size": r.contract.lot_size} for r in reqs]


def requirements_from_rows(rows: Iterable[dict]) -> list[Requirement]:
    out = []
    for r in rows:
        c = HistoricalContract(instrument_key=r["instrument_key"], underlying=r["underlying"],
                               expiry=date.fromisoformat(r["expiry"]), instrument_type=r["option_type"],
                               lot_size=int(float(r["lot_size"] or 0)), source=r["source"],
                               strike=float(r["strike"]), trading_symbol=r["trading_symbol"])
        out.append(Requirement(date.fromisoformat(r["trading_date"]), c, r["role"], int(r["dte"])))
    return out


# ------------------------------------------------------------------ coverage / quality summaries
def load_underlying(layout: Layout, start: date, end: date) -> pd.DataFrame:
    parts = [read_parquet(p) for p in layout.underlying_files()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True)
    d = f["timestamp"].dt.date
    return (f[(d >= start) & (d <= end)].drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp").reset_index(drop=True))


def _weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def summarize(layout: Layout, manifest: Manifest, reqs: Sequence[Requirement], unresolved: Sequence[dict],
              start: date, end: date, discovery_issues: Sequence[dict] = ()) -> dict[str, Any]:
    rows = manifest.rows()
    u_rows = [r for r in rows if r["instrument_type"] == "INDEX" and start.isoformat()[:7] <= r["month"] <= end.isoformat()[:7]]
    spot = load_underlying(layout, start, end)
    uq = quality_check(spot, start=start, end=end)
    present = sorted(set(spot["timestamp"].dt.date)) if not spot.empty else []
    no_data = [d for d in _weekdays(start, end) if d not in set(present)]
    u_months = []
    for m, a, b in month_windows(start, end):
        sub = spot[spot["timestamp"].dt.date.map(lambda x: a <= x <= b)] if not spot.empty else spot
        mq = quality_check(sub)
        mr = next((r for r in u_rows if r["month"] == m), None)
        u_months.append({"month": m, "candles": len(sub), "trading_days": mq["trading_days"],
                         "missing_intervals": mq["missing_intervals"], "status": mr["status"] if mr else "NOT_RUN"})
    exp = uq["expected_candles"]
    u_missing_share = uq["missing_intervals"] / exp if exp else None
    u_ok = (bool(present) and all(m["status"] == SUCCESS for m in u_months) and uq["ok"]
            and u_missing_share is not None and u_missing_share <= MAX_UNDERLYING_MISSING_SHARE)

    # option files: per-contract per-session candle counts + stored-file quality
    counts: dict[str, Counter] = defaultdict(Counter)
    file_quality, failing = [], []
    option_candles = 0
    req_meta = {r.contract.instrument_key: r.contract for r in reqs}
    for p in layout.option_files():
        try:
            f = read_parquet(p)
        except Exception as exc:
            failing.append({"file": p.as_posix(), "reason": f"unreadable: {type(exc).__name__}"})
            continue
        if f.empty:
            continue
        key = str(f["instrument_key"].iloc[0])
        c = req_meta.get(key)
        month = p.parent.name
        q = quality_check(f, expiry=c.expiry if c else None, option_type=c.instrument_type if c else None,
                          start=date.fromisoformat(month + "-01"), end=_month_end(month))
        file_quality.append(q)
        if not q["ok"]:
            failing.append({"file": p.as_posix(), "reason": ", ".join(k for k in ("tz_ist", "sorted", "on_grid")
                                                                       if not q[k]) or "rows rejected/duplicated"})
        option_candles += len(f)
        for d, n in f["timestamp"].dt.date.value_counts().items():
            counts[key][d] += int(n)

    item_status = {(r["instrument_key"], r["month"]): r for r in rows if r["instrument_type"] in ("CE", "PE")}
    missing_rows: list[dict] = []
    cov_month: dict[str, Counter] = defaultdict(Counter)
    atm_sessions: dict[date, bool] = {}
    atm_slots = atm_total = atm_cov = all_cov = 0
    for r in reqs:
        n = counts.get(r.contract.instrument_key, {}).get(r.trading_date, 0)
        m = month_label(r.trading_date)
        cov_month[m]["reqs"] += 1
        cov_month[m]["covered"] += n > 0
        all_cov += n > 0
        if r.role == ATM:
            atm_total += 1
            atm_cov += n > 0
            atm_slots += min(n, SLOTS_PER_SESSION)
            cov_month[m]["atm"] += 1
            cov_month[m]["atm_covered"] += n > 0
            cov_month[m]["atm_slots"] += min(n, SLOTS_PER_SESSION)
            atm_sessions[r.trading_date] = atm_sessions.get(r.trading_date, True) and n > 0
        if n == 0:
            st = item_status.get((r.contract.instrument_key, m))
            reason = (f"{st['status']}:{st['error_code']}" if st and st["status"] != SUCCESS
                      else "NO_CANDLES_ON_SESSION" if st else "NOT_DOWNLOADED")
            missing_rows.append({"scope": f"OPTION_{r.role}", "instrument_key": r.contract.instrument_key,
                                 "trading_symbol": r.contract.trading_symbol, "trading_date": r.trading_date.isoformat(),
                                 "month": m, "reason": reason, "count": SLOTS_PER_SESSION, "detail": ""})
        elif r.role == ATM and n < SLOTS_PER_SESSION:
            missing_rows.append({"scope": "OPTION_ATM_INTERVALS", "instrument_key": r.contract.instrument_key,
                                 "trading_symbol": r.contract.trading_symbol, "trading_date": r.trading_date.isoformat(),
                                 "month": m, "reason": "MISSING_INTERVALS", "count": SLOTS_PER_SESSION - n,
                                 "detail": "Upstox returned no candle for these 5m slots (no fill applied)"})
    for d, n in sorted(uq["missing_by_day"].items()):
        missing_rows.append({"scope": "UNDERLYING", "instrument_key": u_rows[0]["instrument_key"] if u_rows else "",
                             "trading_date": d, "month": d[:7], "reason": "MISSING_INTERVALS", "count": n})
    for d in no_data:
        missing_rows.append({"scope": "UNDERLYING", "trading_date": d.isoformat(), "month": month_label(d),
                             "reason": "WEEKDAY_WITHOUT_CANDLES",
                             "detail": "Upstox returned no candles (exchange holiday or unavailable); not filled"})
    for u in unresolved:
        missing_rows.append({"scope": "PLAN", "trading_date": str(u["trading_date"]),
                             "month": str(u["trading_date"])[:7], "reason": u["reason"], "detail": u.get("detail", "")})
    for i in discovery_issues:
        missing_rows.append({"scope": "DISCOVERY", "reason": i.get("code"), "detail": f"{i.get('scope')} "
                             f"{i.get('expiry', '')} {i.get('message', '')}".strip()})

    contracts = {r.contract.instrument_key: r.contract for r in reqs}
    contract_status: dict[str, set[str]] = defaultdict(set)
    for (k, _), r in item_status.items():
        contract_status[k].add(r["status"])
    missing_contracts = sorted(k for k in contracts if not counts.get(k))
    sessions = sorted({r.trading_date for r in reqs})
    req_cov = all_cov / len(reqs) if reqs else None
    atm_c = atm_cov / atm_total if atm_total else None
    atm_int = atm_slots / (atm_total * SLOTS_PER_SESSION) if atm_total else None
    opt_months = []
    for m, _, _ in month_windows(start, end):
        c = cov_month.get(m, Counter())
        mk = {k for k, mm in item_status if mm == m}
        opt_months.append({"month": m, "contracts": len(mk),
                           "candles": sum(int(item_status[(k, m)]["candle_count"] or 0) for k in mk),
                           "requirements": c["reqs"], "coverage": c["covered"] / c["reqs"] if c["reqs"] else None,
                           "atm_coverage": c["atm_covered"] / c["atm"] if c["atm"] else None,
                           "atm_interval_completeness": c["atm_slots"] / (c["atm"] * SLOTS_PER_SESSION) if c["atm"] else None})
    sessions_unresolved = sorted({str(u["trading_date"]) for u in unresolved})
    option_sufficient = bool(
        reqs and not sessions_unresolved and req_cov is not None and req_cov >= MIN_REQUIREMENT_COVERAGE
        and atm_c is not None and atm_c >= MIN_ATM_COVERAGE and atm_int is not None
        and atm_int >= MIN_ATM_INTERVAL_COMPLETENESS and set(sessions) >= set(present))

    raw_rows = sum(int(r["raw_rows"] or 0) for r in rows)
    raw_rej = sum(int(r["rejected_rows"] or 0) for r in rows)
    dups = sum(int(r["duplicates_removed"] or 0) for r in rows)
    agg = {k: sum(q[k] for q in [uq, *file_quality]) for k in
           ("duplicate_timestamps", "duplicate_rows", "invalid_ohlc", "negative_volume", "negative_oi",
            "outside_session", "after_expiry", "out_of_range", "rejected_rows", "spacing_gaps")}
    agg.update(tz_ist=all(q["tz_ist"] for q in [uq, *file_quality]),
               sorted=all(q["sorted"] for q in [uq, *file_quality]),
               on_grid=all(q["on_grid"] for q in [uq, *file_quality]))
    raw_share = raw_rej / raw_rows if raw_rows else 0.0
    quality_pass = bool(uq["ok"] and not failing and raw_share <= MAX_RAW_REJECT_SHARE)
    status_counts = Counter(r["status"] for r in rows)
    return {
        "banner": BANNER, "period": {"from": start.isoformat(), "to": end.isoformat()},
        "underlying": {"instrument_key": u_rows[0]["instrument_key"] if u_rows else None, "candles": len(spot),
                       "trading_days": len(present), "weekdays_without_candles": [d.isoformat() for d in no_data],
                       "missing_intervals": uq["missing_intervals"], "expected_intervals": exp,
                       "missing_share": u_missing_share, "missing_by_day": uq["missing_by_day"], "months": u_months,
                       "quality": {k: v for k, v in uq.items() if k != "missing_by_day"},
                       "status": "OK" if u_ok else "INCOMPLETE"},
        "options": {"contracts": len(contracts), "ce": sum(c.instrument_type == "CE" for c in contracts.values()),
                    "pe": sum(c.instrument_type == "PE" for c in contracts.values()),
                    "expiries": len({c.expiry for c in contracts.values()}),
                    "expired_source": sum(c.source == EXPIRED for c in contracts.values()),
                    "active_source": sum(c.source == ACTIVE for c in contracts.values()),
                    "candles": option_candles, "requirements": len(reqs), "sessions_planned": len(sessions),
                    "sessions_unresolved": sessions_unresolved, "missing_contracts": missing_contracts,
                    "missing_dates": len([m for m in missing_rows if m["scope"].startswith("OPTION_")
                                          and m["reason"] != "MISSING_INTERVALS"]),
                    "requirement_coverage": req_cov, "atm_coverage": atm_c, "atm_interval_completeness": atm_int,
                    "atm_full_sessions": sum(atm_sessions.values()), "months": opt_months,
                    "roles": dict(Counter(r.role for r in reqs)),
                    "status": "OK" if option_sufficient else "INCOMPLETE"},
        "quality": {"pass": quality_pass, "aggregate": agg, "failing_files": failing, "raw_rows": raw_rows,
                    "raw_rejected_rows": raw_rej, "raw_reject_share": raw_share, "duplicates_removed": dups,
                    "files_checked": 1 + len(file_quality)},
        "manifest_status": dict(status_counts),
        "option_coverage": "SUFFICIENT" if option_sufficient else "INSUFFICIENT",
        "discovery_issues": list(discovery_issues),
        "_missing_rows": missing_rows,
    }


# ------------------------------------------------------------------ reports
def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def quality_markdown(s: dict) -> str:
    q, a, u = s["quality"], s["quality"]["aggregate"], s["underlying"]["quality"]
    md = ["# Upstox 6-month data quality", "", f"**{BANNER}** | period {s['period']['from']} -> {s['period']['to']}",
          "", f"Overall: **{_pf(q['pass'])}** ({q['files_checked']} stored files checked: NIFTY + option files)", "",
          "| Check | Result |", "|---|---|",
          f"| Timestamps tz-aware IST (+05:30) | {_pf(a['tz_ist'])} |",
          f"| Timestamps sorted | {_pf(a['sorted'])} |",
          f"| On the 5-minute grid | {_pf(a['on_grid'])} |",
          f"| Duplicate candles (timestamp) stored | {a['duplicate_timestamps']} |",
          f"| Duplicate rows stored | {a['duplicate_rows']} |",
          f"| Invalid OHLC (high < max(open, close) or low > min(open, close)) stored | {a['invalid_ohlc']} |",
          f"| Negative volume stored | {a['negative_volume']} |",
          f"| Negative OI stored (when available) | {a['negative_oi']} |",
          f"| Candles outside 09:15-15:30 stored | {a['outside_session']} |",
          f"| Option candles after expiry stored | {a['after_expiry']} |",
          f"| Candles outside their month file | {a['out_of_range']} |",
          f"| Intra-day spacing gaps (5m spacing broken; equals missing-interval runs) | {a['spacing_gaps']} |",
          f"| NIFTY missing 5m intervals on trading days | {u['missing_intervals']} of {u['expected_candles']} |",
          "", "## Raw Upstox responses (before storage)", "",
          f"- Rows returned: {q['raw_rows']}",
          f"- Duplicate timestamps removed: {q['duplicates_removed']}",
          f"- Rows rejected by OHLC / price / volume checks (never stored, never repaired): {q['raw_rejected_rows']} "
          f"({_pct(q['raw_reject_share'])}; limit {_pct(MAX_RAW_REJECT_SHARE)})",
          "", "## Files failing checks", ""]
    md += [f"- `{f['file']}`: {f['reason']}" for f in q["failing_files"][:100]] or ["- None"]
    md += ["", "Missing candles are reported, never filled. See `metadata/missing_data.csv`."]
    return "\n".join(md) + "\n"


def coverage_markdown(s: dict) -> str:
    u, o = s["underlying"], s["options"]
    md = ["# Upstox 6-month coverage", "", f"**{BANNER}** | period {s['period']['from']} -> {s['period']['to']}", "",
          f"- NIFTY underlying coverage: **{u['status']}**",
          f"- Option premium coverage: **{o['status']}** ({s['option_coverage']})", ""]
    if u["status"] == "OK" and o["status"] != "OK":
        md += [f"**{INSUFFICIENT_TEXT}**", ""]
    md += ["## NIFTY", "", "| Metric | Value |", "|---|---|",
           f"| Instrument key | `{u['instrument_key']}` |", f"| Total candles | {u['candles']} |",
           f"| Trading days | {u['trading_days']} |",
           f"| Weekdays without candles (holiday or unavailable) | {len(u['weekdays_without_candles'])}: "
           f"{', '.join(u['weekdays_without_candles']) or '-'} |",
           f"| Missing intervals | {u['missing_intervals']} ({_pct(u['missing_share'])}; limit "
           f"{_pct(MAX_UNDERLYING_MISSING_SHARE)}) |", "",
           "## Options", "", "| Metric | Value |", "|---|---|",
           f"| Total contracts | {o['contracts']} (expired source {o['expired_source']}, live {o['active_source']}) |",
           f"| CE contracts | {o['ce']} |", f"| PE contracts | {o['pe']} |", f"| Expiries | {o['expiries']} |",
           f"| Total option candles | {o['candles']} |",
           f"| (session, contract) requirements | {o['requirements']} (roles {o['roles']}) |",
           f"| Requirement coverage | {_pct(o['requirement_coverage'])} (min {_pct(MIN_REQUIREMENT_COVERAGE)}) |",
           f"| ATM coverage | {_pct(o['atm_coverage'])} (min {_pct(MIN_ATM_COVERAGE)}) |",
           f"| ATM 5m interval completeness | {_pct(o['atm_interval_completeness'])} "
           f"(min {_pct(MIN_ATM_INTERVAL_COMPLETENESS)}) |",
           f"| Sessions with every ATM contract present | {o['atm_full_sessions']} / {o['sessions_planned']} |",
           f"| Missing contracts (no candles at all) | {len(o['missing_contracts'])} |",
           f"| Missing (session, contract) dates | {o['missing_dates']} |",
           f"| Unresolved sessions | {len(o['sessions_unresolved'])}: {', '.join(o['sessions_unresolved'][:20]) or '-'} |",
           "", "## Month by month", "",
           "| Month | NIFTY candles | NIFTY days | NIFTY missing intervals | NIFTY status | Option contracts | "
           "Option candles | Requirement coverage | ATM coverage | ATM interval completeness |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for um, om in zip(u["months"], o["months"]):
        md.append(f"| {um['month']} | {um['candles']} | {um['trading_days']} | {um['missing_intervals']} | "
                  f"{um['status']} | {om['contracts']} | {om['candles']} | {_pct(om['coverage'])} | "
                  f"{_pct(om['atm_coverage'])} | {_pct(om['atm_interval_completeness'])} |")
    md += ["", "## Selection rule", "",
           "Per NIFTY session: nearest listed expiry on that date (official expiry calendar); for CE and PE every "
           "listed strike that was ATM for some 5m close of the session (S2 rule: listed strike nearest the signal "
           "close, ties to the lower strike) plus the nearest ITM and nearest OTM listed strike beyond that range. "
           "Past expiries/strikes come from the Upstox expired-instrument API; today's contracts are never used for "
           "past sessions. Strike listings are the exchange's final listing for that expiry (Upstox does not "
           "publish intraday listing dates).", "",
           f"Manifest status counts: {s['manifest_status']}"]
    if s["discovery_issues"]:
        md += ["", "## Discovery issues", ""]
        md += [f"- {i.get('scope')} {i.get('expiry', '')}: {i.get('code')} {i.get('message', '')}"
               for i in s["discovery_issues"]]
    return "\n".join(md) + "\n"


def write_outputs(layout: Layout, s: dict, reqs: Sequence[Requirement], report_dir: Path) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {"contracts": layout.contracts, "missing": layout.missing, "summary": layout.summary,
             "manifest": layout.manifest, "quality": report_dir / "upstox_6m_data_quality.md",
             "coverage": report_dir / "upstox_6m_coverage.md"}
    _write_csv(layout.contracts, CONTRACT_COLUMNS, requirement_rows(reqs))
    _write_csv(layout.missing, MISSING_COLUMNS, s["_missing_rows"])
    layout.summary.write_text(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}, indent=2,
                                         default=str), encoding="utf-8")
    paths["quality"].write_text(quality_markdown(s), encoding="utf-8")
    paths["coverage"].write_text(coverage_markdown(s), encoding="utf-8")
    return paths


def token_leak_check(paths: Iterable[Path], secrets: Iterable[str]) -> tuple[bool, list[str]]:
    """True when no secret value appears in any file (byte scan). Only file names are returned, never values."""
    needles = [s.encode() for s in secrets if s and len(s) >= 8]
    leaks = []
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else (sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [])
        for f in files:
            data = f.read_bytes()
            if any(n in data for n in needles):
                leaks.append(f.as_posix())
    return not leaks, leaks


# ------------------------------------------------------------------ orchestration
def save_plan(layout: Layout, start: date, end: date, reqs: Sequence[Requirement], unresolved: Sequence[dict],
              disc: Discovery | None) -> None:
    layout.metadata.mkdir(parents=True, exist_ok=True)
    _write_csv(layout.contracts, CONTRACT_COLUMNS, requirement_rows(reqs))
    layout.plan.write_text(json.dumps({
        "from": start.isoformat(), "to": end.isoformat(),
        "unresolved": [{**u, "trading_date": str(u["trading_date"])} for u in unresolved],
        "calendar": [d.isoformat() for d in disc.calendar] if disc else [],
        "calendar_from": disc.calendar_from.isoformat() if disc else None,
        "discovery_issues": disc.issues if disc else []}, indent=2, default=str), encoding="utf-8")


def load_plan(layout: Layout) -> tuple[dict | None, list[Requirement]]:
    if not layout.plan.is_file() or not layout.contracts.is_file():
        return None, []
    return json.loads(layout.plan.read_text(encoding="utf-8")), requirements_from_rows(_read_csv(layout.contracts))


def run_download(*, client, catalog, layout: Layout, start: date, end: date, report_dir: Path,
                 expiry_mode: str = "nearest", offset: int = 1, resume: bool = False, validate_only: bool = False,
                 progress: Callable[[str], None] = lambda m: None) -> dict[str, Any]:
    manifest = Manifest(layout.manifest)
    plan, reqs = load_plan(layout)
    same_range = plan is not None and plan["from"] == start.isoformat() and plan["to"] == end.isoformat()
    reqs = reqs if same_range else []
    unresolved = plan["unresolved"] if same_range else []
    issues = plan["discovery_issues"] if same_range else []
    if not validate_only:
        key = catalog.underlying_key(layout.underlying)
        progress(f"NIFTY instrument key resolved: {key}")
        for item in underlying_items(key, start, end, layout):
            row, action = fetch_item(client, item, manifest, resume=resume)
            progress(f"NIFTY {item.month}: {action} ({row['candle_count']} candles)")
        if not (resume and same_range and reqs):
            progress("discovering point-in-time option contracts (expired-instrument API + instrument master) ...")
            disc = discover_contracts(catalog, layout.underlying, start, end)
            spot = load_underlying(layout, start, end)
            reqs, unresolved = plan_requirements(spot, disc, expiry_mode=expiry_mode, offset=offset)
            issues = disc.issues
            save_plan(layout, start, end, reqs, unresolved, disc)
        items = option_items(reqs, layout)
        progress(f"{len({r.contract.instrument_key for r in reqs})} option contracts, {len(items)} contract-month "
                 f"windows to check")
        actions = Counter()
        for i, item in enumerate(items, 1):
            _, action = fetch_item(client, item, manifest, resume=resume)
            actions[action] += 1
            if i % 50 == 0 or i == len(items):
                progress(f"options {i}/{len(items)}: {dict(actions)}")
    elif not reqs:
        progress("no stored plan found; option coverage cannot be validated without a download plan")
    s = summarize(layout, manifest, reqs, unresolved, start, end, issues)
    s["paths"] = {k: v.as_posix() for k, v in write_outputs(layout, s, reqs, report_dir).items()}
    return s
