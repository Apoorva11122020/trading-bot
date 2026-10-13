"""S13.1 - historical option-chain data availability audit (PAPER | RESEARCH ONLY | NO ORDERS | no downloads, no ML).

Audits the STORED NIFTY option candles in data/historical/upstox_expanded + data/historical/upstox_6m for two periods
(clean usable 2025-09-26 -> 2026-06-30, existing 2026 store 2026-04-01 -> 2026-09-30) and measures whether a
point-in-time option chain can be reconstructed. Nothing is repaired, filled, substituted or inferred: a missing
volume / open-interest value stays missing; an invalid candle is counted and excluded, never fixed. October 2026 is
never read. No strategy, no ML, no thresholds are tuned.

Point-in-time chain at 5-minute timestamp t (as of the close of candle t):
  nearest expiry = first listed expiry >= session date (expiry calendar + date only - never prices / outcomes)
  ATM            = strike-grid point nearest to the NIFTY close of candle t (ties -> lower strike)
  ATM +/- k      = ATM + k * strike step (step measured from the stored strikes)
  present        = a valid stored candle of (nearest expiry, strike, CE|PE) exists AT t (t - 5 min for changes)
Only candles at t (and t - 5 min) are used; a contract without a candle at t is "not present" - never substituted.
Note: the downloader only stored the strikes that were ATM during each session plus one neighbour each side, so
wide chains (+/-3, +/-5) are expected to be incomplete; this audit measures exactly how incomplete.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_left
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from src.historical_data import upstox_6m as u6
from src.options.selection import atm_strike
from src.research import s6_clean_strategy_lab as s6
from src.research import s11_history as hx

HEADING = "S13.1 OPTION-CHAIN AUDIT COMPLETE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | data audit only (no strategy, no ML, no repair)"
IST = u6.IST
PERIODS = {"CLEAN_USABLE": (date(2025, 9, 26), date(2026, 6, 30)),
           "EXISTING_2026": (date(2026, 4, 1), date(2026, 9, 30))}
PRIMARY = "CLEAN_USABLE"
LOAD_END = date(2026, 9, 30)
SIDES = ("CE", "PE")
OFFSETS = tuple(range(-5, 6))
BANDS = (1, 2, 3, 5)
COMPLETE_BAND = 3
STEP = pd.Timedelta(minutes=5)
SESSION_FIRST, SESSION_LAST = time(9, 15), time(15, 25)
REQUIRED = ("timestamp", "strike", "expiry", "option_type", "open", "high", "low", "close")
OPTIONAL = ("volume", "open_interest")
# Pre-existing project thresholds (src/historical_data/upstox_6m.py), not tuned here.
ATM_STRONG = u6.MIN_ATM_COVERAGE              # 0.98
COVERAGE_STRONG = u6.MIN_REQUIREMENT_COVERAGE  # 0.95
LARGELY_UNAVAILABLE_BELOW = 0.5                # "largely unavailable" = present on fewer than half
AVAILABLE, PARTIAL, UNAVAILABLE = "AVAILABLE", "PARTIAL", "UNAVAILABLE"
READY, PARTIAL_READY, NOT_READY = "S13_READY", "S13_PARTIAL", "S13_NOT_READY"
FIELD_AVAILABLE, FIELD_PARTIAL, FIELD_UNAVAILABLE = "A_AVAILABLE", "C_PARTIALLY_AVAILABLE", "B_UNAVAILABLE"
NAME_RE = re.compile(r"^(?P<u>[A-Z]+)_(?P<exp>\d{4}-\d{2}-\d{2})_(?P<strike>[0-9.]+)_(?P<typ>CE|PE)$")
KEY_EXP_RE = re.compile(r"\|(\d{2})-(\d{2})-(\d{4})$")
HARD_CHECKS = ("duplicate_rows", "conflicting_duplicates", "invalid_strike", "invalid_expiry", "invalid_option_type",
               "invalid_ohlc", "negative_volume", "negative_oi", "timezone_problems", "outside_session",
               "off_grid_timestamps", "contract_key_mismatch", "missing_required_columns")


def lab(k: int) -> str:
    return "0" if k == 0 else (f"m{-k}" if k < 0 else f"p{k}")


# ------------------------------------------------------------------ loading (read-only, never October)
def _month(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _guard(hi: date) -> None:
    hx.guard_no_october(hi)
    if hi > LOAD_END:
        raise SystemExit(f"Refusing {hi}: S13.1 never reads candles after {LOAD_END}")


def _to_ist(ts: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """IST timestamps + per-row flag of a timezone problem (naive or non-IST source). Values are not altered."""
    t = pd.to_datetime(ts)
    if t.dt.tz is None:
        return t.dt.tz_localize(IST), np.ones(len(t), dtype=bool)
    bad = str(t.dt.tz) not in ("Asia/Kolkata", "UTC+05:30", "pytz.FixedOffset(330)")
    return t.dt.tz_convert(IST), np.full(len(t), bad, dtype=bool)


def load_option_candles(roots: Sequence[str | Path], lo: date, hi: date,
                        underlying: str = "NIFTY") -> tuple[pd.DataFrame, dict]:
    """Every stored option candle with session date in [lo, hi]. Raw values unchanged; only month folders that can
    hold candles in [lo, hi] are opened."""
    _guard(hi)
    parts, info = [], {"files_opened": 0, "files_missing_volume": [], "files_missing_oi": [],
                       "files_missing_required": [], "files_bad_name": [], "rows_outside_window": 0, "max_raw": None}
    for r in roots:
        lay = u6.Layout(r, underlying)
        for p in lay.option_files():
            if not _month(lo) <= p.parent.name <= _month(hi):
                continue
            raw = pd.read_parquet(p)
            info["files_opened"] += 1
            if raw.empty:
                continue
            miss = [c for c in REQUIRED if c not in raw.columns]
            if miss:
                info["files_missing_required"].append(p.as_posix())
                continue
            f = raw.copy()
            f["timestamp"], f["tz_problem"] = _to_ist(raw["timestamp"])
            mx = f["timestamp"].max()
            info["max_raw"] = mx if info["max_raw"] is None or mx > info["max_raw"] else info["max_raw"]
            for col, bucket in (("volume", "files_missing_volume"), ("open_interest", "files_missing_oi")):
                if col not in f.columns:
                    f[col] = np.nan
                    info[bucket].append(p.as_posix())
            m = NAME_RE.match(p.stem)
            if not m:
                info["files_bad_name"].append(p.as_posix())
            f["name_expiry"] = m["exp"] if m else None
            f["name_strike"] = float(m["strike"]) if m else np.nan
            f["name_type"] = m["typ"] if m else None
            f["file"] = p.as_posix()
            if "instrument_key" not in f.columns:
                f["instrument_key"] = None
            d = f["timestamp"].dt.date
            keep = (d >= lo) & (d <= hi)
            info["rows_outside_window"] += int((~keep).sum())
            parts.append(f.loc[keep, ["timestamp", "tz_problem", "instrument_key", "strike", "expiry", "option_type",
                                      "open", "high", "low", "close", "volume", "open_interest", "name_expiry",
                                      "name_strike", "name_type", "file"]])
    cols = ["timestamp", "tz_problem", "instrument_key", "strike", "expiry", "option_type", "open", "high", "low",
            "close", "volume", "open_interest", "name_expiry", "name_strike", "name_type", "file"]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    out["expiry_raw"] = out["expiry"].astype(str)
    out["expiry"] = pd.to_datetime(out["expiry_raw"], errors="coerce").dt.date
    out["date"] = out["timestamp"].dt.date
    for c in ("strike", "open", "high", "low", "close", "volume", "open_interest"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    info["max_raw"] = info["max_raw"].isoformat() if info["max_raw"] is not None else None
    return out.sort_values(["timestamp", "expiry_raw", "strike", "option_type"], kind="mergesort").reset_index(drop=True), info


def load_spot(roots: Sequence[str | Path], lo: date, hi: date, underlying: str = "NIFTY") -> pd.DataFrame:
    _guard(hi)
    parts = []
    for r in roots:
        for p in u6.Layout(r, underlying).underlying_files():
            if _month(lo) <= p.stem <= _month(hi):
                f = u6.read_parquet(p)
                if len(f):
                    parts.append(f[["timestamp", "close"]])
    if not parts:
        return pd.DataFrame(columns=["timestamp", "close"])
    f = pd.concat(parts, ignore_index=True)
    f["timestamp"] = f["timestamp"].dt.tz_convert(IST)
    f = f.drop_duplicates("timestamp", keep="last")
    d = f["timestamp"].dt.date
    return f[(d >= lo) & (d <= hi)].sort_values("timestamp").reset_index(drop=True)


def load_calendar(roots: Sequence[str | Path], underlying: str = "NIFTY") -> list[date]:
    cal: set[date] = set()
    for r in roots:
        lay = u6.Layout(r, underlying)
        if lay.plan.is_file():
            cal |= {date.fromisoformat(x) for x in json.loads(lay.plan.read_text(encoding="utf-8")).get("calendar", [])}
    return sorted(cal)


def load_requirements(roots: Sequence[str | Path], lo: date, hi: date, underlying: str = "NIFTY") -> pd.DataFrame:
    rows = []
    for r in roots:
        lay = u6.Layout(r, underlying)
        if lay.contracts.is_file():
            for x in u6._read_csv(lay.contracts):
                d = date.fromisoformat(x["trading_date"])
                if lo <= d <= hi:
                    rows.append({"date": d, "expiry": date.fromisoformat(x["expiry"]), "strike": float(x["strike"]),
                                 "option_type": x["option_type"], "role": x["role"]})
    return pd.DataFrame(rows, columns=["date", "expiry", "strike", "option_type", "role"])


# ------------------------------------------------------------------ candle quality flags (no repair)
def strike_step(candles: pd.DataFrame) -> float | None:
    diffs: Counter = Counter()
    ok = candles.dropna(subset=["strike", "expiry"])
    for _, g in ok.groupby(["expiry", "option_type"]):
        s = np.unique(g["strike"].to_numpy(dtype=float))
        diffs.update(np.round(np.diff(s), 6).tolist())
    return float(diffs.most_common(1)[0][0]) if diffs else None


def flag_candles(c: pd.DataFrame, calendar: Sequence[date], step: float | None) -> pd.DataFrame:
    """Boolean flag columns per candle; ``usable`` = no flag. Nothing is changed or filled."""
    f = c.copy()
    cal = set(calendar)
    key = ["timestamp", "expiry_raw", "strike", "option_type"]
    dup_any = f.duplicated(key, keep=False)
    f["f_duplicate"] = f.duplicated(key, keep="first")
    px = ["open", "high", "low", "close", "volume", "open_interest"]
    conflict_keys = (f[dup_any].groupby(key, dropna=False)[px].nunique(dropna=False) > 1).any(axis=1)
    conflict_keys = set(conflict_keys[conflict_keys].index)
    f["f_conflict"] = [tuple(r) in conflict_keys for r in f[key].itertuples(index=False, name=None)] if conflict_keys \
        else False
    s = f["strike"]
    grid_bad = (np.abs(np.round(s / step) * step - s) > 1e-6) if step else pd.Series(False, index=f.index)
    f["f_strike"] = s.isna() | (s <= 0) | grid_bad | (f["name_strike"].notna() & (f["name_strike"] != s))
    f["f_expiry"] = f["expiry"].isna() | (pd.to_datetime(f["expiry"]) < pd.to_datetime(f["date"])) \
        | (f["name_expiry"].notna() & (f["name_expiry"] != f["expiry_raw"].str[:10])) \
        | ~f["expiry"].isin(cal)
    f["f_type"] = ~f["option_type"].isin(SIDES) | (f["name_type"].notna() & (f["name_type"] != f["option_type"]))
    o, h, l, cl = (f[k] for k in ("open", "high", "low", "close"))
    f["f_ohlc"] = o.isna() | h.isna() | l.isna() | cl.isna() | (o <= 0) | (h <= 0) | (l <= 0) | (cl <= 0) \
        | (h < np.maximum(o, cl)) | (l > np.minimum(o, cl)) | (h < l)
    f["f_negvol"] = f["volume"] < 0
    f["f_negoi"] = f["open_interest"] < 0
    t = f["timestamp"]
    f["f_tz"] = f["tz_problem"].astype(bool)
    tt = t.dt.time
    f["f_session"] = (tt < SESSION_FIRST) | (tt > SESSION_LAST)
    f["f_offgrid"] = (t.dt.minute % 5 != 0) | (t.dt.second != 0)
    kexp = f["instrument_key"].astype(str).str.extract(KEY_EXP_RE)
    has = kexp[0].notna()
    kd = kexp[2] + "-" + kexp[1] + "-" + kexp[0]
    f["f_key"] = has & (kd != f["expiry_raw"].str[:10])
    flags = ["f_duplicate", "f_conflict", "f_strike", "f_expiry", "f_type", "f_ohlc", "f_negvol", "f_negoi", "f_tz",
             "f_session", "f_offgrid", "f_key"]
    f["usable"] = ~f[flags].any(axis=1)
    return f


def quality(flagged: pd.DataFrame, spot: pd.DataFrame, reqs: pd.DataFrame, info: dict,
            lo: date, hi: date) -> dict[str, Any]:
    f = flagged[(flagged["date"] >= lo) & (flagged["date"] <= hi)]
    counts = {"candles": int(len(f)), "duplicate_rows": int(f["f_duplicate"].sum()),
              "conflicting_duplicates": int(f["f_conflict"].sum()), "invalid_strike": int(f["f_strike"].sum()),
              "invalid_expiry": int(f["f_expiry"].sum()), "invalid_option_type": int(f["f_type"].sum()),
              "invalid_ohlc": int(f["f_ohlc"].sum()), "negative_volume": int(f["f_negvol"].sum()),
              "negative_oi": int(f["f_negoi"].sum()), "timezone_problems": int(f["f_tz"].sum()),
              "outside_session": int(f["f_session"].sum()), "off_grid_timestamps": int(f["f_offgrid"].sum()),
              "contract_key_mismatch": int(f["f_key"].sum()),
              "missing_required_columns": len(info.get("files_missing_required", [])),
              "unusable_candles": int((~f["usable"]).sum())}
    sp = spot[(spot["timestamp"].dt.date >= lo) & (spot["timestamp"].dt.date <= hi)]
    expected = sp.groupby(sp["timestamp"].dt.date).size().to_dict()
    r = reqs[(reqs["date"] >= lo) & (reqs["date"] <= hi)].drop_duplicates(["date", "expiry", "strike", "option_type"])
    u = f[f["usable"]]
    have = u.groupby(["date", "expiry", "strike", "option_type"]).size().to_dict()
    miss_bars = no_candles = exp_total = 0
    atm_miss = atm_exp = 0
    for row in r.itertuples(index=False):
        e = expected.get(row.date, 0)
        n = have.get((row.date, row.expiry, row.strike, row.option_type), 0)
        exp_total += e
        miss_bars += max(0, e - n)
        no_candles += int(n == 0)
        if row.role == u6.ATM:
            atm_exp += e
            atm_miss += max(0, e - n)
    counts.update(requirements=int(len(r)), requirements_without_candles=no_candles, expected_bars=exp_total,
                  missing_candles=miss_bars, missing_candle_share=miss_bars / exp_total if exp_total else None,
                  atm_missing_candles=atm_miss, atm_missing_share=atm_miss / atm_exp if atm_exp else None)
    counts["passed"] = all(counts[k] == 0 for k in HARD_CHECKS) and len(f) > 0
    counts["rule"] = ("PASS = zero duplicate / invalid strike / invalid expiry / invalid type / invalid OHLC / negative "
                      "volume / negative OI / timezone / out-of-session / off-grid / contract-key problems; missing "
                      "candles are reported (never filled) and judged through chain coverage")
    return counts


# ------------------------------------------------------------------ point-in-time chain reconstruction
def nearest_expiry(calendar: Sequence[date], d: date) -> date | None:
    """First listed expiry on / after the session date. Depends on the calendar and the date only."""
    cal = sorted(calendar)
    i = bisect_left(cal, d)
    return cal[i] if i < len(cal) else None


def _tkey(ts: pd.Series) -> np.ndarray:
    """Timezone / unit independent join key (UTC nanoseconds)."""
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()


def atm_of(spot: float, step: float) -> float:
    base = np.floor(spot / step) * step
    return float(atm_strike([base - step, base, base + step, base + 2 * step], float(spot)))


def reconstruct(stamps: pd.DataFrame, usable: pd.DataFrame, calendar: Sequence[date], step: float) -> pd.DataFrame:
    """One row per timestamp in ``stamps`` (timestamp, close). Uses only candles AT t and t - 5 min."""
    b = stamps[["timestamp", "close"]].copy().reset_index(drop=True)
    b["date"] = b["timestamp"].dt.date
    exp_by_day = {d: nearest_expiry(calendar, d) for d in sorted(set(b["date"]))}
    b["expiry"] = b["date"].map(exp_by_day)
    b["dte"] = [(e - d).days if e is not None else None for e, d in zip(b["expiry"], b["date"])]
    b["atm"] = [atm_of(c, step) for c in b["close"].astype(float)]
    src = usable[["expiry", "strike", "option_type", "close", "volume", "open_interest"]] \
        .rename(columns={"close": "_c", "volume": "_v", "open_interest": "_o"}).assign(_t=_tkey(usable["timestamp"]))
    now, before = _tkey(b["timestamp"]), _tkey(b["timestamp"] - STEP)
    cols = {}
    for side in SIDES:
        for k in OFFSETS:
            for prev in (False, True):
                q = pd.DataFrame({"_t": before if prev else now, "expiry": b["expiry"],
                                  "strike": b["atm"] + k * step, "option_type": side})
                m = q.merge(src, how="left", on=["_t", "expiry", "strike", "option_type"], validate="m:1")
                pre = f"{side}{lab(k)}_{'prev_' if prev else ''}"
                cols[pre + "p"] = m["_c"].notna().to_numpy()
                cols[pre + "v"] = (m["_c"].notna() & m["_v"].notna() & (m["_v"] >= 0)).to_numpy()
                cols[pre + "o"] = (m["_c"].notna() & m["_o"].notna() & (m["_o"] >= 0)).to_numpy()
                if not prev:
                    cols[pre + "oi_pos"] = (m["_c"].notna() & (m["_o"] > 0)).to_numpy()
    out = pd.concat([b, pd.DataFrame(cols)], axis=1)
    band = [f"{s}{lab(k)}_p" for s in SIDES for k in range(-COMPLETE_BAND, COMPLETE_BAND + 1)]
    n = out[band].sum(axis=1)
    out["chain_status"] = np.where(n == len(band), "complete_chain", np.where(n == 0, "missing_chain",
                                                                               "partial_chain"))
    out["complete_pm5"] = out[[f"{s}{lab(k)}_p" for s in SIDES for k in OFFSETS]].all(axis=1)
    for s in SIDES:
        out[f"{s}_strikes_present_pm5"] = out[[f"{s}{lab(k)}_p" for k in OFFSETS]].sum(axis=1)
    return out


def _band(R: pd.DataFrame, side: str, k: int, kind: str = "p", prev: bool = False) -> pd.Series:
    pre = "prev_" if prev else ""
    return R[[f"{side}{lab(j)}_{pre}{kind}" for j in range(-k, k + 1)]].all(axis=1)


def _share(m: pd.Series) -> float | None:
    return float(m.mean()) if len(m) else None


def chain_summary(R: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"timestamps": int(len(R)), "sessions": int(R["date"].nunique()) if len(R) else 0}
    for s in SIDES:
        out[f"atm_{s.lower()}"] = _share(R[f"{s}0_p"])
        for k in BANDS:
            out[f"band_pm{k}_{s.lower()}"] = _share(_band(R, s, k))
            out[f"offset_pm{k}_{s.lower()}"] = _share(R[f"{s}{lab(-k)}_p"] & R[f"{s}{lab(k)}_p"])
    out["atm_both"] = _share(R["CE0_p"] & R["PE0_p"])
    for k in BANDS:
        out[f"band_pm{k}_both"] = _share(_band(R, "CE", k) & _band(R, "PE", k))
    vc = R["chain_status"].value_counts().to_dict() if len(R) else {}
    for st in ("complete_chain", "partial_chain", "missing_chain"):
        out[st] = int(vc.get(st, 0))
    out["complete_pm5"] = int(R["complete_pm5"].sum()) if len(R) else 0
    return out


def daily_rows(R: pd.DataFrame, period: str) -> list[dict]:
    rows = []
    for d, g in R.groupby("date"):
        s = chain_summary(g)
        rows.append({"period": period, "level": "DAILY", "date": d.isoformat(), **{k: v for k, v in s.items()
                                                                                    if k != "sessions"}})
    return rows


# ------------------------------------------------------------------ OI / volume coverage (per candle)
def field_rows(flagged: pd.DataFrame, R: pd.DataFrame, field: str, period: str, lo: date, hi: date,
               step: float) -> list[dict]:
    f = flagged[(flagged["date"] >= lo) & (flagged["date"] <= hi)]
    near = pd.DataFrame({"_t": _tkey(R["timestamp"]), "_near": R["expiry"].to_numpy(), "atm": R["atm"].to_numpy()})
    pos = f.assign(_t=_tkey(f["timestamp"])).merge(near, how="left", on="_t", validate="m:1")
    off = np.where(pos["expiry"] == pos["_near"], (pos["strike"] - pos["atm"]) / step, np.nan)
    groups = {"ALL": np.ones(len(pos), bool), "CE": (pos["option_type"] == "CE").to_numpy(),
              "PE": (pos["option_type"] == "PE").to_numpy(), "ATM": off == 0, "ATM_CE": (off == 0) & (pos["option_type"] == "CE").to_numpy(),
              "ATM_PE": (off == 0) & (pos["option_type"] == "PE").to_numpy(),
              "ATM_PM3": np.abs(off) <= 3}
    v = pos[field].to_numpy(dtype=float)
    out = []
    for name, m in groups.items():
        x = v[m]
        n = len(x)
        missing = int(np.isnan(x).sum())
        invalid = int((~np.isnan(x) & (~np.isfinite(x) | (x < 0))).sum())
        zero = int((x == 0).sum())
        present = int((np.isfinite(x) & (x > 0)).sum())
        out.append({"period": period, "field": field, "group": name, "candles": n, "present_positive": present,
                    "zero": zero, "missing": missing, "invalid": invalid,
                    "coverage": (present + zero) / n if n else None, "positive_share": present / n if n else None,
                    "status": field_status(present + zero, n)})
    return out


def field_status(valid: int, n: int) -> str:
    if n == 0 or valid == 0:
        return FIELD_UNAVAILABLE
    return FIELD_AVAILABLE if valid == n else FIELD_PARTIAL


# ------------------------------------------------------------------ expiry structure
def expiry_rows(R: pd.DataFrame, flagged: pd.DataFrame, reqs: pd.DataFrame, calendar: Sequence[date],
                period: str) -> list[dict]:
    cal = sorted(calendar)
    monthly = {max(e for e in cal if (e.year, e.month) == (y, m)) for y, m in {(e.year, e.month) for e in cal}}
    planned = reqs.groupby("date")["expiry"].agg(lambda s: sorted(set(s))).to_dict() if len(reqs) else {}
    seen = flagged[flagged["usable"]].groupby("date")["expiry"].agg(lambda s: sorted(set(s))).to_dict()
    rows = []
    for d, g in R.groupby("date"):
        e = g["expiry"].iloc[0]
        rows.append({"period": period, "date": d.isoformat(), "nearest_expiry": e.isoformat() if e else None,
                     "dte": (e - d).days if e else None, "expiry_kind": ("MONTHLY" if e in monthly else "WEEKLY")
                     if e else None, "expiry_weekday": e.strftime("%A") if e else None,
                     "planned_expiries": " ".join(x.isoformat() for x in planned.get(d, [])),
                     "planned_matches_nearest": planned.get(d) == [e] if d in planned else None,
                     "stored_expiries": " ".join(x.isoformat() for x in seen.get(d, [])),
                     "stored_expired_contracts": sum(1 for x in seen.get(d, []) if x < d),
                     "mean_ce_strikes_pm5": float(g["CE_strikes_present_pm5"].mean()),
                     "mean_pe_strikes_pm5": float(g["PE_strikes_present_pm5"].mean()),
                     "min_ce_strikes_pm5": int(g["CE_strikes_present_pm5"].min()),
                     "min_pe_strikes_pm5": int(g["PE_strikes_present_pm5"].min())})
    return rows


def expiry_summary(rows: Sequence[dict], calendar: Sequence[date], lo: date, hi: date) -> dict[str, Any]:
    cal = [e for e in calendar if lo <= e <= hi + timedelta(days=10)]
    per_month = Counter((e.year, e.month) for e in cal)
    kinds = Counter(r["expiry_kind"] for r in rows)
    mism = [r["date"] for r in rows if r["planned_matches_nearest"] is False]
    return {"sessions": len(rows), "no_expiry": sum(1 for r in rows if r["nearest_expiry"] is None),
            "weekly_sessions": kinds.get("WEEKLY", 0), "monthly_sessions": kinds.get("MONTHLY", 0),
            "months_with_multiple_expiries": sum(1 for v in per_month.values() if v >= 2),
            "calendar_months": len(per_month),
            "weekly_monthly_identifiable": bool(per_month) and all(v >= 2 for v in per_month.values()),
            "expiry_weekdays": dict(Counter(e.strftime("%A") for e in cal)),
            "planned_expiry_mismatch_sessions": mism[:20], "planned_expiry_mismatches": len(mism),
            "expired_contract_sessions": sum(1 for r in rows if r["stored_expired_contracts"]),
            "dte_min": min((r["dte"] for r in rows if r["dte"] is not None), default=None),
            "dte_max": max((r["dte"] for r in rows if r["dte"] is not None), default=None),
            "rule": "nearest expiry = first listed expiry >= session date (calendar + date only; never prices); "
                    "MONTHLY = last listed expiry of its calendar month"}


# ------------------------------------------------------------------ point-in-time checks
def truncation_check(R: pd.DataFrame, stamps: pd.DataFrame, usable: pd.DataFrame, calendar: Sequence[date],
                     step: float, sample: int) -> dict[str, Any]:
    """Each sampled timestamp rebuilt from that session's candles <= t only (every later candle and every other
    session removed) must equal the full-data reconstruction."""
    if R.empty:
        return {"checked": 0, "mismatches": [], "passed": False}
    idx = np.unique(np.linspace(0, len(R) - 1, min(sample, len(R))).astype(int))
    bad = []
    cols = [c for c in R.columns if c.endswith(("_p", "_v", "_o", "_oi_pos"))] + ["atm", "expiry", "chain_status"]
    for i in idx:
        t = R["timestamp"].iloc[i]
        cut = usable[(usable["date"] == t.date()) & (usable["timestamp"] <= t)]
        one = reconstruct(stamps[stamps["timestamp"] == t], cut, calendar, step)
        if not one[cols].iloc[0].equals(R[cols].iloc[i]):
            bad.append(t.isoformat())
    return {"checked": int(len(idx)), "mismatches": bad[:20], "passed": not bad}


def pit_checks(R: pd.DataFrame, stamps: pd.DataFrame, usable: pd.DataFrame, calendar: Sequence[date], step: float,
               sample: int) -> dict[str, Any]:
    cal = sorted(calendar)
    wrong = [d for d, e in zip(R["date"], R["expiry"]) if e is None or e < d
             or e != min((x for x in cal if x >= d), default=None)]
    used = usable.assign(_t=_tkey(usable["timestamp"])).merge(
        pd.DataFrame({"_t": _tkey(R["timestamp"]), "expiry": R["expiry"].to_numpy()}), on=["_t", "expiry"],
        how="inner")
    expired_used = int((used["expiry"] < used["date"]).sum())
    trunc = truncation_check(R, stamps, usable, calendar, step, sample)
    det = bool(reconstruct(stamps, usable, calendar, step).equals(R))
    return {"expiry_causal_violations": len(wrong), "expired_contracts_used": expired_used,
            "future_truncation": trunc, "deterministic": det,
            "expiry_rule_inputs": "expiry calendar + session date only (no prices, no outcomes)",
            "passed": not wrong and expired_used == 0 and trunc["passed"] and det}


# ------------------------------------------------------------------ feature availability
def feature_specs() -> list[dict]:
    """(name, field, band_limited, mask builder, definition)."""
    both = lambda R, k, kind, prev=False: _band(R, "CE", k, kind, prev) & _band(R, "PE", k, kind, prev)  # noqa: E731
    chg = lambda R, s, kind: R[f"{s}0_{kind}"] & R[f"{s}0_prev_{kind}"]  # noqa: E731
    return [
        {"feature": "PCR_OI", "field": "open_interest", "band_limited": True,
         "mask": lambda R: both(R, 1, "o"), "inputs": "CE+PE OI at ATM-1..ATM+1 at t",
         "note": "only a band PCR is possible: strikes beyond the stored band (full chain) were never downloaded"},
        {"feature": "PCR_VOLUME", "field": "volume", "band_limited": True,
         "mask": lambda R: both(R, 1, "v"), "inputs": "CE+PE volume at ATM-1..ATM+1 at t",
         "note": "band PCR only (full chain not stored)"},
        {"feature": "CE_OI_CHANGE", "field": "open_interest", "band_limited": False,
         "mask": lambda R: chg(R, "CE", "o"), "inputs": "ATM CE OI at t and t-5min (same contract)", "note": ""},
        {"feature": "PE_OI_CHANGE", "field": "open_interest", "band_limited": False,
         "mask": lambda R: chg(R, "PE", "o"), "inputs": "ATM PE OI at t and t-5min (same contract)", "note": ""},
        {"feature": "CE_VOLUME_CHANGE", "field": "volume", "band_limited": False,
         "mask": lambda R: chg(R, "CE", "v"), "inputs": "ATM CE volume at t and t-5min", "note": ""},
        {"feature": "PE_VOLUME_CHANGE", "field": "volume", "band_limited": False,
         "mask": lambda R: chg(R, "PE", "v"), "inputs": "ATM PE volume at t and t-5min", "note": ""},
        {"feature": "ATM_STRADDLE_PREMIUM", "field": "close", "band_limited": False,
         "mask": lambda R: R["CE0_p"] & R["PE0_p"], "inputs": "ATM CE close + ATM PE close at t", "note": ""},
        {"feature": "CE_PE_RELATIVE_STRENGTH", "field": "close", "band_limited": False,
         "mask": lambda R: chg(R, "CE", "p") & chg(R, "PE", "p"), "inputs": "ATM CE / PE close at t and t-5min",
         "note": ""},
        {"feature": "ATM_PM1_BREADTH", "field": "close", "band_limited": False,
         "mask": lambda R: both(R, 1, "p") & both(R, 1, "p", True), "inputs": "CE+PE close ATM-1..ATM+1, t and t-5min",
         "note": ""},
        {"feature": "ATM_PM3_BREADTH", "field": "close", "band_limited": False,
         "mask": lambda R: both(R, 3, "p") & both(R, 3, "p", True), "inputs": "CE+PE close ATM-3..ATM+3, t and t-5min",
         "note": ""},
        {"feature": "OI_CONCENTRATION", "field": "open_interest", "band_limited": True,
         "mask": lambda R: both(R, 3, "o"), "inputs": "CE+PE OI ATM-3..ATM+3 at t",
         "note": "concentration over a band only (full chain not stored)"},
        {"feature": "STRIKE_PRICE_MOMENTUM", "field": "close", "band_limited": False,
         "mask": lambda R: both(R, 1, "p") & both(R, 1, "p", True), "inputs": "per-strike close at t and t-5min, ATM+/-1",
         "note": ""},
        {"feature": "CALL_PUT_OI_IMBALANCE", "field": "open_interest", "band_limited": False,
         "mask": lambda R: R["CE0_o"] & R["PE0_o"], "inputs": "ATM CE OI and ATM PE OI at t", "note": ""},
        {"feature": "CALL_PUT_VOLUME_IMBALANCE", "field": "volume", "band_limited": False,
         "mask": lambda R: R["CE0_v"] & R["PE0_v"], "inputs": "ATM CE volume and ATM PE volume at t", "note": ""},
    ]


def feature_status(cov: float | None, field_valid: int, band_limited: bool) -> tuple[str, str]:
    if not field_valid or not cov:
        return UNAVAILABLE, "required field never present"
    if cov < LARGELY_UNAVAILABLE_BELOW:
        return UNAVAILABLE, f"inputs present on only {cov:.1%} of timestamps (< {LARGELY_UNAVAILABLE_BELOW:.0%})"
    if band_limited:
        return PARTIAL, "band-limited (full chain not stored)"
    if cov >= COVERAGE_STRONG:
        return AVAILABLE, f"inputs present on {cov:.1%} of timestamps (>= {COVERAGE_STRONG:.0%})"
    return PARTIAL, f"inputs present on {cov:.1%} of timestamps"


def feature_rows(R: pd.DataFrame, flagged: pd.DataFrame, period: str, lo: date, hi: date) -> list[dict]:
    f = flagged[(flagged["date"] >= lo) & (flagged["date"] <= hi) & flagged["usable"]]
    valid = {"open_interest": int((f["open_interest"] >= 0).sum()), "volume": int((f["volume"] >= 0).sum()),
             "close": int(f["close"].notna().sum())}
    out = []
    for sp in feature_specs():
        cov = _share(sp["mask"](R)) if len(R) else None
        st, why = feature_status(cov, valid[sp["field"]], sp["band_limited"])
        out.append({"period": period, "feature": sp["feature"], "status": st, "timestamp_coverage": cov,
                    "inputs": sp["inputs"], "reason": why, "note": sp["note"], "calculated": False})
    return out


# ------------------------------------------------------------------ feasibility
def classify(*, dq_ok: bool, pit_ok: bool, recon_ok: bool, chain: dict, oi: dict, vol: dict) -> tuple[str, list[str]]:
    atm = min(chain["atm_ce"] or 0, chain["atm_pe"] or 0)
    band3 = chain["band_pm3_both"] or 0
    oi_atm, vol_atm = oi["ATM"]["coverage"] or 0, vol["ATM"]["coverage"] or 0
    why = [f"ATM CE/PE coverage {atm:.2%}", f"ATM+/-3 both-side coverage {band3:.2%}",
           f"ATM OI coverage {oi_atm:.2%}", f"ATM volume coverage {vol_atm:.2%}"]
    if not (dq_ok and pit_ok and recon_ok):
        return NOT_READY, why + ["data quality / point-in-time / reconstruction check failed"]
    if atm < COVERAGE_STRONG:
        return NOT_READY, why + [f"ATM coverage below {COVERAGE_STRONG:.0%}"]
    if oi_atm < LARGELY_UNAVAILABLE_BELOW and vol_atm < LARGELY_UNAVAILABLE_BELOW:
        return NOT_READY, why + ["OI and volume largely unavailable"]
    if atm >= ATM_STRONG and vol_atm >= COVERAGE_STRONG and oi_atm >= COVERAGE_STRONG and band3 >= COVERAGE_STRONG:
        return READY, why
    gaps = []
    if band3 < COVERAGE_STRONG:
        gaps.append(f"ATM+/-3 chain complete on only {band3:.2%} of timestamps")
    if oi_atm < COVERAGE_STRONG:
        gaps.append("OI incomplete")
    if vol_atm < COVERAGE_STRONG:
        gaps.append("volume incomplete")
    if atm < ATM_STRONG:
        gaps.append(f"ATM coverage below {ATM_STRONG:.0%}")
    return PARTIAL_READY, why + gaps


# ------------------------------------------------------------------ audit
def build_audit(roots: Sequence[str | Path], underlying: str = "NIFTY", sample: int = 40,
                progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    lo = min(a for a, _ in PERIODS.values())
    hi = max(b for _, b in PERIODS.values())
    _guard(hi)
    say(f"reading stored option candles {lo} .. {hi} (October never read) ...")
    candles, info = load_option_candles(roots, lo, hi, underlying)
    spot = load_spot(roots, lo, hi, underlying)
    calendar = load_calendar(roots, underlying)
    reqs = load_requirements(roots, lo, hi, underlying)
    step = strike_step(candles)
    say(f"{len(candles)} option candles, {len(spot)} NIFTY bars, strike step {step}")
    flagged = flag_candles(candles, calendar, step)
    usable = flagged[flagged["usable"]]
    conflict = flagged[flagged["f_conflict"]]
    usable = usable.drop_duplicates(["timestamp", "expiry", "strike", "option_type"], keep="first")
    usable = usable[~usable.set_index(["timestamp", "expiry", "strike", "option_type"]).index.isin(
        conflict.set_index(["timestamp", "expiry", "strike", "option_type"]).index)] if len(conflict) else usable
    on_grid = spot[(spot["timestamp"].dt.minute % 5 == 0) & (spot["timestamp"].dt.second == 0)]
    periods: dict[str, dict] = {}
    rows = {k: [] for k in ("chain", "oi", "volume", "expiry", "feature", "quality", "stamps")}
    for name, (a, b) in PERIODS.items():
        say(f"{name}: reconstructing the point-in-time chain {a} .. {b} ...")
        st = on_grid[(on_grid["timestamp"].dt.date >= a) & (on_grid["timestamp"].dt.date <= b)]
        if step is None or st.empty:
            periods[name] = {"range": [a.isoformat(), b.isoformat()], "ran": False, "status": NOT_READY,
                             "reason": "no option candles or no NIFTY bars"}
            continue
        R = reconstruct(st, usable, calendar, step)
        chain = chain_summary(R)
        dq = quality(flagged, spot, reqs, info, a, b)
        oi = {r["group"]: r for r in field_rows(flagged, R, "open_interest", name, a, b, step)}
        vol = {r["group"]: r for r in field_rows(flagged, R, "volume", name, a, b, step)}
        ex = expiry_rows(R, flagged[(flagged["date"] >= a) & (flagged["date"] <= b)],
                         reqs[(reqs["date"] >= a) & (reqs["date"] <= b)], calendar, name)
        exs = expiry_summary(ex, calendar, a, b)
        pit = pit_checks(R, st, usable[(usable["date"] >= a) & (usable["date"] <= b)], calendar, step, sample)
        grid_ok = dq["invalid_strike"] == 0
        recon_ok = bool(pit["deterministic"] and grid_ok and exs["no_expiry"] == 0
                        and min(chain["atm_ce"] or 0, chain["atm_pe"] or 0) >= COVERAGE_STRONG)
        feats = feature_rows(R, flagged, name, a, b)
        status, why = classify(dq_ok=dq["passed"], pit_ok=pit["passed"], recon_ok=recon_ok, chain=chain, oi=oi,
                               vol=vol)
        periods[name] = {"range": [a.isoformat(), b.isoformat()], "ran": True, "chain": chain, "oi": oi,
                         "volume": vol, "expiry": exs, "point_in_time": pit, "data_quality": dq,
                         "reconstruction": {"passed": recon_ok, "strike_step": step, "strike_grid_ok": grid_ok,
                                            "rule": "PASS = deterministic, strikes on the measured grid, an expiry "
                                                    f"for every session, ATM CE and PE present >= {COVERAGE_STRONG:.0%}"},
                         "features": {r["feature"]: r for r in feats}, "status": status, "status_reasons": why}
        rows["chain"].append({"period": name, "level": "SUMMARY", "date": "", **chain})
        rows["chain"] += daily_rows(R, name)
        rows["oi"] += list(oi.values())
        rows["volume"] += list(vol.values())
        rows["expiry"] += ex
        rows["feature"] += feats
        rows["quality"] += [{"period": name, "check": k, "value": v} for k, v in dq.items() if k != "rule"]
        keep = ["timestamp", "expiry", "dte", "close", "atm", "chain_status", "CE_strikes_present_pm5",
                "PE_strikes_present_pm5", "CE0_p", "PE0_p", "CE0_v", "PE0_v", "CE0_o", "PE0_o"]
        rows["stamps"] += [{"period": name, **r} for r in R[keep].to_dict("records")]
    prim = periods.get(PRIMARY, {})
    return {"heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
            "roots": [Path(r).as_posix() for r in roots], "periods": periods, "primary_period": PRIMARY,
            "load_range": [lo.isoformat(), hi.isoformat()], "load_info": info, "strike_step": step,
            "calendar_expiries": len(calendar), "status": prim.get("status", NOT_READY),
            "october_loaded": False, "production_unchanged": None, "live_orders": 0, "trading_mode": "PAPER",
            "live_trading_enabled": False, "fabricated_values": 0, "repairs": 0,
            "download_scope_note": "stored strikes = ATM during each session +/- 1 neighbour (download plan); wider "
                                   "strikes were never stored, so +/-3 / +/-5 chains are measured, not filled",
            "_rows": rows}


# ------------------------------------------------------------------ reports / text
REPORT_FILES = ("summary.json", "s13_chain_audit.md", "chain_coverage.csv", "oi_coverage.csv", "volume_coverage.csv",
                "expiry_coverage.csv", "feature_availability.csv", "data_quality.csv", "chain_timestamps.csv")


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    rows = a.get("_rows") or {}
    p["summary.json"].write_text(json.dumps(s6._jsonable(_strip(a)), indent=2, default=str), encoding="utf-8")
    for name, key in (("chain_coverage.csv", "chain"), ("oi_coverage.csv", "oi"), ("volume_coverage.csv", "volume"),
                      ("expiry_coverage.csv", "expiry"), ("feature_availability.csv", "feature"),
                      ("data_quality.csv", "quality"), ("chain_timestamps.csv", "stamps")):
        s6._csv(p[name], rows.get(key, []))
    p["s13_chain_audit.md"].write_text(markdown(a), encoding="utf-8")
    return p


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.2f}%"


def _ok(v) -> str:
    return "PASS" if v else "FAIL"


def _feat(P: dict, *names: str) -> str:
    fs = [P["features"][n] for n in names]
    order = {AVAILABLE: 0, PARTIAL: 1, UNAVAILABLE: 2}
    worst = max(fs, key=lambda r: order[r["status"]])
    cov = min((r["timestamp_coverage"] or 0) for r in fs)
    return f"{worst['status']} ({_pct(cov)} of timestamps)"


def period_block(P: dict) -> list[str]:
    c, oi, vol = P["chain"], P["oi"], P["volume"]
    return ["ATM COVERAGE:", f"CE: {_pct(c['atm_ce'])}", f"PE: {_pct(c['atm_pe'])}", "",
            "ATM ±1:", f"CE: {_pct(c['band_pm1_ce'])}", f"PE: {_pct(c['band_pm1_pe'])}", "",
            "ATM ±3:", f"CE: {_pct(c['band_pm3_ce'])}", f"PE: {_pct(c['band_pm3_pe'])}",
            f"(ATM ±2 CE {_pct(c['band_pm2_ce'])} / PE {_pct(c['band_pm2_pe'])}; ATM ±5 CE {_pct(c['band_pm5_ce'])} / "
            f"PE {_pct(c['band_pm5_pe'])}; timestamps {c['timestamps']}: complete {c['complete_chain']}, partial "
            f"{c['partial_chain']}, missing {c['missing_chain']} [complete = CE+PE at every strike ATM±3])", "",
            "OI:", f"CE: {_pct(oi['CE']['coverage'])} ({oi['CE']['status']})",
            f"PE: {_pct(oi['PE']['coverage'])} ({oi['PE']['status']})",
            f"overall: {_pct(oi['ALL']['coverage'])} (zero {oi['ALL']['zero']}, missing {oi['ALL']['missing']}, "
            f"invalid {oi['ALL']['invalid']} of {oi['ALL']['candles']} candles)", "",
            "VOLUME:", f"CE: {_pct(vol['CE']['coverage'])} ({vol['CE']['status']})",
            f"PE: {_pct(vol['PE']['coverage'])} ({vol['PE']['status']})",
            f"overall: {_pct(vol['ALL']['coverage'])} (zero {vol['ALL']['zero']}, missing {vol['ALL']['missing']}, "
            f"invalid {vol['ALL']['invalid']} of {vol['ALL']['candles']} candles)", "",
            "FEATURE AVAILABILITY:", "",
            f"PCR OI: {_feat(P, 'PCR_OI')}", f"PCR volume: {_feat(P, 'PCR_VOLUME')}",
            f"OI change: {_feat(P, 'CE_OI_CHANGE', 'PE_OI_CHANGE')}",
            f"volume imbalance: {_feat(P, 'CALL_PUT_VOLUME_IMBALANCE')}",
            f"CE/PE relative strength: {_feat(P, 'CE_PE_RELATIVE_STRENGTH')}",
            f"ATM straddle: {_feat(P, 'ATM_STRADDLE_PREMIUM')}",
            f"ATM breadth: ±1 {_feat(P, 'ATM_PM1_BREADTH')}; ±3 {_feat(P, 'ATM_PM3_BREADTH')}", "",
            "CHAIN RECONSTRUCTION:", _ok(P["reconstruction"]["passed"]), "",
            "POINT-IN-TIME:", _ok(P["point_in_time"]["passed"]), "",
            "DATA QUALITY:", _ok(P["data_quality"]["passed"])]


def terminal_text(a: dict) -> str:
    out = [HEADING, ""]
    P = a["periods"].get(PRIMARY, {})
    lo, hi = PERIODS[PRIMARY]
    out += ["PERIOD:", f"start: {lo}", f"end: {hi}", f"({PRIMARY}; strike step {a.get('strike_step')})", ""]
    if P.get("ran"):
        out += period_block(P)
    else:
        out.append(f"NOT RUN: {P.get('reason')}")
    out += ["", "S13 STATUS:", a.get("status", NOT_READY)]
    if P.get("status_reasons"):
        out.append("(" + "; ".join(P["status_reasons"]) + ")")
    for name, Q in a["periods"].items():
        if name != PRIMARY and Q.get("ran"):
            c = Q["chain"]
            out += ["", f"{name} {Q['range'][0]} -> {Q['range'][1]} (reported separately): ATM CE {_pct(c['atm_ce'])} "
                    f"/ PE {_pct(c['atm_pe'])}; ±1 CE {_pct(c['band_pm1_ce'])} / PE {_pct(c['band_pm1_pe'])}; ±3 CE "
                    f"{_pct(c['band_pm3_ce'])} / PE {_pct(c['band_pm3_pe'])}; OI {_pct(Q['oi']['ALL']['coverage'])}; "
                    f"volume {_pct(Q['volume']['ALL']['coverage'])}; reconstruction "
                    f"{_ok(Q['reconstruction']['passed'])}; point-in-time {_ok(Q['point_in_time']['passed'])}; data "
                    f"quality {_ok(Q['data_quality']['passed'])}; status {Q['status']}"]
    out += ["", "OCTOBER LOADED:", "NO", "", "PRODUCTION CHANGED:",
            "NO" if a.get("production_unchanged") is not False else "YES", "", "LIVE ORDERS:", "0"]
    return "\n".join(out)


def markdown(a: dict) -> str:
    md = ["# S13.1 historical option-chain data availability audit", "", f"**{BANNER}**", "",
          f"S13 status (primary period {PRIMARY}): **{a.get('status')}**", "",
          f"> {a.get('download_scope_note')}", "",
          "Definitions: point-in-time chain at t uses the nearest listed expiry >= session date, ATM = strike-grid "
          "point nearest to the NIFTY close of candle t, and only candles stored AT t (t-5min for change features). "
          "Missing values stay missing (never inferred / filled). Features were NOT calculated - only input coverage "
          "was measured.", ""]
    for name, P in a["periods"].items():
        md += [f"## {name} {P['range'][0]} -> {P['range'][1]}", ""]
        if not P.get("ran"):
            md += [f"Not run: {P.get('reason')}", ""]
            continue
        md += ["```", *period_block(P), "```", "", f"Status: **{P['status']}** - " + "; ".join(P["status_reasons"]), "",
               "| Feature | Status | Timestamp coverage | Inputs | Note |", "|---|---|---|---|---|"]
        md += [f"| {r['feature']} | {r['status']} | {_pct(r['timestamp_coverage'])} | {r['inputs']} | "
               f"{r['note'] or r['reason']} |" for r in P["features"].values()]
        dq = P["data_quality"]
        md += ["", "Data quality: " + ", ".join(f"{k}={dq[k]}" for k in HARD_CHECKS)
               + f"; missing candles {dq['missing_candles']} ({_pct(dq['missing_candle_share'])})", "",
               f"Expiry: {json.dumps(s6._jsonable(P['expiry']), default=str)}", ""]
    md += ["No strategy built, no ML trained, no thresholds tuned, no data repaired. October not loaded. Production "
           "unchanged; live orders 0."]
    return "\n".join(md) + "\n"
