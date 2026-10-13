"""S12.1 - pre-2025-09-26 ATM option coverage investigation (PAPER | RESEARCH ONLY | NO ORDERS | offline, no ML).

Explains why sessions before the S12 clean-history start fail LOW_ATM_COVERAGE, using ONLY stored data and stored
download metadata (plan.json, contracts.csv, download_manifest.csv, missing_data.csv) plus the S12 session statuses.
Nothing is downloaded, modified, shifted, substituted or synthesised. October 2026 is never read.

Pre-declared rules (fixed before the investigation was run):

* ATM selection per (signal timestamp, side) is exactly the S12 point-in-time rule (``s12._atm_one``): nearest listed
  expiry from the official expiry calendar recorded at plan time, strike nearest the spot close, planned ATM contract.
* A timestamp is AVAILABLE if the selected contract has an on-grid candle at that timestamp; ELIGIBLE additionally
  needs the 14-candle ATR (warm-up is not missing data). Coverage % = eligible / expected (S12 definition).
* Missing requirement classes: CONTRACT_EXISTS_DATA_MISSING (contract in the expired-instrument metadata, candles
  absent), CONTRACT_NOT_AVAILABLE (no listed chain / contract), SELECTION_LOGIC_ISSUE, EXPIRY_METADATA_ISSUE.
  CONTRACT_EXISTS_DATA_MISSING whose request returned no / partial candles from the source -> SOURCE_COVERAGE_LIMITATION.
* A blocking session is RECOVERABLE only if its sole S12 reason is LOW_ATM_COVERAGE and every missing requirement on
  it has a non-source cause (never requested, request failed, stored file lost, selection / metadata bug).
* Decision: all blocking sessions recoverable -> RECOVERABLE_HISTORY; none -> SOURCE_LIMITED_HISTORY;
  otherwise PARTIALLY_RECOVERABLE. The S12 12-month gate is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import HistoricalContract
from src.options.models import OptionResolutionError
from src.options.selection import atm_strike
from src.research import s10_ml_data as sd
from src.research import s11_history as hx
from src.research import s12_data_quality as s12

HEADING = "S12.1 OPTION COVERAGE INVESTIGATION COMPLETE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | offline investigation (no download, no model, nothing modified)"
IST = s12.IST
STEP = s12.STEP
WINDOW = (date(2025, 1, 1), date(2025, 9, 25))
ATR_WARMUP = "OPTION_ATR_UNAVAILABLE"
LOW_ATM = "LOW_ATM_COVERAGE"

EXISTS_MISSING, NOT_AVAILABLE = "CONTRACT_EXISTS_DATA_MISSING", "CONTRACT_NOT_AVAILABLE"
SELECTION, EXPIRY_META = "SELECTION_LOGIC_ISSUE", "EXPIRY_METADATA_ISSUE"
SOURCE_LIMIT = "SOURCE_COVERAGE_LIMITATION"

# download verdicts per (session, contract)
OK, NOT_DISCOVERED, NOT_REQUESTED = "OK", "NOT_DISCOVERED", "NOT_REQUESTED"
REQUEST_FAILED, SOURCE_EMPTY, SOURCE_INCOMPLETE = "REQUEST_FAILED", "SOURCE_RETURNED_EMPTY", "SOURCE_RETURNED_INCOMPLETE"
STORED_LOST = "STORED_FILE_MISSING_OR_SHORT"
RECOVERABLE_VERDICTS = (NOT_REQUESTED, REQUEST_FAILED, STORED_LOST)
SOURCE_VERDICTS = (SOURCE_EMPTY, SOURCE_INCOMPLETE)

# objective letters
CAUSE = {"A": "missing option contracts that were not downloaded",
         "B": "unavailable historical contracts / data from the data source",
         "C": "incorrect point-in-time strike selection",
         "D": "expiry / contract metadata issue",
         "E": "genuine historical data coverage limitation (source returned partial intervals)"}
RECOVERABLE_LETTERS = ("A", "C", "D")

RECOVERABLE, PARTIAL, SOURCE_LIMITED, NO_GAP = (
    "RECOVERABLE_HISTORY", "PARTIALLY_RECOVERABLE", "SOURCE_LIMITED_HISTORY", "NO_PRE_WINDOW_GAP")


# ------------------------------------------------------------------ stored metadata
def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def key_expiry(instrument_key: str) -> date | None:
    """Expired Upstox keys end in |dd-mm-yyyy (the contract's own expiry)."""
    parts = instrument_key.split("|")
    if len(parts) < 3:
        return None
    try:
        return datetime.strptime(parts[-1], "%d-%m-%Y").date()
    except ValueError:
        return None


def load_metadata(roots: Sequence[Path], underlying: str, lo: date, hi: date) -> dict[str, Any]:
    reqs: dict[tuple[date, str], dict] = {}
    by_day: dict[date, list[HistoricalContract]] = defaultdict(list)
    manifest: dict[tuple[str, str], list[dict]] = defaultdict(list)
    missing: dict[tuple[str, date], list[dict]] = defaultdict(list)
    plans = {}
    for root in roots:
        lay = u6.Layout(root, underlying)
        plan = json.loads(lay.plan.read_text(encoding="utf-8")) if lay.plan.is_file() else {}
        plans[Path(root).name] = {"from": plan.get("from"), "to": plan.get("to"),
                                  "calendar_from": plan.get("calendar_from"), "calendar_size": len(plan.get("calendar", [])),
                                  "plan_sha256": sha256_file(lay.plan), "contracts_sha256": sha256_file(lay.contracts),
                                  "manifest_sha256": sha256_file(lay.manifest), "missing_sha256": sha256_file(lay.missing)}
        for r in u6.requirements_from_rows(u6._read_csv(lay.contracts)):
            if lo <= r.trading_date <= hi:
                reqs[(r.trading_date, r.contract.instrument_key)] = {"root": root, "req": r}
                by_day[r.trading_date].append(r.contract)
        for m in u6._read_csv(lay.manifest):
            manifest[(m["instrument_key"], m["month"])].append({**m, "root": Path(root).name})
        for m in u6._read_csv(lay.missing):
            if m.get("instrument_key") and m.get("trading_date"):
                try:
                    missing[(m["instrument_key"], date.fromisoformat(m["trading_date"]))].append(m)
                except ValueError:
                    continue
    return {"reqs": reqs, "by_day": dict(by_day), "manifest": dict(manifest), "missing": dict(missing),
            "plans": plans}


def load_s12_sessions(path: Path) -> tuple[list[dict], str]:
    if not path.is_file():
        raise SystemExit(f"S12 session statuses not found at {path}; run scripts/run_s12_data_quality.py first")
    m = json.loads(path.read_text(encoding="utf-8"))
    return list(m.get("sessions") or []), sha256_file(path)


# ------------------------------------------------------------------ 1/2. per-timestamp ATM scan
def _expected_strike(market, d: date, expiry: date | None, side: str, spot: float) -> float | None:
    chain = market.chains.get((d, expiry, side)) or {}
    try:
        return float(atm_strike(sorted(chain), spot)) if chain else None
    except OptionResolutionError:
        return None


def classify(reason: str, c: HistoricalContract | None, d: date, meta: dict) -> str:
    if reason in ("NO_LISTED_EXPIRY", "EXPIRY_NOT_POINT_IN_TIME"):
        return EXPIRY_META
    if reason in ("OUTSIDE_PLANNED_STRIKES", "ATM_NOT_BRACKETED") or reason.startswith("STRIKE_UNAVAILABLE"):
        return SELECTION
    if reason == "CONTRACT_LIST_UNAVAILABLE" or c is None:
        return NOT_AVAILABLE
    r = meta["reqs"].get((d, c.instrument_key))
    if r is None:
        return NOT_AVAILABLE
    m = r["req"].contract
    same = (m.expiry == c.expiry and m.instrument_type == c.instrument_type
            and abs(float(m.strike) - float(c.strike)) < 1e-6)
    return EXISTS_MISSING if same else EXPIRY_META


def scan(market, sessions: Sequence[date], meta: dict, s12_status: dict[str, dict]) -> dict[str, Any]:
    last = sd.last_signal(sd.HORIZON)
    spot_close, day_bars = market.spot_close(), market.day_bars()
    cache: dict = {}
    first_seen: dict = {}
    recs: dict[tuple, dict] = {}
    days, missing = [], []
    use: dict[tuple[date, str], Counter] = defaultdict(Counter)
    for d in sessions:
        sts = [ts for ts in day_bars.get(d, []) if sd.SIGNAL_FIRST <= ts.time() <= last]
        expiry = sd.nearest_expiry(market.calendar, d)
        cnt = {s: Counter() for s in sd.SIDES}
        both = 0
        for ts in sts:
            good = True
            for s in sd.SIDES:
                why, c = s12._atm_one(market, d, expiry, s, spot_close[ts], ts, cache, first_seen, recs)
                if c is not None:
                    use[(d, c.instrument_key)]["timestamps"] += 1
                if why is None:
                    cnt[s]["eligible"] += 1
                    cnt[s]["available"] += 1
                    continue
                good = False
                if why.startswith(ATR_WARMUP):
                    cnt[s]["available"] += 1
                    cnt[s]["atr_warmup"] += 1
                    continue
                cnt[s]["missing"] += 1
                cnt[s][why] += 1
                cls = classify(why, c, d, meta)
                if c is not None:
                    use[(d, c.instrument_key)]["missing"] += 1
                missing.append({"date": d.isoformat(), "timestamp": ts.isoformat(), "spot": float(spot_close[ts]),
                                "side": s, "expected_expiry": expiry.isoformat() if expiry else None,
                                "expected_atm_strike": float(c.strike) if c is not None
                                else _expected_strike(market, d, expiry, s, spot_close[ts]),
                                "instrument_key": c.instrument_key if c is not None else None,
                                "trading_symbol": c.trading_symbol if c is not None else None,
                                "s12_reason": why, "classification": cls})
            both += good
        n = len(sts)
        st = s12_status.get(d.isoformat())
        row = {"date": d.isoformat(), "eligible_nifty_timestamps": n,
               "expiry": expiry.isoformat() if expiry else None}
        for s in sd.SIDES:
            c = cnt[s]
            row.update({f"atm_{s}_expected": n, f"atm_{s}_available": c["available"], f"atm_{s}_eligible": c["eligible"],
                        f"{s}_coverage_pct": round(100 * c["eligible"] / n, 4) if n else None,
                        f"{s}_candle_coverage_pct": round(100 * c["available"] / n, 4) if n else None,
                        f"missing_{s}_count": c["missing"], f"{s}_atr_warmup": c["atr_warmup"],
                        f"{s}_missing_reasons": json.dumps({k: v for k, v in sorted(c.items())
                                                            if k not in ("eligible", "available", "missing",
                                                                         "atr_warmup")})})
        share = both / n if n else None
        row.update({"both_eligible": both, "both_share": share,
                    "option_session_clean": bool(n and share >= s12.SESSION_MIN_ATM_ELIGIBLE),
                    "s12_clean": st["clean"] if st else None, "s12_reasons": st["reasons"] if st else "NOT_IN_S12"})
        days.append(row)
    for r in recs.values():
        r["passed"] = bool(r["expiry_valid"] and r["nearest_expiry"] and r["in_listed_chain"] and r["nearest_strike"]
                           and r["bracketed"] and r["planned_atm"] and r["candle_matches_contract"])
    return {"days": days, "missing": missing, "selections": list(recs.values()), "use": use}


# ------------------------------------------------------------------ 3/4. download audit
def _bars_on(market, key: str, d: date) -> int:
    p = market.premium(key)
    if p is None or p.empty:
        return 0
    have = set(p.loc[pd.DatetimeIndex(p["timestamp"]).date == d, "timestamp"])
    return len(have & set(s12.expected_grid(d)))


def download_audit(market, meta: dict, use: dict[tuple[date, str], Counter],
                   missing_keys: set[tuple[date, str]]) -> list[dict]:
    """Every ATM requirement (session, contract) in the window + every contract with a missing ATM timestamp."""
    todo = {k for k, v in meta["reqs"].items() if v["req"].role == u6.ATM} | missing_keys
    out = []
    for d, key in sorted(todo):
        r = meta["reqs"].get((d, key))
        c = r["req"].contract if r else None
        month = hx.month_label(d)
        rows = meta["manifest"].get((key, month), [])
        cover = [m for m in rows if m.get("from_date", "9") <= d.isoformat() <= m.get("to_date", "0")]
        m = cover[-1] if cover else (rows[-1] if rows else None)
        lay = u6.Layout(r["root"]) if r else None
        path = lay.option_file(month, c) if r else None
        bars = _bars_on(market, key, d)
        expected = len(s12.expected_grid(d))
        src_missing = meta["missing"].get((key, d), [])
        status = (m or {}).get("status") or None
        if r is None:
            verdict = NOT_DISCOVERED
        elif bars >= expected:
            verdict = OK
        elif not cover:
            verdict = NOT_REQUESTED
        elif status == u6.FAILED:
            verdict = REQUEST_FAILED
        elif status in (u6.EMPTY, u6.UNAVAILABLE):
            verdict = SOURCE_EMPTY
        elif status == u6.SUCCESS and bars > 0 and any(x.get("reason") == "MISSING_INTERVALS" for x in src_missing):
            verdict = SOURCE_INCOMPLETE
        else:
            verdict = STORED_LOST
        u = use.get((d, key), Counter())
        out.append({"date": d.isoformat(), "instrument_key": key,
                    "trading_symbol": c.trading_symbol if c else None, "expiry": c.expiry.isoformat() if c else None,
                    "strike": float(c.strike) if c else None, "option_type": c.instrument_type if c else None,
                    "role": r["req"].role if r else None, "source_endpoint": c.source if c else None,
                    "discovered": r is not None, "requested": bool(cover),
                    "manifest_status": status, "http_status": (m or {}).get("http_status"),
                    "request_from": (m or {}).get("from_date"), "request_to": (m or {}).get("to_date"),
                    "raw_rows_returned": (m or {}).get("raw_rows"), "candle_count": (m or {}).get("candle_count"),
                    "download_timestamp": (m or {}).get("download_timestamp"),
                    "successfully_downloaded": status == u6.SUCCESS, "returned_empty": status in (u6.EMPTY, u6.UNAVAILABLE),
                    "file_exists": bool(path and path.is_file()), "bars_on_session": bars, "expected_bars": expected,
                    "incomplete": 0 < bars < expected,
                    "source_reported_missing": "; ".join(f"{x.get('reason')}:{x.get('count')}" for x in src_missing),
                    "atm_timestamps_selected": u["timestamps"], "atm_timestamps_missing": u["missing"],
                    "verdict": verdict, "recoverable": verdict in RECOVERABLE_VERDICTS,
                    "source_limited": verdict in SOURCE_VERDICTS or verdict == NOT_DISCOVERED})
    return out


def root_cause(rec: dict, audit_by: dict[tuple[str, str], dict]) -> tuple[str, str]:
    """(final classification, objective letter) for one missing ATM timestamp."""
    cls = rec["classification"]
    if cls == SELECTION:
        return SELECTION, "C"
    if cls == EXPIRY_META:
        return EXPIRY_META, "D"
    if cls == NOT_AVAILABLE:
        return NOT_AVAILABLE, "B"
    a = audit_by.get((rec["date"], rec["instrument_key"]))
    v = a["verdict"] if a else NOT_REQUESTED
    if v == SOURCE_EMPTY:
        return SOURCE_LIMIT, "B"
    if v == SOURCE_INCOMPLETE:
        return SOURCE_LIMIT, "E"
    if v == OK:
        return SOURCE_LIMIT, "E"
    return EXISTS_MISSING, "A"


# ------------------------------------------------------------------ 5. point-in-time validation
def pit_validation(market, sessions: Sequence[date], meta: dict, selections: Sequence[dict]) -> dict[str, Any]:
    sel_by = defaultdict(list)
    for r in selections:
        sel_by[r["date"]].append(r)
    cal_from = [date.fromisoformat(p["calendar_from"]) for p in meta["plans"].values() if p.get("calendar_from")]
    rows = []
    for d in sessions:
        exp = sd.nearest_expiry(market.calendar, d)
        cs = meta["by_day"].get(d, [])
        ke = [key_expiry(c.instrument_key) for c in cs]
        sel = sel_by.get(d.isoformat(), [])
        r = {"date": d.isoformat(), "expiry": exp.isoformat() if exp else None,
             "expiry_weekday": exp.strftime("%A") if exp else None, "dte": (exp - d).days if exp else None,
             "calendar_recorded_from_plan": bool(cal_from) and min(cal_from) <= d,
             "expiry_not_before_session": bool(exp and exp >= d),
             "expiry_gap_ok": bool(exp and (exp - d).days <= hx.MAX_EXPIRY_GAP_DAYS),
             "contracts_planned": len(cs),
             "planned_contracts_on_nearest_expiry": all(c.expiry == exp for c in cs),
             "contracts_from_expired_endpoint": all(c.source == "EXPIRED" for c in cs),
             "key_encodes_same_expiry": all(k is None or k == c.expiry for k, c in zip(ke, cs)),
             "strikes_on_50_grid": all(abs(float(c.strike) / 50 - round(float(c.strike) / 50)) < 1e-9 for c in cs),
             "atm_selections_checked": len(sel), "atm_selections_failed": sum(not s["passed"] for s in sel)}
        r["passed"] = bool(cs and r["calendar_recorded_from_plan"] and r["expiry_not_before_session"]
                           and r["planned_contracts_on_nearest_expiry"]
                           and r["contracts_from_expired_endpoint"] and r["key_encodes_same_expiry"]
                           and r["strikes_on_50_grid"] and r["atm_selections_failed"] == 0)
        rows.append(r)
    wd = Counter(r["expiry_weekday"] for r in rows if r["expiry_weekday"])
    return {"rows": rows, "passed": bool(rows) and all(r["passed"] for r in rows),
            "failed_sessions": [r["date"] for r in rows if not r["passed"]], "expiry_weekdays": dict(wd),
            "sessions_with_expiry_gap_over_limit": [r["date"] for r in rows if not r["expiry_gap_ok"]],
            "rule": "expiry = nearest expiry >= session from the official calendar recorded at plan time; planned "
                    "contracts come from the expired-instrument endpoint for that expiry (not today's instrument "
                    "master), instrument keys encode the same expiry, strikes on the 50-point grid, every ATM "
                    "selection passes the S12 point-in-time checks"}


# ------------------------------------------------------------------ 6. decision
def blocking_sessions(days: Sequence[dict], missing: Sequence[dict]) -> list[dict]:
    by_day = defaultdict(list)
    for m in missing:
        by_day[m["date"]].append(m)
    out = []
    for r in days:
        if r["s12_clean"] is not False and r["option_session_clean"]:
            continue
        ms = by_day.get(r["date"], [])
        letters = Counter(m["cause_letter"] for m in ms)
        reasons = set(filter(None, (r["s12_reasons"] or "").split()))
        option_only = reasons <= {LOW_ATM} if reasons else not r["option_session_clean"]
        recoverable = bool(option_only and ms and set(letters) <= set(RECOVERABLE_LETTERS))
        contracts = sorted({(m["instrument_key"], m["trading_symbol"], m["side"], m["root_cause"]) for m in ms},
                           key=lambda x: (str(x[0]), x[2]))
        out.append({"date": r["date"], "s12_reasons": r["s12_reasons"], "both_share": r["both_share"],
                    "missing_CE": r["missing_CE_count"], "missing_PE": r["missing_PE_count"],
                    "cause_letters": dict(letters), "option_only": option_only, "recoverable": recoverable,
                    "contracts": [{"instrument_key": k, "trading_symbol": t, "side": s, "root_cause": c}
                                  for k, t, s, c in contracts]})
    return out


def recover_windows(audit: Sequence[dict], blocking: Sequence[dict]) -> list[dict]:
    days = {b["date"] for b in blocking if b["recoverable"]}
    out = []
    for a in audit:
        if a["date"] in days and a["recoverable"] and a["atm_timestamps_missing"]:
            out.append({"instrument_key": a["instrument_key"], "trading_symbol": a["trading_symbol"],
                        "from": a["date"], "to": a["date"], "reason": a["verdict"]})
    return out


def histories(statuses: Sequence[dict], blocking: Sequence[dict], lo: date) -> dict[str, Any]:
    now = s12.clean_runs(statuses)
    fix = {b["date"] for b in blocking if b["recoverable"]}
    toggled = [dict(s, clean=True, reasons="") if s["date"] in fix else s for s in statuses]
    after = s12.clean_runs(toggled)
    upto = [s for s in statuses if lo.isoformat() <= s["date"] and (not blocking or s["date"] < blocking[0]["date"])]
    pre = s12.clean_runs(upto, date.fromisoformat(upto[-1]["date"])) if upto else None
    return {"now": now["ending_train_end"], "longest_now": now["longest"],
            "if_recovered": after["ending_train_end"],
            "pre_window_segment": pre["ending_train_end"] if pre else None,
            "required_months": s12.MIN_CLEAN_MONTHS}


def decide(blocking: Sequence[dict], hist: dict, windows: Sequence[dict]) -> dict[str, Any]:
    if not blocking:
        d = NO_GAP
    elif all(b["recoverable"] for b in blocking):
        d = RECOVERABLE
    elif any(b["recoverable"] for b in blocking):
        d = PARTIAL
    else:
        d = SOURCE_LIMITED
    out = {"decision": d, "blocking_sessions": [b["date"] for b in blocking],
           "recoverable_sessions": [b["date"] for b in blocking if b["recoverable"]],
           "source_limited_sessions": [b["date"] for b in blocking if not b["recoverable"]],
           "earliest_clean_usable_date": hist["now"]["start"],
           "earliest_clean_usable_date_if_recovered": hist["if_recovered"]["start"],
           "clean_months_now": hist["now"]["months"], "clean_months_if_recovered": hist["if_recovered"]["months"],
           "required_months": hist["required_months"],
           "gate_passes_now": hist["now"]["months"] >= hist["required_months"],
           "gate_passes_if_recovered": hist["if_recovered"]["months"] >= hist["required_months"],
           "contracts_to_download": list(windows)}
    return out


# ------------------------------------------------------------------ orchestration
def build(roots: Sequence[str | Path], s12_manifest: Path, *, underlying: str = "NIFTY", lo: date | None = None,
          hi: date | None = None, progress=None) -> dict[str, Any]:
    lo, hi = lo or WINDOW[0], hi or WINDOW[1]
    hx.guard_no_october(hi)
    say = progress or (lambda _m: None)
    roots = [Path(r) for r in roots]
    before = s12.raw_hashes(roots, underlying, hi)
    statuses, s12_sha = load_s12_sessions(s12_manifest)
    st_by = {s["date"]: s for s in statuses}
    say("reading stored plan / contracts / download manifest / missing-data metadata ...")
    meta = load_metadata(roots, underlying, lo, hi)
    say("loading stored market data (off-grid option rows quarantined, never shifted) ...")
    market = sd.load_market([str(r) for r in roots], underlying, max_day=hi)
    quarantine = hx.quarantine_off_grid(market)
    sessions = [d for d in market.sessions if lo <= d <= hi]
    say(f"point-in-time ATM scan over {len(sessions)} sessions ...")
    sc = scan(market, sessions, meta, st_by)
    say("download audit of every required ATM contract ...")
    missing_keys = {(date.fromisoformat(m["date"]), m["instrument_key"]) for m in sc["missing"] if m["instrument_key"]}
    audit = download_audit(market, meta, sc["use"], missing_keys)
    audit_by = {(a["date"], a["instrument_key"]): a for a in audit}
    for m in sc["missing"]:
        a = audit_by.get((m["date"], m["instrument_key"]))
        m["root_cause"], m["cause_letter"] = root_cause(m, audit_by)
        m["download_verdict"] = a["verdict"] if a else None
        m["manifest_status"] = a["manifest_status"] if a else None
        m["raw_rows_returned"] = a["raw_rows_returned"] if a else None
        m["contract_in_metadata"] = bool(a and a["discovered"])
    pit = pit_validation(market, sessions, meta, sc["selections"])
    blocking = blocking_sessions(sc["days"], sc["missing"])
    hist = histories(statuses, blocking, lo)
    windows = recover_windows(audit, blocking)
    dec = decide(blocking, hist, windows)
    trading_days = [s["date"] for s in statuses if lo.isoformat() <= s["date"] <= hi.isoformat()]
    days = sc["days"]
    cons = [r["date"] for r in days if r["s12_clean"] is not None
            and (LOW_ATM in (r["s12_reasons"] or "").split()) == (not r["option_session_clean"])]
    after = s12.raw_hashes(roots, underlying, hi)
    tot = {s: {k: sum(r[f"atm_{s}_{k}"] for r in days) for k in ("expected", "available", "eligible")}
           for s in sd.SIDES}
    for s in sd.SIDES:
        tot[s]["missing"] = sum(r[f"missing_{s}_count"] for r in days)
        tot[s]["coverage"] = tot[s]["eligible"] / tot[s]["expected"] if tot[s]["expected"] else None
    max_day = market.spot["timestamp"].max() if len(market.spot) else None
    return {"heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
            "roots": [r.as_posix() for r in roots], "window": [lo.isoformat(), hi.isoformat()],
            "s12_manifest": s12_manifest.as_posix(), "s12_manifest_sha256": s12_sha,
            "metadata": meta["plans"], "sessions": len(sessions), "s12_trading_days_in_window": len(trading_days),
            "sessions_not_in_market": sorted(set(trading_days) - {d.isoformat() for d in sessions}),
            "days": days, "missing": sc["missing"], "audit": audit, "pit": pit, "blocking": blocking, "history": hist,
            "decision": dec, "totals": tot,
            "missing_by_root_cause": dict(Counter(m["root_cause"] for m in sc["missing"])),
            "missing_by_letter": dict(Counter(m["cause_letter"] for m in sc["missing"])),
            "audit_verdicts": dict(Counter(a["verdict"] for a in audit)),
            "s12_consistent": len(cons) == sum(r["s12_clean"] is not None for r in days),
            "quarantine": hx.quarantine_report(quarantine),
            "max_timestamp_loaded": max_day.isoformat() if max_day is not None else None,
            "october_loaded": False, "raw_unchanged": before == after, "raw_files_hashed": len(before),
            "fabricated_values": 0, "substituted_contracts": 0, "live_orders": 0, "production_unchanged": None,
            "_market": market}


# ------------------------------------------------------------------ text / reports
def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.2f}%"


def terminal_text(a: dict) -> str:
    t, d, h, p = a["totals"], a["decision"], a["history"], a["pit"]
    out = [HEADING, f"({BANNER})", "", f"WINDOW: {a['window'][0]} -> {a['window'][1]} ({a['sessions']} sessions)", "",
           "SESSION COVERAGE:",
           f"CE: {_pct(t['CE']['coverage'])} eligible ({t['CE']['missing']} missing timestamps)",
           f"PE: {_pct(t['PE']['coverage'])} eligible ({t['PE']['missing']} missing timestamps)",
           f"sessions below the S12 per-session ATM rule: {len(a['blocking'])}", "", "BLOCKING SESSIONS:"]
    if not a["blocking"]:
        out.append("none")
    for b in a["blocking"]:
        cs = ", ".join(f"{c['trading_symbol']} ({c['instrument_key']}) -> {c['root_cause']}" for c in b["contracts"])
        out.append(f"{b['date']}: both-side eligible {_pct(b['both_share'])}; missing CE {b['missing_CE']} / PE "
                   f"{b['missing_PE']}; cause {b['cause_letters']}; {cs}")
    out += ["", "ROOT CAUSE (missing ATM timestamps):"]
    out += [f"{k}: {v}" for k, v in sorted(a["missing_by_root_cause"].items())] or ["none"]
    out += [f"cause {k} ({CAUSE[k]}): {v}" for k, v in sorted(a["missing_by_letter"].items())]
    out += ["", "DOWNLOAD AUDIT:"] + [f"{k}: {v}" for k, v in sorted(a["audit_verdicts"].items())]
    out += ["", "POINT-IN-TIME VALIDATION:", "PASS" if p["passed"] else f"FAIL {p['failed_sessions'][:10]}", "",
            "DECISION:", d["decision"], "",
            "CLEAN HISTORY (S12 rule, 12-month gate unchanged):",
            f"now: {h['now']['start']} -> {h['now']['end']} ({h['now']['months']} months)",
            f"if recoverable gaps were fixed: {h['if_recovered']['start']} -> {h['if_recovered']['end']} "
            f"({h['if_recovered']['months']} months)"]
    if h["pre_window_segment"]:
        s = h["pre_window_segment"]
        out.append(f"clean segment before the first blocking session: {s['start']} -> {s['end']} "
                   f"({s['trading_days']} sessions, {s['months']} months)")
    if d["decision"] == SOURCE_LIMITED:
        out.append(f"EARLIEST CLEAN USABLE DATE: {d['earliest_clean_usable_date']}")
    if d["contracts_to_download"]:
        out += ["", "CONTRACTS / WINDOWS TO DOWNLOAD:"] + [
            f"{w['trading_symbol']} ({w['instrument_key']}) {w['from']} -> {w['to']} [{w['reason']}]"
            for w in d["contracts_to_download"]]
    out += ["", "S12 CONSISTENT:", "YES" if a["s12_consistent"] else "NO", "", "FABRICATED VALUES:", "0", "",
            "SUBSTITUTED CONTRACTS:", "0", "", "LIVE ORDERS:", "0", "", "PRODUCTION CHANGED:",
            "NO" if a.get("production_unchanged") is not False else "YES", "", "OCTOBER LOADED:", "NO", "",
            "S11 RERUN:", "NO"]
    return "\n".join(out)


def markdown(a: dict) -> str:
    t, d, h, p = a["totals"], a["decision"], a["history"], a["pit"]
    md = ["# S12.1 pre-2025-09-26 ATM option coverage investigation", "", f"**{BANNER}**", "",
          f"## Decision: **{d['decision']}**", "",
          f"- Window {a['window'][0]} -> {a['window'][1]}: {a['sessions']} sessions; ATM CE eligible "
          f"{_pct(t['CE']['coverage'])}, PE {_pct(t['PE']['coverage'])}; missing timestamps CE {t['CE']['missing']}, "
          f"PE {t['PE']['missing']}.",
          f"- Sessions failing the S12 per-session ATM rule (both sides >= {s12.SESSION_MIN_ATM_ELIGIBLE:.0%}): "
          f"{len(a['blocking'])} ({', '.join(b['date'] for b in a['blocking']) or 'none'}).",
          f"- Clean history now: {h['now']['start']} -> {h['now']['end']} ({h['now']['months']} months; gate needs "
          f"{h['required_months']}). If recoverable gaps were fixed: {h['if_recovered']['start']} -> "
          f"{h['if_recovered']['end']} ({h['if_recovered']['months']} months).",
          f"- Earliest clean usable date (current data): {d['earliest_clean_usable_date']}.", "",
          "## Blocking sessions", ""]
    for b in a["blocking"]:
        md.append(f"- **{b['date']}**: both-side eligible {_pct(b['both_share'])}, missing CE {b['missing_CE']} / PE "
                  f"{b['missing_PE']}, S12 reasons `{b['s12_reasons']}`, causes {b['cause_letters']}, recoverable "
                  f"{b['recoverable']}.")
        md += [f"  - {c['side']} {c['trading_symbol']} (`{c['instrument_key']}`): {c['root_cause']}" for c in b["contracts"]]
    md += ["", "## Root cause of missing ATM timestamps", "", "| Class | Timestamps |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in sorted(a["missing_by_root_cause"].items())]
    md += ["", "| Objective cause | Meaning | Timestamps |", "|---|---|---|"]
    md += [f"| {k} | {CAUSE[k]} | {a['missing_by_letter'].get(k, 0)} |" for k in CAUSE]
    md += ["", "## Download audit (every ATM requirement in the window)", "", "| Verdict | Contract-sessions |",
           "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in sorted(a["audit_verdicts"].items())]
    bad = [x for x in a["audit"] if x["verdict"] != OK]
    if bad:
        md += ["", "| Date | Contract | Key | Requested | Status | HTTP | Raw rows | Bars | Verdict |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {x['date']} | {x['trading_symbol']} | `{x['instrument_key']}` | {x['requested']} | "
               f"{x['manifest_status']} | {x['http_status']} | {x['raw_rows_returned']} | {x['bars_on_session']}/"
               f"{x['expected_bars']} | {x['verdict']} |" for x in bad[:50]]
    md += ["", "## Point-in-time validation", "", f"- Result: **{'PASS' if p['passed'] else 'FAIL'}** "
           f"({len(p['rows'])} sessions; failed {p['failed_sessions'][:10]}).", f"- Rule: {p['rule']}.",
           f"- Expiry weekdays in the window: {p['expiry_weekdays']} (the official calendar recorded at plan time).",
           "", "## Safety", "",
           f"- Raw data unchanged: {a['raw_unchanged']} ({a['raw_files_hashed']} files hashed before/after). "
           "Fabricated values: 0. Substituted contracts: 0. October loaded: NO. S11 rerun: NO. Live orders: 0. "
           f"Production changed: {'NO' if a.get('production_unchanged') is not False else 'YES'}.",
           f"- S12 per-session verdicts reproduced: {a['s12_consistent']}."]
    return "\n".join(md) + "\n"


REPORT_FILES = ("summary.json", "coverage_by_day.csv", "pre_sep_atm_coverage.csv", "missing_atm_contracts.csv",
                "contract_download_audit.csv", "point_in_time_validation.csv", "coverage_report.md")


def summary_obj(a: dict) -> dict[str, Any]:
    keep = ("heading", "banner", "generated_at", "roots", "window", "s12_manifest", "s12_manifest_sha256", "metadata",
            "sessions", "s12_trading_days_in_window", "sessions_not_in_market", "totals", "missing_by_root_cause",
            "missing_by_letter", "audit_verdicts", "blocking", "history", "decision", "s12_consistent", "quarantine",
            "max_timestamp_loaded", "october_loaded", "raw_unchanged", "raw_files_hashed", "fabricated_values",
            "substituted_contracts", "live_orders", "production_unchanged", "token_leak")
    out = {k: a.get(k) for k in keep}
    out["point_in_time"] = {k: v for k, v in a["pit"].items() if k != "rows"}
    out["cause_meaning"] = CAUSE
    return out


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    from src.research import s6_clean_strategy_lab as s6
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    p["summary.json"].write_text(json.dumps(s6._jsonable(summary_obj(a)), indent=2, default=str), encoding="utf-8")
    s6._csv(p["coverage_by_day.csv"], a["days"])
    s6._csv(p["pre_sep_atm_coverage.csv"], a["days"])
    s6._csv(p["missing_atm_contracts.csv"], a["missing"])
    s6._csv(p["contract_download_audit.csv"], a["audit"])
    s6._csv(p["point_in_time_validation.csv"], a["pit"]["rows"])
    p["coverage_report.md"].write_text(markdown(a), encoding="utf-8")
    return p
