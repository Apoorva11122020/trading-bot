"""Option-contract comparability for S2 research (analysis only, PAPER, no orders).

The live S2 setup trades the nearest-expiry ATM option. Historical research can
only use contracts whose real candles are available, which may have a very
different days-to-expiry (DTE). This module classifies every evaluated signal
by DTE, separates samples comparable to the live setup from non-comparable
ones, and refuses rows that are not backed by real option candles. Missing
expired-contract data is reported as unavailable, never synthesised.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

from src.backtest.live_vs_historical import AMBIGUOUS, LOSS, PENDING, WIN, group_metrics
from src.historical_data.models import HistoricalDataError
from src.options.models import OptionContract, OptionResolutionError, OptionType
from src.options.selection import atm_strike

REAL_PREMIUM_SOURCES = frozenset({"STORED_UPSTOX_V3", "UPSTOX_V3_HISTORICAL", "UPSTOX_V3_INTRADAY",
                                  "UPSTOX_EXPIRED_HISTORICAL"})
DTE_BUCKETS = ((0, 0, "0 DTE"), (1, 1, "1 DTE"), (2, 4, "2-4 DTE"), (5, 7, "5-7 DTE"), (8, 14, "8-14 DTE"),
               (15, 21, "15-21 DTE"), (22, 10_000, "22+ DTE"))
LIVE_REFERENCE = {"underlying": "NIFTY", "interval_minutes": 5, "dte": 1, "expiry": "2026-10-06",
                  "date": "2026-10-05", "contract_rule": "nearest expiry, ATM, BUY->CE / SELL->PE"}
CSV_FIELDS = ("sample", "comparable", "session_date", "signal_time", "direction", "contract", "expiry",
              "calendar_dte", "trading_dte", "dte_bucket", "premium_source", "premium_candles", "status", "reason",
              "exit_reason", "entry_price", "stop_loss", "target", "premium_atr", "premium_atr_pct", "stop_pct",
              "target_pct", "exit_price", "exit_time", "pnl", "r_multiple", "holding_minutes")


class FabricatedDataError(ValueError):
    """Raised when an evaluated row is not backed by real option candles."""


# ----------------------------------------------------------------- DTE
def calendar_dte(expiry: date, signal_day: date) -> int:
    dte = (expiry - signal_day).days
    if dte < 0:
        raise ValueError(f"expiry {expiry} is before signal day {signal_day}")
    return dte


def trading_dte(expiry: date, signal_day: date, holidays: Iterable[date] = ()) -> int:
    """Trading sessions after the signal day up to and including expiry (weekends/holidays skipped)."""
    calendar_dte(expiry, signal_day)
    closed = set(holidays)
    day, count = signal_day, 0
    while day < expiry:
        day += timedelta(days=1)
        if day.weekday() < 5 and day not in closed:
            count += 1
    return count


def dte_bucket(dte: int | None) -> str | None:
    if dte is None:
        return None
    for low, high, label in DTE_BUCKETS:
        if low <= dte <= high:
            return label
    return None


def is_comparable(dte: int | None, live_dte: int = LIVE_REFERENCE["dte"]) -> bool:
    return dte is not None and dte == live_dte


# ------------------------------------------------------------ data guards
def annotate(rows: list[dict[str, Any]], sources: dict[str, tuple[str, int]], sample: str,
             holidays: Iterable[date] = (), live_dte: int = LIVE_REFERENCE["dte"]) -> list[dict[str, Any]]:
    holidays = tuple(holidays)
    for row in rows:
        row["sample"] = sample
        key = row.get("instrument_key")
        source, count = sources.get(key, (None, 0)) if key else (None, 0)
        row["premium_source"], row["premium_candles"] = source, count
        if row.get("expiry"):
            expiry, day = date.fromisoformat(row["expiry"]), date.fromisoformat(row["session_date"])
            row["calendar_dte"] = calendar_dte(expiry, day)
            row["trading_dte"] = trading_dte(expiry, day, holidays)
            row["dte_bucket"] = dte_bucket(row["calendar_dte"])
        row["comparable"] = is_comparable(row.get("calendar_dte"), live_dte)
    return rows


def has_real_levels(row: dict[str, Any]) -> bool:
    return row.get("entry_price") is not None


def verify_real_data(rows: Iterable[dict[str, Any]]) -> int:
    """Every row with levels/outcome must come from a real premium source with candles."""
    checked = 0
    for row in rows:
        if not has_real_levels(row):
            continue
        parts = str(row.get("premium_source") or "").split("+")
        if not all(p in REAL_PREMIUM_SOURCES for p in parts) or not row.get("premium_candles"):
            raise FabricatedDataError(f"{row.get('signal_time')} {row.get('contract')}: levels without real premium "
                                      f"candles (source={row.get('premium_source')})")
        checked += 1
    return checked


def separate(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """A = comparable to the live setup, B = non-comparable; rows without real data are 'unavailable'."""
    out = {"comparable": [], "non_comparable": [], "unavailable": []}
    for row in rows:
        if not has_real_levels(row):
            out["unavailable"].append(row)
        elif row.get("comparable"):
            out["comparable"].append(row)
        else:
            out["non_comparable"].append(row)
    return out


# ---------------------------------------------------------------- metrics
def _dist(values: list[float]) -> dict[str, Any] | None:
    values = sorted(v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    if not values:
        return None
    return {"n": len(values), "mean": mean(values), "median": median(values), "min": values[0], "max": values[-1]}


def sample_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in rows if has_real_levels(r)]
    if not evaluated:
        return {"signals": len(rows), "resolved": 0, "status": "NO_REAL_DATA"}
    m = group_metrics(evaluated)
    dte_counts: dict[str, int] = {}
    for r in evaluated:
        dte_counts[str(r.get("calendar_dte"))] = dte_counts.get(str(r.get("calendar_dte")), 0) + 1
    return {
        "signals": len(rows),
        "evaluated_signals": len(evaluated),
        "resolved": m["resolved"], "wins": m["wins"], "losses": m["losses"],
        "ambiguous": m["ambiguous"], "pending": m["pending"],
        "win_rate": m["win_rate"], "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"],
        "max_drawdown": m["max_drawdown"], "average_r": m["average_r"],
        "trading_days": len({r["session_date"] for r in evaluated}),
        "dte_distribution": dict(sorted(dte_counts.items(), key=lambda kv: int(kv[0]) if kv[0] != "None" else -1)),
        "premium_atr_pct": _dist([r.get("premium_atr_pct") for r in evaluated]),
        "stop_pct": _dist([r.get("stop_pct") for r in evaluated]),
        "target_pct": _dist([r.get("target_pct") for r in evaluated]),
        "by_direction": {d: _direction(evaluated, d) for d in ("BUY", "SELL")},
        "premium_sources": sorted({r.get("premium_source") for r in evaluated}),
        "status": "OK",
    }


def _direction(rows, direction):
    sub = [r for r in rows if r["direction"] == direction]
    if not sub:
        return {"signals": 0}
    m = group_metrics(sub)
    return {k: m[k] for k in ("signals", "resolved", "wins", "losses", "win_rate", "profit_factor", "net_pnl",
                              "average_r")}


def dte_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for _, _, label in DTE_BUCKETS:
        sub = [r for r in rows if r.get("dte_bucket") == label and has_real_levels(r)]
        if not sub:
            out.append({"dte_bucket": label, "status": "NO_REAL_DATA"})
            continue
        rep = sample_report(sub)
        days = sorted({r["session_date"] for r in sub})
        out.append({"dte_bucket": label, "first_day": days[0], "last_day": days[-1],
                    "expiries": sorted({r.get("expiry") for r in sub}), **{k: rep[k] for k in (
            "evaluated_signals", "resolved", "wins", "losses", "win_rate", "profit_factor", "net_pnl",
            "average_r", "trading_days", "premium_atr_pct", "stop_pct", "target_pct")}, "status": "OK"})
    return out


def paired_expiry_comparison(rows_by_expiry: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Same signals, same session, different expiries: outcome agreement table."""
    keyed = {exp: {r["signal_time"]: r for r in rows if r["status"] in (WIN, LOSS)}
             for exp, rows in rows_by_expiry.items()}
    common = set.intersection(*(set(v) for v in keyed.values())) if keyed else set()
    expiries = list(keyed)
    table = []
    for ts in sorted(common):
        table.append({"signal_time": ts, **{exp: keyed[exp][ts]["status"] for exp in expiries}})
    per = {}
    for exp in expiries:
        sub = [keyed[exp][ts] for ts in common]
        m = group_metrics(sub) if sub else None
        per[exp] = None if m is None else {k: m[k] for k in ("resolved", "wins", "losses", "win_rate",
                                                              "profit_factor", "net_pnl", "average_r")}
    agree = sum(1 for row in table if len({row[e] for e in expiries}) == 1)
    return {"common_resolved_signals": len(common), "same_outcome_all_expiries": agree, "per_expiry": per,
            "signals": table}


# ------------------------------------------------------ live-equivalent contracts
def probe_expired_access(client, underlying_key: str) -> dict[str, Any]:
    try:
        expiries = client.expired_expiries(underlying_key)
    except HistoricalDataError as exc:
        return {"available": False, "code": exc.code, "http_status": exc.status, "expiries": []}
    return {"available": True, "code": None, "http_status": 200, "expiries": [e.isoformat() for e in expiries]}


def expired_contract_resolver(client, underlying: str, underlying_key: str, expiries: list[date],
                              tick_default: float = 0.05) -> Callable:
    """Nearest-expiry ATM contract from Upstox expired-instrument data (only when the API is available)."""
    cache: dict[date, list[dict]] = {}

    def resolve(signal) -> OptionContract:
        day = signal.timestamp.date()
        expiry = next((e for e in sorted(expiries) if e >= day), None)
        if expiry is None:
            raise OptionResolutionError("NO_EXPIRED_EXPIRY", f"No expired expiry on/after {day}")
        if expiry not in cache:
            cache[expiry] = client.expired_option_contracts(underlying_key, expiry)
        want = "CE" if signal.direction == "BUY" else "PE"
        rows = [c for c in cache[expiry] if str(c.get("instrument_type", "")).upper() == want]
        if not rows:
            raise OptionResolutionError("NO_EXPIRED_CONTRACTS", f"No {want} contracts for {expiry}")
        strike = atm_strike(sorted({float(c["strike_price"]) for c in rows}), signal.spot)
        row = next(c for c in rows if float(c["strike_price"]) == strike)
        return OptionContract(
            instrument_key=str(row["instrument_key"]), trading_symbol=str(row.get("trading_symbol", "")),
            underlying=underlying.upper(), expiry=expiry, strike=strike, option_type=OptionType(want),
            lot_size=int(row.get("lot_size") or 65), tick_size=float(row.get("tick_size") or tick_default),
        )

    return resolve


# ---------------------------------------------------------------- reports
def _j(value):
    if isinstance(value, dict):
        return {str(k): _j(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_j(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 6)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _f(v, kind="num"):
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "money":
        return f"Rs {v:,.0f}"
    return f"{v:,.2f}" if isinstance(v, float) else str(v)


def _dist_text(d):
    return "n/a" if not d else f"median {d['median']:.2f} (range {d['min']:.2f}-{d['max']:.2f}, n={d['n']})"


def write_reports(report: dict[str, Any], out_dir: Path, stem: str = "NIFTY_S2_contract_comparability") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / f"{stem}.json", "csv": out_dir / f"{stem}.csv", "md": out_dir / f"{stem}.md"}
    paths["json"].write_text(json.dumps(_j(report), indent=2, default=str), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({k: _j(row.get(k)) for k in CSV_FIELDS})
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    return paths


def _sample_table(samples: dict[str, dict[str, Any]]) -> list[str]:
    names = list(samples)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for key, kind in (("signals", "num"), ("evaluated_signals", "num"), ("resolved", "num"), ("wins", "num"),
                      ("losses", "num"), ("win_rate", "pct"), ("profit_factor", "num"), ("net_pnl", "money"),
                      ("max_drawdown", "money"), ("average_r", "num"), ("trading_days", "num")):
        lines.append(f"| {key} | " + " | ".join(_f(samples[n].get(key), kind) for n in names) + " |")
    for key in ("premium_atr_pct", "stop_pct", "target_pct"):
        lines.append(f"| {key} | " + " | ".join(_dist_text(samples[n].get(key)) for n in names) + " |")
    lines.append("| DTE distribution | " + " | ".join(
        ", ".join(f"{k}d:{v}" for k, v in (samples[n].get("dte_distribution") or {}).items()) or "n/a"
        for n in names) + " |")
    for d in ("BUY", "SELL"):
        lines.append(f"| {d} (W/L, win rate, net) | " + " | ".join(
            (lambda x: "n/a" if not x or not x.get("signals") else
             f"{x['wins']}/{x['losses']}, {_f(x['win_rate'], 'pct')}, {_f(x['net_pnl'], 'money')}")(
                (samples[n].get("by_direction") or {}).get(d)) for n in names) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# NIFTY S2 option-contract comparability", "", f"Mode: {report['mode']}", "",
             "## Live reference setup", ""]
    lines += [f"- {k}: {v}" for k, v in report["live_reference"].items()]
    lines += ["", "## 1. Look-ahead fix", ""] + [f"- {n}" for n in report["lookahead_fix"]["notes"]]
    e = report["engine_research"]
    lines += ["", "| Research engine (S2, FULL window) | Before fix | After fix |", "|---|---|---|"]
    for key, kind in (("resolved_trades", "num"), ("win_rate", "pct"), ("profit_factor", "num"),
                      ("net_pnl", "money"), ("max_drawdown", "money")):
        lines.append(f"| {key} | {_f(e['before'].get(key), kind)} | {_f(e['after'].get(key), kind)} |")
    lines += ["", f"Window: {e['window']}. Contracts used by the engine: {e['contract_expiries']}.", "",
              "## 2. Historical near-expiry data availability", ""]
    av = report["expired_data"]
    lines += [f"- Upstox expired-instrument API: {'AVAILABLE' if av['available'] else 'UNAVAILABLE'}"
              f" (code {av.get('code')}, HTTP {av.get('http_status')}).",
              f"- Historical signals that need the live-equivalent (nearest weekly) contract: "
              f"{report['live_equivalent_historical']['signals']}; evaluated: "
              f"{report['live_equivalent_historical']['evaluated']}.",
              f"- {report['live_equivalent_historical']['statement']}", "",
              "## 3. Samples", "", "### A. Comparable option sample (1 DTE, real candles)", ""]
    lines += _sample_table(report["samples"]["A_comparable"])
    lines += ["", "### B. Non-comparable option sample (real candles, DTE != 1)", ""]
    lines += _sample_table(report["samples"]["B_non_comparable"])
    lines += ["", "## 4. DTE comparison (all real-data rows)", "",
              "| DTE bucket | Sessions | Expiries | Signals | W/L | Win rate | PF | Net | Avg R | Days | "
              "Premium ATR % | Stop % | Target % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in report["dte_comparison"]:
        if b["status"] != "OK":
            lines.append(f"| {b['dte_bucket']} | no real data | | | | | | | | | | | |")
            continue
        md = lambda d: "n/a" if not d else f"{d['median']:.2f}"
        lines.append(f"| {b['dte_bucket']} | {b['first_day']}..{b['last_day']} | {', '.join(b['expiries'])} | "
                     f"{b['evaluated_signals']} | {b['wins']}/{b['losses']} | "
                     f"{_f(b['win_rate'], 'pct')} | {_f(b['profit_factor'])} | {_f(b['net_pnl'], 'money')} | "
                     f"{_f(b['average_r'])} | {b['trading_days']} | {md(b['premium_atr_pct'])} | {md(b['stop_pct'])} | "
                     f"{md(b['target_pct'])} |")
    p = report["paired_today"]
    lines += ["", "## 5. Same signals, same session, different expiries (today, real intraday candles)", "",
              f"Common resolved signals: {p['common_resolved_signals']}; same outcome on every expiry: "
              f"{p['same_outcome_all_expiries']}.", "", "| Expiry | Resolved | W/L | Win rate | PF | Net | Avg R |",
              "|---|---|---|---|---|---|---|"]
    for exp, m in p["per_expiry"].items():
        if m:
            lines.append(f"| {exp} | {m['resolved']} | {m['wins']}/{m['losses']} | {_f(m['win_rate'], 'pct')} | "
                         f"{_f(m['profit_factor'])} | {_f(m['net_pnl'], 'money')} | {_f(m['average_r'])} |")
    b = report["benchmark"]
    lines += ["", "## 6. Live validation benchmark (unchanged)", "",
              f"Source: {b['source']}. Check: **{b['check']}**.", ""]
    lines += [f"- {k}: {v}" for k, v in b["values"].items()]
    lines += ["", "## Findings", ""] + [f"- {f}" for f in report["findings"]]
    lines += ["", "## Notes", ""] + [f"- {n}" for n in report["notes"]]
    lines += ["", "No S2 entry/exit rule, filter or parameter was changed.", ""]
    return "\n".join(lines)
