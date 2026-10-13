"""STRICT CLEAN HISTORICAL BACKTEST of the frozen S2 strategy (PAPER | RESEARCH ONLY | NO ORDERS).

Historical backtest accuracy on the real Upstox 6-month dataset (data/historical/upstox_6m), never live accuracy.

Engine = the project's exact S2 replay (``s2_baseline_drift_audit.replay`` -> ``s2_1dte_improvement`` pipeline):
  signal on a completed 5m candle (S2 signal_fn, entry cutoff applied) -> point-in-time nearest-expiry ATM contract
  -> entry at the NEXT option candle's open -> premium ATR14 from premium candles <= signal -> SL / target fixed at
  entry -> first touch on later same-day option candles -> SL+target in one candle = AMBIGUOUS (-1R) -> EOD square-off.
Missing option data -> DATA_UNAVAILABLE (never interpolated).

Accuracy here = wins / (wins + losses). Ambiguous, scratch and unavailable signals are never wins and are reported
separately; the project's conservative rate wins / (all resolved incl. ambiguous) is reported alongside.
Robustness status reuses ``strategy_improvement_lab.candidate_status`` (LabCriteria) unchanged.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.research import s2_baseline_drift_audit as dra
from src.research import s2_diagnostic_lab as dg
from src.research import s5_targeted_strategy as s5

HEADING = "CLEAN HISTORICAL BACKTEST COMPLETE"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS | historical backtest accuracy (not live accuracy)"
IST = imp.IST
SCOPES = ("BUY", "SELL", "COMBINED")
SPLITS = {"TRAIN": (date(2026, 4, 1), date(2026, 6, 30)), "VALIDATION": (date(2026, 7, 1), date(2026, 7, 31)),
          "OOS": (date(2026, 8, 1), date(2026, 8, 31)), "FINAL_HOLDOUT": (date(2026, 9, 1), date(2026, 9, 30))}
SPLIT_LABEL = {"TRAIN": "A. TRAIN", "VALIDATION": "B. VALIDATION", "OOS": "C. OOS (AUGUST)",
               "FINAL_HOLDOUT": "D. FINAL HOLDOUT (SEPTEMBER)"}
FROZEN_S5 = "B5"
TARGET = 0.70
CRITERIA = lab.LabCriteria()
STATUS_MAP = {lab.ROBUST: "PROMISING", lab.PROMISING: "PROMISING", lab.NOT_PROMISING: "REJECTED",
              lab.REJECTED: "REJECTED", lab.INSUFFICIENT_DATA: "INSUFFICIENT_DATA"}
SESSION_OPEN, SESSION_CLOSE = time(9, 15), time(15, 30)


# ------------------------------------------------------------------ helpers
def day_of(r: dict) -> date:
    return date.fromisoformat(str(r["session_date"])[:10])


def split_of(r: dict) -> str | None:
    d = day_of(r)
    return next((k for k, (lo, hi) in SPLITS.items() if lo <= d <= hi), None)


def split_rows(rows: Sequence[dict], split: str) -> list[dict]:
    lo, hi = SPLITS[split]
    return [r for r in rows if lo <= day_of(r) <= hi]


def side(rows: Sequence[dict], scope: str) -> list[dict]:
    return list(rows) if scope == "COMBINED" else [r for r in rows if r["direction"] == scope]


def _pct(v) -> str:
    return "N/A" if v is None else f"{v * 100:.1f}%"


def _pf(v) -> str:
    return "N/A" if v is None else ("INF" if v == math.inf else f"{v:.2f}")


# ------------------------------------------------------------------ metrics (user definitions)
def metrics(rows: Sequence[dict]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
    ev = [r for r in rows if r.get("state") == sh.EVALUATED]
    wins = sum(r["outcome"] == imp.WIN for r in ev)
    losses = sum(r["outcome"] == imp.LOSS for r in ev)
    amb = sum(r["outcome"] == imp.AMBIGUOUS for r in ev)
    scratch = sum(r["outcome"] == imp.SCRATCH for r in ev)
    pnl = [float(r["pnl"]) for r in ev]
    gp, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    eq = peak = dd = 0.0
    streak = max_streak = 0
    for r, p in zip(ev, pnl):
        eq += p
        peak, dd = max(peak, eq), max(dd, peak - eq)
        streak = streak + 1 if r["outcome"] in (imp.LOSS, imp.AMBIGUOUS) else 0
        max_streak = max(max_streak, streak)
    rs = [lab.r_value(r) for r in ev]
    resolved = wins + losses
    reasons = Counter(str(r.get("reason") or r.get("reason_raw")) for r in rows if r.get("state") == sh.DATA_UNAVAILABLE)
    return {
        "signals": len(rows), "executable": len(ev), "skipped": sum(r.get("state") == sh.SKIPPED for r in rows),
        "unresolved_or_unavailable": sum(r.get("state") == sh.DATA_UNAVAILABLE for r in rows),
        "unavailable_reasons": dict(reasons), "total_trades": len(ev), "wins": wins, "losses": losses,
        "ambiguous": amb, "scratch": scratch, "win_rate": wins / resolved if resolved else None,
        "win_rate_denominator": f"{wins} wins / {resolved} resolved (wins + losses)",
        "conservative_win_rate": wins / len(ev) if ev else None,
        "profit_factor": (gp / gl if gl > 0 else (math.inf if gp > 0 else None)) if ev else None,
        "gross_profit": round(gp, 2), "gross_loss": round(gl, 2), "net_pnl": round(gp - gl, 2),
        "max_drawdown": round(dd, 2), "max_losing_streak": max_streak,
        "average_r": round(sum(rs) / len(rs), 4) if rs else None,
        "buy_signals": sum(r["direction"] == "BUY" for r in rows), "sell_signals": sum(r["direction"] == "SELL" for r in rows),
    }


def scoped(rows: Sequence[dict]) -> dict[str, dict]:
    return {s: metrics(side(rows, s)) for s in SCOPES}


# ------------------------------------------------------------------ robustness (existing LabCriteria, unchanged)
def wf_share(rows: Sequence[dict], end: date) -> float | None:
    return s5.walk_forward(rows, end)["profitable_share"]


def regime_counts(rows: Sequence[dict]) -> tuple[int, int]:
    _, groups = lab.group_stability(rows, lambda r: [r["features"]["trend_regime"]], CRITERIA.min_group_trades)
    big = {k: v for k, v in groups.items() if v["evaluated"] >= CRITERIA.min_group_trades and k != dg.UNAVAILABLE}
    return len(big), sum(v["net_pnl"] > 0 for v in big.values())


def robustness(rows: Sequence[dict], test_split: str, lookahead_ok: bool, retention: float | None) -> dict[str, Any]:
    """lab.candidate_status with TRAIN / VALIDATION / <test split>. Maps to PROMISING / REJECTED / INSUFFICIENT_DATA."""
    test = split_rows(rows, test_split)
    groups, profitable = regime_counts(test)
    s = {"TRAIN": lab.perf(split_rows(rows, "TRAIN")), "VALIDATION": lab.perf(split_rows(rows, "VALIDATION")),
         "OOS": lab.perf(test), "retention": retention, "lookahead_passed": lookahead_ok, "regime_groups": groups,
         "profitable_regime_groups": profitable, "wf_positive_share": wf_share(rows, SPLITS[test_split][1]),
         "robustness_ok": None}
    st = lab.candidate_status(s, CRITERIA)
    return {"status": STATUS_MAP[st["status"]], "project_status": st["status"], "failed_checks": st["failed_checks"],
            "wf_positive_share": s["wf_positive_share"], "regime_groups": groups, "profitable_regime_groups": profitable}


def target_70(per_split: dict[str, dict]) -> str:
    """ACHIEVED only if OOS and FINAL HOLDOUT combined win rates are >= 70% on >= min_oos_trades resolved trades."""
    tests = [per_split[k]["COMBINED"] for k in ("OOS", "FINAL_HOLDOUT")]
    if any((m["wins"] + m["losses"]) < CRITERIA.min_oos_trades for m in tests):
        return "INSUFFICIENT DATA"
    return "ACHIEVED" if all((m["win_rate"] or 0.0) >= TARGET for m in tests) else "NOT ACHIEVED"


# ------------------------------------------------------------------ anti-lookahead / integrity checks
def timestamp_checks(rows: Sequence[dict], square_off: str, stop_new_entries: str) -> dict[str, Any]:
    so, cutoff = time.fromisoformat(square_off), time.fromisoformat(stop_new_entries)
    bad: Counter = Counter()
    ex: list[str] = []
    for r in rows:
        st = dra._ts(r["signal_time"])
        probs = []
        if st.weekday() >= 5:
            probs.append("WEEKEND_SIGNAL")
        if not (SESSION_OPEN <= st.time() < SESSION_CLOSE):
            probs.append("SIGNAL_OUTSIDE_SESSION")
        if st.time() >= cutoff:
            probs.append("SIGNAL_AFTER_ENTRY_CUTOFF")
        if st.minute % 5 or st.second:
            probs.append("SIGNAL_OFF_GRID")
        if st.date().isoformat() != str(r["session_date"])[:10]:
            probs.append("SESSION_DATE_MISMATCH")
        if r.get("entry_time"):
            et = dra._ts(r["entry_time"])
            if et <= st:
                probs.append("ENTRY_NOT_AFTER_SIGNAL")
            if et.date() != st.date():
                probs.append("ENTRY_OTHER_DAY")
            if r.get("exit_time"):
                xt = dra._ts(r["exit_time"])
                if xt < et:
                    probs.append("EXIT_BEFORE_ENTRY")
                if xt.date() != et.date() or xt.time() > so:
                    probs.append("EXIT_AFTER_SQUARE_OFF")
        for p in probs:
            bad[p] += 1
        if probs and len(ex) < 10:
            ex.append(f"{r['signal_time']}|{r['direction']}: {','.join(probs)}")
    return {"checked": len(rows), "problems": dict(bad), "examples": ex, "passed": not bad}


def duplicate_check(rows: Sequence[dict]) -> dict[str, Any]:
    keys = Counter(dra.key_of(r) for r in rows)
    trades = Counter((r.get("instrument_key"), r.get("entry_time"), r["direction"]) for r in rows
                     if r.get("state") == sh.EVALUATED)
    dk = [k for k, n in keys.items() if n > 1]
    dt = [k for k, n in trades.items() if n > 1]
    return {"duplicate_signals": len(dk), "duplicate_trades": len(dt), "examples": [str(x) for x in (dk + dt)[:5]],
            "passed": not dk and not dt}


def chronology(rows: Sequence[dict]) -> dict[str, Any]:
    ts = [dra._ts(r["signal_time"]) for r in rows]
    return {"ordered": all(a <= b for a, b in zip(ts, ts[1:])), "passed": all(a <= b for a, b in zip(ts, ts[1:]))}


def contract_validity(rows: Sequence[dict], calendar, listed: Callable, exists: Callable) -> dict[str, Any]:
    res = [dra.contract_check(r, calendar, listed, exists) for r in rows if r.get("instrument_key")]
    bad = [x for x in res if not x["valid"]]
    fails = Counter(k for x in bad for k, v in x.items() if isinstance(v, bool) and not v and k not in ("valid",))
    return {"checked": len(res), "invalid": len(bad), "failed_fields": dict(fails),
            "examples": [f"{x['timestamp']}|{x['direction']}|{x['contract']}" for x in bad[:5]], "passed": not bad}


def selection_isolation(rows: Sequence[dict]) -> dict[str, Any]:
    """The frozen S5 selection must use only rows <= 2026-07-31 and be unchanged if every Aug/Sep outcome flips."""
    pre = [r for r in rows if day_of(r) <= s5.SELECTION_END]
    sel = s5.select(pre)
    flipped = s5._flip(rows, s5.SELECTION_END)
    sel2 = s5.select([r for r in flipped if day_of(r) <= s5.SELECTION_END])
    refuses = False
    try:
        s5.select(rows)
    except ValueError:
        refuses = True
    ids = s5._ids(sel)
    return {"selected": ids, "frozen_s5": FROZEN_S5, "matches_frozen": ids.get("BUY") == FROZEN_S5,
            "invariant_to_oos_and_holdout": ids == s5._ids(sel2), "refuses_post_validation_rows": refuses,
            "passed": ids == s5._ids(sel2) and refuses}


def option_stats(rows: Sequence[dict]) -> dict[str, Any]:
    ev = [r for r in rows if r.get("instrument_key")]
    return {"resolved_contracts": len(ev), "unique_contracts": len({r["instrument_key"] for r in ev}),
            "unique_expiries": len({str(r.get("expiry_date")) for r in ev}),
            "by_option_type": dict(Counter(r.get("option_type") for r in ev)),
            "by_expiry": dict(sorted(Counter(str(r.get("expiry_date")) for r in ev).items())),
            "by_dte_bucket": dict(Counter(sh.dte_bucket(r.get("dte")) for r in ev)),
            "unresolved_contract": sum(1 for r in rows if not r.get("instrument_key"))}


# ------------------------------------------------------------------ study
def group_results(rows: Sequence[dict], key: Callable[[dict], str], name: str) -> list[dict]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    return [{name: k, "scope": s, **_flat(metrics(side(g[k], s)))} for k in sorted(g) for s in SCOPES]


def _flat(m: dict) -> dict:
    return {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in m.items()}


def build(*, rows: Sequence[dict], contexts: Sequence[dict], params, square_off: str, stop_new_entries: str,
          calendar, listed: Callable, exists: Callable, integrity: dict, mutation: dict | None,
          truncation: dict, sessions: Sequence[str], dataset: dict, production_unchanged: bool | None = None
          ) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (dra._ts(r["signal_time"]), r["direction"]))
    align = dra.alignment_checks(contexts, params)
    ts_chk = timestamp_checks(rows, square_off, stop_new_entries)
    dup = duplicate_check(rows)
    chron = chronology(rows)
    cval = contract_validity(rows, calendar, listed, exists)
    iso = selection_isolation(rows)
    lookahead = {"entry_next_candle_and_atr_no_future (alignment)": align["passed"],
                 "pipeline_mutation_after_cut": None if mutation is None else mutation["passed"],
                 "feature_truncation": truncation["passed"], "chronology": chron["passed"],
                 "timestamps": ts_chk["passed"], "selection_isolation": iso["passed"]}
    lookahead_ok = all(v is not False for v in lookahead.values())
    data_ok = bool(integrity.get("ok")) and dup["passed"] and cval["passed"] and ts_chk["passed"]

    per_split = {k: scoped(split_rows(rows, k)) for k in SPLITS}
    status = {k: {s: robustness(side(rows, s), k, lookahead_ok, 1.0) for s in SCOPES} for k in ("OOS", "FINAL_HOLDOUT")}

    b5 = next(c for c in s5.BASE_CANDIDATES if c.cid == FROZEN_S5)
    b5_rows = s5.candidate_rows(b5, rows)
    s2_buy = [r for r in rows if r["direction"] == "BUY"]
    retention = (len(b5_rows) / len(s2_buy)) if s2_buy else None
    s5_block = {"candidate": f"{b5.cid} {b5.name}", "rules": b5.describe(), "direction": b5.direction,
                "contaminated": True,
                "contamination_note": ("B5's conditions come from the Apr-Sep S2 diagnostic, so August and September "
                                       "took part in discovering it. Its OOS and September results are NOT clean "
                                       "validation."),
                "per_split": {k: metrics(split_rows(b5_rows, k)) for k in SPLITS},
                "status": {k: robustness(b5_rows, k, lookahead_ok, retention) for k in ("OOS", "FINAL_HOLDOUT")},
                "selection_check": iso}

    ev = [r for r in rows if r.get("state") == sh.EVALUATED]
    summary = {
        "trading_sessions": len(sessions), "generated_signals": len(rows), "executable_signals": len(ev),
        "excluded_or_unresolved_signals": len(rows) - len(ev),
        "buy_signals": sum(r["direction"] == "BUY" for r in rows), "sell_signals": sum(r["direction"] == "SELL" for r in rows),
        "sessions_per_split": {k: sum(1 for d in sessions if SPLITS[k][0] <= date.fromisoformat(d) <= SPLITS[k][1])
                               for k in SPLITS},
    }
    return {
        "heading": HEADING, "banner": BANNER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "accuracy_definition": "wins / (wins + losses); ambiguous, scratch and unavailable are never wins",
        "dataset": dataset, "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        "summary": summary, "per_split": per_split, "status": status, "target_70": target_70(per_split),
        "monthly": group_results(rows, lambda r: str(r["session_date"])[:7], "month"),
        "regime": group_results(rows, lambda r: r["features"]["trend_regime"], "trend_regime")
        + group_results(rows, lambda r: r["features"]["vol_regime"], "vol_regime"),
        "time_bucket": group_results(rows, lambda r: str(r.get("time_bucket")), "time_bucket"),
        "options": option_stats(rows), "s5": s5_block,
        "checks": {"alignment": align, "mutation": mutation, "truncation": truncation, "timestamps": ts_chk,
                   "duplicates": dup, "chronology": chron, "contract_validity": cval, "selection_isolation": iso},
        "lookahead": lookahead, "lookahead_ok": lookahead_ok, "data_integrity": integrity, "data_ok": data_ok,
        "production_unchanged": production_unchanged, "live_orders": 0, "trading_mode": "PAPER",
        "live_trading_enabled": False, "rows": rows,
        "clean_note": ("S2 rules are unchanged by this research (file hashes verified). OOS and FINAL HOLDOUT months "
                       "were already viewed in earlier diagnostic labs, but no S2 rule was changed because of them."),
    }


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "clean_backtest_report.md", "trades.csv", "monthly_results.csv", "buy_sell_results.csv",
                "data_quality_report.json")
TRADE_FIELDS = ("split", "session_date", "signal_time", "direction", "spot", "breakout_level", "state", "outcome",
                "reason", "contract", "instrument_key", "option_type", "strike", "expiry_date", "dte", "entry_time",
                "entry", "premium_atr", "stop_loss", "target", "exit_time", "exit_price", "exit_reason", "pnl",
                "r_multiple", "time_bucket", "trend_regime", "vol_regime")


def _jsonable(v):
    if isinstance(v, float) and math.isinf(v):
        return "INF"
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (date, datetime, pd.Timestamp)):
        return v.isoformat()
    return v


def _csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    cols: list[str] = list(fields or [])
    if not fields:
        for r in rows:
            cols += [k for k in r if k not in cols]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols or ["empty"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("INF" if v == math.inf else v) for k, v in r.items()})


def trade_rows(a: dict) -> list[dict]:
    out = []
    for r in a["rows"]:
        out.append({**{k: r.get(k) for k in TRADE_FIELDS if k in r}, "split": split_of(r),
                    "trend_regime": r["features"]["trend_regime"], "vol_regime": r["features"]["vol_regime"]})
    return out


def buy_sell_rows(a: dict) -> list[dict]:
    out = []
    for k in SPLITS:
        for s in SCOPES:
            st = a["status"].get(k, {}).get(s, {})
            out.append({"strategy": "S2", "split": SPLIT_LABEL[k], "scope": s, **_flat(a["per_split"][k][s]),
                        "robustness_status": st.get("status"), "project_status": st.get("project_status"),
                        "failed_checks": ",".join(st.get("failed_checks", []))})
    for k in SPLITS:
        st = a["s5"]["status"].get(k, {})
        out.append({"strategy": f"S5 {a['s5']['candidate']} (CONTAMINATED)", "split": SPLIT_LABEL[k], "scope": "BUY",
                    **_flat(a["s5"]["per_split"][k]), "robustness_status": st.get("status"),
                    "project_status": st.get("project_status"), "failed_checks": ",".join(st.get("failed_checks", []))})
    return out


def line(m: dict) -> str:
    return (f"Trades {m['total_trades']} | Win Rate {_pct(m['win_rate'])} ({m['wins']}/{m['wins'] + m['losses']}) | "
            f"PF {_pf(m['profit_factor'])} | P&L Rs {m['net_pnl']:,.2f} | Max DD Rs {m['max_drawdown']:,.2f}")


def markdown(a: dict) -> str:
    s, ck = a["summary"], a["checks"]
    md = ["# Clean historical backtest - frozen S2", "", f"**{BANNER}**", "",
          f"Generated {a['generated_at']}. Dataset {a['dataset'].get('from')} -> {a['dataset'].get('to')} "
          f"(data/historical/upstox_6m). TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, live orders: 0.", "",
          "**Accuracy = wins / (wins + losses).** Ambiguous (SL and target in one candle, booked at -1R), scratch "
          "and unavailable signals are never wins; the denominator is shown next to every rate. The project's "
          "conservative rate (wins / all resolved incl. ambiguous) is in buy_sell_results.csv.", "",
          "## Summary", "",
          f"- Trading sessions: {s['trading_sessions']} {s['sessions_per_split']}",
          f"- Generated S2 signals: {s['generated_signals']} (BUY {s['buy_signals']} / SELL {s['sell_signals']})",
          f"- Executable (resolved) signals: {s['executable_signals']}; excluded / unresolved: "
          f"{s['excluded_or_unresolved_signals']}",
          f"- 70% target: **{a['target_70']}** | Lookahead: **{'PASS' if a['lookahead_ok'] else 'FAIL'}** | Data "
          f"integrity: **{'PASS' if a['data_ok'] else 'FAIL'}** | Production changed: "
          f"**{'NO' if a['production_unchanged'] is not False else 'YES'}**", "", f"> {a['clean_note']}", ""]
    for k in SPLITS:
        md += [f"## {SPLIT_LABEL[k]} ({a['splits'][k][0]} -> {a['splits'][k][1]})", "",
               "| Scope | Trades | W | L | Amb | Unavail. | Win rate (W/(W+L)) | PF | Gross profit | Gross loss | Net P&L "
               "| Max DD | Max losing streak | Avg R |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for sc in SCOPES:
            m = a["per_split"][k][sc]
            md.append(f"| {sc} | {m['total_trades']} | {m['wins']} | {m['losses']} | {m['ambiguous']} | "
                      f"{m['unresolved_or_unavailable']} | {_pct(m['win_rate'])} ({m['wins']}/{m['wins'] + m['losses']}) "
                      f"| {_pf(m['profit_factor'])} | {m['gross_profit']:,.2f} | {m['gross_loss']:,.2f} | "
                      f"{m['net_pnl']:,.2f} | {m['max_drawdown']:,.2f} | {m['max_losing_streak']} | {m['average_r']} |")
        if k in a["status"]:
            md += ["", "Robustness (existing LabCriteria via strategy_improvement_lab.candidate_status):"]
            md += [f"- {sc}: **{a['status'][k][sc]['status']}** (project status {a['status'][k][sc]['project_status']}"
                   f"; {a['status'][k][sc]['failed_checks'] or 'no failed checks'})" for sc in SCOPES]
        md.append("")
    b = a["s5"]
    md += ["## Frozen S5 candidate (secondary, CONTAMINATED)", "", f"- {b['candidate']}: {b['rules']}",
           f"- **{b['contamination_note']}**",
           f"- Re-running the S5 selection on <= 2026-07-31 picks {b['selection_check']['selected']} "
           f"(matches frozen {FROZEN_S5}: {b['selection_check']['matches_frozen']})", ""]
    md += [f"- {SPLIT_LABEL[k]}: {line(b['per_split'][k])}" for k in SPLITS]
    md += [f"- {k} status: {b['status'][k]['status']} ({b['status'][k]['failed_checks']})" for k in b["status"]]
    md += ["", "## Monthly results (S2)", "", "| Month | Scope | Trades | Win rate | PF | Net P&L | Max DD |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['month']} | {r['scope']} | {r['total_trades']} | {_pct(r['win_rate'])} | {_pf(r['profit_factor'])} | "
           f"{r['net_pnl']:,.2f} | {r['max_drawdown']:,.2f} |" for r in a["monthly"]]
    md += ["", "## Regime (causal, signal candle) and time bucket (S2, all months)", "",
           "| Group | Scope | Trades | Win rate | PF | Net P&L |", "|---|---|---|---|---|---|"]
    for r in a["regime"] + a["time_bucket"]:
        g = r.get("trend_regime") or r.get("vol_regime") or r.get("time_bucket")
        md.append(f"| {g} | {r['scope']} | {r['total_trades']} | {_pct(r['win_rate'])} | {_pf(r['profit_factor'])} | "
                  f"{r['net_pnl']:,.2f} |")
    o = a["options"]
    md += ["", "## Option contracts", "", f"- Resolved contracts {o['resolved_contracts']}, unique {o['unique_contracts']}, "
           f"expiries {o['unique_expiries']}, unresolved {o['unresolved_contract']}",
           f"- By type {o['by_option_type']}; by DTE {o['by_dte_bucket']}", f"- By expiry {o['by_expiry']}", "",
           "## Anti-lookahead and integrity checks", ""]
    md += [f"- {k}: {'PASS' if v else ('NOT RUN' if v is None else 'FAIL')}" for k, v in a["lookahead"].items()]
    md += [f"- duplicates: {ck['duplicates']}", f"- contract validity: {ck['contract_validity']}",
           f"- timestamps: {ck['timestamps']}", f"- data integrity (6-month store): "
           f"{'PASS' if a['data_integrity'].get('ok') else 'FAIL'} - see data_quality_report.json"]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    summary = {k: v for k, v in a.items() if k not in ("rows", "data_integrity")}
    p["summary.json"].write_text(json.dumps(_jsonable(summary), indent=2, default=str), encoding="utf-8")
    p["clean_backtest_report.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["trades.csv"], trade_rows(a), TRADE_FIELDS)
    _csv(p["monthly_results.csv"], a["monthly"])
    _csv(p["buy_sell_results.csv"], buy_sell_rows(a))
    dq = {"data_integrity": a["data_integrity"], "unavailable_reasons": {k: a["per_split"][k]["COMBINED"]
          ["unavailable_reasons"] for k in SPLITS}, "duplicates": a["checks"]["duplicates"],
          "contract_validity": a["checks"]["contract_validity"], "timestamps": a["checks"]["timestamps"],
          "alignment": a["checks"]["alignment"], "options": a["options"], "data_ok": a["data_ok"]}
    p["data_quality_report.json"].write_text(json.dumps(_jsonable(dq), indent=2, default=str), encoding="utf-8")
    return p


def terminal_text(a: dict) -> str:
    out = [HEADING, "", "S2 (historical backtest accuracy, not live accuracy):"]
    names = {"TRAIN": "TRAIN", "VALIDATION": "VALIDATION", "OOS": "OOS AUGUST", "FINAL_HOLDOUT": "FINAL HOLDOUT SEPTEMBER"}
    for k in SPLITS:
        out.append(f"{names[k]}:")
        for s in SCOPES:
            st = a["status"].get(k, {}).get(s)
            out.append(f"{s}: {line(a['per_split'][k][s])}" + (f" | {st['status']}" if st else ""))
        out.append("")
    b = a["s5"]
    out.append(f"S5 {b['candidate']} (CONTAMINATED - not clean validation): OOS {line(b['per_split']['OOS'])}; "
               f"SEPTEMBER {line(b['per_split']['FINAL_HOLDOUT'])}")
    out += ["", "70% TARGET:", a["target_70"], "", "LOOKAHEAD:", "PASS" if a["lookahead_ok"] else "FAIL", "",
            "DATA INTEGRITY:", "PASS" if a["data_ok"] else "FAIL", "", "PRODUCTION CHANGED:",
            "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:", "0"]
    return "\n".join(out)
