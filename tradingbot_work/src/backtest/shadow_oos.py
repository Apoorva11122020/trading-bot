"""OUT-OF-SAMPLE / FORWARD VALIDATION of the shadow strategies (PAPER, research only, no orders).

The three shadow strategies are fixed in advance (``CANDIDATES``); nothing here selects, tunes or filters them.
One shared replay over TRAIN + VALIDATION + OOS is partitioned chronologically by session date. Ranking and the
robustness status read only VALIDATION/OOS metrics, never TRAIN. The current-day forward shadow result is reported
separately and never merged into any historical split.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.paper_trading.shadow_validation import CANDIDATES

HEADING = "OUT-OF-SAMPLE / FORWARD VALIDATION"
SPLIT_LABELS = {"TRAIN": "HISTORICAL TRAIN", "VALIDATION": "HISTORICAL VALIDATION", "OOS": "HISTORICAL OOS"}
CURRENT_LABEL = "CURRENT-DAY FORWARD SHADOW"
SPLIT_FILES = {"TRAIN": "train.json", "VALIDATION": "validation.json", "OOS": "oos.json"}
PROMISING, ROBUST, NOT_ROBUST, INSUFFICIENT_DATA = "PROMISING", "ROBUST", "NOT_ROBUST", "INSUFFICIENT_DATA"
BEARISH = "strong_bearish_trend"
OOS_PREVIOUSLY_OBSERVED_NOTE = (
    "The OOS window 2026-09-25..2026-10-01 was already visible in earlier research (direction study and the 22-day "
    "historical replay) before this study, so it is not a pristine hold-out. Status is therefore capped at PROMISING; "
    "only sessions after 2026-10-05 recorded by the forward shadow can confirm ROBUST.")


@dataclass(frozen=True)
class RobustnessCriteria:
    min_oos_pf: float = 1.20
    min_oos_trades: int = 30
    min_oos_traded_days: int = 3
    min_coverage: float = 0.95
    insufficient_coverage: float = 0.80
    min_return_to_drawdown: float = 1.0
    max_best_day_share: float = 0.50


# ------------------------------------------------------------------ splits
def assign_days(days: Sequence[str], splits: dict[str, tuple[date, date]]) -> tuple[dict[str, list[str]], list[str]]:
    """Chronological partition of replay sessions; sessions outside every split are returned separately."""
    out: dict[str, list[str]] = {k: [] for k in imp.SPLITS}
    unassigned = []
    for d in sorted(days):
        name = imp.split_of(d, splits)
        (out[name] if name else unassigned).append(d)
    return out, unassigned


def split_lookahead(lookahead: dict[str, dict], split_days: Sequence[str]) -> dict[str, dict]:
    """Mutation-test result restricted to the cuts that fall on sessions of one split."""
    wanted = set(split_days)
    out = {}
    for name, r in lookahead.items():
        cuts = len(wanted) if r.get("cuts") else 0
        failures = [f for f in r.get("failures") or [] if f[:10] in wanted]
        out[name] = {"cuts": cuts, "failures": failures, "passed": bool(cuts) and not failures}
    return out


# ------------------------------------------------------------------ metrics
def _best_day(m: dict) -> str | None:
    return (m.get("best_day") or {}).get("date")


def split_metrics(rows: Sequence[dict], days: Sequence[str]) -> dict[str, Any]:
    m = sh.strategy_summary(rows, days)
    m["regime"] = {k: v for k, v in m["regime"].items() if v["signals"]}
    best = _best_day(m)
    rest = sh.stats(r for r in rows if r["session_date"] != best) if best else sh.stats(rows)
    m["excluding_best_day"] = {"excluded_date": best, "evaluated": rest["evaluated"], "win_rate": rest["win_rate"],
                               "profit_factor": rest["profit_factor"], "net_pnl": rest["net_pnl"]}
    best_pnl = (m.get("best_day") or {}).get("net_pnl")
    m["best_day_share"] = best_pnl / m["net_pnl"] if best_pnl is not None and m["net_pnl"] > 0 else None
    m["return_to_drawdown"] = (m["net_pnl"] / m["max_drawdown"] if m["max_drawdown"] > 0
                               else (math.inf if m["net_pnl"] > 0 else None))
    m["traded_days"] = sum(1 for x in m["daily"].values() if x["evaluated"])
    return m


def regime_dependence(rows: Sequence[dict]) -> dict[str, Any]:
    """OOS P&L on strong-bearish sessions versus all other sessions (labels are end-of-day, reporting only)."""
    bear = sh.stats(r for r in rows if BEARISH in r["regimes"])
    other = sh.stats(r for r in rows if BEARISH not in r["regimes"])
    total = bear["net_pnl"] + other["net_pnl"]
    return {"strong_bearish": {k: bear[k] for k in ("evaluated", "win_rate", "profit_factor", "net_pnl")},
            "non_bearish": {k: other[k] for k in ("evaluated", "win_rate", "profit_factor", "net_pnl")},
            "bearish_share_of_pnl": bear["net_pnl"] / total if total > 0 else None,
            "profitable_without_bearish_days": other["net_pnl"] > 0}


# ------------------------------------------------------------------ ranking / status (never TRAIN)
def _num(v: float | None, missing: float) -> float:
    return missing if v is None else v


def rank_by_oos(oos: dict[str, dict]) -> list[str]:
    """OOS PF, then OOS P&L, lower OOS drawdown, OOS win rate, OOS trade count; untraded strategies last."""
    def key(name):
        m = oos[name]
        return (m["evaluated"] == 0, -_num(m["profit_factor"], -1.0), -m["net_pnl"], m["max_drawdown"],
                -_num(m["win_rate"], -1.0), -m["evaluated"])
    return sorted(oos, key=key)


def _check(passed: bool, value: Any, threshold: Any) -> dict:
    return {"passed": bool(passed), "value": value, "threshold": threshold}


def robustness(oos: dict, validation: dict, lookahead_passed: bool, criteria: RobustnessCriteria = RobustnessCriteria(),
               oos_previously_observed: bool = True) -> dict[str, Any]:
    c = criteria
    pf, cov, rdd = oos["profit_factor"], oos["data_coverage"], oos["return_to_drawdown"]
    sufficiency = {
        "oos_trades": _check(oos["evaluated"] >= c.min_oos_trades, oos["evaluated"], f">= {c.min_oos_trades}"),
        "oos_traded_days": _check(oos["traded_days"] >= c.min_oos_traded_days, oos["traded_days"],
                                  f">= {c.min_oos_traded_days}"),
        "oos_data_coverage_usable": _check(cov is not None and cov >= c.insufficient_coverage, cov,
                                           f">= {c.insufficient_coverage:.0%}"),
    }
    core = {
        "no_lookahead_failures": _check(lookahead_passed, lookahead_passed, "all splits pass"),
        "oos_profit_factor": _check(pf is not None and pf > c.min_oos_pf, pf, f"> {c.min_oos_pf}"),
        "oos_pnl_positive": _check(oos["net_pnl"] > 0, oos["net_pnl"], "> 0"),
        "profitable_excluding_best_day": _check(oos["excluding_best_day"]["net_pnl"] > 0,
                                                oos["excluding_best_day"]["net_pnl"], "> 0"),
        "no_catastrophic_drawdown": _check(rdd is not None and rdd >= c.min_return_to_drawdown, rdd,
                                           f"OOS P&L / OOS max DD >= {c.min_return_to_drawdown}"),
    }
    vpf = validation["profit_factor"]
    strict = {
        "no_material_data_gaps": _check(cov is not None and cov >= c.min_coverage, cov, f">= {c.min_coverage:.0%}"),
        "no_single_day_dependence": _check(oos["best_day_share"] is not None
                                           and oos["best_day_share"] <= c.max_best_day_share,
                                           oos["best_day_share"], f"best day <= {c.max_best_day_share:.0%} of OOS P&L"),
        "validation_consistent": _check(vpf is not None and vpf > 1.0 and validation["net_pnl"] > 0,
                                        {"profit_factor": vpf, "net_pnl": validation["net_pnl"]}, "PF > 1 and P&L > 0"),
        "oos_window_unseen": _check(not oos_previously_observed, not oos_previously_observed,
                                    "OOS not inspected before the study"),
    }
    failed = lambda group: [k for k, v in group.items() if not v["passed"]]  # noqa: E731
    if failed(sufficiency):
        status, reasons = INSUFFICIENT_DATA, failed(sufficiency)
    elif failed(core):
        status, reasons = NOT_ROBUST, failed(core)
    elif failed(strict):
        status, reasons = PROMISING, failed(strict)
    else:
        status, reasons = ROBUST, []
    return {"status": status, "failed_checks": reasons,
            "checks": {"sufficiency": sufficiency, "core": core, "strict": strict}}


# ------------------------------------------------------------------ study
def current_day_forward(current: dict | None, splits: dict[str, tuple[date, date]]) -> dict[str, Any]:
    if not current:
        return {"label": CURRENT_LABEL, "available": False, "note": "No current-day shadow report found."}
    days = list(current.get("validated_days") or [])
    overlap = [d for d in days if imp.split_of(d, splits)]
    return {"label": CURRENT_LABEL, "available": True, "validated_days": days,
            "overlapping_historical_days": overlap, "candidates": current.get("candidates") or {},
            "note": "Kept separate: never merged into TRAIN, VALIDATION or OOS."}


def build_study(result: dict[str, Any], meta: dict[str, Any], *, splits: dict[str, tuple[date, date]],
                current: dict | None = None, criteria: RobustnessCriteria = RobustnessCriteria(),
                oos_previously_observed: bool = True) -> dict[str, Any]:
    split_days, unassigned = assign_days(result["days"], splits)
    names = [c.name for c in CANDIDATES]
    per_split: dict[str, dict] = {}
    for split, days in split_days.items():
        wanted = set(days)
        per_split[split] = {
            "label": SPLIT_LABELS[split], "from": splits[split][0].isoformat(), "to": splits[split][1].isoformat(),
            "days": days, "lookahead": split_lookahead(result["lookahead"], days),
            "strategies": {n: split_metrics([r for r in result["rows"].get(n, []) if r["session_date"] in wanted], days)
                           for n in names},
            "signals": {n: [r for r in result["rows"].get(n, []) if r["session_date"] in wanted] for n in names},
        }
    oos, val = per_split["OOS"]["strategies"], per_split["VALIDATION"]["strategies"]
    lookahead_ok = {n: all(per_split[s]["lookahead"][n]["passed"] for s in per_split if per_split[s]["days"])
                    for n in names}
    status = {n: robustness(oos[n], val[n], lookahead_ok[n], criteria, oos_previously_observed) for n in names}
    oos_rows = per_split["OOS"]["signals"]
    daily_oos = {d: {n: {"trades": oos[n]["daily"][d]["evaluated"], "win_rate": oos[n]["daily"][d]["win_rate"],
                         "net_pnl": oos[n]["daily"][d]["net_pnl"]} for n in names}
                 for d in per_split["OOS"]["days"]}
    final_table = []
    for n in names:
        t, v, o = per_split["TRAIN"]["strategies"][n], val[n], oos[n]
        final_table.append({"strategy": n, "train_wr": t["win_rate"], "train_pf": t["profit_factor"],
                            "validation_wr": v["win_rate"], "validation_pf": v["profit_factor"],
                            "oos_wr": o["win_rate"], "oos_pf": o["profit_factor"], "oos_pnl": o["net_pnl"],
                            "oos_dd": o["max_drawdown"], "oos_trades": o["evaluated"], "status": status[n]["status"]})
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"),
        "strategies": names,
        "strategy_selection": "Pre-declared (BASELINE_S2_BUY_SELL, SELL_ONLY, SELL_CONFIRMED); no parameter, filter "
                              "or strategy is chosen from any split.",
        "splits": {s: {k: per_split[s][k] for k in ("label", "from", "to", "days")} for s in per_split},
        "unassigned_days": unassigned,
        "criteria": asdict(criteria),
        "oos_previously_observed": oos_previously_observed,
        "oos_previously_observed_note": OOS_PREVIOUSLY_OBSERVED_NOTE if oos_previously_observed else None,
        "ranking_by_oos": rank_by_oos(oos),
        "ranking_rule": "OOS PF, then OOS P&L, lower OOS max DD, OOS WR, OOS trade count (TRAIN never used)",
        "status": status,
        "lookahead": {s: per_split[s]["lookahead"] for s in per_split},
        "regime_dependence_oos": {n: regime_dependence(oos_rows[n]) for n in names},
        "daily_oos": daily_oos,
        "final_table": final_table,
        "metrics": {s: per_split[s]["strategies"] for s in per_split},
        "replay_meta": {k: meta.get(k) for k in ("date_range", "excluded_days", "contract_resolution",
                                                   "expired_candle_downloads")},
        "current_day_forward": current_day_forward(current, splits),
        "_per_split": per_split,
    }


# ------------------------------------------------------------------ reports
_pct, _pf, _rs = sh._pct, sh._pf_s, sh._rs


def summary_markdown(s: dict[str, Any]) -> str:
    md = [f"# {s['heading']}", "",
          "PAPER mode | observation only | no orders placed | production S2 unchanged | no optimisation", "",
          f"Strategy selection: {s['strategy_selection']}", "", "## Splits (chronological, no shuffling)", "",
          "| Split | Label | From | To | Sessions |", "|---|---|---|---|---|"]
    for name, sp in s["splits"].items():
        md.append(f"| {name} | {sp['label']} | {sp['from']} | {sp['to']} | {len(sp['days'])} ({', '.join(sp['days'])}) |")
    if s["unassigned_days"]:
        md.append(f"\nSessions outside every split (ignored): {', '.join(s['unassigned_days'])}")
    if s["oos_previously_observed_note"]:
        md += ["", f"> {s['oos_previously_observed_note']}"]
    md += ["", "## Final table", "",
           "| Strategy | Train WR | Train PF | Validation WR | Validation PF | OOS WR | OOS PF | OOS P&L (Rs) | "
           "OOS DD (Rs) | OOS Trades | Status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["final_table"]:
        md.append(f"| {r['strategy']} | {_pct(r['train_wr'])} | {_pf(r['train_pf'])} | {_pct(r['validation_wr'])} | "
                  f"{_pf(r['validation_pf'])} | {_pct(r['oos_wr'])} | {_pf(r['oos_pf'])} | {_rs(r['oos_pnl'])} | "
                  f"{_rs(r['oos_dd'])} | {r['oos_trades']} | {r['status']} |")
    md += ["", f"Ranking by OOS ({s['ranking_rule']}): " + " > ".join(s["ranking_by_oos"]), "",
           "## Robustness status", ""]
    for n, st in s["status"].items():
        md.append(f"- **{n}: {st['status']}**" + (f" - failed: {', '.join(st['failed_checks'])}"
                                                   if st["failed_checks"] else ""))
    for split, by in s["metrics"].items():
        md += ["", f"## {SPLIT_LABELS[split]} ({s['splits'][split]['from']}..{s['splits'][split]['to']})", "",
               "| Strategy | Signals | Evaluated | Wins | Losses | Skipped | Data unavail. | WR | PF | P&L (Rs) | "
               "Avg/trade | Max DD | +Days | -Days | Best day | Worst day | Coverage | Ex-best-day P&L |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, m in by.items():
            best, worst = m["best_day"] or {}, m["worst_day"] or {}
            md.append(f"| {n} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | {m['skipped']} | "
                      f"{m['data_unavailable']} | {_pct(m['win_rate'])} | {_pf(m['profit_factor'])} | "
                      f"{_rs(m['net_pnl'])} | {_rs(m['avg_pnl_per_trade'])} | {_rs(m['max_drawdown'])} | "
                      f"{m['profitable_days']} | {m['losing_days']} | {best.get('date', 'N/A')} "
                      f"{_rs(best.get('net_pnl'))} | {worst.get('date', 'N/A')} {_rs(worst.get('net_pnl'))} | "
                      f"{_pct(m['data_coverage'])} | {_rs(m['excluding_best_day']['net_pnl'])} |")
        la = s["lookahead"][split]
        md.append("\nLookahead mutation: " + "; ".join(
            "%s %s (%d cuts)" % (n, "PASS" if r["passed"] else ("NOT RUN" if not r["cuts"] else "FAIL"), r["cuts"])
            for n, r in la.items()))
        for n, m in by.items():
            md += ["", f"### {n} - {SPLIT_LABELS[split]}", ""]
            md += sh._group_table("Direction", m["direction"])
            md += [""] + sh._group_table("DTE", m["dte"])
    md += ["", "## HISTORICAL OOS by regime (end-of-day labels, reporting only)", ""]
    for n, m in s["metrics"]["OOS"].items():
        md += [f"### {n}", ""] + (sh._group_table("Regime", m["regime"]) if m["regime"]
                                   else ["- No regime label present in OOS sessions."])
        dep = s["regime_dependence_oos"][n]
        md += ["", f"Strong-bearish P&L Rs {_rs(dep['strong_bearish']['net_pnl'])} vs non-bearish Rs "
                   f"{_rs(dep['non_bearish']['net_pnl'])} (bearish share {_pct(dep['bearish_share_of_pnl'])}); "
                   f"profitable without bearish days: {dep['profitable_without_bearish_days']}", ""]
    md += ["## HISTORICAL OOS daily", "",
           "| Date | " + " | ".join(f"{n} trades | {n} WR | {n} P&L (Rs)" for n in s["strategies"]) + " |",
           "|---|" + "---|---|---|" * len(s["strategies"])]
    for d, by in s["daily_oos"].items():
        md.append(f"| {d} | " + " | ".join(f"{x['trades']} | {_pct(x['win_rate'])} | {_rs(x['net_pnl'])}"
                                           for x in by.values()) + " |")
    cur = s["current_day_forward"]
    md += ["", f"## {CURRENT_LABEL} (separate - not merged)", ""]
    if not cur["available"]:
        md.append(f"- {cur['note']}")
    else:
        md += [f"Days: {', '.join(cur['validated_days'])}. {cur['note']}", "",
               "| Strategy | Evaluated | Wins | Losses | WR | PF | P&L (Rs) |", "|---|---|---|---|---|---|---|"]
        for n, m in cur["candidates"].items():
            pf = m.get("profit_factor")
            md.append(f"| {n} | {m.get('evaluated')} | {m.get('wins')} | {m.get('losses')} | "
                      f"{_pct(m.get('win_rate'))} | {_pf(None if pf == 'INF' else pf)} | {_rs(m.get('net_pnl'))} |")
        if cur["overlapping_historical_days"]:
            md.append(f"\nWARNING: current-day dates inside a historical split: {cur['overlapping_historical_days']}")
    md += ["", "---", "Research only. PAPER mode, LIVE_TRADING_ENABLED=false, no orders placed, production S2 unchanged. "
                      "Missing option data stays DATA_UNAVAILABLE; costs and slippage excluded."]
    return "\n".join(md) + "\n"


CSV_FIELDS = ("split", "label", "strategy", "sessions", "signals", "evaluated", "wins", "losses", "skipped",
              "data_unavailable", "win_rate", "profit_factor", "net_pnl", "avg_pnl_per_trade", "max_drawdown",
              "profitable_days", "losing_days", "data_coverage", "excluding_best_day_pnl", "lookahead_passed",
              "status")


def write_reports(study: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Writes only summary.{md,json,csv} and the three split files; nothing else in *out_dir* is touched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_split = study["_per_split"]
    public = {k: v for k, v in study.items() if k != "_per_split"}
    paths = {"md": out_dir / "summary.md", "json": out_dir / "summary.json", "csv": out_dir / "summary.csv"}
    paths["md"].write_text(summary_markdown(public), encoding="utf-8")
    paths["json"].write_text(json.dumps(sh._jsonable(public), indent=2, default=str), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for split, sp in per_split.items():
            for n, m in sp["strategies"].items():
                w.writerow({"split": split, "label": sp["label"], "strategy": n, "sessions": len(sp["days"]),
                            **{k: m[k] for k in CSV_FIELDS if k in m and k not in ("strategy",)},
                            "profit_factor": _pf(m["profit_factor"]),
                            "excluding_best_day_pnl": m["excluding_best_day"]["net_pnl"],
                            "lookahead_passed": sp["lookahead"][n]["passed"],
                            "status": study["status"][n]["status"] if split == "OOS" else ""})
    for split, name in SPLIT_FILES.items():
        sp = per_split[split]
        payload = {"heading": HEADING, "label": sp["label"], "split": split, "from": sp["from"], "to": sp["to"],
                   "days": sp["days"], "trading_mode": "PAPER", "live_trading_enabled": False,
                   "lookahead": sp["lookahead"], "strategies": sp["strategies"], "signals": sp["signals"]}
        paths[split] = out_dir / name
        paths[split].write_text(json.dumps(sh._jsonable(payload), indent=2, default=str), encoding="utf-8")
    return paths
