"""LIVE QUALITY GATE V1 daily / cumulative reports, built only from stored shadow_strategy_signals rows.

PAPER | OBSERVATION ONLY | NO ORDERS. Generated S2 signals, gate-accepted signals, resolved trades and pending
trades are always reported separately. Accuracy = wins / resolved; pending is never a loss. A 70% target is only
"ACHIEVED" with at least ``min_live_samples`` resolved trades, accuracy >= 70% and PF >= 1. Nothing is guaranteed.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from src.backtest import live_quality_gate as qg
from src.backtest import s2_1dte_improvement as imp
from src.database import Database
from src.paper_trading import live_quality_monitor as mon

IST = imp.IST
SCOPES = ("BUY", "SELL", "COMBINED")
BASELINE = "BASELINE_S2_BUY_SELL"
DISCLAIMER = ("Research-derived observation filter. Not a proven live edge; 70% accuracy is a target, never a "
              "guarantee. One day of results proves nothing.")
SAFETY = {"trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
          "production_s2_changed": False, "live_orders": 0}


def _loads(v: Any, default: Any) -> Any:
    if not v:
        return default
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def gate_rows(db: Database, day: str | None = None) -> list[dict]:
    rows = [r for r in db.get_shadow_signals(day) if r["candidate"] == qg.GATE_CANDIDATE]
    for r in rows:
        r["gate_reasons"] = _loads(r.get("gate_reasons"), [])
        r["gate_failed_checks"] = _loads(r.get("gate_failed_checks"), [])
        r["gate_features"] = _loads(r.get("gate_features"), {})
    return sorted(rows, key=lambda r: (r["session_date"], r["timestamp"]))


def scoped(rows: Sequence[dict]) -> dict[str, dict]:
    return {"BUY": mon.perf([r for r in rows if r["direction"] == "BUY"]),
            "SELL": mon.perf([r for r in rows if r["direction"] == "SELL"]), "COMBINED": mon.perf(rows)}


def score_distribution(rows: Sequence[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for score in sorted({r.get("gate_score") for r in rows}, key=lambda s: -1 if s is None else s):
        sub = [r for r in rows if r.get("gate_score") == score]
        m = mon.perf(sub)
        out["UNSCORED" if score is None else str(score)] = {k: m[k] for k in (
            "generated", "accepted", "resolved", "wins", "losses", "pending", "accuracy", "profit_factor", "pnl")}
    return out


def skip_reasons(rows: Sequence[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        if r["decision"] == imp.ACCEPTED:
            continue
        reason = r.get("skip_reason") or "UNKNOWN"
        if reason.startswith(qg.REJECTED + ":"):
            c.update(x for x in reason.split(":", 1)[1].split(",") if x)
        else:
            c[reason] += 1
    return dict(c.most_common())


def missing_features(rows: Sequence[dict]) -> dict[str, int]:
    c: Counter = Counter()
    for r in rows:
        for chk in r["gate_failed_checks"]:
            if chk.startswith(qg.DATA_UNAVAILABLE + ":"):
                c[chk.split(":", 1)[1]] += 1
    return dict(c.most_common())


def accepted_details(rows: Sequence[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["decision"] != imp.ACCEPTED:
            continue
        resolved = r["status"] in mon.RESOLVED
        out.append({"signal_time": r["timestamp"], "direction": r["direction"], "score": r.get("gate_score"),
                    "max_score": r.get("gate_max_score"), "confidence": r.get("gate_confidence"),
                    "passed_checks": r["gate_reasons"], "instrument_key": r["instrument_key"],
                    "entry_price": r["entry_price"], "stop_loss": r["stop_loss"], "target": r["target"],
                    "exit_price": r["exit_price"], "exit_time": r["exit_time"], "exit_reason": r["exit_reason"],
                    "status": r["status"], "pnl": round(mon.trade_pnl(r), 2) if resolved else None,
                    "r_multiple": r["r_multiple"] if resolved else None})
    return out


def _counts(rows: Sequence[dict]) -> dict[str, Any]:
    gate_unavail = sum(any(c.startswith(qg.DATA_UNAVAILABLE) for c in r["gate_failed_checks"]) for r in rows)
    exec_unavail = sum(r["decision"] == imp.ACCEPTED and r["status"] == imp.UNAVAILABLE for r in rows)
    generated = len(rows)
    accepted = sum(r["decision"] == imp.ACCEPTED for r in rows)
    return {"generated": generated, "accepted": accepted, "skipped": generated - accepted,
            "skip_percentage": round((generated - accepted) / generated * 100, 2) if generated else None,
            "missing_feature_count": gate_unavail, "missing_features": missing_features(rows),
            "data_unavailable_count": gate_unavail + exec_unavail,
            "gate_data_unavailable_count": gate_unavail, "execution_data_unavailable_count": exec_unavail}


def live_signal_accuracy(m: dict, status: str) -> dict[str, Any]:
    return {"signals_generated": m["generated"], "signals_accepted": m["accepted"], "signals_skipped": m["skipped"],
            "resolved": m["resolved"], "wins": m["wins"], "losses": m["losses"], "pending": m["pending"],
            "data_unavailable": m["data_unavailable"], "accuracy": m["accuracy"],
            "profit_factor": m["profit_factor"], "pnl": m["pnl"], "target_status": status}


def daily_report(db: Database, day: str, config: qg.QualityGateConfig = qg.QualityGateConfig(),
                 run_status: dict | None = None) -> dict[str, Any]:
    rows = gate_rows(db, day)
    upto = [r for r in gate_rows(db) if r["session_date"] <= day]
    sc = scoped(rows)
    cum = mon.perf(upto)
    counts = _counts(rows)
    if rows:
        status, reason = "VALIDATED", None
    elif run_status and run_status.get("status") == qg.DATA_UNAVAILABLE:
        status, reason = qg.DATA_UNAVAILABLE, run_status.get("reason")
    else:
        status, reason = "NO_STORED_DATA", "no LIVE_QUALITY_GATE_V1 rows stored for this date"
    daily_status = mon.target_status(sc["COMBINED"], config.min_live_samples, config.target_accuracy)
    cum_status = mon.target_status(cum, config.min_live_samples, config.target_accuracy)
    return {
        "report": "LIVE QUALITY GATE V1", "date": day, "gate_version": qg.GATE_VERSION, "status": status,
        "data_reason": reason, **SAFETY, "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "config": {"min_score_buy": config.min_score_buy, "min_score_sell": config.min_score_sell,
                   "target_accuracy": config.target_accuracy, "min_live_samples": config.min_live_samples,
                   "min_accept_rate": config.min_accept_rate, "max_skip_ratio": config.max_skip_ratio,
                   "allow_partial_features": config.allow_partial_features},
        "baseline_s2_signals": sum(r["candidate"] == BASELINE for r in db.get_shadow_signals(day)),
        **counts,
        "restrictiveness": qg.restrictiveness(counts["generated"], counts["accepted"], config),
        "scopes": sc,
        "quality_gate_score_distribution": score_distribution(rows),
        "accepted_signal_details": accepted_details(rows),
        "skipped_signal_reasons": skip_reasons(rows),
        "live_signal_accuracy": live_signal_accuracy(sc["COMBINED"], daily_status),
        "target_status_daily": daily_status,
        "cumulative_to_date": {**live_signal_accuracy(cum, cum_status),
                               "sessions": len({r["session_date"] for r in upto})},
        "target_status_cumulative": cum_status,
    }


def cumulative_report(db: Database, config: qg.QualityGateConfig = qg.QualityGateConfig()) -> dict[str, Any]:
    rows = gate_rows(db)
    sessions = sorted({r["session_date"] for r in rows})
    total = mon.perf(rows)
    counts = _counts(rows)
    months = sorted({d[:7] for d in sessions})
    return {
        "report": "LIVE QUALITY GATE V1 - cumulative", "gate_version": qg.GATE_VERSION, **SAFETY,
        "disclaimer": DISCLAIMER, "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "sessions": len(sessions), "session_dates": sessions, **counts,
        "restrictiveness": qg.restrictiveness(counts["generated"], counts["accepted"], config),
        "combined": total,
        "target_status": mon.target_status(total, config.min_live_samples, config.target_accuracy),
        "min_live_samples": config.min_live_samples, "target_accuracy": config.target_accuracy,
        "direction_breakdown": scoped(rows),
        "monthly_breakdown": {m: mon.perf([r for r in rows if r["session_date"].startswith(m)]) for m in months},
        "gate_score_breakdown": score_distribution(rows),
        "rolling": mon.rolling(rows, min_samples=config.min_live_samples, target=config.target_accuracy),
        "daily": [{"date": d, **{k: v for k, v in mon.perf([r for r in rows if r["session_date"] == d]).items()
                                 if k in ("generated", "accepted", "resolved", "wins", "losses", "pending",
                                          "accuracy", "profit_factor", "pnl", "max_drawdown")}}
                  for d in sessions],
        "skipped_signal_reasons": skip_reasons(rows),
    }


# ------------------------------------------------------------------ rendering
def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and math.isinf(v):
        return None
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def pf(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}"


def money(v: float | None, symbol: str = "₹") -> str:
    if v is None:
        return "N/A"
    return f"{'+' if v > 0 else '-' if v < 0 else ''}{symbol}{abs(v):,.2f}"


def _scope_table(sc: dict[str, dict]) -> list[str]:
    out = ["| Scope | Generated | Accepted | Resolved | Wins | Losses | Pending | Data unavailable | Accuracy | PF | "
           "P&L | Max DD |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for name in SCOPES:
        m = sc[name]
        out.append(f"| {name} | {m['generated']} | {m['accepted']} | {m['resolved']} | {m['wins']} | {m['losses']} | "
                   f"{m['pending']} | {m['data_unavailable']} | {pct(m['accuracy'])} | {pf(m['profit_factor'])} | "
                   f"{money(m['pnl'])} | {money(m['max_drawdown'])} |")
    return out


def _score_table(dist: dict[str, dict]) -> list[str]:
    out = ["| Gate score | Generated | Accepted | Resolved | Wins | Losses | Pending | Accuracy | P&L |",
           "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for s, m in dist.items():
        out.append(f"| {s} | {m['generated']} | {m['accepted']} | {m['resolved']} | {m['wins']} | {m['losses']} | "
                   f"{m['pending']} | {pct(m['accuracy'])} | {money(m['pnl'])} |")
    return out


def _summary(title: str, s: dict, status: str, min_samples: int) -> list[str]:
    return [f"## {title}", "", f"- Signals generated: {s['signals_generated']}",
            f"- Signals accepted: {s['signals_accepted']}", f"- Signals skipped: {s['signals_skipped']}",
            f"- Resolved: {s['resolved']}", f"- Wins: {s['wins']}", f"- Losses: {s['losses']}",
            f"- Pending (not counted as losses): {s['pending']}",
            f"- Data unavailable (not counted): {s['data_unavailable']}",
            f"- Accuracy (wins / resolved): {pct(s['accuracy'])}", f"- Profit Factor: {pf(s['profit_factor'])}",
            f"- P&L: {money(s['pnl'])}", "",
            f"**70% Target Status: {status}** (needs >= {min_samples} resolved, accuracy >= 70%, PF >= 1.0)", ""]


def daily_markdown(rep: dict) -> str:
    md = [f"# LIVE QUALITY GATE V1 - {rep['date']}", "",
          "PAPER | OBSERVATION ONLY | NO ORDERS | production S2 unchanged | live orders: 0", "",
          f"> {rep['disclaimer']}", "",
          f"- Gate version: {rep['gate_version']}", f"- Report status: {rep['status']}"]
    if rep["data_reason"]:
        md.append(f"- Data reason: {rep['data_reason']}")
    r = rep["restrictiveness"]
    md += [f"- Generated S2 signals: {rep['generated']} (baseline S2 rows stored: {rep['baseline_s2_signals']})",
           f"- Accepted signals: {rep['accepted']}", f"- Skipped signals: {rep['skipped']}",
           f"- Skip percentage: {'N/A' if rep['skip_percentage'] is None else str(rep['skip_percentage']) + '%'}",
           f"- Overfilter guard: {r['status']}"
           + ("" if r["status"] != qg.TOO_RESTRICTIVE else " - accepted sample too small; accuracy is NOT meaningful"),
           ""]
    md += _summary("Live Signal Accuracy (this session)", rep["live_signal_accuracy"], rep["target_status_daily"],
                   rep["config"]["min_live_samples"])
    cum = rep["cumulative_to_date"]
    md += _summary(f"Live Signal Accuracy (cumulative to date, {cum['sessions']} session(s))", cum,
                   rep["target_status_cumulative"], rep["config"]["min_live_samples"])
    md += ["## BUY / SELL / COMBINED", ""] + _scope_table(rep["scopes"])
    md += ["", "## Quality gate score distribution", ""] + _score_table(rep["quality_gate_score_distribution"])
    md += ["", "## Accepted signal details", "",
           "| Time | Dir | Score | Passed checks | Entry | SL | Target | Exit | Exit reason | Status | P&L |",
           "|---|---|---|---|---:|---:|---:|---:|---|---|---:|"]
    for a in rep["accepted_signal_details"]:
        md.append(f"| {a['signal_time'][11:16]} | {a['direction']} | {a['score']}/{a['max_score']} | "
                  f"{', '.join(a['passed_checks'])} | {a['entry_price'] or ''} | {a['stop_loss'] or ''} | "
                  f"{a['target'] or ''} | {a['exit_price'] or ''} | {a['exit_reason'] or ''} | {a['status']} | "
                  f"{'' if a['pnl'] is None else money(a['pnl'])} |")
    if not rep["accepted_signal_details"]:
        md.append("| - | - | - | none accepted | | | | | | | |")
    md += ["", "## Skipped signal reasons", ""]
    md += [f"- {k}: {v}" for k, v in rep["skipped_signal_reasons"].items()] or ["- none"]
    md += ["", "## Data availability", "", f"- Missing-feature signals: {rep['missing_feature_count']} "
           f"({', '.join(f'{k}: {v}' for k, v in rep['missing_features'].items()) or 'none'})",
           f"- Data unavailable total: {rep['data_unavailable_count']} (gate features "
           f"{rep['gate_data_unavailable_count']}, option premium/exit data {rep['execution_data_unavailable_count']})",
           "", "Missing data is never synthesised."]
    return "\n".join(md) + "\n"


def cumulative_markdown(rep: dict) -> str:
    c = rep["combined"]
    md = ["# LIVE QUALITY GATE V1 - cumulative", "",
          "PAPER | OBSERVATION ONLY | NO ORDERS | production S2 unchanged | live orders: 0", "",
          f"> {rep['disclaimer']}", "",
          f"- Sessions: {rep['sessions']} ({', '.join(rep['session_dates']) or 'none'})",
          f"- Generated: {rep['generated']} | Accepted: {rep['accepted']} | Skipped: {rep['skipped']}",
          f"- Resolved: {c['resolved']} | Wins: {c['wins']} | Losses: {c['losses']} | Pending: {c['pending']} | "
          f"Data unavailable: {c['data_unavailable']}",
          f"- Accuracy (wins / resolved): {pct(c['accuracy'])} | PF: {pf(c['profit_factor'])} | "
          f"P&L: {money(c['pnl'])} | Max DD: {money(c['max_drawdown'])} | Max losing streak: {c['max_losing_streak']}",
          f"- Overfilter guard: {rep['restrictiveness']['status']}", "",
          f"**70% Target Status: {rep['target_status']}** (needs >= {rep['min_live_samples']} resolved, "
          "accuracy >= 70%, PF >= 1.0)", "",
          "## BUY vs SELL", ""] + _scope_table(rep["direction_breakdown"])
    md += ["", "## Monthly", "", "| Month | Generated | Accepted | Resolved | Wins | Losses | Pending | Accuracy | PF | "
           "P&L | Max DD |", "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for mth, m in rep["monthly_breakdown"].items():
        md.append(f"| {mth} | {m['generated']} | {m['accepted']} | {m['resolved']} | {m['wins']} | {m['losses']} | "
                  f"{m['pending']} | {pct(m['accuracy'])} | {pf(m['profit_factor'])} | {money(m['pnl'])} | "
                  f"{money(m['max_drawdown'])} |")
    md += ["", "## Gate score breakdown", ""] + _score_table(rep["gate_score_breakdown"])
    md += ["", "## Rolling monitor (reporting only - never changes the strategy)", "",
           "| Window | Resolved | Accuracy | PF | P&L | Max DD | Status |", "|---|---:|---|---|---:|---:|---|"]
    for name, w in rep["rolling"].items():
        md.append(f"| {name} | {w['resolved']} | {pct(w['accuracy'])} | {pf(w['profit_factor'])} | "
                  f"{money(w['pnl'])} | {money(w['max_drawdown'])} | {w['status']} |")
    md += ["", "## Daily", "", "| Date | Generated | Accepted | Resolved | Wins | Losses | Pending | Accuracy | P&L |",
           "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for d in rep["daily"]:
        md.append(f"| {d['date']} | {d['generated']} | {d['accepted']} | {d['resolved']} | {d['wins']} | "
                  f"{d['losses']} | {d['pending']} | {pct(d['accuracy'])} | {money(d['pnl'])} |")
    return "\n".join(md) + "\n"


def write_daily(rep: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rep['date']}.json"
    path.write_text(json.dumps(_jsonable(rep), indent=2, default=str), encoding="utf-8")
    (out_dir / f"{rep['date']}.md").write_text(daily_markdown(rep), encoding="utf-8")
    return path


def write_cumulative(rep: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cumulative.json"
    path.write_text(json.dumps(_jsonable(rep), indent=2, default=str), encoding="utf-8")
    (out_dir / "cumulative.md").write_text(cumulative_markdown(rep), encoding="utf-8")
    return path


def run_status_path(out_dir: Path, day: str) -> Path:
    return out_dir / f"{day}.run_status.json"


def read_run_status(out_dir: Path, day: str) -> dict | None:
    p = run_status_path(out_dir, day)
    return _loads(p.read_text(encoding="utf-8"), None) if p.exists() else None


def console_summary(rep: dict) -> str:
    s = rep["live_signal_accuracy"]
    lines = ["LIVE QUALITY GATE V1", f"Date: {rep['date']}", f"Report status: {rep['status']}"]
    if rep["data_reason"]:
        lines.append(f"Data reason: {rep['data_reason']}")
    lines += [f"Generated: {s['signals_generated']}", f"Accepted: {s['signals_accepted']}",
              f"Skipped: {s['signals_skipped']}", f"Resolved: {s['resolved']}", f"Wins: {s['wins']}",
              f"Losses: {s['losses']}", f"Pending: {s['pending']}", f"Accuracy: {pct(s['accuracy'])}",
              f"PF: {pf(s['profit_factor'])}", f"P&L: {money(s['pnl'], 'Rs ')}",
              f"Overfilter guard: {rep['restrictiveness']['status']}",
              f"Target status (day): {rep['target_status_daily']}",
              f"Target status (cumulative): {rep['target_status_cumulative']}",
              "Production S2 changed: NO", "Live orders: 0"]
    return "\n".join(lines)
