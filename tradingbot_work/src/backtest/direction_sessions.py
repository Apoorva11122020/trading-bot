"""BUY vs SELL breakdown of the original S2 signals over the latest N stored sessions (PAPER, descriptive only).

Rows come from the historical shadow replay of BASELINE_S2_BUY_SELL (unchanged S2 signals, nearest-expiry ATM
contract, stop 1.5 x premium ATR, target 2R). Nothing is filtered, tuned or ranked; DATA_UNAVAILABLE signals are
excluded from wins, losses, accuracy and P&L.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh

HEADING = "S2 DIRECTION BREAKDOWN - LATEST HISTORICAL SESSIONS"
STRATEGY = dr.BASELINE.name
DIRECTIONS = sh.DIRECTIONS
MIN_EVALUATED_FOR_SAMPLE = 30
METRIC_KEYS = ("signals", "evaluated", "wins", "losses", "win_rate", "profit_factor", "net_pnl", "avg_pnl_per_trade",
               "max_drawdown", "profitable_days", "losing_days", "data_unavailable", "skipped", "data_coverage",
               "data_unavailable_reasons")
GROUP_KEYS = ("signals", "evaluated", "wins", "losses", "win_rate", "profit_factor", "net_pnl")


def outcome_counts(rows: Sequence[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        key = r["outcome"] if r["state"] == sh.EVALUATED else r["state"]
        out[str(key)] = out.get(str(key), 0) + 1
    return out


def direction_metrics(rows: Sequence[dict], days: Sequence[str]) -> dict[str, Any]:
    m = sh.strategy_summary(rows, days)
    out = {k: m[k] for k in METRIC_KEYS}
    out.update(outcomes=outcome_counts(rows), daily=m["daily"],
               dte={k: {g: v[g] for g in GROUP_KEYS} for k, v in m["dte"].items()},
               dte_unknown={g: m["dte_unknown"][g] for g in GROUP_KEYS},
               regime={k: {g: v[g] for g in GROUP_KEYS} for k, v in m["regime"].items() if v["signals"]})
    return out


def _better(a: float | None, b: float | None) -> str:
    if a is None and b is None:
        return "N/A (no evaluated trades)"
    if b is None or (a is not None and a > b):
        return "BUY"
    if a is None or b > a:
        return "SELL"
    return "TIE"


def conclusion(buy: dict, sell: dict, min_sample: int = MIN_EVALUATED_FOR_SAMPLE) -> dict[str, Any]:
    return {
        "higher_accuracy": _better(buy["win_rate"], sell["win_rate"]),
        "better_profit_factor": _better(buy["profit_factor"], sell["profit_factor"]),
        "better_pnl": _better(buy["net_pnl"], sell["net_pnl"]),
        "buy_sample_sufficient": buy["evaluated"] >= min_sample,
        "buy_sample_rule": f"descriptive minimum: >= {min_sample} evaluated BUY trades",
        "buy_evaluated": buy["evaluated"],
        "buy_data_unavailable": buy["data_unavailable"],
        "note": "Historical descriptive analysis only; it does not establish whether BUY is robust or not robust.",
    }


def build_analysis(result: dict[str, Any], meta: dict[str, Any], requested_sessions: int) -> dict[str, Any]:
    days = result["days"]
    rows = sorted(result["rows"].get(STRATEGY, []), key=lambda r: r["signal_time"])
    by_dir = {d: direction_metrics([r for r in rows if r["direction"] == d], days) for d in DIRECTIONS}
    total = direction_metrics(rows, days)
    daily = []
    for d in days:
        row = {"date": d}
        for name in DIRECTIONS:
            x = by_dir[name]["daily"][d]
            row.update({f"{name.lower()}_{k}": x[k] for k in ("signals", "evaluated", "wins", "losses",
                                                               "win_rate", "net_pnl", "data_unavailable")})
        daily.append(row)
    la = result.get("direction_lookahead") or {}
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"),
        "strategy": STRATEGY,
        "strategy_note": "Original S2 signals (BUY -> CE, SELL -> PE, nearest expiry ATM); existing exit logic; "
                         "no filter, optimisation or parameter change.",
        "requested_sessions": requested_sessions, "sessions_selected": len(days), "selected_dates": days,
        "fewer_sessions_than_requested": len(days) < requested_sessions,
        "directions": by_dir, "total": total, "daily": daily,
        "lookahead": {"BUY": la.get("BUY"), "SELL": la.get("SELL"),
                      "TOTAL": (result.get("lookahead") or {}).get(STRATEGY)},
        "conclusion": conclusion(by_dir["BUY"], by_dir["SELL"]),
        "replay_meta": {k: meta.get(k) for k in ("date_range", "excluded_days", "contract_resolution",
                                                   "expired_candle_downloads")},
        "accuracy_definition": "Accuracy = Wins / Evaluated x 100. Evaluated = WIN, LOSS, AMBIGUOUS (conservative "
                               "-1R loss) and SCRATCH (non-win); DATA_UNAVAILABLE and SKIPPED are excluded.",
    }


# ------------------------------------------------------------------ reports
_pct, _pf, _rs = sh._pct, sh._pf_s, sh._rs


def _la(r: dict | None) -> str:
    if not r:
        return "NOT RUN"
    return f"{'PASS' if r['passed'] else 'FAIL'} ({r['cuts']} cuts)"


def summary_markdown(a: dict[str, Any]) -> str:
    md = [f"# {a['heading']}", "", "PAPER mode | research only | no orders placed | production S2 unchanged", "",
          f"Strategy: {a['strategy']} - {a['strategy_note']}", "",
          f"Sessions: {a['sessions_selected']} of {a['requested_sessions']} requested"
          + (" (FEWER THAN REQUESTED)" if a["fewer_sessions_than_requested"] else ""), "",
          "Selected dates: " + ", ".join(a["selected_dates"]), "", "## Direction summary", "",
          "| Direction | Signals | Evaluated | Wins | Losses | Accuracy | PF | P&L (Rs) | Avg/trade | Max DD | "
          "+Days | -Days | Data unavailable |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in (*a["directions"].items(), ("TOTAL", a["total"])):
        md.append(f"| {name} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | "
                  f"{_pct(m['win_rate'])} | {_pf(m['profit_factor'])} | {_rs(m['net_pnl'])} | "
                  f"{_rs(m['avg_pnl_per_trade'])} | {_rs(m['max_drawdown'])} | {m['profitable_days']} | "
                  f"{m['losing_days']} | {m['data_unavailable']} |")
    md += ["", a["accuracy_definition"], "",
           f"**BUY ACCURACY = {_pct(a['directions']['BUY']['win_rate'])}**  ",
           f"**SELL ACCURACY = {_pct(a['directions']['SELL']['win_rate'])}**  ",
           f"**TOTAL ACCURACY = {_pct(a['total']['win_rate'])}**", "",
           "Outcomes: " + "; ".join(f"{n}: {m['outcomes']}" for n, m in a["directions"].items()), "",
           "Lookahead mutation: " + "; ".join(f"{k} {_la(v)}" for k, v in a["lookahead"].items()), "",
           "## Daily breakdown", "",
           "| Date | BUY Signals | BUY Evaluated | BUY Wins | BUY Losses | BUY WR | BUY P&L | SELL Signals | "
           "SELL Evaluated | SELL Wins | SELL Losses | SELL WR | SELL P&L |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["daily"]:
        md.append(f"| {r['date']} | {r['buy_signals']} | {r['buy_evaluated']} | {r['buy_wins']} | {r['buy_losses']} | "
                  f"{_pct(r['buy_win_rate'])} | {_rs(r['buy_net_pnl'])} | {r['sell_signals']} | "
                  f"{r['sell_evaluated']} | {r['sell_wins']} | {r['sell_losses']} | {_pct(r['sell_win_rate'])} | "
                  f"{_rs(r['sell_net_pnl'])} |")
    for name, m in a["directions"].items():
        md += ["", f"## {name} by DTE", ""] + sh._group_table("DTE", m["dte"])
        if m["dte_unknown"]["signals"]:
            md.append(f"\nDTE unknown: {m['dte_unknown']['signals']} signals")
        md += ["", f"## {name} by regime (end-of-day labels, reporting only)", ""]
        md += sh._group_table("Regime", m["regime"]) if m["regime"] else ["- No regime label present."]
    c = a["conclusion"]
    md += ["", "## BUY vs SELL conclusion", "",
           f"- Higher accuracy: {c['higher_accuracy']}", f"- Better PF: {c['better_profit_factor']}",
           f"- Better P&L: {c['better_pnl']}",
           f"- BUY sample sufficient ({c['buy_sample_rule']}): {'YES' if c['buy_sample_sufficient'] else 'NO'}",
           f"- BUY evaluated: {c['buy_evaluated']}", f"- BUY data unavailable: {c['buy_data_unavailable']}",
           f"- {c['note']}", "", "---",
           "PAPER mode, LIVE_TRADING_ENABLED=false, no orders placed. Real NIFTY and expired option candles only; "
           "missing option data stays DATA_UNAVAILABLE. Costs and slippage excluded."]
    return "\n".join(md) + "\n"


def _write_csv(path: Path, fields: Sequence[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: _pf(v) if k.endswith("profit_factor") else v for k, v in r.items()})


def write_reports(a: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Writes only the six files of this report; nothing else in *out_dir* is touched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in ("summary.md", "summary.json", "summary.csv", "daily.csv", "dte.csv", "regime.csv")}
    p["summary.md"].write_text(summary_markdown(a), encoding="utf-8")
    p["summary.json"].write_text(json.dumps(sh._jsonable(a), indent=2, default=str), encoding="utf-8")
    flat = ("signals", "evaluated", "wins", "losses", "win_rate", "profit_factor", "net_pnl", "avg_pnl_per_trade",
            "max_drawdown", "profitable_days", "losing_days", "data_unavailable", "skipped", "data_coverage")
    _write_csv(p["summary.csv"], ("direction", *flat, "lookahead_passed"),
               [{"direction": n, **{k: m[k] for k in flat},
                 "lookahead_passed": (a["lookahead"].get(n) or {}).get("passed")}
                for n, m in (*a["directions"].items(), ("TOTAL", a["total"]))])
    _write_csv(p["daily.csv"], tuple(a["daily"][0]) if a["daily"] else ("date",), a["daily"])
    _write_csv(p["dte.csv"], ("direction", "dte_bucket", *GROUP_KEYS),
               [{"direction": n, "dte_bucket": b, **g} for n, m in a["directions"].items() for b, g in m["dte"].items()])
    _write_csv(p["regime.csv"], ("direction", "regime", *GROUP_KEYS),
               [{"direction": n, "regime": b, **g} for n, m in a["directions"].items()
                for b, g in m["regime"].items()])
    return p
