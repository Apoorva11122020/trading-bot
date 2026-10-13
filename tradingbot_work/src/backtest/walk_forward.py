"""Walk-forward (development vs out-of-sample) validation for S2 research."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.robustness import MIN_RESOLVED_TRADES

IST = ZoneInfo("Asia/Kolkata")

MIN_OOS_TRADING_DAYS = 20
MIN_OOS_TRADES = MIN_RESOLVED_TRADES

PERIOD_METRICS: list[tuple[str, str]] = [
    ("trading_days", "Trading days"),
    ("total_signals", "Signals"),
    ("resolved_trades", "Resolved trades"),
    ("win_rate", "Win rate"),
    ("profit_factor", "Profit factor"),
    ("net_pnl", "Net P&L"),
    ("gross_profit", "Gross profit"),
    ("gross_loss", "Gross loss"),
    ("max_drawdown", "Max drawdown"),
    ("average_r", "Average R"),
    ("median_r", "Median R"),
    ("average_trades_per_day", "Average trades/day"),
    ("max_consecutive_wins", "Max consecutive wins"),
    ("max_consecutive_losses", "Max consecutive losses"),
    ("unresolved_signals", "Unresolved signals"),
    ("blocked_signals", "Blocked signals (all reasons)"),
    ("regime_filtered_signals", "Blocked by regime filter"),
    ("daily_loss_limit_skips", "Skipped by ₹3,000 daily limit"),
    ("option_coverage_percent", "Option coverage %"),
]


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date

    def contains(self, value: str | pd.Timestamp) -> bool:
        ts = pd.Timestamp(value)
        ts = ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)
        return self.start <= ts.date() <= self.end


def validate_split(development: Period, out_of_sample: Period) -> None:
    if development.start > development.end or out_of_sample.start > out_of_sample.end:
        raise ValueError("Period start must be on or before its end")
    if out_of_sample.start <= development.end:
        raise ValueError("Out-of-sample period must start after the development period ends")


def _local_dates(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series)
    ts = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    return ts.dt.date


def slice_candles(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    days = _local_dates(frame["timestamp"])
    return frame.loc[(days >= start) & (days <= end)].reset_index(drop=True)


def slice_premium(premium: dict[str, pd.DataFrame], start: date, end: date) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, frame in premium.items():
        sliced = slice_candles(frame, start, end)
        if sliced is not None and not sliced.empty:
            out[key] = sliced
    return out


def coverage_for_period(coverage: dict[str, Any], signal_times: list[str], period: Period) -> dict[str, Any]:
    """Restrict a full-window coverage report to signals inside *period*."""
    in_period = [t for t in signal_times if period.contains(t)]
    unavailable = [u for u in coverage.get("unavailable", [])
                   if u.get("signal_time") and period.contains(u["signal_time"])]
    total = len(in_period)
    missing = min(total, len({u["signal_time"] for u in unavailable}))
    covered = total - missing
    return {
        **coverage,
        "from_date": period.start.isoformat(),
        "to_date": period.end.isoformat(),
        "unavailable": unavailable,
        "total_signals": total,
        "signals_missing_coverage": missing,
        "signals_with_complete_coverage": covered,
        "coverage_percent": round(100.0 * covered / total, 2) if total else 0.0,
    }


def _holds(summary: dict[str, Any]) -> bool:
    return summary["profit_factor"] > 1.0 and summary["net_pnl"] > 0


def _filtered_better(original: dict[str, Any], filtered: dict[str, Any]) -> bool:
    return (filtered["profit_factor"] > original["profit_factor"]
            and filtered["net_pnl"] >= original["net_pnl"]
            and filtered["max_drawdown"] <= original["max_drawdown"])


def _original_better(original: dict[str, Any], filtered: dict[str, Any]) -> bool:
    return original["profit_factor"] > filtered["profit_factor"] and original["net_pnl"] > filtered["net_pnl"]


def walk_forward_verdict(results: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, list[str]]:
    """results[strategy][period] -> summary; strategies 'original'/'filtered', periods 'development'/'oos'."""
    o_dev, o_oos = results["original"]["development"], results["original"]["oos"]
    f_dev, f_oos = results["filtered"]["development"], results["filtered"]["oos"]
    reasons: list[str] = []
    if o_oos["trading_days"] < MIN_OOS_TRADING_DAYS:
        reasons.append(f"OOS has only {o_oos['trading_days']} trading days (< {MIN_OOS_TRADING_DAYS})")
    if min(o_oos["resolved_trades"], f_oos["resolved_trades"]) < MIN_OOS_TRADES:
        reasons.append(f"OOS resolved trades original={o_oos['resolved_trades']}, "
                       f"filtered={f_oos['resolved_trades']} (< {MIN_OOS_TRADES})")
    if reasons:
        return "NEEDS MORE DATA", reasons
    if _filtered_better(o_dev, f_dev) and _filtered_better(o_oos, f_oos):
        return "PREFER FILTERED", ["Filtered S2 beats original on PF and net P&L without larger drawdown in both periods"]
    if _original_better(o_oos, f_oos):
        return "PREFER ORIGINAL", ["Original S2 has higher OOS profit factor and net P&L"]
    if _holds(o_oos) and _holds(f_oos):
        return "ROBUST", ["Both variants keep PF > 1 and positive net P&L out of sample"]
    return "NEEDS MORE DATA", ["OOS performance does not hold up for either variant"]


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    if key == "win_rate":
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _change(key: str, a: Any, b: Any) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or float("inf") in (a, b):
        return "n/a"
    delta = b - a
    if key == "win_rate":
        return f"{delta * 100:+.1f} pp"
    if isinstance(a, int) and isinstance(b, int):
        return f"{delta:+d}"
    return f"{delta:+,.2f}"


def _period_table(title: str, original: dict[str, Any], filtered: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", "", "| Metric | Original S2 | Filtered S2 | Change |", "| --- | ---: | ---: | ---: |"]
    for key, label in PERIOD_METRICS:
        lines.append(f"| {label} | {_fmt(key, original.get(key))} | {_fmt(key, filtered.get(key))} | "
                     f"{_change(key, original.get(key), filtered.get(key))} |")
    return lines + [""]


def _daily_table(original: dict[str, Any], filtered: dict[str, Any]) -> list[str]:
    o_days = {r["label"]: r["net_pnl"] for r in original["daily_pnl"]}
    f_days = {r["label"]: r["net_pnl"] for r in filtered["daily_pnl"]}
    lines = ["| Day | Original S2 | Filtered S2 |", "| --- | ---: | ---: |"]
    for day in sorted(set(o_days) | set(f_days)):
        lines.append(f"| {day} | {o_days.get(day, 0.0):,.2f} | {f_days.get(day, 0.0):,.2f} |")
    return lines + [""]


def write_walk_forward_reports(report: dict[str, Any], out_dir: Path,
                               stem: str = "NIFTY_S2_walk_forward") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, md_path = (out_dir / f"{stem}.{ext}" for ext in ("json", "csv", "md"))
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    results = report["results"]
    rows: list[dict[str, Any]] = []
    for period in ("development", "oos"):
        for strategy in ("original", "filtered"):
            summary = results[strategy][period]
            for key, label in PERIOD_METRICS:
                rows.append({"section": "metric", "period": period, "strategy": strategy,
                             "metric": label, "value": summary.get(key)})
            for day in summary["daily_pnl"]:
                rows.append({"section": "daily_pnl", "period": period, "strategy": strategy,
                             "metric": day["label"], "value": day["net_pnl"]})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "period", "strategy", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    dev, oos = report["development"], report["out_of_sample"]
    o, f = results["original"], results["filtered"]
    md = [
        f"# {report['underlying']} {report['strategy_id']} — Walk-Forward Validation",
        "",
        f"**Development:** {dev['start']} to {dev['end']}  ",
        f"**Out-of-sample:** {oos['start']} to {oos['end']}  ",
        f"**Analysis run:** `{report['run_id']}`  ",
        "**Mode:** PAPER research only — no broker orders, live trading disabled. "
        "Historical simulation; not a claim of future profitability.",
        "",
        f"Regime filter frozen: avoids {', '.join(report['avoid_regimes'])}. "
        "No thresholds or parameters were changed for the OOS run.",
        "",
    ]
    md += _period_table("1. Development results", o["development"], f["development"])
    md += ["**Development daily P&L**", ""] + _daily_table(o["development"], f["development"])
    md += _period_table("2. Out-of-sample results", o["oos"], f["oos"])
    md += ["**OOS daily P&L**", ""] + _daily_table(o["oos"], f["oos"])
    md += ["## 3. Original S2 vs Filtered S2", "",
           "| Period | Original PF | Filtered PF | Original net | Filtered net | Original DD | Filtered DD |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for period, label in (("development", "Development"), ("oos", "Out-of-sample")):
        md.append(f"| {label} | {_fmt('profit_factor', o[period]['profit_factor'])} | "
                  f"{_fmt('profit_factor', f[period]['profit_factor'])} | {_fmt('net_pnl', o[period]['net_pnl'])} | "
                  f"{_fmt('net_pnl', f[period]['net_pnl'])} | {_fmt('max_drawdown', o[period]['max_drawdown'])} | "
                  f"{_fmt('max_drawdown', f[period]['max_drawdown'])} |")
    md += ["", "## 4. OOS performance vs development", "",
           "| Strategy | Dev win rate | OOS win rate | Dev PF | OOS PF | Dev avg R | OOS avg R | Dev trades/day | OOS trades/day |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for strategy, label in (("original", "Original S2"), ("filtered", "Filtered S2")):
        d, s = results[strategy]["development"], results[strategy]["oos"]
        md.append(f"| {label} | {_fmt('win_rate', d['win_rate'])} | {_fmt('win_rate', s['win_rate'])} | "
                  f"{_fmt('profit_factor', d['profit_factor'])} | {_fmt('profit_factor', s['profit_factor'])} | "
                  f"{_fmt('average_r', d['average_r'])} | {_fmt('average_r', s['average_r'])} | "
                  f"{_fmt('x', d['average_trades_per_day'])} | {_fmt('x', s['average_trades_per_day'])} |")
    md += ["", f"## 5. Verdict: **{report['verdict']}**", ""]
    md += [f"- {r}" for r in report["verdict_reasons"]]
    md += [f"- Caveat: {c}" for c in report.get("caveats", [])]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}
