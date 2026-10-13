"""Original vs regime-filtered S2 comparison and reports."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

from src.backtest.options_engine import DAILY_LOSS, REGIME_FILTERED
from src.backtest.robustness import MIN_RESOLVED_TRADES, analyze_robustness

MIN_TRADING_DAYS = 30
LOW_FREQUENCY_RATIO = 0.5

COMPARISON_METRICS: list[tuple[str, str, bool | None]] = [
    # (key, label, higher_is_better; None = informational)
    ("trading_days", "Trading days", None),
    ("total_signals", "Total signals", None),
    ("resolved_trades", "Resolved trades", None),
    ("win_rate", "Win rate", True),
    ("profit_factor", "Profit factor", True),
    ("net_pnl", "Net P&L", True),
    ("gross_profit", "Gross profit", True),
    ("gross_loss", "Gross loss", False),
    ("max_drawdown", "Max drawdown", False),
    ("average_r", "Average R", True),
    ("median_r", "Median R", True),
    ("average_trades_per_day", "Average trades/day", None),
    ("max_consecutive_wins", "Max consecutive wins", True),
    ("max_consecutive_losses", "Max consecutive losses", False),
    ("max_daily_loss", "Worst day P&L", True),
    ("blocked_signals", "Blocked signals (all reasons)", None),
    ("regime_filtered_signals", "Signals blocked by regime filter", None),
    ("daily_loss_limit_skips", "Signals skipped by ₹3,000 daily limit", None),
    ("option_coverage_percent", "Option coverage %", None),
    ("trades_removed_by_filter", "Original trades removed", None),
    ("trades_added_by_filter", "New trades (sequencing)", None),
]


def filtered_coverage(coverage: dict[str, Any], regime_blocked_times: set[str]) -> dict[str, Any]:
    """Coverage restricted to signals the filtered strategy still evaluates."""
    unavailable = [u for u in coverage.get("unavailable", []) if u.get("signal_time") not in regime_blocked_times]
    total = max(0, int(coverage.get("total_signals", 0)) - len(regime_blocked_times))
    missing = min(total, len({u.get("signal_time") for u in unavailable if u.get("signal_time")}))
    covered = max(0, total - missing)
    return {
        **coverage,
        "unavailable": unavailable,
        "total_signals": total,
        "signals_missing_coverage": missing,
        "signals_with_complete_coverage": covered,
        "coverage_percent": round(100.0 * covered / total, 2) if total else 0.0,
    }


def summarize(
    result: dict[str, Any],
    *,
    coverage: dict[str, Any],
    trading_days: int,
    from_date: date,
    to_date: date,
    initial_equity: float,
    daily_loss_limit_inr: float,
    label: str,
) -> dict[str, Any]:
    report = analyze_robustness(
        run_id=result.get("run_id", label),
        source_run_id=label,
        underlying=result.get("underlying", "NIFTY"),
        strategy_id=result.get("strategy_id", "S2"),
        from_date=from_date,
        to_date=to_date,
        trading_days=trading_days,
        result=result,
        coverage=coverage,
        initial_equity=initial_equity,
        daily_loss_limit_inr=daily_loss_limit_inr,
    )
    skipped = result.get("skipped", [])
    return {
        "trading_days": report.total_trading_days,
        "total_signals": report.total_signals,
        "resolved_trades": report.resolved_trades,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
        "net_pnl": report.net_pnl,
        "gross_profit": report.gross_profit,
        "gross_loss": report.gross_loss,
        "max_drawdown": report.max_drawdown,
        "average_r": report.average_r,
        "median_r": report.median_r,
        "average_trades_per_day": report.average_trades_per_day,
        "max_consecutive_wins": report.max_consecutive_wins,
        "max_consecutive_losses": report.max_consecutive_losses,
        "max_daily_loss": report.max_daily_loss,
        "blocked_signals": len(skipped),
        "regime_filtered_signals": sum(1 for s in skipped if s.get("reason") == REGIME_FILTERED),
        "daily_loss_limit_skips": sum(1 for s in skipped if s.get("reason") == DAILY_LOSS),
        "option_coverage_percent": report.option_premium_coverage.get("coverage_percent", 0.0),
        "option_coverage": report.option_premium_coverage,
        "daily_pnl": report.pnl_by_day,
    }


def trade_diff(original: dict[str, Any], filtered: dict[str, Any]) -> dict[str, Any]:
    orig_times = {t["signal_time"] for t in original.get("trades", [])}
    filt_times = {t["signal_time"] for t in filtered.get("trades", [])}
    blocked_times = {s["time"] for s in filtered.get("skipped", []) if s.get("reason") == REGIME_FILTERED}
    removed = sorted(orig_times - filt_times)
    return {
        "removed": removed,
        "added": sorted(filt_times - orig_times),
        "removed_directly_by_filter": sorted(set(removed) & blocked_times),
    }


def _assessment(original: dict[str, Any], filtered: dict[str, Any]) -> dict[str, Any]:
    o_trades, f_trades = original["resolved_trades"], filtered["resolved_trades"]
    too_low = f_trades < MIN_RESOLVED_TRADES or (o_trades > 0 and f_trades < LOW_FREQUENCY_RATIO * o_trades)
    return {
        "win_rate_improved": filtered["win_rate"] > original["win_rate"],
        "profit_factor_improved": filtered["profit_factor"] > original["profit_factor"],
        "max_drawdown_improved": filtered["max_drawdown"] < original["max_drawdown"],
        "net_pnl_improved": filtered["net_pnl"] > original["net_pnl"],
        "trade_frequency_too_low": too_low,
    }


def verdict(original: dict[str, Any], filtered: dict[str, Any], assessment: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if filtered["trading_days"] < MIN_TRADING_DAYS:
        reasons.append(f"Only {filtered['trading_days']} trading days (< {MIN_TRADING_DAYS})")
    if filtered["resolved_trades"] < MIN_RESOLVED_TRADES:
        reasons.append(f"Filtered strategy has {filtered['resolved_trades']} resolved trades (< {MIN_RESOLVED_TRADES})")
    if assessment["trade_frequency_too_low"]:
        reasons.append("Filtered trade frequency is too low")
    if reasons:
        reasons.append("Filter hypothesis was derived from the same window (in-sample); needs out-of-sample confirmation")
        return "NEEDS MORE DATA", reasons
    core = ("profit_factor_improved", "net_pnl_improved")
    if all(assessment[k] for k in core) and not filtered["max_drawdown"] > original["max_drawdown"]:
        return "KEEP FILTERED", ["Filtered S2 improves profit factor and net P&L without a larger drawdown"]
    return "KEEP ORIGINAL", ["Filtered S2 does not improve profit factor and net P&L without a larger drawdown"]


def build_comparison(
    *,
    original: dict[str, Any],
    filtered: dict[str, Any],
    diff: dict[str, Any],
    filtered_regimes: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    original = {**original, "trades_removed_by_filter": 0, "trades_added_by_filter": 0}
    filtered = {
        **filtered,
        "trades_removed_by_filter": len(diff["removed"]),
        "trades_added_by_filter": len(diff["added"]),
    }
    assessment = _assessment(original, filtered)
    final, reasons = verdict(original, filtered, assessment)
    return {
        **meta,
        "original": original,
        "filtered": filtered,
        "trade_diff": diff,
        "filtered_regimes": filtered_regimes,
        "assessment": assessment,
        "verdict": final,
        "verdict_reasons": reasons,
    }


def _fmt(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    if key == "win_rate":
        return f"{value * 100:.1f}%"
    if key in {"net_pnl", "gross_profit", "gross_loss", "max_drawdown", "max_daily_loss"}:
        return f"{value:,.2f}"
    if key in {"profit_factor", "average_r", "median_r", "average_trades_per_day", "option_coverage_percent"}:
        return f"{value:.2f}"
    return str(value)


def _change(key: str, a: Any, b: Any) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "—"
    if float("inf") in (a, b):
        return "n/a"
    delta = b - a
    if key == "win_rate":
        return f"{delta * 100:+.1f} pp"
    if isinstance(a, int) and isinstance(b, int):
        return f"{delta:+d}"
    return f"{delta:+,.2f}"


def _yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"


def write_reports(comparison: dict[str, Any], out_dir: Path, stem: str = "NIFTY_S2_regime_filtered") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, md_path = (out_dir / f"{stem}.{ext}" for ext in ("json", "csv", "md"))
    json_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")

    o, f = comparison["original"], comparison["filtered"]
    rows: list[dict[str, Any]] = []
    for key, label, _ in COMPARISON_METRICS:
        rows.append({"section": "metric", "metric": label, "original": o.get(key), "filtered": f.get(key),
                     "change": _change(key, o.get(key), f.get(key))})
    o_days = {r["label"]: r["net_pnl"] for r in o["daily_pnl"]}
    f_days = {r["label"]: r["net_pnl"] for r in f["daily_pnl"]}
    for day in sorted(set(o_days) | set(f_days)):
        rows.append({"section": "daily_pnl", "metric": day, "original": o_days.get(day, 0.0),
                     "filtered": f_days.get(day, 0.0), "change": f_days.get(day, 0.0) - o_days.get(day, 0.0)})
    for rid, info in comparison["filtered_regimes"].items():
        rows.append({"section": "filtered_regime", "metric": rid, "original": info.get("net_pnl"),
                     "filtered": None, "change": info.get("reason")})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "original", "filtered", "change"])
        writer.writeheader()
        writer.writerows(rows)

    a = comparison["assessment"]
    md = [
        f"# {comparison['underlying']} {comparison['strategy_id']} — Original vs Regime-Filtered",
        "",
        f"**Window:** {comparison['from_date']} to {comparison['to_date']}  ",
        f"**Analysis run:** `{comparison['run_id']}`  ",
        f"**Regime study:** `{comparison['regime_study_run_id']}`  ",
        "**Mode:** PAPER research only — no broker orders, live trading disabled.",
        "",
        "## Comparison",
        "",
        "| Metric | Original S2 | Filtered S2 | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label, _ in COMPARISON_METRICS:
        md.append(f"| {label} | {_fmt(key, o.get(key))} | {_fmt(key, f.get(key))} | {_change(key, o.get(key), f.get(key))} |")
    md += [
        "",
        "## Filtered regimes",
        "",
        "| Regime | Regime-study trades | Regime-study net P&L | Regime-study PF | Why filtered |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for rid, info in comparison["filtered_regimes"].items():
        pf = info.get("profit_factor", 0.0)
        md.append(f"| {info.get('regime_label', rid)} | {info.get('resolved_trades', 0)} | "
                  f"{info.get('net_pnl', 0.0):,.2f} | {'inf' if pf == float('inf') else f'{pf:.2f}'} | {info.get('reason')} |")
    md += [
        "",
        "Regimes are detected **point-in-time**: at each signal bar only bars up to that bar are used, and "
        "thresholds come from prior sessions at the same time of day. The whole-day labels of the regime "
        "study are used only to choose which regimes to avoid, never to filter individual signals.",
        "",
        "## Assessment",
        "",
        f"- Win rate improved: **{_yes_no(a['win_rate_improved'])}**",
        f"- Profit factor improved: **{_yes_no(a['profit_factor_improved'])}**",
        f"- Max drawdown improved: **{_yes_no(a['max_drawdown_improved'])}**",
        f"- Net P&L improved: **{_yes_no(a['net_pnl_improved'])}**",
        f"- Trade frequency too low: **{_yes_no(a['trade_frequency_too_low'])}**",
        f"- Original trades removed: {len(comparison['trade_diff']['removed'])} "
        f"(directly blocked: {len(comparison['trade_diff']['removed_directly_by_filter'])}); "
        f"new trades from changed sequencing: {len(comparison['trade_diff']['added'])}",
        "",
        "## Daily P&L",
        "",
        "| Day | Original S2 | Filtered S2 | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for day in sorted(set(o_days) | set(f_days)):
        ov, fv = o_days.get(day, 0.0), f_days.get(day, 0.0)
        md.append(f"| {day} | {ov:,.2f} | {fv:,.2f} | {fv - ov:+,.2f} |")
    md += ["", f"## Verdict: **{comparison['verdict']}**", ""]
    md += [f"- {r}" for r in comparison["verdict_reasons"]]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}
