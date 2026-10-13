"""Robustness analysis for strategy research backtests."""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.options_engine import DAILY_LOSS, DATA_INVALID

IST = ZoneInfo("Asia/Kolkata")

ENTRY_TIME_BUCKETS: list[tuple[str, str, str]] = [
    ("09:15-10:00", "09:15", "10:00"),
    ("10:00-11:00", "10:00", "11:00"),
    ("11:00-12:00", "11:00", "12:00"),
    ("12:00-13:00", "12:00", "13:00"),
    ("13:00-14:30", "13:00", "14:30"),
]

MIN_RESOLVED_TRADES = 30
MIN_COVERAGE_PERCENT = 80.0
MAX_TOP3_PNL_SHARE = 0.50
MIN_PF_EX_TOP3 = 1.0


@dataclass
class RobustnessReport:
    run_id: str
    underlying: str
    strategy_id: str
    from_date: str
    to_date: str
    source_run_id: str
    total_trading_days: int = 0
    total_signals: int = 0
    resolved_trades: int = 0
    unresolved_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    average_r: float = 0.0
    median_r: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    max_daily_profit: float = 0.0
    max_daily_loss: float = 0.0
    average_trades_per_day: float = 0.0
    pnl_by_day: list[dict[str, Any]] = field(default_factory=list)
    pnl_by_week: list[dict[str, Any]] = field(default_factory=list)
    performance_by_entry_bucket: list[dict[str, Any]] = field(default_factory=list)
    largest_winning_trades: list[dict[str, Any]] = field(default_factory=list)
    largest_losing_trades: list[dict[str, Any]] = field(default_factory=list)
    profit_factor_excluding_top_1_winner: float = 0.0
    profit_factor_excluding_top_3_winners: float = 0.0
    top_1_trade_pnl_share_percent: float = 0.0
    top_3_trades_pnl_share_percent: float = 0.0
    daily_loss_limit: dict[str, Any] = field(default_factory=dict)
    option_premium_coverage: dict[str, Any] = field(default_factory=dict)
    trade_concentration: dict[str, Any] = field(default_factory=dict)
    verdict: str = "NEEDS MORE DATA"
    verdict_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ts(value: str | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def entry_time_bucket(entry_time: str) -> str | None:
    """Map an entry timestamp to a configured bucket label."""
    local = _ts(entry_time)
    minutes = local.hour * 60 + local.minute
    for label, start, end in ENTRY_TIME_BUCKETS:
        sh, sm = _parse_hhmm(start)
        eh, em = _parse_hhmm(end)
        start_m = sh * 60 + sm
        end_m = eh * 60 + em
        if start_m <= minutes < end_m:
            return label
    return None


def _profit_factor(wins_pnl: Iterable[float], losses_pnl: Iterable[float]) -> float:
    gross_profit = sum(p for p in wins_pnl if p > 0)
    gross_loss = abs(sum(p for p in losses_pnl if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _consecutive_streak(trades: list[dict], positive: bool) -> int:
    best = streak = 0
    for trade in trades:
        pnl = float(trade.get("pnl", 0))
        hit = pnl > 0 if positive else pnl < 0
        if hit:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _max_drawdown(trades: list[dict], initial_equity: float = 1_000_000.0) -> float:
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    for trade in trades:
        equity += float(trade.get("pnl", 0))
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _aggregate_pnl(trades: list[dict], key_fn) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for trade in trades:
        key, label = key_fn(trade)
        if key not in grouped:
            grouped[key] = {"label": label, "trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0}
        grouped[key]["trades"] += 1
        pnl = float(trade.get("pnl", 0))
        grouped[key]["net_pnl"] += pnl
        if pnl > 0:
            grouped[key]["wins"] += 1
        elif pnl < 0:
            grouped[key]["losses"] += 1
    rows = list(grouped.values())
    rows.sort(key=lambda r: r["label"])
    return rows


def _day_key(trade: dict) -> tuple[str, str]:
    day = _ts(trade["entry_time"]).date().isoformat()
    return day, day


def _week_key(trade: dict) -> tuple[str, str]:
    dt = _ts(trade["entry_time"]).date()
    iso = dt.isocalendar()
    label = f"{iso.year}-W{iso.week:02d}"
    return label, label


def _bucket_metrics(trades: list[dict]) -> list[dict[str, Any]]:
    buckets = {label: {"bucket": label, "trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0}
               for label, _, _ in ENTRY_TIME_BUCKETS}
    for trade in trades:
        bucket = entry_time_bucket(trade["entry_time"])
        if bucket is None:
            continue
        pnl = float(trade.get("pnl", 0))
        buckets[bucket]["trades"] += 1
        buckets[bucket]["net_pnl"] += pnl
        if pnl > 0:
            buckets[bucket]["wins"] += 1
        elif pnl < 0:
            buckets[bucket]["losses"] += 1
    for row in buckets.values():
        decided = row["wins"] + row["losses"]
        row["win_rate"] = row["wins"] / decided if decided else 0.0
    return [buckets[label] for label, _, _ in ENTRY_TIME_BUCKETS]


def _top_trades(trades: list[dict], n: int, reverse: bool) -> list[dict[str, Any]]:
    if reverse:
        pool = [t for t in trades if float(t.get("pnl", 0)) > 0]
    else:
        pool = [t for t in trades if float(t.get("pnl", 0)) < 0]
    ordered = sorted(pool, key=lambda t: float(t.get("pnl", 0)), reverse=reverse)
    fields = (
        "signal_time", "entry_time", "exit_time", "direction", "option_type",
        "trading_symbol", "pnl", "r_multiple", "exit_reason",
    )
    out = []
    for trade in ordered[:n]:
        out.append({k: trade.get(k) for k in fields})
    return out


def _pnl_share_percent(total_pnl: float, top_pnls: list[float]) -> float:
    if total_pnl == 0:
        return 0.0 if not top_pnls else 100.0
    return round(100.0 * sum(top_pnls) / total_pnl, 2)


def profit_factor_excluding_top_n_winners(trades: list[dict], n: int) -> float:
    """Profit factor after removing the top N winning trades by P&L."""
    winners = sorted(
        [t for t in trades if float(t.get("pnl", 0)) > 0],
        key=lambda t: float(t.get("pnl", 0)),
        reverse=True,
    )
    exclude_ids = {id(t) for t in winners[:n]}
    remaining = [t for t in trades if id(t) not in exclude_ids]
    wins = [float(t.get("pnl", 0)) for t in remaining if float(t.get("pnl", 0)) > 0]
    losses = [float(t.get("pnl", 0)) for t in remaining if float(t.get("pnl", 0)) < 0]
    return _profit_factor(wins, losses)


def daily_loss_limit_behavior(
    skipped: list[dict],
    trades: list[dict],
    *,
    limit_inr: float,
) -> dict[str, Any]:
    hits = [s for s in skipped if s.get("reason") == DAILY_LOSS]
    hit_days = sorted({_ts(s["time"]).date().isoformat() for s in hits})
    trade_days = {_ts(t["entry_time"]).date().isoformat() for t in trades}
    day_pnl: dict[str, float] = {}
    for trade in trades:
        day = _ts(trade["entry_time"]).date().isoformat()
        day_pnl[day] = day_pnl.get(day, 0.0) + float(trade.get("pnl", 0))
    days_at_or_below_limit = sorted(
        day for day, pnl in day_pnl.items() if pnl <= -limit_inr
    )
    return {
        "limit_inr": limit_inr,
        "signals_skipped_after_limit": len(hits),
        "days_with_limit_skip": len(hit_days),
        "days_with_limit_skip_dates": hit_days,
        "days_with_realized_loss_at_or_below_limit": len(days_at_or_below_limit),
        "days_with_realized_loss_at_or_below_limit_dates": days_at_or_below_limit,
        "trading_days_with_resolved_trades": len(trade_days),
    }


def option_coverage_summary(
    coverage: dict[str, Any],
    trades: list[dict],
    skipped: list[dict],
) -> dict[str, Any]:
    missing_times = set()
    for item in coverage.get("unavailable", []):
        if item.get("signal_time"):
            missing_times.add(item["signal_time"])
    data_invalid_times = {s["time"] for s in skipped if s.get("reason") == DATA_INVALID}
    trade_signal_times = {t.get("signal_time") for t in trades}
    affected = [t for t in trades if t.get("signal_time") in missing_times]
    return {
        "total_signals": int(coverage.get("total_signals", 0)),
        "signals_with_complete_coverage": int(coverage.get("signals_with_complete_coverage", 0)),
        "signals_missing_coverage": int(coverage.get("signals_missing_coverage", 0)),
        "coverage_percent": float(coverage.get("coverage_percent", 0.0)),
        "unique_required_contracts": int(coverage.get("unique_required_contracts", 0)),
        "contracts_downloaded": int(coverage.get("contracts_downloaded", 0)),
        "contracts_unavailable": int(coverage.get("contracts_unavailable", 0)),
        "data_invalid_skips": len(data_invalid_times),
        "resolved_trades_affected_by_missing_coverage": len(affected),
        "affected_trade_signal_times": [t.get("signal_time") for t in affected],
    }


def assess_trade_concentration(
    trades: list[dict],
    *,
    top_1_share: float,
    top_3_share: float,
    pf_ex_top_3: float,
) -> dict[str, Any]:
    highly_dependent = (
        top_3_share >= MAX_TOP3_PNL_SHARE * 100
        or pf_ex_top_3 < MIN_PF_EX_TOP3
    )
    reasons = []
    if top_1_share >= 40:
        reasons.append(f"Top 1 trade contributes {top_1_share:.1f}% of net P&L")
    if top_3_share >= MAX_TOP3_PNL_SHARE * 100:
        reasons.append(f"Top 3 trades contribute {top_3_share:.1f}% of net P&L")
    if pf_ex_top_3 < MIN_PF_EX_TOP3:
        reasons.append(f"Profit factor excluding top 3 winners is {pf_ex_top_3:.2f}")
    return {
        "highly_dependent_on_few_trades": highly_dependent,
        "reasons": reasons,
        "top_1_pnl_share_percent": top_1_share,
        "top_3_pnl_share_percent": top_3_share,
        "profit_factor_excluding_top_3_winners": pf_ex_top_3,
    }


def robustness_verdict(
    *,
    resolved_trades: int,
    coverage_percent: float,
    trade_concentration: dict[str, Any],
    trading_days: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if resolved_trades < MIN_RESOLVED_TRADES:
        reasons.append(f"Only {resolved_trades} resolved trades (< {MIN_RESOLVED_TRADES})")
    if coverage_percent < MIN_COVERAGE_PERCENT:
        reasons.append(f"Premium coverage {coverage_percent:.1f}% (< {MIN_COVERAGE_PERCENT:.0f}%)")
    if trade_concentration.get("highly_dependent_on_few_trades"):
        reasons.extend(trade_concentration.get("reasons", []))
    if trading_days < 30:
        reasons.append(f"Short calendar sample: {trading_days} trading days")
    if reasons:
        return "NEEDS MORE DATA", reasons
    return "ROBUST", ["Metrics pass robustness thresholds"]


def analyze_robustness(
    *,
    run_id: str,
    source_run_id: str,
    underlying: str,
    strategy_id: str,
    from_date: date,
    to_date: date,
    trading_days: int,
    result: dict[str, Any],
    coverage: dict[str, Any],
    initial_equity: float = 1_000_000.0,
    daily_loss_limit_inr: float = 3000.0,
) -> RobustnessReport:
    trades = list(result.get("trades", []))
    signals = int(result.get("metrics", {}).get("total_signals", len(result.get("signals", []))))
    resolved = len(trades)
    unresolved = max(0, signals - resolved)

    wins = [t for t in trades if float(t.get("pnl", 0)) > 0]
    losses = [t for t in trades if float(t.get("pnl", 0)) < 0]
    gross_profit = sum(float(t.get("pnl", 0)) for t in wins)
    gross_loss = abs(sum(float(t.get("pnl", 0)) for t in losses))
    net_pnl = sum(float(t.get("pnl", 0)) for t in trades)
    decided = len(wins) + len(losses)
    win_rate = len(wins) / decided if decided else 0.0
    pf = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0)
    r_values = [float(t.get("r_multiple", 0)) for t in trades]
    avg_r = sum(r_values) / len(r_values) if r_values else 0.0
    med_r = float(statistics.median(r_values)) if r_values else 0.0

    pnl_by_day = _aggregate_pnl(trades, _day_key)
    day_pnls = [row["net_pnl"] for row in pnl_by_day]
    max_daily_profit = max(day_pnls) if day_pnls else 0.0
    max_daily_loss = min(day_pnls) if day_pnls else 0.0

    top_winners = sorted(trades, key=lambda t: float(t.get("pnl", 0)), reverse=True)
    top_win_pnls = [float(t.get("pnl", 0)) for t in top_winners if float(t.get("pnl", 0)) > 0]
    top_1_share = _pnl_share_percent(net_pnl, top_win_pnls[:1])
    top_3_share = _pnl_share_percent(net_pnl, top_win_pnls[:3])
    pf_ex_1 = profit_factor_excluding_top_n_winners(trades, 1)
    pf_ex_3 = profit_factor_excluding_top_n_winners(trades, 3)

    skipped = list(result.get("skipped", []))
    loss_limit = daily_loss_limit_behavior(skipped, trades, limit_inr=daily_loss_limit_inr)
    cov_summary = option_coverage_summary(coverage, trades, skipped)
    concentration = assess_trade_concentration(
        trades, top_1_share=top_1_share, top_3_share=top_3_share, pf_ex_top_3=pf_ex_3,
    )
    verdict, verdict_reasons = robustness_verdict(
        resolved_trades=resolved,
        coverage_percent=cov_summary["coverage_percent"],
        trade_concentration=concentration,
        trading_days=trading_days,
    )

    return RobustnessReport(
        run_id=run_id,
        underlying=underlying,
        strategy_id=strategy_id,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        source_run_id=source_run_id,
        total_trading_days=trading_days,
        total_signals=signals,
        resolved_trades=resolved,
        unresolved_signals=unresolved,
        win_rate=win_rate,
        profit_factor=pf,
        net_pnl=net_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        max_drawdown=_max_drawdown(trades, initial_equity),
        average_r=avg_r,
        median_r=med_r,
        max_consecutive_wins=_consecutive_streak(trades, True),
        max_consecutive_losses=_consecutive_streak(trades, False),
        max_daily_profit=max_daily_profit,
        max_daily_loss=max_daily_loss,
        average_trades_per_day=resolved / trading_days if trading_days else 0.0,
        pnl_by_day=pnl_by_day,
        pnl_by_week=_aggregate_pnl(trades, _week_key),
        performance_by_entry_bucket=_bucket_metrics(trades),
        largest_winning_trades=_top_trades(trades, 5, True),
        largest_losing_trades=_top_trades(trades, 5, False),
        profit_factor_excluding_top_1_winner=pf_ex_1,
        profit_factor_excluding_top_3_winners=pf_ex_3,
        top_1_trade_pnl_share_percent=top_1_share,
        top_3_trades_pnl_share_percent=top_3_share,
        daily_loss_limit=loss_limit,
        option_premium_coverage=cov_summary,
        trade_concentration=concentration,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
    )


def load_final_comparison(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_for(final: dict[str, Any], underlying: str, strategy_id: str) -> dict[str, Any]:
    for item in final.get("coverage", []):
        if item.get("underlying") == underlying and item.get("strategy_id") == strategy_id:
            return item
    raise KeyError(f"No coverage for {underlying}/{strategy_id}")


def row_for(final: dict[str, Any], underlying: str, strategy_id: str) -> dict[str, Any]:
    for row in final.get("rows", []):
        if row.get("underlying") == underlying and row.get("strategy_id") == strategy_id:
            return row
    raise KeyError(f"No row for {underlying}/{strategy_id}")


def write_robustness_reports(report: RobustnessReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.underlying}_{report.strategy_id}_robustness"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    payload = report.as_dict()
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat_rows = [
        {"section": "summary", "metric": "total_trading_days", "value": report.total_trading_days},
        {"section": "summary", "metric": "total_signals", "value": report.total_signals},
        {"section": "summary", "metric": "resolved_trades", "value": report.resolved_trades},
        {"section": "summary", "metric": "unresolved_signals", "value": report.unresolved_signals},
        {"section": "summary", "metric": "win_rate", "value": report.win_rate},
        {"section": "summary", "metric": "profit_factor", "value": report.profit_factor},
        {"section": "summary", "metric": "net_pnl", "value": report.net_pnl},
        {"section": "summary", "metric": "gross_profit", "value": report.gross_profit},
        {"section": "summary", "metric": "gross_loss", "value": report.gross_loss},
        {"section": "summary", "metric": "max_drawdown", "value": report.max_drawdown},
        {"section": "summary", "metric": "average_r", "value": report.average_r},
        {"section": "summary", "metric": "median_r", "value": report.median_r},
        {"section": "summary", "metric": "max_consecutive_wins", "value": report.max_consecutive_wins},
        {"section": "summary", "metric": "max_consecutive_losses", "value": report.max_consecutive_losses},
        {"section": "summary", "metric": "max_daily_profit", "value": report.max_daily_profit},
        {"section": "summary", "metric": "max_daily_loss", "value": report.max_daily_loss},
        {"section": "summary", "metric": "average_trades_per_day", "value": report.average_trades_per_day},
        {"section": "summary", "metric": "profit_factor_excluding_top_1_winner", "value": report.profit_factor_excluding_top_1_winner},
        {"section": "summary", "metric": "profit_factor_excluding_top_3_winners", "value": report.profit_factor_excluding_top_3_winners},
        {"section": "summary", "metric": "top_1_trade_pnl_share_percent", "value": report.top_1_trade_pnl_share_percent},
        {"section": "summary", "metric": "top_3_trades_pnl_share_percent", "value": report.top_3_trades_pnl_share_percent},
        {"section": "summary", "metric": "verdict", "value": report.verdict},
    ]
    for row in report.pnl_by_day:
        flat_rows.append({"section": "pnl_by_day", **row})
    for row in report.pnl_by_week:
        flat_rows.append({"section": "pnl_by_week", **row})
    for row in report.performance_by_entry_bucket:
        flat_rows.append({"section": "entry_bucket", **row})
    for idx, row in enumerate(report.largest_winning_trades, start=1):
        flat_rows.append({"section": "top_winners", "rank": idx, **row})
    for idx, row in enumerate(report.largest_losing_trades, start=1):
        flat_rows.append({"section": "top_losers", "rank": idx, **row})

    fieldnames = sorted({k for row in flat_rows for k in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    pf_text = "inf" if report.profit_factor == float("inf") else f"{report.profit_factor:.2f}"
    md_lines = [
        f"# {report.underlying} {report.strategy_id} Robustness Report",
        "",
        f"**Source run:** `{report.source_run_id}`  ",
        f"**Analysis run:** `{report.run_id}`  ",
        f"**Period:** {report.from_date} to {report.to_date} ({report.total_trading_days} trading days)",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total signals | {report.total_signals} |",
        f"| Resolved trades | {report.resolved_trades} |",
        f"| Unresolved signals | {report.unresolved_signals} |",
        f"| Win rate | {report.win_rate * 100:.1f}% |",
        f"| Profit factor | {pf_text} |",
        f"| Net P&L | {report.net_pnl:,.2f} |",
        f"| Gross profit | {report.gross_profit:,.2f} |",
        f"| Gross loss | {report.gross_loss:,.2f} |",
        f"| Max drawdown | {report.max_drawdown:,.2f} |",
        f"| Average R | {report.average_r:.3f} |",
        f"| Median R | {report.median_r:.3f} |",
        f"| Max consecutive wins | {report.max_consecutive_wins} |",
        f"| Max consecutive losses | {report.max_consecutive_losses} |",
        f"| Max daily profit | {report.max_daily_profit:,.2f} |",
        f"| Max daily loss | {report.max_daily_loss:,.2f} |",
        f"| Avg trades/day | {report.average_trades_per_day:.2f} |",
        "",
        "## P&L by trading day",
        "",
        "| Day | Trades | Wins | Losses | Net P&L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report.pnl_by_day:
        md_lines.append(
            f"| {row['label']} | {row['trades']} | {row['wins']} | {row['losses']} | {row['net_pnl']:,.2f} |"
        )
    md_lines.extend([
        "",
        "## P&L by week",
        "",
        "| Week | Trades | Wins | Losses | Net P&L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in report.pnl_by_week:
        md_lines.append(
            f"| {row['label']} | {row['trades']} | {row['wins']} | {row['losses']} | {row['net_pnl']:,.2f} |"
        )
    md_lines.extend([
        "",
        "## Performance by entry time bucket",
        "",
        "| Bucket | Trades | Wins | Losses | Win rate | Net P&L |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report.performance_by_entry_bucket:
        md_lines.append(
            f"| {row['bucket']} | {row['trades']} | {row['wins']} | {row['losses']} | "
            f"{row['win_rate'] * 100:.1f}% | {row['net_pnl']:,.2f} |"
        )
    md_lines.extend([
        "",
        "## Largest winning trades",
        "",
    ])
    for idx, row in enumerate(report.largest_winning_trades, start=1):
        md_lines.append(
            f"{idx}. {row.get('trading_symbol')} — {row.get('pnl'):,.2f} "
            f"(R={float(row.get('r_multiple', 0)):.2f}, {row.get('entry_time')})"
        )
    md_lines.extend(["", "## Largest losing trades", ""])
    for idx, row in enumerate(report.largest_losing_trades, start=1):
        md_lines.append(
            f"{idx}. {row.get('trading_symbol')} — {row.get('pnl'):,.2f} "
            f"(R={float(row.get('r_multiple', 0)):.2f}, {row.get('entry_time')})"
        )
    md_lines.extend([
        "",
        "## Trade concentration",
        "",
        f"- Profit factor excluding top 1 winner: {report.profit_factor_excluding_top_1_winner:.2f}",
        f"- Profit factor excluding top 3 winners: {report.profit_factor_excluding_top_3_winners:.2f}",
        f"- Top 1 trade share of net P&L: {report.top_1_trade_pnl_share_percent:.1f}%",
        f"- Top 3 trades share of net P&L: {report.top_3_trades_pnl_share_percent:.1f}%",
        f"- Highly dependent on few trades: **{report.trade_concentration.get('highly_dependent_on_few_trades')}**",
    ])
    if report.trade_concentration.get("reasons"):
        for reason in report.trade_concentration["reasons"]:
            md_lines.append(f"  - {reason}")
    md_lines.extend([
        "",
        "## Daily ₹3,000 loss limit",
        "",
        f"- Signals skipped after limit: {report.daily_loss_limit.get('signals_skipped_after_limit')}",
        f"- Days with limit skip: {report.daily_loss_limit.get('days_with_limit_skip')}",
        f"- Days with realized loss at/below limit: {report.daily_loss_limit.get('days_with_realized_loss_at_or_below_limit')}",
        "",
        "## Option premium coverage",
        "",
        f"- Fully covered signals: {report.option_premium_coverage.get('signals_with_complete_coverage')}",
        f"- Missing-data signals: {report.option_premium_coverage.get('signals_missing_coverage')}",
        f"- Coverage: {report.option_premium_coverage.get('coverage_percent')}%",
        f"- Resolved trades affected by missing coverage: "
        f"{report.option_premium_coverage.get('resolved_trades_affected_by_missing_coverage')}",
        "",
        f"## Verdict: **{report.verdict}**",
        "",
    ])
    for reason in report.verdict_reasons:
        md_lines.append(f"- {reason}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}


def print_conclusion(report: RobustnessReport) -> None:
    pf = "inf" if report.profit_factor == float("inf") else f"{report.profit_factor:.2f}"
    print("\n=== NIFTY S2 Robustness Conclusion ===")
    print(f"Resolved trades: {report.resolved_trades}")
    print(f"Win rate: {report.win_rate * 100:.1f}%")
    print(f"Profit factor: {pf}")
    print(f"Net P&L: {report.net_pnl:,.2f}")
    print(f"Max drawdown: {report.max_drawdown:,.2f}")
    print(f"Top-3-trade dependency: {report.top_3_trades_pnl_share_percent:.1f}% of net P&L")
    print(
        f"Data coverage: {report.option_premium_coverage.get('coverage_percent')}% "
        f"({report.option_premium_coverage.get('signals_with_complete_coverage')}/"
        f"{report.option_premium_coverage.get('total_signals')} signals)"
    )
    print(f"Verdict: {report.verdict}")
