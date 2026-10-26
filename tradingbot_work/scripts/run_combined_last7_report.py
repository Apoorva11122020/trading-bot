#!/usr/bin/env python3
"""Combined backtest: pivot + NIFTY PE recovery with effectiveness metrics."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from scripts.classic_pivot_historical_backtest import GUIDE_BASELINE_VERSION, backtest_guide_day
from scripts.doc_guide_filters import (
    GUIDE_FILTER_LABELS,
    all_filter_options,
    backtest_guide_filtered_day,
    resolve_profile,
)
from scripts.client_strategy_config import (
    CLIENT_V1_CHANGES,
    CLIENT_V1_VERSION,
    PIVOT_PRESET_ALIASES,
)
from scripts.frozen_pivot_config import CONFIG_VERSION, config_for_symbol
from scripts.nifty_pe_contract_resolver import build_nifty_pe_resolver
from scripts.nifty_pe_recovery_rule import DROP_POINTS, RECOVERY_DOWN_POINTS
from scripts.pe_recovery_backtest import backtest_pe_day, scan_pe_signals
from scripts.run_frozen_last7_report import backtest_day, load_spot_frames
from scripts.strategy_metrics import (
    combined_effectiveness_score,
    compute_daily_effectiveness,
    compute_exit_analytics,
    compute_leg_metrics,
    exit_analytics_markdown,
    metrics_markdown_table,
)
from src.safety.live_gate import assert_paper_trading_only

IST = ZoneInfo("Asia/Kolkata")
END = datetime.now(IST).date()
PRESET_CHOICES = ("client_v1", "guide_baseline", "v5_short_only")
DEFAULT_RR = 1.5
RR_CHOICES = (1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_ATR_BUFFER = 0.1
ATR_BUFFER_CHOICES = (0.05, 0.08, 0.1, 0.15, 0.2)


def resolve_pivot_engine(preset: str) -> str:
    return PIVOT_PRESET_ALIASES.get(preset, preset)


def pivot_label_for(preset: str) -> str:
    engine = resolve_pivot_engine(preset)
    if preset == CLIENT_V1_VERSION:
        return CLIENT_V1_VERSION
    if engine == "guide_baseline":
        return GUIDE_BASELINE_VERSION
    return CONFIG_VERSION


def out_dir_for(preset: str) -> Path:
    mapping = {
        "guide_baseline": ROOT / "reports" / "combined_last7_guide_pe",
        CLIENT_V1_VERSION: ROOT / "reports" / "combined_client_v1_pe",
        "v5_short_only": ROOT / "reports" / "combined_last7_v5_pe",
    }
    return mapping.get(preset, ROOT / "reports" / f"combined_{preset}_pe")


def run_pivot_day(
    preset: str,
    frame: pd.DataFrame,
    label: str,
    symbol: str,
    day: date,
    guide_filter: str = "none",
    rr: float = DEFAULT_RR,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
) -> pd.DataFrame:
    engine = resolve_pivot_engine(preset)
    if engine == "guide_baseline":
        if guide_filter and guide_filter != "none":
            return backtest_guide_filtered_day(
                frame, label, day, resolve_profile(guide_filter), rr=rr, atr_buffer=atr_buffer,
            )
        return backtest_guide_day(frame, label, day, rr=rr, atr_buffer=atr_buffer)
    cfg = config_for_symbol(symbol)
    if rr != cfg.rr or atr_buffer != cfg.atr_buffer:
        cfg = replace(cfg, rr=rr, atr_buffer=atr_buffer)
    return backtest_day(frame, label, symbol, day, cfg)


def _pivot_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "net_points": 0.0}
    wins = int((frame.net_pnl_points > 0).sum())
    return {
        "trades": len(frame),
        "wins": wins,
        "losses": len(frame) - wins,
        "net_points": round(float(frame.net_pnl_points.sum()), 2),
    }


def _pe_stats(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"signals": 0, "wins": 0, "losses": 0, "net_inr": 0.0}
    resolved = frame[~frame.exit_reason.astype(str).str.startswith("unresolved")]
    wins = int(resolved["win"].sum()) if not resolved.empty else 0
    return {
        "signals": len(frame),
        "wins": wins,
        "losses": len(resolved) - wins,
        "net_inr": round(float(resolved.net_inr.sum()), 2) if not resolved.empty else 0.0,
    }


def run_backtest(
    preset: str,
    report_start: date,
    report_end: date,
    symbols: list[str],
    guide_filter: str = "none",
    rr: float = DEFAULT_RR,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
) -> tuple[pd.DataFrame, pd.DataFrame, list[date], list[str]]:
    frames = load_spot_frames(report_start, report_end)
    days = [d.date() for d in pd.date_range(report_start, report_end, freq="D") if d.weekday() < 5]
    pivot_rows: list[pd.DataFrame] = []
    label = pivot_label_for(preset)

    for symbol in symbols:
        if symbol not in frames:
            continue
        sym_label, frame = frames[symbol]
        for day in days:
            trades = run_pivot_day(
                preset, frame, sym_label, symbol, day, guide_filter, rr, atr_buffer,
            )
            if not trades.empty:
                trades = trades.copy()
                trades["session_date"] = str(day)
                trades["strategy"] = label
                pivot_rows.append(trades)

    nifty_label, nifty_frame = frames["NIFTY"]
    from src.historical_data.upstox_client import UpstoxHistoricalClient

    client = UpstoxHistoricalClient()
    resolver = build_nifty_pe_resolver()
    pe_rows: list[dict] = []
    pe_armed_days: list[str] = []
    for day in days:
        if scan_pe_signals(nifty_frame, day):
            pe_armed_days.append(str(day))
        row = backtest_pe_day(nifty_frame, day, client, resolver)
        if row is not None:
            pe_rows.append(row)

    pivot = pd.concat(pivot_rows, ignore_index=True) if pivot_rows else pd.DataFrame()
    pe = pd.DataFrame(pe_rows)
    return pivot, pe, days, pe_armed_days


def write_report_files(
    out: Path,
    *,
    pivot_label: str,
    report_start: date,
    report_end: date,
    days: list[date],
    pivot: pd.DataFrame,
    pe: pd.DataFrame,
    pe_armed_days: list[str],
    pivot_preset: str,
    guide_filter: str = "none",
    rr: float = DEFAULT_RR,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
) -> dict[str, Any]:
    daily_rows: list[dict] = []
    for day in days:
        day_key = str(day)
        day_pivot = pivot[pivot.session_date == day_key] if not pivot.empty else pivot
        day_pe = pe[pe.session_date == day_key] if not pe.empty else pe
        pstats = _pivot_stats(day_pivot)
        estats = _pe_stats(day_pe)
        combined_day_net = pstats["net_points"]
        pe_note = ""
        if estats["signals"] == 0:
            pe_note = "armed (-200 seen)" if day_key in pe_armed_days else "no PE activity"
        daily_rows.append({
            "date": day_key,
            "pivot_trades": pstats["trades"],
            "pivot_wins": pstats["wins"],
            "pivot_losses": pstats["losses"],
            "pivot_net_points": pstats["net_points"],
            "pe_signals": estats["signals"],
            "pe_wins": estats["wins"],
            "pe_losses": estats["losses"],
            "pe_net_inr": estats["net_inr"],
            "combined_day_profitable": combined_day_net > 0,
            "pe_note": pe_note,
        })
    daily = pd.DataFrame(daily_rows)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily_summary.csv", index=False)

    pivot_metrics = compute_leg_metrics(
        pivot.net_pnl_points if not pivot.empty else pd.Series(dtype=float),
        label="pivot",
        unit="spot-index pts",
    )
    pe_net = pe.net_inr if not pe.empty and "net_inr" in pe.columns else pd.Series(dtype=float)
    pe_metrics = compute_leg_metrics(pe_net, label="pe", unit="INR")
    daily_metrics = compute_daily_effectiveness(daily, "pivot_net_points")
    score = combined_effectiveness_score(pivot_metrics, pe_metrics)
    exit_analytics = compute_exit_analytics(pivot, rr=rr)

    metrics_payload = {
        "pivot": pivot_metrics,
        "pe": pe_metrics,
        "daily": daily_metrics,
        "score": score,
        "exit_analytics": exit_analytics,
    }
    pd.DataFrame([{k: v for k, v in exit_analytics.items() if k != "exit_reasons"}]).to_csv(
        out / "exit_analytics.csv", index=False,
    )
    pd.DataFrame([{**pivot_metrics, "leg": "pivot"}, {**pe_metrics, "leg": "pe"}]).to_csv(
        out / "effectiveness_metrics.csv", index=False,
    )

    monthly = pd.DataFrame()
    if not pivot.empty:
        pivot_monthly = pivot.copy()
        pivot_monthly["month"] = pd.to_datetime(pivot_monthly.session_date).dt.to_period("M").astype(str)
        monthly = pivot_monthly.groupby("month").agg(
            pivot_trades=("net_pnl_points", "count"),
            pivot_net_points=("net_pnl_points", "sum"),
        ).round(2)
        monthly["pivot_wins"] = pivot_monthly.groupby("month").apply(
            lambda g: int((g.net_pnl_points > 0).sum()), include_groups=False,
        )
        monthly["win_rate_pct"] = pivot_monthly.groupby("month").apply(
            lambda g: round((g.net_pnl_points > 0).mean() * 100, 1), include_groups=False,
        )
        monthly = monthly.reset_index()
        monthly.to_csv(out / "monthly_pivot_summary.csv", index=False)

    symbol_monthly = pd.DataFrame()
    if not pivot.empty:
        symbol_monthly = pivot.groupby("symbol").agg(
            trades=("net_pnl_points", "count"),
            wins=("net_pnl_points", lambda s: int((s > 0).sum())),
            net_points=("net_pnl_points", "sum"),
        ).round(2).reset_index()
        symbol_monthly["win_rate_pct"] = pivot.groupby("symbol").net_pnl_points.apply(
            lambda s: round((s > 0).mean() * 100, 1),
        ).values
        symbol_monthly.to_csv(out / "symbol_summary.csv", index=False)

    best_day = daily.loc[daily.pivot_net_points.idxmax()] if not daily.empty and pivot_metrics["trades"] else None
    worst_day = daily.loc[daily.pivot_net_points.idxmin()] if not daily.empty and pivot_metrics["trades"] else None

    lines = [
        "# Combined backtest report",
        "",
        f"- Generated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}",
        f"- Period: {report_start} to {report_end}",
        f"- Trading days: {len(days)}",
        f"- Strategy preset: **{pivot_preset}** (pivot engine: {pivot_label})",
        f"- Guide filter: **{GUIDE_FILTER_LABELS.get(guide_filter, guide_filter)}**",
        f"- Risk/Reward target: **1:{rr:g}** (stop fixed, target = risk × {rr:g})",
        f"- Stop ATR buffer: **{atr_buffer:g}** (stop = signal candle ± ATR × {atr_buffer:g})",
        f"- PE rule: morning -{DROP_POINTS:.0f} pts, cross up open-{RECOVERY_DOWN_POINTS:.0f}, "
        f"BUY PE 2 lots, SL 20 premium pts (NIFTY only)",
        "",
        "## Effectiveness score",
        "",
        f"| Score | Value |",
        f"|-------|-------|",
        f"| Overall effectiveness | **{score['overall_score']:.1f} / 100** ({score['verdict']}) |",
        f"| Pivot component | {score['pivot_component_score']:.1f} / 100 |",
        f"| PE component | {score['pe_component_score']:.1f} / 100 |",
        f"| Profitable trading days (pivot) | **{daily_metrics['profitable_day_pct']:.1f}%** "
        f"({daily_metrics['profitable_days']}/{daily_metrics['trading_days']} days) |",
        "",
    ]

    if pivot_preset == CLIENT_V1_VERSION:
        lines.extend(["## Client v1 changes (vs doc guide)", ""])
        for item in CLIENT_V1_CHANGES:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(["## Pivot metrics", "", metrics_markdown_table(pivot_metrics), ""])
    lines.extend([
        "## Exit breakdown (why win % ≠ profit)",
        "",
        exit_analytics_markdown(exit_analytics, rr=rr),
        "",
        "_Net R includes slippage (2 pts/side) and commission. Real break-even win % is usually much "
        "higher than the theoretical RR formula suggests._",
        "",
    ])
    lines.extend(["## PE metrics", "", metrics_markdown_table(pe_metrics), ""])

    lines.extend([
        "## Grand total (quick view)",
        "",
        f"| Leg | Trades | Win rate | Net |",
        f"|-----|--------|----------|-----|",
        f"| Pivot | {pivot_metrics['trades']} | {pivot_metrics['win_rate_pct']:.1f}% | "
        f"{pivot_metrics['net']:+.2f} pts |",
        f"| PE | {pe_metrics['trades']} | {pe_metrics['win_rate_pct']:.1f}% | "
        f"INR {pe_metrics['net']:+.0f} |",
        "",
    ])

    if best_day is not None:
        lines.append(
            f"- Best pivot day: {best_day['date']} ({best_day['pivot_net_points']:+.2f} pts)"
        )
        lines.append(
            f"- Worst pivot day: {worst_day['date']} ({worst_day['pivot_net_points']:+.2f} pts)"
        )

    lines.extend(["", "## Files", "", "| File | Purpose |", "|------|---------|",
                  "| `REPORT.md` | This summary |",
                  "| `effectiveness_metrics.csv` | Win %, profit factor, scores |",
                  "| `daily_summary.csv` | Day-by-day Excel view |",
                  "| `pivot_trades.csv` | All pivot trades |",
                  "| `pe_trades.csv` | All PE signals |"])

    if not monthly.empty:
        lines.extend(["", "## Pivot by month", "", monthly.to_markdown(index=False)])
    if not symbol_monthly.empty:
        lines.extend(["", "## Pivot by symbol", "", symbol_monthly.to_markdown(index=False)])
    if not pe.empty:
        lines.extend(["", "## PE trades", ""])
        pe_view = pe[["session_date", "trading_symbol", "entry_premium", "exit_premium", "net_inr", "exit_reason"]]
        lines.append(pe_view.to_markdown(index=False))

    lines.extend(["", "## Last 10 trading days", ""])
    lines.append(daily.tail(10).to_markdown(index=False))
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_payload


def write_comparison_report(
    out: Path,
    report_start: date,
    report_end: date,
    guide_metrics: dict[str, Any],
    client_metrics: dict[str, Any],
) -> None:
    g = guide_metrics["pivot"]
    c = client_metrics["pivot"]
    lines = [
        "# Strategy comparison: doc guide vs client v1",
        "",
        f"- Period: {report_start} to {report_end}",
        "",
        "| Metric | Doc guide | Client v1 | Better |",
        "|--------|-----------|-----------|--------|",
        f"| Trades | {g['trades']} | {c['trades']} | fewer if similar net |",
        f"| Win rate | {g['win_rate_pct']:.1f}% | **{c['win_rate_pct']:.1f}%** | "
        f"{'client_v1' if c['win_rate_pct'] >= g['win_rate_pct'] else 'guide'} |",
        f"| Net pivot P&L | {g['net']:+.2f} | **{c['net']:+.2f}** | "
        f"{'client_v1' if c['net'] >= g['net'] else 'guide'} |",
        f"| Profit factor | {g['profit_factor'] or 'n/a'} | {c['profit_factor'] or 'n/a'} | "
        f"{'client_v1' if (c['profit_factor'] or 0) >= (g['profit_factor'] or 0) else 'guide'} |",
        f"| Profitable days % | {guide_metrics['daily']['profitable_day_pct']:.1f}% | "
        f"{client_metrics['daily']['profitable_day_pct']:.1f}% | — |",
        f"| Effectiveness score | {guide_metrics['score']['overall_score']:.1f} | "
        f"**{client_metrics['score']['overall_score']:.1f}** | "
        f"{'client_v1' if client_metrics['score']['overall_score'] >= guide_metrics['score']['overall_score'] else 'guide'} |",
        "",
        "**Recommendation:** Use `client_v1` for paper/live (default). Keep `guide_baseline` only to compare against the original doc.",
    ]
    (out / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(preset: str, pivot_label: str, pivot: pd.DataFrame, pe: pd.DataFrame, metrics: dict[str, Any]) -> None:
    pm = metrics["pivot"]
    pe_m = metrics["pe"]
    sc = metrics["score"]
    print(f"=== {preset.upper()} | {pivot_label} + PE ===")
    print(
        f"Pivot: {pm['trades']} trades | win {pm['win_rate_pct']:.1f}% | "
        f"net {pm['net']:+.2f} pts | PF {pm['profit_factor']}"
    )
    print(
        f"PE: {pe_m['trades']} signals | win {pe_m['win_rate_pct']:.1f}% | net INR {pe_m['net']:+.0f}"
    )
    print(
        f"Effectiveness: {sc['overall_score']:.1f}/100 ({sc['verdict']}) | "
        f"profitable days {metrics['daily']['profitable_day_pct']:.1f}%"
    )


def main() -> int:
    assert_paper_trading_only()
    parser = argparse.ArgumentParser(description="Combined pivot + PE backtest with effectiveness %")
    parser.add_argument(
        "--pivot-preset",
        choices=PRESET_CHOICES,
        default="guide_baseline",
        help="guide_baseline=doc strategy (default) | client_v1=short-only improved",
    )
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument(
        "--symbols",
        default="NIFTY,BANKNIFTY",
        help="Comma-separated: NIFTY, BANKNIFTY",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run doc guide vs client_v1 and write COMPARISON.md",
    )
    parser.add_argument(
        "--guide-filter",
        choices=[opt[0] for opt in all_filter_options()],
        default="none",
        help="Extra filters on doc guide pivot (ignored for client_v1/v5)",
    )
    parser.add_argument(
        "--rr",
        type=float,
        choices=RR_CHOICES,
        default=DEFAULT_RR,
        help="Reward/risk multiple for pivot target (default 1.5 = 1:1.5)",
    )
    parser.add_argument(
        "--atr-buffer",
        type=float,
        choices=ATR_BUFFER_CHOICES,
        default=DEFAULT_ATR_BUFFER,
        help="ATR multiplier for stop distance (default 0.1; 0.05 = tighter stop)",
    )
    args = parser.parse_args()
    report_end = args.end or END
    report_start = args.start or (report_end - timedelta(days=182))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    pivot_label = pivot_label_for(args.pivot_preset)
    out = out_dir_for(args.pivot_preset)

    pivot, pe, days, pe_armed_days = run_backtest(
        args.pivot_preset, report_start, report_end, symbols,
        args.guide_filter, args.rr, args.atr_buffer,
    )
    out.mkdir(parents=True, exist_ok=True)
    if not pivot.empty:
        pivot.to_csv(out / "pivot_trades.csv", index=False)
    if not pe.empty:
        pe.to_csv(out / "pe_trades.csv", index=False)

    metrics = write_report_files(
        out,
        pivot_label=pivot_label,
        report_start=report_start,
        report_end=report_end,
        days=days,
        pivot=pivot,
        pe=pe,
        pe_armed_days=pe_armed_days,
        pivot_preset=args.pivot_preset,
        guide_filter=args.guide_filter,
        rr=args.rr,
        atr_buffer=args.atr_buffer,
    )
    print_summary(args.pivot_preset, pivot_label, pivot, pe, metrics)
    if args.rr != DEFAULT_RR:
        print(f"Risk/Reward: 1:{args.rr:g}")
    if args.atr_buffer != DEFAULT_ATR_BUFFER:
        print(f"ATR buffer: {args.atr_buffer:g}")
    if args.pivot_preset == "guide_baseline" and args.guide_filter != "none":
        print(f"Guide filter: {GUIDE_FILTER_LABELS[args.guide_filter]}")

    if args.compare:
        cmp_out = ROOT / "reports" / "strategy_comparison"
        cmp_out.mkdir(parents=True, exist_ok=True)
        guide_pivot, guide_pe, _, guide_armed = run_backtest("guide_baseline", report_start, report_end, symbols)
        client_pivot, client_pe, _, client_armed = run_backtest(CLIENT_V1_VERSION, report_start, report_end, symbols)
        guide_metrics = write_report_files(
            cmp_out / "guide_baseline",
            pivot_label=GUIDE_BASELINE_VERSION,
            report_start=report_start,
            report_end=report_end,
            days=days,
            pivot=guide_pivot,
            pe=guide_pe,
            pe_armed_days=guide_armed,
            pivot_preset="guide_baseline",
        )
        client_metrics = write_report_files(
            cmp_out / "client_v1",
            pivot_label=CLIENT_V1_VERSION,
            report_start=report_start,
            report_end=report_end,
            days=days,
            pivot=client_pivot,
            pe=client_pe,
            pe_armed_days=client_armed,
            pivot_preset=CLIENT_V1_VERSION,
        )
        write_comparison_report(cmp_out, report_start, report_end, guide_metrics, client_metrics)
        print(f"\nComparison saved: {cmp_out / 'COMPARISON.md'}")

    print(f"\nSaved: {out.resolve()}")
    print("Open REPORT.md and effectiveness_metrics.csv for win % and scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
