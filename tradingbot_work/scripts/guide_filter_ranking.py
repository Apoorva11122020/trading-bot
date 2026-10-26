"""Rank doc-guide filter profiles (singles + combinations) on one data load."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from scripts.classic_pivot_historical_backtest import prepare
from scripts.doc_guide_filters import (
    COMBO_FILTER_LABELS,
    GUIDE_FILTER_LABELS,
    all_filter_options,
    backtest_guide_filtered_day,
    resolve_profile,
)
from scripts.run_combined_last7_report import DEFAULT_ATR_BUFFER, DEFAULT_RR
from scripts.run_frozen_last7_report import load_spot_frames
from scripts.strategy_metrics import compute_exit_analytics, compute_leg_metrics


def filter_label(filter_id: str) -> str:
    if filter_id in GUIDE_FILTER_LABELS:
        return GUIDE_FILTER_LABELS[filter_id]
    return COMBO_FILTER_LABELS.get(filter_id, filter_id)


def list_filters_for_ui() -> list[dict[str, str]]:
    return [
        {"id": key, "label": label, "kind": "combo" if "+" in key else "single"}
        for key, label in all_filter_options()
    ]


def _balanced_score(net: float, win_rate: float, trades: int) -> float:
    if trades == 0:
        return -9999.0
    return net + win_rate * 12.0 - max(0, trades - 30) * 2.0


def rank_guide_filters(
    report_start: date,
    report_end: date,
    symbols: list[str],
    rr: float = DEFAULT_RR,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
) -> dict[str, Any]:
    frames = load_spot_frames(report_start, report_end)
    prepared = {
        symbol: prepare(frames[symbol][1], volume_filter=False)
        for symbol in symbols
        if symbol in frames
    }
    days = [d.date() for d in pd.date_range(report_start, report_end, freq="D") if d.weekday() < 5]

    rows: list[dict[str, Any]] = []
    for filter_id, label in all_filter_options():
        profile = resolve_profile(filter_id)
        pivot_parts: list[pd.DataFrame] = []
        for symbol in symbols:
            if symbol not in prepared:
                continue
            sym_label = frames[symbol][0]
            frame = prepared[symbol]
            for day in days:
                trades = backtest_guide_filtered_day(
                    frame, sym_label, day, profile,
                    already_prepared=True, rr=rr, atr_buffer=atr_buffer,
                )
                if not trades.empty:
                    part = trades.copy()
                    part["session_date"] = str(day)
                    pivot_parts.append(part)
        pivot = pd.concat(pivot_parts, ignore_index=True) if pivot_parts else pd.DataFrame()
        metrics = compute_leg_metrics(
            pivot.net_pnl_points if not pivot.empty else pd.Series(dtype=float),
            label=filter_id,
            unit="spot-index pts",
        )
        exit_stats = compute_exit_analytics(pivot, rr=rr)
        rows.append({
            "filter_id": filter_id,
            "filter_label": label,
            "kind": "combo" if "+" in filter_id else "single",
            "trades": metrics["trades"],
            "win_rate_pct": metrics["win_rate_pct"],
            "net_points": metrics["net"],
            "profit_factor": metrics["profit_factor"],
            "avg_trade": metrics["avg_trade"],
            "max_drawdown": metrics["max_drawdown"],
            "profitable": metrics["net"] > 0,
            "balanced_score": round(_balanced_score(metrics["net"], metrics["win_rate_pct"], metrics["trades"]), 2),
            "target_hits": exit_stats["target_hits"],
            "stop_hits": exit_stats["stop_hits"],
            "flat_exits": exit_stats["flat_exits"],
            "avg_net_win_r": exit_stats["avg_net_win_r"],
            "avg_net_loss_r": exit_stats["avg_net_loss_r"],
            "avg_cost_points": exit_stats["avg_cost_points"],
            "breakeven_win_rate_pct": exit_stats["breakeven_win_rate_pct"],
            "theoretical_breakeven_win_rate_pct": exit_stats["theoretical_breakeven_win_rate_pct"],
        })

    def _net_key(r: dict[str, Any]) -> tuple:
        if r["trades"] == 0:
            return (-999999.0, 0.0)
        return (r["net_points"], r["win_rate_pct"])

    def _win_key(r: dict[str, Any]) -> tuple:
        if r["trades"] == 0:
            return (0.0, -999999.0)
        return (r["win_rate_pct"], r["net_points"])

    def _balanced_key(r: dict[str, Any]) -> tuple:
        if r["trades"] == 0:
            return (-999999.0, -999999.0)
        return (r["balanced_score"], r["net_points"])

    ranked_net = sorted(rows, key=_net_key, reverse=True)
    ranked_win = sorted(rows, key=_win_key, reverse=True)
    ranked_balanced = sorted(rows, key=_balanced_key, reverse=True)

    return {
        "report_start": report_start.isoformat(),
        "report_end": report_end.isoformat(),
        "symbols": symbols,
        "rr": rr,
        "atr_buffer": atr_buffer,
        "rows": ranked_net,
        "best_net": ranked_net[0] if ranked_net else None,
        "best_win": ranked_win[0] if ranked_win else None,
        "best_balanced": ranked_balanced[0] if ranked_balanced else None,
        "any_profitable": any(r["profitable"] for r in rows),
    }
