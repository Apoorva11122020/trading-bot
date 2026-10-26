"""Run any catalog strategy over a date range (shared by UI ranking and backtest)."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from scripts.alt_strategies import (
    add_ema_pullback_signals,
    add_opening_range_signals,
    add_vwap_reversion_signals,
    backtest_signal_day,
)
from scripts.classic_pivot_historical_backtest import GUIDE_BASELINE_VERSION, prepare
from scripts.doc_guide_filters import backtest_guide_filtered_day, resolve_profile
from scripts.nifty_pe_contract_resolver import build_nifty_pe_resolver
from scripts.pe_recovery_backtest import backtest_pe_day, scan_pe_signals
from scripts.run_combined_last7_report import backtest_guide_day, pivot_label_for, run_pivot_day
from scripts.run_frozen_last7_report import load_spot_frames
from scripts.strategy_catalog import STRATEGIES, StrategySpec
from scripts.strategy_metrics import compute_exit_analytics, compute_leg_metrics

_ALT_BUILDERS = {
    "opening_range": add_opening_range_signals,
    "vwap_reversion": add_vwap_reversion_signals,
    "ema_pullback": add_ema_pullback_signals,
}


def run_strategy_pivot(
    spec: StrategySpec,
    report_start: date,
    report_end: date,
    symbols: list[str],
    *,
    rr: float | None = None,
    atr_buffer: float | None = None,
    guide_filter: str | None = None,
    frames: dict | None = None,
    prepared_alt: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rr = rr if rr is not None else spec.rr
    atr_buffer = atr_buffer if atr_buffer is not None else spec.atr_buffer
    guide_filter = guide_filter if guide_filter is not None else spec.guide_filter

    if spec.id == "pe_only":
        return pd.DataFrame()

    if frames is None:
        frames = load_spot_frames(report_start, report_end)
    days = [d.date() for d in pd.date_range(report_start, report_end, freq="D") if d.weekday() < 5]
    pivot_rows: list[pd.DataFrame] = []

    if spec.alt_engine:
        if prepared_alt is not None:
            prepared = prepared_alt
        else:
            prepared = {
                symbol: _ALT_BUILDERS[spec.alt_engine](prepare(frames[symbol][1], volume_filter=False))
                for symbol in symbols
                if symbol in frames
            }
        for symbol in symbols:
            if symbol not in frames:
                continue
            sym_label, _ = frames[symbol]
            frame = prepared[symbol]
            for day in days:
                trades = backtest_signal_day(
                    frame, sym_label, day,
                    signal_builder=lambda x: x,
                    already_prepared=True,
                    rr=rr,
                    atr_buffer=atr_buffer,
                )
                if not trades.empty:
                    part = trades.copy()
                    part["session_date"] = str(day)
                    part["strategy"] = spec.id
                    pivot_rows.append(part)
        return pd.concat(pivot_rows, ignore_index=True) if pivot_rows else pd.DataFrame()

    preset = spec.preset or "guide_baseline"
    label = pivot_label_for(preset)

    for symbol in symbols:
        if symbol not in frames:
            continue
        sym_label, frame = frames[symbol]
        for day in days:
            if preset == "guide_baseline":
                if guide_filter and guide_filter != "none":
                    trades = backtest_guide_filtered_day(
                        frame, sym_label, day, resolve_profile(guide_filter),
                        rr=rr, atr_buffer=atr_buffer,
                    )
                else:
                    trades = backtest_guide_day(frame, sym_label, day, rr=rr, atr_buffer=atr_buffer)
            else:
                trades = run_pivot_day(
                    preset, frame, sym_label, symbol, day, guide_filter, rr, atr_buffer,
                )
            if not trades.empty:
                part = trades.copy()
                part["session_date"] = str(day)
                part["strategy"] = label
                pivot_rows.append(part)

    return pd.concat(pivot_rows, ignore_index=True) if pivot_rows else pd.DataFrame()


def run_pe_backtest(
    report_start: date,
    report_end: date,
) -> tuple[pd.DataFrame, list[date], list[str]]:
    frames = load_spot_frames(report_start, report_end)
    days = [d.date() for d in pd.date_range(report_start, report_end, freq="D") if d.weekday() < 5]
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
    pe = pd.DataFrame(pe_rows)
    return pe, days, pe_armed_days


def summarize_strategy_pivot(
    pivot: pd.DataFrame,
    spec: StrategySpec,
    rr: float,
) -> dict[str, Any]:
    metrics = compute_leg_metrics(
        pivot.net_pnl_points if not pivot.empty else pd.Series(dtype=float),
        label=spec.id,
        unit="spot-index pts",
    )
    exit_stats = compute_exit_analytics(pivot, rr=rr)
    return {
        "strategy_id": spec.id,
        "strategy_label": spec.label,
        "kind": spec.kind,
        "include_pe": spec.include_pe,
        **metrics,
        "net_points": metrics["net"],
        "target_hits": exit_stats["target_hits"],
        "stop_hits": exit_stats["stop_hits"],
        "flat_exits": exit_stats["flat_exits"],
        "avg_net_win_r": exit_stats["avg_net_win_r"],
        "avg_net_loss_r": exit_stats["avg_net_loss_r"],
        "breakeven_win_rate_pct": exit_stats["breakeven_win_rate_pct"],
        "avg_cost_points": exit_stats["avg_cost_points"],
        "profitable": metrics["net"] > 0,
    }


def _prepared_alt_cache(frames: dict, symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    cache: dict[str, dict[str, pd.DataFrame]] = {}
    for engine, builder in _ALT_BUILDERS.items():
        cache[engine] = {
            symbol: builder(prepare(frames[symbol][1], volume_filter=False))
            for symbol in symbols
            if symbol in frames
        }
    return cache


def rank_strategies(
    report_start: date,
    report_end: date,
    symbols: list[str] | None = None,
    *,
    with_pe: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pe_cache: pd.DataFrame | None = None
    if with_pe:
        pe_cache, _, _ = run_pe_backtest(report_start, report_end)

    all_symbols = sorted({s for spec in STRATEGIES for s in (symbols or spec.symbols_default)})
    frames = load_spot_frames(report_start, report_end)
    alt_prepared = _prepared_alt_cache(frames, all_symbols)

    for spec in STRATEGIES:
        syms = list(symbols or spec.symbols_default)
        prepared = alt_prepared.get(spec.alt_engine or "", {}) if spec.alt_engine else None
        pivot = run_strategy_pivot(
            spec, report_start, report_end, syms,
            frames=frames, prepared_alt=prepared,
        )
        row = summarize_strategy_pivot(pivot, spec, spec.rr)
        pe_net_inr = None
        pe_trades = 0
        if with_pe and spec.include_pe and pe_cache is not None:
            if not pe_cache.empty and "net_inr" in pe_cache.columns:
                resolved = pe_cache[~pe_cache.exit_reason.astype(str).str.startswith("unresolved")]
                pe_net_inr = round(float(resolved.net_inr.sum()), 2) if not resolved.empty else 0.0
                pe_trades = len(pe_cache)
        row["pe_trades"] = pe_trades
        row["pe_net_inr"] = pe_net_inr
        rows.append(row)

    ranked = sorted(rows, key=lambda r: (r["net_points"], r["win_rate_pct"]), reverse=True)
    return {
        "report_start": report_start.isoformat(),
        "report_end": report_end.isoformat(),
        "symbols": symbols,
        "rows": ranked,
        "best_net": ranked[0] if ranked else None,
        "best_win": max(ranked, key=lambda r: (r["win_rate_pct"], r["net_points"])) if ranked else None,
        "any_profitable": any(r["profitable"] for r in rows),
    }


def rank_strategies_with_params(
    report_start: date,
    report_end: date,
    symbols: list[str] | None = None,
    *,
    rr: float = 2.0,
    atr_buffer: float = 0.05,
    with_pe: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_symbols = sorted({s for spec in STRATEGIES for s in (symbols or spec.symbols_default)})
    frames = load_spot_frames(report_start, report_end)
    alt_prepared = _prepared_alt_cache(frames, all_symbols)

    for spec in STRATEGIES:
        syms = list(symbols or spec.symbols_default)
        use_rr = rr if spec.preset == "guide_baseline" or spec.alt_engine else spec.rr
        use_atr = atr_buffer if spec.preset == "guide_baseline" or spec.alt_engine else spec.atr_buffer
        prepared = alt_prepared.get(spec.alt_engine or "", {}) if spec.alt_engine else None
        pivot = run_strategy_pivot(
            spec, report_start, report_end, syms,
            rr=use_rr, atr_buffer=use_atr,
            frames=frames, prepared_alt=prepared,
        )
        row = summarize_strategy_pivot(pivot, spec, use_rr)
        row["rr_used"] = use_rr
        row["atr_used"] = use_atr
        rows.append(row)

    ranked = sorted(rows, key=lambda r: (r["net_points"], r["win_rate_pct"]), reverse=True)
    return {
        "report_start": report_start.isoformat(),
        "report_end": report_end.isoformat(),
        "symbols": symbols,
        "rr": rr,
        "atr_buffer": atr_buffer,
        "rows": ranked,
        "best_net": ranked[0] if ranked else None,
        "best_win": max(ranked, key=lambda r: (r["win_rate_pct"], r["net_points"])) if ranked else None,
        "any_profitable": any(r["profitable"] for r in rows),
    }
