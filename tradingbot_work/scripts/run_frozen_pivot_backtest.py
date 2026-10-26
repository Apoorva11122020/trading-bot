#!/usr/bin/env python3
"""Frozen pivot v3 hybrid production backtest (paper/research only)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT,
    DEFAULT_SPOT,
    END,
    OOS_START,
    START,
    load_front_month,
    load_spot,
    make_signals,
    run_one,
    summarize,
)
from scripts.frozen_pivot_config import (
    CONFIG_VERSION,
    V2_CONFIG,
    V3_HYBRID_CONFIG,
    V3_HYBRID_VERSION,
    V4_SPLIT_VERSION,
    V5_SHORT_ONLY_VERSION,
    FrozenPivotConfig,
    apply_frozen_filters,
    config_for_symbol,
    entry_window_minutes,
    max_risk_for,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "frozen_pivot_production_v3"


def run_variant(baseline: pd.DataFrame, symbol: str, config: FrozenPivotConfig, rule: str) -> pd.DataFrame:
    if rule == "guide_baseline_both_sides":
        return run_one(baseline, symbol, False, "upstox", 1.0, 1.5, 0.1, 1)
    frozen = apply_frozen_filters(baseline, symbol, config, execution_basis="futures")
    return run_one(
        frozen,
        symbol,
        volume_filter=config.require_volume,
        cost_model="upstox",
        slip=1.0,
        rr=config.rr,
        atr_buffer=config.atr_buffer,
        lot=1,
        breakeven_trigger_r=config.breakeven_trigger_r,
        max_risk_points=max_risk_for(symbol, config),
        one_trade_per_session=config.one_trade_per_session and not config.separate_long_short_quota,
        one_trade_per_side_per_session=config.separate_long_short_quota,
    )


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    )
    summaries, trade_frames = [], []
    variants = (
        ("guide_baseline_both_sides", "baseline"),
        ("frozen_v2", "v2"),
        (f"frozen_{V3_HYBRID_VERSION}", V3_HYBRID_VERSION),
        (f"frozen_{V4_SPLIT_VERSION}", V4_SPLIT_VERSION),
        (f"frozen_{V5_SHORT_ONLY_VERSION}", V5_SHORT_ONLY_VERSION),
    )
    for symbol, spot_file in settings:
        spot = load_spot(DEFAULT_SPOT / spot_file)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        baseline = make_signals(spot, futures)
        window_label = (
            f"L{entry_window_minutes(symbol, 'LONG')[0] // 60:02d}-"
            f"{entry_window_minutes(symbol, 'LONG')[1] // 60:02d}/"
            f"S{entry_window_minutes(symbol, 'SHORT')[0] // 60:02d}-"
            f"{entry_window_minutes(symbol, 'SHORT')[1] // 60:02d}"
        )
        for rule, preset in variants:
            config = (
                V3_HYBRID_CONFIG
                if preset == "baseline"
                else V2_CONFIG if preset == "v2" else config_for_symbol(symbol, preset)
            )
            trades = run_variant(baseline, symbol, config, rule)
            if not trades.empty:
                trades = trades.copy()
                trades["rule"] = rule
                trade_frames.append(trades)
            vol = config.require_volume if rule != "guide_baseline_both_sides" else False
            for split, start, end in (
                ("DEVELOPMENT", START, OOS_START.fromordinal(OOS_START.toordinal() - 1)),
                ("OOS", OOS_START, END),
                ("FULL", START, END),
            ):
                st = pd.to_datetime(trades["entry_ts"]).dt.date if not trades.empty else pd.Series([], dtype=object)
                et = pd.to_datetime(trades["exit_ts"]).dt.date if not trades.empty else pd.Series([], dtype=object)
                subset = trades[(st >= start) & (et <= end)] if not trades.empty else trades.copy()
                stats = summarize(subset, symbol, vol, split)
                stats["rule"] = rule
                stats["windows_ist"] = window_label if rule.startswith("frozen") else "ALL"
                summaries.append(stats)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT.resolve()}")
    print(f"Frozen default {CONFIG_VERSION}: both symbols short-only | v4=NIFTY short+BN hybrid | BE@0.75R")
    print("Futures fills, Upstox fees, 1 pt/side slip. PAPER/RESEARCH ONLY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
