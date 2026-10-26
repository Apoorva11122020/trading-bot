#!/usr/bin/env python3
"""Compare frozen v3 hybrid vs v4 split on recent spot sessions and full-year futures."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT,
    DEFAULT_SPOT,
    END,
    START,
    load_front_month,
    load_spot,
    make_signals,
    run_one,
    summarize,
)
from scripts.frozen_pivot_config import (
    CONFIG_VERSION,
    V3_HYBRID_VERSION,
    V4_SPLIT_VERSION,
    V5_SHORT_ONLY_VERSION,
    apply_frozen_filters,
    config_for_symbol,
    max_risk_for,
)
from scripts.run_frozen_last7_report import END as LAST7_END
from scripts.run_frozen_last7_report import START as LAST7_START
from scripts.run_frozen_last7_report import backtest_day, load_spot_frames
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "frozen_pivot_split_compare"
PRESETS = (
    ("v3_hybrid", V3_HYBRID_VERSION),
    ("v4_split", V4_SPLIT_VERSION),
    ("v5_short_only", V5_SHORT_ONLY_VERSION),
)


def run_last7_spot(preset_key: str, preset_name: str) -> pd.DataFrame:
    frames = load_spot_frames(LAST7_START, LAST7_END)
    all_trades: list[pd.DataFrame] = []
    for symbol, (label, frame) in frames.items():
        cfg = config_for_symbol(symbol, preset_name)
        for day in pd.date_range(LAST7_START, LAST7_END, freq="D"):
            if day.weekday() >= 5:
                continue
            trades = backtest_day(frame, label, symbol, day.date(), cfg)
            if not trades.empty:
                trades = trades.copy()
                trades["session_date"] = str(day.date())
                trades["preset"] = preset_key
                all_trades.append(trades)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def run_futures_full(preset_key: str, preset_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, spot_file in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        cfg = config_for_symbol(symbol, preset_name)
        spot = load_spot(DEFAULT_SPOT / spot_file)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        baseline = make_signals(spot, futures)
        frozen = apply_frozen_filters(baseline, symbol, cfg, execution_basis="futures")
        trades = run_one(
            frozen,
            symbol,
            volume_filter=cfg.require_volume,
            cost_model="upstox",
            slip=1.0,
            rr=cfg.rr,
            atr_buffer=cfg.atr_buffer,
            lot=1,
            breakeven_trigger_r=cfg.breakeven_trigger_r,
            max_risk_points=max_risk_for(symbol, cfg),
            one_trade_per_session=cfg.one_trade_per_session and not cfg.separate_long_short_quota,
            one_trade_per_side_per_session=cfg.separate_long_short_quota,
        )
        stats = summarize(trades, symbol, cfg.require_volume, "FULL")
        stats["preset"] = preset_key
        stats["nifty_mode"] = "v1_short_only" if symbol == "NIFTY" and preset_name == V4_SPLIT_VERSION else "v3_hybrid"
        rows.append(stats)
    return pd.DataFrame(rows)


def print_last7_table(trades: pd.DataFrame) -> None:
    print("\n=== LAST 7 DAYS | SPOT ===")
    print(f"Range: {LAST7_START} to {LAST7_END}")
    for preset_key, _ in PRESETS:
        part = trades[trades.preset == preset_key] if not trades.empty else trades
        net = float(part.net_pnl_points.sum()) if not part.empty else 0.0
        wins = int((part.net_pnl_points > 0).sum()) if not part.empty else 0
        losses = len(part) - wins if not part.empty else 0
        print(f"  {preset_key:12} {len(part):2} trades | {wins}W-{losses}L | net={net:+.2f} pts")
        if part.empty:
            continue
        for sym in sorted(part.symbol.unique()):
            sub = part[part.symbol == sym]
            print(f"    {sym:10} {len(sub)} trades | net={sub.net_pnl_points.sum():+.2f}")


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)

    last7_frames = [run_last7_spot(key, name) for key, name in PRESETS]
    last7 = pd.concat(last7_frames, ignore_index=True) if any(not f.empty for f in last7_frames) else pd.DataFrame()
    if not last7.empty:
        last7.to_csv(OUT / "last7_trades.csv", index=False)
        daily = (
            last7.groupby(["preset", "session_date"], as_index=False)
            .agg(trades=("net_pnl_points", "count"), net_points=("net_pnl_points", "sum"))
        )
        daily.to_csv(OUT / "last7_daily.csv", index=False)

    futures_frames = [run_futures_full(key, name) for key, name in PRESETS]
    futures_summary = pd.concat(futures_frames, ignore_index=True)
    futures_summary.to_csv(OUT / "futures_full_year.csv", index=False)

    print("=== FROZEN SPLIT COMPARE ===")
    print("v3_hybrid     : both symbols morning long + afternoon short")
    print("v4_split      : NIFTY v1 short-only | BANKNIFTY v3 hybrid")
    print("v5_short_only : both symbols short-only (no morning longs) [DEFAULT]")
    print_last7_table(last7)

    print("\n=== FULL YEAR | FUTURES + COSTS ===")
    print(f"Window: {START} to {END}")
    for preset_key, _ in PRESETS:
        part = futures_summary[futures_summary.preset == preset_key]
        total = float(part.net_points.sum())
        print(f"  {preset_key:12} total={total:+.0f} pts")
        for _, row in part.iterrows():
            print(f"    {row['symbol']:10} {int(row['trades'])} trades | WR {row['win_rate_pct']:.1f}% | net={row['net_points']:+.0f}")

    print(f"\nSaved: {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
