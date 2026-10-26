#!/usr/bin/env python3
"""Development-only study of stricter RSI confirmation for the pivot strategy.

Uses only dates before 2026-06-01. It intentionally does not run or report OOS.
No broker calls or orders are made.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT,
    DEFAULT_SPOT,
    OOS_START,
    ROOT,
    load_front_month,
    load_spot,
    make_signals,
    run_one,
    summarize,
)
from src.safety.live_gate import assert_paper_trading_only


OUT_DIR = ROOT / "reports" / "pivot_rsi_filter_development_20261014"
RSI_THRESHOLDS = (55, 60, 65)
RR = 1.5
ATR_BUFFER = 0.1
SLIPPAGE_PER_SIDE = 1.0


def main() -> int:
    assert_paper_trading_only()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_trades: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    coverage_rows: list[dict] = []

    for symbol, spot_name in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        merged = make_signals(spot, futures)
        # Limit the input bars before simulation so OOS trades are never generated.
        development = merged[
            (merged["ts"].dt.date < OOS_START)
            & (merged["ts"].dt.date >= pd.Timestamp("2025-10-01").date())
        ].copy()
        missing = selection.loc[
            (selection["session_date"] < OOS_START)
            & (selection["status"] != "SELECTED")
        ]
        coverage_rows.append({
            "symbol": symbol,
            "development_coverage": "COMPLETE" if missing.empty else "INCOMPLETE",
            "unavailable_sessions": len(missing),
            "unavailable_dates": ";".join(str(d) for d in missing["session_date"]),
        })

        for threshold in RSI_THRESHOLDS:
            x = development.copy()
            short_threshold = 100 - threshold
            x["long_signal"] = (
                x["in_session"] & x["cross_up"]
                & (x["ema20_spot"] > x["ema50_spot"])
                & (x["rsi14_spot"] > threshold)
            )
            x["short_signal"] = (
                x["in_session"] & x["cross_down"]
                & (x["ema20_spot"] < x["ema50_spot"])
                & (x["rsi14_spot"] < short_threshold)
            )
            trades = run_one(
                x, symbol, True, "upstox", SLIPPAGE_PER_SIDE, RR, ATR_BUFFER, 1
            )
            label = f"RSI>{threshold}/<{short_threshold}"
            if not trades.empty:
                trades["rsi_filter"] = label
                trades["volume_filter"] = True
                trades["split"] = "DEVELOPMENT"
                all_trades.append(trades)
            stats = summarize(trades, symbol, True, "DEVELOPMENT")
            stats.update({
                "rsi_filter": label,
                "long_rsi_min_exclusive": threshold,
                "short_rsi_max_exclusive": short_threshold,
                "rr": RR,
                "atr_buffer": ATR_BUFFER,
                "slippage_points_per_side": SLIPPAGE_PER_SIDE,
            })
            summary_rows.append(stats)

    summary = pd.DataFrame(summary_rows)
    coverage = pd.DataFrame(coverage_rows)
    summary = summary.merge(coverage, on="symbol", how="left", validate="many_to_one")
    summary.to_csv(OUT_DIR / "development_summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(OUT_DIR / "development_trades.csv", index=False)
    print("DEVELOPMENT ONLY | RSI confirmation thresholds | estimated Upstox costs | 1 point/side slippage")
    print(summary.sort_values(["symbol", "net_points"], ascending=[True, False]).to_string(index=False))
    print(f"Saved: {OUT_DIR / 'development_summary.csv'}")
    print("PAPER | RESEARCH ONLY | ORDERS: 0 | OOS was not simulated or used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
