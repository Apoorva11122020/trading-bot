#!/usr/bin/env python3
"""Development-only test of a fixed 30-minute opening-range breakout idea.

The 09:15-09:45 IST spot range defines the high/low. After 09:45, enter only
when a completed spot bar crosses that range; existing futures fills, volume
filter, ATR stops, estimated Upstox costs, and slippage remain unchanged.
Dates from 2026-06-01 onward are excluded before simulation.
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
    load_front_month,
    load_spot,
    make_signals,
    run_one,
    summarize,
)
from src.safety.live_gate import assert_paper_trading_only


OUT_DIR = ROOT / "reports" / "opening_range_breakout_development_20261014"
RR = 1.5
ATR_BUFFER = 0.1
SLIPPAGE_PER_SIDE = 1.0


def main() -> int:
    assert_paper_trading_only()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    trades_all: list[pd.DataFrame] = []
    coverage: list[dict] = []

    for symbol, spot_name in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        merged = make_signals(spot, futures)
        dev = merged[
            (merged["ts"].dt.date < OOS_START)
            & (merged["ts"].dt.date >= pd.Timestamp("2025-10-01").date())
        ].copy()

        minute_of_day = dev["ts"].dt.hour * 60 + dev["ts"].dt.minute
        opening = dev.loc[(minute_of_day >= 9 * 60 + 15) & (minute_of_day < 9 * 60 + 45)]
        levels = opening.groupby("session", sort=True).agg(
            opening_range_high=("high_spot", "max"),
            opening_range_low=("low_spot", "min"),
            opening_bars=("high_spot", "size"),
        )
        dev = dev.join(levels, on="session")
        prev_close = dev.groupby("session", sort=False)["close_spot"].shift(1)
        after_opening = (minute_of_day >= 9 * 60 + 45) & (minute_of_day < 15 * 60)
        dev["long_signal"] = (
            after_opening
            & (dev["opening_bars"] == 6)
            & (prev_close <= dev["opening_range_high"])
            & (dev["close_spot"] > dev["opening_range_high"])
        )
        dev["short_signal"] = (
            after_opening
            & (dev["opening_bars"] == 6)
            & (prev_close >= dev["opening_range_low"])
            & (dev["close_spot"] < dev["opening_range_low"])
        )

        unavailable = selection.loc[
            (selection["session_date"] < OOS_START)
            & (selection["status"] != "SELECTED")
        ]
        coverage.append({
            "symbol": symbol,
            "development_coverage": "COMPLETE" if unavailable.empty else "INCOMPLETE",
            "unavailable_sessions": len(unavailable),
            "unavailable_dates": ";".join(str(d) for d in unavailable["session_date"]),
        })

        trades = run_one(
            dev, symbol, True, "upstox", SLIPPAGE_PER_SIDE, RR, ATR_BUFFER, 1
        )
        if not trades.empty:
            trades["entry_logic"] = "30-minute opening-range breakout"
            trades["volume_filter"] = True
            trades["split"] = "DEVELOPMENT"
            trades_all.append(trades)
        stats = summarize(trades, symbol, True, "DEVELOPMENT")
        stats.update({
            "entry_logic": "30-minute opening-range breakout",
            "opening_range_ist": "09:15-09:45",
            "rr": RR,
            "atr_buffer": ATR_BUFFER,
            "volume_filter_threshold": "current volume > prior 20-bar average * 1.2",
            "slippage_points_per_side": SLIPPAGE_PER_SIDE,
        })
        summaries.append(stats)

    out = pd.DataFrame(summaries).merge(
        pd.DataFrame(coverage), on="symbol", how="left", validate="one_to_one"
    )
    out.to_csv(OUT_DIR / "development_summary.csv", index=False)
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(
            OUT_DIR / "development_trades.csv", index=False
        )
    print("DEVELOPMENT ONLY | 30-minute opening-range breakout | estimated Upstox costs | 1 point/side slippage")
    print(out.to_string(index=False))
    print(f"Saved: {OUT_DIR / 'development_summary.csv'}")
    print("PAPER | RESEARCH ONLY | ORDERS: 0 | OOS was not simulated or used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
