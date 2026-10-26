#!/usr/bin/env python3
"""TRAIN-selected short-only pivot rule; re-simulated on Oct-May, paper only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, load_front_month, load_spot, make_signals,
    run_one, summarize,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "pivot_short_only_validation_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    )
    summaries, trade_frames = [], []
    for symbol, spot_file in settings:
        spot = load_spot(DEFAULT_SPOT / spot_file)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        dev = make_signals(spot, futures)
        dev = dev[(dev.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                  (dev.ts.dt.date < DEV_END)].copy()
        short_only = dev.copy()
        short_only["long_signal"] = False
        # Both indices' short side had higher TRAIN mean P&L/trade than their long side.
        variants = (
            ("baseline", dev, None),
            ("train_selected_short_only", short_only, None),
            ("short_only_close_confirmed_be075", short_only, 0.75),
        )
        for rule, frame, be_trigger in variants:
            trades = run_one(frame, symbol, True, "upstox", 1.0, 1.5, 0.1, 1,
                             breakeven_trigger_r=be_trigger)
            if not trades.empty:
                entries = pd.to_datetime(trades.entry_ts).dt.date
                exits = pd.to_datetime(trades.exit_ts).dt.date
                trades["period"] = entries.map(lambda d: "TRAIN_OCT_JAN" if d < TRAIN_END else "VALIDATION_FEB_MAY")
                trades.loc[(entries < TRAIN_END) & (exits >= TRAIN_END), "period"] = "BOUNDARY_EXCLUDED"
                trades["rule"] = rule
                trade_frames.append(trades)
            for period in ("TRAIN_OCT_JAN", "VALIDATION_FEB_MAY"):
                if trades.empty:
                    part = trades
                else:
                    part = trades[(trades.period == period) &
                                  (pd.to_datetime(trades.entry_ts).dt.date == pd.to_datetime(trades.exit_ts).dt.date)]
                stats = summarize(part, symbol, True, period)
                stats["rule"] = rule
                summaries.append(stats)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("TRAIN Oct-Jan selected short-only by mean net points/trade; VALIDATION Feb-May re-simulated.")
    print("June onward/recent seven sessions excluded before simulation; actual selected futures + modeled fees + 1pt/side slip.")
    print("PAPER/RESEARCH ONLY; no broker orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
