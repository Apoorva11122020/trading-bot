#!/usr/bin/env python3
"""Frozen NIFTY short-only below-session-open forward check; PAPER only.

The rule was fixed on Oct-Jan TRAIN and passed Feb-May validation. This script
does not tune it: evaluate June 1-Oct 5 with selected NIFTY futures, volume
filter, modeled Upstox charges and 1 point/side slippage. Oct 6 onward is
excluded before signal construction. No orders.
"""
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

OUT = ROOT / "reports" / "frozen_nifty_bearish_day_open_forward_20261014"
FORWARD_START = pd.Timestamp("2026-06-01").date()
FORWARD_END = pd.Timestamp("2026-10-06").date()
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0


def apply_rule(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    session_open = x.groupby("session", sort=False).open_spot.transform("first")
    x["long_signal"] = False
    x["short_signal"] = x.short_signal.astype(bool) & x.close_spot.lt(session_open)
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    spot = load_spot(DEFAULT_SPOT / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv")
    spot = spot[spot.ts.dt.date < FORWARD_END].copy()
    futures, selection = load_front_month("NIFTY", DEFAULT_ROOT)
    futures = futures[futures.ts.dt.date < FORWARD_END].copy()
    frame = make_signals(spot, futures)
    forward = frame[(frame.ts.dt.date >= FORWARD_START) & (frame.ts.dt.date < FORWARD_END)].copy()
    results = []
    for label, data in (("baseline", forward), ("candidate", apply_rule(forward))):
        trades = run_one(data, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        stats = summarize(trades, "NIFTY", True, "FORWARD_JUN01_OCT05")
        stats.update({"rule": "pivot_baseline" if label == "baseline" else "short_only_close_below_session_open",
                      "recent_seven_sessions_used": False,
                      "unavailable_sessions": int(((selection.session_date >= FORWARD_START)
                                                   & (selection.session_date < FORWARD_END)
                                                   & (selection.status != "SELECTED")).sum())})
        results.append(stats)
        if label == "candidate" and not trades.empty:
            trades = trades.copy()
            trades["rule"] = "short_only_close_below_session_open"
            trades.to_csv(OUT / "candidate_trades.csv", index=False)
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "summary.csv", index=False)
    print("FROZEN NIFTY FORWARD | no tuning | PAPER ONLY | estimated Upstox fees + 1 point/side")
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}; Oct 6 onward excluded before signal construction; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
