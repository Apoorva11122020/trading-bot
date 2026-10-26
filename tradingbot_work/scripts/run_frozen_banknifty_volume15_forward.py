#!/usr/bin/env python3
"""Frozen BANKNIFTY 1.5x prior-volume forward check; PAPER only.

The 1.5x volume rule was fixed on Oct-Jan TRAIN and passed Feb-May validation.
This script evaluates it once on June 1-Oct 5, using selected futures and the
same modeled Upstox charges + 1 point/side slippage. No tuning; Oct 6+ is
excluded before signal construction. No broker orders.
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

OUT = ROOT / "reports" / "frozen_banknifty_volume15_forward_20261014"
FORWARD_START = pd.Timestamp("2026-06-01").date()
FORWARD_END = pd.Timestamp("2026-10-06").date()
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    spot = load_spot(DEFAULT_SPOT / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv")
    spot = spot[spot.ts.dt.date < FORWARD_END].copy()
    futures, selection = load_front_month("BANKNIFTY", DEFAULT_ROOT)
    futures = futures[futures.ts.dt.date < FORWARD_END].copy()
    frame = make_signals(spot, futures)
    forward = frame[(frame.ts.dt.date >= FORWARD_START) & (frame.ts.dt.date < FORWARD_END)].copy()
    strict = forward.copy()
    # run_one applies volume > vol_avg20 * 1.2; rescaling sets the effective cutoff to 1.5x.
    strict["vol_avg20_fut"] = strict["vol_avg20_fut"] * (1.5 / 1.2)
    rows = []
    for name, data in (("baseline_volume_1.2x", forward), ("strict_volume_1.5x", strict)):
        trades = run_one(data, "BANKNIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        stats = summarize(trades, "BANKNIFTY", True, "FORWARD_JUN01_OCT05")
        stats.update({"rule": name, "recent_seven_sessions_used": False,
                      "unavailable_sessions": int(((selection.session_date >= FORWARD_START)
                                                   & (selection.session_date < FORWARD_END)
                                                   & (selection.status != "SELECTED")).sum())})
        rows.append(stats)
        if name == "strict_volume_1.5x" and not trades.empty:
            trades = trades.copy()
            trades["rule"] = name
            trades.to_csv(OUT / "candidate_trades.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print("FROZEN BANKNIFTY VOLUME 1.5x FORWARD | PAPER ONLY | estimated Upstox fees + 1 point/side")
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}; Oct 6 onward excluded before signal construction; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
