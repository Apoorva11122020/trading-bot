#!/usr/bin/env python3
"""Frozen forward check for TRAIN-selected BANKNIFTY LONG body>=0.8 rule.

Selection came from a fixed 3x3 side-specific body grid on Oct-Jan TRAIN; the
rule passed Feb-May validation. This script does not tune it: it checks June 1
through Oct 5 once using actual selected futures, volume filter and modeled
Upstox costs + 1 point/side slippage. Oct 6 onward is excluded before signals.
PAPER only; no broker order APIs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, load_front_month, load_spot, make_signals,
    run_one, summarize,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "frozen_banknifty_side_body_forward_20261014"
FORWARD_START = pd.Timestamp("2026-06-01").date()
FORWARD_END = pd.Timestamp("2026-10-06").date()
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0
BODY_MIN = 0.8


def apply_rule(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    span = (x.high_spot - x.low_spot).replace(0, np.nan)
    body = (x.close_spot - x.open_spot).abs() / span
    # TRAIN-selected policy: require a strong bullish signal candle for LONG;
    # all SHORT signals are disabled.
    x["long_signal"] = (x.long_signal.astype(bool) & x.close_spot.gt(x.open_spot)
                        & body.ge(BODY_MIN))
    x["short_signal"] = False
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    spot = load_spot(DEFAULT_SPOT / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv")
    # No Oct 6+ spot candle is passed into signal construction.
    spot = spot[spot.ts.dt.date < FORWARD_END].copy()
    futures, selection = load_front_month("BANKNIFTY", DEFAULT_ROOT)
    futures = futures[futures.ts.dt.date < FORWARD_END].copy()
    frame = make_signals(spot, futures)
    forward = frame[(frame.ts.dt.date >= FORWARD_START) & (frame.ts.dt.date < FORWARD_END)].copy()
    baseline_trades = run_one(forward, "BANKNIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
    candidate_trades = run_one(apply_rule(forward), "BANKNIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
    rows = []
    for name, trades, rule in (("baseline", baseline_trades, "pivot_baseline"),
                               ("candidate", candidate_trades, "long_body_ge_0.8_short_off")):
        stats = summarize(trades, "BANKNIFTY", True, "FORWARD_JUN01_OCT05")
        stats.update({"rule": rule, "recent_seven_sessions_used": False,
                      "unavailable_sessions": int(((selection.session_date >= FORWARD_START)
                                                   & (selection.session_date < FORWARD_END)
                                                   & (selection.status != "SELECTED")).sum())})
        rows.append(stats)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    if not candidate_trades.empty:
        candidate_trades = candidate_trades.copy()
        candidate_trades["rule"] = "long_body_ge_0.8_short_off"
        candidate_trades.to_csv(OUT / "candidate_trades.csv", index=False)
    print("FROZEN BANKNIFTY FORWARD | fixed rule; PAPER ONLY; estimated Upstox costs + 1 point/side")
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}; Oct 6 onward excluded before signal construction; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
