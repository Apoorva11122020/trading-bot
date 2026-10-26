#!/usr/bin/env python3
"""Frozen BANKNIFTY short-only below futures session VWAP forward test.

Fixed rule passed TRAIN relative-improvement and Feb-May validation gates.
This script checks June 1-Oct 5 once, without tuning. Oct 6 onward is excluded
before signal construction. Actual selected futures, volume filter, modeled
Upstox charges + 1 point/side slippage. PAPER only; no orders.
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

OUT = ROOT / "reports" / "frozen_banknifty_short_below_vwap_forward_20261014"
FORWARD_START = pd.Timestamp("2026-06-01").date()
FORWARD_END = pd.Timestamp("2026-10-06").date()
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0


def add_vwap(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy().sort_values("ts").reset_index(drop=True)
    typical = (x.high_fut + x.low_fut + x.close_fut) / 3
    pv = typical * x.volume_fut
    pv_sum = pv.groupby(x.session, sort=False).cumsum()
    vol_sum = x.volume_fut.groupby(x.session, sort=False).cumsum()
    x["session_vwap_fut"] = pv_sum / vol_sum.where(vol_sum > 0)
    return x


def apply_rule(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["long_signal"] = False
    x["short_signal"] = x.short_signal.astype(bool) & x.close_fut.lt(x.session_vwap_fut)
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    spot = load_spot(DEFAULT_SPOT / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv")
    spot = spot[spot.ts.dt.date < FORWARD_END].copy()
    futures, selection = load_front_month("BANKNIFTY", DEFAULT_ROOT)
    futures = futures[futures.ts.dt.date < FORWARD_END].copy()
    frame = add_vwap(make_signals(spot, futures))
    forward = frame[(frame.ts.dt.date >= FORWARD_START) & (frame.ts.dt.date < FORWARD_END)].copy()
    rows = []
    for name, data in (("pivot_baseline", forward), ("short_only_close_below_futures_session_vwap", apply_rule(forward))):
        trades = run_one(data, "BANKNIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        stats = summarize(trades, "BANKNIFTY", True, "FORWARD_JUN01_OCT05")
        stats.update({"rule": name, "recent_seven_sessions_used": False,
                      "unavailable_sessions": int(((selection.session_date >= FORWARD_START)
                                                   & (selection.session_date < FORWARD_END)
                                                   & (selection.status != "SELECTED")).sum())})
        rows.append(stats)
        if name != "pivot_baseline" and not trades.empty:
            trades = trades.copy()
            trades["rule"] = name
            trades.to_csv(OUT / "candidate_trades.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print("FROZEN BANKNIFTY SHORT BELOW VWAP | PAPER ONLY | estimated Upstox fees + 1 point/side")
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}; Oct 6 onward excluded; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
