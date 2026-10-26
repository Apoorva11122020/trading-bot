#!/usr/bin/env python3
"""Test prior-session close-location continuation on pivot signals.

Fixed causal rule: retain baseline long signals only when the previous
session's close was in the top quartile of that session's spot range; retain
short signals only when the previous close was in the bottom quartile. No
threshold search. Evaluate TRAIN Oct-Jan and VALIDATION Feb-May separately;
June onward and the recent seven sessions are excluded before feature creation.
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

OUT = ROOT / "reports" / "prior_close_location_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, trade_frames = [], []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        spot = spot[(spot.ts.dt.date >= START) & (spot.ts.dt.date < VALID_END)].copy()
        daily = spot.groupby("session", sort=True).agg(
            day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")
        )
        width = (daily.day_high - daily.day_low).where(lambda s: s > 0)
        close_location = (daily.day_close - daily.day_low) / width
        previous_close_location = close_location.shift(1)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        frame["previous_close_location"] = frame.session.map(previous_close_location)
        filtered = frame.copy()
        filtered["long_signal"] = (
            filtered.long_signal.astype(bool) & (filtered.previous_close_location >= 0.75)
        )
        filtered["short_signal"] = (
            filtered.short_signal.astype(bool) & (filtered.previous_close_location <= 0.25)
        )
        for label, begin, end in (
            ("TRAIN_OCT_JAN", START, TRAIN_END),
            ("VALIDATION_FEB_MAY", TRAIN_END, VALID_END),
        ):
            missing = selection[(selection.session_date >= begin) &
                                (selection.session_date < end) &
                                (selection.status != "SELECTED")]
            base_period = frame[(frame.ts.dt.date >= begin) & (frame.ts.dt.date < end)].copy()
            candidate_period = filtered[(filtered.ts.dt.date >= begin) &
                                        (filtered.ts.dt.date < end)].copy()
            for rule, data in (("baseline", base_period),
                               ("prior_close_top_bottom_quartile", candidate_period)):
                trades = run_one(data, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
                stat = summarize(trades, symbol, True, label)
                stat.update({"rule": rule, "unavailable_sessions": len(missing)})
                summaries.append(stat)
                if not trades.empty:
                    trades = trades.copy()
                    trades["period"], trades["rule"] = label, rule
                    trade_frames.append(trades)
    results = pd.DataFrame(summaries)
    results.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(results.to_string(index=False))
    print("Rule: prior session close location >=75% of range for longs, <=25% for shorts.")
    print("Actual selected futures; modeled Upstox charges + 1 point/side; paper only; no orders.")
    print("Oct-Jan TRAIN; Feb-May validation; June onward and recent seven sessions excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
