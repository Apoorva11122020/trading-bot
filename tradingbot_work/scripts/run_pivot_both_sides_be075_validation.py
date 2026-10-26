#!/usr/bin/env python3
"""Evaluate fixed close-confirmed, fee-adjusted breakeven on both pivot sides.

Baseline signals and entry/target rules are unchanged. After a completed
futures candle closes >=0.75R in favor, the stop is moved to entry plus
modeled round-trip cost for the next candle. Evaluate separate chronological
TRAIN (Oct-Jan) and VALIDATION (Feb-May) slices; do not inspect later data.
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

OUT = ROOT / "reports" / "pivot_both_sides_be075_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
SLIPPAGE, RR, ATR_BUFFER, BE_TRIGGER = 1.0, 1.5, 0.1, 0.75


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, trade_frames = [], []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        spot = spot[(spot.ts.dt.date >= START) & (spot.ts.dt.date < VALID_END)]
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        for label, begin, end in (
            ("TRAIN_OCT_JAN", START, TRAIN_END),
            ("VALIDATION_FEB_MAY", TRAIN_END, VALID_END),
        ):
            period = frame[(frame.ts.dt.date >= begin) & (frame.ts.dt.date < end)].copy()
            missing = selection[(selection.session_date >= begin) &
                                (selection.session_date < end) &
                                (selection.status != "SELECTED")]
            for rule, trigger in (("baseline", None),
                                  ("both_sides_close_confirmed_be075", BE_TRIGGER)):
                trades = run_one(period, symbol, True, "upstox", SLIPPAGE,
                                 RR, ATR_BUFFER, 1,
                                 breakeven_trigger_r=trigger)
                stat = summarize(trades, symbol, True, label)
                stat.update({"rule": rule, "unavailable_sessions": len(missing),
                             "breakeven_trigger_r": trigger})
                summaries.append(stat)
                if not trades.empty:
                    trades = trades.copy()
                    trades["period"], trades["rule"] = label, rule
                    trade_frames.append(trades)
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(result.to_string(index=False))
    print("Fixed BE: after 0.75R close confirmation, next-bar stop = entry +/- modeled roundtrip cost.")
    print("Actual selected futures; estimated Upstox fees + 1 point/side; volume ON; no orders.")
    print(f"June onward/recent seven sessions excluded; report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
