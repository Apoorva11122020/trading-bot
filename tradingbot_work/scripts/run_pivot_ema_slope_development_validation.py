#!/usr/bin/env python3
"""Test one fixed, causal EMA20-slope confirmation on Oct-May only; paper research."""
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

OUT = ROOT / "reports" / "pivot_ema_slope_development_20261014"
DEV_END = pd.Timestamp("2026-06-01").date()
VALIDATION_START = pd.Timestamp("2026-02-01").date()
TRAIN_START = pd.Timestamp("2025-10-01").date()


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
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        # Drop all rows from June onward before either strategy is simulated.
        dev = make_signals(spot, futures)
        dev = dev[(dev.ts.dt.date >= TRAIN_START) & (dev.ts.dt.date < DEV_END)].copy()
        slope_ref = dev.groupby("session", sort=False)["ema20_spot"].shift(3)
        candidate = dev.copy()
        candidate["long_signal"] = candidate.long_signal & candidate.ema20_spot.gt(slope_ref)
        candidate["short_signal"] = candidate.short_signal & candidate.ema20_spot.lt(slope_ref)
        for rule, frame in (("baseline", dev), ("ema20_slope_3bars_directional", candidate)):
            trades = run_one(frame, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
            if not trades.empty:
                entries = pd.to_datetime(trades.entry_ts).dt.date
                exits = pd.to_datetime(trades.exit_ts).dt.date
                trades["period"] = entries.map(lambda d: "TRAIN_OCT_JAN" if d < VALIDATION_START else "VALIDATION_FEB_MAY")
                trades["rule"] = rule
                # Ignore any boundary-straddling position in the split summary.
                trades.loc[(entries < VALIDATION_START) & (exits >= VALIDATION_START), "period"] = "BOUNDARY_EXCLUDED"
                trade_frames.append(trades)
            for period in ("TRAIN_OCT_JAN", "VALIDATION_FEB_MAY"):
                if trades.empty:
                    part = trades
                else:
                    part = trades[(trades.period == period) &
                                  (pd.to_datetime(trades.entry_ts).dt.date == pd.to_datetime(trades.exit_ts).dt.date)]
                stats = summarize(part, symbol, True, period)
                stats["rule"] = rule
                stats["coverage_unavailable_sessions"] = int(((selection.session_date < pd.Timestamp("2026-06-01").date()) &
                                                                  (selection.status != "SELECTED")).sum())
                summaries.append(stats)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Development only: train Oct-Jan, validation Feb-May; June onward excluded before simulation.")
    print("Actual selected futures, estimated Upstox statutory fees + 1 point/side; volume filter ON; RR 1.5 / stop buffer 0.1.")
    print("Paper/research only; no orders submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
