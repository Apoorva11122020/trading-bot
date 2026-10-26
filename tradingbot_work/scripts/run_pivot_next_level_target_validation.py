#!/usr/bin/env python3
"""Test next-classical-pivot target exits on Oct-May; actual futures, paper only."""
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

OUT = ROOT / "reports" / "pivot_next_level_target_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def add_next_level_targets(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    # Since R1-S1 equals the prior session's high-low range:
    # R2 = PP + range and S2 = PP - range.
    range_points = x.r1 - x.s1
    r2 = x.pp + range_points
    s2 = x.pp - range_points
    long_target_spot = x.r1.where(x.close_spot <= x.r1, r2)
    short_target_spot = x.s1.where(x.close_spot >= x.s1, s2)
    x["pivot_target_fut"] = x.close_fut + (
        long_target_spot - x.close_spot
    ).where(x.long_signal, (short_target_spot - x.close_spot).where(x.short_signal))
    return x


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
        frame = make_signals(spot, futures)
        frame = frame[(frame.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                      (frame.ts.dt.date < DEV_END)].copy()
        pivot_target = add_next_level_targets(frame)
        variants = (("baseline_rr1.5", frame, None),
                    ("next_pivot_level_target", pivot_target, "pivot_target_fut"))
        for rule, data, target_col in variants:
            trades = run_one(data, symbol, True, "upstox", 1.0, 1.5, 0.1, 1,
                             target_column=target_col)
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
    print("Train Oct-Jan; validation Feb-May; June onward/recent seven sessions excluded before simulation.")
    print("Actual selected futures; estimated Upstox statutory charges + 1 point/side; volume ON; stop buffer 0.1 ATR.")
    print("Paper/research only; no orders submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
