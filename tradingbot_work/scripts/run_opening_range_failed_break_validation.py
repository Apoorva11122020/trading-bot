#!/usr/bin/env python3
"""Test failed 30m opening-range breaks fading to midpoint; Oct-May only."""
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

OUT = ROOT / "reports" / "opening_range_failed_break_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def failed_break_signals(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy().sort_values("ts").reset_index(drop=True)
    prev_close = x.groupby("session", sort=False).close_spot.shift(1)
    minute = x.ts.dt.hour * 60 + x.ts.dt.minute
    allowed = (minute >= 9 * 60 + 45) & (minute < 15 * 60)
    long_sig = (
        allowed & (prev_close >= x.or_low) & (x.low_spot < x.or_low)
        & (x.close_spot > x.or_low) & (x.close_spot >= x.open_spot)
        & (x.close_spot < x.or_mid)
    )
    short_sig = (
        allowed & (prev_close <= x.or_high) & (x.high_spot > x.or_high)
        & (x.close_spot < x.or_high) & (x.close_spot <= x.open_spot)
        & (x.close_spot > x.or_mid)
    )
    x["long_signal"] = long_sig
    x["short_signal"] = short_sig
    target_spot = x.or_mid.where(long_sig | short_sig)
    x["or_mid_target_fut"] = x.close_fut + (target_spot - x.close_spot)
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
        merged = make_signals(spot, futures)
        minute = merged.ts.dt.hour * 60 + merged.ts.dt.minute
        opening = merged[(minute >= 9 * 60 + 15) & (minute < 9 * 60 + 45)]
        levels = opening.groupby("session", sort=True).agg(
            or_high=("high_spot", "max"), or_low=("low_spot", "min"), bars=("high_spot", "size")
        )
        levels["or_mid"] = (levels.or_high + levels.or_low) / 2
        merged = merged.join(levels[["or_high", "or_low", "or_mid", "bars"]], on="session")
        dev = merged[(merged.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                     (merged.ts.dt.date < DEV_END) & (merged.bars == 6)].copy()
        candidate = failed_break_signals(dev)
        baseline = dev.copy()
        for rule, data, target in (
            ("pivot_baseline", baseline, None),
            ("failed_or_fade_to_midpoint", candidate, "or_mid_target_fut"),
        ):
            trades = run_one(data, symbol, True, "upstox", 1.0, 1.5, 0.1, 1,
                             target_column=target)
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
                st = summarize(part, symbol, True, period)
                st["rule"] = rule
                summaries.append(st)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Train Oct-Jan; validation Feb-May; June onward/recent seven sessions excluded before simulation.")
    print("Failed opening-range wick rejection after 09:45; midpoint target; actual futures; modeled fees + 1 point/side.")
    print("Paper/research only; no broker orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
