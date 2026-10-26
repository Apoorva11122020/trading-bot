#!/usr/bin/env python3
"""Test a one-short-entry-per-session rule on Oct-May; paper research only."""
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

OUT = ROOT / "reports" / "pivot_one_short_per_day_20261014"
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
        frame = make_signals(spot, futures)
        frame = frame[(frame.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                      (frame.ts.dt.date < DEV_END)].copy()
        short_only = frame.copy()
        short_only["long_signal"] = False
        # Mirror the simulator's volume condition, then retain the first eligible signal per day.
        eligible = (
            short_only.short_signal
            & short_only.vol_avg20_fut.notna()
            & short_only.vol_avg20_fut.gt(0)
            & short_only.volume_fut.gt(short_only.vol_avg20_fut * 1.2)
        )
        first_signal = eligible.groupby(short_only.session, sort=False).transform(
            lambda s: s & (s.cumsum() == 1)
        )
        once_daily = short_only.copy()
        once_daily["short_signal"] = first_signal
        variants = (("baseline", frame), ("train_selected_short_only", short_only),
                    ("short_only_first_eligible_per_session", once_daily))
        for rule, data in variants:
            trades = run_one(data, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
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
    print("Short-only, max one entry/session; actual futures, modeled Upstox costs + 1 point/side.")
    print("Paper/research only; no broker orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
