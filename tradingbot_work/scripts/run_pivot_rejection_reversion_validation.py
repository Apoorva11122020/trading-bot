#!/usr/bin/env python3
"""Test fixed PP-target pivot rejection reversals on Oct-May; paper research only."""
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

OUT = ROOT / "reports" / "pivot_rejection_reversion_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def reversion_signals(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy().sort_values("ts").reset_index(drop=True)
    previous_close = x.groupby("session", sort=False).close_spot.shift(1)
    # Buy a bullish rejection of S1; mean-reversion target is PP.
    long = (
        x.in_session & (x.low_spot <= x.s1) & (previous_close >= x.s1)
        & (x.close_spot > x.s1) & (x.close_spot >= x.open_spot)
        & (x.close_spot < x.pp)
    )
    # Short a bearish rejection of R1; mean-reversion target is PP.
    short = (
        x.in_session & (x.high_spot >= x.r1) & (previous_close <= x.r1)
        & (x.close_spot < x.r1) & (x.close_spot <= x.open_spot)
        & (x.close_spot > x.pp)
    )
    x["long_signal"] = long
    x["short_signal"] = short
    # Translate the spot pivot target to futures price using the signal-candle basis.
    x["target_pivot_fut"] = x.pp + (x.close_fut - x.close_spot)
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    )
    summaries, trades_out = [], []
    for symbol, spot_file in settings:
        spot = load_spot(DEFAULT_SPOT / spot_file)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        merged = make_signals(spot, futures)
        merged = merged[(merged.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                        (merged.ts.dt.date < DEV_END)].copy()
        for rule, frame, target_col in (
            ("baseline_pivot_breakout_rr1.5", merged, None),
            ("reject_S1_R1_close_back_target_PP", reversion_signals(merged), "target_pivot_fut"),
        ):
            trades = run_one(frame, symbol, True, "upstox", 1.0, 1.5, 0.1, 1, target_column=target_col)
            if not trades.empty:
                entries = pd.to_datetime(trades.entry_ts).dt.date
                exits = pd.to_datetime(trades.exit_ts).dt.date
                trades["period"] = entries.map(lambda d: "TRAIN_OCT_JAN" if d < TRAIN_END else "VALIDATION_FEB_MAY")
                trades.loc[(entries < TRAIN_END) & (exits >= TRAIN_END), "period"] = "BOUNDARY_EXCLUDED"
                trades["rule"] = rule
                trades_out.append(trades)
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
    if trades_out:
        pd.concat(trades_out, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Train Oct-Jan; validation Feb-May. June onward/recent seven sessions excluded before simulation.")
    print("Actual selected futures; modeled Upstox statutory charges + 1 point/side; volume ON; stop = signal rejection extreme +/- 0.1 ATR; target = PP.")
    print("Paper/research only; no orders submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
