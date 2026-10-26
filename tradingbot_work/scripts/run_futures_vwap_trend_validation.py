#!/usr/bin/env python3
"""Test futures session-VWAP cross with aligned EMA/RSI trend; Oct-May only."""
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

OUT = ROOT / "reports" / "futures_vwap_trend_validation_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def with_vwap_cross(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy().sort_values("ts").reset_index(drop=True)
    typical = (x.high_fut + x.low_fut + x.close_fut) / 3.0
    pv = typical * x.volume_fut
    cumulative_pv = pv.groupby(x.session, sort=False).cumsum()
    cumulative_volume = x.volume_fut.groupby(x.session, sort=False).cumsum()
    x["session_vwap"] = cumulative_pv / cumulative_volume.where(cumulative_volume > 0)
    previous_close = x.groupby("session", sort=False).close_fut.shift(1)
    previous_vwap = x.groupby("session", sort=False).session_vwap.shift(1)
    minute = x.ts.dt.hour * 60 + x.ts.dt.minute
    entry_window = (minute >= 9 * 60 + 45) & (minute < 15 * 60)
    cross_up = (x.close_fut > x.session_vwap) & (previous_close <= previous_vwap)
    cross_down = (x.close_fut < x.session_vwap) & (previous_close >= previous_vwap)
    x["long_signal"] = entry_window & cross_up & (x.ema20_spot > x.ema50_spot) & (x.rsi14_spot > 55)
    x["short_signal"] = entry_window & cross_down & (x.ema20_spot < x.ema50_spot) & (x.rsi14_spot < 45)
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
        dev = merged[(merged.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                     (merged.ts.dt.date < DEV_END)].copy()
        candidates = (("pivot_baseline", dev), ("futures_vwap_cross_trend", with_vwap_cross(dev)))
        for rule, frame in candidates:
            trades = run_one(frame, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
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
    print("Train Oct-Jan; validation Feb-May; June onward and recent seven sessions excluded before simulation.")
    print("Futures session-VWAP crosses only after 09:45; same-direction spot EMA20/50 and RSI>55/<45.")
    print("Actual selected futures; estimated Upstox charges + 1 point/side; RR 1.5 / stop buffer 0.1 ATR; volume confirmation ON.")
    print("Paper/research only; no orders submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
