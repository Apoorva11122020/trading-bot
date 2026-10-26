#!/usr/bin/env python3
"""Causal completed-30m EMA trend filter for pivot signals; Oct-May only."""
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
from scripts.pivot_indicators import ema
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "pivot_30m_trend_validation_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()
IST = "Asia/Kolkata"


def completed_30m_features(spot: pd.DataFrame) -> pd.DataFrame:
    x = spot.copy().sort_values("ts")
    minute = x.ts.dt.hour * 60 + x.ts.dt.minute
    x["slot"] = ((minute - (9 * 60 + 15)) // 30).astype(int)
    bars = x.groupby(["session", "slot"], sort=True).agg(
        close=("close", "last"), count=("close", "size"), last_start=("ts", "last")
    ).reset_index()
    # Keep full six-candle bars only. Features become available at the end of the
    # last constituent candle and are never mapped backward into that interval.
    bars = bars[bars["count"] == 6].copy()
    bars["available_ts"] = bars.last_start + pd.Timedelta(minutes=5)
    bars["ema20_30m"] = ema(bars.close, 20)
    bars["ema50_30m"] = ema(bars.close, 50)
    bars = bars.sort_values("available_ts")
    signals = x[["ts"]].copy().sort_values("ts")
    signals["signal_close_ts"] = signals.ts + pd.Timedelta(minutes=5)
    aligned = pd.merge_asof(
        signals.sort_values("signal_close_ts"),
        bars[["available_ts", "ema20_30m", "ema50_30m"]].sort_values("available_ts"),
        left_on="signal_close_ts", right_on="available_ts", direction="backward",
        allow_exact_matches=True,
    )
    return aligned[["ts", "ema20_30m", "ema50_30m"]]


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
        features = completed_30m_features(spot)
        frame = make_signals(spot, futures).merge(features, on="ts", how="left", validate="one_to_one")
        frame = frame[(frame.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                      (frame.ts.dt.date < DEV_END)].copy()
        candidate = frame.copy()
        candidate["long_signal"] &= candidate.ema20_30m.gt(candidate.ema50_30m)
        candidate["short_signal"] &= candidate.ema20_30m.lt(candidate.ema50_30m)
        for rule, data in (("baseline", frame), ("completed_30m_ema20_50_alignment", candidate)):
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
    print("Train Oct-Jan; validation Feb-May; June onward excluded before simulation.")
    print("Completed 30m bars only; actual selected futures, estimated Upstox charges + 1 point/side; volume ON.")
    print("Paper/research only; no broker orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
