#!/usr/bin/env python3
"""TRAIN-selected RSI65 directional rules; exact re-simulation on Oct-May."""
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

OUT = ROOT / "reports" / "pivot_train_selected_rsi_sides_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def selected_signals(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    x = frame.copy()
    if symbol == "NIFTY":
        # TRAIN: RSI>65/<35 short-side had positive net and higher hit rate than long-side.
        x["long_signal"] = False
        x["short_signal"] = (
            x.in_session & x.cross_down & (x.ema20_spot < x.ema50_spot)
            & (x.rsi14_spot < 35)
        )
    else:
        # TRAIN: RSI>65/<35 long-side was less negative and higher-hit than short-side.
        x["long_signal"] = (
            x.in_session & x.cross_up & (x.ema20_spot > x.ema50_spot)
            & (x.rsi14_spot > 65)
        )
        x["short_signal"] = False
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
        candidate = selected_signals(frame, symbol)
        for rule, data in (("baseline_rsi55_both_sides", frame),
                           ("train_selected_rsi65_one_side", candidate)):
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
                is_candidate = rule != "baseline_rsi55_both_sides"
                stats["selected_side"] = ("SHORT" if symbol == "NIFTY" else "LONG") if is_candidate else "BOTH"
                stats["rsi_rule"] = ("RSI<35" if symbol == "NIFTY" else "RSI>65") if is_candidate else "RSI>55/<45"
                summaries.append(stats)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Side and RSI thresholds selected on TRAIN Oct-Jan; exact VALIDATION Feb-May re-simulation.")
    print("June onward/recent seven sessions excluded. Actual selected futures; estimated Upstox fees + 1 point/side; RR 1.5 / stop 0.1 ATR.")
    print("Paper/research only; no broker orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
