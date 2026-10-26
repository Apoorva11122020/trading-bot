#!/usr/bin/env python3
"""One predeclared, TRAIN-selected entry-time filter; chronological validation, paper only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, OOS_START, load_front_month, load_spot,
    make_signals, run_one, summarize,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "pivot_train_selected_time_filter_20261014"
WINDOWS = {"NIFTY": (11 * 60, 13 * 60), "BANKNIFTY": (13 * 60, 15 * 60)}
TRAIN_END = pd.Timestamp("2026-02-01").date()


def split_label(ts: object) -> str:
    day = pd.Timestamp(ts).date()
    if day < TRAIN_END:
        return "TRAIN_OCT_JAN"
    if day < OOS_START:
        return "VALIDATION_FEB_MAY"
    return "OOS_JUN_ONWARD"


def time_filtered(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    x = frame.copy()
    start, end = WINDOWS[symbol]
    # Signals become known at candle close; the simulator enters at the next 5m open.
    signal_minute = x.ts.dt.hour * 60 + x.ts.dt.minute
    entry_minute = signal_minute + 5
    keep = entry_minute.ge(start) & entry_minute.lt(end)
    x.loc[~keep, ["long_signal", "short_signal"]] = False
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
        baseline = make_signals(spot, futures)
        variants = (("baseline", baseline), ("train_selected_time_window", time_filtered(baseline, symbol)))
        for rule, frame in variants:
            trades = run_one(frame, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
            if not trades.empty:
                trades["period"] = trades.entry_ts.map(split_label)
                trades["rule"] = rule
                trade_frames.append(trades)
            for period in ("TRAIN_OCT_JAN", "VALIDATION_FEB_MAY", "OOS_JUN_ONWARD"):
                part = trades.loc[trades.period.eq(period)] if not trades.empty else trades
                stats = summarize(part, symbol, True, period)
                stats["rule"] = rule
                stats["entry_window_ist"] = f"{WINDOWS[symbol][0] // 60:02d}:00-{WINDOWS[symbol][1] // 60:02d}:00" if rule != "baseline" else "ALL"
                summaries.append(stats)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Train-selected rule: NIFTY 11:00-13:00; BANKNIFTY 13:00-15:00 IST.")
    print("One lot, actual selected futures, estimated Upstox fees + 1 point/side; paper only, no orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
