#!/usr/bin/env python3
"""Frozen TRAIN-selected short-only rule final forward check; excludes recent 7 sessions."""
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

OUT = ROOT / "reports" / "pivot_short_only_forward_eval_20261014"
FINAL_EVAL_END = pd.Timestamp("2026-10-06").date()  # exclusive; leave Oct 6-14 holdout untouched


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
        # Do not simulate any Oct 6-14 signal or position.
        merged = merged[merged.ts.dt.date < FINAL_EVAL_END].copy()
        short_only = merged.copy()
        short_only["long_signal"] = False
        typical = (merged.high_fut + merged.low_fut + merged.close_fut) / 3.0
        cumulative_pv = (typical * merged.volume_fut).groupby(merged.session, sort=False).cumsum()
        cumulative_volume = merged.volume_fut.groupby(merged.session, sort=False).cumsum()
        short_vwap = short_only.copy()
        short_vwap["long_signal"] = False
        short_vwap["short_signal"] &= (
            merged.close_fut < cumulative_pv / cumulative_volume.where(cumulative_volume > 0)
        )
        session_vwap = cumulative_pv / cumulative_volume.where(cumulative_volume > 0)
        vwap_3bars_ago = session_vwap.groupby(merged.session, sort=False).shift(3)
        short_vwap_falling = short_only.copy()
        short_vwap_falling["short_signal"] &= session_vwap < vwap_3bars_ago
        session_open = merged.groupby("session", sort=False).open_spot.transform("first")
        short_below_open = short_only.copy()
        short_below_open["short_signal"] &= short_below_open.close_spot < session_open
        daily = spot.groupby("session", sort=True).agg(day_open=("open", "first"), day_close=("close", "last"))
        overnight_gap = daily.day_open - daily.day_close.shift(1)
        bearish_gap = overnight_gap < 0
        short_bearish_gap = short_only.copy()
        short_bearish_gap["short_signal"] &= short_bearish_gap.session.map(bearish_gap).fillna(False).astype(bool)
        selected_gap_condition = overnight_gap.gt(0) if symbol == "NIFTY" else overnight_gap.lt(0)
        short_train_selected_gap = short_only.copy()
        short_train_selected_gap["short_signal"] &= short_train_selected_gap.session.map(
            selected_gap_condition
        ).fillna(False).astype(bool)
        rsi65_candidate = merged.copy()
        if symbol == "NIFTY":
            rsi65_candidate["long_signal"] = False
            rsi65_candidate["short_signal"] = (
                rsi65_candidate.in_session & rsi65_candidate.cross_down
                & (rsi65_candidate.ema20_spot < rsi65_candidate.ema50_spot)
                & (rsi65_candidate.rsi14_spot < 35)
            )
        else:
            rsi65_candidate["long_signal"] = (
                rsi65_candidate.in_session & rsi65_candidate.cross_up
                & (rsi65_candidate.ema20_spot > rsi65_candidate.ema50_spot)
                & (rsi65_candidate.rsi14_spot > 65)
            )
            rsi65_candidate["short_signal"] = False
        variants = (
            ("baseline", merged, None),
            ("frozen_short_only", short_only, None),
            ("frozen_short_only_close_confirmed_be075", short_only, 0.75),
            ("frozen_short_only_below_session_vwap", short_vwap, None),
            ("frozen_short_only_vwap_falling_3bars", short_vwap_falling, None),
            ("frozen_short_only_below_session_open", short_below_open, None),
            ("frozen_short_only_bearish_overnight_gap", short_bearish_gap, None),
            ("frozen_train_selected_rsi65_side", rsi65_candidate, None),
            ("frozen_train_selected_gap_side", short_train_selected_gap, None),
        )
        for rule, frame, be_trigger in variants:
            trades = run_one(frame, symbol, True, "upstox", 1.0, 1.5, 0.1, 1,
                             breakeven_trigger_r=be_trigger)
            if not trades.empty:
                entries = pd.to_datetime(trades.entry_ts).dt.date
                exits = pd.to_datetime(trades.exit_ts).dt.date
                evaluated = trades[(entries >= OOS_START) & (exits < FINAL_EVAL_END)]
                evaluated = evaluated.copy()
                evaluated["period"] = "FORWARD_JUN_01_TO_OCT_05"
                evaluated["rule"] = rule
                trade_frames.append(evaluated)
            else:
                evaluated = trades
            stats = summarize(evaluated, symbol, True, "FORWARD_JUN_01_TO_OCT_05")
            stats["rule"] = rule
            stats["recent_7_sessions_used"] = False
            summaries.append(stats)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Frozen rule candidates; June 1-Oct 5 final forward evaluation; Oct 6-14 excluded.")
    print("Actual selected futures; estimated Upstox statutory fees + 1 point/side; volume ON; RR 1.5; stop 0.1 ATR.")
    print("Paper/research only; no orders submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
