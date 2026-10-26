#!/usr/bin/env python3
"""Recheck fixed NIFTY futures-VWAP trend rule with expanded TRAIN history.

Rule is frozen from the earlier study: after 09:45, futures close crosses
session VWAP while spot EMA20/EMA50 and RSI14 agree (RSI >55 long, <45 short).
TRAIN uses Jan 2025-Jan 2026 expanded history; Feb-May 2026 is validation;
the frozen June-Oct 5 forward period is reported as a one-time diagnostic and
is never used to select or modify the rule. Oct 6 onward is never loaded into
signal construction. PAPER-only, no orders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, IST, load_front_month, load_spot, make_signals,
    run_one, summarize,
)
from scripts.run_extended_nifty_train_selection import load_early_futures, load_early_spot
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "extended_nifty_vwap_train_check_20261014"
TRAIN_START = pd.Timestamp("2025-01-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
FORWARD_END = pd.Timestamp("2026-10-06").date()
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0


def with_vwap_cross(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy().sort_values("ts").reset_index(drop=True)
    typical = (x.high_fut + x.low_fut + x.close_fut) / 3.0
    pv = typical * x.volume_fut
    x["session_vwap"] = pv.groupby(x.session, sort=False).cumsum() / x.volume_fut.groupby(x.session, sort=False).cumsum().where(
        x.volume_fut.groupby(x.session, sort=False).cumsum() > 0)
    prev_close = x.groupby("session", sort=False).close_fut.shift(1)
    prev_vwap = x.groupby("session", sort=False).session_vwap.shift(1)
    minute = x.ts.dt.hour * 60 + x.ts.dt.minute
    window = (minute >= 9 * 60 + 45) & (minute < 15 * 60)
    x["long_signal"] = window & (x.close_fut > x.session_vwap) & (prev_close <= prev_vwap) & \
        (x.ema20_spot > x.ema50_spot) & (x.rsi14_spot > 55)
    x["short_signal"] = window & (x.close_fut < x.session_vwap) & (prev_close >= prev_vwap) & \
        (x.ema20_spot < x.ema50_spot) & (x.rsi14_spot < 45)
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    old_spot = load_early_spot()
    new_spot = load_spot(DEFAULT_SPOT / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv")
    new_spot = new_spot[(new_spot.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                        (new_spot.ts.dt.date < FORWARD_END)]
    spot = pd.concat([old_spot, new_spot], ignore_index=True).sort_values("ts").drop_duplicates("ts")
    old_futures, early_selection = load_early_futures()
    later_futures, _ = load_front_month("NIFTY", DEFAULT_ROOT)
    later_futures = later_futures[(later_futures.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                                  (later_futures.ts.dt.date < FORWARD_END)]
    futures = pd.concat([old_futures, later_futures], ignore_index=True).sort_values("ts")
    if futures.duplicated("ts").any():
        raise SystemExit("Duplicate selected NIFTY futures timestamps across source boundary")
    merged = make_signals(spot, futures)
    merged = merged[(merged.ts.dt.date >= TRAIN_START) & (merged.ts.dt.date < FORWARD_END)].copy()
    candidate = with_vwap_cross(merged)
    periods = {
        "TRAIN_JAN25_JAN26": (TRAIN_START, TRAIN_END),
        "VALIDATION_FEB_MAY26": (TRAIN_END, VALID_END),
        "FORWARD_JUN01_OCT05": (VALID_END, FORWARD_END),
    }
    rows = []
    baseline_trades = {}
    candidate_trades = {}
    for name, (start, end) in periods.items():
        base_frame = merged[(merged.ts.dt.date >= start) & (merged.ts.dt.date < end)]
        cand_frame = candidate[(candidate.ts.dt.date >= start) & (candidate.ts.dt.date < end)]
        baseline_trades[name] = run_one(base_frame, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        candidate_trades[name] = run_one(cand_frame, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        for label, trades in (("pivot_baseline", baseline_trades[name]), ("futures_vwap_cross_trend", candidate_trades[name])):
            stats = summarize(trades, "NIFTY", True, name)
            rows.append({**stats, "rule": label})
    stats_df = pd.DataFrame(rows)
    def get(rule: str, period: str, col: str):
        return stats_df.loc[(stats_df.rule == rule) & (stats_df.split == period), col].iloc[0]
    train_pass = bool(get("futures_vwap_cross_trend", "TRAIN_JAN25_JAN26", "trades") >= 20
                      and get("futures_vwap_cross_trend", "TRAIN_JAN25_JAN26", "win_rate_pct") > get("pivot_baseline", "TRAIN_JAN25_JAN26", "win_rate_pct")
                      and get("futures_vwap_cross_trend", "TRAIN_JAN25_JAN26", "net_points") > get("pivot_baseline", "TRAIN_JAN25_JAN26", "net_points"))
    val_pass = bool(train_pass
                    and get("futures_vwap_cross_trend", "VALIDATION_FEB_MAY26", "trades") >= 8
                    and get("futures_vwap_cross_trend", "VALIDATION_FEB_MAY26", "win_rate_pct") >= 50
                    and get("futures_vwap_cross_trend", "VALIDATION_FEB_MAY26", "net_points") > 0)
    gate = pd.DataFrame([{"train_gate_pass": train_pass, "validation_gate_pass": val_pass,
                          "forward_evaluated": True, "forward_used_for_selection": False,
                          "recent_seven_sessions_used": False,
                          "unavailable_early_sessions": int((early_selection.status != "SELECTED").sum())}])
    stats_df.to_csv(OUT / "period_results.csv", index=False)
    gate.to_csv(OUT / "gate.csv", index=False)
    if val_pass:
        trades = candidate_trades["FORWARD_JUN01_OCT05"].copy()
        trades.to_csv(OUT / "frozen_forward_trades.csv", index=False)
    print("FIXED NIFTY VWAP TREND RULE | PAPER ONLY | modeled Upstox fees + 1 point/side")
    print(stats_df.to_string(index=False))
    print(gate.to_string(index=False))
    print(f"Saved: {OUT}; Oct 6 onward excluded before signal construction; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
