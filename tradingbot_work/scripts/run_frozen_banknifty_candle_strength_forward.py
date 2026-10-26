#!/usr/bin/env python3
"""Frozen BANKNIFTY 0.8 directional candle-body rule forward check.

Rule was selected by TRAIN ranking (among 0.4/0.6/0.8 body fractions) and
showed positive Feb-May validation metrics. This script performs one forward
check on Jun 1-Oct 5; it does not tune the rule. Spot candles are truncated
before the recent seven-session window, and no orders are submitted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, load_front_month, load_spot, make_signals,
    run_one, summarize,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "frozen_banknifty_candle_strength_forward_20261014"
OOS_START = pd.Timestamp("2026-06-01").date()
OOS_END = pd.Timestamp("2026-10-06").date()
THRESHOLD = 0.8
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1


def apply_rule(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    span = (x.high_spot - x.low_spot).replace(0, np.nan)
    body = (x.close_spot - x.open_spot).abs() / span
    x["long_signal"] = x.long_signal.astype(bool) & (x.close_spot > x.open_spot) & (body >= THRESHOLD)
    x["short_signal"] = x.short_signal.astype(bool) & (x.close_spot < x.open_spot) & (body >= THRESHOLD)
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    spot = load_spot(DEFAULT_SPOT / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv")
    # Do not pass recent seven-session candles to signal construction.
    spot = spot[spot.ts.dt.date < OOS_END].copy()
    futures, selection = load_front_month("BANKNIFTY", DEFAULT_ROOT)
    full = make_signals(spot, futures)
    oos = full[(full.ts.dt.date >= OOS_START) & (full.ts.dt.date < OOS_END)].copy()
    candidate = apply_rule(oos)
    trades = run_one(candidate, "BANKNIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
    stats = summarize(trades, "BANKNIFTY", True, "FORWARD_JUN_01_TO_OCT_05")
    stats.update({"rule": "directional_body_fraction_0.8", "threshold": THRESHOLD,
                  "recent_7_sessions_used": False})
    missing = selection[(selection.session_date >= OOS_START) &
                        (selection.session_date < OOS_END) &
                        (selection.status != "SELECTED")]
    stats["unavailable_sessions"] = len(missing)
    pd.DataFrame([stats]).to_csv(OUT / "summary.csv", index=False)
    if not trades.empty:
        trades = trades.copy()
        trades["period"] = "FORWARD_JUN_01_TO_OCT_05"
        trades["rule"] = "directional_body_fraction_0.8"
        trades.to_csv(OUT / "trades.csv", index=False)
    print(pd.DataFrame([stats]).to_string(index=False))
    print("Frozen rule; actual selected futures; modeled Upstox fees + 1 point/side; paper only.")
    print(f"Recent seven sessions excluded; unavailable OOS sessions={len(missing)}; saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
