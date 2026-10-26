#!/usr/bin/env python3
"""Test a causal prior-session direction filter on Oct-May only; paper research."""
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

OUT = ROOT / "reports" / "pivot_prior_session_trend_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


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
        daily_close = spot.groupby("session", sort=True).close.last()
        prior_session_move = daily_close.diff().shift(1)
        frame = make_signals(spot, futures)
        frame = frame[frame.ts.dt.date < DEV_END].copy()
        frame["prior_session_move"] = frame.session.map(prior_session_move)
        candidate = frame.copy()
        candidate["long_signal"] = candidate.long_signal & candidate.prior_session_move.gt(0)
        candidate["short_signal"] = candidate.short_signal & candidate.prior_session_move.lt(0)
        for rule, data in (("baseline", frame), ("prior_session_direction_alignment", candidate)):
            trades = run_one(data, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
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
    print("TRAIN Oct-Jan; VALIDATION Feb-May. June onward and recent seven sessions excluded before simulation.")
    print("Actual selected futures; Upstox estimated statutory charges + 1 point/side; volume ON; RR 1.5; stop buffer 0.1 ATR.")
    print("PAPER/RESEARCH ONLY; no broker orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
