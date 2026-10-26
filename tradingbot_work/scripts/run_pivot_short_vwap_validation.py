#!/usr/bin/env python3
"""TRAIN-selected short-only pivot plus causal futures-VWAP filter; Oct-May only."""
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

OUT = ROOT / "reports" / "pivot_short_vwap_validation_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def add_session_vwap(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    typical = (x.high_fut + x.low_fut + x.close_fut) / 3.0
    pv = typical * x.volume_fut
    cumulative_pv = pv.groupby(x.session, sort=False).cumsum()
    cumulative_volume = x.volume_fut.groupby(x.session, sort=False).cumsum()
    x["session_vwap_fut"] = cumulative_pv / cumulative_volume.where(cumulative_volume > 0)
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
        frame = add_session_vwap(make_signals(spot, futures))
        frame = frame[(frame.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                      (frame.ts.dt.date < DEV_END)].copy()
        vwap_3bars_ago = frame.groupby("session", sort=False).session_vwap_fut.shift(3)
        short_only = frame.copy()
        short_only["long_signal"] = False
        vwap_confirmed = short_only.copy()
        vwap_confirmed["short_signal"] &= vwap_confirmed.close_fut < vwap_confirmed.session_vwap_fut
        falling_vwap = short_only.copy()
        falling_vwap["short_signal"] &= frame.session_vwap_fut < vwap_3bars_ago
        variants = (
            ("baseline", frame),
            ("train_selected_short_only", short_only),
            ("short_only_close_below_futures_session_vwap", vwap_confirmed),
            ("short_only_futures_vwap_falling_3bars", falling_vwap),
        )
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
    print("Train Oct-Jan; validation Feb-May; June onward and recent seven sessions excluded before simulation.")
    print("Short-only; completed futures-candle VWAP; actual futures fills; estimated Upstox costs + 1 point/side; RR 1.5 / stop 0.1 ATR.")
    print("Paper/research only; no broker orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
