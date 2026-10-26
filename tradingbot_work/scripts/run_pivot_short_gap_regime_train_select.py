#!/usr/bin/env python3
"""Select gap direction on TRAIN only, then validate the frozen short-only rule."""
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

OUT = ROOT / "reports" / "pivot_short_gap_regime_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
DEV_END = pd.Timestamp("2026-06-01").date()


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    )
    summaries, trades_out, selection_rows = [], [], []
    for symbol, spot_file in settings:
        spot = load_spot(DEFAULT_SPOT / spot_file)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        daily = spot.groupby("session", sort=True).agg(day_open=("open", "first"), day_close=("close", "last"))
        gap = daily.day_open - daily.day_close.shift(1)
        frame = make_signals(spot, futures)
        frame = frame[(frame.ts.dt.date >= pd.Timestamp("2025-10-01").date()) &
                      (frame.ts.dt.date < DEV_END)].copy()
        short_only = frame.copy()
        short_only["long_signal"] = False
        short_train = short_only[short_only.ts.dt.date < TRAIN_END].copy()
        candidates = {}
        for name, condition in (("GAP_DOWN", gap < 0), ("GAP_UP", gap > 0)):
            filtered = short_train.copy()
            filtered["short_signal"] &= filtered.session.map(condition).fillna(False).astype(bool)
            train_trades = run_one(filtered, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
            stats = summarize(train_trades, symbol, True, "TRAIN_OCT_JAN")
            selection_rows.append({"symbol": symbol, "gap_direction": name,
                                   "train_trades": stats["trades"], "train_net_points": stats["net_points"],
                                   "train_win_rate_pct": stats["win_rate_pct"],
                                   "train_avg_points": stats["avg_trade_points"]})
            candidates[name] = (int(stats["trades"]), float(stats["avg_trade_points"]), filtered)
        # Freeze the gap direction with highest TRAIN mean net/trade; no validation data used.
        eligible = {name: values for name, values in candidates.items() if values[0] >= 5}
        selected_name = max(eligible, key=lambda key: eligible[key][1]) if eligible else "NONE"
        full_selected = short_only.copy()
        if selected_name != "NONE":
            full_selected["short_signal"] &= full_selected.session.map(
                gap.lt(0) if selected_name == "GAP_DOWN" else gap.gt(0)
            ).fillna(False).astype(bool)
        rule = f"short_only_train_selected_{selected_name.lower()}"
        for label, data in (("baseline", frame), ("short_only", short_only), (rule, full_selected)):
            trades = run_one(data, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
            if not trades.empty:
                entries = pd.to_datetime(trades.entry_ts).dt.date
                exits = pd.to_datetime(trades.exit_ts).dt.date
                trades["period"] = entries.map(lambda d: "TRAIN_OCT_JAN" if d < TRAIN_END else "VALIDATION_FEB_MAY")
                trades.loc[(entries < TRAIN_END) & (exits >= TRAIN_END), "period"] = "BOUNDARY_EXCLUDED"
                trades["rule"] = label
                trades_out.append(trades)
            for period in ("TRAIN_OCT_JAN", "VALIDATION_FEB_MAY"):
                if trades.empty:
                    part = trades
                else:
                    part = trades[(trades.period == period) &
                                  (pd.to_datetime(trades.entry_ts).dt.date == pd.to_datetime(trades.exit_ts).dt.date)]
                st = summarize(part, symbol, True, period)
                st["rule"] = label
                st["train_selected_gap_direction"] = selected_name if label == rule else ""
                summaries.append(st)
    summary = pd.DataFrame(summaries)
    selection = pd.DataFrame(selection_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    selection.to_csv(OUT / "train_gap_selection.csv", index=False)
    if trades_out:
        pd.concat(trades_out, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print("TRAIN-ONLY GAP DIRECTION SELECTION")
    print(selection.to_string(index=False))
    print("FROZEN-CANDIDATE CHRONOLOGICAL RESULTS")
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Validation Feb-May tested only after selecting the gap direction by TRAIN mean net/trade.")
    print("June onward/recent seven-session data excluded; actual futures, estimated Upstox charges + 1 point/side.")
    print("Paper/research only; no orders submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
