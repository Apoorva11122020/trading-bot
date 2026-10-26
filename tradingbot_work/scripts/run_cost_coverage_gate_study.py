#!/usr/bin/env python3
"""Research a fixed economic viability gate on pivot entries, Oct-May only.

Keep a signal only when its next-bar 1.5R gross target is at least 2x the
estimated round-trip charges plus 1 point/side slippage. The rule is fixed
before evaluation and uses only information available at signal close and the
next-bar entry. TRAIN Oct-Jan and validation Feb-May are simulated separately.
June onward and the recent seven sessions are excluded before simulation.
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
    run_one, summarize, upstox_futures_fees_inr,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "cost_coverage_gate_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
RR, ATR_BUFFER, SLIPPAGE, COST_MULTIPLE = 1.5, 0.1, 1.0, 2.0


def add_cost_gate(data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy().sort_values("ts").reset_index(drop=True)
    nxt = x.groupby("session", sort=False).shift(-1)
    # Grouping by session means shifted rows already belong to the same session.
    valid_next = (nxt.ts.notna() & nxt.instrument_key.eq(x.instrument_key)
                  & ((nxt.ts - x.ts) == pd.Timedelta(minutes=5)))
    keep_long = np.zeros(len(x), dtype=bool)
    keep_short = np.zeros(len(x), dtype=bool)
    for i, row in x.iterrows():
        if not valid_next.iloc[i] or not pd.notna(row.atr14_fut) or row.atr14_fut <= 0:
            continue
        entry = float(nxt.open_fut.iloc[i])
        lot = int(row.lot_size)
        for side, signal, stop in (
            ("LONG", bool(row.long_signal), float(row.low_fut - row.atr14_fut * ATR_BUFFER)),
            ("SHORT", bool(row.short_signal), float(row.high_fut + row.atr14_fut * ATR_BUFFER)),
        ):
            if not signal:
                continue
            risk = entry - stop if side == "LONG" else stop - entry
            if not np.isfinite(risk) or risk <= 0:
                continue
            reward = risk * RR
            hypothetical_exit = entry + reward if side == "LONG" else entry - reward
            fees_inr = upstox_futures_fees_inr(
                entry, hypothetical_exit, side, lot, row.ts.date(), nxt.ts.iloc[i].date()
            )
            roundtrip_cost_points = 2 * SLIPPAGE + fees_inr / lot
            if reward >= COST_MULTIPLE * roundtrip_cost_points:
                if side == "LONG":
                    keep_long[i] = True
                else:
                    keep_short[i] = True
    x["long_signal"] = keep_long
    x["short_signal"] = keep_short
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, trade_frames = [], []
    for symbol, spot_name in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        merged = make_signals(spot, futures)
        # Cut future data before making the filtered strategy input.
        dev = merged[(merged.ts.dt.date >= START) & (merged.ts.dt.date < VALID_END)].copy()
        gate_data = add_cost_gate(dev)
        missing = selection[(selection.session_date >= START) &
                            (selection.session_date < VALID_END) &
                            (selection.status != "SELECTED")]
        for label, begin, end in (
            ("TRAIN_OCT_JAN", START, TRAIN_END),
            ("VALIDATION_FEB_MAY", TRAIN_END, VALID_END),
        ):
            period = gate_data[(gate_data.ts.dt.date >= begin) & (gate_data.ts.dt.date < end)].copy()
            for rule, source in (("baseline", dev[(dev.ts.dt.date >= begin) & (dev.ts.dt.date < end)]),
                                 ("gross_1.5R_at_least_2x_cost", period)):
                trades = run_one(source, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
                stat = summarize(trades, symbol, True, label)
                stat.update({"rule": rule, "estimated_data_unavailable_sessions": len(missing)})
                summaries.append(stat)
                if not trades.empty:
                    trades = trades.copy()
                    trades["rule"], trades["period"] = rule, label
                    trade_frames.append(trades)
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(result.to_string(index=False))
    print(f"Saved: {OUT}; cost gate = 1.5R gross >= {COST_MULTIPLE}x modeled round-trip cost.")
    print("Estimated Upstox fees + 1 point/side slippage; paper only; no orders; June onward excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
