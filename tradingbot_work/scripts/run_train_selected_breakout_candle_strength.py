#!/usr/bin/env python3
"""Train-select strong signal-candle confirmation for pivot crosses.

Fixed candidates require a signal candle aligned with trade direction and a
spot candle body/range fraction >=0.4, 0.6, or 0.8. TRAIN chooses a threshold
only if it improves both net and win rate vs baseline with >=8 trades. The
chosen threshold is validated once on Feb-May. June onward/recent-seven are
excluded before feature construction. Paper only; no orders.
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

OUT = ROOT / "reports" / "train_selected_candle_strength_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
THRESHOLDS = (0.4, 0.6, 0.8)
MIN_TRAIN = MIN_VALIDATION = 8
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1


def add_strength(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    x = frame.copy()
    spread = (x.high_spot - x.low_spot).replace(0, np.nan)
    body_fraction = (x.close_spot - x.open_spot).abs() / spread
    bullish = x.close_spot > x.open_spot
    bearish = x.close_spot < x.open_spot
    x["body_fraction"] = body_fraction
    x["long_signal"] = x.long_signal.astype(bool) & bullish & (body_fraction >= threshold)
    x["short_signal"] = x.short_signal.astype(bool) & bearish & (body_fraction >= threshold)
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    grid, choices, validation_rows, trade_frames = [], [], [], []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        spot = spot[(spot.ts.dt.date >= START) & (spot.ts.dt.date < VALID_END)].copy()
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        train = frame[(frame.ts.dt.date >= START) & (frame.ts.dt.date < TRAIN_END)].copy()
        val = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
        base_train = summarize(run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                               symbol, True, "TRAIN_OCT_JAN")
        candidate_rows = []
        for threshold in THRESHOLDS:
            data = add_strength(train, threshold)
            stats = summarize(run_one(data, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                              symbol, True, "TRAIN_OCT_JAN")
            stats.update({"threshold": threshold, "rule": "directional_body_fraction"})
            stats["train_net_improved"] = stats["net_points"] > base_train["net_points"]
            stats["train_win_improved"] = stats["win_rate_pct"] > base_train["win_rate_pct"]
            grid.append(stats)
            candidate_rows.append(stats)
        eligible = [r for r in candidate_rows if r["trades"] >= MIN_TRAIN and
                    r["train_net_improved"] and r["train_win_improved"]]
        train_pass = bool(eligible)
        pool = eligible if eligible else [r for r in candidate_rows if r["trades"] >= MIN_TRAIN]
        chosen = max(pool or candidate_rows,
                     key=lambda r: (r["net_points"], r["win_rate_pct"], -r["max_drawdown_points"]))
        choices.append({**chosen, "train_gate_pass": train_pass,
                        "baseline_train_net": base_train["net_points"],
                        "baseline_train_win_rate_pct": base_train["win_rate_pct"]})
        base_val = summarize(run_one(val, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                             symbol, True, "VALIDATION_FEB_MAY")
        val_data = add_strength(val, float(chosen["threshold"]))
        val_trades = run_one(val_data, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        val_stats = summarize(val_trades, symbol, True, "VALIDATION_FEB_MAY")
        gate = bool(train_pass and val_stats["trades"] >= MIN_VALIDATION and
                    val_stats["win_rate_pct"] >= 50 and val_stats["net_points"] > 0)
        validation_rows.append({**val_stats, "threshold": chosen["threshold"],
                                "train_gate_pass": train_pass,
                                "baseline_validation_trades": base_val["trades"],
                                "baseline_validation_win_rate_pct": base_val["win_rate_pct"],
                                "baseline_validation_net_points": base_val["net_points"],
                                "validation_gate_pass": gate})
        if not val_trades.empty:
            val_trades = val_trades.copy()
            val_trades["period"] = "VALIDATION_FEB_MAY"
            val_trades["rule"] = f"directional_body_fraction_{chosen['threshold']}"
            trade_frames.append(val_trades)
        missing = selection[(selection.session_date >= START) &
                            (selection.session_date < VALID_END) &
                            (selection.status != "SELECTED")]
        print(f"{symbol}: threshold={chosen['threshold']}; train pass={train_pass}; "
              f"validation={val_stats['trades']} / {val_stats['win_rate_pct']}% / "
              f"{val_stats['net_points']} pts; gate={gate}; missing={len(missing)}")

    pd.DataFrame(grid).to_csv(OUT / "train_grid.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "selected_rules.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_results.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "validation_trades.csv", index=False)
    print(pd.DataFrame(validation_rows).to_string(index=False))
    print(f"Saved: {OUT}; estimated Upstox charges + 1 point/side; paper only; no orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
