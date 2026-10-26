#!/usr/bin/env python3
"""Train-select prior-session close-location cutoff, then validate once.

Fixed development grid: mirrored long/short extremes at 55/45, 65/35, 75/25,
and 85/15 percent of the prior session's spot range. Selection uses Oct-Jan
only; a selected cutoff must improve both TRAIN net and win rate vs baseline
and have >=8 trades. Only that cutoff is evaluated Feb-May. June onward and
the recent seven sessions are excluded before feature construction.
"""
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

OUT = ROOT / "reports" / "train_selected_close_location_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
THRESHOLDS = (0.55, 0.65, 0.75, 0.85)
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1
MIN_TRAIN, MIN_VALIDATION = 8, 8


def filter_rule(frame: pd.DataFrame, loc: pd.Series, threshold: float) -> pd.DataFrame:
    x = frame.copy()
    x["prior_close_loc"] = x.session.map(loc)
    x["long_signal"] = x.long_signal.astype(bool) & (x.prior_close_loc >= threshold)
    x["short_signal"] = x.short_signal.astype(bool) & (x.prior_close_loc <= 1 - threshold)
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    grid_rows, selected_rows, validation_rows, trade_frames = [], [], [], []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        spot = spot[(spot.ts.dt.date >= START) & (spot.ts.dt.date < VALID_END)].copy()
        daily = spot.groupby("session", sort=True).agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last")
        )
        rng = (daily.high - daily.low).where(lambda s: s > 0)
        loc = ((daily.close - daily.low) / rng).shift(1)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        train = frame[(frame.ts.dt.date >= START) & (frame.ts.dt.date < TRAIN_END)].copy()
        val = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
        base_train = summarize(run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                               symbol, True, "TRAIN_OCT_JAN")
        candidates = []
        for threshold in THRESHOLDS:
            data = filter_rule(train, loc, threshold)
            stats = summarize(run_one(data, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                              symbol, True, "TRAIN_OCT_JAN")
            stats.update({"threshold": threshold, "rule": "prior_close_location_extremes"})
            stats["train_net_improved"] = stats["net_points"] > base_train["net_points"]
            stats["train_win_improved"] = stats["win_rate_pct"] > base_train["win_rate_pct"]
            grid_rows.append(stats)
            candidates.append(stats)
        eligible = [r for r in candidates if r["trades"] >= MIN_TRAIN and
                    r["train_net_improved"] and r["train_win_improved"]]
        train_pass = bool(eligible)
        pool = eligible if eligible else [r for r in candidates if r["trades"] >= MIN_TRAIN]
        chosen = max(pool or candidates,
                     key=lambda r: (r["net_points"], r["win_rate_pct"], -r["max_drawdown_points"]))
        selected_rows.append({**chosen, "train_gate_pass": train_pass,
                              "baseline_train_net": base_train["net_points"],
                              "baseline_train_win_rate_pct": base_train["win_rate_pct"]})

        base_val = summarize(run_one(val, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                             symbol, True, "VALIDATION_FEB_MAY")
        val_data = filter_rule(val, loc, float(chosen["threshold"]))
        val_trades = run_one(val_data, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        val_stats = summarize(val_trades, symbol, True, "VALIDATION_FEB_MAY")
        val_gate = bool(train_pass and val_stats["trades"] >= MIN_VALIDATION and
                        val_stats["win_rate_pct"] >= 50 and val_stats["net_points"] > 0)
        validation_rows.append({**val_stats, "threshold": chosen["threshold"],
                                "train_gate_pass": train_pass,
                                "baseline_validation_trades": base_val["trades"],
                                "baseline_validation_win_rate_pct": base_val["win_rate_pct"],
                                "baseline_validation_net_points": base_val["net_points"],
                                "validation_gate_pass": val_gate})
        if not val_trades.empty:
            val_trades = val_trades.copy()
            val_trades["rule"] = f"prior_close_location_{chosen['threshold']:.2f}"
            val_trades["period"] = "VALIDATION_FEB_MAY"
            trade_frames.append(val_trades)
        missing = selection[(selection.session_date >= START) &
                            (selection.session_date < VALID_END) &
                            (selection.status != "SELECTED")]
        print(f"{symbol}: cutoff={chosen['threshold']:.2f}; train gate={train_pass}; "
              f"validation {val_stats['trades']} / {val_stats['win_rate_pct']}% / "
              f"{val_stats['net_points']} pts; gate={val_gate}; missing={len(missing)}")

    pd.DataFrame(grid_rows).to_csv(OUT / "train_grid.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_rules.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_results.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "validation_trades.csv", index=False)
    print(pd.DataFrame(validation_rows).to_string(index=False))
    print(f"Saved: {OUT}; estimated Upstox charges + 1 point/side; paper only; no orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
