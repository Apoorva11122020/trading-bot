#!/usr/bin/env python3
"""Train-select prior-session range-expansion gate for pivot entries.

For each session, compare the prior session's spot range with the median of
the ten completed daily ranges preceding it. Fixed cutoffs 1.0/1.25/1.5 are
ranked on TRAIN only. Selection requires >=8 trades and better TRAIN net and
win rate than baseline; the selected rule is then evaluated once on Feb-May.
June onward and the recent seven sessions are excluded before feature creation.
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

OUT = ROOT / "reports" / "train_selected_prior_range_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
THRESHOLDS = (1.0, 1.25, 1.5)
MIN_TRAIN, MIN_VALIDATION = 8, 8
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    grid_rows, choices, validation_rows, trade_frames = [], [], [], []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        spot = spot[(spot.ts.dt.date >= START) & (spot.ts.dt.date < VALID_END)].copy()
        daily = spot.groupby("session", sort=True).agg(
            high=("high", "max"), low=("low", "min")
        )
        daily["range"] = daily.high - daily.low
        # For session D: prior range is D-1; reference median includes only
        # completed ranges through D-1, with no current-session information.
        daily["prior_range"] = daily["range"].shift(1)
        daily["prior_median10"] = daily["range"].shift(1).rolling(10, min_periods=10).median()
        daily["prior_range_ratio"] = daily.prior_range / daily.prior_median10
        ratio = daily.prior_range_ratio

        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        frame["prior_range_ratio"] = frame.session.map(ratio)
        train = frame[(frame.ts.dt.date >= START) & (frame.ts.dt.date < TRAIN_END)].copy()
        val = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
        baseline_train = summarize(run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                   symbol, True, "TRAIN_OCT_JAN")
        candidates = []
        for threshold in THRESHOLDS:
            x = train.copy()
            eligible = x.prior_range_ratio >= threshold
            x["long_signal"] = x.long_signal.astype(bool) & eligible
            x["short_signal"] = x.short_signal.astype(bool) & eligible
            stats = summarize(run_one(x, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                              symbol, True, "TRAIN_OCT_JAN")
            stats.update({"threshold_ratio": threshold, "rule": "prior_range_expansion"})
            stats["train_net_improved"] = stats["net_points"] > baseline_train["net_points"]
            stats["train_win_improved"] = stats["win_rate_pct"] > baseline_train["win_rate_pct"]
            grid_rows.append(stats)
            candidates.append(stats)
        eligible = [r for r in candidates if r["trades"] >= MIN_TRAIN and
                    r["train_net_improved"] and r["train_win_improved"]]
        train_pass = bool(eligible)
        pool = eligible if eligible else [r for r in candidates if r["trades"] >= MIN_TRAIN]
        chosen = max(pool or candidates,
                     key=lambda r: (r["net_points"], r["win_rate_pct"], -r["max_drawdown_points"]))
        choices.append({**chosen, "train_gate_pass": train_pass,
                        "baseline_train_net": baseline_train["net_points"],
                        "baseline_train_win_rate_pct": baseline_train["win_rate_pct"]})

        baseline_val = summarize(run_one(val, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                 symbol, True, "VALIDATION_FEB_MAY")
        xval = val.copy()
        regime = xval.prior_range_ratio >= float(chosen["threshold_ratio"])
        xval["long_signal"] = xval.long_signal.astype(bool) & regime
        xval["short_signal"] = xval.short_signal.astype(bool) & regime
        val_trades = run_one(xval, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        stats = summarize(val_trades, symbol, True, "VALIDATION_FEB_MAY")
        gate = bool(train_pass and stats["trades"] >= MIN_VALIDATION and
                    stats["win_rate_pct"] >= 50 and stats["net_points"] > 0)
        validation_rows.append({**stats, "threshold_ratio": chosen["threshold_ratio"],
                                "train_gate_pass": train_pass,
                                "baseline_validation_trades": baseline_val["trades"],
                                "baseline_validation_win_rate_pct": baseline_val["win_rate_pct"],
                                "baseline_validation_net_points": baseline_val["net_points"],
                                "validation_gate_pass": gate})
        if not val_trades.empty:
            val_trades = val_trades.copy()
            val_trades["period"] = "VALIDATION_FEB_MAY"
            val_trades["rule"] = f"prior_range_expansion_{chosen['threshold_ratio']}x"
            trade_frames.append(val_trades)
        missing = selection[(selection.session_date >= START) &
                            (selection.session_date < VALID_END) &
                            (selection.status != "SELECTED")]
        print(f"{symbol}: ratio>={chosen['threshold_ratio']}x; TRAIN pass={train_pass}; "
              f"validation {stats['trades']} / {stats['win_rate_pct']}% / "
              f"{stats['net_points']} pts; gate={gate}; missing={len(missing)}")

    pd.DataFrame(grid_rows).to_csv(OUT / "train_grid.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "selected_rules.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_results.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "validation_trades.csv", index=False)
    print(pd.DataFrame(validation_rows).to_string(index=False))
    print(f"Saved: {OUT}; estimated Upstox charges + 1 point/side; paper only; no orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
