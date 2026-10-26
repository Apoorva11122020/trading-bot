#!/usr/bin/env python3
"""Choose range-expansion vs contraction context on TRAIN, then validate.

Fixed causal features use prior session range / median of its preceding ten
completed ranges. The two hypotheses are range expansion >=1.25x or contraction
<=0.80x; both preserve the pivot signal directions. Per symbol, TRAIN selects
only a hypothesis that improves net and win rate over baseline with >=8 trades.
Only that hypothesis is checked on Feb-May validation. June onward and the
recent seven sessions are excluded before feature construction.
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

OUT = ROOT / "reports" / "train_selected_range_regime_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
REGIMES = (("EXPANSION", lambda r: r >= 1.25), ("CONTRACTION", lambda r: r <= 0.80))
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1
MIN_TRAIN = MIN_VALIDATION = 8


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
        daily = spot.groupby("session", sort=True).agg(high=("high", "max"), low=("low", "min"))
        daily["range"] = daily.high - daily.low
        daily["prior_range"] = daily["range"].shift(1)
        daily["prior_median10"] = daily["range"].shift(1).rolling(10, min_periods=10).median()
        ratio = daily.prior_range / daily.prior_median10
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        frame["prior_range_ratio"] = frame.session.map(ratio)
        train = frame[(frame.ts.dt.date >= START) & (frame.ts.dt.date < TRAIN_END)].copy()
        val = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
        baseline_train = summarize(run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                   symbol, True, "TRAIN_OCT_JAN")
        candidates = []
        for label, predicate in REGIMES:
            x = train.copy()
            keep = predicate(x.prior_range_ratio)
            x["long_signal"] = x.long_signal.astype(bool) & keep
            x["short_signal"] = x.short_signal.astype(bool) & keep
            stats = summarize(run_one(x, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                              symbol, True, "TRAIN_OCT_JAN")
            stats.update({"regime": label, "rule": "pivot_in_prior_range_regime"})
            stats["train_net_improved"] = stats["net_points"] > baseline_train["net_points"]
            stats["train_win_improved"] = stats["win_rate_pct"] > baseline_train["win_rate_pct"]
            grid.append(stats)
            candidates.append((stats, predicate))
        eligible = [(s, p) for s, p in candidates if s["trades"] >= MIN_TRAIN and
                    s["train_net_improved"] and s["train_win_improved"]]
        train_pass = bool(eligible)
        pool = eligible if eligible else [(s, p) for s, p in candidates if s["trades"] >= MIN_TRAIN]
        chosen, predicate = max(pool or candidates,
            key=lambda pair: (pair[0]["net_points"], pair[0]["win_rate_pct"],
                              -pair[0]["max_drawdown_points"]))
        choices.append({**chosen, "train_gate_pass": train_pass,
                        "baseline_train_net": baseline_train["net_points"],
                        "baseline_train_win_rate_pct": baseline_train["win_rate_pct"]})
        base_val = summarize(run_one(val, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                             symbol, True, "VALIDATION_FEB_MAY")
        xval = val.copy()
        keep = predicate(xval.prior_range_ratio)
        xval["long_signal"] = xval.long_signal.astype(bool) & keep
        xval["short_signal"] = xval.short_signal.astype(bool) & keep
        val_trades = run_one(xval, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        val_stats = summarize(val_trades, symbol, True, "VALIDATION_FEB_MAY")
        gate = bool(train_pass and val_stats["trades"] >= MIN_VALIDATION and
                    val_stats["win_rate_pct"] >= 50 and val_stats["net_points"] > 0)
        validation_rows.append({**val_stats, "regime": chosen["regime"],
                                "train_gate_pass": train_pass,
                                "baseline_validation_trades": base_val["trades"],
                                "baseline_validation_win_rate_pct": base_val["win_rate_pct"],
                                "baseline_validation_net_points": base_val["net_points"],
                                "validation_gate_pass": gate})
        if not val_trades.empty:
            val_trades = val_trades.copy()
            val_trades["period"] = "VALIDATION_FEB_MAY"
            val_trades["rule"] = f"pivot_prior_range_{chosen['regime'].lower()}"
            trade_frames.append(val_trades)
        missing = selection[(selection.session_date >= START) &
                            (selection.session_date < VALID_END) &
                            (selection.status != "SELECTED")]
        print(f"{symbol}: selected {chosen['regime']}; train pass={train_pass}; "
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
