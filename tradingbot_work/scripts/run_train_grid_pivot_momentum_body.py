#!/usr/bin/env python3
"""Small TRAIN-only grid for pivot momentum/body confirmation; PAPER research only.

Predeclared grid: directional signal candle body/range >=0.6 or 0.8 and EMA20
slope lookback 3 or 6 completed 5m bars. A per-symbol rule is selected only if
TRAIN (Oct-Jan) has >=8 trades and improves both net points and win rate over
baseline. Then it is evaluated once on Feb-May; June onward is hard-excluded.
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

OUT = ROOT / "reports" / "train_grid_pivot_momentum_body_20261014"
TRAIN_START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
GRID = ((0.6, 3), (0.6, 6), (0.8, 3), (0.8, 6))
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0


def gated(frame: pd.DataFrame, threshold: float, lag: int) -> pd.DataFrame:
    x = frame.copy()
    slope_ref = x.groupby("session", sort=False)["ema20_spot"].shift(lag)
    span = (x.high_spot - x.low_spot).replace(0, float("nan"))
    body = (x.close_spot - x.open_spot).abs() / span
    x["long_signal"] = (x.long_signal.astype(bool) & x.ema20_spot.gt(slope_ref)
                         & x.close_spot.gt(x.open_spot) & body.ge(threshold))
    x["short_signal"] = (x.short_signal.astype(bool) & x.ema20_spot.lt(slope_ref)
                          & x.close_spot.lt(x.open_spot) & body.ge(threshold))
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        spot = spot[(spot.ts.dt.date >= TRAIN_START) & (spot.ts.dt.date < VALID_END)].copy()
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        frame = frame[(frame.ts.dt.date >= TRAIN_START) & (frame.ts.dt.date < VALID_END)].copy()
        train = frame[frame.ts.dt.date < TRAIN_END].copy()
        validation = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
        base_train_trades = run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        base_train = summarize(base_train_trades, symbol, True, "TRAIN_OCT_JAN")
        base_validation = summarize(run_one(validation, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                    symbol, True, "VALIDATION_FEB_MAY")
        candidates = []
        for threshold, lag in GRID:
            rule_train = gated(train, threshold, lag)
            trades = run_one(rule_train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
            stats = summarize(trades, symbol, True, "TRAIN_OCT_JAN")
            eligible = (stats["trades"] >= 8
                        and stats["win_rate_pct"] > base_train["win_rate_pct"]
                        and stats["net_points"] > base_train["net_points"])
            candidates.append({**stats, "threshold": threshold, "slope_lag": lag,
                               "train_gate_pass": eligible})
        eligible = [c for c in candidates if c["train_gate_pass"]]
        chosen = max(eligible, key=lambda c: (c["net_points"], c["win_rate_pct"], -c["max_drawdown_points"])) if eligible else None
        row = {
            "symbol": symbol,
            "baseline_train_trades": base_train["trades"],
            "baseline_train_wins_pct": base_train["win_rate_pct"],
            "baseline_train_net_points": base_train["net_points"],
            "baseline_validation_trades": base_validation["trades"],
            "baseline_validation_wins_pct": base_validation["win_rate_pct"],
            "baseline_validation_net_points": base_validation["net_points"],
            "train_candidate_count": len(candidates),
            "train_pass_count": len(eligible),
            "recent_seven_sessions_used": False,
        }
        if chosen is not None:
            val_df = gated(validation, chosen["threshold"], chosen["slope_lag"])
            val_stats = summarize(run_one(val_df, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                  symbol, True, "VALIDATION_FEB_MAY")
            row.update({
                "selected_threshold": chosen["threshold"],
                "selected_slope_lag": chosen["slope_lag"],
                "selected_train_trades": chosen["trades"],
                "selected_train_win_rate_pct": chosen["win_rate_pct"],
                "selected_train_net_points": chosen["net_points"],
                "validation_trades": val_stats["trades"],
                "validation_win_rate_pct": val_stats["win_rate_pct"],
                "validation_net_points": val_stats["net_points"],
                "validation_drawdown_points": val_stats["max_drawdown_points"],
                "validation_gate_pass": bool(val_stats["trades"] >= 8 and val_stats["win_rate_pct"] >= 50
                                              and val_stats["net_points"] > 0),
            })
        else:
            row.update({"selected_threshold": "", "selected_slope_lag": "", "selected_train_trades": "",
                        "selected_train_win_rate_pct": "", "selected_train_net_points": "",
                        "validation_trades": "NOT_RUN_TRAIN_GATE_FAILED", "validation_win_rate_pct": "",
                        "validation_net_points": "", "validation_drawdown_points": "", "validation_gate_pass": False})
        missing = selection[(selection.session_date >= TRAIN_START) & (selection.session_date < VALID_END)
                            & (selection.status != "SELECTED")]
        row["unavailable_sessions"] = len(missing)
        summaries.append(row)
    result = pd.DataFrame(summaries)
    result.to_csv(OUT / "results.csv", index=False)
    print("PAPER ONLY | selected futures | volume filter ON | modeled Upstox charges + 1 point/side")
    print(result.to_string(index=False))
    print(f"Saved: {OUT / 'results.csv'}; June onward and recent seven sessions excluded; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
