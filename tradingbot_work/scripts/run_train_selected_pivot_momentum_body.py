#!/usr/bin/env python3
"""TRAIN-gated pivot confirmation: directional EMA20 slope plus candle body.

The fixed candidate requires both (a) EMA20 slope over three completed bars to
agree with the pivot trade direction and (b) signal-candle body/range >= 0.6,
with candle color aligned to the trade. It is evaluated on Feb-May only when
Oct-Jan TRAIN improves both net points and win rate with >=8 trades. June onward
is excluded before signal construction. PAPER research only; no orders.
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

OUT = ROOT / "reports" / "train_selected_pivot_momentum_body_20261014"
TRAIN_START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
MIN_TRAIN_TRADES = 8
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0
BODY_THRESHOLD = 0.6


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        # Hard truncate before signal construction: no June+ feature/signal rows.
        spot = spot[(spot.ts.dt.date >= TRAIN_START) & (spot.ts.dt.date < VALID_END)].copy()
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        frame = frame[(frame.ts.dt.date >= TRAIN_START) & (frame.ts.dt.date < VALID_END)].copy()
        slope_ref = frame.groupby("session", sort=False)["ema20_spot"].shift(3)
        span = (frame.high_spot - frame.low_spot).replace(0, float("nan"))
        body = (frame.close_spot - frame.open_spot).abs() / span
        candidate = frame.copy()
        candidate["long_signal"] = (
            frame.long_signal.astype(bool)
            & frame.ema20_spot.gt(slope_ref)
            & frame.close_spot.gt(frame.open_spot)
            & body.ge(BODY_THRESHOLD)
        )
        candidate["short_signal"] = (
            frame.short_signal.astype(bool)
            & frame.ema20_spot.lt(slope_ref)
            & frame.close_spot.lt(frame.open_spot)
            & body.ge(BODY_THRESHOLD)
        )
        train = frame[frame.ts.dt.date < TRAIN_END].copy()
        train_candidate = candidate[candidate.ts.dt.date < TRAIN_END].copy()
        val = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
        val_candidate = candidate[(candidate.ts.dt.date >= TRAIN_END) & (candidate.ts.dt.date < VALID_END)].copy()

        base_train = summarize(run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), symbol, True, "TRAIN_OCT_JAN")
        cand_train = summarize(run_one(train_candidate, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), symbol, True, "TRAIN_OCT_JAN")
        train_gate = bool(
            cand_train["trades"] >= MIN_TRAIN_TRADES
            and cand_train["win_rate_pct"] > base_train["win_rate_pct"]
            and cand_train["net_points"] > base_train["net_points"]
        )
        row = {
            "symbol": symbol,
            "rule": "pivot + directional_ema20_slope_3 + directional_body_fraction_0.6",
            "train_baseline_trades": base_train["trades"],
            "train_baseline_win_rate_pct": base_train["win_rate_pct"],
            "train_baseline_net_points": base_train["net_points"],
            "train_candidate_trades": cand_train["trades"],
            "train_candidate_win_rate_pct": cand_train["win_rate_pct"],
            "train_candidate_net_points": cand_train["net_points"],
            "train_gate_pass": train_gate,
            "validation_evaluated": train_gate,
            "recent_seven_sessions_used": False,
        }
        if train_gate:
            val_trades = run_one(val_candidate, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
            val_stats = summarize(val_trades, symbol, True, "VALIDATION_FEB_MAY")
            row.update({
                "validation_trades": val_stats["trades"],
                "validation_win_rate_pct": val_stats["win_rate_pct"],
                "validation_net_points": val_stats["net_points"],
                "validation_drawdown_points": val_stats["max_drawdown_points"],
                "validation_gate_pass": bool(val_stats["trades"] >= 8 and val_stats["win_rate_pct"] >= 50 and val_stats["net_points"] > 0),
            })
        else:
            row.update({"validation_trades": "NOT_RUN_TRAIN_GATE_FAILED", "validation_win_rate_pct": "", "validation_net_points": "", "validation_drawdown_points": "", "validation_gate_pass": False})
        missing = selection[(selection.session_date >= TRAIN_START) & (selection.session_date < VALID_END) & (selection.status != "SELECTED")]
        row["unavailable_sessions"] = len(missing)
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results.csv", index=False)
    print("PAPER ONLY | actual selected futures | modeled Upstox charges + 1 point/side | volume filter ON")
    print(result.to_string(index=False))
    print(f"Saved: {OUT / 'results.csv'}; Jun 1 onward and recent seven sessions excluded; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
