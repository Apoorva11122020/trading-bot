#!/usr/bin/env python3
"""TRAIN-gated NIFTY transfer check for BANKNIFTY LONG body>=0.8 rule.

This is one fixed cross-symbol hypothesis, not a threshold search. Extended
NIFTY TRAIN (Jan 2025-Jan 2026) must improve win rate and net over baseline
with >=20 trades before Feb-May validation is run. June+ is excluded. PAPER
only; no orders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import make_signals, run_one, summarize
from scripts.run_extended_nifty_train_selection import load_early_futures, load_early_spot
from scripts.run_train_side_specific_body_filter import load_nifty
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "nifty_banknifty_rule_transfer_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0
BODY_MIN = 0.8


def apply_rule(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    body = (x.close_spot - x.open_spot).abs() / (x.high_spot - x.low_spot).replace(0, np.nan)
    x["long_signal"] = (x.long_signal.astype(bool) & x.close_spot.gt(x.open_spot) & body.ge(BODY_MIN))
    x["short_signal"] = False
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    spot, futures, _ = load_nifty()
    frame = make_signals(spot, futures)
    frame = frame[frame.ts.dt.date < VALID_END].copy()
    train = frame[frame.ts.dt.date < TRAIN_END].copy()
    validation = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
    baseline_train = summarize(run_one(train, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                               "NIFTY", True, "TRAIN_JAN25_JAN26")
    candidate_train = summarize(run_one(apply_rule(train), "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                "NIFTY", True, "TRAIN_JAN25_JAN26")
    passed = bool(candidate_train["trades"] >= 20
                  and candidate_train["win_rate_pct"] > baseline_train["win_rate_pct"]
                  and candidate_train["net_points"] > baseline_train["net_points"])
    row = {"symbol": "NIFTY", "rule": "long_only_bullish_body_fraction_ge_0.8",
           "baseline_train_trades": baseline_train["trades"], "baseline_train_win_rate_pct": baseline_train["win_rate_pct"],
           "baseline_train_net_points": baseline_train["net_points"], "candidate_train_trades": candidate_train["trades"],
           "candidate_train_win_rate_pct": candidate_train["win_rate_pct"], "candidate_train_net_points": candidate_train["net_points"],
           "train_gate_pass": passed, "recent_seven_sessions_used": False}
    if passed:
        baseline_val = summarize(run_one(validation, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                 "NIFTY", True, "VALIDATION_FEB_MAY")
        candidate_val = summarize(run_one(apply_rule(validation), "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                                  "NIFTY", True, "VALIDATION_FEB_MAY")
        row.update({"baseline_validation_trades": baseline_val["trades"],
                    "baseline_validation_win_rate_pct": baseline_val["win_rate_pct"],
                    "baseline_validation_net_points": baseline_val["net_points"],
                    "validation_trades": candidate_val["trades"], "validation_win_rate_pct": candidate_val["win_rate_pct"],
                    "validation_net_points": candidate_val["net_points"],
                    "validation_gate_pass": bool(candidate_val["trades"] >= 8 and candidate_val["win_rate_pct"] >= 50
                                                 and candidate_val["net_points"] > 0)})
    else:
        row.update({"validation_trades": "NOT_RUN_TRAIN_GATE_FAILED", "validation_win_rate_pct": "",
                    "validation_net_points": "", "validation_gate_pass": False})
    result = pd.DataFrame([row])
    result.to_csv(OUT / "results.csv", index=False)
    print("FIXED BANKNIFTY RULE TRANSFER TO NIFTY | PAPER ONLY | realistic modeled futures costs")
    print(result.to_string(index=False))
    print(f"Saved: {OUT}; Feb-May only after TRAIN gate; June+ excluded; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
