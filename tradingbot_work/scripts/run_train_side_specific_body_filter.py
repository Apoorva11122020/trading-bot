#!/usr/bin/env python3
"""TRAIN-selected side-specific candle-body confirmation, PAPER research only.

Fixed 3x3 grid: LONG and SHORT independently use no extra body filter, or a
same-color body/range minimum of 0.6 or 0.8. Select from TRAIN only when >=20
trades and both win rate and net points exceed baseline. Then one Feb-May
validation; no June+ or recent-seven rows are passed to signal construction.
NIFTY TRAIN is extended to Jan 2025 using validated selected futures; BANKNIFTY
uses its available Oct 2025-Jan 2026 TRAIN because older futures are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path
from itertools import product
from datetime import date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, load_front_month, load_spot, make_signals,
    run_one, summarize,
)
from scripts.run_extended_nifty_train_selection import load_early_futures, load_early_spot
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "train_side_specific_body_filter_20261014"
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
NIFTY_START = pd.Timestamp("2025-01-01").date()
BANK_START = pd.Timestamp("2025-10-01").date()
OPTIONS = (None, 0.6, 0.8)
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0
MIN_TRAIN = 20


def body_filter(frame: pd.DataFrame, long_min: float | None, short_min: float | None) -> pd.DataFrame:
    x = frame.copy()
    span = (x.high_spot - x.low_spot).replace(0, np.nan)
    body = (x.close_spot - x.open_spot).abs() / span
    long_ok = pd.Series(True, index=x.index) if long_min is None else x.close_spot.gt(x.open_spot) & body.ge(long_min)
    short_ok = pd.Series(True, index=x.index) if short_min is None else x.close_spot.lt(x.open_spot) & body.ge(short_min)
    x["long_signal"] = x.long_signal.astype(bool) & long_ok
    x["short_signal"] = x.short_signal.astype(bool) & short_ok
    return x


def load_nifty() -> tuple[pd.DataFrame, pd.DataFrame, date]:
    old_spot = load_early_spot()
    new_spot = load_spot(DEFAULT_SPOT / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv")
    new_spot = new_spot[(new_spot.ts.dt.date >= pd.Timestamp("2025-10-01").date()) & (new_spot.ts.dt.date < VALID_END)]
    spot = pd.concat([old_spot, new_spot], ignore_index=True).sort_values("ts").drop_duplicates("ts")
    old_fut, _ = load_early_futures()
    new_fut, _ = load_front_month("NIFTY", DEFAULT_ROOT)
    new_fut = new_fut[(new_fut.ts.dt.date >= pd.Timestamp("2025-10-01").date()) & (new_fut.ts.dt.date < VALID_END)]
    fut = pd.concat([old_fut, new_fut], ignore_index=True).sort_values("ts")
    if fut.duplicated("ts").any():
        raise SystemExit("Duplicate NIFTY futures timestamps across source boundary")
    return spot, fut, NIFTY_START


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for symbol in ("NIFTY", "BANKNIFTY"):
        if symbol == "NIFTY":
            spot, fut, train_start = load_nifty()
        else:
            spot = load_spot(DEFAULT_SPOT / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv")
            spot = spot[(spot.ts.dt.date >= BANK_START) & (spot.ts.dt.date < VALID_END)]
            fut, _ = load_front_month("BANKNIFTY", DEFAULT_ROOT)
            fut = fut[(fut.ts.dt.date >= BANK_START) & (fut.ts.dt.date < VALID_END)]
            train_start = BANK_START
        merged = make_signals(spot, fut)
        merged = merged[(merged.ts.dt.date >= train_start) & (merged.ts.dt.date < VALID_END)].copy()
        train = merged[merged.ts.dt.date < TRAIN_END].copy()
        validation = merged[(merged.ts.dt.date >= TRAIN_END) & (merged.ts.dt.date < VALID_END)].copy()
        base_train = summarize(run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), symbol, True, "TRAIN")
        base_val = summarize(run_one(validation, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), symbol, True, "VALIDATION_FEB_MAY")
        candidates = []
        for long_min, short_min in product(OPTIONS, repeat=2):
            f = body_filter(train, long_min, short_min)
            stats = summarize(run_one(f, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), symbol, True, "TRAIN")
            passed = (stats["trades"] >= MIN_TRAIN and stats["win_rate_pct"] > base_train["win_rate_pct"]
                      and stats["net_points"] > base_train["net_points"])
            candidates.append({**stats, "long_body_min": "none" if long_min is None else long_min,
                               "short_body_min": "none" if short_min is None else short_min,
                               "train_gate_pass": passed})
        eligible = [c for c in candidates if c["train_gate_pass"]]
        chosen = max(eligible, key=lambda c: (c["net_points"], c["win_rate_pct"], -c["max_drawdown_points"])) if eligible else None
        row = {"symbol": symbol, "train_start": str(train_start), "train_end_exclusive": str(TRAIN_END),
               "baseline_train_trades": base_train["trades"], "baseline_train_win_rate_pct": base_train["win_rate_pct"],
               "baseline_train_net_points": base_train["net_points"], "baseline_validation_trades": base_val["trades"],
               "baseline_validation_win_rate_pct": base_val["win_rate_pct"], "baseline_validation_net_points": base_val["net_points"],
               "grid_size": len(candidates), "train_pass_count": len(eligible), "recent_seven_sessions_used": False}
        if chosen:
            lm = None if chosen["long_body_min"] == "none" else float(chosen["long_body_min"])
            sm = None if chosen["short_body_min"] == "none" else float(chosen["short_body_min"])
            vstats = summarize(run_one(body_filter(validation, lm, sm), symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1),
                               symbol, True, "VALIDATION_FEB_MAY")
            row.update({"selected_long_body_min": chosen["long_body_min"], "selected_short_body_min": chosen["short_body_min"],
                        "selected_train_trades": chosen["trades"], "selected_train_win_rate_pct": chosen["win_rate_pct"],
                        "selected_train_net_points": chosen["net_points"], "validation_trades": vstats["trades"],
                        "validation_win_rate_pct": vstats["win_rate_pct"], "validation_net_points": vstats["net_points"],
                        "validation_drawdown_points": vstats["max_drawdown_points"],
                        "validation_gate_pass": bool(vstats["trades"] >= 8 and vstats["win_rate_pct"] >= 50 and vstats["net_points"] > 0)})
        else:
            row.update({"selected_long_body_min": "", "selected_short_body_min": "", "selected_train_trades": "",
                        "selected_train_win_rate_pct": "", "selected_train_net_points": "",
                        "validation_trades": "NOT_RUN_TRAIN_GATE_FAILED", "validation_win_rate_pct": "",
                        "validation_net_points": "", "validation_drawdown_points": "", "validation_gate_pass": False})
        all_rows.append(row)
    result = pd.DataFrame(all_rows)
    result.to_csv(OUT / "results.csv", index=False)
    print("SIDE-SPECIFIC BODY FILTER | PAPER ONLY | selected futures, volume filter, modeled Upstox cost + 1 point/side")
    print(result.to_string(index=False))
    print(f"Saved: {OUT}; June onward excluded; recent seven untouched; orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
