#!/usr/bin/env python3
"""TRAIN-select a short-only overnight-gap/time gate; validate once on Feb-May.

Fixed in advance: gap-up or gap-down, and one of six non-overlapping hourly
windows from 09:30 to 15:00 IST. Candidate selection uses Oct-Jan only, with
>=8 trades, positive net and >=50% wins. The chosen rule is then evaluated on
Feb-May, with a predeclared 8-trade validation minimum. June onward, including
the recent seven-session window, is excluded before constructing the study.
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

OUT = ROOT / "reports" / "train_gap_time_short_gate_20261014"
TRAIN_START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
WINDOWS = ((570, 630), (630, 690), (690, 750), (750, 810), (810, 870), (870, 900))
MIN_TRAIN, MIN_VALIDATION = 8, 8
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1


def apply_rule(frame: pd.DataFrame, gap: pd.Series, gap_rule: str,
               start: int, end: int) -> pd.DataFrame:
    x = frame.copy()
    minutes = x.ts.dt.hour * 60 + x.ts.dt.minute
    is_gap = gap.gt(0) if gap_rule == "GAP_UP" else gap.lt(0)
    allowed = x.session.map(is_gap).fillna(False).astype(bool)
    x["long_signal"] = False
    x["short_signal"] = (
        x.short_signal.astype(bool) & allowed & (minutes >= start) & (minutes < end)
    )
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    grid_rows, selected_rows, validation_rows, validation_trades = [], [], [], []
    settings = (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    )
    for symbol, filename in settings:
        spot = load_spot(DEFAULT_SPOT / filename)
        spot = spot[spot.ts.dt.date < VALID_END].copy()
        daily = spot.groupby("session", sort=True).agg(day_open=("open", "first"),
                                                       day_close=("close", "last"))
        gap = daily.day_open - daily.day_close.shift(1)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        merged = make_signals(spot, futures)
        dev = merged[(merged.ts.dt.date >= TRAIN_START) &
                     (merged.ts.dt.date < VALID_END)].copy()
        train = dev[dev.ts.dt.date < TRAIN_END].copy()
        validation = dev[dev.ts.dt.date >= TRAIN_END].copy()
        symbol_rows = []
        for gap_rule in ("GAP_UP", "GAP_DOWN"):
            for start, end in WINDOWS:
                name = f"SHORT_{gap_rule}_{start // 60:02d}{start % 60:02d}-{end // 60:02d}{end % 60:02d}"
                filtered = apply_rule(train, gap, gap_rule, start, end)
                trades = run_one(filtered, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
                stats = summarize(trades, symbol, True, "TRAIN_OCT_JAN")
                row = {**stats, "rule": name, "gap_rule": gap_rule,
                       "window_ist": f"{start // 60:02d}:{start % 60:02d}-{end // 60:02d}:{end % 60:02d}"}
                grid_rows.append(row)
                symbol_rows.append(row)
        eligible = [r for r in symbol_rows if r["trades"] >= MIN_TRAIN and
                    r["net_points"] > 0 and r["win_rate_pct"] >= 50]
        train_gate = bool(eligible)
        pool = eligible if eligible else [r for r in symbol_rows if r["trades"] >= MIN_TRAIN]
        chosen = max(pool or symbol_rows,
                     key=lambda r: (r["net_points"], r["win_rate_pct"], -r["max_drawdown_points"]))
        selected_rows.append({**chosen, "train_gate_pass": train_gate})
        start, end = next((a, b) for a, b in WINDOWS
                          if f"{a // 60:02d}:{a % 60:02d}-{b // 60:02d}:{b % 60:02d}" == chosen["window_ist"])
        frozen = apply_rule(validation, gap, chosen["gap_rule"], start, end)
        trades = run_one(frozen, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        result = summarize(trades, symbol, True, "VALIDATION_FEB_MAY")
        val_pass = bool(result["trades"] >= MIN_VALIDATION and result["net_points"] > 0
                        and result["win_rate_pct"] >= 50)
        validation_rows.append({**result, "rule": chosen["rule"],
                                "train_gate_pass": train_gate,
                                "validation_gate_pass": val_pass})
        if not trades.empty:
            trades = trades.copy()
            trades["rule"], trades["period"] = chosen["rule"], "VALIDATION_FEB_MAY"
            validation_trades.append(trades)
        unavailable = selection[(selection.session_date >= TRAIN_START) &
                                (selection.session_date < VALID_END) &
                                (selection.status != "SELECTED")]
        print(f"{symbol}: {chosen['rule']} | TRAIN gate {train_gate} | "
              f"validation {result['trades']} trades, {result['win_rate_pct']}%, "
              f"{result['net_points']} pts | missing sessions {len(unavailable)}")

    pd.DataFrame(grid_rows).to_csv(OUT / "train_grid.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_rules.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_results.csv", index=False)
    if validation_trades:
        pd.concat(validation_trades, ignore_index=True).to_csv(OUT / "validation_trades.csv", index=False)
    print(pd.DataFrame(selected_rows)[["symbol", "rule", "trades", "win_rate_pct",
                                       "net_points", "train_gate_pass"]].to_string(index=False))
    print(pd.DataFrame(validation_rows)[["symbol", "rule", "trades", "win_rate_pct",
                                         "net_points", "validation_gate_pass"]].to_string(index=False))
    print(f"Saved: {OUT}; actual futures; estimated Upstox charges + 1 point/side; paper only; no orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
