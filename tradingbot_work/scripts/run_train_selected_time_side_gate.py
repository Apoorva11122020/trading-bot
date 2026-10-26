#!/usr/bin/env python3
"""Select a simple pivot time/side gate on TRAIN only, then validate once.

Grid is fixed in advance: one-hour windows starting every 30 minutes from
09:30 through 14:00 IST; LONG, SHORT, or BOTH. Selection requires >=8 TRAIN
trades, positive TRAIN net, and >=50% TRAIN wins. If no candidate meets that
development gate, report the best eligible TRAIN candidate but mark it as a
development failure. June onward is cut before simulation; recent sessions are
never read for selection or validation.
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

OUT = ROOT / "reports" / "train_selected_time_side_gate_20261014"
TRAIN_START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
WINDOW_STARTS = range(9 * 60 + 30, 14 * 60 + 1, 30)
WINDOW_LENGTH = 60
RR = 1.5
ATR_BUFFER = 0.1
SLIPPAGE = 1.0


def time_side_frame(source: pd.DataFrame, start: int, side: str) -> pd.DataFrame:
    x = source.copy()
    minute = x.ts.dt.hour * 60 + x.ts.dt.minute
    inside = (minute >= start) & (minute < start + WINDOW_LENGTH)
    long = x.long_signal.astype(bool) & inside
    short = x.short_signal.astype(bool) & inside
    if side == "LONG":
        short[:] = False
    elif side == "SHORT":
        long[:] = False
    x["long_signal"], x["short_signal"] = long, short
    return x


def score(trades: pd.DataFrame, symbol: str, period: str, rule: str) -> dict:
    row = summarize(trades, symbol, True, period)
    row["rule"] = rule
    return row


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    all_candidates, selected_rows, val_rows, trade_outputs = [], [], [], []
    for symbol, spot_name in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        merged = make_signals(spot, futures)
        # Exclude June onward before constructing any candidate simulation input.
        dev = merged[(merged.ts.dt.date >= TRAIN_START) &
                     (merged.ts.dt.date < VALID_END)].copy()
        train = dev[dev.ts.dt.date < TRAIN_END].copy()
        validation = dev[(dev.ts.dt.date >= TRAIN_END) &
                         (dev.ts.dt.date < VALID_END)].copy()
        for start in WINDOW_STARTS:
            window = f"{start // 60:02d}:{start % 60:02d}-{(start + WINDOW_LENGTH) // 60:02d}:{(start + WINDOW_LENGTH) % 60:02d}"
            for side in ("LONG", "SHORT", "BOTH"):
                rule = f"{window}_{side}"
                train_data = time_side_frame(train, start, side)
                tr = run_one(train_data, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
                stat = score(tr, symbol, "TRAIN_OCT_JAN", rule)
                stat.update({"window_ist": window, "side_gate": side})
                all_candidates.append(stat)

        candidates = pd.DataFrame([r for r in all_candidates if r["symbol"] == symbol])
        eligible = candidates[(candidates.trades >= 8) &
                              (candidates.net_points > 0) &
                              (candidates.win_rate_pct >= 50)]
        dev_pass = not eligible.empty
        pool = eligible if dev_pass else candidates[candidates.trades >= 8]
        if pool.empty:
            selected = candidates.sort_values(["net_points", "win_rate_pct"], ascending=False).iloc[0]
        else:
            selected = pool.sort_values(
                ["net_points", "win_rate_pct", "max_drawdown_points"],
                ascending=[False, False, True],
            ).iloc[0]
        selected_rows.append({**selected.to_dict(), "train_gate_pass": dev_pass})
        start_hour, start_minute = map(int, selected.window_ist.split("-")[0].split(":"))
        start = start_hour * 60 + start_minute
        val_data = time_side_frame(validation, start, str(selected.side_gate))
        vt = run_one(val_data, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        val_rule = f"{selected.window_ist}_{selected.side_gate}"
        val_stat = score(vt, symbol, "VALIDATION_FEB_MAY", val_rule)
        val_stat.update({"window_ist": selected.window_ist, "side_gate": selected.side_gate,
                         "train_gate_pass": dev_pass,
                         "validation_gate_pass": bool(val_stat["trades"] >= 8 and
                             val_stat["net_points"] > 0 and val_stat["win_rate_pct"] >= 50)})
        val_rows.append(val_stat)
        if not vt.empty:
            vt = vt.copy()
            vt["symbol"] = symbol
            vt["rule"] = val_rule
            vt["period"] = "VALIDATION_FEB_MAY"
            trade_outputs.append(vt)
        missing = selection[(selection.session_date >= TRAIN_START) &
                            (selection.session_date < VALID_END) &
                            (selection.status != "SELECTED")]
        print(f"{symbol}: selected {val_rule}; TRAIN gate={dev_pass}; "
              f"missing sessions={len(missing)}")

    grid = pd.DataFrame(all_candidates)
    chosen = pd.DataFrame(selected_rows)
    validation = pd.DataFrame(val_rows)
    grid.to_csv(OUT / "train_candidate_grid.csv", index=False)
    chosen.to_csv(OUT / "selected_train_rules.csv", index=False)
    validation.to_csv(OUT / "validation_results.csv", index=False)
    if trade_outputs:
        pd.concat(trade_outputs, ignore_index=True).to_csv(OUT / "validation_trades.csv", index=False)
    print("TRAIN ONLY candidate ranking; exact chosen rule checked on Feb-May validation.")
    print(chosen[["symbol", "rule", "trades", "win_rate_pct", "net_points", "train_gate_pass"]].to_string(index=False))
    print(validation[["symbol", "rule", "trades", "win_rate_pct", "net_points", "validation_gate_pass"]].to_string(index=False))
    print(f"Saved: {OUT}; modeled Upstox charges + 1 point/side; paper only; no orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
