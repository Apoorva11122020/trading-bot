#!/usr/bin/env python3
"""Select a BE trigger threshold on TRAIN only, then evaluate one frozen rule.

The predeclared candidates are close-confirmed, cost-adjusted BE at 0.5R,
0.75R, and 1.0R on both sides of the baseline pivot strategy. Per symbol,
selection prefers thresholds that improve TRAIN net and wins over baseline;
ties rank by net, win rate, then drawdown. If no threshold improves both,
selection is flagged as a TRAIN failure. Only that selected threshold is run
on Feb-May validation. June onward and recent seven sessions are excluded.
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

OUT = ROOT / "reports" / "train_selected_be_thresholds_20261014"
START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
TRIGGERS = (0.5, 0.75, 1.0)
SLIPPAGE, RR, ATR_BUFFER = 1.0, 1.5, 0.1
MIN_VALIDATION_TRADES = 8


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    grid, choices, validation_rows, trade_rows = [], [], [], []
    for symbol, filename in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / filename)
        # Exclude June onward before the study input is constructed.
        spot = spot[(spot.ts.dt.date >= START) & (spot.ts.dt.date < VALID_END)]
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        frame = make_signals(spot, futures)
        train = frame[(frame.ts.dt.date >= START) & (frame.ts.dt.date < TRAIN_END)].copy()
        validation = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
        baseline_train_trades = run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        baseline_train = summarize(baseline_train_trades, symbol, True, "TRAIN_OCT_JAN")
        candidate_rows = []
        for trigger in TRIGGERS:
            trades = run_one(train, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1,
                             breakeven_trigger_r=trigger)
            stat = summarize(trades, symbol, True, "TRAIN_OCT_JAN")
            stat.update({"rule": "both_sides_close_confirmed_be", "trigger_r": trigger})
            stat["train_net_improved"] = stat["net_points"] > baseline_train["net_points"]
            stat["train_win_rate_improved"] = stat["win_rate_pct"] > baseline_train["win_rate_pct"]
            grid.append(stat)
            candidate_rows.append(stat)
        eligible = [r for r in candidate_rows if r["train_net_improved"] and
                    r["train_win_rate_improved"] and r["trades"] >= 20]
        train_pass = bool(eligible)
        pool = eligible if eligible else candidate_rows
        selected = max(pool, key=lambda r: (r["net_points"], r["win_rate_pct"],
                                             -r["max_drawdown_points"]))
        choices.append({**selected, "train_gate_pass": train_pass,
                        "baseline_train_net": baseline_train["net_points"],
                        "baseline_train_win_rate_pct": baseline_train["win_rate_pct"]})

        val_base_trades = run_one(validation, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1)
        val_base = summarize(val_base_trades, symbol, True, "VALIDATION_FEB_MAY")
        val_trades = run_one(validation, symbol, True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1,
                             breakeven_trigger_r=float(selected["trigger_r"]))
        val = summarize(val_trades, symbol, True, "VALIDATION_FEB_MAY")
        gate = bool(train_pass and val["trades"] >= MIN_VALIDATION_TRADES and
                    val["win_rate_pct"] >= 50.0 and val["net_points"] > 0)
        validation_rows.append({**val, "rule": "both_sides_close_confirmed_be",
                                "trigger_r": selected["trigger_r"],
                                "train_gate_pass": train_pass,
                                "baseline_validation_trades": val_base["trades"],
                                "baseline_validation_win_rate_pct": val_base["win_rate_pct"],
                                "baseline_validation_net_points": val_base["net_points"],
                                "validation_gate_pass": gate})
        if not val_trades.empty:
            val_trades = val_trades.copy()
            val_trades["rule"] = f"both_sides_close_confirmed_be_{selected['trigger_r']}R"
            val_trades["period"] = "VALIDATION_FEB_MAY"
            trade_rows.append(val_trades)
        missing = selection[(selection.session_date >= START) &
                            (selection.session_date < VALID_END) &
                            (selection.status != "SELECTED")]
        print(f"{symbol}: TRAIN selected BE {selected['trigger_r']}R; train pass={train_pass}; "
              f"validation={val['trades']} trades / {val['win_rate_pct']}% / "
              f"{val['net_points']} pts; gate={gate}; missing sessions={len(missing)}")

    pd.DataFrame(grid).to_csv(OUT / "train_grid.csv", index=False)
    pd.DataFrame(choices).to_csv(OUT / "selected_rules.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_results.csv", index=False)
    if trade_rows:
        pd.concat(trade_rows, ignore_index=True).to_csv(OUT / "validation_trades.csv", index=False)
    print(pd.DataFrame(validation_rows).to_string(index=False))
    print(f"Saved: {OUT}; modeled Upstox charges + 1 point/side; paper only; no orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
