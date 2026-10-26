#!/usr/bin/env python3
"""Development-only test of a fixed intraday futures VWAP-reversion trigger.

Signal: after price has closed at least 1 futures ATR beyond session VWAP,
enter only when a later completed bar crosses back through VWAP. Existing
futures fills, volume filter, ATR stops, estimated fees, and slippage remain
unchanged. Dates from 2026-06-01 onward are excluded before simulation.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT,
    DEFAULT_SPOT,
    OOS_START,
    load_front_month,
    load_spot,
    make_signals,
    run_one,
    summarize,
)
from src.safety.live_gate import assert_paper_trading_only


OUT_DIR = ROOT / "reports" / "vwap_reversion_development_20261014"
RR = 1.5
ATR_BUFFER = 0.1
SLIPPAGE_PER_SIDE = 1.0
EXCURSION_ATR = 1.0


def main() -> int:
    assert_paper_trading_only()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    trades_all: list[pd.DataFrame] = []
    coverage: list[dict] = []

    for symbol, spot_name in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        merged = make_signals(spot, futures)
        dev = merged[
            (merged["ts"].dt.date < OOS_START)
            & (merged["ts"].dt.date >= pd.Timestamp("2025-10-01").date())
        ].copy()

        typical_price = (dev["high_fut"] + dev["low_fut"] + dev["close_fut"]) / 3
        dev["_pv"] = typical_price * dev["volume_fut"]
        dev["_cum_pv"] = dev.groupby("session", sort=False)["_pv"].cumsum()
        dev["_cum_volume"] = dev.groupby("session", sort=False)["volume_fut"].cumsum()
        dev["vwap_fut"] = dev["_cum_pv"] / dev["_cum_volume"].replace(0, np.nan)
        dev["_vwap_deviation_atr"] = (dev["close_fut"] - dev["vwap_fut"]) / dev["atr14_fut"]
        prev_close = dev.groupby("session", sort=False)["close_fut"].shift(1)
        prev_vwap = dev.groupby("session", sort=False)["vwap_fut"].shift(1)
        prev_atr = dev.groupby("session", sort=False)["atr14_fut"].shift(1)
        prev_deviation = (prev_close - prev_vwap) / prev_atr
        minutes = dev["ts"].dt.hour * 60 + dev["ts"].dt.minute
        valid_time = (minutes >= 9 * 60 + 45) & (minutes < 15 * 60)

        # Prior completed bar showed a >=1 ATR excursion; current completed bar
        # reclaims VWAP. Entries are then filled at the next futures bar open.
        dev["long_signal"] = (
            valid_time
            & (prev_deviation <= -EXCURSION_ATR)
            & (dev["close_fut"] > dev["vwap_fut"])
        )
        dev["short_signal"] = (
            valid_time
            & (prev_deviation >= EXCURSION_ATR)
            & (dev["close_fut"] < dev["vwap_fut"])
        )

        unavailable = selection.loc[
            (selection["session_date"] < OOS_START)
            & (selection["status"] != "SELECTED")
        ]
        coverage.append({
            "symbol": symbol,
            "development_coverage": "COMPLETE" if unavailable.empty else "INCOMPLETE",
            "unavailable_sessions": len(unavailable),
            "unavailable_dates": ";".join(str(d) for d in unavailable["session_date"]),
        })

        trades = run_one(
            dev, symbol, True, "upstox", SLIPPAGE_PER_SIDE, RR, ATR_BUFFER, 1
        )
        if not trades.empty:
            trades["entry_logic"] = "1-ATR excursion then close back through session VWAP"
            trades["volume_filter"] = True
            trades["split"] = "DEVELOPMENT"
            trades_all.append(trades)
        stats = summarize(trades, symbol, True, "DEVELOPMENT")
        stats.update({
            "entry_logic": "1-ATR excursion then close back through session VWAP",
            "excursion_threshold_atr": EXCURSION_ATR,
            "rr": RR,
            "atr_buffer": ATR_BUFFER,
            "volume_filter_threshold": "current volume > prior 20-bar average * 1.2",
            "slippage_points_per_side": SLIPPAGE_PER_SIDE,
        })
        summaries.append(stats)

    out = pd.DataFrame(summaries).merge(
        pd.DataFrame(coverage), on="symbol", how="left", validate="one_to_one"
    )
    out.to_csv(OUT_DIR / "development_summary.csv", index=False)
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(
            OUT_DIR / "development_trades.csv", index=False
        )
    print("DEVELOPMENT ONLY | VWAP reversion | estimated Upstox costs | 1 point/side slippage")
    print(out.to_string(index=False))
    print(f"Saved: {OUT_DIR / 'development_summary.csv'}")
    print("PAPER | RESEARCH ONLY | ORDERS: 0 | OOS was not simulated or used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
