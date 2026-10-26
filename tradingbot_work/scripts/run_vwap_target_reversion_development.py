#!/usr/bin/env python3
"""Development-only test: enter on a 1-ATR VWAP stretch, exit at VWAP.

Long/short entries trigger when a completed futures candle first stretches at
least 1 ATR below/above session VWAP. Entry is next-bar open, target is the
signal-time VWAP, and stop is 1.5 ATR beyond the fill. Applies the existing
volume filter, estimated Upstox charges, and 1 point/side slippage. OOS dates
are excluded before simulation.
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
    finish,
    load_front_month,
    load_spot,
    make_signals,
    summarize,
)
from src.safety.live_gate import assert_paper_trading_only

OUT_DIR = ROOT / "reports" / "vwap_target_reversion_development_20261014"
EXCURSION_ATR = 1.0
STOP_ATR = 1.5
SLIPPAGE_PER_SIDE = 1.0


def simulate(x: pd.DataFrame, symbol: str) -> pd.DataFrame:
    trades: list[dict] = []
    for _, day in x.groupby("session", sort=True):
        day = day.sort_values("ts").reset_index(drop=True)
        pos = None
        pending = None
        previous_deviation = None
        for i, row in day.iterrows():
            # Fill only on the bar immediately after the completed signal bar.
            if pending is not None:
                entry = float(row["open_fut"])
                target = float(pending["target"])
                atr_value = float(pending["atr"])
                side = pending["side"]
                stop = entry - STOP_ATR * atr_value if side == "LONG" else entry + STOP_ATR * atr_value
                target_is_ahead = target > entry if side == "LONG" else target < entry
                if target_is_ahead and np.isfinite(entry) and np.isfinite(stop):
                    pos = {
                        "symbol": symbol,
                        "side": side,
                        "entry_ts": row["ts"],
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "contract": row["trading_symbol"],
                        "expiry": str(row["expiry"])[:10],
                        "lot_size": int(row["lot_size"]),
                    }
                pending = None

            exited = False
            if pos is not None:
                op, hi, lo = (float(row[k]) for k in ("open_fut", "high_fut", "low_fut"))
                if pos["side"] == "LONG":
                    if op <= pos["stop"]:
                        px, reason = op, "stop_gap"
                    elif op >= pos["target"]:
                        px, reason = op, "vwap_target_gap"
                    elif lo <= pos["stop"]:
                        px, reason = pos["stop"], "stop"
                    elif hi >= pos["target"]:
                        px, reason = pos["target"], "vwap_target"
                    else:
                        px, reason = None, None
                else:
                    if op >= pos["stop"]:
                        px, reason = op, "stop_gap"
                    elif op <= pos["target"]:
                        px, reason = op, "vwap_target_gap"
                    elif hi >= pos["stop"]:
                        px, reason = pos["stop"], "stop"
                    elif lo <= pos["target"]:
                        px, reason = pos["target"], "vwap_target"
                    else:
                        px, reason = None, None
                if px is not None:
                    trades.append(finish(pos, row["ts"], px, reason, "upstox", SLIPPAGE_PER_SIDE, pos["lot_size"]))
                    pos = None
                    exited = True

            mins = row["ts"].hour * 60 + row["ts"].minute
            if pos is not None and mins >= 15 * 60 + 15:
                trades.append(finish(pos, row["ts"], float(row["close_fut"]), "intraday_flat",
                                      "upstox", SLIPPAGE_PER_SIDE, pos["lot_size"]))
                pos = None
                exited = True

            if pos is None and not exited and pending is None and i + 1 < len(day):
                vol_ok = (
                    pd.notna(row["vol_avg20_fut"])
                    and row["vol_avg20_fut"] > 0
                    and row["volume_fut"] > row["vol_avg20_fut"] * 1.2
                )
                if 9 * 60 + 45 <= mins < 15 * 60 and vol_ok and pd.notna(row["_dev_atr"]):
                    dev_atr = float(row["_dev_atr"])
                    crossed_into_long_stretch = (
                        previous_deviation is not None
                        and previous_deviation > -EXCURSION_ATR
                        and dev_atr <= -EXCURSION_ATR
                    )
                    crossed_into_short_stretch = (
                        previous_deviation is not None
                        and previous_deviation < EXCURSION_ATR
                        and dev_atr >= EXCURSION_ATR
                    )
                    if crossed_into_long_stretch:
                        pending = {"side": "LONG", "target": row["vwap_fut"], "atr": row["atr14_fut"]}
                    elif crossed_into_short_stretch:
                        pending = {"side": "SHORT", "target": row["vwap_fut"], "atr": row["atr14_fut"]}

            if i == len(day) - 1 and pos is not None:
                trades.append(finish(pos, row["ts"], float(row["close_fut"]), "session_boundary",
                                      "upstox", SLIPPAGE_PER_SIDE, pos["lot_size"]))
                pos = None
            previous_deviation = (
                float(row["_dev_atr"]) if pd.notna(row["_dev_atr"]) else None
            )
    return pd.DataFrame(trades)


def main() -> int:
    assert_paper_trading_only()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    trades_all: list[pd.DataFrame] = []
    coverage: list[dict] = []
    for symbol, spot_name in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, selection = load_front_month(symbol, DEFAULT_ROOT)
        x = make_signals(spot, futures)
        x = x[(x["ts"].dt.date < OOS_START) & (x["ts"].dt.date >= pd.Timestamp("2025-10-01").date())].copy()
        typical = (x["high_fut"] + x["low_fut"] + x["close_fut"]) / 3
        x["_pv"] = typical * x["volume_fut"]
        x["_cum_pv"] = x.groupby("session", sort=False)["_pv"].cumsum()
        x["_cum_volume"] = x.groupby("session", sort=False)["volume_fut"].cumsum()
        x["vwap_fut"] = x["_cum_pv"] / x["_cum_volume"].replace(0, np.nan)
        x["_dev_atr"] = (x["close_fut"] - x["vwap_fut"]) / x["atr14_fut"]
        unavailable = selection.loc[(selection["session_date"] < OOS_START) & (selection["status"] != "SELECTED")]
        coverage.append({
            "symbol": symbol,
            "development_coverage": "COMPLETE" if unavailable.empty else "INCOMPLETE",
            "unavailable_sessions": len(unavailable),
            "unavailable_dates": ";".join(str(d) for d in unavailable["session_date"]),
        })
        trades = simulate(x, symbol)
        if not trades.empty:
            trades["entry_logic"] = "enter at 1-ATR excursion; target signal-time VWAP; stop 1.5 ATR"
            trades["split"] = "DEVELOPMENT"
            trades_all.append(trades)
        stats = summarize(trades, symbol, True, "DEVELOPMENT")
        stats.update({
            "entry_logic": "enter at 1-ATR excursion; target signal-time VWAP; stop 1.5 ATR",
            "excursion_threshold_atr": EXCURSION_ATR,
            "stop_atr": STOP_ATR,
            "volume_filter_threshold": "current volume > prior 20-bar average * 1.2",
            "slippage_points_per_side": SLIPPAGE_PER_SIDE,
        })
        rows.append(stats)

    out = pd.DataFrame(rows).merge(pd.DataFrame(coverage), on="symbol", validate="one_to_one")
    out.to_csv(OUT_DIR / "development_summary.csv", index=False)
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(OUT_DIR / "development_trades.csv", index=False)
    print("DEVELOPMENT ONLY | enter at stretch, VWAP target | estimated Upstox costs | 1 point/side slippage")
    print(out.to_string(index=False))
    print(f"Saved: {OUT_DIR / 'development_summary.csv'}")
    print("PAPER | RESEARCH ONLY | ORDERS: 0 | OOS was not simulated or used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
