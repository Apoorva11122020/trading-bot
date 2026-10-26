#!/usr/bin/env python3
"""Research-only study of next-candle confirmation after a classic pivot cross."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, OOS_START, finish, load_front_month, load_spot,
    make_signals, run_one, summarize,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "pivot_confirmation_study_20261014"


def with_cross_level(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    previous = x.groupby("session", sort=False)["close_spot"].shift(1)
    up_r1 = (x.close_spot > x.r1) & (previous <= x.r1)
    down_s1 = (x.close_spot < x.s1) & (previous >= x.s1)
    x["cross_level"] = float("nan")
    long_mask, short_mask = x.long_signal, x.short_signal
    x.loc[long_mask & up_r1, "cross_level"] = x.loc[long_mask & up_r1, "r1"]
    x.loc[long_mask & ~up_r1, "cross_level"] = x.loc[long_mask & ~up_r1, "pp"]
    x.loc[short_mask & down_s1, "cross_level"] = x.loc[short_mask & down_s1, "s1"]
    x.loc[short_mask & ~down_s1, "cross_level"] = x.loc[short_mask & ~down_s1, "pp"]
    return x


def confirmation_frame(frame: pd.DataFrame, candle_color: bool) -> pd.DataFrame:
    x = frame.copy()
    g = x.groupby("session", sort=False)
    previous_long = g.long_signal.shift(1).fillna(False).astype(bool)
    previous_short = g.short_signal.shift(1).fillna(False).astype(bool)
    previous_level = g.cross_level.shift(1)
    long_confirmed = previous_long & x.close_spot.gt(previous_level)
    short_confirmed = previous_short & x.close_spot.lt(previous_level)
    if candle_color:
        long_confirmed &= x.close_spot.ge(x.open_spot)
        short_confirmed &= x.close_spot.le(x.open_spot)
    x["long_signal"] = long_confirmed
    x["short_signal"] = short_confirmed
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    settings = (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv", 1.5, 0.1),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv", 1.5, 0.1),
    )
    for symbol, spot_name, rr, atr_buffer in settings:
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        merged = with_cross_level(make_signals(spot, futures))
        for volume_filter in (False, True):
            candidates = {
                "baseline": merged,
                "hold_pivot": confirmation_frame(merged, candle_color=False),
                "hold_pivot_directional_candle": confirmation_frame(merged, candle_color=True),
            }
            for variant, candidate_frame in candidates.items():
                trades = run_one(candidate_frame, symbol, volume_filter, "upstox", 1.0,
                                 rr, atr_buffer, 1)
                if not trades.empty:
                    trades = trades.copy()
                    trades["variant"] = variant
                    trades["volume_filter"] = volume_filter
                    trades["split"] = trades.entry_ts.map(
                        lambda t: "DEVELOPMENT" if pd.Timestamp(t).date() < OOS_START else "OOS")
                    all_trades.append(trades)
                for split, mask in (
                    ("DEVELOPMENT", trades.entry_ts.map(lambda t: pd.Timestamp(t).date() < OOS_START) if not trades.empty else []),
                    ("OOS", trades.entry_ts.map(lambda t: pd.Timestamp(t).date() >= OOS_START) if not trades.empty else []),
                ):
                    part = trades.loc[mask] if not trades.empty else trades
                    stats = summarize(part, symbol, volume_filter, split)
                    stats.update(variant=variant, rr=rr, atr_buffer=atr_buffer)
                    rows.append(stats)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {OUT}")
    print("One extra 5m confirmation candle; real selected futures OHLCV; Upstox statutory charges + 1 point/side slippage.")
    print("OOS June 2026 onward is evaluation only. Research only; no orders submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
