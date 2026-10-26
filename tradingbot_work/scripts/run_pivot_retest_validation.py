#!/usr/bin/env python3
"""Fixed pivot-breakout retest hypothesis; chronological validation, paper research only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, OOS_START, load_front_month, load_spot,
    make_signals, run_one, summarize,
)
from scripts.pivot_indicators import atr
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "pivot_retest_validation_20261014"


def make_retest_frame(spot: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
    base = make_signals(spot, futures)
    long_base = base["long_signal"].copy()
    short_base = base["short_signal"].copy()
    base["atr14_spot"] = atr(base.rename(columns={"high_spot": "high", "low_spot": "low", "close_spot": "close"}), 14)
    prev_close = base.groupby("session", sort=False)["close_spot"].shift(1)
    up_r1 = (base.close_spot > base.r1) & (prev_close <= base.r1)
    down_s1 = (base.close_spot < base.s1) & (prev_close >= base.s1)
    base["cross_level"] = float("nan")
    base.loc[long_base & up_r1, "cross_level"] = base.loc[long_base & up_r1, "r1"]
    base.loc[long_base & ~up_r1, "cross_level"] = base.loc[long_base & ~up_r1, "pp"]
    base.loc[short_base & down_s1, "cross_level"] = base.loc[short_base & down_s1, "s1"]
    base.loc[short_base & ~down_s1, "cross_level"] = base.loc[short_base & ~down_s1, "pp"]
    base["long_signal"] = False
    base["short_signal"] = False
    indexes = base.groupby("session", sort=False).indices
    for positions in indexes.values():
        pos = list(positions)
        for j, i in enumerate(pos):
            side = "LONG" if bool(long_base.iloc[i]) else ("SHORT" if bool(short_base.iloc[i]) else None)
            if side is None:
                continue
            level = base.at[i, "cross_level"]
            # Examine only the next three completed candles, within the same session.
            for k in pos[j + 1:j + 4]:
                a = base.at[k, "atr14_spot"]
                if pd.isna(a) or a <= 0:
                    continue
                band = float(a) * 0.10
                if side == "LONG":
                    touched = base.at[k, "low_spot"] <= level + band
                    reclaimed = base.at[k, "close_spot"] > level and base.at[k, "close_spot"] >= base.at[k, "open_spot"]
                    if touched and reclaimed:
                        base.at[k, "long_signal"] = True
                        break
                else:
                    touched = base.at[k, "high_spot"] >= level - band
                    reclaimed = base.at[k, "close_spot"] < level and base.at[k, "close_spot"] <= base.at[k, "open_spot"]
                    if touched and reclaimed:
                        base.at[k, "short_signal"] = True
                        break
    return base


def split_label(ts: object) -> str:
    day = pd.Timestamp(ts).date()
    if day < pd.Timestamp("2026-02-01").date():
        return "TRAIN_OCT_JAN"
    if day < OOS_START:
        return "VALIDATION_FEB_MAY"
    return "OOS_JUN_ONWARD"


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    )
    summaries, trade_frames = [], []
    for symbol, spot_file in settings:
        spot = load_spot(DEFAULT_SPOT / spot_file)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        candidates = (
            ("baseline", make_signals(spot, futures)),
            ("pivot_retest_3bars_0.10ATR_directional", make_retest_frame(spot, futures)),
        )
        for rule, frame in candidates:
            trades = run_one(frame, symbol, True, "upstox", 1.0, 1.5, 0.1, 1)
            if not trades.empty:
                trades["period"] = trades.entry_ts.map(split_label)
                trades["rule"] = rule
                trade_frames.append(trades)
            for period in ("TRAIN_OCT_JAN", "VALIDATION_FEB_MAY", "OOS_JUN_ONWARD"):
                part = trades.loc[trades.period.eq(period)] if not trades.empty else trades
                stats = summarize(part, symbol, True, period)
                stats["rule"] = rule
                summaries.append(stats)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Fixed rule; actual selected futures candles; estimated Upstox statutory fees + 1 point/side slippage.")
    print("No broker orders. Fees + 1 point/side slippage; RR 1.5, ATR stop buffer 0.1, volume filter ON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
