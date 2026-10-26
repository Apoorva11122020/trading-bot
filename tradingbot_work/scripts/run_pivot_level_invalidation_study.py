#!/usr/bin/env python3
"""Research-only test: exit a pivot-cross position after spot closes back through its crossed level."""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, END, OOS_START, START,
    finish, load_front_month, load_spot, make_signals, run_one, summarize,
)
from src.safety.live_gate import assert_paper_trading_only

OUT = ROOT / "reports" / "pivot_level_invalidation_20261014"


def with_reference_level(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    prev_close = x.groupby("session", sort=False)["close_spot"].shift(1)
    # Prefer the outer level when one candle crosses both levels.
    up_r1 = (x.close_spot > x.r1) & (prev_close <= x.r1)
    down_s1 = (x.close_spot < x.s1) & (prev_close >= x.s1)
    x["reference_level"] = float("nan")
    long_mask = x.long_signal
    short_mask = x.short_signal
    x.loc[long_mask & up_r1, "reference_level"] = x.loc[long_mask & up_r1, "r1"]
    x.loc[long_mask & ~up_r1, "reference_level"] = x.loc[long_mask & ~up_r1, "pp"]
    x.loc[short_mask & down_s1, "reference_level"] = x.loc[short_mask & down_s1, "s1"]
    x.loc[short_mask & ~down_s1, "reference_level"] = x.loc[short_mask & ~down_s1, "pp"]
    return x


def replay_with_invalidation(frame: pd.DataFrame, symbol: str, volume_filter: bool,
                             cost_model: str = "upstox", slippage: float = 1.0,
                             rr: float = 1.5, atr_buffer: float = 0.1) -> pd.DataFrame:
    x = frame.sort_values("ts").reset_index(drop=True)
    trades: list[dict] = []
    position = None
    pending = None
    for i, row in x.iterrows():
        ts, session = row.ts, row.session
        if position is not None and session != position["session"]:
            prev = x.iloc[i - 1]
            trades.append(finish(position, prev.ts, float(prev.close_fut), "session_boundary",
                                 cost_model, slippage, int(position["lot_size"])))
            position = None
        if pending is not None and (
            session != pending["session"] or row.instrument_key != pending["instrument_key"]
            or (ts - pending["signal_ts"]).total_seconds() > 301
        ):
            pending = None

        exited_this_bar = False
        if position is not None and position.get("exit_at_open"):
            trades.append(finish(position, ts, float(row.open_fut), "pivot_reclaim_next_open",
                                 cost_model, slippage, int(position["lot_size"])))
            position = None
            exited_this_bar = True

        # Fill queued entry at the next futures bar open.
        if position is None and pending is not None and not exited_this_bar:
            signal = pending["signal_row"]
            side = pending["side"]
            entry = float(row.open_fut)
            atr_value = signal.atr14_fut
            ref = signal.reference_level
            pending = None
            if pd.notna(atr_value) and float(atr_value) > 0 and pd.notna(ref):
                stop = (float(signal.low_fut - float(atr_value) * atr_buffer) if side == "LONG"
                        else float(signal.high_fut + float(atr_value) * atr_buffer))
                risk = entry - stop if side == "LONG" else stop - entry
                if risk > 0:
                    target = entry + risk * rr if side == "LONG" else entry - risk * rr
                    position = {"symbol": symbol, "side": side, "session": session,
                                "entry_ts": ts, "entry": entry, "stop": stop, "target": target,
                                "risk_points": risk, "contract": row.trading_symbol,
                                "expiry": str(row.expiry)[:10], "lot_size": int(row.lot_size),
                                "reference_level": float(ref)}

        if position is not None and not exited_this_bar:
            exit_price = reason = None
            if position["side"] == "LONG":
                if row.open_fut <= position["stop"]:
                    exit_price, reason = float(row.open_fut), "stop_gap"
                elif row.low_fut <= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif row.high_fut >= position["target"]:
                    exit_price, reason = position["target"], "target"
            else:
                if row.open_fut >= position["stop"]:
                    exit_price, reason = float(row.open_fut), "stop_gap"
                elif row.high_fut >= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif row.low_fut <= position["target"]:
                    exit_price, reason = position["target"], "target"
            if exit_price is None and bool(row.flat_time):
                exit_price, reason = float(row.close_fut), "intraday_flat"
            if exit_price is not None:
                trades.append(finish(position, ts, float(exit_price), reason, cost_model,
                                     slippage, int(position["lot_size"])))
                position = None
                exited_this_bar = True
            elif position["side"] == "LONG" and row.close_spot < position["reference_level"]:
                position["exit_at_open"] = True
            elif position["side"] == "SHORT" and row.close_spot > position["reference_level"]:
                position["exit_at_open"] = True

        if position is None and pending is None and not exited_this_bar:
            vol_ok = (pd.notna(row.vol_avg20_fut) and row.vol_avg20_fut > 0
                      and row.volume_fut > row.vol_avg20_fut * 1.2)
            long_ok = bool(row.long_signal) and (vol_ok or not volume_filter)
            short_ok = bool(row.short_signal) and (vol_ok or not volume_filter)
            if long_ok or short_ok:
                pending = {"side": "LONG" if long_ok else "SHORT", "session": session,
                           "signal_ts": ts, "instrument_key": row.instrument_key,
                           "signal_row": row}

    if position is not None and len(x):
        last = x.iloc[-1]
        trades.append(finish(position, last.ts, float(last.close_fut), "end_of_data",
                             cost_model, slippage, int(position["lot_size"])))
    return pd.DataFrame(trades)


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, all_trades = [], []
    for symbol, spot_name, rr, buffer in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv", 1.5, 0.1),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv", 1.5, 0.1),
    ):
        spot = load_spot(DEFAULT_SPOT / spot_name)
        futures, _ = load_front_month(symbol, DEFAULT_ROOT)
        merged = with_reference_level(make_signals(spot, futures))
        # Compare the new exit with the unchanged entry, target and stop logic.
        for volume_filter in (False, True):
            baseline = run_one(merged, symbol, volume_filter, "upstox", 1.0, rr, buffer, 1)
            candidate = replay_with_invalidation(merged, symbol, volume_filter, "upstox", 1.0, rr, buffer)
            for variant, trades in (("baseline", baseline), ("pivot_reclaim_exit", candidate)):
                trades = trades.copy()
                if not trades.empty:
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
                    stats.update(variant=variant, rr=rr, atr_buffer=buffer)
                    rows.append(stats)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {OUT}")
    print("Actual selected futures candles; estimated Upstox statutory charges + 1 point/side slippage.")
    print("Research only; no orders submitted. OOS is evaluation only and was not used to select this rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
