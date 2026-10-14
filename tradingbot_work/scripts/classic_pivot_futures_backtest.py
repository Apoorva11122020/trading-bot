#!/usr/bin/env python3
"""PAPER-only classic pivot research using real, causally selected futures OHLCV.

Signals and daily pivot levels are computed from the spot index (same basis as the existing
classic-pivot research). Fills, stop/target checks, ATR risk and volume filter use the selected
real futures contract at the same timestamp. Contracts are never price-adjusted or stitched;
indicator warm-up is computed independently within each contract. No broker calls or orders.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pivot_indicators import atr, ema, rsi
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_ROOT = ROOT / "data" / "historical" / "pivot_futures"
DEFAULT_SPOT = ROOT / "data" / "historical" / "upstox_pivot_5m"
DEFAULT_OUT = ROOT / "reports" / "pivot_futures_backtest"
START = date(2025, 10, 1)
END = date(2026, 10, 8)
OOS_START = date(2026, 6, 1)
IST = "Asia/Kolkata"


def load_spot(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing spot reference CSV: {path}")
    x = pd.read_csv(path)
    required = {"open", "high", "low", "close"}
    missing = required - set(x.columns)
    tc = "timestamp_ist" if "timestamp_ist" in x.columns else "timestamp"
    if tc not in x:
        missing.add("timestamp (or timestamp_ist)")
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    ts = pd.to_datetime(x[tc], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(IST if tc == "timestamp_ist" else "UTC")
    x["ts"] = ts.dt.tz_convert(IST)
    for c in ("open", "high", "low", "close"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["ts", "open", "high", "low", "close"])
    x = x.sort_values("ts").drop_duplicates("ts", keep="last")
    x = x[(x["ts"].dt.date >= START) & (x["ts"].dt.date <= END)].copy()
    mins = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    x = x[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)].copy()
    x["session"] = x["ts"].dt.date
    if x.empty:
        raise SystemExit(f"No spot rows in research window: {path}")
    return x.reset_index(drop=True)


def load_front_month(symbol: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = root / symbol.lower() / "futures"
    selection_path = folder / "futures_contract_selection.csv"
    contracts_path = folder / "contracts.csv"
    quality_path = folder / "data_quality.json"
    for p in (selection_path, contracts_path, quality_path):
        if not p.is_file():
            raise SystemExit(f"Required futures validation file missing: {p}")
    selection = pd.read_csv(selection_path)
    contracts = pd.read_csv(contracts_path)
    if selection.empty or contracts.empty:
        raise SystemExit(f"{symbol}: empty contract-selection or contract metadata file")
    selection["session_date"] = pd.to_datetime(selection["session_date"], errors="coerce").dt.date
    selection = selection[(selection["session_date"] >= START) & (selection["session_date"] <= END)].copy()
    if selection.empty:
        raise SystemExit(f"{symbol}: no contract-selection sessions in research window")
    # Preserve missing sessions explicitly. They are excluded from execution, never imputed;
    # the resulting report is clearly marked INCOMPLETE rather than silently presented as full coverage.
    invalid_status = ~selection["status"].isin(["SELECTED", "DATA_UNAVAILABLE"])
    if invalid_status.any():
        bad = selection.loc[invalid_status, ["session_date", "status", "reason"]].head(10)
        raise SystemExit(f"{symbol}: unknown selection status values:\n{bad}")
    unavailable = selection.loc[selection["status"] != "SELECTED", ["session_date", "status", "reason"]]
    if not unavailable.empty:
        print(f"WARNING {symbol}: {len(unavailable)} unavailable session(s); no candles will be fabricated:")
        print(unavailable.to_string(index=False))
    selected = selection.loc[selection["status"] == "SELECTED"].copy()
    if selected.empty:
        raise SystemExit(f"{symbol}: no SELECTED sessions available for backtest")
    if selection["session_date"].duplicated().any():
        raise SystemExit(f"{symbol}: duplicate session dates in selection CSV")
    metadata = contracts.set_index("instrument_key").to_dict("index")
    frames = []
    for key in selected["instrument_key"].dropna().unique():
        if key not in metadata:
            raise SystemExit(f"{symbol}: selected key missing from contracts.csv: {key}")
        expiry = str(metadata[key]["expiry"])[:10]
        file = folder / f"{symbol}_FUT_{expiry}.parquet"
        if not file.is_file():
            raise SystemExit(f"{symbol}: selected contract parquet missing: {file}")
        f = pd.read_parquet(file)
        required = {"timestamp", "instrument_key", "open", "high", "low", "close", "volume", "expiry"}
        missing = required - set(f.columns)
        if missing:
            raise SystemExit(f"{file}: missing columns {sorted(missing)}")
        f["ts"] = pd.to_datetime(f["timestamp"], errors="coerce")
        if f["ts"].dt.tz is None:
            f["ts"] = f["ts"].dt.tz_localize(IST)
        f["ts"] = f["ts"].dt.tz_convert(IST)
        for c in ("open", "high", "low", "close", "volume"):
            f[c] = pd.to_numeric(f[c], errors="coerce")
        f = f.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
        f = f.sort_values("ts").drop_duplicates("ts", keep="last")
        if not f["instrument_key"].eq(key).all():
            raise SystemExit(f"{file}: instrument_key mismatch")
        # Indicator warm-up is contract-local: no cross-expiry synthetic price series.
        f["lot_size"] = int(metadata[key]["lot_size"])
        f["atr14_fut"] = atr(f, 14)
        # Compare the current completed candle against the PREVIOUS 20 candles only.
        # Including current volume in its own baseline weakens the intended volume filter.
        f["vol_avg20_fut"] = f["volume"].shift(1).rolling(20, min_periods=20).mean()
        f["session"] = f["ts"].dt.date
        frames.append(f)
    all_bars = pd.concat(frames, ignore_index=True)
    chosen = selected[["session_date", "instrument_key"]].rename(columns={"session_date": "session"})
    # Keep only the contract explicitly selected for each session; never fall back to a later expiry.
    all_bars = all_bars.merge(chosen, on=["session", "instrument_key"], how="inner", validate="many_to_one")
    if all_bars.empty:
        raise SystemExit(f"{symbol}: no candles match the causal contract selection")
    if all_bars.duplicated(["ts"]).any():
        dup = all_bars.loc[all_bars.duplicated(["ts"], keep=False), ["ts", "instrument_key"]].head(10)
        raise SystemExit(f"{symbol}: duplicate timestamps after contract selection:\n{dup}")
    return all_bars.sort_values("ts").reset_index(drop=True), selection


def make_signals(spot: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
    x = spot.copy().sort_values("ts").reset_index(drop=True)
    # Previous completed regular-session spot H/L/C; out-of-session candles do not enter pivots.
    mins = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    regular = x.loc[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)]
    daily = regular.groupby("session", sort=True).agg(
        day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")
    )
    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily["prev_close"] = daily["day_close"].shift(1)
    daily["pp"] = (daily["prev_high"] + daily["prev_low"] + daily["prev_close"]) / 3
    daily["r1"] = 2 * daily["pp"] - daily["prev_low"]
    daily["s1"] = 2 * daily["pp"] - daily["prev_high"]
    x = x.join(daily[["pp", "r1", "s1"]], on="session")
    x["ema20_spot"] = ema(x["close"], 20)
    x["ema50_spot"] = ema(x["close"], 50)
    x["rsi14_spot"] = rsi(x["close"], 14)
    # Compare both candles against the SAME session's pivot levels. Shifting pivot
    # levels across sessions can fabricate a "cross" when the pivot itself changes.
    # Also require a prior candle in the same session; never manufacture an opening-bar cross.
    prev_close = x.groupby("session", sort=False)["close"].shift(1)
    prev_pp, prev_r1, prev_s1 = x["pp"], x["r1"], x["s1"]
    x["cross_up"] = (((x["close"] > x["pp"]) & (prev_close <= prev_pp)) |
                     ((x["close"] > x["r1"]) & (prev_close <= prev_r1)))
    x["cross_down"] = (((x["close"] < x["pp"]) & (prev_close >= prev_pp)) |
                       ((x["close"] < x["s1"]) & (prev_close >= prev_s1)))
    mins = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    x["in_session"] = (mins >= 9 * 60 + 20) & (mins < 15 * 60)
    x["flat_time"] = mins >= 15 * 60 + 15
    x["bull_ok"] = (x["ema20_spot"] > x["ema50_spot"]) & (x["rsi14_spot"] > 55)
    x["bear_ok"] = (x["ema20_spot"] < x["ema50_spot"]) & (x["rsi14_spot"] < 45)
    x["long_signal"] = x["in_session"] & x["cross_up"] & x["bull_ok"]
    x["short_signal"] = x["in_session"] & x["cross_down"] & x["bear_ok"]
    fcols = ["ts", "instrument_key", "trading_symbol", "expiry", "lot_size", "open", "high", "low", "close",
             "volume", "atr14_fut", "vol_avg20_fut"]
    merged = x.merge(futures[fcols], on="ts", how="inner", suffixes=("_spot", "_fut"), validate="one_to_one")
    if merged.empty:
        raise SystemExit("Spot/futures timestamp merge produced no rows")
    return merged.sort_values("ts").reset_index(drop=True)


def upstox_futures_fees_inr(
    entry: float, exit_price: float, side: str, lot: int, entry_date: date, exit_date: date
) -> float:
    """Estimate NSE equity-futures charges for two executed Upstox orders, one lot.

    Applies the rate effective on each transaction date: NSE transaction charge
    0.00183% per side from 2026-03-01 (0.00173% before), STT on the sell order
    0.02% before 2026-04-01 and 0.05% thereafter, SEBI fee 0.0001% per side,
    stamp duty 0.002% on the buy order, and 18% GST on brokerage + exchange +
    SEBI fees. Brokerage is capped at Rs 20 per order.
    Does not model contract-note rounding or account-specific plans.
    """
    buy_price = entry if side == "LONG" else exit_price
    sell_price = exit_price if side == "LONG" else entry
    buy_date = entry_date if side == "LONG" else exit_date
    sell_date = exit_date if side == "LONG" else entry_date
    buy = buy_price * lot
    sell = sell_price * lot
    turnover = buy + sell
    brokerage = min(20.0, buy * 0.0005) + min(20.0, sell * 0.0005)
    # Use total exchange + IPFT outflow on both sides. Before 2026-03-01 this
    # was 0.00173% transaction + 0.00010% IPFT; from that date the combined
    # rate is 0.00183%. Total cost is unchanged across the effective date.
    exchange_and_ipft = turnover * 0.0000183
    sebi = turnover * 0.000001
    stamp = buy * 0.00002
    stt = sell * (0.0005 if sell_date >= date(2026, 4, 1) else 0.0002)
    gst = (brokerage + exchange_and_ipft + sebi) * 0.18
    return brokerage + exchange_and_ipft + sebi + stamp + stt + gst


def finish(pos: dict, ts, price: float, reason: str, cost_model: str, slip: float, lot: int) -> dict:
    direction = 1 if pos["side"] == "LONG" else -1
    gross = (price - pos["entry"]) * direction
    exit_date = pd.Timestamp(ts).date()
    entry_date = pd.Timestamp(pos["entry_ts"]).date()
    fees_inr = (
        upstox_futures_fees_inr(pos["entry"], price, pos["side"], lot, entry_date, exit_date)
        if cost_model == "upstox" else 0.0
    )
    fees_points = fees_inr / lot
    cost = 2 * slip + fees_points
    net = gross - cost
    return {
        "symbol": pos["symbol"], "side": pos["side"], "entry_ts": pos["entry_ts"], "exit_ts": ts,
        "contract": pos["contract"], "expiry": pos["expiry"], "lot_size": lot,
        "entry": pos["entry"], "exit": price, "stop": pos["stop"], "target": pos["target"],
        "exit_reason": reason, "gross_points": round(gross, 4), "estimated_fees_inr": round(fees_inr, 2),
        "estimated_cost_points": round(cost, 4),
        "net_points": round(net, 4), "approx_net_inr_1_lot": round(net * lot, 2),
    }


def run_one(df: pd.DataFrame, symbol: str, volume_filter: bool, cost_model: str, slip: float,
            rr: float, atr_buffer: float, lot: int) -> pd.DataFrame:
    """Simulate signals at candle close and enter at the next same-session 5-minute bar open."""
    x = df.copy().sort_values("ts").reset_index(drop=True)
    trades, pos, pending = [], None, None

    for i, row in x.iterrows():
        ts, session = row["ts"], row["session"]

        # Never carry positions or pending entries across sessions.
        if pos is not None and session != pos["session"]:
            prev = x.iloc[i - 1]
            trades.append(finish(pos, prev["ts"], float(prev["close_fut"]), "session_boundary",
                                 cost_model, slip, int(pos["lot_size"])))
            pos = None
        if pending is not None and (
            session != pending["session"]
            or row["instrument_key"] != pending["instrument_key"]
            or (ts - pending["signal_ts"]).total_seconds() > 5 * 60 + 1
        ):
            pending = None

        # A signal is only known after its candle closes; execute at the next bar's open.
        if pos is None and pending is not None:
            entry = float(row["open_fut"])
            signal_row = pending["signal_row"]
            side = pending["side"]
            a = signal_row["atr14_fut"]
            pending = None
            if pd.notna(a) and float(a) > 0 and np.isfinite(entry):
                if side == "LONG":
                    stop = float(signal_row["low_fut"] - float(a) * atr_buffer)
                    risk = entry - stop
                    target = entry + risk * rr
                else:
                    stop = float(signal_row["high_fut"] + float(a) * atr_buffer)
                    risk = stop - entry
                    target = entry - risk * rr
                if np.isfinite(risk) and risk > 0:
                    pos = {
                        "symbol": symbol, "side": side, "session": session, "entry_ts": ts,
                        "entry": entry, "stop": stop, "target": target,
                        "contract": row["trading_symbol"], "expiry": str(row["expiry"])[:10],
                        "lot_size": int(row["lot_size"]),
                    }

        exited_this_bar = False
        if pos is not None:
            exit_price, reason = None, None
            # Conservative OHLC handling: if both stop and target are touched, assume stop first.
            # Account for adverse gaps through the stop at the current bar's open.
            if pos["side"] == "LONG":
                if row["open_fut"] <= pos["stop"]:
                    exit_price, reason = float(row["open_fut"]), "stop_gap"
                elif row["low_fut"] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif row["high_fut"] >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            else:
                if row["open_fut"] >= pos["stop"]:
                    exit_price, reason = float(row["open_fut"]), "stop_gap"
                elif row["high_fut"] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif row["low_fut"] <= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            if exit_price is None and bool(row["flat_time"]):
                exit_price, reason = float(row["close_fut"]), "intraday_flat"
            if exit_price is not None:
                trades.append(finish(pos, ts, float(exit_price), reason, cost_model, slip,
                                     int(pos["lot_size"])))
                pos = None
                exited_this_bar = True

        # Queue a new signal after this bar closes; it can only fill on the next bar.
        if pos is None and not exited_this_bar and pending is None:
            vol_ok = (
                pd.notna(row["vol_avg20_fut"])
                and row["vol_avg20_fut"] > 0
                and row["volume_fut"] > row["vol_avg20_fut"] * 1.2
            )
            long_signal = bool(row["long_signal"]) and (vol_ok or not volume_filter)
            short_signal = bool(row["short_signal"]) and (vol_ok or not volume_filter)
            if long_signal or short_signal:
                pending = {
                    "side": "LONG" if long_signal else "SHORT",
                    "session": session,
                    "signal_ts": ts,
                    "instrument_key": row["instrument_key"],
                    "signal_row": row,
                }

    if pos is not None and len(x):
        row = x.iloc[-1]
        trades.append(finish(pos, row["ts"], float(row["close_fut"]), "end_of_data",
                             cost_model, slip, int(pos["lot_size"])))
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, symbol: str, filter_on: bool, split: str) -> dict:
    if trades.empty:
        return {"symbol": symbol, "volume_filter": filter_on, "split": split, "trades": 0,
                "net_points": 0.0, "approx_net_inr_1_lot": 0.0, "win_rate_pct": 0.0,
                "profit_factor": None, "max_drawdown_points": 0.0, "avg_trade_points": 0.0}
    t = trades.sort_values("exit_ts")
    wins = float(t.loc[t["net_points"] > 0, "net_points"].sum())
    losses = float(-t.loc[t["net_points"] < 0, "net_points"].sum())
    eq = t["net_points"].cumsum()
    dd = eq - eq.cummax().clip(lower=0)
    return {"symbol": symbol, "volume_filter": filter_on, "split": split, "trades": int(len(t)),
            "net_points": round(float(t["net_points"].sum()), 2),
            "approx_net_inr_1_lot": round(float(t["approx_net_inr_1_lot"].sum()), 2),
            "win_rate_pct": round(float((t["net_points"] > 0).mean() * 100), 2),
            "profit_factor": round(wins / losses, 3) if losses > 0 else None,
            "max_drawdown_points": round(float(-dd.min()), 2),
            "avg_trade_points": round(float(t["net_points"].mean()), 3)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--spot-root", type=Path, default=DEFAULT_SPOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cost-model", choices=("upstox", "zero"), default="upstox",
                   help="Upstox NSE futures estimated statutory charges or zero-fee diagnostic")
    p.add_argument("--slippage-points", type=float, default=2.0, help="assumed points per side")
    p.add_argument("--rr", type=float, default=1.5)
    p.add_argument("--atr-buffer", type=float, default=0.1)
    a = p.parse_args()
    assert_paper_trading_only()
    if a.slippage_points < 0 or a.rr <= 0 or a.atr_buffer < 0:
        p.error("costs/buffer must be nonnegative and rr must be positive")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, all_trades, coverage_rows = [], [], []
    for symbol, spot_name in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv"),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv"),
    ):
        spot = load_spot(a.spot_root / spot_name)
        futures, selection = load_front_month(symbol, a.data_root)
        unavailable = selection.loc[selection["status"] != "SELECTED", ["session_date", "status", "reason"]].copy()
        if unavailable.empty:
            coverage_rows.append({"symbol": symbol, "coverage_status": "COMPLETE",
                                  "unavailable_sessions": 0, "unavailable_dates": "", "reasons": ""})
        else:
            coverage_rows.append({
                "symbol": symbol, "coverage_status": "INCOMPLETE_DATA_UNAVAILABLE",
                "unavailable_sessions": len(unavailable),
                "unavailable_dates": ";".join(str(d) for d in unavailable["session_date"]),
                "reasons": ";".join(str(r) for r in unavailable["reason"]),
            })
        merged = make_signals(spot, futures)
        print(f"{symbol}: spot={len(spot)} bars, selected futures={len(futures)} bars, "
              f"selection_sessions={len(selection)}, aligned={len(merged)} bars, "
              f"coverage={'INCOMPLETE' if not unavailable.empty else 'COMPLETE'}")
        # Restrict to the dates covered by the verified causal selection and matching spot bars.
        for vf in (False, True):
            trades = run_one(merged, symbol, vf, a.cost_model, a.slippage_points, a.rr, a.atr_buffer, 1)
            if not trades.empty:
                trades["volume_filter"] = vf
                all_trades.append(trades)
            for split, start, end in (
                ("DEVELOPMENT", START, OOS_START.fromordinal(OOS_START.toordinal() - 1)),
                ("OOS", OOS_START, END),
                ("FULL", START, END),
            ):
                st = pd.to_datetime(trades["entry_ts"]).dt.date if not trades.empty else pd.Series([], dtype=object)
                et = pd.to_datetime(trades["exit_ts"]).dt.date if not trades.empty else pd.Series([], dtype=object)
                if trades.empty:
                    subset = trades.copy()
                else:
                    # Both entry and exit must belong to the same split; never let a trade straddle the boundary.
                    subset = trades[(st >= start) & (et <= end)].copy()
                summaries.append(summarize(subset, symbol, vf, split))
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = a.out_dir / "data_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    summary_df = pd.DataFrame(summaries)
    if not coverage_df.empty:
        summary_df = summary_df.merge(
            coverage_df[["symbol", "coverage_status", "unavailable_sessions", "unavailable_dates", "reasons"]],
            on="symbol", how="left", validate="many_to_one",
        )
    summary_path = a.out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(a.out_dir / "trades.csv", index=False)
    print("\nSUMMARY (estimated costs; INR is approximate one-lot P&L)")
    print(summary_df.to_string(index=False))
    print(f"\nData coverage: {coverage_path.resolve()}")
    print(f"Reports: {a.out_dir.resolve()}")
    print("PAPER | RESEARCH ONLY | ORDERS: 0")
    print("Caveat: signals use spot-index pivots/EMA/RSI; fills, ATR stops and volume use selected futures. "
          "Fees use estimated Upstox NSE futures rates, not contract-note exact; check historical rate changes. OOS is 2026-06-01 onward.")
    if any(r["coverage_status"] != "COMPLETE" for r in coverage_rows):
        print("IMPORTANT: INCOMPLETE DATA COVERAGE. Do not treat these metrics as a fully covered backtest; "
              "see data_coverage.csv. Missing sessions were not imputed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
