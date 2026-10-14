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
    if selection.empty or not selection["status"].eq("SELECTED").all():
        bad = selection.loc[selection["status"] != "SELECTED", ["session_date", "status", "reason"]].head(10)
        raise SystemExit(f"{symbol}: selection has unavailable sessions; refusing partial backtest:\n{bad}")
    if selection["session_date"].duplicated().any():
        raise SystemExit(f"{symbol}: duplicate session dates in selection CSV")
    metadata = contracts.set_index("instrument_key").to_dict("index")
    frames = []
    for key in selection["instrument_key"].dropna().unique():
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
        f["atr14_fut"] = atr(f, 14)
        f["vol_avg20_fut"] = f["volume"].rolling(20, min_periods=20).mean()
        f["session"] = f["ts"].dt.date
        frames.append(f)
    all_bars = pd.concat(frames, ignore_index=True)
    chosen = selection[["session_date", "instrument_key"]].rename(columns={"session_date": "session"})
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
    prev_close = x["close"].shift(1)
    prev_pp, prev_r1, prev_s1 = x["pp"].shift(1), x["r1"].shift(1), x["s1"].shift(1)
    x["cross_up"] = (((x["close"] > x["pp"]) & (prev_close <= prev_pp)) |
                     ((x["close"] > x["r1"]) & (prev_close <= prev_r1)))
    x["cross_down"] = (((x["close"] < x["pp"]) & (prev_close >= prev_pp)) |
                       ((x["close"] < x["s1"]) & (prev_close >= prev_s1)))
    mins = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    x["in_session"] = (mins >= 9 * 60 + 20) & (mins < 15 * 60)
    x["flat_time"] = mins >= 15 * 60
    x["bull_ok"] = (x["ema20_spot"] > x["ema50_spot"]) & (x["rsi14_spot"] > 55)
    x["bear_ok"] = (x["ema20_spot"] < x["ema50_spot"]) & (x["rsi14_spot"] < 45)
    x["long_signal"] = x["in_session"] & x["cross_up"] & x["bull_ok"]
    x["short_signal"] = x["in_session"] & x["cross_down"] & x["bear_ok"]
    fcols = ["ts", "instrument_key", "trading_symbol", "expiry", "open", "high", "low", "close",
             "volume", "atr14_fut", "vol_avg20_fut"]
    merged = x.merge(futures[fcols], on="ts", how="inner", suffixes=("_spot", "_fut"), validate="one_to_one")
    if merged.empty:
        raise SystemExit("Spot/futures timestamp merge produced no rows")
    return merged.sort_values("ts").reset_index(drop=True)


def finish(pos: dict, ts, price: float, reason: str, commission_pct: float, slip: float, lot: int) -> dict:
    direction = 1 if pos["side"] == "LONG" else -1
    gross = (price - pos["entry"]) * direction
    fees_points = (pos["entry"] + price) * commission_pct / 100.0
    cost = 2 * slip + fees_points
    net = gross - cost
    return {
        "symbol": pos["symbol"], "side": pos["side"], "entry_ts": pos["entry_ts"], "exit_ts": ts,
        "contract": pos["contract"], "expiry": pos["expiry"], "lot_size": lot,
        "entry": pos["entry"], "exit": price, "stop": pos["stop"], "target": pos["target"],
        "exit_reason": reason, "gross_points": round(gross, 4), "estimated_cost_points": round(cost, 4),
        "net_points": round(net, 4), "approx_net_inr_1_lot": round(net * lot, 2),
    }


def run_one(df: pd.DataFrame, symbol: str, volume_filter: bool, commission_pct: float, slip: float,
            rr: float, atr_buffer: float, lot: int) -> pd.DataFrame:
    x = df.copy()
    trades, pos = [], None
    for i, row in x.iterrows():
        ts, session = row["ts"], row["session"]
        if pos is not None and session != pos["session"]:
            prev = x.iloc[i - 1]
            trades.append(finish(pos, prev["ts"], float(prev["close"]), "session_boundary",
                                 commission_pct, slip, lot))
            pos = None
        if pos is not None:
            exit_price, reason = None, None
            if pos["side"] == "LONG":
                if row["low"] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif row["high"] >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            else:
                if row["high"] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop"
                elif row["low"] <= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            if exit_price is None and bool(row["flat_time"]):
                exit_price, reason = float(row["close"]), "intraday_flat"
            if exit_price is not None:
                trades.append(finish(pos, ts, float(exit_price), reason, commission_pct, slip, lot))
                pos = None
        if pos is None and not (trades and trades[-1]["exit_ts"] == ts):
            vol_ok = (pd.notna(row["vol_avg20_fut"]) and row["vol_avg20_fut"] > 0 and
                      row["volume"] > row["vol_avg20_fut"] * 1.2)
            long_signal = bool(row["long_signal"]) and (vol_ok or not volume_filter)
            short_signal = bool(row["short_signal"]) and (vol_ok or not volume_filter)
            if long_signal or short_signal:
                side = "LONG" if long_signal else "SHORT"
                entry, a = float(row["close"]), row["atr14_fut"]
                if pd.notna(a) and float(a) > 0:
                    if side == "LONG":
                        stop = float(row["low"] - a * atr_buffer)
                        risk = entry - stop
                        target = entry + risk * rr
                    else:
                        stop = float(row["high"] + a * atr_buffer)
                        risk = stop - entry
                        target = entry - risk * rr
                    if np.isfinite(risk) and risk > 0:
                        pos = {"symbol": symbol, "side": side, "session": session, "entry_ts": ts,
                               "entry": entry, "stop": stop, "target": target,
                               "contract": row["trading_symbol"], "expiry": str(row["expiry"])[:10]}
    if pos is not None and len(x):
        row = x.iloc[-1]
        trades.append(finish(pos, row["ts"], float(row["close"]), "end_of_data",
                             commission_pct, slip, lot))
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
    p.add_argument("--commission-pct", type=float, default=0.03,
                   help="estimated commission percent per side (not broker-accurate)")
    p.add_argument("--slippage-points", type=float, default=2.0, help="assumed points per side")
    p.add_argument("--rr", type=float, default=1.5)
    p.add_argument("--atr-buffer", type=float, default=0.1)
    a = p.parse_args()
    assert_paper_trading_only()
    if a.commission_pct < 0 or a.slippage_points < 0 or a.rr <= 0 or a.atr_buffer < 0:
        p.error("costs/buffer must be nonnegative and rr must be positive")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, all_trades = [], []
    for symbol, spot_name, lot in (
        ("NIFTY", "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv", 75),
        ("BANKNIFTY", "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv", 35),
    ):
        spot = load_spot(a.spot_root / spot_name)
        futures, selection = load_front_month(symbol, a.data_root)
        merged = make_signals(spot, futures)
        print(f"{symbol}: spot={len(spot)} bars, selected futures={len(futures)} bars, "
              f"selection_sessions={len(selection)}, aligned={len(merged)} bars")
        # Restrict to the dates covered by the verified causal selection and matching spot bars.
        for vf in (False, True):
            trades = run_one(merged, symbol, vf, a.commission_pct, a.slippage_points, a.rr, a.atr_buffer, lot)
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
    summary_path = a.out_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(a.out_dir / "trades.csv", index=False)
    print("\nSUMMARY (estimated costs; INR is approximate one-lot P&L)")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nReports: {a.out_dir.resolve()}")
    print("PAPER | RESEARCH ONLY | ORDERS: 0")
    print("Caveat: signals use spot-index pivots/EMA/RSI; fills, ATR stops and volume use selected futures. "
          "Fees are estimates, not broker-accurate. OOS is 2026-06-01 onward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
