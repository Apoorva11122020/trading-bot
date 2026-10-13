"""Free, reproducible Pine-vs-Python candidate signal audit.

This script audits the supplied 5-minute OHLCV CSVs without TradingView's paid
chart export. It computes Python-side candidate signals and data-quality checks.
It does NOT claim exact TradingView parity; that requires TradingView output for
the same symbol/session/timeframe.

Example:
  python scripts/free_pine_parity_audit.py \
    --nifty path/to/NIFTY_50_5m.csv \
    --bank-nifty path/to/BANK_NIFTY_5m.csv \
    --out-dir reports/free_pine_parity

No broker APIs or live orders are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.pivot_indicators import atr, ema, rsi
except ModuleNotFoundError:  # direct execution from scripts/
    from pivot_indicators import atr, ema, rsi

IST = "Asia/Kolkata"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(IST)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    return df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)


def audit_symbol(
    df: pd.DataFrame,
    symbol: str,
    volume_filter: bool,
    session_end: str,
    atr_buffer: float,
    rr: float,
    min_tick: float,
) -> tuple[pd.DataFrame, dict]:
    x = df.copy()
    x["session"] = x["ts"].dt.date
    minute_of_day = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    regular = x.loc[(minute_of_day >= 9 * 60 + 15) & (minute_of_day < 15 * 60 + 30)]
    daily = regular.groupby("session", sort=True).agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
    )
    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily["prev_close"] = daily["day_close"].shift(1)
    daily["pp"] = (daily["prev_high"] + daily["prev_low"] + daily["prev_close"]) / 3.0
    daily["r1"] = 2.0 * daily["pp"] - daily["prev_low"]
    daily["s1"] = 2.0 * daily["pp"] - daily["prev_high"]
    x = x.join(daily[["pp", "r1", "s1"]], on="session")

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["rsi14"] = rsi(x["close"], 14)
    x["atr14"] = atr(x, 14)
    x["vol_avg20"] = x["volume"].rolling(20, min_periods=20).mean()

    previous_close = x["close"].shift(1)
    previous_pp = x["pp"].shift(1)
    previous_r1 = x["r1"].shift(1)
    previous_s1 = x["s1"].shift(1)
    x["cross_up"] = (
        ((x["close"] > x["pp"]) & (previous_close <= previous_pp))
        | ((x["close"] > x["r1"]) & (previous_close <= previous_r1))
    )
    x["cross_down"] = (
        ((x["close"] < x["pp"]) & (previous_close >= previous_pp))
        | ((x["close"] < x["s1"]) & (previous_close >= previous_s1))
    )

    try:
        end_hour, end_minute = (int(part) for part in session_end.split(":"))
        if not (0 <= end_hour <= 23 and 0 <= end_minute <= 59):
            raise ValueError
    except ValueError as exc:
        raise ValueError("--session-end must be HH:MM, e.g. 15:00") from exc
    start_minute = 9 * 60 + 20
    end_minute_of_day = end_hour * 60 + end_minute
    x["in_session"] = (minute_of_day >= start_minute) & (minute_of_day < end_minute_of_day)

    x["long_filter"] = (x["ema20"] > x["ema50"]) & (x["rsi14"] > 55)
    x["short_filter"] = (x["ema20"] < x["ema50"]) & (x["rsi14"] < 45)
    x["volume_ok"] = x["volume"].gt(x["vol_avg20"] * 1.2) & x["vol_avg20"].notna()
    if volume_filter:
        x["long_filter"] &= x["volume_ok"]
        x["short_filter"] &= x["volume_ok"]

    x["long_candidate"] = (
        x["in_session"] & x["cross_up"] & x["long_filter"]
    )
    x["short_candidate"] = (
        x["in_session"] & x["cross_down"] & x["short_filter"]
    )
    x["long_risk"] = x["close"] - (x["low"] - x["atr14"] * atr_buffer)
    x["short_risk"] = (x["high"] + x["atr14"] * atr_buffer) - x["close"]
    x["long_valid"] = x["long_candidate"] & x["long_risk"].gt(min_tick) & x["atr14"].notna()
    x["short_valid"] = x["short_candidate"] & x["short_risk"].gt(min_tick) & x["atr14"].notna()
    x["long_stop"] = x["low"] - x["atr14"] * atr_buffer
    x["long_target"] = x["close"] + x["long_risk"] * rr
    x["short_stop"] = x["high"] + x["atr14"] * atr_buffer
    x["short_target"] = x["close"] - x["short_risk"] * rr
    x["symbol"] = symbol

    signal_cols = [
        "symbol", "ts", "open", "high", "low", "close", "volume",
        "pp", "r1", "s1", "ema20", "ema50", "rsi14", "atr14", "vol_avg20",
        "cross_up", "cross_down", "long_filter", "short_filter", "volume_ok",
        "long_candidate", "short_candidate", "long_valid", "short_valid",
        "long_stop", "long_target", "short_stop", "short_target",
    ]
    candidates = x.loc[x["long_candidate"] | x["short_candidate"], signal_cols].copy()

    session_bars = regular.groupby("session").size()
    ohlc_invalid = (
        (x["high"] < x["low"])
        | (x["high"] < x["open"])
        | (x["high"] < x["close"])
        | (x["low"] > x["open"])
        | (x["low"] > x["close"])
    )
    summary = {
        "symbol": symbol,
        "bars": int(len(x)),
        "first_bar_ist": str(x["ts"].min()),
        "last_bar_ist": str(x["ts"].max()),
        "sessions": int(x["session"].nunique()),
        "duplicate_timestamps_removed": int(len(df) - df["ts"].nunique()),
        "missing_ohlc_volume_rows_removed": 0,
        "ohlc_invalid_rows": int(ohlc_invalid.sum()),
        "non_regular_session_bars": int((~((minute_of_day >= 555) & (minute_of_day < 930))).sum()),
        "sessions_not_75_bars": int((session_bars != 75).sum()),
        "short_sessions": ";".join(f"{day}:{count}" for day, count in session_bars[session_bars != 75].items()),
        "volume_nonzero_bars": int(x["volume"].gt(0).sum()),
        "long_candidates_volume_off": int((x["in_session"] & x["cross_up"] & (x["ema20"] > x["ema50"]) & (x["rsi14"] > 55)).sum()),
        "short_candidates_volume_off": int((x["in_session"] & x["cross_down"] & (x["ema20"] < x["ema50"]) & (x["rsi14"] < 45)).sum()),
        "long_candidates_configured": int(x["long_candidate"].sum()),
        "short_candidates_configured": int(x["short_candidate"].sum()),
        "long_valid_tick_guard": int(x["long_valid"].sum()),
        "short_valid_tick_guard": int(x["short_valid"].sum()),
        "volume_filter": volume_filter,
        "session_end": session_end,
        "min_tick": min_tick,
    }
    return candidates, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Free Python-side Pivot Algo parity/data-quality audit; no live orders.")
    parser.add_argument("--nifty", type=Path, required=True)
    parser.add_argument("--bank-nifty", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/free_pine_parity"))
    parser.add_argument("--volume-filter", action="store_true", help="Enable volume confirmation (zero-volume files will normally produce no signals).")
    parser.add_argument("--session-end", default="15:00", help="Signal entry window end, local exchange time (default: 15:00 for Strategy appendix).")
    parser.add_argument("--atr-buffer", type=float, default=0.1)
    parser.add_argument("--rr", type=float, default=1.5)
    parser.add_argument("--min-tick", type=float, default=0.05)
    args = parser.parse_args()
    if args.atr_buffer < 0 or args.rr <= 0 or args.min_tick <= 0:
        parser.error("--atr-buffer must be >= 0, --rr and --min-tick must be > 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_candidates = []
    summaries = []
    for symbol, path in [("NIFTY 50", args.nifty), ("BANK NIFTY", args.bank_nifty)]:
        raw = pd.read_csv(path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        initial_rows = len(raw)
        raw["ts"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.tz_convert(IST)
        for col in ["open", "high", "low", "close", "volume"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        cleaned = raw.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
        cleaned = cleaned.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
        candidates, summary = audit_symbol(
            cleaned, symbol, args.volume_filter, args.session_end,
            args.atr_buffer, args.rr, args.min_tick,
        )
        summary["input_rows"] = initial_rows
        summary["rows_removed_during_cleaning"] = initial_rows - len(cleaned)
        all_candidates.append(candidates)
        summaries.append(summary)

    pd.concat(all_candidates, ignore_index=True).to_csv(args.out_dir / "free_pine_parity_candidate_signals.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "free_pine_parity_audit_summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nWrote audit files to: {args.out_dir.resolve()}")
    print("NOTE: This is Python-side candidate auditing, not proof of exact TradingView parity.")


if __turn__ == "__main__":
    main()
