from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path("data/historical/upstox_pivot_5m")
DEFAULT_OUT = Path("reports/pivot_short_oos")


try:
    from scripts.pivot_indicators import atr, ema, rma, rsi, classic_pivots_from_intraday
except ModuleNotFoundError:  # Support direct execution as python scripts/<file>.py
    from pivot_indicators import atr, ema, rma, rsi, classic_pivots_from_intraday


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["session"] = df["ts"].dt.date
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return (
        df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )


def prepare(df: pd.DataFrame, volume_filter: bool, min_session_bars: int = 75) -> pd.DataFrame:
    x = df.copy()
    x = x.join(classic_pivots_from_intraday(x, min_session_bars=min_session_bars), on="session")

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["rsi14"] = rsi(x["close"], 14)
    x["atr14"] = atr(x, 14)
    x["vol_avg20"] = x["volume"].rolling(20, min_periods=20).mean()

    prev_close = x["close"].shift(1)
    prev_pp = x["pp"].shift(1)
    prev_s1 = x["s1"].shift(1)
    x["cross_down"] = (
        ((x["close"] < x["pp"]) & (prev_close >= prev_pp))
        | ((x["close"] < x["s1"]) & (prev_close >= prev_s1))
    )

    minutes = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    x["in_session"] = (minutes >= 9 * 60 + 20) & (minutes < 15 * 60)
    x["flat_time"] = minutes >= 15 * 60
    x["bear_ok"] = (x["ema20"] < x["ema50"]) & (x["rsi14"] < 45)
    if volume_filter:
        vok = x["vol_avg20"].gt(0) & x["volume"].gt(x["vol_avg20"] * 1.2)
        x["bear_ok"] &= vok

    x["short_signal"] = x["in_session"] & x["cross_down"] & x["bear_ok"]
    return x


def finish(
    pos: dict,
    exit_ts,
    exit_price: float,
    reason: str,
    commission_pct: float,
    slippage_points: float,
) -> dict:
    raw_points = pos["entry"] - exit_price  # SHORT P&L
    commission_points = (pos["entry"] + exit_price) * (commission_pct / 100.0)
    net_points = raw_points - 2 * slippage_points - commission_points
    return {
        "symbol": pos["symbol"],
        "side": "SHORT",
        "entry_ts": pos["entry_ts"],
        "exit_ts": exit_ts,
        "entry": pos["entry"],
        "exit": exit_price,
        "stop": pos["stop"],
        "target": pos["target"],
        "risk_points": pos["risk_points"],
        "exit_reason": reason,
        "volume_filter": pos["volume_filter"],
        "gross_pnl_points": round(raw_points, 4),
        "estimated_cost_points": round(2 * slippage_points + commission_points, 4),
        "net_pnl_points": round(net_points, 4),
        "net_pnl_rupees_qty1_approx": round(net_points, 2),
    }


def backtest(
    df: pd.DataFrame,
    symbol: str,
    volume_filter: bool,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    commission_pct: float,
    slippage_points: float,
    atr_buffer: float = 0.1,
    rr: float = 1.5,
    min_tick: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    # Indicators/signals are calculated on the full history to preserve warm-up.
    # Only entries are restricted to the requested date window; open trades are
    # force-closed at the requested window end. The Pine Strategy appendix closes at 15:00.
    x = prepare(df, volume_filter)
    start_ts = pd.Timestamp(start_date).date()
    end_ts = pd.Timestamp(end_date).date()
    x["in_test_window"] = x["ts"].dt.date.between(start_ts, end_ts)
    minutes = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    x["flat_window_end"] = (
        x["ts"].dt.date.eq(end_ts) & (minutes >= 15 * 60)
    )

    trades = []
    pos = None
    last_trade_exit_ts = None

    for i, row in x.iterrows():
        ts = row["ts"]

        if pos is not None and row["session"] != pos["session"]:
            prev = x.iloc[i - 1]
            trades.append(
                finish(pos, prev["ts"], float(prev["close"]), "session_boundary",
                       commission_pct, slippage_points)
            )
            last_trade_exit_ts = prev["ts"]
            pos = None

        # Process exits before considering a new entry on this candle.
        if pos is not None:
            exit_price = None
            reason = None

            # Conservative OHLC assumption: stop wins if stop and target both hit.
            hit_stop = row["high"] >= pos["stop"]
            hit_target = row["low"] <= pos["target"]

            if float(row["open"]) >= pos["stop"]:
                exit_price, reason = float(row["open"]), "stop_gap"
            elif hit_stop:
                exit_price, reason = pos["stop"], "stop"
            elif hit_target:
                exit_price, reason = pos["target"], "target"
            elif bool(row["flat_window_end"]):
                exit_price, reason = float(row["close"]), "test_window_flat"
            elif bool(row["flat_time"]):
                exit_price, reason = float(row["close"]), "intraday_flat"

            if exit_price is not None:
                trades.append(
                    finish(
                        pos, ts, float(exit_price), reason,
                        commission_pct, slippage_points
                    )
                )
                last_trade_exit_ts = ts
                pos = None

        if (
            pos is None
            and bool(row["in_test_window"])
            and bool(row["short_signal"])
            and last_trade_exit_ts != ts
        ):
            entry = float(row["close"])
            current_atr = row["atr14"]
            if pd.notna(current_atr):
                stop = float(row["high"] + current_atr * atr_buffer)
                risk = stop - entry
                target = entry - risk * rr
                if risk > min_tick and np.isfinite(risk):
                    pos = {
                        "symbol": symbol,
                        "session": row["session"],
                        "entry_ts": ts,
                        "entry": entry,
                        "stop": stop,
                        "target": target,
                        "risk_points": risk,
                        "volume_filter": volume_filter,
                    }

    # Close at the last candle *within the requested end date*, never a later
    # row outside the test window. This also covers datasets ending before 15:00.
    if pos is not None:
        eligible = x.loc[x["ts"].dt.date.between(start_ts, end_ts)]
        if not eligible.empty:
            row = eligible.iloc[-1]
            trades.append(
                finish(pos, row["ts"], float(row["close"]), "test_window_end",
                       commission_pct, slippage_points)
            )

    t = pd.DataFrame(trades)
    if t.empty:
        summary = {
            "symbol": symbol, "side_filter": "SHORT_ONLY",
            "volume_filter": volume_filter, "start_date": str(start_ts),
            "end_date": str(end_ts), "trades": 0, "net_pnl_points": 0.0,
            "win_rate_pct": 0.0, "profit_factor": None,
            "max_drawdown_points": 0.0, "avg_trade_points": 0.0,
        }
        return t, summary

    winning_sum = t.loc[t["net_pnl_points"] > 0, "net_pnl_points"].sum()
    losing_sum = -t.loc[t["net_pnl_points"] < 0, "net_pnl_points"].sum()
    cumulative = t["net_pnl_points"].cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    summary = {
        "symbol": symbol, "side_filter": "SHORT_ONLY",
        "volume_filter": volume_filter, "start_date": str(start_ts),
        "end_date": str(end_ts), "trades": int(len(t)),
        "net_pnl_points": round(float(t["net_pnl_points"].sum()), 2),
        "win_rate_pct": round(float((t["net_pnl_points"] > 0).mean() * 100), 2),
        "profit_factor": round(float(winning_sum / losing_sum), 3) if losing_sum else None,
        "max_drawdown_points": round(float(-drawdown.min()), 2),
        "avg_trade_points": round(float(t["net_pnl_points"].mean()), 3),
        "first_entry": str(t["entry_ts"].min()),
        "last_exit": str(t["exit_ts"].max()),
    }
    return t, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHORT-only, date-split historical research backtest. No live orders."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--commission-pct", type=float, default=0.0)
    parser.add_argument("--slippage-points", type=float, default=0.0)
    parser.add_argument("--min-tick", type=float, default=0.05)
    parser.add_argument("--development-start", default="2025-10-01")
    parser.add_argument("--development-end", default="2026-05-31")
    parser.add_argument("--test-start", default="2026-06-01")
    parser.add_argument("--test-end", default="2026-10-08")
    args = parser.parse_args()

    if args.slippage_points < 0 or args.commission_pct < 0:
        parser.error("commission and slippage must be >= 0")
    if args.min_tick <= 0:
        parser.error("min-tick must be > 0")

    dev_start = pd.Timestamp(args.development_start).date()
    dev_end = pd.Timestamp(args.development_end).date()
    test_start = pd.Timestamp(args.test_start).date()
    test_end = pd.Timestamp(args.test_end).date()
    if dev_start > dev_end or test_start > test_end:
        parser.error("start dates must be on or before end dates")
    if dev_end >= test_start:
        parser.error("development period must end before test period starts")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_files = {
        "NIFTY_50": args.data_root / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv",
        "BANK_NIFTY": args.data_root / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv",
    }

    summaries = []
    all_trades = []
    for symbol, path in data_files.items():
        if not path.exists():
            raise SystemExit(f"CSV not found: {path}")
        raw = load_data(path)

        for period_name, start, end in (
            ("development", dev_start, dev_end),
            ("out_of_sample", test_start, test_end),
        ):
            # Volume filter OFF by default in analysis because source volume is zero.
            trades, summary = backtest(
                raw, symbol, False, start, end,
                args.commission_pct, args.slippage_points, min_tick=args.min_tick
            )
            summary["period"] = period_name
            summaries.append(summary)
            if not trades.empty:
                trades = trades.copy()
                trades["period"] = period_name
                all_trades.append(trades)

            print(
                f"{symbol} | {period_name} ({start} to {end}) | SHORT_ONLY: "
                f"trades={summary['trades']}, net_points={summary['net_pnl_points']}, "
                f"win_rate={summary['win_rate_pct']}%, PF={summary['profit_factor']}, "
                f"maxDD_points={summary['max_drawdown_points']}"
            )

    pd.DataFrame(summaries).to_csv(args.out_dir / "summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(args.out_dir / "trades.csv", index=False)

    print(f"\nSaved reports to: {args.out_dir.resolve()}")
    print("RESEARCH ONLY. Results are approximate spot-index points, not broker-accurate P&L.")
    print("No live trading or order placement is implemented.")
    print("If stop and target both touch in one candle, stop is counted first.")


if __name__ == "__main__":
    main()
