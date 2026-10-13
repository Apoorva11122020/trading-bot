from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path("data/historical/upstox_pivot_5m")
DEFAULT_OUT = Path("reports/pivot_short_only_research")


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask(al.eq(0) & ag.gt(0), 100)
    out = out.mask(ag.eq(0) & al.gt(0), 0)
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    # CSV timestamps are UTC; convert to India time for session calculations.
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["ts"] = ts
    df["session"] = ts.dt.date
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


def prepare(df: pd.DataFrame, volume_filter: bool) -> pd.DataFrame:
    x = df.copy()

    # Daily H/L/C from this dataset; shift one full session to avoid lookahead.
    daily = x.groupby("session", sort=True).agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_close=("close", "last"),
    )
    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily["prev_close"] = daily["day_close"].shift(1)
    daily["pp"] = (
        daily["prev_high"] + daily["prev_low"] + daily["prev_close"]
    ) / 3
    daily["r1"] = 2 * daily["pp"] - daily["prev_low"]
    daily["s1"] = 2 * daily["pp"] - daily["prev_high"]
    x = x.join(daily[["pp", "r1", "s1"]], on="session")

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["rsi14"] = rsi(x["close"], 14)
    x["atr14"] = atr(x, 14)
    x["vol_avg20"] = x["volume"].rolling(20, min_periods=20).mean()

    prev_close = x["close"].shift(1)
    prev_r1 = x["r1"].shift(1)
    prev_s1 = x["s1"].shift(1)
    prev_pp = x["pp"].shift(1)

    x["cross_up"] = (
        ((x["close"] > x["pp"]) & (prev_close <= prev_pp))
        | ((x["close"] > x["r1"]) & (prev_close <= prev_r1))
    )
    x["cross_down"] = (
        ((x["close"] < x["pp"]) & (prev_close >= prev_pp))
        | ((x["close"] < x["s1"]) & (prev_close >= prev_s1))
    )

    # Entry window: 09:20 through 14:55; flatten at/after 15:15 IST.
    minutes = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    x["in_session"] = (minutes >= 9 * 60 + 20) & (minutes < 15 * 60)
    x["flat_time"] = minutes >= 15 * 60 + 15
    x["bear_ok"] = (x["ema20"] < x["ema50"]) & (x["rsi14"] < 45)

    if volume_filter:
        vok = x["vol_avg20"].gt(0) & x["volume"].gt(x["vol_avg20"] * 1.2)
        x["bear_ok"] &= vok

    # SHORT-ONLY variant: long entries are intentionally disabled.
    x["long_signal"] = False
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
    side_mult = 1 if pos["side"] == "LONG" else -1
    raw_points = (exit_price - pos["entry"]) * side_mult

    # Approximation only: percentage commission mapped to index points.
    # Not a broker-accurate futures/options P&L model.
    commission_points = (pos["entry"] + exit_price) * (commission_pct / 100.0)
    net_points = raw_points - (2 * slippage_points) - commission_points

    return {
        "symbol": pos["symbol"],
        "side": pos["side"],
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
    commission_pct: float = 0.03,
    slippage_points: float = 2.0,
    atr_buffer: float = 0.1,
    rr: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    x = prepare(df, volume_filter)
    trades = []
    pos = None
    equity_curve = []

    # Conservative OHLC handling: if stop and target touch in one candle,
    # assume stop first. Entries are at signal close; no same-candle exit.
    for i, row in x.iterrows():
        ts = row["ts"]

        if pos is not None and row["session"] != pos["session"]:
            previous = x.iloc[i - 1]
            trades.append(
                finish(
                    pos,
                    previous["ts"],
                    float(previous["close"]),
                    "session_boundary",
                    commission_pct,
                    slippage_points,
                )
            )
            pos = None

        if pos is not None:
            exit_price = None
            reason = None

            # This file only enters SHORT positions, but retain generic exit handling.
            if pos["side"] == "LONG":
                hit_stop = row["low"] <= pos["stop"]
                hit_target = row["high"] >= pos["target"]
                if hit_stop:
                    exit_price, reason = pos["stop"], "stop"
                elif hit_target:
                    exit_price, reason = pos["target"], "target"
            else:
                hit_stop = row["high"] >= pos["stop"]
                hit_target = row["low"] <= pos["target"]
                if hit_stop:
                    exit_price, reason = pos["stop"], "stop"
                elif hit_target:
                    exit_price, reason = pos["target"], "target"

            if exit_price is None and bool(row["flat_time"]):
                exit_price, reason = float(row["close"]), "intraday_flat"

            if exit_price is not None:
                trade = finish(
                    pos, ts, float(exit_price), reason, commission_pct, slippage_points
                )
                trades.append(trade)
                pos = None

        # One position at a time; do not re-enter on a bar where an exit just happened.
        if pos is None and not (trades and trades[-1].get("exit_ts") == ts):
            if bool(row["short_signal"]):
                side = "SHORT"
                entry = float(row["close"])
                current_atr = row["atr14"]

                if pd.notna(current_atr):
                    stop = float(row["high"] + current_atr * atr_buffer)
                    risk = stop - entry
                    target = entry - risk * rr

                    if risk > 0 and np.isfinite(risk):
                        pos = {
                            "symbol": symbol,
                            "side": side,
                            "session": row["session"],
                            "entry_ts": ts,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "risk_points": risk,
                            "volume_filter": volume_filter,
                        }

        unrealized = (
            (float(row["close"]) - pos["entry"])
            * (1 if pos["side"] == "LONG" else -1)
            if pos is not None
            else 0.0
        )
        equity_curve.append({"timestamp": ts, "equity_points": unrealized})

    if pos is not None and len(x):
        row = x.iloc[-1]
        trades.append(
            finish(
                pos,
                row["ts"],
                float(row["close"]),
                "end_of_data",
                commission_pct,
                slippage_points,
            )
        )

    t = pd.DataFrame(trades)
    if t.empty:
        summary = {
            "symbol": symbol,
            "side_filter": "SHORT_ONLY",
            "volume_filter": volume_filter,
            "trades": 0,
            "net_pnl_points": 0.0,
            "net_pnl_rupees_qty1_approx": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "max_drawdown_points": 0.0,
            "avg_trade_points": 0.0,
        }
        return t, summary

    wins = t.loc[t["net_pnl_points"] > 0, "net_pnl_points"].sum()
    losses = -t.loc[t["net_pnl_points"] < 0, "net_pnl_points"].sum()
    cumulative = t["net_pnl_points"].cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)

    summary = {
        "symbol": symbol,
        "side_filter": "SHORT_ONLY",
        "volume_filter": volume_filter,
        "trades": int(len(t)),
        "net_pnl_points": round(float(t["net_pnl_points"].sum()), 2),
        "net_pnl_rupees_qty1_approx": round(
            float(t["net_pnl_rupees_qty1_approx"].sum()), 2
        ),
        "win_rate_pct": round(float((t["net_pnl_points"] > 0).mean() * 100), 2),
        "profit_factor": round(float(wins / losses), 3) if losses > 0 else None,
        "max_drawdown_points": round(float(-drawdown.min()), 2),
        "avg_trade_points": round(float(t["net_pnl_points"].mean()), 3),
        "start": str(x["ts"].min()),
        "end": str(x["ts"].max()),
    }
    return t, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-only SHORT-only classic pivot historical research backtest."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--commission-pct",
        type=float,
        default=0.03,
        help="Approximate commission percent per side (default 0.03%%).",
    )
    parser.add_argument(
        "--slippage-points",
        type=float,
        default=2.0,
        help="Assumed index points per side (default 2).",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "NIFTY_50": args.data_root
        / "NIFTY_50_5m_2025-10-01_to-2026-10-09.csv",
        "BANK_NIFTY": args.data_root
        / "BANK_NIFTY_5m_2025-10-01_to-2026-10-09.csv",
    }

    summaries = []
    all_trades = []

    for symbol, path in files.items():
        if not path.exists():
            raise SystemExit(f"CSV not found: {path}")

        raw = load_data(path)
        for volume_filter in (False, True):
            trades, summary = backtest(
                raw,
                symbol,
                volume_filter,
                args.commission_pct,
                args.slippage_points,
            )
            summaries.append(summary)
            if not trades.empty:
                all_trades.append(trades)

            label = "volume_on" if volume_filter else "volume_off"
            print(
                f"{symbol} | SHORT_ONLY | {label}: "
                f"trades={summary['trades']}, "
                f"net_points={summary['net_pnl_points']}, "
                f"win_rate={summary['win_rate_pct']}%, "
                f"PF={summary['profit_factor']}, "
                f"maxDD_points={summary['max_drawdown_points']}"
            )

    pd.DataFrame(summaries).to_csv(args.out_dir / "summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            args.out_dir / "trades.csv", index=False
        )

    print(f"\nSaved reports to: {args.out_dir.resolve()}")
    print(
        "RESEARCH ONLY. Approximate spot-index points/costs; "
        "not broker-accurate futures/options P&L."
    )
    print(
        "Conservative assumption: if stop and target touch in same candle, "
        "stop is counted first."
    )


if __name__ == "__main__":
    main()
