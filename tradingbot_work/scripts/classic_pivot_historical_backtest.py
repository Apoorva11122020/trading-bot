from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_DATA = Path("data/historical/upstox_pivot_5m")
DEFAULT_OUT = Path("reports/pivot_research_backtest")


try:
    from scripts.pivot_indicators import atr, ema, rsi
except ModuleNotFoundError:  # Support direct execution as python scripts/<file>.py
    from pivot_indicators import atr, ema, rsi


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    # timestamp is stored as UTC; convert to India time for sessions.
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["ts"] = ts
    df["session"] = ts.dt.date
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df


def prepare(df: pd.DataFrame, volume_filter: bool) -> pd.DataFrame:
    x = df.copy()
    # Daily H/L/C from this dataset; shift by one full session to avoid lookahead.
    # Use regular NSE session candles only when deriving daily pivot levels.
    # Out-of-session rows must not affect the prior-day H/L/C used by Pine.
    minutes = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    regular = x.loc[(minutes >= 9 * 60 + 15) & (minutes < 15 * 60 + 30)]
    daily = regular.groupby("session", sort=True).agg(
        day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last")
    )
    daily["prev_high"] = daily["day_high"].shift(1)
    daily["prev_low"] = daily["day_low"].shift(1)
    daily["prev_close"] = daily["day_close"].shift(1)
    daily["pp"] = (daily["prev_high"] + daily["prev_low"] + daily["prev_close"]) / 3
    daily["r1"] = 2*daily["pp"] - daily["prev_low"]
    daily["s1"] = 2*daily["pp"] - daily["prev_high"]
    x = x.join(daily[["pp", "r1", "s1"]], on="session")

    x["ema20"] = ema(x["close"], 20)
    x["ema50"] = ema(x["close"], 50)
    x["rsi14"] = rsi(x["close"], 14)
    x["atr14"] = atr(x, 14)
    x["vol_avg20"] = x["volume"].rolling(20, min_periods=20).mean()

    prev_close = x["close"].shift(1)
    prev_pp = x["pp"].shift(1)
    prev_r1 = x["r1"].shift(1)
    prev_s1 = x["s1"].shift(1)

    x["cross_up"] = (
        ((x["close"] > x["pp"]) & (prev_close <= prev_pp)) |
        ((x["close"] > x["r1"]) & (prev_close <= prev_r1))
    )
    x["cross_down"] = (
        ((x["close"] < x["pp"]) & (prev_close >= prev_pp)) |
        ((x["close"] < x["s1"]) & (prev_close >= prev_s1))
    )

    # TradingView defaults: entries from 09:20 until before 15:00, flatten at 15:15.
    minutes = x["ts"].dt.hour * 60 + x["ts"].dt.minute
    x["in_session"] = (minutes >= 9*60+20) & (minutes < 15*60)
    x["flat_time"] = minutes >= 15*60+15
    x["bull_ok"] = (x["ema20"] > x["ema50"]) & (x["rsi14"] > 55)
    x["bear_ok"] = (x["ema20"] < x["ema50"]) & (x["rsi14"] < 45)
    if volume_filter:
        vok = x["vol_avg20"].gt(0) & x["volume"].gt(x["vol_avg20"] * 1.2)
        x["bull_ok"] &= vok
        x["bear_ok"] &= vok
    x["long_signal"] = x["in_session"] & x["cross_up"] & x["bull_ok"]
    x["short_signal"] = x["in_session"] & x["cross_down"] & x["bear_ok"]
    return x


def backtest(df: pd.DataFrame, symbol: str, volume_filter: bool,
             commission_pct: float = 0.03, slippage_points: float = 2.0,
             atr_buffer: float = 0.1, rr: float = 1.5) -> tuple[pd.DataFrame, dict]:
    x = prepare(df, volume_filter)
    trades = []
    pos = None
    equity = 0.0
    equity_curve = []
    # Conservative OHLC handling: if stop and target are both touched in one candle,
    # assume stop first. Entries are at signal close; no same-candle exit.
    for i, row in x.iterrows():
        ts = row["ts"]
        if pos is not None and row["session"] != pos["session"]:
            # Should generally be prevented by force-flat; safety close at prior close.
            prev = x.iloc[i-1]
            exit_price = float(prev["close"])
            trades.append(finish(pos, prev["ts"], exit_price, "session_boundary", commission_pct, slippage_points))
            pos = None

        if pos is not None:
            exit_price = None
            reason = None
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
                tr = finish(pos, ts, float(exit_price), reason, commission_pct, slippage_points)
                trades.append(tr)
                equity += tr["net_pnl_points"]
                pos = None

        # One position at a time; do not re-enter on a bar where an exit just happened.
        if pos is None and not (trades and trades[-1].get("exit_ts") == ts):
            if bool(row["long_signal"]) or bool(row["short_signal"]):
                side = "LONG" if bool(row["long_signal"]) else "SHORT"
                entry = float(row["close"])
                a = row["atr14"]
                if pd.notna(a):
                    if side == "LONG":
                        stop = float(row["low"] - a * atr_buffer)
                        risk = entry - stop
                        target = entry + risk * rr
                    else:
                        stop = float(row["high"] + a * atr_buffer)
                        risk = stop - entry
                        target = entry - risk * rr
                    if risk > 0 and np.isfinite(risk):
                        pos = {
                            "symbol": symbol, "side": side, "session": row["session"],
                            "entry_ts": ts, "entry": entry, "stop": stop, "target": target,
                            "risk_points": risk, "volume_filter": volume_filter
                        }
        equity_curve.append({"timestamp": ts, "equity_points": equity + (
            (float(row["close"]) - pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
            if pos is not None else 0.0)})
    if pos is not None and len(x):
        row = x.iloc[-1]
        tr = finish(pos, row["ts"], float(row["close"]), "end_of_data", commission_pct, slippage_points)
        trades.append(tr)
        equity += tr["net_pnl_points"]

    t = pd.DataFrame(trades)
    if t.empty:
        summary = {
            "symbol": symbol, "volume_filter": volume_filter, "trades": 0,
            "net_pnl_points": 0.0, "net_pnl_rupees_qty1_approx": 0.0,
            "win_rate_pct": 0.0, "profit_factor": None, "max_drawdown_points": 0.0,
            "avg_trade_points": 0.0
        }
        return t, summary
    wins = t.loc[t["net_pnl_points"] > 0, "net_pnl_points"].sum()
    losses = -t.loc[t["net_pnl_points"] < 0, "net_pnl_points"].sum()
    cumulative = t["net_pnl_points"].cumsum()
    dd = cumulative - cumulative.cummax().clip(lower=0)
    summary = {
        "symbol": symbol, "volume_filter": volume_filter, "trades": int(len(t)),
        "net_pnl_points": round(float(t["net_pnl_points"].sum()), 2),
        "net_pnl_rupees_qty1_approx": round(float(t["net_pnl_rupees_qty1_approx"].sum()), 2),
        "win_rate_pct": round(float((t["net_pnl_points"] > 0).mean()*100), 2),
        "profit_factor": round(float(wins/losses), 3) if losses > 0 else None,
        "max_drawdown_points": round(float(-dd.min()), 2),
        "avg_trade_points": round(float(t["net_pnl_points"].mean()), 3),
        "start": str(x["ts"].min()), "end": str(x["ts"].max())
    }
    return t, summary


def finish(pos: dict, exit_ts, exit_price: float, reason: str,
           commission_pct: float, slippage_points: float) -> dict:
    side_mult = 1 if pos["side"] == "LONG" else -1
    # Approximate two-sided point slippage. Costs in index points are approximated
    # using entry/exit price * commission percent per side; rupee value is quantity 1
    # and is illustrative only, not a valid futures/options contract P&L.
    raw_points = (exit_price - pos["entry"]) * side_mult
    commission_points = (pos["entry"] + exit_price) * (commission_pct / 100.0)
    net_points = raw_points - (2 * slippage_points) - commission_points
    return {
        "symbol": pos["symbol"], "side": pos["side"], "entry_ts": pos["entry_ts"],
        "exit_ts": exit_ts, "entry": pos["entry"], "exit": exit_price,
        "stop": pos["stop"], "target": pos["target"], "risk_points": pos["risk_points"],
        "exit_reason": reason, "volume_filter": pos["volume_filter"],
        "gross_pnl_points": round(raw_points, 4),
        "estimated_cost_points": round(2*slippage_points + commission_points, 4),
        "net_pnl_points": round(net_points, 4),
        "net_pnl_rupees_qty1_approx": round(net_points, 2)
    }


def main():
    p = argparse.ArgumentParser(description="Paper-only classic pivot historical research backtest.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--commission-pct", type=float, default=0.03,
                   help="commission percent per side (default 0.03%%)")
    p.add_argument("--slippage-points", type=float, default=2.0,
                   help="assumed index points per side (default 2)")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "NIFTY_50": a.data_root / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv",
        "BANK_NIFTY": a.data_root / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv",
    }
    summaries, all_trades = [], []
    for symbol, path in files.items():
        if not path.exists():
            raise SystemExit(f"CSV not found: {path}")
        raw = load_data(path)
        for vf in (False, True):
            trades, summary = backtest(raw, symbol, vf, a.commission_pct, a.slippage_points)
            summaries.append(summary)
            if not trades.empty:
                all_trades.append(trades)
            label = "volume_on" if vf else "volume_off"
            print(f"{symbol} | {label}: trades={summary['trades']}, "
                  f"net_points={summary['net_pnl_points']}, win_rate={summary['win_rate_pct']}%, "
                  f"PF={summary['profit_factor']}, maxDD_points={summary['max_drawdown_points']}")

    pd.DataFrame(summaries).to_csv(a.out_dir / "summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(a.out_dir / "trades.csv", index=False)
    print(f"\nSaved reports to: {a.out_dir.resolve()}")
    print("RESEARCH ONLY. Approximate spot-index points/costs; not broker-accurate futures/options P&L.")
    print("Conservative assumption: if stop and target touch in same candle, stop is counted first.")


if __name__ == "__main__":
    main()
