#!/usr/bin/env python3
"""Fetch recent read-only spot candles and backtest fixed guide rules by session."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from scripts.classic_pivot_historical_backtest import finish, prepare
from src.historical_data.downloader import month_chunks
from src.historical_data.upstox_client import UpstoxHistoricalClient

IST = ZoneInfo("Asia/Kolkata")
INSTRUMENTS = {"NIFTY_50": "NSE_INDEX|Nifty 50", "BANK_NIFTY": "NSE_INDEX|Nifty Bank"}


def backtest_session(frame: pd.DataFrame, symbol: str, day: date, rr: float = 1.5,
                    atr_buffer: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """Use only the requested day's signals; enter at the next 5m candle open."""
    prepared = prepare(frame, volume_filter=False)
    trades: list[dict] = []
    position = None
    exited_this_bar = False
    day_rows = prepared.index[prepared.session == day].tolist()
    for idx in day_rows:
        row = prepared.loc[idx]
        ts = row.ts
        exited_this_bar = False
        if position is not None and ts >= position["entry_ts"]:
            exit_price, reason = None, None
            if position["side"] == "LONG":
                if row.low <= position["stop"]:
                    exit_price, reason = position["stop"], "stop (stop-first)"
                elif row.high >= position["target"]:
                    exit_price, reason = position["target"], "target"
            else:
                if row.high >= position["stop"]:
                    exit_price, reason = position["stop"], "stop (stop-first)"
                elif row.low <= position["target"]:
                    exit_price, reason = position["target"], "target"
            if exit_price is None and ts.time() >= pd.Timestamp("15:15").time():
                exit_price, reason = row.close, "intraday_flat"
            if exit_price is not None:
                trades.append(finish(position, ts, float(exit_price), reason, 0.03, 2.0))
                position = None
                exited_this_bar = True

        if position is not None or exited_this_bar:
            continue
        if not (bool(row.long_signal) or bool(row.short_signal)):
            continue
        next_idx = idx + 1
        if next_idx not in prepared.index or prepared.loc[next_idx].session != day:
            continue
        side = "LONG" if bool(row.long_signal) else "SHORT"
        entry_row = prepared.loc[next_idx]
        entry = float(entry_row.open)
        atr_value = row.atr14
        if pd.isna(atr_value):
            continue
        stop = float(row.low - atr_value * atr_buffer) if side == "LONG" else float(row.high + atr_value * atr_buffer)
        risk = entry - stop if side == "LONG" else stop - entry
        if risk <= 0:
            continue
        target = entry + rr * risk if side == "LONG" else entry - rr * risk
        position = {"symbol": symbol, "side": side, "session": day, "entry_ts": entry_row.ts,
                    "entry": entry, "stop": stop, "target": target, "risk_points": risk,
                    "volume_filter": False}

    if position is not None and day_rows:
        last = prepared.loc[day_rows[-1]]
        trades.append(finish(position, last.ts, float(last.close), "period_end", 0.03, 2.0))
    result = pd.DataFrame(trades)
    if result.empty:
        return result, {"symbol": symbol, "trades": 0, "long_trades": 0, "short_trades": 0,
                        "net_points": 0.0, "win_rate_pct": 0.0, "profit_factor": None,
                        "max_drawdown_points": 0.0}
    wins = result.loc[result.net_pnl_points > 0, "net_pnl_points"].sum()
    losses = -result.loc[result.net_pnl_points < 0, "net_pnl_points"].sum()
    cumulative = result.net_pnl_points.cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    return result, {"symbol": symbol, "trades": len(result),
                    "long_trades": int((result.side == "LONG").sum()),
                    "short_trades": int((result.side == "SHORT").sum()),
                    "net_points": round(float(result.net_pnl_points.sum()), 2),
                    "win_rate_pct": round(float((result.net_pnl_points > 0).mean() * 100), 2),
                    "profit_factor": round(float(wins / losses), 3) if losses else None,
                    "max_drawdown_points": round(float(-drawdown.min()), 2),
                    "avg_trade_points": round(float(result.net_pnl_points.mean()), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest fixed pivot rules on recent completed sessions")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2026, 10, 12))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 10, 13))
    parser.add_argument("--symbols", nargs="+", choices=INSTRUMENTS, default=list(INSTRUMENTS))
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "recent_pivot_backtest_20261012_13")
    parser.add_argument("--profile", choices=("baseline", "development_winrate"), default="baseline")
    parser.add_argument("--source-dir", type=Path, help="Reuse previously downloaded source CSVs instead of calling Upstox")
    args = parser.parse_args()
    if args.end_date < args.start_date:
        parser.error("end date must be on or after start date")

    start = args.start_date
    end = args.end_date
    fetch_start = start - timedelta(days=45)
    client = None if args.source_dir else UpstoxHistoricalClient()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, all_trades = [], []

    for symbol in args.symbols:
        rr, atr_buffer = ((1.5, 0.2) if symbol == "NIFTY_50" else (1.0, 0.2)) if args.profile == "development_winrate" else (1.5, 0.1)
        instrument_key = INSTRUMENTS[symbol]
        if args.source_dir:
            candidates = sorted(args.source_dir.glob(f"{symbol}_*_source.csv"))
            if not candidates:
                parser.error(f"no {symbol} source CSV found in {args.source_dir}")
            source_path = max(candidates, key=lambda path: path.stat().st_mtime)
            frame = pd.read_csv(source_path)
        else:
            candles = []
            for chunk_start, chunk_end in month_chunks(fetch_start, end):
                candles.extend(client.candles(instrument_key, chunk_start, chunk_end, unit="minutes", interval=5))
            # The date-bounded endpoint can lag the current session; use Upstox's
            # read-only intraday endpoint as the source for today's completed bars.
            if end == datetime.now(IST).date():
                candles.extend(client.intraday_candles(instrument_key, unit="minutes", interval=5))
            rows = [{"timestamp": c.timestamp.isoformat(), "open": c.open, "high": c.high,
                     "low": c.low, "close": c.close, "volume": c.volume} for c in candles]
            if not rows:
                print(f"{symbol}: Upstox returned no candles for requested window.")
                continue
            frame = pd.DataFrame(rows)
        frame["ts"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
        frame = frame[frame.ts + pd.Timedelta(minutes=5) <= pd.Timestamp.now(tz=IST)]
        frame["session"] = frame.ts.dt.date
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
        frame = frame.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
        frame[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(
            args.out / f"{symbol}_{fetch_start}_to_{end}_source.csv", index=False)
        available_sessions = sorted(day for day in frame.session.unique() if start <= day <= end)
        for day in available_sessions[-7:]:
            session_bars = frame[frame.session == day]
            if session_bars.empty:
                print(f"{symbol} | {day}: no candles (market holiday or unavailable data).")
                continue
            trades, summary = backtest_session(frame, symbol, day, rr=rr, atr_buffer=atr_buffer)
            summary.update(date=str(day), bars=len(session_bars),
                           first_bar=str(session_bars.ts.min()), last_bar=str(session_bars.ts.max()),
                           profile=args.profile, rr=rr, atr_buffer=atr_buffer)
            summaries.append(summary)
            if not trades.empty:
                trades = trades.copy()
                trades["session_date"] = str(day)
                all_trades.append(trades)
            print(f"{symbol} | {day}: bars={len(session_bars)}, trades={summary['trades']} "
                  f"(L={summary.get('long_trades', 0)}, S={summary.get('short_trades', 0)}), "
                  f"net={summary['net_points']} pts, win={summary['win_rate_pct']}%, "
                  f"PF={summary['profit_factor']}, maxDD={summary['max_drawdown_points']} pts")

    if summaries:
        pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(args.out / "trades.csv", index=False)
    print(f"Output: {args.out}")
    print(f"Profile: {args.profile}. Research only: spot-index points, volume filter OFF, estimated costs 0.03% per side + 2 points per side.")
    print("No orders submitted; stop-first when stop and target touch in one bar.")


if __name__ == "__main__":
    main()
