#!/usr/bin/env python3
"""Daily win/loss report for frozen pivot on recent Upstox spot sessions."""
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
from scripts.frozen_pivot_config import (
    CONFIG_VERSION,
    V3_HYBRID_VERSION,
    V4_SPLIT_VERSION,
    V5_SHORT_ONLY_VERSION,
    FrozenPivotConfig,
    FLAT_SESSION_MINUTE,
    SIGNAL_SESSION_END_MINUTE,
    config_for_symbol,
    mark_side_taken,
    max_risk_for,
    side_quota_available,
    spot_paper_entry_window_ok,
)
from scripts.pivot_indicators import atr, ema, rsi
from src.historical_data.downloader import month_chunks
from src.historical_data.upstox_client import UpstoxHistoricalClient

IST = ZoneInfo("Asia/Kolkata")
INSTRUMENTS = {"NIFTY": ("NIFTY_50", "NSE_INDEX|Nifty 50"), "BANKNIFTY": ("BANK_NIFTY", "NSE_INDEX|Nifty Bank")}
END = datetime.now(IST).date()
START = END - timedelta(days=10)


def backtest_day(
    frame: pd.DataFrame,
    label: str,
    symbol: str,
    day: date,
    cfg: FrozenPivotConfig,
) -> pd.DataFrame:
    trades: list[dict] = []
    prepared = prepare(frame, volume_filter=False)
    prepared["typical"] = (prepared.high + prepared.low + prepared.close) / 3.0
    prepared["session_typical_avg"] = prepared.groupby("session", sort=False)["typical"].expanding().mean().reset_index(
        level=0, drop=True
    )
    day_rows = prepared.index[prepared.session == day].tolist()
    position = None
    long_taken = False
    short_taken = False
    trade_taken = False
    risk_cap = max_risk_for(symbol, cfg)

    for idx in day_rows:
        row = prepared.loc[idx]
        ts = row.ts
        if position is not None and ts >= position["entry_ts"]:
            if not position.get("breakeven_armed"):
                trigger = position["entry"] + position["initial_risk"] * cfg.breakeven_trigger_r * (
                    1 if position["side"] == "LONG" else -1
                )
                if position["side"] == "LONG" and row.close >= trigger and row.high >= trigger:
                    position["stop"] = position["entry"] + 4.0
                    position["breakeven_armed"] = True
                elif position["side"] == "SHORT" and row.close <= trigger and row.low <= trigger:
                    position["stop"] = position["entry"] - 4.0
                    position["breakeven_armed"] = True

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
            if exit_price is None and (ts.hour * 60 + ts.minute) >= FLAT_SESSION_MINUTE:
                exit_price, reason = row.close, "intraday_flat"
            if exit_price is not None:
                trades.append(finish(position, ts, float(exit_price), reason, 0.03, 2.0))
                position = None

        if position is not None or idx == 0:
            continue
        prev = prepared.loc[idx - 1]
        if row.session != day or prev.session != day:
            continue
        if ts.hour * 60 + ts.minute < 9 * 60 + 20 or ts.hour * 60 + ts.minute >= SIGNAL_SESSION_END_MINUTE:
            continue

        cross_up = (row.close > row.pp and prev.close <= prev.pp) or (row.close > row.r1 and prev.close <= prev.r1)
        cross_down = (row.close < row.pp and prev.close >= prev.pp) or (row.close < row.s1 and prev.close >= prev.s1)
        long_vwap = not cfg.require_vwap_above_for_long or (
            pd.notna(row.session_typical_avg) and row.close > row.session_typical_avg
        )
        short_vwap = not cfg.require_vwap_below_for_short or (
            pd.notna(row.session_typical_avg) and row.close < row.session_typical_avg
        )
        long_ok = (
            side_quota_available(cfg, "LONG", long_taken=long_taken, short_taken=short_taken, trade_taken=trade_taken)
            and spot_paper_entry_window_ok(symbol, ts, "LONG")
            and cross_up and row.ema20 > row.ema50 and row.rsi14 > 55 and long_vwap
        )
        short_ok = (
            side_quota_available(cfg, "SHORT", long_taken=long_taken, short_taken=short_taken, trade_taken=trade_taken)
            and spot_paper_entry_window_ok(symbol, ts, "SHORT")
            and cross_down and row.ema20 < row.ema50 and row.rsi14 < 45 and short_vwap
        )
        if not (long_ok or short_ok) or pd.isna(row.atr14):
            continue
        next_idx = idx + 1
        if next_idx not in prepared.index or prepared.loc[next_idx].session != day:
            continue
        side = "LONG" if long_ok else "SHORT"
        entry_row = prepared.loc[next_idx]
        entry = float(entry_row.open)
        if side == "LONG":
            stop = float(row.low - row.atr14 * cfg.atr_buffer)
            risk = entry - stop
            target = entry + cfg.rr * risk
        else:
            stop = float(row.high + row.atr14 * cfg.atr_buffer)
            risk = stop - entry
            target = entry - cfg.rr * risk
        if risk <= 0 or risk > risk_cap:
            continue
        position = {
            "symbol": label, "side": side, "session": day, "entry_ts": entry_row.ts,
            "entry": entry, "stop": stop, "target": target, "risk_points": risk,
            "initial_risk": risk, "breakeven_armed": False, "volume_filter": False,
        }
        long_taken, short_taken, trade_taken = mark_side_taken(
            cfg, side, long_taken=long_taken, short_taken=short_taken, trade_taken=trade_taken,
        )

    if position is not None and day_rows:
        last = prepared.loc[day_rows[-1]]
        trades.append(finish(position, last.ts, float(last.close), "period_end", 0.03, 2.0))
    return pd.DataFrame(trades)


def load_spot_frames(start: date, end: date) -> dict[str, tuple[str, pd.DataFrame]]:
    client = UpstoxHistoricalClient()
    fetch_start = start - timedelta(days=45)
    frames: dict[str, tuple[str, pd.DataFrame]] = {}
    for symbol, (label, key) in INSTRUMENTS.items():
        candles = []
        for chunk_start, chunk_end in month_chunks(fetch_start, end):
            candles.extend(client.candles(key, chunk_start, chunk_end, unit="minutes", interval=5))
        if end == datetime.now(IST).date():
            candles.extend(client.intraday_candles(key, unit="minutes", interval=5))
        frame = pd.DataFrame([{
            "timestamp": c.timestamp.isoformat(), "open": c.open, "high": c.high,
            "low": c.low, "close": c.close, "volume": c.volume,
        } for c in candles])
        frame["ts"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(IST)
        frame = frame[frame.ts + pd.Timedelta(minutes=5) <= pd.Timestamp.now(tz=IST)]
        frame["session"] = frame.ts.dt.date
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
        frames[symbol] = (label, frame)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen pivot last-7-day spot report")
    parser.add_argument(
        "--preset",
        choices=["v3_hybrid", "v4_split", "v5_short_only"],
        default=CONFIG_VERSION,
        help="default=v5_short_only | v4_split=NIFTY short + BN hybrid",
    )
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=None)
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=None)
    args = parser.parse_args()
    report_end = args.end or END
    report_start = args.start or (report_end - timedelta(days=10))
    preset_map = {
        "v3_hybrid": V3_HYBRID_VERSION,
        "v4_split": V4_SPLIT_VERSION,
        "v5_short_only": V5_SHORT_ONLY_VERSION,
    }
    preset_name = preset_map[args.preset]
    out = ROOT / "reports" / f"frozen_pivot_last7_{args.preset}"

    all_trades: list[pd.DataFrame] = []
    for symbol, (label, frame) in load_spot_frames(report_start, report_end).items():
        cfg = config_for_symbol(symbol, preset_name)
        for day in pd.date_range(report_start, report_end, freq="D"):
            if day.weekday() >= 5:
                continue
            trades = backtest_day(frame, label, symbol, day.date(), cfg)
            if not trades.empty:
                trades["session_date"] = str(day.date())
                all_trades.append(trades)

    out.mkdir(parents=True, exist_ok=True)
    t = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not t.empty:
        t.to_csv(out / "trades.csv", index=False)

    days = [d.date() for d in pd.date_range(report_start, report_end, freq="D") if d.weekday() < 5]
    print(f"=== FROZEN {preset_name} | LAST 7 DAYS | SPOT ===")
    print(f"Range: {report_start} to {report_end}")
    daily_rows = []
    for day in days:
        day_trades = t[t.session_date == str(day)].sort_values(["symbol", "entry_ts"]) if not t.empty else t
        print(f"\n--- {day} ---")
        if day_trades.empty:
            print("  No trades")
            daily_rows.append({"date": str(day), "trades": 0, "wins": 0, "losses": 0, "net_points": 0})
            continue
        for _, r in day_trades.iterrows():
            result = "WIN" if r.net_pnl_points > 0 else "LOSS"
            print(f"  {r['symbol']:10} {r['side']:5} {result} {r.net_pnl_points:+.2f} pts | {r['exit_reason']}")
        wins = int((day_trades.net_pnl_points > 0).sum())
        net = float(day_trades.net_pnl_points.sum())
        print(f"  DAY TOTAL: {len(day_trades)} trades | {wins}W-{len(day_trades)-wins}L | net={net:+.2f} pts")
        daily_rows.append({"date": str(day), "trades": len(day_trades), "wins": wins,
                           "losses": len(day_trades) - wins, "net_points": round(net, 2)})

    pd.DataFrame(daily_rows).to_csv(out / "daily_summary.csv", index=False)
    if not t.empty:
        wins = int((t.net_pnl_points > 0).sum())
        print(f"\nGRAND TOTAL: {len(t)} trades | {wins}W-{len(t)-wins}L | net={t.net_pnl_points.sum():+.2f} pts")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
