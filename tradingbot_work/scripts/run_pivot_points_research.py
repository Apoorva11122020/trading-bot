#!/usr/bin/env python3
"""Development-only research for improving net points in the classic pivot strategy.

Compares the existing RR/ATR settings with two pre-declared execution rules:
1) skip entries whose next-bar open gaps more than 0.25 ATR from signal close;
2) after six bars, exit if the trade has never reached +0.5R favorable excursion.

October 2026 recent sessions are excluded. No broker APIs or orders are used.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_historical_backtest import finish, load_data, prepare

DATA = ROOT / "data" / "historical" / "upstox_pivot_5m"
OUT = ROOT / "reports" / "pivot_points_research_20261014"
DEV_START = pd.Timestamp("2025-10-01").date()
TRAIN_END = pd.Timestamp("2026-02-28").date()
VALIDATION_START = pd.Timestamp("2026-03-01").date()
DEV_END = pd.Timestamp("2026-05-31").date()


def replay(prepared: pd.DataFrame, symbol: str, rr: float, atr_buffer: float,
          variant: str) -> pd.DataFrame:
    trades: list[dict] = []
    days = sorted(d for d in prepared.session.unique() if DEV_START <= d <= DEV_END)
    for day in days:
        indices = prepared.index[prepared.session == day].tolist()
        position = None
        exited_index = None
        for idx in indices:
            row = prepared.loc[idx]
            if position is not None and row.ts >= position["entry_ts"]:
                position["bars_held"] += 1
                side = position["side"]
                if side == "LONG":
                    hit_stop = row.low <= position["stop"]
                    hit_target = row.high >= position["target"]
                    favorable = max(0.0, float(row.high) - position["entry"])
                else:
                    hit_stop = row.high >= position["stop"]
                    hit_target = row.low <= position["target"]
                    favorable = max(0.0, position["entry"] - float(row.low))
                position["max_favorable"] = max(position["max_favorable"], favorable)
                exit_price = reason = None
                if hit_stop:  # conservative stop-first OHLC handling
                    exit_price, reason = position["stop"], "stop"
                elif hit_target:
                    exit_price, reason = position["target"], "target"
                elif row.ts.time() >= pd.Timestamp("15:15").time():
                    exit_price, reason = float(row.close), "intraday_flat"
                elif (variant == "followthrough" and position["bars_held"] >= 6
                      and position["max_favorable"] < 0.5 * position["risk_points"]):
                    exit_price, reason = float(row.close), "failed_followthrough_6bars"
                if exit_price is not None:
                    trades.append(finish(position, row.ts, float(exit_price), reason, 0.03, 2.0))
                    position = None
                    exited_index = idx

            if position is not None or idx == exited_index:
                continue
            if not (bool(row.long_signal) or bool(row.short_signal)):
                continue
            next_idx = idx + 1
            if (next_idx not in prepared.index or prepared.loc[next_idx].session != day
                    or prepared.loc[next_idx].ts - row.ts != pd.Timedelta(minutes=5)):
                continue
            if pd.isna(row.atr14):
                continue
            entry_row = prepared.loc[next_idx]
            entry = float(entry_row.open)
            if variant == "gap_guard" and abs(entry - float(row.close)) > 0.25 * float(row.atr14):
                continue
            side = "LONG" if bool(row.long_signal) else "SHORT"
            stop = (float(row.low - row.atr14 * atr_buffer) if side == "LONG"
                    else float(row.high + row.atr14 * atr_buffer))
            risk = entry - stop if side == "LONG" else stop - entry
            if risk <= 0:
                continue
            target = entry + rr * risk if side == "LONG" else entry - rr * risk
            position = {
                "symbol": symbol, "side": side, "session": day,
                "entry_ts": entry_row.ts, "entry": entry, "stop": stop,
                "target": target, "risk_points": risk, "volume_filter": False,
                "bars_held": 0, "max_favorable": 0.0,
            }

        if position is not None and indices:
            last = prepared.loc[indices[-1]]
            trades.append(finish(position, last.ts, float(last.close), "period_end", 0.03, 2.0))
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, symbol: str, variant: str, split: str) -> dict:
    values = trades.net_pnl_points.astype(float) if not trades.empty else pd.Series(dtype=float)
    wins = values[values > 0]
    losses = -values[values < 0]
    return {
        "symbol": symbol, "variant": variant, "split": split,
        "trades": len(values), "wins": int((values > 0).sum()),
        "win_rate_pct": round(float((values > 0).mean() * 100), 2) if len(values) else 0.0,
        "net_points": round(float(values.sum()), 2),
        "profit_factor": round(float(wins.sum() / losses.sum()), 3) if losses.sum() else None,
        "avg_trade_points": round(float(values.mean()), 3) if len(values) else 0.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "NIFTY_50": DATA / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv",
        "BANK_NIFTY": DATA / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv",
    }
    configs = {"NIFTY_50": (1.5, 0.2), "BANK_NIFTY": (1.0, 0.2)}
    variants = ("baseline", "gap_guard", "followthrough")
    summary_rows: list[dict] = []
    trade_rows: list[pd.DataFrame] = []
    for symbol, path in files.items():
        if not path.exists():
            raise SystemExit(f"Missing local historical candles: {path}")
        prepared = prepare(load_data(path), volume_filter=False)
        rr, buffer = configs[symbol]
        for variant in variants:
            trades = replay(prepared, symbol, rr, buffer, variant)
            trades["variant"] = variant
            trades["rr"] = rr
            trades["atr_buffer"] = buffer
            trades["session_date"] = pd.to_datetime(trades.entry_ts).dt.date if not trades.empty else None
            if not trades.empty:
                trade_rows.append(trades)
            for split, condition in (
                ("TRAIN", trades.session_date <= TRAIN_END if not trades.empty else []),
                ("VALIDATION", ((trades.session_date >= VALIDATION_START) &
                                (trades.session_date <= DEV_END)) if not trades.empty else []),
            ):
                part = trades.loc[condition] if not trades.empty else trades
                summary_rows.append(summarize(part, symbol, variant, split))
    pd.DataFrame(summary_rows).to_csv(OUT / "summary.csv", index=False)
    if trade_rows:
        pd.concat(trade_rows, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Saved: {OUT}")
    print("Development only through 2026-05-31. Estimated costs: 0.03% per side + 2 points per side.")
    print("Spot-index points, volume filter OFF; not actual futures rupee P&L. No orders submitted.")


if __name__ == "__main__":
    main()
