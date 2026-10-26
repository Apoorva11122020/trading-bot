#!/usr/bin/env python3
"""Bounded development-only RR/stop screen for the classic pivot baseline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts.classic_pivot_historical_backtest import finish, load_data, prepare
except ModuleNotFoundError:
    from classic_pivot_historical_backtest import finish, load_data, prepare

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "historical" / "upstox_pivot_5m"
OUT = ROOT / "reports" / "development_winrate_screen_20261014"
DEV_START = pd.Timestamp("2025-10-01").date()
DEV_END = pd.Timestamp("2026-05-31").date()
RR_VALUES = (0.5, 0.75, 1.0, 1.5)
ATR_BUFFERS = (0.1, 0.2)


def evaluate(prepared: pd.DataFrame, symbol: str, rr: float, atr_buffer: float,
             commission_pct: float = 0.03, slippage_points: float = 2.0) -> pd.DataFrame:
    trades = []
    days = sorted(d for d in prepared.session.unique() if DEV_START <= d <= DEV_END)
    for day in days:
        pos = None
        day_idx = prepared.index[prepared.session == day].tolist()
        exited = False
        for idx in day_idx:
            row = prepared.loc[idx]
            exited = False
            if pos is not None and row.ts >= pos["entry_ts"]:
                exit_price, reason = None, None
                if pos["side"] == "LONG":
                    if row.low <= pos["stop"]:
                        exit_price, reason = pos["stop"], "stop (stop-first)"
                    elif row.high >= pos["target"]:
                        exit_price, reason = pos["target"], "target"
                else:
                    if row.high >= pos["stop"]:
                        exit_price, reason = pos["stop"], "stop (stop-first)"
                    elif row.low <= pos["target"]:
                        exit_price, reason = pos["target"], "target"
                if exit_price is None and row.ts.time() >= pd.Timestamp("15:15").time():
                    exit_price, reason = row.close, "intraday_flat"
                if exit_price is not None:
                    trades.append(finish(pos, row.ts, float(exit_price), reason,
                                         commission_pct, slippage_points))
                    pos, exited = None, True
            if pos is not None or exited or not (bool(row.long_signal) or bool(row.short_signal)):
                continue
            next_idx = idx + 1
            if next_idx not in prepared.index:
                continue
            following = prepared.loc[next_idx]
            if following.session != day or following.ts - row.ts != pd.Timedelta(minutes=5):
                continue
            side = "LONG" if bool(row.long_signal) else "SHORT"
            entry = float(following.open)
            if pd.isna(row.atr14):
                continue
            stop = float(row.low - row.atr14 * atr_buffer) if side == "LONG" else float(row.high + row.atr14 * atr_buffer)
            risk = entry - stop if side == "LONG" else stop - entry
            if risk <= 0:
                continue
            target = entry + risk * rr if side == "LONG" else entry - risk * rr
            pos = {"symbol": symbol, "side": side, "session": day, "entry_ts": following.ts,
                   "entry": entry, "stop": stop, "target": target, "risk_points": risk,
                   "volume_filter": False}
        if pos is not None and day_idx:
            last = prepared.loc[day_idx[-1]]
            trades.append(finish(pos, last.ts, float(last.close), "intraday_flat",
                                 commission_pct, slippage_points))
    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame, symbol: str, rr: float, atr_buffer: float) -> dict:
    if trades.empty:
        return {"symbol": symbol, "rr": rr, "atr_buffer": atr_buffer, "trades": 0,
                "wins": 0, "win_rate_pct": 0.0, "net_points": 0.0,
                "profit_factor": None, "max_drawdown_points": 0.0}
    values = trades.net_pnl_points.astype(float)
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    cum = values.cumsum()
    dd = cum - cum.cummax().clip(lower=0)
    return {"symbol": symbol, "rr": rr, "atr_buffer": atr_buffer, "trades": len(trades),
            "wins": int((values > 0).sum()), "win_rate_pct": round(float((values > 0).mean() * 100), 2),
            "net_points": round(float(values.sum()), 2),
            "profit_factor": round(gains / losses, 3) if losses else None,
            "max_drawdown_points": round(float(-dd.min()), 2),
            "avg_trade_points": round(float(values.mean()), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "NIFTY_50": DATA / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv",
        "BANK_NIFTY": DATA / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv",
    }
    summaries, details = [], []
    for symbol, path in paths.items():
        frame = load_data(path)
        # Development-only cut: OOS candles never enter indicator calculation or ranking.
        frame = frame[frame.session <= DEV_END].reset_index(drop=True)
        prepared = prepare(frame, volume_filter=False)
        for rr in RR_VALUES:
            for buffer in ATR_BUFFERS:
                trades = evaluate(prepared, symbol, rr, buffer)
                summary = metrics(trades, symbol, rr, buffer)
                summaries.append(summary)
                if not trades.empty:
                    trades = trades.copy()
                    trades["rr"] = rr
                    trades["atr_buffer"] = buffer
                    details.append(trades)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(OUT / "development_summary.csv", index=False)
    if details:
        pd.concat(details, ignore_index=True).to_csv(OUT / "development_trades.csv", index=False)
    ranked = summary_frame.sort_values(["symbol", "win_rate_pct", "profit_factor"], ascending=[True, False, False])
    ranked.to_csv(OUT / "development_ranked.csv", index=False)
    print("DEVELOPMENT ONLY (2025-10-01 through 2026-05-31); volume OFF; next-bar open entries.")
    print("Ranked by win rate for inspection. Positive net points and PF > 1 are required for a viable candidate.")
    print(ranked.to_string(index=False))
    print(f"Saved: {OUT}")
    print("Estimated costs: 0.03% per side + 2 index points per side. No OOS data used for selection; no orders placed.")


if __name__ == "__main__":
    main()
