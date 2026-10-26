"""Period-split, paper-only evaluation of the classic pivot guide rules."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts.classic_pivot_historical_backtest import finish, load_data, prepare
except ModuleNotFoundError:
    from classic_pivot_historical_backtest import finish, load_data, prepare


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "historical" / "upstox_pivot_5m"
OUT = ROOT / "reports" / "guide_split_evaluation_20261014"
PERIODS = {
    "development": (pd.Timestamp("2025-10-01").date(), pd.Timestamp("2026-05-31").date()),
    "out_of_sample": (pd.Timestamp("2026-06-01").date(), pd.Timestamp("2026-10-08").date()),
}


def run_period(frame: pd.DataFrame, symbol: str, start, end) -> tuple[pd.DataFrame, dict]:
    # Prepare on the complete chronological series so indicators and pivots are warmed up.
    prepared = prepare(frame, volume_filter=False)
    rows = prepared.loc[(prepared["session"] >= start) & (prepared["session"] <= end)]
    trades: list[dict] = []
    position = None
    for _, row in rows.iterrows():
        ts = row["ts"]
        if position is not None:
            exit_price = None
            reason = None
            if position["side"] == "LONG":
                if row["low"] <= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif row["high"] >= position["target"]:
                    exit_price, reason = position["target"], "target"
            else:
                if row["high"] >= position["stop"]:
                    exit_price, reason = position["stop"], "stop"
                elif row["low"] <= position["target"]:
                    exit_price, reason = position["target"], "target"
            if exit_price is None and bool(row["flat_time"]):
                exit_price, reason = float(row["close"]), "intraday_flat"
            if exit_price is not None:
                trades.append(finish(position, ts, float(exit_price), reason, 0.03, 2.0))
                position = None

        if position is None and not (trades and trades[-1]["exit_ts"] == ts):
            if bool(row["long_signal"]) or bool(row["short_signal"]):
                side = "LONG" if bool(row["long_signal"]) else "SHORT"
                entry = float(row["close"])
                atr = row["atr14"]
                if pd.notna(atr):
                    stop = float(row["low"] - atr * 0.1) if side == "LONG" else float(row["high"] + atr * 0.1)
                    risk = entry - stop if side == "LONG" else stop - entry
                    if risk > 0:
                        target = entry + 1.5 * risk if side == "LONG" else entry - 1.5 * risk
                        position = {"symbol": symbol, "side": side, "session": row["session"],
                                    "entry_ts": ts, "entry": entry, "stop": stop, "target": target,
                                    "risk_points": risk, "volume_filter": False}

    if position is not None and not rows.empty:
        last = rows.iloc[-1]
        trades.append(finish(position, last["ts"], float(last["close"]), "period_end", 0.03, 2.0))

    result = pd.DataFrame(trades)
    if result.empty:
        return result, {"symbol": symbol, "trades": 0, "net_points": 0.0, "win_rate_pct": 0.0,
                        "profit_factor": None, "max_drawdown_points": 0.0}
    wins = result.loc[result.net_pnl_points > 0, "net_pnl_points"].sum()
    losses = -result.loc[result.net_pnl_points < 0, "net_pnl_points"].sum()
    cumulative = result.net_pnl_points.cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0)
    return result, {
        "symbol": symbol, "trades": len(result),
        "long_trades": int((result.side == "LONG").sum()), "short_trades": int((result.side == "SHORT").sum()),
        "net_points": round(float(result.net_pnl_points.sum()), 2),
        "win_rate_pct": round(float((result.net_pnl_points > 0).mean() * 100), 2),
        "profit_factor": round(float(wins / losses), 3) if losses else None,
        "max_drawdown_points": round(float(-drawdown.min()), 2),
        "avg_trade_points": round(float(result.net_pnl_points.mean()), 3),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "NIFTY_50": DATA / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv",
        "BANK_NIFTY": DATA / "BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv",
    }
    summaries, all_trades = [], []
    for symbol, path in files.items():
        raw = load_data(path)
        for period, (start, end) in PERIODS.items():
            trades, summary = run_period(raw, symbol, start, end)
            summary.update(period=period, start_date=str(start), end_date=str(end))
            summaries.append(summary)
            if not trades.empty:
                trades = trades.copy()
                trades["period"] = period
                all_trades.append(trades)
            print(f"{symbol} | {period}: trades={summary['trades']} (L={summary.get('long_trades', 0)}, "
                  f"S={summary.get('short_trades', 0)}), net={summary['net_points']} pts, "
                  f"win={summary['win_rate_pct']}%, PF={summary['profit_factor']}, "
                  f"maxDD={summary['max_drawdown_points']} pts")
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(OUT / "trades.csv", index=False)
    print(f"Saved: {OUT}")
    print("Research only. Volume filter OFF because cached spot-index volume is zero; estimated costs are 0.03% per side + 2 points per side.")


if __name__ == "__main__":
    main()
