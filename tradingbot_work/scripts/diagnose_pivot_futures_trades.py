#!/usr/bin/env python3
"""PAPER-only diagnostics for classic-pivot futures backtest trades. No broker calls/orders."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = ROOT / "reports" / "pivot_futures_backtest"


def stats(g: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(g["net_points"], errors="coerce").dropna()
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    eq = pnl.cumsum()
    dd = eq - eq.cummax().clip(lower=0)
    hold = (pd.to_datetime(g["exit_ts"]) - pd.to_datetime(g["entry_ts"])).dt.total_seconds() / 60
    return {
        "trades": int(len(pnl)),
        "net_points": round(float(pnl.sum()), 2),
        "avg_points": round(float(pnl.mean()), 3) if len(pnl) else 0.0,
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 2) if len(pnl) else 0.0,
        "profit_factor": round(gains / losses, 3) if losses > 0 else None,
        "max_drawdown_points": round(float(-dd.min()), 2) if len(dd) else 0.0,
        "avg_hold_minutes": round(float(hold.mean()), 1) if len(hold) else 0.0,
    }


def grouped(frame: pd.DataFrame, by: list[str], group_name: str) -> pd.DataFrame:
    rows = []
    for keys, g in frame.groupby(by, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({"group": group_name, **{k: v for k, v in zip(by, keys)}, **stats(g)})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    a = p.parse_args()
    trades_path = a.reports_dir / "trades.csv"
    coverage_path = a.reports_dir / "data_coverage.csv"
    if not trades_path.is_file():
        raise SystemExit(f"Missing {trades_path}; run classic_pivot_futures_backtest.py first.")
    t = pd.read_csv(trades_path)
    required = {"symbol", "side", "entry_ts", "exit_ts", "exit_reason", "net_points", "volume_filter"}
    missing = required - set(t.columns)
    if missing:
        raise SystemExit(f"{trades_path}: missing columns {sorted(missing)}")
    t["entry_ts"] = pd.to_datetime(t["entry_ts"], errors="coerce")
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], errors="coerce")
    t["net_points"] = pd.to_numeric(t["net_points"], errors="coerce")
    t = t.dropna(subset=["entry_ts", "exit_ts", "net_points"]).copy()
    t["month"] = t["entry_ts"].dt.strftime("%Y-%m")
    t["split"] = np.where(t["entry_ts"].dt.date >= pd.Timestamp("2026-06-01").date(), "OOS", "DEVELOPMENT")

    outputs = []
    for (symbol, vf, split), sub in t.groupby(["symbol", "volume_filter", "split"], sort=True):
        for field in ("side", "exit_reason", "month"):
            out = grouped(sub, [field], field)
            out.insert(0, "split", split)
            out.insert(0, "volume_filter", vf)
            out.insert(0, "symbol", symbol)
            outputs.append(out)
        overall = pd.DataFrame([{
            "group": "overall", "symbol": symbol, "volume_filter": vf, "split": split, **stats(sub)
        }])
        outputs.append(overall)
    result = pd.concat(outputs, ignore_index=True, sort=False)
    out_path = a.reports_dir / "trade_diagnostics.csv"
    result.to_csv(out_path, index=False)

    print("PAPER | RESEARCH ONLY | ORDERS: 0")
    print("Diagnostics saved:", out_path.resolve())
    print("\nOOS overview (estimated-cost net points):")
    overview = result[(result["group"] == "overall") & (result["split"] == "OOS")]
    cols = ["symbol", "volume_filter", "trades", "net_points", "win_rate_pct", "profit_factor",
            "max_drawdown_points", "avg_hold_minutes"]
    print(overview[cols].to_string(index=False) if not overview.empty else "No OOS trades.")
    print("\nUse group=side / exit_reason / month rows in trade_diagnostics.csv to investigate where losses occur.")
    if coverage_path.is_file():
        coverage = pd.read_csv(coverage_path)
        if "coverage_status" in coverage and (coverage["coverage_status"] != "COMPLETE").any():
            print("WARNING: input backtest has incomplete data coverage; diagnostics inherit that limitation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
