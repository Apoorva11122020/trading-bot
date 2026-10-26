#!/usr/bin/env python3
"""Summarize daily pivot paper logs and enforce a 20-session evidence gate."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "reports" / "daily_pivot_paper"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize forward paper sessions (never enables live trading)")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--required-sessions", type=int, default=20)
    args = parser.parse_args()
    summary_path = args.reports_dir / "daily_summary.csv"
    out_path = args.reports_dir / "forward_20_session_review.md"
    if not summary_path.exists():
        raise SystemExit(f"No paper summary file found: {summary_path}")
    daily = pd.read_csv(summary_path)
    if daily.empty:
        lines = ["# Forward Paper Review", "",
                 f"Required evidence: {args.required_sessions} adequately covered sessions per index.", "",
                 "No forward paper sessions have been recorded yet. Start the tracker during market hours; this report will count only sessions with adequate live-candle coverage.", "",
                 "| Index | Adequate sessions | Remaining | Status |", "|---|---:|---:|---|"]
        for symbol in ("NIFTY", "BANKNIFTY"):
            lines.append(f"| {symbol} | 0 | {args.required_sessions} | IN PROGRESS; no evidence yet |")
        lines.extend(["", "The strategy remains paper-only. Costs are estimates and actual contract-note charges are not validated.", ""])
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved forward review: {out_path}")
        print(f"NIFTY: 0/{args.required_sessions} sessions; BANKNIFTY: 0/{args.required_sessions} sessions")
        return
    required = {"date", "symbol", "coverage_status", "completed_live_bars", "data_gaps"}
    missing = required - set(daily.columns)
    if missing:
        raise SystemExit(f"Paper summary lacks forward-coverage fields {sorted(missing)}; collect new sessions with the updated tracker.")

    lines = ["# Forward Paper Review", "", f"Required evidence: {args.required_sessions} adequately covered sessions per index.",
             "", "This report summarizes simulated spot-index trades only. It does not recommend or place live orders.", ""]
    adequate = daily[daily.coverage_status == "ADEQUATE"].copy()
    adequate["date"] = pd.to_datetime(adequate.date).dt.date
    rows = []
    for symbol in sorted(set(daily.symbol.astype(str))):
        sessions = (adequate[adequate.symbol.astype(str) == symbol]
                    .sort_values("date").drop_duplicates("date").tail(args.required_sessions))
        count = len(sessions)
        if count < args.required_sessions:
            rows.append({"symbol": symbol, "adequate_sessions": count,
                         "remaining": args.required_sessions - count, "trades": "—",
                         "wins": "—", "win_rate": "—", "net_points": "—", "profit_factor": "—",
                         "status": "IN PROGRESS; not enough evidence"})
            continue

        session_dates = {d.isoformat() for d in sessions.date}
        event_files = [args.reports_dir / f"{d}_{symbol}_paper.csv" for d in sorted(session_dates)]
        exits = []
        for file in event_files:
            if file.exists():
                frame = pd.read_csv(file)
                if "event" in frame and "net_points" in frame:
                    exits.append(frame[frame.event == "PAPER_EXIT"])
        trades = pd.concat(exits, ignore_index=True) if exits else pd.DataFrame()
        net = pd.to_numeric(trades.get("net_points", pd.Series(dtype=float)), errors="coerce").dropna()
        wins = net[net > 0]
        losses = net[net < 0]
        positive_total = float(wins.sum())
        negative_total = abs(float(losses.sum()))
        profit_factor = positive_total / negative_total if negative_total else None
        win_rate = float((net > 0).mean() * 100) if len(net) else None
        net_points = float(net.sum()) if len(net) else 0.0
        rows.append({"symbol": symbol, "adequate_sessions": count, "remaining": 0,
                     "trades": len(net), "wins": int((net > 0).sum()),
                     "win_rate": f"{win_rate:.1f}%" if win_rate is not None else "—",
                     "net_points": f"{net_points:.2f}",
                     "profit_factor": f"{profit_factor:.3f}" if profit_factor is not None else "—",
                     "status": "Review research assumptions and fees; never auto-live"})

    lines.extend(["| Index | Adequate sessions | Remaining | Closed trades | Wins | Win rate | Net points | Profit factor | Status |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in rows:
        lines.append("| {symbol} | {adequate_sessions} | {remaining} | {trades} | {wins} | {win_rate} | {net_points} | {profit_factor} | {status} |".format(**row))
    lines.extend(["", "## Evidence rules", "",
                  f"- A session counts only when the tracker recorded at least 60 completed live bars and zero detected 5-minute gaps.",
                  "- Net points use the configured estimated costs. They are not contract-note-verified futures/options P&L.",
                  "- Fewer than 20 adequate sessions means no performance conclusion. Twenty sessions still do not establish robust profitability.",
                  "- The prior seven-session historical backtest was negative for both indices. Keep this candidate paper-only while collecting forward observations.", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved forward review: {out_path}")
    for row in rows:
        print(f"{row['symbol']}: {row['adequate_sessions']}/{args.required_sessions} sessions; {row['status']}")


if __name__ == "__main__":
    main()
