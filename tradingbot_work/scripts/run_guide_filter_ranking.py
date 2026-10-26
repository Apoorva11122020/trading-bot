#!/usr/bin/env python3
"""Rank doc-guide filter profiles (singles + combinations)."""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from scripts.guide_filter_ranking import rank_guide_filters
from scripts.run_combined_last7_report import ATR_BUFFER_CHOICES, DEFAULT_ATR_BUFFER, DEFAULT_RR, RR_CHOICES
from src.safety.live_gate import assert_paper_trading_only

IST = ZoneInfo("Asia/Kolkata")
OUT = ROOT / "reports" / "guide_filter_ranking.csv"


def main() -> int:
    assert_paper_trading_only()
    parser = argparse.ArgumentParser(description="Compare doc guide filters and combos")
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s), default=date(2026, 7, 25))
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s), default=datetime.now(IST).date())
    parser.add_argument("--symbols", default="NIFTY,BANKNIFTY")
    parser.add_argument("--rr", type=float, choices=RR_CHOICES, default=DEFAULT_RR)
    parser.add_argument("--atr-buffer", type=float, choices=ATR_BUFFER_CHOICES, default=DEFAULT_ATR_BUFFER)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    result = rank_guide_filters(
        args.start, args.end, symbols, rr=args.rr, atr_buffer=args.atr_buffer,
    )
    ranked = pd.DataFrame(result["rows"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUT, index=False)

    print(
        f"=== DOC GUIDE FILTER RANKING | {args.start} to {args.end} | "
        f"RR 1:{args.rr:g} | ATR {args.atr_buffer:g} ===\n"
    )
    for row in result["rows"]:
        pf = row["profit_factor"] if row["profit_factor"] is not None else "n/a"
        tag = "PROFIT" if row["profitable"] else "loss"
        print(
            f"{row['filter_label']:52} | trades={int(row['trades']):3d} | "
            f"win={row['win_rate_pct']:5.1f}% | net={row['net_points']:+8.1f} | "
            f"PF={pf} | {tag}"
        )

    best_net = result["best_net"]
    best_win = result["best_win"]
    best_bal = result["best_balanced"]
    print(f"\nBEST NET: {best_net['filter_id']} — {best_net['filter_label']}")
    print(f"BEST WIN%: {best_win['filter_id']} — {best_win['win_rate_pct']}%")
    print(f"BEST BALANCED: {best_bal['filter_id']} — score {best_bal['balanced_score']}")
    if not result["any_profitable"]:
        print("\nNote: No filter was net profitable in this period on spot data.")
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
