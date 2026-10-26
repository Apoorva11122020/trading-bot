#!/usr/bin/env python3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multi_strategy_runner import rank_strategies

start = date(2026, 4, 25)
end = date(2026, 10, 25)
print("Ranking strategies...")
r = rank_strategies(start, end)
for row in r["rows"]:
    print(
        f"{row['strategy_id']:28} trades={row['trades']:3} "
        f"win={row['win_rate_pct']:5.1f}% net={row['net_points']:+8.1f} prof={row['profitable']}"
    )
print("Best:", r["best_net"]["strategy_id"], r["best_net"]["net_points"])
