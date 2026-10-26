"""Client-facing combined strategy: improved pivot rules + NIFTY PE recovery."""
from __future__ import annotations

from typing import Final

from scripts.frozen_pivot_config import V5_SHORT_ONLY_VERSION

CLIENT_V1_VERSION: Final[str] = "client_v1"
CLIENT_V1_PIVOT_PRESET: Final[str] = V5_SHORT_ONLY_VERSION

CLIENT_V1_CHANGES: Final[tuple[str, ...]] = (
    "Pivot: SHORT ONLY — morning LONG trades removed (doc longs lost ~1,783 pts in 3m test).",
    "Pivot: NIFTY short window 11:00–13:00 IST; BANKNIFTY short 13:00–15:00 IST.",
    "Pivot: price below session VWAP for shorts; max risk NIFTY 40 / BANKNIFTY 100 pts.",
    "Pivot: breakeven at 0.75R; one short per symbol per session.",
    "PE (NIFTY only): morning -200 pts, cross up open-70, BUY PE 2 lots, SL 20 premium pts.",
)

PIVOT_PRESET_ALIASES: Final[dict[str, str]] = {
    "client_v1": CLIENT_V1_PIVOT_PRESET,
    "guide_baseline": "guide_baseline",
    "v5_short_only": V5_SHORT_ONLY_VERSION,
}
