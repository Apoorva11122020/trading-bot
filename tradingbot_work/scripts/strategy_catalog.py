"""All backtestable strategy packs for UI and ranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from scripts.run_combined_last7_report import DEFAULT_ATR_BUFFER, DEFAULT_RR


@dataclass(frozen=True)
class StrategySpec:
    id: str
    label: str
    kind: str  # doc | enhanced | alt
    description: str
    preset: str | None
    guide_filter: str
    rr: float
    atr_buffer: float
    include_pe: bool
    symbols_default: tuple[str, ...] = ("NIFTY", "BANKNIFTY")
    alt_engine: str | None = None  # opening_range | vwap_reversion | ema_pullback


STRATEGIES: Final[tuple[StrategySpec, ...]] = (
    StrategySpec(
        id="doc_pure",
        label="Doc guide (pure — as written)",
        kind="doc",
        description="Classic pivot both sides, no extra filters, RR 1:1.5, ATR 0.1. Shows raw doc performance.",
        preset="guide_baseline",
        guide_filter="none",
        rr=1.5,
        atr_buffer=0.1,
        include_pe=True,
    ),
    StrategySpec(
        id="doc_optimized",
        label="Doc guide (optimized — best backtest)",
        kind="doc",
        description="NIFTY short 11–13, 1 trade/day, RR 1:2, ATR 0.08. Best 6-mo NIFTY (+27.5 pts Apr–Oct). Use 6-month range for client demo.",
        preset="guide_baseline",
        guide_filter="short_nifty_11_13+one_per_day",
        rr=2.0,
        atr_buffer=0.08,
        include_pe=True,
        symbols_default=("NIFTY",),
    ),
    StrategySpec(
        id="client_v1",
        label="Client v1 (short-only + VWAP + breakeven)",
        kind="enhanced",
        description="Frozen v5: short only, time windows, VWAP filter, breakeven 0.75R, risk caps.",
        preset="client_v1",
        guide_filter="none",
        rr=1.5,
        atr_buffer=0.1,
        include_pe=True,
    ),
    StrategySpec(
        id="v5_short_only",
        label="Frozen v5 short-only",
        kind="enhanced",
        description="Same engine as client v1 with v5 label for research comparison.",
        preset="v5_short_only",
        guide_filter="none",
        rr=1.5,
        atr_buffer=0.1,
        include_pe=True,
    ),
    StrategySpec(
        id="opening_range_breakout",
        label="Opening range breakout (09:15–09:45)",
        kind="alt",
        description="Break of 30-min opening range after 09:45. Spot-based, pivot-only leg.",
        preset=None,
        guide_filter="none",
        rr=1.5,
        atr_buffer=0.1,
        include_pe=False,
        alt_engine="opening_range",
    ),
    StrategySpec(
        id="vwap_reversion",
        label="VWAP reversion (1 ATR stretch)",
        kind="alt",
        description="Price stretches 1 ATR from session VWAP then crosses back. Spot-based.",
        preset=None,
        guide_filter="none",
        rr=1.5,
        atr_buffer=0.1,
        include_pe=False,
        alt_engine="vwap_reversion",
    ),
    StrategySpec(
        id="ema_pullback",
        label="EMA pullback (trend + RSI)",
        kind="alt",
        description="Pullback to EMA20 in EMA20/50 trend with RSI confirmation. Spot-based.",
        preset=None,
        guide_filter="none",
        rr=1.5,
        atr_buffer=0.1,
        include_pe=False,
        alt_engine="ema_pullback",
    ),
    StrategySpec(
        id="pe_only",
        label="NIFTY PE recovery only",
        kind="alt",
        description="Morning -200 pts, cross open-70, buy PE. No pivot trades.",
        preset=None,
        guide_filter="none",
        rr=DEFAULT_RR,
        atr_buffer=DEFAULT_ATR_BUFFER,
        include_pe=True,
        symbols_default=("NIFTY",),
    ),
)


def strategy_by_id(strategy_id: str) -> StrategySpec:
    for spec in STRATEGIES:
        if spec.id == strategy_id:
            return spec
    raise ValueError(f"Unknown strategy: {strategy_id}")


def list_strategies_for_ui() -> list[dict[str, str | float | bool | list[str]]]:
    return [
        {
            "id": s.id,
            "label": s.label,
            "kind": s.kind,
            "description": s.description,
            "include_pe": s.include_pe,
            "default_symbols": list(s.symbols_default),
            "preset": s.preset,
            "guide_filter": s.guide_filter,
            "rr": s.rr,
            "atr_buffer": s.atr_buffer,
        }
        for s in STRATEGIES
    ]
