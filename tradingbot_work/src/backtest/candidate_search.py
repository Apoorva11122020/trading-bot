"""Leak-safe candidate screening for the NIFTY S2 research workflow.

This module only uses point-in-time fields available at signal/entry time.
It deliberately excludes hindsight fields such as MFE/MAE, exit reason,
holding time and post-entry false-breakout labels.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class CandidateResult:
    rules: tuple[str, ...]
    train_n: int
    train_win_rate: float
    train_profit_factor: float
    train_pnl: float
    oos_n: int
    oos_win_rate: float
    oos_profit_factor: float
    oos_pnl: float
    current_n: int
    current_win_rate: float
    current_profit_factor: float
    current_pnl: float

    def as_dict(self):
        return asdict(self)


def _stats(frame: pd.DataFrame, pnl_col: str) -> tuple[int, float, float, float]:
    if frame.empty:
        return 0, 0.0, 0.0, 0.0
    wins = int(frame["win"].sum())
    gross_profit = float(frame.loc[frame[pnl_col] > 0, pnl_col].sum())
    gross_loss = float(-frame.loc[frame[pnl_col] < 0, pnl_col].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    return len(frame), wins / len(frame), pf, float(frame[pnl_col].sum())


def _historical_predicates(frame: pd.DataFrame) -> dict[str, pd.Series]:
    # All fields below exist at signal/entry time. Do not add post-entry fields.
    return {
        "SELL": frame["direction"].eq("SELL"),
        "RSI>=30": frame["rsi14"] >= 30,
        "RSI>=40": frame["rsi14"] >= 40,
        "RSI<=55": frame["rsi14"] <= 55,
        "RSI<=60": frame["rsi14"] <= 60,
        "ATR>=0.05%": frame["atr_pct"] >= 0.05,
        "ATR>=0.08%": frame["atr_pct"] >= 0.08,
        "EMA aligned": frame["ema_trend_aligned"].eq(True),
        "EMA opposed": frame["ema_trend_aligned"].eq(False),
        "Time11-14:30": frame["entry_time_bucket"].isin(
            ["11:00-12:00", "12:00-13:00", "13:00-14:30"]
        ),
        "Break>=0.05%": frame["distance_from_breakout_pct"] >= 0.05,
        "Option above VWAP": frame["option_vs_vwap"].eq("above"),
        "Gap!=flat": frame["gap_condition"].ne("flat"),
        "Gap-up": frame["gap_condition"].eq("gap_up"),
        "Gap-down": frame["gap_condition"].eq("gap_down"),
    }


def _current_predicates(frame: pd.DataFrame) -> dict[str, pd.Series]:
    # Current-day forward tracker has a smaller feature set. Missing fields are
    # intentionally not invented; candidates requiring unavailable fields are
    # omitted from current-day comparison.
    return {
        "SELL": frame["direction"].eq("SELL"),
        "RSI>=30": frame["rsi"] >= 30,
        "RSI>=40": frame["rsi"] >= 40,
        "RSI<=55": frame["rsi"] <= 55,
        "RSI<=60": frame["rsi"] <= 60,
        "EMA aligned": frame["ema_alignment"].str.contains("ALIGNED", na=False)
        & ~frame["ema_alignment"].str.contains("OPPOSED", na=False),
        "EMA opposed": frame["ema_alignment"].str.contains("OPPOSED", na=False),
        "Time11-14:30": frame["time_of_day"].isin(["11:00-12:00", "12:00-13:30"]),
        "Option above VWAP": frame["option_vs_vwap"].eq("ABOVE"),
    }


def screen(
    historical: pd.DataFrame,
    current: pd.DataFrame,
    train_end: str = "2026-09-23",
    min_train: int = 15,
    min_oos: int = 10,
    min_current: int = 5,
    max_rules: int = 3,
) -> list[CandidateResult]:
    """Screen small, predeclared rule combinations without using current data for selection."""
    h = historical.copy()
    c = current.copy()
    h["timestamp"] = pd.to_datetime(h["signal_time"])
    h["win"] = (h["outcome"] == "WIN").astype(int)
    h["split"] = h["timestamp"].dt.date <= pd.Timestamp(train_end).date()
    c["win"] = (c["outcome"] == "WIN").astype(int)

    hp = _historical_predicates(h)
    cp = _current_predicates(c)
    names = [name for name in hp if name in cp]
    results: list[CandidateResult] = []

    for width in range(1, max_rules + 1):
        for rules in combinations(names, width):
            hm = pd.Series(True, index=h.index)
            cm = pd.Series(True, index=c.index)
            for rule in rules:
                hm &= hp[rule].fillna(False)
                cm &= cp[rule].fillna(False)
            train = h[hm & h["split"]]
            oos = h[hm & ~h["split"]]
            cur = c[cm]
            if len(train) < min_train or len(oos) < min_oos or len(cur) < min_current:
                continue
            tn, tw, tpf, tp = _stats(train, "pnl")
            on, ow, opf, opp = _stats(oos, "pnl")
            cn, cw, cpf, cpp = _stats(cur, "pnl_1_lot")
            results.append(CandidateResult(rules, tn, tw, tpf, tp, on, ow, opf, opp, cn, cw, cpf, cpp))

    return sorted(results, key=lambda x: (x.oos_win_rate, x.oos_profit_factor, x.oos_n), reverse=True)
