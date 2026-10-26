"""Optional filters on top of the doc guide (classic pivot) backtest."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

from scripts.classic_pivot_historical_backtest import finish, prepare

GUIDE_FILTER_CHOICES: Final[tuple[str, ...]] = (
    "none",
    "no_after_14",
    "short_only",
    "one_per_day",
    "risk_cap",
    "short_nifty_11_13",
    "bn_short_13_15",
    "doc_improved",
)

GUIDE_FILTER_LABELS: Final[dict[str, str]] = {
    "none": "No extra filter (pure doc)",
    "no_after_14": "No new entry after 2:00 PM",
    "short_only": "Short only (no long trades)",
    "one_per_day": "One trade per symbol per day",
    "risk_cap": "Max risk NIFTY 40 / BANKNIFTY 100 pts",
    "short_nifty_11_13": "NIFTY short only 11:00-13:00",
    "bn_short_13_15": "BANKNIFTY short only 13:00-15:00",
    "doc_improved": "Recommended combo (short windows + risk + 1/day)",
}


@dataclass(frozen=True)
class DocGuideFilter:
    name: str
    short_only: bool = False
    no_entry_after_minute: int | None = None
    one_trade_per_day: bool = False
    risk_cap: dict[str, float] | None = None
    short_windows: dict[str, tuple[int, int]] | None = None
    long_windows: dict[str, tuple[int, int]] | None = None


def filter_profile(name: str) -> DocGuideFilter:
    if name == "none":
        return DocGuideFilter(name="none")
    if name == "no_after_14":
        return DocGuideFilter(name=name, no_entry_after_minute=14 * 60)
    if name == "short_only":
        return DocGuideFilter(name=name, short_only=True)
    if name == "one_per_day":
        return DocGuideFilter(name=name, one_trade_per_day=True)
    if name == "risk_cap":
        return DocGuideFilter(name=name, risk_cap={"NIFTY": 40.0, "BANKNIFTY": 100.0})
    if name == "short_nifty_11_13":
        return DocGuideFilter(
            name=name,
            short_only=True,
            short_windows={"NIFTY": (11 * 60, 13 * 60)},
        )
    if name == "bn_short_13_15":
        return DocGuideFilter(
            name=name,
            short_only=True,
            short_windows={"BANKNIFTY": (13 * 60, 15 * 60)},
        )
    if name == "doc_improved":
        return DocGuideFilter(
            name=name,
            short_only=True,
            no_entry_after_minute=15 * 60,
            one_trade_per_day=True,
            risk_cap={"NIFTY": 40.0, "BANKNIFTY": 100.0},
            short_windows={
                "NIFTY": (11 * 60, 13 * 60),
                "BANKNIFTY": (13 * 60, 15 * 60),
            },
        )
    return _resolve_combo_profile(name)


COMBO_FILTER_SPECS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("short_nifty_11_13+one_per_day", ("short_nifty_11_13", "one_per_day")),
    ("short_nifty_11_13+risk_cap", ("short_nifty_11_13", "risk_cap")),
    ("short_nifty_11_13+one_per_day+risk_cap", ("short_nifty_11_13", "one_per_day", "risk_cap")),
    ("short_nifty_11_13+no_after_14", ("short_nifty_11_13", "no_after_14")),
    ("short_only+one_per_day+no_after_14", ("short_only", "one_per_day", "no_after_14")),
    ("short_only+no_after_14", ("short_only", "no_after_14")),
    ("one_per_day+risk_cap", ("one_per_day", "risk_cap")),
)

COMBO_FILTER_LABELS: Final[dict[str, str]] = {
    "short_nifty_11_13+one_per_day": "COMBO: NIFTY short 11-13 + 1 trade/day",
    "short_nifty_11_13+risk_cap": "COMBO: NIFTY short 11-13 + risk cap",
    "short_nifty_11_13+one_per_day+risk_cap": "COMBO: NIFTY short 11-13 + 1/day + risk cap",
    "short_nifty_11_13+no_after_14": "COMBO: NIFTY short 11-13 + no entry after 2 PM",
    "short_only+one_per_day+no_after_14": "COMBO: Short only + 1/day + no after 2 PM",
    "short_only+no_after_14": "COMBO: Short only + no entry after 2 PM",
    "one_per_day+risk_cap": "COMBO: One trade/day + risk cap",
}


def merge_profiles(*names: str) -> DocGuideFilter:
    parts = [filter_profile(name) for name in names if name in GUIDE_FILTER_LABELS]
    combo_id = "+".join(names)
    short_only = any(p.short_only for p in parts)
    no_mins = [p.no_entry_after_minute for p in parts if p.no_entry_after_minute is not None]
    no_entry_after_minute = min(no_mins) if no_mins else None
    one_trade_per_day = any(p.one_trade_per_day for p in parts)
    risk_cap: dict[str, float] = {}
    short_windows: dict[str, tuple[int, int]] = {}
    long_windows: dict[str, tuple[int, int]] = {}
    for part in parts:
        if part.risk_cap:
            risk_cap.update(part.risk_cap)
        if part.short_windows:
            short_windows.update(part.short_windows)
        if part.long_windows:
            long_windows.update(part.long_windows)
    return DocGuideFilter(
        name=combo_id,
        short_only=short_only,
        no_entry_after_minute=no_entry_after_minute,
        one_trade_per_day=one_trade_per_day,
        risk_cap=risk_cap or None,
        short_windows=short_windows or None,
        long_windows=long_windows or None,
    )


def _resolve_combo_profile(name: str) -> DocGuideFilter:
    for combo_id, parts in COMBO_FILTER_SPECS:
        if combo_id == name:
            return merge_profiles(*parts)
    raise ValueError(f"Unknown guide filter: {name}")


def resolve_profile(name: str) -> DocGuideFilter:
    if name in GUIDE_FILTER_LABELS:
        return filter_profile(name)
    return _resolve_combo_profile(name)


def all_filter_options() -> list[tuple[str, str]]:
    options = [(k, GUIDE_FILTER_LABELS[k]) for k in GUIDE_FILTER_CHOICES]
    options.extend((k, COMBO_FILTER_LABELS[k]) for k, _ in COMBO_FILTER_SPECS)
    return options


def _symbol_key(label: str) -> str:
    if label in {"NIFTY", "NIFTY_50"}:
        return "NIFTY"
    if label in {"BANKNIFTY", "BANK_NIFTY"}:
        return "BANKNIFTY"
    return label.upper()


def _entry_allowed(
    profile: DocGuideFilter,
    *,
    symbol_key: str,
    side: str,
    signal_minute: int,
    risk: float,
    trade_taken: bool,
) -> bool:
    if profile.short_only and side == "LONG":
        return False
    if profile.no_entry_after_minute is not None and signal_minute >= profile.no_entry_after_minute:
        return False
    if profile.one_trade_per_day and trade_taken:
        return False
    if profile.risk_cap is not None:
        cap = profile.risk_cap.get(symbol_key)
        if cap is not None and risk > cap:
            return False
    if side == "SHORT" and profile.short_windows is not None:
        window = profile.short_windows.get(symbol_key)
        if window is None:
            return False
        start, end = window
        if not (start <= signal_minute < end):
            return False
    if side == "LONG" and profile.long_windows is not None:
        window = profile.long_windows.get(symbol_key)
        if window is None:
            return False
        start, end = window
        if not (start <= signal_minute < end):
            return False
    return True


def backtest_guide_filtered_day(
    frame: pd.DataFrame,
    label: str,
    day: date,
    profile: DocGuideFilter,
    *,
    volume_filter: bool = False,
    commission_pct: float = 0.03,
    slippage_points: float = 2.0,
    atr_buffer: float = 0.1,
    rr: float = 1.5,
    already_prepared: bool = False,
) -> pd.DataFrame:
    x = frame if already_prepared else prepare(frame, volume_filter)
    day_rows = x.index[x.session == day].tolist()
    trades: list[dict] = []
    pos = None
    trade_taken = False
    symbol_key = _symbol_key(label)

    for idx in day_rows:
        row = x.loc[idx]
        ts = row.ts
        signal_minute = int(ts.hour * 60 + ts.minute)

        if pos is not None:
            exit_price = None
            reason = None
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
            if exit_price is None and bool(row.flat_time):
                exit_price, reason = float(row.close), "intraday_flat"
            if exit_price is not None:
                trades.append(finish(pos, ts, float(exit_price), reason, commission_pct, slippage_points))
                pos = None

        if pos is None and not (trades and trades[-1].get("exit_ts") == ts):
            if bool(row.long_signal) or bool(row.short_signal):
                side = "LONG" if bool(row.long_signal) else "SHORT"
                entry = float(row.close)
                atr_val = row.atr14
                if pd.notna(atr_val):
                    if side == "LONG":
                        stop = float(row.low - atr_val * atr_buffer)
                        risk = entry - stop
                        target = entry + risk * rr
                    else:
                        stop = float(row.high + atr_val * atr_buffer)
                        risk = stop - entry
                        target = entry - risk * rr
                    if risk > 0 and np.isfinite(risk) and _entry_allowed(
                        profile,
                        symbol_key=symbol_key,
                        side=side,
                        signal_minute=signal_minute,
                        risk=float(risk),
                        trade_taken=trade_taken,
                    ):
                        pos = {
                            "symbol": label,
                            "side": side,
                            "session": row.session,
                            "entry_ts": ts,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "risk_points": risk,
                            "volume_filter": volume_filter,
                            "guide_filter": profile.name,
                        }
                        trade_taken = True

    if pos is not None and day_rows:
        last = x.loc[day_rows[-1]]
        trades.append(finish(pos, last.ts, float(last.close), "period_end", commission_pct, slippage_points))
    return pd.DataFrame(trades)
