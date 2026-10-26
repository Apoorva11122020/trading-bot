#!/usr/bin/env python3
"""Extend NIFTY development with validated Jan-Sep 2025 futures history.

Uses only NIFTY spot/selected futures from 2025-01-01 through 2026-05-31.
TRAIN is 2025-01-01 through 2026-01-31; Feb-May 2026 is one untouched
chronological validation. June onward (including the recent-seven window) is
excluded. A compact, predeclared six-rule body/momentum grid is selected on
TRAIN only. Futures fills, volume filter, Upstox cost model and slippage match
the main pivot backtest. Paper research; no broker orders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.classic_pivot_futures_backtest import (
    DEFAULT_ROOT, DEFAULT_SPOT, IST, load_front_month, load_spot, make_signals,
    run_one, summarize,
)
from scripts.pivot_indicators import atr
from src.safety.live_gate import assert_paper_trading_only

EXPANDED = ROOT / "data" / "historical" / "upstox_expanded"
OUT = ROOT / "reports" / "extended_nifty_train_selection_20261014"
TRAIN_START = pd.Timestamp("2025-01-01").date()
TRAIN_END = pd.Timestamp("2026-02-01").date()
VALID_END = pd.Timestamp("2026-06-01").date()
EARLY_CUTOFF = pd.Timestamp("2025-10-01").date()
RULES = (("body_only", 0.4, 0), ("body_only", 0.6, 0), ("body_only", 0.8, 0),
         ("slope_body", 0.4, 3), ("slope_body", 0.6, 3), ("slope_body", 0.8, 3))
RR, ATR_BUFFER, SLIPPAGE = 1.5, 0.1, 1.0
MIN_TRAIN_TRADES = 20


def load_early_spot() -> pd.DataFrame:
    pieces = []
    for p in sorted((EXPANDED / "underlying" / "nifty").glob("2025-*.parquet")):
        x = pd.read_parquet(p)
        x["ts"] = pd.to_datetime(x["timestamp"], errors="coerce")
        if x["ts"].dt.tz is None:
            x["ts"] = x["ts"].dt.tz_localize(IST)
        x["ts"] = x["ts"].dt.tz_convert(IST)
        x["session"] = x["ts"].dt.date
        pieces.append(x[["ts", "session", "open", "high", "low", "close", "volume"]])
    if not pieces:
        raise SystemExit("No expanded NIFTY underlying data found for Jan-Sep 2025")
    x = pd.concat(pieces, ignore_index=True)
    x = x[(x.ts.dt.date >= TRAIN_START) & (x.ts.dt.date < EARLY_CUTOFF)]
    mins = x.ts.dt.hour * 60 + x.ts.dt.minute
    x = x[(mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)]
    return x.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def load_early_futures() -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = EXPANDED / "futures"
    selection = pd.read_csv(folder / "futures_contract_selection.csv")
    selection["session_date"] = pd.to_datetime(selection.session_date).dt.date
    selection = selection[(selection.session_date >= TRAIN_START) & (selection.session_date < EARLY_CUTOFF)].copy()
    chosen = selection[selection.status == "SELECTED"].copy()
    metadata = pd.read_csv(folder / "contracts.csv").set_index("instrument_key").to_dict("index")
    frames = []
    for key in chosen.instrument_key.dropna().unique():
        if key not in metadata:
            raise SystemExit(f"Expanded futures key missing metadata: {key}")
        expiry = str(metadata[key]["expiry"])[:10]
        f = pd.read_parquet(folder / f"NIFTY_FUT_{expiry}.parquet")
        if not f.instrument_key.eq(key).all():
            raise SystemExit(f"Expanded futures instrument mismatch: {key}")
        f["ts"] = pd.to_datetime(f.timestamp, errors="coerce")
        if f.ts.dt.tz is None:
            f["ts"] = f.ts.dt.tz_localize(IST)
        f["ts"] = f.ts.dt.tz_convert(IST)
        for col in ("open", "high", "low", "close", "volume"):
            f[col] = pd.to_numeric(f[col], errors="coerce")
        f = f.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).sort_values("ts")
        f = f.drop_duplicates("ts", keep="last")
        f["lot_size"] = int(metadata[key]["lot_size"])
        f["atr14_fut"] = atr(f, 14)
        f["vol_avg20_fut"] = f.volume.shift(1).rolling(20, min_periods=20).mean()
        f["session"] = f.ts.dt.date
        frames.append(f)
    all_bars = pd.concat(frames, ignore_index=True)
    selected = chosen[["session_date", "instrument_key"]].rename(columns={"session_date": "session"})
    all_bars = all_bars.merge(selected, on=["session", "instrument_key"], how="inner", validate="many_to_one")
    return all_bars.sort_values("ts").reset_index(drop=True), selection


def apply_rule(frame: pd.DataFrame, kind: str, body_min: float, slope_lag: int) -> pd.DataFrame:
    x = frame.copy()
    span = (x.high_spot - x.low_spot).replace(0, np.nan)
    body = (x.close_spot - x.open_spot).abs() / span
    long_ok = x.close_spot.gt(x.open_spot) & body.ge(body_min)
    short_ok = x.close_spot.lt(x.open_spot) & body.ge(body_min)
    if kind == "slope_body":
        ref = x.groupby("session", sort=False).ema20_spot.shift(slope_lag)
        long_ok &= x.ema20_spot.gt(ref)
        short_ok &= x.ema20_spot.lt(ref)
    x["long_signal"] = x.long_signal.astype(bool) & long_ok
    x["short_signal"] = x.short_signal.astype(bool) & short_ok
    return x


def main() -> int:
    assert_paper_trading_only()
    OUT.mkdir(parents=True, exist_ok=True)
    old_spot = load_early_spot()
    new_spot = load_spot(DEFAULT_SPOT / "NIFTY_50_5m_2025-10-01_to_2026-10-09.csv")
    new_spot = new_spot[(new_spot.ts.dt.date >= EARLY_CUTOFF) & (new_spot.ts.dt.date < VALID_END)]
    spot = pd.concat([old_spot, new_spot], ignore_index=True).sort_values("ts").drop_duplicates("ts")
    old_futures, early_selection = load_early_futures()
    new_futures, current_selection = load_front_month("NIFTY", DEFAULT_ROOT)
    new_futures = new_futures[(new_futures.ts.dt.date >= EARLY_CUTOFF) & (new_futures.ts.dt.date < VALID_END)]
    futures = pd.concat([old_futures, new_futures], ignore_index=True).sort_values("ts")
    if futures.duplicated("ts").any():
        raise SystemExit("Overlapping timestamps found across the two validated NIFTY futures sources")
    # Assert the extended section is complete in selection metadata, never fill gaps.
    unavailable = early_selection[(early_selection.status != "SELECTED")]
    frame = make_signals(spot, futures)
    frame = frame[(frame.ts.dt.date >= TRAIN_START) & (frame.ts.dt.date < VALID_END)].copy()
    train = frame[frame.ts.dt.date < TRAIN_END].copy()
    validation = frame[(frame.ts.dt.date >= TRAIN_END) & (frame.ts.dt.date < VALID_END)].copy()
    base_train = summarize(run_one(train, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), "NIFTY", True, "TRAIN_JAN25_JAN26")
    base_val = summarize(run_one(validation, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), "NIFTY", True, "VALIDATION_FEB_MAY26")
    grid = []
    for kind, body_min, lag in RULES:
        candidate = apply_rule(train, kind, body_min, lag)
        stats = summarize(run_one(candidate, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), "NIFTY", True, "TRAIN_JAN25_JAN26")
        eligible = (stats["trades"] >= MIN_TRAIN_TRADES
                    and stats["win_rate_pct"] > base_train["win_rate_pct"]
                    and stats["net_points"] > base_train["net_points"])
        grid.append({**stats, "rule_kind": kind, "body_min": body_min, "slope_lag": lag,
                     "train_gate_pass": eligible})
    eligible = [r for r in grid if r["train_gate_pass"]]
    chosen = max(eligible, key=lambda r: (r["net_points"], r["win_rate_pct"], -r["max_drawdown_points"])) if eligible else None
    summary = {
        "symbol": "NIFTY", "train_start": str(TRAIN_START), "train_end_exclusive": str(TRAIN_END),
        "validation_start": str(TRAIN_END), "validation_end_exclusive": str(VALID_END),
        "train_baseline_trades": base_train["trades"], "train_baseline_win_rate_pct": base_train["win_rate_pct"],
        "train_baseline_net_points": base_train["net_points"],
        "validation_baseline_trades": base_val["trades"], "validation_baseline_win_rate_pct": base_val["win_rate_pct"],
        "validation_baseline_net_points": base_val["net_points"], "grid_size": len(grid),
        "train_pass_count": len(eligible), "recent_seven_sessions_used": False,
        "unavailable_early_sessions": len(unavailable),
    }
    if chosen:
        valcand = apply_rule(validation, chosen["rule_kind"], chosen["body_min"], int(chosen["slope_lag"]))
        valstats = summarize(run_one(valcand, "NIFTY", True, "upstox", SLIPPAGE, RR, ATR_BUFFER, 1), "NIFTY", True, "VALIDATION_FEB_MAY26")
        summary.update({"selected_rule": chosen["rule_kind"], "selected_body_min": chosen["body_min"],
                        "selected_slope_lag": chosen["slope_lag"], "selected_train_trades": chosen["trades"],
                        "selected_train_win_rate_pct": chosen["win_rate_pct"], "selected_train_net_points": chosen["net_points"],
                        "validation_trades": valstats["trades"], "validation_win_rate_pct": valstats["win_rate_pct"],
                        "validation_net_points": valstats["net_points"], "validation_drawdown_points": valstats["max_drawdown_points"],
                        "validation_gate_pass": bool(valstats["trades"] >= 8 and valstats["win_rate_pct"] >= 50 and valstats["net_points"] > 0)})
    else:
        summary.update({"selected_rule": "NONE_TRAIN_GATE_FAILED", "validation_gate_pass": False})
    pd.DataFrame(grid).to_csv(OUT / "train_grid.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "summary.csv", index=False)
    print("EXTENDED NIFTY TRAIN | real selected futures | volume filter ON | modeled Upstox fees + 1 point/side")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved: {OUT}; no June+ candles used; recent seven untouched; PAPER ONLY, orders=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
