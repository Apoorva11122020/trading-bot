#!/usr/bin/env python3
"""Matched BANKNIFTY futures paper observer for the frozen body-strength candidate.

Read-only Upstox market data. Signals use BANKNIFTY spot; fills, ATR, volume,
stops and targets use the nearest currently listed BANKNIFTY future. No orders.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from datetime import date, datetime, time as daytime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

from scripts.classic_pivot_futures_backtest import make_signals, upstox_futures_fees_inr
from scripts.pivot_indicators import atr
from src.historical_data.downloader import month_chunks
from src.historical_data.service import build_catalog
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.market_data.models import Tick
from src.paper_trading.upstox_feed import create_upstox_v3_feed
from src.safety.live_gate import assert_paper_trading_only

IST = ZoneInfo("Asia/Kolkata")
SPOT_KEY = "NSE_INDEX|Nifty Bank"
FIELDS = ["timestamp", "event", "side", "contract", "entry", "stop", "target", "exit",
          "gross_points", "estimated_fees_inr", "estimated_cost_points", "net_points", "reason", "note"]


def bucket(ts: datetime) -> datetime:
    return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)


def historical(client, key: str, start: date, end: date) -> dict[datetime, dict]:
    result = {}
    for a, b in month_chunks(start, end):
        for candle in client.candles(key, a, b, unit="minutes", interval=5):
            ts = candle.timestamp.astimezone(IST)
            if ts + timedelta(minutes=5) <= datetime.now(IST):
                result[ts] = {"open": candle.open, "high": candle.high, "low": candle.low,
                              "close": candle.close, "volume": candle.volume}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "daily_banknifty_body_paper")
    parser.add_argument("--slippage-points-per-side", type=float, default=1.0)
    args = parser.parse_args()
    if args.slippage_points_per_side < 0:
        parser.error("slippage must be non-negative")
    assert_paper_trading_only()
    now = datetime.now(IST)
    if not daytime(9, 0) <= now.time() <= daytime(15, 15):
        raise SystemExit("Paper observer not started: run 09:00–15:15 IST on a trading day. No daily result recorded.")
    today = now.date()
    client = UpstoxHistoricalClient()
    contracts = [c for c in build_catalog(client).active_contracts()
                 if c.underlying == "BANKNIFTY" and c.instrument_type == "FUT" and c.expiry >= today]
    if not contracts:
        raise SystemExit("No active BANKNIFTY futures contract found in Upstox instrument master.")
    contract = min(contracts, key=lambda c: c.expiry)
    start = today - timedelta(days=60)
    spot = historical(client, SPOT_KEY, start, today)
    fut = historical(client, contract.instrument_key, start, today)
    if len(spot) < 60 or len(fut) < 60:
        raise SystemExit(f"Insufficient warm-up bars: spot={len(spot)}, futures={len(fut)}; no observer started.")

    def rows_frame(s: dict, f: dict) -> pd.DataFrame:
        paired = sorted(set(s).intersection(f))
        spot_df = pd.DataFrame([{**s[t], "ts": t, "session": t.date()} for t in paired])
        fut_df = pd.DataFrame([{**f[t], "ts": t, "instrument_key": contract.instrument_key,
                                "trading_symbol": contract.trading_symbol, "expiry": contract.expiry,
                                "lot_size": contract.lot_size} for t in paired])
        if not paired:
            return pd.DataFrame()
        fut_df["atr14_fut"] = atr(fut_df, 14)
        fut_df["vol_avg20_fut"] = fut_df["volume"].shift(1).rolling(20, min_periods=20).mean()
        return make_signals(spot_df, fut_df)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{today.isoformat()}_BANKNIFTY_FUT_body_paper.csv"
    summary_file = args.out / "daily_summary.csv"
    if not out_file.exists():
        with out_file.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    state = {"bars": {"spot": spot, "fut": fut}, "open": {"spot": None, "fut": None},
             "ticks": {"spot": 0, "fut": 0}, "last_vtt": None, "volume_gaps": 0,
             "first_tick": {"spot": None, "fut": None}, "gaps": 0, "position": None,
             "pending": None, "completed": set(), "trades": []}
    lock = threading.RLock()

    def log(event: str, ts: datetime, **values) -> None:
        row = {"timestamp": ts.isoformat(), "event": event, **values}
        with out_file.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow({k: row.get(k, "") for k in FIELDS})
        print(f"[{event}] {ts:%H:%M} {values.get('side','')} net={values.get('net_points','')} {values.get('reason','')}", flush=True)

    def finish(ts: datetime, price: float, reason: str) -> None:
        p = state["position"]
        gross = price - p["entry"]
        fees = upstox_futures_fees_inr(p["entry"], price, "LONG", contract.lot_size, p["entry_ts"].date(), ts.date())
        fees_pts = fees / contract.lot_size
        cost = fees_pts + 2 * args.slippage_points_per_side
        net = gross - cost
        trade = {"timestamp": ts.isoformat(), "side": "LONG", "contract": contract.trading_symbol,
                 "entry": round(p["entry"], 2), "stop": round(p["stop"], 2), "target": round(p["target"], 2),
                 "exit": round(price, 2), "gross_points": round(gross, 2), "estimated_fees_inr": round(fees, 2),
                 "estimated_cost_points": round(cost, 2), "net_points": round(net, 2), "reason": reason,
                 "note": "One-lot paper estimate; 1 point/side slippage; fees are modeled, not contract-note actual"}
        state["trades"].append(trade)
        log("PAPER_EXIT", ts, **{k: v for k, v in trade.items() if k != "timestamp"})
        state["position"] = None

    def evaluate(ts: datetime) -> None:
        if ts in state["completed"] or ts not in state["bars"]["spot"] or ts not in state["bars"]["fut"]:
            return
        state["completed"].add(ts)
        if ts.date() != today:
            return
        frame = rows_frame(state["bars"]["spot"], state["bars"]["fut"])
        if frame.empty:
            return
        row = frame.iloc[-1]
        if len(frame) < 2 or frame.iloc[-2]["ts"].date() != today or ts - frame.iloc[-2]["ts"] != timedelta(minutes=5):
            return
        fb = state["bars"]["fut"][ts]
        pos = state["position"]
        if pos:
            px = None
            why = ""
            if fb["open"] <= pos["stop"]:
                px, why = fb["open"], "stop_gap"
            elif fb["low"] <= pos["stop"]:
                px, why = pos["stop"], "stop (stop-first)"
            elif fb["high"] >= pos["target"]:
                px, why = pos["target"], "target"
            elif ts.time() >= daytime(15, 0):
                px, why = fb["close"], "intraday_square_off"
            if px is not None:
                finish(ts + timedelta(minutes=5), float(px), why)
        if state["position"] is None and state["pending"]:
            pending = state["pending"]
            state["pending"] = None
            if pending["ts"].date() == today and ts == pending["ts"] + timedelta(minutes=5):
                entry = float(fb["open"])
                stop = float(pending["low"] - 0.1 * pending["atr"])
                risk = entry - stop
                if risk > 0:
                    state["position"] = {"entry": entry, "stop": stop, "target": entry + 1.5*risk,
                                          "entry_ts": ts}
                    log("PAPER_ENTRY", ts, side="LONG", contract=contract.trading_symbol,
                        entry=round(entry, 2), stop=round(stop, 2), target=round(entry+1.5*risk, 2),
                        note="Frozen signal; next futures bar open proxy")
                    # Entry bar is checked conservatively, matching the backtest model.
                    p = state["position"]
                    if fb["low"] <= p["stop"]:
                        finish(ts + timedelta(minutes=5), min(float(fb["open"]), p["stop"]), "stop (stop-first)")
                    elif fb["high"] >= p["target"]:
                        finish(ts + timedelta(minutes=5), p["target"], "target")
        if state["position"] is None and state["pending"] is None and ts.time() < daytime(14, 55):
            body_range = float(row["high_spot"] - row["low_spot"])
            body_ratio = abs(float(row["close_spot"] - row["open_spot"])) / body_range if body_range > 0 else 0
            vol_ok = pd.notna(row["vol_avg20_fut"]) and row["vol_avg20_fut"] > 0 and row["volume_fut"] > row["vol_avg20_fut"] * 1.2
            signal_ok = bool(row["long_signal"]) and float(row["close_spot"]) > float(row["open_spot"]) and body_ratio >= 0.8 and vol_ok
            if signal_ok and pd.notna(row["atr14_fut"]) and float(row["atr14_fut"]) > 0:
                state["pending"] = {"ts": ts, "low": float(row["low_fut"]), "atr": float(row["atr14_fut"])}
                log("PAPER_SIGNAL", ts + timedelta(minutes=5), side="LONG", contract=contract.trading_symbol,
                    note=f"body_ratio={body_ratio:.3f}; futures volume={row['volume_fut']:.0f}; prior20avg={row['vol_avg20_fut']:.0f}; awaiting next bar")

    def on_tick(tick: Tick) -> None:
        typ = "spot" if tick.symbol == SPOT_KEY else "fut" if tick.symbol == contract.instrument_key else None
        if typ is None or tick.ltp <= 0:
            return
        ts = tick.timestamp.astimezone(IST)
        if ts.date() != today or not daytime(9, 15) <= ts.time() < daytime(15, 30):
            return
        with lock:
            state["ticks"][typ] += 1
            if state["first_tick"][typ] is None:
                state["first_tick"][typ] = ts
            volume_delta = 0.0
            if typ == "fut":
                total = float(tick.volume or 0.0)
                previous = state["last_vtt"]
                if previous is not None and total >= previous:
                    volume_delta = total - previous
                elif previous is not None:
                    state["volume_gaps"] += 1
                state["last_vtt"] = total
            start_bar = bucket(ts)
            current = state["open"][typ]
            if current is None:
                state["open"][typ] = {"timestamp": start_bar, "open": tick.ltp, "high": tick.ltp,
                                       "low": tick.ltp, "close": tick.ltp, "volume": volume_delta}
                return
            if start_bar < current["timestamp"]:
                return
            if start_bar == current["timestamp"]:
                current["high"] = max(current["high"], tick.ltp)
                current["low"] = min(current["low"], tick.ltp)
                current["close"] = tick.ltp
                current["volume"] += volume_delta
                return
            elapsed = start_bar - current["timestamp"]
            if elapsed != timedelta(minutes=5):
                state["gaps"] += 1
                log("DATA_GAP", current["timestamp"], note=f"{typ} feed advanced by {elapsed}; skipped bar")
            else:
                state["bars"][typ][current["timestamp"]] = {k: current[k] for k in ("open", "high", "low", "close", "volume")}
                evaluate(current["timestamp"])
            state["open"][typ] = {"timestamp": start_bar, "open": tick.ltp, "high": tick.ltp,
                                   "low": tick.ltp, "close": tick.ltp, "volume": volume_delta}

    feed = create_upstox_v3_feed(stale_after_seconds=15, mode="full")
    feed.set_tick_handler(on_tick)
    feed.subscribe([SPOT_KEY, contract.instrument_key])
    print(f"PAPER ONLY | BANKNIFTY FUT {contract.trading_symbol} | frozen LONG body>=0.8 | volume > prior20 x1.2 | RR 1.5 | 0.1 ATR buffer")
    print(f"Warm-up aligned bars: {len(set(spot).intersection(fut))} | expiry={contract.expiry} lot={contract.lot_size} | estimated Upstox fees + {args.slippage_points_per_side:g} point/side | orders=0")
    print(f"Journal: {out_file}")
    stopped = threading.Event()
    def shutdown(*_):
        stopped.set()
        feed.disconnect()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    feed.connect()
    try:
        while not stopped.is_set():
            time.sleep(1)
            if datetime.now(IST).time() >= daytime(15, 35):
                break
    finally:
        feed.disconnect()
    trades = pd.DataFrame(state["trades"])
    net = pd.to_numeric(trades.get("net_points", pd.Series(dtype=float)), errors="coerce").dropna()
    full_opening_coverage = all(t is not None and t.time() <= daytime(9, 20) for t in state["first_tick"].values())
    summary = {"date": today.isoformat(), "symbol": "BANKNIFTY", "contract": contract.trading_symbol,
               "closed_trades": len(trades), "wins": int((net > 0).sum()), "losses": int((net <= 0).sum()),
               "net_points": round(float(net.sum()), 2), "coverage_status": "ADEQUATE" if full_opening_coverage and state["gaps"] == 0 and state["volume_gaps"] == 0 else "PARTIAL",
               "spot_ticks": state["ticks"]["spot"], "futures_ticks": state["ticks"]["fut"], "data_gaps": state["gaps"],
               "first_spot_tick": state["first_tick"]["spot"].isoformat() if state["first_tick"]["spot"] else "",
               "first_futures_tick": state["first_tick"]["fut"].isoformat() if state["first_tick"]["fut"] else "",
               "volume_resets_or_gaps": state["volume_gaps"], "open_position_at_shutdown": bool(state["position"]),
               "orders_submitted": 0, "estimated_costs_only": True}
    prior = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
    if not prior.empty:
        prior = prior[~((prior.date.astype(str) == today.isoformat()) & (prior.symbol == "BANKNIFTY"))]
    pd.concat([prior, pd.DataFrame([summary])], ignore_index=True).to_csv(summary_file, index=False)
    print(f"DAILY PAPER SUMMARY | BANKNIFTY FUT closed={summary['closed_trades']} wins={summary['wins']} losses={summary['losses']} net={summary['net_points']} pts coverage={summary['coverage_status']} orders=0")
    print("Stopped. No broker orders were submitted.")


if __name__ == "__main__":
    main()
