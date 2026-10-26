#!/usr/bin/env python3
"""Daily frozen pivot v2 paper tracker using Upstox read-only market data. No orders."""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from datetime import datetime, time as daytime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from scripts.classic_pivot_historical_backtest import GUIDE_BASELINE_VERSION
from scripts.doc_guide_filters import _entry_allowed, resolve_profile
from scripts.run_combined_last7_report import ATR_BUFFER_CHOICES, DEFAULT_ATR_BUFFER, DEFAULT_RR, RR_CHOICES
from scripts.frozen_pivot_config import (
    CONFIG_VERSION,
    FLAT_SESSION_MINUTE,
    SIGNAL_SESSION_END_MINUTE,
    config_for_symbol,
    mark_side_taken,
    max_risk_for,
    side_quota_available,
    spot_paper_entry_window_ok,
)
from scripts.nifty_pe_contract_resolver import (
    build_nifty_pe_resolver,
    build_pe_recovery_order,
    format_pe_order_alert,
)
from scripts.nifty_pe_recovery_rule import (
    PeRecoverySessionState,
    evaluate_pe_recovery,
    format_alert,
    update_session_state,
)
from src.options.models import OptionResolutionError
from scripts.pivot_indicators import atr, ema, rsi
from src.historical_data.downloader import month_chunks
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.market_data.models import Tick
from src.paper_trading.upstox_feed import create_upstox_v3_feed

IST = ZoneInfo("Asia/Kolkata")
INSTRUMENTS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
}
FIELDS = ["timestamp", "symbol", "event", "side", "entry", "stop", "target", "exit",
          "gross_points", "estimated_cost_points", "net_points", "reason", "note"]


def bucket(ts: datetime) -> datetime:
    return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only daily pivot + NIFTY PE paper tracker")
    parser.add_argument("--symbol", choices=INSTRUMENTS, required=False, help="NIFTY or BANKNIFTY spot index")
    parser.add_argument(
        "--pivot-preset",
        choices=["client_v1", "guide_baseline", "v5_short_only"],
        default="guide_baseline",
        help="guide_baseline=doc strategy (default) | client_v1=short-only improved",
    )
    parser.add_argument("--slippage-points-per-side", type=float, default=2.0)
    parser.add_argument("--commission-percent-per-side", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "daily_pivot_paper")
    parser.add_argument("--strategy-id", default=None, help="Load RR/ATR/filter from strategy catalog")
    parser.add_argument("--guide-filter", default="none")
    parser.add_argument("--rr", type=float, default=DEFAULT_RR, choices=RR_CHOICES)
    parser.add_argument("--atr-buffer", type=float, default=DEFAULT_ATR_BUFFER, choices=ATR_BUFFER_CHOICES)
    parser.add_argument("--no-pe", action="store_true", help="Disable NIFTY PE recovery leg")
    args = parser.parse_args()

    if args.strategy_id:
        from scripts.strategy_catalog import strategy_by_id

        spec = strategy_by_id(args.strategy_id)
        args.pivot_preset = spec.preset or "guide_baseline"
        args.guide_filter = spec.guide_filter
        args.rr = spec.rr
        args.atr_buffer = spec.atr_buffer
        if not spec.include_pe:
            args.no_pe = True
        if spec.symbols_default:
            sym = spec.symbols_default[0]
            if sym in INSTRUMENTS:
                args.symbol = sym
    if args.slippage_points_per_side < 0 or args.commission_percent_per_side < 0:
        parser.error("cost assumptions must be non-negative")
    if not args.symbol:
        parser.error("--symbol is required unless --strategy-id sets it")

    symbol, key = args.symbol, INSTRUMENTS[args.symbol]
    now = datetime.now(IST)
    if not (daytime(9, 0) <= now.time() <= daytime(15, 15)):
        raise SystemExit("Market monitor not started: run between 09:00 and 15:15 IST on a trading day.")
    today = now.date()
    client = UpstoxHistoricalClient()
    pe_resolver = build_nifty_pe_resolver() if symbol == "NIFTY" and not args.no_pe else None
    history_start = today - timedelta(days=45)
    bars = []
    for chunk_start, chunk_end in month_chunks(history_start, today):
        bars.extend(client.candles(key, chunk_start, chunk_end, unit="minutes", interval=5))
    data = {c.timestamp.astimezone(IST): {"open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume} for c in bars if c.timestamp.astimezone(IST) + timedelta(minutes=5) <= now}
    if len(data) < 60:
        raise SystemExit(f"Not enough completed 5m history from Upstox ({len(data)} bars).")

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"{today.isoformat()}_{symbol}_paper.csv"
    summary_file = args.out / "daily_summary.csv"
    if not out_file.exists():
        with out_file.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    lock = threading.RLock()
    guide_mode = args.pivot_preset == "guide_baseline"
    pivot_label = (
        GUIDE_BASELINE_VERSION if guide_mode
        else ("client_v1" if args.pivot_preset == "client_v1" else CONFIG_VERSION)
    )
    symbol_cfg = config_for_symbol(symbol) if not guide_mode else None
    state = {"open_bar": None, "position": None, "bars": data, "live_ticks": 0,
             "live_bars": 0, "first_live_tick": None, "last_live_tick": None, "data_gaps": 0,
             "long_taken_today": False, "short_taken_today": False, "trade_taken_today": False,
             "pe_state": PeRecoverySessionState(), "pe_session_day": None, "pe_position": None}
    max_risk = max_risk_for(symbol, symbol_cfg) if symbol_cfg is not None else None
    rr = float(args.rr)
    atr_buffer = float(args.atr_buffer)
    guide_profile = resolve_profile(args.guide_filter) if guide_mode and args.guide_filter != "none" else None

    def log_event(event: str, ts: datetime, *, side="", entry="", stop="", target="", exit_price="",
                  gross="", cost="", net="", reason="", note="") -> None:
        row = {"timestamp": ts.isoformat(), "symbol": symbol, "event": event, "side": side,
               "entry": entry, "stop": stop, "target": target, "exit": exit_price,
               "gross_points": gross, "estimated_cost_points": cost, "net_points": net,
               "reason": reason, "note": note}
        with out_file.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"[{event}] {ts:%H:%M} {symbol} {side} entry={entry} stop={stop} target={target} "
              f"exit={exit_price} net_pts={net} {reason} {note}", flush=True)

    def complete_bar(ts: datetime, bar: dict, next_open_price: float, next_start: datetime) -> None:
        pivot_exited_this_bar = False
        state["bars"][ts] = bar
        if ts.date() == today:
            state["live_bars"] += 1
        all_rows = [(t, v) for t, v in sorted(state["bars"].items()) if daytime(9, 15) <= t.timetz().replace(tzinfo=None) < daytime(15, 30)]
        frame = pd.DataFrame([{**v, "ts": t, "session": t.date()} for t, v in all_rows])
        day = ts.date()
        previous = frame[frame.session < day]
        if previous.empty:
            return
        prev_date = previous.session.max()
        prior = previous[previous.session == prev_date]
        pp_base = (float(prior.high.max()) + float(prior.low.min()) + float(prior.iloc[-1].close)) / 3
        pp, r1, s1 = pp_base, 2 * pp_base - float(prior.low.min()), 2 * pp_base - float(prior.high.max())
        frame["ema20"] = ema(frame.close, 20)
        frame["ema50"] = ema(frame.close, 50)
        frame["rsi14"] = rsi(frame.close, 14)
        frame["atr14"] = atr(frame, 14)
        frame["typical"] = (frame.high + frame.low + frame.close) / 3.0
        frame["session_typical_avg"] = frame.groupby("session", sort=False)["typical"].expanding().mean().reset_index(
            level=0, drop=True
        )
        current_idx = frame.index[frame.ts == ts]
        if len(current_idx) == 0:
            return
        i = int(current_idx[-1])
        row = frame.iloc[i]

        pe_pos = state["pe_position"]
        if symbol == "NIFTY" and pe_pos is not None and pe_pos["session"] == day and ts >= pe_pos["entry_ts"]:
            exit_premium, reason = None, ""
            try:
                candles = client.intraday_candles(pe_pos["instrument_key"], unit="minutes", interval=1)
                if candles:
                    last = candles[-1]
                    current = float(last.close)
                    bar_low = float(last.low)
                    if bar_low <= pe_pos["stop_premium"]:
                        exit_premium, reason = pe_pos["stop_premium"], "pe_stop (premium)"
                    elif (ts.hour * 60 + ts.minute) >= FLAT_SESSION_MINUTE:
                        exit_premium, reason = current, "pe_session_flat"
            except OptionResolutionError:
                exit_premium = None
            if exit_premium is not None:
                gross_inr = (float(exit_premium) - pe_pos["entry_premium"]) * pe_pos["quantity"]
                log_event(
                    "PE_EXIT", ts, side="PE", entry=pe_pos["entry_premium"], stop=pe_pos["stop_premium"],
                    target="", exit_price=round(float(exit_premium), 2), gross=round(gross_inr, 2),
                    cost=0, net=round(gross_inr, 2), reason=reason,
                    note=f"{pe_pos['trading_symbol']} qty={pe_pos['quantity']} premium P&L INR",
                )
                state["pe_position"] = None

        pos = state["position"]
        if pos is not None and pos["session"] == day and ts >= pos["entry_ts"]:
            if not guide_mode and not pos.get("breakeven_armed"):
                trigger = pos["entry"] + pos["initial_risk"] * symbol_cfg.breakeven_trigger_r * (
                    1 if pos["side"] == "LONG" else -1
                )
                if pos["side"] == "LONG" and bar["close"] >= trigger and bar["high"] >= trigger:
                    reserve = 2 * args.slippage_points_per_side + (pos["entry"] * args.commission_percent_per_side / 100) * 2
                    pos["stop"] = pos["entry"] + reserve
                    pos["breakeven_armed"] = True
                elif pos["side"] == "SHORT" and bar["close"] <= trigger and bar["low"] <= trigger:
                    reserve = 2 * args.slippage_points_per_side + (pos["entry"] * args.commission_percent_per_side / 100) * 2
                    pos["stop"] = pos["entry"] - reserve
                    pos["breakeven_armed"] = True

            exit_price, reason = None, ""
            if pos["side"] == "LONG":
                if bar["low"] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop (stop-first if both touched)"
                elif bar["high"] >= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            else:
                if bar["high"] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "stop (stop-first if both touched)"
                elif bar["low"] <= pos["target"]:
                    exit_price, reason = pos["target"], "target"
            if exit_price is None and (ts.hour * 60 + ts.minute) >= FLAT_SESSION_MINUTE:
                exit_price, reason = bar["close"], "daily paper square-off"
            if exit_price is not None:
                side_mult = 1 if pos["side"] == "LONG" else -1
                gross = (float(exit_price) - pos["entry"]) * side_mult
                cost = 2 * args.slippage_points_per_side + (pos["entry"] + float(exit_price)) * args.commission_percent_per_side / 100
                log_event("PAPER_EXIT", ts, side=pos["side"], entry=round(pos["entry"], 2),
                          stop=round(pos["stop"], 2), target=round(pos["target"], 2),
                          exit_price=round(float(exit_price), 2), gross=round(gross, 2), cost=round(cost, 2),
                          net=round(gross - cost, 2), reason=reason,
                          note=f"{pivot_label}; spot-index points; estimated costs")
                state["position"] = None
                pivot_exited_this_bar = True
                if not guide_mode:
                    return

        prev_row = frame.iloc[i - 1] if i > 0 else None
        if symbol == "NIFTY" and prev_row is not None and row.ts.date() == day and prev_row.ts.date() == day:
            if row.ts.date() != state["pe_session_day"]:
                state["pe_state"] = PeRecoverySessionState()
                state["pe_session_day"] = row.ts.date()
            day_frame = frame[frame.session == day]
            if not day_frame.empty:
                update_session_state(
                    state["pe_state"],
                    open_price=float(day_frame.iloc[0].open),
                    low=float(day_frame.low.min()),
                )
            pe_signal = evaluate_pe_recovery(
                state["pe_state"],
                prev_close=float(prev_row.close),
                close=float(row.close),
            )
            if pe_signal is not None and state["pe_position"] is None and pe_resolver is not None:
                try:
                    order = build_pe_recovery_order(pe_signal, pe_resolver, client, as_of=day)
                    state["pe_position"] = {
                        "session": day,
                        "entry_ts": ts,
                        "instrument_key": order.contract.instrument_key,
                        "trading_symbol": order.contract.trading_symbol,
                        "strike": order.strike,
                        "expiry": order.expiry.isoformat(),
                        "entry_premium": order.entry_premium,
                        "stop_premium": order.stop_premium,
                        "quantity": order.quantity,
                        "lots": order.lots,
                    }
                    log_event(
                        "PE_ENTRY", ts, side="PE", entry=order.entry_premium,
                        stop=order.stop_premium, target="",
                        note=format_pe_order_alert(order),
                    )
                except OptionResolutionError as exc:
                    log_event(
                        "PE_SIGNAL", ts, side="PE", entry=round(pe_signal.spot_ref, 2),
                        stop="", target="",
                        note=f"{format_alert(pe_signal)} | auto-resolve failed: {exc.code}",
                    )

        if state["position"] is not None or i == 0 or pivot_exited_this_bar:
            return
        if prev_row is None:
            return
        if row.ts.date() != day or prev_row.ts.date() != day:
            return
        signal_minute = ts.hour * 60 + ts.minute
        if signal_minute < 9 * 60 + 20 or signal_minute >= SIGNAL_SESSION_END_MINUTE:
            return

        cross_up = (row.close > pp and prev_row.close <= pp) or (row.close > r1 and prev_row.close <= r1)
        cross_down = (row.close < pp and prev_row.close >= pp) or (row.close < s1 and prev_row.close >= s1)
        if guide_mode:
            long_ok = cross_up and row.ema20 > row.ema50 and row.rsi14 > 55
            short_ok = cross_down and row.ema20 < row.ema50 and row.rsi14 < 45
        else:
            long_vwap = not symbol_cfg.require_vwap_above_for_long or (
                pd.notna(row.session_typical_avg) and row.close > row.session_typical_avg
            )
            short_vwap = not symbol_cfg.require_vwap_below_for_short or (
                pd.notna(row.session_typical_avg) and row.close < row.session_typical_avg
            )
            long_ok = (
                side_quota_available(
                    symbol_cfg, "LONG",
                    long_taken=state["long_taken_today"],
                    short_taken=state["short_taken_today"],
                    trade_taken=state["trade_taken_today"],
                )
                and spot_paper_entry_window_ok(symbol, ts, "LONG")
                and cross_up and row.ema20 > row.ema50 and row.rsi14 > 55 and long_vwap
            )
            short_ok = (
                side_quota_available(
                    symbol_cfg, "SHORT",
                    long_taken=state["long_taken_today"],
                    short_taken=state["short_taken_today"],
                    trade_taken=state["trade_taken_today"],
                )
                and spot_paper_entry_window_ok(symbol, ts, "SHORT")
                and cross_down and row.ema20 < row.ema50 and row.rsi14 < 45 and short_vwap
            )
        if not (long_ok or short_ok) or pd.isna(row.atr14):
            return

        side = "LONG" if long_ok else "SHORT"
        if guide_mode and guide_profile is not None and guide_profile.short_only and side == "LONG":
            return
        if guide_mode:
            entry = float(row.close)
            entry_ts = ts
            stop_mult = atr_buffer
            target_rr = rr
        else:
            entry = float(next_open_price)
            entry_ts = next_start
            stop_mult = symbol_cfg.atr_buffer
            target_rr = symbol_cfg.rr
        if side == "LONG":
            stop = float(row.low - row.atr14 * stop_mult)
            risk = entry - stop
            target = entry + target_rr * risk
        else:
            stop = float(row.high + row.atr14 * stop_mult)
            risk = stop - entry
            target = entry - target_rr * risk
        if risk <= 0 or (max_risk is not None and risk > max_risk):
            return
        if guide_mode and guide_profile is not None and not _entry_allowed(
            guide_profile,
            symbol_key=symbol,
            side=side,
            signal_minute=signal_minute,
            risk=float(risk),
            trade_taken=state["trade_taken_today"],
        ):
            return

        state["position"] = {
            "side": side, "entry": entry, "entry_ts": entry_ts, "session": day,
            "stop": stop, "target": target, "initial_risk": risk, "breakeven_armed": False,
        }
        if not guide_mode:
            state["long_taken_today"], state["short_taken_today"], state["trade_taken_today"] = mark_side_taken(
                symbol_cfg, side,
                long_taken=state["long_taken_today"],
                short_taken=state["short_taken_today"],
                trade_taken=state["trade_taken_today"],
            )
        elif guide_profile is not None and guide_profile.one_trade_per_day:
            state["trade_taken_today"] = True
        filter_tag = f" filter={args.guide_filter}" if guide_profile else ""
        entry_note = (
            f"{pivot_label} {side} | close={row.close:.2f}; PP={pp_base:.2f}; "
            f"RSI={row.rsi14:.1f}; risk={risk:.1f}; RR={rr}; ATR={atr_buffer}{filter_tag}; "
            f"signal-bar close entry; spot volume OFF"
            if guide_mode else
            f"FROZEN {CONFIG_VERSION} {side} | close={row.close:.2f}; PP={pp_base:.2f}; "
            f"RSI={row.rsi14:.1f}; risk={risk:.1f}/{max_risk:.0f}; next-bar first tick; spot volume OFF"
        )
        log_event(
            "PAPER_ENTRY", entry_ts, side=side, entry=round(entry, 2), stop=round(stop, 2),
            target=round(target, 2),
            note=entry_note,
        )

    def on_tick(tick: Tick) -> None:
        if tick.symbol != key or tick.ltp <= 0:
            return
        ts = tick.timestamp.astimezone(IST)
        if ts.date() != today or not (daytime(9, 15) <= ts.time() < daytime(15, 30)):
            return
        start = bucket(ts)
        with lock:
            state["live_ticks"] += 1
            state["first_live_tick"] = state["first_live_tick"] or ts.isoformat()
            state["last_live_tick"] = ts.isoformat()
            current = state["open_bar"]
            if current is None:
                state["open_bar"] = {"timestamp": start, "open": tick.ltp, "high": tick.ltp,
                                     "low": tick.ltp, "close": tick.ltp, "volume": 0.0}
            elif start < current["timestamp"]:
                return
            elif start == current["timestamp"]:
                current["high"] = max(current["high"], tick.ltp)
                current["low"] = min(current["low"], tick.ltp)
                current["close"] = tick.ltp
            else:
                elapsed = start - current["timestamp"]
                if elapsed == timedelta(minutes=5):
                    complete_bar(current["timestamp"], current, tick.ltp, start)
                else:
                    state["data_gaps"] += 1
                    log_event("DATA_GAP", current["timestamp"], note=f"Skipped candle evaluation after {elapsed} gap")
                state["open_bar"] = {"timestamp": start, "open": tick.ltp, "high": tick.ltp,
                                     "low": tick.ltp, "close": tick.ltp, "volume": 0.0}

    feed = create_upstox_v3_feed(stale_after_seconds=15)
    feed.set_tick_handler(on_tick)
    feed.subscribe([key])
    pe_extra = " | NIFTY PE recovery ON" if pe_resolver is not None else ""
    if guide_mode:
        quota = f"doc guide RR={rr} ATR={atr_buffer} filter={args.guide_filter}"
    else:
        quota = "1 short/day" if not symbol_cfg.allow_long else "1 long + 1 short/day"
    print(f"PAPER ONLY | {pivot_label} | {symbol} spot | 5m | {quota}{pe_extra}")
    print(f"Historical warm-up: {len(data)} candles | output: {out_file}")
    stopped = threading.Event()

    def shutdown(*_):
        stopped.set()
        feed.disconnect()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    feed.connect()
    print(
        f"Listening on Upstox WebSocket until 15:35 IST (Ctrl+C to stop). "
        f"Logs appear only on signals/trades — not every tick.",
        flush=True,
    )
    started = time.monotonic()
    last_status = started
    no_tick_warned = False
    try:
        while not stopped.is_set():
            time.sleep(1)
            now_mono = time.monotonic()
            if now_mono - last_status >= 60:
                health = feed.health()
                print(
                    f"[STATUS] {datetime.now(IST):%H:%M:%S} | ticks={state['live_ticks']} "
                    f"bars={state['live_bars']} | feed={health.status} | "
                    f"last_tick={state['last_live_tick'] or 'none'} | gaps={state['data_gaps']}",
                    flush=True,
                )
                last_status = now_mono
            if not no_tick_warned and now_mono - started >= 30 and state["live_ticks"] == 0:
                print(
                    "[WARN] No live ticks yet. Check UPSTOX_MARKET_DATA_ACCESS_TOKEN (or ACCESS_TOKEN) "
                    "and that market is open.",
                    flush=True,
                )
                no_tick_warned = True
            if datetime.now(IST).time() >= daytime(15, 35):
                break
    finally:
        feed.disconnect()
    if out_file.exists():
        events = pd.read_csv(out_file)
        exits = events[events.event.isin(["PAPER_EXIT", "PE_EXIT"])]
        pivot_net = pd.to_numeric(events.loc[events.event == "PAPER_EXIT", "net_points"], errors="coerce").dropna()
        pe_net = pd.to_numeric(events.loc[events.event == "PE_EXIT", "net_points"], errors="coerce").dropna()
        combined = pd.concat([pivot_net, pe_net])
        summary_row = {"date": today.isoformat(), "symbol": symbol, "closed_trades": len(exits),
                       "wins": int((combined > 0).sum()) if len(combined) else 0,
                       "losses": int((combined <= 0).sum()) if len(combined) else 0,
                       "net_points": round(float(pivot_net.sum()), 2),
                       "pe_net_inr_estimate": round(float(pe_net.sum()), 2),
                       "config_version": pivot_label,
                       "coverage_status": "ADEQUATE" if state["live_bars"] >= 60 and state["data_gaps"] == 0 else "PARTIAL",
                       "live_ticks": state["live_ticks"], "completed_live_bars": state["live_bars"],
                       "data_gaps": state["data_gaps"],
                       "open_position_at_shutdown": bool(state["position"] or state["pe_position"]),
                       "orders_submitted": 0, "estimated_costs_only": True}
        if summary_file.exists():
            previous = pd.read_csv(summary_file)
            previous = previous[~((previous.date == today.isoformat()) & (previous.symbol == symbol))]
            previous = pd.concat([previous, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            previous = pd.DataFrame([summary_row])
        previous.to_csv(summary_file, index=False)
        pe_part = f" pe_inr={summary_row['pe_net_inr_estimate']}" if symbol == "NIFTY" else ""
        print(f"DAILY PAPER SUMMARY | {symbol} closed={len(exits)} wins={summary_row['wins']} "
              f"losses={summary_row['losses']} net={summary_row['net_points']} pts{pe_part} orders=0")
    print("Stopped. No broker orders were submitted.")


if __name__ == "__main__":
    main()
