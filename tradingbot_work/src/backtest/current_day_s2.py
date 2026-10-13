"""Current-day NIFTY S2 signal research from real Upstox intraday candles (read-only, PAPER only).

Signals come from the unchanged S2 generator, evaluated bar by bar on candles
up to and including each bar. Option entry/SL/target mirror
OptionsResearchEngine and are computed only when exact real premium candles
exist. No trade outcomes, win rate or P&L are computed and nothing is stored.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.research_strategies import ResearchSignal, generate_signals
from src.historical_data.models import HistoricalCandle, candles_to_frame
from src.historical_data.option_coverage import entry_allowed
from src.historical_data.validation import expected_slots, validate_candles
from src.indicators.technical import atr
from src.options.models import OptionResolutionError
from src.options.strategy import OptionSignalError, option_stop_loss, option_target, round_to_tick

IST = ZoneInfo("Asia/Kolkata")
IST_OFFSET = timedelta(hours=5, minutes=30)
DISCLAIMER = ("Signal-level research only. This is not a completed trade-performance backtest: no outcomes, "
              "win rate or P&L are calculated from today's signals.")


@dataclass(frozen=True)
class LevelParams:
    entry_timing: str
    stop_new_entries: str
    atr_period: int
    stop_atr_multiplier: float
    target_r: float
    slippage_bps: float
    interval: int = 5

    @classmethod
    def from_configs(cls, research_config, strategy_config) -> "LevelParams":
        return cls(
            entry_timing=str(research_config.option_mapping.get("entry_timing", "next_open")),
            stop_new_entries=research_config.session.stop_new_entries,
            atr_period=int(strategy_config.indicators.atr_period),
            stop_atr_multiplier=float(strategy_config.options.risk.stop_atr_multiplier),
            target_r=float(strategy_config.options.risk.target_r),
            slippage_bps=float(research_config.costs.slippage_bps),
            interval=int(research_config.interval_minutes),
        )


@dataclass
class CandleValidation:
    requested_day: str
    received: int = 0
    received_order: str = "empty"
    non_ist_timestamps: int = 0
    wrong_day_rows: int = 0
    duplicates: int = 0
    rejected_rows: int = 0
    issues: dict[str, int] = field(default_factory=dict)
    flat_candles: list[str] = field(default_factory=list)
    expected_candles: int = 0
    valid_candles: int = 0
    missing_candles: list[str] = field(default_factory=list)
    first: str | None = None
    last: str | None = None


def _order(timestamps: list) -> str:
    if len(timestamps) < 2:
        return "single" if timestamps else "empty"
    if all(a < b for a, b in zip(timestamps, timestamps[1:])):
        return "ascending"
    if all(a > b for a, b in zip(timestamps, timestamps[1:])):
        return "descending"
    return "unordered"


def validate_intraday(candles: list[HistoricalCandle], day: date,
                      interval: int = 5) -> tuple[pd.DataFrame, CandleValidation]:
    """Sort, de-duplicate and validate; flat candles are kept exactly as received."""
    report = CandleValidation(requested_day=day.isoformat(), received=len(candles))
    raw_ts = [c.timestamp for c in candles]
    report.received_order = _order(raw_ts)
    report.non_ist_timestamps = sum(1 for ts in raw_ts if ts.utcoffset() != IST_OFFSET)
    frame = candles_to_frame(candles)
    if frame.empty:
        report.expected_candles = len(expected_slots(day, interval))
        report.missing_candles = [s.isoformat() for s in expected_slots(day, interval)]
        return frame, report
    on_day = frame["timestamp"].dt.date == day
    report.wrong_day_rows = int((~on_day).sum())
    frame = frame[on_day]
    clean, vr = validate_candles(frame, interval=interval)
    report.duplicates = vr.duplicates_removed
    report.rejected_rows = vr.rejected_rows
    report.issues = {k: v for k, v in vr.issues.items() if k != "UNSORTED_TIMESTAMPS"}
    if vr.outside_session_rows:
        report.issues["OUTSIDE_SESSION"] = vr.outside_session_rows
    flat = (clean["open"] == clean["high"]) & (clean["high"] == clean["low"]) & (clean["low"] == clean["close"])
    report.flat_candles = [ts.isoformat() for ts in clean.loc[flat, "timestamp"]]
    slots = expected_slots(day, interval)
    present = set(clean["timestamp"])
    report.expected_candles = len(slots)
    report.valid_candles = len(clean)
    report.missing_candles = [s.isoformat() for s in slots if s not in present]
    if not clean.empty:
        report.first = clean["timestamp"].iloc[0].isoformat()
        report.last = clean["timestamp"].iloc[-1].isoformat()
    return clean.reset_index(drop=True), report


def point_in_time_signals(candles: pd.DataFrame, underlying: str, research_config, session,
                          stop_new_entries: str, strategy_id: str = "S2") -> tuple[list[ResearchSignal], bool]:
    """Signals produced at each bar using only bars up to that bar; returns (signals, matches_full_run)."""
    frame = candles.sort_values("timestamp").reset_index(drop=True).copy()
    frame["symbol"] = underlying.upper()
    signals: list[ResearchSignal] = []
    for i in range(len(frame)):
        bar_ts = frame["timestamp"].iloc[i]
        for sig in generate_signals(strategy_id, frame.iloc[: i + 1], underlying.upper(), research_config):
            if pd.Timestamp(sig.timestamp) == bar_ts and entry_allowed(sig.timestamp, session, stop_new_entries):
                signals.append(sig)
    full = [s for s in generate_signals(strategy_id, frame, underlying.upper(), research_config)
            if entry_allowed(s.timestamp, session, stop_new_entries)]
    same = [(s.timestamp, s.direction) for s in full] == [(s.timestamp, s.direction) for s in signals]
    return signals, same


def premium_frame(stored: pd.DataFrame | None, intraday: list[HistoricalCandle]) -> pd.DataFrame:
    """Real stored history plus today's intraday candles (in memory only)."""
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    parts = []
    for part in (stored, candles_to_frame(intraday)):
        if part is not None and not part.empty:
            part = part[columns].copy()
            part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True).dt.tz_convert(IST)
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(parts, ignore_index=True)
    return frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def option_levels(signal: ResearchSignal, contract, premium: pd.DataFrame, params: LevelParams) -> dict[str, Any]:
    """Entry/SL/target as OptionsResearchEngine computes them; exact premium candles are required."""
    sig_ts = pd.Timestamp(signal.timestamp).tz_convert(IST)
    at_signal = premium.index[premium["timestamp"] == sig_ts]
    if len(at_signal) == 0:
        return {"status": "UNRESOLVED", "reason": "NO_PREMIUM_CANDLE_AT_SIGNAL"}
    signal_idx = int(at_signal[0])
    if params.entry_timing == "next_open":
        entry_ts = sig_ts + pd.Timedelta(minutes=params.interval)
        at_entry = premium.index[premium["timestamp"] == entry_ts]
        if len(at_entry) == 0:
            return {"status": "UNRESOLVED", "reason": "NO_PREMIUM_ENTRY_CANDLE"}
        entry_row = premium.loc[int(at_entry[0])]
        raw_entry = float(entry_row["open"])
    else:
        entry_row = premium.loc[signal_idx]
        raw_entry = float(entry_row["close"])
    tick = getattr(contract, "tick_size", None)
    entry = round_to_tick(raw_entry, tick, "nearest")
    history = premium.loc[:signal_idx]
    premium_atr = atr(history, params.atr_period).iloc[-1]
    if pd.isna(premium_atr):
        return {"status": "UNRESOLVED", "reason": "INSUFFICIENT_ATR_HISTORY", "entry_time": entry_row["timestamp"].isoformat(),
                "entry": entry, "premium_bars_available": len(history)}
    try:
        stop = option_stop_loss(entry, stop_mode="premium_atr", multiplier=params.stop_atr_multiplier,
                                premium_atr=float(premium_atr), tick_size=tick)
        target = option_target(entry, stop, params.target_r, tick)
    except OptionSignalError as exc:
        return {"status": "UNRESOLVED", "reason": exc.code}
    return {
        "status": "RESOLVED",
        "reason": None,
        "entry_time": entry_row["timestamp"].isoformat(),
        "entry": entry,
        "entry_with_slippage": entry + entry * params.slippage_bps / 10_000.0,
        "stop_loss": stop,
        "target": target,
        "premium_atr": float(premium_atr),
        "premium_at_signal_close": float(premium.loc[signal_idx, "close"]),
    }


def _contract_fields(contract) -> dict[str, Any]:
    option_type = getattr(contract, "option_type", None)
    return {
        "instrument_key": getattr(contract, "instrument_key", None),
        "trading_symbol": getattr(contract, "trading_symbol", None),
        "strike": float(contract.strike) if getattr(contract, "strike", None) is not None else None,
        "option_type": getattr(option_type, "value", option_type),
        "expiry": contract.expiry.isoformat() if getattr(contract, "expiry", None) else None,
    }


def resolve_signals(signals: Iterable[ResearchSignal], resolve: Callable[[ResearchSignal], Any],
                    premium_for: Callable[[Any], pd.DataFrame], params: LevelParams) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sig in signals:
        row: dict[str, Any] = {"signal_time": pd.Timestamp(sig.timestamp).isoformat(), "direction": sig.direction,
                               "spot": sig.spot, "breakout_level": sig.conditions.get("above", sig.conditions.get("below")),
                               "instrument_key": None, "trading_symbol": None, "strike": None, "option_type": None,
                               "expiry": None, "status": "UNRESOLVED", "reason": None, "entry_time": None,
                               "entry": None, "entry_with_slippage": None, "stop_loss": None, "target": None,
                               "premium_atr": None, "premium_at_signal_close": None}
        try:
            contract = resolve(sig)
        except OptionResolutionError as exc:
            row["reason"] = f"CONTRACT_UNRESOLVED:{exc.code}"
            rows.append(row)
            continue
        row.update(_contract_fields(contract))
        premium = premium_for(contract)
        if isinstance(premium, str):
            row["reason"] = premium
        elif premium is None or premium.empty:
            row["reason"] = "NO_PREMIUM_DATA"
        else:
            row.update(option_levels(sig, contract, premium, params))
        rows.append(row)
    return rows


def summarize(validation: CandleValidation, rows: list[dict[str, Any]], lookahead_ok: bool) -> dict[str, Any]:
    resolved = [r for r in rows if r["status"] == "RESOLVED"]
    reasons: dict[str, int] = {}
    for r in rows:
        if r["status"] != "RESOLVED":
            reasons[str(r["reason"])] = reasons.get(str(r["reason"]), 0) + 1
    return {
        "candle_count": validation.valid_candles,
        "first_candle": validation.first,
        "last_candle": validation.last,
        "signal_count": len(rows),
        "buy_signals": sum(1 for r in rows if r["direction"] == "BUY"),
        "sell_signals": sum(1 for r in rows if r["direction"] == "SELL"),
        "resolved_option_signals": len(resolved),
        "unresolved_signals": len(rows) - len(resolved),
        "option_coverage_percent": round(100.0 * len(resolved) / len(rows), 2) if rows else None,
        "unresolved_reasons": reasons,
        "unique_contracts": sorted({r["trading_symbol"] for r in rows if r["trading_symbol"]}),
        "point_in_time_matches_full_run": lookahead_ok,
    }


def write_reports(report: dict[str, Any], out_dir: Path, stem: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {ext: out_dir / f"{stem}.{ext}" for ext in ("json", "csv", "md")}
    paths["json"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    rows = report["signals"]
    columns = list(rows[0].keys()) if rows else ["signal_time", "direction", "status"]
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    s, v = report["summary"], report["validation"]
    md = [f"# NIFTY S2 current-day research {report['date']}", "",
          f"> {DISCLAIMER}", "",
          f"Source: Upstox V3 intraday candles (`{report['instrument_key']}`, 5 minutes), fetched "
          f"{report['fetched_at']}. Token source variable: `{report['token_source']}`. Mode: {report['mode']}.", "",
          "## Candles", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {val} |" for k, val in [
        ("Candles (valid / expected)", f"{v['valid_candles']} / {v['expected_candles']}"),
        ("First / last", f"{v['first']} / {v['last']}"),
        ("Order as received", v["received_order"]),
        ("Non-IST timestamps", v["non_ist_timestamps"]),
        ("Duplicates removed", v["duplicates"]),
        ("Rejected rows", v["rejected_rows"]),
        ("Missing slots", ", ".join(v["missing_candles"]) or "none"),
        ("Flat candles (kept unaltered)", ", ".join(t[11:16] for t in v["flat_candles"]) or "none"),
    ]]
    md += ["", "## Signals", "", "| Metric | Value |", "|---|---|"]
    md += [f"| {k} | {val} |" for k, val in [
        ("Signals", s["signal_count"]), ("BUY", s["buy_signals"]), ("SELL", s["sell_signals"]),
        ("Resolved option signals", s["resolved_option_signals"]), ("Unresolved", s["unresolved_signals"]),
        ("Option coverage", "n/a" if s["option_coverage_percent"] is None else f"{s['option_coverage_percent']}%"),
        ("Unresolved reasons", ", ".join(f"{k}={n}" for k, n in s["unresolved_reasons"].items()) or "none"),
        ("Point-in-time signals match full-day run", s["point_in_time_matches_full_run"]),
    ]]
    md += ["", "| Signal time | Dir | Spot | Contract | Status | Entry time | Entry | SL | Target | Reason |",
           "|---|---|---:|---|---|---|---:|---:|---:|---|"]
    for r in rows:
        fmt = lambda x: "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
        md.append(f"| {r['signal_time'][11:16]} | {r['direction']} | {fmt(r['spot'])} | {r['trading_symbol'] or '-'} | "
                  f"{r['status']} | {(r['entry_time'] or '-')[11:16] or '-'} | {fmt(r['entry'])} | {fmt(r['stop_loss'])} | "
                  f"{fmt(r['target'])} | {r['reason'] or ''} |")
    md += ["", "## Notes", ""] + [f"- {n}" for n in report["notes"]]
    paths["md"].write_text("\n".join(md) + "\n", encoding="utf-8")
    return paths


def validation_dict(validation: CandleValidation) -> dict[str, Any]:
    return asdict(validation)
