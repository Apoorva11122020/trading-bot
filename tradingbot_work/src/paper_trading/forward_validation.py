"""S2 forward paper validation (observation / analytics only).

Records every S2 signal of the production setup (NIFTY, 5 minutes, nearest
expiry, ATM, BUY -> CE, SELL -> PE) with its point-in-time context and
resolves it from real option premium candles using the live tracker's
outcome rules. Nothing here places, modifies or cancels orders: no broker,
engine or order-manager code is imported.

* All signals are kept, including repeated and losing ones.
* Features use only candles completed at the signal bar (no look-ahead);
  premium ATR excludes the entry candle.
* A signal whose real option candles are missing is stored as
  DATA_UNAVAILABLE; no outcome is ever created without candle evidence.
* Records are immutable and keyed by strategy/underlying/time/direction, so
  re-running a session cannot insert a signal twice.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.current_day_s2 import LevelParams, option_levels
from src.backtest.losing_trade_analysis import enrich_underlying
from src.backtest.research_strategies import ResearchSignal
from src.options.models import OptionResolutionError
from src.paper_trading.signal_tracker import (
    AMBIGUOUS,
    INVALID,
    LOSS,
    PENDING,
    WIN,
    _aware,
    _bar,
    _utc_iso,
    evaluate_candle,
    signal_id_for,
    verify_observation_only,
)

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
STATUSES = (PENDING, WIN, LOSS, AMBIGUOUS, INVALID, DATA_UNAVAILABLE)
METHOD = "S2 unchanged; next-bar option open entry; premium-ATR stop from candles completed at the signal bar; target R"
HISTORICAL_COMPARABLE_STATUS = "Historical comparable 1-DTE data: UNAVAILABLE"
TARGET_DTE = 1
ACCURACY_MIN_RESOLVED = 100
SAMPLE_LEVELS = ((200, "MEANINGFUL SAMPLE"), (100, "PRELIMINARY"), (30, "EARLY SAMPLE"), (0, "INSUFFICIENT SAMPLE"))
TIME_BUCKETS = ((time(9, 15), time(10, 0), "09:15-10:00"), (time(10, 0), time(11, 0), "10:00-11:00"),
                (time(11, 0), time(12, 0), "11:00-12:00"), (time(12, 0), time(13, 30), "12:00-13:30"),
                (time(13, 30), time(15, 30), "13:30-15:30"))
# Reasons that mean real option data was missing (as opposed to a rule rejecting the levels).
_DATA_REASONS = ("CONTRACT_UNRESOLVED", "NO_PREMIUM", "PREMIUM_API_ERROR", "INSUFFICIENT_ATR_HISTORY",
                 "NO_EXIT_EVIDENCE")


def sample_status(resolved: int) -> str:
    for minimum, label in SAMPLE_LEVELS:
        if resolved >= minimum:
            return label
    return SAMPLE_LEVELS[-1][1]


def accuracy_label(resolved: int) -> str:
    if resolved >= ACCURACY_MIN_RESOLVED:
        return "Strategy accuracy"
    return f"Observed win rate (not strategy accuracy: {resolved} < {ACCURACY_MIN_RESOLVED} resolved signals)"


def status_for_reason(reason: str | None) -> str:
    if reason is None:
        return PENDING
    return DATA_UNAVAILABLE if reason.startswith(_DATA_REASONS) else INVALID


def time_bucket(ts: datetime) -> str:
    t = ts.astimezone(IST).time()
    return next((label for start, end, label in TIME_BUCKETS if start <= t < end), "other")


def calendar_dte(expiry: str | date | None, day: date) -> int | None:
    if expiry is None:
        return None
    exp = expiry if isinstance(expiry, date) else date.fromisoformat(str(expiry)[:10])
    return (exp - day).days


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else round(out, 4)


@dataclass(frozen=True)
class ForwardSignal:
    """Immutable point-in-time snapshot of one S2 signal and its predefined option levels."""

    signal_id: str
    timestamp: str
    session_date: str
    strategy: str
    underlying: str
    direction: str
    underlying_price: float | None
    breakout_level: float | None
    instrument_key: str | None
    trading_symbol: str | None
    option_type: str | None
    strike: float | None
    expiry: str | None
    dte: int | None
    quantity: int | None
    entry_time: str | None
    entry_price: float | None
    premium_atr: float | None
    premium_atr_pct: float | None
    stop_loss: float | None
    target: float | None
    rsi: float | None
    ema20: float | None
    ema50: float | None
    ema_alignment: str | None
    breakout_distance: float | None
    breakout_distance_atr: float | None
    option_vwap: float | None
    option_vs_vwap: str | None
    option_vs_vwap_pct: float | None
    market_regime: str
    signal_number_of_day: int
    same_direction_number: int
    premium_source: str | None
    data_reason: str | None
    method: str = METHOD

    @property
    def initial_status(self) -> str:
        reason = self.data_reason
        if reason is None and not (self.stop_loss and self.entry_price and self.target
                                   and 0 < self.stop_loss < self.entry_price < self.target):
            reason = "INVALID_LEVELS"
        return status_for_reason(reason)

    @property
    def initial_reason(self) -> str | None:
        if self.data_reason:
            return self.data_reason
        return "INVALID_LEVELS" if self.initial_status == INVALID else None

    def to_row(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "timestamp": _utc_iso(self.timestamp),
            "session_date": self.session_date,
            "strategy": self.strategy,
            "underlying": self.underlying,
            "direction": self.direction,
            "instrument_key": self.instrument_key,
            "expiry": self.expiry,
            "dte": self.dte,
            "record_json": json.dumps(asdict(self), sort_keys=True, default=str),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": self.initial_status,
            "exit_reason": self.initial_reason,
            "resolved_at": None if self.initial_status == PENDING else datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_json(cls, raw: str) -> "ForwardSignal":
        data = json.loads(raw)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ------------------------------------------------------------------ features
def option_vwap(premium: pd.DataFrame, signal_ts: pd.Timestamp) -> float | None:
    """Session VWAP of real premium candles up to and including the signal bar; None without volume."""
    if premium is None or premium.empty or "volume" not in premium:
        return None
    day = premium[(premium["timestamp"].dt.date == signal_ts.date()) & (premium["timestamp"] <= signal_ts)]
    volume = day["volume"].astype(float)
    if day.empty or volume.sum() <= 0:
        return None
    typical = (day["high"] + day["low"] + day["close"]).astype(float) / 3
    return float((typical * volume).sum() / volume.sum())


def _ema_alignment(direction: str, ema20: float | None, ema50: float | None) -> str | None:
    if ema20 is None or ema50 is None:
        return None
    if ema20 == ema50:
        return "FLAT"
    bullish = ema20 > ema50
    trend = "BULLISH" if bullish else "BEARISH"
    aligned = bullish == (direction == "BUY")
    return f"{trend} ({'ALIGNED' if aligned else 'OPPOSED'})"


def build_signals(
    signals: Iterable[ResearchSignal],
    underlying_candles: pd.DataFrame,
    *,
    resolve: Callable[[ResearchSignal], Any],
    premium_for: Callable[[Any], pd.DataFrame | str],
    params: LevelParams,
    s2_cfg: Mapping[str, Any],
    regime_at: Callable[[datetime], Iterable[str]],
    underlying: str = "NIFTY",
    strategy: str = "S2",
    premium_source: str = "UPSTOX_V3_INTRADAY",
) -> list[ForwardSignal]:
    """Snapshot every signal; features come only from bars at or before the signal bar."""
    frame = enrich_underlying(underlying_candles, dict(s2_cfg))
    by_ts = frame.set_index("timestamp")
    counters: dict[tuple, int] = {}
    out: list[ForwardSignal] = []
    for sig in sorted(signals, key=lambda s: pd.Timestamp(s.timestamp)):
        ts = pd.Timestamp(sig.timestamp).tz_convert(IST)
        day = ts.date()
        counters[(day,)] = counters.get((day,), 0) + 1
        counters[(day, sig.direction)] = counters.get((day, sig.direction), 0) + 1
        bar = by_ts.loc[ts] if ts in by_ts.index else None
        get = (lambda k: _num(bar[k])) if bar is not None else (lambda k: None)
        spot = _num(sig.spot)
        level = _num(sig.conditions.get("above", sig.conditions.get("below")))
        sign = 1.0 if sig.direction == "BUY" else -1.0
        distance = round(sign * (spot - level), 4) if spot is not None and level is not None else None
        atr14, ema20, ema50 = get("atr14"), get("ema20"), get("ema50")
        tags = sorted(regime_at(ts.to_pydatetime()))
        row: dict[str, Any] = dict(
            signal_id=signal_id_for(strategy, underlying, ts, sig.direction),
            timestamp=ts.isoformat(), session_date=day.isoformat(), strategy=strategy,
            underlying=underlying.upper(), direction=sig.direction, underlying_price=spot, breakout_level=level,
            instrument_key=None, trading_symbol=None, option_type=None, strike=None, expiry=None, dte=None,
            quantity=None, entry_time=None, entry_price=None, premium_atr=None, premium_atr_pct=None,
            stop_loss=None, target=None, rsi=get("rsi14"), ema20=ema20, ema50=ema50,
            ema_alignment=_ema_alignment(sig.direction, ema20, ema50), breakout_distance=distance,
            breakout_distance_atr=round(distance / atr14, 4) if distance is not None and atr14 else None,
            option_vwap=None, option_vs_vwap=None, option_vs_vwap_pct=None,
            market_regime="+".join(tags) if tags else "none",
            signal_number_of_day=counters[(day,)], same_direction_number=counters[(day, sig.direction)],
            premium_source=None, data_reason=None,
        )
        try:
            contract = resolve(sig)
        except OptionResolutionError as exc:
            row["data_reason"] = f"CONTRACT_UNRESOLVED:{exc.code}"
            out.append(ForwardSignal(**row))
            continue
        option_type = getattr(contract, "option_type", None)
        row.update(instrument_key=contract.instrument_key, trading_symbol=contract.trading_symbol,
                   option_type=getattr(option_type, "value", option_type), strike=float(contract.strike),
                   expiry=contract.expiry.isoformat(), dte=calendar_dte(contract.expiry, day),
                   quantity=int(contract.lot_size))
        premium = premium_for(contract)
        if isinstance(premium, str):
            row["data_reason"] = premium
        elif premium is None or premium.empty:
            row["data_reason"] = "NO_PREMIUM_DATA"
        else:
            row["premium_source"] = premium.attrs.get("source", premium_source)
            levels = option_levels(sig, contract, premium, params)
            if levels["status"] != "RESOLVED":
                row["data_reason"] = levels["reason"]
            else:
                entry, patr = levels["entry"], levels["premium_atr"]
                vwap = option_vwap(premium, ts)
                close = levels["premium_at_signal_close"]
                row.update(entry_time=levels["entry_time"], entry_price=entry, premium_atr=_num(patr),
                           premium_atr_pct=_num(patr / entry * 100) if entry else None,
                           stop_loss=levels["stop_loss"], target=levels["target"], option_vwap=_num(vwap))
                if vwap:
                    row["option_vs_vwap"] = "ABOVE" if close > vwap else "BELOW" if close < vwap else "AT"
                    row["option_vs_vwap_pct"] = _num((close - vwap) / vwap * 100)
        out.append(ForwardSignal(**row))
    return out


# ------------------------------------------------------------------ outcomes
def resolve_outcome(record: ForwardSignal, premium: pd.DataFrame | None, square_off: time,
                    session_complete: bool) -> dict[str, Any] | None:
    """First outcome evidenced by real candles from the entry candle on; None while still pending."""
    if premium is not None and not premium.empty and record.entry_time:
        entry_ts = pd.Timestamp(record.entry_time)
        day = date.fromisoformat(record.session_date)
        window = premium[(premium["timestamp"] >= entry_ts) & (premium["timestamp"].dt.date == day)]
        for candle in window.sort_values("timestamp").to_dict("records"):
            outcome = evaluate_candle(record, _bar(candle), square_off)
            if outcome is not None:
                return outcome
    if session_complete:
        return {"status": DATA_UNAVAILABLE, "exit_price": None, "exit_time": None, "pnl": None,
                "r_multiple": None, "resolved_at": datetime.now(timezone.utc).isoformat(),
                "reason": "NO_EXIT_EVIDENCE"}
    return None


class ForwardValidationTracker:
    """Persists forward signals once and resolves pending ones from real premium candles."""

    def __init__(self, database, *, square_off: str = "15:15"):
        self.safety = verify_observation_only()
        self.database = database
        self.square_off = time.fromisoformat(square_off)

    def record(self, signal: ForwardSignal) -> bool:
        """True when newly stored; duplicates are ignored and existing records are never changed."""
        return self.database.save_forward_signal(signal.to_row())

    def record_all(self, signals: Iterable[ForwardSignal]) -> int:
        return sum(1 for s in signals if self.record(s))

    def resolve_pending(self, session_date: str, premium_by_key: Mapping[str, pd.DataFrame],
                        session_complete: bool) -> list[dict[str, Any]]:
        resolved = []
        for row in self.database.get_forward_signals(session_date):
            if row["status"] != PENDING:
                continue
            record = ForwardSignal.from_json(row["record_json"])
            outcome = resolve_outcome(record, premium_by_key.get(record.instrument_key), self.square_off,
                                      session_complete)
            if outcome and self.database.resolve_forward_signal(record.signal_id, outcome):
                resolved.append({"signal_id": record.signal_id, **outcome})
        return resolved


# ------------------------------------------------------------------ statistics
def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Signal snapshot plus outcome for API/reports (no internal JSON, IST times)."""
    data = json.loads(row["record_json"])
    data.pop("method", None)
    exit_time = row.get("exit_time")
    data.update(
        status=row["status"],
        exit_time=_aware(exit_time).astimezone(IST).isoformat() if exit_time else None,
        exit_price=row.get("exit_price"),
        exit_reason=row.get("exit_reason"),
        outcome=row["status"],
        r_multiple=row.get("r_multiple"),
        pnl_1_lot=row.get("pnl"),
        resolved_at=row.get("resolved_at"),
        time_of_day=time_bucket(_aware(data["timestamp"])),
        target_setup_match=data.get("dte") is not None and data["dte"] <= TARGET_DTE,
    )
    return data


def _decided(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted((r for r in rows if r["status"] in (WIN, LOSS)),
                  key=lambda r: (r.get("exit_time") or "", r.get("timestamp") or ""))


def metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    decided = _decided(rows)
    resolved = len(decided)
    pnls = [float(r["pnl"]) for r in decided if r.get("pnl") is not None]
    r_values = [float(r["r_multiple"]) for r in decided if r.get("r_multiple") is not None]
    gross_profit = round(sum(p for p in pnls if p > 0), 2)
    gross_loss = round(-sum(p for p in pnls if p < 0), 2)
    equity = peak = drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    win_rate = round(counts[WIN] / resolved * 100, 4) if resolved else None
    return {
        "signals": len(rows),
        "resolved": resolved,
        "wins": counts[WIN],
        "losses": counts[LOSS],
        "pending": counts[PENDING],
        "ambiguous": counts[AMBIGUOUS],
        "invalid": counts[INVALID],
        "data_unavailable": counts[DATA_UNAVAILABLE],
        "accuracy": win_rate,
        "accuracy_display": f"{win_rate:.2f}%" if win_rate is not None else "N/A",
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": round(gross_profit - gross_loss, 2),
        "max_drawdown": round(drawdown, 2),
        "average_r": round(sum(r_values) / len(r_values), 4) if r_values else None,
    }


def _grouped(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(str(key(row)), []).append(row)
    return {k: {**metrics(v), "sample_status": sample_status(metrics(v)["resolved"])} for k, v in sorted(groups.items())}


def _merged(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """DB rows with their snapshot fields expanded, for grouping."""
    out = []
    for row in rows:
        snap = json.loads(row["record_json"])
        out.append({**snap, **{k: row[k] for k in ("status", "pnl", "r_multiple", "exit_time", "timestamp")},
                    "time_of_day": time_bucket(_aware(snap["timestamp"]))})
    return out


def daily_stats(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged = _merged(rows)
    days: dict[str, list] = {}
    for row in merged:
        days.setdefault(row["session_date"], []).append(row)
    cumulative, out = 0.0, []
    for day in sorted(days):
        m = metrics(days[day])
        cumulative = round(cumulative + m["net_pnl"], 2)
        out.append({"session_date": day, **m, "cumulative_pnl": cumulative})
    return out


def forward_stats(rows: Iterable[Mapping[str, Any]], today: str | None = None) -> dict[str, Any]:
    rows = list(rows)
    merged = _merged(rows)
    overall = metrics(merged)
    resolved = overall["resolved"]
    daily = daily_stats(rows)
    today_rows = [r for r in merged if r["session_date"] == today] if today else []
    return {
        **overall,
        "sample_status": sample_status(resolved),
        "accuracy_label": accuracy_label(resolved),
        "accuracy_is_strategy_accuracy": resolved >= ACCURACY_MIN_RESOLVED,
        "days_tracked": len(daily),
        "first_session": daily[0]["session_date"] if daily else None,
        "last_session": daily[-1]["session_date"] if daily else None,
        "today_signals": len(today_rows),
        "today": {"date": today, **metrics(today_rows)} if today else None,
        "cumulative_pnl": daily[-1]["cumulative_pnl"] if daily else 0.0,
        "by_direction": _grouped(merged, lambda r: r["direction"]),
        "by_dte": _grouped(merged, lambda r: "unknown" if r.get("dte") is None else f"{r['dte']} DTE"),
        "by_time_of_day": _grouped(merged, lambda r: r["time_of_day"]),
        "by_regime": _grouped(merged, lambda r: r.get("market_regime") or "none"),
        "historical_comparable_1dte": HISTORICAL_COMPARABLE_STATUS,
        "method": METHOD,
    }


# ------------------------------------------------------------------ reports
CSV_COLUMNS = (
    "signal_id", "timestamp", "session_date", "direction", "underlying_price", "breakout_level", "trading_symbol",
    "instrument_key", "option_type", "strike", "expiry", "dte", "quantity", "entry_time", "entry_price",
    "premium_atr", "premium_atr_pct", "stop_loss", "target", "exit_time", "exit_price", "exit_reason", "outcome",
    "r_multiple", "pnl_1_lot", "rsi", "ema20", "ema50", "ema_alignment", "breakout_distance",
    "breakout_distance_atr", "option_vwap", "option_vs_vwap", "option_vs_vwap_pct", "market_regime",
    "signal_number_of_day", "same_direction_number", "time_of_day", "target_setup_match", "premium_source",
    "data_reason",
)
DAILY_COLUMNS = ("session_date", "signals", "resolved", "wins", "losses", "pending", "ambiguous", "invalid",
                 "data_unavailable", "accuracy", "profit_factor", "net_pnl", "cumulative_pnl", "max_drawdown")


def write_reports(rows: list[Mapping[str, Any]], out_dir: Path, today: str | None = None,
                  notes: list[str] | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = forward_stats(rows, today)
    table = [public_row(r) for r in rows]
    daily = daily_stats(rows)
    paths = {
        "json": out_dir / "forward_validation_summary.json",
        "md": out_dir / "forward_validation_summary.md",
        "signals_csv": out_dir / "forward_validation_signals.csv",
        "daily_csv": out_dir / "forward_validation_daily.csv",
    }
    payload = {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "mode": "PAPER observation only - no orders placed",
        "safety": verify_observation_only(),
        "stats": stats,
        "daily": daily,
        "notes": notes or [],
        "signals": table,
    }
    paths["json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    with paths["signals_csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS} for r in table)
    with paths["daily_csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DAILY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: ("" if d.get(k) is None else d.get(k)) for k in DAILY_COLUMNS} for d in daily)
    paths["md"].write_text(render_markdown(stats, daily, notes or []), encoding="utf-8")
    return paths


def _fmt(value: Any, money: bool = False) -> str:
    if value is None:
        return "N/A"
    if money:
        return f"Rs {value:,.2f}"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def _group_table(title: str, groups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [f"### {title}", "", "| Group | Signals | Resolved | W | L | Win rate | PF | Net P&L | Sample |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, g in groups.items():
        lines.append(f"| {name} | {g['signals']} | {g['resolved']} | {g['wins']} | {g['losses']} | "
                     f"{g['accuracy_display']} | {_fmt(g['profit_factor'])} | {_fmt(g['net_pnl'], True)} | "
                     f"{g['sample_status']} |")
    return lines + [""]


def render_markdown(stats: Mapping[str, Any], daily: list[Mapping[str, Any]], notes: list[str]) -> str:
    lines = [
        "# S2 Forward Paper Validation", "",
        "> PAPER observation only. No orders are placed. S2 rules, stop and target are unchanged.", "",
        f"**Sample status: {stats['sample_status']}** ({stats['resolved']} resolved signals). "
        f"{stats['accuracy_label']}.", "",
        f"{HISTORICAL_COMPARABLE_STATUS}.", "",
        "| Metric | Value |", "|---|---|",
        f"| Days tracked | {stats['days_tracked']} ({stats['first_session']} .. {stats['last_session']}) |",
        f"| Signals | {stats['signals']} |",
        f"| Resolved (W+L) | {stats['resolved']} |",
        f"| Wins / Losses | {stats['wins']} / {stats['losses']} |",
        f"| Pending / Ambiguous / Invalid / Data unavailable | {stats['pending']} / {stats['ambiguous']} / "
        f"{stats['invalid']} / {stats['data_unavailable']} |",
        f"| Win rate | {stats['accuracy_display']} |",
        f"| Profit factor | {_fmt(stats['profit_factor'])} |",
        f"| Net P&L (1 lot, gross) | {_fmt(stats['net_pnl'], True)} |",
        f"| Max drawdown | {_fmt(stats['max_drawdown'], True)} |",
        f"| Average R | {_fmt(stats['average_r'])} |", "",
        "## Daily", "",
        "| Date | Signals | Resolved | W | L | Data unavailable | Win rate | PF | Net P&L | Cumulative |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in daily:
        lines.append(f"| {d['session_date']} | {d['signals']} | {d['resolved']} | {d['wins']} | {d['losses']} | "
                     f"{d['data_unavailable']} | {d['accuracy_display']} | {_fmt(d['profit_factor'])} | "
                     f"{_fmt(d['net_pnl'], True)} | {_fmt(d['cumulative_pnl'], True)} |")
    lines += ["", "## Breakdowns", "", "Group results below the sample thresholds are descriptive only.", ""]
    lines += _group_table("Direction", stats["by_direction"])
    lines += _group_table("Days to expiry", stats["by_dte"])
    lines += _group_table("Time of day", stats["by_time_of_day"])
    lines += _group_table("Market regime", stats["by_regime"])
    lines += ["## Method and notes", "", f"- {stats['method']}."] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def premium_window(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise a premium candle frame to IST timestamps sorted ascending."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(IST)
    return out.sort_values("timestamp").reset_index(drop=True)


__all__ = [
    "ACCURACY_MIN_RESOLVED", "DATA_UNAVAILABLE", "ForwardSignal", "ForwardValidationTracker",
    "HISTORICAL_COMPARABLE_STATUS", "accuracy_label", "build_signals", "daily_stats", "forward_stats",
    "metrics", "public_row", "resolve_outcome", "sample_status", "status_for_reason", "write_reports",
]
