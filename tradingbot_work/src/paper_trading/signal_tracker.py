"""Live NIFTY S2 signal accuracy tracker (observation / analytics only).

The tracker never places, modifies or cancels orders. It imports no broker,
engine or order-manager code; it only reads option premium candles and writes
signal outcomes to the ``live_signal_outcomes`` table.

Outcome rules (S2 always buys premium: BUY -> CE, SELL -> PE):
* a candle is evaluated only when its timestamp is at or after the record's
  ``active_from`` (no look-ahead) and after the last candle already seen;
* the first candle at or after the session square-off exits at its open,
  as the research engine's EOD force exit does;
* an open beyond the stop/target fills at the open; otherwise a touch fills
  at the level;
* stop and target inside the same candle -> AMBIGUOUS (excluded from accuracy);
* nothing is resolved without a real candle as evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from src.safety.live_gate import live_trading_enabled, trading_mode

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
PENDING, WIN, LOSS, AMBIGUOUS, INVALID = "PENDING", "WIN", "LOSS", "AMBIGUOUS", "INVALID"
MARK_PREFIX = "live_signal_mark:"
MARK_HISTORY = 100
STATUSES = (PENDING, WIN, LOSS, AMBIGUOUS, INVALID)
REPORT_FIELDS = (
    "total", "resolved", "pending", "wins", "losses", "ambiguous", "invalid", "accuracy",
    "profit_factor", "gross_profit", "gross_loss", "net_pnl", "average_r", "max_drawdown",
    "max_consecutive_wins", "max_consecutive_losses",
)


def verify_observation_only() -> dict[str, Any]:
    """Startup safety check. The tracker has no order path, so it stays observation-only either way."""
    mode, live = trading_mode(), live_trading_enabled()
    ok = mode == "PAPER" and not live
    if not ok:
        log.warning("Signal tracker: TRADING_MODE=%s LIVE_TRADING_ENABLED=%s; tracker remains observation-only "
                    "and never places orders", mode, live)
    return {"trading_mode": mode, "live_trading_enabled": live, "paper_mode_ok": ok, "places_orders": False}


def _aware(ts: Any) -> datetime:
    value = pd.Timestamp(ts)
    if value.tzinfo is None:
        value = value.tz_localize(IST)
    return value.to_pydatetime()


def _utc_iso(ts: Any) -> str:
    return _aware(ts).astimezone(timezone.utc).isoformat()


def signal_id_for(strategy: str, underlying: str, timestamp: Any, direction: str) -> str:
    key = f"{strategy.upper()}|{underlying.upper()}|{_utc_iso(timestamp)}|{direction.upper()}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SignalRecord:
    """Immutable audit record of one S2 signal and its predefined levels."""

    signal_id: str
    timestamp: datetime
    underlying: str
    direction: str
    option_type: str | None
    strike: float | None
    expiry: str | None
    instrument_key: str | None
    trading_symbol: str | None
    signal_price: float | None
    entry_price: float | None
    stop_loss: float | None
    target: float | None
    quantity: int | None
    active_from: datetime
    strategy: str = "S2"
    source: str = "LIVE"
    status: str = PENDING

    @classmethod
    def create(cls, *, timestamp: Any, underlying: str, direction: str, active_from: Any,
               strategy: str = "S2", **fields: Any) -> "SignalRecord":
        ts = _aware(timestamp)
        return cls(
            signal_id=signal_id_for(strategy, underlying, ts, direction),
            timestamp=ts,
            underlying=underlying.upper(),
            direction=direction.upper(),
            active_from=_aware(active_from),
            strategy=strategy,
            option_type=fields.get("option_type"),
            strike=fields.get("strike"),
            expiry=fields.get("expiry"),
            instrument_key=fields.get("instrument_key"),
            trading_symbol=fields.get("trading_symbol"),
            signal_price=fields.get("signal_price"),
            entry_price=fields.get("entry_price"),
            stop_loss=fields.get("stop_loss"),
            target=fields.get("target"),
            quantity=fields.get("quantity"),
            source=fields.get("source", "LIVE"),
        )

    @property
    def session_date(self) -> date:
        return self.timestamp.astimezone(IST).date()

    def validation_error(self) -> str | None:
        if not self.instrument_key:
            return "NO_CONTRACT"
        if None in (self.entry_price, self.stop_loss, self.target):
            return "MISSING_LEVELS"
        if not (0 < self.stop_loss < self.entry_price < self.target):
            return "INVALID_LEVELS"
        return None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["active_from"] = self.active_from.isoformat()
        return data

    @classmethod
    def from_json(cls, raw: str) -> "SignalRecord":
        data = json.loads(raw)
        data["timestamp"] = _aware(data["timestamp"])
        data["active_from"] = _aware(data["active_from"])
        return cls(**data)

    def to_row(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "timestamp": _utc_iso(self.timestamp),
            "session_date": self.session_date.isoformat(),
            "strategy": self.strategy,
            "underlying": self.underlying,
            "direction": self.direction,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry,
            "instrument_key": self.instrument_key,
            "trading_symbol": self.trading_symbol,
            "signal_price": self.signal_price,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "quantity": self.quantity,
            "active_from": _utc_iso(self.active_from),
            "source": self.source,
            "record_json": json.dumps(self.as_dict(), sort_keys=True, default=str),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": PENDING,
            "reason": None,
        }


def _bar(candle: Any) -> dict[str, Any]:
    get = candle.get if isinstance(candle, Mapping) else (lambda k: getattr(candle, k))
    return {"timestamp": _aware(get("timestamp")), **{k: float(get(k)) for k in ("open", "high", "low", "close")}}


def evaluate_candle(record: SignalRecord, candle: Mapping[str, Any], square_off: time) -> dict[str, Any] | None:
    """Outcome evidenced by one completed premium candle, or None if the signal stays pending."""
    ts = candle["timestamp"]
    entry, stop, target = record.entry_price, record.stop_loss, record.target
    o, h, l = candle["open"], candle["high"], candle["low"]
    if ts.astimezone(IST).time() >= square_off:
        if o == entry:
            return _outcome(record, INVALID, o, ts, "EOD_EXIT_AT_ENTRY")
        return _outcome(record, WIN if o > entry else LOSS, o, ts, "EOD_SQUARE_OFF")
    if o <= stop:
        return _outcome(record, LOSS, o, ts, "STOP_LOSS_GAP")
    if o >= target:
        return _outcome(record, WIN, o, ts, "TARGET_GAP")
    hit_stop, hit_target = l <= stop, h >= target
    if hit_stop and hit_target:
        return _outcome(record, AMBIGUOUS, None, ts, "SL_AND_TARGET_SAME_CANDLE")
    if hit_stop:
        return _outcome(record, LOSS, stop, ts, "STOP_LOSS")
    if hit_target:
        return _outcome(record, WIN, target, ts, "TARGET")
    return None


def _outcome(record: SignalRecord, status: str, exit_price: float | None, ts: datetime, reason: str) -> dict[str, Any]:
    pnl = r_multiple = None
    if exit_price is not None:
        move = exit_price - record.entry_price
        r_multiple = round(move / (record.entry_price - record.stop_loss), 4)
        if record.quantity:
            pnl = round(move * record.quantity, 2)
    return {
        "status": status,
        "exit_price": exit_price,
        "exit_time": _utc_iso(ts),
        "pnl": pnl,
        "r_multiple": r_multiple,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }


class LiveSignalTracker:
    """Registers S2 signals once and resolves them from real premium candles."""

    def __init__(self, database, *, square_off: str = "15:15"):
        self.safety = verify_observation_only()
        self.database = database
        self.square_off = time.fromisoformat(square_off)
        self._lock = threading.RLock()
        self._pending: dict[str, SignalRecord] = {}
        self._last_seen: dict[str, datetime] = {}
        for row in database.get_live_signals():
            if row["status"] == PENDING:
                record = SignalRecord.from_json(row["record_json"])
                self._pending[record.signal_id] = record

    def register(self, record: SignalRecord, invalid_reason: str | None = None) -> bool:
        """Persist a new signal; duplicates (same strategy/underlying/time/direction) are ignored."""
        with self._lock:
            if not self.database.save_live_signal(record.to_row()):
                return False
            error = invalid_reason or record.validation_error()
            if error:
                self.database.resolve_live_signal(record.signal_id, {
                    "status": INVALID, "resolved_at": datetime.now(timezone.utc).isoformat(), "reason": error,
                })
            else:
                self._pending[record.signal_id] = record
            return True

    def pending_keys(self) -> set[str]:
        with self._lock:
            return {r.instrument_key for r in self._pending.values() if r.instrument_key}

    def on_tick(self, instrument_key: str, price: float, timestamp: Any) -> None:
        """Latest observed option premium tick (display only; outcomes are never resolved from ticks)."""
        ts = _aware(timestamp).astimezone(IST)
        with self._lock:
            for signal_id, record in self._pending.items():
                if record.instrument_key != instrument_key or ts < record.active_from:
                    continue
                self._mark_tick(signal_id, price, ts)

    def on_candle(self, instrument_key: str, candle: Any) -> list[dict[str, Any]]:
        """Feed one completed premium candle (in time order); returns outcomes resolved by it."""
        bar = _bar(candle)
        resolved: list[dict[str, Any]] = []
        with self._lock:
            for signal_id, record in list(self._pending.items()):
                if record.instrument_key != instrument_key or bar["timestamp"] < record.active_from:
                    continue
                last = self._last_seen.get(signal_id)
                if last is not None and bar["timestamp"] <= last:
                    continue
                if bar["timestamp"].astimezone(IST).date() != record.session_date:
                    self.database.note_pending_live_signal(signal_id, "SESSION_ENDED_WITHOUT_EXIT_EVIDENCE")
                    continue
                self._last_seen[signal_id] = bar["timestamp"]
                self._mark(signal_id, bar)
                outcome = evaluate_candle(record, bar, self.square_off)
                if outcome is None:
                    continue
                if self.database.resolve_live_signal(signal_id, outcome):
                    resolved.append({"signal_id": signal_id, **outcome})
                self._pending.pop(signal_id, None)
                self._last_seen.pop(signal_id, None)
        return resolved

    def _mark_tick(self, signal_id: str, price: float, timestamp: datetime) -> None:
        key = f"{MARK_PREFIX}{signal_id}"
        prev = self.database.get_state(key, {}) or {}
        tick_time = timestamp.astimezone(IST).isoformat()
        self.database.set_state(key, {
            **prev, "signal_id": signal_id, "price": float(price),
            "last_tick_time": tick_time, "last_tick_price": float(price),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def _mark(self, signal_id: str, bar: Mapping[str, Any]) -> None:
        """Latest completed premium candle of a tracked signal (display only; outcomes never read it)."""
        key = f"{MARK_PREFIX}{signal_id}"
        prev = self.database.get_state(key, {}) or {}
        point = {"time": _aware(bar["timestamp"]).astimezone(IST).isoformat(), "close": bar["close"]}
        history = [p for p in prev.get("history", []) if p.get("time") != point["time"]][-(MARK_HISTORY - 1):]
        tick_time = prev.get("last_tick_time")
        self.database.set_state(key, {
            "signal_id": signal_id, "price": bar["close"], "candle_time": point["time"],
            "last_tick_time": tick_time, "last_tick_price": prev.get("last_tick_price"),
            "updated_at": datetime.now(timezone.utc).isoformat(), "history": [*history, point],
        })

    def finalize_session(self, day: date) -> int:
        """Mark still-pending signals of ``day`` as lacking exit evidence; they stay PENDING."""
        count = 0
        with self._lock:
            for signal_id, record in self._pending.items():
                if record.session_date == day:
                    self.database.note_pending_live_signal(signal_id, "SESSION_ENDED_WITHOUT_EXIT_EVIDENCE")
                    count += 1
        return count


def _streaks(statuses: Iterable[str]) -> tuple[int, int]:
    best = {WIN: 0, LOSS: 0}
    current, run = None, 0
    for status in statuses:
        run = run + 1 if status == current else 1
        current = status
        best[status] = max(best[status], run)
    return best[WIN], best[LOSS]


def compute_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    decided = sorted((r for r in rows if r["status"] in (WIN, LOSS)),
                     key=lambda r: (r.get("exit_time") or r.get("resolved_at") or "", r["timestamp"]))
    resolved = len(decided)
    pnls = [float(r["pnl"]) for r in decided if r.get("pnl") is not None]
    gross_profit = round(sum(p for p in pnls if p > 0), 2)
    gross_loss = round(-sum(p for p in pnls if p < 0), 2)
    r_values = [float(r["r_multiple"]) for r in decided if r.get("r_multiple") is not None]
    equity = peak = drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    wins_run, losses_run = _streaks(r["status"] for r in decided)
    return {
        "total": len(rows),
        "resolved": resolved,
        "pending": counts[PENDING],
        "wins": counts[WIN],
        "losses": counts[LOSS],
        "ambiguous": counts[AMBIGUOUS],
        "invalid": counts[INVALID],
        "accuracy": round(counts[WIN] / resolved * 100, 4) if resolved else None,
        "accuracy_display": f"{counts[WIN] / resolved * 100:.2f}%" if resolved else "N/A",
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": round(gross_profit - gross_loss, 2),
        "average_r": round(sum(r_values) / len(r_values), 4) if r_values else None,
        "max_drawdown": round(drawdown, 2),
        "max_consecutive_wins": wins_run,
        "max_consecutive_losses": losses_run,
        "pnl_signals": len(pnls),
    }


PUBLIC_COLUMNS = (
    "signal_id", "time", "session_date", "strategy", "underlying", "direction", "option_type", "strike",
    "expiry", "contract", "signal_price", "entry_price", "stop_loss", "target", "quantity", "status",
    "exit_price", "exit_time", "pnl", "r_multiple", "resolved_at", "reason", "source",
)


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Table row for the API/dashboard (IST times, no internal JSON)."""
    def ist(value):
        return _aware(value).astimezone(IST).isoformat() if value else None

    return {
        "signal_id": row["signal_id"],
        "time": ist(row["timestamp"]),
        "session_date": row["session_date"],
        "strategy": row["strategy"],
        "underlying": row["underlying"],
        "direction": row["direction"],
        "option_type": row["option_type"],
        "strike": row["strike"],
        "expiry": row["expiry"],
        "contract": row["trading_symbol"] or row["instrument_key"],
        "signal_price": row["signal_price"],
        "entry_price": row["entry_price"],
        "stop_loss": row["stop_loss"],
        "target": row["target"],
        "quantity": row["quantity"],
        "status": row["status"],
        "exit_price": row["exit_price"],
        "exit_time": ist(row["exit_time"]),
        "pnl": row["pnl"],
        "r_multiple": row["r_multiple"],
        "resolved_at": row["resolved_at"],
        "reason": row["reason"],
        "source": row["source"],
    }


def write_daily_report(rows: list[Mapping[str, Any]], day: date, out_dir: Path,
                       notes: list[str] | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_stats(rows)
    table = [public_row(r) for r in rows]
    payload = {
        "date": day.isoformat(),
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "mode": "PAPER observation only - no orders placed",
        **{k: stats[k] for k in REPORT_FIELDS},
        "accuracy_display": stats["accuracy_display"],
        "notes": notes or [],
        "signals": table,
    }
    json_path = out_dir / f"{day.isoformat()}.json"
    csv_path = out_dir / f"{day.isoformat()}.csv"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    columns = list(table[0].keys()) if table else list(PUBLIC_COLUMNS)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# " + ", ".join(f"{k}={stats[k] if stats[k] is not None else 'N/A'}" for k in REPORT_FIELDS) + "\n")
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    return {"json": json_path, "csv": csv_path}
