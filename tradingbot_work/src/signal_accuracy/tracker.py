"""S2 signal events, outcome tracking and accuracy reporting on NIFTY spot prices.

Observation only: nothing here places, modifies or cancels orders.

Level convention (unchanged S2 research settings, ``strategy_research.yaml``):
* entry  = ``option_mapping.entry_timing`` -> ``next_open``: the first real price
  observed at/after the signal candle closes (``signal_close``: the candle close);
* stop   = entry -/+ ``exit.stop_loss.multiplier`` x S2 ATR of the signal candle;
* target = entry +/- ``exit.target.risk_reward`` x stop distance;
* breakeven/trailing are not applied (same as the existing live signal tracker).

Outcome policy:
* tick path (realtime): prices are evaluated in arrival order, so the first
  touch decides; there is no same-candle ambiguity;
* candle path (no tick ordering): stop and target inside one candle ->
  AMBIGUOUS, never the favourable result;
* stop exits are debited at the observed price when price is beyond the stop;
  target exits are credited at the target level only (no gap bonus);
* the first price at/after square-off exits as TIME_EXIT at that price;
* a data gap, restart or run end while a signal is open -> UNRESOLVED;
  nothing is resolved without a real observed price.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

SIGNAL_GENERATED = "SIGNAL_GENERATED"
TRACKING = "TRACKING"
TARGET_HIT = "TARGET_HIT"
STOP_HIT = "STOP_HIT"
TIME_EXIT = "TIME_EXIT"
AMBIGUOUS = "AMBIGUOUS"
UNRESOLVED = "UNRESOLVED"
ACTIVE = (SIGNAL_GENERATED, TRACKING)
FINAL = (TARGET_HIT, STOP_HIT, TIME_EXIT, AMBIGUOUS, UNRESOLVED)

WIN, LOSS, FLAT = "WIN", "LOSS", "FLAT"

POLICY_NOTES = [
    "PAPER observation only: no broker, engine or order code is imported; nothing is ordered.",
    "S2 is evaluated only on completed 5-minute candles built from real ticks, using candles up to that candle.",
    "Entry = first real price at/after the signal candle close (entry_timing=next_open).",
    "Stop = entry -/+ exit.stop_loss.multiplier x S2 ATR of the signal candle; target = exit.target.risk_reward x risk.",
    "Breakeven/trailing are not applied; every signal is tracked independently (signal accuracy, not a trade simulation).",
    "Realtime outcomes use tick order (first touch). Candle-only evaluation marks stop+target in one candle AMBIGUOUS.",
    "Stop exits use the observed price (can be worse than the stop); target exits are credited at the target only.",
    "First price at/after square-off -> TIME_EXIT at that price; WIN/LOSS by sign of the points.",
    "A data gap, restart or run end while a signal is open -> UNRESOLVED (excluded from accuracy).",
    "Accuracy = wins / (wins + losses); N/A when there are no resolved WIN/LOSS signals.",
    "Conservative accuracy also counts AMBIGUOUS as losses: wins / (wins + losses + ambiguous).",
    "Candles that started before the tracker was listening or that overlap a data gap are not evaluated.",
]


def to_ist(ts: Any) -> datetime:
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.astimezone(IST)


def iso(ts: Any) -> str:
    return to_ist(ts).isoformat()


def signal_id_for(strategy: str, instrument: str, timeframe: str, timestamp: Any, direction: str) -> str:
    key = f"{strategy.upper()}|{instrument}|{timeframe}|{iso(timestamp)}|{direction.upper()}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class LevelConvention:
    entry_timing: str
    stop_atr_multiplier: float
    target_r: float
    square_off: time
    interval_minutes: int

    @classmethod
    def from_research(cls, research, *, square_off: str, interval_minutes: int) -> "LevelConvention":
        exit_cfg = research.exit
        stop = exit_cfg.get("stop_loss") or {}
        if str(stop.get("type", "")).lower() != "atr":
            raise ValueError(f"Unsupported S2 stop type {stop.get('type')!r}; only 'atr' is defined")
        timing = str(research.option_mapping.get("entry_timing", "next_open"))
        if timing not in ("next_open", "signal_close"):
            raise ValueError(f"Unsupported entry_timing {timing!r}")
        return cls(
            entry_timing=timing,
            stop_atr_multiplier=float(stop["multiplier"]),
            target_r=float(exit_cfg["target"]["risk_reward"]),
            square_off=time.fromisoformat(square_off),
            interval_minutes=int(interval_minutes),
        )


@dataclass
class SignalEvent:
    session_id: str
    signal_id: str
    timestamp_ist: str
    instrument: str
    timeframe: str
    strategy_name: str
    direction: str
    signal_price: float
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    atr: float | None
    stop_distance: float | None
    entry_timing: str
    active_from: str
    entry_price: float | None = None
    entry_time: str | None = None
    stop_price: float | None = None
    target_price: float | None = None
    risk_points: float | None = None
    status: str = SIGNAL_GENERATED
    outcome: str | None = None
    outcome_timestamp: str | None = None
    exit_price: float | None = None
    pnl_points: float | None = None
    mfe_points: float | None = None
    mae_points: float | None = None
    last_price: float | None = None
    last_price_time: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalEvent":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @property
    def session_date(self) -> str:
        return to_ist(self.timestamp_ist).date().isoformat()


CSV_FIELDS = [f.name for f in fields(SignalEvent)]


def new_signal(*, session_id: str, candle: Mapping[str, Any], direction: str, atr: float | None,
               convention: LevelConvention, instrument: str, timeframe: str, strategy: str) -> SignalEvent:
    """Signal event for a completed candle; levels are set only once the entry price is observed."""
    ts = to_ist(candle["timestamp"])
    valid_atr = atr is not None and atr == atr and atr > 0
    distance = round(float(atr) * convention.stop_atr_multiplier, 4) if valid_atr else None
    return SignalEvent(
        session_id=session_id,
        signal_id=signal_id_for(strategy, instrument, timeframe, ts, direction),
        timestamp_ist=ts.isoformat(),
        instrument=instrument,
        timeframe=timeframe,
        strategy_name=strategy.upper(),
        direction=direction.upper(),
        signal_price=float(candle["close"]),
        candle_open=float(candle["open"]),
        candle_high=float(candle["high"]),
        candle_low=float(candle["low"]),
        candle_close=float(candle["close"]),
        atr=round(float(atr), 4) if valid_atr else None,
        stop_distance=distance,
        entry_timing=convention.entry_timing,
        active_from=(ts + timedelta(minutes=convention.interval_minutes)).isoformat(),
    )


EventSink = Callable[[str, SignalEvent], None]


class OutcomeTracker:
    """Resolves registered signals from subsequent real prices only."""

    def __init__(self, convention: LevelConvention, on_event: EventSink | None = None):
        self.convention = convention
        self.signals: dict[str, SignalEvent] = {}
        self._on_event = on_event or (lambda _kind, _ev: None)

    def known(self, signal_id: str) -> bool:
        return signal_id in self.signals

    def restore(self, event: SignalEvent) -> None:
        self.signals[event.signal_id] = event

    def active(self) -> list[SignalEvent]:
        return [ev for ev in self.signals.values() if ev.status in ACTIVE]

    def register(self, event: SignalEvent) -> bool:
        """Store a new signal once; returns False for a duplicate signal_id."""
        if event.signal_id in self.signals:
            return False
        self.signals[event.signal_id] = event
        self._on_event("SIGNAL", event)
        active_from = to_ist(event.active_from)
        if event.stop_distance is None:
            self._close(event, UNRESOLVED, None, active_from, "INSUFFICIENT_ATR_HISTORY")
        elif event.entry_timing == "signal_close":
            self._enter(event, event.signal_price, active_from)
        return True

    # -- realtime tick path ---------------------------------------------------

    def on_tick(self, ts: datetime, price: float) -> list[SignalEvent]:
        ts = to_ist(ts)
        resolved = []
        for ev in self.active():
            if ts < to_ist(ev.active_from):
                continue
            if self._session_over(ev, ts):
                resolved.append(ev)
                continue
            if ev.status == SIGNAL_GENERATED:
                if ts.time() >= self.convention.square_off:
                    self._close(ev, UNRESOLVED, None, ts, "NO_ENTRY_BEFORE_SQUARE_OFF")
                    resolved.append(ev)
                else:
                    self._enter(ev, price, ts)
                continue
            self._excursion(ev, price, ts)
            if ts.time() >= self.convention.square_off:
                self._close(ev, TIME_EXIT, price, ts, "SQUARE_OFF")
            elif self._stop_hit(ev, price):
                self._close(ev, STOP_HIT, price, ts, "STOP_LOSS")
            elif self._target_hit(ev, price):
                self._close(ev, TARGET_HIT, ev.target_price, ts, "TARGET")
            if ev.status not in ACTIVE:
                resolved.append(ev)
        return resolved

    # -- candle path (tick ordering unavailable) -------------------------------

    def on_bar(self, bar: Mapping[str, Any]) -> list[SignalEvent]:
        start = to_ist(bar["timestamp"])
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        resolved = []
        for ev in self.active():
            if start < to_ist(ev.active_from):
                continue
            if self._session_over(ev, start):
                resolved.append(ev)
                continue
            if ev.status == SIGNAL_GENERATED:
                if start.time() >= self.convention.square_off:
                    self._close(ev, UNRESOLVED, None, start, "NO_ENTRY_BEFORE_SQUARE_OFF")
                    resolved.append(ev)
                    continue
                self._enter(ev, o, start)
            if start.time() >= self.convention.square_off:
                self._excursion(ev, o, start)
                self._close(ev, TIME_EXIT, o, start, "SQUARE_OFF")
                resolved.append(ev)
                continue
            if self._stop_hit(ev, o):
                self._excursion(ev, o, start)
                self._close(ev, STOP_HIT, o, start, "STOP_LOSS_GAP")
                resolved.append(ev)
                continue
            if self._target_hit(ev, o):
                self._excursion(ev, o, start)
                self._close(ev, TARGET_HIT, ev.target_price, start, "TARGET_GAP")
                resolved.append(ev)
                continue
            worst, best = (l, h) if ev.direction == "BUY" else (h, l)
            hit_stop = self._stop_hit(ev, worst)
            hit_target = self._target_hit(ev, best)
            if hit_stop and hit_target:
                self._excursion(ev, worst, start)
                self._excursion(ev, best, start)
                self._close(ev, AMBIGUOUS, None, start, "SL_AND_TARGET_SAME_CANDLE")
            elif hit_stop:
                self._excursion(ev, worst, start)
                self._close(ev, STOP_HIT, ev.stop_price, start, "STOP_LOSS")
            elif hit_target:
                self._excursion(ev, best, start)
                self._close(ev, TARGET_HIT, ev.target_price, start, "TARGET")
            else:
                self._excursion(ev, worst, start)
                self._excursion(ev, best, start)
                ev.last_price, ev.last_price_time = float(bar["close"]), start.isoformat()
            if ev.status not in ACTIVE:
                resolved.append(ev)
        return resolved

    # -- data gaps / end of run ---------------------------------------------------

    def mark_gap(self, ts: datetime, reason: str) -> list[SignalEvent]:
        """Open signals cannot be resolved honestly across missing data."""
        out = []
        for ev in self.active():
            self._close(ev, UNRESOLVED, None, to_ist(ts), f"DATA_GAP:{reason}")
            out.append(ev)
        return out

    def finalize(self, ts: datetime, reason: str = "RUN_ENDED_BEFORE_OUTCOME") -> list[SignalEvent]:
        out = []
        for ev in self.active():
            self._close(ev, UNRESOLVED, None, to_ist(ts), reason)
            out.append(ev)
        return out

    # -- internals -------------------------------------------------------------

    def _session_over(self, ev: SignalEvent, ts: datetime) -> bool:
        if ts.date().isoformat() == ev.session_date:
            return False
        self._close(ev, UNRESOLVED, None, ts, "SESSION_ENDED_WITHOUT_EXIT")
        return True

    def _enter(self, ev: SignalEvent, price: float, ts: datetime) -> None:
        d = ev.stop_distance
        sign = 1 if ev.direction == "BUY" else -1
        ev.entry_price = round(float(price), 2)
        ev.entry_time = ts.isoformat()
        ev.stop_price = round(ev.entry_price - sign * d, 2)
        ev.target_price = round(ev.entry_price + sign * d * self.convention.target_r, 2)
        ev.risk_points = round(abs(ev.entry_price - ev.stop_price), 2)
        ev.mfe_points = ev.mae_points = 0.0
        ev.last_price, ev.last_price_time = ev.entry_price, ts.isoformat()
        ev.status = TRACKING
        self._on_event("TRACKING", ev)

    @staticmethod
    def _favourable(ev: SignalEvent, price: float) -> float:
        move = float(price) - ev.entry_price
        return move if ev.direction == "BUY" else -move

    @staticmethod
    def _stop_hit(ev: SignalEvent, price: float) -> bool:
        return price <= ev.stop_price if ev.direction == "BUY" else price >= ev.stop_price

    @staticmethod
    def _target_hit(ev: SignalEvent, price: float) -> bool:
        return price >= ev.target_price if ev.direction == "BUY" else price <= ev.target_price

    def _excursion(self, ev: SignalEvent, price: float, ts: datetime) -> None:
        fav = self._favourable(ev, price)
        ev.mfe_points = round(max(ev.mfe_points or 0.0, fav), 2)
        ev.mae_points = round(max(ev.mae_points or 0.0, -fav), 2)
        ev.last_price, ev.last_price_time = float(price), ts.isoformat()

    def _close(self, ev: SignalEvent, status: str, exit_price: float | None, ts: datetime, reason: str) -> None:
        ev.status = status
        ev.reason = reason
        ev.outcome_timestamp = ts.isoformat()
        ev.exit_price = round(float(exit_price), 2) if exit_price is not None else None
        if ev.exit_price is not None and ev.entry_price is not None:
            ev.pnl_points = round(self._favourable(ev, ev.exit_price), 2)
        if status == TARGET_HIT:
            ev.outcome = WIN
        elif status == STOP_HIT:
            ev.outcome = LOSS
        elif status == TIME_EXIT:
            ev.outcome = WIN if ev.pnl_points > 0 else LOSS if ev.pnl_points < 0 else FLAT
        else:
            ev.outcome = status
        self._on_event("OUTCOME", ev)


# -- summary -------------------------------------------------------------------

def _ratio(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 4) if den else None


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _accuracy(rows: list[SignalEvent]) -> float | None:
    wins = sum(1 for r in rows if r.outcome == WIN)
    losses = sum(1 for r in rows if r.outcome == LOSS)
    return _ratio(wins, wins + losses)


def summarize(signals: Iterable[SignalEvent], *, session_date: str | None, strategy: str, instrument: str,
              timeframe: str) -> dict[str, Any]:
    rows = list(signals)
    wins = [r for r in rows if r.outcome == WIN]
    losses = [r for r in rows if r.outcome == LOSS]
    ambiguous = [r for r in rows if r.status == AMBIGUOUS]
    decided = sorted((r for r in rows if r.outcome in (WIN, LOSS, FLAT) and r.pnl_points is not None),
                     key=lambda r: (r.outcome_timestamp or "", r.timestamp_ist))
    gross_win = sum(r.pnl_points for r in decided if r.pnl_points > 0)
    gross_loss = -sum(r.pnl_points for r in decided if r.pnl_points < 0)
    equity = peak = drawdown = 0.0
    for r in decided:
        equity += r.pnl_points
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "session_date": session_date,
        "strategy": strategy,
        "instrument": instrument,
        "timeframe": timeframe,
        "total_signals": len(rows),
        "buy_signals": sum(1 for r in rows if r.direction == "BUY"),
        "sell_signals": sum(1 for r in rows if r.direction == "SELL"),
        "resolved_signals": sum(1 for r in rows if r.status in (TARGET_HIT, STOP_HIT, TIME_EXIT, AMBIGUOUS)),
        "unresolved_signals": sum(1 for r in rows if r.status == UNRESOLVED),
        "open_signals": sum(1 for r in rows if r.status in ACTIVE),
        "wins": len(wins),
        "losses": len(losses),
        "flat": sum(1 for r in rows if r.outcome == FLAT),
        "ambiguous": len(ambiguous),
        "time_exits": sum(1 for r in rows if r.status == TIME_EXIT),
        "target_hits": sum(1 for r in rows if r.status == TARGET_HIT),
        "stop_hits": sum(1 for r in rows if r.status == STOP_HIT),
        "accuracy": _ratio(len(wins), len(wins) + len(losses)),
        "accuracy_conservative": _ratio(len(wins), len(wins) + len(losses) + len(ambiguous)),
        "buy_accuracy": _accuracy([r for r in rows if r.direction == "BUY"]),
        "sell_accuracy": _accuracy([r for r in rows if r.direction == "SELL"]),
        "average_winning_points": _mean([r.pnl_points for r in wins if r.pnl_points is not None]),
        "average_losing_points": _mean([r.pnl_points for r in losses if r.pnl_points is not None]),
        "total_points": round(sum(r.pnl_points for r in decided), 2),
        "gross_winning_points": round(gross_win, 2),
        "gross_losing_points": round(gross_loss, 2),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "max_drawdown_points": round(drawdown, 2),
        "average_mfe_points": _mean([r.mfe_points for r in decided if r.mfe_points is not None]),
        "average_mae_points": _mean([r.mae_points for r in decided if r.mae_points is not None]),
    }


def profit_factor_display(summary: Mapping[str, Any]) -> str:
    if summary["profit_factor"] is not None:
        return f"{summary['profit_factor']:.2f}"
    return "no losses" if summary["gross_winning_points"] > 0 else "N/A"


# -- persistence -------------------------------------------------------------------

class SessionStore:
    """Append-only JSONL audit log of one session; restart reads it back."""

    def __init__(self, out_dir: Path, session_id: str):
        self.out_dir = Path(out_dir)
        self.session_id = session_id
        self.jsonl = self.out_dir / f"session_{session_id}.jsonl"
        self.json = self.out_dir / f"session_{session_id}.json"
        self.corrupt_lines = 0

    def append(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        row = {"type": kind, "session_id": self.session_id,
               "logged_at": datetime.now(timezone.utc).astimezone(IST).isoformat(), **payload}
        with self.jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()

    def load(self) -> tuple[dict[str, SignalEvent], list[dict[str, Any]]]:
        """Latest state of every signal plus complete candles already written."""
        signals: dict[str, SignalEvent] = {}
        candles: list[dict[str, Any]] = []
        if not self.jsonl.exists():
            return signals, candles
        for line in self.jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                self.corrupt_lines += 1
                continue
            if row.get("type") in ("SIGNAL", "TRACKING", "OUTCOME") and "signal" in row:
                ev = SignalEvent.from_dict(row["signal"])
                signals[ev.signal_id] = ev
            elif row.get("type") == "CANDLE" and row.get("complete"):
                candles.append(row["candle"])
        return signals, candles


def _md_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summary_markdown(payload: Mapping[str, Any]) -> str:
    s, c, dq = payload["session"], payload["cumulative"], payload["data_quality"]
    lines = [f"# S2 live PAPER signal accuracy — session {payload['session_id']}", "",
             f"> {payload['mode']}", "",
             "| Metric | Session | All sessions |", "|---|---:|---:|"]
    rows = [
        ("Session date", s["session_date"], f"{c['sessions']} session(s)"),
        ("Strategy / instrument / timeframe", f"{s['strategy']} / {s['instrument']} / {s['timeframe']}", ""),
        ("Total signals", s["total_signals"], c["total_signals"]),
        ("BUY / SELL", f"{s['buy_signals']} / {s['sell_signals']}", f"{c['buy_signals']} / {c['sell_signals']}"),
        ("Resolved", s["resolved_signals"], c["resolved_signals"]),
        ("Unresolved", s["unresolved_signals"], c["unresolved_signals"]),
        ("Wins / losses", f"{s['wins']} / {s['losses']}", f"{c['wins']} / {c['losses']}"),
        ("Ambiguous", s["ambiguous"], c["ambiguous"]),
        ("Time exits", s["time_exits"], c["time_exits"]),
        ("Accuracy", pct(s["accuracy"]), pct(c["accuracy"])),
        ("Conservative accuracy (ambiguous = loss)", pct(s["accuracy_conservative"]), pct(c["accuracy_conservative"])),
        ("BUY accuracy", pct(s["buy_accuracy"]), pct(c["buy_accuracy"])),
        ("SELL accuracy", pct(s["sell_accuracy"]), pct(c["sell_accuracy"])),
        ("Average winning points", _md_value(s["average_winning_points"]), _md_value(c["average_winning_points"])),
        ("Average losing points", _md_value(s["average_losing_points"]), _md_value(c["average_losing_points"])),
        ("Total points", _md_value(s["total_points"]), _md_value(c["total_points"])),
        ("Profit factor", profit_factor_display(s), profit_factor_display(c)),
        ("Max drawdown (points)", _md_value(s["max_drawdown_points"]), _md_value(c["max_drawdown_points"])),
        ("Average MFE (points)", _md_value(s["average_mfe_points"]), _md_value(c["average_mfe_points"])),
        ("Average MAE (points)", _md_value(s["average_mae_points"]), _md_value(c["average_mae_points"])),
    ]
    lines += [f"| {a} | {b} | {d} |" for a, b, d in rows]
    lines += ["", "## Data quality", "", "| Counter | Value |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in dq.items()]
    lines += ["", "## Signals (this session)", "",
              "| Time | Dir | Signal close | Entry | SL | Target | Status | Exit | Points | MFE | MAE | Reason |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in payload["signals"]:
        lines.append(
            f"| {r['timestamp_ist'][11:19]} | {r['direction']} | {_md_value(r['signal_price'])} | "
            f"{_md_value(r['entry_price'])} | {_md_value(r['stop_price'])} | {_md_value(r['target_price'])} | "
            f"{r['status']} | {_md_value(r['exit_price'])} | {_md_value(r['pnl_points'])} | "
            f"{_md_value(r['mfe_points'])} | {_md_value(r['mae_points'])} | {r['reason'] or ''} |")
    lines += ["", "## Safety", ""] + [f"- {k}: {v}" for k, v in payload["safety"].items()]
    lines += ["", "## Policy", ""] + [f"- {n}" for n in payload["policy"]]
    return "\n".join(lines) + "\n"


def write_reports(store: SessionStore, *, session: Mapping[str, Any], signals: list[SignalEvent],
                  data_quality: Mapping[str, Any], safety: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Path]:
    out = store.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted((s.as_dict() for s in signals), key=lambda r: (r["timestamp_ist"], r["direction"]))
    mode = "PAPER observation only - no orders placed"
    store.json.write_text(json.dumps({
        "session_id": store.session_id, "mode": mode, **meta, "summary": session,
        "data_quality": dict(data_quality), "safety": dict(safety), "policy": POLICY_NOTES, "signals": rows,
    }, indent=2, default=str), encoding="utf-8")

    every: dict[str, SignalEvent] = {}
    sessions = 0
    for path in sorted(out.glob("session_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sessions += 1
        for raw in data.get("signals", []):
            ev = SignalEvent.from_dict(raw)
            every[ev.signal_id] = ev
    all_rows = sorted(every.values(), key=lambda e: (e.timestamp_ist, e.direction))
    cumulative = summarize(all_rows, session_date=None, strategy=session["strategy"],
                           instrument=session["instrument"], timeframe=session["timeframe"])
    cumulative["sessions"] = sessions

    csv_path = out / "signals.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for ev in all_rows:
            writer.writerow({k: ("" if v is None else v) for k, v in ev.as_dict().items()})

    payload = {
        "session_id": store.session_id, "mode": mode,
        "generated_at": datetime.now(timezone.utc).astimezone(IST).isoformat(timespec="seconds"),
        "session": dict(session), "cumulative": cumulative, "data_quality": dict(data_quality),
        "safety": dict(safety), "policy": POLICY_NOTES, "signals": rows, **meta,
    }
    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    summary_md = out / "summary.md"
    summary_md.write_text(summary_markdown(payload), encoding="utf-8")
    return {"jsonl": store.jsonl, "session_json": store.json, "signals_csv": csv_path,
            "summary_json": summary_json, "summary_md": summary_md}
