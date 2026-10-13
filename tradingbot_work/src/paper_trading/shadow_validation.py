"""Forward shadow validation for the three S2 direction candidates.

Observation-only: no broker/order calls, no production S2 changes.
Candidates:
- BASELINE_S2_BUY_SELL
- SELL_ONLY
- SELL_CONFIRMED

All candidates share one pipeline run (same NIFTY candles, contract resolution, entry timing, premium candles and
exit rules); only the direction/confirmation filter differs. Every S2 signal is stored once per candidate under
``shadow_id`` (candidate|session_date|signal_time|direction, also enforced by a UNIQUE constraint), so re-running a
day reconciles PENDING / DATA_UNAVAILABLE outcomes instead of adding rows. Missing data is never synthesised.

LIVE_QUALITY_GATE_V1 is an optional observation-only filter above BASELINE_S2_BUY_SELL, produced only when
``run_once(quality_gate=...)`` is given. BUY_BREAKOUT_CONFIRM is research-only and excluded from these reports.
"""
from __future__ import annotations

import csv
import importlib
import json
import logging
import math
import re
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest import live_quality_gate as qg
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.database import Database
from src.safety.live_gate import LiveTradingSafetyError, assert_paper_trading_only

IST = ZoneInfo("Asia/Kolkata")
CANDIDATES = (
    dr.BASELINE,
    dr.DirectionCandidate("SELL_ONLY", ("SELL",)),
    dr.DirectionCandidate("SELL_CONFIRMED", ("SELL",), confirm=True),
)
# Observation-only layers above BASELINE_S2_BUY_SELL; produced only when explicitly enabled (never by default).
OBSERVATION_CANDIDATES = (qg.GATE_CANDIDATE,)
# Research-only rows that may exist in older databases; never part of the production comparison.
RESEARCH_ONLY_CANDIDATES = ("BUY_BREAKOUT_CONFIRM",)
STATUSES = {imp.WIN, imp.LOSS, imp.AMBIGUOUS, imp.SCRATCH, imp.PENDING, imp.UNAVAILABLE, imp.INVALID}
DECIDED = (imp.WIN, imp.LOSS, imp.AMBIGUOUS, imp.SCRATCH)
LOT_SIZE = 65
HISTORY_DAYS = 35

EVALUATED, PENDING, DATA_UNAVAILABLE, SKIPPED = "EVALUATED", imp.PENDING, imp.UNAVAILABLE, "SKIPPED"
SIGNAL_STATES = (EVALUATED, PENDING, DATA_UNAVAILABLE, SKIPPED)

REASON_TEXT = {
    "NO_PREMIUM_DATA": "missing option candle (no premium series)",
    "NO_PREMIUM_CANDLE_AT_SIGNAL": "missing option candle at signal time",
    "NO_PREMIUM_ENTRY_CANDLE": "missing option entry candle",
    "INSUFFICIENT_ATR_HISTORY": "insufficient ATR history",
    "MISSING_LOCAL_PARQUET": "missing local parquet",
    "NO_EXIT_EVIDENCE": "no stop/target touch in available candles yet",
}
API_REASON_TEXT = {
    "AUTH_FAILED": "Upstox historical authorization unavailable",
    "PERMISSION_DENIED": "Upstox historical authorization unavailable",
    "CREDENTIALS_MISSING": "Upstox historical authorization unavailable",
    "PLAN_NOT_SUPPORTED": "expired instrument unavailable",
    "NOT_FOUND": "expired instrument unavailable",
    "INVALID_RESPONSE": "malformed API response",
}
UPSTOX_EVENTS = ("upstox_historical_error", "upstox_historical_retry")

# Every method that can submit, modify or cancel an order. Patched to raise while a shadow run is active.
ORDER_ENTRY_POINTS = (
    ("src.broker.paper", "PaperBroker", "place_order"),
    ("src.broker.upstox_sandbox", "UpstoxSandboxBroker", "place_order"),
    ("src.broker.adapters", "UnconfiguredBrokerAdapter", "place_order"),
    ("src.execution.order_manager", "OrderManager", "submit_signal"),
    ("src.execution.order_manager", "OrderManager", "modify"),
    ("src.execution.order_manager", "OrderManager", "cancel"),
    ("src.engine", "TradingEngine", "start"),
    ("src.engine", "TradingEngine", "submit_entry"),
)


@contextmanager
def order_placement_blocked() -> Iterator[None]:
    saved = []

    def guard_for(name: str):
        def guard(*_args, **_kwargs):
            raise LiveTradingSafetyError(f"SHADOW_VALIDATION_ORDER_BLOCKED: {name}")
        return guard

    try:
        for module, cls_name, method in ORDER_ENTRY_POINTS:
            cls = getattr(importlib.import_module(module), cls_name)
            saved.append((cls, method, cls.__dict__[method]))
            setattr(cls, method, guard_for(f"{cls_name}.{method}"))
        yield
    finally:
        for cls, method, original in reversed(saved):
            setattr(cls, method, original)


class _UpstoxEventCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.events: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() in UPSTOX_EVENTS:
            self.events.append({"event": record.getMessage(), **(getattr(record, "context", None) or {})})


@contextmanager
def capture_upstox_events() -> Iterator[list[dict[str, Any]]]:
    handler = _UpstoxEventCapture()
    upstox_logger = logging.getLogger("historical_data.upstox")
    upstox_logger.addHandler(handler)
    try:
        yield handler.events
    finally:
        upstox_logger.removeHandler(handler)


# ------------------------------------------------------------------ signal states
def shadow_id(candidate: str, row: dict) -> str:
    return f"{candidate}|{row['session_date']}|{row['signal_time']}|{row['direction']}"


def _status(row: dict) -> str:
    if row["decision"] != imp.ACCEPTED:
        return "SKIPPED"
    return str(row.get("outcome") or imp.PENDING)


def signal_state(row: dict) -> str:
    """EVALUATED / PENDING / DATA_UNAVAILABLE / SKIPPED for a stored shadow row."""
    if row["decision"] != imp.ACCEPTED:
        return SKIPPED
    if row["status"] in DECIDED:
        return EVALUATED
    if row["status"] == imp.PENDING:
        return PENDING
    return DATA_UNAVAILABLE


def describe_reason(code: Any) -> str | None:
    if not code:
        return None
    code = str(code)
    if code.startswith("PREMIUM_API_ERROR:"):
        api = code.split(":", 1)[1]
        return f"{API_REASON_TEXT.get(api, 'Upstox API error')} ({api})"
    if code.startswith("CONTRACT_UNRESOLVED:"):
        return f"option contract unresolved ({code.split(':', 1)[1]})"
    return REASON_TEXT.get(code, code)


def data_reason(row: dict) -> str | None:
    """Reason code for PENDING / DATA_UNAVAILABLE rows (latest reconciliation first, then the original record)."""
    if signal_state(row) not in (PENDING, DATA_UNAVAILABLE):
        return None
    if row.get("exit_reason"):
        return str(row["exit_reason"])
    try:
        return json.loads(row.get("record_json") or "{}").get("data_reason") or "UNKNOWN"
    except ValueError:
        return "UNKNOWN"


def save_rows(db: Database, rows_by_candidate: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Insert new signals; reconcile PENDING / DATA_UNAVAILABLE outcomes of already-stored ones."""
    counts: dict[str, dict[str, int]] = {}
    for candidate, rows in rows_by_candidate.items():
        c = counts.setdefault(candidate, {"created": 0, "updated": 0, "unchanged": 0})
        for r in rows:
            sid = shadow_id(candidate, r)
            payload = dict(r)
            payload.pop("premium", None)
            if hasattr(payload.get("contract"), "__dict__"):
                payload["contract"] = str(payload["contract"])
            status = _status(r)
            pnl, r_multiple = r.get("pnl"), r.get("r_multiple")
            if status == imp.AMBIGUOUS and pnl is None and r.get("stop_loss") is not None and r.get("entry") is not None:
                pnl, r_multiple = (float(r["stop_loss"]) - float(r["entry"])) * LOT_SIZE, -1.0
            exit_reason = r.get("exit_reason")
            if status in (imp.PENDING, imp.UNAVAILABLE) and not exit_reason:
                exit_reason = r.get("data_reason")
            record = {
                "shadow_id": sid, "candidate": candidate,
                "timestamp": r["signal_time"], "session_date": r["session_date"].isoformat()
                    if hasattr(r["session_date"], "isoformat") else str(r["session_date"]),
                "underlying": "NIFTY", "direction": r["direction"],
                "instrument_key": r.get("instrument_key"), "expiry": str(r.get("expiry")) if r.get("expiry") else None,
                "dte": r.get("dte"), "decision": r.get("decision"), "skip_reason": r.get("skip_reason"),
                "status": status, "entry_price": r.get("entry"), "stop_loss": r.get("stop_loss"),
                "target": r.get("target"), "exit_price": r.get("exit_price"), "exit_time": r.get("exit_time"),
                "exit_reason": exit_reason, "pnl": pnl, "r_multiple": r_multiple,
                "market_regime": None, "record_json": json.dumps(payload, default=str, sort_keys=True),
                "created_at": datetime.now(IST).isoformat(timespec="seconds"),
            }
            gate = r.get("quality_gate")
            if gate:
                record.update(gate_score=gate["score"], gate_max_score=gate["max_score"],
                              gate_confidence=gate["confidence"], gate_version=gate["version"],
                              gate_reasons=json.dumps(gate["reasons"]),
                              gate_failed_checks=json.dumps(gate["failed_checks"]),
                              gate_features=json.dumps(gate["features"], default=str, sort_keys=True))
            if db.save_shadow_signal(record):
                c["created"] += 1
            elif db.update_shadow_outcome(sid, record):
                c["updated"] += 1
            else:
                c["unchanged"] += 1
    return counts


# ------------------------------------------------------------------ metrics
def trade_pnl(row: dict) -> float:
    """Realised P&L; an ambiguous candle (stop and target inside one bar) is booked as the stop (-1R)."""
    if row["status"] == imp.AMBIGUOUS and row.get("pnl") is None:
        if row.get("stop_loss") is not None and row.get("entry_price") is not None:
            return (float(row["stop_loss"]) - float(row["entry_price"])) * LOT_SIZE
    return float(row.get("pnl") or 0.0)


def _pf(gross_profit: float, gross_loss: float) -> float | None:
    return gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit else None)


def metrics(rows: Sequence[dict]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: (r["session_date"], r["timestamp"]))
    states = [signal_state(r) for r in rows]
    accepted = [r for r in rows if r["decision"] == imp.ACCEPTED]
    decided = [r for r, s in zip(rows, states) if s == EVALUATED]
    wins = sum(r["status"] == imp.WIN for r in decided)
    pnl = [trade_pnl(r) for r in decided]
    gross_profit = sum(x for x in pnl if x > 0)
    gross_loss = -sum(x for x in pnl if x < 0)
    eq = peak = dd = 0.0
    for x in pnl:
        eq += x
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    reasons: dict[str, int] = {}
    for r, s in zip(rows, states):
        if s == DATA_UNAVAILABLE:
            text = describe_reason(data_reason(r)) or "unknown"
            reasons[text] = reasons.get(text, 0) + 1
    pending = states.count(PENDING)
    unavailable = states.count(DATA_UNAVAILABLE)
    return {
        "signals": len(rows), "accepted": len(accepted), "skipped": states.count(SKIPPED),
        "evaluated": len(decided), "wins": wins, "losses": len(decided) - wins,
        "ambiguous": sum(r["status"] == imp.AMBIGUOUS for r in decided),
        "scratches": sum(r["status"] == imp.SCRATCH for r in decided),
        "pending": pending, "data_unavailable": unavailable, "data_unavailable_reasons": reasons,
        "win_rate": wins / len(decided) if decided else None,
        "profit_factor": _pf(gross_profit, gross_loss) if decided else None,
        "gross_profit": round(gross_profit, 2), "gross_loss": round(gross_loss, 2),
        "net_pnl": round(sum(pnl), 2),
        "avg_pnl_per_trade": round(sum(pnl) / len(decided), 2) if decided else None,
        "max_drawdown": round(dd, 2),
        "pending_or_unavailable": pending + unavailable,
        "days": len({r["session_date"] for r in rows}),
    }


def report_candidates(rows: Sequence[dict]) -> list[str]:
    """Production S2 candidates, plus observation-only layers that actually have stored rows."""
    present = {r["candidate"] for r in rows}
    return [c.name for c in CANDIDATES] + [c for c in OBSERVATION_CANDIDATES if c in present]


def comparison(db: Database, day: str | None = None) -> dict[str, Any]:
    rows = db.get_shadow_signals(day)
    result = {}
    for candidate in report_candidates(rows):
        cr = [r for r in rows if r["candidate"] == candidate]
        result[candidate] = metrics(cr)
    return {"trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
            "candidates": result}


def cumulative(db: Database, validated_days: Sequence[str] = ()) -> dict[str, Any]:
    """Aggregate over every validated day; each signal exists once in the DB, so no day is counted twice."""
    rows = db.get_shadow_signals()
    days = sorted(set(validated_days) | {r["session_date"] for r in rows})
    result = {}
    for candidate in report_candidates(rows):
        cr = [r for r in rows if r["candidate"] == candidate]
        m = metrics(cr)
        daily = []
        for d in days:
            dm = metrics([r for r in cr if r["session_date"] == d])
            daily.append({"date": d, **{k: dm[k] for k in ("signals", "accepted", "evaluated", "wins", "losses",
                                                           "pending", "data_unavailable", "win_rate",
                                                           "profit_factor", "net_pnl", "max_drawdown")}})
        traded = [x for x in daily if x["evaluated"]]
        best = max(traded, key=lambda x: x["net_pnl"]) if traded else None
        worst = min(traded, key=lambda x: x["net_pnl"]) if traded else None
        m.update(
            days=len(days), trading_days=len(days),
            best_day=None if best is None else {"date": best["date"], "net_pnl": best["net_pnl"]},
            worst_day=None if worst is None else {"date": worst["date"], "net_pnl": worst["net_pnl"]},
            profitable_days=sum(x["net_pnl"] > 0 for x in daily),
            losing_days=sum(x["net_pnl"] < 0 for x in daily),
            average_daily_pnl=round(m["net_pnl"] / len(days), 2) if days else None,
            daily=daily,
        )
        result[candidate] = m
    return {"trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"), "validated_days": days,
            "candidates": result}


# ------------------------------------------------------------------ run
def history_gaps(spot: pd.DataFrame, day: date) -> list[str]:
    """Weekdays between the last stored session and ``day`` with no stored candles (holiday or missing parquet)."""
    if spot is None or spot.empty:
        return []
    stored = set(pd.to_datetime(spot["timestamp"]).dt.date)
    last = max(d for d in stored if d < day) if any(d < day for d in stored) else None
    if last is None:
        return []
    gaps, d = [], last + timedelta(days=1)
    while d < day:
        if d.weekday() < 5 and d not in stored:
            gaps.append(d.isoformat())
        d += timedelta(days=1)
    return gaps


def signal_view(row: dict) -> dict[str, Any]:
    state = signal_state(row)
    code = data_reason(row)
    return {
        "strategy": row["candidate"], "trading_date": row["session_date"], "signal_time": row["timestamp"],
        "direction": row["direction"], "instrument_key": row["instrument_key"], "expiry": row["expiry"],
        "dte": row["dte"], "decision": row["decision"], "state": state, "outcome": row["status"],
        "skip_reason": row["skip_reason"], "data_reason_code": code, "data_reason": describe_reason(code),
        "entry_price": row["entry_price"], "stop_loss": row["stop_loss"], "target": row["target"],
        "exit_price": row["exit_price"], "exit_time": row["exit_time"], "exit_reason": row["exit_reason"],
        "pnl": None if state != EVALUATED else round(trade_pnl(row), 2), "r_multiple": row["r_multiple"],
        "shadow_id": row["shadow_id"],
    }


S2_CONTEXT_FIELDS = ("signal_time", "session_date", "direction", "spot", "breakout_level", "atr14", "break_pct",
                     "break_atr", "time_bucket", "data_status", "reason", "contract", "instrument_key", "expiry",
                     "dte", "entry_time", "entry", "premium_atr")


def quality_gate_rows(contexts: Sequence[dict], baseline_rows: Sequence[dict], spot: pd.DataFrame | None,
                      config: qg.QualityGateConfig) -> list[dict[str, Any]]:
    """LIVE_QUALITY_GATE_V1 rows: the unchanged baseline S2 row when accepted, SKIPPED with exact reasons otherwise.

    Accepted rows reuse the baseline evaluation (same contract, entry, stop, target, first-touch exit), so the gate
    can only remove signals. S2 contexts are read, never modified.
    """
    out = []
    for ctx, base in zip(contexts, baseline_rows):
        decision = qg.evaluate_quality(qg.gate_context(ctx, spot), config)
        row = dict(base, variant=qg.GATE_CANDIDATE, quality_gate=decision.to_dict(),
                   s2_context={k: ctx.get(k) for k in S2_CONTEXT_FIELDS})
        if base["decision"] != imp.ACCEPTED:
            row["skip_reason"] = base["skip_reason"]
        elif not decision.accepted:
            row.update(decision="SKIPPED", skip_reason=decision.skip_reason,
                       **{k: None for k in dr._OUTCOME_KEYS})
        out.append(row)
    return out


def run_once(*, day: date, database: Database, prepare_inputs_fn,
             quality_gate: qg.QualityGateConfig | None = None) -> dict[str, Any]:
    """Shadow run for the production S2 candidates. ``quality_gate`` (off by default) adds LIVE_QUALITY_GATE_V1."""
    assert_paper_trading_only()
    hist_to = day - timedelta(days=1)
    hist_from = day - timedelta(days=HISTORY_DAYS)
    with order_placement_blocked(), capture_upstox_events() as events:
        inputs = prepare_inputs_fn(day=day, hist_from=hist_from, hist_to=hist_to, underlying="NIFTY",
                                   download_missing_options=False, prefer_stored_current_day=True)
        research = inputs["research"]
        pipeline = imp._pipeline_factory(research["signal_fn"], research["resolve"], research["params"],
                                         research["range_start"], research["range_end"])
        cur = pipeline(research["current_spot"], research["current_premium_for"], {day})
        rows_by_candidate = {
            c.name: dr.run_candidate(
                cur[0],
                cur[1],
                c,
                square_off=research["square_off"],
                lot_size=LOT_SIZE,
                cache={},
            )
            for c in CANDIDATES
        }

        if quality_gate is not None and quality_gate.enabled:
            rows_by_candidate[qg.GATE_CANDIDATE] = quality_gate_rows(
                cur[0], rows_by_candidate[dr.BASELINE.name], research["current_spot"], quality_gate)

        saved = save_rows(database, rows_by_candidate)
    data = dict(inputs.get("data") or {})
    data.update(
        history_gaps_unverified=history_gaps(research.get("hist_spot"), day),
        upstox_events=events,
        upstox_final_errors=sum(e["event"] == "upstox_historical_error" for e in events),
        upstox_retries=sum(e["event"] == "upstox_historical_retry" for e in events),
    )
    return {"date": day.isoformat(), "status": "VALIDATED",
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
            "created": {k: v["created"] for k, v in saved.items()}, "reconciliation": saved,
            "comparison": comparison(database, day.isoformat()),
            "point_in_time_matches_full_run": inputs["point_in_time_matches_full_run"],
            "data_status": data,
            "signals": [signal_view(r) for r in database.get_shadow_signals(day.isoformat())]}


# ------------------------------------------------------------------ reports
SIGNAL_FIELDS = ("strategy", "trading_date", "signal_time", "direction", "instrument_key", "expiry", "dte",
                 "decision", "state", "outcome", "skip_reason", "data_reason_code", "data_reason", "entry_price",
                 "stop_loss", "target", "exit_price", "exit_time", "exit_reason", "pnl", "r_multiple", "shadow_id")
SUMMARY_FIELDS = ("strategy", "trading_days", "signals", "evaluated", "wins", "losses", "pending",
                  "data_unavailable", "skipped", "win_rate", "profit_factor", "net_pnl", "avg_pnl_per_trade",
                  "max_drawdown", "best_day", "best_day_pnl", "worst_day", "worst_day_pnl", "profitable_days",
                  "losing_days", "average_daily_pnl")
DAILY_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "INF"
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _fmt_wr(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def _fmt_pf(v: float | None) -> str:
    return "N/A" if v is None else ("INF" if math.isinf(v) else f"{v:.2f}")


def _fmt_rs(v: float | None) -> str:
    return "N/A" if v is None else f"Rs {v:,.2f}"


def _fmt_day(d: dict | None) -> str:
    return "N/A" if d is None else f"{d['date']} ({_fmt_rs(d['net_pnl'])})"


def _table(candidates: dict[str, dict]) -> list[str]:
    out = ["| Strategy | Days | Signals | Evaluated | Wins | Losses | Pending | Data unavailable | WR | PF | "
           "P&L (Rs) | Max DD (Rs) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in candidates.items():
        out.append(f"| {name} | {m['days']} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | "
                   f"{m['pending']} | {m['data_unavailable']} | {_fmt_wr(m['win_rate'])} | "
                   f"{_fmt_pf(m['profit_factor'])} | {m['net_pnl']:,.2f} | {m['max_drawdown']:,.2f} |")
    return out


def _unavailable_lines(candidates: dict[str, dict]) -> list[str]:
    out = []
    for name, m in candidates.items():
        reasons = ", ".join(f"{k}: {n}" for k, n in m["data_unavailable_reasons"].items()) or "none"
        out.append(f"- {name}: {m['data_unavailable']} DATA_UNAVAILABLE, {m['pending']} PENDING ({reasons})")
    return out


SAFETY_LINE = "TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, observation only - no orders were placed."
CURRENT_HEADING = "CURRENT-DAY SHADOW VALIDATION"


def daily_markdown(result: dict[str, Any]) -> str:
    data = result.get("data_status", {})
    c = result["comparison"]["candidates"]
    md = [f"# {CURRENT_HEADING} - {result['date']}", "", SAFETY_LINE, "",
          "## Comparison", ""] + _table(c)
    md += ["", "## Data status", "",
           f"- Current-day NIFTY source: {data.get('current_day_source', 'UNKNOWN')}",
           f"- Stored history window: {data.get('historical_window', 'UNKNOWN')}",
           f"- Weekdays without stored history (exchange holiday or missing parquet): "
           f"{', '.join(data.get('history_gaps_unverified') or []) or 'none'}",
           f"- Upstox final errors: {data.get('upstox_final_errors', 0)}, retried requests: "
           f"{data.get('upstox_retries', 0)}",
           f"- Point-in-time signals equal full-day run: {result.get('point_in_time_matches_full_run')}", ""]
    md += _unavailable_lines(c)
    for e in data.get("upstox_events") or []:
        md.append(f"- {e['event']}: path={e.get('path')} code={e.get('code')} http={e.get('status')} "
                  f"attempt={e.get('attempt')}")
    md += ["", "## Signals", "", "| Strategy | Time | Dir | State | Reason | Entry | Exit | P&L |",
           "|---|---|---|---|---|---|---|---|"]
    for s in result.get("signals") or []:
        reason = s["skip_reason"] if s["state"] == SKIPPED else (s["data_reason"] or s["exit_reason"] or "")
        md.append(f"| {s['strategy']} | {s['signal_time'][11:16]} | {s['direction']} | {s['state']} | {reason} | "
                  f"{s['entry_price'] if s['entry_price'] is not None else ''} | "
                  f"{s['exit_price'] if s['exit_price'] is not None else ''} | "
                  f"{s['pnl'] if s['pnl'] is not None else ''} |")
    return "\n".join(md) + "\n"


def write_report(result: dict[str, Any], out_dir: Path) -> Path:
    """Daily JSON / MD / CSV; overwriting the same date reconciles that day instead of appending."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['date']}.json"
    path.write_text(json.dumps(_jsonable(result), indent=2, default=str), encoding="utf-8")
    (out_dir / f"{result['date']}.md").write_text(daily_markdown(result), encoding="utf-8")
    with (out_dir / f"{result['date']}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SIGNAL_FIELDS)
        writer.writeheader()
        for s in result.get("signals") or []:
            writer.writerow({k: s.get(k) for k in SIGNAL_FIELDS})
    return path


def validated_days(out_dir: Path) -> list[str]:
    days = []
    for p in sorted(out_dir.glob("*.json")) if out_dir.is_dir() else []:
        if not DAILY_FILE.match(p.name):
            continue
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if body.get("status", "VALIDATED") == "VALIDATED" and body.get("date") == p.stem:
            days.append(p.stem)
    return days


def cumulative_markdown(cum: dict[str, Any]) -> str:
    c = cum["candidates"]
    md = [f"# {CURRENT_HEADING} - cumulative", "", SAFETY_LINE, "",
          "Current-market forward validation (real intraday candles). Kept separate from the HISTORICAL SHADOW "
          "REPLAY; the two are never merged.", "",
          f"Validated trading days: {', '.join(cum['validated_days']) or 'none'}", "",
          "## Comparison", "",
          "| Strategy | Days | Signals | Evaluated | Wins | Losses | WR | PF | P&L (Rs) | Max DD (Rs) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in c.items():
        md.append(f"| {name} | {m['trading_days']} | {m['signals']} | {m['evaluated']} | {m['wins']} | "
                  f"{m['losses']} | {_fmt_wr(m['win_rate'])} | {_fmt_pf(m['profit_factor'])} | "
                  f"{m['net_pnl']:,.2f} | {m['max_drawdown']:,.2f} |")
    md += ["", "## Data status", ""] + _unavailable_lines(c)
    for name, m in c.items():
        best, worst = m["best_day"], m["worst_day"]
        md += ["", f"## {name}", "",
               f"- Trading days: {m['trading_days']} (profitable {m['profitable_days']}, losing {m['losing_days']})",
               f"- Signals {m['signals']}, evaluated {m['evaluated']}, wins {m['wins']}, losses {m['losses']}, "
               f"pending {m['pending']}, data unavailable {m['data_unavailable']}, skipped {m['skipped']}",
               f"- Win rate {_fmt_wr(m['win_rate'])}, profit factor {_fmt_pf(m['profit_factor'])}",
               f"- Total P&L Rs {m['net_pnl']:,.2f}, average per trade {_fmt_rs(m['avg_pnl_per_trade'])}, "
               f"average per day {_fmt_rs(m['average_daily_pnl'])}",
               f"- Max drawdown Rs {m['max_drawdown']:,.2f}",
               f"- Best day: {_fmt_day(best)}; worst day: {_fmt_day(worst)}",
               "", "| Date | Signals | Evaluated | Wins | Losses | Pending | Data unavailable | P&L (Rs) | Max DD (Rs) |",
               "|---|---|---|---|---|---|---|---|---|"]
        for d in m["daily"]:
            md.append(f"| {d['date']} | {d['signals']} | {d['evaluated']} | {d['wins']} | {d['losses']} | "
                      f"{d['pending']} | {d['data_unavailable']} | {d['net_pnl']:,.2f} | {d['max_drawdown']:,.2f} |")
    return "\n".join(md) + "\n"


def write_cumulative(database: Database, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cum = cumulative(database, validated_days(out_dir))
    (out_dir / "cumulative.json").write_text(json.dumps(_jsonable(cum), indent=2, default=str), encoding="utf-8")
    (out_dir / "cumulative.md").write_text(cumulative_markdown(cum), encoding="utf-8")
    with (out_dir / "cumulative.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for name, m in cum["candidates"].items():
            row = {k: m.get(k) for k in SUMMARY_FIELDS}
            row.update(strategy=name, profit_factor=_jsonable(m["profit_factor"]),
                       best_day=(m["best_day"] or {}).get("date"), best_day_pnl=(m["best_day"] or {}).get("net_pnl"),
                       worst_day=(m["worst_day"] or {}).get("date"),
                       worst_day_pnl=(m["worst_day"] or {}).get("net_pnl"))
            writer.writerow(row)
    return cum
