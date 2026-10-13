"""Read-only PAPER signal-validation view for the existing dashboard.

Single source of truth stays with the two existing trackers; nothing here generates
a signal or decides an outcome:
* option-premium tracker (``scripts/run_live_premium_signal_accuracy.py``) ->
  ``live_signal_outcomes`` rows + latest completed premium candle per tracked signal
  (``system_state`` keys ``live_signal_mark:<signal_id>``);
* realtime NIFTY spot tracker (``scripts/run_live_signal_accuracy.py``) -> append-only
  ``session_*.jsonl`` audit logs + the ``live_status.json`` heartbeat.

A signal recorded by both trackers (same S2 candle and direction) is shown once:
the option-premium record decides its levels / status, the spot record adds NIFTY levels.
Simulated P&L uses option-premium records only (gross, 1 lot, costs excluded).

Like the trackers, this module must not import ``src.paper_trading`` (that package
imports the broker / engine / order manager as a side effect).
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.backtest.research_strategies import enrich_s2
from src.signal_accuracy.realtime import DEFAULT_OUT as DEFAULT_SPOT_DIR, STATUS_FILE
from src.signal_accuracy.tracker import ACTIVE, IST, SessionStore, SignalEvent, to_ist

PENDING, WIN, LOSS, AMBIGUOUS, UNRESOLVED = "PENDING", "WIN", "LOSS", "AMBIGUOUS", "UNRESOLVED"
STATUSES = (PENDING, WIN, LOSS, AMBIGUOUS, UNRESOLVED)
MEANINGFUL_SAMPLE = 100
LOW_SAMPLE = 30
HEARTBEAT_TIMEOUT_SECONDS = 15
MARK_PREFIX = "live_signal_mark:"
SOURCE_PREMIUM, SOURCE_SPOT = "OPTION_PREMIUM_TRACKER", "NIFTY_SPOT_TRACKER"
BASIS_PREMIUM, BASIS_SPOT = "OPTION_PREMIUM", "NIFTY_SPOT"
CONNECTED, DISCONNECTED, STALE = "CONNECTED", "DISCONNECTED", "STALE"
RUNNING, STOPPED = "RUNNING", "STOPPED"
DEGRADED = "DEGRADED"
STALE_THRESHOLD_SECONDS = 15
PREMIUM_DISCONNECT_SECONDS = 60
PREMIUM_STATUS_FILE = "premium_status.json"
DEFAULT_PREMIUM_DIR = Path(__file__).resolve().parents[2] / "reports" / "live_signal_accuracy"
BROKER_PAPER_LABEL = "NOT CONNECTED — PAPER SIGNAL MODE"
BANNER = ("PAPER MODE", "SIGNAL VALIDATION ONLY", "NO REAL ORDERS ARE BEING PLACED")
NOTES = (
    "Accuracy = wins / (wins + losses). Pending, ambiguous and unresolved signals are excluded.",
    f"Today's live PAPER session uses provisional accuracy; fewer than {LOW_SAMPLE} resolved trades are labelled "
    f"PROVISIONAL — LOW SAMPLE. Stored prior sessions use historical accuracy. Neither is live-trading profit.",
    "Simulated P&L = option premium move x 1 lot from the premium tracker (gross, costs and slippage excluded). "
    "It is not an account balance.",
    "Signals come only from the backend trackers; the dashboard never calculates a signal.",
)


@dataclass(frozen=True)
class DashboardConfig:
    market_open: time = time(9, 15)
    market_close: time = time(15, 30)
    stale_data_seconds: int = 15
    s2: Mapping[str, Any] | None = None

    @classmethod
    def load(cls) -> "DashboardConfig":
        from src.config.research_loader import load_research_config
        from src.signal_accuracy.realtime import RESEARCH_CONFIG, TrackerSettings

        s = TrackerSettings.load()
        research = load_research_config(RESEARCH_CONFIG)
        return cls(market_open=time.fromisoformat(s.market_open), market_close=time.fromisoformat(s.market_close),
                   stale_data_seconds=s.stale_data_seconds, s2=dict(research.strategies[s.strategy_id]))


# ------------------------------------------------------------------ sources (read-only)
class SpotSource:
    """Realtime spot tracker output; each JSONL log is re-read only when its size / mtime changes."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self._cache: dict[Path, tuple[tuple[int, int], dict[str, SignalEvent], list[dict[str, Any]]]] = {}

    def _files(self) -> list[Path]:
        return sorted(self.out_dir.glob("session_*.jsonl")) if self.out_dir.is_dir() else []

    def fingerprint(self) -> tuple:
        out = []
        for p in [*self._files(), self.out_dir / STATUS_FILE]:
            try:
                st = p.stat()
            except OSError:
                continue
            out.append((p.name, st.st_mtime_ns, st.st_size))
        return tuple(out)

    def load(self) -> tuple[list[SignalEvent], list[dict[str, Any]]]:
        signals: dict[str, SignalEvent] = {}
        candles: dict[str, dict[str, Any]] = {}
        for p in self._files():
            try:
                st = p.stat()
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            cached = self._cache.get(p)
            if cached is None or cached[0] != sig:
                s, c = SessionStore(self.out_dir, p.stem[len("session_"):]).load()
                cached = (sig, s, c)
                self._cache[p] = cached
            signals.update(cached[1])
            for raw in cached[2]:
                candles[to_ist(raw["timestamp"]).isoformat()] = raw
        return list(signals.values()), [candles[k] for k in sorted(candles)]

    def status(self) -> dict[str, Any] | None:
        try:
            data = json.loads((self.out_dir / STATUS_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


class PremiumSource:
    """Option-premium tracker heartbeat (``reports/live_signal_accuracy/premium_status.json``)."""

    def __init__(self, out_dir: str | Path = DEFAULT_PREMIUM_DIR):
        self.path = Path(out_dir) / PREMIUM_STATUS_FILE

    def status(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None


# ------------------------------------------------------------------ unified signal rows
def _iso(ts: Any) -> str | None:
    return to_ist(pd.Timestamp(ts).to_pydatetime()).isoformat() if ts else None


def _key(ts: Any, direction: str) -> str:
    return f"{to_ist(pd.Timestamp(ts).to_pydatetime()).isoformat(timespec='seconds')}|{direction.upper()}"


def spot_status(ev: SignalEvent) -> str:
    if ev.status in ACTIVE:
        return PENDING
    if ev.outcome in (WIN, LOSS):
        return ev.outcome
    return AMBIGUOUS if ev.status == AMBIGUOUS else UNRESOLVED


def premium_status(status: str) -> str:
    return status if status in (PENDING, WIN, LOSS, AMBIGUOUS) else UNRESOLVED


def _spot_levels(ev: SignalEvent) -> dict[str, Any]:
    return {"status": spot_status(ev), "tracker_status": ev.status, "entry": ev.entry_price, "stop_loss": ev.stop_price,
            "target": ev.target_price, "current": ev.last_price, "current_time": ev.last_price_time,
            "entry_time": ev.entry_time, "exit_price": ev.exit_price, "pnl_points": ev.pnl_points}


def spot_row(ev: SignalEvent) -> dict[str, Any]:
    lv = _spot_levels(ev)
    return {
        "signal_id": ev.signal_id, "key": _key(ev.timestamp_ist, ev.direction), "time": _iso(ev.timestamp_ist),
        "session_date": ev.session_date, "strategy": ev.strategy_name, "underlying": ev.instrument,
        "direction": ev.direction, "basis": BASIS_SPOT, "sources": [SOURCE_SPOT],
        "option": None, "option_type": None, "strike": None, "expiry": None, "quantity": None,
        "signal_price": ev.signal_price, "entry": lv["entry"], "stop_loss": lv["stop_loss"], "target": lv["target"],
        "current": lv["current"], "current_time": lv["current_time"], "entry_time": lv["entry_time"],
        "status": lv["status"], "reason": ev.reason, "exit_price": ev.exit_price,
        "exit_time": ev.outcome_timestamp if lv["status"] != PENDING else None,
        "pnl_rupees": None, "pnl_points": ev.pnl_points, "r_multiple": None, "spot": lv,
    }


def premium_row(row: Mapping[str, Any], mark: Mapping[str, Any] | None = None, *, now: datetime | None = None,
                cfg: DashboardConfig | None = None) -> dict[str, Any]:
    mark = mark or {}
    entry = row.get("entry_price")
    out = {
        "signal_id": row["signal_id"], "key": _key(row["timestamp"], row["direction"]), "time": _iso(row["timestamp"]),
        "session_date": row["session_date"], "strategy": row.get("strategy"), "underlying": row.get("underlying"),
        "direction": row["direction"], "basis": BASIS_PREMIUM, "sources": [SOURCE_PREMIUM],
        "option": row.get("trading_symbol") or row.get("instrument_key"), "option_type": row.get("option_type"),
        "strike": row.get("strike"), "expiry": row.get("expiry"), "quantity": row.get("quantity"),
        "signal_price": row.get("signal_price"), "entry": entry, "stop_loss": row.get("stop_loss"),
        "target": row.get("target"), "current": mark.get("price"),
        "current_time": mark.get("last_tick_time") or mark.get("candle_time"),
        "last_tick_time": mark.get("last_tick_time"), "last_completed_candle": mark.get("candle_time"),
        "entry_time": _iso(row.get("active_from")) if entry is not None else None,
        "status": premium_status(row["status"]), "reason": row.get("reason"), "exit_price": row.get("exit_price"),
        "exit_time": _iso(row.get("exit_time")), "pnl_rupees": row.get("pnl"), "pnl_points": None,
        "r_multiple": row.get("r_multiple"), "spot": None,
    }
    return apply_premium_freshness(out, now, cfg)


def apply_premium_freshness(row: Mapping[str, Any], now: datetime | None, cfg: DashboardConfig | None) -> dict[str, Any]:
    """Stale when the last option *tick* is older than the threshold (not the 5m candle bucket)."""
    out = dict(row)
    if out.get("basis") != BASIS_PREMIUM or out.get("status") != PENDING:
        out["premium_stale"] = False
        out["current_display"] = out.get("current")
        out["premium_freshness"] = CONNECTED
        return out
    tick_ts = out.get("last_tick_time") or out.get("current_time")
    age = _age(now, tick_ts) if now and cfg else None
    fresh = freshness(age, live=market_state(now, cfg) == "LIVE") if now and cfg else DISCONNECTED
    stale = fresh != CONNECTED
    out["premium_stale"] = stale
    out["premium_freshness"] = fresh
    cur = out.get("current")
    label = {CONNECTED: "FRESH", STALE: "STALE", DISCONNECTED: "STALE"}.get(fresh, "STALE")
    out["current_display"] = label if stale and cur is None else (f"{label} ({cur})" if stale else cur)
    return out


def merge_rows(premium: Iterable[dict[str, Any]], spot: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (S2 candle, direction), newest first; the option-premium record wins when both exist."""
    out: dict[str, dict[str, Any]] = {r["key"]: r for r in spot}
    for r in premium:
        other = out.get(r["key"])
        if other is not None:
            r = {**r, "sources": [SOURCE_PREMIUM, SOURCE_SPOT], "spot": other["spot"]}
        out[r["key"]] = r
    return sorted(out.values(), key=lambda r: (r["time"], r["direction"]), reverse=True)


# ------------------------------------------------------------------ summaries
def accuracy_label(summary: Mapping[str, Any], *, live: bool) -> str:
    """Display label only; accuracy math is unchanged."""
    resolved = int(summary.get("resolved") or 0)
    if not resolved:
        return "N/A"
    if live:
        return "PROVISIONAL — LOW SAMPLE" if resolved < LOW_SAMPLE else "PROVISIONAL ACCURACY"
    return "HISTORICAL ACCURACY"


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = Counter(r["status"] for r in rows)
    wins, losses = n[WIN], n[LOSS]
    resolved = wins + losses
    decided = sorted((r for r in rows if r["status"] in (WIN, LOSS) and r.get("pnl_rupees") is not None),
                     key=lambda r: (r.get("exit_time") or r["time"], r["time"]))
    pnls = [float(r["pnl_rupees"]) for r in decided]
    gp, gl = sum(p for p in pnls if p > 0), -sum(p for p in pnls if p < 0)
    equity = peak = dd = 0.0
    for p in pnls:
        equity += p
        peak, dd = max(peak, equity), max(dd, peak - equity)
    return {
        "total": len(rows), "resolved": resolved, "pending": n[PENDING], "wins": wins, "losses": losses,
        "ambiguous": n[AMBIGUOUS], "unresolved": n[UNRESOLVED],
        "accuracy": round(100.0 * wins / resolved, 2) if resolved else None,
        "accuracy_status": "N/A" if not resolved else "PROVISIONAL" if resolved < MEANINGFUL_SAMPLE else "OBSERVED",
        "meaningful_sample": MEANINGFUL_SAMPLE, "low_sample": LOW_SAMPLE,
        "simulated_pnl": round(gp - gl, 2) if pnls else None,
        "max_drawdown": round(dd, 2) if pnls else None,
        "profit_factor": round(gp / gl, 4) if gl > 0 else None,
        "gross_profit": round(gp, 2), "gross_loss": round(gl, 2), "pnl_signals": len(pnls),
        "pnl_label": "SIMULATED (option premium x 1 lot, gross, costs excluded)",
    }


def daily_summaries(rows: Iterable[Mapping[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    by_day: dict[str, list] = {}
    for r in rows:
        by_day.setdefault(r["session_date"], []).append(r)
    out = []
    for d in sorted(by_day, reverse=True)[:limit]:
        s = summarize_rows(by_day[d])
        s["accuracy_label"] = accuracy_label(s, live=False)
        out.append({"date": d, **s})
    return out


def active_signal(rows: Iterable[Mapping[str, Any]], marks: Mapping[str, Mapping[str, Any]], *,
                  now: datetime | None = None, cfg: DashboardConfig | None = None) -> dict[str, Any] | None:
    pending = [r for r in rows if r["status"] == PENDING]
    if not pending:
        return None
    r = dict(max(pending, key=lambda x: x["time"]))
    sign = -1 if r["basis"] == BASIS_SPOT and r["direction"] == "SELL" else 1
    cur, sl, tgt = r.get("current"), r.get("stop_loss"), r.get("target")
    stale = bool(r.get("premium_stale"))
    ok = r.get("entry") is not None and None not in (cur, sl, tgt) and not stale
    r.update(
        awaiting_entry=r.get("entry") is None,
        distance_to_sl=round((cur - sl) * sign, 2) if ok else None,
        distance_to_target=round((tgt - cur) * sign, 2) if ok else None,
        unit="option premium (Rs)" if r["basis"] == BASIS_PREMIUM else "NIFTY points",
        premium_history=list((marks.get(r["signal_id"]) or {}).get("history", [])) if r["basis"] == BASIS_PREMIUM
        else [],
        current_display=r.get("current_display", cur),
        premium_stale=stale,
    )
    return r


# ------------------------------------------------------------------ market / health
def market_state(now: datetime, cfg: DashboardConfig) -> str:
    now = to_ist(now)
    return "LIVE" if now.weekday() < 5 and cfg.market_open <= now.time() < cfg.market_close else "CLOSED"


def _age(now: datetime, ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return round((to_ist(now) - to_ist(ts)).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def _latest_ts(stamps: Iterable[str | None]) -> str | None:
    vals = [s for s in stamps if s]
    return _iso(max(vals, key=lambda s: pd.Timestamp(s))) if vals else None


def premium_times(marks: Iterable[Mapping[str, Any]]) -> tuple[str | None, str | None, str | None]:
    """Last option tick, last completed premium candle, last tracker write."""
    return (_latest_ts(m.get("last_tick_time") for m in marks),
            _latest_ts(m.get("candle_time") for m in marks),
            _latest_ts(m.get("updated_at") for m in marks))


def map_subscription_status(raw: str | None, *, tracker_running: bool) -> str:
    """Map premium tracker subscription state to dashboard CONNECTED/STALE/DISCONNECTED."""
    if not tracker_running:
        return DISCONNECTED
    if raw in {None, "", "REST_ONLY", "FAILED"}:
        return DISCONNECTED
    if raw in {"CONNECTED", "IDLE", "SUBSCRIBED"}:
        return CONNECTED
    if raw == "RECONNECTING":
        return STALE
    return DISCONNECTED if raw == "PENDING" else str(raw)


def freshness(age: float | None, *, live: bool) -> str:
    """CONNECTED (fresh <=15s), STALE (>15s), DISCONNECTED (>60s or missing when live)."""
    if age is None:
        return DISCONNECTED if live else DISCONNECTED
    if age <= STALE_THRESHOLD_SECONDS:
        return CONNECTED
    if age <= PREMIUM_DISCONNECT_SECONDS:
        return STALE
    return DISCONNECTED


def broker_execution_label(safety: Mapping[str, Any] | None) -> str:
    safety = safety or {}
    if safety.get("trading_mode") == "PAPER" or safety.get("paper_only") or safety.get("signal_generation_only"):
        return BROKER_PAPER_LABEL
    return DISCONNECTED


def health(status: Mapping[str, Any] | None, now: datetime, cfg: DashboardConfig, marks: Iterable[Mapping[str, Any]],
           premium_status: Mapping[str, Any] | None = None, safety: Mapping[str, Any] | None = None) -> dict[str, Any]:
    status, premium_status = status or {}, premium_status or {}
    live = market_state(now, cfg) == "LIVE"

    hb_age = _age(now, status.get("updated_at"))
    engine_running = bool(status.get("running")) and hb_age is not None and hb_age <= HEARTBEAT_TIMEOUT_SECONDS
    feed = status.get("market_data_status") if engine_running else DISCONNECTED
    ws = CONNECTED if engine_running and feed in ("CONNECTED", "STALE") else DISCONNECTED
    stale_after = int(status.get("stale_data_seconds") or cfg.stale_data_seconds)
    data_age = _age(now, status.get("last_tick_time"))
    spot_stale = live and (data_age is None or data_age > stale_after or bool(status.get("in_data_gap")))
    market_data = CONNECTED if ws == CONNECTED and (feed == "CONNECTED" or not live) and not spot_stale else DISCONNECTED

    prem_hb_age = _age(now, premium_status.get("updated_at"))
    prem_tracker_running = bool(premium_status.get("running")) and prem_hb_age is not None \
        and prem_hb_age <= HEARTBEAT_TIMEOUT_SECONDS
    last_tick, last_candle, last_write = premium_times(marks)
    tick_ref = last_tick or premium_status.get("last_option_tick_time")
    prem_tick_age = _age(now, tick_ref)
    prem_candle_age = _age(now, last_candle)
    option_premium_feed = freshness(prem_tick_age, live=live)
    subscription = map_subscription_status(premium_status.get("subscription_status"),
                                           tracker_running=prem_tracker_running)

    signal_engine = RUNNING if engine_running else STOPPED
    premium_tracker = RUNNING if prem_tracker_running else STOPPED
    broker_orders = max(int(status.get("orders_placed") or 0), int(premium_status.get("orders_placed") or 0))

    if signal_engine == RUNNING and option_premium_feed == CONNECTED:
        overall = RUNNING
    elif signal_engine == RUNNING:
        overall = DEGRADED
    else:
        overall = STOPPED

    candle = status.get("last_completed_candle") or {}
    return {
        "market_data": market_data, "websocket": ws, "feed_status": feed or DISCONNECTED,
        "signal_engine": signal_engine, "premium_tracker": premium_tracker,
        "premium_data": option_premium_feed, "option_premium_feed": option_premium_feed,
        "premium_tick_age_seconds": prem_tick_age, "premium_data_age_seconds": prem_tick_age,
        "premium_candle_age_seconds": prem_candle_age,
        "premium_last_tick": tick_ref, "premium_last_tick_price": premium_status.get("last_option_tick_price"),
        "premium_last_candle": last_candle,
        "premium_subscription": subscription,
        "subscribed_instruments": premium_status.get("subscribed_instruments") or [],
        "premium_tracker_last_update": premium_status.get("updated_at") or last_write,
        "broker_execution": broker_execution_label(safety), "broker_orders": broker_orders,
        "last_tick": status.get("last_tick_time"), "last_tick_price": status.get("last_tick_price"),
        "last_completed_candle": candle.get("timestamp"), "data_age_seconds": data_age, "stale": spot_stale,
        "stale_after_seconds": stale_after, "engine_heartbeat_age_seconds": hb_age,
        "premium_heartbeat_age_seconds": prem_hb_age, "engine_session_id": status.get("session_id"),
        "system": signal_engine, "overall_status": overall,
    }


# ------------------------------------------------------------------ chart (observed values only)
def chart(candles: list[Mapping[str, Any]], rows: list[Mapping[str, Any]], active: Mapping[str, Any] | None,
          cfg: DashboardConfig) -> dict[str, Any]:
    bars = [{"time": _iso(c["timestamp"]), **{k: float(c[k]) for k in ("open", "high", "low", "close")}}
            for c in candles]
    orb = None
    s2 = dict(cfg.s2 or {})
    if bars and s2.get("range_start") and s2.get("range_end"):
        end = time.fromisoformat(s2["range_end"])
        if any(to_ist(b["time"]).time() >= end for b in bars):
            frame = pd.DataFrame(bars).rename(columns={"time": "timestamp"})
            frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_convert(IST)
            last = enrich_s2(frame, {"atr_period": 14, **s2}).iloc[-1]
            if pd.notna(last["orb_high"]):
                orb = {"high": float(last["orb_high"]), "low": float(last["orb_low"]),
                       "label": f"S2 opening range {s2['range_start']}-{s2['range_end']}"}
    markers = [{"time": r["time"], "direction": r["direction"], "price": r["signal_price"], "status": r["status"]}
               for r in rows if r.get("signal_price") is not None]
    levels = premium = None
    if active:
        sp = active.get("spot") or {}
        if sp.get("entry") is not None:
            levels = {k: sp.get(k) for k in ("entry", "stop_loss", "target")}
        if active["basis"] == BASIS_PREMIUM:
            premium = {"option": active.get("option"), "history": active.get("premium_history") or [],
                       **{k: active.get(k) for k in ("entry", "stop_loss", "target")}}
    return {"candles": bars, "opening_range": orb, "vwap": None,
            "vwap_note": "Not shown: the frozen S2 strategy (opening range breakout) does not use VWAP.",
            "markers": markers, "levels": levels, "premium": premium}


# ------------------------------------------------------------------ snapshot
def build_snapshot(*, premium_rows: list[Mapping[str, Any]], marks: list[Mapping[str, Any]],
                   spot_signals: list[SignalEvent], spot_candles: list[Mapping[str, Any]],
                   status: Mapping[str, Any] | None, premium_status: Mapping[str, Any] | None,
                   now: datetime, cfg: DashboardConfig, safety: Mapping[str, Any], day: str | None = None) -> dict[str, Any]:
    now = to_ist(now)
    day = day or now.date().isoformat()
    by_id = {m["signal_id"]: m for m in marks if m.get("signal_id")}
    rows = merge_rows([premium_row(r, by_id.get(r["signal_id"]), now=now, cfg=cfg) for r in premium_rows],
                      [spot_row(e) for e in spot_signals])
    today = [r for r in rows if r["session_date"] == day]
    active = active_signal(today, by_id, now=now, cfg=cfg) if day == now.date().isoformat() else None
    hl = health(status, now, cfg, marks, premium_status=premium_status, safety=safety)
    day_candles = [c for c in spot_candles if to_ist(c["timestamp"]).date().isoformat() == day]
    last = (status or {}).get("last_completed_candle") or (
        {**day_candles[-1], "timestamp": _iso(day_candles[-1]["timestamp"])} if day_candles else None)
    return {
        "generated_at": now.isoformat(timespec="seconds"), "date": day, "mode": safety.get("trading_mode", "PAPER"),
        "banner": list(BANNER), "labels": ["PAPER", "SIGNAL VALIDATION", "PROVISIONAL ACCURACY"],
        "notes": list(NOTES), "safety": dict(safety),
        "market": {"state": market_state(now, cfg),
                   "nifty": {"price": hl["last_tick_price"], "time": hl["last_tick"]}, "latest_candle": last},
        "summary": {**(_today := summarize_rows(today)), "accuracy_label": accuracy_label(_today, live=True)},
        "cumulative": {**(_all := summarize_rows(rows)), "accuracy_label": accuracy_label(_all, live=False)},
        "signals": today,
        "active": active, "health": hl, "chart": chart(day_candles, today, active, cfg),
        "daily": daily_summaries(rows),
    }


VOLATILE = {"market": ("nifty",), "health": ("data_age_seconds", "last_tick", "last_tick_price",
                                              "engine_heartbeat_age_seconds", "premium_data_age_seconds",
                                              "premium_heartbeat_age_seconds")}


def stable_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot without per-tick fields: a change here means a new candle / signal / premium / outcome / health."""
    out = copy.deepcopy(dict(snapshot))
    out.pop("generated_at", None)
    for section, keys in VOLATILE.items():
        for k in keys:
            (out.get(section) or {}).pop(k, None)
    return out


def tick_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    h = snapshot["health"]
    return {"generated_at": snapshot["generated_at"], "nifty": snapshot["market"]["nifty"],
            "data_age_seconds": h["data_age_seconds"], "last_tick": h["last_tick"]}


# ------------------------------------------------------------------ realtime push (SSE)
def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, separators=(',', ':'))}\n\n"


KEEPALIVE = ": keepalive\n\n"


class StreamDiff:
    """Successive snapshots -> SSE frames: full ``snapshot`` on a candle / signal / premium / outcome / health
    change, a small ``tick`` when only the NIFTY tick moved, and a keep-alive comment when nothing changed."""

    def __init__(self, keepalive_after: int = 15):
        self.keepalive_after = max(1, int(keepalive_after))
        self._stable: dict[str, Any] | None = None
        self._tick: tuple | None = None
        self._idle = 0

    def frames(self, snapshot: Mapping[str, Any]) -> list[str]:
        stable = stable_view(snapshot)
        tick = tick_view(snapshot)
        tick_key = (tick["last_tick"], (tick["nifty"] or {}).get("price"))
        if stable != self._stable:
            self._stable, self._tick, self._idle = stable, tick_key, 0
            return [sse("snapshot", snapshot)]
        if tick_key != self._tick:
            self._tick, self._idle = tick_key, 0
            return [sse("tick", tick)]
        self._idle += 1
        if self._idle >= self.keepalive_after:
            self._idle = 0
            return [KEEPALIVE]
        return []
