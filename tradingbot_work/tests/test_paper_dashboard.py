"""Live PAPER signal-validation dashboard: read-only view over the existing trackers."""
from __future__ import annotations

import ast
import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.database import Database
from src.paper_trading.signal_tracker import MARK_PREFIX
from src.signal_accuracy import paper_dashboard as pd_
from src.signal_accuracy.realtime import STATUS_FILE, write_status
from src.signal_accuracy.tracker import SessionStore, SignalEvent

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 10, 7)
PREV = date(2026, 10, 6)
NOW = datetime(2026, 10, 7, 11, 0, tzinfo=IST)
CFG = pd_.DashboardConfig()
SAFETY = {"trading_mode": "PAPER", "live_trading_enabled": False}


def at(hh, mm, day=DAY, ss=0):
    return datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=IST)


def row(status, pnl=None, hh=10, mm=0, direction="BUY", day=DAY, **kw):
    """Unified dashboard row as produced by premium_row / spot_row."""
    return {"signal_id": f"{day}{hh}{mm}{direction}", "time": at(hh, mm, day).isoformat(), "session_date": day.isoformat(),
            "direction": direction, "status": status, "pnl_rupees": pnl, "exit_time": None, **kw}


def save_premium(db, sid, hh, mm, direction="BUY", status=None, pnl=None, day=DAY, entry=100.0):
    ts = at(hh, mm, day)
    db.save_live_signal({
        "signal_id": sid, "timestamp": ts.isoformat(), "session_date": day.isoformat(), "strategy": "S2",
        "underlying": "NIFTY", "direction": direction, "option_type": "CE" if direction == "BUY" else "PE",
        "strike": 25000.0, "expiry": "2026-10-09", "instrument_key": "NSE_FO|TEST",
        "trading_symbol": f"NIFTY 25000 {'CE' if direction == 'BUY' else 'PE'}", "signal_price": 25010.0,
        "entry_price": entry, "stop_loss": 90.0, "target": 120.0, "quantity": 75,
        "active_from": (ts + timedelta(minutes=5)).isoformat(), "source": "TEST", "record_json": "{}",
        "created_at": ts.isoformat(), "status": "PENDING",
    })
    if status:
        exit_ts = ts + timedelta(minutes=30)
        db.resolve_live_signal(sid, {"status": status, "exit_price": 120.0 if status == "WIN" else 90.0,
                                     "exit_time": exit_ts.isoformat(), "pnl": pnl, "r_multiple": None,
                                     "resolved_at": exit_ts.isoformat(), "reason": status})


def spot_event(hh, mm, direction="BUY", status="TRACKING", outcome=None, day=DAY):
    ts = at(hh, mm, day)
    return SignalEvent(
        session_id=day.strftime("%Y%m%d"), signal_id=f"spot-{hh}{mm}{direction}", timestamp_ist=ts.isoformat(),
        instrument="NIFTY", timeframe="5m", strategy_name="S2", direction=direction, signal_price=25010.0,
        candle_open=25000.0, candle_high=25020.0, candle_low=24990.0, candle_close=25010.0, atr=20.0,
        stop_distance=20.0, entry_timing="NEXT_CANDLE_OPEN", active_from=(ts + timedelta(minutes=5)).isoformat(),
        entry_price=25012.0, entry_time=(ts + timedelta(minutes=5)).isoformat(), stop_price=24992.0,
        target_price=25052.0, status=status, outcome=outcome, last_price=25030.0,
        last_price_time=(ts + timedelta(minutes=7)).isoformat(),
    )


def heartbeat(now=NOW, *, running=True, feed="CONNECTED", tick_age=1.0, orders=0, gap=False):
    return {"running": running, "updated_at": now.isoformat(), "market_data_status": feed, "in_data_gap": gap,
            "last_tick_time": (now - timedelta(seconds=tick_age)).isoformat(), "last_tick_price": 25031.5,
            "stale_data_seconds": 15, "orders_placed": orders, "session_id": "20261007",
            "last_completed_candle": {"timestamp": at(10, 55).isoformat(), "open": 1, "high": 2, "low": 0.5,
                                      "close": 1.5, "volume": 0}}


def snapshot(db, *, spot=(), candles=(), status=None, premium_status=None, now=NOW, day=None):
    return pd_.build_snapshot(premium_rows=db.get_live_signals(), marks=db.get_states(MARK_PREFIX),
                              spot_signals=list(spot), spot_candles=list(candles), status=status,
                              premium_status=premium_status, now=now, cfg=CFG, safety=SAFETY, day=day)


def premium_heartbeat(running=True, age=1.0, now=NOW):
    return {"running": running, "updated_at": (now - timedelta(seconds=age)).isoformat(), "orders_placed": 0}


def mark(db, signal_id, candle_time, price=100.0, updated=None, tick_time=None):
    tick_time = tick_time or candle_time
    db.set_state(f"{MARK_PREFIX}{signal_id}", {
        "signal_id": signal_id, "price": price, "candle_time": candle_time,
        "last_tick_time": tick_time, "last_tick_price": price,
        "updated_at": updated or tick_time, "history": [{"time": candle_time, "close": price}],
    })


@pytest.fixture()
def db():
    return Database(":memory:")


# ------------------------------------------------------------------ summary / accuracy
def test_summary_calculation():
    rows = [row("WIN", 1500.0, 9, 30), row("WIN", 750.0, 9, 45), row("LOSS", -750.0, 10, 0), row("PENDING", hh=10, mm=5),
            row("AMBIGUOUS", hh=10, mm=10), row("UNRESOLVED", hh=10, mm=15)]
    s = pd_.summarize_rows(rows)
    assert (s["total"], s["resolved"], s["pending"], s["wins"], s["losses"]) == (6, 3, 1, 2, 1)
    assert (s["ambiguous"], s["unresolved"]) == (1, 1)
    assert s["simulated_pnl"] == 1500.0
    assert s["profit_factor"] == 3.0
    assert s["max_drawdown"] == 750.0
    assert "SIMULATED" in s["pnl_label"]


def test_accuracy_is_wins_over_wins_plus_losses():
    rows = [row("WIN", 1.0, 9, m) for m in range(18)] + [row("LOSS", -1.0, 10, m) for m in range(10)]
    s = pd_.summarize_rows(rows)
    assert s["accuracy"] == round(100 * 18 / 28, 2)
    assert s["accuracy_status"] == "PROVISIONAL"


def test_pending_ambiguous_unresolved_excluded_from_accuracy():
    base = [row("WIN", 1.0, 9, 30), row("LOSS", -1.0, 9, 35)]
    noisy = base + [row(st, hh=11, mm=m) for m, st in enumerate(["PENDING", "PENDING", "AMBIGUOUS", "UNRESOLVED"])]
    assert pd_.summarize_rows(base)["accuracy"] == pd_.summarize_rows(noisy)["accuracy"] == 50.0
    only_pending = pd_.summarize_rows([row("PENDING")])
    assert only_pending["accuracy"] is None and only_pending["accuracy_status"] == "N/A"
    assert only_pending["simulated_pnl"] is None


def test_accuracy_provisional_until_meaningful_sample():
    rows = [row("WIN", 1.0, 9 + i // 60, i % 60) for i in range(pd_.MEANINGFUL_SAMPLE - 1)]
    assert pd_.summarize_rows(rows)["accuracy_status"] == "PROVISIONAL"
    rows.append(row("LOSS", -1.0, 15, 0))
    assert pd_.summarize_rows(rows)["accuracy_status"] == "OBSERVED"


# ------------------------------------------------------------------ active signal
def test_active_signal_is_latest_pending_with_distances(db):
    save_premium(db, "old", 9, 30)
    save_premium(db, "new", 10, 15, direction="SELL")
    save_premium(db, "done", 10, 30, status="WIN", pnl=1500.0)
    mark(db, "new", at(10, 59, ss=45).isoformat(), price=110.0)
    a = snapshot(db)["active"]
    assert a["signal_id"] == "new" and a["option_type"] == "PE"
    assert a["current"] == 110.0
    # long option premium: SL below, target above, whatever the NIFTY direction
    assert a["distance_to_sl"] == 20.0 and a["distance_to_target"] == 10.0
    assert a["premium_history"] == [{"time": at(10, 59, ss=45).isoformat(), "close": 110.0}]


def test_spot_sell_active_signal_distances_are_signed_for_short():
    ev = spot_event(10, 0, direction="SELL")
    ev.stop_price, ev.target_price, ev.last_price = 25030.0, 24970.0, 25000.0
    a = pd_.active_signal([pd_.spot_row(ev)], {})
    assert a["distance_to_sl"] == 30.0 and a["distance_to_target"] == 30.0
    assert a["unit"] == "NIFTY points"


def test_no_active_signal(db):
    save_premium(db, "w", 9, 30, status="WIN", pnl=10.0)
    assert snapshot(db)["active"] is None


def test_previous_session_never_shows_an_active_signal(db):
    save_premium(db, "stale-pending", 13, 10, day=PREV)
    s = snapshot(db, day=PREV.isoformat())
    assert s["signals"][0]["status"] == "PENDING"
    assert s["active"] is None


# ------------------------------------------------------------------ history
def test_signal_history_today_and_previous_sessions(db):
    save_premium(db, "p1", 10, 0, day=PREV, status="WIN", pnl=1500.0)
    save_premium(db, "p2", 11, 0, day=PREV, status="LOSS", pnl=-750.0)
    save_premium(db, "t1", 9, 30, status="WIN", pnl=600.0)
    save_premium(db, "t2", 9, 50)
    s = snapshot(db)
    assert [r["signal_id"] for r in s["signals"]] == ["t2", "t1"]  # newest first, today only
    assert (s["summary"]["total"], s["summary"]["wins"], s["summary"]["pending"]) == (2, 1, 1)
    assert s["summary"]["simulated_pnl"] == 600.0
    assert s["cumulative"]["total"] == 4 and s["cumulative"]["accuracy"] == round(200 / 3, 2)
    assert [d["date"] for d in s["daily"]] == [DAY.isoformat(), PREV.isoformat()]
    prev = snapshot(db, day=PREV.isoformat())
    assert prev["date"] == PREV.isoformat() and prev["summary"]["accuracy"] == 50.0


def test_no_duplicate_signal_when_both_trackers_record_the_same_candle(db):
    save_premium(db, "prem", 10, 0)
    s = snapshot(db, spot=[spot_event(10, 0), spot_event(10, 20)])
    keyed = [r for r in s["signals"] if r["time"] == at(10, 0).isoformat()]
    assert len(keyed) == 1
    assert keyed[0]["basis"] == "OPTION_PREMIUM"
    assert keyed[0]["sources"] == [pd_.SOURCE_PREMIUM, pd_.SOURCE_SPOT]
    assert keyed[0]["spot"]["entry"] == 25012.0
    assert s["summary"]["total"] == 2


def test_duplicate_tracker_insert_is_ignored(db):
    save_premium(db, "dup", 10, 0)
    save_premium(db, "dup", 10, 0)
    assert snapshot(db)["summary"]["total"] == 1


def test_spot_source_reads_tracker_logs_and_heartbeat(tmp_path):
    store = SessionStore(tmp_path, "20261007")
    ev = spot_event(10, 0)
    store.append("SIGNAL", {"signal": ev.as_dict()})
    ev.status, ev.outcome = "TARGET_HIT", "WIN"
    store.append("OUTCOME", {"signal": ev.as_dict()})
    store.append("CANDLE", {"candle": {"timestamp": at(10, 0).isoformat(), "open": 1, "high": 2, "low": 0.5,
                                       "close": 1.5, "volume": 0}, "complete": True})
    write_status(tmp_path / STATUS_FILE, heartbeat())
    src = pd_.SpotSource(tmp_path)
    signals, candles = src.load()
    assert [pd_.spot_status(e) for e in signals] == ["WIN"] and len(candles) == 1
    assert src.status()["running"] is True
    fp = src.fingerprint()
    store.append("CANDLE", {"candle": {"timestamp": at(10, 5).isoformat(), "open": 1, "high": 2, "low": 0.5,
                                       "close": 1.5, "volume": 0}, "complete": True})
    assert src.fingerprint() != fp and len(src.load()[1]) == 2


# ------------------------------------------------------------------ health
def test_connected_fresh_feed(db):
    save_premium(db, "live", 10, 0)
    mark(db, "live", NOW.isoformat())
    h = snapshot(db, status=heartbeat(), premium_status=premium_heartbeat())["health"]
    assert (h["market_data"], h["websocket"], h["signal_engine"]) == ("CONNECTED", "CONNECTED", "RUNNING")
    assert h["stale"] is False and h["data_age_seconds"] == 1.0
    assert h["last_completed_candle"] == at(10, 55).isoformat()
    assert h["broker_execution"] == pd_.BROKER_PAPER_LABEL and h["broker_orders"] == 0
    assert h["system"] == "RUNNING" and h["overall_status"] == "RUNNING"


def test_stale_data_indicator(db):
    h = snapshot(db, status=heartbeat(tick_age=60))["health"]
    assert h["stale"] is True and h["data_age_seconds"] == 60.0
    assert h["market_data"] == "DISCONNECTED" and h["websocket"] == "CONNECTED"
    assert snapshot(db, status=heartbeat(gap=True))["health"]["stale"] is True


def test_disconnected_state(db):
    for status in (None, heartbeat(running=False), heartbeat(feed="DISCONNECTED"),
                   heartbeat(now=NOW - timedelta(seconds=pd_.HEARTBEAT_TIMEOUT_SECONDS + 5))):
        h = snapshot(db, status=status)["health"]
        assert h["market_data"] == "DISCONNECTED"
        if status is None or not status["running"] or status["updated_at"] != NOW.isoformat():
            assert h["signal_engine"] == "STOPPED" and h["websocket"] == "DISCONNECTED"


def test_market_live_or_closed():
    assert pd_.market_state(NOW, CFG) == "LIVE"
    assert pd_.market_state(at(15, 30), CFG) == "CLOSED"
    assert pd_.market_state(datetime(2026, 10, 10, 11, 0, tzinfo=IST), CFG) == "CLOSED"  # Saturday


def test_broker_disconnected_not_market_failure(db):
    h = snapshot(db, status=heartbeat(), premium_status=None)["health"]
    assert h["market_data"] == "CONNECTED"
    assert h["broker_execution"] == pd_.BROKER_PAPER_LABEL
    assert h["broker_orders"] == 0


def test_subscription_status_mapping():
    assert pd_.map_subscription_status("CONNECTED", tracker_running=True) == pd_.CONNECTED
    assert pd_.map_subscription_status("RECONNECTING", tracker_running=True) == pd_.STALE
    assert pd_.map_subscription_status("CONNECTED", tracker_running=False) == pd_.DISCONNECTED
    assert pd_.map_subscription_status("REST_ONLY", tracker_running=True) == pd_.DISCONNECTED


def test_premium_freshness_thresholds():
    assert pd_.freshness(5.0, live=True) == pd_.CONNECTED
    assert pd_.freshness(20.0, live=True) == pd_.STALE
    assert pd_.freshness(90.0, live=True) == pd_.DISCONNECTED
    assert pd_.freshness(None, live=True) == pd_.DISCONNECTED


def test_stale_premium_detection_and_display(db):
    save_premium(db, "stale", 9, 35)
    mark(db, "stale", at(10, 59, ss=30).isoformat(), price=148.1)
    a = snapshot(db, status=heartbeat(), premium_status=premium_heartbeat(running=False, age=600))["active"]
    assert a["premium_stale"] is True
    assert a["current_display"] == "STALE (148.1)"
    assert a["distance_to_sl"] is None
    h = snapshot(db, status=heartbeat(), premium_status=premium_heartbeat(running=False, age=600))["health"]
    assert h["premium_data"] == pd_.STALE
    assert h["premium_tracker"] == "STOPPED"
    assert h["overall_status"] == pd_.DEGRADED


def test_fresh_premium_when_tracker_running(db):
    save_premium(db, "fresh", 10, 0)
    mark(db, "fresh", NOW.isoformat(), price=120.0)
    h = snapshot(db, status=heartbeat(), premium_status=premium_heartbeat())["health"]
    assert h["premium_data"] == pd_.CONNECTED
    assert h["premium_tracker"] == "RUNNING"
    assert h["premium_data_age_seconds"] <= 1.0


def test_system_running_when_signal_engine_running(db):
    h = snapshot(db, status=heartbeat(), premium_status=None)["health"]
    assert h["system"] == "RUNNING"
    assert h["signal_engine"] == "RUNNING"
    assert h["overall_status"] in ("RUNNING", pd_.DEGRADED)


def test_tick_age_separate_from_completed_candle(db):
    save_premium(db, "sep", 10, 0)
    mark(db, "sep", at(10, 30).isoformat(), price=129.25, tick_time=at(10, 59, ss=50).isoformat())
    h = snapshot(db, status=heartbeat(), premium_status=premium_heartbeat())["health"]
    assert h["premium_tick_age_seconds"] == 10.0
    assert h["premium_candle_age_seconds"] == 1800.0
    assert h["premium_data"] == pd_.CONNECTED
    assert h["premium_last_candle"] == at(10, 30).isoformat()


def test_health_fields_consistent_between_snapshot_and_api(client):
    c, db, spot = client
    write_status(spot / pd_.STATUS_FILE, heartbeat())
    save_premium(db, "x", 10, 0)
    snap = c.get("/api/paper/snapshot").json()
    health = c.get("/api/paper/health").json()
    for key in ("market_data", "websocket", "signal_engine", "premium_tracker", "premium_data",
                "option_premium_feed", "premium_tick_age_seconds", "premium_last_tick",
                "premium_last_candle", "broker_execution", "broker_orders", "system", "overall_status"):
        assert health[key] == snap["health"][key]


# ------------------------------------------------------------------ real-time push
def test_stream_pushes_on_change_only(db):
    diff = pd_.StreamDiff(keepalive_after=3)
    first = diff.frames(snapshot(db, status=heartbeat()))
    assert len(first) == 1 and first[0].startswith("event: snapshot\n")
    assert diff.frames(snapshot(db, status=heartbeat(), now=NOW + timedelta(seconds=1))) == []

    moved = heartbeat(now=NOW + timedelta(seconds=2))
    moved["last_tick_price"] = 25040.0
    tick = diff.frames(snapshot(db, status=moved, now=NOW + timedelta(seconds=2)))
    assert len(tick) == 1 and tick[0].startswith("event: tick\n")
    assert json.loads(tick[0].split("data: ", 1)[1])["nifty"]["price"] == 25040.0

    save_premium(db, "fresh", 10, 55)
    new = diff.frames(snapshot(db, status=moved, now=NOW + timedelta(seconds=3)))
    assert new[0].startswith("event: snapshot\n")
    assert json.loads(new[0].split("data: ", 1)[1])["active"]["signal_id"] == "fresh"

    db.resolve_live_signal("fresh", {"status": "WIN", "exit_price": 120.0, "exit_time": NOW.isoformat(), "pnl": 1500.0,
                                     "resolved_at": NOW.isoformat(), "reason": "TARGET"})
    resolved = diff.frames(snapshot(db, status=moved, now=NOW + timedelta(seconds=4)))
    assert json.loads(resolved[0].split("data: ", 1)[1])["summary"]["wins"] == 1

    quiet = [diff.frames(snapshot(db, status=moved, now=NOW + timedelta(seconds=5 + i))) for i in range(3)]
    assert quiet[:2] == [[], []] and quiet[2] == [pd_.KEEPALIVE]


def test_stream_pushes_health_change(db):
    diff = pd_.StreamDiff()
    diff.frames(snapshot(db, status=heartbeat()))
    frames = diff.frames(snapshot(db, status=heartbeat(feed="DISCONNECTED")))
    assert frames[0].startswith("event: snapshot\n")
    assert json.loads(frames[0].split("data: ", 1)[1])["health"]["market_data"] == "DISCONNECTED"


# ------------------------------------------------------------------ API
@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = Database(tmp_path / "api.db")
    monkeypatch.setattr(api, "database", db)
    monkeypatch.setattr(api, "_paper_spot", pd_.SpotSource(tmp_path / "spot"))
    monkeypatch.setattr(api, "_paper_premium", pd_.PremiumSource(tmp_path / "premium"))
    monkeypatch.setattr(api, "_paper_cfg", CFG)
    monkeypatch.setattr(api, "_paper_now", lambda: NOW)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DASHBOARD_API_AUTH_REQUIRED", raising=False)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    yield TestClient(api.app), db, tmp_path / "spot"
    db.close()


def test_api_endpoints(client):
    c, db, spot = client
    save_premium(db, "a", 9, 30, status="WIN", pnl=1500.0)
    save_premium(db, "b", 10, 30)
    write_status(spot / STATUS_FILE, heartbeat())
    summary = c.get("/api/paper/summary").json()
    assert summary["summary"]["accuracy"] == 100.0 and summary["summary"]["pending"] == 1
    assert summary["market"]["state"] == "LIVE" and summary["market"]["nifty"]["price"] == 25031.5
    assert summary["banner"] == list(pd_.BANNER)
    assert [r["signal_id"] for r in c.get("/api/paper/signals").json()["signals"]] == ["b", "a"]
    assert c.get("/api/paper/active").json()["active"]["signal_id"] == "b"
    health = c.get("/api/paper/health").json()
    assert health["signal_engine"] == "RUNNING" and health["broker_orders"] == 0
    assert health["broker_execution"] == pd_.BROKER_PAPER_LABEL
    assert health["system"] == "RUNNING"
    snap = c.get("/api/paper/snapshot").json()
    assert {"summary", "signals", "active", "health", "chart", "daily", "banner"} <= set(snap)
    assert c.get("/api/paper/signals?date=bad").status_code == 400


def test_api_paper_mode_and_zero_orders(client):
    c, db, _ = client
    before = (db.get_orders(limit=10), db.get_trades(limit=10))
    for path in ("/api/paper/summary", "/api/paper/signals", "/api/paper/active", "/api/paper/health"):
        body = c.get(path).json()
        assert body["mode"] == "PAPER"
        assert body["safety"]["trading_mode"] == "PAPER" and body["safety"]["live_trading_enabled"] is False
        assert body["safety"]["paper_only"] is True
    assert c.get("/api/paper/health").json()["broker_orders"] == 0
    assert (db.get_orders(limit=10), db.get_trades(limit=10)) == before
    paper_routes = [r for r in api.app.routes if getattr(r, "path", "").startswith("/api/paper/")]
    assert paper_routes and all(r.methods <= {"GET", "HEAD"} for r in paper_routes)


def test_api_no_fabricated_values_when_trackers_idle(client):
    c, _, _ = client
    s = c.get("/api/paper/snapshot").json()
    assert s["signals"] == [] and s["active"] is None
    assert s["summary"]["accuracy"] is None and s["summary"]["simulated_pnl"] is None
    assert s["market"]["nifty"]["price"] is None and s["chart"]["candles"] == []
    assert s["health"]["signal_engine"] == "STOPPED" and s["health"]["market_data"] == "DISCONNECTED"


def test_stream_endpoint_generator_emits_snapshot(client):
    _, db, _ = client
    save_premium(db, "s1", 10, 0)

    class FakeRequest:
        calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 2

    async def collect():
        return [f async for f in api._paper_events(FakeRequest(), None, interval=0)]

    frames = asyncio.run(collect())
    assert frames[0].startswith("retry:")
    snaps = [f for f in frames if f.startswith("event: snapshot\n")]
    assert len(snaps) == 1  # unchanged second poll pushes nothing
    assert json.loads(snaps[0].split("data: ", 1)[1])["signals"][0]["signal_id"] == "s1"


def test_stream_requires_key_when_auth_required(client, monkeypatch):
    c, _, _ = client
    monkeypatch.setenv("DASHBOARD_API_AUTH_REQUIRED", "true")
    monkeypatch.setenv("DASHBOARD_API_KEY", "k")
    assert c.get("/api/paper/stream").status_code == 401
    assert c.get("/api/paper/summary").status_code == 401


# ------------------------------------------------------------------ no duplicate engine / no order path
@pytest.mark.parametrize("path", ["src/signal_accuracy/paper_dashboard.py"])
def test_dashboard_module_has_no_order_or_signal_engine_imports(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m and (m.startswith(("src.broker", "src.execution", "src.paper_trading", "src.engine"))) for m in mods)
    assert "generate_signals" not in (ROOT / path).read_text(encoding="utf-8")


def test_frontend_never_calculates_signals():
    js = (ROOT / "dashboard" / "paper_signals.js").read_text(encoding="utf-8")
    assert "/api/paper/stream" in js and "EventSource" in js
    for forbidden in ("generate_signals", "orb_high", "atr", "Guaranteed", "85%", "Live Trading Profit"):
        assert forbidden not in js
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    for label in ("PAPER MODE", "SIGNAL VALIDATION ONLY", "NO REAL ORDERS ARE BEING PLACED", "PROVISIONAL ACCURACY"):
        assert label in html
