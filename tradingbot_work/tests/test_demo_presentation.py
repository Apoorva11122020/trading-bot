"""Dashboard presentation of persisted demo runs (/api/demo/overview + UI).

The broker is an in-process PaperBroker standing in for the Upstox Sandbox
(deterministic, no network). The demo controllers, StrategyEngine,
RiskManager, OrderManager and Database are the real components; the overview
must only echo what they persisted.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.database import Database
from src.demo.controller import DemoController
from src.demo.summary import PERFORMANCE_NOTE, PNL_UNAVAILABLE, build_overview
from src.options.demo import OptionDemoController
from src.options.models import OptionResolutionError

from tests.options_support import make_resolver

API_KEY = "presentation-test-key-123"
SECRET = "sandbox-secret-token-value-pres"
EQUITY_INSTRUMENT = "NSE_EQ|INE669E01016"
DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
STAGE_EVENTS = {"MARKET_DATA_READY", "SIGNAL_GENERATED", "RISK_VALIDATED",
                "ORDER_SUBMITTED", "ORDER_VERIFIED", "ORDER_CANCELLED"}


class SandboxLikeBroker(PaperBroker):
    def __init__(self):
        super().__init__()
        self.set_ltp(EQUITY_INSTRUMENT, 100.0)
        self.set_next_execution(OrderStatus.OPEN)

    def place_order(self, symbol, side, quantity, order_type, price=None, tag=None):
        if symbol not in self.prices and price:
            self.set_ltp(symbol, price)
        return super().place_order(symbol, side, quantity, order_type, price, tag)

    def get_order(self, order_id):
        return self.orders[order_id]


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = Database(tmp_path / "presentation.db")
    monkeypatch.setattr(api, "database", db)
    for name in ("APP_ENV", "DASHBOARD_API_AUTH_REQUIRED", "DEMO_EQUITY", "DEMO_STRATEGY_SYMBOL",
                 "DEMO_OPTION_EQUITY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("UPSTOX_RUN_SANDBOX_ORDER_TEST", "true")
    monkeypatch.setenv("UPSTOX_SANDBOX_INSTRUMENT", EQUITY_INSTRUMENT)
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", SECRET)
    monkeypatch.setenv("DASHBOARD_API_KEY", API_KEY)

    e = SimpleNamespace(db=db, path=tmp_path / "presentation.db", brokers=[], resolver=make_resolver())

    def broker_factory():
        broker = SandboxLikeBroker()
        e.brokers.append(broker)
        return broker

    def resolver_factory():
        if isinstance(e.resolver, Exception):
            raise e.resolver
        return e.resolver

    e.equity = DemoController(db, broker_factory=broker_factory)
    e.option = OptionDemoController(db, resolver_factory=resolver_factory, broker_factory=broker_factory)
    monkeypatch.setattr(api, "demo_controller", e.equity)
    monkeypatch.setattr(api, "option_demo_controller", e.option)
    e.client = TestClient(api.app)
    e.headers = {"X-API-Key": API_KEY}

    def run_equity():
        response = e.client.post("/api/demo/start", headers=e.headers)
        assert response.status_code == 202, response.text
        assert e.equity.wait(15)
        return response.json()["demo_run_id"], e.client.get("/api/demo/status").json()

    def run_option(body):
        response = e.client.post("/api/options/demo/start", json=body, headers=e.headers)
        assert response.status_code == 202, response.text
        assert e.option.wait(20)
        return response.json()["demo_run_id"], e.client.get("/api/options/demo/status").json()

    e.run_equity, e.run_option = run_equity, run_option
    e.overview = lambda: e.client.get("/api/demo/overview").json()
    yield e
    e.equity.wait(15)
    e.option.wait(20)
    db.close()


def _keys(timeline):
    return [step["key"] for step in timeline]


# --------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------

def test_overview_is_empty_before_any_demo(env):
    ov = env.overview()
    assert ov["latest"] is None and ov["latest_equity"] is None and ov["latest_option"] is None
    assert ov["runs"] == [] and ov["latest_events"] == [] and ov["order_labels"] == {}
    stats = ov["stats"]
    assert stats["demo_runs"] == stats["demo_orders"] == stats["total_trades"] == stats["open_positions"] == 0
    assert stats["realized_pnl"] is None and stats["pnl_note"] == PNL_UNAVAILABLE
    assert ov["safety"] == {"trading_mode": "PAPER", "live_trading_enabled": False}
    assert ov["running"] == {"equity": False, "option": False}


# --------------------------------------------------------------------------
# Equity demo (regression) on Overview + Signals
# --------------------------------------------------------------------------

def test_latest_equity_demo_visible_with_actual_signal(env):
    run_id, status = env.run_equity()
    (broker,) = env.brokers
    (order_id,) = broker.orders
    signal = status["signal"]

    latest = env.overview()["latest"]
    assert latest["demo_run_id"] == run_id and latest["demo_type"] == "EQUITY"
    assert latest["market_type"] == "EQUITY" and latest["underlying"] == "RELIANCE"
    assert latest["instrument_key"] == EQUITY_INSTRUMENT
    assert latest["signal"] == latest["side"] == signal["side"] == "BUY"
    assert latest["entry"] == signal["entry_price"]
    assert latest["stop_loss"] == signal["stop_loss"] and latest["target"] == signal["target"]
    assert latest["quantity"] == signal["quantity"] > 0
    assert latest["conditions"] == signal["conditions"]
    assert latest["signal_id"] == signal["signal_id"]
    assert latest["order_id"] == order_id and latest["order_type"] == "MARKET"
    assert latest["final_status"] == "CANCELLED" and latest["execution_result"] == "SUCCESS"
    assert latest["broker"] == "UPSTOX SANDBOX" and latest["stage"] == "COMPLETED"
    assert latest["expiry"] is None and latest["option_type"] is None and latest["lot_size"] is None
    assert latest["live_trading_enabled"] is False and latest["trading_mode"] == "PAPER"

    assert _keys(latest["timeline"]) == [
        "MARKET_DATA", "STRATEGY", "SIGNAL", "RISK", "ORDER_SUBMITTED", "BROKER_VERIFIED",
        "ORDER_CANCELLED", "FINAL"]
    assert all(step["state"] == "done" for step in latest["timeline"])
    assert latest["timeline"][-1]["detail"] == "DEMO COMPLETED"
    assert latest["timeline"][4]["detail"] == order_id


# --------------------------------------------------------------------------
# Option demos: metadata
# --------------------------------------------------------------------------

def test_index_option_demo_metadata_visible(env):
    run_id, s = env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    assert s["stage"] == "COMPLETED", s["error"]
    latest = env.overview()["latest"]

    assert latest["demo_run_id"] == run_id and latest["demo_type"] == "OPTION"
    assert latest["market_type"] == "INDEX_OPTION" and latest["underlying"] == "NIFTY"
    assert latest["contract"] == s["trading_symbol"] and latest["instrument_key"] == s["instrument_key"]
    assert latest["instrument_key"].startswith("NSE_FO|")
    assert latest["expiry"] == s["expiry"] and latest["strike"] == s["strike"]
    assert latest["option_type"] == "CE" and latest["signal"] == "BUY CALL" and latest["side"] == "BUY"
    assert latest["lot_size"] == 65 and latest["lots"] * latest["lot_size"] == latest["quantity"]
    assert latest["quantity"] == s["signal"]["quantity"]
    assert latest["entry"] == s["signal"]["entry_price"] and latest["entry_basis"] == "PREMIUM"
    assert latest["stop_loss"] == s["signal"]["stop_loss"] and latest["target"] == s["signal"]["target"]
    assert latest["conditions"] == s["decision"]["conditions"] and latest["conditions"]
    assert latest["order_id"] == s["order"]["order_id"] and latest["final_status"] == "CANCELLED"
    assert latest["data_label"] == "DEMO DATA"

    keys = _keys(latest["timeline"])
    assert keys.index("CONTRACT") < keys.index("RISK") < keys.index("ORDER_SUBMITTED")
    assert all(step["state"] == "done" for step in latest["timeline"])
    contract_step = next(step for step in latest["timeline"] if step["key"] == "CONTRACT")
    assert contract_step["detail"] == s["trading_symbol"]


def test_stock_option_put_and_latest_per_market(env):
    equity_id, _ = env.run_equity()
    option_id, s = env.run_option({"market_type": "STOCK_OPTION", "underlying": "RELIANCE", "option_type": "PE"})
    assert s["stage"] == "COMPLETED", s["error"]

    ov = env.overview()
    assert ov["latest"]["demo_run_id"] == option_id
    assert ov["latest"]["market_type"] == "STOCK_OPTION" and ov["latest"]["signal"] == "BUY PUT"
    assert ov["latest"]["option_type"] == "PE"
    assert ov["latest_equity"]["demo_run_id"] == equity_id
    assert ov["latest_option"]["demo_run_id"] == option_id
    assert [r["demo_run_id"] for r in ov["runs"]] == [option_id, equity_id]
    assert ov["stats"]["demo_runs"] == ov["stats"]["completed_demo_runs"] == 2


# --------------------------------------------------------------------------
# Positions, Orders, Events, Analytics
# --------------------------------------------------------------------------

def test_cancelled_demo_does_not_appear_as_open_position(env):
    env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    assert env.client.get("/api/positions").json() == []
    ov = env.overview()
    assert ov["stats"]["open_positions"] == 0
    assert ov["stats"]["cancelled_demo_orders"] == ov["stats"]["demo_orders"] == 1
    assert ov["latest"]["final_status"] == "CANCELLED"


def test_demo_orders_visible_in_orders_with_labels(env):
    env.run_equity()
    _, s = env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    orders = {o["order_id"]: o for o in env.client.get("/api/orders").json()}
    ov = env.overview()

    assert set(orders) == set(ov["order_labels"]) and len(orders) == 2
    option_order = orders[s["order"]["order_id"]]
    assert option_order["status"] == "CANCELLED" and option_order["market_type"] == "INDEX_OPTION"
    assert option_order["expiry"] == s["expiry"] and option_order["strike"] == s["strike"]
    assert option_order["option_type"] == "CE" and option_order["lot_size"] == 65
    equity_label = ov["order_labels"][ov["latest_equity"]["order_id"]]
    assert equity_label == {"demo_run_id": ov["latest_equity"]["demo_run_id"], "market_type": "EQUITY",
                            "underlying": "RELIANCE", "contract": "RELIANCE"}
    assert orders[ov["latest_equity"]["order_id"]]["status"] == "CANCELLED"


def test_demo_stage_events_are_recorded_in_system_events(env):
    equity_id, _ = env.run_equity()
    option_id, _ = env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})

    rows = env.client.get("/api/events").json()
    by_run: dict[str, set] = {}
    for row in rows:
        meta = json.loads(row["metadata_json"] or "{}")
        if meta.get("demo_run_id"):
            by_run.setdefault(meta["demo_run_id"], set()).add(row["event_type"])
    assert STAGE_EVENTS | {"DEMO_STARTED", "DEMO_COMPLETED"} <= by_run[equity_id]
    assert "OPTION_RESOLVED" not in by_run[equity_id]
    assert STAGE_EVENTS | {"OPTION_RESOLVED", "OPTION_DEMO_STARTED", "OPTION_DEMO_COMPLETED"} <= by_run[option_id]

    latest_events = env.overview()["latest_events"]
    types = [e["event_type"] for e in latest_events]
    assert types[0] == "OPTION_DEMO_STARTED" and "OPTION_DEMO_COMPLETED" in types
    assert types.index("SIGNAL_GENERATED") < types.index("OPTION_RESOLVED") < types.index("ORDER_CANCELLED")
    assert "ORDER" in types
    assert all(e["demo_run_id"] == option_id or e["event_type"] == "ORDER" for e in latest_events)


def test_failed_option_demo_records_no_stage_success_events(env):
    env.resolver = OptionResolutionError("INSTRUMENT_DATA_UNAVAILABLE", "Upstox instrument data could not be loaded")
    run_id, _ = env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    latest = env.overview()["latest"]

    assert latest["execution_result"] == "FAILED" and latest["order_id"] is None
    assert latest["timeline"][0]["state"] == "failed"
    assert all(step["state"] == "pending" for step in latest["timeline"][1:-1])
    assert latest["timeline"][-1]["state"] == "failed"
    assert latest["error"]["code"] == "INSTRUMENT_DATA_UNAVAILABLE"
    types = {e["event_type"] for e in env.overview()["latest_events"]}
    assert "OPTION_DEMO_FAILED" in types and not (types & STAGE_EVENTS)
    assert env.overview()["stats"]["failed_demo_runs"] == 1


def test_analytics_does_not_fabricate_pnl_for_cancelled_orders(env):
    env.run_equity()
    env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    stats = env.overview()["stats"]

    assert stats["filled_orders"] == 0 and stats["total_trades"] == 0 and stats["closed_trades"] == 0
    assert stats["realized_pnl"] is None
    assert stats["pnl_note"] == PNL_UNAVAILABLE == "N/A — Sandbox orders were cancelled without fills."
    assert stats["performance_note"] == PERFORMANCE_NOTE
    assert env.client.get("/api/equity").json() == []


def test_analytics_reports_pnl_only_from_closed_trades(env):
    env.db.save_trade({"trade_id": "T1", "order_id": "O1", "symbol": "RELIANCE", "side": "BUY", "quantity": 10,
                       "entry_price": 100.0, "exit_price": 102.5, "realized_pnl": 25.0,
                       "opened_at": "2026-10-01T04:00:00+00:00", "closed_at": "2026-10-01T05:00:00+00:00",
                       "status": "CLOSED"})
    env.db.save_trade({"trade_id": "T2", "order_id": "O2", "symbol": "TCS", "side": "BUY", "quantity": 5,
                       "entry_price": 100.0, "opened_at": "2026-10-01T04:00:00+00:00", "status": "OPEN"})
    stats = env.overview()["stats"]
    assert stats["closed_trades"] == 1 and stats["total_trades"] == 2
    assert stats["realized_pnl"] == 25.0 and stats["pnl_note"] is None


# --------------------------------------------------------------------------
# Persistence, auth, secrets
# --------------------------------------------------------------------------

def test_overview_reads_persisted_results_after_restart(env):
    run_id, _ = env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    reopened = Database(env.path)
    try:
        ov = build_overview(reopened, running={"equity": False, "option": False},
                            trading_mode="PAPER", live_trading_enabled=False)
    finally:
        reopened.close()
    assert ov["latest"]["demo_run_id"] == run_id and ov["latest"]["final_status"] == "CANCELLED"
    assert all(step["state"] == "done" for step in ov["latest"]["timeline"])


def test_overview_never_exposes_secrets(env):
    env.run_equity()
    env.run_option({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    body = env.client.get("/api/demo/overview").text
    assert SECRET not in body and API_KEY not in body


def test_overview_requires_key_when_dashboard_auth_required(env, monkeypatch):
    monkeypatch.setenv("DASHBOARD_API_AUTH_REQUIRED", "true")
    assert env.client.get("/api/demo/overview").status_code == 401
    assert env.client.get("/api/demo/overview", headers=env.headers).status_code == 200


def test_overview_is_read_only(env):
    assert env.client.post("/api/demo/overview", headers=env.headers).status_code == 405


# --------------------------------------------------------------------------
# Dashboard static assets
# --------------------------------------------------------------------------

def test_dashboard_wires_presentation_views():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    app_js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    js = (DASHBOARD / "presentation.js").read_text(encoding="utf-8")
    for element in ("latestDemoCard", "signalsContent", "signalsEmpty", "positionsContext",
                    "analyticsContent", "analyticsEmpty", "eventFilter", "ordersBadge"):
        assert f'id="{element}"' in html, element
    assert '<script src="/dashboard/presentation.js"></script>' in html
    for header in ("Time", "Market", "Underlying", "Instrument", "Side", "Qty", "Type", "Order ID", "Status",
                   "Option Details"):
        assert f"<th>{header}</th>" in html, header
    assert "/api/demo/overview" in app_js and "renderPresentation" in app_js
    for text in ("LATEST DEMO RUN", "Latest Demo Signal", "DEMO / PAPER", "DEMO / SANDBOX", "NO OPEN POSITIONS",
                 "Demo orders completed successfully and were cancelled in Sandbox",
                 "PAPER TRADING", "LIVE EXECUTION LOCKED", "Demo Execution Timeline"):
        assert text in js, text
    assert "No open positions. The latest Sandbox demo order was cancelled successfully" in app_js


def test_dashboard_presentation_has_no_hardcoded_demo_values():
    js = (DASHBOARD / "presentation.js").read_text(encoding="utf-8")
    for literal in ("NSE_FO|", "22900", "40721", "106699", "1240", "RELIANCE", "NIFTY", "4500", "195"):
        assert literal not in js, literal
    lowered = js.lower()
    assert "enable live" not in lowered and "live_trading_enabled=true" not in lowered
    assert "fetch(" not in js and "method:'POST'" not in js


def test_demo_has_its_own_tab_and_overview_is_simplified():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    app_js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert 'data-view="demo"' in html
    demo_view = html.split('id="demoView"')[1].split('id="overviewView"')[0]
    for element in ("tradeTypePanel", "optionConsole", "demoConsole", "demoStartBtn", "optStartBtn"):
        assert f'id="{element}"' in demo_view, element
    overview = html.split('id="overviewView"')[1].split('id="positionsView"')[0]
    assert 'id="latestDemoCard"' in overview and 'id="demoConsole"' not in overview
    assert "riskPct" not in html and "12%" not in html
    assert "demo:'Sandbox Demo'" in app_js and "'index-options','stock-options'" in app_js


def test_dashboard_supports_light_and_dark_themes():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    app_js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert 'id="themeToggle"' in html and "prefers-color-scheme: dark" in html and "atlasTheme" in html
    assert '[data-theme="light"]' in css and '[data-theme="dark"]' in css
    assert "localStorage.setItem('atlasTheme'" in app_js
    for text in ("PAPER TRADING · LIVE EXECUTION LOCKED", "REAL MONEY: NOT USED"):
        assert text in html


def test_dashboard_serves_presentation_script():
    assert TestClient(api.app).get("/dashboard/presentation.js").status_code == 200
