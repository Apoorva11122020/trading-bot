"""Option demo + option API: full flow, safety gates, errors, persistence, UI.

The broker is an in-process PaperBroker standing in for the Upstox Sandbox
(deterministic, no network). Contracts come from the instrument-master
fixture. StrategyEngine, RiskManager, OrderManager and Database are real.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.broker.upstox_sandbox import UpstoxSandboxBroker, UpstoxSandboxError
from src.database import Database
from src.demo.controller import DemoController
from src.demo.sandbox_demo import DemoSafetyError
from src.options.demo import OPTION_NOTICE, OptionDemoController
from src.options.models import OptionResolutionError

from tests.options_support import make_resolver

API_KEY = "option-test-key-123"
SECRET = "sandbox-secret-token-value-opt"
DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
OPTION_COLUMNS = ("market_type", "underlying", "option_type", "strike", "expiry", "lot_size", "instrument_key")


class SandboxLikeBroker(PaperBroker):
    """PaperBroker plus the single-order lookup the Upstox Sandbox adapter has.
    The paper fill price is the order's reference premium."""

    next_status = OrderStatus.OPEN

    def __init__(self):
        super().__init__()
        self.set_next_execution(self.next_status)
        self.calls = []

    def place_order(self, symbol, side, quantity, order_type, price=None, tag=None):
        self.calls.append({"symbol": symbol, "side": side, "quantity": quantity, "order_type": order_type})
        if symbol not in self.prices and price:
            self.set_ltp(symbol, price)
        return super().place_order(symbol, side, quantity, order_type, price, tag)

    def get_order(self, order_id):
        return self.orders[order_id]


@pytest.fixture
def opt_env(tmp_path, monkeypatch):
    db = Database(tmp_path / "options.db")
    monkeypatch.setattr(api, "database", db)
    for name in ("APP_ENV", "DASHBOARD_API_AUTH_REQUIRED", "DEMO_OPTION_EQUITY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("UPSTOX_RUN_SANDBOX_ORDER_TEST", "true")
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", SECRET)
    monkeypatch.setenv("DASHBOARD_API_KEY", API_KEY)

    resolver = make_resolver()
    env = SimpleNamespace(db=db, brokers=[], broker_cls=SandboxLikeBroker, resolver=resolver)
    monkeypatch.setattr(api, "get_option_resolver", lambda: env.resolver)

    def factory():
        broker = env.broker_cls()
        env.brokers.append(broker)
        return broker

    def resolver_factory():
        if isinstance(env.resolver, Exception):
            raise env.resolver
        return env.resolver

    env.controller = OptionDemoController(db, resolver_factory=resolver_factory, broker_factory=factory)
    env.equity = DemoController(db, broker_factory=factory)
    monkeypatch.setattr(api, "option_demo_controller", env.controller)
    monkeypatch.setattr(api, "demo_controller", env.equity)
    env.client = TestClient(api.app)
    env.headers = {"X-API-Key": API_KEY}

    def run(body):
        response = env.client.post("/api/options/demo/start", json=body, headers=env.headers)
        assert response.status_code == 202, response.text
        assert env.controller.wait(20)
        return response.json()["demo_run_id"], env.client.get("/api/options/demo/status").json()

    env.run = run
    yield env
    env.controller.wait(20)
    env.equity.wait(20)
    db.close()


# --------------------------------------------------------------------------
# Full flows
# --------------------------------------------------------------------------

def test_index_option_demo_runs_full_flow_and_persists_nse_fo_order(opt_env):
    run_id, s = opt_env.run({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})

    assert s["stage"] == "COMPLETED" and s["running"] is False and s["error"] is None
    assert s["headline"] == "OPTION DEMO COMPLETED" and s["progress"] == 100
    assert [step["state"] for step in s["timeline"]] == ["done"] * 9
    assert s["market_type"] == "INDEX_OPTION" and s["underlying"] == "NIFTY"
    assert s["spot"] == 23000.0 and s["spot_source"] == "DEMO_REFERENCE_MEDIAN_STRIKE"
    assert s["expiry"] == "2026-10-06" and s["strike"] == 23000.0 and s["option_type"] == "CE"
    assert s["trading_symbol"] == "NIFTY 23000 CE 06 OCT 26" and s["lot_size"] == 65
    assert s["instrument_key"].startswith("NSE_FO|")
    assert s["decision"]["action"] == "BUY CALL"
    assert s["signal"]["quantity"] % 65 == 0 and s["signal"]["quantity"] > 0
    assert s["signal"]["stop_loss"] < s["signal"]["entry_price"] < s["signal"]["target"]
    assert s["risk"]["status"] == "ALLOWED" and s["risk"]["lot_size"] == 65
    assert s["option_resolution"]["status"] == "RESOLVED"
    assert s["sandbox_execution"]["status"] == "CANCELLED"
    assert s["data_labels"] == {"underlying": "DEMO DATA", "premium": "DEMO DATA"}
    assert s["notice"] == OPTION_NOTICE == "Option execution is currently validated in Paper/Sandbox mode."
    assert s["trading_mode"] == "PAPER" and s["live_trading_enabled"] is False

    (broker,) = opt_env.brokers
    (call,) = broker.calls
    assert call == {"symbol": s["instrument_key"], "side": "BUY", "quantity": s["signal"]["quantity"],
                    "order_type": "MARKET"}
    (order_id,) = broker.orders
    assert broker.orders[order_id].status == OrderStatus.CANCELLED and broker.connected is False
    assert s["order"]["order_id"] == order_id and s["order"]["status"] == "CANCELLED"
    assert s["broker"]["order"]["status"] == "OPEN"

    client = opt_env.client
    (row,) = client.get("/api/orders").json()
    assert row["order_id"] == order_id and row["symbol"] == s["instrument_key"] and row["status"] == "CANCELLED"
    assert {c: row[c] for c in OPTION_COLUMNS} == {
        "market_type": "INDEX_OPTION", "underlying": "NIFTY", "option_type": "CE", "strike": 23000.0,
        "expiry": "2026-10-06", "lot_size": 65, "instrument_key": s["instrument_key"]}
    history = [h["status"] for h in client.get(f"/api/orders/{order_id}/history").json()]
    assert history[0] == "SUBMITTED" and history[-1] == "CANCELLED"
    events = {e["event_type"] for e in client.get("/api/events").json()}
    assert {"OPTION_DEMO_STARTED", "OPTION_DEMO_COMPLETED"} <= events

    result = client.get(f"/api/options/demo/result/{run_id}").json()
    assert result["execution_result"] == "SUCCESS" and result["sandbox_order_id"] == order_id
    assert result["final_cancellation_status"] == "CANCELLED" and result["demo_type"] == "OPTION"
    assert opt_env.db.get_state(f"option_demo_result:{run_id}")["instrument_key"] == s["instrument_key"]
    assert SECRET not in json.dumps(s) and SECRET not in json.dumps(result)


def test_stock_option_put_flow_with_otm_strike(opt_env):
    _, s = opt_env.run({"market_type": "STOCK_OPTION", "underlying": "reliance", "option_type": "PE",
                        "strike_mode": "OTM", "strike_offset": 1})
    assert s["stage"] == "COMPLETED", s["error"]
    assert s["market_type"] == "STOCK_OPTION" and s["option_type"] == "PE"
    assert s["decision"]["action"] == "BUY PUT" and s["decision"]["side"] == "BUY"
    assert s["strike"] == 1230.0 and s["lot_size"] == 500
    assert s["signal"]["quantity"] % 500 == 0
    (row,) = opt_env.client.get("/api/orders").json()
    assert row["market_type"] == "STOCK_OPTION" and row["option_type"] == "PE" and row["lot_size"] == 500


def test_option_candle_signal_source(opt_env):
    _, s = opt_env.run({"market_type": "INDEX_OPTION", "underlying": "BANKNIFTY", "option_type": "CE",
                        "signal_source": "OPTION", "expiry_mode": "next"})
    assert s["stage"] == "COMPLETED", s["error"]
    assert s["decision"]["source"] == "OPTION" and s["expiry"] == "2026-11-24" and s["lot_size"] == 30


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------

def test_start_requires_control_api_key(opt_env):
    body = {"market_type": "INDEX_OPTION", "underlying": "NIFTY"}
    assert opt_env.client.post("/api/options/demo/start", json=body).status_code == 401
    assert opt_env.client.post("/api/options/demo/start", json=body,
                               headers={"X-API-Key": "wrong"}).status_code == 401
    assert opt_env.brokers == []


@pytest.mark.parametrize("env_name, value, detail", [
    ("LIVE_TRADING_ENABLED", "true", "LIVE_TRADING_ENABLED must be false"),
    ("TRADING_MODE", "LIVE", "TRADING_MODE must be PAPER"),
])
def test_option_demo_refused_when_live(opt_env, monkeypatch, env_name, value, detail):
    monkeypatch.setenv(env_name, value)
    response = opt_env.client.post("/api/options/demo/start", headers=opt_env.headers,
                                   json={"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    assert response.status_code == 403 and detail in response.json()["detail"]
    assert opt_env.controller.status()["stage"] == "IDLE" and opt_env.brokers == []
    with pytest.raises(DemoSafetyError):
        opt_env.controller.start(market_type="INDEX_OPTION", underlying="NIFTY")
    assert opt_env.db.get_orders() == []


def test_option_demo_refused_when_options_disabled_or_sandbox_off(opt_env, monkeypatch):
    body = {"market_type": "INDEX_OPTION", "underlying": "NIFTY"}
    monkeypatch.setattr(api, "_options_enabled", lambda: False)
    response = opt_env.client.post("/api/options/demo/start", json=body, headers=opt_env.headers)
    assert response.status_code == 403 and "options.enabled" in response.json()["detail"]
    monkeypatch.setattr(api, "_options_enabled", lambda: True)
    monkeypatch.setenv("UPSTOX_RUN_SANDBOX_ORDER_TEST", "false")
    response = opt_env.client.post("/api/options/demo/start", json=body, headers=opt_env.headers)
    assert response.status_code == 403 and "Sandbox order execution is disabled" in response.json()["detail"]
    assert opt_env.brokers == []


def test_kill_switch_blocks_option_demo(opt_env):
    from src.risk.controller import set_kill_switch
    set_kill_switch(opt_env.db, True, "TEST")
    response = opt_env.client.post("/api/options/demo/start", headers=opt_env.headers,
                                   json={"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    assert response.status_code == 409


@pytest.mark.parametrize("body", [
    {"market_type": "INDEX_OPTION", "underlying": "NIFTY", "live_trading": True},
    {"market_type": "EQUITY", "underlying": "NIFTY"},
    {"market_type": "INDEX_OPTION", "underlying": "NIFTY", "signal_source": "OPTION"},
    {"market_type": "INDEX_OPTION", "underlying": "NIFTY", "strike_mode": "EXACT"},
    {"market_type": "INDEX_OPTION", "underlying": "NIFTY", "expiry_mode": "date"},
    {"market_type": "INDEX_OPTION", "underlying": "NIFTY; DROP"},
    {"market_type": "INDEX_OPTION", "underlying": "NIFTY", "option_type": "FUT"},
])
def test_invalid_requests_are_rejected(opt_env, body):
    response = opt_env.client.post("/api/options/demo/start", json=body, headers=opt_env.headers)
    assert response.status_code == 422
    assert opt_env.controller.status()["stage"] == "IDLE"


def test_default_broker_is_the_sandbox_adapter_which_refuses_production_hosts():
    controller = OptionDemoController(None, resolver_factory=make_resolver)
    assert controller.broker_factory.__name__ == "_sandbox_broker"
    with pytest.raises(UpstoxSandboxError):
        UpstoxSandboxBroker(access_token="x", base_url="https://api.upstox.com/v3")
    with pytest.raises(UpstoxSandboxError):
        UpstoxSandboxBroker(access_token="x", v2_base_url="https://api-hft.upstox.com/v2")


# --------------------------------------------------------------------------
# Concurrency and duplicates
# --------------------------------------------------------------------------

def test_duplicate_and_cross_demo_runs_are_rejected(opt_env):
    release = threading.Event()
    connecting = threading.Event()

    class SlowBroker(SandboxLikeBroker):
        def connect(self):
            connecting.set()
            release.wait(10)
            super().connect()

    opt_env.broker_cls = SlowBroker
    body = {"market_type": "INDEX_OPTION", "underlying": "NIFTY"}
    assert opt_env.client.post("/api/options/demo/start", json=body, headers=opt_env.headers).status_code == 202
    assert connecting.wait(10)
    second = opt_env.client.post("/api/options/demo/start", json=body, headers=opt_env.headers)
    assert second.status_code == 409
    assert opt_env.client.post("/api/demo/start", headers=opt_env.headers).status_code == 409
    assert opt_env.equity.status()["stage"] == "IDLE"
    assert opt_env.client.post("/api/options/demo/reset", headers=opt_env.headers).status_code == 409
    release.set()
    assert opt_env.controller.wait(20)
    assert opt_env.controller.status()["stage"] == "COMPLETED"
    assert len(opt_env.brokers) == 1 and len(opt_env.db.get_orders()) == 1

    reset = opt_env.client.post("/api/options/demo/reset", headers=opt_env.headers)
    assert reset.status_code == 200 and reset.json()["stage"] == "IDLE"


# --------------------------------------------------------------------------
# Failure reporting: option resolution vs sandbox execution
# --------------------------------------------------------------------------

def test_market_type_mismatch_fails_in_resolution_without_broker(opt_env):
    _, s = opt_env.run({"market_type": "INDEX_OPTION", "underlying": "RELIANCE"})
    assert s["stage"] == "FAILED"
    assert s["error"]["code"] == "MARKET_TYPE_MISMATCH" and s["error"]["headline"] == "OPTION RESOLUTION FAILED"
    assert s["option_resolution"]["status"] == "FAILED"
    assert s["sandbox_execution"]["status"] == "NOT_ATTEMPTED"
    assert opt_env.brokers == [] and opt_env.db.get_orders() == []


def test_unlisted_exact_strike_fails_at_strike_resolution(opt_env):
    _, s = opt_env.run({"market_type": "INDEX_OPTION", "underlying": "NIFTY", "option_type": "CE",
                        "strike_mode": "EXACT", "strike": 23025})
    assert s["stage"] == "FAILED" and s["error"]["stage"] == "STRIKE_RESOLVED"
    assert s["error"]["code"] == "STRIKE_NOT_FOUND"
    assert s["sandbox_execution"]["status"] == "NOT_ATTEMPTED" and opt_env.brokers == []
    contract_step = next(step for step in s["timeline"] if step["key"] == "CONTRACT")
    assert contract_step["state"] == "failed"


def test_instrument_data_unavailable_is_reported_not_faked(opt_env):
    opt_env.resolver = OptionResolutionError("INSTRUMENT_DATA_UNAVAILABLE", "Upstox instrument data could not be loaded")
    run_id, s = opt_env.run({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    assert s["stage"] == "FAILED" and s["error"]["stage"] == "UNDERLYING_DATA"
    assert s["error"]["code"] == "INSTRUMENT_DATA_UNAVAILABLE"
    assert s["option_resolution"]["status"] == "FAILED" and s["order"] is None
    result = opt_env.client.get(f"/api/options/demo/result/{run_id}").json()
    assert result["execution_result"] == "FAILED" and result["sandbox_order_id"] is None
    events = {e["event_type"] for e in opt_env.client.get("/api/events").json()}
    assert "OPTION_DEMO_FAILED" in events


def test_broker_rejection_is_reported_as_sandbox_execution_failure(opt_env):
    class RejectingBroker(SandboxLikeBroker):
        next_status = OrderStatus.REJECTED

    opt_env.broker_cls = RejectingBroker
    _, s = opt_env.run({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    assert s["stage"] == "FAILED" and s["error"]["code"] == "ORDER_SUBMISSION_FAILED"
    assert s["option_resolution"]["status"] == "RESOLVED"
    assert s["sandbox_execution"]["status"] == "FAILED"


def test_small_equity_blocks_at_risk_without_order(opt_env, monkeypatch):
    monkeypatch.setenv("DEMO_OPTION_EQUITY", "50000")
    _, s = opt_env.run({"market_type": "INDEX_OPTION", "underlying": "NIFTY"})
    assert s["stage"] == "FAILED" and s["error"]["code"] == "RISK_BLOCKED"
    assert "QUANTITY_BELOW_ONE_LOT" in s["error"]["message"]
    assert opt_env.brokers == [] and opt_env.db.get_orders() == []


def test_unknown_result_is_404(opt_env):
    assert opt_env.client.get("/api/options/demo/result/OPTDEMO-nope").status_code == 404


# --------------------------------------------------------------------------
# Instrument / chain / resolve endpoints
# --------------------------------------------------------------------------

def test_underlyings_and_expiries_endpoints(opt_env):
    index = opt_env.client.get("/api/options/underlyings?market_type=INDEX_OPTION").json()
    assert [u["symbol"] for u in index["underlyings"]] == ["NIFTY", "BANKNIFTY"]
    assert index["enabled"] is True and "RELIANCE" in index["configured"]["stock"]
    stock = opt_env.client.get("/api/options/underlyings?market_type=STOCK_OPTION").json()
    assert [u["symbol"] for u in stock["underlyings"]] == ["RELIANCE"]
    assert opt_env.client.get("/api/options/underlyings?market_type=FUTURES").status_code == 422

    expiries = opt_env.client.get("/api/options/expiries?underlying=NIFTY").json()
    assert expiries["expiries"] == ["2026-10-06", "2026-10-13", "2026-10-27"]
    assert expiries["lot_size"] == 65 and expiries["market_type"] == "INDEX_OPTION"
    missing = opt_env.client.get("/api/options/expiries?underlying=NOSUCH")
    assert missing.status_code == 404 and missing.json()["detail"]["code"] == "UNDERLYING_NOT_FOUND"


def test_chain_endpoint_never_fabricates_quotes(opt_env):
    chain = opt_env.client.get("/api/options/chain?underlying=NIFTY&expiry=2026-10-13").json()
    assert chain["market_data_available"] is False
    assert chain["market_data_message"] == "Option chain requires market-data credentials"
    assert chain["expiry"] == "2026-10-13" and chain["atm_strike"] == 23000.0
    assert len(chain["rows"]) == 21
    assert all(r["ce_quote"] is None and r["pe_quote"] is None for r in chain["rows"])
    bad = opt_env.client.get("/api/options/chain?underlying=NIFTY&expiry=2026-10-14")
    assert bad.status_code == 404 and bad.json()["detail"]["code"] == "EXPIRY_NOT_FOUND"


def test_resolve_endpoint(opt_env):
    data = opt_env.client.get(
        "/api/options/resolve?underlying=NIFTY&option_type=CE&strike_mode=OTM&strike_offset=2").json()
    assert data["contract"]["trading_symbol"] == "NIFTY 23100 CE 06 OCT 26"
    assert data["spot_source"] == "DEMO_REFERENCE_MEDIAN_STRIKE" and data["contract"]["lot_size"] == 65
    bad = opt_env.client.get("/api/options/resolve?underlying=NIFTY&option_type=CE&strike_mode=EXACT&strike=23025")
    assert bad.status_code == 404 and bad.json()["detail"]["code"] == "STRIKE_NOT_FOUND"


def test_config_and_status_expose_option_settings_without_secrets(opt_env):
    cfg = opt_env.client.get("/api/config").json()
    assert cfg["options"]["allow_option_selling"] is False
    assert cfg["options"]["lot_size_required"] is True
    assert cfg["options"]["risk"]["target_r"] == 2.0
    status = opt_env.client.get("/api/options/demo/status").json()
    assert status["preconditions"]["options_enabled"] is True
    assert status["preconditions"]["live_trading_enabled"] is False
    assert SECRET not in json.dumps(cfg) and SECRET not in json.dumps(status)


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------

def test_dashboard_has_option_controls_and_safety_copy():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "options.js").read_text(encoding="utf-8")
    for text in ("▶ START DEMO", "▶ START OPTION DEMO", "PAPER TRADING · LIVE EXECUTION LOCKED",
                 "REAL MONEY: NOT USED", "OPTION DEMO · SANDBOX",
                 "Option execution is currently validated in Paper/Sandbox mode.",
                 "Option chain requires market-data credentials", 'data-trade-type="EQUITY"',
                 'data-trade-type="INDEX_OPTION"', 'data-trade-type="STOCK_OPTION"',
                 "<th>Market</th>", '<script src="/dashboard/options.js"></script>'):
        assert text in html, text
    header = html.split('class="chain-table"')[1].split("</thead>")[0]
    assert re.findall(r"<th(?:\s[^>]*)?>(.*?)</th>", header) == [
        "Strike", "CE LTP", "CE Volume", "CE OI", "PE LTP", "PE Volume", "PE OI"]
    for control in ("optUnderlying", "optExpiry", "optType", "optStrikeMode", "optStrike", "optSource"):
        assert f'id="{control}"' in html
    assert "/api/options/demo/start" in js and "/api/options/chain" in js
    lowered = (html + js).lower()
    assert "enable live" not in lowered and "live_trading_enabled=true" not in lowered


def test_dashboard_serves_options_script():
    client = TestClient(api.app)
    assert client.get("/dashboard/options.js").status_code == 200
