"""UI-triggered Sandbox demo: API endpoints, controller lifecycle, safety and errors.

The broker is an in-process PaperBroker standing in for the Upstox Sandbox
(deterministic, no network). StrategyRuntime, RiskManager, OrderManager and
Database are the real components.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.api.app as api
from src.broker.models import OrderStatus
from src.broker.paper import PaperBroker
from src.database import Database
from src.demo import sandbox_demo
from src.demo.controller import DemoController

INSTRUMENT = "NSE_EQ|INE669E01016"
API_KEY = "demo-test-key-123"
SECRET = "sandbox-secret-token-value-xyz"


class SandboxLikeBroker(PaperBroker):
    """PaperBroker plus the single-order lookup the Upstox Sandbox adapter has."""

    def __init__(self):
        super().__init__()
        self.set_ltp(INSTRUMENT, 100.0)
        self.set_next_execution(OrderStatus.OPEN)

    def get_order(self, order_id: str):
        return self.orders[order_id]


@pytest.fixture
def demo_env(tmp_path, monkeypatch):
    db = Database(tmp_path / "demo.db")
    monkeypatch.setattr(api, "database", db)
    for name in ("APP_ENV", "DASHBOARD_API_AUTH_REQUIRED", "DEMO_EQUITY", "DEMO_STRATEGY_SYMBOL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRADING_MODE", "PAPER")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
    monkeypatch.setenv("UPSTOX_RUN_SANDBOX_ORDER_TEST", "true")
    monkeypatch.setenv("UPSTOX_SANDBOX_INSTRUMENT", INSTRUMENT)
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", SECRET)
    monkeypatch.setenv("DASHBOARD_API_KEY", API_KEY)

    env = SimpleNamespace(db=db, brokers=[], broker_cls=SandboxLikeBroker)

    def factory():
        broker = env.broker_cls()
        env.brokers.append(broker)
        return broker

    env.controller = DemoController(db, broker_factory=factory)
    monkeypatch.setattr(api, "demo_controller", env.controller)
    env.client = TestClient(api.app)
    env.headers = {"X-API-Key": API_KEY}

    def run():
        response = env.client.post("/api/demo/start", headers=env.headers)
        assert response.status_code == 202, response.text
        assert env.controller.wait(15)
        return response.json()["demo_run_id"], env.client.get("/api/demo/status").json()

    env.run = run
    yield env
    env.controller.wait(15)
    db.close()


def test_start_in_paper_mode_runs_actual_flow_and_persists_order(demo_env):
    run_id, status = demo_env.run()

    assert status["running"] is False
    assert status["stage"] == "COMPLETED"
    assert status["progress"] == 100
    assert status["error"] is None
    assert status["demo_run_id"] == run_id
    assert status["signal"]["side"] == "BUY"
    assert status["signal"]["symbol"] == "RELIANCE"
    assert status["signal"]["conditions"] == {
        "ema_trend": True, "vwap": True, "rsi": True, "breakout": True, "volume": True}
    assert status["risk"]["status"] == "ALLOWED"
    assert status["risk"]["limits"]["risk_per_trade_percent"] == 0.5
    assert [step["state"] for step in status["timeline"]] == ["done"] * 8

    (broker,) = demo_env.brokers
    (order_id,) = broker.orders
    assert broker.orders[order_id].status == OrderStatus.CANCELLED
    assert status["order"]["order_id"] == order_id
    assert status["order"]["status"] == "CANCELLED"
    assert status["broker"]["order"]["order_id"] == order_id
    assert status["broker"]["order"]["status"] == "OPEN"
    assert broker.connected is False

    client = demo_env.client
    orders = client.get("/api/orders").json()
    assert [(o["order_id"], o["symbol"], o["status"]) for o in orders] == [
        (order_id, INSTRUMENT, "CANCELLED")]
    assert orders[0]["quantity"] == status["signal"]["quantity"]
    history = [h["status"] for h in client.get(f"/api/orders/{order_id}/history").json()]
    assert history[0] == "SUBMITTED" and history[-1] == "CANCELLED"

    event_types = {e["event_type"] for e in client.get("/api/events").json()}
    assert {"DEMO_STARTED", "DEMO_COMPLETED", "ORDER"} <= event_types


def test_result_endpoint_returns_actual_values(demo_env):
    run_id, status = demo_env.run()
    result = demo_env.client.get(f"/api/demo/result/{run_id}").json()
    (broker,) = demo_env.brokers
    (order_id,) = broker.orders

    assert result["demo_run_id"] == run_id
    assert result["strategy_symbol"] == "RELIANCE"
    assert result["timeframe"] == "5m"
    assert result["signal_id"] == status["signal"]["signal_id"]
    assert result["signal_side"] == "BUY"
    assert result["entry_price"] == status["signal"]["entry_price"]
    assert result["quantity"] == status["signal"]["quantity"] > 0
    assert result["stop_loss"] == status["signal"]["stop_loss"]
    assert result["target"] == status["signal"]["target"]
    assert result["risk_status"] == "ALLOWED"
    assert result["risk_reason"] == "ALL_ENTRY_CONDITIONS_MET"
    assert result["broker_instrument"] == INSTRUMENT
    assert result["sandbox_order_id"] == order_id
    assert result["broker_order_status"] == "OPEN"
    assert result["requested_quantity"] == result["quantity"]
    assert result["filled_quantity"] == 0
    assert result["average_fill"] is None
    assert result["final_cancellation_status"] == "CANCELLED"
    assert result["execution_result"] == "SUCCESS"
    assert result["live_trading_enabled"] is False
    assert result["trading_mode"] == "PAPER"


def test_result_is_persisted_and_survives_api_restart(demo_env):
    run_id, _ = demo_env.run()
    restarted = DemoController(demo_env.db, broker_factory=lambda: None)
    stored = restarted.result(run_id)
    assert stored["execution_result"] == "SUCCESS"
    assert stored == demo_env.client.get(f"/api/demo/result/{run_id}").json()
    assert demo_env.db.get_state("demo_last_run_id") == run_id
    assert demo_env.client.get("/api/demo/result/DEMO-unknown").status_code == 404


@pytest.mark.parametrize("env,value", [("LIVE_TRADING_ENABLED", "true"), ("TRADING_MODE", "LIVE")])
def test_start_refused_unless_paper_and_live_disabled(demo_env, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    response = demo_env.client.post("/api/demo/start", headers=demo_env.headers)
    assert response.status_code == 403
    assert "Demo refused" in response.json()["detail"]
    assert demo_env.brokers == []
    assert demo_env.controller.status()["stage"] == "IDLE"
    assert demo_env.client.get("/api/orders").json() == []


def test_controller_itself_refuses_live_trading(demo_env, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(sandbox_demo.DemoSafetyError):
        demo_env.controller.start()
    assert demo_env.brokers == []


def test_duplicate_start_is_rejected_while_running(demo_env):
    release = threading.Event()
    entered = threading.Event()

    class SlowBroker(SandboxLikeBroker):
        def connect(self):
            entered.set()
            assert release.wait(10)
            super().connect()

    demo_env.broker_cls = SlowBroker
    client = demo_env.client
    first = client.post("/api/demo/start", headers=demo_env.headers)
    assert first.status_code == 202
    assert entered.wait(10)

    second = client.post("/api/demo/start", headers=demo_env.headers)
    assert second.status_code == 409
    assert second.json()["detail"] == "Demo already running"
    assert client.post("/api/demo/reset", headers=demo_env.headers).status_code == 409
    assert client.get("/api/demo/status").json()["running"] is True

    release.set()
    assert demo_env.controller.wait(15)
    assert len(demo_env.brokers) == 1
    assert len(demo_env.brokers[0].orders) == 1
    assert len(client.get("/api/orders").json()) == 1


def test_status_lifecycle_follows_backend_stages(demo_env, monkeypatch):
    stages = []
    original = DemoController._update

    def recording(self, **changes):
        if "stage" in changes:
            stages.append(changes["stage"])
        original(self, **changes)

    monkeypatch.setattr(DemoController, "_update", recording)
    idle = demo_env.client.get("/api/demo/status").json()
    assert idle["stage"] == "IDLE" and idle["running"] is False and idle["progress"] == 0
    assert idle["preconditions"] == {
        "trading_mode": "PAPER", "live_trading_enabled": False,
        "sandbox_execution_enabled": True, "kill_switch_active": False}

    demo_env.run()
    assert stages == [
        "GENERATING_MARKET_DATA", "STRATEGY_EVALUATION", "SIGNAL_GENERATED", "RISK_VALIDATION",
        "BROKER_CONNECTING", "ORDER_SUBMITTING", "ORDER_VERIFIED", "ORDER_CANCELLING", "COMPLETED",
    ]

    reset = demo_env.client.post("/api/demo/reset", headers=demo_env.headers)
    assert reset.status_code == 200 and reset.json()["stage"] == "IDLE"
    (broker,) = demo_env.brokers
    assert [o.status for o in broker.orders.values()] == [OrderStatus.CANCELLED]
    assert len(demo_env.client.get("/api/orders").json()) == 1


def _failed(demo_env):
    response = demo_env.client.post("/api/demo/start", headers=demo_env.headers)
    assert response.status_code == 202
    run_id = response.json()["demo_run_id"]
    assert demo_env.controller.wait(15)
    status = demo_env.client.get("/api/demo/status").json()
    result = demo_env.client.get(f"/api/demo/result/{run_id}").json()
    assert status["stage"] == "FAILED" and status["running"] is False
    assert result["execution_result"] == "FAILED"
    return status, result


def test_broker_connection_failure(demo_env):
    class Down(SandboxLikeBroker):
        def connect(self):
            raise ConnectionError(f"sandbox unreachable Bearer {SECRET} token={SECRET}")

    demo_env.broker_cls = Down
    status, result = _failed(demo_env)
    assert status["error"]["code"] == "BROKER_CONNECTION_FAILED"
    assert status["error"]["headline"] == "BROKER CONNECTION FAILED"
    assert status["error"]["stage"] == "BROKER_CONNECTING"
    assert status["order"] is None and result["sandbox_order_id"] is None
    assert SECRET not in json.dumps(status) and SECRET not in json.dumps(result)
    assert SECRET not in json.dumps(demo_env.client.get("/api/events").json())
    assert demo_env.client.get("/api/orders").json() == []
    timeline = {s["key"]: s["state"] for s in status["timeline"]}
    assert timeline["RISK"] == "done" and timeline["ORDER_MANAGER"] == "failed"
    assert timeline["COMPLETE"] == "pending"


def test_order_submission_rejected_by_broker(demo_env):
    class Rejecting(SandboxLikeBroker):
        def __init__(self):
            super().__init__()
            self.set_next_execution(OrderStatus.REJECTED, rejection_reason="MARGIN")

    demo_env.broker_cls = Rejecting
    status, _ = _failed(demo_env)
    assert status["error"]["code"] == "ORDER_SUBMISSION_FAILED"
    assert status["error"]["headline"] == "ORDER SUBMISSION FAILED"
    assert "MARGIN" in status["error"]["message"]
    assert [o["status"] for o in demo_env.client.get("/api/orders").json()] == ["REJECTED"]


def test_verification_failure_still_cancels_order(demo_env):
    class NoLookup(SandboxLikeBroker):
        def get_order(self, order_id):
            raise RuntimeError("order lookup unavailable")

    demo_env.broker_cls = NoLookup
    status, _ = _failed(demo_env)
    assert status["error"]["code"] == "ORDER_VERIFICATION_FAILED"
    assert status["error"]["headline"] == "ORDER VERIFICATION FAILED"
    assert status["error"]["cancel_required"] is False
    (broker,) = demo_env.brokers
    assert [o.status for o in broker.orders.values()] == [OrderStatus.CANCELLED]
    assert status["order"]["status"] == "CANCELLED"
    assert {s["key"]: s["state"] for s in status["timeline"]}["SANDBOX"] == "failed"


def test_cancellation_failure_requires_attention(demo_env):
    class NoCancel(SandboxLikeBroker):
        def cancel_order(self, order_id):
            raise RuntimeError("cancel endpoint unavailable")

    demo_env.broker_cls = NoCancel
    status, result = _failed(demo_env)
    assert status["error"]["code"] == "ORDER_CANCELLATION_FAILED"
    assert status["error"]["headline"] == "ORDER CANCELLATION REQUIRES ATTENTION"
    assert status["error"]["cancel_required"] is True
    (broker,) = demo_env.brokers
    (order_id,) = broker.orders
    assert broker.orders[order_id].status == OrderStatus.OPEN
    assert result["sandbox_order_id"] == order_id
    assert result["final_cancellation_status"] == "OPEN"
    events = demo_env.client.get("/api/events").json()
    assert any(e["event_type"] == "DEMO_FAILED" and e["severity"] == "CRITICAL" for e in events)


def test_strategy_failure_is_reported(demo_env, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("indicator calculation failed")

    monkeypatch.setattr(sandbox_demo, "evaluate_strategy", broken)
    status, _ = _failed(demo_env)
    assert status["error"]["code"] == "STRATEGY_FAILED"
    assert status["error"]["stage"] == "STRATEGY_EVALUATION"
    assert status["error"]["headline"] == "DEMO FAILED"
    assert "indicator calculation failed" in status["error"]["message"]
    assert demo_env.brokers == []


def test_risk_block_from_actual_risk_manager(demo_env, monkeypatch):
    monkeypatch.setenv("DEMO_EQUITY", "100")
    status, result = _failed(demo_env)
    assert status["error"]["code"] == "RISK_BLOCKED"
    assert status["error"]["headline"] == "DEMO BLOCKED BY RISK"
    assert status["error"]["stage"] == "RISK_VALIDATION"
    assert "CALCULATED_QUANTITY_IS_ZERO" in status["risk"]["reason"]
    assert result["risk_status"] == "BLOCKED"
    assert demo_env.brokers == []
    assert demo_env.client.get("/api/orders").json() == []


def test_sandbox_execution_flag_required(demo_env, monkeypatch):
    monkeypatch.setenv("UPSTOX_RUN_SANDBOX_ORDER_TEST", "false")
    response = demo_env.client.post("/api/demo/start", headers=demo_env.headers)
    assert response.status_code == 403
    assert "UPSTOX_RUN_SANDBOX_ORDER_TEST" in response.json()["detail"]
    assert demo_env.brokers == []


def test_demo_control_endpoints_require_api_key(demo_env, monkeypatch):
    client = demo_env.client
    assert client.post("/api/demo/start").status_code == 401
    assert client.post("/api/demo/start", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/demo/reset").status_code == 401
    monkeypatch.delenv("DASHBOARD_API_KEY")
    assert client.post("/api/demo/start", headers=demo_env.headers).status_code == 503
    assert demo_env.brokers == []


def test_kill_switch_blocks_demo_and_still_works(demo_env):
    client = demo_env.client
    activate = client.post("/api/kill-switch/activate", headers=demo_env.headers, json={"reason": "TEST"})
    assert activate.status_code == 200 and activate.json()["kill_switch_active"] is True
    assert client.get("/api/status").json()["kill_switch_active"] is True

    refused = client.post("/api/demo/start", headers=demo_env.headers)
    assert refused.status_code == 409
    assert "kill switch" in refused.json()["detail"]
    assert demo_env.brokers == []

    resume = client.post("/api/kill-switch/resume", headers=demo_env.headers, json={"reason": "TEST"})
    assert resume.status_code == 200 and resume.json()["kill_switch_active"] is False
    demo_env.run()
    assert len(demo_env.brokers) == 1


def test_existing_dashboard_apis_still_work_after_demo(demo_env):
    demo_env.run()
    client = demo_env.client
    status = client.get("/api/status").json()
    assert status["trading_mode"] == "PAPER" and status["live_trading_enabled"] is False
    assert client.get("/health").json()["status"] == "ok"
    for path in ("/api/positions", "/api/trades", "/api/events", "/api/equity"):
        assert isinstance(client.get(path).json(), list), path
    assert client.get("/api/risk").json()["kill_switch_active"] is False
    config = client.get("/api/config").json()
    assert config["live_trading_enabled"] is False
    page = client.get("/")
    assert "START DEMO" in page.text and "LIVE EXECUTION LOCKED" in page.text
    assert "REAL MONEY: NOT USED" in page.text


def test_demo_status_never_exposes_credentials(demo_env):
    demo_env.run()
    payloads = [
        demo_env.client.get("/api/demo/status").text,
        demo_env.client.get(f"/api/demo/result/{demo_env.controller.status()['demo_run_id']}").text,
        demo_env.client.get("/").text,
        demo_env.client.get("/dashboard/app.js").text,
    ]
    for text in payloads:
        assert SECRET not in text
        assert API_KEY not in text
