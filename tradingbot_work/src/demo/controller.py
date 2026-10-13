"""Background, single-run controller for the UI-triggered Sandbox demo."""
from __future__ import annotations

import copy
import os
import threading
import weakref
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from src.logging.logger import _redact
from src.safety.live_gate import live_trading_enabled, trading_mode

from . import sandbox_demo
from .sandbox_demo import (
    DEFAULT_CONFIG_PATH,
    DEMO_TIMEFRAME,
    DemoStageError,
    assert_demo_safe,
)

PROGRESS = {
    "IDLE": 0,
    "STARTING": 5,
    "GENERATING_MARKET_DATA": 15,
    "STRATEGY_EVALUATION": 25,
    "SIGNAL_GENERATED": 35,
    "RISK_VALIDATION": 45,
    "BROKER_CONNECTING": 55,
    "ORDER_SUBMITTING": 65,
    "ORDER_VERIFIED": 78,
    "ORDER_CANCELLING": 88,
    "COMPLETED": 100,
    "FAILED": 100,
}

TIMELINE = (
    ("SYSTEM", "System ready"),
    ("MARKET_DATA", "Market data"),
    ("STRATEGY", "Strategy"),
    ("RISK", "Risk engine"),
    ("ORDER_MANAGER", "Order manager"),
    ("SANDBOX", "Upstox Sandbox"),
    ("EXIT", "Exit"),
    ("COMPLETE", "Complete"),
)

_STEP_FOR_STAGE = {
    "STARTING": "SYSTEM",
    "GENERATING_MARKET_DATA": "MARKET_DATA",
    "STRATEGY_EVALUATION": "STRATEGY",
    "SIGNAL_GENERATED": "STRATEGY",
    "RISK_VALIDATION": "RISK",
    "BROKER_CONNECTING": "ORDER_MANAGER",
    "ORDER_SUBMITTING": "ORDER_MANAGER",
    "ORDER_VERIFIED": "SANDBOX",
    "ORDER_CANCELLING": "EXIT",
}

_STEP_FOR_CODE = {
    "ORDER_VERIFICATION_FAILED": "SANDBOX",
    "ORDER_CANCELLATION_FAILED": "EXIT",
}

# Audit events written when a timeline step actually completes.
STAGE_EVENTS = {
    "MARKET_DATA": "MARKET_DATA_READY",
    "STRATEGY": "SIGNAL_GENERATED",
    "CONTRACT": "OPTION_RESOLVED",
    "RISK": "RISK_VALIDATED",
    "ORDER_MANAGER": "ORDER_SUBMITTED",
    "SANDBOX": "ORDER_VERIFIED",
    "EXIT": "ORDER_CANCELLED",
}

HEADLINES = {
    "SAFETY_REFUSED": "DEMO REFUSED - SAFETY",
    "STRATEGY_FAILED": "DEMO FAILED",
    "NO_SIGNAL": "DEMO FAILED",
    "RISK_BLOCKED": "DEMO BLOCKED BY RISK",
    "SANDBOX_EXECUTION_DISABLED": "DEMO FAILED",
    "BROKER_CONNECTION_FAILED": "BROKER CONNECTION FAILED",
    "ORDER_SUBMISSION_FAILED": "ORDER SUBMISSION FAILED",
    "ORDER_VERIFICATION_FAILED": "ORDER VERIFICATION FAILED",
    "ORDER_CANCELLATION_FAILED": "ORDER CANCELLATION REQUIRES ATTENTION",
    "UNEXPECTED_ERROR": "DEMO FAILED",
}


class DemoAlreadyRunning(RuntimeError):
    pass


# Equity and option demos share the Sandbox account: only one may run.
_RUN_GUARD = threading.Lock()
_CONTROLLERS: "weakref.WeakSet[DemoController]" = weakref.WeakSet()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(text: Any) -> str:
    message = _redact(str(text))
    for name in ("UPSTOX_SANDBOX_ACCESS_TOKEN", "UPSTOX_ACCESS_TOKEN", "DASHBOARD_API_KEY"):
        secret = os.getenv(name, "").strip()
        if len(secret) >= 8:
            message = message.replace(secret, "***REDACTED***")
    return message[:500]


def _sandbox_broker():
    from src.broker.upstox_sandbox import UpstoxSandboxBroker

    return UpstoxSandboxBroker()


class DemoController:
    PROGRESS = PROGRESS
    TIMELINE = TIMELINE
    STEP_FOR_STAGE = _STEP_FOR_STAGE
    STEP_FOR_CODE = _STEP_FOR_CODE
    HEADLINES = HEADLINES
    RUN_PREFIX = "DEMO-"
    RESULT_PREFIX = "demo_result:"
    LAST_RUN_KEY = "demo_last_run_id"
    EVENT_PREFIX = "DEMO"
    LABEL = "Sandbox demo"

    def __init__(
        self,
        database,
        broker_factory: Callable[[], Any] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        max_results: int = 20,
    ) -> None:
        self.database = database
        self.broker_factory = broker_factory or _sandbox_broker
        self.config_path = config_path
        self.max_results = max_results
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._results: OrderedDict[str, dict] = OrderedDict()
        self._state = self._initial_state(None)
        _CONTROLLERS.add(self)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _initial_state(self, run_id: str | None, running: bool = False) -> dict:
        return {
            "running": running,
            "demo_run_id": run_id,
            "stage": "STARTING" if running else "IDLE",
            "progress": self.PROGRESS["STARTING"] if running else 0,
            "message": "Demo starting" if running else "Ready",
            "headline": "DEMO RUNNING" if running else "SYSTEM READY",
            "started_at": _now() if running else None,
            "finished_at": None,
            "signal": None,
            "risk": None,
            "order": None,
            "broker": {"name": "UPSTOX SANDBOX", "status": "IDLE", "order": None},
            "market_data": {"source": "PAPER / DEMO", "candles": 0, "timeframe": DEMO_TIMEFRAME},
            "system": {
                "status": "DEMO RUNNING" if running else "READY",
                "broker": "SANDBOX",
                "market_data": "PAPER / DEMO",
                "live_trading": "ENABLED" if live_trading_enabled() else "DISABLED",
            },
            "timeline": [
                {"key": key, "label": label,
                 "state": "active" if running and key == "SYSTEM" else "pending",
                 "detail": None}
                for key, label in self.TIMELINE
            ],
            "error": None,
            "trading_mode": trading_mode(),
            "live_trading_enabled": live_trading_enabled(),
        }

    def _update(self, **changes) -> None:
        with self._lock:
            stage = changes.get("stage")
            if stage:
                changes.setdefault("progress", self.PROGRESS.get(stage, self._state["progress"]))
            for key, value in changes.items():
                if key in ("system", "broker", "market_data") and isinstance(value, dict) \
                        and isinstance(self._state.get(key), dict):
                    self._state[key].update(value)
                else:
                    self._state[key] = value

    def _step(self, key: str, state: str, detail: str | None = None) -> None:
        with self._lock:
            run_id = self._state.get("demo_run_id")
            for step in self._state["timeline"]:
                if step["key"] == key:
                    step["state"] = state
                    if detail is not None:
                        step["detail"] = detail
        event_type = STAGE_EVENTS.get(key)
        if state == "done" and event_type:
            self._event(event_type, "INFO", detail or event_type.replace("_", " ").capitalize(),
                        demo_run_id=run_id, demo=self.EVENT_PREFIX)

    def status(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._state)

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._state["running"])

    def result(self, run_id: str) -> dict | None:
        with self._lock:
            found = self._results.get(run_id)
            if found is not None:
                return copy.deepcopy(found)
        if self.database is None:
            return None
        return self.database.get_state(self.RESULT_PREFIX + run_id)

    def last_result(self) -> dict | None:
        with self._lock:
            run_id = self._state.get("demo_run_id")
        return self.result(run_id) if run_id else None

    def wait(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self, **params) -> dict:
        assert_demo_safe()
        with _RUN_GUARD:
            with self._lock:
                if self._state["running"]:
                    raise DemoAlreadyRunning("Demo already running")
            for other in list(_CONTROLLERS):
                if other is not self and other.running:
                    raise DemoAlreadyRunning(f"{other.LABEL} already running")
            with self._lock:
                run_id = self.RUN_PREFIX + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
                self._state = self._initial_state(run_id, running=True)
                thread = threading.Thread(
                    target=self._run, args=(run_id,), kwargs=params,
                    name=self.RUN_PREFIX.lower().strip("-") + "-run", daemon=True)
                self._thread = thread
        self._event(f"{self.EVENT_PREFIX}_STARTED", "INFO", f"UI-triggered {self.LABEL} started",
                    demo_run_id=run_id, **params)
        thread.start()
        return self.status()

    def reset(self) -> dict:
        """Return the console to IDLE. Never touches broker orders."""
        with self._lock:
            if self._state["running"]:
                raise DemoAlreadyRunning("Demo already running")
            self._state = self._initial_state(None)
        return self.status()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _event(self, event_type: str, severity: str, message: str, **metadata) -> None:
        if self.database is None:
            return
        try:
            self.database.add_event(event_type, severity, message, metadata)
        except Exception:
            pass

    def _run(self, run_id: str) -> None:
        symbol = sandbox_demo.demo_symbol()
        instrument = sandbox_demo.sandbox_instrument()
        equity = sandbox_demo.demo_equity()
        outcome: dict | None = None
        try:
            try:
                assert_demo_safe()
            except Exception as exc:
                raise DemoStageError("STARTING", "SAFETY_REFUSED", str(exc), cause=exc) from exc
            self._step("SYSTEM", "done", "PAPER mode, live trading disabled")

            self._update(stage="GENERATING_MARKET_DATA", message="Loading demo candles")
            self._step("MARKET_DATA", "active")
            candles = sandbox_demo.build_demo_candles(symbol)
            self._update(market_data={"candles": len(candles), "symbol": symbol})
            self._step("MARKET_DATA", "done", f"{len(candles)} demo candles loaded ({symbol}, {DEMO_TIMEFRAME})")

            self._update(stage="STRATEGY_EVALUATION", message="Feeding candles into StrategyRuntime")
            self._step("STRATEGY", "active")
            try:
                strategy = sandbox_demo.evaluate_strategy(
                    candles, equity=equity, config_path=self.config_path)
            except Exception as exc:
                raise DemoStageError(
                    "STRATEGY_EVALUATION", "STRATEGY_FAILED", f"{type(exc).__name__}: {exc}", cause=exc,
                ) from exc
            signal = strategy.signal
            if signal is None:
                raise DemoStageError(
                    "STRATEGY_EVALUATION", "NO_SIGNAL", "StrategyRuntime produced no signal for the demo candles")

            summary = sandbox_demo.signal_summary(signal)
            self._update(stage="SIGNAL_GENERATED", signal=summary,
                         message=f"{summary['side']} signal generated by StrategyRuntime")
            self._step("STRATEGY", "done", f"{summary['side']} signal generated")

            self._update(stage="RISK_VALIDATION", message="Running RiskManager entry validation")
            self._step("RISK", "active")
            check = sandbox_demo.preview_risk(signal, strategy.risk_manager, strategy.session_manager, equity)
            limits = strategy.config.risk.model_dump()
            allowed = check["allowed"] and not signal.blocked
            reasons = check["reasons"] or ([signal.block_reason] if signal.block_reason else [])
            risk = {
                "status": "ALLOWED" if allowed else "BLOCKED",
                "signal_risk_status": signal.risk_status,
                "reason": signal.reason if allowed else ";".join(r for r in reasons if r),
                "reasons": reasons,
                "risk_amount": check["risk_amount"],
                "estimated_exposure": check["estimated_exposure"],
                "limits": limits,
            }
            self._update(risk=risk)
            if not allowed:
                raise DemoStageError("RISK_VALIDATION", "RISK_BLOCKED", risk["reason"] or "Risk validation failed")
            if summary["signal_type"] != "ENTRY":
                raise DemoStageError("RISK_VALIDATION", "RISK_BLOCKED", "Demo only executes ENTRY signals")
            self._step("RISK", "done", "Risk ALLOWED")

            if not sandbox_demo.sandbox_order_execution_enabled():
                raise DemoStageError(
                    "BROKER_CONNECTING", "SANDBOX_EXECUTION_DISABLED",
                    "Set UPSTOX_RUN_SANDBOX_ORDER_TEST=true to allow the controlled Sandbox order")

            broker = self.broker_factory()
            outcome = sandbox_demo.execute_sandbox_order(
                signal,
                broker=broker,
                risk_manager=strategy.risk_manager,
                session_manager=strategy.session_manager,
                database=self.database,
                instrument=instrument,
                equity=equity,
                max_slippage_percent=0.10,
                on_stage=self._on_stage,
            )

            self._update(
                stage="COMPLETED", finished_at=_now(),
                headline="DEMO COMPLETED", message="Demo completed successfully",
                system={"status": "DEMO COMPLETED", "broker": "SANDBOX"},
                broker={"status": "DISCONNECTED"},
            )
            self._step("COMPLETE", "done", "Demo successful")
        except DemoStageError as exc:
            self._fail(exc.stage, exc.code, exc.message, exc.cancel_required, exc.order, exc.broker)
        except Exception as exc:
            with self._lock:
                stage = self._state["stage"]
            self._fail(stage, "UNEXPECTED_ERROR", f"{type(exc).__name__}: {exc}", False, None, None)
        finally:
            try:
                self._finalise(run_id, symbol, instrument, outcome)
            finally:
                with self._lock:
                    self._state["running"] = False

    def _on_stage(self, stage: str, **data) -> None:
        if stage == "BROKER_CONNECTING":
            self._update(stage="BROKER_CONNECTING", message="Connecting to Upstox Sandbox",
                         broker={"status": "CONNECTING"})
            self._step("ORDER_MANAGER", "active")
        elif stage == "BROKER_CONNECTED":
            self._update(message="Upstox Sandbox connected", broker={"status": "SANDBOX CONNECTED"},
                         system={"broker": "SANDBOX CONNECTED"})
        elif stage == "ORDER_SUBMITTING":
            self._update(stage="ORDER_SUBMITTING", message="Submitting through OrderManager")
        elif stage == "ORDER_SUBMITTED":
            order = dict(data["order"])
            self._update(order=order, message=f"Sandbox order {order['order_id']} {order['status']}")
            self._step("ORDER_MANAGER", "done", f"Order submitted: {order['order_id']} ({order['status']})")
            self._step("SANDBOX", "active")
        elif stage == "ORDER_VERIFIED":
            verified = dict(data["broker"])
            self._update(stage="ORDER_VERIFIED", broker={"order": verified},
                         message=f"Broker verified order status {verified['status']}")
            self._step("SANDBOX", "done", f"Broker verified: {verified['status']}")
        elif stage == "ORDER_CANCELLING":
            self._update(stage="ORDER_CANCELLING", message="Cancelling controlled Sandbox order")
            self._step("EXIT", "active")
        elif stage == "ORDER_CANCELLED":
            order = dict(data["order"])
            self._update(order=order, message=f"Sandbox order {order['order_id']} {order['status']}")
            self._step("EXIT", "done", f"Order {order['status'].lower()}")

    def _fail(self, stage, code, message, cancel_required, order, broker_order) -> None:
        headline = self.HEADLINES.get(code, "DEMO FAILED")
        error = {
            "stage": stage,
            "code": code,
            "headline": headline,
            "message": _sanitize(message),
            "cancel_required": bool(cancel_required),
        }
        changes: dict[str, Any] = dict(
            stage="FAILED", finished_at=_now(), error=error,
            headline=headline, message=error["message"],
            system={"status": "DEMO FAILED", "broker": "SANDBOX"},
        )
        if order is not None:
            changes["order"] = dict(order)
        if broker_order is not None:
            changes["broker"] = {"order": dict(broker_order)}
        with self._lock:
            if self._state["broker"]["status"] in ("CONNECTING", "SANDBOX CONNECTED"):
                changes.setdefault("broker", {})["status"] = "DISCONNECTED"
        self._update(**changes)
        step = self.STEP_FOR_CODE.get(code) or self.STEP_FOR_STAGE.get(stage, "SYSTEM")
        self._step(step, "failed", error["message"])

    def _store_result(self, run_id: str, result: dict) -> None:
        with self._lock:
            self._results[run_id] = result
            while len(self._results) > self.max_results:
                self._results.popitem(last=False)
        if self.database is not None:
            try:
                self.database.set_state(self.RESULT_PREFIX + run_id, result)
                self.database.set_state(self.LAST_RUN_KEY, run_id)
            except Exception:
                pass

    def _finalise(self, run_id: str, symbol: str, instrument: str, outcome: dict | None) -> None:
        state = self.status()
        signal = state["signal"] or {}
        risk = state["risk"] or {}
        order = state["order"] or {}
        submitted = (outcome or {}).get("submitted") or {}
        verified = state["broker"].get("order") or {}
        error = state["error"]
        success = state["stage"] == "COMPLETED" and order.get("status") == "CANCELLED"

        result = {
            "demo_run_id": run_id,
            "started_at": state["started_at"],
            "finished_at": state["finished_at"],
            "stage": state["stage"],
            "strategy_symbol": symbol,
            "timeframe": DEMO_TIMEFRAME,
            "candles": state["market_data"].get("candles", 0),
            "signal_id": signal.get("signal_id"),
            "signal_side": signal.get("side"),
            "entry_price": signal.get("entry_price"),
            "quantity": signal.get("quantity"),
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target"),
            "conditions": signal.get("conditions"),
            "risk_status": risk.get("status"),
            "risk_reason": risk.get("reason"),
            "broker": "UPSTOX SANDBOX",
            "broker_instrument": instrument,
            "sandbox_order_id": order.get("order_id"),
            "order_type": order.get("order_type"),
            "submitted_status": submitted.get("status"),
            "broker_order_status": verified.get("status"),
            "requested_quantity": order.get("requested_quantity"),
            "filled_quantity": order.get("filled_quantity"),
            "average_fill": order.get("average_fill_price"),
            "final_cancellation_status": order.get("status") if order else None,
            "execution_result": "SUCCESS" if success else "FAILED",
            "error": error,
            "live_trading_enabled": live_trading_enabled(),
            "trading_mode": trading_mode(),
        }
        self._store_result(run_id, result)
        if success:
            self._event("DEMO_COMPLETED", "INFO", "Sandbox demo completed: order submitted, verified and cancelled",
                        demo_run_id=run_id, order_id=result["sandbox_order_id"])
        else:
            self._event("DEMO_FAILED", "CRITICAL" if (error or {}).get("cancel_required") else "WARNING",
                        (error or {}).get("headline", "DEMO FAILED"),
                        demo_run_id=run_id, order_id=result["sandbox_order_id"],
                        code=(error or {}).get("code"), stage=(error or {}).get("stage"))
