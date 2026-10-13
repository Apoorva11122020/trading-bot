"""UI-triggered option Sandbox demo (BUY CE / BUY PE on a real NSE_FO contract).

underlying data -> StrategyEngine -> signal -> expiry -> strike -> CE/PE ->
contract (Upstox instrument master) -> option risk -> lot validation ->
OrderManager -> Upstox Sandbox -> verify -> cancel -> database.

Option resolution and Sandbox order execution are reported separately so a
data problem is never mistaken for (or hidden behind) an execution result.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from src.config.loader import load_strategy_config
from src.demo import sandbox_demo
from src.demo.controller import HEADLINES, DemoController, _now
from src.demo.sandbox_demo import DEFAULT_CONFIG_PATH, DemoStageError, assert_demo_safe
from src.models.risk import AccountState
from src.risk.risk_manager import RiskManager
from src.safety.live_gate import live_trading_enabled, trading_mode
from src.session.session_manager import SessionManager
from src.strategy.strategy_engine import StrategyEngine

from .demo_data import DEMO_DATA_LABEL, demo_premium_candles, demo_underlying_candles
from .models import MarketType, OptionResolutionError, OptionType
from .option_chain import DEMO_REFERENCE_SPOT, demo_reference_spot
from .selection import select_expiry, select_strike
from .strategy import OptionSignalEngine, OptionSignalError

OPTION_NOTICE = "Option execution is currently validated in Paper/Sandbox mode."
OPTION_TIMEFRAME = "5m"

OPTION_PROGRESS = {
    "IDLE": 0,
    "STARTING": 4,
    "UNDERLYING_DATA": 10,
    "STRATEGY_EVALUATION": 18,
    "SIGNAL_GENERATED": 25,
    "EXPIRY_RESOLVED": 32,
    "STRIKE_RESOLVED": 38,
    "OPTION_RESOLVED": 45,
    "RISK_VALIDATION": 52,
    "LOT_SIZE_VALIDATED": 58,
    "BROKER_CONNECTING": 64,
    "ORDER_SUBMITTING": 72,
    "ORDER_VERIFIED": 82,
    "ORDER_CANCELLING": 90,
    "COMPLETED": 100,
    "FAILED": 100,
}

OPTION_TIMELINE = (
    ("SYSTEM", "System ready"),
    ("MARKET_DATA", "Underlying data"),
    ("STRATEGY", "Strategy"),
    ("CONTRACT", "Option contract"),
    ("RISK", "Risk & lot size"),
    ("ORDER_MANAGER", "Order manager"),
    ("SANDBOX", "Upstox Sandbox"),
    ("EXIT", "Exit"),
    ("COMPLETE", "Complete"),
)

OPTION_STEP_FOR_STAGE = {
    "STARTING": "SYSTEM",
    "UNDERLYING_DATA": "MARKET_DATA",
    "STRATEGY_EVALUATION": "STRATEGY",
    "SIGNAL_GENERATED": "STRATEGY",
    "EXPIRY_RESOLVED": "CONTRACT",
    "STRIKE_RESOLVED": "CONTRACT",
    "OPTION_RESOLVED": "CONTRACT",
    "RISK_VALIDATION": "RISK",
    "LOT_SIZE_VALIDATED": "RISK",
    "BROKER_CONNECTING": "ORDER_MANAGER",
    "ORDER_SUBMITTING": "ORDER_MANAGER",
    "ORDER_VERIFIED": "SANDBOX",
    "ORDER_CANCELLING": "EXIT",
}

RESOLUTION_CODES = (
    "INSTRUMENT_DATA_UNAVAILABLE", "INSTRUMENT_SOURCE_REFUSED", "UNDERLYING_NOT_FOUND", "MARKET_TYPE_MISMATCH",
    "NO_EXPIRIES", "EXPIRY_NOT_FOUND", "INVALID_EXPIRY", "INVALID_EXPIRY_MODE", "NO_STRIKES", "STRIKE_NOT_FOUND",
    "STRIKE_OUT_OF_RANGE", "INVALID_STRIKE_MODE", "INVALID_STRIKE_OFFSET", "CONTRACT_NOT_FOUND",
    "OPTION_TYPE_NOT_FOUND", "OPTION_TYPE_REQUIRED", "SPOT_REQUIRED", "NOT_NSE_FO", "INVALID_LOT_SIZE",
)

OPTION_HEADLINES = {
    **HEADLINES,
    "SAFETY_REFUSED": "OPTION DEMO REFUSED - SAFETY",
    "OPTIONS_DISABLED": "OPTION DEMO DISABLED",
    "NO_SIGNAL": "OPTION DEMO - NO SIGNAL",
    "RISK_BLOCKED": "OPTION DEMO BLOCKED BY RISK",
    "LOT_SIZE_INVALID": "OPTION DEMO BLOCKED - LOT SIZE",
    "OPTION_PREMIUM_ATR_UNAVAILABLE": "OPTION STOP UNAVAILABLE",
    "UNDERLYING_ATR_UNAVAILABLE": "OPTION STOP UNAVAILABLE",
    "OPTION_STOP_INVALID": "OPTION STOP UNAVAILABLE",
    "OPTION_PREMIUM_DATA_UNAVAILABLE": "OPTION DATA UNAVAILABLE",
    "SANDBOX_EXECUTION_DISABLED": "SANDBOX EXECUTION DISABLED",
    "UNEXPECTED_ERROR": "OPTION DEMO FAILED",
    **{code: "OPTION RESOLUTION FAILED" for code in RESOLUTION_CODES},
}


def option_demo_equity() -> float:
    return float(os.getenv("DEMO_OPTION_EQUITY", "1000000"))


def _iso(value) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else value


class OptionDemoController(DemoController):
    PROGRESS = OPTION_PROGRESS
    TIMELINE = OPTION_TIMELINE
    STEP_FOR_STAGE = OPTION_STEP_FOR_STAGE
    STEP_FOR_CODE = {"ORDER_VERIFICATION_FAILED": "SANDBOX", "ORDER_CANCELLATION_FAILED": "EXIT"}
    HEADLINES = OPTION_HEADLINES
    RUN_PREFIX = "OPTDEMO-"
    RESULT_PREFIX = "option_demo_result:"
    LAST_RUN_KEY = "option_demo_last_run_id"
    EVENT_PREFIX = "OPTION_DEMO"
    LABEL = "Option Sandbox demo"

    def __init__(
        self,
        database,
        resolver_factory: Callable[[], Any],
        broker_factory: Callable[[], Any] | None = None,
        config_path=DEFAULT_CONFIG_PATH,
        max_results: int = 20,
    ) -> None:
        self.resolver_factory = resolver_factory
        super().__init__(database, broker_factory=broker_factory, config_path=config_path, max_results=max_results)

    def _initial_state(self, run_id, running: bool = False) -> dict:
        state = super()._initial_state(run_id, running)
        state.update({
            "message": "Option demo starting" if running else "Ready",
            "headline": "OPTION DEMO · SANDBOX" if running else "SYSTEM READY",
            "request": None,
            "market_type": None,
            "underlying": None,
            "spot": None,
            "spot_source": None,
            "expiry": None,
            "strike": None,
            "option_type": None,
            "instrument_key": None,
            "trading_symbol": None,
            "lot_size": None,
            "freeze_quantity": None,
            "contract": None,
            "decision": None,
            "option_resolution": {"status": "PENDING" if running else "IDLE", "source": None, "message": None},
            "sandbox_execution": {"status": "PENDING" if running else "IDLE", "message": None},
            "data_labels": {"underlying": DEMO_DATA_LABEL, "premium": DEMO_DATA_LABEL},
            "notice": OPTION_NOTICE,
        })
        state["market_data"] = {"source": DEMO_DATA_LABEL, "candles": 0, "timeframe": OPTION_TIMEFRAME}
        state["system"]["status"] = "OPTION DEMO RUNNING" if running else "READY"
        state["system"]["market_data"] = DEMO_DATA_LABEL
        return state

    def _resolution(self, status: str, message: str | None = None, **extra) -> None:
        with self._lock:
            self._state["option_resolution"].update({"status": status, "message": message, **extra})

    def _execution(self, status: str, message: str | None = None) -> None:
        with self._lock:
            self._state["sandbox_execution"].update({"status": status, "message": message})

    # ------------------------------------------------------------------

    def _run(self, run_id: str, **params) -> None:
        outcome: dict | None = None
        underlying = str(params.get("underlying") or "").upper()
        contract = None
        try:
            try:
                assert_demo_safe()
            except Exception as exc:
                raise DemoStageError("STARTING", "SAFETY_REFUSED", str(exc), cause=exc) from exc

            config = load_strategy_config(self.config_path)
            if not config.options.enabled:
                raise DemoStageError("STARTING", "OPTIONS_DISABLED", "options.enabled is false in config/strategy.yaml")
            self._update(request=dict(params), underlying=underlying, market_type=params.get("market_type"))
            self._step("SYSTEM", "done", "PAPER mode, live trading disabled")

            signal_source = str(params.get("signal_source") or "UNDERLYING").upper()
            option_mode = str(params.get("option_type") or "AUTO").upper()
            trend = str(params.get("demo_trend") or "AUTO").upper()
            bearish = option_mode == "PE" or (option_mode == "AUTO" and trend == "BEARISH")

            # ---------------- Underlying data ----------------
            self._update(stage="UNDERLYING_DATA", message="Loading option instruments and underlying demo data")
            self._step("MARKET_DATA", "active")
            try:
                resolver = self.resolver_factory()
                listed = resolver.contracts(underlying)
            except OptionResolutionError as exc:
                self._resolution("FAILED", exc.message)
                raise DemoStageError("UNDERLYING_DATA", exc.code, exc.message, cause=exc) from exc
            kind = listed[0].market_type
            requested = params.get("market_type")
            if requested and MarketType(requested) != kind:
                message = f"{underlying} is a {kind.value}, not a {requested}"
                self._resolution("FAILED", message)
                raise DemoStageError("UNDERLYING_DATA", "MARKET_TYPE_MISMATCH", message)
            self._resolution("IN_PROGRESS", None, source=resolver.metadata.get("source"))

            expiries = resolver.expiries(underlying)
            spot = params.get("spot")
            spot_source = "REQUEST"
            if not spot:
                spot = demo_reference_spot(c.strike for c in listed if c.expiry == expiries[0])
                spot_source = DEMO_REFERENCE_SPOT
            spot = float(spot)
            candles = demo_underlying_candles(underlying, spot, bearish=bearish)
            self._update(
                market_type=kind.value, spot=spot, spot_source=spot_source,
                market_data={"candles": len(candles), "symbol": underlying, "spot": spot,
                             "spot_source": spot_source, "trend": "BEARISH" if bearish else "BULLISH"},
            )
            self._step("MARKET_DATA", "done",
                       f"{len(candles)} {DEMO_DATA_LABEL} candles ({underlying}, reference spot {spot:g})")

            risk_manager = RiskManager(config.risk)
            session_manager = SessionManager(config.session)
            engine = OptionSignalEngine(
                config, StrategyEngine(config, risk_manager, session_manager), risk_manager, session_manager)

            decision = None
            if signal_source == "UNDERLYING":
                decision = self._evaluate(engine, candles, option_mode)
                option_type = decision.option_type
            else:
                if option_mode not in ("CE", "PE"):
                    raise DemoStageError(
                        "STRATEGY_EVALUATION", "OPTION_TYPE_REQUIRED",
                        "Option-candle signals need an explicit CE or PE option type")
                option_type = OptionType(option_mode)

            # ---------------- Contract resolution ----------------
            self._step("CONTRACT", "active")
            resolving = "EXPIRY_RESOLVED"
            try:
                expiry = select_expiry(expiries, params.get("expiry_mode") or "nearest", params.get("expiry_date"))
                self._update(stage="EXPIRY_RESOLVED", expiry=expiry.isoformat(),
                             message=f"Expiry resolved: {expiry.isoformat()}")
                resolving = "STRIKE_RESOLVED"
                strikes = resolver.strikes(underlying, expiry, option_type)
                strike = select_strike(
                    strikes, option_type=option_type, mode=params.get("strike_mode") or "ATM", spot=spot,
                    offset=int(params.get("strike_offset") or 0), exact=params.get("strike"))
                self._update(stage="STRIKE_RESOLVED", strike=strike, option_type=option_type.value,
                             message=f"Strike resolved: {strike:g} {option_type.value}")
                resolving = "OPTION_RESOLVED"
                contract = resolver.resolve(underlying, expiry, strike, option_type)
            except OptionResolutionError as exc:
                self._resolution("FAILED", exc.message)
                raise DemoStageError(resolving, exc.code, exc.message, cause=exc) from exc
            self._update(
                stage="OPTION_RESOLVED", contract=contract.as_dict(), instrument_key=contract.instrument_key,
                trading_symbol=contract.trading_symbol, lot_size=contract.lot_size,
                freeze_quantity=contract.freeze_quantity,
                message=f"Contract resolved: {contract.trading_symbol} ({contract.instrument_key})",
            )
            self._resolution("RESOLVED", f"{contract.trading_symbol} from {resolver.metadata.get('source')}")
            self._step("CONTRACT", "done", f"{contract.trading_symbol} · lot {contract.lot_size}")

            premium = demo_premium_candles(contract.instrument_key, candles, contract.option_type, spot,
                                           contract.tick_size)
            if decision is None:
                decision = self._evaluate(engine, premium, option_mode, option_candles=True)

            # ---------------- Risk & lot size ----------------
            self._update(stage="RISK_VALIDATION", message="Running option risk validation")
            self._step("RISK", "active")
            equity = option_demo_equity()
            try:
                signal = engine.build_signal(
                    contract=contract, decision=decision, premium_candles=premium,
                    account=AccountState(equity=equity), spot=spot,
                    details={"premium_source": DEMO_DATA_LABEL, "spot_source": spot_source})
            except OptionSignalError as exc:
                raise DemoStageError("RISK_VALIDATION", exc.code, exc.message, cause=exc) from exc

            summary = signal.summary()
            check = signal.details["risk"]
            risk = {
                "status": signal.risk_status,
                "reason": signal.block_reason or signal.reason,
                "reasons": check["reasons"],
                "equity": equity,
                "entry_premium": signal.entry_price,
                "stop_premium": signal.stop_loss,
                "target_premium": signal.target,
                "risk_per_unit": signal.risk_per_unit,
                "lots": signal.lots,
                "lot_size": signal.lot_size,
                "quantity": signal.quantity,
                "max_loss": signal.max_loss,
                "premium_exposure": signal.premium_exposure,
                "daily_loss_limit": check["daily_loss_limit"],
                "sizing": signal.details["sizing"],
                "limits": {**config.risk.model_dump(), **config.options.risk.model_dump()},
                "premium_source": DEMO_DATA_LABEL,
            }
            self._update(signal=summary, risk=risk)
            if signal.blocked:
                raise DemoStageError("RISK_VALIDATION", "RISK_BLOCKED", signal.block_reason or "Option risk blocked")
            if signal.quantity <= 0 or signal.quantity % contract.lot_size or (
                    contract.freeze_quantity and signal.quantity > contract.freeze_quantity):
                raise DemoStageError("LOT_SIZE_VALIDATED", "LOT_SIZE_INVALID",
                                     f"Quantity {signal.quantity} is not a valid lot multiple")
            self._update(stage="LOT_SIZE_VALIDATED",
                         message=f"{signal.lots} lot(s) x {contract.lot_size} = {signal.quantity}")
            self._step("RISK", "done",
                       f"ALLOWED · {signal.lots} lot(s) x {contract.lot_size} = {signal.quantity}")

            # ---------------- Sandbox execution ----------------
            if not sandbox_demo.sandbox_order_execution_enabled():
                self._execution("NOT_ATTEMPTED", "UPSTOX_RUN_SANDBOX_ORDER_TEST is not true")
                raise DemoStageError(
                    "BROKER_CONNECTING", "SANDBOX_EXECUTION_DISABLED",
                    "Set UPSTOX_RUN_SANDBOX_ORDER_TEST=true to allow the controlled Sandbox order")

            broker = self.broker_factory()
            self._execution("IN_PROGRESS")
            outcome = sandbox_demo.execute_sandbox_order(
                signal.to_order_signal(),
                broker=broker,
                risk_manager=risk_manager,
                session_manager=session_manager,
                database=self.database,
                instrument=contract.instrument_key,
                equity=equity,
                max_slippage_percent=config.execution.max_slippage_percent,
                on_stage=self._on_stage,
                instrument_meta=contract.order_metadata(),
                order_type=config.options.execution.order_type,
                order_params={"product": config.options.execution.product},
            )
            self._execution("CANCELLED", "Order submitted, verified and cancelled")
            self._update(
                stage="COMPLETED", finished_at=_now(),
                headline="OPTION DEMO COMPLETED", message="Option demo completed successfully",
                system={"status": "OPTION DEMO COMPLETED", "broker": "SANDBOX"},
                broker={"status": "DISCONNECTED"},
            )
            self._step("COMPLETE", "done", "Option demo successful")
        except DemoStageError as exc:
            self._close_execution(exc.message)
            self._fail(exc.stage, exc.code, exc.message, exc.cancel_required, exc.order, exc.broker)
        except Exception as exc:
            self._close_execution(f"{type(exc).__name__}: {exc}")
            stage = self.status()["stage"]
            self._fail(stage, "UNEXPECTED_ERROR", f"{type(exc).__name__}: {exc}", False, None, None)
        finally:
            try:
                self._finalise(run_id, underlying, contract.instrument_key if contract else None, outcome)
            finally:
                with self._lock:
                    self._state["running"] = False

    def _close_execution(self, message: str) -> None:
        status = self.status()["sandbox_execution"]["status"]
        if status in ("IN_PROGRESS", "SUBMITTED", "VERIFIED"):
            self._execution("FAILED", message)
        elif status == "PENDING":
            self._execution("NOT_ATTEMPTED", "Stopped before Sandbox execution")
        if self.status()["option_resolution"]["status"] in ("PENDING", "IN_PROGRESS"):
            self._resolution("NOT_COMPLETED", message)

    def _evaluate(self, engine: OptionSignalEngine, candles, option_mode: str, option_candles: bool = False):
        self._update(stage="STRATEGY_EVALUATION",
                     message="Evaluating option premium candles" if option_candles
                     else "Evaluating the underlying with StrategyEngine")
        self._step("STRATEGY", "active")
        try:
            decision = (engine.option_decision(candles, option_mode) if option_candles
                        else engine.underlying_decision(candles, option_mode))
        except OptionSignalError as exc:
            raise DemoStageError("STRATEGY_EVALUATION", exc.code, exc.message, cause=exc) from exc
        info = {
            "source": decision.source,
            "side": decision.side.value if decision.side else None,
            "option_type": decision.option_type.value if decision.option_type else None,
            "action": ("BUY CALL" if decision.option_type == OptionType.CE else "BUY PUT")
            if decision.tradeable else None,
            "reason": decision.reason,
            "conditions": decision.conditions,
            "timestamp": _iso(decision.timestamp),
            "spot": decision.spot,
            "underlying_atr": decision.underlying_atr,
        }
        self._update(decision=info)
        if not decision.tradeable:
            raise DemoStageError("STRATEGY_EVALUATION", "NO_SIGNAL", f"No option signal: {decision.reason}")
        self._update(stage="SIGNAL_GENERATED", option_type=decision.option_type.value,
                     message=f"{info['action']} signal generated ({decision.reason})")
        self._step("STRATEGY", "done", f"{info['action']} · {decision.reason}")
        return decision

    def _on_stage(self, stage: str, **data) -> None:
        super()._on_stage(stage, **data)
        if stage == "ORDER_SUBMITTED":
            self._execution("SUBMITTED", f"Order {data['order']['order_id']} submitted")
        elif stage == "ORDER_VERIFIED":
            self._execution("VERIFIED", f"Broker status {data['broker']['status']}")

    def _finalise(self, run_id: str, underlying: str, instrument: str | None, outcome: dict | None) -> None:
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
            "demo_type": "OPTION",
            "started_at": state["started_at"],
            "finished_at": state["finished_at"],
            "stage": state["stage"],
            "request": state["request"],
            "market_type": state["market_type"],
            "underlying": underlying,
            "spot": state["spot"],
            "spot_source": state["spot_source"],
            "expiry": state["expiry"],
            "strike": state["strike"],
            "option_type": state["option_type"],
            "instrument_key": instrument,
            "trading_symbol": state["trading_symbol"],
            "lot_size": state["lot_size"],
            "freeze_quantity": state["freeze_quantity"],
            "signal_id": signal.get("signal_id"),
            "action": signal.get("action") or (state["decision"] or {}).get("action"),
            "conditions": signal.get("conditions") or (state["decision"] or {}).get("conditions"),
            "signal_reason": signal.get("reason") or (state["decision"] or {}).get("reason"),
            "entry_premium": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "target": signal.get("target"),
            "lots": signal.get("lots"),
            "quantity": signal.get("quantity"),
            "risk_status": risk.get("status"),
            "risk_reason": risk.get("reason"),
            "option_resolution": state["option_resolution"],
            "sandbox_execution": state["sandbox_execution"],
            "broker": "UPSTOX SANDBOX",
            "sandbox_order_id": order.get("order_id"),
            "order_type": order.get("order_type"),
            "submitted_status": submitted.get("status"),
            "broker_order_status": verified.get("status"),
            "final_cancellation_status": order.get("status") if order else None,
            "execution_result": "SUCCESS" if success else "FAILED",
            "data_labels": state["data_labels"],
            "notice": OPTION_NOTICE,
            "error": error,
            "live_trading_enabled": live_trading_enabled(),
            "trading_mode": trading_mode(),
        }
        self._store_result(run_id, result)
        if success:
            self._event("OPTION_DEMO_COMPLETED", "INFO",
                        "Option Sandbox demo completed: NSE_FO order submitted, verified and cancelled",
                        demo_run_id=run_id, order_id=result["sandbox_order_id"], instrument_key=instrument)
        else:
            self._event("OPTION_DEMO_FAILED", "CRITICAL" if (error or {}).get("cancel_required") else "WARNING",
                        (error or {}).get("headline", "OPTION DEMO FAILED"),
                        demo_run_id=run_id, order_id=result["sandbox_order_id"],
                        code=(error or {}).get("code"), stage=(error or {}).get("stage"))
