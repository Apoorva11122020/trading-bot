"""Controlled Upstox Sandbox demo flow, shared by the CLI demo and the API.

Actual StrategyRuntime -> RiskManager -> OrderManager -> Upstox Sandbox
-> broker verification -> controlled cancellation. Nothing in here
fabricates signals, orders or broker responses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.broker.models import BrokerOrder, OrderStatus
from src.config.loader import load_strategy_config
from src.execution.order_manager import ExecutionRecord, OrderManager
from src.market_data.models import MarketDataHealth, MarketDataStatus
from src.models.market import Candle
from src.models.risk import AccountState
from src.models.signals import Signal
from src.risk.risk_manager import RiskManager
from src.safety.live_gate import live_trading_enabled, trading_mode
from src.session.session_manager import SessionManager
from src.strategy.runtime import StrategyRuntime
from src.strategy.signal_generator import SignalGenerator
from src.strategy.strategy_engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "strategy.yaml"
DEMO_CANDLE_COUNT = 55
DEMO_TIMEFRAME = "5m"

# Same environment source as scripts/demo_e2e_sandbox.py; never overrides
# variables already set in the process.
load_dotenv(PROJECT_ROOT / ".env")

StageCallback = Callable[..., None]


class DemoSafetyError(RuntimeError):
    """The environment is not safe for the Sandbox demo."""


class DemoStageError(RuntimeError):
    """A demo stage failed. `stage` is the stage that was in progress."""

    def __init__(
        self,
        stage: str,
        code: str,
        message: str,
        *,
        cause: BaseException | None = None,
        order: dict | None = None,
        broker: dict | None = None,
        cancel_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message
        self.cause = cause
        self.order = order
        self.broker = broker
        self.cancel_required = cancel_required


def demo_symbol() -> str:
    return os.getenv("DEMO_STRATEGY_SYMBOL", "RELIANCE")


def sandbox_instrument() -> str:
    return os.getenv("UPSTOX_SANDBOX_INSTRUMENT", "NSE_EQ|INE669E01016")


def demo_equity() -> float:
    return float(os.getenv("DEMO_EQUITY", "100000"))


def sandbox_order_execution_enabled() -> bool:
    return os.getenv("UPSTOX_RUN_SANDBOX_ORDER_TEST", "false").strip().lower() == "true"


def assert_demo_safe() -> None:
    """Refuse unless PAPER mode with live trading disabled."""
    if live_trading_enabled():
        raise DemoSafetyError("LIVE_TRADING_ENABLED must be false for the Sandbox demo")
    if trading_mode() != "PAPER":
        raise DemoSafetyError("TRADING_MODE must be PAPER for the Sandbox demo")


# ---------------------------------------------------------------------------
# Market data + strategy
# ---------------------------------------------------------------------------

def build_demo_candles(symbol: str, count: int = DEMO_CANDLE_COUNT) -> list[Candle]:
    if count < 55:
        raise ValueError("At least 55 candles are required.")

    start = datetime(2026, 10, 1, 9, 30, tzinfo=IST)
    candles: list[Candle] = []
    price = 100.0

    for i in range(count - 1):
        price += 0.15
        candles.append(
            Candle(
                timestamp=start + timedelta(minutes=5 * i),
                symbol=symbol,
                open=price - 0.10,
                high=price + 0.20,
                low=price - 0.20,
                close=price,
                volume=100.0,
            )
        )

    # Final breakout candle
    price += 5.0
    candles.append(
        Candle(
            timestamp=start + timedelta(minutes=5 * (count - 1)),
            symbol=symbol,
            open=price - 0.20,
            high=price + 0.20,
            low=price - 0.30,
            close=price,
            volume=200.0,
        )
    )
    return candles


def connected_market_health(symbol: str) -> MarketDataHealth:
    return MarketDataHealth(
        status=MarketDataStatus.CONNECTED,
        last_tick_at=datetime.now(IST),
        subscribed_symbols=[symbol],
    )


def create_strategy_runtime(config_path: str | Path = DEFAULT_CONFIG_PATH):
    config = load_strategy_config(config_path)
    risk_manager = RiskManager(config.risk)
    session_manager = SessionManager(config.session)
    strategy_engine = StrategyEngine(config, risk_manager, session_manager)
    runtime = StrategyRuntime(SignalGenerator(strategy_engine))
    return config, risk_manager, session_manager, runtime


@dataclass
class StrategyOutcome:
    signal: Signal | None
    config: Any
    risk_manager: RiskManager
    session_manager: SessionManager
    candles: list[Candle] = field(default_factory=list)


def evaluate_strategy(
    candles: list[Candle],
    *,
    equity: float,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    on_signal: Callable[[Candle, Signal], None] | None = None,
) -> StrategyOutcome:
    """Feed candles through the actual StrategyRuntime; the last signal wins."""
    config, risk_manager, session_manager, runtime = create_strategy_runtime(config_path)
    account = AccountState(equity=equity)
    symbol = candles[-1].symbol if candles else demo_symbol()
    health = connected_market_health(symbol)

    signal = None
    for candle in candles:
        result = runtime.on_candle(candle, account, health)
        if result is not None:
            signal = result
            if on_signal is not None:
                on_signal(candle, result)

    return StrategyOutcome(signal, config, risk_manager, session_manager, candles)


def preview_risk(
    signal: Signal,
    risk_manager: RiskManager,
    session_manager: SessionManager,
    equity: float,
) -> dict:
    """Run the actual RiskManager entry validation with the same inputs the
    OrderManager uses for its fresh pre-submission check."""
    if signal.stop_loss is None:
        return {"allowed": False, "reasons": ["MISSING_STOP_LOSS"], "quantity": 0,
                "risk_amount": 0.0, "estimated_exposure": 0.0}
    result = risk_manager.validate_entry(
        account=AccountState(equity=equity),
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        proposed_exposure=signal.entry_price * signal.quantity,
        session_allowed=session_manager.is_entry_allowed(signal.timestamp),
        quantity=signal.quantity,
    )
    return {
        "allowed": bool(result.allowed),
        "reasons": list(result.reasons),
        "quantity": int(result.quantity),
        "risk_amount": float(result.risk_amount),
        "estimated_exposure": float(result.estimated_exposure),
    }


# ---------------------------------------------------------------------------
# Serialisation helpers (no credentials, only order/signal data)
# ---------------------------------------------------------------------------

def _value(item):
    return getattr(item, "value", item)


def signal_summary(signal: Signal) -> dict:
    return {
        "signal_id": signal.signal_id,
        "timestamp": signal.timestamp.isoformat(),
        "symbol": signal.symbol,
        "side": _value(signal.side),
        "signal_type": _value(signal.signal_type),
        "entry_price": signal.entry_price,
        "quantity": signal.quantity,
        "stop_loss": signal.stop_loss,
        "target": signal.target,
        "reason": signal.reason,
        "risk_status": signal.risk_status,
        "blocked": signal.blocked,
        "block_reason": signal.block_reason,
        "conditions": dict(signal.confidence_metadata or {}),
    }


def record_summary(record: ExecutionRecord) -> dict:
    return {
        "order_id": record.order_id,
        "signal_id": record.signal_id,
        "symbol": record.symbol,
        "side": record.side,
        "order_type": record.order_type,
        "status": _value(record.status),
        "requested_quantity": record.requested_quantity,
        "filled_quantity": record.filled_quantity,
        "average_fill_price": record.average_fill_price,
        "rejection_reason": record.rejection_reason,
        "instrument": dict(record.instrument) if record.instrument else None,
    }


def broker_order_summary(order: BrokerOrder) -> dict:
    return {
        "order_id": order.order_id,
        "status": _value(order.status),
        "side": _value(order.side),
        "order_type": _value(order.order_type),
        "quantity": order.quantity,
        "filled_quantity": order.filled_quantity,
        "average_fill_price": order.average_fill_price,
        "price": order.price,
    }


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

def _disconnect(broker) -> None:
    try:
        broker.disconnect()
    except Exception:
        pass


def execute_sandbox_order(
    signal: Signal,
    *,
    broker,
    risk_manager: RiskManager,
    session_manager: SessionManager,
    database=None,
    instrument: str,
    equity: float,
    max_slippage_percent: float = 0.10,
    on_stage: StageCallback | None = None,
    instrument_meta: dict | None = None,
    order_type: str = "MARKET",
    order_params: dict | None = None,
    entry_gate: Callable[[Signal], list[str]] | None = None,
) -> dict:
    """Submit the strategy signal to the Sandbox through the actual
    OrderManager, verify it with the broker, then cancel it.

    Raises DemoStageError on any failure. Once an order exists, a controlled
    cancellation is always attempted before returning or raising.
    """
    notify = on_stage or (lambda stage, **data: None)

    notify("BROKER_CONNECTING")
    try:
        broker.connect()
    except Exception as exc:
        _disconnect(broker)
        raise DemoStageError(
            "BROKER_CONNECTING", "BROKER_CONNECTION_FAILED",
            f"{type(exc).__name__}: {exc}", cause=exc,
        ) from exc
    notify("BROKER_CONNECTED")

    try:
        order_manager = OrderManager(
            broker=broker,
            risk_manager=risk_manager,
            session_manager=session_manager,
            max_slippage_percent=max_slippage_percent,
            database=database,
            entry_gate=entry_gate,
            order_type=order_type,
            order_params=order_params,
        )
        account = AccountState(equity=equity)

        notify("ORDER_SUBMITTING", instrument=instrument)
        original_symbol = signal.symbol
        signal.symbol = instrument
        try:
            record = order_manager.submit_signal(signal, account, instrument=instrument_meta)
        except Exception as exc:
            text = str(exc)
            if text.startswith((
                "ENTRY_BLOCKED", "STALE_SIGNAL_QUANTITY", "SIGNAL_BLOCKED", "QUANTITY_NOT_LOT_MULTIPLE",
            )):
                raise DemoStageError(
                    "RISK_VALIDATION", "RISK_BLOCKED", text, cause=exc,
                ) from exc
            raise DemoStageError(
                "ORDER_SUBMITTING", "ORDER_SUBMISSION_FAILED",
                f"{type(exc).__name__}: {exc}", cause=exc,
            ) from exc
        finally:
            signal.symbol = original_symbol

        if record.status in (OrderStatus.REJECTED, OrderStatus.FAILED):
            raise DemoStageError(
                "ORDER_SUBMITTING", "ORDER_SUBMISSION_FAILED",
                f"Broker returned {record.status.value}: {record.rejection_reason or 'no reason given'}",
                order=record_summary(record),
            )

        submitted = record_summary(record)
        notify("ORDER_SUBMITTED", order=submitted, record=record)

        notify("ORDER_VERIFYING", order_id=record.order_id)
        details = None
        verify_error: Exception | None = None
        try:
            details = broker.get_order(record.order_id)
        except Exception as exc:
            verify_error = exc
        verified = broker_order_summary(details) if details is not None else None
        if details is not None:
            notify("ORDER_VERIFIED", broker=verified, details=details)

        notify("ORDER_CANCELLING", order_id=record.order_id)
        try:
            cancelled = order_manager.cancel(record.order_id)
        except Exception as exc:
            raise DemoStageError(
                "ORDER_CANCELLING", "ORDER_CANCELLATION_FAILED",
                f"{type(exc).__name__}: {exc}", cause=exc,
                order=record_summary(record), broker=verified, cancel_required=True,
            ) from exc

        final = record_summary(cancelled)
        if verify_error is not None:
            raise DemoStageError(
                "ORDER_SUBMITTING", "ORDER_VERIFICATION_FAILED",
                f"{type(verify_error).__name__}: {verify_error} "
                f"(order cancellation status: {final['status']})",
                cause=verify_error, order=final,
            )
        if cancelled.status != OrderStatus.CANCELLED:
            raise DemoStageError(
                "ORDER_CANCELLING", "ORDER_CANCELLATION_FAILED",
                f"Order is {final['status']} after the cancel request",
                order=final, broker=verified,
                cancel_required=cancelled.status not in (OrderStatus.FILLED, OrderStatus.REJECTED),
            )

        notify("ORDER_CANCELLED", order=final, record=cancelled)
        return {
            "submitted": submitted,
            "broker": verified,
            "cancelled": final,
            "record": cancelled,
            "details": details,
        }
    finally:
        _disconnect(broker)
