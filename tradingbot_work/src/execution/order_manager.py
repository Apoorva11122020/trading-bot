from __future__ import annotations

import hashlib
import inspect
import threading
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from typing import Any, Callable

from src.broker.interface import BrokerInterface
from src.broker.models import (
    BrokerOrder,
    OrderStatus,
)
from src.models.risk import AccountState
from src.models.signals import (
    Signal,
    SignalType,
)
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager


try:
    from src.database import Database
except ImportError:
    Database = Any


try:
    from src.logging.logger import (
        get_logger,
        log_event,
    )
except ImportError:
    get_logger = None
    log_event = None


@dataclass
class ExecutionRecord:
    signal_id: str
    order_id: str
    symbol: str
    signal_type: str
    status: OrderStatus
    requested_quantity: int
    filled_quantity: int
    average_fill_price: float | None
    created_at: datetime
    updated_at: datetime
    rejection_reason: str | None = None
    side: str | None = None
    order_type: str | None = None
    requested_price: float | None = None
    reason: str | None = None
    journaled: bool = False
    tag: str | None = None
    slippage_abs: float | None = None
    slippage_pct: float | None = None
    realized_pnl: float = 0.0
    instrument: dict[str, Any] | None = None


INSTRUMENT_COLUMNS = ("market_type", "underlying", "option_type", "strike", "expiry", "lot_size", "instrument_key")

TERMINAL_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.FAILED,
    }
)

WORKING_STATUSES = frozenset(
    {
        OrderStatus.SUBMITTED,
        OrderStatus.OPEN,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.MODIFIED,
    }
)


class OrderInDoubtError(RuntimeError):
    """The broker may or may not have accepted an order (timeout, 5xx,
    dropped connection). The order is never resubmitted automatically."""


def client_tag(signal_id: str) -> str:
    """Deterministic client order reference (<= 40 chars for Upstox)."""
    return "ATB" + hashlib.sha256(signal_id.encode()).hexdigest()[:24]


def _is_definite_rejection(exc: Exception) -> bool:
    """True when the broker certainly did not accept the order."""
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return True
    status_code = getattr(exc, "status_code", None)
    return isinstance(status_code, int) and 400 <= status_code < 500


class OrderManager:
    """Single state machine for every broker order.

    Submission, streamed updates, polling and restart recovery all feed
    `_apply_broker_state`, which accounts fills by cumulative-quantity delta.
    Repeated or out-of-order broker reports therefore never double count a
    fill, and quantities only move forward.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        risk_manager: RiskManager,
        session_manager: SessionManager,
        max_slippage_percent: float = 0.10,
        database: Database | None = None,
        logger=None,
        entry_gate: Callable[[Signal], list[str]] | None = None,
        order_type: str = "MARKET",
        order_params: dict[str, Any] | None = None,
    ):

        if max_slippage_percent < 0:
            raise ValueError(
                "max_slippage_percent must be >= 0"
            )
        if str(order_type).upper() not in ("MARKET", "LIMIT"):
            raise ValueError("order_type must be MARKET or LIMIT")

        self.broker = broker
        self.default_order_type = str(order_type).upper()
        self.order_params = dict(order_params or {})
        self.instruments: dict[str, dict[str, Any]] = {}
        self.risk_manager = risk_manager
        self.session_manager = session_manager
        self.max_slippage_percent = max_slippage_percent
        self.entry_gate = entry_gate

        self.records: dict[str, ExecutionRecord] = {}
        self.signal_to_order: dict[str, str] = {}
        self.expected_positions: dict[str, int] = {}
        self.position_avg: dict[str, float] = {}
        self.in_doubt: dict[str, dict[str, Any]] = {}

        self.database = database
        self._lock = threading.RLock()
        self._listeners: list[Callable[[ExecutionRecord], None]] = []

        self.logger = (
            logger
            or (
                get_logger()
                if get_logger
                else None
            )
        )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _order_type(
        self,
        signal: Signal,
    ) -> str:
        return self.default_order_type

    def add_listener(self, callback: Callable[[ExecutionRecord], None]) -> None:
        self._listeners.append(callback)

    def _notify(self, record: ExecutionRecord) -> None:
        for callback in list(self._listeners):
            callback(record)

    def _tx(self):
        if self.database is not None and hasattr(self.database, "transaction"):
            return self.database.transaction()
        return nullcontext()

    def _event(self, event_type: str, severity: str, message: str, **metadata) -> None:
        if self.database is not None:
            try:
                self.database.add_event(event_type, severity, message, metadata)
            except Exception:
                pass
        if self.logger and log_event:
            level = {"INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}.get(severity, 20)
            log_event(self.logger, level, event_type.lower(), detail=message, **metadata)

    def _broker_order(
        self,
        value,
    ) -> BrokerOrder:
        if isinstance(value, BrokerOrder):
            return value
        if isinstance(value, dict):
            return BrokerOrder(**value)
        return value

    def _broker_orders(
        self,
    ) -> dict[str, BrokerOrder]:
        return {
            str(order.order_id): order
            for order in (
                self._broker_order(item)
                for item in self.broker.get_orders()
            )
        }

    @staticmethod
    def _order_id_of(
        value,
    ) -> str:
        if isinstance(value, dict):
            return str(value["order_id"])
        if hasattr(value, "order_id"):
            return str(value.order_id)
        return str(value)

    def _fetch_broker_order(
        self,
        order_id: str,
    ) -> BrokerOrder:
        order = self._broker_orders().get(str(order_id))
        if order is None:
            raise RuntimeError(
                "ORDER_STATUS_VERIFICATION_FAILED:"
                + str(order_id)
            )
        return order

    def _find_by_tag(self, tag: str) -> BrokerOrder | None:
        for order in self._broker_orders().values():
            if getattr(order, "tag", None) == tag:
                return order
        return None

    def _broker_positions(
        self,
    ) -> dict[str, int]:
        positions: dict[str, int] = {}
        for item in self.broker.get_positions():
            if isinstance(item, dict):
                symbol = str(item["symbol"])
                quantity = int(item["quantity"])
            else:
                symbol = str(item.symbol)
                quantity = int(item.quantity)
            if quantity != 0:
                positions[symbol] = quantity
        return positions

    def _supports_tag(self) -> bool:
        return "tag" in self._place_order_params()

    def _place_order_params(self) -> set[str]:
        try:
            return set(inspect.signature(self.broker.place_order).parameters)
        except (TypeError, ValueError):
            return set()

    # --------------------------------------------------
    # Fresh risk check
    # --------------------------------------------------

    def _fresh_entry_risk_check(
        self,
        signal: Signal,
        account: AccountState,
    ) -> None:

        if signal.stop_loss is None:
            raise ValueError(
                "ENTRY signal must contain stop_loss"
            )

        if signal.quantity <= 0:
            raise RuntimeError(
                "INVALID_SIGNAL_QUANTITY"
            )

        if self.entry_gate is not None:
            reasons = list(self.entry_gate(signal) or [])
            if reasons:
                raise RuntimeError(
                    "ENTRY_BLOCKED_BY_RISK_GATE:"
                    + ";".join(reasons)
                )

        pending = self._in_doubt_for(signal.symbol, SignalType.ENTRY.value)
        if pending is not None:
            raise RuntimeError("ENTRY_IN_DOUBT:" + pending)

        position = self.expected_positions.get(signal.symbol, 0)
        signed = signal.quantity if signal.side.value == "BUY" else -signal.quantity
        if position * signed < 0:
            raise RuntimeError(
                "ENTRY_CONFLICTS_WITH_OPEN_POSITION:"
                + signal.symbol
            )

        result = self.risk_manager.validate_entry(
            account=account,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            proposed_exposure=signal.entry_price * signal.quantity,
            session_allowed=self.session_manager.is_entry_allowed(signal.timestamp),
            quantity=signal.quantity,
        )

        if not result.allowed:
            raise RuntimeError(
                "ENTRY_BLOCKED_BY_FRESH_RISK_CHECK:"
                + ";".join(result.reasons)
            )

        if result.quantity != signal.quantity:
            raise RuntimeError(
                "STALE_SIGNAL_QUANTITY:"
                f"{signal.quantity}"
                f"!={result.quantity}"
            )

    # --------------------------------------------------
    # Slippage (measured at fill time, never rejects a fill)
    # --------------------------------------------------

    @staticmethod
    def slippage(expected: float | None, actual: float | None) -> tuple[float, float] | None:
        """Absolute and percentage deviation of a fill from the reference
        price. Returns None when either price is unknown."""
        if actual is None or expected is None or expected <= 0:
            return None
        absolute = abs(actual - expected)
        return absolute, absolute / expected * 100

    # --------------------------------------------------
    # Records
    # --------------------------------------------------

    def _new_record(
        self,
        signal: Signal,
        order_id: str,
        order: BrokerOrder | None,
        tag: str | None,
    ) -> ExecutionRecord:
        now = datetime.now(timezone.utc)
        record = ExecutionRecord(
            signal_id=signal.signal_id,
            order_id=order_id,
            symbol=order.symbol if order else signal.symbol,
            signal_type=signal.signal_type.value,
            status=OrderStatus.SUBMITTED,
            requested_quantity=order.quantity if order else signal.quantity,
            filled_quantity=0,
            average_fill_price=None,
            created_at=order.created_at if order else now,
            updated_at=now,
            side=order.side.value if order else signal.side.value,
            order_type=order.order_type.value if order else self._order_type(signal),
            requested_price=signal.entry_price,
            reason=signal.reason,
            tag=tag,
            instrument=self.instruments.get(signal.signal_id),
        )
        self.records[order_id] = record
        self.signal_to_order[signal.signal_id] = order_id
        return record

    def _persist_order(self, record: ExecutionRecord) -> None:
        if not self.database or not record.side or not record.order_type:
            return
        instrument = record.instrument or {}
        self.database.save_order(
            {
                **{column: instrument.get(column) for column in INSTRUMENT_COLUMNS},
                "order_id": record.order_id,
                "signal_id": record.signal_id,
                "symbol": record.symbol,
                "side": record.side,
                "order_type": record.order_type,
                "quantity": record.requested_quantity,
                "status": record.status.value,
                "filled_quantity": record.filled_quantity,
                "average_fill_price": record.average_fill_price,
                "requested_price": record.requested_price,
                "reason": record.reason,
                "signal_type": record.signal_type,
                "tag": record.tag,
                "rejection_reason": record.rejection_reason,
                "slippage_abs": record.slippage_abs,
                "slippage_pct": record.slippage_pct,
                "realized_pnl": record.realized_pnl,
                "created_at": record.created_at.isoformat()
                if isinstance(record.created_at, datetime)
                else record.created_at,
            }
        )

    # --------------------------------------------------
    # Positions and trade journal (fill driven)
    # --------------------------------------------------

    def _update_expected_position(
        self,
        symbol: str,
        side: str,
        quantity: int,
    ) -> None:
        signed = quantity if side.upper() == "BUY" else -quantity
        new_quantity = self.expected_positions.get(symbol, 0) + signed
        if new_quantity == 0:
            self.expected_positions.pop(symbol, None)
        else:
            self.expected_positions[symbol] = new_quantity

    def _apply_fill(
        self,
        record: ExecutionRecord,
        side: str,
        quantity: int,
        price: float,
    ) -> None:
        symbol = record.symbol
        signed = quantity if side == "BUY" else -quantity
        old = self.expected_positions.get(symbol, 0)
        old_avg = self.position_avg.get(symbol, price)
        new = old + signed

        opened_qty = 0
        closed_qty = 0
        realized = 0.0

        if old == 0 or old * signed > 0:
            opened_qty = quantity
            average = (old_avg * abs(old) + price * quantity) / abs(new)
        else:
            closed_qty = min(abs(old), quantity)
            direction = 1 if old > 0 else -1
            realized = (price - old_avg) * closed_qty * direction
            record.realized_pnl += realized
            if new * old < 0:
                opened_qty = abs(new)
                average = price
                self._event(
                    "POSITION_FLIP_DETECTED", "CRITICAL",
                    "Fill reversed the position direction",
                    symbol=symbol, order_id=record.order_id, before=old, after=new,
                )
            else:
                average = old_avg

        self._update_expected_position(symbol, side, quantity)
        if new == 0:
            self.position_avg.pop(symbol, None)
        else:
            self.position_avg[symbol] = average

        if self.database is None:
            return

        if new == 0:
            self.database.delete_position(symbol)
        else:
            self.database.upsert_position(
                {
                    "symbol": symbol,
                    "side": "LONG" if new > 0 else "SHORT",
                    "quantity": abs(new),
                    "average_entry": average,
                    "current_price": price,
                    "unrealized_pnl": (price - average) * new,
                }
            )

        now = datetime.now(timezone.utc).isoformat()
        if closed_qty:
            self._journal_exit(record, closed_qty, price, realized, closed=(new == 0 or new * old < 0), now=now)
        if opened_qty:
            self._journal_entry(record, side, opened_qty, price, now)

    def _journal_entry(self, record: ExecutionRecord, side: str, quantity: int, price: float, now: str) -> None:
        existing = self.database.get_trade(record.order_id)
        if existing is not None:
            old_qty = int(existing["quantity"])
            old_price = float(existing["entry_price"] or price)
            new_qty = old_qty + quantity
            entry_price = (old_price * old_qty + price * quantity) / new_qty
            opened_at = existing["opened_at"]
        else:
            new_qty = quantity
            entry_price = price
            opened_at = now
        self.database.save_trade(
            {
                "trade_id": record.order_id,
                "order_id": record.order_id,
                "signal_id": record.signal_id,
                "symbol": record.symbol,
                "side": side,
                "quantity": new_qty,
                "entry_price": entry_price,
                "opened_at": opened_at,
                "status": "OPEN",
                "exit_quantity": int(existing["exit_quantity"] or 0) if existing is not None else 0,
                "entry_slippage_pct": record.slippage_pct,
            }
        )

    def _journal_exit(
        self,
        record: ExecutionRecord,
        quantity: int,
        price: float,
        realized: float,
        closed: bool,
        now: str,
    ) -> None:
        entry_side = "BUY" if record.side == "SELL" else "SELL"
        trade = self.database.get_open_trade(record.symbol, entry_side)
        if trade is None:
            self._event(
                "TRADE_JOURNAL", "WARNING", "Exit fill without matching open trade",
                order_id=record.order_id, symbol=record.symbol,
            )
            return

        previous_exit_qty = int(trade["exit_quantity"] or 0)
        previous_exit_price = float(trade["exit_price"] or 0.0)
        exit_qty = previous_exit_qty + quantity
        exit_price = (previous_exit_price * previous_exit_qty + price * quantity) / exit_qty
        realized_total = float(trade["realized_pnl"] or 0.0) + realized

        self.database.save_trade(
            {
                "trade_id": trade["trade_id"],
                "order_id": trade["order_id"],
                "signal_id": trade["signal_id"],
                "symbol": record.symbol,
                "side": entry_side,
                "quantity": trade["quantity"],
                "entry_price": trade["entry_price"],
                "exit_price": exit_price,
                "realized_pnl": realized_total,
                "opened_at": trade["opened_at"],
                "closed_at": now if closed else None,
                "status": "CLOSED" if closed else "OPEN",
                "exit_reason": record.reason,
                "exit_quantity": exit_qty,
                "last_exit_at": now,
                "exit_slippage_pct": record.slippage_pct,
            }
        )

    # --------------------------------------------------
    # The state machine
    # --------------------------------------------------

    def _apply_broker_state(
        self,
        record: ExecutionRecord,
        *,
        status: OrderStatus,
        filled_quantity: int,
        average_price: float | None,
        quantity: int | None = None,
        side: str | None = None,
        rejection_reason: str | None = None,
        source: str = "broker",
    ) -> ExecutionRecord:

        with self._lock:
            if record.status in TERMINAL_STATUSES:
                if record.status == OrderStatus.FILLED:
                    return record
                if status not in TERMINAL_STATUSES or status != record.status:
                    return record

            previous_filled = record.filled_quantity
            previous_avg = record.average_fill_price
            incoming_filled = int(filled_quantity or 0)
            if incoming_filled < previous_filled:
                return record

            if quantity:
                record.requested_quantity = int(quantity)
            incoming_filled = min(incoming_filled, record.requested_quantity)
            delta = incoming_filled - previous_filled

            side = (side or record.side or "").upper()
            if delta > 0 and side not in {"BUY", "SELL"}:
                raise ValueError("INVALID_BROKER_ORDER_SIDE:" + side)
            if side in {"BUY", "SELL"} and not record.side:
                record.side = side

            # Brokers report 0 until something fills; 0 is not a fill price.
            new_avg = average_price if average_price else previous_avg

            fill_price = None
            if delta > 0:
                if new_avg and previous_avg and previous_filled:
                    fill_price = (new_avg * incoming_filled - previous_avg * previous_filled) / delta
                    if fill_price <= 0:
                        fill_price = new_avg
                else:
                    fill_price = new_avg
                if not fill_price:
                    fill_price = record.requested_price
                    new_avg = new_avg or fill_price
                    self._event(
                        "FILL_PRICE_MISSING", "WARNING",
                        "Broker reported a fill without a price; requested price used",
                        order_id=record.order_id,
                    )

            status_changed = status != record.status
            reason_changed = (
                status == OrderStatus.REJECTED
                and rejection_reason
                and rejection_reason != record.rejection_reason
            )
            if not (status_changed or delta > 0 or reason_changed):
                return record

            snapshot = replace(record)
            positions_snapshot = dict(self.expected_positions)
            avg_snapshot = dict(self.position_avg)

            record.status = status
            record.filled_quantity = incoming_filled
            record.average_fill_price = new_avg
            if status == OrderStatus.REJECTED and rejection_reason:
                record.rejection_reason = rejection_reason
            record.updated_at = datetime.now(timezone.utc)

            slip = self.slippage(record.requested_price, record.average_fill_price)
            if delta > 0 and slip is not None:
                record.slippage_abs, record.slippage_pct = slip

            try:
                self._commit_state(record, side, delta, incoming_filled, fill_price, status, source)
            except Exception:
                # Keep memory consistent with the (rolled back) database so
                # the same broker report is applied again on the next poll.
                for item in fields(record):
                    setattr(record, item.name, getattr(snapshot, item.name))
                self.expected_positions.clear()
                self.expected_positions.update(positions_snapshot)
                self.position_avg.clear()
                self.position_avg.update(avg_snapshot)
                raise

            if delta > 0 and slip is not None and slip[1] > self.max_slippage_percent:
                self._event(
                    "SLIPPAGE_LIMIT_EXCEEDED", "WARNING",
                    "Confirmed fill deviated from the reference price beyond the configured limit",
                    order_id=record.order_id,
                    symbol=record.symbol,
                    reference_price=record.requested_price,
                    average_fill_price=record.average_fill_price,
                    slippage_abs=slip[0],
                    slippage_pct=round(slip[1], 6),
                    limit_pct=self.max_slippage_percent,
                )

        self._notify(record)
        return record

    def _commit_state(
        self,
        record: ExecutionRecord,
        side: str,
        delta: int,
        incoming_filled: int,
        fill_price: float | None,
        status: OrderStatus,
        source: str,
    ) -> None:
        with self._tx():
            apply_fill = delta > 0
            if apply_fill and self.database is not None:
                fill_slip = self.slippage(record.requested_price, fill_price)
                apply_fill = self.database.save_execution(
                    {
                        "order_id": record.order_id,
                        "signal_id": record.signal_id,
                        "symbol": record.symbol,
                        "side": side,
                        "quantity": delta,
                        "price": fill_price,
                        "cumulative_filled": incoming_filled,
                        "reference_price": record.requested_price,
                        "slippage_abs": fill_slip[0] if fill_slip else None,
                        "slippage_pct": fill_slip[1] if fill_slip else None,
                    }
                )
            if apply_fill:
                self._apply_fill(record, side, delta, fill_price)

            self._persist_order(record)
            severity = "WARNING" if status in {OrderStatus.REJECTED, OrderStatus.FAILED} else "INFO"
            self._event(
                "ORDER", severity, "Order state recorded",
                order_id=record.order_id,
                signal_id=record.signal_id,
                symbol=record.symbol,
                status=status.value,
                filled_quantity=incoming_filled,
                fill_delta=delta,
                fill_price=fill_price,
                source=source,
                rejection_reason=record.rejection_reason,
            )

    def _apply_broker_order(self, record: ExecutionRecord, order: BrokerOrder, source: str) -> ExecutionRecord:
        return self._apply_broker_state(
            record,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_price=order.average_fill_price,
            quantity=order.quantity,
            side=order.side.value,
            rejection_reason=order.rejection_reason,
            source=source,
        )

    # --------------------------------------------------
    # Submit
    # --------------------------------------------------

    def submit_signal(
        self,
        signal: Signal,
        account: AccountState,
        instrument: dict[str, Any] | None = None,
    ) -> ExecutionRecord:
        """`instrument` carries derivative metadata (market_type, underlying,
        option_type, strike, expiry, lot_size, instrument_key) persisted with
        the order. Lot-sized instruments must be ordered in whole lots."""

        with self._lock:
            if instrument:
                lot_size = int(instrument.get("lot_size") or 0)
                if lot_size > 0 and signal.quantity % lot_size:
                    raise RuntimeError(f"QUANTITY_NOT_LOT_MULTIPLE:{signal.quantity}%{lot_size}")
                if instrument.get("instrument_key") and instrument["instrument_key"] != signal.symbol:
                    raise RuntimeError("INSTRUMENT_KEY_MISMATCH")

            if signal.signal_id in self.signal_to_order:
                raise RuntimeError("DUPLICATE_SIGNAL:" + signal.signal_id)

            if signal.signal_id in self.in_doubt:
                adopted = self._resolve_in_doubt(signal)
                if adopted is not None:
                    return adopted
                raise OrderInDoubtError("ORDER_IN_DOUBT_UNRESOLVED:" + signal.signal_id)

            if signal.blocked:
                raise RuntimeError("SIGNAL_BLOCKED:" + (signal.block_reason or "UNKNOWN"))

            if not self.broker.is_connected():
                raise RuntimeError("BROKER_DISCONNECTED")

            if signal.quantity <= 0:
                raise RuntimeError("INVALID_SIGNAL_QUANTITY")

            if signal.signal_type == SignalType.ENTRY:
                self._fresh_entry_risk_check(signal, account)
            elif signal.signal_type == SignalType.EXIT:
                self._exit_guard(signal)

            tag = client_tag(signal.signal_id)
            kwargs = dict(
                symbol=signal.symbol,
                side=signal.side.value,
                quantity=signal.quantity,
                order_type=self._order_type(signal),
                price=signal.entry_price,
            )
            if self._supports_tag():
                kwargs["tag"] = tag
            accepted = self._place_order_params()
            kwargs.update({k: v for k, v in self.order_params.items() if k in accepted})
            if instrument:
                self.instruments[signal.signal_id] = dict(instrument)

            try:
                placed = self.broker.place_order(**kwargs)
            except Exception as exc:
                if _is_definite_rejection(exc):
                    self._event(
                        "ORDER_SUBMIT_FAILED", "WARNING", "Broker refused the order request",
                        signal_id=signal.signal_id, symbol=signal.symbol, error=type(exc).__name__,
                    )
                    raise
                return self._handle_in_doubt(signal, tag, exc)

            order_id = self._order_id_of(placed)
            try:
                order = self._fetch_broker_order(order_id)
            except Exception as exc:
                record = self._new_record(signal, order_id, None, tag)
                self._persist_order(record)
                self._event(
                    "ORDER_STATUS_UNAVAILABLE", "WARNING",
                    "Order accepted but status could not be verified; polling will reconcile it",
                    order_id=order_id, signal_id=signal.signal_id, error=type(exc).__name__,
                )
                return record

            record = self._new_record(signal, order_id, order, tag)
            self._persist_order(record)
            return self._apply_broker_order(record, order, "submit")

    def _handle_in_doubt(self, signal: Signal, tag: str, exc: Exception) -> ExecutionRecord:
        found = None
        try:
            found = self._find_by_tag(tag)
        except Exception:
            found = None
        if found is not None:
            record = self._new_record(signal, found.order_id, found, tag)
            self._persist_order(record)
            self._event(
                "ORDER_IN_DOUBT_RESOLVED", "WARNING",
                "Submission error but broker has the order; adopted without resubmitting",
                order_id=found.order_id, signal_id=signal.signal_id,
            )
            return self._apply_broker_order(record, found, "in_doubt")

        self.in_doubt[signal.signal_id] = {
            "tag": tag,
            "signal_id": signal.signal_id,
            "symbol": signal.symbol,
            "side": signal.side.value,
            "quantity": signal.quantity,
            "requested_price": signal.entry_price,
            "reason": signal.reason,
            "signal_type": signal.signal_type.value,
            "at": datetime.now(timezone.utc).isoformat(),
            "error": type(exc).__name__,
        }
        if self.database is not None:
            self.database.set_state("in_doubt_orders", self.in_doubt)
        self._event(
            "ORDER_IN_DOUBT", "CRITICAL",
            "Broker submission outcome unknown; order will not be resubmitted automatically",
            signal_id=signal.signal_id, symbol=signal.symbol, tag=tag, error=type(exc).__name__,
        )
        raise OrderInDoubtError("ORDER_IN_DOUBT:" + signal.signal_id) from exc

    def _resolve_in_doubt(self, signal: Signal) -> ExecutionRecord | None:
        info = self.in_doubt[signal.signal_id]
        found = self._find_by_tag(info["tag"])
        if found is None:
            return None
        del self.in_doubt[signal.signal_id]
        if self.database is not None:
            self.database.set_state("in_doubt_orders", self.in_doubt)
        record = self._new_record(signal, found.order_id, found, info["tag"])
        self._persist_order(record)
        self._event(
            "ORDER_IN_DOUBT_RESOLVED", "INFO", "In-doubt order found at broker by client tag",
            order_id=found.order_id, signal_id=signal.signal_id,
        )
        return self._apply_broker_order(record, found, "in_doubt")

    def _adopt_in_doubt(self, signal_id: str, order: BrokerOrder) -> ExecutionRecord:
        """Track an in-doubt order that has appeared at the broker, using
        the signal details captured at submission time."""
        with self._lock:
            info = self.in_doubt.pop(signal_id)
            if self.database is not None:
                self.database.set_state("in_doubt_orders", self.in_doubt)
            now = datetime.now(timezone.utc)
            record = ExecutionRecord(
                signal_id=signal_id,
                order_id=order.order_id,
                symbol=order.symbol,
                signal_type=info.get("signal_type") or "UNKNOWN",
                status=OrderStatus.SUBMITTED,
                requested_quantity=order.quantity,
                filled_quantity=0,
                average_fill_price=None,
                created_at=order.created_at,
                updated_at=now,
                side=order.side.value,
                order_type=order.order_type.value,
                requested_price=info.get("requested_price"),
                reason=info.get("reason"),
                tag=info.get("tag"),
                instrument=self.instruments.get(signal_id),
            )
            self.records[order.order_id] = record
            self.signal_to_order[signal_id] = order.order_id
            self._persist_order(record)
            self._event(
                "ORDER_IN_DOUBT_RESOLVED", "WARNING",
                "In-doubt order found at broker by client tag and adopted",
                order_id=order.order_id, signal_id=signal_id,
            )
            return self._apply_broker_order(record, order, "in_doubt")

    def _in_doubt_for(self, symbol: str, signal_type: str) -> str | None:
        for signal_id, info in self.in_doubt.items():
            if info.get("symbol") == symbol and info.get("signal_type") == signal_type:
                return signal_id
        return None

    def release_in_doubt(self, signal_id: str) -> bool:
        """Operator action after verifying the broker has no such order.
        Re-checks the broker; releases only when no tagged order exists."""
        with self._lock:
            info = self.in_doubt.get(signal_id)
            if info is None:
                return False
            if self._find_by_tag(info["tag"]) is not None:
                return False
            del self.in_doubt[signal_id]
            if self.database is not None:
                self.database.set_state("in_doubt_orders", self.in_doubt)
            self._event("ORDER_IN_DOUBT_RELEASED", "WARNING", "In-doubt order released", signal_id=signal_id)
            return True

    def _exit_guard(
        self,
        signal: Signal,
    ) -> None:
        symbol = signal.symbol
        position = self.expected_positions.get(symbol, 0)
        closing_side = "SELL" if position > 0 else "BUY" if position < 0 else None

        if signal.side.value != closing_side:
            raise RuntimeError("NO_OPEN_POSITION_TO_EXIT:" + symbol)

        pending = self._in_doubt_for(symbol, SignalType.EXIT.value)
        if pending is not None:
            raise RuntimeError("EXIT_IN_DOUBT:" + pending)

        for record in self.records.values():
            if (
                record.symbol == symbol
                and record.signal_type == SignalType.EXIT.value
                and record.status not in TERMINAL_STATUSES
            ):
                raise RuntimeError("EXIT_ALREADY_PENDING:" + record.order_id)

        if signal.quantity > abs(position):
            raise RuntimeError(
                "EXIT_QUANTITY_EXCEEDS_POSITION:"
                f"{signal.quantity}>{abs(position)}"
            )

    def has_pending_entry(self, symbol: str) -> bool:
        return any(
            r.symbol == symbol
            and r.signal_type == SignalType.ENTRY.value
            and r.status not in TERMINAL_STATUSES
            for r in self.records.values()
        )

    # --------------------------------------------------
    # Broker polling
    # --------------------------------------------------

    def open_order_ids(self) -> list[str]:
        with self._lock:
            return [
                order_id
                for order_id, record in self.records.items()
                if record.status not in TERMINAL_STATUSES
            ]

    def refresh_order(
        self,
        order_id: str,
    ) -> ExecutionRecord:
        """Pull the broker's current view of an order through the same
        state machine used for streamed order updates."""
        record = self.records.get(order_id)
        if record is None:
            raise KeyError("UNKNOWN_BROKER_ORDER:" + str(order_id))
        order = self._fetch_broker_order(order_id)
        return self._apply_broker_order(record, order, "refresh")

    def refresh_open_orders(self) -> dict[str, Any]:
        """One broker order-book call; applies the result to every
        non-terminal local order. Never submits anything."""
        open_ids = self.open_order_ids()
        if not open_ids and not self.in_doubt:
            return {"refreshed": [], "missing": []}
        broker_orders = self._broker_orders()
        refreshed, missing = [], []
        for order_id in open_ids:
            order = broker_orders.get(order_id)
            if order is None:
                missing.append(order_id)
                continue
            self._apply_broker_order(self.records[order_id], order, "poll")
            refreshed.append(order_id)
        for signal_id in list(self.in_doubt):
            tag = self.in_doubt[signal_id]["tag"]
            match = next((o for o in broker_orders.values() if o.tag == tag), None)
            if match is not None:
                self._adopt_in_doubt(signal_id, match)
                refreshed.append(match.order_id)
        if missing:
            self._event(
                "ORDER_NOT_FOUND_AT_BROKER", "WARNING",
                "Local open orders were not present in the broker order book",
                order_ids=missing,
            )
        return {"refreshed": refreshed, "missing": missing}

    # --------------------------------------------------
    # Modify / cancel
    # --------------------------------------------------

    def modify(
        self,
        order_id: str,
        price: float | None = None,
        quantity: int | None = None,
    ) -> ExecutionRecord:
        if not self.broker.is_connected():
            raise RuntimeError("BROKER_DISCONNECTED")
        record = self.records.get(order_id)
        if record is None:
            raise KeyError("UNKNOWN_BROKER_ORDER:" + str(order_id))

        changes = {
            key: value
            for key, value in {"price": price, "quantity": quantity}.items()
            if value is not None
        }
        self.broker.modify_order(order_id, **changes)
        order = self._fetch_broker_order(order_id)
        return self._apply_broker_order(record, order, "modify")

    def cancel(
        self,
        order_id: str,
    ) -> ExecutionRecord:
        if not self.broker.is_connected():
            raise RuntimeError("BROKER_DISCONNECTED")
        record = self.records.get(order_id)
        if record is None:
            raise KeyError("UNKNOWN_BROKER_ORDER:" + str(order_id))
        self.broker.cancel_order(order_id)
        order = self._fetch_broker_order(order_id)
        return self._apply_broker_order(record, order, "cancel")

    # --------------------------------------------------
    # Reconcile
    # --------------------------------------------------

    def reconcile(self) -> dict:
        actual_positions = self._broker_positions()
        mismatches: dict[str, dict[str, int]] = {}
        for symbol in set(self.expected_positions) | set(actual_positions):
            expected = int(self.expected_positions.get(symbol, 0))
            actual = int(actual_positions.get(symbol, 0))
            if expected != actual:
                mismatches[symbol] = {"expected": expected, "actual": actual}
        return {
            "ok": not mismatches,
            "expected_positions": dict(self.expected_positions),
            "broker_positions": actual_positions,
            "mismatches": mismatches,
        }

    @staticmethod
    def _map_broker_status(
        status: str,
    ) -> OrderStatus:
        value = str(status).strip().lower().replace("_", " ")
        mapping = {
            "open": OrderStatus.OPEN,
            "pending": OrderStatus.SUBMITTED,
            "submitted": OrderStatus.SUBMITTED,
            "put order req received": OrderStatus.SUBMITTED,
            "validation pending": OrderStatus.SUBMITTED,
            "open pending": OrderStatus.SUBMITTED,
            "trigger pending": OrderStatus.OPEN,
            "partial": OrderStatus.PARTIALLY_FILLED,
            "partially filled": OrderStatus.PARTIALLY_FILLED,
            "complete": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "cancelled after market order": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "failed": OrderStatus.FAILED,
            "modified": OrderStatus.MODIFIED,
        }
        return mapping.get(value, OrderStatus.SUBMITTED)

    # --------------------------------------------------
    # Streamed order / position updates
    # --------------------------------------------------

    def apply_order_update(
        self,
        update: Any,
    ) -> ExecutionRecord:
        order_id = str(update.order_id)
        record = self.records.get(order_id)

        if record is None:
            self._event(
                "BROKER_ORDER_RECONCILIATION", "WARNING", "Unknown broker order update ignored",
                order_id=order_id, status=getattr(update, "status", None),
            )
            raise KeyError("UNKNOWN_BROKER_ORDER:" + order_id)

        status = self._map_broker_status(update.status)
        # An "open" report with fills is a partial fill.
        filled = int(getattr(update, "filled_quantity", 0) or 0)
        quantity = int(getattr(update, "quantity", 0) or record.requested_quantity)
        if status == OrderStatus.OPEN and 0 < filled < quantity:
            status = OrderStatus.PARTIALLY_FILLED

        return self._apply_broker_state(
            record,
            status=status,
            filled_quantity=filled,
            average_price=getattr(update, "average_price", None),
            quantity=quantity,
            side=str(getattr(update, "side", "") or ""),
            rejection_reason=getattr(update, "status_message", None),
            source="stream",
        )

    def apply_position_update(
        self,
        update: Any,
    ) -> dict[str, Any]:
        symbol = str(getattr(update, "symbol", None) or getattr(update, "instrument_key", ""))
        actual = int(update.quantity)
        expected = int(self.expected_positions.get(symbol, 0))
        mismatch = expected != actual

        if mismatch:
            self._event(
                "POSITION_RECONCILIATION", "WARNING",
                "Broker position differs from expected position",
                symbol=symbol, expected=expected, actual=actual,
            )

        return {
            "symbol": symbol,
            "expected": expected,
            "actual": actual,
            "mismatch": mismatch,
        }

    # --------------------------------------------------
    # Restart recovery
    # --------------------------------------------------

    def restore_from_database(self) -> dict[str, Any]:
        """Rebuild in-memory order/position state from persisted rows.
        Idempotent: already-known orders are left untouched."""
        if self.database is None:
            return {"orders": 0, "positions": 0}

        restored = 0
        with self._lock:
            for row in self.database.get_all_orders():
                order_id = str(row["order_id"])
                if order_id in self.records:
                    continue
                keys = row.keys()
                created = datetime.fromisoformat(row["created_at"])
                updated = datetime.fromisoformat(row["updated_at"])
                record = ExecutionRecord(
                    signal_id=row["signal_id"] or "",
                    order_id=order_id,
                    symbol=row["symbol"],
                    signal_type=(row["signal_type"] if "signal_type" in keys else None) or "UNKNOWN",
                    status=OrderStatus(row["status"]),
                    requested_quantity=int(row["quantity"]),
                    filled_quantity=int(row["filled_quantity"] or 0),
                    average_fill_price=row["average_fill_price"],
                    created_at=created,
                    updated_at=updated,
                    rejection_reason=row["rejection_reason"] if "rejection_reason" in keys else None,
                    side=row["side"],
                    order_type=row["order_type"],
                    requested_price=row["requested_price"],
                    reason=row["reason"],
                    tag=row["tag"] if "tag" in keys else None,
                    slippage_abs=row["slippage_abs"] if "slippage_abs" in keys else None,
                    slippage_pct=row["slippage_pct"] if "slippage_pct" in keys else None,
                    realized_pnl=float(row["realized_pnl"] or 0.0) if "realized_pnl" in keys else 0.0,
                    instrument=(
                        {column: row[column] for column in INSTRUMENT_COLUMNS}
                        if "instrument_key" in keys and row["instrument_key"]
                        else None
                    ),
                )
                self.records[order_id] = record
                if record.signal_id:
                    self.signal_to_order[record.signal_id] = order_id
                restored += 1

            positions = 0
            for row in self.database.get_positions():
                qty = int(row["quantity"])
                if qty == 0:
                    continue
                signed = -abs(qty) if str(row["side"]).upper() in {"SHORT", "SELL"} else abs(qty)
                self.expected_positions[row["symbol"]] = signed
                self.position_avg[row["symbol"]] = float(row["average_entry"])
                positions += 1

            self.in_doubt.update(self.database.get_state("in_doubt_orders", {}) or {})

        return {"orders": restored, "positions": positions}

    def recover_open_orders(
        self,
    ) -> list[BrokerOrder]:
        """Apply the broker's current state to every known order that was
        not terminal. Fills that happened while the process was down go
        through the normal fill path (execution ids keep this idempotent)."""
        broker_orders = self._broker_orders()
        recovered: list[BrokerOrder] = []
        for order_id in list(self.records):
            order = broker_orders.get(str(order_id))
            if order is None:
                continue
            record = self.records[order_id]
            if record.status not in TERMINAL_STATUSES or record.filled_quantity != order.filled_quantity:
                self._apply_broker_order(record, order, "recovery")
            recovered.append(order)
        return recovered
