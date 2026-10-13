from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from src.execution.order_manager import (
    TERMINAL_STATUSES,
    ExecutionRecord,
    OrderManager,
)
from src.models.positions import Position
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.position.monitor import PositionMonitor


EOD_FORCE_EXIT = "EOD_FORCE_EXIT"


@dataclass
class ManagedPosition:
    entry_signal: Signal
    entry_order_id: str
    position: Position | None = None
    exit_order_id: str | None = None
    exit_reason: str | None = None
    closed: bool = False
    realized_pnl: float | None = None
    exit_order_ids: list[str] = field(default_factory=list)
    close_reason: str | None = None


class PositionLifecycle:
    """Tracks one position per symbol from entry order to closure.

    Positions exist only once the broker reports filled quantity; a
    SUBMITTED/OPEN entry order never creates a position. Every order goes
    through OrderManager, and PositionMonitor decides when to exit. Position
    quantity always mirrors OrderManager's fill-driven net quantity, so
    partial entries and partial exits are reflected exactly.

    State (including stops, extremes and breakeven) is persisted after every
    change and can be rebuilt with `recover()` after a restart.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        monitor: PositionMonitor,
        account_provider: Callable[[], AccountState],
        database: Any | None = None,
    ):
        self.order_manager = order_manager
        self.monitor = monitor
        self.account_provider = account_provider
        self.database = database if database is not None else getattr(order_manager, "database", None)
        self.managed: dict[str, ManagedPosition] = {}
        self._persisted: dict[str, str] = {}
        if hasattr(order_manager, "add_listener"):
            order_manager.add_listener(self._on_record)

    # --------------------------------------------------
    # Events / persistence
    # --------------------------------------------------

    def _event(self, event_type: str, severity: str, message: str, **metadata) -> None:
        self.order_manager._event(event_type, severity, message, **metadata)

    def _payload(self, managed: ManagedPosition) -> dict[str, Any]:
        symbol = managed.entry_signal.symbol
        return {
            "entry_signal": managed.entry_signal.model_dump(mode="json"),
            "entry_order_id": managed.entry_order_id,
            "exit_order_id": managed.exit_order_id,
            "exit_order_ids": list(managed.exit_order_ids),
            "exit_reason": managed.exit_reason,
            "closed": managed.closed,
            "close_reason": managed.close_reason,
            "realized_pnl": managed.realized_pnl,
            "position": managed.position.model_dump(mode="json") if managed.position else None,
            "monitor_state": self.monitor.export_state(symbol),
        }

    def _persist(self, symbol: str) -> None:
        if self.database is None or symbol not in self.managed:
            return
        managed = self.managed[symbol]
        payload = self._payload(managed)
        blob = json.dumps(payload, sort_keys=True, default=str)
        if self._persisted.get(symbol) == blob:
            return
        self.database.save_managed_position(symbol, payload, managed.closed)
        self._persisted[symbol] = blob

    def _on_record(self, record: ExecutionRecord) -> None:
        managed = self.managed.get(record.symbol)
        if managed is not None and not managed.closed:
            self._sync(record.symbol)

    # --------------------------------------------------
    # Entry
    # --------------------------------------------------

    def submit_entry(self, signal: Signal, instrument: dict | None = None) -> ExecutionRecord:
        current = self.managed.get(signal.symbol)
        if current is not None and not current.closed:
            raise RuntimeError("POSITION_ALREADY_MANAGED:" + signal.symbol)

        record = self.order_manager.submit_signal(
            signal,
            self.account_provider(),
            instrument=instrument,
        )

        self.managed[signal.symbol] = ManagedPosition(
            entry_signal=signal,
            entry_order_id=record.order_id,
        )
        self._persisted.pop(signal.symbol, None)
        self._event(
            "POSITION_ENTRY_SUBMITTED", "INFO", "Entry order submitted",
            symbol=signal.symbol, order_id=record.order_id, side=signal.side.value,
            quantity=signal.quantity,
        )
        self._sync(signal.symbol)
        return record

    # --------------------------------------------------
    # Broker state
    # --------------------------------------------------

    def on_order_update(self, update: Any) -> ExecutionRecord:
        record = self.order_manager.apply_order_update(update)
        if record.symbol in self.managed:
            self._sync(record.symbol)
        return record

    def refresh(self, symbol: str) -> ManagedPosition:
        managed = self.managed[symbol]
        for order_id in (managed.entry_order_id, managed.exit_order_id):
            if order_id is None:
                continue
            record = self.order_manager.records.get(order_id)
            if record is not None and record.status not in TERMINAL_STATUSES:
                self.order_manager.refresh_order(order_id)
        self._sync(symbol)
        return managed

    # --------------------------------------------------
    # Monitoring / exit
    # --------------------------------------------------

    def on_price(
        self,
        symbol: str,
        price: float,
        atr_value: float,
        timestamp: datetime,
    ) -> ExecutionRecord | None:
        managed = self.managed.get(symbol)
        if managed is None or managed.closed or managed.position is None:
            return None

        if managed.exit_order_id is not None:
            return None

        position = managed.position
        position.current_price = price
        direction = 1 if position.side == "LONG" else -1
        position.unrealized_pnl = (
            (price - position.average_entry) * position.quantity * direction
        )

        exit_signal = self.monitor.evaluate(position, price, atr_value, timestamp)
        if exit_signal is None:
            self._persist(symbol)
            return None

        if exit_signal.reason == "FORCE_EXIT_EOD":
            exit_signal = exit_signal.model_copy(update={"reason": EOD_FORCE_EXIT})

        return self._submit_exit(managed, exit_signal)

    def _submit_exit(self, managed: ManagedPosition, exit_signal: Signal) -> ExecutionRecord:
        symbol = managed.entry_signal.symbol
        om = self.order_manager

        entry = om.records.get(managed.entry_order_id)
        if entry is not None and entry.status not in TERMINAL_STATUSES:
            # Stop the unfilled entry remainder before flattening, otherwise
            # late entry fills would reopen the position after the exit.
            try:
                om.cancel(managed.entry_order_id)
                self._event(
                    "ENTRY_REMAINDER_CANCELLED", "INFO",
                    "Unfilled entry remainder cancelled before exit",
                    symbol=symbol, order_id=managed.entry_order_id,
                )
            except Exception as exc:
                self._event(
                    "ENTRY_REMAINDER_CANCEL_FAILED", "WARNING",
                    "Could not cancel entry remainder before exit",
                    symbol=symbol, order_id=managed.entry_order_id, error=str(exc)[:200],
                )
            net = om.expected_positions.get(symbol, 0)
            if net != 0 and abs(net) != exit_signal.quantity:
                exit_signal = exit_signal.model_copy(update={"quantity": abs(net)})

        managed.exit_reason = exit_signal.reason
        try:
            record = om.submit_signal(exit_signal, self.account_provider())
        except Exception as exc:
            managed.exit_reason = None
            self._allow_exit_retry(symbol)
            self._event(
                "POSITION_EXIT_SUBMIT_FAILED", "WARNING", "Exit submission failed; retry allowed",
                symbol=symbol, reason=exit_signal.reason, error=str(exc)[:200],
            )
            self._persist(symbol)
            raise

        if record.order_id not in managed.exit_order_ids:
            managed.exit_order_ids.append(record.order_id)
        if not managed.closed:
            managed.exit_order_id = record.order_id
        self._event(
            "POSITION_EXIT_SUBMITTED", "INFO", "Exit order submitted",
            symbol=symbol, order_id=record.order_id, reason=exit_signal.reason,
            quantity=exit_signal.quantity,
        )
        self._sync(symbol)
        return record

    def eod_sweep(self, timestamp: datetime) -> list[ExecutionRecord]:
        """Flatten every managed position inside the force-exit window.
        Safe to call repeatedly: pending exits are not duplicated, and a
        rejected EOD exit is retried on the next call."""
        session = self.monitor.session_manager
        if not session.is_force_exit_window(timestamp):
            return []

        submitted: list[ExecutionRecord] = []
        om = self.order_manager
        for symbol, managed in list(self.managed.items()):
            if managed.closed:
                continue

            entry = om.records.get(managed.entry_order_id)
            if managed.position is None and entry is not None and entry.status not in TERMINAL_STATUSES:
                try:
                    om.cancel(managed.entry_order_id)
                except Exception as exc:
                    self._event(
                        "ENTRY_REMAINDER_CANCEL_FAILED", "WARNING",
                        "Could not cancel pending entry at EOD",
                        symbol=symbol, order_id=managed.entry_order_id, error=str(exc)[:200],
                    )

            self._sync(symbol)
            if managed.closed or managed.position is None or managed.exit_order_id is not None:
                continue

            net = om.expected_positions.get(symbol, 0)
            if net == 0:
                continue

            state = self.monitor.state(symbol)
            if state is not None:
                state.exit_emitted = True

            position = managed.position
            signal = Signal(
                signal_id=f"EOD-{uuid4()}",
                timestamp=timestamp,
                symbol=symbol,
                side=SignalSide.SELL if net > 0 else SignalSide.BUY,
                signal_type=SignalType.EXIT,
                entry_price=position.current_price or position.average_entry,
                quantity=abs(net),
                reason=EOD_FORCE_EXIT,
                risk_status="ALLOWED",
            )
            try:
                submitted.append(self._submit_exit(managed, signal))
            except Exception as exc:
                self._event(
                    "EOD_EXIT_FAILED", "CRITICAL", "EOD force exit could not be submitted",
                    symbol=symbol, error=str(exc)[:200],
                )
        return submitted

    def open_symbols(self) -> list[str]:
        return [s for s, m in self.managed.items() if not m.closed]

    # --------------------------------------------------
    # State reconciliation
    # --------------------------------------------------

    def _sync(self, symbol: str) -> None:
        managed = self.managed[symbol]
        if managed.closed:
            return

        om = self.order_manager
        records = om.records
        entry = records.get(managed.entry_order_id)
        net = om.expected_positions.get(symbol, 0)

        if managed.position is None:
            if entry is None or entry.filled_quantity <= 0 or net == 0:
                if entry is not None and entry.filled_quantity <= 0 and entry.status in TERMINAL_STATUSES:
                    managed.closed = True
                    managed.close_reason = "ENTRY_NOT_FILLED"
                self._persist(symbol)
                return
            managed.position = self._open_position(managed, entry, abs(net))
            if self.monitor.state(symbol) is None:
                self.monitor.register(managed.position)
            self._event(
                "POSITION_OPENED", "INFO", "Position opened from broker-confirmed fill",
                symbol=symbol, side=managed.position.side, quantity=managed.position.quantity,
                average_entry=managed.position.average_entry,
            )
            self._persist(symbol)
            return

        position = managed.position
        if net != 0:
            position.quantity = abs(net)
            position.average_entry = om.position_avg.get(symbol, position.average_entry)

        if managed.exit_order_ids:
            managed.realized_pnl = sum(
                records[oid].realized_pnl for oid in managed.exit_order_ids if oid in records
            )

        if managed.exit_order_id is not None and managed.exit_order_id not in records:
            managed.exit_order_id = None

        if managed.exit_order_id is None:
            external = self._latest_exit_record(managed)
            if external is not None:
                managed.exit_order_id = external.order_id
                managed.exit_order_ids.append(external.order_id)
                managed.exit_reason = managed.exit_reason or external.reason
                state = self.monitor.state(symbol)
                if state is not None and external.status not in TERMINAL_STATUSES:
                    state.exit_emitted = True
            elif net == 0:
                managed.closed = True
                managed.close_reason = "POSITION_NOT_FOUND"
                self.monitor.mark_exit_completed(symbol)
                self._event(
                    "POSITION_CLOSED_EXTERNALLY", "WARNING",
                    "Net position is zero without a tracked exit order",
                    symbol=symbol,
                )
                self._persist(symbol)
                return

        if managed.exit_order_id is None:
            self._persist(symbol)
            return

        exit_record = records[managed.exit_order_id]

        if net == 0 and exit_record.filled_quantity > 0:
            self._close(managed, exit_record)
            return

        if exit_record.status in TERMINAL_STATUSES:
            managed.exit_order_id = None
            managed.exit_reason = None
            self._allow_exit_retry(symbol)
            self._event(
                "POSITION_EXIT_INCOMPLETE", "WARNING",
                "Exit order ended without flattening the position; exit may be retried",
                symbol=symbol, order_id=exit_record.order_id, status=exit_record.status.value,
                remaining=abs(net),
            )
        self._persist(symbol)

    def _latest_exit_record(self, managed: ManagedPosition) -> ExecutionRecord | None:
        ids = list(self.order_manager.records)
        start = ids.index(managed.entry_order_id) + 1 if managed.entry_order_id in ids else 0
        for order_id in reversed(ids[start:]):
            record = self.order_manager.records[order_id]
            if (
                record.symbol == managed.entry_signal.symbol
                and record.signal_type == SignalType.EXIT.value
                and (record.filled_quantity > 0 or record.status not in TERMINAL_STATUSES)
                and order_id not in managed.exit_order_ids
            ):
                return record
        return None

    def _open_position(
        self,
        managed: ManagedPosition,
        entry: ExecutionRecord,
        quantity: int,
    ) -> Position:
        signal = managed.entry_signal
        fill_price = (
            self.order_manager.position_avg.get(signal.symbol)
            or entry.average_fill_price
            or signal.entry_price
        )
        return Position(
            symbol=signal.symbol,
            side="LONG" if signal.side == SignalSide.BUY else "SHORT",
            quantity=quantity,
            average_entry=fill_price,
            current_price=fill_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            entry_time=signal.timestamp,
        )

    def _close(
        self,
        managed: ManagedPosition,
        exit_record: ExecutionRecord,
    ) -> None:
        position = managed.position
        records = self.order_manager.records
        realized = sum(
            records[oid].realized_pnl for oid in managed.exit_order_ids if oid in records
        )
        exit_price = exit_record.average_fill_price or exit_record.requested_price

        position.current_price = exit_price
        position.realized_pnl = realized
        position.unrealized_pnl = 0.0
        managed.realized_pnl = realized
        managed.closed = True
        managed.close_reason = managed.exit_reason or exit_record.reason
        self.monitor.mark_exit_completed(position.symbol)
        self._event(
            "POSITION_CLOSED", "INFO", "Position closed from broker-confirmed exit fill",
            symbol=position.symbol, reason=managed.close_reason, realized_pnl=realized,
            exit_order_id=exit_record.order_id,
        )
        self._persist(position.symbol)

    def _allow_exit_retry(self, symbol: str) -> None:
        state = self.monitor.state(symbol)
        if state is not None:
            state.exit_emitted = False

    # --------------------------------------------------
    # Restart recovery
    # --------------------------------------------------

    def recover(self) -> list[str]:
        """Rebuild managed positions from the database. OrderManager must be
        restored/reconciled first so order and fill state is current.
        Idempotent: symbols already managed in memory are skipped."""
        if self.database is None:
            return []

        recovered: list[str] = []
        om = self.order_manager
        for payload in self.database.get_managed_positions():
            signal = Signal.model_validate(payload["entry_signal"])
            symbol = signal.symbol
            current = self.managed.get(symbol)
            if current is not None and not current.closed:
                continue

            managed = ManagedPosition(
                entry_signal=signal,
                entry_order_id=payload["entry_order_id"],
                position=Position.model_validate(payload["position"]) if payload.get("position") else None,
                exit_order_id=payload.get("exit_order_id"),
                exit_reason=payload.get("exit_reason"),
                closed=bool(payload.get("closed")),
                realized_pnl=payload.get("realized_pnl"),
                exit_order_ids=list(payload.get("exit_order_ids") or []),
                close_reason=payload.get("close_reason"),
            )
            self.managed[symbol] = managed
            self._persisted.pop(symbol, None)

            state_data = payload.get("monitor_state")
            if state_data:
                state = self.monitor.restore_state(state_data)
                exit_record = om.records.get(managed.exit_order_id) if managed.exit_order_id else None
                state.exit_emitted = bool(exit_record and exit_record.status not in TERMINAL_STATUSES)
            elif managed.position is not None:
                self.monitor.register(managed.position)

            if managed.entry_order_id not in om.records:
                self._event(
                    "LIFECYCLE_RECOVERY_ENTRY_UNKNOWN", "WARNING",
                    "Persisted managed position references an unknown entry order",
                    symbol=symbol, order_id=managed.entry_order_id,
                )

            self._sync(symbol)
            self._event(
                "LIFECYCLE_RECOVERED", "INFO", "Managed position restored after restart",
                symbol=symbol,
                closed=managed.closed,
                quantity=managed.position.quantity if managed.position else 0,
                active_stop=(self.monitor.state(symbol).active_stop if self.monitor.state(symbol) else None),
            )
            recovered.append(symbol)
        return recovered
