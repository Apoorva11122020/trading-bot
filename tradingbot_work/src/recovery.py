from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.broker.interface import BrokerInterface
from src.broker.models import BrokerOrder, OrderStatus
from src.database import Database
from src.execution.order_manager import OrderManager


ACTIONABLE = {
    OrderStatus.OPEN,
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.MODIFIED,
    OrderStatus.SUBMITTED,
}


@dataclass(frozen=True)
class RecoveryReport:
    broker_connected: bool
    broker_positions: dict[str, int]
    persisted_positions: dict[str, int]
    reconstructed_positions: dict[str, int]
    open_orders: int
    order_mismatches: dict[str, Any]
    position_mismatches: dict[str, Any]
    restored_orders: int = 0
    refreshed_orders: list[str] = field(default_factory=list)
    recovered_lifecycles: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.broker_connected
            and not self.order_mismatches
            and not self.position_mismatches
        )


def _signed(row) -> int:
    qty = abs(int(row["quantity"]))
    return -qty if str(row["side"]).upper() in {"SHORT", "SELL"} else qty


class RecoveryManager:
    """Conservative restart/recovery coordinator.

    Order of operations (idempotent, safe to run repeatedly):
    1. restore local orders/positions from the database,
    2. replay the broker's current order state through OrderManager's
       fill state machine (fills made while down are applied exactly once),
    3. reconcile net positions against the broker (broker is authoritative),
    4. rebuild managed position lifecycles (stops, trailing state).

    Unknown broker orders are reported and never resubmitted or cancelled.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        database: Database,
        order_manager: OrderManager,
        lifecycle: Any | None = None,
    ):
        self.broker = broker
        self.database = database
        self.order_manager = order_manager
        self.lifecycle = lifecycle

    def recover(self) -> RecoveryReport:
        if not self.broker.is_connected():
            raise RuntimeError("BROKER_DISCONNECTED_DURING_RECOVERY")

        om = self.order_manager
        persisted_positions = {
            str(row["symbol"]): _signed(row)
            for row in self.database.get_positions()
            if int(row["quantity"]) != 0
        }

        restored = om.restore_from_database()
        broker_orders = [BrokerOrder(**row) for row in self.broker.get_orders()]
        refreshed = [o.order_id for o in om.recover_open_orders()]

        broker_positions = {
            str(row["symbol"]): int(row["quantity"])
            for row in self.broker.get_positions()
            if int(row["quantity"]) != 0
        }
        broker_avg = {
            str(row["symbol"]): float(row.get("average_price") or 0.0)
            for row in self.broker.get_positions()
        }

        unknown_actionable = {
            o.order_id: o.status.value
            for o in broker_orders
            if o.status in ACTIONABLE and o.order_id not in om.records
        }

        expected = dict(om.expected_positions)
        position_mismatches = {
            symbol: {
                "persisted": persisted_positions.get(symbol, 0),
                "expected": expected.get(symbol, 0),
                "broker": broker_positions.get(symbol, 0),
            }
            for symbol in set(expected) | set(broker_positions)
            if expected.get(symbol, 0) != broker_positions.get(symbol, 0)
        }

        # A clean restart may find broker positions with no local snapshot at
        # all (e.g. database lost). Adopt broker truth so exits stay possible;
        # these positions have no stored stop/target and are flagged.
        if not persisted_positions and not expected and broker_positions:
            self.database.add_event(
                "RECOVERY_UNMANAGED_POSITIONS", "WARNING",
                "Broker positions adopted without local state; no stop/target is known for them",
                {"positions": broker_positions},
            )
            position_mismatches = {}

        # Broker is authoritative for net quantity.
        for symbol in set(expected) | set(broker_positions):
            qty = broker_positions.get(symbol, 0)
            if qty == 0:
                om.expected_positions.pop(symbol, None)
                om.position_avg.pop(symbol, None)
                self.database.delete_position(symbol)
                continue
            om.expected_positions[symbol] = qty
            average = om.position_avg.get(symbol) or broker_avg.get(symbol) or 0.0
            if average > 0:
                om.position_avg[symbol] = average
            self.database.upsert_position({
                "symbol": symbol,
                "side": "LONG" if qty > 0 else "SHORT",
                "quantity": abs(qty),
                "average_entry": average,
                "current_price": None,
                "unrealized_pnl": None,
            })

        if position_mismatches:
            self.database.add_event(
                "RECOVERY_MISMATCH", "WARNING", "Broker and local positions differ; broker state adopted",
                {"mismatches": position_mismatches},
            )

        if unknown_actionable:
            self.database.add_event(
                "RECOVERY_UNKNOWN_ORDERS", "WARNING",
                "Open broker orders were found without local managed state; no automatic retry was submitted",
                {"orders": unknown_actionable},
            )

        lifecycles: list[str] = []
        if self.lifecycle is not None:
            lifecycles = self.lifecycle.recover()

        report = RecoveryReport(
            broker_connected=True,
            broker_positions=broker_positions,
            persisted_positions=persisted_positions,
            reconstructed_positions=dict(om.expected_positions),
            open_orders=len([o for o in broker_orders if o.status in ACTIONABLE]),
            order_mismatches=unknown_actionable,
            position_mismatches=position_mismatches,
            restored_orders=restored.get("orders", 0),
            refreshed_orders=refreshed,
            recovered_lifecycles=lifecycles,
        )
        self.database.add_event(
            "RECOVERY_COMPLETED", "INFO" if report.ok else "WARNING", "Startup reconciliation finished",
            {
                "ok": report.ok,
                "restored_orders": report.restored_orders,
                "refreshed_orders": len(refreshed),
                "open_orders": report.open_orders,
                "positions": report.reconstructed_positions,
                "lifecycles": lifecycles,
            },
        )
        return report
