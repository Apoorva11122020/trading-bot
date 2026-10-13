from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.broker.interface import BrokerInterface
from src.broker.models import BrokerOrder, OrderSide, OrderStatus, OrderType


WORKING_STATUSES = {
    OrderStatus.OPEN,
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.MODIFIED,
    OrderStatus.SUBMITTED,
}


class PaperBroker(BrokerInterface):
    """Local in-memory broker. Never contacts an exchange or real broker.

    MARKET orders fill at the last traded price. LIMIT orders fill only when
    marketable (BUY at/below the limit, SELL at/above it) and otherwise rest
    until `set_ltp` makes them marketable.

    The test-only controls make rejected, resting and partial-fill paths
    deterministic:
    - `set_next_execution` overrides the outcome of the next placed order.
    - `fill_order` adds an incremental (possibly partial) fill to a working order.
    - `reject_order` rejects a working order that has not filled.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.cash = initial_cash
        self.connected = False
        self.prices: dict[str, float] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self.positions: dict[str, dict[str, Any]] = {}
        self.realized_pnl: dict[str, float] = {}
        self._next_status = OrderStatus.FILLED
        self._next_filled_quantity: int | None = None
        self._next_rejection_reason: str | None = None
        self._next_explicit = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def _require_connection(self) -> None:
        if not self.connected:
            raise RuntimeError("Paper broker is not connected")

    def set_ltp(self, symbol: str, price: float) -> None:
        if price <= 0:
            raise ValueError("price must be > 0")
        self.prices[symbol] = price
        self._match_resting_limits(symbol)

    def get_ltp(self, symbol: str) -> float:
        self._require_connection()
        if symbol not in self.prices:
            raise KeyError(f"No paper price available for {symbol}")
        return self.prices[symbol]

    def get_account(self) -> dict[str, Any]:
        self._require_connection()
        return {
            "cash": self.cash,
            "realized_pnl": sum(self.realized_pnl.values()),
            "mode": "PAPER",
        }

    def get_positions(self) -> list[dict[str, Any]]:
        self._require_connection()
        return [dict(p) for p in self.positions.values()]

    def get_orders(self) -> list[dict[str, Any]]:
        self._require_connection()
        return [o.model_dump() for o in self.orders.values()]

    def set_next_execution(
        self,
        status: OrderStatus,
        filled_quantity: int | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        """Configure the next paper order for deterministic testing."""
        self._next_status = status
        self._next_filled_quantity = filled_quantity
        self._next_rejection_reason = rejection_reason
        self._next_explicit = True

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str,
        price: float | None = None,
        tag: str | None = None,
    ) -> str:
        self._require_connection()
        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        side_enum = OrderSide(side.upper())
        type_enum = OrderType(order_type.upper())

        if type_enum == OrderType.LIMIT and (price is None or price <= 0):
            raise ValueError("LIMIT orders require a positive price")
        if type_enum not in {OrderType.MARKET, OrderType.LIMIT}:
            raise ValueError(f"Paper broker does not support {type_enum.value} orders")

        ltp = self.prices.get(symbol)
        if type_enum == OrderType.MARKET and ltp is None:
            raise KeyError(f"No paper price available for {symbol}")

        if self._next_explicit:
            status = self._next_status
            filled_quantity = (
                quantity
                if status == OrderStatus.FILLED
                else (self._next_filled_quantity or 0)
            )
        elif type_enum == OrderType.LIMIT and not self._marketable(side_enum, price, ltp):
            status, filled_quantity = OrderStatus.OPEN, 0
        else:
            status, filled_quantity = OrderStatus.FILLED, quantity

        filled_quantity = max(0, min(quantity, filled_quantity))
        if status == OrderStatus.REJECTED:
            filled_quantity = 0

        fill_price = price if type_enum == OrderType.LIMIT else ltp

        order_id = f"PAPER-{uuid4()}"
        order = BrokerOrder(
            order_id=order_id,
            symbol=symbol,
            side=side_enum,
            quantity=quantity,
            order_type=type_enum,
            price=price,
            status=status,
            created_at=datetime.now(timezone.utc),
            filled_quantity=filled_quantity,
            average_fill_price=fill_price if filled_quantity else None,
            rejection_reason=self._next_rejection_reason,
            tag=tag,
        )
        self.orders[order_id] = order

        if filled_quantity:
            self._apply_position(symbol, side_enum, filled_quantity, fill_price)

        # Reset one-shot simulation settings.
        self._next_status = OrderStatus.FILLED
        self._next_filled_quantity = None
        self._next_rejection_reason = None
        self._next_explicit = False

        return order_id

    @staticmethod
    def _marketable(side: OrderSide, limit: float, ltp: float | None) -> bool:
        if ltp is None:
            return False
        return ltp <= limit if side == OrderSide.BUY else ltp >= limit

    def _match_resting_limits(self, symbol: str) -> None:
        ltp = self.prices[symbol]
        for order in list(self.orders.values()):
            if (
                order.symbol == symbol
                and order.order_type == OrderType.LIMIT
                and order.status in WORKING_STATUSES
                and self._marketable(order.side, order.price, ltp)
            ):
                self.fill_order(order.order_id, order.quantity - order.filled_quantity, order.price)

    # ------------------------------------------------------------------
    # Controlled lifecycle (deterministic tests)
    # ------------------------------------------------------------------

    def fill_order(self, order_id: str, quantity: int, price: float | None = None) -> BrokerOrder:
        """Add an incremental fill of `quantity` to a working order."""
        order = self.orders[order_id]
        if order.status not in WORKING_STATUSES:
            raise RuntimeError(f"Cannot fill order in status {order.status.value}")
        remaining = order.quantity - order.filled_quantity
        if quantity <= 0 or quantity > remaining:
            raise ValueError(f"fill quantity must be within 1..{remaining}")

        fill_price = float(price if price is not None else self.prices[order.symbol])
        new_filled = order.filled_quantity + quantity
        previous_value = (order.average_fill_price or 0.0) * order.filled_quantity
        average = (previous_value + fill_price * quantity) / new_filled

        self.orders[order_id] = order.model_copy(update={
            "filled_quantity": new_filled,
            "average_fill_price": average,
            "status": OrderStatus.FILLED if new_filled == order.quantity else OrderStatus.PARTIALLY_FILLED,
        })
        self._apply_position(order.symbol, order.side, quantity, fill_price)
        return self.orders[order_id]

    def reject_order(self, order_id: str, reason: str) -> BrokerOrder:
        order = self.orders[order_id]
        if order.status not in WORKING_STATUSES or order.filled_quantity:
            raise RuntimeError("Only unfilled working orders can be rejected")
        self.orders[order_id] = order.model_copy(update={
            "status": OrderStatus.REJECTED,
            "rejection_reason": reason,
        })
        return self.orders[order_id]

    # ------------------------------------------------------------------
    # Positions / P&L
    # ------------------------------------------------------------------

    def _apply_position(self, symbol: str, side: OrderSide, quantity: int, price: float) -> None:
        signed = quantity if side == OrderSide.BUY else -quantity
        current = self.positions.get(symbol)
        if current is None:
            self.positions[symbol] = {"symbol": symbol, "quantity": signed, "average_price": price}
            return

        old_qty = current["quantity"]
        old_avg = current["average_price"]
        new_qty = old_qty + signed

        if old_qty * signed > 0:
            current["average_price"] = (old_avg * abs(old_qty) + price * quantity) / abs(new_qty)
        else:
            closed = min(abs(old_qty), quantity)
            direction = 1 if old_qty > 0 else -1
            self.realized_pnl[symbol] = (
                self.realized_pnl.get(symbol, 0.0) + (price - old_avg) * closed * direction
            )
            if new_qty != 0 and new_qty * old_qty < 0:
                current["average_price"] = price

        if new_qty == 0:
            del self.positions[symbol]
        else:
            current["quantity"] = new_qty

    # ------------------------------------------------------------------
    # Modify / cancel / close
    # ------------------------------------------------------------------

    def modify_order(self, order_id: str, **kwargs: Any) -> None:
        self._require_connection()
        order = self.orders[order_id]
        if order.status in {
            OrderStatus.CANCELLED,
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
        }:
            raise RuntimeError(f"Cannot modify order in status {order.status}")
        self.orders[order_id] = order.model_copy(
            update=kwargs | {"status": OrderStatus.MODIFIED}
        )

    def cancel_order(self, order_id: str) -> None:
        self._require_connection()
        order = self.orders[order_id]
        if order.status == OrderStatus.FILLED:
            raise RuntimeError("Cannot cancel a filled order")
        if order.status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}:
            return
        self.orders[order_id] = order.model_copy(
            update={"status": OrderStatus.CANCELLED}
        )

    def close_position(self, symbol: str) -> str:
        self._require_connection()
        position = self.positions.get(symbol)
        if not position or position["quantity"] == 0:
            raise RuntimeError(f"No open paper position for {symbol}")
        side = "SELL" if position["quantity"] > 0 else "BUY"
        return self.place_order(symbol, side, abs(position["quantity"]), "MARKET")
