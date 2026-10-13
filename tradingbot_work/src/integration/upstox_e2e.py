from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from src.broker.upstox_portfolio import OrderUpdate, PositionUpdate
from src.broker.upstox_sandbox import UpstoxSandboxBroker


@dataclass(frozen=True)
class SandboxE2EResult:
    order_id: str
    order_placed: bool
    order_update_seen: bool
    position_update_seen: bool
    notes: tuple[str, ...] = ()


class UpstoxSandboxE2EHarness:
    """Deterministic E2E harness with an optional real sandbox smoke path.

    The default mode never touches the network. The real mode is opt-in via
    ``UPSTOX_RUN_REAL_SANDBOX_E2E=true`` and requires a sandbox token.
    """

    def __init__(self, broker: UpstoxSandboxBroker | None = None):
        self.broker = broker or UpstoxSandboxBroker()

    def validate_configuration(self) -> None:
        if not os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "").strip():
            raise RuntimeError("UPSTOX_SANDBOX_ACCESS_TOKEN is required for real sandbox E2E")
        if os.getenv("TRADING_MODE", "PAPER").upper() == "LIVE":
            raise RuntimeError("Sandbox E2E cannot run while TRADING_MODE=LIVE")

    def run_real_order_smoke(
        self,
        *,
        instrument_token: str,
        side: str = "BUY",
        quantity: int = 1,
        order_type: str = "MARKET",
    ) -> str:
        """Place exactly one explicitly requested sandbox order.

        This method is intentionally separate from the application's trading
        loop and requires an explicit environment flag before execution.
        """
        if os.getenv("UPSTOX_RUN_REAL_SANDBOX_E2E", "false").lower() != "true":
            raise RuntimeError("Set UPSTOX_RUN_REAL_SANDBOX_E2E=true to enable the real sandbox smoke")
        self.validate_configuration()
        self.broker.connect()
        return self.broker.place_order(instrument_token, side, quantity, order_type)

    @staticmethod
    def normalize_event(event: OrderUpdate | PositionUpdate) -> dict[str, Any]:
        if isinstance(event, OrderUpdate):
            return {
                "type": "order",
                "order_id": event.order_id,
                "status": event.status,
                "filled_quantity": event.filled_quantity,
                "average_price": event.average_price,
            }
        return {
            "type": "position",
            "symbol": event.symbol,
            "quantity": event.quantity,
            "average_price": event.average_price,
            "unrealized_pnl": event.unrealized_pnl,
        }
