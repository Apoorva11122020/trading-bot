from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from src.broker.interface import BrokerInterface


class BrokerAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrokerCredentials:
    api_key: str
    api_secret: str | None = None
    access_token: str | None = None

    def validate(self) -> None:
        if not self.api_key.strip():
            raise BrokerAdapterError("BROKER_API_KEY is required")


class UnconfiguredBrokerAdapter(BrokerInterface):
    """Fail-closed placeholder until a client's broker/API contract is supplied.

    This class intentionally makes no network requests and cannot place live orders.
    """
    def __init__(self, provider: str, credentials: BrokerCredentials | None = None):
        self.provider = provider.upper()
        self.credentials = credentials
        self._connected = False

    def _not_configured(self) -> None:
        raise BrokerAdapterError(
            f"{self.provider} adapter is not configured: supply the broker API contract and credentials before enabling it"
        )

    def connect(self) -> None: self._not_configured()
    def disconnect(self) -> None: self._connected = False
    def is_connected(self) -> bool: return self._connected
    def get_ltp(self, symbol: str) -> float: self._not_configured(); return 0.0
    def get_account(self) -> dict[str, Any]: self._not_configured(); return {}
    def get_positions(self) -> list[dict[str, Any]]: self._not_configured(); return []
    def get_orders(self) -> list[dict[str, Any]]: self._not_configured(); return []
    def place_order(self, symbol: str, side: str, quantity: int, order_type: str, price: float | None = None, tag: str | None = None) -> str:
        self._not_configured(); return ""
    def modify_order(self, order_id: str, **kwargs: Any) -> None: self._not_configured()
    def cancel_order(self, order_id: str) -> None: self._not_configured()
    def close_position(self, symbol: str) -> str: self._not_configured(); return ""
