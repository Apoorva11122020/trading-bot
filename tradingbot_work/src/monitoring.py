"""Read-only monitoring snapshot used by the dashboard/API."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class MonitoringSnapshot:
    bot_status: str = "RUNNING"
    broker_status: str = "DISCONNECTED"
    market_data_status: str = "DISCONNECTED"
    trading_mode: str = "PAPER"
    trading_enabled: bool = True
    risk_status: str = "NORMAL"
    available_balance: float = 0.0
    used_margin: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    positions: list[dict[str, Any]] | None = None
    orders: list[dict[str, Any]] | None = None

    def as_dict(self):
        data = asdict(self)
        data["positions"] = self.positions or []
        data["orders"] = self.orders or []
        return data
