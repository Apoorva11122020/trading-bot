from __future__ import annotations
import os
from src.broker.interface import BrokerInterface
from src.broker.paper import PaperBroker
from src.broker.adapters import BrokerAdapterError, BrokerCredentials, UnconfiguredBrokerAdapter
from src.broker.upstox_sandbox import UpstoxSandboxBroker

SUPPORTED_PROVIDERS = {"PAPER", "UPSTOX_SANDBOX", "ZERODHA", "UPSTOX", "ANGEL_ONE", "DHAN", "FYERS"}

class BrokerNotConfiguredError(BrokerAdapterError):
    pass

def _credentials() -> BrokerCredentials:
    return BrokerCredentials(
        api_key=os.getenv("BROKER_API_KEY", ""),
        api_secret=os.getenv("BROKER_API_SECRET"),
        access_token=os.getenv("ACCESS_TOKEN"),
    )

def create_broker(name: str | None = None) -> BrokerInterface:
    provider = (name or os.getenv("BROKER", "PAPER")).strip().upper()
    if provider not in SUPPORTED_PROVIDERS:
        raise BrokerNotConfiguredError(f"Unsupported broker provider: {provider}")
    if provider == "PAPER":
        return PaperBroker()
    if provider == "UPSTOX_SANDBOX":
        return UpstoxSandboxBroker()
    # Provider adapters are deliberately unavailable until the client broker/API contract is confirmed.
    raise BrokerNotConfiguredError(
        f"Broker {provider} is recognized but not configured. Supply the client API contract and credentials before enabling it."
    )
