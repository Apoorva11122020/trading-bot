import pytest
from src.broker.adapters import BrokerAdapterError, BrokerCredentials, UnconfiguredBrokerAdapter
from src.broker.factory import create_broker
from src.broker.paper import PaperBroker


def test_factory_defaults_to_paper(monkeypatch):
    monkeypatch.delenv("BROKER", raising=False)
    assert isinstance(create_broker(), PaperBroker)

@pytest.mark.parametrize("provider", ["ZERODHA", "UPSTOX", "ANGEL_ONE", "DHAN", "FYERS"])
def test_real_provider_is_fail_closed(provider):
    from src.broker.factory import BrokerNotConfiguredError
    with pytest.raises(BrokerNotConfiguredError):
        create_broker(provider)


def test_credentials_require_api_key():
    with pytest.raises(BrokerAdapterError):
        BrokerCredentials("").validate()


def test_unknown_provider_rejected():
    with pytest.raises(BrokerAdapterError):
        create_broker("UNKNOWN")
