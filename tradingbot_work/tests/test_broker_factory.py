import pytest

from src.broker.factory import BrokerNotConfiguredError, create_broker
from src.broker.paper import PaperBroker


def test_factory_defaults_to_paper(monkeypatch):
    monkeypatch.delenv("BROKER", raising=False)
    assert isinstance(create_broker(), PaperBroker)


def test_factory_rejects_unconfigured_provider(monkeypatch):
    monkeypatch.setenv("BROKER", "ZERODHA")
    with pytest.raises(BrokerNotConfiguredError):
        create_broker()
