import pytest

from src.broker.factory import create_broker
from src.broker.upstox_sandbox import UpstoxSandboxBroker, UpstoxSandboxError


def test_factory_creates_upstox_sandbox(monkeypatch):
    monkeypatch.setenv("UPSTOX_SANDBOX_ACCESS_TOKEN", "demo-token")
    broker = create_broker("UPSTOX_SANDBOX")
    assert isinstance(broker, UpstoxSandboxBroker)
    broker.connect()
    assert broker.is_connected()
    broker.disconnect()
    assert not broker.is_connected()


def test_sandbox_requires_token(monkeypatch):
    monkeypatch.delenv("UPSTOX_SANDBOX_ACCESS_TOKEN", raising=False)
    broker = UpstoxSandboxBroker()
    with pytest.raises(UpstoxSandboxError):
        broker.connect()


def test_sandbox_place_order_uses_sandbox_host(monkeypatch):
    broker = UpstoxSandboxBroker("token")
    broker.connect()
    captured = {}

    class Response:
        status_code = 200
        text = ''
        def json(self):
            return {"data": {"order_id": "SBX-123"}}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, kwargs=kwargs)
        return Response()

    monkeypatch.setattr("requests.request", fake_request)
    order_id = broker.place_order("NSE_EQ|INE669E01016", "BUY", 1, "MARKET")
    assert order_id == "SBX-123"
    assert captured["url"] == "https://api-sandbox.upstox.com/v3/order/place"
    assert captured["kwargs"]["json"]["transaction_type"] == "BUY"


def test_sandbox_rejects_live_host(monkeypatch):
    broker = UpstoxSandboxBroker("token")
    assert broker.BASE_URL == "https://api-sandbox.upstox.com/v3"
    assert "api.upstox.com" not in broker.BASE_URL
