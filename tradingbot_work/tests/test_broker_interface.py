import pytest
from src.broker.paper import PaperBroker

def test_paper_broker_connection_and_ltp():
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("RELIANCE", 2500)
    assert broker.is_connected()
    assert broker.get_ltp("RELIANCE") == 2500

def test_paper_market_order_fills():
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("RELIANCE", 2500)
    order_id = broker.place_order("RELIANCE", "BUY", 10, "MARKET")
    order = broker.get_orders()[0]
    assert order["order_id"] == order_id
    assert order["status"] == "FILLED"
    assert order["filled_quantity"] == 10

def test_paper_broker_requires_connection():
    broker = PaperBroker()
    with pytest.raises(RuntimeError):
        broker.get_account()

def test_close_position():
    broker = PaperBroker()
    broker.connect()
    broker.set_ltp("TCS", 3000)
    broker.place_order("TCS", "BUY", 5, "MARKET")
    broker.close_position("TCS")
    assert broker.get_positions() == []
