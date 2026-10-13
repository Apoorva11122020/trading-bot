import json
import pytest

from src.broker.upstox_portfolio import (
    OrderUpdate, PositionUpdate, UpstoxPortfolioStreamError,
    parse_portfolio_message,
)


def test_parse_order_update():
    u = parse_portfolio_message(json.dumps({
        "update_type":"order", "order_id":"OID1", "status":"complete",
        "instrument_key":"NSE_EQ|ABC", "transaction_type":"BUY", "quantity":10,
        "filled_quantity":10, "pending_quantity":0, "average_price":123.45,
        "exchange_order_id":"EX1"
    }))
    assert isinstance(u, OrderUpdate)
    assert u.order_id == "OID1" and u.filled_quantity == 10 and u.average_price == 123.45


def test_parse_position_update():
    u = parse_portfolio_message(json.dumps({
        "update_type":"position", "instrument_key":"NSE_EQ|ABC", "trading_symbol":"ABC-EQ",
        "quantity":10, "average_price":123.0, "buy_value":1230
    }))
    assert isinstance(u, PositionUpdate)
    assert u.symbol == "ABC-EQ" and u.quantity == 10


def test_unknown_update_is_ignored():
    assert parse_portfolio_message('{"update_type":"holding"}') is None


def test_invalid_json_fails_closed():
    with pytest.raises(UpstoxPortfolioStreamError):
        parse_portfolio_message(b"not-json")
