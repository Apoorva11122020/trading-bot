from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.broker.models import OrderStatus
from src.broker.upstox_portfolio import OrderUpdate
from src.broker.upstox_sandbox import UpstoxSandboxBroker
from src.config.loader import load_strategy_config
from src.database import Database
from src.execution.order_manager import OrderManager
from src.market_data.models import MarketDataHealth, MarketDataStatus
from src.models.market import Candle
from src.models.risk import AccountState
from src.models.signals import Signal, SignalSide, SignalType
from src.position.lifecycle import PositionLifecycle
from src.position.monitor import PositionMonitor
from src.risk.risk_manager import RiskManager
from src.safety.live_gate import assert_startup_safe
from src.session.session_manager import SessionManager
from src.strategy.runtime import StrategyRuntime
from src.strategy.signal_generator import SignalGenerator
from src.strategy.strategy_engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")
SANDBOX_HOST = "https://api-sandbox.upstox.com"
INSTRUMENT = "NSE_EQ|INE669E01016"


# ---------------------------------------------------------------------------
# Controlled Upstox Sandbox (HTTP layer only)
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload


class FakeUpstoxSandbox:
    """Stands in for api-sandbox.upstox.com. Orders stay `open` until the
    test explicitly fills or rejects them, so nothing is assumed filled."""

    def __init__(self):
        self.orders: dict[str, dict] = {}
        self.requests: list[tuple[str, str]] = []
        self._seq = 0

    def __call__(self, method, url, headers=None, json=None, params=None, timeout=None):
        self.requests.append((method, url))
        assert url.startswith(SANDBOX_HOST + "/"), f"non-sandbox URL: {url}"
        assert headers["Authorization"] == "Bearer test-token"
        path = url[len(SANDBOX_HOST):]

        if method == "POST" and path == "/v3/order/place":
            self._seq += 1
            order_id = f"SBX-{self._seq}"
            self.orders[order_id] = {
                "order_id": order_id,
                "instrument_token": json["instrument_token"],
                "transaction_type": json["transaction_type"],
                "order_type": json["order_type"],
                "quantity": json["quantity"],
                "price": json["price"],
                "filled_quantity": 0,
                "average_price": 0.0,
                "status": "open",
                "order_timestamp": "2026-10-01 14:00:00",
            }
            return _Response({"status": "success", "data": {"order_ids": [order_id]}})

        if method == "GET" and path == "/v2/order/retrieve-all":
            return _Response({"status": "success", "data": list(self.orders.values())})

        if method == "GET" and path == "/v3/portfolio/short-term-positions":
            return _Response({"status": "success", "data": []})

        raise AssertionError(f"unexpected sandbox call {method} {path}")

    def fill(self, order_id, price):
        order = self.orders[order_id]
        order.update(
            status="complete",
            filled_quantity=order["quantity"],
            average_price=price,
        )

    def reject(self, order_id, message):
        self.orders[order_id].update(status="rejected", status_message=message)

    def placed(self, side):
        return [o for o in self.orders.values() if o["transaction_type"] == side]


# ---------------------------------------------------------------------------
# Actual strategy entry signal
# ---------------------------------------------------------------------------

def _breakout_candles(count=55):
    start = datetime(2026, 10, 1, 9, 30, tzinfo=IST)
    rows, price = [], 100.0
    for i in range(count - 1):
        price += 0.15
        rows.append({"timestamp": start + timedelta(minutes=5 * i), "symbol": "RELIANCE",
                     "open": price - 0.10, "high": price + 0.20, "low": price - 0.20,
                     "close": price, "volume": 100.0})
    price += 5.0
    rows.append({"timestamp": start + timedelta(minutes=5 * (count - 1)), "symbol": "RELIANCE",
                 "open": price - 0.20, "high": price + 0.20, "low": price - 0.30,
                 "close": price, "volume": 200.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def strategy_entry() -> Signal:
    cfg = load_strategy_config("config/strategy.yaml")
    engine = StrategyEngine(cfg, RiskManager(cfg.risk), SessionManager(cfg.session))
    runtime = StrategyRuntime(SignalGenerator(engine))
    health = MarketDataHealth(status=MarketDataStatus.CONNECTED,
                              last_tick_at=datetime.now().astimezone(),
                              subscribed_symbols=["RELIANCE"])
    signal = None
    for row in _breakout_candles().to_dict("records"):
        signal = runtime.on_candle(Candle(**row), AccountState(equity=100_000), health) or signal

    assert signal is not None
    assert signal.signal_type == SignalType.ENTRY and signal.side == SignalSide.BUY
    assert signal.risk_status == "ALLOWED" and not signal.blocked
    # Broker orders are keyed by the Upstox instrument key.
    return signal.model_copy(update={"symbol": INSTRUMENT})


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self, monkeypatch, tmp_path, entry: Signal):
        monkeypatch.setenv("TRADING_MODE", "PAPER")
        monkeypatch.setenv("LIVE_TRADING_ENABLED", "false")
        assert_startup_safe()

        self.cfg = load_strategy_config("config/strategy.yaml")
        assert self.cfg.system.trading_mode == "PAPER"
        assert self.cfg.system.live_trading_enabled is False

        self.sandbox = FakeUpstoxSandbox()
        monkeypatch.setattr("requests.request", self.sandbox)

        self.broker = UpstoxSandboxBroker("test-token")
        self.broker.connect()
        self.db = Database(tmp_path / "lifecycle.db")
        session = SessionManager(self.cfg.session)
        self.om = OrderManager(self.broker, RiskManager(self.cfg.risk), session, database=self.db)
        self.monitor = PositionMonitor(self.cfg, session)
        self.lifecycle = PositionLifecycle(
            self.om, self.monitor, lambda: AccountState(equity=100_000)
        )

        self.entry = entry
        self.E = entry.entry_price
        self.R = entry.entry_price - entry.stop_loss
        self.atr = self.R / self.cfg.exit.stop_loss.multiplier
        self.qty = entry.quantity

    def ts(self, minute):
        return datetime(2026, 10, 1, 14, minute, tzinfo=IST)

    def open_filled_position(self):
        record = self.lifecycle.submit_entry(self.entry)
        self.sandbox.fill(record.order_id, self.E)
        self.lifecycle.refresh(INSTRUMENT)
        return record

    def manual_exit(self, signal_id, quantity=None, side=SignalSide.SELL):
        return Signal(signal_id=signal_id, timestamp=self.ts(30), symbol=INSTRUMENT, side=side,
                      signal_type=SignalType.EXIT, entry_price=self.E,
                      quantity=quantity or self.qty, reason="MANUAL", risk_status="ALLOWED")

    @property
    def managed(self):
        return self.lifecycle.managed[INSTRUMENT]


@pytest.fixture
def h(monkeypatch, tmp_path, strategy_entry):
    harness = Harness(monkeypatch, tmp_path, strategy_entry)
    yield harness
    harness.db.close()
    assert all(url.startswith(SANDBOX_HOST) for _, url in harness.sandbox.requests)
    assert not any("//api.upstox.com" in url for _, url in harness.sandbox.requests)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_submitted_entry_is_not_treated_as_filled(h):
    record = h.lifecycle.submit_entry(h.entry)

    assert record.status == OrderStatus.OPEN
    assert record.filled_quantity == 0
    assert record.average_fill_price is None
    assert h.managed.position is None
    assert h.om.expected_positions == {}

    h.lifecycle.refresh(INSTRUMENT)
    assert h.managed.position is None

    assert h.lifecycle.on_price(INSTRUMENT, h.entry.stop_loss - 1, h.atr, h.ts(5)) is None
    with pytest.raises(RuntimeError, match="NO_OPEN_POSITION_TO_EXIT"):
        h.om.submit_signal(h.manual_exit("EXIT-UNFILLED"), AccountState(equity=100_000))

    assert h.sandbox.placed("SELL") == []
    assert h.db.counts()["trades"] == 0


def test_target_exit_full_lifecycle(h):
    entry_record = h.lifecycle.submit_entry(h.entry)
    assert entry_record.status == OrderStatus.OPEN
    assert h.sandbox.placed("BUY")[0]["quantity"] == h.qty

    # Controlled broker fill -> position created from broker-reported fill.
    h.sandbox.fill(entry_record.order_id, h.E)
    h.lifecycle.refresh(INSTRUMENT)

    assert entry_record.status == OrderStatus.FILLED
    assert entry_record.filled_quantity == h.qty
    assert entry_record.average_fill_price == pytest.approx(h.E)
    position = h.managed.position
    assert position.side == "LONG"
    assert position.quantity == h.qty
    assert position.average_entry == pytest.approx(h.E)
    assert position.stop_loss == h.entry.stop_loss
    assert position.target == h.entry.target
    assert h.om.expected_positions == {INSTRUMENT: h.qty}
    open_trade = h.db.get_trades()[0]
    assert open_trade["status"] == "OPEN"
    assert open_trade["entry_price"] == pytest.approx(h.E)

    # Monitoring below target: no exit, unrealized P&L tracked.
    assert h.lifecycle.on_price(INSTRUMENT, h.E + 0.5 * h.R, h.atr, h.ts(5)) is None
    assert position.unrealized_pnl == pytest.approx(0.5 * h.R * h.qty)

    # Target -> exit signal -> sandbox SELL order through OrderManager.
    exit_record = h.lifecycle.on_price(INSTRUMENT, h.entry.target, h.atr, h.ts(10))
    assert h.managed.exit_reason == "TARGET"
    assert exit_record.side == "SELL"
    assert exit_record.requested_quantity == h.qty
    assert exit_record.status == OrderStatus.OPEN
    assert ("POST", f"{SANDBOX_HOST}/v3/order/place") in h.sandbox.requests
    sell = h.sandbox.placed("SELL")
    assert len(sell) == 1 and sell[0]["order_type"] == "MARKET" and sell[0]["quantity"] == h.qty

    # Exit submitted but not filled: position stays open.
    assert h.managed.closed is False
    assert h.om.expected_positions == {INSTRUMENT: h.qty}

    # Duplicate-exit prevention while the exit is pending.
    assert h.lifecycle.on_price(INSTRUMENT, h.entry.target + 1, h.atr, h.ts(11)) is None
    with pytest.raises(RuntimeError, match="EXIT_ALREADY_PENDING"):
        h.om.submit_signal(h.manual_exit("EXIT-DUP"), AccountState(equity=100_000))
    assert len(h.sandbox.placed("SELL")) == 1

    # Controlled exit fill -> broker verification -> closure.
    exit_price = h.entry.target
    h.sandbox.fill(exit_record.order_id, exit_price)
    h.lifecycle.refresh(INSTRUMENT)

    expected_pnl = (exit_price - h.E) * h.qty
    assert exit_record.status == OrderStatus.FILLED
    assert exit_record.average_fill_price == pytest.approx(exit_price)
    assert h.managed.closed is True
    assert h.managed.realized_pnl == pytest.approx(expected_pnl)
    assert position.realized_pnl == pytest.approx(expected_pnl)
    assert h.om.expected_positions == {}
    assert h.om.reconcile()["ok"] is True
    assert h.monitor.state(INSTRUMENT) is None

    trades = h.db.get_trades()
    assert len(trades) == 1
    assert trades[0]["status"] == "CLOSED"
    assert trades[0]["quantity"] == h.qty
    assert trades[0]["entry_price"] == pytest.approx(h.E)
    assert trades[0]["exit_price"] == pytest.approx(exit_price)
    assert trades[0]["realized_pnl"] == pytest.approx(expected_pnl)
    assert h.db.get_order(exit_record.order_id)["status"] == "FILLED"

    # Closed position cannot be exited again.
    assert h.lifecycle.on_price(INSTRUMENT, h.entry.target + 2, h.atr, h.ts(20)) is None
    with pytest.raises(RuntimeError, match="NO_OPEN_POSITION_TO_EXIT"):
        h.om.submit_signal(h.manual_exit("EXIT-AFTER-CLOSE"), AccountState(equity=100_000))
    assert len(h.sandbox.placed("SELL")) == 1


def test_stop_loss_exit_via_order_stream(h):
    entry_record = h.lifecycle.submit_entry(h.entry)
    h.sandbox.fill(entry_record.order_id, h.E)
    h.lifecycle.on_order_update(OrderUpdate(
        entry_record.order_id, "complete", INSTRUMENT, "BUY", h.qty, h.qty, 0, h.E))
    assert h.managed.position.quantity == h.qty

    stop_price = h.entry.stop_loss - 0.05
    exit_record = h.lifecycle.on_price(INSTRUMENT, stop_price, h.atr, h.ts(5))
    assert h.managed.exit_reason == "STOP_LOSS"
    assert exit_record.status == OrderStatus.OPEN

    h.sandbox.fill(exit_record.order_id, stop_price)
    h.lifecycle.on_order_update(OrderUpdate(
        exit_record.order_id, "complete", INSTRUMENT, "SELL", h.qty, h.qty, 0, stop_price))

    expected_pnl = (stop_price - h.E) * h.qty
    assert expected_pnl < 0
    assert h.managed.closed is True
    assert h.managed.realized_pnl == pytest.approx(expected_pnl)
    assert h.db.get_trades()[0]["realized_pnl"] == pytest.approx(expected_pnl)
    assert h.om.expected_positions == {}


def test_trailing_stop_exit(h):
    h.open_filled_position()

    trail_atr = 0.25 * h.R
    assert h.lifecycle.on_price(INSTRUMENT, h.E + 1.5 * h.R, trail_atr, h.ts(5)) is None
    state = h.monitor.state(INSTRUMENT)
    assert state.breakeven_activated is True
    assert state.active_stop == pytest.approx(h.E + 1.25 * h.R)
    assert state.active_stop > state.initial_stop

    exit_price = h.E + 1.2 * h.R
    exit_record = h.lifecycle.on_price(INSTRUMENT, exit_price, trail_atr, h.ts(6))
    assert h.managed.exit_reason == "TRAILING_STOP"

    h.sandbox.fill(exit_record.order_id, exit_price)
    h.lifecycle.refresh(INSTRUMENT)

    expected_pnl = 1.2 * h.R * h.qty
    assert h.managed.closed is True
    assert h.managed.realized_pnl == pytest.approx(expected_pnl)
    assert h.db.get_trades()[0]["realized_pnl"] == pytest.approx(expected_pnl)


def test_rejected_exit_keeps_position_and_allows_one_retry(h):
    h.open_filled_position()
    stop_price = h.entry.stop_loss - 0.05

    first = h.lifecycle.on_price(INSTRUMENT, stop_price, h.atr, h.ts(5))
    h.sandbox.reject(first.order_id, "RMS: margin")
    h.lifecycle.refresh(INSTRUMENT)

    assert first.status == OrderStatus.REJECTED
    assert first.rejection_reason == "RMS: margin"
    assert h.managed.closed is False
    assert h.managed.exit_order_id is None
    assert h.om.expected_positions == {INSTRUMENT: h.qty}

    second = h.lifecycle.on_price(INSTRUMENT, stop_price, h.atr, h.ts(6))
    assert second is not None and second.order_id != first.order_id
    assert h.lifecycle.on_price(INSTRUMENT, stop_price, h.atr, h.ts(7)) is None
    assert len(h.sandbox.placed("SELL")) == 2

    h.sandbox.fill(second.order_id, stop_price)
    h.lifecycle.refresh(INSTRUMENT)
    assert h.managed.closed is True
    assert len(h.db.get_trades()) == 1


def test_exit_cannot_exceed_or_flip_position(h):
    h.open_filled_position()
    account = AccountState(equity=100_000)

    with pytest.raises(RuntimeError, match="EXIT_QUANTITY_EXCEEDS_POSITION"):
        h.om.submit_signal(h.manual_exit("EXIT-TOO-BIG", quantity=h.qty + 1), account)
    with pytest.raises(RuntimeError, match="NO_OPEN_POSITION_TO_EXIT"):
        h.om.submit_signal(h.manual_exit("EXIT-WRONG-SIDE", side=SignalSide.BUY), account)

    assert h.sandbox.placed("SELL") == []
    assert len(h.sandbox.placed("BUY")) == 1


def test_second_entry_blocked_while_position_managed(h):
    h.open_filled_position()
    again = h.entry.model_copy(update={"signal_id": "SECOND-ENTRY"})
    with pytest.raises(RuntimeError, match="POSITION_ALREADY_MANAGED"):
        h.lifecycle.submit_entry(again)
    assert len(h.sandbox.placed("BUY")) == 1
