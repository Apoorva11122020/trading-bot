import pytest

from src.broker.models import OrderStatus

from trading_support import SYMBOL, Stack, entry, ts


@pytest.fixture
def s(tmp_path):
    stack = Stack(tmp_path)
    yield stack
    stack.db.close()


def test_restart_restores_position_and_trailing_state_without_new_orders(s):
    s.lifecycle.submit_entry(entry("E1"))
    assert s.price(107.0, ts(11, 0)) is None
    before = s.monitor.export_state(SYMBOL)
    assert before["breakeven_activated"] is True
    assert before["active_stop"] == pytest.approx(106.0)
    orders_before = len(s.broker.get_orders())

    r = s.restart()
    assert r.report.ok
    assert r.report.recovered_lifecycles == [SYMBOL]
    assert r.om.expected_positions == {SYMBOL: 100}
    assert r.om.position_avg[SYMBOL] == pytest.approx(100.0)
    managed = r.managed()
    assert managed.closed is False
    assert managed.position.quantity == 100
    assert managed.entry_order_id in r.om.records
    after = r.monitor.export_state(SYMBOL)
    assert after == before
    assert len(r.broker.get_orders()) == orders_before

    # Idempotent: a second reconciliation changes nothing.
    r.report = None
    again = r.restart()
    assert again.monitor.export_state(SYMBOL) == before
    assert len(again.broker.get_orders()) == orders_before
    assert len(again.lifecycle.open_symbols()) == 1

    # Restored trailing stop (106) is honoured, not reset to the initial stop.
    record = again.price(105.5, ts(11, 5))
    assert record is not None
    assert again.managed().exit_reason == "TRAILING_STOP"
    assert again.managed().realized_pnl == pytest.approx(550.0)
    assert len(again.db.get_trades()) == 1


def test_entry_filled_while_down_is_applied_once(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    rec = s.lifecycle.submit_entry(entry("E1"))
    s.broker.fill_order(rec.order_id, 100, 100.0)

    r = s.restart()
    assert r.om.records[rec.order_id].status == OrderStatus.FILLED
    assert r.managed().position.quantity == 100
    assert r.monitor.state(SYMBOL) is not None
    assert r.report.ok

    r2 = r.restart()
    assert len(r2.db.get_executions(rec.order_id)) == 1
    assert r2.db.get_trade(rec.order_id)["quantity"] == 100
    assert r2.om.expected_positions == {SYMBOL: 100}


def test_partial_fill_while_down_then_completion(s):
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    rec = s.lifecycle.submit_entry(entry("E1"))
    s.broker.fill_order(rec.order_id, 40, 100.0)
    r = s.restart()
    assert r.managed().position.quantity == 40
    s.broker.fill_order(rec.order_id, 60, 100.0)
    r.om.refresh_open_orders()
    assert r.managed().position.quantity == 100
    assert r.db.get_trade(rec.order_id)["quantity"] == 100


def test_exit_filled_while_down_closes_position_after_restart(s):
    s.lifecycle.submit_entry(entry("E1"))
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    exit_record = s.price(94.0, ts(11, 0))
    s.broker.fill_order(exit_record.order_id, 100, 94.0)

    r = s.restart()
    managed = r.managed()
    assert managed.closed is True
    assert managed.close_reason == "STOP_LOSS"
    assert managed.realized_pnl == pytest.approx(-600.0)
    trade = r.db.get_trades()[0]
    assert trade["status"] == "CLOSED" and trade["exit_reason"] == "STOP_LOSS"
    assert r.om.expected_positions == {}
    assert r.lifecycle.open_symbols() == []


def test_pending_exit_survives_restart_without_duplicate_exit(s):
    s.lifecycle.submit_entry(entry("E1"))
    s.broker.set_next_execution(OrderStatus.OPEN, filled_quantity=0)
    exit_record = s.price(94.0, ts(11, 0))

    r = s.restart()
    assert r.managed().exit_order_id == exit_record.order_id
    assert r.monitor.state(SYMBOL).exit_emitted is True
    assert r.price(93.0, ts(11, 1)) is None
    sells = [o for o in r.broker.get_orders() if o["side"] == "SELL"]
    assert len(sells) == 1


def test_broker_position_mismatch_adopts_broker_truth(s):
    s.lifecycle.submit_entry(entry("E1"))
    s.broker.positions[SYMBOL]["quantity"] = 60
    r = s.restart()
    assert not r.report.ok
    assert r.report.position_mismatches[SYMBOL]["broker"] == 60
    assert r.om.expected_positions == {SYMBOL: 60}
    assert r.managed().position.quantity == 60
    assert r.events("RECOVERY_MISMATCH")
    assert r.events("RECOVERY_COMPLETED")


def test_recovery_restores_persisted_kill_switch(tmp_path):
    from src.risk.controller import PersistentKillSwitch
    s = Stack(tmp_path)
    PersistentKillSwitch(s.db).activate("TEST")
    assert PersistentKillSwitch(s.db).is_halted is True
    s.db.close()
