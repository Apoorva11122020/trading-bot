from scripts.nifty_pe_recovery_rule import (
    PeRecoverySessionState,
    evaluate_pe_recovery,
    recovery_level,
    update_session_state,
)


def test_pe_recovery_fires_after_deep_drop_and_cross_up():
    state = PeRecoverySessionState()
    morning = 23000.0
    update_session_state(state, open_price=morning, low=morning - 210.0)
    assert state.saw_deep_drop
    level = recovery_level(morning)
    signal = evaluate_pe_recovery(state, prev_close=level - 5, close=level + 1)
    assert signal is not None
    assert signal.lots == 2
    assert signal.stop_premium_points == 20.0
    assert evaluate_pe_recovery(state, prev_close=level, close=level + 1) is None


def test_pe_recovery_skips_without_200_point_drop():
    state = PeRecoverySessionState()
    morning = 23000.0
    update_session_state(state, open_price=morning, low=morning - 100.0)
    assert not state.saw_deep_drop
    level = recovery_level(morning)
    assert evaluate_pe_recovery(state, prev_close=level - 1, close=level + 1) is None
