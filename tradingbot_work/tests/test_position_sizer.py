from src.risk.position_sizer import calculate_position_size


def test_position_size():
    qty, risk = calculate_position_size(100000, 0.5, 1000, 990)
    assert risk == 500
    assert qty == 50
