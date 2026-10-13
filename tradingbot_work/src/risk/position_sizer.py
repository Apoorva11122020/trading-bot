import math


def calculate_position_size(equity: float, risk_percent: float, entry_price: float, stop_loss: float) -> tuple[int, float]:
    if equity <= 0:
        raise ValueError("equity must be > 0")
    if entry_price <= 0 or stop_loss <= 0:
        raise ValueError("entry and stop-loss must be > 0")

    risk_per_share = abs(entry_price - stop_loss)
    if risk_per_share <= 0:
        raise ValueError("stop-loss distance must be > 0")

    risk_amount = equity * risk_percent / 100
    quantity = math.floor(risk_amount / risk_per_share)
    return quantity, risk_amount
