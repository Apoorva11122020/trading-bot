from pydantic import BaseModel


class AccountState(BaseModel):
    equity: float
    daily_pnl: float = 0.0
    open_positions: int = 0
    trades_today: int = 0
    current_exposure: float = 0.0


class RiskValidationResult(BaseModel):
    allowed: bool
    reasons: list[str] = []
    risk_amount: float = 0.0
    estimated_exposure: float = 0.0
    quantity: int = 0


class OptionSizing(BaseModel):
    """BUY-side option position size, always a whole number of lots."""

    risk_budget: float
    risk_per_unit: float
    raw_quantity: int
    lot_size: int
    lots: int
    quantity: int
    freeze_quantity: int | None = None
    max_quantity_by_risk: int = 0
    max_quantity_by_premium: int | None = None
    max_quantity_by_exposure: int | None = None
    max_quantity_by_freeze: int | None = None
    limiting_factor: str = "RISK"
    max_loss: float = 0.0
    premium_exposure: float = 0.0


class OptionRiskResult(BaseModel):
    allowed: bool
    reasons: list[str] = []
    quantity: int = 0
    lots: int = 0
    lot_size: int = 0
    risk_per_unit: float = 0.0
    max_loss: float = 0.0
    premium_exposure: float = 0.0
    position_notional: float = 0.0
    daily_loss_limit: float = 0.0
    daily_loss_after_stop: float = 0.0
    open_option_positions: int = 0
    max_option_positions: int | None = None
