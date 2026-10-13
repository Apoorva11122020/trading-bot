from src.config.loader import load_strategy_config
from src.models.risk import AccountState
from src.risk.risk_manager import RiskManager


def test_daily_loss_blocks():
    cfg = load_strategy_config("config/strategy.yaml")
    manager = RiskManager(cfg.risk)
    account = AccountState(equity=100000, daily_pnl=-3001)
    result = manager.validate_entry(account, 1000, 990, 1000, True)
    assert not result.allowed
    assert "DAILY_LOSS_LIMIT_REACHED" in result.reasons
