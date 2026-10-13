from src.config.loader import load_strategy_config


def test_valid_config():
    config = load_strategy_config("config/strategy.yaml")
    assert config.system.trading_mode == "PAPER"
    assert config.indicators.ema_fast == 20
