from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.config.loader import load_strategy_config
from src.models.market import Candle
from src.models.risk import AccountState
from src.market_data.models import MarketDataHealth, MarketDataStatus
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager
from src.strategy.signal_generator import SignalGenerator
from src.strategy.strategy_engine import StrategyEngine
from src.strategy.runtime import StrategyRuntime


def build_candles(count=55):
    start = datetime(2026, 10, 1, 9, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    rows = []
    price = 100.0
    for i in range(count - 1):
        price += 0.15
        rows.append({
            "timestamp": start + timedelta(minutes=5 * i),
            "symbol": "RELIANCE",
            "open": price - 0.10,
            "high": price + 0.20,
            "low": price - 0.20,
            "close": price,
            "volume": 100.0,
        })

    # Final candle breaks the previous 20-candle high with a volume surge.
    price += 5.0
    rows.append({
        "timestamp": start + timedelta(minutes=5 * (count - 1)),
        "symbol": "RELIANCE",
        "open": price - 0.20,
        "high": price + 0.20,
        "low": price - 0.30,
        "close": price,
        "volume": 200.0,
    })
    return pd.DataFrame(rows)


def make_runtime():
    cfg = load_strategy_config("config/strategy.yaml")
    engine = StrategyEngine(
        cfg,
        RiskManager(cfg.risk),
        SessionManager(cfg.session),
    )
    return cfg, StrategyRuntime(SignalGenerator(engine))


def test_signal_generator_produces_long_entry():
    cfg, runtime = make_runtime()
    account = AccountState(equity=100000)
    health = MarketDataHealth(
        status=MarketDataStatus.CONNECTED,
        last_tick_at=datetime.now().astimezone(),
        subscribed_symbols=["RELIANCE"],
    )

    result = None
    for row in build_candles().to_dict("records"):
        result = runtime.on_candle(Candle(**row), account, health) or result

    assert result is not None
    assert result.signal_type.value == "ENTRY"
    assert result.side.value == "BUY"
    assert result.quantity > 0
    assert result.stop_loss < result.entry_price
    assert result.target > result.entry_price
    assert result.risk_status == "ALLOWED"


def test_duplicate_completed_candle_does_not_generate_again():
    cfg, runtime = make_runtime()
    account = AccountState(equity=100000)
    health = MarketDataHealth(
        status=MarketDataStatus.CONNECTED,
        last_tick_at=datetime.now().astimezone(),
        subscribed_symbols=["RELIANCE"],
    )

    signal = None
    candles = [Candle(**r) for r in build_candles().to_dict("records")]
    for candle in candles:
        signal = runtime.on_candle(candle, account, health) or signal

    duplicate = runtime.on_candle(candles[-1], account, health)
    assert signal is not None
    assert duplicate is None


def test_unhealthy_market_data_blocks_signal_generation():
    _, runtime = make_runtime()
    account = AccountState(equity=100000)
    unhealthy = MarketDataHealth(
        status=MarketDataStatus.STALE,
        last_tick_at=datetime.now().astimezone(),
        subscribed_symbols=["RELIANCE"],
    )

    candle = Candle(**build_candles().iloc[-1].to_dict())
    assert runtime.on_candle(candle, account, unhealthy) is None
