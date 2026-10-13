from datetime import datetime, timedelta
import pandas as pd
from src.backtest.engine import BacktestEngine
from src.security import validate_security

class FakeStrategy:
    def evaluate_entry(self, history, account, ts):
        return None

def test_empty_backtest_metrics():
    result = BacktestEngine(FakeStrategy()).run(pd.DataFrame(columns=["timestamp","symbol","open","high","low","close","volume"]))
    assert result.total_trades == 0
    assert result.net_pnl == 0

def test_security_requires_env_ignored():
    result = validate_security(".")
    assert result["env_ignored"] is True
