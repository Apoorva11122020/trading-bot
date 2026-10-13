from pathlib import Path
import copy
import yaml
import pytest

from src.config.loader import load_strategy_config
from src.config.strategy_validation import StrategyValidationError, assert_strategy_valid

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "strategy.yaml"


def test_current_strategy_semantics_are_valid():
    assert_strategy_valid(load_strategy_config(CFG))


def _raw():
    return yaml.safe_load(CFG.read_text())


def _load(raw):
    p = CFG.parent / "_phase28_tmp.yaml"
    p.write_text(yaml.safe_dump(raw))
    try:
        return load_strategy_config(p)
    finally:
        p.unlink(missing_ok=True)


def test_fast_ema_must_be_less_than_slow():
    raw = _raw()
    raw["strategy"]["indicators"]["ema_fast"] = 50
    raw["strategy"]["indicators"]["ema_slow"] = 20
    with pytest.raises(StrategyValidationError):
        assert_strategy_valid(_load(raw))


def test_session_order_is_validated():
    raw = _raw()
    raw["session"]["stop_new_entries"] = "09:10"
    with pytest.raises(StrategyValidationError):
        assert_strategy_valid(_load(raw))


def test_force_exit_must_match_session():
    raw = _raw()
    raw["strategy"]["exit"]["force_exit_time"] = "15:10"
    with pytest.raises(StrategyValidationError):
        assert_strategy_valid(_load(raw))


def test_unknown_breakout_is_rejected():
    raw = _raw()
    raw["strategy"]["entry"]["long"]["breakout"] = "magic_breakout"
    with pytest.raises(StrategyValidationError):
        assert_strategy_valid(_load(raw))


def test_unknown_stop_type_is_rejected():
    raw = _raw()
    raw["strategy"]["exit"]["stop_loss"]["type"] = "unknown"
    with pytest.raises(StrategyValidationError):
        assert_strategy_valid(_load(raw))
