from __future__ import annotations
from datetime import datetime
from typing import Any

from src.config.loader import StrategyConfig


class StrategyValidationError(ValueError):
    pass


def _time(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%H:%M")
    except ValueError as exc:
        raise StrategyValidationError(f"Invalid HH:MM time: {value}") from exc


def validate_strategy_semantics(config: StrategyConfig) -> list[str]:
    """Validate cross-field strategy relationships without inventing trading rules."""
    errors: list[str] = []
    ind = config.indicators

    if ind.ema_fast >= ind.ema_slow:
        errors.append("indicators.ema_fast must be smaller than indicators.ema_slow")

    if config.timeframe not in {"1m", "3m", "5m", "10m", "15m", "30m", "1h"}:
        errors.append("strategy.timeframe must use a supported canonical timeframe")

    if not config.instruments:
        errors.append("strategy.instruments must not be empty")

    for side, entry in config.entry.items():
        if side not in {"long", "short"}:
            errors.append(f"entry.{side} is unsupported; use long or short")
            continue
        if entry.breakout not in {"previous_20_high", "previous_20_low", "none"}:
            errors.append(f"entry.{side}.breakout has unsupported value: {entry.breakout}")
        if side == "long" and entry.rsi_min is not None and entry.rsi_max is not None and entry.rsi_min > entry.rsi_max:
            errors.append("entry.long.rsi_min cannot exceed rsi_max")
        if side == "short" and entry.rsi_min is not None and entry.rsi_max is not None and entry.rsi_min > entry.rsi_max:
            errors.append("entry.short.rsi_min cannot exceed rsi_max")

    if config.exit.stop_loss.type not in {"atr", "fixed_percent", "fixed_points"}:
        errors.append("exit.stop_loss.type must be atr, fixed_percent, or fixed_points")
    if config.exit.trailing_stop.enabled and config.exit.trailing_stop.atr_multiplier <= 0:
        errors.append("exit.trailing_stop.atr_multiplier must be positive when enabled")
    if config.exit.target.risk_reward <= 0:
        errors.append("exit.target.risk_reward must be positive")

    try:
        market_open = _time(config.session.market_open)
        entry_start = _time(config.session.allow_entries_from)
        entry_stop = _time(config.session.stop_new_entries)
        force_exit = _time(config.session.force_exit)
        market_close = _time(config.session.market_close)
        if not (market_open <= entry_start < entry_stop <= force_exit <= market_close):
            errors.append("session times must satisfy market_open <= allow_entries_from < stop_new_entries <= force_exit <= market_close")
        if _time(config.exit.force_exit_time) != force_exit:
            errors.append("exit.force_exit_time must match session.force_exit")
    except StrategyValidationError as exc:
        errors.append(str(exc))

    if config.risk.max_daily_loss_percent >= config.risk.max_portfolio_exposure_percent:
        # Not a trading recommendation; this is only a suspicious configuration warning.
        errors.append("risk.max_daily_loss_percent must be lower than max_portfolio_exposure_percent")

    if config.system.trading_mode == "LIVE" and not config.system.live_trading_enabled:
        # Valid fail-safe configuration; deliberately not an error.
        pass

    return errors


def assert_strategy_valid(config: StrategyConfig) -> StrategyConfig:
    errors = validate_strategy_semantics(config)
    if errors:
        raise StrategyValidationError("; ".join(errors))
    return config
