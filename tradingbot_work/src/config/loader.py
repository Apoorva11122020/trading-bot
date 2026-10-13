from datetime import date as _date
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigError(ValueError):
    """Raised when project configuration is invalid."""


class IndicatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ema_fast: int = Field(gt=0)
    ema_slow: int = Field(gt=0)
    rsi_period: int = Field(gt=0)
    atr_period: int = Field(gt=0)
    volume_sma_period: int = Field(gt=0)
    breakout_period: int = Field(gt=0)
    vwap_enabled: bool = True


class EntrySideConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    ema_fast_above_slow: bool | None = None
    ema_fast_below_slow: bool | None = None
    price_above_vwap: bool | None = None
    price_below_vwap: bool | None = None
    rsi_min: float | None = Field(default=None, ge=0, le=100)
    rsi_max: float | None = Field(default=None, ge=0, le=100)
    breakout: str
    volume_multiplier: float = Field(gt=0)


class StopLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    multiplier: float = Field(gt=0)


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_reward: float = Field(gt=0)


class BreakevenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    activation_r: float = Field(gt=0)


class TrailingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    atr_multiplier: float = Field(gt=0)


class ExitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stop_loss: StopLossConfig
    target: TargetConfig
    breakeven: BreakevenConfig
    trailing_stop: TrailingConfig
    force_exit_time: str


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_per_trade_percent: float = Field(gt=0, le=100)
    max_open_positions: int = Field(gt=0)
    max_trades_per_day: int = Field(gt=0)
    max_daily_loss_inr: float | None = Field(default=None, gt=0)
    max_daily_loss_percent: float = Field(gt=0, le=100)
    max_portfolio_exposure_percent: float = Field(gt=0, le=100)
    max_loss_gap_buffer_percent: float = Field(default=10.0, ge=0)


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timezone: str
    market_open: str
    allow_entries_from: str
    stop_new_entries: str
    force_exit: str
    market_close: str


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_slippage_percent: float = Field(ge=0)
    stale_data_seconds: int = Field(gt=0)
    duplicate_order_protection: bool = True
    # Operational (non-strategy) order status polling settings.
    order_poll_interval_seconds: float = Field(default=2.0, gt=0)
    order_poll_max_backoff_seconds: float = Field(default=30.0, gt=0)
    order_status_timeout_seconds: float = Field(default=120.0, gt=0)


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trading_mode: str
    live_trading_enabled: bool = False

    @field_validator("trading_mode")
    @classmethod
    def valid_mode(cls, value: str) -> str:
        value = value.upper()
        if value not in {"PAPER", "LIVE"}:
            raise ValueError("trading_mode must be PAPER or LIVE")
        return value


class OptionExpiryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["nearest", "next", "date"] = "nearest"
    date: str | None = None

    @model_validator(mode="after")
    def specific_date(self):
        if self.mode == "date":
            if not self.date:
                raise ValueError("options.expiry.date is required when mode is 'date'")
            _date.fromisoformat(self.date)
        return self


class OptionStrikeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["ATM", "ITM", "OTM", "EXACT"] = "ATM"
    offset: int = Field(default=0, ge=0)
    exact: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def exact_strike(self):
        if self.mode == "EXACT" and self.exact is None:
            raise ValueError("options.strike.exact is required when mode is 'EXACT'")
        return self


class OptionTypeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["signal_based", "CE", "PE"] = "signal_based"


class OptionRiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_premium_exposure_pct: float = Field(default=10.0, gt=0, le=100)
    max_option_positions: int = Field(default=2, gt=0)
    stop_mode: Literal["premium_atr", "underlying_atr"] = "premium_atr"
    stop_atr_multiplier: float = Field(default=1.5, gt=0)
    # Only used by stop_mode=underlying_atr: premium move per 1.0 underlying move.
    underlying_delta: float = Field(default=0.5, gt=0, le=1)
    target_r: float = Field(default=2.0, gt=0)
    breakeven_r: float = Field(default=1.0, gt=0)
    trailing_atr_multiplier: float = Field(default=1.0, gt=0)


class OptionExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product: Literal["D", "I"] = "D"
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"


class OptionInstrumentDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["upstox_instrument_master"] = "upstox_instrument_master"
    url: str = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    cache_dir: str = "data/cache/instruments"
    cache_hours: float = Field(default=12.0, gt=0)


class OptionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    signal_source: Literal["underlying", "option"] = "underlying"
    market_type: Literal["index_option", "stock_option"] = "index_option"
    underlying: str = "NIFTY"
    expiry: OptionExpiryConfig = Field(default_factory=OptionExpiryConfig)
    strike: OptionStrikeConfig = Field(default_factory=OptionStrikeConfig)
    option_type: OptionTypeConfig = Field(default_factory=OptionTypeConfig)
    risk: OptionRiskConfig = Field(default_factory=OptionRiskConfig)
    execution: OptionExecutionConfig = Field(default_factory=OptionExecutionConfig)
    lot_size_required: Literal[True] = True
    # Buy-side only; option writing is not implemented.
    allow_option_selling: Literal[False] = False
    index_underlyings: list[str] = Field(default_factory=lambda: ["NIFTY", "BANKNIFTY", "FINNIFTY"])
    stock_underlyings: list[str] = Field(default_factory=lambda: ["RELIANCE", "HDFCBANK", "ICICIBANK"])
    instrument_data: OptionInstrumentDataConfig = Field(default_factory=OptionInstrumentDataConfig)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    exchange: str
    market: str
    timeframe: str
    instruments: dict[str, Any]
    indicators: IndicatorConfig
    entry: dict[str, EntrySideConfig]
    exit: ExitConfig
    risk: RiskConfig
    session: SessionConfig
    execution: ExecutionConfig
    system: SystemConfig
    options: OptionsConfig = Field(default_factory=OptionsConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a mapping.")
    return data


def load_strategy_config(path: str | Path) -> StrategyConfig:
    raw = load_yaml(path)
    try:
        config = StrategyConfig.model_validate(raw["strategy"] | {
            "entry": raw["strategy"]["entry"],
            "exit": raw["strategy"]["exit"],
            "risk": raw["risk"],
            "session": raw["session"],
            "execution": raw["execution"],
            "system": raw["system"],
        } | ({"options": raw["options"]} if raw.get("options") is not None else {}))
    except KeyError as exc:
        raise ConfigError(f"Missing configuration section: {exc}") from exc
    except Exception as exc:
        raise ConfigError(str(exc)) from exc

    if config.system.trading_mode == "LIVE" and not config.system.live_trading_enabled:
        # Safe state: LIVE mode without explicit enable remains non-executable.
        pass
    return config
