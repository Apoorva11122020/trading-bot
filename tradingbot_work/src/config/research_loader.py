from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .loader import ConfigError, load_yaml


class ResearchSessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timezone: str = "Asia/Kolkata"
    market_open: str = "09:15"
    allow_entries_from: str = "09:15"
    stop_new_entries: str = "14:30"
    square_off: str = "15:15"
    market_close: str = "15:30"


class ResearchRiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_equity: float = Field(gt=0)
    max_daily_loss_inr: float = Field(gt=0)
    risk_per_trade_percent: float = Field(gt=0, le=100)
    max_open_positions: int = Field(gt=0)
    max_trades_per_day: int = Field(gt=0)
    max_portfolio_exposure_percent: float = Field(gt=0, le=100)
    max_premium_exposure_pct: float = Field(gt=0, le=100)
    max_option_positions: int = Field(gt=0)


class ResearchCostsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = "Costs excluded"
    brokerage_per_order: float = Field(ge=0)
    stt_percent: float = Field(ge=0)
    exchange_charges_percent: float = Field(ge=0)
    gst_percent: float = Field(ge=0)
    stamp_duty_percent: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)


class ResearchSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    out_of_sample: float = Field(gt=0, lt=1)

    @field_validator("out_of_sample")
    @classmethod
    def total_one(cls, value: float, info) -> float:
        train = info.data.get("train", 0.7)
        validation = info.data.get("validation", 0.15)
        if abs(train + validation + value - 1.0) > 1e-6:
            raise ValueError("train + validation + out_of_sample must equal 1.0")
        return value


class StrategyResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    underlyings: list[str]
    timeframe: str
    interval_minutes: int = Field(gt=0)
    data_source_label: str
    session: ResearchSessionConfig
    risk: ResearchRiskConfig
    option_mapping: dict[str, Any]
    exit: dict[str, Any]
    costs: ResearchCostsConfig
    split: ResearchSplitConfig
    ranking: list[str]
    strategies: dict[str, dict[str, Any]]
    download: dict[str, Any] = Field(default_factory=dict)


def load_research_config(path: str | Path) -> StrategyResearchConfig:
    raw = load_yaml(path)
    try:
        return StrategyResearchConfig.model_validate(raw["research"])
    except KeyError as exc:
        raise ConfigError(f"Missing configuration section: {exc}") from exc
    except Exception as exc:
        raise ConfigError(str(exc)) from exc
