"""Paper trading configuration (S2 NIFTY)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.config.loader import ConfigError, load_strategy_config
from src.config.research_loader import load_research_config

ROOT = Path(__file__).resolve().parents[2]
PAPER_CONFIG = ROOT / "config" / "paper_trading.yaml"
STRATEGY_CONFIG = ROOT / "config" / "strategy.yaml"
RESEARCH_CONFIG = ROOT / "config" / "strategy_research.yaml"


class PaperSessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timezone: str = "Asia/Kolkata"
    market_open: str = "09:15"
    allow_entries_from: str = "09:15"
    stop_new_entries: str = "14:30"
    square_off: str = "15:15"
    market_close: str = "15:30"


class PaperTradingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_id: str = "S2"
    underlying: str = "NIFTY"
    timeframe: str = "5m"
    interval_minutes: int = Field(gt=0)
    instrument_key: str
    initial_equity: float = Field(gt=0)
    stale_data_seconds: int = Field(gt=0)
    history_candles: int = Field(ge=20)
    session: PaperSessionConfig
    reports_dir: str = "reports/paper_trading"


def load_paper_trading_settings(path: str | Path = PAPER_CONFIG) -> PaperTradingSettings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        return PaperTradingSettings.model_validate(raw["paper_trading"])
    except Exception as exc:
        raise ConfigError(str(exc)) from exc


def merged_strategy_config(settings: PaperTradingSettings | None = None):
    """Strategy config with S2 paper session window and research-aligned risk."""
    settings = settings or load_paper_trading_settings()
    research = load_research_config(RESEARCH_CONFIG)
    cfg = load_strategy_config(STRATEGY_CONFIG)
    session = cfg.session.model_copy(update={
        "market_open": settings.session.market_open,
        "allow_entries_from": settings.session.allow_entries_from,
        "stop_new_entries": settings.session.stop_new_entries,
        "force_exit": settings.session.square_off,
        "market_close": settings.session.market_close,
    })
    risk = cfg.risk.model_copy(update={
        "max_daily_loss_inr": research.risk.max_daily_loss_inr,
        "risk_per_trade_percent": research.risk.risk_per_trade_percent,
        "max_trades_per_day": research.risk.max_trades_per_day,
        "max_open_positions": research.risk.max_open_positions,
        "max_portfolio_exposure_percent": research.risk.max_portfolio_exposure_percent,
    })
    return cfg.model_copy(update={"session": session, "risk": risk}), research, settings
