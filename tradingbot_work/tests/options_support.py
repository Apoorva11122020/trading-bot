"""Deterministic NSE F&O fixtures shaped like the Upstox instrument master.

Row fields mirror https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz
(expiry as epoch ms at IST end of day, tick_size in paise). Values are test
fixtures only; production resolution always loads the official file.
"""
from __future__ import annotations

import gzip
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config.loader import load_strategy_config
from src.options.instrument_resolver import OptionInstrumentResolver, StaticContractSource, parse_instrument_rows
from src.options.strategy import OptionSignalEngine
from src.risk.risk_manager import RiskManager
from src.session.session_manager import SessionManager
from src.strategy.strategy_engine import StrategyEngine

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "strategy.yaml"
TODAY = date(2026, 10, 3)

NIFTY_EXPIRIES = (date(2026, 10, 6), date(2026, 10, 13), date(2026, 10, 27))
BANKNIFTY_EXPIRIES = (date(2026, 10, 27), date(2026, 11, 24))
RELIANCE_EXPIRIES = (date(2026, 10, 27), date(2026, 11, 24))
EXPIRED = date(2026, 9, 29)

NIFTY_STRIKES = [22500.0 + 50 * i for i in range(21)]          # 22500..23500
BANKNIFTY_STRIKES = [56000.0 + 100 * i for i in range(11)]     # 56000..57000
RELIANCE_STRIKES = [1200.0 + 10 * i for i in range(7)] + [1280.0, 1300.0]  # uneven step


def epoch_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=IST).timestamp() * 1000)


def _token(name, expiry, strike, kind) -> int:
    base = {"NIFTY": 1, "BANKNIFTY": 2, "RELIANCE": 3}[name]
    return int(f"{base}{expiry.strftime('%y%m%d')}{int(strike):06d}{0 if kind == 'CE' else 1 if kind == 'PE' else 2}")


def option_row(name, expiry, strike, kind, lot, freeze, underlying_type, underlying_key, tick=5.0):
    token = _token(name, expiry, strike, kind)
    return {
        "segment": "NSE_FO",
        "name": name,
        "exchange": "NSE",
        "expiry": epoch_ms(expiry),
        "weekly": name == "NIFTY",
        "instrument_key": f"NSE_FO|{token}",
        "exchange_token": str(token),
        "trading_symbol": f"{name} {strike:g} {kind} {expiry.strftime('%d %b %y').upper()}",
        "tick_size": tick,
        "lot_size": lot,
        "instrument_type": kind,
        "freeze_quantity": float(freeze),
        "underlying_key": underlying_key,
        "underlying_type": underlying_type,
        "underlying_symbol": name,
        "strike_price": strike,
        "minimum_lot": lot,
        "qty_multiplier": 1.0,
        "asset_type": underlying_type,
    }


def instrument_rows() -> list[dict]:
    rows: list[dict] = []
    for expiry in NIFTY_EXPIRIES:
        for strike in NIFTY_STRIKES:
            for kind in ("CE", "PE"):
                rows.append(option_row("NIFTY", expiry, strike, kind, 65, 1755, "INDEX", "NSE_INDEX|Nifty 50"))
    for expiry in BANKNIFTY_EXPIRIES:
        for strike in BANKNIFTY_STRIKES:
            for kind in ("CE", "PE"):
                rows.append(option_row("BANKNIFTY", expiry, strike, kind, 30, 900, "INDEX", "NSE_INDEX|Nifty Bank"))
    for expiry in RELIANCE_EXPIRIES:
        for strike in RELIANCE_STRIKES:
            for kind in ("CE", "PE"):
                rows.append(option_row("RELIANCE", expiry, strike, kind, 500, 20000, "EQUITY",
                                       "NSE_EQ|INE002A01018"))
    # Strike 1290 CE only (PE not listed), an expired contract, futures,
    # equity rows and a malformed row: all must be handled correctly.
    rows.append(option_row("RELIANCE", RELIANCE_EXPIRIES[0], 1290.0, "CE", 500, 20000, "EQUITY",
                           "NSE_EQ|INE002A01018"))
    rows.append(option_row("NIFTY", EXPIRED, 23000.0, "CE", 65, 1755, "INDEX", "NSE_INDEX|Nifty 50"))
    rows.append({**option_row("NIFTY", NIFTY_EXPIRIES[0], 0.0, "CE", 65, 1755, "INDEX", "NSE_INDEX|Nifty 50"),
                 "instrument_type": "FUT"})
    rows.append({"segment": "NSE_EQ", "instrument_key": "NSE_EQ|INE002A01018", "trading_symbol": "RELIANCE",
                 "instrument_type": "EQ", "lot_size": 1})
    rows.append({**option_row("NIFTY", NIFTY_EXPIRIES[0], 23600.0, "CE", 65, 1755, "INDEX",
                              "NSE_INDEX|Nifty 50"), "lot_size": 0})
    return rows


def gz_payload(rows: list[dict] | None = None) -> bytes:
    return gzip.compress(json.dumps(rows if rows is not None else instrument_rows()).encode())


def contracts():
    return parse_instrument_rows(instrument_rows())[0]


def make_resolver(today: date = TODAY) -> OptionInstrumentResolver:
    return OptionInstrumentResolver(StaticContractSource(contracts()), today=lambda: today)


def strategy_stack(config_path=CONFIG_PATH):
    config = load_strategy_config(config_path)
    risk_manager = RiskManager(config.risk)
    session_manager = SessionManager(config.session)
    strategy_engine = StrategyEngine(config, risk_manager, session_manager)
    engine = OptionSignalEngine(config, strategy_engine, risk_manager, session_manager)
    return config, risk_manager, session_manager, strategy_engine, engine
