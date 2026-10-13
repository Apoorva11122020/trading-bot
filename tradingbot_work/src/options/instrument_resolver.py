"""Resolve NSE_FO option contracts from official Upstox instrument data.

Source: the public Upstox instrument master (BOD file). It is read-only,
needs no credentials and contains every listed contract with its expiry,
strike, lot size and freeze quantity. Contracts are never invented: a
request that does not match a listed contract raises OptionResolutionError.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from .models import NSE_FO, MarketType, OptionContract, OptionResolutionError, OptionType

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
ALLOWED_INSTRUMENT_HOST = "https://assets.upstox.com/"
SOURCE_NAME = "UPSTOX_INSTRUMENT_MASTER"
# The instrument master reports tick_size in paise (5.0 == Rs 0.05).
TICK_SIZE_DIVISOR = 100.0


def _expiry_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).astimezone(IST).date()
    text = str(value).strip()
    if text.isdigit():
        return _expiry_date(int(text))
    return date.fromisoformat(text[:10])


def parse_instrument_rows(rows: Iterable[dict]) -> tuple[list[OptionContract], int]:
    """NSE_FO CE/PE rows -> OptionContract. Returns (contracts, skipped)."""
    contracts: list[OptionContract] = []
    skipped = 0
    for row in rows:
        if row.get("segment") != NSE_FO or str(row.get("instrument_type", "")).upper() not in ("CE", "PE"):
            continue
        try:
            freeze = row.get("freeze_quantity")
            contracts.append(
                OptionContract(
                    instrument_key=str(row["instrument_key"]),
                    trading_symbol=str(row.get("trading_symbol") or ""),
                    underlying=str(row.get("underlying_symbol") or row.get("name") or "").upper(),
                    expiry=_expiry_date(row["expiry"]),
                    strike=float(row["strike_price"]),
                    option_type=OptionType(str(row["instrument_type"]).upper()),
                    lot_size=int(row["lot_size"]),
                    segment=str(row["segment"]),
                    exchange=str(row.get("exchange") or "NSE"),
                    freeze_quantity=int(float(freeze)) if freeze not in (None, "") else None,
                    underlying_key=row.get("underlying_key"),
                    underlying_type=row.get("underlying_type"),
                    tick_size=(float(row["tick_size"]) / TICK_SIZE_DIVISOR
                               if row.get("tick_size") not in (None, "") else None),
                    weekly=row.get("weekly"),
                )
            )
        except (KeyError, TypeError, ValueError):
            skipped += 1
    return contracts, skipped


def _http_fetch(url: str) -> bytes:
    import requests

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


class UpstoxInstrumentMasterSource:
    """Downloads and caches the public Upstox NSE instrument master."""

    def __init__(
        self,
        url: str = DEFAULT_INSTRUMENT_URL,
        cache_dir: str | Path = "data/cache/instruments",
        cache_hours: float = 12.0,
        fetcher: Callable[[str], bytes] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not str(url).startswith(ALLOWED_INSTRUMENT_HOST):
            raise OptionResolutionError(
                "INSTRUMENT_SOURCE_REFUSED", f"Instrument data must come from {ALLOWED_INSTRUMENT_HOST}")
        self.url = url
        self.cache_dir = Path(cache_dir)
        self.cache_seconds = float(cache_hours) * 3600.0
        self.fetcher = fetcher or _http_fetch
        self.clock = clock or time.time
        self._lock = threading.Lock()
        self._contracts: list[OptionContract] | None = None
        self._loaded_at: float | None = None
        self.metadata: dict[str, Any] = {"source": SOURCE_NAME, "url": url}

    @property
    def cache_file(self) -> Path:
        return self.cache_dir / Path(self.url).name

    def _fresh(self, loaded_at: float | None) -> bool:
        return loaded_at is not None and (self.clock() - loaded_at) < self.cache_seconds

    def _read(self, payload: bytes) -> list[dict]:
        data = json.loads(gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload)
        if not isinstance(data, list):
            raise ValueError("Unexpected instrument master format")
        return data

    def option_contracts(self) -> list[OptionContract]:
        with self._lock:
            if self._contracts is not None and self._fresh(self._loaded_at):
                return self._contracts

            payload = None
            stale = False
            from_cache = False
            cache = self.cache_file
            if cache.exists() and self._fresh(cache.stat().st_mtime):
                payload = cache.read_bytes()
                from_cache = True
            else:
                try:
                    payload = self.fetcher(self.url)
                    self._read(payload)
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache.with_suffix(cache.suffix + ".tmp")
                    tmp.write_bytes(payload)
                    tmp.replace(cache)
                except Exception as exc:
                    if cache.exists():
                        payload = cache.read_bytes()
                        stale = from_cache = True
                    elif self._contracts is not None:
                        self.metadata["stale"] = True
                        return self._contracts
                    else:
                        raise OptionResolutionError(
                            "INSTRUMENT_DATA_UNAVAILABLE",
                            f"Upstox instrument data could not be loaded ({type(exc).__name__})",
                        ) from exc

            contracts, skipped = parse_instrument_rows(self._read(payload))
            self._contracts = contracts
            self._loaded_at = self.clock()
            self.metadata = {
                "source": SOURCE_NAME,
                "url": self.url,
                "loaded_at": datetime.fromtimestamp(self._loaded_at, tz=timezone.utc).isoformat(),
                "from_cache": from_cache,
                "stale": stale,
                "contracts": len(contracts),
                "skipped_rows": skipped,
            }
            return contracts


class StaticContractSource:
    """In-memory contract list (tests and historical backtests)."""

    def __init__(self, contracts: Iterable[OptionContract], metadata: dict | None = None) -> None:
        self._contracts = list(contracts)
        self.metadata = metadata or {"source": "STATIC", "contracts": len(self._contracts)}

    def option_contracts(self) -> list[OptionContract]:
        return self._contracts


class OptionInstrumentResolver:
    def __init__(self, source, today: Callable[[], date] | None = None) -> None:
        self.source = source
        self.today = today or (lambda: datetime.now(IST).date())
        self._lock = threading.Lock()
        self._index_for: int | None = None
        self._index: dict[str, list[OptionContract]] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(getattr(self.source, "metadata", {}) or {})

    def _by_underlying(self) -> dict[str, list[OptionContract]]:
        contracts = self.source.option_contracts()
        with self._lock:
            if self._index_for != id(contracts):
                index: dict[str, list[OptionContract]] = {}
                for contract in contracts:
                    index.setdefault(contract.underlying, []).append(contract)
                self._index = index
                self._index_for = id(contracts)
            return self._index

    @staticmethod
    def _norm(underlying: str) -> str:
        return str(underlying or "").strip().upper()

    def _live(self, contracts: Iterable[OptionContract]) -> list[OptionContract]:
        today = self.today()
        return [c for c in contracts if c.expiry >= today]

    def underlyings(self, market_type: MarketType | str | None = None) -> list[dict[str, Any]]:
        wanted = MarketType(market_type) if market_type else None
        result = []
        for symbol, contracts in sorted(self._by_underlying().items()):
            live = self._live(contracts)
            if not live:
                continue
            kind = live[0].market_type
            if wanted and kind != wanted:
                continue
            result.append({
                "symbol": symbol,
                "market_type": kind.value,
                "underlying_key": live[0].underlying_key,
                "lot_size": live[0].lot_size,
                "expiries": len({c.expiry for c in live}),
            })
        return result

    def contracts(
        self,
        underlying: str,
        expiry: date | None = None,
        option_type: OptionType | str | None = None,
    ) -> list[OptionContract]:
        name = self._norm(underlying)
        found = self._live(self._by_underlying().get(name, []))
        if not found:
            raise OptionResolutionError("UNDERLYING_NOT_FOUND", f"No listed NSE_FO options for {name}")
        if expiry is not None:
            found = [c for c in found if c.expiry == expiry]
        if option_type is not None:
            kind = self.option_type(option_type)
            found = [c for c in found if c.option_type == kind]
        return sorted(found, key=lambda c: (c.expiry, c.strike, c.option_type.value))

    @staticmethod
    def option_type(value: OptionType | str) -> OptionType:
        try:
            return value if isinstance(value, OptionType) else OptionType(str(value).strip().upper())
        except ValueError as exc:
            raise OptionResolutionError("OPTION_TYPE_NOT_FOUND", f"Option type must be CE or PE, got {value!r}") from exc

    def market_type(self, underlying: str) -> MarketType:
        return self.contracts(underlying)[0].market_type

    def expiries(self, underlying: str) -> list[date]:
        return sorted({c.expiry for c in self.contracts(underlying)})

    def strikes(self, underlying: str, expiry: date, option_type: OptionType | str | None = None) -> list[float]:
        return sorted({c.strike for c in self.contracts(underlying, expiry, option_type)})

    def resolve(
        self,
        underlying: str,
        expiry: date | str,
        strike: float,
        option_type: OptionType | str,
    ) -> OptionContract:
        name = self._norm(underlying)
        kind = self.option_type(option_type)
        expiry_date = _expiry_date(expiry)
        if expiry_date not in self.expiries(name):
            raise OptionResolutionError(
                "EXPIRY_NOT_FOUND", f"{name} has no listed expiry {expiry_date.isoformat()}")
        candidates = self.contracts(name, expiry_date)
        same_strike = [c for c in candidates if abs(c.strike - float(strike)) < 1e-6]
        if not same_strike:
            raise OptionResolutionError(
                "STRIKE_NOT_FOUND", f"{name} {expiry_date.isoformat()} has no listed strike {float(strike):g}")
        for contract in same_strike:
            if contract.option_type == kind:
                return contract
        raise OptionResolutionError(
            "CONTRACT_NOT_FOUND",
            f"{name} {expiry_date.isoformat()} {float(strike):g} {kind.value} is not listed")


def build_resolver(options_config) -> OptionInstrumentResolver:
    data = options_config.instrument_data
    return OptionInstrumentResolver(
        UpstoxInstrumentMasterSource(url=data.url, cache_dir=data.cache_dir, cache_hours=data.cache_hours))
