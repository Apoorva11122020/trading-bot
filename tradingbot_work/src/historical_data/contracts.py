"""Historical contract discovery (no hardcoded strikes, expiries or keys).

Two real sources:
  ACTIVE  - contracts listed today, from the official Upstox instrument
            master (same source as the live OptionInstrumentResolver).
  EXPIRED - contracts that already expired, from the Upstox
            expired-instruments API (requires the Upstox Plus plan).

When the expired source is unavailable the reason is recorded in
``availability`` and surfaced to callers; it is never silently ignored, and
currently listed contracts are never presented as past contracts.
"""
from __future__ import annotations

import gzip
import json
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.options.instrument_resolver import TICK_SIZE_DIVISOR, _expiry_date
from src.options.models import OptionContract, OptionType

from .models import HistoricalDataError

IST = ZoneInfo("Asia/Kolkata")
ACTIVE = "ACTIVE"
EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class HistoricalContract:
    instrument_key: str
    underlying: str
    expiry: date
    instrument_type: str          # CE | PE | FUT
    lot_size: int
    source: str                   # ACTIVE | EXPIRED
    strike: float | None = None
    trading_symbol: str = ""
    freeze_quantity: int | None = None
    tick_size: float | None = None
    underlying_key: str | None = None
    weekly: bool | None = None

    @property
    def is_option(self) -> bool:
        return self.instrument_type in ("CE", "PE")

    def to_option_contract(self) -> OptionContract:
        return OptionContract(
            instrument_key=self.instrument_key, trading_symbol=self.trading_symbol, underlying=self.underlying,
            expiry=self.expiry, strike=float(self.strike or 0), option_type=OptionType(self.instrument_type),
            lot_size=int(self.lot_size), freeze_quantity=self.freeze_quantity, underlying_key=self.underlying_key,
            underlying_type="INDEX", tick_size=self.tick_size, weekly=self.weekly)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expiry"] = self.expiry.isoformat()
        return data


def contract_from_row(row: dict, source: str, *, tick_in_paise: bool) -> HistoricalContract:
    kind = str(row.get("instrument_type") or "").upper()
    tick = row.get("tick_size")
    freeze = row.get("freeze_quantity")
    return HistoricalContract(
        instrument_key=str(row["instrument_key"]),
        underlying=str(row.get("underlying_symbol") or row.get("name") or "").upper(),
        expiry=_expiry_date(row["expiry"]),
        instrument_type=kind,
        lot_size=int(float(row.get("lot_size") or row.get("minimum_lot") or 0)),
        source=source,
        strike=float(row["strike_price"]) if kind in ("CE", "PE") else None,
        trading_symbol=str(row.get("trading_symbol") or ""),
        freeze_quantity=int(float(freeze)) if freeze not in (None, "") else None,
        tick_size=(float(tick) / (TICK_SIZE_DIVISOR if tick_in_paise else 1.0)) if tick not in (None, "") else None,
        underlying_key=row.get("underlying_key"),
        weekly=row.get("weekly"),
    )


def load_instrument_master_rows(source) -> list[dict]:
    """Raw rows from the cached Upstox instrument master (downloaded on demand)."""
    source.option_contracts()
    path = source.cache_file
    payload = path.read_bytes()
    data = json.loads(gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload)
    if not isinstance(data, list):
        raise HistoricalDataError("INSTRUMENT_DATA_INVALID", "Instrument master is not a list")
    return data


class ContractCatalog:
    def __init__(
        self,
        client,
        active_rows: Callable[[], list[dict]],
        *,
        today: Callable[[], date] | None = None,
        underlying_keys: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self._active_rows = active_rows
        self.today = today or (lambda: datetime.now(IST).date())
        self._configured_keys = {k.upper(): v for k, v in (underlying_keys or {}).items()}
        self._lock = threading.Lock()
        self._active: list[HistoricalContract] | None = None
        self._expired: dict[tuple[str, str, date], list[HistoricalContract]] = {}
        self._expiries: dict[str, list[date]] = {}
        self.availability: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------- active
    def active_contracts(self) -> list[HistoricalContract]:
        with self._lock:
            if self._active is None:
                found = []
                for row in self._active_rows():
                    if row.get("segment") != "NSE_FO":
                        continue
                    if str(row.get("instrument_type") or "").upper() not in ("CE", "PE", "FUT"):
                        continue
                    try:
                        found.append(contract_from_row(row, ACTIVE, tick_in_paise=True))
                    except (KeyError, TypeError, ValueError):
                        continue
                self._active = found
            return self._active

    def underlying_key(self, underlying: str) -> str:
        name = underlying.upper()
        for contract in self.active_contracts():
            if contract.underlying == name and contract.underlying_key:
                return contract.underlying_key
        if name in self._configured_keys:
            return self._configured_keys[name]
        raise HistoricalDataError("UNDERLYING_NOT_FOUND", f"No underlying instrument key found for {name}")

    # --------------------------------------------------------------- expired
    def expired_expiries(self, underlying: str) -> list[date]:
        name = underlying.upper()
        if name not in self._expiries:
            try:
                expiries = self.client.expired_expiries(self.underlying_key(name))
            except HistoricalDataError as exc:
                self.availability[name] = {"expired_available": False, **exc.as_dict()}
                raise
            today = self.today()
            self._expiries[name] = [e for e in expiries if e < today]
            self.availability[name] = {"expired_available": True, "expired_expiries": len(self._expiries[name])}
        return self._expiries[name]

    def _expired_contracts(self, underlying: str, kind: str, expiry: date) -> list[HistoricalContract]:
        cache_key = (underlying, kind, expiry)
        if cache_key not in self._expired:
            key = self.underlying_key(underlying)
            rows = (self.client.expired_future_contracts(key, expiry) if kind == "FUT"
                    else self.client.expired_option_contracts(key, expiry))
            contracts = []
            for row in rows:
                try:
                    contracts.append(contract_from_row(row, EXPIRED, tick_in_paise=False))
                except (KeyError, TypeError, ValueError) as exc:
                    raise HistoricalDataError("MALFORMED_CONTRACT", f"Expired contract row invalid: {exc}") from exc
            self._expired[cache_key] = contracts
        return self._expired[cache_key]

    # ---------------------------------------------------------------- public
    def expired_option_contracts(self, underlying: str, expiry: date) -> list[HistoricalContract]:
        """CE/PE contracts of one expired expiry, exactly as returned by the Upstox expired-instruments API."""
        return [c for c in self._expired_contracts(underlying.upper(), "OPTION", expiry) if c.is_option]

    def contracts(self, underlying: str, from_date: date, to_date: date, *, kind: str = "OPTION",
                  include_expired: bool = True) -> list[HistoricalContract]:
        """Contracts whose expiry is on/after ``from_date`` and that could trade in the range.

        Options are returned for every expiry; the caller restricts each
        expiry to the window in which it was the nearest one.
        """
        name = underlying.upper()
        wanted = ("FUT",) if kind == "FUT" else ("CE", "PE")
        found = [c for c in self.active_contracts()
                 if c.underlying == name and c.instrument_type in wanted and c.expiry >= from_date]
        today = self.today()
        if include_expired and from_date < today:
            try:
                expiries = self.expired_expiries(name)
            except HistoricalDataError:
                expiries = []
            for expiry in expiries:
                if from_date <= expiry <= max(to_date, from_date):
                    found.extend(c for c in self._expired_contracts(name, "FUT" if kind == "FUT" else "OPTION", expiry)
                                 if c.instrument_type in wanted)
                elif expiry > to_date:
                    # The first expiry after the range still covers its tail.
                    found.extend(c for c in self._expired_contracts(name, "FUT" if kind == "FUT" else "OPTION", expiry)
                                 if c.instrument_type in wanted)
                    break
        by_key = {c.instrument_key: c for c in found}
        return sorted(by_key.values(), key=lambda c: (c.expiry, c.strike or 0, c.instrument_type))

    def calendar_complete_from(self, underlying: str, from_date: date) -> date:
        """Earliest date from which the expiry calendar is known to be complete.

        With expired data available that is ``from_date``. Without it only
        expiries on/after today are known, so earlier dates cannot be mapped
        to their true nearest expiry.
        """
        state = self.availability.get(underlying.upper(), {})
        return from_date if state.get("expired_available") else max(from_date, self.today())
