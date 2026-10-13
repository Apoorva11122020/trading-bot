"""Expiry and strike selection over the contracts actually listed."""
from __future__ import annotations

from bisect import bisect_left
from datetime import date

from .models import OptionResolutionError, OptionType

EXPIRY_MODES = ("nearest", "next", "date")
STRIKE_MODES = ("ATM", "ITM", "OTM", "EXACT")


def select_expiry(expiries: list[date], mode: str = "nearest", specific: str | date | None = None) -> date:
    available = sorted(set(expiries))
    if not available:
        raise OptionResolutionError("NO_EXPIRIES", "No listed expiries")
    mode = str(mode or "nearest").strip().lower()
    if mode == "nearest":
        return available[0]
    if mode == "next":
        if len(available) < 2:
            raise OptionResolutionError("EXPIRY_NOT_FOUND", "Only one expiry is listed; 'next' is unavailable")
        return available[1]
    if mode == "date":
        if not specific:
            raise OptionResolutionError("EXPIRY_NOT_FOUND", "A specific expiry date (YYYY-MM-DD) is required")
        try:
            wanted = specific if isinstance(specific, date) else date.fromisoformat(str(specific).strip())
        except ValueError as exc:
            raise OptionResolutionError("INVALID_EXPIRY", f"Expiry must be YYYY-MM-DD, got {specific!r}") from exc
        if wanted not in available:
            raise OptionResolutionError("EXPIRY_NOT_FOUND", f"Expiry {wanted.isoformat()} is not listed")
        return wanted
    raise OptionResolutionError("INVALID_EXPIRY_MODE", f"Unknown expiry mode {mode!r}")


def atm_strike(strikes: list[float], spot: float) -> float:
    """Listed strike nearest to spot; ties resolve to the lower strike."""
    available = sorted(set(strikes))
    if not available:
        raise OptionResolutionError("NO_STRIKES", "No listed strikes")
    if spot is None or spot <= 0:
        raise OptionResolutionError("SPOT_REQUIRED", "A positive underlying spot is required")
    i = bisect_left(available, spot)
    candidates = available[max(0, i - 1): i + 1]
    return min(candidates, key=lambda s: (abs(s - spot), s))


def select_strike(
    strikes: list[float],
    *,
    option_type: OptionType | str,
    mode: str = "ATM",
    spot: float | None = None,
    offset: int = 0,
    exact: float | None = None,
) -> float:
    """ATM: nearest listed strike. OTM: CE above / PE below ATM by `offset`
    listed strikes. ITM: CE below / PE above. EXACT: must be listed."""
    available = sorted(set(strikes))
    if not available:
        raise OptionResolutionError("NO_STRIKES", "No listed strikes")
    kind = option_type if isinstance(option_type, OptionType) else OptionType(str(option_type).upper())
    mode = str(mode or "ATM").strip().upper()
    if offset < 0:
        raise OptionResolutionError("INVALID_STRIKE_OFFSET", "Strike offset must be >= 0")

    if mode == "EXACT":
        if exact is None:
            raise OptionResolutionError("STRIKE_NOT_FOUND", "EXACT strike mode requires a strike")
        for strike in available:
            if abs(strike - float(exact)) < 1e-6:
                return strike
        raise OptionResolutionError("STRIKE_NOT_FOUND", f"Strike {float(exact):g} is not listed")

    if mode not in STRIKE_MODES:
        raise OptionResolutionError("INVALID_STRIKE_MODE", f"Unknown strike mode {mode!r}")

    atm = atm_strike(available, spot)
    index = available.index(atm)
    if mode == "ATM":
        return atm
    towards_otm = 1 if kind == OptionType.CE else -1
    step = towards_otm * offset if mode == "OTM" else -towards_otm * offset
    target = index + step
    if target < 0 or target >= len(available):
        raise OptionResolutionError(
            "STRIKE_OUT_OF_RANGE", f"{mode} offset {offset} from ATM {atm:g} is beyond the listed strikes")
    return available[target]
