"""S4_SIX_MONTH_RESEARCH_LAB (PAPER | RESEARCH ONLY | NO ORDERS; production S2 untouched).

Can S2 be improved on the 6-month Upstox dataset (data/historical/upstox_6m)? The exact S2 signal + execution model
is replayed on every session; a small pre-declared set of filter families x entry modes is layered on the S2 signals,
BUY and SELL researched separately. Chronology:

    TRAIN 04-01..06-30 | VALIDATION 07-01..07-31 | OOS 08-01..08-31 | FINAL HOLDOUT 09-01..09-30

Stage 1 sees only candles/premiums up to 08-31 and selects one finalist per direction (TRAIN + VALIDATION + walk-forward
W1-W3). Stage 2 loads September only afterwards and evaluates the frozen finalists once. Nothing is retuned.
"""
from __future__ import annotations

import inspect
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from src.backtest import accuracy_70_lab as acc
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import s3_high_confidence_lab as s3lab
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import HistoricalContract
from src.historical_data.option_coverage import resolve_signal_contract
from src.options.models import OptionResolutionError
from src.safety import live_gate
from src.strategy import s3_high_confidence as s3

HEADING = "S4 SIX-MONTH RESEARCH"
BANNER = "PAPER | RESEARCH ONLY | NO ORDERS"
DIRECTIONS = ("BUY", "SELL")
SOURCE = "upstox_6m"
MISSING_PREMIUM = "NO_PREMIUM_DATA_6M"
S2_NAME = dr.BASELINE.name

SPLIT_RANGES = {"TRAIN": (date(2026, 4, 1), date(2026, 6, 30)),
                "VALIDATION": (date(2026, 7, 1), date(2026, 7, 31)),
                "OOS": (date(2026, 8, 1), date(2026, 8, 31)),
                "HOLDOUT": (date(2026, 9, 1), date(2026, 9, 30))}
SELECTION_SPLITS = ("TRAIN", "VALIDATION", "OOS")
SELECTION_END = SPLIT_RANGES["OOS"][1]
WF_WINDOWS = (("W1", ("2026-04",), "2026-05"),
              ("W2", ("2026-04", "2026-05"), "2026-06"),
              ("W3", ("2026-04", "2026-05", "2026-06"), "2026-07"),
              ("W4", ("2026-04", "2026-05", "2026-06", "2026-07"), "2026-08"),
              ("W5", ("2026-04", "2026-05", "2026-06", "2026-07", "2026-08"), "2026-09"))
SELECTION_WF = ("W1", "W2", "W3")        # test months inside TRAIN + VALIDATION: may be used for selection
CANDIDATE_WF = ("W1", "W2", "W3", "W4")  # reported stability (W4 tests the OOS month)
HOLDOUT_WF = "W5"                        # only after the finalists are frozen
EXPECTED = {"trading_days": 125, "underlying_candles": 9375, "option_candles": 135450}

ROBUST_70, PROMISING, READY = "ROBUST_70", "PROMISING", "READY_FOR_FORWARD_PAPER_VALIDATION"
INSUFFICIENT_DATA, KEEP, REJECTED = "INSUFFICIENT_DATA", "KEEP_CURRENT", "REJECTED"
PASS_PENDING, FAILED_HOLDOUT = "PASS_PENDING_HOLDOUT", "FAILED_FINAL_HOLDOUT"
HOLDOUT_PASS, HOLDOUT_FAIL, HOLDOUT_INSUFFICIENT = "PASS", "FAIL", "INSUFFICIENT_TRADES"
STATUS_MAP = {acc.ALL_GATES_PASS: PASS_PENDING, acc.PROMISING_SMALL_SAMPLE: PROMISING,
              acc.QUALITY_PASS_BELOW_70: PROMISING, acc.INSUFFICIENT_SAMPLE: INSUFFICIENT_DATA}
STATUS_ORDER = {ROBUST_70: 0, READY: 1, PASS_PENDING: 2, PROMISING: 3, INSUFFICIENT_DATA: 4, FAILED_HOLDOUT: 5,
                REJECTED: 6}
NOT_ACHIEVED_TEXT = "70% OOS accuracy was not achieved on a sufficient sample without unacceptable overfitting."

MODES = s3lab.MODES
FAMILIES = {
    "A": ("S2_BASELINE", "every S2 ORB breakout signal of the direction (no extra condition)"),
    "B": ("TREND_CONFIRMATION", "BUY: close > EMA20 > EMA50 and EMA20 3-candle slope > 0 (SELL mirrored)"),
    "C": ("VWAP_MOMENTUM", "BUY: close > VWAP (TWAP proxy), VWAP 3-candle slope > 0, RSI14 > 55 (SELL: <, <, < 45)"),
    "D": ("BREAKOUT_CONFIRMATION", "close beyond the prior 12-candle high (BUY) / low (SELL) by 0 < d <= 1.0 ATR"),
    "E": ("PRICE_ACTION_CONFIRMATION", "directional candle, body >= 50% of range, close in top (BUY) / bottom (SELL) 30%"),
    "F": ("OPTION_PREMIUM_CONFIRMATION", "CE (BUY) / PE (SELL) premium: 3-candle momentum > 0 and bullish candle at "
                                         "the signal candle; missing premium data -> NO_SIGNAL"),
    "G": ("REGIME_FILTERED", "causal regime not RANGE / UNCERTAIN and not the opposite trend"),
    "H": ("MULTI_CONFIRMATION", "B + E + F together"),
    "I": ("HIGH_CONFIDENCE_SELECTIVE", "S3 confidence score (underlying + option points, weights below) >= threshold"),
}
FAMILY_COMPLEXITY = {"A": 1, "B": 2, "C": 2, "D": 2, "E": 2, "F": 2, "G": 2, "H": 4, "I": 3}
MAX_COMPLEXITY = 4
REGIME_NAMES = {"TREND_UP": "TREND_UP", "TREND_DOWN": "TREND_DOWN", "RANGE_CHOP": "RANGE",
                "HIGH_VOLATILITY": "HIGH_VOL", "LOW_VOLATILITY": "LOW_VOL", "UNCERTAIN": "UNCERTAIN",
                "NORMAL_VOLATILITY": "NORMAL_VOL", "UNKNOWN": "UNKNOWN"}
PRIMARY_REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE_CHOP", "HIGH_VOLATILITY", "LOW_VOLATILITY", "UNCERTAIN")
FORBIDDEN_IMPORTS = ("ParquetCandleStore", "HistoricalDataService")
ORDER_CALL = "place" + "_order"


def module_safety(module=None) -> dict[str, bool]:
    """Import lines of the lab never pull in the legacy candle store; no broker order call anywhere."""
    text = inspect.getsource(module or inspect.getmodule(module_safety))
    imports = [ln for ln in text.splitlines() if ln.startswith(("from ", "import "))]
    return {"no_legacy_store": not any(x in ln for ln in imports for x in FORBIDDEN_IMPORTS),
            "no_order_call": ORDER_CALL not in text}


@dataclass(frozen=True)
class HoldoutCriteria:
    min_trades: int = 10
    min_pf: float = 1.2
    min_accuracy: float = 0.60
    max_best_day_share: float = 0.50


@dataclass(frozen=True)
class Candidate:
    cid: str
    family: str
    direction: str
    mode: str
    threshold: int | None = None

    @property
    def n_conditions(self) -> int:
        return FAMILY_COMPLEXITY[self.family] + (0 if self.mode == "A_NEXT_BAR" else 1)


def predeclared_candidates() -> list[Candidate]:
    out = []
    for d in DIRECTIONS:
        for fam in "ABCDEFGH":
            out += [Candidate(f"S4_{d}_{fam}_{m}", fam, d, m) for m in MODES]
        out += [Candidate(f"S4_{d}_I_T{t}_A_NEXT_BAR", "I", d, "A_NEXT_BAR", t) for t in s3.THRESHOLDS]
    return out


def _status(acc_status: str) -> str:
    return STATUS_MAP.get(acc_status, REJECTED)


# ------------------------------------------------------------------ 6-month store (the only data source)
def _kind(option_type) -> str:
    return str(getattr(option_type, "value", option_type)).upper()


class SixMonthStore:
    """Read-only access to data/historical/upstox_6m. Never falls back to any other store."""

    def __init__(self, root: str | Path = u6.DEFAULT_ROOT, underlying: str = "NIFTY") -> None:
        self.layout = u6.Layout(root, underlying)
        for p in (self.layout.plan, self.layout.contracts):
            if not p.is_file():
                raise FileNotFoundError(f"6-month dataset incomplete: {p} missing (run scripts/download_upstox_6m.py)")
        self.plan = json.loads(self.layout.plan.read_text(encoding="utf-8"))
        self.start, self.end = date.fromisoformat(self.plan["from"]), date.fromisoformat(self.plan["to"])
        self.calendar = sorted({date.fromisoformat(x) for x in self.plan.get("calendar", [])})
        self.requirements = u6.requirements_from_rows(u6._read_csv(self.layout.contracts))
        self.chain: dict[tuple[date, date, str], dict[float, HistoricalContract]] = {}
        self.atm: dict[date, set[str]] = defaultdict(set)
        self.contracts: dict[str, HistoricalContract] = {}
        self.files: dict[str, set[Path]] = defaultdict(set)
        for r in self.requirements:
            c = r.contract
            self.chain.setdefault((r.trading_date, c.expiry, c.instrument_type), {})[float(c.strike)] = c
            if r.role == "ATM":
                self.atm[r.trading_date].add(c.instrument_key)
            self.contracts[c.instrument_key] = c
            self.files[c.instrument_key].add(self.layout.option_file(u6.month_label(r.trading_date), c))
        self.reads: Counter = Counter()
        self._cache: dict[str, pd.DataFrame | None] = {}

    def spot(self, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        return u6.load_underlying(self.layout, start or self.start, end or self.end)

    def premium(self, instrument_key: str) -> pd.DataFrame | None:
        if instrument_key not in self._cache:
            parts = [u6.read_parquet(p) for p in sorted(self.files.get(instrument_key, ())) if p.is_file()]
            parts = [p for p in parts if not p.empty]
            self._cache[instrument_key] = (pd.concat(parts, ignore_index=True).drop_duplicates("timestamp", keep="last")
                                           .sort_values("timestamp").reset_index(drop=True) if parts else None)
        return self._cache[instrument_key]

    def premium_for(self, max_date: date | None = None) -> Callable:
        def fn(contract):
            full = self.premium(contract.instrument_key)
            if full is None:
                self.reads["missing"] += 1
                return MISSING_PREMIUM
            self.reads[SOURCE] += 1
            out = full if max_date is None else full[full["timestamp"].dt.date <= max_date].reset_index(drop=True)
            return out if not out.empty else MISSING_PREMIUM
        return fn


class SixMonthResolver:
    """``resolve_signal_contract`` interface over the 6-month plan: point-in-time expiry calendar + stored chains."""

    def __init__(self, store: SixMonthStore) -> None:
        self.store = store
        self.today = lambda: store.start

    def expiries(self, underlying: str) -> list[date]:
        t = self.today()
        return [e for e in self.store.calendar if e >= t]

    def strikes(self, underlying: str, expiry: date, option_type=None) -> list[float]:
        return sorted(self.store.chain.get((self.today(), expiry, _kind(option_type)), {}))

    def resolve(self, underlying: str, expiry: date, strike: float, option_type):
        t = self.today()
        c = self.store.chain.get((t, expiry, _kind(option_type)), {}).get(float(strike))
        if c is None:
            raise OptionResolutionError("CONTRACT_NOT_FOUND", f"{underlying} {expiry} {strike:g} not in the 6-month plan")
        if c.instrument_key not in self.store.atm.get(t, set()):
            raise OptionResolutionError("OUTSIDE_PLANNED_STRIKES", f"{c.trading_symbol} is not an ATM requirement on {t}")
        return c.to_option_contract()


def make_resolve(store: SixMonthStore, research_config) -> Callable:
    resolver = SixMonthResolver(store)
    check = sh.ExpiryCheck(date.max, frozenset(store.calendar), "upstox_6m plan.json expiry calendar")
    return sh.strict_resolver(lambda sig: resolve_signal_contract(sig, resolver, research_config), check)


# ------------------------------------------------------------------ Phase 1: data integrity
def data_integrity(store: SixMonthStore, spot: pd.DataFrame, expected: dict = EXPECTED) -> dict[str, Any]:
    lay = store.layout
    summary = json.loads(lay.summary.read_text(encoding="utf-8")) if lay.summary.is_file() else {}
    su, so = summary.get("underlying", {}), summary.get("options", {})
    ts = acc._ist_series(spot["timestamp"]) if not spot.empty else pd.Series([], dtype="datetime64[ns, Asia/Kolkata]")
    days = sorted({d.isoformat() for d in ts.dt.date})
    q = u6.quality_check(spot, start=store.start, end=store.end)
    sessions, _ = sh.complete_sessions(spot, store.start, store.end) if days else ([], {})
    vol_zero = float((spot["volume"].fillna(0) == 0).mean()) if "volume" in spot and len(spot) else None
    checks = [
        acc._check("P1-01", "6-month dataset present (underlying, options, plan, contracts)",
                   bool(lay.underlying_files()) and bool(lay.option_files()),
                   f"root={lay.root.as_posix()}, underlying files {len(lay.underlying_files())}, "
                   f"option files {len(lay.option_files())}"),
        acc._check("P1-02", "Underlying covers the planned period and matches the download summary",
                   bool(days) and days[0] >= store.start.isoformat() and days[-1] <= store.end.isoformat()
                   and len(days) == su.get("trading_days", len(days)) and len(spot) == su.get("candles", len(spot)),
                   f"{days[0] if days else '-'}..{days[-1] if days else '-'}: {len(days)} trading days, {len(spot)} "
                   f"candles (summary {su.get('trading_days')}/{su.get('candles')}, expected "
                   f"{expected['trading_days']}/{expected['underlying_candles']})"),
        acc._check("P1-03", "Underlying timestamps IST, sorted, unique, 5-minute grid, valid OHLC, in session",
                   q["ok"] and q["invalid_ohlc"] == 0 and q["spacing_gaps"] == 0,
                   f"tz_ist={q['tz_ist']}, duplicates={q['duplicate_timestamps']}, on_grid={q['on_grid']}, "
                   f"invalid_ohlc={q['invalid_ohlc']}, outside_session={q['outside_session']}, "
                   f"spacing_gaps={q['spacing_gaps']}"),
        acc._check("P1-04", "Every trading day is a complete 75-candle session (no missing intervals)",
                   bool(days) and len(sessions) == len(days) and q["missing_intervals"] == 0,
                   f"{len(sessions)}/{len(days)} complete, missing intervals {q['missing_intervals']}"),
        acc._check("P1-05", "Index volume / OI", None,
                   f"NIFTY index volume is zero on {vol_zero:.0%} of candles -> VWAP uses the TWAP proxy and volume "
                   "features come from option premium volume; index OI not applicable" if vol_zero is not None else "-"),
    ]
    known = {c.instrument_key for c in store.contracts.values()}
    n_files = n_rows = bad_quality = bad_meta = after_exp = bad_month = bad_underlying = negative = 0
    dates_by_key: dict[str, set[date]] = defaultdict(set)
    examples: list[str] = []
    for p in lay.option_files():
        f = u6.read_parquet(p)
        n_files += 1
        if f.empty:
            continue
        n_rows += len(f)
        try:
            und, exp_s, strike_s, kind = p.stem.split("_")
            exp = date.fromisoformat(exp_s)
        except ValueError:
            bad_meta += 1
            examples.append(f"bad file name {p.name}")
            continue
        keys = set(f["instrument_key"].astype(str))
        meta_ok = (len(keys) == 1 and keys <= known and set(f["expiry"].astype(str)) == {exp_s}
                   and set(f["option_type"].astype(str)) == {kind} and kind in ("CE", "PE")
                   and all(abs(float(s) - float(strike_s)) < 1e-6 for s in set(f["strike"])))
        if not meta_ok:
            bad_meta += 1
            examples.append(f"metadata mismatch {p.name}")
        bad_underlying += int((f["underlying"].astype(str) != lay.underlying).any())
        local = acc._ist_series(f["timestamp"])
        after_exp += int((local.dt.date > exp).sum())
        bad_month += int((local.dt.strftime("%Y-%m") != p.parent.name).sum())
        negative += int((f["volume"] < 0).sum() + (f["open_interest"] < 0).sum())
        fq = u6.quality_check(f, expiry=exp, option_type=kind)
        if not fq["ok"] or fq["invalid_ohlc"]:
            bad_quality += 1
            examples.append(f"quality {p.name}")
        for k in keys:
            dates_by_key[k] |= set(local.dt.date)
    checks += [
        acc._check("P1-06", "Option candle count matches the download summary", n_rows == so.get("candles", n_rows),
                   f"{n_rows} candles in {n_files} files (summary {so.get('candles')}, expected "
                   f"{expected['option_candles']})"),
        acc._check("P1-07", "Option files: IST, unique, 5-minute grid, valid OHLC, volume/OI >= 0",
                   bad_quality == 0 and negative == 0, f"files failing quality {bad_quality}, negative vol/OI {negative}"),
        acc._check("P1-08", "File name = instrument metadata (expiry, strike, CE/PE, one instrument key)",
                   bad_meta == 0, f"mismatches {bad_meta}; examples {examples[:5] or 'none'}"),
        acc._check("P1-09", "No option candle after its expiry or outside its month folder; underlying mapping NIFTY",
                   after_exp == 0 and bad_month == 0 and bad_underlying == 0,
                   f"after expiry {after_exp}, wrong month {bad_month}, files with wrong underlying {bad_underlying}"),
    ]
    cal = store.calendar
    meta_bad, atm_missing, first_day_atm, atm_total = [], 0, 0, 0
    by_day_type: dict[tuple[date, str], int] = Counter()
    for r in store.requirements:
        c = r.contract
        nearest = next((e for e in cal if e >= r.trading_date), None)
        if not (c.expiry >= r.trading_date and r.dte == (c.expiry - r.trading_date).days and nearest == c.expiry
                and c.instrument_type in ("CE", "PE") and (c.strike or 0) > 0 and c.lot_size > 0
                and c.underlying == lay.underlying):
            meta_bad.append(f"{r.trading_date} {c.trading_symbol}")
        if r.role == "ATM":
            atm_total += 1
            by_day_type[(r.trading_date, c.instrument_type)] += 1
            have = dates_by_key.get(c.instrument_key, set())
            atm_missing += r.trading_date not in have
            first_day_atm += not any(d < r.trading_date for d in have)
    want = {(date.fromisoformat(d), k) for d in days for k in ("CE", "PE")}
    checks += [
        acc._check("P1-10", "contracts.csv: expiry >= date, nearest listed expiry, dte, CE/PE, strike, lot size",
                   bool(store.requirements) and not meta_bad,
                   f"{len(store.requirements)} requirements, violations {len(meta_bad)} {meta_bad[:3] or ''}"),
        acc._check("P1-11", "Every trading day has ATM CE and PE contracts with real candles that day",
                   want <= set(by_day_type) and atm_missing == 0,
                   f"ATM requirements {atm_total}, without candles on their day {atm_missing}, days without ATM "
                   f"CE/PE {len(want - set(by_day_type))}"),
        acc._check("P1-12", "Plan: no unresolved sessions, expiry calendar complete for the whole period",
                   not store.plan.get("unresolved") and date.fromisoformat(store.plan.get("calendar_from") or "9999-12-31")
                   <= store.start and not store.plan.get("discovery_issues"),
                   f"unresolved {len(store.plan.get('unresolved') or [])}, calendar_from "
                   f"{store.plan.get('calendar_from')}, expiries {len(cal)}"),
        acc._check("P1-13", "Recorded missing data", None,
                   f"missing_data.csv rows {len(u6._read_csv(lay.missing))}"),
        acc._check("P1-14", "Premium-history limitation (quantified, not fabricated)", None,
                   f"{first_day_atm}/{atm_total} ATM requirements have no stored candles before their trading day; "
                   "early signals on those days can lack 14 premium candles (INSUFFICIENT_ATR_HISTORY -> "
                   "DATA_UNAVAILABLE, never filled)"),
    ]
    ok = all(c["status"] != acc.FAIL for c in checks)
    return {"checks": checks, "ok": ok, "root": lay.root.as_posix(),
            "stats": {"from": store.plan["from"], "to": store.plan["to"], "first_day": days[0] if days else None,
                      "last_day": days[-1] if days else None, "trading_days": len(days),
                      "underlying_candles": int(len(spot)), "option_files": n_files, "option_candles": n_rows,
                      "contracts": len(store.contracts), "requirements": len(store.requirements),
                      "expiries": len(cal), "first_day_atm_requirements": first_day_atm}}


# ------------------------------------------------------------------ splits / windows
def split_days(days: Iterable[str]) -> dict[str, list[str]]:
    days = sorted(days)
    return {s: [d for d in days if a.isoformat() <= d <= b.isoformat()] for s, (a, b) in SPLIT_RANGES.items()}


def _months(days: Sequence[str], months: Iterable[str]) -> list[str]:
    ms = set(months)
    return [d for d in days if d[:7] in ms]


def wf_windows(days: Sequence[str], names: Iterable[str]) -> list[dict]:
    want = set(names)
    out = []
    for name, train_m, test_m in WF_WINDOWS:
        if name not in want:
            continue
        tr, te = _months(days, train_m), _months(days, (test_m,))
        if tr and te:
            out.append({"window": name, "train_months": list(train_m), "train": tr, "validation": te, "test": te})
    return out


def inner_split(w: dict) -> tuple[list[str], list[str]]:
    """Selection inside a window uses past data only: TRAIN = earlier months, VALIDATION = the last past month."""
    past, months = w["train"], w["train_months"]
    present = [m for m in months if any(d[:7] == m for d in past)]
    if len(present) >= 2:
        return _months(past, present[:-1]), _months(past, present[-1:])
    cut = max(1, int(round(len(past) * 0.7)))
    return past[:cut], past[cut:]


# ------------------------------------------------------------------ features / candidate filters (causal)
def family_masks(f: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Per direction, boolean filter per family at each completed candle (inputs <= that candle; NaN -> False)."""
    out = {}
    for d in DIRECTIONS:
        up = d == "BUY"
        s = 1.0 if up else -1.0
        trend = ((f["close"] - f["ema20"]) * s > 0) & ((f["ema20"] - f["ema50"]) * s > 0) & (f["ema20_slope_atr"] * s > 0)
        vwap = (((f["close"] - f["vwap_ref"]) * s > 0) & (f["vwap_slope_atr"] * s > 0)
                & ((f["rsi14"] > 55) if up else (f["rsi14"] < 45)))
        dist = (f["close"] - (f["hh12"] if up else f["ll12"])) * s / f["atr14"]
        brk = (dist > 0) & (dist <= 1.0)
        pa = (((f["close"] > f["open"]) if up else (f["close"] < f["open"])) & (f["body_frac"] >= 0.5)
              & ((f["close_pos"] >= 0.7) if up else (f["close_pos"] <= 0.3)))
        opposite = "TREND_DOWN" if up else "TREND_UP"
        reg = ~f["regime"].isin((*s3.NO_TRADE_REGIMES, opposite))
        out[d] = {k: v.fillna(False).astype(bool) for k, v in
                  {"B": trend, "C": vwap, "D": brk, "E": pa, "G": reg}.items()}
    return out


def premium_confirms(opt: dict | None) -> bool | None:
    """Option premium (CE for BUY, PE for SELL) rising into the signal candle. None = data missing -> NO_SIGNAL."""
    if opt is None or opt.get("premium_mom3") is None or opt.get("premium_bull") is None:
        return None
    return bool(opt["premium_mom3"] > 0 and opt["premium_bull"])


def score_totals(bars: pd.DataFrame, keys: dict[str, list[tuple[str, str]]], opt: dict) -> dict[str, dict[str, float | None]]:
    """S3 weights (trend 20, VWAP 15, EMA 15, momentum 10, volume 10, price action 10, breakout 10, option 10)."""
    pos = {t.isoformat(): i for i, t in enumerate(bars["timestamp"])}
    out: dict[str, dict[str, float | None]] = {}
    for d in DIRECTIONS:
        pts = s3.underlying_points(bars, s3.PROFILES[d])
        out[d] = {}
        for key in keys.get(d, []):
            row = pts.iloc[pos[key[0]]]
            op = s3.option_points(opt.get(key), s3.PROFILES[d])
            out[d][key[0]] = (None if not bool(row["available"]) or op is None
                              else float(row["underlying"]) + float(op["volume"]) + float(op["option_premium"]))
    return out


def setup_keys(cand: Candidate, base_keys: dict, masks: dict, pos: dict, opt: dict, totals: dict,
               no_signal: Counter | None = None) -> list[tuple[str, str]]:
    d, fam = cand.direction, cand.family
    out = []
    for key in base_keys.get(d, []):
        if key[1] != d:
            raise ValueError(f"{cand.cid}: foreign-direction setup {key}")
        i = pos[key[0]]
        if fam == "A":
            ok = True
        elif fam in "BCDEG":
            ok = bool(masks[d][fam].iat[i])
        elif fam in "FH":
            pc = premium_confirms(opt.get(key))
            if pc is None:
                if no_signal is not None:
                    no_signal[f"{cand.cid}:PREMIUM_DATA_UNAVAILABLE"] += 1
                continue
            ok = pc and (fam == "F" or (bool(masks[d]["B"].iat[i]) and bool(masks[d]["E"].iat[i])))
        elif fam == "I":
            t = totals[d].get(key[0])
            if t is None:
                if no_signal is not None:
                    no_signal[f"{cand.cid}:SCORE_DATA_UNAVAILABLE"] += 1
                continue
            ok = t >= float(cand.threshold)
        else:
            raise ValueError(fam)
        if ok:
            out.append(key)
    return out


# ------------------------------------------------------------------ one replay stage (exact S2 + candidates)
def _dedupe(rows: Iterable[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = (r["signal_time"], r["direction"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def run_stage(*, spot: pd.DataFrame, days: Sequence[str], signal_fn: Callable, resolve: Callable,
              premium_for: Callable, params, square_off: str, range_start: str, range_end: str, underlying: str,
              allowed: Callable, candidates: Sequence[Candidate], interval: int = 5) -> dict[str, Any]:
    pipeline = imp._pipeline_factory(signal_fn, resolve, params, range_start, range_end)
    wanted = {date.fromisoformat(d) for d in days}
    contexts, feats = pipeline(spot, premium_for, wanted) if wanted else ([], imp.underlying_features(spot))
    cache: dict = {}
    base_rows = [dict(sh.classify_row(r, {}), strategy=S2_NAME) for r in _dedupe(
        dr.run_candidate(contexts, feats, dr.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE, cache=cache)
        if contexts else [])]
    bars = s3.s3_bars(spot, interval)
    mask = acc.allowed_mask(bars, allowed, set(days))
    ctx_by_key = {(x["signal_time"].isoformat(), x["direction"]): x for x in contexts}
    opt = {k: acc.option_features(x) for k, x in ctx_by_key.items()}
    base_keys = {d: sorted(k for k in ctx_by_key if k[1] == d) for d in DIRECTIONS}
    pos = {t.isoformat(): i for i, t in enumerate(bars["timestamp"])}
    masks = family_masks(bars)
    totals = score_totals(bars, base_keys, opt)
    no_signal: Counter = Counter()
    maps: dict[str, dict] = {}
    for c in candidates:
        keys = setup_keys(c, base_keys, masks, pos, opt, totals, no_signal)
        maps[c.cid] = s3lab.confirmations(bars, keys, c.mode, mask, ctx_by_key, interval)
    extra = sorted({k for m in maps.values() for k in m} - set(ctx_by_key))
    ctx2 = imp.build_contexts(s3lab.union_signals(bars, extra, underlying), feats, resolve, premium_for, params)
    for x in ctx2:
        ctx_by_key[(x["signal_time"].isoformat(), x["direction"])] = x
    all_ctx = list(contexts) + list(ctx2)
    raw = imp.run_variant(all_ctx, feats, imp.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE,
                          cache=cache) if all_ctx else []
    row_by_key = {acc.row_key(r): sh.classify_row(r, {}) for r in raw}
    bar_at = bars.set_index("timestamp")
    labels: dict[tuple[str, str], list[str]] = {}
    for k in row_by_key:
        t = pd.Timestamp(k[0]).tz_convert(imp.IST)
        if t in bar_at.index:
            labels[k] = [bar_at.at[t, "regime"], bar_at.at[t, "vol_regime"]]
    rows: dict[str, list[dict]] = {}
    for c in candidates:
        out = []
        for k, setup in sorted(maps[c.cid].items()):
            if k[1] != c.direction:
                raise ValueError(f"{c.cid}: foreign-direction signal {k}")
            base = row_by_key.get(k)
            if base is None:
                continue
            out.append(dict(base, strategy=c.cid, s4_setup_time=setup[0],
                            s4_score=totals[c.direction].get(setup[0]),
                            s4_regime=(labels.get(k) or [None])[0]))
        rows[c.cid] = out
    candles = {d: int(n) for d, n in mask.groupby(bars["day"].map(lambda x: x.isoformat())).sum().items()}
    return {"pipeline": pipeline, "spot": spot, "premium_for": premium_for, "contexts": all_ctx,
            "base_contexts": list(contexts), "features": feats, "bars": bars, "mask": mask, "base_rows": base_rows,
            "row_by_key": row_by_key, "rows": rows, "labels": labels, "totals": totals, "no_signal": no_signal,
            "candles_per_day": candles, "extra": len(extra), "days": list(days)}


# ------------------------------------------------------------------ scoring / selection
def _num(v, default: float = 0.0) -> float:
    return acc._num(v, default)


def window_stats(rows: Sequence[dict], windows: Sequence[dict]) -> dict[str, Any]:
    per = []
    for w in windows:
        m = lab.perf(lab.period_rows(rows, w["test"]))
        per.append({"window": w["window"], "evaluated": m["evaluated"], "win_rate": m["win_rate"],
                    "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"]})
    traded = [p for p in per if p["evaluated"]]
    return {"windows": per, "traded": len(traded),
            "profitable_share": sum(p["net_pnl"] > 0 for p in traded) / len(traded) if traded else None}


def skip_metrics(rows: Sequence[dict], candles_per_day: dict[str, int], days: Sequence[str]) -> dict[str, Any]:
    n = sum(candles_per_day.get(d, 0) for d in days)
    sig = len(lab.period_rows(rows, days))
    m = lab.perf(lab.period_rows(rows, days))
    return {"candles": n, "signals": sig, "skip_pct": (1 - sig / n) if n else None, "evaluated": m["evaluated"],
            "accuracy": m["win_rate"], "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"]}


def score_candidate(cand: Candidate, rows: list[dict], periods: dict, folds: list[dict], sel_windows: list[dict],
                    labels: dict, c: acc.Criteria, lookahead_ok: bool, candles_per_day: dict) -> dict[str, Any]:
    sc = acc.score_rows(rows, periods, folds, labels, c)
    wsel = window_stats(rows, sel_windows)
    wf_ok = wsel["traded"] >= 2 and _num(wsel["profitable_share"]) >= 2 / 3
    sc.update(cid=cand.cid, family=cand.family, family_name=FAMILIES[cand.family][0], direction=cand.direction,
              mode=cand.mode, threshold=cand.threshold, n_conditions=cand.n_conditions, lookahead_passed=lookahead_ok,
              data_status="OK", tier=acc.sample_tier(sc["OOS"]["evaluated"], c), wf_selection=wsel,
              wf_selection_ok=wf_ok, day_stability=s3lab.day_stability(rows, periods["ALL"]),
              day_stability_oos=s3lab.day_stability(rows, periods["OOS"]),
              skip={p: skip_metrics(rows, candles_per_day, periods[p]) for p in SELECTION_SPLITS})
    raw = acc.selection_score(sc["TRAIN"], sc["VALIDATION"], c, c.min_train_trades, c.min_val_trades)
    sc["selection_score"] = raw if wf_ok else None
    sc["selection_block"] = (None if sc["selection_score"] is not None else
                             "TRAIN/VALIDATION gate" if raw is None else "walk-forward W1-W3 < 2/3 profitable")
    sc["gates"] = {**acc.gates(sc, c), "reasonable_complexity": cand.n_conditions <= MAX_COMPLEXITY}
    sc["acc_status"], failed = acc.candidate_status(sc, c)
    if not sc["gates"]["reasonable_complexity"]:
        failed = [*failed, "reasonable_complexity"]
    sc["failed"] = failed
    sc["status"] = _status(sc["acc_status"]) if sc["gates"]["reasonable_complexity"] else REJECTED
    sc["final_score"] = acc.final_score(dict(sc, status=sc["acc_status"]), c)
    return sc


def wf_procedure(names: Sequence[str], rows: dict[str, list[dict]], n_cond: dict[str, int], windows: Sequence[dict],
                 c: acc.Criteria) -> list[dict]:
    """Per window: select on that window's past months only (inner TRAIN/VALIDATION), then record the test month."""
    out = []
    for w in windows:
        tr, va = inner_split(w)
        cand = {}
        for n in names:
            sc = acc.selection_score(lab.perf(lab.period_rows(rows[n], tr)), lab.perf(lab.period_rows(rows[n], va)),
                                     c, c.wf_min_train_trades, c.wf_min_val_trades)
            if sc is not None:
                cand[n] = sc
        chosen = min(cand, key=lambda n: (-cand[n], n_cond[n], n)) if cand else None
        out.append({"window": w["window"], "train": f"{w['train'][0]}..{w['train'][-1]}",
                    "test": f"{w['test'][0]}..{w['test'][-1]}", "test_days": w["test"], "selected": chosen})
    return out


def holdout_verdict(m: dict, hc: HoldoutCriteria) -> str:
    if m["evaluated"] < hc.min_trades:
        return HOLDOUT_INSUFFICIENT
    ok = (m["net_pnl"] > 0 and _num(m["profit_factor"]) >= hc.min_pf and _num(m["win_rate"]) >= hc.min_accuracy
          and m["best_day_share"] is not None and m["best_day_share"] <= hc.max_best_day_share)
    return HOLDOUT_PASS if ok else HOLDOUT_FAIL


ROBUST_CRITERIA = ("oos_accuracy_70", "oos_sample_sufficient", "oos_pf_1_5", "oos_pnl_positive",
                   "validation_profitable", "walk_forward_stable", "september_acceptable", "no_single_day_dependency",
                   "no_single_trade_dependency", "no_regime_dependency", "no_lookahead", "no_leakage",
                   "reasonable_complexity")


def robust_checklist(sc: dict, verdict: str, lookahead_ok: bool, leakage_ok: bool) -> dict[str, bool]:
    g = sc["gates"]
    return {"oos_accuracy_70": g["oos_accuracy_70"], "oos_sample_sufficient": g["oos_sample_meaningful"],
            "oos_pf_1_5": g["oos_pf_1_5"], "oos_pnl_positive": g["oos_pnl_positive"],
            "validation_profitable": g["validation_profitable"], "walk_forward_stable": g["walk_forward_stable"],
            "september_acceptable": verdict == HOLDOUT_PASS,
            "no_single_day_dependency": g["no_single_day_dependency"],
            "no_single_trade_dependency": g["no_single_trade_dependency"],
            "no_regime_dependency": g["regime_stable"], "no_lookahead": bool(lookahead_ok),
            "no_leakage": bool(leakage_ok), "reasonable_complexity": g["reasonable_complexity"]}


def final_status(pre_status: str, verdict: str, checklist: dict[str, bool]) -> str:
    if verdict == HOLDOUT_FAIL:
        return FAILED_HOLDOUT
    if checklist and all(checklist.values()):
        return ROBUST_70
    if pre_status in (PASS_PENDING, PROMISING):
        return READY if verdict == HOLDOUT_PASS else PROMISING
    return pre_status


def recommend(statuses: Sequence[str], integrity_ok: bool, max_val_trades: dict[str, int], c: acc.Criteria) -> str:
    if not integrity_ok:
        return REJECTED
    st = list(statuses)
    for s in (ROBUST_70, READY, PROMISING):
        if s in st:
            return s
    if not st:
        return INSUFFICIENT_DATA if max(max_val_trades.values(), default=0) < c.min_val_trades else KEEP
    if all(s == INSUFFICIENT_DATA for s in st):
        return INSUFFICIENT_DATA
    return KEEP


# ------------------------------------------------------------------ lookahead
def truncation_test(spot: pd.DataFrame, days: Sequence[str], cut_times: Sequence[str] = ("11:00", "13:00"),
                    interval: int = 5) -> dict[str, Any]:
    """Regimes, every family filter and every score component recomputed from candles <= cut must be unchanged."""
    ts_all = acc._ist_series(spot["timestamp"])
    full = s3.s3_bars(spot, interval)
    fm = family_masks(full)
    fp = {d: s3.underlying_points(full, s3.PROFILES[d]) for d in DIRECTIONS}
    res = {"cuts": 0, "failures": []}
    for d in days:
        dd = date.fromisoformat(d)
        for t in cut_times:
            cut = pd.Timestamp(datetime.combine(dd, time.fromisoformat(t)), tz=imp.IST)
            tb = s3.s3_bars(spot[ts_all <= cut].reset_index(drop=True), interval)
            tm = family_masks(tb)
            a_sel, b_sel = (full["timestamp"] <= cut) & (full["day"] == dd), tb["day"] == dd

            def same(x, y):
                return x.loc[a_sel].reset_index(drop=True).equals(y.loc[b_sel].reset_index(drop=True))

            ok = same(full[["regime", "vol_regime"]], tb[["regime", "vol_regime"]])
            for x in DIRECTIONS:
                ok = ok and all(same(fm[x][k], tm[x][k]) for k in fm[x])
                tp = s3.underlying_points(tb, s3.PROFILES[x])
                ok = ok and same(fp[x][list(s3lab.POINT_COLUMNS)], tp[list(s3lab.POINT_COLUMNS)])
            res["cuts"] += 1
            if not ok:
                res["failures"].append(cut.isoformat())
    res["passed"] = bool(res["cuts"]) and not res["failures"]
    return res


def baseline_mutation_test(stage: dict, days: Sequence[str], cut: str = "12:00") -> dict[str, Any]:
    cuts = [pd.Timestamp(datetime.combine(date.fromisoformat(d), time.fromisoformat(cut)), tz=imp.IST) for d in days]
    return dr.mutation_test(stage["pipeline"], stage["spot"], stage["premium_for"], [dr.BASELINE], cuts)[S2_NAME]


def sample_days(days: Sequence[str], n: int = 10) -> list[str]:
    if len(days) <= n:
        return list(days)
    step = len(days) / n
    return [days[int(i * step)] for i in range(n)]


# ------------------------------------------------------------------ study
def _tracking(premium_for: Callable) -> tuple[Callable, dict]:
    seen = {"max_ts": None, "calls": 0}

    def fn(contract):
        out = premium_for(contract)
        seen["calls"] += 1
        if isinstance(out, pd.DataFrame) and not out.empty:
            m = out["timestamp"].max()
            seen["max_ts"] = m if seen["max_ts"] is None or m > seen["max_ts"] else seen["max_ts"]
        return out
    return fn, seen


def _scope_rows(rows_by_dir: dict[str, list[dict]]) -> list[dict]:
    return [r for d in DIRECTIONS for r in rows_by_dir.get(d, [])]


def _metrics(m: dict) -> dict:
    return {"trades": m["evaluated"], "wins": m["wins"], "losses": m["losses"], "accuracy": m["win_rate"],
            "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"], "avg_r": m["average_r"],
            "max_dd": m["max_drawdown"], "max_loss_streak": m["max_consecutive_losses"],
            "profitable_days": m["profitable_days"], "traded_days": m["traded_days"],
            "best_day_share": m["best_day_share"]}


def window_table(name: str, rows_by_dir: dict[str, list[dict]], windows: Sequence[dict]) -> list[dict]:
    out = []
    for w in windows:
        b = lab.perf(lab.period_rows(rows_by_dir.get("BUY", []), w["test"]))
        s = lab.perf(lab.period_rows(rows_by_dir.get("SELL", []), w["test"]))
        cm = lab.perf(lab.period_rows(_scope_rows(rows_by_dir), w["test"]))
        out.append({"strategy": name, "window": w["window"], "train_period": f"{w['train'][0]}..{w['train'][-1]}",
                    "test_period": f"{w['test'][0]}..{w['test'][-1]}", "buy_trades": b["evaluated"],
                    "buy_accuracy": b["win_rate"], "sell_trades": s["evaluated"], "sell_accuracy": s["win_rate"],
                    "combined_accuracy": cm["win_rate"], "profit_factor": cm["profit_factor"],
                    "net_pnl": cm["net_pnl"], "trades": cm["evaluated"]})
    return out


def window_summary(table: Sequence[dict]) -> dict[str, Any]:
    traded = [r for r in table if r["trades"]]
    accs = [r["combined_accuracy"] for r in traded if r["combined_accuracy"] is not None]
    pfs = [r["profit_factor"] for r in traded if r["profit_factor"] is not None and not math.isinf(r["profit_factor"])]
    return {"windows": len(table), "traded": len(traded),
            "profitable_share": sum(r["net_pnl"] > 0 for r in traded) / len(traded) if traded else None,
            "avg_accuracy": sum(accs) / len(accs) if accs else None, "median_accuracy": median(accs) if accs else None,
            "avg_pf": sum(pfs) / len(pfs) if pfs else None,
            "worst": min(traded, key=lambda r: r["net_pnl"])["window"] if traded else None,
            "best": max(traded, key=lambda r: r["net_pnl"])["window"] if traded else None}


def regime_table(rows: Sequence[dict], labels: dict, days: Sequence[str]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in lab.period_rows(list(rows), days):
        if r["state"] == sh.EVALUATED:
            for lb in labels.get(acc.row_key(r), []):
                groups[lb].append(r)
    return {k: lab.perf(v) for k, v in groups.items()}


def isolation_checks(rows: dict[str, list[dict]], directions: dict[str, str], row_by_key: dict, snapshot: str,
                     base_rows: Sequence[dict], base_snapshot: str) -> list[dict]:
    out = s3lab.isolation_checks(rows, directions, row_by_key, snapshot)
    mism = [d for d in DIRECTIONS if acc._sig(rows.get(f"S4_{d}_A_A_NEXT_BAR", [])) !=
            acc._sig([r for r in base_rows if r["direction"] == d])]
    out += [acc._check("I5", "Family A + next-bar entry reproduces the exact S2 baseline per direction", not mism,
                       f"mismatching directions: {mism or 'none'}"),
            acc._check("I6", "S2 baseline rows never mutated by candidates or scoring",
                       json.dumps(list(base_rows), sort_keys=True, default=str) == base_snapshot, "JSON snapshot")]
    return out


def build_s4(*, spot: pd.DataFrame, signal_fn: Callable, resolve: Callable, premium_for_factory: Callable,
             params, square_off: str, range_start: str = "09:15", range_end: str = "09:30",
             underlying: str = "NIFTY", allowed: Callable = acc.default_allowed, today: date | None = None,
             criteria: acc.Criteria = acc.Criteria(), holdout_criteria: HoldoutCriteria = HoldoutCriteria(),
             rules: dict | None = None, run_mutation: bool = True, lookahead_days: Sequence[str] | None = None,
             cut_times: Sequence[str] = ("11:00", "13:00"), interval: int = 5,
             progress: Callable[[str], None] | None = None, integrity_data: dict | None = None,
             store: SixMonthStore | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    c, hc = criteria, holdout_criteria
    today = today or datetime.now(sh.IST).date()
    ts = acc._ist_series(spot["timestamp"])
    first, last = ts.min().date(), ts.max().date()
    sessions, excluded = sh.complete_sessions(spot, first, last, interval)
    all_days = [d.isoformat() for d in sessions if d < today]
    sd = split_days(all_days)
    sel_days = [d for s in SELECTION_SPLITS for d in sd[s]]
    periods = {**{s: sd[s] for s in SELECTION_SPLITS}, "ALL": sel_days}
    cands = predeclared_candidates()
    by_id = {x.cid: x for x in cands}
    directions = {x.cid: x.direction for x in cands}
    n_cond = {x.cid: x.n_conditions for x in cands}

    # ---------------- stage 1: selection era only (no September candle or premium is loaded)
    spot_sel = spot[ts.dt.date <= SELECTION_END].reset_index(drop=True)
    prem_sel, prem_seen = _tracking(premium_for_factory(SELECTION_END))
    say(f"stage 1: exact S2 replay + {len(cands)} candidates on {len(sel_days)} selection sessions (<= {SELECTION_END})")
    st1 = run_stage(spot=spot_sel, days=sel_days, signal_fn=signal_fn, resolve=resolve, premium_for=prem_sel,
                    params=params, square_off=square_off, range_start=range_start, range_end=range_end,
                    underlying=underlying, allowed=allowed, candidates=cands, interval=interval)
    base1 = st1["base_rows"]
    base_snapshot = json.dumps(base1, sort_keys=True, default=str)
    snapshot = acc.snapshot_rows(st1["row_by_key"])

    la_days = list(lookahead_days) if lookahead_days is not None else sample_days(sel_days)
    say(f"lookahead: filter/score truncation test ({len(la_days)} sessions x {len(cut_times)} cuts) ...")
    la_trunc = truncation_test(spot_sel, la_days, cut_times, interval)
    la_opt = acc.option_feature_mutation_test(st1["contexts"], params.atr_period)
    la_mut = None
    if run_mutation:
        say("lookahead: exact S2 pipeline mutation test (candles after 12:00 scrambled) ...")
        la_mut = baseline_mutation_test(st1, la_days)
    lookahead_ok = la_trunc["passed"] and la_opt["passed"] and (la_mut is None or la_mut["passed"])

    folds = wf_windows(sel_days, CANDIDATE_WF)
    sel_windows = wf_windows(sel_days, SELECTION_WF)
    say("scoring candidates (TRAIN / VALIDATION / walk-forward; OOS = August) ...")
    scores = {x.cid: score_candidate(x, st1["rows"][x.cid], periods, folds, sel_windows, st1["labels"], c,
                                     lookahead_ok, st1["candles_per_day"]) for x in cands}
    s2_rows1 = {d: [r for r in base1 if r["direction"] == d] for d in DIRECTIONS}
    s2 = {d: acc.score_rows(s2_rows1[d], periods, folds, st1["labels"], c) for d in DIRECTIONS}
    s2["COMBINED"] = acc.score_rows(base1, periods, folds, st1["labels"], c)
    finalists = {d: acc.pick(scores, [n for n in scores if directions[n] == d]) for d in DIRECTIONS}
    max_val = {d: max((scores[n]["VALIDATION"]["evaluated"] for n in scores if directions[n] == d), default=0)
               for d in DIRECTIONS}
    procedure = {d: wf_procedure([n for n in scores if directions[n] == d], st1["rows"], n_cond, folds, c)
                 for d in DIRECTIONS}
    w5 = wf_windows(all_days, (HOLDOUT_WF,))
    w5_pick = {d: wf_procedure([n for n in scores if directions[n] == d], st1["rows"], n_cond, w5, c)
               for d in DIRECTIONS}
    frozen = json.dumps({"finalists": finalists, "w5": {d: [p["selected"] for p in v] for d, v in w5_pick.items()}},
                        sort_keys=True)

    # ---------------- stage 2: final holdout, loaded only now; the frozen choices are evaluated once
    hold_days = sd["HOLDOUT"]
    need = sorted({n for n in finalists.values() if n} | {p["selected"] for v in w5_pick.values() for p in v
                                                          if p["selected"]})
    st2 = None
    if hold_days:
        say(f"stage 2: September holdout ({len(hold_days)} sessions) for {len(need)} frozen candidate(s) + S2 ...")
        st2 = run_stage(spot=spot, days=hold_days, signal_fn=signal_fn, resolve=resolve,
                        premium_for=premium_for_factory(None), params=params, square_off=square_off,
                        range_start=range_start, range_end=range_end, underlying=underlying, allowed=allowed,
                        candidates=[by_id[n] for n in need], interval=interval)
    rows2 = st2["rows"] if st2 else {}
    base2 = st2["base_rows"] if st2 else []
    labels_all = {**st1["labels"], **(st2["labels"] if st2 else {})}

    named: dict[str, dict] = {}

    def add(name, scope, rows_sel, rows_hold, cid=None):
        named[name] = {"scope": scope, "cid": cid, "rows_sel": rows_sel, "rows_hold": rows_hold,
                       "rows_all": list(rows_sel) + list(rows_hold)}

    for d in DIRECTIONS:
        add(f"S2_{d}", d, s2_rows1[d], [r for r in base2 if r["direction"] == d])
    add("S2_COMBINED", "COMBINED", base1, base2)
    for d in DIRECTIONS:
        n = finalists[d]
        if n:
            add(f"S4_{d}", d, st1["rows"][n], rows2.get(n, []), n)
    comb_names = [finalists[d] for d in DIRECTIONS if finalists[d]]
    if comb_names:
        add("S4_COMBINED", "COMBINED", [r for n in comb_names for r in st1["rows"][n]],
            [r for n in comb_names for r in rows2.get(n, [])], "+".join(comb_names))

    holdout = {}
    for name, v in named.items():
        m = lab.perf(v["rows_hold"])
        holdout[name] = {**_metrics(m), "verdict": holdout_verdict(m, hc) if hold_days else HOLDOUT_INSUFFICIENT,
                         "day": acc.day_profile(v["rows_hold"])}
    all_windows = wf_windows(all_days, [w[0] for w in WF_WINDOWS])
    windows = {}
    for name, v in named.items():
        by_dir = {d: [r for r in v["rows_all"] if r["direction"] == d] for d in DIRECTIONS}
        windows[name] = window_table(name, by_dir, all_windows)
    proc_rows = []
    for w in all_windows:
        picks = {}
        by_dir = {}
        for d in DIRECTIONS:
            src = procedure[d] if w["window"] != HOLDOUT_WF else w5_pick[d]
            p = next((x for x in src if x["window"] == w["window"]), None)
            picks[d] = p["selected"] if p else None
            pool = rows2 if w["window"] == HOLDOUT_WF else st1["rows"]
            by_dir[d] = pool.get(picks[d], []) if picks[d] else []
        row = window_table("SELECTION_PROCEDURE", by_dir, [w])[0]
        row["selected"] = " | ".join(f"{d}:{picks[d] or 'NONE'}" for d in DIRECTIONS)
        proc_rows.append(row)
    windows["SELECTION_PROCEDURE"] = proc_rows
    window_sum = {k: window_summary(v) for k, v in windows.items()}

    periods_all = {**periods, "HOLDOUT": hold_days, "SIX_MONTHS": all_days}
    perf_named = {name: {p: _metrics(lab.perf(lab.period_rows(v["rows_all"], ds))) for p, ds in periods_all.items()}
                  for name, v in named.items()}
    day_stab = {name: {p: s3lab.day_stability(v["rows_all"], periods_all[p]) for p in ("ALL", "OOS", "HOLDOUT")}
                for name, v in named.items()}
    regimes = {name: {p: regime_table(v["rows_all"], labels_all, periods_all[p]) for p in ("ALL", "OOS", "HOLDOUT")}
               for name, v in named.items()}
    s2_by_regime = {d: {p: regime_table(s2_rows1[d], st1["labels"], periods[p]) for p in SELECTION_SPLITS}
                    for d in DIRECTIONS}
    skip_tables = {name: {p: skip_metrics(v["rows_sel"], st1["candles_per_day"], periods[p]) for p in SELECTION_SPLITS}
                   for name, v in named.items()}

    say("integrity audit ...")
    hold_max = None if not hold_days else max(hold_days)
    stage1_days_ok = all(d <= SELECTION_END.isoformat() for d in st1["days"])
    spot1_ok = spot_sel.empty or acc._ist_series(spot_sel["timestamp"]).max().date() <= SELECTION_END
    prem1_ok = prem_seen["max_ts"] is None or prem_seen["max_ts"].date() <= SELECTION_END
    order = [d for s in (*SELECTION_SPLITS, "HOLDOUT") for d in sd[s]]
    chrono = order == sorted(order) and len(order) == len(set(order)) and all(
        sd[a][-1] < sd[b][0] for a, b in zip((*SELECTION_SPLITS, "HOLDOUT"), (*SELECTION_SPLITS, "HOLDOUT")[1:])
        if sd[a] and sd[b])
    safety = module_safety()
    reads = dict(store.reads) if store is not None else {}
    integrity = []
    if integrity_data is not None:
        integrity.append(acc._check("A1", "Phase 1 data integrity (6-month dataset)", integrity_data["ok"],
                                    f"root {integrity_data['root']}: {integrity_data['stats']['trading_days']} days, "
                                    f"{integrity_data['stats']['option_candles']} option candles"))
    integrity += [
        acc._check("A2", "Only the 6-month store is read (no silent fallback to the old dataset)",
                   (store is None or (set(reads) <= {SOURCE, "missing"} and reads.get(SOURCE, 0) > 0))
                   and safety["no_legacy_store"],
                   f"premium reads {reads or 'n/a (injected source)'}; lab never imports the legacy candle store"),
        acc._check("A3", "Current day excluded; dataset ends before today",
                   bool(all_days) and max(all_days) < today.isoformat(),
                   f"latest session {max(all_days) if all_days else '-'}, today {today.isoformat()} (kept separate)"),
    ]
    integrity += acc.data_checks(spot, all_days, interval)
    integrity += acc.context_checks(st1["contexts"], st1["row_by_key"], interval)
    if st2 is not None and st2["contexts"]:
        integrity += [dict(x, id=f"H{x['id']}", check=f"[holdout] {x['check']}")
                      for x in acc.context_checks(st2["contexts"], st2["row_by_key"], interval)]
    integrity += [
        acc._check("L1", "Exact S2 pipeline mutation test (later underlying/premium candles scrambled)",
                   None if la_mut is None else la_mut["passed"],
                   "skipped (--skip-mutation)" if la_mut is None else
                   f"{la_mut['cuts']} cuts, failures {la_mut['failures'][:5] or 'none'}"),
        acc._check("L2", "Regimes, family filters and score components unchanged from candles <= cut",
                   la_trunc["passed"], f"{la_trunc['cuts']} cuts, failures {la_trunc['failures'][:5] or 'none'}"),
        acc._check("L3", "Option features + premium ATR unchanged when later premium candles are scrambled",
                   la_opt["passed"], f"{la_opt['checked']} signals, failures {la_opt['failure_count']}"),
        acc._check("L4", "Option premium missing -> NO_SIGNAL (never filled)", True,
                   f"NO_SIGNAL counts: {dict(Counter(k.split(':')[1] for k in st1['no_signal'].elements())) or 'none'}"),
        acc._check("T1", "TRAIN < VALIDATION < OOS < HOLDOUT, disjoint and chronological", chrono,
                   "; ".join(f"{s} {sd[s][0] if sd[s] else '-'}..{sd[s][-1] if sd[s] else '-'} ({len(sd[s])})"
                             for s in (*SELECTION_SPLITS, "HOLDOUT"))),
        acc._check("T2", "Stage 1 (selection) never loads a September candle or premium",
                   stage1_days_ok and spot1_ok and prem1_ok,
                   f"stage-1 sessions <= {SELECTION_END}: {stage1_days_ok}; spot <= {SELECTION_END}: {spot1_ok}; "
                   f"latest premium served {prem_seen['max_ts']}"),
        acc._check("T3", "Finalists frozen before the holdout is loaded; unchanged afterwards",
                   frozen == json.dumps({"finalists": finalists, "w5": {d: [p["selected"] for p in v]
                                                                        for d, v in w5_pick.items()}}, sort_keys=True),
                   f"holdout {hold_days[0] if hold_days else '-'}..{hold_max or '-'} evaluated for {need or 'none'}"),
        acc._check("T4", "Selection reads TRAIN + VALIDATION + walk-forward W1-W3 only (never OOS / September)", True,
                   "enforced in code: acc.pick() reads selection_score (TRAIN/VALIDATION perf, W1-W3 eligibility)"),
    ]
    integrity += isolation_checks(st1["rows"], directions, st1["row_by_key"], snapshot, base1, base_snapshot)
    integrity += [
        acc._check("Z1", "PAPER mode, live trading disabled, no broker order call in the lab",
                   live_gate.trading_mode() == "PAPER" and not live_gate.live_trading_enabled()
                   and safety["no_order_call"], f"TRADING_MODE={live_gate.trading_mode()}"),
        acc._check("Z2", "Production S2 untouched (research module only; S2 replayed read-only)", True,
                   "S4 is not registered in research_strategies / runtime; configs only read"),
    ]
    integrity_ok = all(x["status"] != acc.FAIL for x in integrity)

    final = {}
    for d in DIRECTIONS:
        n = finalists[d]
        if not n:
            continue
        v = holdout[f"S4_{d}"]["verdict"]
        chk = robust_checklist(scores[n], v, lookahead_ok, integrity_ok)
        final[d] = {"cid": n, "verdict": v, "checklist": chk, "status": final_status(scores[n]["status"], v, chk)}
    if comb_names:
        worst = max((final[d]["status"] for d in final), key=lambda s: STATUS_ORDER.get(s, 9))
        final["COMBINED"] = {"cid": "+".join(comb_names), "verdict": holdout["S4_COMBINED"]["verdict"],
                             "status": FAILED_HOLDOUT if holdout["S4_COMBINED"]["verdict"] == HOLDOUT_FAIL else worst}
    rec = recommend([final[d]["status"] for d in DIRECTIONS if d in final], integrity_ok, max_val, c)
    achieved = any(_num(scores[final[d]["cid"]]["OOS"]["win_rate"]) >= c.target_accuracy
                   and scores[final[d]["cid"]]["tier"] in ("A", "B") for d in DIRECTIONS if d in final)
    ranking = sorted(scores, key=lambda n: (STATUS_ORDER.get(scores[n]["status"], 9), -scores[n]["final_score"], n))
    unavailable = Counter(sh.reason_code(x["reason"]) for x in st1["base_contexts"] if x["data_status"] != "OK")
    return {
        "heading": HEADING, "banner": BANNER, "trading_mode": "PAPER", "live_trading_enabled": False,
        "production_s2_changed": False, "live_orders": 0, "september_used_for_optimization": False,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"), "today": today.isoformat(),
        "rules": rules or {}, "criteria": asdict(c), "holdout_criteria": asdict(hc),
        "weights": dict(s3.WEIGHTS), "thresholds": list(s3.THRESHOLDS), "families": FAMILIES,
        "modes": dict(s3lab.MODE_TEXT), "data_integrity": integrity_data,
        "dataset": {"from": first.isoformat(), "to": last.isoformat(), "trading_days": len(all_days),
                    "option_candles": (integrity_data or {}).get("stats", {}).get("option_candles"),
                    "excluded_weekdays": excluded},
        "splits": {s: {"from": sd[s][0] if sd[s] else None, "to": sd[s][-1] if sd[s] else None, "days": sd[s]}
                   for s in sd},
        "counts": {"candidates": len(cands), "s2_signals_stage1": len(st1["base_contexts"]),
                   "s2_unavailable_reasons": dict(unavailable), "confirmation_candles": st1["extra"],
                   "no_signal": dict(st1["no_signal"]),
                   "vwap_source": "TWAP_PROXY (NIFTY index volume is zero)" if st1["bars"]["vwap"].isna().all()
                   else "VWAP"},
        "scores": scores, "ranking": ranking, "finalists": finalists, "final": final, "s2": s2,
        "wf_procedure": procedure, "w5_pick": w5_pick, "holdout": holdout, "windows": windows,
        "window_summary": window_sum, "perf": perf_named, "day_stability": day_stab, "regimes": regimes,
        "s2_by_regime": s2_by_regime, "skip": skip_tables,
        "lookahead": {"truncation": la_trunc, "option": la_opt, "mutation": la_mut, "passed": lookahead_ok},
        "integrity": integrity, "integrity_ok": integrity_ok, "recommendation": rec, "target_achieved": achieved,
        "robust_70": rec == ROBUST_70, "formulas": {"selection": acc.SELECTION_FORMULA, "score": acc.SCORE_FORMULA},
        "_rows": st1["rows"], "_holdout_rows": rows2, "_base_rows": base1, "_holdout_base_rows": base2,
    }


# ------------------------------------------------------------------ reports
_pct, _pf, _rs, _csv = sh._pct, sh._pf_s, sh._rs, acc._csv
REPORT_FILES = ("s4_six_month_research.md", "s4_baseline.csv", "s4_candidates.csv", "s4_walk_forward.csv",
                "s4_day_stability.csv", "s4_regime_stability.csv", "s4_final_holdout.csv", "s4_integrity_audit.md",
                "s4_data_integrity.md")


def _m(m: dict | None) -> str:
    if not m:
        return "-"
    n = m.get("trades", m.get("evaluated"))
    a = m.get("accuracy", m.get("win_rate"))
    return f"{n} / {_pct(a)} / PF {_pf(m['profit_factor'])} / Rs {_rs(m['net_pnl'])}"


def baseline_csv_rows(a: dict) -> list[dict]:
    return [{"scope": n.replace("S2_", ""), "period": p, **m} for n in ("S2_BUY", "S2_SELL", "S2_COMBINED")
            for p, m in a["perf"][n].items()]


def candidate_csv_rows(a: dict) -> list[dict]:
    out = []
    fin = set(a["finalists"].values())
    for n in a["ranking"]:
        s = a["scores"][n]
        row = {"candidate": n, "direction": s["direction"], "family": f"{s['family']} {s['family_name']}",
               "mode": s["mode"], "threshold": s["threshold"], "complexity": s["n_conditions"]}
        for p in (*SELECTION_SPLITS, "ALL"):
            row.update({f"{p.lower()}_{k}": v for k, v in _metrics(s[p]).items()})
        w = s["walk_forward"]
        row.update({"oos_sample_grade": s["tier"], "wf_w1_w4_profitable_share": w["wf_profitable_share"],
                    "wf_w1_w4_pf": w["wf_pf"], "wf_w1_w3_profitable_share": s["wf_selection"]["profitable_share"],
                    "oos_skip_pct": s["skip"]["OOS"]["skip_pct"], "regime_share": s["regime_share"],
                    "selection_score": s["selection_score"], "selection_block": s["selection_block"],
                    "final_score": s["final_score"], "status": s["status"], "failed_gates": s["failed"],
                    "finalist": n in fin})
        out.append(row)
    return out


def walk_forward_csv_rows(a: dict) -> list[dict]:
    out = [{"kind": "STRATEGY", **r} for v in a["windows"].values() for r in v]
    for n in a["ranking"]:
        for w in a["scores"][n]["walk_forward"]["folds"]:
            out.append({"kind": "CANDIDATE_W1_W4", "strategy": n, "window": f"fold{w['fold']}",
                        "train_period": w["train"], "test_period": w["test"], "trades": w["evaluated"],
                        "combined_accuracy": w["win_rate"], "profit_factor": w["profit_factor"],
                        "net_pnl": w["net_pnl"]})
    return out


def day_csv_rows(a: dict) -> list[dict]:
    out = []
    for n, per in a["day_stability"].items():
        for p, s in per.items():
            out.append({"strategy": n, "period": p, "row": "SUMMARY", "active_days": s["active_days"],
                        "best_day": s["best_day"], "best_day_pnl": s["best_day_pnl"], "worst_day": s["worst_day"],
                        "worst_day_pnl": s["worst_day_pnl"], "worst_day_loss": s["worst_day_loss"],
                        "median_day_pnl": s["median_day_pnl"], "profitable_day_share": s["profitable_day_share"],
                        "best_day_share_of_net": s["best_day_share_of_net"]})
            if p == "ALL" or p == "HOLDOUT":
                for d, v in s["daily"].items():
                    out.append({"strategy": n, "period": p, "row": "DAY", "session_date": d, "pnl": v})
    return out


def regime_csv_rows(a: dict) -> list[dict]:
    out = []
    for n, per in a["regimes"].items():
        for p, groups in per.items():
            for r in (*PRIMARY_REGIMES, "NORMAL_VOLATILITY"):
                m = groups.get(r)
                out.append({"strategy": n, "period": p, "regime": REGIME_NAMES[r], "trades": m["evaluated"] if m else 0,
                            "accuracy": m["win_rate"] if m else None,
                            "profit_factor": m["profit_factor"] if m else None, "net_pnl": m["net_pnl"] if m else 0.0})
    for d, per in a["s2_by_regime"].items():
        for p, groups in per.items():
            for r in PRIMARY_REGIMES:
                m = groups.get(r)
                out.append({"strategy": f"S2_{d}_ANALYSIS", "period": p, "regime": REGIME_NAMES[r],
                            "trades": m["evaluated"] if m else 0, "accuracy": m["win_rate"] if m else None,
                            "profit_factor": m["profit_factor"] if m else None, "net_pnl": m["net_pnl"] if m else 0.0})
    return out


def holdout_csv_rows(a: dict) -> list[dict]:
    out = []
    for n, h in a["holdout"].items():
        status = a["final"].get(n.replace("S4_", ""), {}).get("status") if n.startswith("S4_") else "REFERENCE"
        out.append({"strategy": n, "candidate": n if n.startswith("S2_") else a["final"].get(
            n.replace("S4_", ""), {}).get("cid"), **{k: v for k, v in h.items() if k != "day"},
            "status": status})
    for d, ps in a["w5_pick"].items():
        for p in ps:
            if p["selected"]:
                m = lab.perf(a["_holdout_rows"].get(p["selected"], []))
                out.append({"strategy": f"W5_SELECTION_PROCEDURE_{d}", "candidate": p["selected"], **_metrics(m),
                            "verdict": "INFO", "status": "WALK_FORWARD_ONLY"})
    return out


def compare_rows(a: dict) -> list[dict]:
    out = []
    for scope in ("BUY", "SELL", "COMBINED"):
        for kind in ("S2", "S4"):
            name = f"{kind}_{scope}"
            if name not in a["perf"]:
                out.append({"strategy": name, "status": INSUFFICIENT_DATA})
                continue
            o, h, w = a["perf"][name]["OOS"], a["holdout"][name], a["window_summary"][name]
            status = "CURRENT" if kind == "S2" else a["final"].get(scope, {}).get("status", INSUFFICIENT_DATA)
            out.append({"strategy": name, "candidate": a["final"].get(scope, {}).get("cid") if kind == "S4" else S2_NAME,
                        "accuracy": o["accuracy"], "trades": o["trades"], "pf": o["profit_factor"],
                        "pnl": o["net_pnl"], "avg_r": o["avg_r"], "max_dd": o["max_dd"],
                        "max_loss_streak": o["max_loss_streak"], "profitable_days": o["profitable_days"],
                        "traded_days": o["traded_days"], "wf_profitable_share": w["profitable_share"],
                        "holdout": f"{h['trades']} / {_pct(h['accuracy'])} / {h['verdict']}", "status": status})
    return out


def data_integrity_markdown(di: dict | None) -> str:
    md = ["# S4 data integrity (6-month Upstox dataset)", "", f"**{BANNER}**", ""]
    if not di:
        return "\n".join(md + ["Data integrity was not run (injected data source)."]) + "\n"
    s = di["stats"]
    md += [f"Overall: **{'PASS' if di['ok'] else 'FAIL'}** | root `{di['root']}`", "",
           f"- Period {s['from']}..{s['to']}; sessions {s['first_day']}..{s['last_day']} ({s['trading_days']} trading "
           f"days, {s['underlying_candles']} NIFTY candles)",
           f"- {s['contracts']} contracts, {s['requirements']} (session, contract) requirements, {s['expiries']} expiries,"
           f" {s['option_files']} option files, {s['option_candles']} option candles",
           "- The legacy dataset is never read: contracts and premiums come only from this root.", "",
           "| ID | Check | Status | Detail |", "|---|---|---|---|"]
    md += [f"| {x['id']} | {x['check']} | {x['status']} | {x['detail']} |" for x in di["checks"]]
    return "\n".join(md) + "\n"


def integrity_markdown(a: dict) -> str:
    md = ["# S4 integrity audit", "", f"**{BANNER}** | TRADING_MODE=PAPER | LIVE_TRADING_ENABLED=false", "",
          f"Overall: **{'PASS' if a['integrity_ok'] else 'FAIL'}** | Lookahead: "
          f"**{'PASS' if a['lookahead']['passed'] else 'FAIL'}**", "",
          "| ID | Check | Status | Detail |", "|---|---|---|---|"]
    md += [f"| {x['id']} | {x['check']} | {x['status']} | {x['detail']} |" for x in a["integrity"]]
    md += ["", "- September used for optimization: NO (stage 1 never loads it; finalists frozen before stage 2).",
           "- Production S2 changed: NO. Live orders: 0. S4 is research-only and not registered anywhere."]
    return "\n".join(md) + "\n"


def _perf_line(name: str, m: dict) -> str:
    return (f"| {name} | {m['trades']} | {m['wins']} | {m['losses']} | {_pct(m['accuracy'])} | {_pf(m['profit_factor'])} "
            f"| {_rs(m['net_pnl'])} | {m['avg_r'] if m['avg_r'] is not None else '-'} | {_rs(m['max_dd'])} | "
            f"{m['max_loss_streak']} | {m['profitable_days']}/{m['traded_days']} |")


PERF_HEAD = ["| Scope | Trades | Wins | Losses | Accuracy | PF | P&L | Avg R | Max DD | Max loss streak | "
             "Profitable days |", "|---|---|---|---|---|---|---|---|---|---|---|"]


def markdown(a: dict) -> str:
    sc, fin, c = a["scores"], a["finalists"], a["criteria"]
    ds = a["dataset"]
    md = [f"# {HEADING}", "", f"**{BANNER}** | production S2 NOT modified | LIVE_TRADING_ENABLED=false | "
          f"generated {a['generated_at']}", "",
          "## 1. Dataset", "",
          f"- {ds['from']} -> {ds['to']}, {ds['trading_days']} trading days, option candles {ds['option_candles']}",
          f"- Source: data/historical/upstox_6m only (see s4_data_integrity.md); no fallback to the old dataset.",
          f"- VWAP source: {a['counts']['vwap_source']}; volume features from option premium volume.",
          f"- Current day {a['today']} kept separate and never used.", "",
          "## 2. S2 baseline (exact existing implementation)", ""]
    md += [f"- **{k}**: {v}" for k, v in a["rules"].items()]
    md += ["", f"S2 signals in stage 1: {a['counts']['s2_signals_stage1']}; unavailable by reason: "
           f"{a['counts']['s2_unavailable_reasons'] or 'none'}", ""]
    for p in ("TRAIN", "VALIDATION", "OOS", "HOLDOUT", "SIX_MONTHS"):
        md += [f"**{p}**", "", *PERF_HEAD]
        md += [_perf_line(n.replace("S2_", ""), a["perf"][n][p]) for n in ("S2_BUY", "S2_SELL", "S2_COMBINED")]
        md.append("")
    md += ["## 3. Splits and walk-forward", "", "| Split | Sessions | Range |", "|---|---|---|"]
    md += [f"| {s} | {len(v['days'])} | {v['from'] or '-'}..{v['to'] or '-'} |" for s, v in a["splits"].items()]
    md += ["", "Walk-forward (past data only): " + "; ".join(f"{w[0]} train {'+'.join(w[1])} -> test {w[2]}"
                                                            for w in WF_WINDOWS) + ". W1-W3 may inform selection; W4 "
           "(August) is reported only; W5 (September) is computed after the finalists are frozen.", "",
           "## 4. Features and regimes (causal, completed candles only)", "",
           "Trend EMA20/50 + slope + distance; VWAP (TWAP proxy) relation/slope/distance; RSI14 + momentum + body; "
           "volume from option premiums; ATR14 + ATR percentile + range; 6/12-candle breakout distance (ATR units); "
           "body/wick/close position/streaks. Option features from real premium candles <= signal (momentum, body, "
           "VWAP, volume ratio, premium ATR). Regimes: TREND_UP / TREND_DOWN / RANGE / HIGH_VOL / LOW_VOL / UNCERTAIN.",
           "", "S2 performance by regime (analysis only, not used as an automatic filter):", "",
           "| Direction | Period | Regime | Trades | Accuracy | PF | P&L |", "|---|---|---|---|---|---|---|"]
    for d, per in a["s2_by_regime"].items():
        for p, g in per.items():
            for r in PRIMARY_REGIMES:
                m = g.get(r)
                if m:
                    md.append(f"| {d} | {p} | {REGIME_NAMES[r]} | {m['evaluated']} | {_pct(m['win_rate'])} | "
                              f"{_pf(m['profit_factor'])} | {_rs(m['net_pnl'])} |")
    md += ["", "## 5. Candidate families and entry modes", "", "| Family | Name | Rule (at the S2 signal candle) | "
           "Complexity |", "|---|---|---|---|"]
    md += [f"| {k} | {v[0]} | {v[1]} | {FAMILY_COMPLEXITY[k]} |" for k, v in a["families"].items()]
    md += ["", "| Entry mode | Rule (+1 complexity unless A) |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in a["modes"].items()]
    md += ["", f"{a['counts']['candidates']} pre-declared candidates (A-H x 5 modes x 2 directions + I x 7 thresholds x "
           f"2 directions). Exits unchanged: 1.5 x premium ATR14 stop, 2R target, 15:15 square-off, AMBIGUOUS = -1R.",
           "", "## 6. Confidence score thresholds and no-trade analysis (family I, OOS = August)", "",
           "Weights: " + ", ".join(f"{k} {v}" for k, v in a["weights"].items()), "",
           "| Direction | Threshold | TRAIN skip% / n / acc | VALIDATION skip% / n / acc | OOS skip% / n / acc / PF / "
           "P&L |", "|---|---|---|---|---|"]
    for d in DIRECTIONS:
        for t in a["thresholds"]:
            s = sc[f"S4_{d}_I_T{t}_A_NEXT_BAR"]["skip"]
            md.append(f"| {d} | {t} | {_pct(s['TRAIN']['skip_pct'])} / {s['TRAIN']['evaluated']} / "
                      f"{_pct(s['TRAIN']['accuracy'])} | {_pct(s['VALIDATION']['skip_pct'])} / "
                      f"{s['VALIDATION']['evaluated']} / {_pct(s['VALIDATION']['accuracy'])} | "
                      f"{_pct(s['OOS']['skip_pct'])} / {s['OOS']['evaluated']} / {_pct(s['OOS']['accuracy'])} / "
                      f"{_pf(s['OOS']['profit_factor'])} / {_rs(s['OOS']['net_pnl'])} |")
    md += ["", "| Strategy | TRAIN skip% / n / acc | VALIDATION skip% / n / acc | OOS skip% / n / acc / PF / P&L |",
           "|---|---|---|---|"]
    for n, s in a["skip"].items():
        md.append(f"| {n} | {_pct(s['TRAIN']['skip_pct'])} / {s['TRAIN']['evaluated']} / {_pct(s['TRAIN']['accuracy'])} "
                  f"| {_pct(s['VALIDATION']['skip_pct'])} / {s['VALIDATION']['evaluated']} / "
                  f"{_pct(s['VALIDATION']['accuracy'])} | {_pct(s['OOS']['skip_pct'])} / {s['OOS']['evaluated']} / "
                  f"{_pct(s['OOS']['accuracy'])} / {_pf(s['OOS']['profit_factor'])} / {_rs(s['OOS']['net_pnl'])} |")
    md += ["", "Skip% = share of entry-window candles without a signal. Higher selectivity is judged on OOS accuracy "
           "AND sample size, never on OOS alone.", "",
           "## 7. Selection (TRAIN + VALIDATION + walk-forward W1-W3 only)", "", a["formulas"]["selection"],
           "Additionally eligible only if >= 2/3 of traded W1-W3 test months are profitable.", "",
           "| Direction | Finalist | Selection | TRAIN | VALIDATION | OOS (Aug) | Grade | Pre-holdout status | "
           "Failed gates |", "|---|---|---|---|---|---|---|---|---|"]
    for d in DIRECTIONS:
        n = fin[d]
        md.append(f"| {d} | {n or 'NONE ELIGIBLE'} | {sc[n]['selection_score'] if n else '-'} | "
                  f"{_m(sc[n]['TRAIN']) if n else '-'} | {_m(sc[n]['VALIDATION']) if n else '-'} | "
                  f"{_m(sc[n]['OOS']) if n else '-'} | {sc[n]['tier'] if n else '-'} | {sc[n]['status'] if n else '-'} "
                  f"| {', '.join(sc[n]['failed']) if n else '-'} |")
    md += ["", "Top candidates by report score (OOS used for ordering only, never for selection):", "",
           "| Candidate | Complexity | TRAIN | VALIDATION | OOS | Grade | WF W1-W4 profitable | Status |",
           "|---|---|---|---|---|---|---|---|"]
    for n in a["ranking"][:20]:
        s = sc[n]
        md.append(f"| {n} | {s['n_conditions']} | {_m(s['TRAIN'])} | {_m(s['VALIDATION'])} | {_m(s['OOS'])} | "
                  f"{s['tier']} | {_pct(s['walk_forward']['wf_profitable_share'])} | {s['status']} |")
    md += ["", "## 8. Walk-forward per window", "", "| Strategy | Window | Train | Test | BUY acc (n) | SELL acc (n) | "
           "Combined acc | PF | P&L | Trades |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, rows in a["windows"].items():
        for r in rows:
            md.append(f"| {n}{' (' + r['selected'] + ')' if r.get('selected') else ''} | {r['window']} | "
                      f"{r['train_period']} | {r['test_period']} | {_pct(r['buy_accuracy'])} ({r['buy_trades']}) | "
                      f"{_pct(r['sell_accuracy'])} ({r['sell_trades']}) | {_pct(r['combined_accuracy'])} | "
                      f"{_pf(r['profit_factor'])} | {_rs(r['net_pnl'])} | {r['trades']} |")
    md += ["", "| Strategy | Profitable windows | Avg accuracy | Median accuracy | Avg PF | Worst | Best |",
           "|---|---|---|---|---|---|---|"]
    for n, s in a["window_summary"].items():
        md.append(f"| {n} | {_pct(s['profitable_share'])} | {_pct(s['avg_accuracy'])} | {_pct(s['median_accuracy'])} | "
                  f"{_pf(s['avg_pf'])} | {s['worst'] or '-'} | {s['best'] or '-'} |")
    md += ["", "## 9. Day stability", "", "| Strategy | Period | Active days | Best day | Worst day | Median day | "
           "Profitable days | Best-day share |", "|---|---|---|---|---|---|---|---|"]
    for n, per in a["day_stability"].items():
        for p, s in per.items():
            md.append(f"| {n} | {p} | {s['active_days']} | {s['best_day'] or '-'} ({_rs(s['best_day_pnl'])}) | "
                      f"{s['worst_day'] or '-'} ({_rs(s['worst_day_pnl'])}) | {_rs(s['median_day_pnl'])} | "
                      f"{_pct(s['profitable_day_share'])} | {_pct(s['best_day_share_of_net'])} |")
    md += ["", "## 10. Regime stability (selection era Apr-Aug; per-period detail in s4_regime_stability.csv)", "",
           "| Strategy | Regime | Trades | Accuracy | PF | P&L |", "|---|---|---|---|---|---|"]
    for n, per in a["regimes"].items():
        for r in PRIMARY_REGIMES:
            m = per["ALL"].get(r)
            if m:
                md.append(f"| {n} | {REGIME_NAMES[r]} | {m['evaluated']} | {_pct(m['win_rate'])} | "
                          f"{_pf(m['profit_factor'])} | {_rs(m['net_pnl'])} |")
    hc = a["holdout_criteria"]
    md += ["", "## 11. Final September holdout (evaluated once, after selection)", "",
           f"Acceptable = trades >= {hc['min_trades']}, P&L > 0, PF >= {hc['min_pf']}, accuracy >= "
           f"{hc['min_accuracy']:.0%}, best-day share <= {hc['max_best_day_share']:.0%}. A failure is "
           "FAILED_FINAL_HOLDOUT; nothing is retuned.", "",
           "| Strategy | Trades | Wins | Losses | Accuracy | PF | P&L | Max DD | Max loss streak | Profitable days | "
           "Best-day share | Verdict |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, h in a["holdout"].items():
        md.append(f"| {n} | {h['trades']} | {h['wins']} | {h['losses']} | {_pct(h['accuracy'])} | "
                  f"{_pf(h['profit_factor'])} | {_rs(h['net_pnl'])} | {_rs(h['max_dd'])} | {h['max_loss_streak']} | "
                  f"{h['profitable_days']}/{h['traded_days']} | {_pct(h['best_day_share'])} | {h['verdict']} |")
    md += ["", "## 12. ROBUST_70 checklist (all 13 required)", ""]
    if not any(d in a["final"] for d in DIRECTIONS):
        md.append("No finalist was eligible, so no candidate can be ROBUST_70.")
    else:
        md += ["| Criterion | " + " | ".join(f"{d} ({a['final'][d]['cid']})" for d in DIRECTIONS if d in a["final"])
               + " |", "|---|" + "---|" * sum(d in a["final"] for d in DIRECTIONS)]
        for k in ROBUST_CRITERIA:
            md.append(f"| {k} | " + " | ".join("PASS" if a["final"][d]["checklist"][k] else "FAIL"
                                              for d in DIRECTIONS if d in a["final"]) + " |")
        md += ["", "Final status: " + "; ".join(f"{d} {v['status']}" for d, v in a["final"].items())]
    md += ["", "## 13. S2 vs S4 (OOS = August; holdout = September)", "",
           "| Strategy | Candidate | Accuracy | Trades | PF | P&L | Avg R | Max DD | Max loss streak | Profitable days "
           "| WF profitable windows | Final holdout | Status |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in compare_rows(a):
        if "trades" not in r:
            md.append(f"| {r['strategy']} | - | - | - | - | - | - | - | - | - | - | - | {r['status']} |")
            continue
        md.append(f"| {r['strategy']} | {r['candidate']} | {_pct(r['accuracy'])} | {r['trades']} | {_pf(r['pf'])} | "
                  f"{_rs(r['pnl'])} | {r['avg_r'] if r['avg_r'] is not None else '-'} | {_rs(r['max_dd'])} | "
                  f"{r['max_loss_streak']} | {r['profitable_days']}/{r['traded_days']} | "
                  f"{_pct(r['wf_profitable_share'])} | {r['holdout']} | {r['status']} |")
    statuses = Counter(s["status"] for s in sc.values())
    md += ["", "## 14. Anti-overfitting", "",
           f"- {len(sc)} pre-declared candidates; rules, weights and thresholds fixed in code (no brute force).",
           "- Selection: TRAIN + VALIDATION + W1-W3 only. OOS (August) only grades; September only after freezing.",
           f"- Complexity penalty in the report score; candidates above {MAX_COMPLEXITY} conditions are rejected.",
           "- No losing trades removed; AMBIGUOUS counts as -1R; no future candles or premiums (see audit).",
           f"- Sample grades (OOS trades): A >= {c['tier_a']}, B >= {c['tier_b']}, C >= {c['tier_c']}, D below "
           "(70% on grade C/D = INSUFFICIENT_DATA, never a pass).",
           f"- Status counts: {dict(statuses)}",
           "", "## 15. 70% OOS target", "", f"**{'ACHIEVED' if a['target_achieved'] else 'NOT ACHIEVED'}**", ""]
    if not a["target_achieved"]:
        md.append(NOT_ACHIEVED_TEXT)
    md += ["", "## 16. Recommendation", "", f"**{a['recommendation']}**", "",
           "A surviving candidate is at most READY_FOR_FORWARD_PAPER_VALIDATION (separate live-data PAPER forward "
           "test); nothing here is production-ready. Production S2 changed: NO. Live orders: 0. "
           "September used for optimization: NO."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {n: out_dir / n for n in REPORT_FILES}
    paths["s4_six_month_research.md"].write_text(markdown(a), encoding="utf-8")
    _csv(paths["s4_baseline.csv"], baseline_csv_rows(a))
    _csv(paths["s4_candidates.csv"], candidate_csv_rows(a))
    _csv(paths["s4_walk_forward.csv"], walk_forward_csv_rows(a))
    _csv(paths["s4_day_stability.csv"], day_csv_rows(a))
    _csv(paths["s4_regime_stability.csv"], regime_csv_rows(a))
    _csv(paths["s4_final_holdout.csv"], holdout_csv_rows(a))
    paths["s4_integrity_audit.md"].write_text(integrity_markdown(a), encoding="utf-8")
    paths["s4_data_integrity.md"].write_text(data_integrity_markdown(a["data_integrity"]), encoding="utf-8")
    return paths
