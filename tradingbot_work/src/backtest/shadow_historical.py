"""HISTORICAL SHADOW REPLAY of BASELINE_S2_BUY_SELL / SELL_ONLY / SELL_CONFIRMED (PAPER, read-only, no orders).

Reuses the tested research pipeline unchanged: S2 signals (``signal_fn``), contract resolution (``resolve``),
``build_contexts`` (entry = next option candle open, premium ATR from candles <= signal candle),
``dr.run_candidate`` (direction / confirmation filter) and ``imp.evaluate`` (stop 1.5 x premium ATR, target 2R,
first touch on later real premium candles, square-off). Only stored candles are read.

Point-in-time contract rule: the strategy trades the *nearest* expiry. ``PointInTimeResolver`` adds the official
expired expiries and their expired contracts (Upstox expired-instruments API, keys used verbatim) to the live
instrument master, so the unchanged ``resolve_signal_contract`` picks the contract that was actually nearest on the
signal date. ``strict_resolver`` additionally refuses any resolution when no earlier expiry can be ruled out (e.g.
the expired-expiry list is unavailable): such signals are DATA_UNAVAILABLE / expired_instrument_unavailable. No other
expiry is ever substituted.

Results are kept separate from the CURRENT-DAY SHADOW VALIDATION and are never merged with it.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.historical_data.models import HistoricalDataError
from src.historical_data.validation import expected_slots
from src.options.models import OptionResolutionError, OptionType
from src.paper_trading.shadow_validation import CANDIDATES

IST = ZoneInfo("Asia/Kolkata")
HEADING = "HISTORICAL SHADOW REPLAY"
CURRENT_HEADING = "CURRENT-DAY SHADOW VALIDATION"
EVALUATED, DATA_UNAVAILABLE, SKIPPED = "EVALUATED", "DATA_UNAVAILABLE", "SKIPPED"
STATES = (EVALUATED, DATA_UNAVAILABLE, SKIPPED)
DECIDED = (imp.WIN, imp.LOSS, imp.AMBIGUOUS, imp.SCRATCH)
LOT_SIZE = 65
EXPIRED_UNAVAILABLE = "EXPIRED_INSTRUMENT_UNAVAILABLE"
EXPIRED_CONTRACTS_UNAVAILABLE = "EXPIRED_CONTRACTS_UNAVAILABLE"
EXPIRED_CANDLES_EMPTY = "EXPIRED_CANDLES_EMPTY"
DTE_BUCKETS = (("0 DTE", 0, 0), ("1 DTE", 1, 1), ("2-4 DTE", 2, 4), ("5-7 DTE", 5, 7), ("8-14 DTE", 8, 14),
               ("15-21 DTE", 15, 21), ("22+ DTE", 22, None))
REGIME_LABELS = ("strong_bullish_trend", "strong_bearish_trend", "range_bound", "choppy_whipsaw")
DAILY_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.(json|csv|md)$")

REASON_CODES = {
    "NO_PREMIUM_DATA": "local_historical_file_missing",
    "MISSING_LOCAL_PARQUET": "local_historical_file_missing",
    "NO_PREMIUM_CANDLE_AT_SIGNAL": "option_candle_missing",
    "NO_PREMIUM_ENTRY_CANDLE": "option_candle_missing",
    "NO_EXIT_EVIDENCE": "option_candle_missing",
    "INSUFFICIENT_ATR_HISTORY": "insufficient_option_atr_history",
}


def reason_code(raw: Any) -> str:
    raw = str(raw or "UNKNOWN")
    if raw.startswith("CONTRACT_UNRESOLVED:"):
        sub = raw.split(":", 1)[1]
        if sub == EXPIRED_UNAVAILABLE:
            return "expired_instrument_unavailable"
        if sub.startswith(EXPIRED_CONTRACTS_UNAVAILABLE):
            return "historical_api_unavailable"
        return "option_contract_not_found"
    if raw == EXPIRED_CANDLES_EMPTY:
        return "option_candle_missing"
    if raw.startswith("PREMIUM_API_ERROR:"):
        api = raw.split(":", 1)[1]
        return "expired_instrument_unavailable" if api in ("PLAN_NOT_SUPPORTED", "NOT_FOUND") \
            else "historical_api_unavailable"
    return REASON_CODES.get(raw, raw.lower())


# ------------------------------------------------------------------ trading days
def complete_sessions(spot: pd.DataFrame, from_date: date, to_date: date,
                      interval: int = 5) -> tuple[list[date], dict[str, str]]:
    """Weekdays in range with every session candle stored; other weekdays are listed with the reason."""
    ts = pd.to_datetime(spot["timestamp"]) if not spot.empty else pd.Series([], dtype="datetime64[ns, Asia/Kolkata]")
    present: dict[date, set[int]] = {}
    for t in ts:
        present.setdefault(t.date(), set()).add(pd.Timestamp(t).value)
    sessions, excluded = [], {}
    d = from_date
    while d <= to_date:
        if d.weekday() < 5:
            need = {pd.Timestamp(s).value for s in expected_slots(d, interval)}
            have = present.get(d, set()) & need
            if have == need:
                sessions.append(d)
            elif have:
                excluded[d.isoformat()] = f"incomplete_session ({len(have)}/{len(need)} candles)"
            else:
                excluded[d.isoformat()] = "no_stored_candles (exchange holiday or local_historical_file_missing)"
        d += timedelta(days=1)
    return sessions, excluded


def select_days(sessions: Sequence[date], days: int | None = None) -> list[date]:
    ordered = sorted(sessions)
    return ordered if not days else ordered[-int(days):]


# ------------------------------------------------------------------ point-in-time expiry
@dataclass(frozen=True)
class ExpiryCheck:
    """``listing_date``: snapshot date of the instrument master (expiries before it are not listed).
    ``expired_expiries``: complete official expired-expiry list, or None when unavailable."""
    listing_date: date
    expired_expiries: frozenset | None
    source: str

    def verify(self, signal_date: date, expiry: date) -> bool:
        if signal_date >= self.listing_date:
            return True
        if self.expired_expiries is None:
            return False
        return not any(signal_date <= e < expiry for e in self.expired_expiries)


class PointInTimeResolver:
    """Resolver interface for ``resolve_signal_contract`` as of a historical date.

    Expiries = official expired expiries (Upstox expired-instruments API) plus live expiries from the instrument
    master; contracts of an expired expiry come verbatim from the expired option contract API. ``today`` is set
    to the signal date by ``resolve_signal_contract``, so only expiries on/after that date are selectable.
    """

    def __init__(self, live, expired_expiries: Iterable[date], expired_contracts: Callable[[date], list]) -> None:
        self.live = live
        self.expired = frozenset(expired_expiries)
        self._fetch = expired_contracts
        self._cache: dict[date, list] = {}
        self._errors: dict[date, str] = {}
        self.today = lambda: datetime.now(IST).date()

    def _expired_for(self, expiry: date) -> list:
        if expiry in self._errors:
            raise OptionResolutionError(self._errors[expiry], f"expired contracts for {expiry} unavailable")
        if expiry not in self._cache:
            try:
                self._cache[expiry] = list(self._fetch(expiry))
            except HistoricalDataError as exc:
                self._errors[expiry] = f"{EXPIRED_CONTRACTS_UNAVAILABLE}_{exc.code}"
                raise OptionResolutionError(self._errors[expiry], exc.message) from None
        return self._cache[expiry]

    def _live_expiries(self, underlying: str) -> list[date]:
        self.live.today = self.today
        try:
            return self.live.expiries(underlying)
        except OptionResolutionError:
            return []

    def expiries(self, underlying: str) -> list[date]:
        today = self.today()
        return sorted(e for e in self.expired | set(self._live_expiries(underlying)) if e >= today)

    def contracts(self, underlying: str, expiry: date, option_type=None) -> list:
        if expiry in self.expired:
            kind = None if option_type is None else OptionType(str(getattr(option_type, "value", option_type)))
            return [c for c in self._expired_for(expiry) if kind is None or c.option_type == kind]
        self.live.today = self.today
        return self.live.contracts(underlying, expiry, option_type)

    def strikes(self, underlying: str, expiry: date, option_type=None) -> list[float]:
        return sorted({float(c.strike) for c in self.contracts(underlying, expiry, option_type)})

    def resolve(self, underlying: str, expiry: date, strike: float, option_type) -> Any:
        if expiry not in self.expired:
            self.live.today = self.today
            return self.live.resolve(underlying, expiry, strike, option_type)
        for c in self.contracts(underlying, expiry, option_type):
            if abs(float(c.strike) - float(strike)) < 1e-6:
                return c
        raise OptionResolutionError("CONTRACT_NOT_FOUND", f"{underlying} {expiry} {strike:g} {option_type} not returned")


def strict_resolver(resolve: Callable, check: ExpiryCheck) -> Callable:
    def wrapped(sig):
        contract = resolve(sig)
        if not check.verify(imp._ist(sig.timestamp).date(), contract.expiry):
            raise OptionResolutionError(EXPIRED_UNAVAILABLE,
                                        f"nearest expiry on the signal date is not a listed contract ({check.source})")
        return contract
    return wrapped


# ------------------------------------------------------------------ replay
def dte_bucket(dte: int | None) -> str | None:
    if dte is None:
        return None
    for label, lo, hi in DTE_BUCKETS:
        if dte >= lo and (hi is None or dte <= hi):
            return label
    return None


def classify_row(row: dict, regimes: dict[date, set[str]] | None = None) -> dict:
    out = dict(row)
    out.pop("premium", None)
    if row["decision"] != imp.ACCEPTED:
        state, raw = SKIPPED, None
    elif row.get("outcome") in DECIDED:
        state, raw = EVALUATED, None
    else:
        state = DATA_UNAVAILABLE
        raw = row.get("data_reason") or row.get("exit_reason") or row.get("outcome")
    day = date.fromisoformat(str(row["session_date"])[:10])
    out.update(
        state=state, reason_raw=raw, reason=None if raw is None else reason_code(raw),
        pnl=round(imp._conservative(row)[0], 2) if state == EVALUATED else None,
        dte_bucket=dte_bucket(row.get("dte")),
        regimes=sorted((regimes or {}).get(day, ())),
        expiry=str(row["expiry"]) if row.get("expiry") else None,
    )
    if hasattr(out.get("contract"), "__dict__"):
        out["contract"] = str(out["contract"])
    return out


def replay(*, spot: pd.DataFrame, signal_fn: Callable, resolve: Callable, premium_for: Callable, params,
           square_off: str, range_start: str, range_end: str, days: Sequence[date],
           regimes: dict[date, set[str]] | None = None, mutation_cut: str | None = "12:00",
           direction_lookahead: bool = False, keep_pipeline: bool = False) -> dict[str, Any]:
    """One shared pipeline run; candidates differ only in their direction/confirmation filter."""
    days = sorted(days)
    if days:
        spot = spot[pd.to_datetime(spot["timestamp"]).dt.date <= days[-1]].reset_index(drop=True)
    pipeline = imp._pipeline_factory(signal_fn, resolve, params, range_start, range_end)
    contexts, feats = pipeline(spot, premium_for, set(days)) if days else ([], None)
    cache: dict = {}
    rows: dict[str, list[dict]] = {}
    for c in CANDIDATES:
        seen: set = set()
        out = []
        for r in (dr.run_candidate(contexts, feats, c, square_off=square_off, lot_size=LOT_SIZE, cache=cache)
                  if contexts else []):
            key = (c.name, r["signal_time"], r["direction"])
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(classify_row(r, regimes), strategy=c.name))
        rows[c.name] = out
    lookahead = {c.name: {"cuts": 0, "failures": [], "passed": True} for c in CANDIDATES}
    if mutation_cut and days:
        cuts = [pd.Timestamp(datetime.combine(d, time.fromisoformat(mutation_cut)), tz=IST) for d in days]
        lookahead = dr.mutation_test(pipeline, spot, premium_for, CANDIDATES, cuts)
    out = {"days": [d.isoformat() for d in days], "rows": rows, "lookahead": lookahead}
    if direction_lookahead:
        out["direction_lookahead"] = (direction_mutation_test(pipeline, spot, premium_for, cuts)
                                      if mutation_cut and days else
                                      {d: {"cuts": 0, "failures": [], "passed": False} for d in DIRECTIONS})
    if keep_pipeline:
        out["pipeline"] = {"pipeline": pipeline, "spot": spot, "premium_for": premium_for, "contexts": contexts,
                           "features": feats, "cache": cache, "regimes": regimes or {}, "square_off": square_off}
    return out


DIRECTIONS = ("BUY", "SELL")


def direction_mutation_test(pipeline: Callable, spot: pd.DataFrame, premium_for: Callable,
                            cuts: Sequence[pd.Timestamp], seed: int = 11) -> dict[str, dict]:
    """``dr.mutation_test`` for the original S2 (BUY + SELL), with BUY and SELL decisions checked separately."""
    result = {d: {"cuts": 0, "failures": []} for d in DIRECTIONS}
    for i, cut in enumerate(cuts):
        cut = imp._ist(cut)
        day = cut.date()
        base_ctx, base_feat = pipeline(spot, premium_for, {day})
        mut_ctx, mut_feat = pipeline(imp.mutate_after(spot, cut, seed + i),
                                     lambda c, _cut=cut, _i=i: imp._mut_prem(premium_for(c), _cut, seed + _i), {day})
        a = dr.fingerprint(base_ctx, base_feat, dr.BASELINE, cut, day)
        b = dr.fingerprint(mut_ctx, mut_feat, dr.BASELINE, cut, day)
        for d, r in result.items():
            fa, fb = [x for x in a if x[1] == d], [x for x in b if x[1] == d]
            r["cuts"] += 1
            if fa != fb or not all(x[-1] for x in fa):
                r["failures"].append(cut.isoformat())
    for r in result.values():
        r["passed"] = bool(r["cuts"]) and not r["failures"]
    return result


# ------------------------------------------------------------------ metrics
def _pf(gp: float, gl: float) -> float | None:
    return gp / gl if gl > 0 else (math.inf if gp > 0 else None)


def stats(rows: Iterable[dict]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["signal_time"])
    ev = [r for r in rows if r["state"] == EVALUATED]
    wins = sum(r["outcome"] == imp.WIN for r in ev)
    pnl = [float(r["pnl"]) for r in ev]
    gp, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
    eq = peak = dd = 0.0
    for p in pnl:
        eq += p
        peak, dd = max(peak, eq), max(dd, peak - eq)
    skipped = sum(r["state"] == SKIPPED for r in rows)
    unavailable = sum(r["state"] == DATA_UNAVAILABLE for r in rows)
    eligible = len(rows) - skipped
    reasons: dict[str, int] = {}
    for r in rows:
        if r["state"] == DATA_UNAVAILABLE:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    return {
        "signals": len(rows), "evaluated": len(ev), "wins": wins, "losses": len(ev) - wins,
        "data_unavailable": unavailable, "skipped": skipped,
        "win_rate": wins / len(ev) if ev else None, "profit_factor": _pf(gp, gl) if ev else None,
        "net_pnl": round(sum(pnl), 2), "avg_pnl_per_trade": round(sum(pnl) / len(ev), 2) if ev else None,
        "max_drawdown": round(dd, 2),
        "data_coverage": len(ev) / eligible if eligible else None,
        "data_unavailable_pct": unavailable / eligible if eligible else None,
        "data_unavailable_reasons": reasons,
    }


def strategy_summary(rows: Sequence[dict], days: Sequence[str]) -> dict[str, Any]:
    m = stats(rows)
    daily = {d: stats(r for r in rows if r["session_date"] == d) for d in days}
    traded = [(d, x) for d, x in daily.items() if x["evaluated"]]
    best = max(traded, key=lambda t: t[1]["net_pnl"]) if traded else None
    worst = min(traded, key=lambda t: t[1]["net_pnl"]) if traded else None
    m.update(
        trading_days=len(days),
        profitable_days=sum(x["net_pnl"] > 0 for x in daily.values()),
        losing_days=sum(x["net_pnl"] < 0 for x in daily.values()),
        best_day=None if best is None else {"date": best[0], "net_pnl": best[1]["net_pnl"]},
        worst_day=None if worst is None else {"date": worst[0], "net_pnl": worst[1]["net_pnl"]},
        average_daily_pnl=round(m["net_pnl"] / len(days), 2) if days else None,
        daily=daily,
        direction={d: stats(r for r in rows if r["direction"] == d) for d in ("BUY", "SELL")},
        dte={label: stats(r for r in rows if r["dte_bucket"] == label) for label, _, _ in DTE_BUCKETS},
        dte_unknown=stats(r for r in rows if r["dte_bucket"] is None),
        session={label: stats(r for r in rows if r.get("time_bucket") == label)
                 for label in sorted({r.get("time_bucket") for r in rows if r.get("time_bucket")})},
        regime={label: stats(r for r in rows if label in r["regimes"]) for label in REGIME_LABELS},
    )
    return m


def build_summary(result: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    days = result["days"]
    strategies = {name: strategy_summary(rows, days) for name, rows in result["rows"].items()}
    ranked = {k: v for k, v in strategies.items() if v["evaluated"]}
    pf_key = lambda kv: -1 if kv[1]["profit_factor"] is None else kv[1]["profit_factor"]  # noqa: E731
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "date_range": meta.get("date_range"), "trading_days": days, **{k: v for k, v in meta.items()
                                                                       if k != "date_range"},
        "strategies": strategies,
        "best": {
            "by_profit_factor": max(ranked.items(), key=pf_key)[0] if ranked else None,
            "by_pnl": max(ranked.items(), key=lambda kv: kv[1]["net_pnl"])[0] if ranked else None,
            "by_drawdown": min(ranked.items(), key=lambda kv: kv[1]["max_drawdown"])[0] if ranked else None,
            "note": "Ranks only strategies with evaluated trades; a ranking on this sample is not evidence of "
                    "robustness.",
        },
        "lookahead": result["lookahead"],
    }


# ------------------------------------------------------------------ reports
def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and math.isinf(v):
        return "INF"
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def _pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.1%}"


def _pf_s(v: float | None) -> str:
    return "N/A" if v is None else ("INF" if math.isinf(v) else f"{v:.2f}")


def _rs(v: float | None) -> str:
    return "N/A" if v is None else f"{v:,.2f}"


SUMMARY_FIELDS = ("strategy", "trading_days", "signals", "evaluated", "wins", "losses", "data_unavailable", "skipped",
                  "win_rate", "profit_factor", "net_pnl", "avg_pnl_per_trade", "max_drawdown", "data_coverage",
                  "data_unavailable_pct", "profitable_days", "losing_days", "best_day", "best_day_pnl", "worst_day",
                  "worst_day_pnl", "average_daily_pnl")
SIGNAL_FIELDS = ("strategy", "session_date", "signal_time", "direction", "decision", "state", "skip_reason",
                 "reason", "reason_raw", "contract", "instrument_key", "expiry", "dte", "dte_bucket", "time_bucket",
                 "entry_time", "entry", "premium_atr", "stop_loss", "target", "outcome", "exit_time", "exit_price",
                 "exit_reason", "pnl", "r_multiple")


def _group_table(title: str, groups: dict[str, dict]) -> list[str]:
    md = [f"| {title} | Signals | Evaluated | Wins | Losses | WR | PF | P&L (Rs) |", "|---|---|---|---|---|---|---|---|"]
    for label, m in groups.items():
        md.append(f"| {label} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | "
                  f"{_pct(m['win_rate'])} | {_pf_s(m['profit_factor'])} | {_rs(m['net_pnl'])} |")
    return md


def current_day_section(current: dict | None) -> list[str]:
    md = ["", f"## {CURRENT_HEADING} (separate - not merged with the replay)", ""]
    if not current:
        return md + ["- No current-day shadow report found."]
    md += [f"Current-day validation: {', '.join(current.get('validated_days') or [])} "
           "(1-DTE / current-market validation; real intraday candles).", "",
           "| Strategy | Days | Evaluated | Wins | Losses | WR | PF | P&L (Rs) | Max DD (Rs) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, m in (current.get("candidates") or {}).items():
        pf = m.get("profit_factor")
        pf = None if pf == "INF" else pf
        md.append(f"| {name} | {m.get('trading_days', m.get('days'))} | {m['evaluated']} | {m['wins']} | "
                  f"{m['losses']} | {_pct(m['win_rate'])} | {_pf_s(pf)} | {_rs(m['net_pnl'])} | "
                  f"{_rs(m['max_drawdown'])} |")
    return md


def summary_markdown(s: dict[str, Any], current: dict | None = None) -> str:
    st = s["strategies"]
    names = list(st)
    md = [f"# {HEADING}", "",
          "TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, read-only replay - no orders were placed. "
          "Production S2 unchanged; no filters, thresholds, stops, targets or expiry rules were changed.", "",
          f"- Date range: {s['date_range']}",
          f"- Trading days replayed: {len(s['trading_days'])} ({', '.join(s['trading_days']) or 'none'})",
          f"- Weekdays excluded: {', '.join(f'{d} ({r})' for d, r in s.get('excluded_days', {}).items()) or 'none'}",
          f"- Contract point-in-time check: {s.get('expiry_check', {}).get('summary', 'n/a')}",
          f"- Contract resolution: {s.get('contract_resolution', 'n/a')}",
          "- Expired option candles: " + (
              "downloaded {contracts_downloaded}, from local cache {contracts_from_cache}, rows written "
              "{rows_written}, download failures {nf}".format(nf=len(s['expired_candle_downloads'].get('failures') or []),
                                                              **s['expired_candle_downloads'])
              if s.get("expired_candle_downloads") else "n/a"), "",
          "## Comparison", "",
          "| Strategy | Days | Signals | Evaluated | Wins | Losses | Data Unavailable | Skipped | WR | PF | P&L (Rs) | "
          "Max DD (Rs) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, m in st.items():
        md.append(f"| {n} | {m['trading_days']} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | "
                  f"{m['data_unavailable']} | {m['skipped']} | {_pct(m['win_rate'])} | {_pf_s(m['profit_factor'])} | "
                  f"{_rs(m['net_pnl'])} | {_rs(m['max_drawdown'])} |")
    md += ["", "## Data coverage", "",
           "Data coverage = evaluated / (signals - skipped). DATA_UNAVAILABLE and SKIPPED are never counted as losses.",
           "", "| Strategy | Eligible (signals - skipped) | Evaluated | Data coverage | Option data unavailable % | "
           "Unavailable reasons |", "|---|---|---|---|---|---|"]
    for n, m in st.items():
        reasons = ", ".join(f"{k}: {v}" for k, v in m["data_unavailable_reasons"].items()) or "none"
        md.append(f"| {n} | {m['signals'] - m['skipped']} | {m['evaluated']} | {_pct(m['data_coverage'])} | "
                  f"{_pct(m['data_unavailable_pct'])} | {reasons} |")
    for n, m in st.items():
        md += ["", f"## {n}", "",
               f"- Trading days {m['trading_days']}: profitable {m['profitable_days']}, losing {m['losing_days']}",
               f"- Signals {m['signals']}, evaluated {m['evaluated']}, wins {m['wins']}, losses {m['losses']}, "
               f"data unavailable {m['data_unavailable']}, skipped {m['skipped']}",
               f"- WR {_pct(m['win_rate'])}, PF {_pf_s(m['profit_factor'])}, total P&L Rs {_rs(m['net_pnl'])}, "
               f"avg/trade Rs {_rs(m['avg_pnl_per_trade'])}, avg/day Rs {_rs(m['average_daily_pnl'])}, "
               f"max DD Rs {_rs(m['max_drawdown'])}",
               f"- Best day: {'N/A' if not m['best_day'] else m['best_day']['date'] + ' (Rs ' + _rs(m['best_day']['net_pnl']) + ')'}; "
               f"worst day: {'N/A' if not m['worst_day'] else m['worst_day']['date'] + ' (Rs ' + _rs(m['worst_day']['net_pnl']) + ')'}",
               ""] + _group_table("Direction", m["direction"])
    md += ["", "## Day-by-day comparison", "", "| Date | " + " | ".join(f"{n} P&L" for n in names) + " |",
           "|---|" + "---|" * len(names)]
    for d in s["trading_days"]:
        md.append(f"| {d} | " + " | ".join(_rs(st[n]["daily"][d]["net_pnl"]) for n in names) + " |")
    md += ["", "| Date | " + " | ".join(f"{n} WR" for n in names) + " |", "|---|" + "---|" * len(names)]
    for d in s["trading_days"]:
        md.append(f"| {d} | " + " | ".join(_pct(st[n]["daily"][d]["win_rate"]) for n in names) + " |")
    md += ["", "| Date | " + " | ".join(f"{n} trades" for n in names) + " |", "|---|" + "---|" * len(names)]
    for d in s["trading_days"]:
        md.append(f"| {d} | " + " | ".join(str(st[n]["daily"][d]["evaluated"]) for n in names) + " |")
    md += ["", "## DTE breakdown (observed only; no causal claim)", "",
           "DTE = actual contract expiry - trading date, for signals whose contract was verified."]
    for n, m in st.items():
        md += ["", f"### {n}", ""] + _group_table("DTE", {**m["dte"], "unknown (contract not verified)":
                                                         m["dte_unknown"]})
    md += ["", "## Session breakdown (existing project time buckets)"]
    for n, m in st.items():
        md += ["", f"### {n}", ""] + (_group_table("Time bucket", m["session"]) if m["session"] else ["- none"])
    md += ["", "## Regime breakdown (existing regime_classifier, end-of-day labels, reporting only)", "",
           "Labels describe the whole session after the fact and are never used to generate or filter signals. "
           "A day can carry several labels."]
    for n, m in st.items():
        md += ["", f"### {n}", ""] + _group_table("Regime", m["regime"])
    la = s["lookahead"]
    md += ["", "## Look-ahead mutation test", "",
           "Every NIFTY and premium candle after 12:00 on each replayed day is randomly rescaled and the pipeline is "
           "re-run; decisions, contracts and premium ATR for earlier signals must be identical and every entry must "
           "be after its signal.", ""]
    md += [f"- {n}: {v['cuts']} cuts, {'PASSED' if v['passed'] else 'FAILED ' + ', '.join(v['failures'])}"
           for n, v in la.items()]
    b = s["best"]
    md += ["", "## Observed ranking (not a robustness claim)", "",
           f"- Best by PF: {b['by_profit_factor'] or 'N/A'}; best by P&L: {b['by_pnl'] or 'N/A'}; "
           f"lowest drawdown: {b['by_drawdown'] or 'N/A'}", f"- {b['note']}"]
    md += current_day_section(current)
    md += ["", "---", "", f"Historical replay: {s['date_range']}", "",
           f"Current-day validation: {', '.join((current or {}).get('validated_days') or []) or 'none'} "
           "(1-DTE / current-market validation)"]
    return "\n".join(md) + "\n"


def daily_markdown(day: str, summary: dict[str, Any], rows: Sequence[dict]) -> str:
    st = summary["strategies"]
    md = [f"# {HEADING} - {day}", "", "PAPER, read-only replay - no orders.", "",
          "| Strategy | Signals | Evaluated | Wins | Losses | Data Unavailable | Skipped | WR | PF | P&L (Rs) | "
          "Max DD (Rs) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, m in st.items():
        d = m["daily"][day]
        md.append(f"| {n} | {d['signals']} | {d['evaluated']} | {d['wins']} | {d['losses']} | {d['data_unavailable']} "
                  f"| {d['skipped']} | {_pct(d['win_rate'])} | {_pf_s(d['profit_factor'])} | {_rs(d['net_pnl'])} | "
                  f"{_rs(d['max_drawdown'])} |")
    md += ["", "| Strategy | Time | Dir | State | Reason | Expiry | DTE | Entry | Exit | P&L |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        reason = r["skip_reason"] if r["state"] == SKIPPED else (r["reason"] or r.get("exit_reason") or "")
        md.append(f"| {r['strategy']} | {r['signal_time'][11:16]} | {r['direction']} | {r['state']} | {reason} | "
                  f"{r['expiry'] or ''} | {'' if r.get('dte') is None else r['dte']} | "
                  f"{'' if r.get('entry') is None else r['entry']} | "
                  f"{'' if r.get('exit_price') is None else r['exit_price']} | {'' if r['pnl'] is None else r['pnl']} |")
    return "\n".join(md) + "\n"


def write_reports(summary: dict[str, Any], rows_by_strategy: dict[str, list[dict]], out_dir: Path,
                  current: dict | None = None) -> dict[str, Path]:
    """Overwrites the whole replay output, so a re-run never duplicates days or signals."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.iterdir():
        if DAILY_FILE.match(p.name):
            p.unlink()
    all_rows = sorted((r for rows in rows_by_strategy.values() for r in rows),
                      key=lambda r: (r["signal_time"], r["strategy"]))
    for day in summary["trading_days"]:
        rows = [r for r in all_rows if r["session_date"] == day]
        day_summary = {n: m["daily"][day] for n, m in summary["strategies"].items()}
        (out_dir / f"{day}.json").write_text(json.dumps(_jsonable(
            {"heading": HEADING, "date": day, "strategies": day_summary,
             "signals": [{k: r.get(k) for k in SIGNAL_FIELDS} for r in rows]}), indent=2), encoding="utf-8")
        (out_dir / f"{day}.md").write_text(daily_markdown(day, summary, rows), encoding="utf-8")
        with (out_dir / f"{day}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SIGNAL_FIELDS)
            w.writeheader()
            w.writerows({k: _jsonable(r.get(k)) for k in SIGNAL_FIELDS} for r in rows)
    paths = {"json": out_dir / "summary.json", "csv": out_dir / "summary.csv", "md": out_dir / "summary.md"}
    paths["json"].write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for n, m in summary["strategies"].items():
            row = {k: _jsonable(m.get(k)) for k in SUMMARY_FIELDS}
            row.update(strategy=n, best_day=(m["best_day"] or {}).get("date"),
                       best_day_pnl=(m["best_day"] or {}).get("net_pnl"), worst_day=(m["worst_day"] or {}).get("date"),
                       worst_day_pnl=(m["worst_day"] or {}).get("net_pnl"))
            w.writerow(row)
    paths["md"].write_text(summary_markdown(summary, current), encoding="utf-8")
    return paths
