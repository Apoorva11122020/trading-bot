"""BUY DTE / EXPIRY SELECTION RESEARCH LAB (PAPER, research only, no orders).

The original S2 BUY signal, ATM CE strike rule, next-candle entry, 1.5 x premium-ATR stop, 2R target and square-off
are unchanged; only the expiry used for the CE differs between eight pre-declared policies. Expiries are chosen
point-in-time: from the official expiry list as of the signal date, and a contract counts as available only if it
has a real candle on the signal date at or before the signal candle. When no listed expiry exists in the requested
range the signal is EXPIRY_UNAVAILABLE (never replaced by another DTE) and excluded from wins, losses and P&L.
SELL signals are not evaluated.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import buy_strategy_lab as lab
from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.historical_data.option_coverage import resolve_signal_contract
from src.options.models import OptionResolutionError

HEADING = "BUY DTE / EXPIRY SELECTION RESEARCH LAB"
EXPIRY_UNAVAILABLE = "EXPIRY_UNAVAILABLE"
EXPIRY_NOT_LISTED = "EXPIRY_NOT_LISTED"
EXPIRY_LIST_UNAVAILABLE = "EXPIRY_LIST_UNAVAILABLE"
EXPIRY_REASONS = {EXPIRY_UNAVAILABLE: "expiry_unavailable", EXPIRY_NOT_LISTED: "expiry_not_listed_on_signal_date"}
PROMISING, NOT_PROMISING, INSUFFICIENT_DATA = lab.PROMISING, lab.NOT_PROMISING, lab.INSUFFICIENT_DATA
BASELINE = "BUY_NEAREST_EXPIRY"


@dataclass(frozen=True)
class ExpiryPolicy:
    name: str
    mode: str                      # NEAREST | NEXT_TRADING_DAY | RANGE
    low: int | None = None
    high: int | None = None

    @property
    def requested(self) -> str:
        if self.mode == "NEAREST":
            return "nearest"
        if self.mode == "NEXT_TRADING_DAY":
            return "1 trading day"
        return f"{self.low}+ DTE" if self.high is None else (
            f"{self.low} DTE" if self.low == self.high else f"{self.low}-{self.high} DTE")


POLICIES = (
    ExpiryPolicy(BASELINE, "NEAREST"),
    ExpiryPolicy("BUY_0_DTE", "RANGE", 0, 0),
    ExpiryPolicy("BUY_1_DTE", "NEXT_TRADING_DAY"),
    ExpiryPolicy("BUY_2_4_DTE", "RANGE", 2, 4),
    ExpiryPolicy("BUY_5_7_DTE", "RANGE", 5, 7),
    ExpiryPolicy("BUY_8_14_DTE", "RANGE", 8, 14),
    ExpiryPolicy("BUY_15_21_DTE", "RANGE", 15, 21),
    ExpiryPolicy("BUY_22_PLUS_DTE", "RANGE", 22, None),
)
POLICY_NOTE = ("DTE = calendar days from the signal date to the resolved contract's actual expiry. BUY_1_DTE = the "
               "expiry on the next weekday after the signal date (exchange holidays are not modelled). Within a "
               "range the nearest listed expiry is used; no expiry outside the requested range is ever substituted.")


@dataclass(frozen=True)
class DteCriteria:
    min_oos_trades: int = 30
    min_wr: float = 0.45
    min_oos_pf: float = 1.10
    meaningful_train_improvement_pp: float = 10.0
    min_data_coverage: float = 0.90


# ------------------------------------------------------------------ point-in-time expiry selection
def _weekdays_between(a: date, b: date) -> int:
    return sum((a + timedelta(days=i)).weekday() < 5 for i in range(1, (b - a).days))


def in_policy(policy: ExpiryPolicy, signal_date: date, expiry: date) -> bool:
    dte = (expiry - signal_date).days
    if dte < 0:
        return False
    if policy.mode == "NEAREST":
        return True
    if policy.mode == "NEXT_TRADING_DAY":
        return dte >= 1 and _weekdays_between(signal_date, expiry) == 0
    return dte >= policy.low and (policy.high is None or dte <= policy.high)


def eligible_expiries(expiries: Sequence[date], signal_date: date, policy: ExpiryPolicy) -> list[date]:
    """Expiries on/after the signal date inside the requested range, nearest first."""
    future = sorted(e for e in set(expiries) if e >= signal_date)
    if policy.mode == "NEAREST":
        return future[:1]
    return [e for e in future if in_policy(policy, signal_date, e)]


def listed_at(premium: Any, signal_time: pd.Timestamp) -> bool:
    """The contract traded on the signal date at or before the signal candle (no later candle is read)."""
    if premium is None or isinstance(premium, str) or premium.empty:
        return False
    ts = pd.to_datetime(premium["timestamp"])
    ts = ts.dt.tz_localize(imp.IST) if ts.dt.tz is None else ts.dt.tz_convert(imp.IST)
    return bool(((ts.dt.date == signal_time.date()) & (ts <= signal_time)).any())


class _OneExpiry:
    """Exposes a single expiry so ``resolve_signal_contract`` applies its unchanged ATM CE strike rule."""

    def __init__(self, pit, expiry: date) -> None:
        self.pit, self.expiry, self.today = pit, expiry, pit.today

    def expiries(self, underlying):
        return [self.expiry]

    def contracts(self, underlying, expiry, option_type=None):
        return self.pit.contracts(underlying, expiry, option_type)

    def strikes(self, underlying, expiry, option_type=None):
        return self.pit.strikes(underlying, expiry, option_type)

    def resolve(self, underlying, expiry, strike, option_type):
        return self.pit.resolve(underlying, expiry, strike, option_type)


def policy_resolver(policy: ExpiryPolicy, pit, rc, premium_for: Callable) -> Callable:
    """Resolve the ATM CE of the nearest listed expiry inside the policy's DTE range, as of the signal date."""
    def resolve(sig):
        if pit is None:
            raise OptionResolutionError(EXPIRY_LIST_UNAVAILABLE, "official expiry list unavailable")
        ts = imp._ist(sig.timestamp)
        day = ts.date()
        pit.today = lambda: day
        choices = eligible_expiries(pit.expiries(sig.underlying), day, policy)
        if not choices:
            raise OptionResolutionError(EXPIRY_UNAVAILABLE, f"no listed expiry for {policy.requested} on {day}")
        for expiry in choices:
            try:
                contract = resolve_signal_contract(sig, _OneExpiry(pit, expiry), rc)
            except OptionResolutionError:
                continue
            premium = premium_for(contract)
            if isinstance(premium, str) and premium.startswith("PREMIUM_API_ERROR"):
                return contract
            if listed_at(premium, ts):
                return contract
        raise OptionResolutionError(EXPIRY_NOT_LISTED, f"no {policy.requested} contract traded on {day} by {ts}")
    return resolve


def buy_only(signal_fn: Callable) -> Callable:
    return lambda frame: [s for s in signal_fn(frame) if s.direction == "BUY"]


# ------------------------------------------------------------------ evaluation
def classify(row: dict, regimes: dict | None, policy: ExpiryPolicy) -> dict:
    out = dict(sh.classify_row(row, regimes), strategy=policy.name, requested_dte=policy.requested)
    raw = str(out.get("reason_raw") or "")
    sub = raw.split(":", 1)[1] if raw.startswith("CONTRACT_UNRESOLVED:") else ""
    out["expiry_unavailable"] = sub in EXPIRY_REASONS
    if out["expiry_unavailable"]:
        out["reason"] = EXPIRY_REASONS[sub]
    elif sub == EXPIRY_LIST_UNAVAILABLE:
        out["reason"] = "expired_instrument_unavailable"
    exp = out.get("expiry")
    out["in_requested_range"] = (None if not exp else
                                 in_policy(policy, date.fromisoformat(out["session_date"]), date.fromisoformat(exp[:10])))
    return out


def evaluate_policy(contexts: Sequence[dict], features: pd.DataFrame, policy: ExpiryPolicy, *, square_off: str,
                    regimes: dict | None = None, cache: dict | None = None) -> list[dict]:
    """``cache`` must be private to one policy: ``imp.evaluate`` keys it by signal, not by contract."""
    rows = imp.run_variant(contexts, features, imp.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE, cache=cache)
    out, seen = [], set()
    for ctx, r in zip(contexts, rows):
        key = (r["signal_time"], r["direction"])
        if r["direction"] != "BUY" or key in seen:
            continue
        seen.add(key)
        r["expiry"] = ctx["expiry"].isoformat() if ctx.get("expiry") else None
        out.append(classify(r, regimes, policy))
    return out


def policy_metrics(rows: Sequence[dict], days: Sequence[str]) -> dict[str, Any]:
    m = so.split_metrics(rows, days)
    ev = [r for r in rows if r["state"] == sh.EVALUATED]
    m["non_wins"] = m["losses"]
    m["losses"] = sum(r["outcome"] == imp.LOSS for r in ev)
    m["ambiguous"] = sum(r["outcome"] == imp.AMBIGUOUS for r in ev)
    m["scratch"] = sum(r["outcome"] == imp.SCRATCH for r in ev)
    m["expiry_unavailable"] = sum(bool(r.get("expiry_unavailable")) for r in rows)
    m["data_unavailable_other"] = m["data_unavailable"] - m["expiry_unavailable"]
    eligible = m["signals"] - m["skipped"] - m["expiry_unavailable"]
    m["data_coverage_ex_expiry"] = m["evaluated"] / eligible if eligible else None
    return m


def candidate_status(train: dict, val: dict, oos: dict, base_train: dict, lookahead_passed: bool,
                     c: DteCriteria = DteCriteria()) -> dict[str, Any]:
    twr, bwr = train["win_rate"], base_train["win_rate"]
    improves = twr is not None and bwr is not None and (twr - bwr) * 100 >= c.meaningful_train_improvement_pp
    cov = oos["data_coverage_ex_expiry"]
    checks = {
        "oos_trades": (oos["evaluated"] >= c.min_oos_trades, oos["evaluated"], f">= {c.min_oos_trades}"),
        "oos_wr": (oos["win_rate"] is not None and oos["win_rate"] >= c.min_wr, oos["win_rate"], f">= {c.min_wr:.0%}"),
        "oos_pf": (oos["profit_factor"] is not None and oos["profit_factor"] > c.min_oos_pf, oos["profit_factor"],
                   f"> {c.min_oos_pf}"),
        "oos_pnl": (oos["net_pnl"] > 0, oos["net_pnl"], "> 0"),
        "validation_wr": (val["win_rate"] is not None and val["win_rate"] >= c.min_wr, val["win_rate"],
                          f">= {c.min_wr:.0%}"),
        "train_wr_or_meaningful_improvement": ((twr is not None and twr >= c.min_wr) or improves, twr,
                                               f">= {c.min_wr:.0%} or >= +{c.meaningful_train_improvement_pp:g} pp "
                                               "vs baseline train WR"),
        "lookahead": (lookahead_passed, lookahead_passed, "PASS"),
        "data_coverage": (cov is not None and cov >= c.min_data_coverage, cov,
                          f">= {c.min_data_coverage:.0%} (expiry-unavailable excluded)"),
    }
    checks = {k: {"passed": bool(p), "value": v, "threshold": t} for k, (p, v, t) in checks.items()}
    failed = [k for k, v in checks.items() if not v["passed"]]
    status = (INSUFFICIENT_DATA if not checks["oos_trades"]["passed"] else PROMISING if not failed else NOT_PROMISING)
    return {"status": status, "failed_checks": failed, "checks": checks}


def resolution_summary(rows: Sequence[dict]) -> dict[str, Any]:
    resolved = [r for r in rows if r.get("expiry")]
    hist: dict[str, int] = {}
    for r in resolved:
        hist[str(r["dte"])] = hist.get(str(r["dte"]), 0) + 1
    return {"requested": len(rows), "resolved": len(resolved),
            "expiry_unavailable": sum(bool(r.get("expiry_unavailable")) for r in rows),
            "resolved_dte_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
            "outside_requested_range": sum(r["in_requested_range"] is False for r in resolved),
            "expiry_before_signal_date": sum(r["dte"] is not None and r["dte"] < 0 for r in resolved)}


def mutation_lookahead(pipeline: Callable, spot: pd.DataFrame, premium_for: Callable, policy: ExpiryPolicy,
                       cuts: Sequence[pd.Timestamp]) -> dict:
    """Existing mutation test: rescale candles after each cut; BUY decision, contract and entry must not change."""
    cand = dr.DirectionCandidate(policy.name, ("BUY",))
    return dr.mutation_test(pipeline, spot, premium_for, [cand], cuts)[policy.name]


# ------------------------------------------------------------------ study
def build_lab(x, *, splits: dict, criteria: DteCriteria = DteCriteria(), cut_times: Sequence[str] = ("12:00",),
              original_buy_rows: Sequence[dict] | None = None,
              progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """``x`` = inputs from ``prepare_replay_inputs`` (stored spot, point-in-time resolver, real premium loader)."""
    split_days, unassigned = so.assign_days([d.isoformat() for d in x.selected], splits)
    hist_days = [d for s in imp.SPLITS for d in split_days[s]]
    day_set = set(hist_days)
    signal_fn = buy_only(x.signal_fn)
    cuts = [pd.Timestamp(datetime.combine(date.fromisoformat(d), time.fromisoformat(t)), tz=imp.IST)
            for d in hist_days for t in cut_times]
    rows, lookahead = {}, {}
    for i, p in enumerate(POLICIES, 1):
        if progress:
            progress(f"[{i}/{len(POLICIES)}] {p.name}: resolving contracts and evaluating BUY signals")
        cache: dict = {}
        resolve = (sh.strict_resolver(x.resolve, x.check) if p.mode == "NEAREST"
                   else policy_resolver(p, x.pit, x.rc, x.premium_for))
        pipeline = imp._pipeline_factory(signal_fn, resolve, x.params, x.s2["range_start"], x.s2["range_end"])
        contexts, feats = pipeline(x.spot, x.premium_for, {date.fromisoformat(d) for d in hist_days})
        rows[p.name] = [r for r in evaluate_policy(contexts, feats, p, square_off=x.rc.session.square_off,
                                                   regimes=x.regimes, cache=cache) if r["session_date"] in day_set]
        if progress:
            progress(f"[{i}/{len(POLICIES)}] {p.name}: lookahead mutation test ({len(cuts)} cuts)")
        lookahead[p.name] = (mutation_lookahead(pipeline, x.spot, x.premium_for, p, cuts) if cuts
                             else {"cuts": 0, "failures": [], "passed": False})
        lookahead[p.name]["passed"] = bool(lookahead[p.name]["cuts"]) and not lookahead[p.name]["failures"]
    return assemble(rows, lookahead, split_days, unassigned, splits, criteria, original_buy_rows)


def assemble(rows: dict[str, list[dict]], lookahead: dict[str, dict], split_days: dict, unassigned: list,
             splits: dict, criteria: DteCriteria = DteCriteria(),
             original_buy_rows: Sequence[dict] | None = None) -> dict[str, Any]:
    names = [p.name for p in POLICIES]
    hist_days = [d for s in imp.SPLITS for d in split_days[s]]
    periods = {**split_days, "ALL": hist_days}
    metrics = {s: {n: policy_metrics([r for r in rows[n] if r["session_date"] in set(days)], days) for n in names}
               for s, days in periods.items()}
    oos = metrics["OOS"]
    status = {n: candidate_status(metrics["TRAIN"][n], metrics["VALIDATION"][n], oos[n], metrics["TRAIN"][BASELINE],
                                  lookahead[n]["passed"], criteria) for n in names}
    ranking = lab.rank_by_oos(oos, criteria.min_oos_trades)
    eligible = [n for n in ranking if n != BASELINE and oos[n]["evaluated"] >= criteria.min_oos_trades]
    best = eligible[0] if eligible else None
    top3 = [n for n in ranking if n != BASELINE][:3]
    oos_set = set(split_days["OOS"])
    day_check = {n: {"best_day_removed": lab._without_day([r for r in rows[n] if r["session_date"] in oos_set],
                                                          (oos[n]["best_day"] or {}).get("date")),
                     "worst_day_removed": lab._without_day([r for r in rows[n] if r["session_date"] in oos_set],
                                                           (oos[n]["worst_day"] or {}).get("date"))}
                 for n in names}
    sig = lambda rs: [(r["signal_time"], r["state"], r.get("outcome"), r["pnl"], r.get("instrument_key"))  # noqa: E731
                      for r in rs]
    daily = []
    split_of_day = {d: s for s in imp.SPLITS for d in split_days[s]}
    for d in hist_days:
        row = {"date": d, "split": split_of_day[d]}
        for n in names:
            x = metrics["ALL"][n]["daily"][d]
            row.update({f"{n}_pnl": x["net_pnl"], f"{n}_trades": x["evaluated"], f"{n}_wr": x["win_rate"]})
        daily.append(row)
    table = [{"candidate": n, "requested_dte": p.requested, "train_wr": metrics["TRAIN"][n]["win_rate"],
              "validation_wr": metrics["VALIDATION"][n]["win_rate"], "oos_wr": oos[n]["win_rate"],
              "oos_pf": oos[n]["profit_factor"], "oos_pnl": oos[n]["net_pnl"], "oos_dd": oos[n]["max_drawdown"],
              "oos_trades": oos[n]["evaluated"], "status": status[n]["status"]} for n, p in zip(names, POLICIES)]
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"),
        "policies": {p.name: p.requested for p in POLICIES}, "policy_note": POLICY_NOTE,
        "evaluation": "Original S2 BUY signal; ATM CE (existing strike rule); entry at the next option candle open; "
                      "stop 1.5 x premium ATR; target 2R; square-off 15:15. Only the expiry differs. SELL not evaluated.",
        "accuracy_definition": "WR = wins / evaluated x 100; AMBIGUOUS (conservative -1R) and SCRATCH are not wins; "
                               "EXPIRY_UNAVAILABLE and DATA_UNAVAILABLE are excluded from evaluated, wins, losses, P&L.",
        "never_robust_note": "No policy is called ROBUST: the historical OOS window was already seen by earlier research.",
        "splits": {s: {"label": so.SPLIT_LABELS[s], "from": splits[s][0].isoformat(), "to": splits[s][1].isoformat(),
                       "days": split_days[s]} for s in imp.SPLITS},
        "unassigned_days": unassigned,
        "current_day_note": "2026-10-05 and later are not used for selection or ranking.",
        "criteria": asdict(criteria),
        "baseline_matches_original_s2_buy": (None if original_buy_rows is None else
                                             sig(rows[BASELINE]) == sig(original_buy_rows)),
        "metrics": metrics, "status": status, "ranking_by_oos": ranking, "best_candidate": best,
        "best_candidate_recommendable": bool(best and status[best]["status"] == PROMISING),
        "top3": top3, "day_removal_check_oos": day_check,
        "regime_report_for": top3 + [BASELINE],
        "comparison_vs_baseline": {n: {s: lab.comparison(metrics[s][BASELINE], metrics[s][n]) for s in ("OOS", "ALL")}
                                   for n in names if n != BASELINE},
        "dte_resolution": {n: resolution_summary(rows[n]) for n in names},
        "lookahead": lookahead,
        "lookahead_by_split": {s: so.split_lookahead(lookahead, split_days[s]) for s in imp.SPLITS},
        "daily": daily, "final_table": table,
        "_rows": rows,
    }


# ------------------------------------------------------------------ reports
_pct, _pf, _rs, _signed = sh._pct, sh._pf_s, sh._rs, lab._signed
RES_FIELDS = ("policy", "requested_dte", "session_date", "signal_time", "instrument_key", "contract", "expiry", "dte",
              "dte_bucket", "in_requested_range", "state", "reason", "outcome", "entry_time", "entry", "stop_loss",
              "target", "exit_time", "exit_reason", "pnl")


def resolution_sample(rows: dict[str, list[dict]], per_policy: int = 3) -> list[dict]:
    out = []
    for n, rs in rows.items():
        out += [{"policy": n, "date": r["session_date"], "requested_dte": r["requested_dte"], "expiry": r["expiry"],
                 "actual_dte": r["dte"]} for r in rs if r.get("expiry")][:per_policy]
    return out


def summary_markdown(a: dict[str, Any], rows: dict[str, list[dict]]) -> str:
    md = [f"# {a['heading']}", "", "PAPER mode | research only | no orders placed | production S2, SELL_ONLY and "
          "SELL_CONFIRMED unchanged | no policy applied automatically", "", a["evaluation"], "", a["policy_note"], "",
          a["accuracy_definition"], "", f"> {a['never_robust_note']}", "",
          f"BUY_NEAREST_EXPIRY identical to original S2 BUY rows: {a['baseline_matches_original_s2_buy']}", "",
          "## Results (ranked by OOS PF, P&L, DD, WR, trades)", "",
          "| Rank | Candidate | Requested | Train WR | Validation WR | OOS WR | OOS PF | OOS P&L (Rs) | OOS DD (Rs) | "
          "OOS Trades | Lookahead | Status |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    t = {r["candidate"]: r for r in a["final_table"]}
    for i, n in enumerate(a["ranking_by_oos"], 1):
        r = t[n]
        md.append(f"| {i} | {n} | {r['requested_dte']} | {_pct(r['train_wr'])} | {_pct(r['validation_wr'])} | "
                  f"{_pct(r['oos_wr'])} | {_pf(r['oos_pf'])} | {_rs(r['oos_pnl'])} | {_rs(r['oos_dd'])} | "
                  f"{r['oos_trades']} | {'PASS' if a['lookahead'][n]['passed'] else 'FAIL'} | {r['status']} |")
    md += ["", "Failed checks: " + "; ".join(f"{n}: {', '.join(s['failed_checks']) or 'none'}"
                                             for n, s in a["status"].items())]
    for period in ("TRAIN", "VALIDATION", "OOS", "ALL"):
        md += ["", f"## {period}", "",
               "| Policy | Signals | Evaluated | Wins | Losses | Ambiguous | Scratch | Expiry unavail. | Data unavail. | "
               "Skipped | WR | PF | P&L (Rs) | Avg/trade | Max DD | +Days | -Days | Best day | Worst day |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, m in a["metrics"][period].items():
            b, w = m["best_day"] or {}, m["worst_day"] or {}
            md.append(f"| {n} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | {m['ambiguous']} | "
                      f"{m['scratch']} | {m['expiry_unavailable']} | {m['data_unavailable_other']} | {m['skipped']} | "
                      f"{_pct(m['win_rate'])} | {_pf(m['profit_factor'])} | {_rs(m['net_pnl'])} | "
                      f"{_rs(m['avg_pnl_per_trade'])} | {_rs(m['max_drawdown'])} | {m['profitable_days']} | "
                      f"{m['losing_days']} | {b.get('date', 'N/A')} {_rs(b.get('net_pnl'))} | "
                      f"{w.get('date', 'N/A')} {_rs(w.get('net_pnl'))} |")
    md += ["", "## Comparison vs BUY_NEAREST_EXPIRY", "",
           "| Policy | Period | WR | WR change | PF change | P&L change (Rs) | DD change (Rs) | Trade-count diff |",
           "|---|---|---|---|---|---|---|---|"]
    for n, by in a["comparison_vs_baseline"].items():
        for period, c in by.items():
            md.append(f"| {n} | {period} | {_pct(c['baseline']['win_rate'])} -> {_pct(c['candidate']['win_rate'])} | "
                      f"{_signed(c['wr_change_pp'], 'pp')} | {_signed(c['pf_change'], 'pf')} | "
                      f"{_signed(c['pnl_change'])} | {_signed(c['dd_change'])} | {c['trade_count_change']:+d} |")
    md += ["", "## DTE reality check", "",
           "| Policy | Requested | Signals | Resolved | Expiry unavailable | Outside range | Resolved DTE histogram |",
           "|---|---|---|---|---|---|---|"]
    for n, r in a["dte_resolution"].items():
        md.append(f"| {n} | {a['policies'][n]} | {r['requested']} | {r['resolved']} | {r['expiry_unavailable']} | "
                  f"{r['outside_requested_range']} | {r['resolved_dte_histogram']} |")
    md += ["", "Sample resolved trades:", "", "| Policy | Date | Requested DTE | Resolved expiry | Actual DTE |",
           "|---|---|---|---|---|"]
    md += [f"| {s['policy']} | {s['date']} | {s['requested_dte']} | {s['expiry']} | {s['actual_dte']} |"
           for s in resolution_sample(rows)]
    md += ["", "## OOS day-removal check (reporting only)", "",
           "| Policy | OOS P&L | Best day removed | P&L | Worst day removed | P&L |", "|---|---|---|---|---|---|"]
    for n, c in a["day_removal_check_oos"].items():
        b, w = c["best_day_removed"], c["worst_day_removed"]
        md.append(f"| {n} | {_rs(a['metrics']['OOS'][n]['net_pnl'])} | {b['excluded_date']} | {_rs(b['net_pnl'])} | "
                  f"{w['excluded_date']} | {_rs(w['net_pnl'])} |")
    md += ["", "## Regime report (top 3 + baseline; end-of-day labels, reporting only)", ""]
    for n in a["regime_report_for"]:
        for period in ("ALL", "OOS"):
            reg = a["metrics"][period][n]["regime"]
            md += [f"### {n} - {period}", ""] + (sh._group_table("Regime", reg) if reg else ["- No regime label."])
            md.append("")
    md += ["## Daily (all splits)", "", "| Date | Split | " + " | ".join(f"{n} P&L / trades / WR" for n in
                                                                      a["policies"]) + " |",
           "|---|---|" + "---|" * len(a["policies"])]
    for r in a["daily"]:
        md.append(f"| {r['date']} | {r['split']} | " + " | ".join(
            f"{_rs(r[f'{n}_pnl'])} / {r[f'{n}_trades']} / {_pct(r[f'{n}_wr'])}" for n in a["policies"]) + " |")
    md += ["", "## Lookahead mutation", ""]
    md += [f"- {n}: {'PASS' if r['passed'] else 'FAIL'} ({r['cuts']} cuts, {len(r['failures'])} failures)"
           for n, r in a["lookahead"].items()]
    md += ["", "---", "Research only. PAPER mode, LIVE_TRADING_ENABLED=false, no orders placed. No DTE policy is applied "
                      "to production S2. Real NIFTY and option candles only; costs/slippage excluded."]
    return "\n".join(md) + "\n"


def _csv(path: Path, fields: Sequence[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _pf(v) if k.endswith("profit_factor") else v for k, v in r.items()})


GROUP = ("signals", "evaluated", "wins", "losses", "ambiguous", "scratch", "win_rate", "profit_factor", "net_pnl")


def write_reports(a: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Writes only the six DTE-lab files; nothing else in *out_dir* is touched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = a["_rows"]
    public = {k: v for k, v in a.items() if not k.startswith("_")}
    p = {n: out_dir / n for n in ("summary.md", "summary.json", "summary.csv", "daily.csv", "dte_resolution.csv",
                                   "regime.csv")}
    p["summary.md"].write_text(summary_markdown(public, rows), encoding="utf-8")
    p["summary.json"].write_text(json.dumps(sh._jsonable({**public, "resolution_sample": resolution_sample(rows)}),
                                            indent=2, default=str), encoding="utf-8")
    flat = ("signals", "evaluated", "wins", "losses", "ambiguous", "scratch", "expiry_unavailable",
            "data_unavailable_other", "skipped", "win_rate", "profit_factor", "net_pnl", "avg_pnl_per_trade",
            "max_drawdown", "profitable_days", "losing_days", "data_coverage_ex_expiry")
    _csv(p["summary.csv"], ("policy", "period", *flat, "best_day", "worst_day", "lookahead_passed", "status"),
         [{"policy": n, "period": period, **{k: m[k] for k in flat},
           "best_day": (m["best_day"] or {}).get("date"), "worst_day": (m["worst_day"] or {}).get("date"),
           "lookahead_passed": a["lookahead"][n]["passed"], "status": a["status"][n]["status"]}
          for period, by in a["metrics"].items() for n, m in by.items()])
    _csv(p["daily.csv"], tuple(a["daily"][0]) if a["daily"] else ("date",), a["daily"])
    _csv(p["dte_resolution.csv"], RES_FIELDS,
         [{"policy": n, **r} for n, rs in rows.items() for r in rs])
    _csv(p["regime.csv"], ("policy", "period", "regime", *GROUP),
         [{"policy": n, "period": period, "regime": b, **{k: x[k] for k in GROUP if k in x}}
          for period in ("ALL", "OOS") for n, m in a["metrics"][period].items() for b, x in m["regime"].items()])
    return p
