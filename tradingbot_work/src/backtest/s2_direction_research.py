"""S2 direction comparison (research only, PAPER, production S2 unchanged).

Five predeclared candidates built from the unchanged S2 signal stream:

1. BASELINE_S2_BUY_SELL - every S2 signal
2. SELL_ONLY            - S2 SELL signals only
3. BUY_ONLY             - S2 BUY signals only
4. BUY_CONFIRMED        - S2 BUY signals that also pass the stronger confirmation
5. SELL_CONFIRMED       - S2 SELL signals that also pass the stronger confirmation

Stronger confirmation (fixed before any result is computed; reads only underlying candles <= the signal candle):
  a. two-close confirmation - the previous candle of the same session also closed beyond the same opening-range side;
  b. the signal candle body points in the trade direction (close > open for BUY, close < open for SELL);
  c. trend alignment - the signal close is on the trade side of a causal EMA20 (adjust=False) of the underlying.

Candidates only remove signals. Contract resolution, option entry (next option candle open), look-ahead-safe
premium ATR, stop (1.5 x premium ATR), target (2R), first-touch evaluation, square-off, metrics, splits and status
classification are reused unchanged from ``s2_1dte_improvement``. Missing option data is never synthesised: such
signals stay in the sample as DATA_UNAVAILABLE and are excluded from win rate / P&L but counted and explained.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import s2_1dte_improvement as imp
from src.backtest.current_day_s2 import LevelParams

IST = imp.IST
SPLITS = imp.SPLITS
EMA_PERIOD = 20
DIRECTION_SKIP = "DIRECTION_EXCLUDED"
CONFIRM_UNAVAILABLE = "CONFIRMATION_DATA_UNAVAILABLE"


@dataclass(frozen=True)
class DirectionCandidate:
    name: str
    directions: tuple[str, ...]
    confirm: bool = False

    def describe(self) -> list[str]:
        dirs = " + ".join(self.directions)
        conf = ("stronger confirmation: previous candle also closed beyond the same ORB side, signal candle body in "
                f"trade direction, close on trade side of EMA{EMA_PERIOD}") if self.confirm else "no extra confirmation"
        return [f"directions: {dirs}", conf, "S2 exit unchanged: stop 1.50 x premium ATR, target 2R"]


BASELINE = DirectionCandidate("BASELINE_S2_BUY_SELL", ("BUY", "SELL"))


def predeclared_candidates() -> list[DirectionCandidate]:
    return [BASELINE, DirectionCandidate("SELL_ONLY", ("SELL",)), DirectionCandidate("BUY_ONLY", ("BUY",)),
            DirectionCandidate("BUY_CONFIRMED", ("BUY",), confirm=True),
            DirectionCandidate("SELL_CONFIRMED", ("SELL",), confirm=True)]


# ------------------------------------------------------------------ features
def confirmation_features(features: pd.DataFrame) -> pd.DataFrame:
    """Causal per-bar confirmation inputs: previous same-session bar, EMA20 and its warm-up count."""
    f = features.copy()
    f["ema20"] = f["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    f["bars_seen"] = range(1, len(f) + 1)
    g = f.groupby("day")
    for col in ("close", "orb_high", "orb_low"):
        f[f"prev_{col}"] = g[col].shift(1)
    return f.set_index("timestamp")


def confirmation(ctx: dict, conf: pd.DataFrame) -> tuple[bool | None, dict[str, Any]]:
    """(passed, detail). ``None`` means the inputs were not available at the signal candle."""
    ts, d = ctx["signal_time"], ctx["direction"]
    if ts not in conf.index:
        return None, {"reason": "SIGNAL_BAR_MISSING"}
    bar = conf.loc[ts]
    need = ("open", "close", "ema20", "prev_close", "prev_orb_high", "prev_orb_low")
    if bar["bars_seen"] < EMA_PERIOD or any(pd.isna(bar[k]) for k in need):
        return None, {"reason": "INSUFFICIENT_HISTORY"}
    if d == "BUY":
        checks = {"two_close": bar["prev_close"] > bar["prev_orb_high"], "body": bar["close"] > bar["open"],
                  "ema": bar["close"] > bar["ema20"]}
    else:
        checks = {"two_close": bar["prev_close"] < bar["prev_orb_low"], "body": bar["close"] < bar["open"],
                  "ema": bar["close"] < bar["ema20"]}
    checks = {k: bool(v) for k, v in checks.items()}
    return all(checks.values()), checks


def direction_filter(contexts: Sequence[dict], features: pd.DataFrame,
                     cand: DirectionCandidate) -> list[str | None]:
    """Skip reason per context (None = accepted)."""
    conf = confirmation_features(features) if cand.confirm else None
    out: list[str | None] = []
    for ctx in contexts:
        if ctx["direction"] not in cand.directions:
            out.append(DIRECTION_SKIP)
            continue
        if conf is None:
            out.append(None)
            continue
        ok, detail = confirmation(ctx, conf)
        if ok is None:
            out.append(f"{CONFIRM_UNAVAILABLE}:{detail['reason']}")
        elif not ok:
            out.append("CONFIRMATION_FAILED:" + ",".join(k for k, v in detail.items() if not v))
        else:
            out.append(None)
    return out


_OUTCOME_KEYS = ("outcome", "pnl", "r_multiple", "exit_reason", "exit_time", "exit_price", "stop_loss", "target")


def run_candidate(contexts: Sequence[dict], features: pd.DataFrame, cand: DirectionCandidate, *, square_off: str,
                  lot_size: int = 65, cache: dict | None = None) -> list[dict[str, Any]]:
    """Baseline S2 rows with the candidate's direction/confirmation skips applied; evaluation is unchanged."""
    rows = imp.run_variant(contexts, features, imp.BASELINE, square_off=square_off, lot_size=lot_size, cache=cache)
    for row, skip in zip(rows, direction_filter(contexts, features, cand)):
        row["variant"] = cand.name
        if skip is not None:
            row.update(decision="SKIPPED", skip_reason=skip, **{k: None for k in _OUTCOME_KEYS})
    return rows


def unavailable_reasons(rows: Sequence[dict]) -> dict[str, int]:
    return imp._count(r["data_reason"] or r.get("exit_reason") for r in rows
                      if r["decision"] == imp.ACCEPTED and r.get("outcome") in (imp.UNAVAILABLE, imp.PENDING))


def candidate_metrics(rows: Sequence[dict]) -> dict[str, Any]:
    m = imp.compute_metrics(rows)
    m["unavailable_reasons"] = unavailable_reasons(rows)
    return m


# ---------------------------------------------------------------- look-ahead
def fingerprint(contexts: Sequence[dict], features: pd.DataFrame, cand: DirectionCandidate, upto: pd.Timestamp,
                day: date) -> list[tuple]:
    out = []
    for ctx, skip in zip(contexts, direction_filter(contexts, features, cand)):
        if ctx["session_date"] != day or ctx["signal_time"] > upto:
            continue
        out.append((ctx["signal_time"].isoformat(), ctx["direction"], skip, ctx["instrument_key"],
                    None if ctx["premium_atr"] is None else round(ctx["premium_atr"], 9),
                    ctx["entry_time"] is None or ctx["entry_time"] > ctx["signal_time"]))
    return out


def mutation_test(pipeline: Callable, spot: pd.DataFrame, premium_for: Callable,
                  candidates: Sequence[DirectionCandidate], cuts: Sequence[pd.Timestamp],
                  seed: int = 11) -> dict[str, dict]:
    """Rescale every underlying/premium candle after each cut, re-run, and require identical earlier decisions."""
    result = {c.name: {"cuts": 0, "failures": []} for c in candidates}
    for i, cut in enumerate(cuts):
        cut = imp._ist(cut)
        day = cut.date()
        base_ctx, base_feat = pipeline(spot, premium_for, {day})
        mut_ctx, mut_feat = pipeline(imp.mutate_after(spot, cut, seed + i),
                                     lambda c, _cut=cut, _i=i: imp._mut_prem(premium_for(c), _cut, seed + _i), {day})
        for c in candidates:
            a = fingerprint(base_ctx, base_feat, c, cut, day)
            b = fingerprint(mut_ctx, mut_feat, c, cut, day)
            result[c.name]["cuts"] += 1
            if a != b or not all(x[-1] for x in a):
                result[c.name]["failures"].append(cut.isoformat())
    for r in result.values():
        r["passed"] = not r["failures"]
    return result


# ------------------------------------------------------------------ the study
LIMITATIONS = [
    "Historical option outcomes use the 2026-10-06 weekly at 5-35 DTE (NON-COMPARABLE_1DTE); expired weekly candles "
    "are unavailable on this Upstox plan. Only the current day is 1-DTE.",
    "The historical window (22 sessions, September 2026) was mostly a falling market, which favours SELL (long PE) "
    "signals; a direction edge measured here may be a market-regime effect rather than a property of S2.",
    "Only one comparable 1-DTE session exists; its signals are highly correlated, so the effective sample is small.",
    "Every accepted signal is evaluated independently with 1 lot (overlap allowed, as in the live tracker); costs "
    "and slippage are excluded; ambiguous candles count as losses in the win rate.",
    "Signals whose option premium data is missing stay DATA_UNAVAILABLE; they are not synthesised and are excluded "
    "from win rate and P&L.",
]


def run_direction_research(*, hist_spot: pd.DataFrame, signal_fn: Callable[[pd.DataFrame], list], resolve: Callable,
                           hist_premium_for: Callable, params: LevelParams, square_off: str = "15:15",
                           range_start: str = "09:15", range_end: str = "09:30",
                           current_spot: pd.DataFrame | None = None, current_premium_for: Callable | None = None,
                           current_day: date | None = None,
                           splits: dict[str, tuple[str, str]] = imp.DEFAULT_SPLITS,
                           criteria: imp.Criteria = imp.Criteria(), hist_cut_time: str = "12:00",
                           current_cut_every_min: int = 15) -> dict[str, Any]:
    sp = imp.split_dates(splits)
    hist_days = {d for d in pd.to_datetime(hist_spot["timestamp"]).dt.date if imp.split_of(d, sp)}
    pipeline = imp._pipeline_factory(signal_fn, resolve, params, range_start, range_end)
    hist = pipeline(hist_spot, hist_premium_for, hist_days)
    cur = pipeline(current_spot, current_premium_for, {current_day}) if current_spot is not None else None
    cache: dict = {}
    candidates = predeclared_candidates()

    results = []
    for cand in candidates:
        rows_h = run_candidate(hist[0], hist[1], cand, square_off=square_off, cache=cache)
        for r in rows_h:
            r["split"] = imp.split_of(r["session_date"], sp)
        m = {s: candidate_metrics([r for r in rows_h if r["split"] == s]) for s in SPLITS}
        m["HISTORICAL_ALL"] = candidate_metrics([r for r in rows_h if r["split"] in SPLITS])
        if cur is not None:
            m["CURRENT_DAY"] = candidate_metrics(run_candidate(cur[0], cur[1], cand, square_off=square_off,
                                                               cache=cache))
        results.append({"candidate": cand, "metrics": m})

    hist_cuts = [pd.Timestamp(datetime.combine(d, time.fromisoformat(hist_cut_time)), tz=IST)
                 for d in sorted(hist_days)]
    la_hist = mutation_test(pipeline, hist_spot, hist_premium_for, candidates, hist_cuts)
    la_cur = {c.name: {"cuts": 0, "failures": [], "passed": True} for c in candidates}
    cur_cuts: list = []
    if cur is not None:
        start = pd.Timestamp(datetime.combine(current_day, time(9, 35)), tz=IST)
        cur_cuts = [start + pd.Timedelta(minutes=current_cut_every_min * i) for i in range(300 // current_cut_every_min + 1)
                    if (start + pd.Timedelta(minutes=current_cut_every_min * i)).time() <= time(14, 30)]
        la_cur = mutation_test(pipeline, current_spot, current_premium_for, candidates, cur_cuts)

    base_cur_wr = (results[0]["metrics"].get("CURRENT_DAY") or {}).get("win_rate")
    out = []
    for r in results:
        c = r["candidate"]
        ok = la_hist[c.name]["passed"] and la_cur[c.name]["passed"]
        status, notes = imp.classify(r["metrics"], r["metrics"].get("CURRENT_DAY"), base_cur_wr, ok, criteria)
        if c is BASELINE and status != "LOOKAHEAD_FAIL":
            status = "BASELINE"
        out.append({"name": c.name, "group": "DIRECTION", "configuration": c.describe(), "candidate": asdict(c),
                    "metrics": r["metrics"], "status": status, "status_notes": notes, "lookahead_passed": ok,
                    "lookahead": {"historical": la_hist[c.name], "current_day": la_cur[c.name]}})

    report = {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "mode": "Research only - TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, no orders, production S2 unchanged",
        "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in sp.items()},
        "labels": {"historical": imp.NON_COMPARABLE, "current_day": imp.COMPARABLE},
        "current_day": None if current_day is None else current_day.isoformat(),
        "criteria": asdict(criteria),
        "evaluation_method": ("Unchanged S2 signals; candidates only remove signals. Option entry at next candle "
                              "open, premium ATR from candles <= signal candle, stop 1.5 x premium ATR, target 2R, "
                              f"first touch on real premium candles, square-off {square_off}, 1 lot (65), costs "
                              "excluded, ambiguous counted as loss."),
        "confirmation_rule": BUY_SELL_CONFIRMATION_TEXT,
        "predeclared_candidates": [{"name": c.name, "configuration": c.describe()} for c in candidates],
        "results": out,
        "lookahead": {"cuts": len(hist_cuts) + len(cur_cuts),
                      "cut_summary": f"{len(hist_cuts)} historical at {hist_cut_time}, {len(cur_cuts)} current-day "
                                     f"every {current_cut_every_min} min",
                      "candidates": len(out), "passed": sum(1 for r in out if r["lookahead_passed"]),
                      "failed": [r["name"] for r in out if not r["lookahead_passed"]]},
        "limitations": LIMITATIONS,
    }
    report["summary"] = summarize(report)
    return report


BUY_SELL_CONFIRMATION_TEXT = (
    "A BUY (SELL) passes only if (a) the previous candle of the same session closed above the ORB high (below the "
    "ORB low), (b) the signal candle closed above (below) its open, and (c) the signal close is above (below) a "
    f"causal EMA{EMA_PERIOD} of the underlying close. Signals with fewer than {EMA_PERIOD} prior bars or no previous "
    "in-range bar are skipped as CONFIRMATION_DATA_UNAVAILABLE.")


def summarize(rep: dict) -> dict[str, Any]:
    res = {r["name"]: r for r in rep["results"]}
    base = res[BASELINE.name]["metrics"]
    order = {s: i for i, s in enumerate(imp.STATUS_ORDER + ("BASELINE",))}
    ranked = sorted((r for r in rep["results"] if r["name"] != BASELINE.name),
                    key=lambda r: (order[r["status"]], -(r["metrics"]["VALIDATION"]["win_rate"] or 0),
                                   -(r["metrics"]["TRAIN"]["win_rate"] or 0), r["name"]))
    robust = [r["name"] for r in ranked if r["status"] == "ROBUST"]
    lines = []
    for r in rep["results"]:
        m = r["metrics"]
        cur = m.get("CURRENT_DAY") or {}
        lines.append(f"{r['name']} ({r['status']}): TRAIN/VALIDATION/OOS {imp._pct(m['TRAIN']['win_rate'])}/"
                     f"{imp._pct(m['VALIDATION']['win_rate'])}/{imp._pct(m['OOS']['win_rate'])}, OOS PF "
                     f"{imp._fmt_pf(m['OOS']['profit_factor'])}, current day {imp._pct(cur.get('win_rate'))} "
                     f"(n={cur.get('evaluated', 'n/a')}).")
    if robust:
        rec = (f"{robust[0]} is ROBUST on NON-COMPARABLE_1DTE history; it is a candidate for further PAPER "
               "validation only. Production S2 is not changed.")
    else:
        rec = "No direction candidate is ROBUST. Keep production S2 (BUY + SELL) unchanged."
    return {"lines": lines, "ranking": [r["name"] for r in ranked], "robust": robust, "recommendation": rec,
            "baseline_current_win_rate": (base.get("CURRENT_DAY") or {}).get("win_rate")}


# ---------------------------------------------------------------- reporting
REPORT_SPLITS = ("TRAIN", "VALIDATION", "OOS", "HISTORICAL_ALL", "CURRENT_DAY")
CSV_FIELDS = ("s2_signals", "accepted_signals", "evaluated", "wins", "losses", "ambiguous", "pending_or_unavailable",
              "win_rate", "profit_factor", "net_pnl", "max_drawdown", "average_r")


def csv_rows(results: Sequence[dict]) -> list[dict[str, Any]]:
    out = []
    for r in results:
        row = {"candidate": r["name"], "configuration": " | ".join(r["configuration"]), "status": r["status"],
               "lookahead_passed": r["lookahead_passed"]}
        for s in REPORT_SPLITS:
            m, p = r["metrics"].get(s), s.lower()
            row[f"{p}_label"] = imp.COMPARABLE if s == "CURRENT_DAY" else imp.NON_COMPARABLE
            for k in CSV_FIELDS:
                row[f"{p}_{k}"] = None if m is None else imp._jsonable(m[k])
            for d in ("BUY", "SELL"):
                for k in ("evaluated", "wins", "losses", "accuracy", "profit_factor", "net_pnl"):
                    row[f"{p}_{d.lower()}_{k}"] = None if m is None else imp._jsonable(m[d][k])
            row[f"{p}_unavailable_reasons"] = None if m is None else json.dumps(m["unavailable_reasons"])
        out.append(row)
    return out


def _row(name: str, m: dict | None) -> str:
    if not m:
        return f"| {name} | n/a |" + " |" * 12
    avg = "n/a" if m["average_r"] is None else f"{m['average_r']:.2f}"
    return (f"| {name} | {m['accepted_signals']} | {m['evaluated']} | {m['wins']} | {m['losses'] + m['ambiguous']} | "
            f"{imp._pct(m['win_rate'])} | {imp._fmt_pf(m['profit_factor'])} | {m['net_pnl']:,.0f} | "
            f"{m['max_drawdown']:,.0f} | {avg} | {m['BUY']['wins']}/{m['BUY']['evaluated']} | "
            f"{m['SELL']['wins']}/{m['SELL']['evaluated']} | {m['pending_or_unavailable']} |")


HEAD = ["| Sample | Signals | Evaluated | Wins | Losses | Win rate | PF | Net P&L | Max DD | Avg R | BUY W/n | "
        "SELL W/n | Unavailable |", "|---|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|"]


def render_markdown(rep: dict) -> str:
    sd, summ = rep["splits"], rep["summary"]
    md = ["# S2 direction research", "", f"Generated {rep['generated_at']}. {rep['mode']}.", "",
          "## Summary", ""] + [f"- {x}" for x in summ["lines"]] + ["", f"**{summ['recommendation']}**", "",
          "## Method", "", f"- {rep['evaluation_method']}", f"- Confirmation rule: {rep['confirmation_rule']}",
          f"- Splits ({imp.NON_COMPARABLE}): TRAIN {sd['TRAIN'][0]}..{sd['TRAIN'][1]}, VALIDATION "
          f"{sd['VALIDATION'][0]}..{sd['VALIDATION'][1]}, OOS {sd['OOS'][0]}..{sd['OOS'][1]}. Current day "
          f"{rep['current_day']} is {imp.COMPARABLE}.",
          "- Losses include ambiguous candles. Unavailable = accepted signals without usable option data or exit "
          "evidence (not synthesised).", "",
          "## Comparison", "",
          "| Candidate | Train WR (n) | Validation WR (n) | OOS WR (n) | Current 1-DTE WR (n) | OOS PF | Net P&L (hist) "
          "| Max DD (hist) | Status |", "|---|---|---|---|---|---|---:|---:|---|"]
    for r in rep["results"]:
        m = r["metrics"]
        cur = m.get("CURRENT_DAY") or {}
        md.append(f"| {r['name']} | " + " | ".join(f"{imp._pct(m[s]['win_rate'])} ({m[s]['evaluated']})"
                                                   for s in SPLITS)
                  + f" | {imp._pct(cur.get('win_rate'))} ({cur.get('evaluated', 'n/a')}) | "
                    f"{imp._fmt_pf(m['OOS']['profit_factor'])} | {m['HISTORICAL_ALL']['net_pnl']:,.0f} | "
                    f"{m['HISTORICAL_ALL']['max_drawdown']:,.0f} | {r['status']} |")
    md.append("")
    for r in rep["results"]:
        md += [f"## {r['name']}", ""] + [f"- {x}" for x in r["configuration"]]
        md += [f"- Status notes: {'; '.join(r['status_notes'])}", ""] + HEAD
        md += [_row(s, r["metrics"].get(s)) for s in REPORT_SPLITS]
        reasons = {s: r["metrics"][s]["unavailable_reasons"] for s in REPORT_SPLITS
                   if r["metrics"].get(s) and r["metrics"][s]["unavailable_reasons"]}
        skips = {s: r["metrics"][s]["skip_reasons"] for s in REPORT_SPLITS
                 if r["metrics"].get(s) and r["metrics"][s]["skip_reasons"]}
        md += ["", f"- Skipped by candidate rule: {skips or 'none'}", f"- Unavailable reasons: {reasons or 'none'}", ""]
    la = rep["lookahead"]
    md += ["## Look-ahead test", "",
           f"- {la['cuts']} cuts ({la['cut_summary']}): all underlying and premium candles after each cut randomly "
           "rescaled and the pipeline re-run; accept/skip decision, contract and premium ATR for every signal at or "
           "before the cut must be identical and every entry strictly after its signal.",
           f"- {la['passed']} of {la['candidates']} candidates passed; failed: {la['failed'] or 'none'}.", "",
           "## Limitations", ""] + [f"- {x}" for x in rep["limitations"]]
    return "\n".join(md) + "\n"


def write_reports(report: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / "s2_direction_research.json", "csv": out_dir / "s2_direction_candidates.csv",
             "md": out_dir / "s2_direction_summary.md"}
    paths["json"].write_text(json.dumps(imp._jsonable(report), indent=2), encoding="utf-8")
    rows = csv_rows(report["results"])
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    return paths
