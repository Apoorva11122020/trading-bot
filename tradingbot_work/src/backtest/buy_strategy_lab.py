"""BUY STRATEGY RESEARCH LAB for the original S2 BUY signal (PAPER, research only, no orders).

Twelve pre-declared BUY confirmation candidates are applied as skip filters on top of the unchanged S2 BUY signals.
Entry, contract (nearest-expiry ATM CE), stop 1.5 x premium ATR, target 2R and evaluation are the existing ones;
SELL signals are never evaluated here. Every confirmation input is read from the completed signal candle or earlier
candles. Ranking reads only OOS metrics; status never says ROBUST because the OOS window was already seen.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import s2_1dte_improvement as imp
from src.backtest import s2_direction_research as dr
from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.indicators.technical import ema, rsi

HEADING = "BUY STRATEGY RESEARCH LAB"
ORIGINAL = dr.BASELINE.name
EMA_FAST, EMA_SLOW, RSI_PERIOD, RSI_MIN = 20, 50, 14, 55.0
VOLUME_PERIOD, VOLUME_MULT = 20, 1.2
BREAKOUT_MAX_ATR = 0.50
BREAKOUT_NOTE = ("S2 has no ATR breakout rule of its own; the project's existing pre-declared ATR breakout threshold "
                 "(ENTRY_MAX_BREAK_0.50ATR, breakout distance <= 0.50 x ATR14) is used.")
PROMISING, NOT_PROMISING, INSUFFICIENT_DATA = "PROMISING", "NOT_PROMISING", "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class BuyCandidate:
    name: str
    checks: tuple[str, ...] = ()


CANDIDATES = (
    BuyCandidate("BUY_BASELINE"),
    BuyCandidate("BUY_EMA_CONFIRM", ("ema",)),
    BuyCandidate("BUY_CANDLE_CONFIRM", ("candle",)),
    BuyCandidate("BUY_VOLUME_CONFIRM", ("volume",)),
    BuyCandidate("BUY_RSI_CONFIRM", ("rsi",)),
    BuyCandidate("BUY_BREAKOUT_CONFIRM", ("breakout",)),
    BuyCandidate("BUY_EMA_CANDLE", ("ema", "candle")),
    BuyCandidate("BUY_EMA_VOLUME", ("ema", "volume")),
    BuyCandidate("BUY_EMA_RSI", ("ema", "rsi")),
    BuyCandidate("BUY_EMA_CANDLE_VOLUME", ("ema", "candle", "volume")),
    BuyCandidate("BUY_EMA_CANDLE_RSI", ("ema", "candle", "rsi")),
    BuyCandidate("BUY_STRONG_CONFIRMATION", ("strong",)),
)
CHECK_TEXT = {
    "ema": f"close > EMA{EMA_FAST} and EMA{EMA_FAST} > EMA{EMA_SLOW}",
    "candle": "previous completed same-session candle bullish and signal candle bullish",
    "volume": f"signal candle volume >= {VOLUME_MULT} x average volume of the previous {VOLUME_PERIOD} candles",
    "rsi": f"RSI{RSI_PERIOD} >= {RSI_MIN:g}",
    "breakout": f"close above the ORB high and breakout distance <= {BREAKOUT_MAX_ATR:.2f} x ATR14",
    "strong": f"previous candle closed above the ORB high, signal candle body bullish, close > EMA{EMA_FAST}, "
              f"EMA{EMA_FAST} > EMA{EMA_SLOW}",
}


@dataclass(frozen=True)
class LabCriteria:
    min_oos_trades: int = 30
    min_validation_wr: float = 0.45
    min_oos_wr: float = 0.45
    reasonable_train_wr: float = 0.45
    min_oos_pf: float = 1.10
    min_data_coverage: float = 0.90


# ------------------------------------------------------------------ causal features
def buy_features(spot: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Per-bar inputs known at each bar's close: EMAs, RSI, previous bar, ORB, prior-candle average volume."""
    cols = ["timestamp", "open", "high", "low", "close"] + (["volume"] if "volume" in spot else [])
    f = spot[cols].copy()
    ts = pd.to_datetime(f["timestamp"])
    f["timestamp"] = ts.dt.tz_localize(imp.IST) if ts.dt.tz is None else ts.dt.tz_convert(imp.IST)
    f = f.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    if "volume" not in f:
        f["volume"] = float("nan")
    f["day"] = f["timestamp"].dt.date
    f["ema20"] = ema(f["close"], EMA_FAST)
    f["ema50"] = ema(f["close"], EMA_SLOW)
    f["rsi14"] = rsi(f["close"], RSI_PERIOD)
    vol = f["volume"].astype(float).where(f["volume"].astype(float) > 0)
    f["vol_avg"] = vol.shift(1).rolling(VOLUME_PERIOD, min_periods=VOLUME_PERIOD).mean()
    orb = features[["timestamp", "orb_high"]].drop_duplicates("timestamp", keep="last")
    f = f.merge(orb, on="timestamp", how="left")
    g = f.groupby("day")
    for col in ("open", "close", "orb_high"):
        f[f"prev_{col}"] = g[col].shift(1)
    return f.set_index("timestamp")


def _missing(*values) -> bool:
    return any(v is None or pd.isna(v) for v in values)


def check(name: str, ctx: dict, bar: pd.Series) -> bool | None:
    """True/False, or None when the input was not available at the signal candle."""
    if name == "ema":
        if _missing(bar["ema20"], bar["ema50"]):
            return None
        return bool(bar["close"] > bar["ema20"] and bar["ema20"] > bar["ema50"])
    if name == "candle":
        if _missing(bar["prev_open"], bar["prev_close"]):
            return None
        return bool(bar["prev_close"] > bar["prev_open"] and bar["close"] > bar["open"])
    if name == "volume":
        if _missing(bar["volume"], bar["vol_avg"]) or bar["volume"] <= 0 or bar["vol_avg"] <= 0:
            return None
        return bool(bar["volume"] >= VOLUME_MULT * bar["vol_avg"])
    if name == "rsi":
        return None if _missing(bar["rsi14"]) else bool(bar["rsi14"] >= RSI_MIN)
    if name == "breakout":
        if _missing(ctx.get("breakout_level"), ctx.get("break_atr")):
            return None
        return bool(bar["close"] > ctx["breakout_level"] and ctx["break_atr"] <= BREAKOUT_MAX_ATR)
    if name == "strong":
        if _missing(bar["prev_close"], bar["prev_orb_high"], bar["ema20"], bar["ema50"]):
            return None
        return bool(bar["prev_close"] > bar["prev_orb_high"] and bar["close"] > bar["open"]
                    and bar["close"] > bar["ema20"] and bar["ema20"] > bar["ema50"])
    raise ValueError(f"unknown check {name}")


def buy_filter(contexts: Sequence[dict], conf: pd.DataFrame, cand: BuyCandidate) -> list[str | None]:
    """Skip reason per context (None = accepted). SELL contexts are always skipped."""
    out: list[str | None] = []
    for ctx in contexts:
        if ctx["direction"] != "BUY":
            out.append(dr.DIRECTION_SKIP)
            continue
        if not cand.checks:
            out.append(None)
            continue
        if ctx["signal_time"] not in conf.index:
            out.append(f"{dr.CONFIRM_UNAVAILABLE}:SIGNAL_BAR_MISSING")
            continue
        bar = conf.loc[ctx["signal_time"]]
        results = {c: check(c, ctx, bar) for c in cand.checks}
        missing = [c for c, v in results.items() if v is None]
        failed = [c for c, v in results.items() if v is False]
        out.append(f"{dr.CONFIRM_UNAVAILABLE}:{','.join(missing)}" if missing else
                   "CONFIRMATION_FAILED:" + ",".join(failed) if failed else None)
    return out


def run_candidate(contexts: Sequence[dict], features: pd.DataFrame, conf: pd.DataFrame, cand: BuyCandidate, *,
                  square_off: str, regimes: dict | None = None, cache: dict | None = None) -> list[dict]:
    """Classified BUY rows: unchanged S2 evaluation, with the candidate's confirmation skips applied."""
    rows = imp.run_variant(contexts, features, imp.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE, cache=cache)
    out, seen = [], set()
    for row, skip in zip(rows, buy_filter(contexts, conf, cand)):
        if row["direction"] != "BUY":
            continue
        if skip is not None:
            row.update(decision="SKIPPED", skip_reason=skip, **{k: None for k in dr._OUTCOME_KEYS})
        key = (row["signal_time"], row["direction"])
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(sh.classify_row(row, regimes), strategy=cand.name))
    return out


# ------------------------------------------------------------------ look-ahead
def _fingerprint(contexts, conf, cand, cut, day) -> list[tuple]:
    out = []
    for ctx, skip in zip(contexts, buy_filter(contexts, conf, cand)):
        if ctx["direction"] != "BUY" or ctx["session_date"] != day or ctx["signal_time"] > cut:
            continue
        out.append((ctx["signal_time"].isoformat(), skip, ctx["instrument_key"],
                    None if ctx["premium_atr"] is None else round(ctx["premium_atr"], 9),
                    ctx["entry_time"] is None or ctx["entry_time"] > ctx["signal_time"]))
    return out


def buy_mutation_test(pipeline: Callable, spot: pd.DataFrame, premium_for: Callable,
                      candidates: Sequence[BuyCandidate], cuts: Sequence[pd.Timestamp], seed: int = 11) -> dict:
    """Rescale every underlying/premium candle after each cut; earlier BUY decisions must not change."""
    result = {c.name: {"cuts": 0, "failures": []} for c in candidates}
    for i, cut in enumerate(cuts):
        cut = imp._ist(cut)
        day = cut.date()
        base_ctx, base_feat = pipeline(spot, premium_for, {day})
        mut_spot = imp.mutate_after(spot, cut, seed + i)
        mut_ctx, mut_feat = pipeline(mut_spot, lambda c, _cut=cut, _i=i: imp._mut_prem(premium_for(c), _cut, seed + _i),
                                     {day})
        base_conf, mut_conf = buy_features(spot, base_feat), buy_features(mut_spot, mut_feat)
        for c in candidates:
            a = _fingerprint(base_ctx, base_conf, c, cut, day)
            b = _fingerprint(mut_ctx, mut_conf, c, cut, day)
            result[c.name]["cuts"] += 1
            if a != b or not all(x[-1] for x in a):
                result[c.name]["failures"].append(cut.isoformat())
    for r in result.values():
        r["passed"] = bool(r["cuts"]) and not r["failures"]
    return result


# ------------------------------------------------------------------ status / ranking (OOS only)
def candidate_status(train: dict, val: dict, oos: dict, baseline_train: dict, lookahead_passed: bool,
                     c: LabCriteria = LabCriteria()) -> dict[str, Any]:
    twr, bwr = train["win_rate"], baseline_train["win_rate"]
    checks = {
        "oos_trades": (oos["evaluated"] >= c.min_oos_trades, oos["evaluated"], f">= {c.min_oos_trades}"),
        "train_wr_improves_or_reasonable": (twr is not None and ((bwr is not None and twr >= bwr)
                                                                 or twr >= c.reasonable_train_wr),
                                            twr, f">= baseline train WR or >= {c.reasonable_train_wr:.0%}"),
        "validation_wr": (val["win_rate"] is not None and val["win_rate"] >= c.min_validation_wr, val["win_rate"],
                          f">= {c.min_validation_wr:.0%}"),
        "oos_wr": (oos["win_rate"] is not None and oos["win_rate"] >= c.min_oos_wr, oos["win_rate"],
                   f">= {c.min_oos_wr:.0%}"),
        "oos_pf": (oos["profit_factor"] is not None and oos["profit_factor"] > c.min_oos_pf, oos["profit_factor"],
                   f"> {c.min_oos_pf}"),
        "oos_pnl": (oos["net_pnl"] > 0, oos["net_pnl"], "> 0"),
        "lookahead": (lookahead_passed, lookahead_passed, "PASS"),
        "no_major_data_gaps": (oos["data_coverage"] is not None and oos["data_coverage"] >= c.min_data_coverage,
                               oos["data_coverage"], f">= {c.min_data_coverage:.0%}"),
    }
    checks = {k: {"passed": bool(p), "value": v, "threshold": t} for k, (p, v, t) in checks.items()}
    failed = [k for k, v in checks.items() if not v["passed"]]
    status = (INSUFFICIENT_DATA if not checks["oos_trades"]["passed"] else
              PROMISING if not failed else NOT_PROMISING)
    return {"status": status, "failed_checks": failed, "checks": checks}


def rank_by_oos(oos: dict[str, dict], min_trades: int = LabCriteria.min_oos_trades) -> list[str]:
    """OOS PF, OOS P&L, lower OOS DD, OOS WR, OOS trades; candidates below the sample minimum rank last."""
    def key(name):
        m = oos[name]
        pf = -1.0 if m["profit_factor"] is None else m["profit_factor"]
        wr = -1.0 if m["win_rate"] is None else m["win_rate"]
        return (m["evaluated"] < min_trades, -pf, -m["net_pnl"], m["max_drawdown"], -wr, -m["evaluated"])
    return sorted(oos, key=key)


def _without_day(rows: Sequence[dict], day: str | None) -> dict[str, Any]:
    m = sh.stats(r for r in rows if r["session_date"] != day)
    return {"excluded_date": day, **{k: m[k] for k in ("evaluated", "win_rate", "profit_factor", "net_pnl",
                                                         "max_drawdown")}}


def _delta(a, b):
    if a is None or b is None or (isinstance(a, float) and math.isinf(a)) or (isinstance(b, float) and math.isinf(b)):
        return None
    return round(b - a, 4)


def comparison(base: dict, best: dict) -> dict[str, Any]:
    keys = ("win_rate", "profit_factor", "net_pnl", "max_drawdown", "evaluated")
    out = {"baseline": {k: base[k] for k in keys}, "candidate": {k: best[k] for k in keys}}
    out["wr_change_pp"] = None if _delta(base["win_rate"], best["win_rate"]) is None else \
        round((best["win_rate"] - base["win_rate"]) * 100, 2)
    out["pf_change"] = _delta(base["profit_factor"], best["profit_factor"])
    out["pnl_change"] = _delta(base["net_pnl"], best["net_pnl"])
    out["dd_change"] = _delta(base["max_drawdown"], best["max_drawdown"])
    out["dd_improved"] = best["max_drawdown"] < base["max_drawdown"]
    out["trade_count_change"] = best["evaluated"] - base["evaluated"]
    return out


def skip_reasons(rows: Sequence[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        if r["state"] == sh.SKIPPED:
            out[r["skip_reason"]] = out.get(r["skip_reason"], 0) + 1
    return out


# ------------------------------------------------------------------ study
def build_lab(result: dict[str, Any], meta: dict[str, Any], *, splits: dict, criteria: LabCriteria = LabCriteria(),
              cut_times: Sequence[str] = ("12:00",)) -> dict[str, Any]:
    p = result["pipeline"]
    conf = buy_features(p["spot"], p["features"])
    split_days, unassigned = so.assign_days(result["days"], splits)
    hist_days = [d for s in imp.SPLITS for d in split_days[s]]
    rows = {c.name: [r for r in run_candidate(p["contexts"], p["features"], conf, c, square_off=p["square_off"],
                                              regimes=p["regimes"], cache=p["cache"])
                     if r["session_date"] in set(hist_days)] for c in CANDIDATES}
    original_buy = [r for r in result["rows"].get(ORIGINAL, []) if r["direction"] == "BUY"
                    and r["session_date"] in set(hist_days)]
    sig = lambda rs: [(r["signal_time"], r["state"], r.get("outcome"), r["pnl"]) for r in rs]  # noqa: E731
    baseline_matches = sig(rows["BUY_BASELINE"]) == sig(original_buy)

    periods = {**split_days, "ALL": hist_days}
    metrics = {s: {n: so.split_metrics([r for r in rs if r["session_date"] in set(days)], days)
                   for n, rs in rows.items()} for s, days in periods.items()}
    cut_days = [pd.Timestamp(d) for d in hist_days]
    cuts = [pd.Timestamp(datetime.combine(d.date(), time.fromisoformat(t)), tz=sh.IST) for d in cut_days
            for t in cut_times]
    lookahead = buy_mutation_test(p["pipeline"], p["spot"], p["premium_for"], CANDIDATES, cuts)
    by_split_la = {s: so.split_lookahead(lookahead, split_days[s]) for s in imp.SPLITS}

    names = [c.name for c in CANDIDATES]
    oos = metrics["OOS"]
    status = {n: candidate_status(metrics["TRAIN"][n], metrics["VALIDATION"][n], oos[n],
                                  metrics["TRAIN"]["BUY_BASELINE"], lookahead[n]["passed"], criteria) for n in names}
    ranking = rank_by_oos(oos, criteria.min_oos_trades)
    eligible = [n for n in ranking if n != "BUY_BASELINE" and oos[n]["evaluated"] >= criteria.min_oos_trades]
    best = eligible[0] if eligible else None
    top3 = [n for n in ranking if n != "BUY_BASELINE"][:3]
    oos_rows = {n: [r for r in rows[n] if r["session_date"] in set(split_days["OOS"])] for n in names}
    top3_check = {n: {"oos": {k: oos[n][k] for k in ("evaluated", "win_rate", "profit_factor", "net_pnl",
                                                      "max_drawdown")},
                      "best_day_removed": _without_day(oos_rows[n], (oos[n]["best_day"] or {}).get("date")),
                      "worst_day_removed": _without_day(oos_rows[n], (oos[n]["worst_day"] or {}).get("date"))}
                  for n in top3}
    regime_for = [n for n in names if status[n]["status"] == PROMISING]
    for n in ("BUY_BASELINE", best):
        if n and n not in regime_for:
            regime_for.append(n)
    table = [{"candidate": n, "train_wr": metrics["TRAIN"][n]["win_rate"],
              "validation_wr": metrics["VALIDATION"][n]["win_rate"], "oos_wr": oos[n]["win_rate"],
              "oos_pf": oos[n]["profit_factor"], "oos_pnl": oos[n]["net_pnl"], "oos_dd": oos[n]["max_drawdown"],
              "oos_trades": oos[n]["evaluated"], "status": status[n]["status"]} for n in names]
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"),
        "candidates": {c.name: [CHECK_TEXT[k] for k in c.checks] or ["original S2 BUY signal, unchanged"]
                       for c in CANDIDATES},
        "breakout_note": BREAKOUT_NOTE,
        "evaluation": "Unchanged S2 BUY: nearest-expiry ATM CE, entry next option candle open, stop 1.5 x premium "
                      "ATR, target 2R, square-off 15:15. Candidates only skip BUY signals; SELL is not evaluated.",
        "splits": {s: {"label": so.SPLIT_LABELS[s], "from": splits[s][0].isoformat(), "to": splits[s][1].isoformat(),
                       "days": split_days[s]} for s in imp.SPLITS},
        "unassigned_days": unassigned,
        "current_day_note": "Current-day (2026-10-05+) data is not used for selection or ranking.",
        "criteria": asdict(criteria),
        "never_robust_note": "No candidate is called ROBUST: the historical OOS window was already seen by earlier "
                             "research. PROMISING means worth forward shadow observation only.",
        "baseline_matches_original_s2_buy": baseline_matches,
        "metrics": metrics, "status": status, "ranking_by_oos": ranking, "best_candidate": best,
        "best_candidate_recommendable": bool(best and status[best]["status"] == PROMISING),
        "top3_day_removal_check": top3_check,
        "regime_report_for": regime_for,
        "comparison": None if best is None else {s: comparison(metrics[s]["BUY_BASELINE"], metrics[s][best])
                                                 for s in ("OOS", "ALL")},
        "lookahead": lookahead, "lookahead_by_split": by_split_la,
        "skip_reasons": {n: skip_reasons(rows[n]) for n in names},
        "final_table": table,
        "replay_meta": {k: meta.get(k) for k in ("date_range", "excluded_days", "contract_resolution",
                                                   "expired_candle_downloads")},
        "_rows": rows,
    }


# ------------------------------------------------------------------ reports
_pct, _pf, _rs = sh._pct, sh._pf_s, sh._rs


def _signed(v, kind="rs"):
    if v is None:
        return "N/A"
    if kind == "pp":
        return f"{v:+.2f} pp"
    if kind == "pf":
        return f"{v:+.2f}"
    return f"{v:+,.2f}"


def summary_markdown(a: dict[str, Any]) -> str:
    md = [f"# {a['heading']}", "", "PAPER mode | research only | no orders placed | production S2 and SELL_ONLY "
          "unchanged | not auto-applied", "", a["evaluation"], "", f"> {a['never_robust_note']}", "",
          f"Breakout candidate: {a['breakout_note']}", "",
          f"BUY_BASELINE identical to original S2 BUY rows: {a['baseline_matches_original_s2_buy']}", "",
          "## Candidates (pre-declared)", ""]
    md += [f"- **{n}**: " + "; ".join(t) for n, t in a["candidates"].items()]
    md += ["", "## Splits", "", "| Split | From | To | Sessions |", "|---|---|---|---|"]
    md += [f"| {sp['label']} | {sp['from']} | {sp['to']} | {len(sp['days'])} |" for sp in a["splits"].values()]
    md += ["", "## Results (ranked by OOS PF, P&L, DD, WR, trades)", "",
           "| Rank | Candidate | Train WR | Validation WR | OOS WR | OOS PF | OOS P&L (Rs) | OOS DD (Rs) | OOS Trades | "
           "Lookahead | Status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {r["candidate"]: r for r in a["final_table"]}
    for i, n in enumerate(a["ranking_by_oos"], 1):
        r = rows[n]
        md.append(f"| {i} | {n} | {_pct(r['train_wr'])} | {_pct(r['validation_wr'])} | {_pct(r['oos_wr'])} | "
                  f"{_pf(r['oos_pf'])} | {_rs(r['oos_pnl'])} | {_rs(r['oos_dd'])} | {r['oos_trades']} | "
                  f"{'PASS' if a['lookahead'][n]['passed'] else 'FAIL'} | {r['status']} |")
    md += ["", "Failed checks: " + "; ".join(f"{n}: {', '.join(s['failed_checks']) or 'none'}"
                                             for n, s in a["status"].items())]
    for period in ("TRAIN", "VALIDATION", "OOS", "ALL"):
        md += ["", f"## {period} metrics", "",
               "| Candidate | Signals | Evaluated | Wins | Losses | WR | PF | P&L (Rs) | Avg/trade | Max DD | +Days | "
               "-Days | Data unavail. | Skipped |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n, m in a["metrics"][period].items():
            md.append(f"| {n} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | "
                      f"{_pct(m['win_rate'])} | {_pf(m['profit_factor'])} | {_rs(m['net_pnl'])} | "
                      f"{_rs(m['avg_pnl_per_trade'])} | {_rs(m['max_drawdown'])} | {m['profitable_days']} | "
                      f"{m['losing_days']} | {m['data_unavailable']} | {m['skipped']} |")
    md += ["", "## Top 3 - OOS day-removal check (reporting only, not used for selection)", "",
           "| Candidate | OOS P&L | Best day removed | P&L | PF | Worst day removed | P&L | PF |",
           "|---|---|---|---|---|---|---|---|"]
    for n, t in a["top3_day_removal_check"].items():
        b, w = t["best_day_removed"], t["worst_day_removed"]
        md.append(f"| {n} | {_rs(t['oos']['net_pnl'])} | {b['excluded_date']} | {_rs(b['net_pnl'])} | "
                  f"{_pf(b['profit_factor'])} | {w['excluded_date']} | {_rs(w['net_pnl'])} | {_pf(w['profit_factor'])} |")
    md += ["", "## Regime report (end-of-day labels, reporting only, not a filter)", ""]
    for n in a["regime_report_for"]:
        for period in ("ALL", "OOS"):
            reg = a["metrics"][period][n]["regime"]
            md += [f"### {n} - {period}", ""] + (sh._group_table("Regime", reg) if reg else ["- No regime label."])
            md.append("")
    md += ["## BUY_BASELINE vs best BUY candidate", ""]
    if not a["best_candidate"]:
        md.append(f"- No candidate has >= {a['criteria']['min_oos_trades']} OOS trades; no best candidate.")
    else:
        md.append(f"Best OOS candidate: **{a['best_candidate']}** (recommendable: "
                  f"{a['best_candidate_recommendable']})")
        for period, c in a["comparison"].items():
            b, x = c["baseline"], c["candidate"]
            md += ["", f"### {period}", "", "| Metric | BUY_BASELINE | Candidate | Change |", "|---|---|---|---|",
                   f"| WR | {_pct(b['win_rate'])} | {_pct(x['win_rate'])} | {_signed(c['wr_change_pp'], 'pp')} |",
                   f"| PF | {_pf(b['profit_factor'])} | {_pf(x['profit_factor'])} | {_signed(c['pf_change'], 'pf')} |",
                   f"| P&L (Rs) | {_rs(b['net_pnl'])} | {_rs(x['net_pnl'])} | {_signed(c['pnl_change'])} |",
                   f"| Max DD (Rs) | {_rs(b['max_drawdown'])} | {_rs(x['max_drawdown'])} | {_signed(c['dd_change'])} |",
                   f"| Trades | {b['evaluated']} | {x['evaluated']} | {c['trade_count_change']:+d} |"]
    md += ["", "## Lookahead mutation (BUY decisions only)", ""]
    md += [f"- {n}: {'PASS' if r['passed'] else 'FAIL'} ({r['cuts']} cuts, {len(r['failures'])} failures)"
           for n, r in a["lookahead"].items()]
    md += ["", "## Skip reasons (all splits)", ""]
    md += [f"- {n}: {r or 'none'}" for n, r in a["skip_reasons"].items()]
    md += ["", "---", "Research only. PAPER mode, LIVE_TRADING_ENABLED=false, no orders placed. Production S2 and "
                      "SELL_ONLY unchanged; the best candidate is NOT applied automatically. Costs/slippage excluded."]
    return "\n".join(md) + "\n"


def _csv(path: Path, fields: Sequence[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: _pf(v) if k.endswith("profit_factor") else v for k, v in r.items()})


GROUP = ("signals", "evaluated", "wins", "losses", "win_rate", "profit_factor", "net_pnl")


def write_reports(a: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Writes only the six lab files; nothing else in *out_dir* is touched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in a.items() if not k.startswith("_")}
    p = {n: out_dir / n for n in ("summary.md", "summary.json", "summary.csv", "daily.csv", "dte.csv", "regime.csv")}
    p["summary.md"].write_text(summary_markdown(public), encoding="utf-8")
    p["summary.json"].write_text(json.dumps(sh._jsonable(public), indent=2, default=str), encoding="utf-8")
    flat = ("signals", "evaluated", "wins", "losses", "win_rate", "profit_factor", "net_pnl", "avg_pnl_per_trade",
            "max_drawdown", "profitable_days", "losing_days", "data_unavailable", "skipped", "data_coverage")
    _csv(p["summary.csv"], ("candidate", "period", *flat, "lookahead_passed", "status"),
         [{"candidate": n, "period": period, **{k: m[k] for k in flat},
           "lookahead_passed": a["lookahead"][n]["passed"], "status": a["status"][n]["status"]}
          for period, by in a["metrics"].items() for n, m in by.items()])
    split_of_day = {d: s for s, sp in a["splits"].items() for d in sp["days"]}
    _csv(p["daily.csv"], ("candidate", "date", "split", *GROUP),
         [{"candidate": n, "date": d, "split": split_of_day.get(d), **{k: x[k] for k in GROUP}}
          for n, m in a["metrics"]["ALL"].items() for d, x in m["daily"].items()])
    _csv(p["dte.csv"], ("candidate", "period", "dte_bucket", *GROUP),
         [{"candidate": n, "period": period, "dte_bucket": b, **{k: x[k] for k in GROUP}}
          for period, by in a["metrics"].items() for n, m in by.items() for b, x in m["dte"].items()])
    _csv(p["regime.csv"], ("candidate", "period", "regime", *GROUP),
         [{"candidate": n, "period": period, "regime": b, **{k: x[k] for k in GROUP}}
          for period in ("ALL", "OOS") for n, m in a["metrics"][period].items() for b, x in m["regime"].items()])
    return p
