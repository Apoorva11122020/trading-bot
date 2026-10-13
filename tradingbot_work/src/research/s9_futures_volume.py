"""S9_FUTURES_VOLUME (PAPER | RESEARCH ONLY | NO ORDERS).

The S9 ORB + VWAP rules driven by REAL NIFTY futures OHLCV (S9.1). Separate from S9_PROXY (session TWAP + selected
option volume) and S9_EXACT_INDEX_VOLUME (index volume is zero, so it cannot fire); S2 is not modified.

  underlying signal   causal front-month NIFTY future (nearest unexpired expiry, contract kept separate, no stitching)
                      OR = its 09:15-09:25 candles, session VWAP from its real volume, volume ratio = its completed
                      candle volume / mean of its prior 20 completed candles
  option strike       NIFTY SPOT close at the signal timestamp -> ATM, nearest listed expiry on the signal date
  confirmation        bought CE and bought PE: option close > open AND > previous completed close (rising premium)
  entry / exits       unchanged S9: next option candle open, 1.5 x ATR14 stop, 2R target, AMBIGUOUS = -1R, 14:30 exit

Historical backtest accuracy only - never live accuracy, never a profit guarantee. Nothing is promoted automatically.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import s2_1dte_improvement as imp
from src.historical_data import nifty_futures as nf
from src.research import clean_historical_backtest as cb
from src.research import s2_baseline_drift_audit as dra
from src.research import s9_orb_vwap_option_momentum as s9

HEADING = "S9.1 FUTURES DATA COMPLETE"
STRATEGY = "S9_FUTURES_VOLUME"
BANNER = cb.BANNER
IST = s9.IST
SPLITS = cb.SPLITS
SPLIT_NAMES = {"TRAIN": "TRAIN", "VALIDATION": "VALIDATION", "OOS": "OOS AUGUST",
               "FINAL_HOLDOUT": "HOLDOUT SEPTEMBER"}
SCOPES = s9.SCOPES
HOLDOUT_LABEL = s9.HOLDOUT_LABEL
CFG = s9.Config(STRATEGY, mode=s9.FUTURES, option_confirmation=s9.RISING_PREMIUM)
SYSTEM_LABELS = {
    "S2": "S2 frozen baseline (unchanged)",
    "S9_PROXY": "S9_PROXY = PROXY DATA (session TWAP + selected option volume; literal falling-PE confirmation)",
    STRATEGY: "S9_FUTURES_VOLUME = REAL FUTURES VOLUME / SESSION VWAP (rising-premium confirmation for CE and PE)",
}
SELECTION_KEY = s9.BASE.name


# ------------------------------------------------------------------ features (selected real contract per session)
def _spot_close(spot: pd.DataFrame) -> pd.Series:
    t = dra._ist_series(spot["timestamp"])
    s = pd.Series(spot["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(t))
    return s[~s.index.duplicated(keep="last")]


def futures_features(frames: dict[str, pd.DataFrame], selection: Sequence[dict], spot: pd.DataFrame,
                     cache: dict | None = None) -> pd.DataFrame:
    """S9 features computed on each contract's own candle history (prior-20 volume never mixes contracts), kept only
    for the sessions where that contract is the causal front month. ``spot_close`` (same timestamp, no fill) is the
    strike reference; a missing spot candle stays NaN."""
    cache = {} if cache is None else cache
    sc = _spot_close(spot)
    parts = []
    for r in selection:
        if r["status"] != nf.SELECTED:
            continue
        k = r["instrument_key"]
        if k not in cache:
            cache[k] = s9.nifty_features(frames[k])
        f = cache[k]
        g = f[f["day"] == date.fromisoformat(r["session_date"])].copy()
        g["futures_contract"] = r["trading_symbol"]
        g["futures_key"] = k
        g["spot_close"] = sc.reindex(g.index).to_numpy()
        parts.append(g)
    if not parts:
        cols = ["open", "high", "low", "close", "volume", "day", *s9.FEATURE_COLS, "futures_contract", "futures_key",
                "spot_close"]
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], tz=IST, name="timestamp"))
    return pd.concat(parts).sort_index()


# ------------------------------------------------------------------ anti-lookahead
def feature_truncation_check(frames: dict[str, pd.DataFrame], feat: pd.DataFrame, sample: int = 20) -> dict:
    """Each sampled futures feature row recomputed from its contract's candles up to that timestamp only."""
    window = [ts for ts in feat.index if s9.in_window(ts)]
    picked = window[::max(1, len(window) // max(sample, 1))][:sample]
    bad = []
    for ts in picked:
        row = feat.loc[ts]
        k = row["futures_key"]
        f = frames[k]
        t = dra._ist_series(f["timestamp"])
        tf = s9.nifty_features(f[(t <= ts).to_numpy()].reset_index(drop=True)).loc[ts]
        for c in s9.FEATURE_COLS:
            a = bool(row[c]) if c == "or_complete" else s9._num(row[c])
            b = bool(tf[c]) if c == "or_complete" else s9._num(tf[c])
            if not s9._same(a, b):
                bad.append(f"{ts.isoformat()}|{c}")
        ra = {**row.to_dict(), "timestamp": ts}
        rb = {**tf.to_dict(), "timestamp": ts}
        for side in s9.DIRECTIONS:
            if s9.underlying_check(ra, side, CFG) != s9.underlying_check(rb, side, CFG):
                bad.append(f"{ts.isoformat()}|{side}|decision")
    return {"checked": len(picked), "mismatches": bad[:20], "passed": bool(picked) and not bad}


def scrambling_check(spot: pd.DataFrame, contracts, frames: dict[str, pd.DataFrame], feat: pd.DataFrame,
                     days: Sequence[str], resolve: Callable, premium_for: Callable, seed: int = 23) -> dict:
    """Spot, every futures contract and every option candle after 12:00 rescaled at random (prices and volume):
    contract selection and every decision at or before 12:00 must be identical."""
    fails = []
    for i, d in enumerate(days):
        day = date.fromisoformat(d)
        cut = pd.Timestamp(datetime.combine(day, time(12, 0)), tz=IST)
        a = s9.replay(feat, [d], resolve, premium_for, CFG)[0]
        mspot = imp.mutate_after(spot, cut, seed + i)
        mframes = {k: imp.mutate_after(f, cut, seed + 100 + i) for k, f in frames.items()}
        msel = nf.select_contracts(contracts, mframes, [day])
        mfeat = futures_features(mframes, msel, mspot)
        b = s9.replay(mfeat, [d], resolve, lambda c, _cut=cut, _i=i: imp._mut_prem(premium_for(c), _cut, seed + _i),
                      CFG)[0]
        pa = [s9._decision(r, cut) for r in a if s9._ts(r["signal_time"]) <= cut]
        pb = [s9._decision(r, cut) for r in b if s9._ts(r["signal_time"]) <= cut]
        if pa != pb:
            fails.append(cut.isoformat())
    return {"cuts": len(days), "failures": fails, "passed": bool(days) and not fails}


# ------------------------------------------------------------------ study
def build(*, spot: pd.DataFrame, sessions: Sequence[str], contracts, frames: dict[str, pd.DataFrame],
          selection: Sequence[dict], futures_quality: dict, resolve: Callable, premium_for: Callable,
          s2_rows: Sequence[dict], calendar, listed: Callable, exists: Callable, integrity: dict, dataset: dict,
          mutation_days: Sequence[str], check_sample: int = 20, production_unchanged: bool | None = None,
          progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    say("S9_FUTURES_VOLUME features (front-month contract per session, real volume / VWAP) ...")
    feat = futures_features(frames, selection, spot)
    say("S9_FUTURES_VOLUME chronological replay ...")
    recs, funnel, cache = s9.replay(feat, sessions, resolve, premium_for, CFG)
    say("S9_PROXY replay (comparison only, proxy data) ...")
    proxy = s9.replay(s9.nifty_features(spot), sessions, resolve, premium_for, s9.BASE, cache)[0]

    say("anti-lookahead checks ...")
    lookahead = {
        "futures_feature_truncation": feature_truncation_check(frames, feat, check_sample),
        "future_candle_scrambling": scrambling_check(spot, contracts, frames, feat, mutation_days, resolve,
                                                     premium_for),
        "option_candle_truncation": s9.option_truncation_check(recs, cache, check_sample, configs=(CFG,)),
        "chronological_replay": s9.chronology_check(recs),
        "determinism": s9.determinism_check(recs, feat, mutation_days, resolve, premium_for, CFG),
        "session_vwap_no_future_data": {"passed": bool(futures_quality["vwap"].get("passed")),
                                        **{k: v for k, v in futures_quality["vwap"].items() if k != "passed"}},
        "causal_contract_selection": {"passed": bool(futures_quality["contract_selection"]["passed"])},
    }
    lookahead_ok = all(bool(v.get("passed")) for v in lookahead.values())

    sel = s9.select(s9.pre({SELECTION_KEY: recs}))
    iso = s9.isolation({SELECTION_KEY: recs}, sel)
    executed = [r for r in recs if r["outcome"] in s9.EXECUTED]
    dup = cb.duplicate_check(executed)
    ts_chk = cb.timestamp_checks(executed, s9.SQUARE_OFF.isoformat(timespec="minutes"),
                                 s9.LAST_ENTRY.isoformat(timespec="minutes"))
    cval = s9.contract_checks(recs, calendar, listed, exists)
    data_ok = bool(integrity.get("ok")) and bool(futures_quality["pass"]) and dup["passed"] and ts_chk["passed"] \
        and cval["passed"]
    clean = lookahead_ok and data_ok and iso["passed"]

    tables = {STRATEGY: s9.table(recs), "S9_PROXY": s9.table(proxy)}
    oos = tables[STRATEGY]["OOS"]["COMBINED"]
    oos_lab = s9.oos_label(oos)
    status = s9.s9_status(sel["train"][SELECTION_KEY], sel["validation"][SELECTION_KEY], oos_lab, clean)
    fp = s9.fingerprint(sel)
    holdout = tables[STRATEGY]["FINAL_HOLDOUT"]["COMBINED"]
    ho_lab = s9.holdout_label(holdout)
    if oos["resolved"] < s9.MIN_OOS_RESOLVED or holdout["resolved"] < s9.MIN_HOLDOUT_RESOLVED:
        target = "INSUFFICIENT DATA"
    elif (oos["accuracy"] or 0) >= s9.TARGET and (holdout["accuracy"] or 0) >= s9.TARGET:
        target = "ACHIEVED"
    else:
        target = "NOT ACHIEVED"
    safe = status == s9.PROMISING and ho_lab == s9.PASSED_HO and clean

    s2 = {k: {s: s9.s2_view(cb.metrics(cb.side(cb.split_rows(s2_rows, k), s))) for s in SCOPES} for k in SPLITS}
    views = {"S2": s2, "S9_PROXY": {k: {s: s9.s9_view(tables["S9_PROXY"][k][s]) for s in SCOPES} for k in SPLITS},
             STRATEGY: {k: {s: s9.s9_view(tables[STRATEGY][k][s]) for s in SCOPES} for k in SPLITS}}
    comparison = [{"split": SPLIT_NAMES[k], "scope": s9.SCOPE_NAMES[s], "system": name,
                   "data": SYSTEM_LABELS[name], **views[name][k][s]}
                  for k in SPLITS for s in SCOPES for name in ("S2", "S9_PROXY", STRATEGY)]
    return {
        "heading": HEADING, "strategy": STRATEGY, "banner": BANNER, "ran": True,
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "accuracy_definition": "historical backtest accuracy = wins / (wins + losses); AMBIGUOUS (-1R), SCRATCH and "
                               "UNRESOLVED are never wins or losses",
        "dataset": dataset, "splits": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in SPLITS.items()},
        "sessions": len(sessions), "config": asdict(CFG), "rules": rules_text(), "system_labels": SYSTEM_LABELS,
        "tables": tables, "funnel": dict(funnel), "selection": sel, "fingerprint": fp, "isolation": iso,
        "train_label": sel["train"][SELECTION_KEY], "validation_label": sel["validation"][SELECTION_KEY],
        "oos_label": oos_lab, "holdout_label": ho_lab, "status": status, "target_70": target,
        "lookahead": lookahead, "lookahead_ok": lookahead_ok, "data_integrity": integrity,
        "data_checks": {"duplicates": dup, "timestamps": ts_chk, "contract_validity": cval}, "data_ok": data_ok,
        "futures_quality": futures_quality, "s2": s2, "views": views, "comparison": comparison,
        "monthly": s9.monthly_rows(recs), "forward_paper": "SAFE" if safe else "NOT SAFE",
        "futures_contract_usage": dict(Counter(r["futures_contract"] for r in recs if r["outcome"] in s9.EXECUTED)),
        "disclosure": ("S9 rules were written before S9.1; the only changes are the user-specified data source "
                       "(real futures volume / VWAP) and the corrected rising-premium confirmation for bought PE. "
                       "Nothing was tuned on August or September. Fresh forward PAPER data is still required."),
        "production_unchanged": production_unchanged, "live_orders": 0, "trading_mode": "PAPER",
        "live_trading_enabled": False, "records": recs, "proxy_records": proxy,
    }


def not_run(futures_quality: dict, integrity: dict | None = None, production_unchanged: bool | None = None) -> dict:
    return {"heading": HEADING, "strategy": STRATEGY, "banner": BANNER, "ran": False,
            "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
            "reason": f"futures data quality {futures_quality['status']} - S9 futures replay not run",
            "status": s9.INSUFFICIENT, "lookahead_ok": None, "data_ok": False, "forward_paper": "NOT SAFE",
            "futures_quality": futures_quality, "data_integrity": integrity or {}, "rules": rules_text(),
            "system_labels": SYSTEM_LABELS, "production_unchanged": production_unchanged, "live_orders": 0,
            "trading_mode": "PAPER", "live_trading_enabled": False}


def rules_text() -> dict[str, str]:
    return {
        "underlying": "causal front-month NIFTY future per session (nearest unexpired expiry from instrument "
                      "metadata; no stitching; missing contract / candle -> DATA_UNAVAILABLE, never substituted)",
        "opening_range": "OR_HIGH / OR_LOW = futures 09:15, 09:20, 09:25 candles (all three required)",
        "vwap": "session VWAP = cumsum(typical price x futures volume) / cumsum(futures volume), reset daily",
        "volume": "futures completed-candle volume / mean of the same contract's prior 20 completed candles >= 1.20",
        "signal": "BUY close > OR_HIGH by >= 0.05%, > VWAP, > open, body/range >= 0.50, close in top 30%; SELL "
                  "mirrored; signal candle 09:30..13:55",
        "option_selection": "NIFTY SPOT close at the signal timestamp -> ATM; nearest listed expiry on the signal "
                            "date; BUY -> CE, SELL -> PE; missing spot candle or contract -> NO_SIGNAL",
        "option_confirmation": "bought CE and bought PE: option close > open and > previous completed close",
        "entry_exit": "next option candle OPEN; SL 1.5 x option ATR14; target 2R; SL+target same candle = AMBIGUOUS "
                      "(-1R); last candle ending <= 14:30",
    }


# ------------------------------------------------------------------ reports
REPORT_FILES = ("summary.json", "s9_futures_report.md", "trade_log.csv", "monthly_results.csv", "buy_results.csv",
                "sell_results.csv", "combined_results.csv", "data_quality.json", "comparison_s9_proxy.csv")
TRADE_COLS = (*s9.TRADE_COLS, "underlying_close", "futures_contract", "option_confirmation_rule")


def _strip(q: dict) -> dict:
    return {k: v for k, v in q.items() if not k.startswith("_")}


def scope_csv(a: dict, scope: str) -> list[dict]:
    out = []
    for system in (STRATEGY, "S9_PROXY"):
        for k in SPLITS:
            m = a["tables"][system][k][scope]
            out.append({"system": system, "data": SYSTEM_LABELS[system], "split": SPLIT_NAMES[k],
                        "scope": s9.SCOPE_NAMES[scope], "label": HOLDOUT_LABEL if k == "FINAL_HOLDOUT" else "",
                        **{c: m[c] for c in s9.RESULT_COLS}})
    return out


def markdown(a: dict) -> str:
    q = a["futures_quality"]
    md = [f"# {STRATEGY}", "", f"**{BANNER}**", "", f"Generated {a['generated_at']}. TRADING_MODE=PAPER, "
          "LIVE_TRADING_ENABLED=false, live orders: 0. Production changed: "
          f"**{'NO' if a['production_unchanged'] is not False else 'YES'}**.", "",
          "All results are **historical backtest accuracy**, not live accuracy; no profit is guaranteed.", "",
          "## Futures data", "", f"- Data quality: **{q['status']}**",
          f"- Contracts: {q['contracts_downloaded']} downloaded / {q['contracts_discovered']} discovered",
          f"- Candles: {q['total_candles']} (front-month {q['front_month_candles']}), trading days {q['trading_days']}",
          f"- Front-month zero-volume candles: {q['zero_volume_candles']['front_month']}",
          f"- VWAP {'PASS' if q['vwap'].get('passed') else 'FAIL'} | causal contract selection "
          f"{'PASS' if q['contract_selection']['passed'] else 'FAIL'} | spot/futures alignment "
          f"{'PASS' if q['timestamp_alignment']['passed'] else 'FAIL'} | token leak "
          f"{'PASS' if q['token_leak']['passed'] else 'FAIL'}", ""]
    if not a["ran"]:
        md += [f"**S9 futures replay NOT RUN**: {a['reason']}.", ""]
        return "\n".join(md) + "\n"
    md += [f"## Result: {a['status']}", "",
           f"TRAIN {a['train_label']['label']} | July {a['validation_label']} | August {a['oos_label']} | September "
           f"{a['holdout_label']} ({HOLDOUT_LABEL}). Lookahead **{'PASS' if a['lookahead_ok'] else 'FAIL'}**, data "
           f"integrity **{'PASS' if a['data_ok'] else 'FAIL'}**, forward PAPER validation **{a['forward_paper']}**.",
           "", f"> {a['disclosure']}", "", "## Rules", ""]
    md += [f"- **{k}**: {v}" for k, v in a["rules"].items()]
    md += ["", "## S9_FUTURES_VOLUME", "", "| Split | Scope | Result |", "|---|---|---|"]
    md += [f"| {SPLIT_NAMES[k]} | {s9.SCOPE_NAMES[s]} | {s9.md_line(a['tables'][STRATEGY][k][s])} |"
           for k in SPLITS for s in SCOPES]
    md += ["", "## S2 vs S9_PROXY vs S9_FUTURES_VOLUME", ""]
    md += [f"- **{k}**: {v}" for k, v in SYSTEM_LABELS.items()]
    md += ["", "| Split | Scope | S2 | S9_PROXY (proxy data) | S9_FUTURES_VOLUME (real futures) |",
           "|---|---|---|---|---|"]
    for k in SPLITS:
        for s in SCOPES:
            v = a["views"]
            md.append(f"| {SPLIT_NAMES[k]} | {s9.SCOPE_NAMES[s]} | {s9.view_text(v['S2'][k][s])} | "
                      f"{s9.view_text(v['S9_PROXY'][k][s])} | {s9.view_text(v[STRATEGY][k][s])} |")
    md += ["", "## Signal funnel", ""] + [f"- {k}: {v}" for k, v in sorted(a["funnel"].items())]
    md += ["", "## Anti-lookahead and isolation", ""]
    md += [f"- {k}: {'PASS' if v.get('passed') else 'FAIL'}" for k, v in a["lookahead"].items()]
    md += [f"- {k}: {'PASS' if v else 'FAIL'}" for k, v in a["isolation"]["checks"].items()]
    md += ["", "## Safety", "", "Research only. Production S2, broker, OrderManager, RiskManager, credentials and live "
           "flags are unchanged. Nothing is promoted automatically."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in REPORT_FILES}
    skip = ("records", "proxy_records", "data_integrity")
    summary = {k: (_strip(v) if k == "futures_quality" else v) for k, v in a.items() if k not in skip}
    p["summary.json"].write_text(json.dumps(s9._jsonable(summary), indent=2, default=str), encoding="utf-8")
    p["s9_futures_report.md"].write_text(markdown(a), encoding="utf-8")
    dq = {"futures": _strip(a["futures_quality"]), "store_integrity": a.get("data_integrity"),
          "lookahead": a.get("lookahead"), "data_checks": a.get("data_checks"), "data_ok": a.get("data_ok")}
    p["data_quality.json"].write_text(json.dumps(s9._jsonable(dq), indent=2, default=str), encoding="utf-8")
    if not a["ran"]:
        return {k: v for k, v in p.items() if k in ("summary.json", "s9_futures_report.md", "data_quality.json")}
    s9._csv(p["trade_log.csv"], [{**{c: r.get(c) for c in TRADE_COLS},
                                  "split": SPLIT_NAMES.get(cb.split_of(r) or "", ""),
                                  "label": HOLDOUT_LABEL if cb.split_of(r) == "FINAL_HOLDOUT" else ""}
                                 for r in a["records"]])
    s9._csv(p["monthly_results.csv"], a["monthly"])
    s9._csv(p["buy_results.csv"], scope_csv(a, "BUY"))
    s9._csv(p["sell_results.csv"], scope_csv(a, "SELL"))
    s9._csv(p["combined_results.csv"], scope_csv(a, "COMBINED"))
    s9._csv(p["comparison_s9_proxy.csv"], a["comparison"])
    return p


def terminal_text(a: dict) -> str:
    q = a["futures_quality"]
    out = [HEADING, "(historical backtest accuracy - not live accuracy, no guaranteed profit)", "",
           *nf.data_text(q), ""]
    if q["status"] == nf.FUTURES_VOLUME_UNAVAILABLE:
        out += [nf.FUTURES_VOLUME_UNAVAILABLE, ""]
    out += ["S9 FUTURES REPLAY:", "RUN" if a["ran"] else "NOT RUN", "", f"{STRATEGY}:", ""]
    for k in SPLITS:
        out.append(f"{SPLIT_NAMES[k]}:" + (f"  ({HOLDOUT_LABEL})" if k == "FINAL_HOLDOUT" else ""))
        out += [f"{s}: {s9.line(a['tables'][STRATEGY][k][s]) if a['ran'] else 'NOT RUN'}" for s in SCOPES]
        out.append("")
    look = "NOT RUN (data quality failed)" if a["lookahead_ok"] is None else ("PASS" if a["lookahead_ok"] else "FAIL")
    out += ["STATUS:", a["status"], "", "LOOKAHEAD:", look, "", "DATA INTEGRITY:",
            "PASS" if a["data_ok"] else "FAIL", "", "PRODUCTION CHANGED:",
            "NO" if a["production_unchanged"] is not False else "YES", "", "LIVE ORDERS:", "0", "",
            "FORWARD PAPER VALIDATION:", a["forward_paper"]]
    if a["ran"]:
        s2, px, fu = (a["views"][n]["OOS"]["COMBINED"] for n in ("S2", "S9_PROXY", STRATEGY))
        out += ["", "OOS AUGUST COMBINED (S2 | S9_PROXY proxy data | S9_FUTURES_VOLUME real futures):"]
        out += [f"{n}: {s9.view_text(v)}" for n, v in (("S2", s2), ("S9_PROXY", px), (STRATEGY, fu))]
    return "\n".join(out)
