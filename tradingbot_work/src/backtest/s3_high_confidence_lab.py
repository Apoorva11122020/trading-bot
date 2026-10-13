"""S3_HIGH_CONFIDENCE research lab (PAPER | RESEARCH ONLY | NO ORDERS; production S2 untouched).

Pre-declared grid only: 7 score thresholds x 5 entry-confirmation modes x BUY/SELL = 70 candidates. Every signal is
evaluated with the unchanged S2-compatible execution model (point-in-time nearest-expiry ATM contract, entry at the
next option candle open, stop 1.5 x premium ATR14, 2R target, first touch on later real option candles, 15:15
square-off, AMBIGUOUS = -1R). TRAIN + VALIDATION select the finalist per direction; OOS is evaluated once.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.backtest import accuracy_70_lab as acc
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest import strategy_improvement_lab as lab
from src.backtest.research_strategies import ResearchSignal
from src.strategy import s3_high_confidence as s3

HEADING = "S3_HIGH_CONFIDENCE RESEARCH LAB"
BANNER = s3.BANNER
SPLITS, PERIODS = acc.SPLITS, acc.PERIODS
ROBUST_70, PROMISING, INSUFFICIENT_DATA, KEEP, REJECTED = (
    "ROBUST_70", "PROMISING", "INSUFFICIENT_DATA", "KEEP_CURRENT", "REJECTED")
FORWARD_REQUIRED = "FORWARD_VALIDATION_REQUIRED"
NOT_ACHIEVED_TEXT = "70% OOS accuracy was not achieved without unacceptable overfitting."
MODES = ("A_NEXT_BAR", "B_CONFIRMATION_CANDLE", "C_BREAKOUT_CONTINUATION", "D_BREAKOUT_RETEST",
         "E_PREMIUM_CONFIRMATION")
MODE_TEXT = {
    "A_NEXT_BAR": "enter at the next option candle open after the scored candle",
    "B_CONFIRMATION_CANDLE": "next underlying candle closes in the trade direction beyond the scored close; enter after it",
    "C_BREAKOUT_CONTINUATION": "next underlying candle closes beyond the scored candle's high (BUY) / low (SELL)",
    "D_BREAKOUT_RETEST": ("within 3 candles price retests the broken 12-candle level (or EMA9 if no breakout) within "
                          "0.15 ATR and closes back beyond it with a directional candle"),
    "E_PREMIUM_CONFIRMATION": "next option candle of the scored contract closes higher and bullish (premium follows)",
}
MODE_COMPLEXITY = {m: (1 if m == "A_NEXT_BAR" else 2) for m in MODES}
STATUS_MAP = {acc.ALL_GATES_PASS: ROBUST_70, acc.PROMISING_SMALL_SAMPLE: PROMISING,
              acc.QUALITY_PASS_BELOW_70: PROMISING, acc.INSUFFICIENT_SAMPLE: INSUFFICIENT_DATA}
STATUS_ORDER = {ROBUST_70: 0, PROMISING: 1, INSUFFICIENT_DATA: 2, REJECTED: 3}
COMPARE_REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE_CHOP", "HIGH_VOLATILITY", "LOW_VOLATILITY")


def cid(direction: str, threshold: int, mode: str) -> str:
    return f"S3_{direction}_T{threshold}_{mode}"


def s3_status(acc_status: str) -> str:
    return STATUS_MAP.get(acc_status, REJECTED)


# ------------------------------------------------------------------ signal construction
def union_signals(f: pd.DataFrame, keys, underlying: str) -> list[ResearchSignal]:
    look = {ts.isoformat(): (ts, float(c)) for ts, c in zip(f["timestamp"], f["close"])}
    return [ResearchSignal(look[t][0].to_pydatetime(), d, s3.STRATEGY_ID, underlying, look[t][1], {})
            for t, d in sorted(set(keys))]


def setup_keys(bars: pd.DataFrame, pts: dict[str, pd.DataFrame], mask: pd.Series) -> dict[str, list[tuple[str, str]]]:
    out = {}
    for d in s3.DIRECTIONS:
        p = pts[d]
        ok = p["available"] & p["tradeable_regime"] & (p["underlying"] >= s3.PRE_FILTER) & mask
        out[d] = [(ts.isoformat(), d) for ts in bars.loc[ok, "timestamp"]]
    return out


def threshold_signals(bars: pd.DataFrame, totals: dict[str, dict[str, float]], threshold: float,
                      mask: pd.Series) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Edge-triggered per direction on total >= threshold; both directions qualifying on one candle -> NO_SIGNAL."""
    isos = [t.isoformat() for t in bars["timestamp"]]
    days = list(bars["day"])
    m = list(mask)
    above = {d: [totals[d].get(i, -1.0) >= threshold for i in isos] for d in s3.DIRECTIONS}
    out: dict[str, list[tuple[str, str]]] = {d: [] for d in s3.DIRECTIONS}
    conflicts = 0
    for i, iso in enumerate(isos):
        if not m[i]:
            continue
        both = above[s3.BUY][i] and above[s3.SELL][i]
        for d in s3.DIRECTIONS:
            if not above[d][i] or (i > 0 and days[i - 1] == days[i] and above[d][i - 1]):
                continue
            if both:
                conflicts += 1
                continue
            out[d].append((iso, d))
    return out, conflicts


def confirmations(bars: pd.DataFrame, keys: Sequence[tuple[str, str]], mode: str, mask: pd.Series,
                  ctx_by_key: dict, interval: int = 5) -> dict[tuple[str, str], tuple[str, str]]:
    """Map confirmation (signal) key -> originating setup key. Every check reads candles <= the confirmation bar."""
    if mode == "A_NEXT_BAR":
        return {k: k for k in keys}
    idx = {t.isoformat(): i for i, t in enumerate(bars["timestamp"])}
    step = pd.Timedelta(minutes=interval)
    m = list(mask)
    out: dict[tuple[str, str], tuple[str, str]] = {}

    def contiguous(i, j):
        return (j < len(bars) and m[j] and bars["day"].iat[j] == bars["day"].iat[i]
                and bars["timestamp"].iat[j] == bars["timestamp"].iat[i] + step * (j - i))

    for key in keys:
        iso, d = key
        i = idx[iso]
        up = d == s3.BUY
        s = 1.0 if up else -1.0
        r = bars.iloc[i]
        hit = None
        if mode in ("B_CONFIRMATION_CANDLE", "C_BREAKOUT_CONTINUATION", "E_PREMIUM_CONFIRMATION"):
            j = i + 1
            if contiguous(i, j):
                n = bars.iloc[j]
                if mode == "B_CONFIRMATION_CANDLE":
                    dirc = n["close"] > n["open"] if up else n["close"] < n["open"]
                    hit = j if dirc and (n["close"] - r["close"]) * s > 0 else None
                elif mode == "C_BREAKOUT_CONTINUATION":
                    hit = j if (n["close"] > r["high"] if up else n["close"] < r["low"]) else None
                else:
                    ctx = ctx_by_key.get(key)
                    prem = None if ctx is None or ctx.get("data_status") != "OK" else ctx["premium"]
                    if prem is not None:
                        a = prem[prem["timestamp"] == r["timestamp"]]
                        b = prem[prem["timestamp"] == n["timestamp"]]
                        if len(a) and len(b) and b["close"].iat[0] > a["close"].iat[0] and \
                                b["close"].iat[0] > b["open"].iat[0]:
                            hit = j
        elif mode == "D_BREAKOUT_RETEST":
            lvl12 = r["hh12"] if up else r["ll12"]
            broke = pd.notna(lvl12) and (r["close"] - lvl12) * s > 0
            level = lvl12 if broke else r["ema9"]
            tol = 0.15 * r["atr14"]
            if pd.notna(level) and pd.notna(tol):
                for j in range(i + 1, i + 4):
                    if not contiguous(i, j):
                        break
                    n = bars.iloc[j]
                    if up and n["low"] <= level + tol and n["close"] > level and n["close"] > n["open"]:
                        hit = j
                        break
                    if not up and n["high"] >= level - tol and n["close"] < level and n["close"] < n["open"]:
                        hit = j
                        break
        else:
            raise ValueError(mode)
        if hit is not None:
            k2 = (bars["timestamp"].iat[hit].isoformat(), d)
            out.setdefault(k2, key)
    return out


# ------------------------------------------------------------------ analysis helpers
def tier_table(rows: Sequence[dict], periods: dict[str, Sequence[str]]) -> list[dict]:
    out = []
    for d in s3.DIRECTIONS:
        for name, lo, hi in s3.TIERS:
            rs = [r for r in rows if r["direction"] == d and lo <= (r.get("s3_score") or -1) <= hi]
            for p, days in {**periods, "TRAIN+VALIDATION": [*periods["TRAIN"], *periods["VALIDATION"]]}.items():
                m = lab.perf(lab.period_rows(rs, days))
                out.append({"direction": d, "tier": name, "score_range": f"{lo}-{hi}", "period": p,
                            "trades": m["evaluated"], "wins": m["wins"], "losses": m["losses"],
                            "accuracy": m["win_rate"], "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"]})
    return out


def best_tier(tiers: list[dict], min_trades: int = 10) -> dict | None:
    """Chosen on TRAIN+VALIDATION accuracy (never OOS); OOS of that tier is reported alongside."""
    pool = [t for t in tiers if t["period"] == "TRAIN+VALIDATION" and t["trades"] >= min_trades]
    if not pool:
        return None
    b = max(pool, key=lambda t: (t["accuracy"] or 0, t["trades"]))
    oos = next(t for t in tiers if t["direction"] == b["direction"] and t["tier"] == b["tier"] and t["period"] == "OOS")
    return {**b, "oos": oos}


def calibration(tiers: list[dict], period: str, min_trades: int = 5) -> dict[str, bool | None]:
    out = {}
    for d in s3.DIRECTIONS:
        acc_ = [t["accuracy"] for t in tiers if t["direction"] == d and t["period"] == period
                and t["trades"] >= min_trades]
        out[d] = None if len(acc_) < 2 else all(b >= a for a, b in zip(acc_, acc_[1:]))
    return out


def day_stability(rows: Sequence[dict], days: Sequence[str]) -> dict[str, Any]:
    p = acc.day_profile(lab.period_rows(rows, days))
    return {**p, "active_days": p["days"], "worst_day_loss": min(p["worst_day_pnl"] or 0.0, 0.0)}


def recommend(finalists: dict[str, str | None], scores: dict[str, dict], integrity_ok: bool,
              max_val_trades: dict[str, int], c: acc.Criteria) -> str:
    if not integrity_ok:
        return REJECTED
    st = [scores[n]["status"] for n in finalists.values() if n]
    if ROBUST_70 in st:
        return ROBUST_70
    if PROMISING in st:
        return PROMISING
    if not st:
        return INSUFFICIENT_DATA if max(max_val_trades.values(), default=0) < c.min_val_trades else REJECTED
    if all(x == INSUFFICIENT_DATA for x in st):
        return INSUFFICIENT_DATA
    if any(scores[n]["OOS"]["net_pnl"] > 0 for n in finalists.values() if n):
        return KEEP
    return REJECTED


# ------------------------------------------------------------------ lookahead
POINT_COLUMNS = (*s3.UNDERLYING_COMPONENTS, "underlying", "available", "tradeable_regime", "regime")


def truncation_test(spot: pd.DataFrame, days: Sequence[str], allowed: Callable = acc.default_allowed,
                    cut_times: Sequence[str] = ("11:00", "13:00"), interval: int = 5) -> dict[str, Any]:
    """Regime, every score component and every setup recomputed from candles <= cut must be unchanged."""
    ts_all = acc._ist_series(spot["timestamp"])
    full = s3.s3_bars(spot, interval)
    fpts = {d: s3.underlying_points(full, s3.PROFILES[d]) for d in s3.DIRECTIONS}
    fset = setup_keys(full, fpts, acc.allowed_mask(full, allowed))
    res = {"cuts": 0, "failures": []}
    for d in days:
        dd = date.fromisoformat(d)
        for t in cut_times:
            cut = pd.Timestamp(datetime.combine(dd, time.fromisoformat(t)), tz=imp.IST)
            tb = s3.s3_bars(spot[ts_all <= cut].reset_index(drop=True), interval)
            tpts = {x: s3.underlying_points(tb, s3.PROFILES[x]) for x in s3.DIRECTIONS}
            tset = setup_keys(tb, tpts, acc.allowed_mask(tb, allowed))
            a_sel = (full["timestamp"] <= cut) & (full["day"] == dd)
            b_sel = tb["day"] == dd
            same = (full.loc[a_sel, ["regime", "vol_regime"]].reset_index(drop=True)
                    .equals(tb.loc[b_sel, ["regime", "vol_regime"]].reset_index(drop=True)))
            for x in s3.DIRECTIONS:
                same = same and (fpts[x].loc[a_sel, list(POINT_COLUMNS)].reset_index(drop=True)
                                 .equals(tpts[x].loc[b_sel, list(POINT_COLUMNS)].reset_index(drop=True)))
                want = {k for k in fset[x] if k[0][:10] == d and pd.Timestamp(k[0]) <= cut}
                same = same and want == {k for k in tset[x] if k[0][:10] == d}
            res["cuts"] += 1
            if not same:
                res["failures"].append(cut.isoformat())
    res["passed"] = bool(res["cuts"]) and not res["failures"]
    return res


def pipeline_mutation_test(spot, allowed, resolve, premium_for, params, underlying, days, interval: int = 5,
                           cut: str = "12:00") -> dict[str, Any]:
    def signal_fn(frame):
        b = s3.s3_bars(frame, interval)
        pts = {d: s3.underlying_points(b, s3.PROFILES[d]) for d in s3.DIRECTIONS}
        keys = setup_keys(b, pts, acc.allowed_mask(b, allowed))
        return union_signals(b, [k for ks in keys.values() for k in ks], underlying)

    pipeline = imp._pipeline_factory(signal_fn, resolve, params, "09:15", "09:30")
    cuts = [pd.Timestamp(datetime.combine(date.fromisoformat(d), time.fromisoformat(cut)), tz=imp.IST) for d in days]
    return imp.mutation_test(pipeline, spot, premium_for, [imp.BASELINE], cuts)[imp.BASELINE.name]


def isolation_checks(rows: dict[str, list[dict]], directions: dict[str, str], row_by_key: dict,
                     snapshot: str) -> list[dict]:
    bad = [n for n, rs in rows.items() if any(r["direction"] != directions[n] for r in rs)]
    order = all(acc._fp(lab.perf(rs)) == acc._fp(lab.perf(list(reversed(rs)))) for rs in rows.values())
    return [
        acc._check("I1", "Candidate IDs are unique", len(rows) == len(set(rows)), f"{len(rows)} candidates"),
        acc._check("I2", "BUY and SELL candidates are fully isolated (own direction only)", not bad,
                   f"violations: {bad[:10] or 'none'}"),
        acc._check("I3", "Shared evaluated rows are never mutated", acc.snapshot_rows(row_by_key) == snapshot,
                   "JSON snapshot before/after"),
        acc._check("I4", "Metrics recompute independently of row order", order, "perf() per candidate"),
    ]


# ------------------------------------------------------------------ study
def build_s3(*, spot: pd.DataFrame, split_days: dict[str, list[str]], splits: dict, resolve: Callable,
             premium_for: Callable, params, square_off: str, underlying: str = "NIFTY",
             allowed: Callable = acc.default_allowed, base_rows: Sequence[dict] = (),
             baseline_lookahead_passed: bool = True, rules: dict | None = None, today: date | None = None,
             criteria: acc.Criteria = acc.Criteria(), progress: Callable[[str], None] | None = None,
             run_mutation: bool = True, lookahead_days: Sequence[str] | None = None,
             cut_times: Sequence[str] = ("11:00", "13:00"), interval: int = 5) -> dict[str, Any]:
    say = progress or (lambda _m: None)
    c = criteria
    hist = [d for s in SPLITS for d in split_days[s]]
    today = today or datetime.now(sh.IST).date()
    periods = {**{s: split_days[s] for s in SPLITS}, "ALL": hist}

    say("computing causal S3 regime + direction scores ...")
    bars = s3.s3_bars(spot, interval)
    mask = acc.allowed_mask(bars, allowed, set(hist))
    pts = {d: s3.underlying_points(bars, s3.PROFILES[d]) for d in s3.DIRECTIONS}
    setups = setup_keys(bars, pts, mask)
    feats = imp.underlying_features(spot)
    union1 = union_signals(bars, [k for ks in setups.values() for k in ks], underlying)
    say(f"resolving contracts / premiums for {len(union1)} scored candles (point-in-time) ...")
    ctx1 = imp.build_contexts(union1, feats, resolve, premium_for, params)
    ctx_by_key = {(x["signal_time"].isoformat(), x["direction"]): x for x in ctx1}
    opt = {k: acc.option_features(x) for k, x in ctx_by_key.items()}
    pos = {t.isoformat(): i for i, t in enumerate(bars["timestamp"])}
    totals: dict[str, dict[str, float]] = {d: {} for d in s3.DIRECTIONS}
    components: dict[tuple[str, str], dict] = {}
    no_signal = Counter()
    for d in s3.DIRECTIONS:
        for key in setups[d]:
            row = pts[d].loc[pos[key[0]]]
            dec = s3.decide(d, row, opt.get(key), threshold=0)
            if dec.score is None or dec.reason == "PREMIUM_DATA_UNAVAILABLE":
                no_signal[dec.reason] += 1
                continue
            totals[d][key[0]] = dec.score
            components[key] = dec.components

    signal_map: dict[str, dict[tuple[str, str], tuple[str, str]]] = {}
    conflicts = {}
    for t in s3.THRESHOLDS:
        a_keys, conflicts[t] = threshold_signals(bars, totals, t, mask)
        for d in s3.DIRECTIONS:
            for mode in MODES:
                signal_map[cid(d, t, mode)] = confirmations(bars, a_keys[d], mode, mask, ctx_by_key, interval)
    extra = sorted({k for m in signal_map.values() for k in m} - set(ctx_by_key))
    if extra:
        say(f"resolving {len(extra)} confirmation candles (point-in-time) ...")
    ctx2 = imp.build_contexts(union_signals(bars, extra, underlying), feats, resolve, premium_for, params)
    contexts = ctx1 + ctx2
    for x in ctx2:
        ctx_by_key[(x["signal_time"].isoformat(), x["direction"])] = x
    cache: dict = {}
    raw = imp.run_variant(contexts, feats, imp.BASELINE, square_off=square_off, lot_size=sh.LOT_SIZE, cache=cache)
    row_by_key = {acc.row_key(r): sh.classify_row(r, {}) for r in raw}
    snapshot = acc.snapshot_rows(row_by_key)

    bar_at = bars.set_index("timestamp")
    labels: dict[tuple[str, str], list[str]] = {}
    for k in [*row_by_key, *(acc.row_key(r) for r in base_rows)]:
        ts = pd.Timestamp(k[0]).tz_convert(imp.IST)
        if k not in labels and ts in bar_at.index:
            b = bar_at.loc[ts]
            labels[k] = [b["regime"], b["vol_regime"]]
    rows: dict[str, list[dict]] = {}
    directions: dict[str, str] = {}
    for name, mp in signal_map.items():
        d = name.split("_")[1]
        directions[name] = d
        out = []
        for k, setup in sorted(mp.items()):
            if k[1] != d:
                raise ValueError(f"{name}: foreign-direction signal {k}")
            base = row_by_key.get(k)
            if base is None:
                continue
            score = totals[d].get(setup[0])
            out.append(dict(base, strategy=name, s3_setup_time=setup[0], s3_score=score,
                            s3_tier=s3.score_tier(score), s3_regime=(labels.get(k) or [None])[0]))
        rows[name] = out

    say(f"truncation lookahead test ({len(lookahead_days or hist)} sessions x {len(cut_times)} cuts) ...")
    la_days = list(lookahead_days) if lookahead_days is not None else hist
    la_trunc = truncation_test(spot, la_days, allowed, cut_times, interval)
    la_opt = acc.option_feature_mutation_test(contexts, params.atr_period)
    la_mut = None
    if run_mutation:
        say("pipeline mutation test (scrambled candles after 12:00, contexts rebuilt) ...")
        la_mut = pipeline_mutation_test(spot, allowed, resolve, premium_for, params, underlying, la_days, interval)
    lookahead_ok = la_trunc["passed"] and la_opt["passed"] and (la_mut is None or la_mut["passed"])

    folds = lab.walk_forward_folds(hist, c.wf_train_days, c.wf_val_days, c.wf_test_days)
    scores = {}
    for name, rs in rows.items():
        _, d, tt, mode = name.split("_", 3)
        sc = acc.score_rows(rs, periods, folds, labels, c)
        sc.update(cid=name, direction=d, threshold=int(tt[1:]), mode=mode, n_conditions=MODE_COMPLEXITY[mode],
                  lookahead_passed=lookahead_ok, data_status="OK", tier=acc.sample_tier(sc["OOS"]["evaluated"], c),
                  day_stability=day_stability(rs, hist), day_stability_oos=day_stability(rs, periods["OOS"]))
        sc["selection_score"] = acc.selection_score(sc["TRAIN"], sc["VALIDATION"], c, c.min_train_trades,
                                                    c.min_val_trades)
        sc["gates"] = {**acc.gates(sc, c), "no_excessive_complexity": sc["n_conditions"] <= 2}
        sc["acc_status"], sc["failed"] = acc.candidate_status(sc, c)
        sc["status"] = s3_status(sc["acc_status"])
        sc["final_score"] = acc.final_score(sc, c)
        scores[name] = sc

    finalists = {d: acc.pick(scores, [n for n in scores if scores[n]["direction"] == d]) for d in s3.DIRECTIONS}
    max_val = {d: max((scores[n]["VALIDATION"]["evaluated"] for n in scores if scores[n]["direction"] == d), default=0)
               for d in s3.DIRECTIONS}
    n_cond = {n: scores[n]["n_conditions"] for n in scores}
    procedure = {d: acc.wf_procedure([n for n in scores if scores[n]["direction"] == d], rows, n_cond, folds, c)
                 for d in s3.DIRECTIONS}
    hist_base = lab.period_rows(list(base_rows), hist)
    s2 = {d: acc.score_rows([r for r in hist_base if r["direction"] == d], periods, folds, labels, c)
          for d in s3.DIRECTIONS}
    s2["COMBINED"] = acc.score_rows(hist_base, periods, folds, labels, c)
    comb_rows = [r for d in s3.DIRECTIONS if finalists[d] for r in rows[finalists[d]]]
    combined = acc.score_rows(comb_rows, periods, folds, labels, c) if comb_rows else None
    t60 = [r for d in s3.DIRECTIONS for r in rows[cid(d, 60, "A_NEXT_BAR")]]
    tiers = tier_table(t60, periods)

    say("integrity audit ...")
    integrity = (acc.data_checks(spot, hist, interval) + acc.context_checks(contexts, row_by_key, interval)
                 + [acc._check("L1", "Regime, score components and setups unchanged when recomputed from candles <= cut",
                               la_trunc["passed"], f"{la_trunc['cuts']} cuts, failures {la_trunc['failures'][:5] or 'none'}"),
                    acc._check("L2", "Option features + premium ATR unchanged when later premium candles are scrambled",
                               la_opt["passed"], f"{la_opt['checked']} signals, failures {la_opt['failure_count']}"),
                    acc._check("L3", "Pipeline mutation test (contract, premium ATR, stop distance, entry > signal)",
                               None if la_mut is None else la_mut["passed"],
                               "skipped" if la_mut is None else f"{la_mut['cuts']} cuts, failures {len(la_mut['failures'])}"),
                    acc._check("L4", "Existing S2 baseline replay mutation test", baseline_lookahead_passed, "12:00 cut"),
                    acc._check("L5", "Premium confirmation requires real option candles (missing -> NO_SIGNAL)", True,
                               f"NO_SIGNAL reasons at scored candles: {dict(no_signal) or 'none'}")]
                 + acc.split_checks(split_days, splits, today)
                 + isolation_checks(rows, directions, row_by_key, snapshot))
    integrity_ok = all(x["status"] != acc.FAIL for x in integrity)
    rec = recommend(finalists, scores, integrity_ok, max_val, c)
    achieved = rec == ROBUST_70
    ranking = sorted(scores, key=lambda n: (STATUS_ORDER[scores[n]["status"]], -scores[n]["final_score"], n))
    return {
        "heading": HEADING, "banner": BANNER, "trading_mode": "PAPER", "live_trading_enabled": False,
        "production_s2_changed": False, "live_orders": 0,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"), "today": today.isoformat(),
        "rules": rules or {}, "criteria": asdict(c), "weights": dict(s3.WEIGHTS), "thresholds": list(s3.THRESHOLDS),
        "profiles": {d: asdict(p) for d, p in s3.PROFILES.items()}, "modes": dict(MODE_TEXT),
        "splits": {s: {"from": splits[s][0].isoformat(), "to": splits[s][1].isoformat(), "days": split_days[s]}
                   for s in SPLITS},
        "counts": {"candidates": len(scores), "scored_candles": len(union1), "confirmation_candles": len(extra),
                   "no_signal_reasons": dict(no_signal), "conflicts": conflicts,
                   "vwap_source": "TWAP_PROXY (NIFTY index volume is zero)" if bars["vwap"].isna().all() else "VWAP",
                   "volume_source": "option premium volume (index volume is zero)"},
        "scores": scores, "ranking": ranking, "finalists": finalists, "wf_procedure": procedure, "s2": s2,
        "combined": combined, "tiers": tiers, "best_tier": best_tier(tiers),
        "calibration": {"TRAIN+VALIDATION": calibration(tiers, "TRAIN+VALIDATION"), "OOS": calibration(tiers, "OOS")},
        "lookahead": {"truncation": la_trunc, "option": la_opt, "mutation": la_mut, "passed": lookahead_ok},
        "integrity": integrity, "integrity_ok": integrity_ok, "recommendation": rec, "target_achieved": achieved,
        "forward_validation": FORWARD_REQUIRED if rec in (ROBUST_70, PROMISING) else None,
        "formulas": {"score": acc.SCORE_FORMULA, "selection": acc.SELECTION_FORMULA},
        "_rows": rows,
    }


# ------------------------------------------------------------------ reports
_pct, _pf, _rs, _csv = sh._pct, sh._pf_s, sh._rs, acc._csv


def _metrics_row(m: dict) -> dict:
    return {"trades": m["evaluated"], "wins": m["wins"], "losses": m["losses"], "accuracy": m["win_rate"],
            "profit_factor": m["profit_factor"], "net_pnl": m["net_pnl"], "avg_r": m["average_r"],
            "max_dd": m["max_drawdown"], "max_losing_streak": m["max_consecutive_losses"]}


def threshold_csv_rows(a: dict) -> list[dict]:
    out = []
    for t in s3.THRESHOLDS:
        for mode in MODES:
            for p in PERIODS:
                b, s = a["scores"][cid("BUY", t, mode)], a["scores"][cid("SELL", t, mode)]
                rb = a["_rows"][cid("BUY", t, mode)] + a["_rows"][cid("SELL", t, mode)]
                comb = lab.perf(lab.period_rows(rb, a["splits"][p]["days"] if p != "ALL" else
                                                [d for x in SPLITS for d in a["splits"][x]["days"]]))
                for scope, m in (("BUY", b[p]), ("SELL", s[p]), ("COMBINED", comb)):
                    out.append({"threshold": t, "mode": mode, "period": p, "scope": scope, **_metrics_row(m)})
    return out


def candidate_csv_rows(a: dict) -> list[dict]:
    out = []
    for n in a["ranking"]:
        s = a["scores"][n]
        row = {"candidate": n, "direction": s["direction"], "threshold": s["threshold"], "mode": s["mode"],
               "complexity": s["n_conditions"]}
        for p in PERIODS:
            row.update({f"{p.lower()}_{k}": v for k, v in _metrics_row(s[p]).items()})
        w, dy = s["walk_forward"], s["day_stability"]
        row.update({"oos_sample_grade": s["tier"], "wf_accuracy": w["wf_accuracy"], "wf_pf": w["wf_pf"],
                    "wf_profitable_share": w["wf_profitable_share"], "wf_consistency": w["wf_consistency"],
                    "profitable_day_share": dy["profitable_day_share"], "best_day_share": dy["best_day_share_of_net"],
                    "regime_share": s["regime_share"], "selection_score": s["selection_score"],
                    "final_score": s["final_score"], "status": s["status"], "failed_gates": s["failed"],
                    "finalist": n in a["finalists"].values()})
        out.append(row)
    return out


def _named(a: dict) -> dict[str, dict]:
    out = {f"S2_{k}": v for k, v in a["s2"].items()}
    for d, n in a["finalists"].items():
        if n:
            out[n] = a["scores"][n]
    if a["combined"]:
        out["S3_COMBINED_FINALISTS"] = a["combined"]
    return out


def walk_forward_csv_rows(a: dict) -> list[dict]:
    out = []
    for n in a["ranking"]:
        for f in a["scores"][n]["walk_forward"]["folds"]:
            out.append({"kind": "CANDIDATE", "candidate": n, **f})
    for d, p in a["wf_procedure"].items():
        for f in p["folds"]:
            out.append({"kind": f"SELECTION_PROCEDURE_{d}", "candidate": f["selected"] or "NONE_ELIGIBLE", **f})
    return out


def day_csv_rows(a: dict) -> list[dict]:
    out = []
    split_of = {d: s for s in SPLITS for d in a["splits"][s]["days"]}
    for n, s in _named(a).items():
        p = s["day_all"]
        for d, v in p["daily"].items():
            out.append({"candidate": n, "session_date": d, "split": split_of.get(d), "pnl": v,
                        "best_day": p["best_day"], "worst_day": p["worst_day"], "median_day_pnl": p["median_day_pnl"],
                        "profitable_day_share": p["profitable_day_share"], "active_days": p["days"],
                        "best_day_share_of_net": p["best_day_share_of_net"]})
    return out


def regime_csv_rows(a: dict) -> list[dict]:
    out = []
    for n, s in _named(a).items():
        for r in (*COMPARE_REGIMES, "NORMAL_VOLATILITY"):
            m = s["regimes"].get(r)
            out.append({"candidate": n, "regime": r, "trades": m["evaluated"] if m else 0,
                        "accuracy": m["win_rate"] if m else None, "profit_factor": m["profit_factor"] if m else None,
                        "net_pnl": m["net_pnl"] if m else 0.0})
    return out


def compare_rows(a: dict) -> list[dict]:
    out = []
    items = [(f"S2 {k}", v, "CURRENT") for k, v in a["s2"].items()]
    for d in s3.DIRECTIONS:
        n = a["finalists"][d]
        items.append((f"S3 {d}" + (f" ({n})" if n else " (no eligible candidate)"), a["scores"][n] if n else None,
                      a["scores"][n]["status"] if n else INSUFFICIENT_DATA))
    items.append(("S3 COMBINED", a["combined"], "FINALISTS_TOGETHER" if a["combined"] else INSUFFICIENT_DATA))
    for name, s, status in items:
        if not s:
            out.append({"strategy": name, "status": status})
            continue
        o, w = s["OOS"], s["walk_forward"]
        out.append({"strategy": name, "accuracy": o["win_rate"], "trades": o["evaluated"], "pf": o["profit_factor"],
                    "pnl": o["net_pnl"], "avg_r": o["average_r"], "max_dd": o["max_drawdown"],
                    "max_loss_streak": o["max_consecutive_losses"], "best_day_share": o["best_day_share"],
                    "wf_profitable_share": w["wf_profitable_share"], "wf_pf": w["wf_pf"],
                    "sample_grade": acc.sample_tier(o["evaluated"], acc.Criteria()), "status": status})
    return out


def _m(m: dict | None) -> str:
    if not m:
        return "-"
    return f"{m['evaluated']} / {_pct(m['win_rate'])} / PF {_pf(m['profit_factor'])} / Rs {_rs(m['net_pnl'])}"


def markdown(a: dict) -> str:
    sc, fin, c = a["scores"], a["finalists"], a["criteria"]
    md = [f"# {HEADING}", "", f"**{BANNER}** | production S2 NOT modified | LIVE_TRADING_ENABLED=false | "
          f"generated {a['generated_at']}", "",
          "## 1. Architecture", "",
          "Per completed 5m candle and direction: causal regime -> separate BUY/SELL direction score -> option "
          "premium confirmation -> threshold -> entry confirmation mode -> BUY / SELL / NO_SIGNAL. "
          "Missing data, RANGE_CHOP/UNCERTAIN regime, missing premium data or score below threshold = NO_SIGNAL. "
          "Both directions qualifying on one candle = NO_SIGNAL.", "",
          f"- VWAP source: {a['counts']['vwap_source']}", f"- Volume source: {a['counts']['volume_source']}",
          "- Exits unchanged (S2-compatible): 1.5 x premium ATR14 stop, 2R target, 15:15 square-off, AMBIGUOUS = -1R.",
          "", "| Component | Weight (starting research weight, not optimised) |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in a["weights"].items()]
    md += ["", "| Profile | Rules (BUY and SELL intentionally different) |", "|---|---|"]
    for d, p in a["profiles"].items():
        md.append(f"| {d} | trend regime {p['trend_regime']}; RSI full {p['rsi_full']} half {p['rsi_half']}; EMA ext <= "
                  f"{p['max_ext_atr']} ATR; VWAP ext <= {p['max_vwap_ext_atr']} ATR; breakout <= {p['max_break_atr']} "
                  f"ATR; premium rule {p['premium_rule']}; {'; '.join(p['notes'])} |")
    md += ["", f"Regimes: TREND_UP/DOWN (EMA20 vs EMA50, price beyond EMA20, |slope| >= {s3.TREND_SLOPE} ATR), "
           f"RANGE_CHOP (>= {s3.CHOP_FLIPS} direction flips in 12 candles or flat EMA near price), HIGH/LOW_VOLATILITY "
           f"(ATR percentile >= {s3.HIGH_VOL} / <= {s3.LOW_VOL}), UNCERTAIN (inputs missing or a candle gap).", "",
           "| Entry mode | Rule |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in a["modes"].items()]
    md += ["", "## 2. Integrity", "", f"Overall: **{'PASS' if a['integrity_ok'] else 'FAIL'}**", "",
           "| ID | Check | Status | Detail |", "|---|---|---|---|"]
    md += [f"| {x['id']} | {x['check']} | {x['status']} | {x['detail']} |" for x in a["integrity"]]
    md += ["", "## 3. Threshold comparison (A_NEXT_BAR; all modes in s3_threshold_comparison.csv)", "",
           "| Threshold | Period | BUY n/acc/PF/P&L | SELL n/acc/PF/P&L |", "|---|---|---|---|"]
    for t in s3.THRESHOLDS:
        for p in PERIODS:
            md.append(f"| {t} | {p} | {_m(sc[cid('BUY', t, 'A_NEXT_BAR')][p])} | {_m(sc[cid('SELL', t, 'A_NEXT_BAR')][p])} |")
    md += ["", "## 4. Confidence tiers (T60 A_NEXT_BAR signals bucketed by score)", "",
           "| Direction | Tier | TRAIN+VAL n / acc | OOS n / acc / PF / P&L |", "|---|---|---|---|"]
    for d in s3.DIRECTIONS:
        for name, _, _ in s3.TIERS:
            tv = next(t for t in a["tiers"] if t["direction"] == d and t["tier"] == name and t["period"] == "TRAIN+VALIDATION")
            o = next(t for t in a["tiers"] if t["direction"] == d and t["tier"] == name and t["period"] == "OOS")
            md.append(f"| {d} | {name} | {tv['trades']} / {_pct(tv['accuracy'])} | {o['trades']} / {_pct(o['accuracy'])} / "
                      f"{_pf(o['profit_factor'])} / {_rs(o['net_pnl'])} |")
    bt = a["best_tier"]
    md += ["", f"Best tier (chosen on TRAIN+VALIDATION, >= 10 trades): " + (
        f"{bt['direction']} {bt['tier']} ({bt['trades']} / {_pct(bt['accuracy'])}); OOS {bt['oos']['trades']} / "
        f"{_pct(bt['oos']['accuracy'])}" if bt else "none with enough trades"),
           f"Higher confidence -> higher accuracy? TRAIN+VALIDATION: {a['calibration']['TRAIN+VALIDATION']}; "
           f"OOS: {a['calibration']['OOS']} (None = fewer than two tiers with >= 5 trades)."]
    md += ["", "## 5. Selection (TRAIN + VALIDATION only)", "", a["formulas"]["selection"], "",
           "| Direction | Finalist | Selection | TRAIN | VALIDATION | OOS | Status | Failed gates |",
           "|---|---|---|---|---|---|---|---|"]
    for d in s3.DIRECTIONS:
        n = fin[d]
        md.append(f"| {d} | {n or 'NONE ELIGIBLE'} | {sc[n]['selection_score'] if n else '-'} | "
                  f"{_m(sc[n]['TRAIN']) if n else '-'} | {_m(sc[n]['VALIDATION']) if n else '-'} | "
                  f"{_m(sc[n]['OOS']) if n else '-'} | {sc[n]['status'] if n else '-'} | "
                  f"{', '.join(sc[n]['failed']) if n else '-'} |")
    md += ["", "## 6. Walk-forward", "", "| Strategy | WF acc | WF trades | WF PF | WF P&L | Profitable windows | "
           "Consistency |", "|---|---|---|---|---|---|---|"]
    for n, s in _named(a).items():
        w = s["walk_forward"]
        md.append(f"| {n} | {_pct(w['wf_accuracy'])} | {w['wf_trades']} | {_pf(w['wf_pf'])} | {_rs(w['wf_pnl'])} | "
                  f"{_pct(w['wf_profitable_share'])} | {_pct(w['wf_consistency'])} |")
    for d, p in a["wf_procedure"].items():
        md.append(f"| SELECTION PROCEDURE {d} | {_pct(p['wf_accuracy'])} | {p['wf_trades']} | {_pf(p['wf_pf'])} | "
                  f"{_rs(p['wf_pnl'])} | {_pct(p['wf_profitable_share'])} | - |")
    md += ["", "## 7. Day stability", "", "| Strategy | Active days | Best day | Worst day | Median day | "
           "Profitable days | Best-day share |", "|---|---|---|---|---|---|---|"]
    for n, s in _named(a).items():
        p = s["day_all"]
        md.append(f"| {n} | {p['days']} | {p['best_day'] or '-'} ({_rs(p['best_day_pnl'])}) | {p['worst_day'] or '-'} "
                  f"({_rs(p['worst_day_pnl'])}) | {_rs(p['median_day_pnl'])} | {_pct(p['profitable_day_share'])} | "
                  f"{_pct(p['best_day_share_of_net'])} |")
    md += ["", "## 8. Regime stability", "", "| Strategy | Regime | Trades | Accuracy | PF | P&L |",
           "|---|---|---|---|---|---|"]
    for r in regime_csv_rows(a):
        md.append(f"| {r['candidate']} | {r['regime']} | {r['trades']} | {_pct(r['accuracy'])} | "
                  f"{_pf(r['profit_factor'])} | {_rs(r['net_pnl'])} |")
    md += ["", "## 9. S2 vs S3 (OOS)", "", "| Strategy | Accuracy | Trades | PF | P&L | Avg R | Max DD | Max loss "
           "streak | Best-day share | WF profitable windows | Sample grade | Status |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in compare_rows(a):
        if "trades" not in r:
            md.append(f"| {r['strategy']} | - | - | - | - | - | - | - | - | - | - | {r['status']} |")
            continue
        md.append(f"| {r['strategy']} | {_pct(r['accuracy'])} | {r['trades']} | {_pf(r['pf'])} | {_rs(r['pnl'])} | "
                  f"{r['avg_r']} | {_rs(r['max_dd'])} | {r['max_loss_streak']} | {_pct(r['best_day_share'])} | "
                  f"{_pct(r['wf_profitable_share'])} | {r['sample_grade']} | {r['status']} |")
    statuses = Counter(s["status"] for s in sc.values())
    reasons = Counter(g for s in sc.values() for g in s["failed"])
    md += ["", "## 10. Anti-overfitting", "",
           f"- {len(sc)} pre-declared candidates (7 thresholds x 5 modes x 2 directions); weights/profiles fixed in code.",
           "- Finalists are chosen on TRAIN+VALIDATION; OOS is never used for selection; forward data never loaded.",
           f"- Sample grades (OOS trades): A >= {c['tier_a']}, B >= {c['tier_b']}, C >= {c['tier_c']}, D below.",
           "- OOS 2026-09-25..10-01 was visible in earlier project research; treat any pass as needing forward PAPER.",
           f"- Status counts: {dict(statuses)}", f"- Failed gates: {dict(reasons.most_common())}"]
    md += ["", "## 11. 70% OOS target", "", f"**{'ACHIEVED' if a['target_achieved'] else 'NOT ACHIEVED'}**", ""]
    if not a["target_achieved"]:
        md.append(NOT_ACHIEVED_TEXT)
    md += ["", "## 12. Recommendation", "", f"**{a['recommendation']}**" + (
        f" -> {a['forward_validation']} (separate live-data PAPER forward test; not production-ready)"
        if a["forward_validation"] else ""), "",
        "S2 production changed: NO. Live orders: 0. TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false."]
    return "\n".join(md) + "\n"


REPORT_FILES = ("s3_high_confidence.md", "s3_threshold_comparison.csv", "s3_candidates.csv", "s3_walk_forward.csv",
                "s3_day_stability.csv", "s3_regime_stability.csv")


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {n: out_dir / n for n in REPORT_FILES}
    paths["s3_high_confidence.md"].write_text(markdown(a), encoding="utf-8")
    _csv(paths["s3_threshold_comparison.csv"], threshold_csv_rows(a))
    _csv(paths["s3_candidates.csv"], candidate_csv_rows(a))
    _csv(paths["s3_walk_forward.csv"], walk_forward_csv_rows(a))
    _csv(paths["s3_day_stability.csv"], day_csv_rows(a))
    _csv(paths["s3_regime_stability.csv"], regime_csv_rows(a))
    return paths
