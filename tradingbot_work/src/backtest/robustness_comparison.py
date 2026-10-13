"""ROBUSTNESS COMPARISON LAB: four fixed S2 configurations (PAPER, research only, no orders, no new thresholds).

A. CURRENT_S2                      - unchanged S2 (BUY + SELL)
B. SELL + BUY_BREAKOUT_CONFIRM     - SELL unchanged; BUY breakout distance <= 0.50 x ATR14
C. SELL + BUY_NON_CHOPPY_ONLY      - SELL unchanged; BUY only with < 6 sign changes so far today
D. SELL + BUY_MORNING_ONLY         - SELL unchanged; BUY only 09:30-11:00

Rules and thresholds are exactly those of ``strategy_improvement_lab``; rows come from the same historical replay.
Every configuration is scored as BUY-only, SELL-only and COMBINED, so the BUY filter's own contribution is visible
separately from the (identical) SELL leg. Nothing here selects or tunes anything; 2026-10-05+ is never read.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.backtest import strategy_improvement_lab as lab
from src.backtest.strategy_improvement_lab import Candidate

HEADING = "ROBUSTNESS COMPARISON LAB"
CURRENT = "A_CURRENT_S2"
CONFIGS = (
    Candidate(CURRENT, "CONFIG", note="current S2, unchanged"),
    Candidate("B_SELL+BUY_BREAKOUT_CONFIRM", "CONFIG", buy=("breakout",)),
    Candidate("C_SELL+BUY_NON_CHOPPY_ONLY", "CONFIG", buy=("regime:non_choppy",)),
    Candidate("D_SELL+BUY_MORNING_ONLY", "CONFIG", buy=("time:0930-1100",)),
)
SCOPES = ("BUY", "SELL", "COMBINED")
KEEP, PAPER_TEST = "KEEP_CURRENT", "PAPER_TEST_CANDIDATE"


@dataclass(frozen=True)
class ComparisonCriteria:
    lab: lab.LabCriteria = field(default_factory=lab.LabCriteria)
    min_buy_oos_trades: int = 30
    min_wf_positive_share: float = 0.5
    wf_train: int = 6
    wf_validation: int = 2
    wf_test: int = 2


def scope_rows(rows: Sequence[dict], scope: str) -> list[dict]:
    return list(rows) if scope == "COMBINED" else [r for r in rows if r["direction"] == scope]


def _pf(m: dict) -> float:
    return lab._n(m["profit_factor"], 0.0)


def walk_forward(rows: Sequence[dict], folds: Sequence[dict], collapse_ratio: float) -> list[dict]:
    """Fixed configuration on rolling windows: train/validation/test metrics per fold, nothing is selected."""
    out = []
    for i, f in enumerate(folds, 1):
        m = {k: lab.perf(lab.period_rows(rows, f[k])) for k in ("train", "validation", "test")}
        t, te = m["train"], m["test"]
        collapse = bool(te["evaluated"] and _pf(t) > 1 and _pf(te) < collapse_ratio * min(_pf(t), 10.0))
        out.append({"fold": i, **{f"{k}_days": f"{f[k][0]}..{f[k][-1]}" for k in ("train", "validation", "test")},
                    "test_day_list": list(f["test"]), **m, "test_collapse": collapse})
    return out


def score_scope(rows: Sequence[dict], base_rows: Sequence[dict], periods: dict, folds: Sequence[dict], facts: dict,
                p: dict, lookahead_passed: bool, c: ComparisonCriteria, n_checks: int) -> dict[str, Any]:
    m = {s: lab.perf(lab.period_rows(rows, d)) for s, d in periods.items()}
    wf = walk_forward(rows, folds, c.lab.collapse_ratio)
    traded = [f for f in wf if f["test"]["evaluated"]]
    wf_share = sum(f["test"]["net_pnl"] > 0 for f in traded) / len(traded) if traded else None
    reg_share, reg = lab.group_stability(rows, lambda r: r["regimes"] or ["unlabelled"], c.lab.min_group_trades)
    _, pit = lab.group_stability(
        rows, lambda r: [lab.factor_buckets(facts.get(lab.row_key(r)), r, p).get("10_regime_point_in_time",
                                                                                   "UNAVAILABLE")],
        c.lab.min_group_trades)
    time_share, times = lab.group_stability(rows, lambda r: [r.get("time_bucket") or "other"], c.lab.min_group_trades)
    big = {k: v for k, v in reg.items() if v["evaluated"] >= c.lab.min_group_trades}
    daily: dict[str, dict] = {}
    for r in rows:
        if r["state"] == sh.EVALUATED:
            d = daily.setdefault(r["session_date"], {"evaluated": 0, "wins": 0, "net_pnl": 0.0})
            d["evaluated"] += 1
            d["wins"] += r["outcome"] == lab.imp.WIN
            d["net_pnl"] += float(r["pnl"])
    base_n = lab.perf(base_rows)["evaluated"]
    s = {**m, "retention": m["ALL"]["evaluated"] / base_n if base_n else None, "regime_stability": reg_share,
         "regime_groups": len(big), "profitable_regime_groups": sum(v["net_pnl"] > 0 for v in big.values()),
         "time_stability": time_share, "day_stability": m["ALL"]["day_stability"], "wf_positive_share": wf_share,
         "wf_collapses": sum(f["test_collapse"] for f in wf), "robustness_ok": None,
         "lookahead_passed": lookahead_passed, "n_checks": n_checks, "family": "CONFIG"}
    s.update(lab.candidate_status(s, c.lab))
    if s["status"] in (lab.ROBUST, lab.PROMISING, lab.NOT_PROMISING) and wf_share is not None \
            and wf_share < c.min_wf_positive_share:
        s["status"], s["failed_checks"] = lab.REJECTED, s["failed_checks"] + ["walk_forward_collapse"]
    s.update(walk_forward=wf, regime_eod=reg, regime_pit=pit, time_of_day=times, daily=daily)
    return s


def verdicts(results: dict[tuple[str, str], dict], c: ComparisonCriteria) -> dict[str, Any]:
    """PAPER_TEST_CANDIDATE only when a configuration's own BUY leg is sufficiently sampled and profitable on
    VALIDATION, OOS and walk-forward, and the COMBINED configuration beats CURRENT_S2 without day dependence."""
    cur = results[(CURRENT, "COMBINED")]
    out = {}
    for cfg in CONFIGS[1:]:
        comb, buy, sell = (results[(cfg.name, s)] for s in ("COMBINED", "BUY", "SELL"))
        checks = {
            "combined_status_promising": comb["status"] in (lab.ROBUST, lab.PROMISING),
            "buy_oos_trades": buy["OOS"]["evaluated"] >= c.min_buy_oos_trades,
            "buy_validation_profitable": _pf(buy["VALIDATION"]) > 1 and buy["VALIDATION"]["net_pnl"] > 0,
            "buy_oos_profitable": _pf(buy["OOS"]) > 1 and buy["OOS"]["net_pnl"] > 0,
            "buy_walk_forward_stable": (buy["wf_positive_share"] or 0) >= c.min_wf_positive_share,
            "combined_beats_current_oos": (comb["OOS"]["net_pnl"] > cur["OOS"]["net_pnl"]
                                           and _pf(comb["OOS"]) > _pf(cur["OOS"])),
            "combined_not_worse_validation": _pf(comb["VALIDATION"]) >= _pf(cur["VALIDATION"]),
            "combined_not_day_dependent": comb["OOS"]["pnl_excl_best_2_days"] > 0,
            "lookahead": bool(comb["lookahead_passed"]),
        }
        out[cfg.name] = {
            "eligible": all(checks.values()), "checks": checks,
            "failed": [k for k, v in checks.items() if not v],
            "buy_leg_pnl": {pr: buy[pr]["net_pnl"] for pr in lab.PERIODS},
            "buy_leg_trades": {pr: buy[pr]["evaluated"] for pr in lab.PERIODS},
            "vs_sell_only_pnl": {pr: round(comb[pr]["net_pnl"] - sell[pr]["net_pnl"], 2) for pr in lab.PERIODS},
            "vs_current_pnl": {pr: round(comb[pr]["net_pnl"] - cur[pr]["net_pnl"], 2) for pr in lab.PERIODS},
        }
    eligible = [n for n, v in out.items() if v["eligible"]]
    best = lab.rank({n: results[(n, "COMBINED")] for n in eligible})[0] if eligible else None
    return {"recommendation": PAPER_TEST if best else KEEP, "candidate": best, "configs": out}


def build_comparison(*, base_rows: Sequence[dict], facts: dict, days: Sequence[str], splits: dict,
                     lookahead: Callable[[Sequence[Candidate], dict], dict] | None = None,
                     baseline_lookahead_passed: bool = True, criteria: ComparisonCriteria = ComparisonCriteria(),
                     forward: list[dict] | None = None) -> dict[str, Any]:
    split_days, unassigned = so.assign_days(days, splits)
    hist = [d for s in lab.SPLITS for d in split_days[s]]
    base = lab.period_rows(base_rows, hist)
    p = lab.train_params(base, facts, split_days["TRAIN"])
    rows = {c.name: lab.apply_candidate(base, facts, c, p) for c in CONFIGS}
    la = lookahead(CONFIGS, p) if lookahead else {c.name: {"cuts": 0, "failures": [], "passed": True} for c in CONFIGS}
    feats_ok = la.get("_features", {}).get("passed", True)
    passed = {c.name: bool(la[c.name]["passed"] and (baseline_lookahead_passed if c.name == CURRENT else feats_ok))
              for c in CONFIGS}
    periods = {**split_days, "ALL": hist}
    folds = lab.walk_forward_folds(hist, criteria.wf_train, criteria.wf_validation, criteria.wf_test)
    results = {(c.name, s): score_scope(scope_rows(rows[c.name], s), scope_rows(rows[CURRENT], s), periods, folds,
                                        facts, p, passed[c.name], criteria, c.n_checks)
               for c in CONFIGS for s in SCOPES}
    v = verdicts(results, criteria)
    return {
        "heading": HEADING, "trading_mode": "PAPER", "live_trading_enabled": False, "observation_only": True,
        "generated_at": datetime.now(sh.IST).isoformat(timespec="seconds"),
        "configs": {c.name: lab.describe(c, p) for c in CONFIGS},
        "splits": {s: {"label": so.SPLIT_LABELS[s], "from": splits[s][0].isoformat(), "to": splits[s][1].isoformat(),
                       "days": split_days[s]} for s in lab.SPLITS},
        "unassigned_days": unassigned, "criteria": asdict(criteria), "params": p,
        "lookahead": la, "lookahead_passed": passed,
        "results": {f"{n}|{s}": r for (n, s), r in results.items()},
        "verdicts": v, "recommendation": v["recommendation"], "forward": forward or [],
        "notes": {
            "no_tuning": "Thresholds are the lab's pre-declared values; nothing was re-optimised here.",
            "oos": "OOS 2026-09-25..10-01 was seen by earlier research: PROMISING is the ceiling, never ROBUST.",
            "forward": "2026-10-05 onwards (including 2026-10-06 live PAPER data) is excluded from every number.",
            "regime": "EOD regime labels are reporting-only; the point-in-time regime is what a filter could use.",
            "sell": "The SELL leg is identical in all four configurations; differences come only from BUY.",
        },
    }


# ------------------------------------------------------------------ reports
_pct, _pf_s, _rs = sh._pct, sh._pf_s, sh._rs
SUMMARY_KEYS = ("signals", "evaluated", "wins", "losses", "win_rate", "profit_factor", "gross_profit", "gross_loss",
                "net_pnl", "max_drawdown", "average_r", "median_r", "max_consecutive_losses", "max_consecutive_wins",
                "day_stability", "best_day", "best_day_pnl", "best_day_share", "worst_day", "worst_day_pnl",
                "worst_day_share", "pnl_excl_best_day", "pnl_excl_best_2_days", "largest_win", "pnl_excl_largest_win")


def _items(a: dict):
    for key, r in a["results"].items():
        name, scope = key.split("|")
        yield name, scope, r


def comparison_rows(a: dict) -> list[dict]:
    out = []
    for name, scope, r in _items(a):
        for pr in lab.PERIODS:
            out.append({"row_type": "PERIOD", "config": name, "scope": scope, "period": pr, "bucket": "",
                        **{k: r[pr][k] for k in SUMMARY_KEYS}, "wf_positive_share": r["wf_positive_share"],
                        "wf_collapses": r["wf_collapses"], "regime_stability": r["regime_stability"],
                        "time_stability": r["time_stability"], "lookahead": "PASS" if r["lookahead_passed"] else "FAIL",
                        "status": r["status"], "failed_checks": ";".join(r["failed_checks"])})
        for bucket, m in sorted(r["time_of_day"].items()):
            out.append({"row_type": "TIME_OF_DAY", "config": name, "scope": scope, "period": "ALL", "bucket": bucket,
                        **{k: m[k] for k in SUMMARY_KEYS}, "wf_positive_share": None, "wf_collapses": None,
                        "regime_stability": None, "time_stability": None, "lookahead": None, "status": None,
                        "failed_checks": None})
    return out


def wf_rows(a: dict) -> list[dict]:
    out = []
    for name, scope, r in _items(a):
        for f in r["walk_forward"]:
            row = {"config": name, "scope": scope, "fold": f["fold"], "train": f["train_days"],
                   "validation": f["validation_days"], "test": f["test_days"]}
            for k in ("train", "validation", "test"):
                row.update({f"{k}_{x}": f[k][y] for x, y in (("trades", "evaluated"), ("win_rate", "win_rate"),
                                                            ("profit_factor", "profit_factor"), ("net_pnl", "net_pnl"))})
            row["test_collapse"] = f["test_collapse"]
            out.append(row)
    return out


def day_rows(a: dict) -> list[dict]:
    split_of = {d: s for s, sp in a["splits"].items() for d in sp["days"]}
    days = [d for sp in a["splits"].values() for d in sp["days"]]
    out = []
    for name, scope, r in _items(a):
        rows_by_day = {d: r["daily"].get(d, {"evaluated": 0, "wins": 0, "net_pnl": 0.0}) for d in days}
        cum = 0.0
        for d in days:
            m = rows_by_day[d]
            cum += m["net_pnl"]
            split_total = r[split_of[d]]["net_pnl"]
            out.append({"config": name, "scope": scope, "date": d, "split": split_of[d], "trades": m["evaluated"],
                        "wins": m["wins"], "win_rate": m["wins"] / m["evaluated"] if m["evaluated"] else None,
                        "net_pnl": round(m["net_pnl"], 2), "cumulative_pnl": round(cum, 2),
                        "share_of_split_pnl": (m["net_pnl"] / split_total) if split_total > 0 else None})
    return out


def regime_rows(a: dict) -> list[dict]:
    out = []
    for name, scope, r in _items(a):
        for kind, groups in (("POINT_IN_TIME", r["regime_pit"]), ("EOD_LABEL_REPORTING_ONLY", r["regime_eod"])):
            for lab_, m in sorted(groups.items()):
                out.append({"config": name, "scope": scope, "kind": kind, "regime": lab_, "trades": m["evaluated"],
                            "wins": m["wins"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
                            "net_pnl": m["net_pnl"], "max_drawdown": m["max_drawdown"], "average_r": m["average_r"]})
    return out


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: lab._fmt(v) for k, v in r.items()})


def markdown(a: dict) -> str:
    md = [f"# {HEADING}", "", "PAPER mode | research only | no orders | production S2 NOT modified | "
          "no thresholds re-optimised", "", "## Configurations", "", "| Config | BUY | SELL |", "|---|---|---|"]
    md += [f"| {n} | {d['BUY']} | {d['SELL']} |" for n, d in a["configs"].items()]
    md += ["", "## Splits", "", "| Split | From | To | Sessions |", "|---|---|---|---|"]
    md += [f"| {sp['label']} | {sp['from']} | {sp['to']} | {len(sp['days'])} |" for sp in a["splits"].values()]
    for pr in lab.PERIODS:
        md += ["", f"## {pr}", "", "| Config | Scope | Trades | WR | PF | Net P&L | Max DD | Avg R | Max loss streak | "
               "Best day share | Worst day share | P&L excl. best 2 days |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, scope, r in _items(a):
            m = r[pr]
            md.append(f"| {name} | {scope} | {m['evaluated']} | {_pct(m['win_rate'])} | {_pf_s(m['profit_factor'])} | "
                      f"{_rs(m['net_pnl'])} | {_rs(m['max_drawdown'])} | {m['average_r']} | "
                      f"{m['max_consecutive_losses']} | {_pct(m['best_day_share'])} | {_pct(m['worst_day_share'])} | "
                      f"{_rs(m['pnl_excl_best_2_days'])} |")
    md += ["", "## Stability and status", "", "| Config | Scope | OOS day stability | Walk-forward positive folds | "
           "WF collapses | Regime stability | Time stability | Lookahead | Status | Reasons |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, scope, r in _items(a):
        md.append(f"| {name} | {scope} | {_pct(r['OOS']['day_stability'])} | {_pct(r['wf_positive_share'])} | "
                  f"{r['wf_collapses']} | {_pct(r['regime_stability'])} | {_pct(r['time_stability'])} | "
                  f"{'PASS' if r['lookahead_passed'] else 'FAIL'} | {r['status']} | "
                  f"{', '.join(r['failed_checks']) or '-'} |")
    md += ["", "## BUY leg contribution (configuration minus SELL-only)", "",
           "| Config | Period | BUY trades | BUY P&L | vs SELL-only | vs CURRENT_S2 |", "|---|---|---|---|---|---|"]
    for n, v in a["verdicts"]["configs"].items():
        for pr in lab.PERIODS:
            md.append(f"| {n} | {pr} | {v['buy_leg_trades'][pr]} | {_rs(v['buy_leg_pnl'][pr])} | "
                      f"{_rs(v['vs_sell_only_pnl'][pr])} | {_rs(v['vs_current_pnl'][pr])} |")
    md += ["", "## Eligibility for PAPER test", ""]
    for n, v in a["verdicts"]["configs"].items():
        md.append(f"- **{n}**: {'ELIGIBLE' if v['eligible'] else 'not eligible'}"
                  + (f" (failed: {', '.join(v['failed'])})" if v["failed"] else ""))
    md += ["", "## Lookahead", ""]
    md += [f"- {n}: {'PASS' if ok else 'FAIL'}" for n, ok in a["lookahead_passed"].items()]
    md += ["", f"## RECOMMENDATION: **{a['recommendation']}**"
           + (f" ({a['verdicts']['candidate']})" if a["verdicts"]["candidate"] else ""), "",
           "Nothing is deployed automatically. Production S2 stays unchanged until explicitly approved.", "", "Notes:"]
    md += [f"- {x}" for x in a["notes"].values()]
    md += ["", "---", "TRADING_MODE=PAPER, LIVE_TRADING_ENABLED=false, zero orders placed. Costs excluded."]
    return "\n".join(md) + "\n"


def write_reports(a: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = {n: out_dir / n for n in ("robustness_comparison.md", "robustness_comparison.csv",
                                  "walk_forward_comparison.csv", "day_stability.csv", "regime_stability.csv")}
    p["robustness_comparison.md"].write_text(markdown(a), encoding="utf-8")
    _csv(p["robustness_comparison.csv"], comparison_rows(a))
    _csv(p["walk_forward_comparison.csv"], wf_rows(a))
    _csv(p["day_stability.csv"], day_rows(a))
    _csv(p["regime_stability.csv"], regime_rows(a))
    return p
