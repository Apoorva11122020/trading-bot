from __future__ import annotations

import csv

import pytest

from src.backtest import robustness_comparison as rc
from src.backtest import strategy_improvement_lab as lab
from tests.test_strategy_improvement_lab import (OOS_DAYS, S2_CFG, SPLITS, TRAIN_DAYS, VAL_DAYS, _dataset, _la_rows,
                                                 _perf, _row, _spot)

DAYS = TRAIN_DAYS + VAL_DAYS + OOS_DAYS


def _build(**kw):
    rows, facts = _dataset()
    return rc.build_comparison(base_rows=rows, facts=facts, days=DAYS, splits=SPLITS, **kw), rows, facts


def test_exactly_four_configs_with_unchanged_sell():
    names = [c.name for c in rc.CONFIGS]
    assert names == ["A_CURRENT_S2", "B_SELL+BUY_BREAKOUT_CONFIRM", "C_SELL+BUY_NON_CHOPPY_ONLY",
                     "D_SELL+BUY_MORNING_ONLY"]
    by = {c.name: c for c in lab.predeclared_candidates()}
    for cfg, lab_name in zip(rc.CONFIGS[1:], ("BUY_BREAKOUT_CONFIRM", "BUY_NON_CHOPPY_ONLY", "BUY_MORNING_ONLY")):
        assert cfg.sell == () and cfg.buy == by[lab_name].buy  # same pre-declared rule, no new threshold
    assert rc.CONFIGS[0].buy == () and rc.CONFIGS[0].sell == ()


def test_scopes_split_buy_and_sell_and_sell_leg_is_identical():
    a, rows, _ = _build()
    for pr in lab.PERIODS:
        sells = {a["results"][f"{c.name}|SELL"][pr]["net_pnl"] for c in rc.CONFIGS}
        assert len(sells) == 1
        for c in rc.CONFIGS:
            r = {s: a["results"][f"{c.name}|{s}"][pr] for s in rc.SCOPES}
            assert r["COMBINED"]["evaluated"] == r["BUY"]["evaluated"] + r["SELL"]["evaluated"]
            assert r["COMBINED"]["net_pnl"] == pytest.approx(r["BUY"]["net_pnl"] + r["SELL"]["net_pnl"])
    v = a["verdicts"]["configs"]["B_SELL+BUY_BREAKOUT_CONFIRM"]
    assert v["vs_sell_only_pnl"]["OOS"] == pytest.approx(v["buy_leg_pnl"]["OOS"])


def test_walk_forward_folds_are_chronological_and_flag_collapse():
    a, _, _ = _build()
    wf = a["results"]["A_CURRENT_S2|COMBINED"]["walk_forward"]
    assert len(wf) == len(lab.walk_forward_folds(DAYS, 6, 2, 2))
    tests = [d for f in wf for d in f["test_day_list"]]
    assert len(tests) == len(set(tests))
    fold = {"train": [TRAIN_DAYS[0]], "validation": [TRAIN_DAYS[1]], "test": [TRAIN_DAYS[2]]}
    rows = [_row(TRAIN_DAYS[0], "10:00", "BUY", True), _row(TRAIN_DAYS[2], "10:00", "BUY", False)]
    assert rc.walk_forward(rows, [fold], 0.5)[0]["test_collapse"] is True


def _res(oos_buy, val_buy, comb_status=lab.PROMISING, comb_oos=None, wf=1.0):
    def scope(m, status=lab.PROMISING, share=wf):
        return {"TRAIN": m, "VALIDATION": m, "OOS": m, "ALL": m, "status": status, "wf_positive_share": share,
                "lookahead_passed": True, "day_stability": 0.6, "regime_stability": 0.7, "n_checks": 1}
    cur = _perf(150, 0.45, 1.5, 50000)
    out = {}
    for c in rc.CONFIGS:
        buy = cur if c.name == rc.CURRENT else oos_buy
        comb = cur if c.name == rc.CURRENT else (comb_oos or _perf(150, 0.55, 2.0, 80000))
        out[(c.name, "BUY")] = dict(scope(buy), VALIDATION=val_buy if c.name != rc.CURRENT else cur)
        out[(c.name, "SELL")] = scope(_perf(110, 0.55, 2.4, 70000))
        out[(c.name, "COMBINED")] = scope(comb, comb_status if c.name != rc.CURRENT else lab.PROMISING)
    for k in out:
        for pr in lab.PERIODS:
            out[k][pr] = dict(out[k][pr], pnl_excl_best_2_days=out[k][pr]["net_pnl"] * 0.5)
    return out


def test_recommendation_requires_a_sufficiently_sampled_profitable_buy_leg():
    c = rc.ComparisonCriteria()
    few = _res(_perf(8, 0.25, 0.7, -2000), _perf(8, 0.8, 15.0, 10000))
    v = rc.verdicts(few, c)
    assert v["recommendation"] == rc.KEEP and v["candidate"] is None
    assert "buy_oos_trades" in v["configs"]["B_SELL+BUY_BREAKOUT_CONFIRM"]["failed"]
    good = _res(_perf(40, 0.5, 1.6, 9000), _perf(35, 0.5, 1.5, 6000))
    assert rc.verdicts(good, c)["recommendation"] == rc.PAPER_TEST
    unstable = _res(_perf(40, 0.5, 1.6, 9000), _perf(35, 0.5, 1.5, 6000), wf=0.25)
    assert rc.verdicts(unstable, c)["recommendation"] == rc.KEEP
    day_dep = _res(_perf(40, 0.5, 1.6, 9000), _perf(35, 0.5, 1.5, 6000))
    for cfg in rc.CONFIGS[1:]:
        day_dep[(cfg.name, "COMBINED")]["OOS"]["pnl_excl_best_2_days"] = -1.0
    assert rc.verdicts(day_dep, c)["recommendation"] == rc.KEEP


def test_walk_forward_collapse_rejects_otherwise_acceptable_scope(monkeypatch):
    rows, facts = _dataset()
    folds = lab.walk_forward_folds(DAYS, 6, 2, 2)
    periods = {"TRAIN": TRAIN_DAYS, "VALIDATION": VAL_DAYS, "OOS": OOS_DAYS, "ALL": DAYS}
    test_days = {d for f in folds for d in f["test"]}
    losing = [dict(r, outcome=lab.imp.LOSS, pnl=-325.0) if r["session_date"] in test_days else r for r in rows]
    monkeypatch.setattr(lab, "candidate_status", lambda s, c: {"status": lab.PROMISING, "failed_checks": []})
    s = rc.score_scope(losing, rows, periods, folds, facts, lab.DEFAULT_PARAMS, True, rc.ComparisonCriteria(), 0)
    assert s["wf_positive_share"] == 0.0
    assert s["status"] == lab.REJECTED and "walk_forward_collapse" in s["failed_checks"]
    ok = rc.score_scope(rows, rows, periods, folds, facts, lab.DEFAULT_PARAMS, True, rc.ComparisonCriteria(), 0)
    assert ok["status"] == lab.PROMISING


def test_lookahead_truncation_passes_for_all_configs():
    spot = _spot(days=8)
    day = sorted({t.date() for t in spot["timestamp"]})[-1].isoformat()
    rows = _la_rows(day)
    facts = lab.facts_from_spot(rows, spot, S2_CFG, {})
    res = lab.truncation_lookahead(rows, spot, S2_CFG, {}, facts, rc.CONFIGS, lab.DEFAULT_PARAMS, [day])
    assert all(v["passed"] for v in res.values())


def test_current_day_data_is_excluded():
    rows, facts = _dataset()
    extra = [dict(r, session_date="2026-10-06", signal_time="2026-10-06" + r["signal_time"][10:]) for r in rows[:5]]
    a = rc.build_comparison(base_rows=rows + extra, facts=facts, days=DAYS + ["2026-10-06"], splits=SPLITS)
    assert a["unassigned_days"] == ["2026-10-06"]
    assert a["results"]["A_CURRENT_S2|COMBINED"]["ALL"]["evaluated"] == len(rows)


def test_report_generation(tmp_path):
    a, _, _ = _build()
    paths = rc.write_reports(a, tmp_path)
    for name in ("robustness_comparison.md", "robustness_comparison.csv", "walk_forward_comparison.csv",
                 "day_stability.csv", "regime_stability.csv"):
        assert paths[name].is_file() and paths[name].stat().st_size > 0
    md = paths["robustness_comparison.md"].read_text(encoding="utf-8")
    assert "RECOMMENDATION" in md and ("KEEP_CURRENT" in md or "PAPER_TEST_CANDIDATE" in md)
    with paths["day_stability.csv"].open(encoding="utf-8") as fh:
        day_rows = list(csv.DictReader(fh))
    assert len(day_rows) == len(rc.CONFIGS) * len(rc.SCOPES) * len(DAYS)
    with paths["robustness_comparison.csv"].open(encoding="utf-8") as fh:
        scopes = {(r["config"], r["scope"]) for r in csv.DictReader(fh)}
    assert len(scopes) == 12
    assert a["live_trading_enabled"] is False and a["trading_mode"] == "PAPER"
