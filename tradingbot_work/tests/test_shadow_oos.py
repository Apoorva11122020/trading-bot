from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.safety.live_gate import LiveTradingSafetyError
from tests.test_shadow_historical import EXPIRED_CE, EXPIRED_PE, _ExpiredApiService, _patch_expired, _spot_frame

NAMES = ["BASELINE_S2_BUY_SELL", "SELL_ONLY", "SELL_CONFIRMED"]
TEST_SPLITS = {"TRAIN": ("2026-09-30", "2026-09-30"), "VALIDATION": ("2026-10-01", "2026-10-01"),
               "OOS": ("2026-10-05", "2026-10-05")}


def _row(day: str, pnl: float, i: int = 0, direction: str = "SELL", regimes=(), dte="1 DTE", state=sh.EVALUATED):
    return {"session_date": day, "signal_time": f"{day}T10:{i:02d}:00+05:30", "direction": direction,
            "state": state, "outcome": (imp.WIN if pnl > 0 else imp.LOSS) if state == sh.EVALUATED else None,
            "pnl": pnl if state == sh.EVALUATED else None, "reason": None if state == sh.EVALUATED else "x",
            "dte_bucket": dte, "regimes": list(regimes), "time_bucket": None}


def _metrics(rows, days):
    return so.split_metrics(rows, days)


def _result(rows_by_name: dict[str, list[dict]], days: list[str], failures=None):
    return {"days": days, "rows": {n: rows_by_name.get(n, []) for n in NAMES},
            "lookahead": {n: {"cuts": len(days), "failures": list((failures or {}).get(n, [])), "passed": True}
                          for n in NAMES}}


SPLITS = imp.split_dates(imp.DEFAULT_SPLITS)
TRAIN_DAYS = ["2026-09-01", "2026-09-02"]
VAL_DAYS = ["2026-09-18", "2026-09-21"]
OOS_DAYS = ["2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01"]


def _good_oos(n_per_day=8):
    rows = []
    for d in OOS_DAYS:
        for i in range(n_per_day):
            rows.append(_row(d, 300.0 if i % 2 == 0 else -100.0, i))
    return rows


# --------------------------------------------------------------------------- splits
def test_existing_project_split_matches_and_is_chronological():
    s = imp.split_dates(imp.DEFAULT_SPLITS)
    assert s == {"TRAIN": (date(2026, 9, 1), date(2026, 9, 17)), "VALIDATION": (date(2026, 9, 18), date(2026, 9, 24)),
                 "OOS": (date(2026, 9, 25), date(2026, 10, 1))}
    assert s["TRAIN"][1] < s["VALIDATION"][0] and s["VALIDATION"][1] < s["OOS"][0]
    with pytest.raises(ValueError):
        imp.split_dates({"TRAIN": ("2026-09-01", "2026-09-20"), "VALIDATION": ("2026-09-18", "2026-09-24"),
                         "OOS": ("2026-09-25", "2026-10-01")})
    with pytest.raises(ValueError):
        imp.split_dates({"TRAIN": ("2026-09-25", "2026-10-01"), "VALIDATION": ("2026-09-18", "2026-09-24"),
                         "OOS": ("2026-09-01", "2026-09-17")})


def test_no_date_overlap_and_current_day_never_assigned():
    days = TRAIN_DAYS + VAL_DAYS + OOS_DAYS + ["2026-10-05"]
    by, unassigned = so.assign_days(list(reversed(days)), SPLITS)
    flat = [d for v in by.values() for d in v]
    assert len(flat) == len(set(flat)) == len(days) - 1
    assert by == {"TRAIN": TRAIN_DAYS, "VALIDATION": VAL_DAYS, "OOS": OOS_DAYS}
    assert unassigned == ["2026-10-05"]


def test_lookahead_is_partitioned_per_split():
    la = {"SELL_ONLY": {"cuts": 9, "failures": ["2026-09-28T12:00:00+05:30"], "passed": False}}
    assert so.split_lookahead(la, OOS_DAYS)["SELL_ONLY"] == {"cuts": 5, "failures": ["2026-09-28T12:00:00+05:30"],
                                                            "passed": False}
    assert so.split_lookahead(la, TRAIN_DAYS)["SELL_ONLY"]["passed"] is True
    assert so.split_lookahead({"X": {"cuts": 0, "failures": []}}, OOS_DAYS)["X"]["passed"] is False


# --------------------------------------------------------------------------- metrics
def test_metrics_drawdown_and_daily_aggregation():
    rows = [_row("2026-09-25", 100, 0), _row("2026-09-25", -50, 1), _row("2026-09-28", -70, 0),
            _row("2026-09-28", 200, 1), _row("2026-09-29", 0, 0, state=sh.DATA_UNAVAILABLE),
            _row("2026-09-29", 0, 1, state=sh.SKIPPED)]
    m = _metrics(rows, OOS_DAYS)
    assert (m["signals"], m["evaluated"], m["wins"], m["losses"], m["skipped"], m["data_unavailable"]) == (6, 4, 2, 2,
                                                                                                          1, 1)
    assert m["net_pnl"] == 180 and m["max_drawdown"] == 120
    assert m["profit_factor"] == pytest.approx(300 / 120) and m["win_rate"] == 0.5
    assert m["avg_pnl_per_trade"] == 45 and m["data_coverage"] == pytest.approx(0.8)
    assert m["daily"]["2026-09-25"]["net_pnl"] == 50 and m["daily"]["2026-09-28"]["net_pnl"] == 130
    assert (m["profitable_days"], m["losing_days"], m["traded_days"]) == (2, 0, 2)
    assert m["best_day"] == {"date": "2026-09-28", "net_pnl": 130}
    assert m["excluding_best_day"]["net_pnl"] == 50 and m["excluding_best_day"]["excluded_date"] == "2026-09-28"
    assert m["best_day_share"] == pytest.approx(130 / 180)
    assert m["return_to_drawdown"] == pytest.approx(1.5)


def test_dte_and_direction_buckets():
    rows = [_row("2026-09-25", 100, 0, "BUY", dte="0 DTE"), _row("2026-09-25", -50, 1, dte="2-4 DTE"),
            _row("2026-09-25", 40, 2, dte="22+ DTE")]
    m = _metrics(rows, OOS_DAYS)
    assert list(m["dte"]) == ["0 DTE", "1 DTE", "2-4 DTE", "5-7 DTE", "8-14 DTE", "15-21 DTE", "22+ DTE"]
    assert (m["dte"]["0 DTE"]["evaluated"], m["dte"]["2-4 DTE"]["net_pnl"], m["dte"]["22+ DTE"]["wins"]) == (1, -50, 1)
    assert m["direction"]["BUY"]["net_pnl"] == 100 and m["direction"]["SELL"]["evaluated"] == 2


def test_regime_report_only_shows_present_regimes_and_bearish_dependence():
    rows = [_row("2026-09-25", 500, 0, regimes=[so.BEARISH]), _row("2026-09-28", -100, 0, regimes=["range_bound"])]
    m = _metrics(rows, OOS_DAYS)
    assert set(m["regime"]) == {so.BEARISH, "range_bound"}
    dep = so.regime_dependence(rows)
    assert dep["strong_bearish"]["net_pnl"] == 500 and dep["non_bearish"]["net_pnl"] == -100
    assert dep["bearish_share_of_pnl"] == pytest.approx(500 / 400)
    assert dep["profitable_without_bearish_days"] is False


# --------------------------------------------------------------------------- ranking / status
def test_ranking_uses_oos_metrics_in_declared_order():
    base = {"evaluated": 10, "profit_factor": 1.5, "net_pnl": 100.0, "max_drawdown": 50.0, "win_rate": 0.5}
    oos = {"A": dict(base, profit_factor=2.0), "B": dict(base, net_pnl=200.0), "C": dict(base),
           "D": dict(base, max_drawdown=10.0), "E": dict(base, evaluated=0, profit_factor=None, win_rate=None)}
    assert so.rank_by_oos(oos) == ["A", "B", "D", "C", "E"]


def _status(oos_rows, val_rows=None, lookahead=True, observed=False, **criteria):
    oos = _metrics(oos_rows, OOS_DAYS)
    val = _metrics(val_rows if val_rows is not None else [_row("2026-09-18", 100)], VAL_DAYS)
    return so.robustness(oos, val, lookahead, so.RobustnessCriteria(**criteria), observed)


def test_robustness_status_rules():
    good = _good_oos()
    assert _status(good)["status"] == so.ROBUST
    capped = _status(good, observed=True)
    assert (capped["status"], capped["failed_checks"]) == (so.PROMISING, ["oos_window_unseen"])
    assert _status(good[:10])["status"] == so.INSUFFICIENT_DATA
    assert _status(good, lookahead=False)["status"] == so.NOT_ROBUST
    weak = [_row(d, 110.0 if i % 2 == 0 else -100.0, i) for d in OOS_DAYS for i in range(8)]
    assert "oos_profit_factor" in _status(weak)["failed_checks"]
    losing = [_row(d, -100.0, i) for d in OOS_DAYS for i in range(8)]
    assert _status(losing)["status"] == so.NOT_ROBUST
    assert _status(good, val_rows=[_row("2026-09-18", -100)])["status"] == so.PROMISING


def test_profit_must_survive_removing_best_day():
    rows = [_row(d, -60.0, i) for d in OOS_DAYS[1:] for i in range(8)] + \
           [_row(OOS_DAYS[0], 2000.0, i) for i in range(8)]
    st = _status(rows)
    assert st["status"] == so.NOT_ROBUST and "profitable_excluding_best_day" in st["failed_checks"]


def test_material_data_gaps_block_robust_and_insufficient_coverage():
    gaps = _good_oos() + [_row("2026-09-25", 0, 30 + i, state=sh.DATA_UNAVAILABLE) for i in range(4)]
    assert "no_material_data_gaps" in _status(gaps)["failed_checks"]
    heavy = _good_oos() + [_row("2026-09-25", 0, i, state=sh.DATA_UNAVAILABLE) for i in range(10, 59)]
    assert _status(heavy)["status"] == so.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- study: selection / isolation
def _study(train_rows, oos_rows=None, failures=None, observed=True):
    days = TRAIN_DAYS + VAL_DAYS + OOS_DAYS
    val = [_row("2026-09-18", 100.0, 0)]
    oos = _good_oos() if oos_rows is None else oos_rows
    rows = {"BASELINE_S2_BUY_SELL": train_rows + val + oos + [_row("2026-09-25", -40, 50, "BUY")],
            "SELL_ONLY": train_rows + val + oos, "SELL_CONFIRMED": val + oos[::2]}
    return so.build_study(_result(rows, days, failures), {"date_range": "x"}, splits=SPLITS,
                          oos_previously_observed=observed)


def test_oos_never_influenced_by_train_and_strategies_fixed():
    a = _study([_row("2026-09-01", 1000.0, i) for i in range(20)])
    b = _study([_row("2026-09-01", -1000.0, i) for i in range(20)])
    assert a["strategies"] == b["strategies"] == NAMES
    assert a["ranking_by_oos"] == b["ranking_by_oos"]
    assert {n: s["status"] for n, s in a["status"].items()} == {n: s["status"] for n, s in b["status"].items()}
    assert a["metrics"]["OOS"] == b["metrics"]["OOS"]
    assert a["metrics"]["TRAIN"]["SELL_ONLY"]["net_pnl"] == 20000 and b["metrics"]["TRAIN"]["SELL_ONLY"]["net_pnl"] == -20000


def test_oos_rows_do_not_leak_into_train_or_validation():
    a = _study([], oos_rows=_good_oos())
    b = _study([], oos_rows=[_row(d, -100.0, i) for d in OOS_DAYS for i in range(8)])
    for split in ("TRAIN", "VALIDATION"):
        assert a["metrics"][split] == b["metrics"][split]


def test_strategy_isolation_and_final_table():
    s = _study([])
    oos = s["metrics"]["OOS"]
    assert oos["BASELINE_S2_BUY_SELL"]["direction"]["BUY"]["evaluated"] == 1
    assert oos["SELL_ONLY"]["direction"]["BUY"]["evaluated"] == 0
    assert oos["SELL_CONFIRMED"]["evaluated"] == 20 and oos["SELL_ONLY"]["evaluated"] == 40
    assert [r["strategy"] for r in s["final_table"]] == NAMES
    assert s["status"]["SELL_ONLY"]["status"] == so.PROMISING
    assert s["status"]["SELL_CONFIRMED"]["status"] == so.INSUFFICIENT_DATA


def test_lookahead_failure_in_any_split_marks_not_robust():
    s = _study([], failures={"SELL_ONLY": ["2026-09-02T12:00:00+05:30"]}, observed=False)
    assert s["lookahead"]["TRAIN"]["SELL_ONLY"]["passed"] is False
    assert s["lookahead"]["OOS"]["SELL_ONLY"]["passed"] is True
    assert s["status"]["SELL_ONLY"]["status"] == so.NOT_ROBUST
    assert s["status"]["SELL_CONFIRMED"]["status"] != so.NOT_ROBUST


def test_reports_write_only_oos_files_and_keep_current_day_separate(tmp_path):
    s = _study([])
    s = dict(s, current_day_forward=so.current_day_forward(
        {"validated_days": ["2026-10-05"], "candidates": {"SELL_ONLY": {"evaluated": 9, "net_pnl": -14569.75}}},
        SPLITS))
    sentinel = tmp_path / "2026-09-25.json"
    sentinel.write_text("{}")
    paths = so.write_reports(s, tmp_path)
    assert sentinel.is_file()
    assert {p.name for p in paths.values()} == {"summary.md", "summary.json", "summary.csv", "train.json",
                                                "validation.json", "oos.json"}
    oos = json.loads((tmp_path / "oos.json").read_text(encoding="utf-8"))
    assert oos["label"] == "HISTORICAL OOS" and set(oos["days"]) == set(OOS_DAYS)
    assert all(r["session_date"] in OOS_DAYS for rows in oos["signals"].values() for r in rows)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["current_day_forward"]["label"] == so.CURRENT_LABEL
    assert summary["current_day_forward"]["overlapping_historical_days"] == []
    assert "2026-10-05" not in [d for sp in summary["splits"].values() for d in sp["days"]]
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    for label in ("HISTORICAL TRAIN", "HISTORICAL VALIDATION", "HISTORICAL OOS", "CURRENT-DAY FORWARD SHADOW"):
        assert label in md
    assert "| Strategy | Train WR | Train PF | Validation WR | Validation PF | OOS WR | OOS PF |" in md


# --------------------------------------------------------------------------- runner (real expired-contract path)
def _oos_runner(monkeypatch, service):
    _patch_expired(monkeypatch, service)
    import scripts.run_shadow_oos_validation as runner
    return runner


def test_runner_uses_real_expired_contracts_and_splits_chronologically(monkeypatch, tmp_path):
    service = _ExpiredApiService(_spot_frame())
    runner = _oos_runner(monkeypatch, service)
    current = tmp_path / "current"
    current.mkdir()
    (current / "cumulative.json").write_text(json.dumps({"validated_days": ["2026-10-06"], "candidates": {}}))
    hist = tmp_path / "hist"
    hist.mkdir()
    (hist / "summary.json").write_text("keep")
    s = runner.run(out_dir=tmp_path / "oos", current_dir=current, splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert (hist / "summary.json").read_text() == "keep"
    assert s["splits"]["TRAIN"]["days"] == [] and s["splits"]["VALIDATION"]["days"] == ["2026-10-01"]
    assert s["splits"]["OOS"]["days"] == ["2026-10-05"]
    oos = s["metrics"]["OOS"]["BASELINE_S2_BUY_SELL"]
    assert (oos["evaluated"], oos["data_unavailable"]) == (3, 0)
    assert oos["net_pnl"] == pytest.approx(390.0 - 195.0)
    assert oos["dte"]["0 DTE"]["evaluated"] == 3
    assert s["metrics"]["OOS"]["SELL_ONLY"]["direction"]["BUY"]["evaluated"] == 0
    assert {c[3] for c in service.downloader.calls} == {"EXPIRED"}
    keys = {r["instrument_key"] for rows in s["_per_split"]["OOS"]["signals"].values() for r in rows
            if r.get("instrument_key")}
    assert keys <= {EXPIRED_CE, EXPIRED_PE} and keys
    assert all(r["passed"] and r["cuts"] == 1 for r in s["lookahead"]["OOS"].values())
    assert s["status"]["BASELINE_S2_BUY_SELL"]["status"] == so.INSUFFICIENT_DATA
    assert s["current_day_forward"]["validated_days"] == ["2026-10-06"]
    assert (tmp_path / "oos" / "oos.json").is_file() and (tmp_path / "oos" / "train.json").is_file()


@pytest.mark.parametrize("env", [{"LIVE_TRADING_ENABLED": "true"}, {"TRADING_MODE": "LIVE"}])
def test_runner_refuses_non_paper_mode(monkeypatch, tmp_path, env):
    runner = _oos_runner(monkeypatch, _ExpiredApiService(_spot_frame()))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(LiveTradingSafetyError):
        runner.run(out_dir=tmp_path / "oos", splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert not (tmp_path / "oos").exists()


def test_runner_blocks_order_entry_points(monkeypatch, tmp_path):
    from src.execution.order_manager import OrderManager
    import scripts.run_shadow_historical_validation as hv
    from tests.test_shadow_historical import PARAMS, RC, _signal_fn

    runner = _oos_runner(monkeypatch, _ExpiredApiService(_spot_frame()))
    attempts = []

    def signal_fn(frame):
        try:
            OrderManager.submit_signal(None)
        except LiveTradingSafetyError as exc:
            attempts.append(str(exc))
        return _signal_fn(frame)

    monkeypatch.setattr(hv, "s2_components", lambda svc, name: (
        RC, PARAMS, None, {"range_start": "09:15", "range_end": "09:30"}, signal_fn, None))
    runner.run(out_dir=tmp_path / "oos", current_dir=tmp_path / "none", splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert attempts and all("SHADOW_VALIDATION_ORDER_BLOCKED" in a for a in attempts)


def test_oos_modules_have_no_order_imports():
    for path in (Path(so.__file__), Path("scripts/run_shadow_oos_validation.py")):
        src = path.read_text(encoding="utf-8")
        for forbidden in ("from src.broker", "from src.execution", "from src.engine", "place_order(",
                          "submit_signal(", "LIVE_TRADING_ENABLED=true"):
            assert forbidden not in src
