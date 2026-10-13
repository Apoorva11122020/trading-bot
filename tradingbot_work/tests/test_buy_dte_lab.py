from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest import buy_dte_lab as dl
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.backtest import shadow_oos as so
from src.historical_data.option_coverage import resolve_signal_contract
from src.options.models import OptionResolutionError
from src.safety.live_gate import LiveTradingSafetyError
from tests.test_shadow_historical import (LISTING, PARAMS, RC, _ExpiredApiService, _patch_expired, _signal_fn,
                                          _spot_frame)
from tests.test_shadow_validation import DAY, PREMIUMS, PREV

NAMES = ["BUY_NEAREST_EXPIRY", "BUY_0_DTE", "BUY_1_DTE", "BUY_2_4_DTE", "BUY_5_7_DTE", "BUY_8_14_DTE",
         "BUY_15_21_DTE", "BUY_22_PLUS_DTE"]
D = lambda m, d: date(2026, m, d)  # noqa: E731
EXPIRIES = [D(9, 29), DAY, D(10, 6), D(10, 8), D(10, 13), D(10, 20), D(10, 27)]
EXPECTED = {"BUY_NEAREST_EXPIRY": DAY, "BUY_0_DTE": DAY, "BUY_1_DTE": D(10, 6), "BUY_2_4_DTE": D(10, 8),
            "BUY_5_7_DTE": None, "BUY_8_14_DTE": D(10, 13), "BUY_15_21_DTE": D(10, 20), "BUY_22_PLUS_DTE": D(10, 27)}
TEST_SPLITS = {"TRAIN": ("2026-09-30", "2026-09-30"), "VALIDATION": ("2026-10-01", "2026-10-01"),
               "OOS": ("2026-10-05", "2026-10-05")}
POL = {p.name: p for p in dl.POLICIES}
CE_FRAME = PREMIUMS["NSE_FO|OK_CE"]


class FakePit:
    """Point-in-time resolver double: expiries on/after ``today``; one ATM grid; unique key per contract."""

    def __init__(self, expiries):
        self.exp = sorted(expiries)
        self.today = lambda: date(2030, 1, 1)
        self.calls: list[tuple] = []

    def expiries(self, underlying):
        t = self.today()
        return [e for e in self.exp if e >= t]

    def contracts(self, underlying, expiry, option_type=None):
        return []

    def strikes(self, underlying, expiry, option_type=None):
        return [24950.0, 25000.0, 25050.0]

    def resolve(self, underlying, expiry, strike, option_type):
        kind = str(getattr(option_type, "value", option_type))
        self.calls.append((expiry, kind))
        key = f"NSE_FO|{kind}|{expiry:%d-%m-%Y}|{int(strike)}"
        return SimpleNamespace(instrument_key=key, expiry=expiry, strike=strike, option_type=option_type,
                               trading_symbol=key, tick_size=0.05, lot_size=65)


def _sig(ts: pd.Timestamp):
    return SimpleNamespace(timestamp=ts.to_pydatetime(), direction="BUY", underlying="NIFTY", spot=25020.0,
                           conditions={"above": 25010.0})


def _at_day(day: date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time.fromisoformat(hhmm)), tz=imp.IST)


def _frame(*stamps):
    return pd.DataFrame([{"timestamp": t, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0} for t in stamps])


def _listed_everywhere(contract):
    return _frame(_at_day(DAY, "09:15"), _at_day(D(10, 2), "09:15"))


# --------------------------------------------------------------------------- policies / DTE selection
def test_all_8_policies_exist():
    assert [p.name for p in dl.POLICIES] == NAMES
    assert [p.requested for p in dl.POLICIES] == ["nearest", "0 DTE", "1 trading day", "2-4 DTE", "5-7 DTE",
                                                  "8-14 DTE", "15-21 DTE", "22+ DTE"]


@pytest.mark.parametrize("name", NAMES)
def test_exact_dte_bucket_selection_without_fallback(name):
    pit = FakePit(EXPIRIES)
    resolve = dl.policy_resolver(POL[name], pit, RC, _listed_everywhere)
    sig = _sig(_at_day(DAY, "10:00"))
    if EXPECTED[name] is None:
        with pytest.raises(OptionResolutionError) as exc:
            resolve(sig)
        assert exc.value.code == dl.EXPIRY_UNAVAILABLE
        assert pit.calls == []
    else:
        c = resolve(sig)
        assert c.expiry == EXPECTED[name] and c.strike == 25000.0
        assert dl.in_policy(POL[name], DAY, c.expiry)
        assert all(kind == "CE" for _, kind in pit.calls)


def test_dte_calculation_and_one_trading_day_over_weekend():
    assert dl.in_policy(POL["BUY_1_DTE"], D(10, 2), D(10, 5))
    assert not dl.in_policy(POL["BUY_1_DTE"], D(10, 2), D(10, 6))
    assert dl.in_policy(POL["BUY_2_4_DTE"], D(10, 2), D(10, 6)) and not dl.in_policy(POL["BUY_2_4_DTE"], DAY, D(10, 10))
    assert dl.in_policy(POL["BUY_22_PLUS_DTE"], DAY, D(11, 30)) and not dl.in_policy(POL["BUY_0_DTE"], DAY, D(9, 29))
    pit = FakePit([D(10, 5), D(10, 6)])
    c = dl.policy_resolver(POL["BUY_1_DTE"], pit, RC, _listed_everywhere)(_sig(_at_day(D(10, 2), "10:00")))
    assert c.expiry == D(10, 5) and (c.expiry - D(10, 2)).days == 3


def test_point_in_time_never_uses_past_or_later_listed_contracts():
    sig_ts = _at_day(DAY, "10:00")
    later_same_day = _frame(_at_day(DAY, "10:30"), _at_day(DAY, "11:00"))
    later_day = _frame(_at_day(D(10, 6), "09:15"))
    frames = {D(10, 13): later_same_day, D(10, 14): _frame(_at_day(DAY, "09:15"))}

    def premium_for(c):
        return frames.get(c.expiry, later_day)

    pit = FakePit([D(9, 29), D(10, 13), D(10, 14)])
    c = dl.policy_resolver(POL["BUY_8_14_DTE"], pit, RC, premium_for)(_sig(sig_ts))
    assert c.expiry == D(10, 14)
    pit = FakePit([D(9, 29), D(10, 13)])
    with pytest.raises(OptionResolutionError) as exc:
        dl.policy_resolver(POL["BUY_8_14_DTE"], pit, RC, premium_for)(_sig(sig_ts))
    assert exc.value.code == dl.EXPIRY_NOT_LISTED
    assert dl.eligible_expiries([D(9, 29), D(10, 1)], DAY, POL["BUY_0_DTE"]) == []
    assert not dl.listed_at(later_same_day, sig_ts) and dl.listed_at(_frame(sig_ts), sig_ts)


def test_future_expiry_mutation_does_not_change_selection():
    sig = _sig(_at_day(DAY, "10:00"))
    unlisted = {D(10, 7), D(10, 10), D(11, 24)}

    def premium_for(c):
        return _frame(_at_day(D(10, 20), "09:15")) if c.expiry in unlisted else _listed_everywhere(c)

    for name in NAMES[1:]:
        if EXPECTED[name] is None:
            continue
        base = dl.policy_resolver(POL[name], FakePit(EXPIRIES), RC, premium_for)(sig)
        mutated = dl.policy_resolver(POL[name], FakePit(EXPIRIES + sorted(unlisted)), RC, premium_for)(sig)
        assert (base.instrument_key, base.expiry) == (mutated.instrument_key, mutated.expiry)
    with pytest.raises(OptionResolutionError) as exc:
        dl.policy_resolver(POL["BUY_5_7_DTE"], FakePit(EXPIRIES + [D(10, 10)]), RC, premium_for)(sig)
    assert exc.value.code == dl.EXPIRY_NOT_LISTED


# --------------------------------------------------------------------------- pipeline-level study
def _inputs(expiries=EXPIRIES, premium_for=None):
    pit = FakePit(expiries)
    check = sh.ExpiryCheck(LISTING, frozenset(e for e in expiries if e <= DAY), "test official list")
    return SimpleNamespace(selected=[PREV, DAY], signal_fn=_signal_fn, spot=_spot_frame(), params=PARAMS, rc=RC,
                           s2={"range_start": "09:15", "range_end": "09:30"}, check=check, pit=pit,
                           resolve=lambda sig: resolve_signal_contract(sig, pit, RC),
                           premium_for=premium_for or (lambda c: CE_FRAME), regimes={})


def _original_buy(x):
    r = sh.replay(spot=x.spot, signal_fn=x.signal_fn, resolve=sh.strict_resolver(x.resolve, x.check),
                  premium_for=x.premium_for, params=x.params, square_off="15:15", range_start="09:15",
                  range_end="09:30", days=x.selected, mutation_cut=None)
    return [row for row in r["rows"][dl.lab.ORIGINAL] if row["direction"] == "BUY"]


@pytest.fixture(scope="module")
def study():
    x = _inputs()
    original = _original_buy(x)
    x.pit.calls.clear()
    return dl.build_lab(x, splits=imp.split_dates(TEST_SPLITS), original_buy_rows=original), x, original


def test_baseline_unchanged_and_buy_only(study):
    a, x, original = study
    assert a["baseline_matches_original_s2_buy"] is True
    for name, rows in a["_rows"].items():
        assert [r["direction"] for r in rows] == ["BUY"]
    assert all(kind == "CE" for _, kind in x.pit.calls)


def test_each_policy_resolves_requested_expiry_or_is_unavailable(study):
    a, _, _ = study
    for name, rows in a["_rows"].items():
        r = rows[0]
        if EXPECTED[name] is None:
            assert (r["state"], r["reason"], r["expiry_unavailable"]) == (sh.DATA_UNAVAILABLE, "expiry_unavailable", True)
            assert r["outcome"] not in (imp.WIN, imp.LOSS) and r["pnl"] is None
        else:
            assert date.fromisoformat(r["expiry"][:10]) == EXPECTED[name]
            assert r["dte"] == (EXPECTED[name] - DAY).days and r["in_requested_range"] is True
    res = a["dte_resolution"]
    assert res["BUY_5_7_DTE"] == {"requested": 1, "resolved": 0, "expiry_unavailable": 1, "resolved_dte_histogram": {},
                                  "outside_requested_range": 0, "expiry_before_signal_date": 0}
    assert res["BUY_22_PLUS_DTE"]["resolved_dte_histogram"] == {"22": 1}
    m = a["metrics"]["OOS"]["BUY_5_7_DTE"]
    assert (m["signals"], m["evaluated"], m["wins"], m["losses"], m["expiry_unavailable"]) == (1, 0, 0, 0, 1)


def test_entry_sl_target_and_square_off_unchanged(study):
    a, _, original = study
    o = original[0]
    for name, rows in a["_rows"].items():
        r = rows[0]
        if r["state"] != sh.EVALUATED:
            continue
        for f in ("entry_time", "entry", "premium_atr", "stop_loss", "target", "exit_time", "exit_reason", "outcome",
                  "pnl"):
            assert r[f] == o[f], (name, f)
        assert pd.Timestamp(r["entry_time"]) == pd.Timestamp(r["signal_time"]) + pd.Timedelta(minutes=5)
        assert r["stop_loss"] == pytest.approx(r["entry"] - 1.5 * r["premium_atr"], abs=0.05)
        assert r["target"] == pytest.approx(r["entry"] + 2 * (r["entry"] - r["stop_loss"]), abs=0.1)
        assert pd.Timestamp(r["exit_time"]).time() <= time(15, 15)
        assert (r["outcome"], r["pnl"]) == (imp.WIN, 390.0)


def test_lookahead_mutation_passes_for_every_policy(study):
    a, _, _ = study
    assert set(a["lookahead"]) == set(NAMES)
    assert all(v["passed"] and v["cuts"] == 2 for v in a["lookahead"].values())


def test_mutation_detects_expiry_choice_that_reads_future_candles():
    x = _inputs()
    policy = POL["BUY_2_4_DTE"]
    cut = _at_day(DAY, "12:00")
    honest = imp._pipeline_factory(dl.buy_only(x.signal_fn), dl.policy_resolver(policy, x.pit, RC, x.premium_for),
                                   PARAMS, "09:15", "09:30")
    assert dl.mutation_lookahead(honest, x.spot, x.premium_for, policy, [cut])["passed"]

    seen: dict = {}
    reference = float(x.spot["close"].iloc[-1])

    def recording_signal_fn(frame):
        seen["last_close"] = float(frame["close"].iloc[-1])
        return _signal_fn(frame)

    def leaky_resolve(sig):
        expiry = D(10, 8) if abs(seen["last_close"] - reference) < 1e-9 else D(10, 6)
        return x.pit.resolve("NIFTY", expiry, 25000.0, "CE")

    leaky = imp._pipeline_factory(dl.buy_only(recording_signal_fn), leaky_resolve, PARAMS, "09:15", "09:30")
    r = dl.mutation_lookahead(leaky, x.spot, x.premium_for, policy, [cut])
    assert r["passed"] is False and r["failures"]


# --------------------------------------------------------------------------- metrics / status
def _r(day, pnl, i=0, outcome=None, state=sh.EVALUATED, expiry_unavailable=False):
    ev = state == sh.EVALUATED
    return {"session_date": day, "signal_time": f"{day}T10:{i:02d}:00+05:30", "direction": "BUY", "state": state,
            "outcome": outcome or ((imp.WIN if pnl > 0 else imp.LOSS) if ev else None), "pnl": pnl if ev else None,
            "reason": "expiry_unavailable" if expiry_unavailable else (None if ev else "option_candle_missing"),
            "expiry_unavailable": expiry_unavailable, "dte_bucket": "1 DTE", "regimes": ["range_bound"],
            "time_bucket": None}


def test_win_loss_ambiguous_scratch_accuracy_pf_pnl_drawdown():
    days = ["2026-09-25", "2026-09-28"]
    rows = [_r(days[0], 390.0, 0), _r(days[0], -195.0, 1), _r(days[0], -195.0, 2, outcome=imp.AMBIGUOUS),
            _r(days[1], 0.0, 0, outcome=imp.SCRATCH), _r(days[1], 600.0, 1),
            _r(days[1], 0, 2, state=sh.DATA_UNAVAILABLE, expiry_unavailable=True),
            _r(days[1], 0, 3, state=sh.DATA_UNAVAILABLE)]
    m = dl.policy_metrics(rows, days)
    assert (m["signals"], m["evaluated"], m["wins"], m["losses"], m["ambiguous"], m["scratch"]) == (7, 5, 2, 1, 1, 1)
    assert (m["expiry_unavailable"], m["data_unavailable_other"]) == (1, 1)
    assert m["win_rate"] == pytest.approx(0.4)
    assert m["profit_factor"] == pytest.approx(990 / 390) and m["net_pnl"] == 600
    assert m["max_drawdown"] == 390 and m["avg_pnl_per_trade"] == 120
    assert m["data_coverage_ex_expiry"] == pytest.approx(5 / 6)
    assert (m["profitable_days"], m["losing_days"]) == (1, 0)
    assert m["best_day"] == {"date": days[1], "net_pnl": 600} and m["worst_day"]["date"] == days[0]


def _m(evaluated=40, wr=0.5, pf=1.5, pnl=1000.0, dd=300.0, cov=1.0):
    return {"evaluated": evaluated, "win_rate": wr, "profit_factor": pf, "net_pnl": pnl, "max_drawdown": dd,
            "data_coverage_ex_expiry": cov}


def test_status_rules_insufficient_sample_and_never_robust():
    base = _m(wr=0.26)
    assert dl.candidate_status(_m(wr=0.45), _m(), _m(), base, True)["status"] == dl.PROMISING
    assert dl.candidate_status(_m(wr=0.37), _m(), _m(), base, True)["status"] == dl.PROMISING
    weak_train = dl.candidate_status(_m(wr=0.30), _m(), _m(), base, True)
    assert weak_train["status"] == dl.NOT_PROMISING
    assert weak_train["failed_checks"] == ["train_wr_or_meaningful_improvement"]
    for kw, check in ((dict(val=_m(wr=0.44)), "validation_wr"), (dict(oos=_m(pf=1.1)), "oos_pf"),
                      (dict(oos=_m(pnl=0.0)), "oos_pnl"), (dict(oos=_m(wr=0.44)), "oos_wr"),
                      (dict(la=False), "lookahead"), (dict(oos=_m(cov=0.5)), "data_coverage")):
        st = dl.candidate_status(_m(), kw.get("val", _m()), kw.get("oos", _m()), base, kw.get("la", True))
        assert st["status"] == dl.NOT_PROMISING and check in st["failed_checks"]
    tiny = dl.candidate_status(_m(), _m(), _m(evaluated=29, pf=9.0), base, True)
    assert tiny["status"] == dl.INSUFFICIENT_DATA
    assert "ROBUST" not in {dl.PROMISING, dl.NOT_PROMISING, dl.INSUFFICIENT_DATA}


def test_insufficient_policy_is_never_best():
    days = {"TRAIN": [], "VALIDATION": [], "OOS": ["2026-09-25"]}
    rows = {n: [_r("2026-09-25", 300.0, i) for i in range(5)] for n in NAMES}
    rows["BUY_2_4_DTE"] = [_r("2026-09-25", 300.0 if i % 2 else -100.0, i) for i in range(40)]
    la = {n: {"cuts": 1, "failures": [], "passed": True} for n in NAMES}
    a = dl.assemble(rows, la, days, [], imp.split_dates(imp.DEFAULT_SPLITS))
    assert a["best_candidate"] == "BUY_2_4_DTE"
    assert a["status"]["BUY_0_DTE"]["status"] == dl.INSUFFICIENT_DATA
    c = a["comparison_vs_baseline"]["BUY_2_4_DTE"]["OOS"]
    assert c["wr_change_pp"] == pytest.approx(-50.0) and c["trade_count_change"] == 35


# --------------------------------------------------------------------------- splits
def test_chronological_split_and_no_date_overlap():
    s = imp.split_dates(imp.DEFAULT_SPLITS)
    assert s["TRAIN"] == (D(9, 1), D(9, 17)) and s["VALIDATION"] == (D(9, 18), D(9, 24))
    assert s["OOS"] == (D(9, 25), D(10, 1))
    by, unassigned = so.assign_days(["2026-10-05", "2026-09-24", "2026-09-25", "2026-09-17"], s)
    assert by == {"TRAIN": ["2026-09-17"], "VALIDATION": ["2026-09-24"], "OOS": ["2026-09-25"]}
    assert unassigned == ["2026-10-05"]


# --------------------------------------------------------------------------- runner / safety
def _dte_runner(monkeypatch, service):
    _patch_expired(monkeypatch, service)
    import scripts.run_buy_dte_lab as runner
    return runner


def test_runner_with_real_expired_contract_path_and_reports(monkeypatch, tmp_path):
    service = _ExpiredApiService(_spot_frame())
    runner = _dte_runner(monkeypatch, service)
    other = tmp_path / "buy_strategy_lab"
    other.mkdir()
    (other / "summary.json").write_text("keep")
    out = tmp_path / "buy_dte_lab"
    a = runner.run(out_dir=out, splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert (other / "summary.json").read_text() == "keep"
    assert a["baseline_matches_original_s2_buy"] is True
    oos = a["metrics"]["OOS"]
    assert (oos["BUY_NEAREST_EXPIRY"]["evaluated"], oos["BUY_NEAREST_EXPIRY"]["net_pnl"]) == (1, 390.0)
    assert (oos["BUY_0_DTE"]["evaluated"], oos["BUY_0_DTE"]["net_pnl"]) == (1, 390.0)
    for n in NAMES[2:]:
        assert (oos[n]["evaluated"], oos[n]["expiry_unavailable"], oos[n]["losses"]) == (0, 1, 0)
    assert {c[3] for c in service.downloader.calls} == {"EXPIRED"}
    assert all(r["passed"] for r in a["lookahead"].values())
    assert a["best_candidate"] is None
    assert {p.name for p in out.iterdir()} == {"summary.md", "summary.json", "summary.csv", "daily.csv",
                                              "dte_resolution.csv", "regime.csv"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["trading_mode"] == "PAPER" and summary["live_trading_enabled"] is False and "_rows" not in summary


@pytest.mark.parametrize("env", [{"LIVE_TRADING_ENABLED": "true"}, {"TRADING_MODE": "LIVE"}])
def test_runner_refuses_non_paper_mode(monkeypatch, tmp_path, env):
    runner = _dte_runner(monkeypatch, _ExpiredApiService(_spot_frame()))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(LiveTradingSafetyError):
        runner.run(out_dir=tmp_path / "out", splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert not (tmp_path / "out").exists()


def test_runner_blocks_broker_order_calls(monkeypatch, tmp_path):
    from src.execution.order_manager import OrderManager
    import scripts.run_shadow_historical_validation as hv

    runner = _dte_runner(monkeypatch, _ExpiredApiService(_spot_frame()))
    attempts = []

    def signal_fn(frame):
        try:
            OrderManager.submit_signal(None)
        except LiveTradingSafetyError as exc:
            attempts.append(str(exc))
        return _signal_fn(frame)

    monkeypatch.setattr(hv, "s2_components", lambda svc, name: (
        RC, PARAMS, None, {"range_start": "09:15", "range_end": "09:30"}, signal_fn, None))
    runner.run(out_dir=tmp_path / "out", splits=TEST_SPLITS, today=date(2026, 10, 6))
    assert attempts and all("SHADOW_VALIDATION_ORDER_BLOCKED" in a for a in attempts)


def test_dte_modules_have_no_order_imports():
    for path in (Path(dl.__file__), Path("scripts/run_buy_dte_lab.py")):
        src = path.read_text(encoding="utf-8")
        for forbidden in ("from src.broker", "from src.execution", "from src.engine", "place_order(",
                          "submit_signal("):
            assert forbidden not in src


def test_timedelta_helper_counts_weekdays():
    assert dl._weekdays_between(D(10, 2), D(10, 5)) == 0
    assert dl._weekdays_between(DAY, DAY + timedelta(days=7)) == 4
