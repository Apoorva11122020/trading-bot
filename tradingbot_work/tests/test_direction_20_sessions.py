from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.backtest import direction_sessions as ds
from src.backtest import s2_1dte_improvement as imp
from src.backtest import shadow_historical as sh
from src.safety.live_gate import LiveTradingSafetyError
from tests.test_shadow_historical import _ExpiredApiService, _FakeService, _patch_expired, _patch_runner, _spot_frame
from tests.test_shadow_validation import DAY, PREV, _at, _spot

DAYS = ["2026-09-28", "2026-09-29", "2026-09-30"]


def _row(day, pnl, i=0, direction="BUY", state=sh.EVALUATED, dte="1 DTE", regimes=(), outcome=None):
    ev = state == sh.EVALUATED
    return {"strategy": ds.STRATEGY, "session_date": day, "signal_time": f"{day}T10:{i:02d}:00+05:30",
            "direction": direction, "state": state,
            "outcome": outcome or ((imp.WIN if pnl > 0 else imp.LOSS) if ev else None),
            "pnl": pnl if ev else None, "reason": None if ev else "option_candle_missing", "dte_bucket": dte,
            "regimes": list(regimes), "time_bucket": None}


def _result(rows, days=DAYS, direction_lookahead=None):
    return {"days": days, "rows": {ds.STRATEGY: rows, "SELL_ONLY": [r for r in rows if r["direction"] == "SELL"]},
            "lookahead": {ds.STRATEGY: {"cuts": len(days), "failures": [], "passed": True}},
            "direction_lookahead": direction_lookahead or {d: {"cuts": len(days), "failures": [], "passed": True}
                                                           for d in ("BUY", "SELL")}}


ROWS = [
    _row("2026-09-28", 300.0, 0, "BUY"), _row("2026-09-28", -100.0, 1, "BUY"),
    _row("2026-09-29", -150.0, 0, "BUY"), _row("2026-09-29", 0.0, 1, "BUY", state=sh.DATA_UNAVAILABLE),
    _row("2026-09-28", 200.0, 2, "SELL", dte="0 DTE", regimes=["strong_bearish_trend"]),
    _row("2026-09-29", 400.0, 2, "SELL", dte="0 DTE", regimes=["strong_bearish_trend"]),
    _row("2026-09-29", -100.0, 3, "SELL", dte="2-4 DTE", regimes=["strong_bearish_trend"]),
    _row("2026-09-30", 0.0, 0, "SELL", outcome=imp.SCRATCH),
]


def _analysis(rows=ROWS, **kw):
    return ds.build_analysis(_result(rows, **kw), {"date_range": "x"}, 20)


# --------------------------------------------------------------------------- session selection
def test_exactly_latest_n_sessions_in_chronological_order():
    days = [date(2026, 9, d) for d in range(1, 31) if date(2026, 9, d).weekday() < 5]
    picked = sh.select_days(list(reversed(days)), 20)
    assert len(picked) == 20 and picked == sorted(picked) and picked == days[-20:]


def test_runner_selects_latest_stored_sessions(monkeypatch, tmp_path):
    stored = [date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30), PREV, DAY]
    spot = pd.DataFrame([r for d in stored for r in _spot(d, d == DAY)])
    runner = _runner(monkeypatch, _patch_runner(monkeypatch, spot, _FakeService(spot)))
    a = runner.run(sessions=3, out_dir=tmp_path / "out", today=date(2026, 10, 6))
    assert a["selected_dates"] == ["2026-09-30", "2026-10-01", "2026-10-05"]
    assert a["sessions_selected"] == 3 and not a["fewer_sessions_than_requested"]
    assert [r["date"] for r in a["daily"]] == a["selected_dates"]


def test_runner_reports_fewer_sessions_than_requested(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, _patch_runner(monkeypatch, _spot_frame()))
    a = runner.run(out_dir=tmp_path / "out", today=date(2026, 10, 6))
    assert a["requested_sessions"] == 20 and a["sessions_selected"] == 2 and a["fewer_sessions_than_requested"]


# --------------------------------------------------------------------------- metrics
def test_buy_sell_separation_and_metrics():
    a = _analysis()
    buy, sell, total = a["directions"]["BUY"], a["directions"]["SELL"], a["total"]
    assert (buy["signals"], buy["evaluated"], buy["wins"], buy["losses"], buy["data_unavailable"]) == (4, 3, 1, 2, 1)
    assert buy["win_rate"] == pytest.approx(1 / 3)
    assert buy["profit_factor"] == pytest.approx(300 / 250)
    assert buy["net_pnl"] == 50 and buy["avg_pnl_per_trade"] == pytest.approx(50 / 3, abs=0.01)
    assert buy["max_drawdown"] == 250
    assert (buy["profitable_days"], buy["losing_days"]) == (1, 1)
    assert (sell["signals"], sell["evaluated"], sell["wins"], sell["losses"]) == (4, 4, 2, 2)
    assert sell["win_rate"] == 0.5 and sell["profit_factor"] == pytest.approx(6.0) and sell["net_pnl"] == 500
    assert sell["outcomes"] == {imp.WIN: 2, imp.LOSS: 1, imp.SCRATCH: 1}
    assert total["evaluated"] == 7 and total["wins"] == 3 and total["win_rate"] == pytest.approx(3 / 7)
    assert total["net_pnl"] == 550


def test_win_loss_classification_follows_existing_outcomes():
    rows = [_row("2026-09-28", 390.0, 0, outcome=imp.WIN), _row("2026-09-28", -195.0, 1, outcome=imp.LOSS),
            _row("2026-09-28", -195.0, 2, outcome=imp.AMBIGUOUS), _row("2026-09-28", 0.0, 3, outcome=imp.SCRATCH)]
    m = _analysis(rows)["directions"]["BUY"]
    assert (m["evaluated"], m["wins"], m["losses"]) == (4, 1, 3)
    assert m["win_rate"] == 0.25 and m["net_pnl"] == 0.0


def test_unavailable_data_excluded_from_accuracy_and_pnl():
    rows = [_row("2026-09-28", 100.0, 0), _row("2026-09-28", 0, 1, state=sh.DATA_UNAVAILABLE),
            _row("2026-09-29", 0, 0, state=sh.DATA_UNAVAILABLE)]
    m = _analysis(rows)["directions"]["BUY"]
    assert (m["signals"], m["evaluated"], m["wins"], m["losses"], m["data_unavailable"]) == (3, 1, 1, 0, 2)
    assert m["win_rate"] == 1.0 and m["net_pnl"] == 100.0
    assert m["data_unavailable_reasons"] == {"option_candle_missing": 2}
    none = _analysis([_row("2026-09-28", 0, 0, state=sh.DATA_UNAVAILABLE)])["directions"]["BUY"]
    assert none["win_rate"] is None and none["profit_factor"] is None and none["net_pnl"] == 0


def test_drawdown_is_peak_to_trough_in_signal_order():
    rows = [_row("2026-09-28", 100.0, 0), _row("2026-09-28", -300.0, 1), _row("2026-09-29", 50.0, 0),
            _row("2026-09-29", -100.0, 1), _row("2026-09-30", 500.0, 0)]
    assert _analysis(rows)["directions"]["BUY"]["max_drawdown"] == 350


def test_daily_includes_every_session_even_without_buy_signals():
    a = _analysis()
    assert [r["date"] for r in a["daily"]] == DAYS
    d30 = a["daily"][2]
    assert (d30["buy_signals"], d30["buy_evaluated"], d30["buy_net_pnl"], d30["buy_win_rate"]) == (0, 0, 0, None)
    assert (d30["sell_signals"], d30["sell_evaluated"]) == (1, 1)
    d29 = a["daily"][1]
    assert (d29["buy_signals"], d29["buy_evaluated"], d29["buy_losses"], d29["buy_net_pnl"]) == (2, 1, 1, -150)
    assert (d29["sell_wins"], d29["sell_net_pnl"]) == (1, 300)


def test_dte_and_regime_aggregation():
    sell = _analysis()["directions"]["SELL"]
    assert list(sell["dte"]) == ["0 DTE", "1 DTE", "2-4 DTE", "5-7 DTE", "8-14 DTE", "15-21 DTE", "22+ DTE"]
    assert (sell["dte"]["0 DTE"]["evaluated"], sell["dte"]["0 DTE"]["net_pnl"]) == (2, 600)
    assert sell["dte"]["2-4 DTE"]["losses"] == 1 and sell["dte"]["1 DTE"]["evaluated"] == 1
    assert set(sell["regime"]) == {"strong_bearish_trend"}
    assert _analysis()["directions"]["BUY"]["regime"] == {}


def test_conclusion_is_descriptive_only():
    c = _analysis()["conclusion"]
    assert (c["higher_accuracy"], c["better_profit_factor"], c["better_pnl"]) == ("SELL", "SELL", "SELL")
    assert c["buy_evaluated"] == 3 and c["buy_data_unavailable"] == 1 and c["buy_sample_sufficient"] is False
    assert "robust" in c["note"] and "does not establish" in c["note"]
    assert ds._better(None, None).startswith("N/A") and ds._better(0.5, None) == "BUY" and ds._better(1, 1) == "TIE"


# --------------------------------------------------------------------------- lookahead
def test_replay_direction_lookahead_passes_on_real_pipeline():
    r = sh.replay(spot=_spot_frame(), **_replay_kwargs(), days=[DAY], mutation_cut="10:20", direction_lookahead=True)
    assert r["direction_lookahead"]["BUY"] == {"cuts": 1, "failures": [], "passed": True}
    assert r["direction_lookahead"]["SELL"]["passed"] is True


def _replay_kwargs():
    from tests.test_shadow_historical import LISTING, PARAMS, _premium_for, _resolve, _signal_fn
    check = sh.ExpiryCheck(LISTING, frozenset(), "test official list")
    return dict(signal_fn=_signal_fn, resolve=sh.strict_resolver(_resolve, check), premium_for=_premium_for(),
                params=PARAMS, square_off="15:15", range_start="09:15", range_end="09:30")


def test_direction_lookahead_detects_future_leak_per_direction():
    def leaky_pipeline(frame, premium_for, days):
        day = next(iter(days))
        future_close = float(frame["close"].iloc[-1])

        def ctx(t, d, atr):
            return {"session_date": day, "signal_time": _at(t), "direction": d, "instrument_key": "K",
                    "premium_atr": atr, "entry_time": _at(t) + pd.Timedelta(minutes=5)}
        return [ctx("10:00", "BUY", future_close), ctx("10:30", "SELL", 1.0)], None

    r = sh.direction_mutation_test(leaky_pipeline, _spot_frame(), lambda c: pd.DataFrame(), [_at("12:00")])
    assert r["BUY"]["passed"] is False and r["BUY"]["failures"]
    assert r["SELL"]["passed"] is True


# --------------------------------------------------------------------------- runner / reports / safety
def _runner(monkeypatch, _patched):
    import scripts.run_direction_20_sessions as runner
    return runner


def test_runner_uses_real_expired_option_data_and_writes_reports(monkeypatch, tmp_path):
    service = _ExpiredApiService(_spot_frame())
    runner = _runner(monkeypatch, _patch_expired(monkeypatch, service))
    other = tmp_path / "shadow_strategy_historical"
    other.mkdir()
    (other / "summary.json").write_text("keep")
    out = tmp_path / "strategy_direction_20_sessions"
    a = runner.run(out_dir=out, today=date(2026, 10, 6))
    assert (other / "summary.json").read_text() == "keep"
    buy, sell = a["directions"]["BUY"], a["directions"]["SELL"]
    assert (buy["signals"], buy["evaluated"], buy["wins"], buy["net_pnl"]) == (1, 1, 1, 390.0)
    assert buy["win_rate"] == 1.0 and buy["dte"]["0 DTE"]["evaluated"] == 1
    assert sell["evaluated"] == 2 and sell["wins"] + sell["losses"] == 2
    assert a["total"]["net_pnl"] == pytest.approx(390.0 - 195.0)
    assert {c[3] for c in service.downloader.calls} == {"EXPIRED"}
    assert a["lookahead"]["BUY"]["passed"] and a["lookahead"]["SELL"]["passed"]
    assert {p.name for p in out.iterdir()} == {"summary.md", "summary.json", "summary.csv", "daily.csv", "dte.csv",
                                              "regime.csv"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["trading_mode"] == "PAPER" and summary["live_trading_enabled"] is False
    daily = list(csv.DictReader((out / "daily.csv").open(encoding="utf-8")))
    assert [r["date"] for r in daily] == a["selected_dates"]
    dte = list(csv.DictReader((out / "dte.csv").open(encoding="utf-8")))
    assert len(dte) == 14
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "BUY ACCURACY = 100.0%" in md and "| Date | BUY Signals | BUY Evaluated |" in md


def test_runner_marks_unavailable_option_data_without_counting_it(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, _patch_runner(monkeypatch, _spot_frame()))
    a = runner.run(out_dir=tmp_path / "out", today=date(2026, 10, 6))
    buy = a["directions"]["BUY"]
    assert (buy["signals"], buy["evaluated"], buy["wins"], buy["losses"], buy["data_unavailable"]) == (1, 0, 0, 0, 1)
    assert buy["win_rate"] is None and a["conclusion"]["buy_data_unavailable"] == 1


@pytest.mark.parametrize("env", [{"LIVE_TRADING_ENABLED": "true"}, {"TRADING_MODE": "LIVE"}])
def test_runner_refuses_non_paper_mode(monkeypatch, tmp_path, env):
    runner = _runner(monkeypatch, _patch_runner(monkeypatch, _spot_frame()))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(LiveTradingSafetyError):
        runner.run(out_dir=tmp_path / "out", today=date(2026, 10, 6))
    assert not (tmp_path / "out").exists()


def test_direction_modules_have_no_order_imports():
    for path in (Path(ds.__file__), Path("scripts/run_direction_20_sessions.py")):
        src = path.read_text(encoding="utf-8")
        for forbidden in ("from src.broker", "from src.execution", "from src.engine", "place_order(",
                          "submit_signal("):
            assert forbidden not in src
