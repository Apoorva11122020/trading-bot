from __future__ import annotations

import copy
import json
import sys

import pandas as pd
import pytest

import test_shadow_validation as tsv
from src.backtest import live_quality_gate as qg
from src.backtest.live_quality_gate import QualityGateConfig, evaluate_quality
from src.database import Database
from src.paper_trading import live_quality_monitor as mon
from src.paper_trading import live_quality_report as rpt
from src.paper_trading import shadow_validation as sv
from src.safety.live_gate import LiveTradingSafetyError

SELL = {"direction": "SELL", "signal_time": "2026-10-05T12:30:00+05:30", "rsi14": 55.0, "ema20": 101.0,
        "ema50": 100.0, "atr14": 10.0, "break_atr": 0.9, "trend_regime": "RANGE", "spot": 25000.0}
BUY = {"direction": "BUY", "signal_time": "2026-10-05T10:00:00+05:30", "ema20": 100.0, "ema50": 115.0,
       "atr14": 10.0, "ema_sep_atr": -1.5, "trend_regime": "TREND_DOWN", "break_atr": 0.30, "rsi14": 40.0}


def _ctx(base, **kw):
    return {**base, **kw}


# ------------------------------------------------------------------ rules
def test_sell_rsi_55_gets_point():
    assert "SELL_RSI_50_60" in evaluate_quality(SELL).reasons


def test_sell_rsi_45_gets_no_rsi_point():
    d = evaluate_quality(_ctx(SELL, rsi14=45.0))
    assert "SELL_RSI_50_60" not in d.reasons and "SELL_RSI_50_60" in d.failed_checks
    assert d.score == 2


def test_sell_ema20_above_ema50_gets_point():
    assert "SELL_EMA20_GT_EMA50" in evaluate_quality(SELL).reasons
    assert "SELL_EMA20_GT_EMA50" not in evaluate_quality(_ctx(SELL, ema20=99.0)).reasons


def test_sell_1230_gets_time_point_and_1300_does_not():
    assert "SELL_TIME_12_13" in evaluate_quality(SELL).reasons
    assert "SELL_TIME_12_13" not in evaluate_quality(_ctx(SELL, signal_time="2026-10-05T13:00:00+05:30")).reasons
    utc = evaluate_quality(_ctx(SELL, signal_time=pd.Timestamp("2026-10-05T07:00:00Z")))
    assert "SELL_TIME_12_13" in utc.reasons


def test_buy_ema_sep_atr_below_minus_one_gets_point():
    assert "BUY_EMA_SEP_ATR_LT_-1" in evaluate_quality(BUY).reasons
    assert "BUY_EMA_SEP_ATR_LT_-1" not in evaluate_quality(_ctx(BUY, ema_sep_atr=-0.5)).reasons


def test_buy_trend_down_gets_point():
    assert "BUY_TREND_DOWN" in evaluate_quality(BUY).reasons
    assert "BUY_TREND_DOWN" not in evaluate_quality(_ctx(BUY, trend_regime="TREND_UP")).reasons


def test_buy_break_atr_at_most_half_gets_point():
    assert "BUY_BREAK_ATR_LE_0.50" in evaluate_quality(_ctx(BUY, break_atr=0.50)).reasons
    assert "BUY_BREAK_ATR_LE_0.50" not in evaluate_quality(_ctx(BUY, break_atr=0.51)).reasons


def test_full_scores_and_decision_fields():
    d = evaluate_quality(SELL)
    assert (d.accepted, d.score, d.max_score, d.confidence, d.direction) == (True, 3, 3, 1.0, "SELL")
    assert d.version == "LIVE_QUALITY_GATE_V1" and d.skip_reason is None
    b = evaluate_quality(BUY)
    assert (b.accepted, b.score) == (True, 3)


def test_missing_feature_is_reported_and_rejected_by_default():
    ctx = _ctx(SELL)
    ctx.pop("rsi14")
    d = evaluate_quality(ctx)
    assert d.missing_features == ["rsi14"]
    assert "DATA_UNAVAILABLE:rsi14" in d.failed_checks
    assert d.score == 2 and not d.accepted
    assert d.skip_reason.startswith("QUALITY_GATE_REJECTED:")
    assert evaluate_quality(ctx, QualityGateConfig(allow_partial_features=True)).accepted
    unavailable = evaluate_quality(_ctx(BUY, trend_regime="UNAVAILABLE", ema_sep_atr=None, ema20=None))
    assert set(unavailable.missing_features) == {"trend_regime", "ema_sep_atr"}


class _Tracking(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.read: set[str] = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def test_future_and_outcome_fields_are_never_read():
    poisoned = {"outcome": "WIN", "status": "WIN", "pnl": 9e9, "r_multiple": 9.0, "exit_price": 1.0,
                "exit_time": "2026-10-05T15:00:00+05:30", "exit_reason": "TARGET", "target": 1.0, "stop_loss": 1.0,
                "premium": "FUTURE", "future_close": 1.0}
    for base in (SELL, BUY):
        ctx = _Tracking({**base, **poisoned})
        d = evaluate_quality(ctx)
        assert ctx.read <= set(qg.CAUSAL_FEATURES)
        assert not set(d.features) & set(poisoned)
        flipped = evaluate_quality({**base, **{k: None for k in poisoned}})
        assert flipped == d


def test_same_input_same_decision():
    assert evaluate_quality(SELL) == evaluate_quality(copy.deepcopy(SELL))
    assert evaluate_quality(BUY).to_dict() == evaluate_quality(dict(BUY)).to_dict()


def test_score_threshold_is_configurable():
    two = _ctx(SELL, rsi14=45.0)
    assert evaluate_quality(two).accepted
    strict = evaluate_quality(two, QualityGateConfig(min_score_sell=3))
    assert not strict.accepted and "SCORE_2_BELOW_MIN_3" in strict.failed_checks
    one = _ctx(BUY, trend_regime="RANGE", break_atr=0.9)
    assert not evaluate_quality(one).accepted
    assert evaluate_quality(one, QualityGateConfig(min_score_buy=1)).accepted


def test_too_restrictive_guard():
    assert qg.restrictiveness(100, 4)["status"] == qg.TOO_RESTRICTIVE
    assert qg.restrictiveness(100, 0)["status"] == qg.TOO_RESTRICTIVE
    assert qg.restrictiveness(100, 5)["status"] == "OK"
    assert qg.restrictiveness(0, 0)["status"] == "NO_SIGNALS"
    assert qg.restrictiveness(10, 1, QualityGateConfig(min_accept_rate=0.2))["status"] == qg.TOO_RESTRICTIVE


def test_buy_breakout_confirm_is_not_a_production_or_gate_candidate():
    names = [c.name for c in sv.CANDIDATES]
    assert "BUY_BREAKOUT_CONFIRM" not in names + list(sv.OBSERVATION_CANDIDATES)
    assert "BUY_BREAKOUT_CONFIRM" in sv.RESEARCH_ONLY_CANDIDATES
    db = Database(":memory:")
    res = sv.run_once(day=tsv.DAY, database=db, prepare_inputs_fn=tsv._fake_prepare(), quality_gate=QualityGateConfig())
    assert "BUY_BREAKOUT_CONFIRM" not in res["created"]
    assert "BUY_BREAKOUT_CONFIRM" not in res["comparison"]["candidates"]
    assert all(r["candidate"] != "BUY_BREAKOUT_CONFIRM" for r in db.get_shadow_signals())


def test_gate_does_not_mutate_s2_context():
    spot = pd.DataFrame(tsv._spot(tsv.PREV, False) + tsv._spot(tsv.DAY, True))
    ctx = {"signal_time": tsv._at("10:30"), "session_date": tsv.DAY, "direction": "SELL", "spot": 24935.0,
           "breakout_level": 24990.0, "atr14": 7.0, "break_atr": 0.5, "outcome": None, "premium": spot}
    before = copy.deepcopy({k: v for k, v in ctx.items() if k != "premium"})
    gctx = qg.gate_context(ctx, spot)
    evaluate_quality(gctx)
    assert {k: v for k, v in ctx.items() if k != "premium"} == before
    assert ctx["premium"] is spot and "premium" not in gctx and "outcome" not in gctx


def test_gate_remains_paper_only(monkeypatch, tmp_path):
    import scripts.run_live_quality_gate as runner

    assert qg.PRODUCTION_EXECUTION_ENABLED is False
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    calls: list = []
    with pytest.raises(LiveTradingSafetyError):
        runner.run(tsv.DAY, Database(":memory:"), tmp_path, QualityGateConfig(), tsv._fake_prepare(calls=calls))
    assert calls == []


# ------------------------------------------------------------------ metrics
def _r(status, pnl=None, day="2026-10-05", at="10:00", decision="ACCEPTED", direction="SELL"):
    return {"status": status, "pnl": pnl, "decision": decision, "session_date": day, "direction": direction,
            "timestamp": f"{day}T{at}:00+05:30", "entry_price": 100.0, "stop_loss": 90.0}


def test_accuracy_excludes_pending():
    rows = [_r("WIN", 500), _r("LOSS", -200, at="10:05"), _r("PENDING", at="10:10"), _r("PENDING", at="10:15"),
            _r("DATA_UNAVAILABLE", at="10:20"), _r("SKIPPED", decision="SKIPPED", at="10:25")]
    m = mon.perf(rows)
    assert (m["generated"], m["accepted"], m["resolved"], m["wins"], m["losses"], m["pending"]) == (6, 5, 2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(0.5)


def test_zero_resolved_is_not_available():
    m = mon.perf([_r("PENDING")])
    assert m["accuracy"] is None and m["profit_factor"] is None
    assert rpt.pct(m["accuracy"]) == "N/A" and rpt.pf(m["profit_factor"]) == "N/A"
    assert mon.accuracy(0, 0) is None


def test_70pct_target_status_logic():
    def m(resolved, acc, pf, pnl=100.0):
        return {"resolved": resolved, "accuracy": acc, "profit_factor": pf, "pnl": pnl}
    assert mon.target_status(m(29, 0.9, 3.0)) == mon.INSUFFICIENT_SAMPLE
    assert mon.target_status(m(30, 0.70, 1.0)) == mon.ACHIEVED
    assert mon.target_status(m(30, 0.69, 2.0)) == mon.BELOW_TARGET
    assert mon.target_status(m(30, 0.80, 0.9, -5.0)) == mon.BELOW_TARGET
    assert mon.target_status(m(30, 1.0, None, 900.0)) == mon.ACHIEVED
    assert mon.rolling_status(m(10, 0.6, 2.0)) == mon.BELOW_70_TARGET
    assert mon.rolling_status(m(10, 0.8, 2.0)) == mon.ROLLING_INSUFFICIENT
    assert mon.rolling_status(m(30, 0.8, 2.0)) == mon.TARGET_MET
    assert mon.rolling_status(m(0, None, None)) == mon.ROLLING_INSUFFICIENT


def test_profit_factor():
    assert mon.profit_factor([300.0, -100.0, 200.0, -150.0]) == pytest.approx(2.0)
    assert mon.profit_factor([300.0]) is None
    assert mon.profit_factor([]) is None
    assert mon.profit_factor([-10.0]) == 0.0


def test_drawdown_and_losing_streak():
    assert mon.max_drawdown([100, -50, -80, 200, -300, 50]) == pytest.approx(300.0)
    assert mon.max_drawdown([100, 200]) == 0.0
    rows = [_r(s) for s in ("WIN", "LOSS", "AMBIGUOUS", "LOSS", "WIN", "LOSS")]
    assert mon.max_losing_streak(rows) == 3


def test_rolling_windows_use_last_resolved():
    rows = [_r("WIN" if i % 4 else "LOSS", 100 if i % 4 else -100, at=f"{10 + i // 12}:{(i % 12) * 5:02d}")
            for i in range(35)] + [_r("PENDING", at="15:00")]
    roll = mon.rolling(rows)
    assert roll["last_10"]["resolved"] == 10 and roll["last_30"]["resolved"] == 30
    assert roll["last_30"]["status"] == mon.TARGET_MET


# ------------------------------------------------------------------ integration
def test_shadow_integration_accepted_rows_match_baseline():
    db = Database(":memory:")
    res = sv.run_once(day=tsv.DAY, database=db, prepare_inputs_fn=tsv._fake_prepare(), quality_gate=QualityGateConfig())
    rows = {r["timestamp"][11:16]: r for r in db.get_shadow_signals() if r["candidate"] == qg.GATE_CANDIDATE}
    base = {r["timestamp"][11:16]: r for r in db.get_shadow_signals() if r["candidate"] == "BASELINE_S2_BUY_SELL"}
    assert set(rows) == set(base) == {"10:00", "10:30", "11:00"}
    assert res["created"][qg.GATE_CANDIDATE] == 3
    for t, r in rows.items():
        assert r["gate_version"] == qg.GATE_VERSION and r["gate_max_score"] == 3
        feats = json.loads(r["gate_features"])
        assert not set(feats) & set(qg.FORBIDDEN_FIELDS)
        if r["decision"] == "ACCEPTED":
            for f in ("instrument_key", "entry_price", "stop_loss", "target", "exit_price", "status", "pnl"):
                assert r[f] == base[t][f]
        else:
            assert r["status"] == "SKIPPED" and r["skip_reason"].startswith("QUALITY_GATE_REJECTED:")
            assert r["pnl"] is None and r["exit_price"] is None
        record = json.loads(r["record_json"])
        assert record["s2_context"]["direction"] == base[t]["direction"]
    again = sv.run_once(day=tsv.DAY, database=db, prepare_inputs_fn=tsv._fake_prepare(), quality_gate=QualityGateConfig())
    assert again["created"][qg.GATE_CANDIDATE] == 0 and len(db.get_shadow_signals()) == 12


def test_gate_decisions_are_point_in_time():
    full, cut = Database(":memory:"), Database(":memory:")
    sv.run_once(day=tsv.DAY, database=full, prepare_inputs_fn=tsv._fake_prepare(), quality_gate=QualityGateConfig())
    sv.run_once(day=tsv.DAY, database=cut, prepare_inputs_fn=tsv._fake_prepare(cutoff=tsv._at("10:40")),
                quality_gate=QualityGateConfig())
    f = {r["timestamp"]: r for r in full.get_shadow_signals() if r["candidate"] == qg.GATE_CANDIDATE}
    c = {r["timestamp"]: r for r in cut.get_shadow_signals() if r["candidate"] == qg.GATE_CANDIDATE}
    assert len(c) == 2
    for ts, r in c.items():
        for k in ("decision", "skip_reason", "gate_score", "gate_reasons", "gate_failed_checks", "gate_features"):
            assert r[k] == f[ts][k]


def test_features_match_research_definitions():
    from src.backtest import strategy_improvement_lab as lab
    from src.research.s2_diagnostic_lab import classify_regime

    spot = pd.DataFrame(tsv._spot(tsv.PREV, False) + tsv._spot(tsv.DAY, True))
    ts = tsv._at("11:00")
    got = qg.gate_features_at(spot, ts)
    bars = lab.bar_features(spot)
    b = bars.loc[ts]
    for k in ("ema20", "ema50", "rsi14", "atr14"):
        assert got[k] == pytest.approx(float(b[k]))
    slope = (bars["ema20"] - bars["ema20"].shift(3)) / bars["atr14"]
    trend, _ = classify_regime({"close": float(b["close"]), "ema20": float(b["ema20"]), "ema50": float(b["ema50"]),
                                "ema20_slope_atr": float(slope.loc[ts]), "atr_pctile": None})
    assert got["trend_regime"] == trend


def test_daily_and_cumulative_reports(tmp_path, capsys, monkeypatch):
    import scripts.run_live_quality_gate as runner

    db = Database(":memory:")
    daily = runner.run(tsv.DAY, db, tmp_path, QualityGateConfig(), tsv._fake_prepare())
    assert daily["status"] == "VALIDATED" and daily["generated"] == 3
    assert daily["accepted"] + daily["skipped"] == 3 and daily["live_orders"] == 0
    assert daily["production_s2_changed"] is False
    for scope in ("BUY", "SELL", "COMBINED"):
        m = daily["scopes"][scope]
        assert m["accuracy"] == (None if m["resolved"] == 0 else m["wins"] / m["resolved"])
    names = {p.name for p in tmp_path.iterdir()}
    assert {f"{tsv.DAY}.json", f"{tsv.DAY}.md", "cumulative.json", "cumulative.md"} <= names
    md = (tmp_path / f"{tsv.DAY}.md").read_text(encoding="utf-8")
    assert "Live Signal Accuracy" in md and "70% Target Status: INSUFFICIENT SAMPLE" in md
    assert "guaranteed" not in md.lower().replace("never a guarantee", "")
    cum = json.loads((tmp_path / "cumulative.json").read_text(encoding="utf-8"))
    assert cum["sessions"] == 1 and set(cum["direction_breakdown"]) == {"BUY", "SELL", "COMBINED"}
    assert "2026-10" in cum["monthly_breakdown"] and set(cum["rolling"]) == {"last_10", "last_20", "last_30"}


def test_missing_current_data_is_reported_not_fabricated(tmp_path):
    import scripts.run_live_quality_gate as runner

    def unavailable(**_):
        raise SystemExit("DATA_UNAVAILABLE: MISSING_LOCAL_PARQUET for NIFTY 2026-10-05")

    daily = runner.run(tsv.DAY, Database(":memory:"), tmp_path, QualityGateConfig(), unavailable)
    assert daily["status"] == qg.DATA_UNAVAILABLE and "MISSING_LOCAL_PARQUET" in daily["data_reason"]
    assert daily["generated"] == 0 and daily["live_signal_accuracy"]["accuracy"] is None


def test_report_script_cli(monkeypatch, tmp_path, capsys):
    import scripts.run_live_quality_report as cli

    path = tmp_path / "t.db"
    sv.run_once(day=tsv.DAY, database=Database(path), prepare_inputs_fn=tsv._fake_prepare(),
                quality_gate=QualityGateConfig())
    monkeypatch.setattr(sys, "argv", ["x", "--date", tsv.DAY.isoformat(), "--db", str(path),
                                      "--out-dir", str(tmp_path / "out")])
    cli.main()
    out = capsys.readouterr().out
    assert "LIVE QUALITY GATE V1" in out and "Production S2 changed: NO" in out and "Live orders: 0" in out
