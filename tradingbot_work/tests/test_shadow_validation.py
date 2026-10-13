from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest.current_day_s2 import LevelParams
from src.backtest.research_strategies import ResearchSignal
from src.database import Database
from src.historical_data.models import HistoricalDataError
from src.historical_data.upstox_client import UpstoxHistoricalClient
from src.paper_trading import shadow_validation as sv
from src.paper_trading.shadow_validation import CANDIDATES, comparison, shadow_id
from src.safety.live_gate import LiveTradingSafetyError

IST = ZoneInfo("Asia/Kolkata")
DAY = date(2026, 10, 5)
PREV = date(2026, 10, 1)


def _row(candidate="SELL_ONLY", direction="SELL", status="WIN", day="2026-10-05", at="10:00", pnl=1300.0):
    return {
        "shadow_id": f"{candidate}|{day}|{day}T{at}:00+05:30|{direction}",
        "candidate": candidate, "timestamp": f"{day}T{at}:00+05:30", "session_date": day,
        "underlying": "NIFTY", "direction": direction, "instrument_key": "NSE_FO|1", "expiry": "2026-10-06",
        "dte": 1, "decision": "ACCEPTED", "skip_reason": None, "status": status,
        "entry_price": 100.0, "stop_loss": 90.0, "target": 120.0, "exit_price": 120.0,
        "exit_time": f"{day}T10:10:00+05:30", "exit_reason": "TARGET", "pnl": pnl,
        "r_multiple": 2.0, "market_regime": None, "record_json": json.dumps({}),
        "created_at": f"{day}T10:00:00+05:30",
    }


def test_shadow_candidates_are_exactly_three():
    assert [c.name for c in CANDIDATES] == ["BASELINE_S2_BUY_SELL", "SELL_ONLY", "SELL_CONFIRMED"]


def test_shadow_insert_is_idempotent_and_outcome_can_resolve():
    db = Database(":memory:")
    row = _row()
    row["status"] = "PENDING"
    row["pnl"] = None
    row["exit_price"] = None
    row["exit_time"] = None
    row["exit_reason"] = None
    row["r_multiple"] = None
    assert db.save_shadow_signal(row)
    assert not db.save_shadow_signal(row)
    assert len(db.get_shadow_signals()) == 1
    resolved = dict(row, status="WIN", pnl=1300.0, exit_price=120.0, exit_time="2026-10-05T10:10:00+05:30", exit_reason="TARGET", r_multiple=2.0)
    assert db.update_shadow_outcome(row["shadow_id"], resolved)
    assert db.get_shadow_signals()[0]["status"] == "WIN"


def test_shadow_comparison_is_observation_only():
    db = Database(":memory:")
    db.save_shadow_signal(_row("SELL_ONLY", "SELL", "WIN"))
    db.save_shadow_signal(_row("SELL_CONFIRMED", "SELL", "LOSS"))
    stats = comparison(db, "2026-10-05")
    assert stats["observation_only"] is True
    assert stats["trading_mode"] == "PAPER"
    assert stats["live_trading_enabled"] is False
    assert stats["candidates"]["SELL_ONLY"]["win_rate"] == 1.0
    assert stats["candidates"]["SELL_CONFIRMED"]["win_rate"] == 0.0


# --------------------------------------------------------------------------- synthetic test fixtures
# Deterministic test-only candles; they exercise the pipeline and are never written to real reports.
def _slots(day: date) -> list[pd.Timestamp]:
    start = pd.Timestamp(datetime.combine(day, time(9, 15)), tz=IST)
    return [start + pd.Timedelta(minutes=5 * i) for i in range(75)]


def _spot(day: date, falling: bool) -> list[dict]:
    rows = []
    for i, ts in enumerate(_slots(day)):
        if not falling or i < 3:
            o = c = 25000.0
            h, lo = 25010.0, 24990.0
        else:
            o = 25000.0 - 5 * (i - 3)
            c = o - 5
            h, lo = o + 1, c - 1
        rows.append({"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 0})
    return rows


def _premium(overrides: dict[str, tuple]) -> pd.DataFrame:
    rows = []
    for day in (PREV, DAY):
        for ts in _slots(day):
            o, h, lo, c = overrides.get(ts.isoformat(), (100.0, 101.0, 99.0, 100.0)) if day == DAY else \
                (100.0, 101.0, 99.0, 100.0)
            rows.append({"timestamp": ts, "open": o, "high": h, "low": lo, "close": c, "volume": 0})
    return pd.DataFrame(rows)


def _at(hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(DAY, time.fromisoformat(hhmm)), tz=IST)


SIGNALS = [
    ResearchSignal(_at("10:00").to_pydatetime(), "BUY", "S2", "NIFTY", 25020.0, {"above": 25010.0}),
    ResearchSignal(_at("10:30").to_pydatetime(), "SELL", "S2", "NIFTY", 24935.0, {"below": 24990.0}),
    ResearchSignal(_at("11:00").to_pydatetime(), "SELL", "S2", "NIFTY", 24905.0, {"below": 24990.0}),
]
KEYS = {"10:00": "NSE_FO|OK_CE", "10:30": "NSE_FO|OK_PE", "11:00": "NSE_FO|MISSING_PE"}
PREMIUMS = {
    "NSE_FO|OK_CE": _premium({_at("10:15").isoformat(): (100.0, 110.0, 99.5, 108.0)}),   # target 106 -> WIN
    "NSE_FO|OK_PE": _premium({_at("10:45").isoformat(): (100.0, 100.5, 90.0, 92.0)}),    # stop 97 -> LOSS
}
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=14, stop_atr_multiplier=1.5,
                     target_r=2.0, slippage_bps=0.0, interval=5)


def _fake_prepare(cutoff: pd.Timestamp | None = None, missing_reason: str | None = "PREMIUM_API_ERROR:AUTH_FAILED",
                  calls: list | None = None, hook=None):
    """Stand-in for prepare_inputs: same contract for every candidate, candles completed before ``cutoff``."""
    def complete(frame: pd.DataFrame) -> pd.DataFrame:
        if cutoff is None:
            return frame.reset_index(drop=True)
        return frame[frame["timestamp"] + pd.Timedelta(minutes=5) <= cutoff].reset_index(drop=True)

    def prepare(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        if hook is not None:
            hook()
        spot = complete(pd.DataFrame(_spot(PREV, False) + _spot(DAY, True)))

        def signal_fn(frame):
            last = frame["timestamp"].max()
            return [s for s in SIGNALS if pd.Timestamp(s.timestamp) <= last]

        def resolve(sig):
            key = KEYS[pd.Timestamp(sig.timestamp).strftime("%H:%M")]
            return SimpleNamespace(instrument_key=key, expiry=date(2026, 10, 6), trading_symbol=key, tick_size=0.05)

        def premium_for(contract):
            if contract.instrument_key not in PREMIUMS:
                return missing_reason
            return complete(PREMIUMS[contract.instrument_key])

        return {"research": {"signal_fn": signal_fn, "resolve": resolve, "params": PARAMS, "range_start": "09:15",
                             "range_end": "09:30", "current_spot": spot, "current_premium_for": premium_for,
                             "square_off": "15:15", "hist_spot": spot[spot["timestamp"].dt.date < DAY]},
                "point_in_time_matches_full_run": True,
                "data": {"current_day_source": "TEST_FIXTURE", "historical_window": "test"}}
    return prepare


def _by(result, strategy):
    return {s["signal_time"][11:16]: s for s in result["signals"] if s["strategy"] == strategy}


# --------------------------------------------------------------------------- historical data
def test_complete_stored_session_requires_every_slot():
    from scripts.run_s2_1dte_improvement import complete_stored_session

    full = pd.DataFrame(_spot(DAY, False))
    assert len(complete_stored_session(full, DAY, 5)) == 75
    assert complete_stored_session(full.iloc[:-1], DAY, 5) is None
    assert complete_stored_session(full, PREV, 5) is None
    assert complete_stored_session(pd.DataFrame(), DAY, 5) is None


class _NoApiClient:
    calls = 0

    def intraday_candles(self, *a, **k):
        _NoApiClient.calls += 1
        raise AssertionError("Upstox must not be called when local data exists")


def _local_service(monkeypatch, tmp_path, past_day: date, with_day: bool):
    import scripts.run_s2_1dte_improvement as mod
    from src.historical_data.service import HistoricalDataService

    def factory():
        svc = HistoricalDataService(store_root=tmp_path)
        svc.client = _NoApiClient()
        svc.underlying_key = lambda name: "NSE_INDEX|Nifty 50"
        return svc

    def frame(days, key):
        rows = [dict(r, instrument_key=key, open_interest=0) for d in days for r in _spot(d, False)]
        return pd.DataFrame(rows)

    store = factory().store
    hist_day = past_day - timedelta(days=1)
    store.write("underlying", "NIFTY", "NSE_INDEX|Nifty 50",
                frame([hist_day, past_day] if with_day else [hist_day], "NSE_INDEX|Nifty 50"))
    store.write("options", "NIFTY", "NSE_FO|STORED", frame([hist_day, past_day], "NSE_FO|STORED"),
                expiry=date(2026, 10, 6))
    monkeypatch.setattr(mod, "HistoricalDataService", factory)
    _NoApiClient.calls = 0
    return mod


def test_prepare_inputs_uses_local_parquet_instead_of_upstox(monkeypatch, tmp_path):
    day = date(2026, 9, 29)
    mod = _local_service(monkeypatch, tmp_path, day, with_day=True)
    inputs = mod.prepare_inputs(day=day, hist_from=day - timedelta(days=35), hist_to=day - timedelta(days=1),
                                prefer_stored_current_day=True)
    assert inputs["data"]["current_day_source"] == "LOCAL_PARQUET"
    premium_for = inputs["research"]["current_premium_for"]
    stored = premium_for(SimpleNamespace(instrument_key="NSE_FO|STORED", expiry=date(2026, 10, 6)))
    assert set(stored["timestamp"].dt.date) == {day - timedelta(days=1), day}
    missing = premium_for(SimpleNamespace(instrument_key="NSE_FO|NOT_STORED", expiry=date(2026, 10, 6)))
    assert missing == "MISSING_LOCAL_PARQUET"
    assert _NoApiClient.calls == 0


def test_prepare_inputs_past_day_without_local_data_is_data_unavailable(monkeypatch, tmp_path):
    day = date(2026, 9, 29)
    mod = _local_service(monkeypatch, tmp_path, day, with_day=False)
    with pytest.raises(SystemExit, match="DATA_UNAVAILABLE: MISSING_LOCAL_PARQUET"):
        mod.prepare_inputs(day=day, hist_from=day - timedelta(days=35), hist_to=day - timedelta(days=1),
                           prefer_stored_current_day=True)
    assert _NoApiClient.calls == 0


def _session(statuses):
    calls = {"n": 0}

    class Resp:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}

        def json(self):
            if self.status_code == 200:
                return {"status": "success", "data": {"candles": []}}
            return {"status": "error", "errors": [{"errorCode": "UDAPI", "message": "nope"}]}

    class Session:
        def get(self, *a, **k):
            calls["n"] += 1
            return Resp(statuses[min(calls["n"], len(statuses)) - 1])

    return Session(), calls


def test_upstox_retry_is_not_reported_as_final_error():
    session, calls = _session([503, 200])
    client = UpstoxHistoricalClient(token_provider=lambda: "token", session=session, sleep=lambda _: None)
    with sv.capture_upstox_events() as events:
        assert client.intraday_candles("NSE_FO|1") == []
    assert calls["n"] == 2
    assert [e["event"] for e in events] == ["upstox_historical_retry"]
    assert events[0]["code"] == "UPSTOX_SERVER_ERROR" and events[0]["status"] == 503
    assert "token" not in json.dumps(events)


def test_upstox_auth_failure_is_final_and_not_retried():
    session, calls = _session([401])
    client = UpstoxHistoricalClient(token_provider=lambda: "secret-token", session=session, sleep=lambda _: None)
    with sv.capture_upstox_events() as events, pytest.raises(HistoricalDataError) as exc:
        client.intraday_candles("NSE_FO|1")
    assert exc.value.code == "AUTH_FAILED"
    assert calls["n"] == 1
    assert [(e["event"], e["code"], e["status"]) for e in events] == [("upstox_historical_error", "AUTH_FAILED", 401)]
    assert "secret-token" not in json.dumps(events)


def test_capture_handler_is_removed_after_run():
    lg = logging.getLogger("historical_data.upstox")
    before = list(lg.handlers)
    with sv.capture_upstox_events():
        pass
    assert lg.handlers == before


# --------------------------------------------------------------------------- run_once
def test_run_once_states_unavailable_reason_and_isolation():
    db = Database(":memory:")
    calls: list = []
    result = sv.run_once(day=DAY, database=db, prepare_inputs_fn=_fake_prepare(calls=calls))
    assert calls[0]["hist_to"] == DAY - timedelta(days=1)
    assert calls[0]["download_missing_options"] is False
    assert calls[0]["prefer_stored_current_day"] is True

    base, sell, conf = (_by(result, n) for n in ("BASELINE_S2_BUY_SELL", "SELL_ONLY", "SELL_CONFIRMED"))
    assert [base[t]["state"] for t in ("10:00", "10:30", "11:00")] == ["EVALUATED", "EVALUATED", "DATA_UNAVAILABLE"]
    assert (base["10:00"]["outcome"], base["10:30"]["outcome"]) == ("WIN", "LOSS")
    assert sell["10:00"]["state"] == "SKIPPED" and sell["10:00"]["skip_reason"] == "DIRECTION_EXCLUDED"
    assert conf["10:30"]["state"] == "EVALUATED"
    for rows in (base, sell, conf):
        assert rows["11:00"]["data_reason"] == "Upstox historical authorization unavailable (AUTH_FAILED)"
    for field in ("instrument_key", "entry_price", "stop_loss", "target", "exit_price", "exit_time", "pnl"):
        assert base["10:30"][field] == sell["10:30"][field] == conf["10:30"][field]

    c = result["comparison"]["candidates"]
    assert (c["BASELINE_S2_BUY_SELL"]["evaluated"], c["BASELINE_S2_BUY_SELL"]["wins"]) == (2, 1)
    assert c["BASELINE_S2_BUY_SELL"]["net_pnl"] == pytest.approx(390.0 - 195.0)
    assert c["BASELINE_S2_BUY_SELL"]["max_drawdown"] == pytest.approx(195.0)
    assert c["SELL_ONLY"]["net_pnl"] == pytest.approx(-195.0)
    for name in c:
        assert c[name]["data_unavailable"] == 1
        assert c[name]["data_unavailable_reasons"] == {"Upstox historical authorization unavailable (AUTH_FAILED)": 1}


def test_repeated_same_day_run_does_not_duplicate_or_double_count(tmp_path):
    db = Database(":memory:")
    first = sv.run_once(day=DAY, database=db, prepare_inputs_fn=_fake_prepare())
    sv.write_report(first, tmp_path)
    second = sv.run_once(day=DAY, database=db, prepare_inputs_fn=_fake_prepare())
    sv.write_report(second, tmp_path)
    cum = sv.write_cumulative(db, tmp_path)
    assert first["created"] == {"BASELINE_S2_BUY_SELL": 3, "SELL_ONLY": 3, "SELL_CONFIRMED": 3}
    assert second["created"] == {"BASELINE_S2_BUY_SELL": 0, "SELL_ONLY": 0, "SELL_CONFIRMED": 0}
    assert len(db.get_shadow_signals()) == 9
    assert second["comparison"] == first["comparison"]
    base = cum["candidates"]["BASELINE_S2_BUY_SELL"]
    assert (base["trading_days"], base["signals"], base["net_pnl"]) == (1, 3, pytest.approx(195.0))
    assert cum["validated_days"] == [DAY.isoformat()]


def test_rerun_reconciles_data_unavailable_without_new_rows():
    db = Database(":memory:")
    sv.run_once(day=DAY, database=db, prepare_inputs_fn=_fake_prepare())
    PREMIUMS["NSE_FO|MISSING_PE"] = _premium({})
    try:
        again = sv.run_once(day=DAY, database=db, prepare_inputs_fn=_fake_prepare())
    finally:
        PREMIUMS.pop("NSE_FO|MISSING_PE")
    assert again["created"]["BASELINE_S2_BUY_SELL"] == 0
    assert again["reconciliation"]["BASELINE_S2_BUY_SELL"]["updated"] == 1
    assert len(db.get_shadow_signals()) == 9
    late = _by(again, "BASELINE_S2_BUY_SELL")["11:00"]
    assert late["state"] in {"PENDING", "EVALUATED"}
    assert late["state"] != "DATA_UNAVAILABLE"


def test_no_lookahead_earlier_decisions_identical_when_later_candles_arrive():
    full = sv.run_once(day=DAY, database=Database(":memory:"), prepare_inputs_fn=_fake_prepare())
    cut = sv.run_once(day=DAY, database=Database(":memory:"), prepare_inputs_fn=_fake_prepare(cutoff=_at("10:40")))
    for name in [c.name for c in CANDIDATES]:
        early, later = _by(cut, name), _by(full, name)
        assert set(early) == {"10:00", "10:30"}
        for t in early:
            for field in ("decision", "skip_reason", "instrument_key", "entry_price", "stop_loss", "target"):
                assert early[t][field] == later[t][field]
    pending = _by(cut, "BASELINE_S2_BUY_SELL")["10:30"]
    assert pending["state"] == "PENDING" and pending["pnl"] is None


# --------------------------------------------------------------------------- aggregation
def test_cumulative_aggregates_pnl_drawdown_and_days():
    db = Database(":memory:")
    for day, at, pnl, status in (("2026-10-05", "10:00", 1000.0, "WIN"), ("2026-10-05", "11:00", -500.0, "LOSS"),
                                 ("2026-10-06", "10:00", -2000.0, "LOSS"), ("2026-10-06", "11:00", 300.0, "WIN")):
        assert db.save_shadow_signal(_row("SELL_ONLY", "SELL", status, day=day, at=at, pnl=pnl))
    pending = _row("SELL_ONLY", "SELL", "DATA_UNAVAILABLE", day="2026-10-06", at="12:00", pnl=None)
    pending["exit_reason"] = "MISSING_LOCAL_PARQUET"
    db.save_shadow_signal(pending)
    m = sv.cumulative(db, ["2026-10-05", "2026-10-06", "2026-10-07"])["candidates"]["SELL_ONLY"]
    assert m["trading_days"] == 3
    assert (m["signals"], m["evaluated"], m["wins"], m["losses"]) == (5, 4, 2, 2)
    assert m["data_unavailable"] == 1
    assert m["data_unavailable_reasons"] == {"missing local parquet": 1}
    assert m["net_pnl"] == pytest.approx(-1200.0)
    assert m["max_drawdown"] == pytest.approx(2500.0)
    assert m["profit_factor"] == pytest.approx(1300.0 / 2500.0)
    assert m["avg_pnl_per_trade"] == pytest.approx(-300.0)
    assert m["best_day"] == {"date": "2026-10-05", "net_pnl": 500.0}
    assert m["worst_day"] == {"date": "2026-10-06", "net_pnl": -1700.0}
    assert (m["profitable_days"], m["losing_days"]) == (1, 1)
    assert m["average_daily_pnl"] == pytest.approx(-400.0)


def test_ambiguous_candle_is_booked_as_stop_loss():
    row = _row(status="AMBIGUOUS", pnl=None)
    assert sv.trade_pnl(row) == pytest.approx((90.0 - 100.0) * 65)
    m = sv.metrics([row])
    assert (m["evaluated"], m["losses"], m["net_pnl"]) == (1, 1, pytest.approx(-650.0))


# --------------------------------------------------------------------------- safety
def test_order_entry_points_are_blocked_during_run_and_restored():
    from src.broker.paper import PaperBroker
    from src.engine import TradingEngine
    from src.execution.order_manager import OrderManager

    originals = (PaperBroker.place_order, OrderManager.submit_signal, TradingEngine.submit_entry)

    def try_order():
        with pytest.raises(LiveTradingSafetyError, match="SHADOW_VALIDATION_ORDER_BLOCKED"):
            OrderManager.submit_signal(None)
        with pytest.raises(LiveTradingSafetyError, match="SHADOW_VALIDATION_ORDER_BLOCKED"):
            PaperBroker.place_order(None, "NIFTY", "BUY", 1, "MARKET")
        with pytest.raises(LiveTradingSafetyError, match="SHADOW_VALIDATION_ORDER_BLOCKED"):
            TradingEngine.submit_entry(None, None)

    sv.run_once(day=DAY, database=Database(":memory:"), prepare_inputs_fn=_fake_prepare(hook=try_order))
    assert (PaperBroker.place_order, OrderManager.submit_signal, TradingEngine.submit_entry) == originals


@pytest.mark.parametrize("env", [{"LIVE_TRADING_ENABLED": "true"}, {"TRADING_MODE": "LIVE"}])
def test_shadow_run_refuses_non_paper_mode(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    calls: list = []
    with pytest.raises(LiveTradingSafetyError):
        sv.run_once(day=DAY, database=Database(":memory:"), prepare_inputs_fn=_fake_prepare(calls=calls))
    assert calls == []


def test_shadow_module_has_no_order_imports():
    src = Path(sv.__file__).read_text(encoding="utf-8")
    for forbidden in ("from src.broker", "from src.execution", "from src.engine", "import src.broker",
                      "place_order(", "submit_signal("):
        assert forbidden not in src


# --------------------------------------------------------------------------- runner
def test_runner_writes_daily_and_cumulative_reports_idempotently(monkeypatch, tmp_path, capsys):
    import scripts.run_shadow_strategy_validation as runner

    monkeypatch.setattr(runner, "prepare_inputs", _fake_prepare())
    out = tmp_path / "reports"
    argv = ["run_shadow_strategy_validation.py", "--date", DAY.isoformat(), "--db", str(tmp_path / "t.db"),
            "--out-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    runner.main()
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([f"{DAY}.json", f"{DAY}.md", f"{DAY}.csv", "cumulative.json", "cumulative.md",
                            "cumulative.csv"])
    cum = json.loads((out / "cumulative.json").read_text(encoding="utf-8"))
    assert cum["validated_days"] == [DAY.isoformat()]
    assert cum["candidates"]["BASELINE_S2_BUY_SELL"]["signals"] == 3
    daily_csv = (out / f"{DAY}.csv").read_text(encoding="utf-8").splitlines()
    assert len(daily_csv) == 1 + 9
    md = (out / "cumulative.md").read_text(encoding="utf-8")
    assert "| Strategy | Days | Signals | Evaluated | Wins | Losses | WR | PF | P&L (Rs) | Max DD (Rs) |" in md
    assert "no orders" in capsys.readouterr().out
