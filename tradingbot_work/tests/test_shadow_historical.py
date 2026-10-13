from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.backtest import shadow_historical as sh
from src.historical_data.models import HistoricalDataError
from src.options.models import OptionResolutionError
from src.safety.live_gate import LiveTradingSafetyError
from tests.test_shadow_validation import DAY, KEYS, PARAMS, PREMIUMS, PREV, SIGNALS, _at, _premium, _spot

LISTING = date(2026, 10, 6)


def _spot_frame(days_falling=((PREV, False), (DAY, True))) -> pd.DataFrame:
    return pd.DataFrame([r for d, f in days_falling for r in _spot(d, f)])


def _signal_fn(frame):
    last = frame["timestamp"].max()
    return [s for s in SIGNALS if pd.Timestamp(s.timestamp) <= last]


def _resolve(sig):
    key = KEYS[pd.Timestamp(sig.timestamp).strftime("%H:%M")]
    return SimpleNamespace(instrument_key=key, expiry=date(2026, 10, 6), trading_symbol=key, tick_size=0.05)


def _premium_for(cutoff=None):
    def get(contract):
        if contract.instrument_key not in PREMIUMS:
            return "NO_PREMIUM_DATA"
        f = PREMIUMS[contract.instrument_key]
        if cutoff is not None:
            f = f[f["timestamp"] + pd.Timedelta(minutes=5) <= cutoff]
        return f.reset_index(drop=True)
    return get


def _replay(check=None, spot=None, premium_for=None, days=(DAY,), mutation_cut=None, regimes=None):
    check = check or sh.ExpiryCheck(LISTING, frozenset(), "test official list")
    return sh.replay(spot=_spot_frame() if spot is None else spot, signal_fn=_signal_fn,
                     resolve=sh.strict_resolver(_resolve, check), premium_for=premium_for or _premium_for(),
                     params=PARAMS, square_off="15:15", range_start="09:15", range_end="09:30", days=list(days),
                     regimes=regimes, mutation_cut=mutation_cut)


def _by(rows):
    return {r["signal_time"][11:16]: r for r in rows}


# --------------------------------------------------------------------------- trading days
def test_complete_sessions_skip_weekends_missing_and_incomplete_days():
    fri, mon, tue, wed = date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)
    spot = pd.DataFrame(_spot(fri, False) + _spot(mon, False) + _spot(wed, False)[:40])
    sessions, excluded = sh.complete_sessions(spot, fri, wed)
    assert sessions == [fri, mon]
    assert tue.isoformat() in excluded and excluded[tue.isoformat()].startswith("no_stored_candles")
    assert excluded[wed.isoformat()] == "incomplete_session (40/75 candles)"
    assert "2026-09-26" not in excluded and "2026-09-27" not in excluded


def test_date_filter_and_latest_n_sessions():
    days = [date(2026, 9, d) for d in (1, 2, 3, 4, 7, 8)]
    assert sh.select_days(days, 3) == [date(2026, 9, 4), date(2026, 9, 7), date(2026, 9, 8)]
    assert sh.select_days(list(reversed(days))) == days
    spot = pd.DataFrame([r for d in days for r in _spot(d, False)])
    sessions, _ = sh.complete_sessions(spot, date(2026, 9, 3), date(2026, 9, 7))
    assert sessions == [date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]


def test_replay_only_emits_signals_for_selected_days():
    assert _replay(days=[PREV])["rows"]["BASELINE_S2_BUY_SELL"] == []
    rows = _replay()["rows"]["BASELINE_S2_BUY_SELL"]
    assert {r["session_date"] for r in rows} == {DAY.isoformat()}
    assert len(rows) == len(SIGNALS)


# --------------------------------------------------------------------------- strategies
def test_strategy_isolation_sell_only_and_sell_confirmed():
    rows = _replay()["rows"]
    base, sell, conf = (_by(rows[n]) for n in ("BASELINE_S2_BUY_SELL", "SELL_ONLY", "SELL_CONFIRMED"))
    assert (base["10:00"]["state"], base["10:00"]["outcome"]) == ("EVALUATED", "WIN")
    assert (sell["10:00"]["state"], sell["10:00"]["skip_reason"]) == ("SKIPPED", "DIRECTION_EXCLUDED")
    assert conf["10:00"]["state"] == "SKIPPED"
    assert conf["10:30"]["state"] == "EVALUATED" and conf["10:30"]["skip_reason"] is None
    for f in ("instrument_key", "entry", "entry_time", "stop_loss", "target", "exit_price", "pnl"):
        assert base["10:30"][f] == sell["10:30"][f] == conf["10:30"][f]


def test_sell_confirmed_skips_when_confirmation_fails():
    flat = _spot_frame(((PREV, False), (DAY, False)))
    conf = _by(_replay(spot=flat)["rows"]["SELL_CONFIRMED"])
    assert conf["10:30"]["state"] == "SKIPPED"
    assert conf["10:30"]["skip_reason"].startswith("CONFIRMATION_FAILED")


# --------------------------------------------------------------------------- contracts / data
def test_expiry_check_rules():
    known = sh.ExpiryCheck(LISTING, frozenset({date(2026, 9, 29)}), "official")
    assert known.verify(date(2026, 9, 30), date(2026, 10, 6))
    assert not known.verify(date(2026, 9, 25), date(2026, 10, 6))
    assert known.verify(LISTING, date(2026, 10, 13))
    unknown = sh.ExpiryCheck(LISTING, None, "PLAN_NOT_SUPPORTED")
    assert not unknown.verify(date(2026, 10, 1), date(2026, 10, 6))
    with pytest.raises(OptionResolutionError) as exc:
        sh.strict_resolver(_resolve, unknown)(SIGNALS[0])
    assert exc.value.code == sh.EXPIRED_UNAVAILABLE
    assert sh.strict_resolver(_resolve, known)(SIGNALS[0]).expiry == date(2026, 10, 6)


def test_unverifiable_expiry_is_data_unavailable_not_substituted():
    rows = _replay(check=sh.ExpiryCheck(LISTING, None, "PLAN_NOT_SUPPORTED"))["rows"]
    base = rows["BASELINE_S2_BUY_SELL"]
    assert {r["state"] for r in base} == {"DATA_UNAVAILABLE"}
    assert {r["reason"] for r in base} == {"expired_instrument_unavailable"}
    assert all(r["instrument_key"] is None and r["expiry"] is None and r["pnl"] is None for r in base)
    m = sh.stats(base)
    assert (m["evaluated"], m["losses"], m["net_pnl"], m["data_coverage"]) == (0, 0, 0.0, 0.0)


def test_missing_option_file_is_data_unavailable_and_not_a_loss():
    rows = _replay()["rows"]
    for name in ("BASELINE_S2_BUY_SELL", "SELL_ONLY", "SELL_CONFIRMED"):
        late = _by(rows[name])["11:00"]
        assert (late["state"], late["reason"], late["reason_raw"]) == \
            ("DATA_UNAVAILABLE", "local_historical_file_missing", "NO_PREMIUM_DATA")
    m = sh.stats(rows["SELL_ONLY"])
    assert (m["signals"], m["skipped"], m["evaluated"], m["losses"], m["data_unavailable"]) == (3, 1, 1, 1, 1)
    assert m["data_coverage"] == pytest.approx(0.5)
    assert m["data_unavailable_pct"] == pytest.approx(0.5)


@pytest.mark.parametrize("raw,code", [
    ("CONTRACT_UNRESOLVED:EXPIRED_INSTRUMENT_UNAVAILABLE", "expired_instrument_unavailable"),
    ("CONTRACT_UNRESOLVED:STRIKE_NOT_FOUND", "option_contract_not_found"),
    ("NO_PREMIUM_DATA", "local_historical_file_missing"),
    ("NO_PREMIUM_ENTRY_CANDLE", "option_candle_missing"),
    ("NO_PREMIUM_CANDLE_AT_SIGNAL", "option_candle_missing"),
    ("NO_EXIT_EVIDENCE", "option_candle_missing"),
    ("INSUFFICIENT_ATR_HISTORY", "insufficient_option_atr_history"),
    ("PREMIUM_API_ERROR:PLAN_NOT_SUPPORTED", "expired_instrument_unavailable"),
    ("PREMIUM_API_ERROR:AUTH_FAILED", "historical_api_unavailable"),
])
def test_reason_codes(raw, code):
    assert sh.reason_code(raw) == code


def test_entry_candle_missing_is_data_unavailable():
    gap = PREMIUMS["NSE_FO|OK_CE"]
    gap = gap[gap["timestamp"] != _at("10:05")].reset_index(drop=True)

    def premium_for(contract):
        return gap if contract.instrument_key == "NSE_FO|OK_CE" else _premium_for()(contract)

    row = _by(_replay(premium_for=premium_for)["rows"]["BASELINE_S2_BUY_SELL"])["10:00"]
    assert (row["state"], row["reason"]) == ("DATA_UNAVAILABLE", "option_candle_missing")


# --------------------------------------------------------------------------- lookahead
def test_mutation_lookahead_test_passes_and_truncation_keeps_decisions():
    full = _replay(mutation_cut="10:20")
    assert all(v["passed"] and v["cuts"] == 1 for v in full["lookahead"].values())
    cut = _at("10:40")
    early = _replay(spot=_spot_frame()[lambda f: f["timestamp"] + pd.Timedelta(minutes=5) <= cut],
                    premium_for=_premium_for(cut))
    for name, rows in early["rows"].items():
        later = _by(full["rows"][name])
        for t, r in _by(rows).items():
            for f in ("decision", "skip_reason", "instrument_key", "entry", "entry_time", "stop_loss", "target"):
                assert r[f] == later[t][f]
            assert r["entry_time"] is None or r["entry_time"] > r["signal_time"]


# --------------------------------------------------------------------------- aggregation
def test_dte_bucket_boundaries_and_row_dte():
    assert [sh.dte_bucket(x) for x in (0, 1, 2, 4, 5, 7, 8, 14, 15, 21, 22, 40, None)] == \
        ["0 DTE", "1 DTE", "2-4 DTE", "2-4 DTE", "5-7 DTE", "5-7 DTE", "8-14 DTE", "8-14 DTE", "15-21 DTE",
         "15-21 DTE", "22+ DTE", "22+ DTE", None]
    row = _by(_replay()["rows"]["BASELINE_S2_BUY_SELL"])["10:00"]
    assert (row["dte"], row["dte_bucket"]) == (1, "1 DTE")


def _r(day, at, direction, outcome, pnl, state="EVALUATED", dte=3, regimes=()):
    return {"session_date": day, "signal_time": f"{day}T{at}:00+05:30", "direction": direction, "state": state,
            "outcome": outcome, "pnl": pnl, "reason": None if state != "DATA_UNAVAILABLE" else "option_candle_missing",
            "dte_bucket": sh.dte_bucket(dte), "time_bucket": "10:00-11:00", "regimes": list(regimes)}


def test_pnl_drawdown_direction_and_day_aggregation():
    rows = [_r("2026-09-01", "10:00", "BUY", "WIN", 1000.0, regimes=("strong_bullish_trend",)),
            _r("2026-09-01", "11:00", "SELL", "LOSS", -500.0),
            _r("2026-09-02", "10:00", "SELL", "LOSS", -2000.0, dte=9),
            _r("2026-09-02", "11:00", "SELL", "WIN", 300.0, dte=9),
            _r("2026-09-02", "12:00", "SELL", None, None, state="DATA_UNAVAILABLE"),
            _r("2026-09-02", "13:00", "BUY", None, None, state="SKIPPED")]
    m = sh.strategy_summary(rows, ["2026-09-01", "2026-09-02", "2026-09-03"])
    assert (m["signals"], m["evaluated"], m["wins"], m["losses"], m["data_unavailable"], m["skipped"]) == \
        (6, 4, 2, 2, 1, 1)
    assert m["net_pnl"] == pytest.approx(-1200.0)
    assert m["max_drawdown"] == pytest.approx(2500.0)
    assert m["profit_factor"] == pytest.approx(1300.0 / 2500.0)
    assert m["data_coverage"] == pytest.approx(4 / 5)
    assert m["best_day"] == {"date": "2026-09-01", "net_pnl": 500.0}
    assert m["worst_day"] == {"date": "2026-09-02", "net_pnl": -1700.0}
    assert (m["profitable_days"], m["losing_days"], m["average_daily_pnl"]) == (1, 1, pytest.approx(-400.0))
    assert (m["direction"]["BUY"]["evaluated"], m["direction"]["BUY"]["net_pnl"]) == (1, 1000.0)
    assert (m["direction"]["SELL"]["wins"], m["direction"]["SELL"]["losses"]) == (1, 2)
    assert m["dte"]["2-4 DTE"]["net_pnl"] == pytest.approx(500.0)
    assert m["dte"]["8-14 DTE"]["net_pnl"] == pytest.approx(-1700.0)
    assert m["regime"]["strong_bullish_trend"]["evaluated"] == 1


# --------------------------------------------------------------------------- reports / duplicates / separation
def _summary(result):
    return sh.build_summary(result, {"date_range": "2026-10-05..2026-10-05", "excluded_days": {},
                                     "expiry_check": {"summary": "test"}})


def test_reports_are_rewritten_without_duplicates_and_kept_separate(tmp_path):
    result = _replay()
    current = {"validated_days": ["2026-10-05"],
               "candidates": {"BASELINE_S2_BUY_SELL": {"trading_days": 1, "evaluated": 40, "wins": 8, "losses": 32,
                                                       "win_rate": 0.2, "profit_factor": 0.59, "net_pnl": -19259.5,
                                                       "max_drawdown": 33478.25}}}
    (tmp_path / "2026-09-01.md").write_text("stale", encoding="utf-8")
    for _ in range(2):
        paths = sh.write_reports(_summary(result), result["rows"], tmp_path, current)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted([f"{DAY}.json", f"{DAY}.csv", f"{DAY}.md", "summary.json", "summary.csv", "summary.md"])
    daily = json.loads((tmp_path / f"{DAY}.json").read_text(encoding="utf-8"))
    keys = [(s["strategy"], s["signal_time"], s["direction"]) for s in daily["signals"]]
    assert len(keys) == len(set(keys)) == 9
    md = paths["md"].read_text(encoding="utf-8")
    assert md.startswith("# HISTORICAL SHADOW REPLAY")
    assert "## CURRENT-DAY SHADOW VALIDATION (separate - not merged with the replay)" in md
    assert "Historical replay: 2026-10-05..2026-10-05" in md
    assert "| Strategy | Days | Signals | Evaluated | Wins | Losses | Data Unavailable | Skipped | WR | PF |" in md
    summary = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert summary["strategies"]["BASELINE_S2_BUY_SELL"]["evaluated"] == 2


def test_duplicate_contexts_are_counted_once(monkeypatch):
    from src.backtest import s2_direction_research as dr
    original = dr.run_candidate
    monkeypatch.setattr(dr, "run_candidate", lambda *a, **k: original(*a, **k) * 2)
    rows = _replay()["rows"]["BASELINE_S2_BUY_SELL"]
    assert len(rows) == len(SIGNALS)


# --------------------------------------------------------------------------- runner / safety
class _FakeService:
    def __init__(self, spot):
        self._spot = spot
        self.client = SimpleNamespace(expired_expiries=self._no_plan)
        self.store = SimpleNamespace(read=self._read)
        self.orders = 0

    @staticmethod
    def _no_plan(_key):
        raise HistoricalDataError("PLAN_NOT_SUPPORTED", "Upstox Plus required", status=401)

    def _read(self, kind, name, key, expiry=None, **_):
        return PREMIUMS.get(key, pd.DataFrame())

    def read_underlying(self, name, start, end):
        d = pd.to_datetime(self._spot["timestamp"]).dt.date
        return self._spot[(d >= start) & (d <= end)].reset_index(drop=True)

    def underlying_key(self, name):
        return "NSE_INDEX|Nifty 50"


RC = SimpleNamespace(session=SimpleNamespace(square_off="15:15"),
                     option_mapping={"expiry_mode": "nearest", "strike_offset": 0})


def _patch_runner(monkeypatch, spot, service=None):
    import scripts.run_shadow_historical_validation as runner

    monkeypatch.setattr(runner, "HistoricalDataService", lambda: service or _FakeService(spot))
    rc = RC
    monkeypatch.setattr(runner, "s2_components", lambda service, name: (
        rc, PARAMS, None, {"range_start": "09:15", "range_end": "09:30"}, _signal_fn, _resolve))
    monkeypatch.setattr(runner, "build_regime_labels", lambda spot, cfg: {})
    return runner


def test_runner_clamps_to_history_and_marks_unverifiable_contracts(monkeypatch, tmp_path):
    runner = _patch_runner(monkeypatch, _spot_frame())
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    (current_dir / "cumulative.json").write_text(json.dumps({"validated_days": ["2026-10-06"], "candidates": {}}))
    s = runner.run(from_date=PREV, to_date=date(2026, 10, 6), out_dir=tmp_path / "hist", current_dir=current_dir,
                   today=date(2026, 10, 6))
    assert s["trading_days"] == [PREV.isoformat(), DAY.isoformat()]
    assert s["date_range"] == "2026-10-01..2026-10-05"
    assert "PLAN_NOT_SUPPORTED" in s["expiry_check"]["source"]
    base = s["strategies"]["BASELINE_S2_BUY_SELL"]
    assert (base["evaluated"], base["data_unavailable"]) == (0, 3)
    assert base["data_unavailable_reasons"] == {"expired_instrument_unavailable": 3}
    assert s["expiry_check"]["upstox_events"] == []
    assert (tmp_path / "hist" / "summary.md").is_file()


def test_runner_evaluates_with_official_expiry_list(monkeypatch, tmp_path):
    runner = _patch_runner(monkeypatch, _spot_frame())
    s = runner.run(from_date=DAY, to_date=DAY, out_dir=tmp_path, current_dir=tmp_path / "none",
                   expired_expiries=[date(2026, 9, 29)], today=date(2026, 10, 6))
    assert s["strategies"]["BASELINE_S2_BUY_SELL"]["evaluated"] == 2
    assert s["strategies"]["BASELINE_S2_BUY_SELL"]["net_pnl"] == pytest.approx(390.0 - 195.0)


def test_runner_days_option_selects_latest_sessions(monkeypatch, tmp_path):
    runner = _patch_runner(monkeypatch, _spot_frame())
    s = runner.run(from_date=None, to_date=DAY, days=1, out_dir=tmp_path, current_dir=tmp_path / "none",
                   today=date(2026, 10, 6))
    assert s["trading_days"] == [DAY.isoformat()]


@pytest.mark.parametrize("env", [{"LIVE_TRADING_ENABLED": "true"}, {"TRADING_MODE": "LIVE"}])
def test_runner_refuses_non_paper_mode(monkeypatch, tmp_path, env):
    runner = _patch_runner(monkeypatch, _spot_frame())
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(LiveTradingSafetyError):
        runner.run(from_date=DAY, to_date=DAY, out_dir=tmp_path, today=date(2026, 10, 6))
    assert not any(tmp_path.iterdir())


def test_runner_blocks_order_entry_points_during_replay(monkeypatch, tmp_path):
    from src.execution.order_manager import OrderManager

    runner = _patch_runner(monkeypatch, _spot_frame())
    attempts = []

    def signal_fn(frame):
        try:
            OrderManager.submit_signal(None)
        except LiveTradingSafetyError as exc:
            attempts.append(str(exc))
        return _signal_fn(frame)

    rc = SimpleNamespace(session=SimpleNamespace(square_off="15:15"))
    monkeypatch.setattr(runner, "s2_components", lambda service, name: (
        rc, PARAMS, None, {"range_start": "09:15", "range_end": "09:30"}, signal_fn, _resolve))
    runner.run(from_date=DAY, to_date=DAY, out_dir=tmp_path, current_dir=tmp_path / "none", today=date(2026, 10, 6))
    assert attempts and all("SHADOW_VALIDATION_ORDER_BLOCKED" in a for a in attempts)


# --------------------------------------------------------------------------- expired-instrument API path
EXPIRED_CE, EXPIRED_PE = "NSE_FO|111|05-10-2026", "NSE_FO|222|05-10-2026"


def _hist_contract(key, kind):
    from src.historical_data.contracts import EXPIRED, HistoricalContract
    return HistoricalContract(instrument_key=key, underlying="NIFTY", expiry=DAY, instrument_type=kind, lot_size=65,
                              source=EXPIRED, strike=25000.0, trading_symbol=f"NIFTY 25000 {kind}", tick_size=0.05)


class _LiveResolver:
    today = None

    def expiries(self, underlying):
        raise OptionResolutionError("UNDERLYING_NOT_FOUND", "no live contracts in test")


class _FakeDownloader:
    def __init__(self, store, fail_code=None):
        self.store, self.fail_code, self.calls = store, fail_code, []

    def download_chunk(self, job, *, kind, underlying, key, start, end, source, contract):
        self.calls.append((key, start, end, source))
        if self.fail_code:
            return {"status": "FAILED", "error": {"code": self.fail_code}}
        frame = PREMIUMS["NSE_FO|OK_CE" if key == EXPIRED_CE else "NSE_FO|OK_PE"]
        written = 0 if key in self.store else len(frame)
        self.store[key] = frame
        return {"status": "OK", "written": written}


class _ExpiredApiService(_FakeService):
    def __init__(self, spot, fail_code=None, contracts_error=None):
        super().__init__(spot)
        self.stored: dict[str, pd.DataFrame] = {}
        self.client = SimpleNamespace(expired_expiries=lambda key: [date(2026, 9, 29), DAY])
        self.contract_calls = []

        def contracts(name, expiry):
            self.contract_calls.append(expiry)
            if contracts_error:
                raise HistoricalDataError(contracts_error, "nope", status=401)
            return [_hist_contract(EXPIRED_CE, "CE"), _hist_contract(EXPIRED_PE, "PE")]

        self.catalog = SimpleNamespace(expired_option_contracts=contracts)
        self.downloader = _FakeDownloader(self.stored, fail_code)

    def _read(self, kind, name, key, expiry=None, **_):
        return self.stored.get(key, pd.DataFrame())

    def resolver_factory(self, as_of):
        return _LiveResolver()

    def _downloader(self):
        return self.downloader


def _patch_expired(monkeypatch, service):
    runner = _patch_runner(monkeypatch, _spot_frame(), service)
    monkeypatch.setattr(runner, "s2_components", lambda svc, name: (
        RC, PARAMS, None, {"range_start": "09:15", "range_end": "09:30"}, _signal_fn,
        lambda sig: pytest.fail("live-master resolver must not be used when the expired API is available")))
    return runner


def test_point_in_time_resolver_uses_expired_contracts_verbatim():
    from src.historical_data.option_coverage import resolve_signal_contract

    pit = sh.PointInTimeResolver(_LiveResolver(), [date(2026, 9, 29), DAY],
                                 lambda e: [_hist_contract(EXPIRED_CE, "CE").to_option_contract(),
                                            _hist_contract(EXPIRED_PE, "PE").to_option_contract()])
    buy = resolve_signal_contract(SIGNALS[0], pit, RC)
    sell = resolve_signal_contract(SIGNALS[1], pit, RC)
    assert (buy.instrument_key, buy.expiry, buy.option_type.value) == (EXPIRED_CE, DAY, "CE")
    assert (sell.instrument_key, sell.option_type.value) == (EXPIRED_PE, "PE")
    pit.today = lambda: date(2026, 9, 30)
    assert pit.expiries("NIFTY") == [DAY]


def test_point_in_time_resolver_maps_contract_api_failure():
    def boom(expiry):
        raise HistoricalDataError("PLAN_NOT_SUPPORTED", "Upstox Plus required", status=401)

    pit = sh.PointInTimeResolver(_LiveResolver(), [DAY], boom)
    pit.today = lambda: DAY
    with pytest.raises(OptionResolutionError) as exc:
        pit.strikes("NIFTY", DAY, "CE")
    assert sh.reason_code(f"CONTRACT_UNRESOLVED:{exc.value.code}") == "historical_api_unavailable"


def test_runner_downloads_real_expired_candles_once_and_reuses_cache(monkeypatch, tmp_path):
    service = _ExpiredApiService(_spot_frame())
    runner = _patch_expired(monkeypatch, service)
    s = runner.run(from_date=DAY, to_date=DAY, out_dir=tmp_path, current_dir=tmp_path / "none",
                   today=date(2026, 10, 6))
    m = s["strategies"]["BASELINE_S2_BUY_SELL"]
    assert (m["evaluated"], m["data_unavailable"]) == (3, 0)
    assert m["net_pnl"] == pytest.approx(390.0 - 195.0)
    assert m["dte"]["0 DTE"]["evaluated"] == 3
    daily = json.loads((tmp_path / f"{DAY}.json").read_text(encoding="utf-8"))["signals"]
    assert {r["instrument_key"] for r in daily} == {EXPIRED_CE, EXPIRED_PE}
    assert {c[3] for c in service.downloader.calls} == {"EXPIRED"}
    d = s["expired_candle_downloads"]
    assert (d["contracts_downloaded"], d["contracts_from_cache"]) == (2, 0)
    assert service.contract_calls == [DAY]

    again = runner.run(from_date=DAY, to_date=DAY, out_dir=tmp_path, current_dir=tmp_path / "none",
                       today=date(2026, 10, 6))
    assert (again["expired_candle_downloads"]["contracts_downloaded"],
            again["expired_candle_downloads"]["contracts_from_cache"]) == (0, 2)
    assert again["strategies"]["BASELINE_S2_BUY_SELL"]["net_pnl"] == pytest.approx(m["net_pnl"])


@pytest.mark.parametrize("code,reason", [("NOT_FOUND", "expired_instrument_unavailable"),
                                         ("AUTH_FAILED", "historical_api_unavailable")])
def test_runner_expired_candle_failure_is_data_unavailable(monkeypatch, tmp_path, code, reason):
    runner = _patch_expired(monkeypatch, _ExpiredApiService(_spot_frame(), fail_code=code))
    s = runner.run(from_date=DAY, to_date=DAY, out_dir=tmp_path, current_dir=tmp_path / "none",
                   today=date(2026, 10, 6))
    m = s["strategies"]["BASELINE_S2_BUY_SELL"]
    assert (m["evaluated"], m["losses"], m["data_unavailable"]) == (0, 0, 3)
    assert m["data_unavailable_reasons"] == {reason: 3}


def test_runner_expired_contract_api_failure_is_data_unavailable(monkeypatch, tmp_path):
    runner = _patch_expired(monkeypatch, _ExpiredApiService(_spot_frame(), contracts_error="PLAN_NOT_SUPPORTED"))
    s = runner.run(from_date=DAY, to_date=DAY, out_dir=tmp_path, current_dir=tmp_path / "none",
                   today=date(2026, 10, 6))
    m = s["strategies"]["BASELINE_S2_BUY_SELL"]
    assert m["data_unavailable_reasons"] == {"historical_api_unavailable": 3}


def test_historical_modules_have_no_order_imports():
    for path in (Path(sh.__file__), Path("scripts/run_shadow_historical_validation.py")):
        src = path.read_text(encoding="utf-8")
        for forbidden in ("from src.broker", "from src.execution", "from src.engine", "place_order(",
                          "submit_signal("):
            assert forbidden not in src
