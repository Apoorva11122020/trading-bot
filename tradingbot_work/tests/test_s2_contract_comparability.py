"""S2 research methodology: look-ahead-free premium ATR, DTE classification, real-data guard, sample separation."""
from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backtest import contract_comparability as cc
from src.backtest import live_vs_historical as lvh
from src.backtest.current_day_s2 import LevelParams, option_levels
from src.backtest.options_engine import OptionsResearchEngine
from src.backtest.research_strategies import ResearchSignal
from src.config.loader import load_strategy_config
from src.config.research_loader import load_research_config
from src.historical_data.models import HistoricalDataError
from src.options.models import OptionResolutionError
from tests.options_support import contracts

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]
PARAMS = LevelParams(entry_timing="next_open", stop_new_entries="14:30", atr_period=3,
                     stop_atr_multiplier=1.5, target_r=2.0, slippage_bps=0.0)
CONTRACT = SimpleNamespace(instrument_key="NSE_FO|X", trading_symbol="NIFTY 22600 CE 06 OCT 26", tick_size=0.05,
                           expiry=date(2026, 10, 6))


@pytest.fixture
def configs():
    return (load_research_config(ROOT / "config" / "strategy_research.yaml"),
            load_strategy_config(ROOT / "config" / "strategy.yaml"))


def at(hh, mm, day=5):
    return pd.Timestamp(datetime(2026, 10, day, hh, mm, tzinfo=IST))


def premium(bars):
    return pd.DataFrame([{"timestamp": at(h, m), "open": o, "high": hi, "low": lo, "close": c, "volume": 1.0}
                         for h, m, o, hi, lo, c in bars])


def signal(hh=9, mm=30, direction="BUY", day=5):
    return ResearchSignal(timestamp=at(hh, mm, day).to_pydatetime(), direction=direction, strategy_id="S2",
                          underlying="NIFTY", spot=22600.0, conditions={"above": 22590.0})


BASE = [(9, 15, 100, 102, 98, 100), (9, 20, 100, 102, 98, 100), (9, 25, 100, 102, 98, 100),
        (9, 30, 100, 102, 98, 100)]


# ------------------------------------------------------- 1. no look-ahead in premium ATR
def test_option_levels_premium_atr_ignores_entry_candle():
    narrow = option_levels(signal(), CONTRACT, premium(BASE + [(9, 35, 100, 101, 99, 100)]), PARAMS)
    wide = option_levels(signal(), CONTRACT, premium(BASE + [(9, 35, 100, 180, 20, 150)]), PARAMS)
    assert narrow["premium_atr"] == pytest.approx(wide["premium_atr"]) == pytest.approx(4.0)
    assert narrow["stop_loss"] == wide["stop_loss"] and narrow["target"] == wide["target"]
    assert narrow["entry"] == wide["entry"] == 100


def _engine_inputs(flat_premium: bool = False):
    """Flat opening range, then a steady upside breakout; one premium path for every test contract."""
    start = datetime(2026, 10, 1, 9, 15, tzinfo=IST)
    under, prem = [], []
    price = 23000.0
    for i in range(75):
        t = start + timedelta(minutes=5 * i)
        price += 6 if i >= 3 else 0
        under.append({"timestamp": t, "symbol": "NIFTY", "open": price, "high": price + 5, "low": price - 5,
                      "close": price, "volume": 0})
        p = 100.0 if flat_premium else 100.0 + (i % 4)
        spread = 0.0 if flat_premium else 2.0
        prem.append({"timestamp": t, "open": p, "high": p + spread, "low": p - spread,
                     "close": p + (0.0 if flat_premium else 1.0), "volume": 100, "open_interest": 1000})
    frame = pd.DataFrame(prem)
    return pd.DataFrame(under), {c.instrument_key: frame.assign(instrument_key=c.instrument_key) for c in contracts()}


def _run(configs, under, prem):
    research, strategy = configs
    return OptionsResearchEngine(research, strategy, contracts(), prem, run_id="la").run(
        under, "NIFTY", "S2", date(2026, 10, 1), date(2026, 10, 1)).as_dict()


def test_engine_premium_atr_ignores_entry_candle(configs):
    under, prem = _engine_inputs()
    base = _run(configs, under, prem)
    assert base["trades"], "fixture must produce at least one trade"
    first = base["trades"][0]
    entry_ts = pd.Timestamp(first["entry_time"])
    mutated = {}
    for key, frame in prem.items():
        frame = frame.copy()
        hit = pd.to_datetime(frame["timestamp"]) == entry_ts
        assert hit.any()
        frame.loc[hit, "high"] = frame.loc[hit, "open"] + 500.0
        frame.loc[hit, "low"] = 1.0
        mutated[key] = frame
    again = _run(configs, under, mutated)
    second = next(t for t in again["trades"] if t["entry_time"] == first["entry_time"])
    assert second["entry_price"] == first["entry_price"]
    assert second["stop_loss"] == first["stop_loss"] and second["target"] == first["target"]


def test_engine_skips_zero_premium_atr_instead_of_crashing(configs):
    under, prem = _engine_inputs(flat_premium=True)
    result = _run(configs, under, prem)
    assert result["trades"] == []
    assert any(s["reason"] == "PREMIUM_ATR_ZERO" for s in result["skipped"])


# ------------------------------------------------------------ 2. DTE classification
def test_calendar_and_trading_dte():
    assert cc.calendar_dte(date(2026, 10, 6), date(2026, 10, 5)) == 1
    assert cc.trading_dte(date(2026, 10, 6), date(2026, 10, 5)) == 1
    assert cc.calendar_dte(date(2026, 10, 6), date(2026, 10, 1)) == 5
    assert cc.trading_dte(date(2026, 10, 6), date(2026, 10, 1), holidays=[date(2026, 10, 2)]) == 2
    assert cc.calendar_dte(date(2026, 10, 6), date(2026, 10, 6)) == 0
    with pytest.raises(ValueError):
        cc.calendar_dte(date(2026, 10, 6), date(2026, 10, 7))


@pytest.mark.parametrize("dte,label", [(0, "0 DTE"), (1, "1 DTE"), (2, "2-4 DTE"), (4, "2-4 DTE"), (5, "5-7 DTE"),
                                       (7, "5-7 DTE"), (8, "8-14 DTE"), (14, "8-14 DTE"), (15, "15-21 DTE"),
                                       (21, "15-21 DTE"), (22, "22+ DTE"), (40, "22+ DTE"), (None, None)])
def test_dte_buckets(dte, label):
    assert cc.dte_bucket(dte) == label


def test_only_live_dte_is_comparable():
    assert cc.is_comparable(1) and not cc.is_comparable(0) and not cc.is_comparable(2)
    assert not cc.is_comparable(None)
    assert cc.LIVE_REFERENCE["dte"] == 1 and cc.LIVE_REFERENCE["expiry"] == "2026-10-06"


# ------------------------------------------------------------ 3. no fabricated data
def test_missing_premium_stays_unresolved_and_unavailable():
    rows = lvh.signal_outcomes([signal()], lambda s: CONTRACT, lambda c: pd.DataFrame(), PARAMS, "15:15", "t")
    assert rows[0]["status"] == lvh.UNRESOLVED and rows[0]["entry_price"] is None if "entry_price" in rows[0] \
        else rows[0]["status"] == lvh.UNRESOLVED
    cc.annotate(rows, {}, "t")
    assert cc.separate(rows)["unavailable"] == rows
    assert cc.verify_real_data(rows) == 0
    assert cc.sample_report(rows)["status"] == "NO_REAL_DATA"


def test_verify_real_data_rejects_unbacked_levels():
    base = {"signal_time": "x", "contract": "c", "entry_price": 100.0}
    with pytest.raises(cc.FabricatedDataError):
        cc.verify_real_data([{**base, "premium_source": None, "premium_candles": 0}])
    with pytest.raises(cc.FabricatedDataError):
        cc.verify_real_data([{**base, "premium_source": "SYNTHETIC", "premium_candles": 50}])
    with pytest.raises(cc.FabricatedDataError):
        cc.verify_real_data([{**base, "premium_source": "UPSTOX_V3_INTRADAY", "premium_candles": 0}])
    assert cc.verify_real_data([{**base, "premium_source": "STORED_UPSTOX_V3+UPSTOX_V3_INTRADAY",
                                 "premium_candles": 80}]) == 1


class _NoPlanClient:
    def expired_expiries(self, key):
        raise HistoricalDataError("PLAN_NOT_SUPPORTED", "plan", status=401)


def test_expired_api_unavailable_is_reported_not_substituted():
    probe = cc.probe_expired_access(_NoPlanClient(), "NSE_INDEX|Nifty 50")
    assert probe == {"available": False, "code": "PLAN_NOT_SUPPORTED", "http_status": 401, "expiries": []}


class _ExpiredClient:
    def __init__(self):
        self.calls = []

    def expired_option_contracts(self, key, expiry):
        self.calls.append(expiry)
        return [{"instrument_key": f"NSE_FO|{t}{s}", "trading_symbol": f"NIFTY {s} {t}", "strike_price": s,
                 "instrument_type": t, "lot_size": 65, "tick_size": 0.05}
                for s in (22500, 22550, 22600, 22650) for t in ("CE", "PE")]


def test_expired_resolver_picks_live_equivalent_nearest_atm():
    client = _ExpiredClient()
    resolve = cc.expired_contract_resolver(client, "NIFTY", "NSE_INDEX|Nifty 50",
                                           [date(2026, 9, 29), date(2026, 9, 22), date(2026, 10, 6)])
    sig = ResearchSignal(timestamp=datetime(2026, 9, 28, 10, 0, tzinfo=IST), direction="SELL", strategy_id="S2",
                         underlying="NIFTY", spot=22590.0, conditions={})
    contract = resolve(sig)
    assert contract.expiry == date(2026, 9, 29) and contract.strike == 22600 and contract.option_type.value == "PE"
    assert cc.calendar_dte(contract.expiry, sig.timestamp.date()) == 1
    late = ResearchSignal(timestamp=datetime(2026, 10, 7, 10, 0, tzinfo=IST), direction="BUY", strategy_id="S2",
                          underlying="NIFTY", spot=22590.0, conditions={})
    with pytest.raises(OptionResolutionError):
        resolve(late)


# ------------------------------------------- 4. comparable vs non-comparable separation
def _row(expiry, day, entry=100.0, status="WIN", key="K", direction="BUY"):
    return {"signal_time": f"{day}T10:00:00+05:30", "session_date": day, "expiry": expiry, "instrument_key": key,
            "entry_price": entry, "status": status, "direction": direction, "exit_time": f"{day}T10:30:00+05:30",
            "pnl": 100.0 if status == "WIN" else -50.0, "r_multiple": 2.0 if status == "WIN" else -1.0,
            "holding_minutes": 30.0, "premium_atr_pct": 10.0, "stop_pct": 15.0, "target_pct": 30.0}


def test_separation_by_dte_and_data():
    rows = [_row("2026-10-06", "2026-10-05"), _row("2026-10-06", "2026-10-05", status="LOSS"),
            _row("2026-10-06", "2026-09-15"), _row("2026-10-13", "2026-10-05", direction="SELL"),
            _row("2026-10-06", "2026-10-06"), _row("2026-10-06", "2026-10-05", entry=None, key=None)]
    sources = {"K": ("UPSTOX_V3_INTRADAY", 75)}
    cc.annotate(rows, sources, "s")
    groups = cc.separate(rows)
    assert [r["calendar_dte"] for r in groups["comparable"]] == [1, 1]
    assert sorted(r["calendar_dte"] for r in groups["non_comparable"]) == [0, 8, 21]
    assert len(groups["unavailable"]) == 1
    rep = cc.sample_report(groups["comparable"])
    assert rep["win_rate"] == pytest.approx(0.5) and rep["dte_distribution"] == {"1": 2}
    assert rep["by_direction"]["BUY"]["signals"] == 2 and rep["by_direction"]["SELL"] == {"signals": 0}
    table = {b["dte_bucket"]: b for b in cc.dte_comparison(rows)}
    assert table["1 DTE"]["evaluated_signals"] == 2 and table["2-4 DTE"]["status"] == "NO_REAL_DATA"
    assert table["15-21 DTE"]["first_day"] == "2026-09-15"


def test_paired_expiry_comparison():
    a = [_row("2026-10-06", "2026-10-05"), {**_row("2026-10-06", "2026-10-05", status="LOSS"),
                                            "signal_time": "2026-10-05T11:00:00+05:30"}]
    b = [_row("2026-10-13", "2026-10-05", status="LOSS"), {**_row("2026-10-13", "2026-10-05", status="LOSS"),
                                                           "signal_time": "2026-10-05T11:00:00+05:30"}]
    out = cc.paired_expiry_comparison({"2026-10-06": a, "2026-10-13": b})
    assert out["common_resolved_signals"] == 2 and out["same_outcome_all_expiries"] == 1
    assert out["per_expiry"]["2026-10-06"]["wins"] == 1 and out["per_expiry"]["2026-10-13"]["wins"] == 0


# ------------------------------------------------------------------------- safety
@pytest.mark.parametrize("path", ["src/backtest/contract_comparability.py", "scripts/run_s2_contract_comparability.py"])
def test_no_order_or_parquet_write_path(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert not [m for m in imported if any(f in m for f in ("src.broker", "src.engine", "order_manager"))]
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"place_order", "submit_entry", "cancel_order", "append_only_write", "write_frame",
                        "save_live_signal", "resolve_live_signal"}
