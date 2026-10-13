from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_clean_historical_backtest as clean_script
from scripts import run_s9_futures_volume as script
from src.historical_data import nifty_futures as nf
from src.historical_data import service as hist_service
from src.historical_data import storage
from src.historical_data import upstox_6m as u6
from src.historical_data.contracts import ACTIVE, EXPIRED, HistoricalContract
from src.historical_data.models import HistoricalCandle, HistoricalDataError
from src.research import s9_futures_volume as s9f
from src.research import s9_orb_vwap_option_momentum as s9
from src.safety import live_gate
from tests.test_s4_six_month_research import EXPIRIES, make_store

IST = nf.IST
STEP = pd.Timedelta(minutes=5)
TOKEN = "TEST-TOKEN-VALUE-NEVER-WRITTEN-0001"
APR = [date(2026, 4, d) for d in (1, 2, 3, 27, 28, 29, 30)]
E_APR, E_MAY, E_JUN, E_JUL = date(2026, 4, 28), date(2026, 5, 26), date(2026, 6, 30), date(2026, 7, 28)


def _boom(*_a, **_k):
    raise AssertionError("S9.1 must read only the 6-month store")


def fut(expiry: date, source: str = EXPIRED, key: str | None = None) -> HistoricalContract:
    return HistoricalContract(instrument_key=key or f"NSE_FO|FUT{expiry:%m%d}", underlying="NIFTY", expiry=expiry,
                              instrument_type="FUT", lot_size=65, source=source, strike=None,
                              trading_symbol=f"NIFTY FUT {expiry:%d %b %y}".upper(), tick_size=0.1)


def at(day: date, hhmm: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time.fromisoformat(hhmm)), tz=IST)


def bars(days, base: float = 24000.0, seed: int = 1, volume: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, price = [], base
    for d in days:
        for i in range(75):
            o = price
            c = o + rng.normal(0, 4)
            rows.append({"timestamp": at(d, "09:15") + i * STEP, "open": o, "high": max(o, c) + abs(rng.normal(0, 2)),
                         "low": min(o, c) - abs(rng.normal(0, 2)), "close": c,
                         "volume": float(rng.integers(500, 3000)) if volume else 0.0, "open_interest": 1e5})
            price = c
    return pd.DataFrame(rows)


def as_futures(spot: pd.DataFrame, basis: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = spot[["timestamp", "open", "high", "low", "close"]].copy()
    for col in ("open", "high", "low", "close"):
        f[col] = f[col] + basis
    f["volume"] = rng.integers(500, 3000, len(f)).astype(float)
    f["open_interest"] = 1e5
    return f


# ------------------------------------------------------------------ fakes (no network)
class FakeClient:
    def __init__(self, data: dict[str, pd.DataFrame], fut_rows: dict[date, list] | None = None) -> None:
        self.data, self.fut_rows, self.calls, self.requests_made = data, fut_rows or {}, [], 0

    def _get(self, key, a, b):
        self.calls.append((key, a, b))
        self.requests_made += 1
        f = self.data.get(key)
        if f is None:
            raise HistoricalDataError("NOT_FOUND", "no data", status=404)
        d = f["timestamp"].dt.date
        sub = f[(d >= a) & (d <= b)]
        return [HistoricalCandle(timestamp=r.timestamp.to_pydatetime(), instrument_key=key, open=r.open, high=r.high,
                                 low=r.low, close=r.close, volume=r.volume, open_interest=r.open_interest)
                for r in sub.itertuples()]

    def candles(self, key, a, b, *, interval=5, unit="minutes"):
        return self._get(key, a, b)

    def expired_candles(self, key, a, b, *, interval=5):
        return self._get(key, a, b)

    def expired_future_contracts(self, key, expiry):
        v = self.fut_rows.get(expiry, [])
        if isinstance(v, Exception):
            raise v
        return v


class FakeCatalog:
    def __init__(self, client, active, expired_expiries) -> None:
        self.client, self._active, self._exp = client, active, expired_expiries

    def active_contracts(self):
        return self._active

    def expired_expiries(self, name):
        return self._exp

    def underlying_key(self, name):
        return "NSE_INDEX|Nifty 50"


def fut_row(c: HistoricalContract, **kw) -> dict:
    return {"instrument_key": c.instrument_key, "underlying_symbol": "NIFTY", "expiry": c.expiry.isoformat(),
            "instrument_type": "FUT", "lot_size": 65, "trading_symbol": c.trading_symbol, "segment": "NSE_FO",
            "tick_size": 0.1, **kw}


def world():
    spot = bars(APR, volume=False)
    apr, may, jun = fut(E_APR), fut(E_MAY, ACTIVE), fut(E_JUN, ACTIVE)
    data = {apr.instrument_key: as_futures(spot, 40.0, 2), may.instrument_key: as_futures(spot, 120.0, 3),
            jun.instrument_key: as_futures(spot, 200.0, 4)}
    bad = data[apr.instrument_key]
    i = bad.index[bad["timestamp"] == at(date(2026, 4, 2), "10:00")][0]
    bad.loc[i, "high"] = bad.loc[i, "low"] - 1.0
    rows = {date(2026, 4, 7): HistoricalDataError("NOT_FOUND", "weekly expiry has no future", status=404),
            E_APR: [fut_row(apr), fut_row(fut(E_APR, key="BSE_FO|X1"), segment="BSE_FO"),
                    fut_row(fut(E_APR, key="NSE_FO|OPT1"), instrument_type="CE", strike_price=24000)]}
    client = FakeClient(data, rows)
    nifty_ce = HistoricalContract(instrument_key="NSE_FO|CE1", underlying="NIFTY", expiry=E_MAY, instrument_type="CE",
                                  lot_size=65, source=ACTIVE, strike=24000.0)
    bank = HistoricalContract(instrument_key="NSE_FO|BNF", underlying="BANKNIFTY", expiry=E_MAY,
                              instrument_type="FUT", lot_size=30, source=ACTIVE)
    catalog = FakeCatalog(client, [may, jun, fut(E_JUL, ACTIVE), nifty_ce, bank], [date(2026, 4, 7), E_APR])
    return SimpleNamespace(spot=spot, data=data, client=client, catalog=catalog, apr=apr, may=may, jun=jun)


@pytest.fixture()
def downloaded(tmp_path):
    w = world()
    root = tmp_path / "upstox_6m"
    q = nf.run_futures_download(client=w.client, catalog=w.catalog, root=root, start=APR[0], end=APR[-1],
                                spot=w.spot, secrets=[TOKEN])
    return SimpleNamespace(w=w, root=root, q=q, layout=nf.FuturesLayout(root))


# ------------------------------------------------------------------ 1-3 discovery, validity, expiry ordering
def test_instrument_discovery_from_metadata_only():
    w = world()
    d = nf.discover_futures(w.catalog, "NIFTY", APR[0], APR[-1])
    assert [c.expiry for c in d.contracts] == [E_APR, E_MAY, E_JUN]
    assert {c.source for c in d.contracts} == {EXPIRED, ACTIVE}
    assert {r["reason"] for r in d.rejected} == {"WRONG_SEGMENT", "NOT_FUTURE"}
    assert any(i["scope"] == "EXPIRED_FUTURE_CONTRACTS" and i["code"] == "NOT_FOUND" for i in d.issues)
    assert all(c.underlying == "NIFTY" and c.instrument_type == "FUT" and c.strike is None for c in d.contracts)


def test_contract_validity_rules():
    assert nf.genuine_future(fut(E_APR), "NIFTY") is None
    assert nf.genuine_future(HistoricalContract("NSE_FO|1", "NIFTY", E_APR, "CE", 65, EXPIRED, 24000.0), "NIFTY") \
        == "NOT_FUTURE"
    assert nf.genuine_future(HistoricalContract("NSE_FO|2", "BANKNIFTY", E_APR, "FUT", 30, EXPIRED), "NIFTY") \
        == "WRONG_UNDERLYING"
    assert nf.genuine_future(fut(E_APR, key="NSE_INDEX|Nifty 50"), "NIFTY") == "WRONG_SEGMENT"
    assert nf.genuine_future(HistoricalContract("NSE_FO|3", "NIFTY", E_APR, "FUT", 0, EXPIRED), "NIFTY") \
        == "NO_LOT_SIZE"


def test_expiry_ordering_and_download_windows():
    w = world()
    d = nf.discover_futures(w.catalog, "NIFTY", APR[0], APR[-1])
    assert [c.expiry for c in d.contracts] == sorted(c.expiry for c in d.contracts)
    assert nf.contract_window(fut(E_APR), APR[0], APR[-1]) == (APR[0], E_APR)
    assert nf.contract_window(fut(E_JUL), APR[0], APR[-1]) == (date(2026, 4, 19), APR[-1])
    assert nf.contract_window(fut(E_JUN), APR[0], APR[-1]) == (APR[0], APR[-1])
    assert nf.contract_window(fut(date(2026, 3, 31)), APR[0], APR[-1]) is None


# ------------------------------------------------------------------ 4-6 candle validation
def test_timestamp_normalization():
    f = bars([APR[0]])
    utc = f.assign(timestamp=f["timestamp"].dt.tz_convert("UTC"))
    clean, invalid, _ = nf.validate_futures(utc)
    assert len(clean) == 75 and invalid.empty and str(clean["timestamp"].dt.tz) == "Asia/Kolkata"
    naive = f.assign(timestamp=f["timestamp"].dt.tz_localize(None))
    clean, invalid, info = nf.validate_futures(naive)
    assert clean.empty and len(invalid) == 75 and info["issues"]["TIMEZONE_NAIVE"] == 75


def test_ohlc_validity_never_repaired():
    f = bars([APR[0]])
    f.loc[3, "high"] = f.loc[3, "low"] - 1
    f.loc[4, "close"] = np.inf
    f.loc[5, "volume"] = -1
    f.loc[6, "timestamp"] = f.loc[6, "timestamp"] + pd.Timedelta(minutes=2)
    f.loc[7, "timestamp"] = at(APR[0], "15:30")
    clean, invalid, info = nf.validate_futures(f, expiry=APR[0])
    assert len(clean) == 70 and len(invalid) == 5
    reasons = " ".join(invalid["reason"])
    for code in ("HIGH_BELOW_LOW", "NON_FINITE_PRICE", "NEGATIVE_VOLUME", "OFF_GRID", "OUTSIDE_SESSION"):
        assert code in reasons
    assert float(invalid["high"].iloc[0]) == float(f.loc[3, "high"])
    late = bars([APR[1]])
    assert nf.validate_futures(late, expiry=APR[0])[2]["issues"]["AFTER_EXPIRY"] == 75


def test_duplicate_detection():
    f = bars([APR[0]])
    exact = pd.concat([f, f.iloc[[10]]], ignore_index=True)
    clean, invalid, _ = nf.validate_futures(exact)
    assert len(clean) == 75 and list(invalid["reason"]) == ["DUPLICATE_EXACT"]
    conflict = pd.concat([f, f.iloc[[10]].assign(close=f.loc[10, "close"] + 1)], ignore_index=True)
    clean, invalid, _ = nf.validate_futures(conflict)
    assert len(clean) == 74 and list(invalid["reason"]) == ["DUPLICATE_CONFLICT"] * 2
    assert nf.file_checks(clean)["ok"] and not nf.file_checks(exact)["ok"]


# ------------------------------------------------------------------ 7 volume
def test_volume_validation():
    v = pd.Series(np.arange(1, 1001, dtype=float))
    s = nf.volume_stats(v)
    assert s["zero"] == 0 and s["total"] == v.sum() and s["median"] == v.median() and "p95" in s["percentiles"]
    assert nf.volume_verdict(s) == "OK"
    assert nf.volume_verdict(nf.volume_stats(pd.Series(np.zeros(100)))) == nf.FUTURES_VOLUME_UNAVAILABLE
    some = v.copy()
    some[:50] = 0
    assert nf.volume_verdict(nf.volume_stats(some)) == "DEGRADED"
    most = v.copy()
    most[:600] = 0
    assert nf.volume_verdict(nf.volume_stats(most)) == nf.FUTURES_VOLUME_UNAVAILABLE


# ------------------------------------------------------------------ 8-10 VWAP
def test_vwap_calculation():
    f = pd.DataFrame({"timestamp": [at(APR[0], "09:15"), at(APR[0], "09:20"), at(APR[0], "09:25")],
                      "open": [100.0, 102.0, 101.0], "high": [103.0, 104.0, 102.0], "low": [99.0, 101.0, 98.0],
                      "close": [102.0, 103.0, 99.0], "volume": [10.0, 30.0, 0.0]})
    v = nf.session_vwap(f)
    tp = (f["high"] + f["low"] + f["close"]) / 3
    assert v["session_vwap"].iloc[0] == pytest.approx(tp[0])
    assert v["session_vwap"].iloc[1] == pytest.approx((tp[0] * 10 + tp[1] * 30) / 40)
    assert v["session_vwap"].iloc[2] == pytest.approx(v["session_vwap"].iloc[1])
    zero = nf.session_vwap(f.assign(volume=0.0))
    assert zero["session_vwap"].isna().all()


def test_session_vwap_resets_each_day():
    f = bars([APR[0], APR[1]])
    v = nf.session_vwap(f)
    first = v[v["timestamp"] == at(APR[1], "09:15")].iloc[0]
    assert first["session_vwap"] == pytest.approx(first["typical_price"])
    assert first["cumulative_volume"] == first["volume"]
    assert nf.vwap_checks(f)["resets_each_session"]


def test_no_future_vwap_contamination():
    f = bars([APR[0], APR[1]])
    full = nf.session_vwap(f)
    cut = at(APR[0], "11:00")
    trunc = nf.session_vwap(f[f["timestamp"] <= cut])
    assert np.allclose(full[full["timestamp"] <= cut]["session_vwap"], trunc["session_vwap"])
    later = f.copy()
    later.loc[later["timestamp"] > cut, ["high", "low", "close", "volume"]] *= 3.0
    again = nf.session_vwap(later)
    assert np.allclose(again[again["timestamp"] <= cut]["session_vwap"], trunc["session_vwap"])
    chk = nf.vwap_checks(f)
    assert chk["passed"] and chk["truncation_mismatches"] == 0 and chk["within_session_range"]


# ------------------------------------------------------------------ 11 volume average
def test_twenty_candle_volume_average_uses_only_previous_candles():
    f = bars([APR[0]])
    feat = s9.nifty_features(f)
    v = f["volume"].to_numpy()
    assert feat["volume_ratio_index"].iloc[20] == pytest.approx(v[20] / v[:20].mean())
    assert np.isnan(feat["volume_ratio_index"].iloc[19])
    spiked = f.copy()
    spiked.loc[21:, "volume"] = 1e9
    assert s9.nifty_features(spiked)["volume_ratio_index"].iloc[20] == pytest.approx(v[20] / v[:20].mean())


# ------------------------------------------------------------------ 12-13 causal selection, rollover
def _frames(days, contracts, basis=(40.0, 120.0, 200.0)):
    spot = bars(days, volume=False)
    return spot, {c.instrument_key: as_futures(spot, b, i) for i, (c, b) in enumerate(zip(contracts, basis))}


def test_causal_contract_selection():
    cs = [fut(E_APR), fut(E_MAY), fut(E_JUN)]
    spot, frames = _frames(APR, cs)
    sel = nf.select_contracts(cs, frames, APR)
    by = {r["session_date"]: r for r in sel}
    assert by["2026-04-28"]["instrument_key"] == cs[0].instrument_key
    assert by["2026-04-29"]["instrument_key"] == cs[1].instrument_key
    assert all(r["status"] == nf.SELECTED for r in sel)
    assert nf.selection_checks(sel, cs, frames, APR)["passed"]
    gap = dict(frames)
    f = gap[cs[0].instrument_key]
    gap[cs[0].instrument_key] = f[f["timestamp"].dt.date != date(2026, 4, 27)]
    r = {x["session_date"]: x for x in nf.select_contracts(cs, gap, APR)}["2026-04-27"]
    assert r["status"] == nf.DATA_UNAVAILABLE and r["reason"] == "NO_CANDLES_ON_SESSION"
    assert r["instrument_key"] == cs[0].instrument_key
    late = dict(frames)
    g = late[cs[0].instrument_key]
    late[cs[0].instrument_key] = g[g["timestamp"].dt.date >= date(2026, 4, 3)]
    r = {x["session_date"]: x for x in nf.select_contracts(cs, late, APR)}["2026-04-01"]
    assert r["status"] == nf.DATA_UNAVAILABLE and r["reason"] == "NOT_YET_TRADING"
    none = nf.select_contracts(cs[1:], {k: v for k, v in frames.items() if k != cs[1].instrument_key}, [APR[0]])
    assert none[0]["reason"] == "CONTRACT_DATA_MISSING"


def test_rollover_handling_keeps_contracts_separate():
    cs = [fut(E_APR), fut(E_MAY), fut(E_JUN)]
    spot, frames = _frames(APR, cs)
    sel = nf.select_contracts(cs, frames, APR)
    rows, summary = nf.rollover_analysis(cs, frames, sel, spot, APR)
    assert summary["rollovers"] == 1 and rows[0]["roll_date"] == "2026-04-29"
    assert rows[0]["basis_jump"] == pytest.approx(80.0, abs=1e-6)
    assert summary["active_contracts_per_day"] == {"3": len(APR)}
    fb = nf.front_month_bars(sel, frames)
    on_roll = fb[fb["timestamp"].dt.date == date(2026, 4, 29)]
    raw = frames[cs[1].instrument_key]
    assert set(on_roll["instrument_key"]) == {cs[1].instrument_key}
    assert np.allclose(on_roll["close"].to_numpy(), raw[raw["timestamp"].dt.date == date(2026, 4, 29)]["close"])


# ------------------------------------------------------------------ 14-15 alignment, missing data
def test_spot_futures_alignment():
    cs = [fut(E_APR), fut(E_MAY), fut(E_JUN)]
    spot, frames = _frames(APR, cs)
    sel = nf.select_contracts(cs, frames, APR)
    rows, s = nf.alignment(sel, nf.front_month_bars(sel, frames), spot, APR)
    assert s["passed"] and s["missing_futures"] == 0 and s["exact_matches"] == len(APR) * 75 and not s["forward_fill"]
    k = cs[0].instrument_key
    frames[k] = frames[k].drop(index=5)
    sel = nf.select_contracts(cs, frames, APR)
    rows, s = nf.alignment(sel, nf.front_month_bars(sel, frames), spot.drop(index=100), APR)
    assert s["missing_futures"] == 1 and s["missing_spot"] == 1
    far = {key: f.assign(close=f["close"] * 1.05) for key, f in frames.items()}
    _, s = nf.alignment(sel, nf.front_month_bars(sel, far), spot, APR)
    assert s["mismatched"] > 0 and not s["passed"]


def test_missing_data_reported_never_filled():
    c = fut(E_APR)
    f = bars(APR[:2])
    gap = f.drop(index=[3, 4, 80]).reset_index(drop=True)
    rows = nf.session_rows(c, gap, APR[:3], "FRONT_MONTH_CONTRACT")
    by = {r["trading_date"]: r for r in rows}
    assert by["2026-04-01"]["missing"] == 2 and by["2026-04-01"]["missing_times"] == "09:30 09:35"
    assert by["2026-04-02"]["missing"] == 1
    assert by["2026-04-03"]["reason"] == "NO_CANDLES_ON_SESSION"
    assert len(gap) == 147


# ------------------------------------------------------------------ 16 no synthetic volume
def test_no_synthetic_volume_and_futures_only_volume_ratio():
    row = {"timestamp": at(APR[0], "11:15"), "open": 100.4, "high": 100.85, "low": 100.35, "close": 100.8,
           "or_complete": True, "or_high": 100.5, "or_low": 99.5, "twap": 100.0, "vwap": 100.0,
           "volume_ratio_index": float("nan"), "spot_close": 24000.0}
    assert s9.underlying_check(row, "BUY", s9f.CFG) == (False, "VOLUME_UNAVAILABLE")
    assert s9.underlying_check(dict(row, volume_ratio_index=1.1), "BUY", s9f.CFG)[1] == "LOW_VOLUME"
    assert s9.underlying_check(dict(row, volume_ratio_index=1.3), "BUY", s9f.CFG) == (True, None)
    assert s9.underlying_check(dict(row, vwap=float("nan"), volume_ratio_index=1.3), "BUY", s9f.CFG)[1] \
        == "VWAP_UNAVAILABLE"
    prem = pd.DataFrame({"timestamp": [at(APR[0], "09:15") + i * STEP for i in range(40)], "open": 100.0,
                         "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0})
    j = 24
    prem.loc[j, ["open", "high", "low", "close", "volume"]] = [100.0, 103.5, 99.8, 103.0, 1.0]
    t = pd.DatetimeIndex(prem["timestamp"])
    out = s9.option_leg(prem, t, t[j], "SELL", s9f.CFG, 0.05)
    assert out["fired"] and out.get("option_volume_ratio") is None
    assert s9.option_leg(prem, t, t[j], "SELL", s9.BASE, 0.05)["reason"] == "LOW_VOLUME"


def test_download_stores_real_candles_only(downloaded):
    d, w = downloaded, downloaded.w
    for c in (w.apr, w.may, w.jun):
        p = d.layout.contract_file(c)
        assert p.name == f"NIFTY_FUT_{c.expiry.isoformat()}.parquet" and p.is_file()
        stored = u6.read_parquet(p)
        src = w.data[c.instrument_key]
        src = src[src["timestamp"].dt.date <= c.expiry]
        merged = stored.merge(src, on="timestamp", suffixes=("", "_src"))
        assert len(merged) == len(stored) and np.allclose(merged["volume"], merged["volume_src"])
        assert set(stored["instrument_key"]) == {c.instrument_key} and set(stored["expiry"]) == {c.expiry.isoformat()}
    inv = u6._read_csv(d.layout.invalid)
    assert len(inv) == 1 and "HIGH_BELOW_LOW" in inv[0]["reason"]
    assert all(b - a <= timedelta(days=31) and a.month == b.month for _, a, b in w.client.calls)
    q = d.q
    assert q["pass"], {k: q[k] for k in ("status", "vwap", "contract_selection", "timestamp_alignment")}
    assert q["contracts_discovered"] == 3 and q["invalid_candles"] == 1 and q["volume"]["verdict"] == "OK"
    assert q["rollover"]["rollovers"] == 1 and q["missing_candles"]["front_month"] == 1
    for f in ("contracts.csv", "futures_missing_data.csv", "futures_contract_selection.csv", "data_quality.json",
              "futures_rollover.csv", "futures_spot_alignment.csv", "futures_vwap_diagnostic.csv"):
        assert (d.layout.dir / f).is_file(), f
    json.loads(d.layout.quality.read_text(encoding="utf-8"))
    calls = len(w.client.calls)
    nf.run_futures_download(client=w.client, catalog=w.catalog, root=d.root, start=APR[0], end=APR[-1], spot=w.spot,
                            secrets=[TOKEN], resume=True)
    assert len(w.client.calls) == calls


# ------------------------------------------------------------------ 17 token leakage
def test_no_token_leakage(downloaded):
    d = downloaded
    assert d.q["token_leak"]["passed"]
    for p in d.layout.dir.rglob("*"):
        if p.is_file():
            assert TOKEN.encode() not in p.read_bytes(), p
    (d.layout.dir / "oops.txt").write_text(f"Bearer {TOKEN}", encoding="utf-8")
    q = nf.validate_store(root=d.root, start=APR[0], end=APR[-1], spot=d.w.spot, secrets=[TOKEN])
    assert not q["token_leak"]["passed"] and not q["pass"]


# ------------------------------------------------------------------ end-to-end on the synthetic 6-month store
def add_futures(root: Path, zero_volume: bool = False) -> None:
    lay = nf.FuturesLayout(root)
    start, end = date(2026, 4, 1), date(2026, 9, 30)
    spot = u6.load_underlying(u6.Layout(root), start, end)
    rng = np.random.default_rng(9)
    rows = []
    for e in [*EXPIRIES, date(2026, 10, 27)]:
        c = fut(e)
        lo, hi = nf.contract_window(c, start, end)
        dd = spot["timestamp"].dt.date
        sub = spot[(dd >= lo) & (dd <= hi)][["timestamp", "open", "high", "low", "close"]].copy()
        for col in ("open", "high", "low", "close"):
            sub[col] = sub[col] + 30.0
        vol = rng.integers(500, 2000, len(sub)).astype(float)
        vol[rng.random(len(sub)) < 0.15] *= 3.0
        sub["volume"] = 0.0 if zero_volume else vol
        sub["open_interest"] = 1e5
        u6.write_parquet(lay.contract_file(c), nf._attach(sub, c))
        rows.append({"instrument_key": c.instrument_key, "trading_symbol": c.trading_symbol,
                     "exchange_segment": "NSE_FO", "instrument_type": "FUT", "underlying": "NIFTY",
                     "expiry": e.isoformat(), "lot_size": 65, "tick_size": 0.1, "source": EXPIRED,
                     "candles": len(sub), "status": u6.SUCCESS})
    u6._write_csv(lay.contracts, nf.CONTRACT_COLUMNS, rows)
    u6._write_csv(lay.invalid, nf.INVALID_COLUMNS, [])


def _run(base: Path, zero_volume: bool = False):
    make_store(base / "upstox_6m")
    add_futures(base / "upstox_6m", zero_volume)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(storage.ParquetCandleStore, "__init__", _boom)
        mp.setattr(hist_service.HistoricalDataService, "__init__", _boom)
        return script.run(data_root=base / "upstox_6m", out_dir=base / "out", mutation_days=2, check_sample=5,
                          secret_values=[TOKEN])


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    before = clean_script.production_hashes()
    base = tmp_path_factory.mktemp("s91")
    return SimpleNamespace(a=_run(base), base=base, before=before)


def test_futures_replay_runs_with_clean_checks(env):
    a = env.a
    assert a["ran"] and a["futures_quality"]["pass"]
    for k, v in a["lookahead"].items():
        assert v["passed"], (k, v)
    assert a["isolation"]["passed"] and a["data_ok"]
    for r in a["records"]:
        assert r["volume_source"] == "FUTURES_VOLUME" and r["option_confirmation_rule"] == s9.RISING_PREMIUM
        if r["outcome"] in s9.EXECUTED:
            assert s9.dra._ts(r["entry_time"]) == s9.dra._ts(r["signal_time"]) + STEP
            assert r["futures_contract"] and r["spot"] != r["underlying_close"]
    assert a["production_unchanged"] is True and a["live_orders"] == 0 and a["trading_mode"] == "PAPER"
    assert clean_script.production_hashes() == env.before


def test_reports_comparison_and_terminal(env):
    out = env.base / "out"
    for n in s9f.REPORT_FILES:
        assert (out / n).is_file(), n
        t = (out / n).read_text(encoding="utf-8")
        assert TOKEN not in t and "bearer " not in t.lower()
    import csv
    with (out / "comparison_s9_proxy.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["system"] for r in rows} == {"S2", "S9_PROXY", "S9_FUTURES_VOLUME"}
    assert all("PROXY DATA" in r["data"] for r in rows if r["system"] == "S9_PROXY")
    assert all("REAL FUTURES" in r["data"] for r in rows if r["system"] == "S9_FUTURES_VOLUME")
    text = s9f.terminal_text(env.a)
    for label in ("S9.1 FUTURES DATA COMPLETE", "FUTURES DATA:", "contracts:", "candles:", "trading days:",
                  "missing:", "duplicates:", "zero volume:", "data quality: PASS", "VWAP:\nPASS",
                  "CAUSAL CONTRACT SELECTION:\nPASS", "SPOT/FUTURES ALIGNMENT:\nPASS", "TOKEN LEAK:\nPASS",
                  "S9 FUTURES REPLAY:\nRUN", "S9_FUTURES_VOLUME:", "TRAIN:", "VALIDATION:", "OOS AUGUST:",
                  "HOLDOUT SEPTEMBER:", "BUY: Trades", "SELL: Trades", "COMBINED: Trades", "STATUS:",
                  "LOOKAHEAD:\nPASS", "DATA INTEGRITY:\nPASS", "PRODUCTION CHANGED:\nNO", "LIVE ORDERS:\n0",
                  "FORWARD PAPER VALIDATION:", "not live accuracy"):
        assert label in text, label


def test_replay_not_run_when_futures_volume_unusable(tmp_path):
    a = _run(tmp_path, zero_volume=True)
    assert not a["ran"] and a["futures_quality"]["status"] == nf.FUTURES_VOLUME_UNAVAILABLE
    text = s9f.terminal_text(a)
    assert "S9 FUTURES REPLAY:\nNOT RUN" in text and nf.FUTURES_VOLUME_UNAVAILABLE in text
    assert not (tmp_path / "out" / "trade_log.csv").exists()


def test_paper_only(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    with pytest.raises(live_gate.LiveTradingSafetyError):
        script.run(data_root=Path("unused"))
