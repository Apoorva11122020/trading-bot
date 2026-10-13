#!/usr/bin/env python3
"""HISTORICAL SHADOW REPLAY for BASELINE_S2_BUY_SELL, SELL_ONLY and SELL_CONFIRMED (PAPER, read-only, no orders).

NIFTY candles come from data/historical/. With the Upstox Plus expired-instruments API (read-only GET) each signal is
resolved to the contract that was actually nearest on its date, and that contract's real 5-minute candles are
downloaded once and cached under data/historical/. If the expired-expiry list is unavailable, historical contracts
whose point-in-time expiry cannot be verified are DATA_UNAVAILABLE instead of substituting a later expiry.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s2_1dte_improvement import s2_components
from src.backtest import shadow_historical as sh
from src.backtest.regime_classifier import build_regime_labels
from src.historical_data.contracts import ACTIVE, EXPIRED, HistoricalContract
from src.historical_data.downloader import DownloadJob, month_chunks
from src.historical_data.models import HistoricalDataError
from src.historical_data.option_coverage import resolve_signal_contract
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "shadow_strategy_historical"
CURRENT_DIR = ROOT / "reports" / "shadow_strategy_validation"
WARMUP_DAYS = 35
CONTRACT_HISTORY_DAYS = 35


def expiry_check(service: HistoricalDataService, underlying_key: str, manual: list[date] | None,
                 today: date) -> tuple[sh.ExpiryCheck, list[dict]]:
    if manual:
        return sh.ExpiryCheck(today, frozenset(manual), "user-supplied official expiry list"), []
    with capture_upstox_events() as events:
        try:
            known = frozenset(service.client.expired_expiries(underlying_key))
            return sh.ExpiryCheck(today, known, "Upstox expired-instruments expiries API"), events
        except HistoricalDataError as exc:
            return sh.ExpiryCheck(today, None, f"expired-expiry list unavailable: {exc.code}"), events


def load_current_day(current_dir: Path) -> dict | None:
    path = current_dir / "cumulative.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run(*, from_date: date | None, to_date: date, days: int | None = None, out_dir: Path = DEFAULT_OUT,
        current_dir: Path = CURRENT_DIR, expired_expiries: list[date] | None = None, underlying: str = "NIFTY",
        today: date | None = None) -> dict:
    result, meta = build_replay(from_date=from_date, to_date=to_date, days=days, current_dir=current_dir,
                                expired_expiries=expired_expiries, underlying=underlying, today=today)
    summary = sh.build_summary(result, meta)
    paths = sh.write_reports(summary, result["rows"], out_dir, load_current_day(current_dir))
    print_summary(summary, paths)
    return summary


def build_replay(*, from_date: date | None, to_date: date, days: int | None = None,
                 current_dir: Path = CURRENT_DIR, expired_expiries: list[date] | None = None,
                 underlying: str = "NIFTY", today: date | None = None,
                 direction_lookahead: bool = False, keep_pipeline: bool = False) -> tuple[dict, dict]:
    """Replay the shadow strategies on stored data without writing any report."""
    x = prepare_replay_inputs(from_date=from_date, to_date=to_date, days=days, expired_expiries=expired_expiries,
                              underlying=underlying, today=today)
    with order_placement_blocked(), capture_upstox_events() as replay_events:
        result = sh.replay(spot=x.spot, signal_fn=x.signal_fn, resolve=sh.strict_resolver(x.resolve, x.check),
                           premium_for=x.premium_for, params=x.params, square_off=x.rc.session.square_off,
                           range_start=x.s2["range_start"], range_end=x.s2["range_end"], days=x.selected,
                           regimes=x.regimes, direction_lookahead=direction_lookahead, keep_pipeline=keep_pipeline)
    return result, replay_meta(x, replay_events, days=days, current_dir=current_dir)


def replay_meta(x: SimpleNamespace, replay_events: list[dict], *, days: int | None = None,
                current_dir: Path = CURRENT_DIR) -> dict:
    downloads, check = x.downloads, x.check
    downloads["upstox_final_errors"] = sum(e["event"] == "upstox_historical_error" for e in replay_events)
    downloads["upstox_retries"] = sum(e["event"] == "upstox_historical_retry" for e in replay_events)
    return {
        "date_range": f"{x.from_date}..{x.to_date}" + (f" (latest {days} sessions)" if days else ""),
        "excluded_days": x.excluded,
        "expiry_check": {"listing_date": check.listing_date.isoformat(), "source": check.source,
                         "known_expired_expiries": sorted(e.isoformat() for e in check.expired_expiries or ()),
                         "upstox_events": x.events,
                         "summary": (f"{check.source}; each signal uses the contract that was actually nearest on its "
                                     "date (expired contracts and candles from the Upstox expired-instruments API)"
                                     if x.use_expired_api else
                                     f"{check.source}; a historical signal is evaluated only when no earlier expiry "
                                     "can exist between the signal date and the resolved contract's expiry")},
        "contract_resolution": ("point-in-time: official expired expiries + expired option contracts (Upstox "
                                "expired-instruments API, keys used verbatim) + live instrument master"
                                if x.use_expired_api else "live instrument master only (strict expiry check)"),
        "expired_candle_downloads": downloads,
        "current_day_report": str(current_dir / "cumulative.json"),
        "sessions_available": [d.isoformat() for d in x.sessions],
    }


def prepare_replay_inputs(*, from_date: date | None, to_date: date, days: int | None = None,
                          expired_expiries: list[date] | None = None, underlying: str = "NIFTY",
                          today: date | None = None, download_active: bool = False) -> SimpleNamespace:
    """Stored spot, sessions, point-in-time resolver and a cached real-candle premium loader.

    ``download_active`` also downloads candles of not-yet-expired contracts (normal historical candle API) when the
    store does not cover the replay window; by default only expired contracts are downloaded."""
    assert_paper_trading_only()
    _refuse_live()
    today = today or datetime.now(sh.IST).date()
    if to_date >= today:
        to_date = today - timedelta(days=1)
    if from_date is None:
        from_date = to_date - timedelta(days=120 if days else 30)
    name = underlying.upper()
    service = HistoricalDataService()
    rc, params, _session, s2, signal_fn, resolve = s2_components(service, name)
    spot = service.read_underlying(name, from_date - timedelta(days=WARMUP_DAYS), to_date)
    if spot.empty:
        raise SystemExit(f"DATA_UNAVAILABLE: no stored {name} candles {from_date}..{to_date}")
    sessions, excluded = sh.complete_sessions(spot, from_date, to_date, params.interval)
    selected = sh.select_days(sessions, days)
    check, events = expiry_check(service, service.underlying_key(name), expired_expiries, today)
    use_expired_api = check.expired_expiries is not None and not expired_expiries
    expired_set = frozenset(check.expired_expiries or ()) if use_expired_api else frozenset()
    if use_expired_api:
        pit = sh.PointInTimeResolver(
            service.resolver_factory(today), expired_set,
            lambda expiry: [c.to_option_contract() for c in service.catalog.expired_option_contracts(name, expiry)])

        def resolve(sig):
            return resolve_signal_contract(sig, pit, rc)

    downloader = None
    downloads = {"contracts_downloaded": 0, "contracts_from_cache": 0, "rows_written": 0, "failures": []}
    if download_active:
        downloads.update(active_contracts_downloaded=0, active_contracts_from_cache=0)
    cache: dict = {}
    last_session = selected[-1] if selected else to_date

    def download_contract(contract, source: str = EXPIRED) -> str | None:
        nonlocal downloader
        downloader = downloader or service._downloader()
        hist = HistoricalContract(
            instrument_key=contract.instrument_key, underlying=name, expiry=contract.expiry,
            instrument_type=contract.option_type.value, lot_size=int(contract.lot_size), source=source,
            strike=float(contract.strike), trading_symbol=contract.trading_symbol,
            freeze_quantity=contract.freeze_quantity, tick_size=contract.tick_size)
        start = contract.expiry - timedelta(days=CONTRACT_HISTORY_DAYS)
        end = contract.expiry if source == EXPIRED else min(contract.expiry, today - timedelta(days=1))
        if end < start:
            return "NO_PREMIUM_DATA"
        job = DownloadJob(job_id="shadow-historical", underlyings=[name], from_date=start, to_date=end)
        for lo, hi in month_chunks(start, end):
            try:
                record = downloader.download_chunk(job, kind="options", underlying=name, key=contract.instrument_key,
                                                   start=lo, end=hi, source=source, contract=hist)
            except HistoricalDataError as exc:
                record = {"status": "FAILED", "error": exc.as_dict()}
            if record.get("status") == "FAILED":
                code = (record.get("error") or {}).get("code", "DOWNLOAD_FAILED")
                downloads["failures"].append({"instrument_key": contract.instrument_key, "code": code})
                return f"PREMIUM_API_ERROR:{code}"
            downloads["rows_written"] += int(record.get("written") or 0)
        downloads["contracts_downloaded" if source == EXPIRED else "active_contracts_downloaded"] += 1
        return None

    def premium_for(contract):
        key = contract.instrument_key
        if key not in cache:
            frame = service.store.read("options", name, key, expiry=contract.expiry)
            expired = contract.expiry in expired_set
            if expired and (frame.empty or not (frame["timestamp"].dt.date == contract.expiry).any()):
                failed = download_contract(contract, EXPIRED)
                if failed:
                    cache[key] = failed
                    return failed
                frame = service.store.read("options", name, key, expiry=contract.expiry)
            elif expired:
                downloads["contracts_from_cache"] += 1
            elif download_active:
                needed = min(contract.expiry, last_session)
                if frame.empty or not (frame["timestamp"].dt.date >= needed).any():
                    failed = download_contract(contract, ACTIVE)
                    if failed:
                        cache[key] = failed
                        return failed
                    frame = service.store.read("options", name, key, expiry=contract.expiry)
                else:
                    downloads["active_contracts_from_cache"] += 1
            if frame.empty:
                cache[key] = sh.EXPIRED_CANDLES_EMPTY if expired else "NO_PREMIUM_DATA"
            else:
                cache[key] = frame[frame["timestamp"].dt.date <= to_date].reset_index(drop=True)
        return cache[key]

    return SimpleNamespace(service=service, name=name, today=today, from_date=from_date, to_date=to_date, rc=rc,
                           params=params, s2=s2, signal_fn=signal_fn, resolve=resolve, spot=spot, sessions=sessions,
                           excluded=excluded, selected=selected, check=check, events=events,
                           use_expired_api=use_expired_api, expired_set=expired_set,
                           pit=pit if use_expired_api else None, premium_for=premium_for, downloads=downloads,
                           cache=cache, regimes=build_regime_labels(spot, s2))


def print_summary(s: dict, paths: dict) -> None:
    print(f"{sh.HEADING} | {s['date_range']} | PAPER | no orders")
    print(f"  Trading days: {len(s['trading_days'])} | contract check: {s['expiry_check']['source']}")
    d = s.get("expired_candle_downloads") or {}
    print(f"  Expired contracts: downloaded {d.get('contracts_downloaded', 0)}, from cache "
          f"{d.get('contracts_from_cache', 0)}, rows written {d.get('rows_written', 0)}, failures "
          f"{len(d.get('failures') or [])}, Upstox errors {d.get('upstox_final_errors', 0)}, retries "
          f"{d.get('upstox_retries', 0)}")
    for n, m in s["strategies"].items():
        print(f"  {n}: signals {m['signals']}, evaluated {m['evaluated']}, data unavailable {m['data_unavailable']}, "
              f"skipped {m['skipped']}, WR {sh._pct(m['win_rate'])}, PF {sh._pf_s(m['profit_factor'])}, "
              f"P&L Rs {sh._rs(m['net_pnl'])}, max DD Rs {sh._rs(m['max_drawdown'])}, "
              f"coverage {sh._pct(m['data_coverage'])}")
    b = s["best"]
    print(f"  Best by PF: {b['by_profit_factor']}, by P&L: {b['by_pnl']}, by drawdown: {b['by_drawdown']}")
    lookahead = ", ".join("%s=%s" % (n, "PASSED" if v["passed"] else "FAILED") for n, v in s["lookahead"].items())
    print(f"  Lookahead: {lookahead}")
    print(f"  Summary: {paths['md']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical shadow replay (read-only, PAPER)")
    parser.add_argument("--from-date", type=date.fromisoformat, default=None)
    parser.add_argument("--to-date", type=date.fromisoformat, default=date(2026, 10, 1))
    parser.add_argument("--days", type=int, default=None, help="latest N complete stored sessions")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expired-expiries", default="",
                        help="comma-separated COMPLETE official expired expiry dates for the range (YYYY-MM-DD)")
    args = parser.parse_args()
    from_date = args.from_date if args.from_date or args.days else date(2026, 9, 1)
    manual = [date.fromisoformat(x.strip()) for x in args.expired_expiries.split(",") if x.strip()] or None
    run(from_date=from_date, to_date=args.to_date, days=args.days, out_dir=args.out_dir, expired_expiries=manual)


if __name__ == "__main__":
    main()
