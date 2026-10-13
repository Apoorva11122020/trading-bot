#!/usr/bin/env python3
"""S2 option-contract comparability research (analysis only, PAPER, no orders, no Parquet writes).

1. Re-runs the S2 research engine with the corrected (look-ahead-free) premium ATR.
2. Re-evaluates every historical S2 signal on real option candles and classifies it by DTE.
3. Tries to evaluate historical signals on the live-equivalent nearest-expiry contract via the Upstox
   expired-instrument API; if that API is unavailable the sample is reported as unavailable.
4. Evaluates today's signals on the nearest (live) expiry and on later listed expiries with real
   intraday candles, giving a same-session DTE comparison.
5. Keeps the recorded live benchmark unchanged and writes the comparability reports.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.contract_comparability import (
    LIVE_REFERENCE,
    annotate,
    dte_comparison,
    expired_contract_resolver,
    paired_expiry_comparison,
    probe_expired_access,
    sample_report,
    separate,
    verify_real_data,
    write_reports,
)
from src.backtest.current_day_s2 import IST, LevelParams, point_in_time_signals, premium_frame, validate_intraday
from src.backtest.live_vs_historical import signal_outcomes
from src.backtest.research_strategies import generate_signals
from src.backtest.robustness import load_final_comparison, row_for
from src.backtest.strategy_comparator import compare_strategies
from src.historical_data.models import HistoricalDataError
from src.historical_data.option_coverage import entry_allowed, resolve_signal_contract
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.historical_data.upstox_client import token_source
from src.options.models import OptionType
from src.options.selection import atm_strike
from src.safety.live_gate import assert_paper_trading_only, live_trading_enabled, trading_mode
from src.session.session_manager import SessionManager

OUT_DIR = ROOT / "reports" / "strategy_research"
INDEX_KEY = "NSE_INDEX|Nifty 50"
BENCHMARK = {"signals": 40, "wins": 9, "losses": 31, "accuracy": 22.5, "profit_factor": 0.66, "net_pnl": -15470.0}
UNRESOLVED_EXITS = {"DATA_INVALID", "SKIPPED"}


def engine_summary(result: dict) -> dict:
    m = result["metrics"]
    trades = [t for t in result.get("trades", []) if t.get("exit_reason") not in UNRESOLVED_EXITS]
    return {"resolved_trades": m.get("resolved_trades"), "win_rate": m.get("win_rate"),
            "profit_factor": m.get("profit_factor"), "max_drawdown": m.get("max_drawdown"),
            "net_pnl": round(sum(float(t.get("pnl") or 0) for t in trades), 2),
            "expiries": sorted({str(t.get("expiry") or t.get("trading_symbol", "")[-9:]) for t in trades})}


def check_benchmark(path: Path) -> dict:
    if not path.exists():
        return {"source": str(path.relative_to(ROOT)), "check": "BENCHMARK REPORT MISSING", "values": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    values = {"signals": data["total"], "wins": data["wins"], "losses": data["losses"],
              "accuracy": round(data["accuracy"], 2), "profit_factor": round(data["profit_factor"], 2),
              "net_pnl": data["net_pnl"]}
    ok = all(values[k] == v for k, v in BENCHMARK.items())
    return {"source": str(path.relative_to(ROOT)), "values": values,
            "check": "UNCHANGED (matches 40 / 9 / 31 / 22.5% / PF 0.66 / -Rs 15,470)" if ok
            else f"MISMATCH vs expected {BENCHMARK}"}


def run(*, day: date, hist_from: date, hist_to: date, underlying: str = "NIFTY", out_dir: Path = OUT_DIR) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    service = HistoricalDataService()
    client = service.client
    rc, sc = service.research_config, service.strategy_config
    params = LevelParams.from_configs(rc, sc)
    session = SessionManager(sc.session)
    square_off = rc.session.square_off

    # ---------------------------------------------------------------- inputs
    hist = service.read_underlying(underlying, hist_from, hist_to)
    if hist.empty:
        raise SystemExit(f"Missing stored underlying candles {hist_from}..{hist_to}")
    hist = hist.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True).dt.tz_convert(IST)
    hist["symbol"] = underlying.upper()
    try:
        raw_today = client.intraday_candles(INDEX_KEY, interval=params.interval)
    except HistoricalDataError as exc:
        raise SystemExit(f"API_REQUEST_ERROR: {exc.code} (HTTP {exc.status})") from None
    today_candles, _ = validate_intraday(raw_today, day, params.interval)
    present = set(hist["timestamp"].dt.date) | ({day} if not today_candles.empty else set())
    holidays = [hist_from + timedelta(days=i) for i in range((day - hist_from).days)
                if (hist_from + timedelta(days=i)).weekday() < 5 and hist_from + timedelta(days=i) not in present]

    sources: dict[str, tuple[str, int]] = {}

    def stored_premium(contract):
        frame = service.store.read("options", underlying.upper(), contract.instrument_key, expiry=contract.expiry)
        frame = premium_frame(frame, [])
        sources[contract.instrument_key] = ("STORED_UPSTOX_V3", len(frame))
        return frame

    intraday_cache: dict[str, object] = {}

    def live_premium(contract):
        key = contract.instrument_key
        if key in intraday_cache:
            return intraday_cache[key]
        stored = service.store.read("options", underlying.upper(), key, expiry=contract.expiry)
        source = "STORED_UPSTOX_V3+UPSTOX_V3_INTRADAY"
        try:
            today = [c for c in client.intraday_candles(key, interval=params.interval)
                     if c.timestamp.astimezone(IST).date() == day]
            if stored is None or stored.empty:
                stored = pd.DataFrame()
                prior = client.candles(key, day - timedelta(days=10), day - timedelta(days=1), interval=params.interval)
                if prior:
                    stored = premium_frame(None, prior)
                source = "UPSTOX_V3_HISTORICAL+UPSTOX_V3_INTRADAY"
        except HistoricalDataError as exc:
            intraday_cache[key] = f"PREMIUM_API_ERROR:{exc.code}"
            return intraday_cache[key]
        frame = premium_frame(stored if stored is not None and not stored.empty else None, today)
        sources[key] = (source if today else "STORED_UPSTOX_V3", len(frame))
        intraday_cache[key] = frame
        return frame

    def resolve_nearest(sig):
        return resolve_signal_contract(sig, service.resolver_factory(sig.timestamp.date()), rc)

    # ------------------------------------------ 1. corrected engine research
    final = load_final_comparison(OUT_DIR / "final_comparison.json")
    row = row_for(final, underlying, "S2")
    eng_from, eng_to = date.fromisoformat(row["from_date"]), date.fromisoformat(row["to_date"])
    eng_candles = service.read_underlying(underlying, eng_from, eng_to)
    eng_candles["symbol"] = underlying.upper()
    results, _ = compare_strategies(
        research_config=rc, strategy_config=sc, underlying=underlying, underlying_candles=eng_candles,
        premium_history=service._premium_history(underlying, eng_from, eng_to),
        contracts=service._contracts_from_store(underlying), strategy_ids=["S2"], run_id=uuid.uuid4().hex[:12],
        master_resolver=service._research_resolver, splits={"FULL": (eng_from, eng_to)})
    after = engine_summary(next(r for r in results if r.split == "FULL").as_dict())
    prefix_path = OUT_DIR / f"{underlying.upper()}_S2_live_vs_historical_{day.isoformat()}.json"
    before = {}
    if prefix_path.exists():
        before = json.loads(prefix_path.read_text(encoding="utf-8")).get("engine", {}).get("historical", {})

    # ----------------------------- 2. historical signals, research mapping
    hist_signals = [s for s in generate_signals("S2", hist, underlying.upper(), rc)
                    if entry_allowed(s.timestamp, session, params.stop_new_entries)]
    hist_rows = annotate(signal_outcomes(hist_signals, resolve_nearest, stored_premium, params, square_off,
                                         "historical_research_mapping"), sources,
                         "historical_research_mapping", holidays)

    # ---------------------- 3. historical signals, live-equivalent contracts
    expired = probe_expired_access(client, INDEX_KEY)
    live_equiv_rows: list[dict] = []
    if expired["available"]:
        resolver = expired_contract_resolver(client, underlying, INDEX_KEY,
                                             [date.fromisoformat(e) for e in expired["expiries"]])
        cache: dict[str, object] = {}

        def expired_premium(contract):
            key = contract.instrument_key
            if key not in cache:
                try:
                    candles = client.expired_candles(key, contract.expiry - timedelta(days=10), contract.expiry,
                                                     interval=params.interval)
                except HistoricalDataError as exc:
                    cache[key] = f"EXPIRED_API_ERROR:{exc.code}"
                    return cache[key]
                cache[key] = premium_frame(None, candles)
                sources[key] = ("UPSTOX_EXPIRED_HISTORICAL", len(cache[key]))
            return cache[key]

        live_equiv_rows = annotate(signal_outcomes(hist_signals, resolver, expired_premium, params, square_off,
                                                   "historical_live_equivalent"), sources,
                                   "historical_live_equivalent", holidays)
    live_equiv = {
        "signals": len(hist_signals),
        "evaluated": sum(1 for r in live_equiv_rows if r.get("entry_price") is not None),
        "statement": (
            "Live-equivalent nearest-weekly contracts were evaluated from real expired-contract candles."
            if expired["available"] else
            f"UNAVAILABLE: Upstox rejected the expired-instrument API ({expired['code']}, HTTP "
            f"{expired['http_status']}). Historical nearest-weekly (incl. 1-DTE) option candles for "
            f"{hist_from}..{hist_to} could not be obtained, so no historical 1-DTE result exists. Nothing was "
            "substituted or synthesised."),
    }

    # ----------------------------- 4. today: nearest expiry vs later expiries
    today_signals, lookahead_ok = point_in_time_signals(today_candles, underlying, rc, session,
                                                        params.stop_new_entries)
    resolver_today = service.resolver_factory(day)
    expiries = sorted(e for e in resolver_today.expiries(underlying.upper()) if e >= day)
    monthly = next((e for e in expiries if e.month == day.month and
                    not any(x.month == e.month and x > e for x in expiries)), None)
    compare_expiries = [e for e in [expiries[0] if expiries else None,
                                    expiries[1] if len(expiries) > 1 else None, monthly] if e]
    compare_expiries = list(dict.fromkeys(compare_expiries))

    def resolver_for(expiry):
        def resolve(sig):
            resolver_today.today = lambda: day
            otype = OptionType.CE if sig.direction == "BUY" else OptionType.PE
            strike = atm_strike(resolver_today.strikes(underlying.upper(), expiry, otype), sig.spot)
            return resolver_today.resolve(underlying.upper(), expiry, strike, otype)
        return resolve

    today_by_expiry: dict[str, list[dict]] = {}
    for expiry in compare_expiries:
        label = f"today_{expiry.isoformat()}"
        rows = signal_outcomes(today_signals, resolver_for(expiry), live_premium, params, square_off, label)
        today_by_expiry[expiry.isoformat()] = annotate(rows, sources, label, holidays)

    all_rows = hist_rows + live_equiv_rows + [r for rows in today_by_expiry.values() for r in rows]
    checked = verify_real_data(all_rows)

    # -------------------------------------------------- 5. samples / reports
    groups = separate(all_rows)
    nearest = today_by_expiry.get(compare_expiries[0].isoformat(), []) if compare_expiries else []
    samples_a = {
        "historical 1-DTE (live-equivalent)": sample_report([r for r in live_equiv_rows if r.get("comparable")]),
        f"today 1-DTE ({compare_expiries[0] if compare_expiries else 'n/a'}, corrected ATR)":
            sample_report([r for r in nearest if r.get("comparable")]),
        "all comparable": sample_report(groups["comparable"]),
    }
    samples_b = {"historical research mapping": sample_report([r for r in hist_rows]),
                 **{f"today {e}": sample_report([r for r in rows if not r.get("comparable")])
                    for e, rows in today_by_expiry.items() if any(not r.get("comparable") for r in rows)},
                 "all non-comparable": sample_report(groups["non_comparable"])}
    if live_equiv_rows:
        samples_b["historical live-equivalent, DTE != 1"] = sample_report(
            [r for r in live_equiv_rows if not r.get("comparable")])
    benchmark = check_benchmark(ROOT / "reports" / "live_signal_accuracy" / f"{day.isoformat()}.json")
    paired = paired_expiry_comparison({e: rows for e, rows in today_by_expiry.items()})
    dte_table = dte_comparison(hist_rows + live_equiv_rows + [r for rows in today_by_expiry.values() for r in rows])
    findings = build_findings(before, after, samples_a, samples_b, dte_table, paired, live_equiv, benchmark,
                              hist_rows, nearest)
    report = {
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "mode": (f"analysis only - TRADING_MODE={trading_mode()}, LIVE_TRADING_ENABLED={live_trading_enabled()}, "
                 "no broker calls, no Parquet writes, S2 rules unchanged"),
        "token_source": token_source(),
        "live_reference": LIVE_REFERENCE,
        "lookahead_fix": {"notes": [
            "OptionsResearchEngine and option_levels now compute premium ATR from candles up to and including the "
            "signal candle only. For next-candle-open entries the entry candle's high/low/close are no longer used.",
            "Entry price, stop multiplier, target R, exits and contract selection are unchanged.",
            "The pre-fix engine values come from the earlier live-vs-historical report generated before the fix."]},
        "engine_research": {"window": f"{eng_from}..{eng_to}", "before": before, "after": after,
                            "contract_expiries": after.get("expiries")},
        "expired_data": expired,
        "live_equivalent_historical": live_equiv,
        "samples": {"A_comparable": samples_a, "B_non_comparable": samples_b},
        "dte_comparison": dte_table,
        "paired_today": paired,
        "benchmark": benchmark,
        "findings": findings,
        "notes": [
            "Signal-level method (same as the live tracker): every point-in-time S2 signal, next-candle option open "
            "entry, unchanged premium-ATR stop and 2R target, first touch on real option candles, EOD exit at the open "
            "of the first candle at/after square-off, same-candle SL+target = AMBIGUOUS, 1 lot, costs excluded.",
            "Comparable = calendar DTE equal to the live setup's 1 DTE. Trading DTE is also recorded; exchange "
            f"holidays are inferred from missing NIFTY sessions ({[h.isoformat() for h in holidays]}).",
            f"Real-data guard: {checked} evaluated rows verified as backed by real Upstox option candles; signals "
            "without real premium candles stay unresolved and are counted as unavailable.",
            f"Today's signals generated bar by bar from completed candles (look-ahead check "
            f"{'PASS' if lookahead_ok else 'MISMATCH'}). Later-expiry contracts use real Upstox V3 intraday candles "
            "plus prior-session V3 candles for ATR warm-up, held in memory only.",
            "The live benchmark is the immutable tracker record; today's re-evaluation with the corrected ATR is "
            "shown separately and does not replace it.",
        ],
        "rows": all_rows,
    }
    for path in write_reports(report, out_dir).values():
        print(f"Wrote {path}")
    print_summary(report)
    return report


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def build_findings(before, after, samples_a, samples_b, dte_table, paired, live_equiv, benchmark,
                   hist_rows, nearest) -> list[str]:
    out = []
    if before:
        out.append(f"Look-ahead fix moved the engine result from {before.get('resolved_trades')} trades / "
                   f"net Rs {before.get('net_pnl'):,.0f} / PF {before.get('profit_factor'):.2f} to "
                   f"{after['resolved_trades']} trades / net Rs {after['net_pnl']:,.0f} / PF {after['profit_factor']:.2f}.")
    hist_rep = samples_b["historical research mapping"]
    dist = hist_rep.get("dte_distribution", {})
    out.append(f"Historical research mapping: {hist_rep['evaluated_signals']} evaluated signals, all at DTE "
               f"{min(map(int, dist))}-{max(map(int, dist))}; win rate {_pct(hist_rep['win_rate'])}, "
               f"PF {hist_rep['profit_factor']:.2f}. None of it is comparable to the 1-DTE live setup." if dist
               else "Historical research mapping produced no evaluated signals.")
    out.append(live_equiv["statement"])
    a_today = next((v for k, v in samples_a.items() if k.startswith("today")), None)
    if a_today and a_today.get("status") == "OK":
        out.append(f"Only comparable real data: today's 1-DTE session ({a_today['evaluated_signals']} signals, 1 day) - "
                   f"win rate {_pct(a_today['win_rate'])}, PF {a_today['profit_factor'] or 0:.2f}, net "
                   f"Rs {a_today['net_pnl']:,.0f} with corrected ATR. One session cannot establish a 1-DTE edge or "
                   "lack of one.")
    ok = [b for b in dte_table if b["status"] == "OK"]
    out.append("DTE buckets with real data: " + "; ".join(
        f"{b['dte_bucket']} n={b['evaluated_signals']} ({b['first_day']}..{b['last_day']}, {b['trading_days']} days) "
        f"win {_pct(b['win_rate'])}, premium ATR {b['premium_atr_pct']['median']:.1f}%, "
        f"stop {b['stop_pct']['median']:.1f}%" for b in ok) + ".")
    hist_expiries = sorted({r.get("expiry") for r in hist_rows if r.get("expiry")})
    if len(hist_expiries) == 1:
        out.append(f"Confounding: every historical contract expires {hist_expiries[0]}, so each historical DTE bucket "
                   "is a distinct calendar period. Win-rate differences between historical DTE buckets reflect "
                   "different market periods as much as DTE and must not be read as a DTE effect. Premium ATR % and "
                   "stop % do scale cleanly with DTE (structural).")
    if paired["per_expiry"]:
        out.append("Same 40 signals on different expiries today: " + "; ".join(
            f"{e}: win {_pct(m['win_rate'])}, net Rs {m['net_pnl']:,.0f}" for e, m in paired["per_expiry"].items() if m)
            + f" ({paired['same_outcome_all_expiries']}/{paired['common_resolved_signals']} identical outcomes). "
            + ("Same market and session: outcomes barely depend on expiry, so today's losses come from the "
               "signals' direction/timing, while DTE changes the rupee scale per lot. One day only."
               if paired["common_resolved_signals"]
               and paired["same_outcome_all_expiries"] >= 0.8 * paired["common_resolved_signals"]
               else "Outcomes differ materially by expiry on the same session. One day only."))
    out.append(f"Live benchmark: {benchmark['check']}.")
    return out


def print_summary(report: dict) -> None:
    e = report["engine_research"]
    print("\nS2 CONTRACT COMPARABILITY RESEARCH COMPLETE")
    b, a = e["before"], e["after"]
    if b:
        print(f"  Engine before fix : {b.get('resolved_trades')} trades, win {_pct(b.get('win_rate'))}, "
              f"PF {b.get('profit_factor'):.2f}, net Rs {b.get('net_pnl'):,.0f}")
    print(f"  Engine after fix  : {a['resolved_trades']} trades, win {_pct(a['win_rate'])}, "
          f"PF {a['profit_factor']:.2f}, net Rs {a['net_pnl']:,.0f}, max DD Rs {a['max_drawdown']:,.0f}")
    print(f"  Expired-contract API: {'AVAILABLE' if report['expired_data']['available'] else 'UNAVAILABLE'} "
          f"({report['expired_data']['code']})")
    for group, samples in report["samples"].items():
        for name, s in samples.items():
            if s.get("status") != "OK":
                print(f"  {group} | {name}: NO REAL DATA ({s.get('signals', 0)} signals)")
                continue
            pf = "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
            print(f"  {group} | {name}: {s['evaluated_signals']} signals, {s['resolved']} resolved, "
                  f"win {_pct(s['win_rate'])}, PF {pf}, net Rs {s['net_pnl']:,.0f}, "
                  f"DTE {list(s['dte_distribution'])[:1]}..{list(s['dte_distribution'])[-1:]}")
    print(f"  Benchmark         : {report['benchmark']['check']}")
    for f in report["findings"]:
        print(f"  - {f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 option-contract comparability research")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date())
    parser.add_argument("--hist-from", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--hist-to", type=date.fromisoformat, default=date(2026, 10, 1))
    args = parser.parse_args()
    run(day=args.date, hist_from=args.hist_from, hist_to=args.hist_to)


if __name__ == "__main__":
    main()
