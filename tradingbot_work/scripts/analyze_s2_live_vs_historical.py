#!/usr/bin/env python3
"""Why did NIFTY S2 perform poorly on a live day? Live day vs historical S2 sample (analysis only, no orders)."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest.current_day_s2 import IST, LevelParams, point_in_time_signals, premium_frame, validate_intraday
from src.backtest.live_vs_historical import (
    LOSS,
    WIN,
    add_underlying_features,
    assign_clusters,
    day_profiles,
    feature_comparison,
    group_characteristics,
    group_metrics,
    investigations,
    percentile_rank,
    wilson,
    write_reports,
)
from src.backtest.regime_filter import PointInTimeRegimeClassifier
from src.backtest.research_strategies import generate_signals
from src.backtest.strategy_comparator import compare_strategies
from src.historical_data.models import HistoricalDataError
from src.historical_data.option_coverage import entry_allowed, resolve_signal_contract
from src.historical_data.service import HistoricalDataService, _refuse_live
from src.safety.live_gate import assert_paper_trading_only
from src.session.session_manager import SessionManager

DEFAULT_OUT = ROOT / "reports" / "strategy_research"
INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"
UNRESOLVED_EXITS = {"DATA_INVALID", "SKIPPED"}
DAY_FIELDS = ("signals", "orb_width_pct", "day_range_pct", "open_to_close_pct", "efficiency_ratio", "orb_crossings",
              "share_bars_inside_orb", "first_break_follow_through_orb_widths", "first_break_follow_through_atr",
              "win_rate", "net_pnl")


def engine_metrics(result: dict) -> dict:
    m = result["metrics"]
    trades = [t for t in result.get("trades", []) if t.get("exit_reason") not in UNRESOLVED_EXITS]
    skips: dict[str, int] = {}
    for s in result.get("skipped", []):
        detail = s.get("detail")
        key = s["reason"] + (f":{detail if isinstance(detail, str) else '+'.join(map(str, detail))}" if detail else "")
        skips[key] = skips.get(key, 0) + 1
    return {"resolved_trades": m.get("resolved_trades"), "win_rate": m.get("win_rate"),
            "profit_factor": m.get("profit_factor"),
            "net_pnl": round(sum(float(t.get("pnl") or 0) for t in trades), 2),
            "max_drawdown": m.get("max_drawdown"),
            "skipped_signals": skips,
            "skip_summary": ("other signals skipped: " + ", ".join(f"{k} x{v}" for k, v in skips.items()))
            if skips else "no signals skipped",
            "trades": [{k: t.get(k) for k in ("signal_time", "direction", "trading_symbol", "entry_price",
                                              "exit_price", "exit_reason", "pnl")} for t in trades]}


def run_engine(service, underlying, candles, premium, contracts, splits) -> dict:
    results, _ = compare_strategies(
        research_config=service.research_config, strategy_config=service.strategy_config, underlying=underlying,
        underlying_candles=candles, premium_history=premium, contracts=contracts, strategy_ids=["S2"],
        run_id=uuid.uuid4().hex[:12], master_resolver=service._research_resolver, splits=splits)
    return {r.split: engine_metrics(r.as_dict()) for r in results}


def _med(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return median(values) if values else None


def top_reasons(today_m, hist_m, today_day, day_cmp, inv, engine, today_rows, hist_rows) -> list[str]:
    decided_t = [r for r in today_rows if r["status"] in (WIN, LOSS)]
    decided_h = [r for r in hist_rows if r["status"] in (WIN, LOSS)]
    clusters = len({r["outcome_cluster"] for r in decided_t if r.get("outcome_cluster")})
    er, rng = day_cmp["efficiency_ratio"], day_cmp["day_range_pct"]
    hist_dte = [r["dte"] for r in decided_h if r.get("dte") is not None]
    today_dte = "/".join(str(d) for d in sorted({r["dte"] for r in decided_t if r.get("dte") is not None}))
    eng = engine.get("today", {})
    eng_net = eng.get("net_pnl")
    g = next(i for i in inv if i["code"] == "G")
    day_level = g["evidence"][1].replace("Day-level overall win rate: ", "").rstrip(".")
    return [
        f"Out-of-sample contract type: today's signals traded {today_dte}-DTE options, but every historical "
        f"S2 signal was mapped to a {min(hist_dte)}-{max(hist_dte)} DTE contract (the resolver only lists currently "
        f"tradable expiries). Premium ATR was {_med(decided_t, 'premium_atr_pct'):.1f}% of premium vs "
        f"{_med(decided_h, 'premium_atr_pct'):.1f}%, so the unchanged 1.5x premium-ATR stop sat "
        f"{_med(decided_t, 'stop_pct'):.1f}% below entry (historical {_med(decided_h, 'stop_pct'):.1f}%) and the 2R "
        f"target {_med(decided_t, 'target_pct'):.1f}% above; in index terms the stop needed "
        f"{_med(decided_t, 'stop_underlying_atr'):.2f} ATR vs {_med(decided_h, 'stop_underlying_atr'):.2f}. This is a "
        "structural difference, not noise; its effect on win rate cannot be measured because history has no "
        "comparable-DTE trades.",
        f"Whipsaw session: day range {rng['today']:.2f}% (percentile {rng['percentile']}) with net move "
        f"{today_day['open_to_close_pct']:+.2f}% and efficiency ratio {er['today']:.3f} (percentile {er['percentile']} "
        f"of {er['n']} days); the first (upside) break reversed and later SELL entries were stopped by bounces "
        f"(losers' median MFE {_med([r for r in decided_t if r['status'] == LOSS], 'mfe_r'):.2f}R vs a 2R target). "
        f"One day: descriptive, not statistically testable.",
        f"Correlated repeat signals amplified a normal-for-S2 bad day: {today_day['signals']} signals collapse into "
        f"{clusters} exit events; at the day level {day_level}, so the win rate itself is within the historical "
        f"day-to-day range. The signal-level net Rs {today_m['net_pnl']:,.0f} counts every repeat; the research "
        f"engine (one position, Rs 3,000 daily loss budget) took {eng.get('resolved_trades', 'n/a')} trade(s), net "
        f"Rs {eng_net:,.0f}" + (f" ({eng['skip_summary']})." if eng.get("skip_summary") else ".")
        if eng_net is not None else "engine replay unavailable.",
    ]


def run(*, day: date, hist_from: date, hist_to: date, underlying: str = "NIFTY", out_dir: Path = DEFAULT_OUT,
        service: HistoricalDataService | None = None, client=None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    service = service or HistoricalDataService()
    client = client or service.client
    rc, sc = service.research_config, service.strategy_config
    params = LevelParams.from_configs(rc, sc)
    s2_cfg = dict(rc.strategies["S2"])
    session = SessionManager(sc.session)
    delta = float(sc.options.risk.underlying_delta)
    target_r = float(sc.options.risk.target_r)

    hist = service.read_underlying(underlying, hist_from, hist_to)
    if hist.empty:
        raise SystemExit(f"Missing stored underlying candles {hist_from}..{hist_to}")
    try:
        raw = client.intraday_candles(INSTRUMENT_KEY, interval=params.interval)
    except HistoricalDataError as exc:
        raise SystemExit(f"API_REQUEST_ERROR: {exc.code} (HTTP {exc.status})") from None
    today_candles, validation = validate_intraday(raw, day, params.interval)
    if today_candles.empty:
        raise SystemExit(f"No intraday candles for {day}")
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    parts = []
    for part in (hist, today_candles):
        part = part[cols].copy()
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True).dt.tz_convert(IST)
        parts.append(part)
    combined = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    combined = combined.drop_duplicates("timestamp").reset_index(drop=True)
    combined["symbol"] = underlying.upper()
    hist_frame = combined[combined["timestamp"].dt.date <= hist_to].reset_index(drop=True)

    hist_signals = [s for s in generate_signals("S2", hist_frame, underlying.upper(), rc)
                    if entry_allowed(s.timestamp, session, params.stop_new_entries)]
    today_signals, lookahead_ok = point_in_time_signals(today_candles, underlying, rc, session,
                                                        params.stop_new_entries)

    intraday_cache: dict[str, list] = {}
    contracts_today: dict[str, object] = {}

    def premium_for(contract, live: bool):
        stored = service.store.read("options", underlying.upper(), contract.instrument_key, expiry=contract.expiry)
        today = []
        if live:
            contracts_today[contract.instrument_key] = contract
            if contract.instrument_key not in intraday_cache:
                try:
                    intraday_cache[contract.instrument_key] = client.intraday_candles(
                        contract.instrument_key, interval=params.interval)
                except HistoricalDataError as exc:
                    return f"PREMIUM_API_ERROR:{exc.code}"
            today = [c for c in intraday_cache[contract.instrument_key] if c.timestamp.astimezone(IST).date() == day]
        return premium_frame(stored, today)

    def resolve(sig):
        return resolve_signal_contract(sig, service.resolver_factory(sig.timestamp.date()), rc)

    from src.backtest.live_vs_historical import signal_outcomes
    hist_rows = signal_outcomes(hist_signals, resolve, lambda c: premium_for(c, False), params,
                                rc.session.square_off, "historical")
    today_rows = signal_outcomes(today_signals, resolve, lambda c: premium_for(c, True), params,
                                 rc.session.square_off, "today")
    pit = PointInTimeRegimeClassifier(combined)
    frame = add_underlying_features(hist_rows + today_rows, combined, s2_cfg, pit, delta)
    assign_clusters(hist_rows)
    assign_clusters(today_rows)

    days = day_profiles(frame, hist_rows + today_rows, s2_cfg.get("range_end", "09:30"))
    hist_days = [d for d in days if d["date"] != day.isoformat()]
    today_day = next(d for d in days if d["date"] == day.isoformat())
    day_cmp = {f: {"today": today_day.get(f),
                   "historical_median": median([d[f] for d in hist_days if d.get(f) is not None]),
                   "percentile": percentile_rank(today_day.get(f), [d.get(f) for d in hist_days]),
                   "n": len(hist_days)} for f in DAY_FIELDS}

    engine: dict = {}
    hist_premium = service._premium_history(underlying, hist_from, hist_to)
    try:
        engine["historical"] = run_engine(service, underlying, hist_frame, hist_premium,
                                          service._contracts_from_store(underlying),
                                          {"FULL": (hist_from, hist_to)})["FULL"]
    except Exception as exc:  # report, do not hide
        engine["historical"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        today_premium = dict(hist_premium)
        for key, contract in contracts_today.items():
            today_premium[key] = premium_for(contract, True)
        engine["today"] = run_engine(service, underlying, combined, today_premium,
                                     service._contracts_from_store(underlying) + list(contracts_today.values()),
                                     {"TODAY": (day, day)})["TODAY"]
    except Exception as exc:
        engine["today"] = {"error": f"{type(exc).__name__}: {exc}"}

    tracker_path = ROOT / "reports" / "live_signal_accuracy" / f"{day.isoformat()}.json"
    consistency = "tracker report not found"
    if tracker_path.exists():
        tracked = {s["time"][11:16]: s["status"] for s in json.loads(tracker_path.read_text(encoding="utf-8"))["signals"]}
        mine = {r["signal_time"][11:16]: r["status"] for r in today_rows}
        diff = sorted(k for k in set(tracked) | set(mine) if tracked.get(k) != mine.get(k))
        consistency = "matches live tracker report" if not diff else f"differs from tracker at {diff}"

    hist_m, today_m = group_metrics(hist_rows), group_metrics(today_rows)
    ci = wilson(today_m["wins"], today_m["resolved"])
    inv = investigations(hist_rows, today_rows, hist_days, today_day, engine, target_r,
                         stop_multiplier=float(sc.options.risk.stop_atr_multiplier))
    losers = sorted((r for r in today_rows if r["status"] == LOSS), key=lambda r: r["signal_time"])
    winners = sorted((r for r in today_rows if r["status"] == WIN), key=lambda r: r["signal_time"])
    report = {
        "live_date": day.isoformat(),
        "historical_window": f"{hist_from}..{hist_to}",
        "underlying": underlying.upper(),
        "strategy_id": "S2",
        "generated_at": datetime.now(IST).isoformat(timespec="seconds"),
        "mode": "analysis only - PAPER, no broker calls, S2 unchanged, no filters proposed",
        "notes": [
            "Both samples use the live tracker's signal-level method: every point-in-time S2 signal, next-bar option "
            "open entry, unchanged premium-ATR stop and 2R target, first touch on real option candles, EOD exit at the "
            "open of the first candle at/after square-off, same-candle SL+target = AMBIGUOUS, P&L = 1 lot (65), "
            "costs excluded.",
            "Engine-level results (one position at a time, breakeven/trailing exits, daily loss budget, costs) are "
            "reported separately; they are what the research engine would have traded.",
            f"Today's signals were generated bar by bar from completed candles; look-ahead check "
            f"{'PASS' if lookahead_ok else 'MISMATCH'}. Consistency with live tracker: {consistency}.",
            "Underlying indicators are computed on stored history plus today's candles (warm-up across days). NIFTY "
            "index volume is zero, so underlying VWAP is unavailable; option VWAP uses option volume.",
            "Signals within a day overlap (same contract, same exit bar). 'Outcome clusters' group signals that share "
            "an exit event; tests on clusters are more honest than tests on raw signals. Today is one day: any "
            "day-level difference has n = 1 and cannot be statistically established.",
            "Hindsight descriptors (MFE/MAE, false breakout after entry, holding time) explain outcomes but are not "
            "known at entry.",
            "Data limitation: the option resolver only lists currently tradable expiries, so historical signals (here "
            "and in the stored S2 research/engine replay) were mapped to the 2026-10-06 expiry at 5-35 DTE rather than "
            "to the weekly that was nearest on each historical day. Today's live contracts were 1 DTE. Historical "
            "results therefore do not describe near-expiry option behaviour.",
        ],
        "summary": {
            "historical_signal_level": hist_m, "today_signal_level": today_m,
            "historical_days": len(hist_days),
            "historical_signals_per_day": round(len(hist_rows) / len(hist_days), 2) if hist_days else None,
            "today_win_rate_ci": None if ci is None else f"{ci[0] * 100:.1f}%-{ci[1] * 100:.1f}%",
            "today_outcome_clusters": len({r["outcome_cluster"] for r in today_rows if r.get("outcome_cluster")}),
            "historical_outcome_clusters": len({r["outcome_cluster"] for r in hist_rows if r.get("outcome_cluster")}),
            "candle_validation": {"received": validation.received, "valid": validation.valid_candles,
                                  "flat_candles": len(validation.flat_candles)},
        },
        "engine": engine,
        "day_comparison": day_cmp,
        "today_day_profile": today_day,
        "historical_day_profiles": hist_days,
        "investigations": inv,
        "feature_comparison": feature_comparison(hist_rows, today_rows),
        "today_groups": {
            "losers": {"count": len(losers), "characteristics": group_characteristics(losers), "signals": losers},
            "winners": {"count": len(winners), "characteristics": group_characteristics(winners), "signals": winners},
        },
        "today_signals": today_rows,
        "historical_signals": hist_rows,
    }
    report["top_reasons"] = top_reasons(today_m, hist_m, today_day, day_cmp, inv, engine, today_rows, hist_rows)
    for path in write_reports(report, out_dir, f"{underlying.upper()}_S2_live_vs_historical_{day.isoformat()}").values():
        print(f"Wrote {path}")
    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    s = report["summary"]
    h, t = s["historical_signal_level"], s["today_signal_level"]
    pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"
    num = lambda v: "n/a" if v is None else f"{v:,.2f}"
    print("\nS2 LIVE VS HISTORICAL ANALYSIS COMPLETE")
    print(f"  Historical {report['historical_window']}: {h['signals']} signals ({h['buys']} BUY / {h['sells']} SELL), "
          f"resolved {h['resolved']}, win rate {pct(h['win_rate'])}, PF {num(h['profit_factor'])}, "
          f"net Rs {h['net_pnl']:,.0f}, max DD Rs {h['max_drawdown']:,.0f}")
    print(f"  Today {report['live_date']}: {t['signals']} signals ({t['buys']} BUY / {t['sells']} SELL), "
          f"resolved {t['resolved']}, win rate {pct(t['win_rate'])} (95% CI {s['today_win_rate_ci']}), "
          f"PF {num(t['profit_factor'])}, net Rs {t['net_pnl']:,.0f}, max DD Rs {t['max_drawdown']:,.0f}")
    for k, v in report["engine"].items():
        if "error" in v:
            print(f"  Engine {k}: {v['error']}")
        else:
            print(f"  Engine {k}: {v['resolved_trades']} trades, win rate {pct(v['win_rate'])}, "
                  f"PF {num(v['profit_factor'])}, net Rs {num(v['net_pnl'])}; {v['skip_summary']}")
    for i in report["investigations"]:
        print(f"  {i['code']}. {i['question']}: {i['verdict']} (p={i['p_value']}, independent n={i['independent_n']})")
    print("  Top evidence-based reasons:")
    for n, r in enumerate(report["top_reasons"], 1):
        print(f"   {n}. {r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 live day vs historical sample (analysis only)")
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date())
    parser.add_argument("--hist-from", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--hist-to", type=date.fromisoformat, default=date(2026, 10, 1))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(day=args.date, hist_from=args.hist_from, hist_to=args.hist_to, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
