#!/usr/bin/env python3
"""Analyse why NIFTY S2 historical trades lose (historical analysis only, PAPER mode, no orders)."""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.losing_trade_analysis import MIN_FILTER_SUPPORT, _fmt, analyze, write_reports
from src.backtest.robustness import load_final_comparison, row_for
from src.backtest.strategy_comparator import compare_strategies
from src.historical_data.service import HistoricalDataService
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_COMPARISON = ROOT / "reports" / "strategy_research" / "final_comparison.json"
DEFAULT_OUT = ROOT / "reports" / "strategy_research"
UNRESOLVED_EXITS = {"DATA_INVALID", "SKIPPED"}


def resolved_trades(result: dict) -> list[dict]:
    return [t for t in result.get("trades", []) if t.get("exit_reason") not in UNRESOLVED_EXITS]


def build_conclusions(analysis: dict, window_move: tuple[float, float] | None = None) -> list[str]:
    lines: list[str] = []
    pattern = analysis["biggest_losing_pattern"]
    if pattern:
        lines.append(f"Biggest losing pattern: {pattern['factor']} = {pattern['bucket']} "
                     f"({pattern['trades']} trades, net {_fmt(pattern['net_pnl'], 'money')}, "
                     f"win rate {_fmt(pattern['win_rate'], 'pct')}, sample {pattern['sample']}).")
    else:
        lines.append("No bucket with at least 5 trades has negative net P&L.")
    best = analysis["strongest_candidate"]
    if best:
        lines.append(f"Strongest candidate for further out-of-sample testing: {best['condition']} "
                     f"(removes {best['trades_removed']} trades: {best['wins_removed']} wins, "
                     f"{best['losses_removed']} losses; net {_fmt(best['original_net_pnl'], 'money')} -> "
                     f"{_fmt(best['filtered_net_pnl'], 'money')}, PF {_fmt(best['original_pf'])} -> "
                     f"{_fmt(best['filtered_pf'])}). It is not applied to S2.")
        if best["non_monotonic"]:
            lines.append("Caution: that bucket is losing while both neighbouring buckets are profitable "
                         "(non-monotonic), which is typical of sampling noise rather than a causal effect.")
    else:
        lines.append(f"No point-in-time filter removing >= {MIN_FILTER_SUPPORT} trades raised net P&L and PF "
                     "without increasing drawdown.")
    small = [c for c in analysis["hypothetical_filters"]
             if c["assessment"] == "TOO_FEW_TRADES" and c["net_pnl_change"] > 0]
    if small:
        lines.append(f"{len(small)} apparent improvements rest on fewer than {MIN_FILTER_SUPPORT} trades and "
                     "are treated as noise.")
    direction = {b["bucket"]: b for f in analysis["factors"] if f["factor_id"] == "direction" for b in f["buckets"]}
    if direction:
        drift = ""
        if window_move is not None:
            first, last = window_move
            drift = (f" NIFTY moved from {first:,.0f} to {last:,.0f} ({(last / first - 1) * 100:+.1f}%) over the "
                     "window, so the direction gap may reflect this window's trend rather than a persistent edge.")
        lines.append("Direction split: " + "; ".join(
            f"{k} {v['trades']} trades net {_fmt(v['net_pnl'], 'money')} PF {_fmt(v['profit_factor'])}"
            for k, v in direction.items()) + "." + drift)
    lines.append("Losing trades are mostly clean stop-outs (-1R) that never moved favourably; outcome descriptors "
                 "(initial move, SL first, false breakout, holding time) describe losses but are not usable as "
                 "entry filters.")
    lines.append("All findings come from one ~1-month window; any candidate needs out-of-sample validation "
                 "before it is considered.")
    return lines


def run_analysis(*, source_run_id: str, comparison_path: Path = DEFAULT_COMPARISON, out_dir: Path = DEFAULT_OUT,
                 underlying: str = "NIFTY", strategy_id: str = "S2") -> dict:
    assert_paper_trading_only()
    final = load_final_comparison(comparison_path)
    if final.get("run_id") != source_run_id:
        raise SystemExit(f"Comparison file run_id {final.get('run_id')} does not match requested {source_run_id}")
    row = row_for(final, underlying, strategy_id)
    from_date, to_date = date.fromisoformat(row["from_date"]), date.fromisoformat(row["to_date"])

    service = HistoricalDataService()
    candles = service.read_underlying(underlying, from_date, to_date)
    if candles.empty:
        raise SystemExit(f"Missing underlying data for {underlying} {from_date}..{to_date}")
    candles["symbol"] = underlying.upper()
    premium = service._premium_history(underlying, from_date, to_date)
    replay_run_id = uuid.uuid4().hex[:12]
    results, _ = compare_strategies(
        research_config=service.research_config, strategy_config=service.strategy_config, underlying=underlying,
        underlying_candles=candles, premium_history=premium, contracts=service._contracts_from_store(underlying),
        strategy_ids=[strategy_id], run_id=replay_run_id, master_resolver=service._research_resolver,
        splits={"FULL": (from_date, to_date)},
    )
    result = next(r for r in results if r.split == "FULL").as_dict()
    trades = resolved_trades(result)
    if len(trades) != result["metrics"]["resolved_trades"]:
        raise SystemExit("Resolved trade count mismatch between trades and engine metrics")

    s2_cfg = dict(service.research_config.strategies[strategy_id])
    initial_equity = float(service.research_config.risk.initial_equity)
    analysis = analyze(trades, candles, premium, s2_cfg, initial_equity)
    notes = [
        f"Stored source run {source_run_id} reported {row['resolved_trades']} trades / net "
        f"{row['net_pnl']:,.0f}; it predates the Rs 3,000 daily-loss sizing fix. The trades analysed here are a "
        f"deterministic replay of the same window with the current (unchanged) configuration: "
        f"{len(trades)} resolved trades.",
        f"Stored underlying candles end {candles['timestamp'].max()}.",
        "NIFTY index candles from Upstox carry zero volume: underlying volume, 20-candle average volume and "
        "underlying VWAP are unavailable. Option premium volume/VWAP are reported separately.",
        "Win = P&L > 0, loss = P&L < 0, breakeven = 0 (same rule as the research engine).",
        "Indicators (EMA20/EMA50/RSI14/ATR14) are evaluated on the signal bar close, which is known at entry.",
    ]
    report = {
        "source_run_id": source_run_id,
        "replay_run_id": replay_run_id,
        "underlying": underlying.upper(),
        "strategy_id": strategy_id,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "mode": "historical analysis only - PAPER, no broker calls",
        "source_row_summary": f"{row['resolved_trades']} trades, net {row['net_pnl']:,.0f} (pre risk-fix)",
        "replay_metrics": result["metrics"],
        "notes": notes,
        "analysis": analysis,
    }
    ordered = candles.sort_values("timestamp")
    report["conclusions"] = build_conclusions(
        analysis, (float(ordered["open"].iloc[0]), float(ordered["close"].iloc[-1])))
    for path in write_reports(report, out_dir).values():
        print(f"Wrote {path}")
    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    a = report["analysis"]
    o = a["overall"]
    print("\nS2 LOSING TRADE ANALYSIS COMPLETE")
    print(f"  Total resolved trades : {o['trades']}")
    print(f"  Winning trades        : {o['wins']}")
    print(f"  Losing trades         : {o['losses']}  (breakeven {o['breakeven']})")
    p = a["biggest_losing_pattern"]
    print("  Biggest losing pattern: " + (
        f"{p['factor']} = {p['bucket']} ({p['trades']} trades, net {_fmt(p['net_pnl'], 'money')}, {p['sample']})"
        if p else "none with >= 5 trades and negative net P&L"))
    c = a["strongest_candidate"]
    if c:
        print(f"  Strongest candidate   : {c['condition']} (not applied)")
        print(f"  Supporting trades     : {c['trades_removed']} removed "
              f"({c['wins_removed']} wins / {c['losses_removed']} losses), {c['trades_remaining']} remain; "
              f"net {_fmt(c['original_net_pnl'], 'money')} -> {_fmt(c['filtered_net_pnl'], 'money')}, "
              f"PF {_fmt(c['original_pf'])} -> {_fmt(c['filtered_pf'])}")
        if c["non_monotonic"]:
            print("  Caution               : non-monotonic bucket (neighbours profitable) - likely noise; "
                  "needs out-of-sample validation")
    else:
        print(f"  Strongest candidate   : none met the support/PF/net/drawdown criteria")


def main() -> None:
    parser = argparse.ArgumentParser(description="NIFTY S2 losing-trade analysis (historical only)")
    parser.add_argument("--run-id", default="b9202ebc90b7")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_analysis(source_run_id=args.run_id, comparison_path=args.comparison, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
