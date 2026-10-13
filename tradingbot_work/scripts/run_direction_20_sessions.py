#!/usr/bin/env python3
"""BUY vs SELL breakdown of the original S2 signals over the latest 20 stored NIFTY sessions (PAPER, no orders).

Sessions are the latest complete trading sessions found in data/historical/ (not hand-picked). Signals are replayed
with the existing historical shadow pipeline: real expired NIFTY option contracts and their cached 5-minute candles,
existing S2 exit logic, DATA_UNAVAILABLE when option data is missing. Reports go to
reports/strategy_direction_20_sessions/.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_shadow_historical_validation as hv
from src.backtest import direction_sessions as ds
from src.backtest import shadow_historical as sh
from src.historical_data.service import _refuse_live
from src.safety.live_gate import assert_paper_trading_only

DEFAULT_OUT = ROOT / "reports" / "strategy_direction_20_sessions"
DEFAULT_SESSIONS = 20


def run(*, sessions: int = DEFAULT_SESSIONS, to_date: date | None = None, out_dir: Path = DEFAULT_OUT,
        today: date | None = None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    today = today or datetime.now(sh.IST).date()
    result, meta = hv.build_replay(from_date=None, to_date=to_date or today - timedelta(days=1), days=sessions,
                                   today=today, direction_lookahead=True)
    analysis = ds.build_analysis(result, meta, sessions)
    paths = ds.write_reports(analysis, out_dir)
    print_result(analysis, paths)
    return analysis


def print_result(a: dict, paths: dict) -> None:
    pct, pf, rs = ds._pct, ds._pf, ds._rs
    print(f"{ds.HEADING} | PAPER | no orders")
    print(f"Selected {a['sessions_selected']} trading sessions:"
          + (f" (only {a['sessions_selected']} available, {a['requested_sessions']} requested)"
             if a["fewer_sessions_than_requested"] else ""))
    for d in a["selected_dates"]:
        print(f"  {d}")
    dl = (a["replay_meta"] or {}).get("expired_candle_downloads") or {}
    print(f"Expired contracts: downloaded {dl.get('contracts_downloaded', 0)}, from cache "
          f"{dl.get('contracts_from_cache', 0)}, failures {len(dl.get('failures') or [])}")
    print()
    print("Direction | Signals | Evaluated | Wins | Losses | Accuracy | PF | P&L | Max DD")
    for name, m in (*a["directions"].items(), ("TOTAL", a["total"])):
        print(f"{name} | {m['signals']} | {m['evaluated']} | {m['wins']} | {m['losses']} | {pct(m['win_rate'])} | "
              f"{pf(m['profit_factor'])} | {rs(m['net_pnl'])} | {rs(m['max_drawdown'])}")
    print()
    print(f"BUY ACCURACY = {pct(a['directions']['BUY']['win_rate'])}")
    print(f"SELL ACCURACY = {pct(a['directions']['SELL']['win_rate'])}")
    print(f"TOTAL ACCURACY = {pct(a['total']['win_rate'])}")
    print("Lookahead: " + ", ".join(f"{k} = {ds._la(v)}" for k, v in a["lookahead"].items()))
    c = a["conclusion"]
    print()
    print("BUY vs SELL conclusion:")
    print(f"- Higher accuracy: {c['higher_accuracy']}")
    print(f"- Better PF: {c['better_profit_factor']}")
    print(f"- Better P&L: {c['better_pnl']}")
    print(f"- BUY sample sufficient ({c['buy_sample_rule']}): {'YES' if c['buy_sample_sufficient'] else 'NO'}")
    print(f"- BUY signals evaluated: {c['buy_evaluated']}")
    print(f"- BUY signals unavailable: {c['buy_data_unavailable']}")
    print(f"- {c['note']}")
    print("Reports: " + ", ".join(str(p) for p in paths.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    p.add_argument("--to-date", type=date.fromisoformat, default=None,
                   help="last calendar date to consider (default: yesterday)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    run(sessions=a.sessions, to_date=a.to_date, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
