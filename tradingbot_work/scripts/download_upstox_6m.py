#!/usr/bin/env python3
"""Download ~6 months of real Upstox 5-minute NIFTY + NIFTY option candles (PAPER | DATA ACQUISITION ONLY | NO ORDERS).

  python scripts/download_upstox_6m.py
  python scripts/download_upstox_6m.py --from-date 2026-04-01 --to-date 2026-09-30
  python scripts/download_upstox_6m.py --resume      # skip valid files and terminal EMPTY/UNAVAILABLE windows
  python scripts/download_upstox_6m.py --validate    # no API calls: re-check stored data, rebuild reports

Read-only GET requests only. The access token is read from the environment / .env by the existing historical client
and is never printed or written anywhere. No strategy is run or optimised.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values, load_dotenv

load_dotenv(ROOT / ".env", override=False)

from src.config.research_loader import load_research_config
from src.historical_data import upstox_6m as u6
from src.historical_data import upstox_client as uc
from src.historical_data.extension import latest_completed_day
from src.historical_data.models import HistoricalDataError
from src.historical_data.service import RESEARCH_CONFIG, _refuse_live, build_catalog
from src.paper_trading.shadow_validation import capture_upstox_events, order_placement_blocked
from src.safety.live_gate import assert_paper_trading_only, trading_mode

DEFAULT_ROOT = ROOT / u6.DEFAULT_ROOT
DEFAULT_REPORTS = ROOT / "reports" / "strategy_improvement"
TEST_FILE = "tests/test_upstox_6m_download.py"


def _secrets() -> list[str]:
    """Every configured Upstox token value (kept in memory only, used for the leak scan)."""
    names = (*uc.TOKEN_ENV_NAMES, *uc.FORBIDDEN_TOKEN_ENV_NAMES)
    values = list(uc._token_settings().values())
    if uc.ENV_FILE.is_file():
        values += [v for k, v in dotenv_values(uc.ENV_FILE).items() if k in names and v]
    values += [os.environ[k] for k in names if os.environ.get(k)]
    return [v.strip() for v in values if v and v.strip()]


def _run_tests() -> str:
    r = subprocess.run([sys.executable, "-m", "pytest", TEST_FILE, "-q"], cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["?"])[-1]
    return f"{'passed' if r.returncode == 0 else 'failed'} ({tail})"


def run(*, start: date, end: date, root: Path = DEFAULT_ROOT, report_dir: Path = DEFAULT_REPORTS,
        resume: bool = False, validate_only: bool = False, client=None, catalog=None,
        now: datetime | None = None) -> dict:
    assert_paper_trading_only()
    _refuse_live()
    now = now or datetime.now(u6.IST)
    if start > end:
        raise SystemExit(f"--from-date {start} is after --to-date {end}")
    latest = latest_completed_day(now)
    if end > latest:
        print(f"NOTE: end date clamped from {end} to {latest} (later sessions are not complete).")
        end = latest
    cfg = load_research_config(RESEARCH_CONFIG)
    mapping = cfg.option_mapping
    expiry_mode = str(mapping.get("expiry_mode", "nearest"))
    offset = max(1, int(mapping.get("strike_offset", 0)))
    if not validate_only and client is None:
        source = uc.token_source()
        if source is None:
            raise SystemExit("No historical token configured (set one of: " + ", ".join(uc.TOKEN_ENV_NAMES)
                             + " in .env). Sandbox tokens are never used for market data.")
        print(f"Token source variable: {source} (value never printed)")
        client = uc.UpstoxHistoricalClient(token_provider=uc.configured_token, max_retries=4, backoff_seconds=1.0,
                                           max_backoff_seconds=30.0, min_interval_seconds=0.35)
    if not validate_only and catalog is None:
        catalog = build_catalog(client)
    layout = u6.Layout(root)
    with order_placement_blocked(), capture_upstox_events():
        try:
            s = u6.run_download(client=client, catalog=catalog, layout=layout, start=start, end=end,
                                report_dir=report_dir, expiry_mode=expiry_mode, offset=offset, resume=resume,
                                validate_only=validate_only, progress=lambda m: print(f"  {m}", flush=True))
        except HistoricalDataError as exc:
            raise SystemExit(f"STOPPED: {exc.code} - {exc.message}. Re-run with --resume after fixing it.") from None
    return s


def print_summary(s: dict, *, tests: str, leak_ok: bool, leaks: list[str], files: list[str]) -> None:
    u, o = s["underlying"], s["options"]
    print("\nUPSTOX 6-MONTH DATA DOWNLOAD")
    print(f"{u6.BANNER}\n")
    print(f"Period:\n{s['period']['from']} -> {s['period']['to']}\n")
    print(f"NIFTY:\ncandles: {u['candles']}\ntrading days: {u['trading_days']}\nstatus: {u['status']}\n")
    print(f"OPTIONS:\ncontracts: {o['contracts']}\nCE: {o['ce']}\nPE: {o['pe']}\ncandles: {o['candles']}\n"
          f"status: {o['status']}\n")
    print(f"NIFTY underlying coverage: {u['status']}")
    print(f"OPTION premium coverage: {o['status']}")
    if u["status"] == "OK" and o["status"] != "OK":
        print(u6.INSUFFICIENT_TEXT)
        print("Do not run a strategy backtest on this dataset until option coverage is sufficient.")
    print(f"\nDATA QUALITY:\n{'PASS' if s['quality']['pass'] else 'FAIL'}\n")
    print(f"OPTION COVERAGE:\n{s['option_coverage']}\n")
    print("FILES:")
    for f in files:
        print(f"  {f}")
    print(f"\nTESTS:\n{tests}\n")
    print(f"TOKEN LEAK:\n{'PASS' if leak_ok else 'FAIL'}")
    for f in leaks:
        print(f"  token value found in: {f}")
    print(f"\nORDERS:\n0\n\nTRADING MODE:\n{trading_mode()}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-date", type=date.fromisoformat, default=u6.DEFAULT_FROM)
    p.add_argument("--to-date", type=date.fromisoformat, default=u6.DEFAULT_TO)
    p.add_argument("--resume", action="store_true",
                   help="reuse the stored contract plan and skip windows already SUCCESS/EMPTY/UNAVAILABLE")
    p.add_argument("--validate", action="store_true", help="no API calls; validate stored data and rebuild reports")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--with-tests", action="store_true", help=f"also run {TEST_FILE} and report the result")
    a = p.parse_args(argv)
    s = run(start=a.from_date, end=a.to_date, root=a.root, report_dir=a.report_dir, resume=a.resume,
            validate_only=a.validate)
    layout = u6.Layout(a.root)
    scanned = [layout.metadata, a.report_dir / "upstox_6m_data_quality.md", a.report_dir / "upstox_6m_coverage.md",
               layout.root / "underlying", layout.root / "options"]
    leak_ok, leaks = u6.token_leak_check(scanned, _secrets())
    files = [*s["paths"].values(),
             f"{(layout.root / 'underlying').as_posix()}/ ({len(layout.underlying_files())} parquet files)",
             f"{(layout.root / 'options').as_posix()}/ ({len(layout.option_files())} parquet files)"]
    tests = _run_tests() if a.with_tests else f"not run by this script (run: python -m pytest {TEST_FILE} -q)"
    print_summary(s, tests=tests, leak_ok=leak_ok, leaks=leaks, files=files)
    return 0 if leak_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
