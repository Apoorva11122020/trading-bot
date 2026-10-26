"""Manage live paper tracker subprocess and expose today's events to the UI."""
from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime, time as daytime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IST = ZoneInfo("Asia/Kolkata")
PAPER_DIR = ROOT / "reports" / "daily_pivot_paper"
LOG_FILE = PAPER_DIR / "live_runner.log"

_process: subprocess.Popen | None = None
_active: dict[str, Any] = {}


def _market_open_for_start() -> bool:
    now = datetime.now(IST)
    return now.weekday() < 5 and daytime(9, 0) <= now.time() <= daytime(15, 15)


def _event_file(symbol: str, day: date | None = None) -> Path:
    day = day or datetime.now(IST).date()
    return PAPER_DIR / f"{day.isoformat()}_{symbol}_paper.csv"


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path).fillna("")
    return frame.to_dict(orient="records")


def _summary_for(symbol: str, day: date | None = None) -> dict[str, Any] | None:
    summary_path = PAPER_DIR / "daily_summary.csv"
    if not summary_path.is_file():
        return None
    day_key = (day or datetime.now(IST).date()).isoformat()
    frame = pd.read_csv(summary_path)
    rows = frame[(frame.date.astype(str) == day_key) & (frame.symbol.astype(str) == symbol)]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def live_status() -> dict[str, Any]:
    today = datetime.now(IST).date()
    symbol = str(_active.get("symbol") or "NIFTY")
    events = _read_events(_event_file(symbol, today))
    pivot_exits = [e for e in events if e.get("event") == "PAPER_EXIT"]
    pe_exits = [e for e in events if e.get("event") == "PE_EXIT"]
    open_entry = next(
        (e for e in reversed(events) if e.get("event") in ("PAPER_ENTRY", "PE_ENTRY")),
        None,
    )
    has_open = open_entry is not None and not any(
        e.get("event") in ("PAPER_EXIT", "PE_EXIT")
        and e.get("timestamp", "") > open_entry.get("timestamp", "")
        for e in events
    )
    pivot_net = sum(float(e.get("net_points") or 0) for e in pivot_exits)
    pe_net = sum(float(e.get("net_points") or 0) for e in pe_exits)
    running = _process is not None and _process.poll() is None
    return {
        "running": running,
        "market_open": _market_open_for_start(),
        "time": datetime.now(IST).isoformat(),
        "strategy_id": _active.get("strategy_id"),
        "strategy_label": _active.get("strategy_label"),
        "symbol": symbol,
        "event_count": len(events),
        "closed_trades": len(pivot_exits) + len(pe_exits),
        "pivot_net_points": round(pivot_net, 2),
        "pe_net_inr": round(pe_net, 2),
        "open_position": has_open,
        "last_event": events[-1] if events else None,
        "summary": _summary_for(symbol, today),
        "log_tail": _tail_log(30),
    }


def _tail_log(lines: int) -> list[str]:
    if not LOG_FILE.is_file():
        return []
    content = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return content[-lines:]


def start_live_paper(strategy_id: str) -> dict[str, Any]:
    global _process, _active
    from scripts.strategy_catalog import strategy_by_id

    if not _market_open_for_start():
        raise ValueError("Live paper can start only Mon–Fri 09:00–15:15 IST.")

    spec = strategy_by_id(strategy_id)
    symbol = spec.symbols_default[0] if spec.symbols_default else "NIFTY"
    stop_live_paper()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_daily_pivot_paper.py"),
        "--symbol", symbol,
        "--pivot-preset", spec.preset or "guide_baseline",
        "--guide-filter", spec.guide_filter,
        "--rr", str(spec.rr),
        "--atr-buffer", str(spec.atr_buffer),
        "--strategy-id", spec.id,
    ]
    if not spec.include_pe:
        cmd.append("--no-pe")

    with LOG_FILE.open("a", encoding="utf-8") as log:
        log.write(f"\n--- START {datetime.now(IST).isoformat()} {strategy_id} ---\n")
        _process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
    _active = {
        "strategy_id": spec.id,
        "strategy_label": spec.label,
        "symbol": symbol,
        "pid": _process.pid,
        "started_at": datetime.now(IST).isoformat(),
    }
    return live_status()


def stop_live_paper() -> dict[str, Any]:
    global _process, _active
    if _process is not None and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _process.kill()
    _process = None
    _active = {}
    return live_status()


def live_events(limit: int = 200) -> dict[str, Any]:
    today = datetime.now(IST).date()
    symbol = str(_active.get("symbol") or "NIFTY")
    events = _read_events(_event_file(symbol, today))
    return {
        "events": events[-limit:],
        "status": live_status(),
    }
