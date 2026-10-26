"""Local web UI for doc pivot + PE combined backtest."""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from scripts.combined_backtest_service import (
    execute_combined_backtest,
    execute_strategy_backtest,
    load_run_tables,
)
from scripts.run_combined_last7_report import (
    ATR_BUFFER_CHOICES,
    DEFAULT_ATR_BUFFER,
    DEFAULT_RR,
    RR_CHOICES,
)
from src.safety.live_gate import assert_paper_trading_only

IST = ZoneInfo("Asia/Kolkata")
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Doc Pivot + PE Backtest UI", version="1.4")


def normalize_rr(rr: float) -> float:
    value = float(rr)
    for choice in RR_CHOICES:
        if abs(value - choice) < 1e-9:
            return choice
    raise HTTPException(400, f"RR must be one of {RR_CHOICES}")


def normalize_atr_buffer(atr_buffer: float) -> float:
    value = float(atr_buffer)
    for choice in ATR_BUFFER_CHOICES:
        if abs(value - choice) < 1e-9:
            return choice
    raise HTTPException(400, f"ATR buffer must be one of {ATR_BUFFER_CHOICES}")


class BacktestRequest(BaseModel):
    start: date
    end: date
    strategy_id: str | None = Field(default=None)
    preset: str = Field(default="guide_baseline")
    symbols: list[str] = Field(default_factory=lambda: ["NIFTY", "BANKNIFTY"])
    guide_filter: str = Field(default="short_nifty_11_13+one_per_day")
    rr: float = Field(default=DEFAULT_RR)
    atr_buffer: float = Field(default=0.05)


class CompareStrategiesRequest(BaseModel):
    start: date
    end: date
    symbols: list[str] | None = None
    sort_by: str = Field(default="net", pattern="^(net|win)$")
    rr: float = Field(default=2.0)
    atr_buffer: float = Field(default=0.05)


class LivePaperStartRequest(BaseModel):
    strategy_id: str = Field(default="doc_optimized")


class CompareFiltersRequest(BaseModel):
    start: date
    end: date
    symbols: list[str] = Field(default_factory=lambda: ["NIFTY", "BANKNIFTY"])
    sort_by: str = Field(default="net", pattern="^(net|win|balanced)$")
    rr: float = Field(default=DEFAULT_RR)
    atr_buffer: float = Field(default=0.05)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "time": datetime.now(IST).isoformat()}


@app.get("/api/rr-options")
async def rr_options() -> dict:
    return {
        "default": DEFAULT_RR,
        "choices": [{"value": v, "label": f"1:{v:g} (risk 1, target {v:g})"} for v in RR_CHOICES],
    }


@app.get("/api/atr-buffer-options")
async def atr_buffer_options() -> dict:
    labels = {
        0.05: "0.05 very tight",
        0.08: "0.08 optimized (6-mo best)",
        0.1: "0.10 doc default",
        0.15: "0.15 wide",
        0.2: "0.20 very wide",
    }
    return {
        "default": 0.08,
        "doc_default": DEFAULT_ATR_BUFFER,
        "choices": [{"value": v, "label": labels.get(v, str(v))} for v in ATR_BUFFER_CHOICES],
    }


@app.get("/api/strategies")
async def strategies() -> dict:
    from scripts.strategy_catalog import list_strategies_for_ui

    return {"strategies": list_strategies_for_ui()}


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest) -> dict:
    assert_paper_trading_only()
    if req.end < req.start:
        raise HTTPException(400, "End date must be on or after start date.")
    if not req.symbols:
        raise HTTPException(400, "Select at least one symbol.")
    rr = normalize_rr(req.rr)
    atr_buffer = normalize_atr_buffer(req.atr_buffer)
    symbols = [s.upper() for s in req.symbols]

    try:
        if req.strategy_id:
            result = execute_strategy_backtest(
                req.strategy_id,
                req.start,
                req.end,
                symbols=symbols,
                ui=True,
                rr=rr,
                atr_buffer=atr_buffer,
            )
        else:
            allowed = {"guide_baseline", "client_v1", "v5_short_only"}
            if req.preset not in allowed:
                raise HTTPException(400, f"Unknown preset: {req.preset}")
            result = execute_combined_backtest(
                req.start,
                req.end,
                preset=req.preset,
                symbols=symbols,
                ui=True,
                guide_filter=req.guide_filter,
                rr=rr,
                atr_buffer=atr_buffer,
            )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    summary = result.to_dict()
    tables = load_run_tables(result.output_dir)
    return {"summary": summary, "tables": tables}


@app.post("/api/compare-strategies")
async def compare_strategies(req: CompareStrategiesRequest) -> dict:
    assert_paper_trading_only()
    if req.end < req.start:
        raise HTTPException(400, "End date must be on or after start date.")
    rr = normalize_rr(req.rr)
    atr_buffer = normalize_atr_buffer(req.atr_buffer)

    from scripts.multi_strategy_runner import rank_strategies_with_params

    syms = [s.upper() for s in req.symbols] if req.symbols else None
    try:
        result = rank_strategies_with_params(
            req.start, req.end, syms, rr=rr, atr_buffer=atr_buffer,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    rows = result["rows"]
    if req.sort_by == "win":
        rows = sorted(rows, key=lambda r: (r["win_rate_pct"], r["net_points"]), reverse=True)
    result["rows"] = rows
    result["sort_by"] = req.sort_by
    result["rr"] = rr
    result["atr_buffer"] = atr_buffer
    return result


@app.get("/api/live/status")
async def live_status() -> dict:
    from scripts.live_paper_service import live_status as _status
    return _status()


@app.get("/api/live/events")
async def live_events(limit: int = 200) -> dict:
    from scripts.live_paper_service import live_events as _events
    return _events(limit=limit)


@app.post("/api/live/start")
async def live_start(req: LivePaperStartRequest) -> dict:
    assert_paper_trading_only()
    from scripts.live_paper_service import start_live_paper
    try:
        return start_live_paper(req.strategy_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/live/stop")
async def live_stop() -> dict:
    from scripts.live_paper_service import stop_live_paper
    return stop_live_paper()


@app.get("/api/recommended-strategy")
async def recommended_strategy() -> dict:
    from scripts.strategy_catalog import strategy_by_id
    spec = strategy_by_id("doc_optimized")
    return {
        "strategy_id": spec.id,
        "label": spec.label,
        "description": spec.description,
        "symbols": list(spec.symbols_default),
        "guide_filter": spec.guide_filter,
        "rr": spec.rr,
        "atr_buffer": spec.atr_buffer,
        "note": (
            "Best on 6-month NIFTY backtest (+27.5 pts Apr–Oct, RR 2, ATR 0.08). "
            "Last 3 months alone can be negative — always show 6-month range to client."
        ),
    }


@app.get("/api/default-range")
async def default_range(months: int = 6) -> dict:
    end = datetime.now(IST).date()
    start = end - timedelta(days=months * 30)
    return {"start": start.isoformat(), "end": end.isoformat(), "preset": "guide_baseline"}


@app.get("/api/guide-filters")
async def guide_filters() -> dict:
    from scripts.guide_filter_ranking import list_filters_for_ui

    return {"filters": list_filters_for_ui()}


@app.post("/api/compare-filters")
async def compare_filters(req: CompareFiltersRequest) -> dict:
    assert_paper_trading_only()
    if req.end < req.start:
        raise HTTPException(400, "End date must be on or after start date.")
    if not req.symbols:
        raise HTTPException(400, "Select at least one symbol.")
    rr = normalize_rr(req.rr)
    atr_buffer = normalize_atr_buffer(req.atr_buffer)

    from scripts.guide_filter_ranking import rank_guide_filters

    try:
        result = rank_guide_filters(
            req.start,
            req.end,
            [s.upper() for s in req.symbols],
            rr=rr,
            atr_buffer=atr_buffer,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    rows = result["rows"]
    if req.sort_by == "win":
        rows = sorted(rows, key=lambda r: (r["win_rate_pct"], r["net_points"]), reverse=True)
    elif req.sort_by == "balanced":
        rows = sorted(rows, key=lambda r: (r["balanced_score"], r["net_points"]), reverse=True)

    result["rows"] = rows
    result["sort_by"] = req.sort_by
    return result


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
