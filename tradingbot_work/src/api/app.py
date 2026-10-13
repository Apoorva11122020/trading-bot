from __future__ import annotations

import asyncio
import hmac
import json
import os
import threading
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.loader import load_strategy_config
from src.database import Database
from src.demo.controller import DemoAlreadyRunning, DemoController
from src.demo.sandbox_demo import DemoSafetyError, sandbox_order_execution_enabled
from src.demo.summary import build_overview
from src.options.demo import OptionDemoController
from src.options.instrument_resolver import build_resolver
from src.options.models import OptionResolutionError
from src.options.option_chain import DEMO_REFERENCE_SPOT, OptionChainService, demo_reference_spot
from src.options.selection import select_expiry
from src.options.strategy import resolve_option_contract
from src.monitoring import MonitoringSnapshot
from src.risk.controller import kill_switch_state, set_kill_switch
from src.historical_data.models import HistoricalDataError
from src.historical_data.service import HistoricalDataService
from src.paper_trading import forward_validation as fv
from src.paper_trading.shadow_validation import comparison as shadow_comparison
from src.paper_trading.signal_tracker import MARK_PREFIX, compute_stats, public_row
from src.safety.live_gate import live_trading_enabled, trading_mode
from src.signal_accuracy import paper_dashboard as pdash


IST = ZoneInfo("Asia/Kolkata")

ROOT = Path(__file__).resolve().parents[2]


def _environment() -> str:
    return os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development")).strip().lower()


_production = _environment() in {"production", "prod"}

app = FastAPI(
    title="Automated Indian Trading Bot",
    version="0.3.0",
    docs_url=None if _production else "/docs",
    redoc_url=None if _production else "/redoc",
    openapi_url=None if _production else "/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Local dashboard runs on port 3000 while FastAPI runs on port 8000.
# Only allow the local dashboard origins required for the demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.getenv("DATABASE_PATH", "data/trading.db")
CONFIG_PATH = os.getenv(
    "STRATEGY_CONFIG_PATH",
    str(ROOT / "config" / "strategy.yaml"),
)

database = Database(DB_PATH)

demo_controller = DemoController(database, config_path=CONFIG_PATH)
historical_service = HistoricalDataService(database=database)

_snapshot = MonitoringSnapshot(
    trading_mode=os.getenv("TRADING_MODE", "PAPER")
)


def _key_matches(x_api_key: str | None) -> None:
    expected = os.getenv("DASHBOARD_API_KEY", "").strip()

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Dashboard API authentication is not configured",
        )

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid dashboard API key",
        )


def _require_dashboard_auth(x_api_key: str | None) -> None:
    """Require an API key for dashboard data in production/explicit auth mode."""
    required = (
        os.getenv("DASHBOARD_API_AUTH_REQUIRED", "false")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )

    if _environment() in {"production", "prod"}:
        required = True

    if not required:
        return

    _key_matches(x_api_key)


def _require_control_auth(x_api_key: str | None) -> None:
    """State-changing endpoints always require the API key, in every environment."""
    _key_matches(x_api_key)


def _config():
    return load_strategy_config(CONFIG_PATH)


def _day_start_utc() -> datetime:
    day = datetime.now(timezone.utc).astimezone(IST).date()

    return datetime.combine(
        day,
        time(0, 0),
        tzinfo=IST,
    ).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Dashboard static files
# ---------------------------------------------------------------------------

DASHBOARD_DIR = ROOT / "dashboard"

if DASHBOARD_DIR.exists():
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(DASHBOARD_DIR)),
        name="dashboard",
    )


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        database.ping()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "trading_mode": _snapshot.trading_mode,
        "database": "ok" if db_ok else "unavailable",
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def status(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)

    data = _snapshot.as_dict()

    runtime = database.get_state("runtime_status", {}) or {}
    kill = kill_switch_state(database)

    data.update(
        trading_mode=trading_mode(),
        live_trading_enabled=live_trading_enabled(),
        broker_status=(
            "CONNECTED"
            if runtime.get("broker_connected")
            else "DISCONNECTED"
        ),
        market_data_status=runtime.get(
            "market_data_status",
            "UNKNOWN",
        ),
        bot_status=(
            "RUNNING"
            if runtime.get("running")
            else "STOPPED"
        ),
        runtime_updated_at=runtime.get("updated_at"),
        kill_switch_active=bool(kill.get("active")),
        trading_enabled=not bool(kill.get("active")),
        daily_pnl=database.realized_pnl_since(_day_start_utc()),
        positions=[
            dict(row)
            for row in database.get_positions()
        ],
    )

    data["database"] = database.counts()

    return data


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

@app.get("/api/orders")
def orders(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)

    return [
        dict(row)
        for row in database.get_orders(limit=100)
    ]


@app.get("/api/orders/{order_id}/history")
def order_history(
    order_id: str,
    x_api_key: str | None = Header(default=None),
):
    _require_dashboard_auth(x_api_key)

    return [
        dict(row)
        for row in database.get_order_history(order_id)
    ]


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

@app.get("/api/trades")
def trades(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)

    return [
        dict(row)
        for row in database.get_trades(limit=100)
    ]


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

@app.get("/api/positions")
def positions(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)

    return [
        dict(row)
        for row in database.get_positions()
    ]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@app.get("/api/events")
def events(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)

    return [
        dict(row)
        for row in database.get_events(limit=100)
    ]


# ---------------------------------------------------------------------------
# Equity
# ---------------------------------------------------------------------------

@app.get("/api/equity")
def equity(x_api_key: str | None = Header(default=None)):
    """Cumulative realized P&L from closed trades (oldest first)."""
    _require_dashboard_auth(x_api_key)

    closed = [
        dict(r)
        for r in database.get_trades(limit=1000)
        if r["status"] == "CLOSED"
    ]

    closed.sort(
        key=lambda r: r["closed_at"] or ""
    )

    curve = []
    total = 0.0

    for row in closed:
        total += float(
            row["realized_pnl"] or 0.0
        )

        curve.append(
            {
                "time": row["closed_at"],
                "cumulative_pnl": total,
            }
        )

    return curve


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

@app.get("/api/risk")
def risk(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)

    cfg = _config()
    kill = kill_switch_state(database)
    day_start = _day_start_utc()

    lock = database.get_state(
        "daily_loss_lock",
        {},
    ) or {}

    today = (
        datetime.now(timezone.utc)
        .astimezone(IST)
        .date()
        .isoformat()
    )

    return {
        "daily_realized_pnl": database.realized_pnl_since(
            day_start
        ),
        "trades_today": database.trades_opened_since(
            day_start
        ),
        "daily_loss_locked": lock.get("date") == today,
        "kill_switch_active": bool(
            kill.get("active")
        ),
        "kill_switch_reason": kill.get("reason"),
        "limits": cfg.risk.model_dump(),
        "open_positions": len(
            database.get_positions()
        ),
    }


# ---------------------------------------------------------------------------
# Paper trading (NIFTY S2)
# ---------------------------------------------------------------------------

@app.get("/api/paper-trading/status")
def paper_trading_status(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    status = database.get_state("paper_trading_status", {}) or {}
    status.setdefault("safety_banner", "PAPER TRADING ONLY | LIVE TRADING DISABLED")
    status["trading_mode"] = trading_mode()
    status["live_trading_enabled"] = live_trading_enabled()
    return status


# ---------------------------------------------------------------------------
# Live signal accuracy (read-only observation data)
# ---------------------------------------------------------------------------

def _signal_day(value: str | None) -> str:
    if not value:
        return datetime.now(IST).date().isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD") from None


@app.get("/api/signals/live")
def live_signals(date: str | None = None, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    day = _signal_day(date)
    return {"date": day, "signals": [public_row(r) for r in database.get_live_signals(day)]}


@app.get("/api/signals/live/stats")
def live_signal_stats(date: str | None = None, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    day = _signal_day(date)
    return {
        "date": day,
        "trading_mode": trading_mode(),
        "live_trading_enabled": live_trading_enabled(),
        "observation_only": True,
        **compute_stats(database.get_live_signals(day)),
    }


# ---------------------------------------------------------------------------
# Live PAPER signal validation (read-only; the two trackers stay the single source of truth)
# ---------------------------------------------------------------------------

PAPER_SIGNAL_DIR = Path(os.getenv("PAPER_SIGNAL_DIR", str(pdash.DEFAULT_SPOT_DIR)))
PAPER_PREMIUM_DIR = Path(os.getenv("PAPER_PREMIUM_DIR", str(pdash.DEFAULT_PREMIUM_DIR)))
PAPER_STREAM_INTERVAL = float(os.getenv("PAPER_STREAM_INTERVAL_SECONDS", "1"))
_paper_spot = pdash.SpotSource(PAPER_SIGNAL_DIR)
_paper_premium = pdash.PremiumSource(PAPER_PREMIUM_DIR)
_paper_cfg: pdash.DashboardConfig | None = None


def _paper_now() -> datetime:
    return datetime.now(IST)


def _paper_config() -> pdash.DashboardConfig:
    global _paper_cfg
    if _paper_cfg is None:
        _paper_cfg = pdash.DashboardConfig.load()
    return _paper_cfg


def _paper_safety() -> dict[str, Any]:
    return {**_forward_safety(), "paper_only": True, "signal_generation_only": True}


def _paper_snapshot(day: str | None = None) -> dict[str, Any]:
    spot_signals, spot_candles = _paper_spot.load()

    # The S2 paper runner publishes its heartbeat to system_state.
    # Adapt its field names for the dashboard; never invent a fresh heartbeat.
    runner_status = database.get_state("paper_trading_status", default=None)
    if isinstance(runner_status, dict):
        status = dict(runner_status)
        status["updated_at"] = status.get("updated_at")
        status["last_tick_time"] = status.get("last_market_data_at")
    else:
        # Keep the existing tracker source as a fallback when no runner state exists.
        status = _paper_spot.status()

    return pdash.build_snapshot(
        premium_rows=database.get_live_signals(), marks=database.get_states(MARK_PREFIX),
        spot_signals=spot_signals, spot_candles=spot_candles, status=status,
        premium_status=_paper_premium.status(), now=_paper_now(), cfg=_paper_config(),
        safety=_paper_safety(), day=day,
    )


def _paper_day(value: str | None) -> str | None:
    return _signal_day(value) if value else None


@app.get("/api/paper/snapshot")
def paper_snapshot(date: str | None = None, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    return _paper_snapshot(_paper_day(date))


@app.get("/api/paper/summary")
def paper_summary(date: str | None = None, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    s = _paper_snapshot(_paper_day(date))
    keys = ("generated_at", "date", "mode", "banner", "labels", "notes", "safety", "market", "summary",
            "cumulative", "daily")
    return {k: s[k] for k in keys}


@app.get("/api/paper/signals")
def paper_signals(date: str | None = None, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    s = _paper_snapshot(_paper_day(date))
    return {"date": s["date"], "mode": s["mode"], "safety": s["safety"], "signals": s["signals"]}


@app.get("/api/paper/active")
def paper_active(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    s = _paper_snapshot()
    return {"date": s["date"], "mode": s["mode"], "safety": s["safety"], "market": s["market"], "active": s["active"]}


@app.get("/api/paper/health")
def paper_health(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    s = _paper_snapshot()
    return {"generated_at": s["generated_at"], "mode": s["mode"], "market": s["market"]["state"],
            "safety": s["safety"], **s["health"]}


async def _paper_events(request: Request, day: str | None, interval: float | None = None):
    diff = pdash.StreamDiff(keepalive_after=max(1, round(15 / max(interval or PAPER_STREAM_INTERVAL, 0.1))))
    yield "retry: 3000\n\n"
    while not await request.is_disconnected():
        for frame in diff.frames(await asyncio.to_thread(_paper_snapshot, day)):
            yield frame
        await asyncio.sleep(interval if interval is not None else PAPER_STREAM_INTERVAL)


@app.get("/api/paper/stream")
async def paper_stream(request: Request, date: str | None = None, api_key: str | None = None,
                       x_api_key: str | None = Header(default=None)):
    """Server-Sent Events; EventSource cannot set headers, so the key may also be passed as ?api_key=."""
    _require_dashboard_auth(x_api_key or api_key)
    return StreamingResponse(_paper_events(request, _paper_day(date)), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# S2 forward paper validation (read-only observation data)
# ---------------------------------------------------------------------------

def _forward_safety() -> dict[str, Any]:
    return {"trading_mode": trading_mode(), "live_trading_enabled": live_trading_enabled(), "observation_only": True}


@app.get("/api/forward-validation")
def forward_validation(date: str | None = None, x_api_key: str | None = Header(default=None)):
    """Recorded forward-validation signals (all sessions, or one session with ?date=)."""
    _require_dashboard_auth(x_api_key)
    day = _signal_day(date) if date else None
    rows = database.get_forward_signals(day)
    return {"date": day, **_forward_safety(), "signals": [fv.public_row(r) for r in rows]}


@app.get("/api/forward-validation/stats")
def forward_validation_stats(date: str | None = None, x_api_key: str | None = Header(default=None)):
    """Cumulative statistics; ``today`` refers to ?date= (default: current IST session)."""
    _require_dashboard_auth(x_api_key)
    return {**_forward_safety(), **fv.forward_stats(database.get_forward_signals(), _signal_day(date))}


@app.get("/api/forward-validation/daily")
def forward_validation_daily(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    return {**_forward_safety(), "days": fv.daily_stats(database.get_forward_signals())}


# ---------------------------------------------------------------------------
# S2 shadow strategy comparison (read-only observation data)
# ---------------------------------------------------------------------------

@app.get("/api/shadow-strategies")
def shadow_strategies(date: str | None = None, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    day = _signal_day(date) if date else None
    rows = database.get_shadow_signals(day)
    return {**_forward_safety(), "date": day, "signals": rows}


@app.get("/api/shadow-strategies/stats")
def shadow_strategy_stats(date: str | None = None, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    day = _signal_day(date) if date else None
    return {**shadow_comparison(database, day)}


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

@app.get("/api/config")
def config(x_api_key: str | None = Header(default=None)):
    """Non-secret runtime configuration. Credentials are never returned."""
    _require_dashboard_auth(x_api_key)

    cfg = _config()

    return {
        "strategy": {
            "name": cfg.name,
            "exchange": cfg.exchange,
            "timeframe": cfg.timeframe,
            "indicators": cfg.indicators.model_dump(),
            "exit": cfg.exit.model_dump(),
        },
        "risk": cfg.risk.model_dump(),
        "session": cfg.session.model_dump(),
        "execution": cfg.execution.model_dump(),
        "options": cfg.options.model_dump(),
        "trading_mode": trading_mode(),
        "live_trading_enabled": live_trading_enabled(),
    }


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

@app.post("/api/kill-switch/activate")
def kill_switch_activate(
    payload: dict | None = Body(default=None),
    x_api_key: str | None = Header(default=None),
):
    _require_control_auth(x_api_key)

    reason = str(
        (payload or {}).get("reason")
        or "DASHBOARD_MANUAL"
    )[:200]

    changed = set_kill_switch(
        database,
        True,
        reason,
    )

    return {
        "kill_switch_active": True,
        "changed": changed,
    }


@app.post("/api/kill-switch/resume")
def kill_switch_resume(
    payload: dict | None = Body(default=None),
    x_api_key: str | None = Header(default=None),
):
    _require_control_auth(x_api_key)

    reason = str(
        (payload or {}).get("reason")
        or "DASHBOARD_RESUME"
    )[:200]

    changed = set_kill_switch(
        database,
        False,
        reason,
    )

    return {
        "kill_switch_active": False,
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# UI-triggered Upstox Sandbox demo
# ---------------------------------------------------------------------------

def _demo_preconditions() -> dict:
    return {
        "trading_mode": trading_mode(),
        "live_trading_enabled": live_trading_enabled(),
        "sandbox_execution_enabled": sandbox_order_execution_enabled(),
        "kill_switch_active": bool(kill_switch_state(database).get("active")),
    }


@app.post("/api/demo/start", status_code=202)
def demo_start(x_api_key: str | None = Header(default=None)):
    _require_control_auth(x_api_key)

    checks = _demo_preconditions()

    if checks["live_trading_enabled"]:
        raise HTTPException(
            status_code=403,
            detail="Demo refused: LIVE_TRADING_ENABLED must be false",
        )

    if checks["trading_mode"] != "PAPER":
        raise HTTPException(
            status_code=403,
            detail="Demo refused: TRADING_MODE must be PAPER",
        )

    if not checks["sandbox_execution_enabled"]:
        raise HTTPException(
            status_code=403,
            detail=(
                "Demo refused: Sandbox order execution is disabled "
                "(UPSTOX_RUN_SANDBOX_ORDER_TEST=true is required)"
            ),
        )

    if checks["kill_switch_active"]:
        raise HTTPException(
            status_code=409,
            detail="Demo refused: kill switch is active",
        )

    try:
        state = demo_controller.start()
    except DemoAlreadyRunning:
        raise HTTPException(
            status_code=409,
            detail="Demo already running",
        )
    except DemoSafetyError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"Demo refused: {exc}",
        )

    return {
        "demo_run_id": state["demo_run_id"],
        "status": state,
    }


@app.get("/api/demo/status")
def demo_status(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)

    data = demo_controller.status()
    data["preconditions"] = _demo_preconditions()

    return data


@app.get("/api/demo/result/{demo_run_id}")
def demo_result(
    demo_run_id: str,
    x_api_key: str | None = Header(default=None),
):
    _require_dashboard_auth(x_api_key)

    result = demo_controller.result(demo_run_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown demo run",
        )

    return result


@app.post("/api/demo/reset")
def demo_reset(x_api_key: str | None = Header(default=None)):
    """Clear the demo console. Never cancels or alters broker orders;
    an in-flight run cannot be reset."""
    _require_control_auth(x_api_key)

    try:
        return demo_controller.reset()
    except DemoAlreadyRunning:
        raise HTTPException(
            status_code=409,
            detail="Demo already running",
        )


# ---------------------------------------------------------------------------
# Options (NSE F&O): instruments, chain, contract resolution, Sandbox demo
# ---------------------------------------------------------------------------

_option_resolver = None
_option_resolver_lock = threading.Lock()


def get_option_resolver():
    """Shared resolver over the official Upstox instrument master (cached)."""
    global _option_resolver
    with _option_resolver_lock:
        if _option_resolver is None:
            _option_resolver = build_resolver(_config().options)
        return _option_resolver


option_demo_controller = OptionDemoController(
    database,
    resolver_factory=lambda: get_option_resolver(),
    config_path=CONFIG_PATH,
)

_RESOLUTION_STATUS = {
    "INSTRUMENT_DATA_UNAVAILABLE": 503,
    "INSTRUMENT_SOURCE_REFUSED": 503,
    "UNDERLYING_NOT_FOUND": 404,
    "EXPIRY_NOT_FOUND": 404,
    "STRIKE_NOT_FOUND": 404,
    "CONTRACT_NOT_FOUND": 404,
    "STRIKE_OUT_OF_RANGE": 404,
}


def _resolution_error(exc: OptionResolutionError) -> HTTPException:
    return HTTPException(
        status_code=_RESOLUTION_STATUS.get(exc.code, 422),
        detail={"code": exc.code, "message": exc.message},
    )


def _options_enabled() -> bool:
    return bool(_config().options.enabled)


class OptionDemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_type: Literal["INDEX_OPTION", "STOCK_OPTION"]
    underlying: str = Field(min_length=1, max_length=30, pattern=r"^[A-Z0-9&\-]+$")
    expiry_mode: Literal["nearest", "next", "date"] = "nearest"
    expiry_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    option_type: Literal["AUTO", "CE", "PE"] = "AUTO"
    strike_mode: Literal["ATM", "ITM", "OTM", "EXACT"] = "ATM"
    strike_offset: int = Field(default=0, ge=0, le=20)
    strike: float | None = Field(default=None, gt=0)
    signal_source: Literal["UNDERLYING", "OPTION"] = "UNDERLYING"
    spot: float | None = Field(default=None, gt=0)
    demo_trend: Literal["AUTO", "BULLISH", "BEARISH"] = "AUTO"

    @field_validator(
        "market_type", "underlying", "option_type", "strike_mode", "signal_source", "demo_trend", mode="before")
    @classmethod
    def _upper(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("expiry_mode", mode="before")
    @classmethod
    def _lower(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _consistent(self):
        if self.expiry_mode == "date" and not self.expiry_date:
            raise ValueError("expiry_date is required when expiry_mode is 'date'")
        if self.strike_mode == "EXACT" and self.strike is None:
            raise ValueError("strike is required when strike_mode is EXACT")
        if self.signal_source == "OPTION" and self.option_type == "AUTO":
            raise ValueError("signal_source OPTION needs option_type CE or PE")
        return self


@app.get("/api/options/underlyings")
def options_underlyings(
    market_type: str | None = None,
    x_api_key: str | None = Header(default=None),
):
    _require_dashboard_auth(x_api_key)
    cfg = _config().options
    try:
        resolver = get_option_resolver()
        kind = market_type.upper() if market_type else None
        if kind not in (None, "INDEX_OPTION", "STOCK_OPTION"):
            raise HTTPException(status_code=422, detail="market_type must be INDEX_OPTION or STOCK_OPTION")
        listed = resolver.underlyings(kind)
    except OptionResolutionError as exc:
        raise _resolution_error(exc)
    preferred = cfg.index_underlyings + cfg.stock_underlyings
    listed.sort(key=lambda u: (preferred.index(u["symbol"]) if u["symbol"] in preferred else len(preferred),
                               u["symbol"]))
    return {
        "enabled": cfg.enabled,
        "configured": {"index": cfg.index_underlyings, "stock": cfg.stock_underlyings},
        "underlyings": listed,
        "source": resolver.metadata,
    }


@app.get("/api/options/expiries")
def options_expiries(underlying: str, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    try:
        resolver = get_option_resolver()
        expiries = resolver.expiries(underlying)
        contract = resolver.contracts(underlying, expiries[0])[0]
    except OptionResolutionError as exc:
        raise _resolution_error(exc)
    return {
        "underlying": underlying.upper(),
        "market_type": contract.market_type.value,
        "lot_size": contract.lot_size,
        "freeze_quantity": contract.freeze_quantity,
        "expiries": [e.isoformat() for e in expiries],
    }


@app.get("/api/options/chain")
def options_chain(
    underlying: str,
    expiry: str | None = None,
    spot: float | None = None,
    x_api_key: str | None = Header(default=None),
):
    _require_dashboard_auth(x_api_key)
    try:
        resolver = get_option_resolver()
        expiries = resolver.expiries(underlying)
        chosen = select_expiry(expiries, "date" if expiry else "nearest", expiry)
        chain = OptionChainService(resolver).get_option_chain(underlying, chosen, spot=spot)
    except OptionResolutionError as exc:
        raise _resolution_error(exc)
    return chain.as_dict()


@app.get("/api/options/resolve")
def options_resolve(
    underlying: str,
    option_type: str,
    expiry_mode: str = "nearest",
    expiry_date: str | None = None,
    strike_mode: str = "ATM",
    strike_offset: int = 0,
    strike: float | None = None,
    spot: float | None = None,
    x_api_key: str | None = Header(default=None),
):
    """Resolve the exact contract the option demo would trade. A missing
    spot uses the demo reference (median listed strike), labelled as such."""
    _require_dashboard_auth(x_api_key)
    try:
        resolver = get_option_resolver()
        spot_source = "REQUEST"
        if spot is None:
            expiries = resolver.expiries(underlying)
            spot = demo_reference_spot(resolver.strikes(underlying, expiries[0]))
            spot_source = DEMO_REFERENCE_SPOT
        resolved = resolve_option_contract(
            resolver, underlying=underlying, option_type=option_type, spot=spot, expiry_mode=expiry_mode,
            expiry_date=expiry_date, strike_mode=strike_mode, strike_offset=strike_offset, exact_strike=strike)
    except OptionResolutionError as exc:
        raise _resolution_error(exc)
    return {
        "contract": resolved.contract.as_dict(),
        "expiry": resolved.expiry.isoformat(),
        "strike": resolved.strike,
        "spot": spot,
        "spot_source": spot_source,
        "source": resolver.metadata.get("source"),
    }


def _option_preconditions() -> dict:
    return {**_demo_preconditions(), "options_enabled": _options_enabled()}


@app.post("/api/options/demo/start", status_code=202)
def option_demo_start(
    request: OptionDemoRequest,
    x_api_key: str | None = Header(default=None),
):
    _require_control_auth(x_api_key)

    checks = _option_preconditions()
    if checks["live_trading_enabled"]:
        raise HTTPException(status_code=403, detail="Option demo refused: LIVE_TRADING_ENABLED must be false")
    if checks["trading_mode"] != "PAPER":
        raise HTTPException(status_code=403, detail="Option demo refused: TRADING_MODE must be PAPER")
    if not checks["options_enabled"]:
        raise HTTPException(status_code=403, detail="Option demo refused: options.enabled is false")
    if not checks["sandbox_execution_enabled"]:
        raise HTTPException(
            status_code=403,
            detail=(
                "Option demo refused: Sandbox order execution is disabled "
                "(UPSTOX_RUN_SANDBOX_ORDER_TEST=true is required)"
            ),
        )
    if checks["kill_switch_active"]:
        raise HTTPException(status_code=409, detail="Option demo refused: kill switch is active")

    try:
        state = option_demo_controller.start(**request.model_dump())
    except DemoAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except DemoSafetyError as exc:
        raise HTTPException(status_code=403, detail=f"Option demo refused: {exc}")

    return {"demo_run_id": state["demo_run_id"], "status": state}


@app.get("/api/options/demo/status")
def option_demo_status(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    data = option_demo_controller.status()
    data["preconditions"] = _option_preconditions()
    return data


@app.get("/api/options/demo/result/{demo_run_id}")
def option_demo_result(demo_run_id: str, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    result = option_demo_controller.result(demo_run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown option demo run")
    return result


@app.post("/api/options/demo/reset")
def option_demo_reset(x_api_key: str | None = Header(default=None)):
    _require_control_auth(x_api_key)
    try:
        return option_demo_controller.reset()
    except DemoAlreadyRunning:
        raise HTTPException(status_code=409, detail="Option demo already running")


# ---------------------------------------------------------------------------
# Dashboard presentation of persisted demo runs (read-only)
# ---------------------------------------------------------------------------

@app.get("/api/demo/overview")
def demo_overview(x_api_key: str | None = Header(default=None)):
    """Latest demo runs, demo orders, statistics and audit events, built only
    from persisted results and database records. Never returns credentials."""
    _require_dashboard_auth(x_api_key)
    return build_overview(
        database,
        running={"equity": demo_controller.running, "option": option_demo_controller.running},
        trading_mode=trading_mode(),
        live_trading_enabled=live_trading_enabled(),
    )


# ---------------------------------------------------------------------------
# Historical options data + strategy research (read-only market data)
# ---------------------------------------------------------------------------


def _research_preconditions() -> None:
    if live_trading_enabled():
        raise HTTPException(status_code=403, detail="Refused: LIVE_TRADING_ENABLED must be false")
    if trading_mode() != "PAPER":
        raise HTTPException(status_code=403, detail="Refused: TRADING_MODE must be PAPER")


class HistoricalDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    underlyings: list[str] = Field(min_length=1)
    from_date: str
    to_date: str
    include_options: bool = True
    include_futures: bool = True

    @field_validator("underlyings")
    @classmethod
    def normalise_underlyings(cls, values: list[str]) -> list[str]:
        return [v.strip().upper() for v in values]


class OptionsBacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    underlyings: list[str] = Field(min_length=1)
    from_date: str
    to_date: str
    strategies: list[str] | None = None

    @field_validator("underlyings")
    @classmethod
    def normalise_underlyings(cls, values: list[str]) -> list[str]:
        return [v.strip().upper() for v in values]


def _historical_http_error(exc: HistoricalDataError) -> HTTPException:
    status = 400
    if exc.code in {"AUTH_FAILED", "CREDENTIALS_MISSING"}:
        status = 401
    elif exc.code in {"LIVE_TRADING_REFUSED", "PERMISSION_DENIED"}:
        status = 403
    elif exc.code in {"DOWNLOAD_RUNNING", "BACKTEST_RUNNING", "RUN_IN_PROGRESS"}:
        status = 409
    elif exc.code == "RUN_NOT_FOUND":
        status = 404
    return HTTPException(status_code=status, detail={"code": exc.code, "message": exc.message})


@app.post("/api/historical/options/download", status_code=202)
def historical_options_download(
    payload: HistoricalDownloadRequest = Body(...),
    x_api_key: str | None = Header(default=None),
):
    _require_control_auth(x_api_key)
    _research_preconditions()
    try:
        return historical_service.start_download(
            payload.underlyings,
            date.fromisoformat(payload.from_date),
            date.fromisoformat(payload.to_date),
            include_options=payload.include_options,
            include_futures=payload.include_futures,
        )
    except HistoricalDataError as exc:
        raise _historical_http_error(exc)


@app.get("/api/historical/options/status")
def historical_options_status(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    return historical_service.status()


@app.get("/api/historical/options/summary")
def historical_options_summary(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    return historical_service.summary()


@app.post("/api/backtest/options/run", status_code=202)
def backtest_options_run(
    payload: OptionsBacktestRequest = Body(...),
    x_api_key: str | None = Header(default=None),
):
    _require_control_auth(x_api_key)
    _research_preconditions()
    try:
        return historical_service.start_backtest(
            payload.underlyings,
            date.fromisoformat(payload.from_date),
            date.fromisoformat(payload.to_date),
            strategies=payload.strategies,
        )
    except HistoricalDataError as exc:
        raise _historical_http_error(exc)


@app.get("/api/backtest/options/status/{run_id}")
def backtest_options_status(run_id: str, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    try:
        return historical_service.backtest_status(run_id)
    except HistoricalDataError as exc:
        raise _historical_http_error(exc)


@app.get("/api/backtest/options/result/{run_id}")
def backtest_options_result(run_id: str, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    try:
        return historical_service.backtest_result(run_id)
    except HistoricalDataError as exc:
        raise _historical_http_error(exc)


@app.get("/api/backtest/options/strategies")
def backtest_options_strategies(x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    return {"strategies": historical_service.strategies()}


@app.get("/api/backtest/options/signals/{run_id}")
def backtest_options_signals(run_id: str, x_api_key: str | None = Header(default=None)):
    _require_dashboard_auth(x_api_key)
    try:
        return {"run_id": run_id, "signals": historical_service.backtest_signals(run_id)}
    except HistoricalDataError as exc:
        raise _historical_http_error(exc)