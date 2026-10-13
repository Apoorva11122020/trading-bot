"""Read-only dashboard summary of persisted demo runs.

Everything here is derived from records already written by the demo
controllers, the OrderManager and the trade journal. Nothing is simulated:
missing values stay ``None`` and P&L is only reported from closed trades.
"""
from __future__ import annotations

import json
from typing import Any

EQUITY_RESULT_PREFIX = "demo_result:"
OPTION_RESULT_PREFIX = "option_demo_result:"

DEMO_EVENT_TYPES = (
    "DEMO_STARTED", "OPTION_DEMO_STARTED",
    "MARKET_DATA_READY", "SIGNAL_GENERATED", "OPTION_RESOLVED", "RISK_VALIDATED",
    "ORDER_SUBMITTED", "ORDER_VERIFIED", "ORDER_CANCELLED",
    "DEMO_COMPLETED", "OPTION_DEMO_COMPLETED", "DEMO_FAILED", "OPTION_DEMO_FAILED",
)

PNL_UNAVAILABLE = "N/A — Sandbox orders were cancelled without fills."
PERFORMANCE_NOTE = "Performance analytics will become available from filled paper/backtest trades."

_REJECTED = {"REJECTED", "FAILED"}


def _option_action(option_type: str | None) -> str | None:
    return {"CE": "BUY CALL", "PE": "BUY PUT"}.get(str(option_type or "").upper())


def normalise_run(result: dict) -> dict:
    """One shape for equity and option results, using only stored fields."""
    option = result.get("demo_type") == "OPTION" or str(result.get("demo_run_id", "")).startswith("OPTDEMO-")
    error = result.get("error") or {}
    if option:
        run = {
            "demo_type": "OPTION",
            "market_type": result.get("market_type"),
            "underlying": result.get("underlying"),
            "instrument_key": result.get("instrument_key"),
            "contract": result.get("trading_symbol"),
            "signal": result.get("action") or _option_action(result.get("option_type")),
            "side": "BUY" if result.get("action") or result.get("option_type") else None,
            "entry": result.get("entry_premium"),
            "entry_basis": "PREMIUM",
            "quantity": result.get("quantity"),
            "lots": result.get("lots"),
            "lot_size": result.get("lot_size"),
            "expiry": result.get("expiry"),
            "strike": result.get("strike"),
            "option_type": result.get("option_type"),
            "spot": result.get("spot"),
            "spot_source": result.get("spot_source"),
            "candles": None,
            "data_label": (result.get("data_labels") or {}).get("underlying"),
            "signal_reason": result.get("signal_reason"),
            "resolution_status": (result.get("option_resolution") or {}).get("status"),
        }
    else:
        run = {
            "demo_type": "EQUITY",
            "market_type": "EQUITY",
            "underlying": result.get("strategy_symbol"),
            "instrument_key": result.get("broker_instrument"),
            "contract": result.get("strategy_symbol"),
            "signal": result.get("signal_side"),
            "side": result.get("signal_side"),
            "entry": result.get("entry_price"),
            "entry_basis": "PRICE",
            "quantity": result.get("requested_quantity") or result.get("quantity"),
            "lots": None,
            "lot_size": None,
            "expiry": None,
            "strike": None,
            "option_type": None,
            "spot": None,
            "spot_source": None,
            "candles": result.get("candles"),
            "data_label": "PAPER / DEMO",
            "signal_reason": result.get("risk_reason"),
            "resolution_status": None,
        }
    run.update(
        demo_run_id=result.get("demo_run_id"),
        stage=result.get("stage"),
        started_at=result.get("started_at"),
        finished_at=result.get("finished_at"),
        signal_id=result.get("signal_id"),
        stop_loss=result.get("stop_loss"),
        target=result.get("target"),
        conditions=result.get("conditions"),
        risk_status=result.get("risk_status"),
        risk_reason=result.get("risk_reason"),
        broker=result.get("broker") or "UPSTOX SANDBOX",
        order_id=result.get("sandbox_order_id"),
        order_type=result.get("order_type"),
        submitted_status=result.get("submitted_status"),
        broker_order_status=result.get("broker_order_status"),
        final_status=result.get("final_cancellation_status"),
        execution_result=result.get("execution_result"),
        error={"headline": error.get("headline"), "message": error.get("message"),
               "stage": error.get("stage"), "code": error.get("code")} if error else None,
        trading_mode=result.get("trading_mode"),
        live_trading_enabled=bool(result.get("live_trading_enabled")),
    )
    run["timeline"] = build_timeline(run)
    return run


def build_timeline(run: dict) -> list[dict]:
    """Execution steps marked done only when the stored result proves it."""
    option = run["demo_type"] == "OPTION"
    submitted = run.get("submitted_status")
    steps: list[tuple[str, str, bool, str | None]] = [
        ("MARKET_DATA", "Market Data",
         (run.get("spot") is not None) if option else bool(run.get("candles")),
         (f"{run.get('underlying')} · reference spot {run['spot']:g}" if option and run.get("spot") is not None
          else f"{run['candles']} candles" if run.get("candles") else None)),
        ("STRATEGY", "Strategy", bool(run.get("signal")), run.get("signal_reason") if run.get("signal") else None),
        ("SIGNAL", "Signal", bool(run.get("signal")), run.get("signal")),
    ]
    if option:
        # Contract resolution precedes risk sizing in the option flow.
        steps.append(("CONTRACT", "Contract Resolution", bool(run.get("instrument_key")),
                      run.get("contract")))
    steps += [
        ("RISK", "Risk", run.get("risk_status") == "ALLOWED", run.get("risk_status")),
        ("ORDER_SUBMITTED", "Order Submitted", bool(submitted) and submitted not in _REJECTED,
         run.get("order_id") if submitted else None),
        ("BROKER_VERIFIED", "Broker Verified", bool(run.get("broker_order_status")),
         run.get("broker_order_status")),
        ("ORDER_CANCELLED", "Order Cancelled", run.get("final_status") == "CANCELLED",
         run.get("final_status")),
    ]

    failed = run.get("execution_result") != "SUCCESS" and bool(run.get("error"))
    timeline, marked = [], False
    for key, label, done, detail in steps:
        if done:
            state = "done"
        elif failed and not marked:
            state, marked = "failed", True
            detail = (run.get("error") or {}).get("message") or detail
        else:
            state = "pending"
        timeline.append({"key": key, "label": label, "state": state, "detail": detail})
    success = run.get("execution_result") == "SUCCESS"
    timeline.append({
        "key": "FINAL", "label": "Final",
        "state": "done" if success else "failed" if run.get("execution_result") else "pending",
        "detail": "DEMO COMPLETED" if success else (run.get("error") or {}).get("headline") or run.get("stage"),
    })
    return timeline


def _sort_key(run: dict) -> str:
    return str(run.get("finished_at") or run.get("started_at") or "")


def _event_row(row) -> dict:
    data = dict(row)
    try:
        meta = json.loads(data.pop("metadata_json", None) or "{}")
    except (TypeError, ValueError):
        meta = {}
    data["demo_run_id"] = meta.get("demo_run_id")
    data["order_id"] = meta.get("order_id")
    data["status"] = meta.get("status")
    return data


def build_overview(database, *, running: dict[str, bool], trading_mode: str, live_trading_enabled: bool) -> dict:
    results = database.get_states(EQUITY_RESULT_PREFIX) + database.get_states(OPTION_RESULT_PREFIX)
    runs = sorted((normalise_run(r) for r in results if isinstance(r, dict)), key=_sort_key, reverse=True)
    latest = runs[0] if runs else None

    orders = [dict(o) for o in database.get_all_orders()]
    demo_order_ids = {r["order_id"] for r in runs if r.get("order_id")}
    demo_orders = [o for o in orders if o.get("order_id") in demo_order_ids]
    positions = [dict(p) for p in database.get_positions()]
    trades = [dict(t) for t in database.get_trades(limit=10000)]
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    filled = [o for o in orders if str(o.get("status")).upper() in ("FILLED", "PARTIALLY_FILLED")]

    stats = {
        "demo_runs": len(runs),
        "completed_demo_runs": sum(1 for r in runs if r.get("execution_result") == "SUCCESS"),
        "failed_demo_runs": sum(1 for r in runs if r.get("execution_result") not in (None, "SUCCESS")),
        "demo_orders": len(demo_orders),
        "cancelled_orders": sum(1 for o in orders if str(o.get("status")).upper() == "CANCELLED"),
        "cancelled_demo_orders": sum(1 for o in demo_orders if str(o.get("status")).upper() == "CANCELLED"),
        "filled_orders": len(filled),
        "total_orders": len(orders),
        "open_positions": len(positions),
        "total_trades": len(trades),
        "closed_trades": len(closed),
        "realized_pnl": sum(float(t.get("realized_pnl") or 0.0) for t in closed) if closed else None,
        "pnl_note": None if closed else PNL_UNAVAILABLE,
        "performance_note": PERFORMANCE_NOTE,
    }

    events = []
    for event_type in DEMO_EVENT_TYPES:
        events.extend(_event_row(e) for e in database.get_events(limit=200, event_type=event_type))
    events.sort(key=lambda e: str(e.get("timestamp") or ""), reverse=True)
    latest_events = [e for e in events if latest and e.get("demo_run_id") == latest["demo_run_id"]]
    if latest and latest.get("order_id"):
        latest_events += [
            e for e in (_event_row(r) for r in database.get_events(limit=500, event_type="ORDER"))
            if e.get("order_id") == latest["order_id"]
        ]

    order_labels = {
        r["order_id"]: {"demo_run_id": r["demo_run_id"], "market_type": r["market_type"],
                        "underlying": r["underlying"], "contract": r["contract"]}
        for r in runs if r.get("order_id")
    }

    return {
        "latest": latest,
        "latest_equity": next((r for r in runs if r["demo_type"] == "EQUITY"), None),
        "latest_option": next((r for r in runs if r["demo_type"] == "OPTION"), None),
        "runs": runs[:20],
        "stats": stats,
        "latest_events": sorted(latest_events, key=lambda e: str(e.get("timestamp") or "")),
        "order_labels": order_labels,
        "running": running,
        "safety": {"trading_mode": trading_mode, "live_trading_enabled": live_trading_enabled},
    }
