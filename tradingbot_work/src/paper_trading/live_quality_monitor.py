"""Live quality-gate performance metrics and rolling target monitor (PAPER, REPORTING ONLY).

Nothing here changes a strategy, a threshold or the gate. Accuracy is always wins / resolved; pending and
data-unavailable signals are never counted as losses.
"""
from __future__ import annotations

from typing import Any, Sequence

from src.backtest import s2_1dte_improvement as imp

RESOLVED = (imp.WIN, imp.LOSS, imp.AMBIGUOUS, imp.SCRATCH)
LOT_SIZE = 65
WINDOWS = (10, 20, 30)

ACHIEVED, BELOW_TARGET, INSUFFICIENT_SAMPLE = "ACHIEVED", "BELOW TARGET", "INSUFFICIENT SAMPLE"
BELOW_70_TARGET, ROLLING_INSUFFICIENT, TARGET_MET, PF_BELOW_1 = (
    "BELOW_70_TARGET", "INSUFFICIENT_SAMPLE", "TARGET_MET", "PF_BELOW_1")


def is_resolved(row: dict) -> bool:
    return row.get("decision") == imp.ACCEPTED and row.get("status") in RESOLVED


def trade_pnl(row: dict) -> float:
    """Realised P&L; an ambiguous candle (stop and target in one bar) is booked as the stop."""
    if row.get("status") == imp.AMBIGUOUS and row.get("pnl") is None:
        if row.get("stop_loss") is not None and row.get("entry_price") is not None:
            return (float(row["stop_loss"]) - float(row["entry_price"])) * LOT_SIZE
    return float(row.get("pnl") or 0.0)


def accuracy(wins: int, resolved: int) -> float | None:
    return wins / resolved if resolved > 0 else None


def profit_factor(pnls: Sequence[float]) -> float | None:
    """Gross profit / gross loss; None (N/A) when there is no gross loss."""
    gross_loss = -sum(x for x in pnls if x < 0)
    if gross_loss <= 0:
        return None
    return sum(x for x in pnls if x > 0) / gross_loss


def max_drawdown(pnls: Sequence[float]) -> float:
    eq = peak = dd = 0.0
    for x in pnls:
        eq += x
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return round(dd, 2)


def max_losing_streak(rows: Sequence[dict]) -> int:
    best = cur = 0
    for r in rows:
        cur = 0 if r["status"] == imp.WIN else cur + 1
        best = max(best, cur)
    return best


def _pf_ok(pf: float | None, pnls: Sequence[float]) -> bool:
    if pf is not None:
        return pf >= 1.0
    return sum(pnls) > 0


def perf(rows: Sequence[dict]) -> dict[str, Any]:
    """Generated / accepted / resolved / pending metrics for stored LIVE_QUALITY_GATE_V1 rows."""
    rows = sorted(rows, key=lambda r: (r["session_date"], r["timestamp"]))
    accepted = [r for r in rows if r.get("decision") == imp.ACCEPTED]
    resolved = [r for r in accepted if r["status"] in RESOLVED]
    wins = sum(r["status"] == imp.WIN for r in resolved)
    pnls = [trade_pnl(r) for r in resolved]
    pf = profit_factor(pnls)
    return {
        "generated": len(rows), "accepted": len(accepted), "skipped": len(rows) - len(accepted),
        "resolved": len(resolved), "wins": wins, "losses": len(resolved) - wins,
        "ambiguous": sum(r["status"] == imp.AMBIGUOUS for r in resolved),
        "pending": sum(r["status"] == imp.PENDING for r in accepted),
        "data_unavailable": sum(r["status"] == imp.UNAVAILABLE for r in accepted),
        "accuracy": accuracy(wins, len(resolved)), "profit_factor": pf,
        "gross_profit": round(sum(x for x in pnls if x > 0), 2),
        "gross_loss": round(-sum(x for x in pnls if x < 0), 2),
        "pnl": round(sum(pnls), 2), "max_drawdown": max_drawdown(pnls),
        "max_losing_streak": max_losing_streak(resolved),
    }


def target_status(m: dict, min_samples: int = 30, target: float = 0.70) -> str:
    """ACHIEVED only with resolved >= min_samples, accuracy >= target and PF >= 1."""
    if m["resolved"] < min_samples:
        return INSUFFICIENT_SAMPLE
    if m["accuracy"] is not None and m["accuracy"] >= target and _pf_ok(m["profit_factor"], [m["pnl"]]):
        return ACHIEVED
    return BELOW_TARGET


def rolling_status(m: dict, min_samples: int = 30, target: float = 0.70) -> str:
    if m["resolved"] == 0:
        return ROLLING_INSUFFICIENT
    if m["accuracy"] < target:
        return BELOW_70_TARGET
    if m["resolved"] < min_samples:
        return ROLLING_INSUFFICIENT
    return TARGET_MET if _pf_ok(m["profit_factor"], [m["pnl"]]) else PF_BELOW_1


def rolling(rows: Sequence[dict], windows: Sequence[int] = WINDOWS, min_samples: int = 30,
            target: float = 0.70) -> dict[str, dict[str, Any]]:
    """Last-N resolved trades. Reporting only: these values never change the strategy or the gate."""
    resolved = sorted((r for r in rows if is_resolved(r)), key=lambda r: (r["session_date"], r["timestamp"]))
    out = {}
    for n in windows:
        last = resolved[-n:]
        m = perf(last)
        out[f"last_{n}"] = {"window": n, "resolved": m["resolved"], "wins": m["wins"], "losses": m["losses"],
                            "accuracy": m["accuracy"], "profit_factor": m["profit_factor"], "pnl": m["pnl"],
                            "max_drawdown": m["max_drawdown"], "status": rolling_status(m, min_samples, target),
                            "complete_window": m["resolved"] == n}
    return out
