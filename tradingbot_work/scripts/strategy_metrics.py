"""Performance metrics for backtest and paper-trade reports."""
from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_leg_metrics(
    values: pd.Series,
    *,
    label: str = "leg",
    unit: str = "points",
) -> dict[str, Any]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return {
            "label": label,
            "unit": unit,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "win_rate_pct": 0.0,
            "net": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": None,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "max_drawdown": 0.0,
            "expectancy_per_trade": 0.0,
            "effective": False,
        }

    wins_mask = series > 0
    losses_mask = series < 0
    wins = int(wins_mask.sum())
    losses = int(losses_mask.sum())
    breakeven = int(len(series) - wins - losses)
    gross_profit = float(series[wins_mask].sum())
    gross_loss = float(-series[losses_mask].sum())
    cumulative = series.cumsum()
    drawdown = cumulative - cumulative.cummax()
    max_dd = float(-drawdown.min()) if len(drawdown) else 0.0
    win_rate = wins / len(series) * 100.0
    avg_win = float(series[wins_mask].mean()) if wins else 0.0
    avg_loss = float(series[losses_mask].mean()) if losses else 0.0
    expectancy = float(series.mean())
    pf = _safe_div(gross_profit, gross_loss)

    return {
        "label": label,
        "unit": unit,
        "trades": len(series),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_pct": round(win_rate, 1),
        "net": round(float(series.sum()), 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(pf, 2) if pf is not None else None,
        "avg_trade": round(expectancy, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "largest_win": round(float(series.max()), 2),
        "largest_loss": round(float(series.min()), 2),
        "max_drawdown": round(max_dd, 2),
        "expectancy_per_trade": round(expectancy, 2),
        "effective": bool(series.sum() > 0 and win_rate >= 35.0),
    }


def compute_daily_effectiveness(
    daily: pd.DataFrame,
    net_col: str,
    *,
    label: str = "daily",
) -> dict[str, Any]:
    if daily.empty or net_col not in daily.columns:
        return {
            "label": label,
            "trading_days": 0,
            "active_days": 0,
            "profitable_days": 0,
            "losing_days": 0,
            "flat_days": 0,
            "profitable_day_pct": 0.0,
            "avg_day_net": 0.0,
            "best_day": 0.0,
            "worst_day": 0.0,
        }
    net = pd.to_numeric(daily[net_col], errors="coerce").fillna(0.0)
    active = net[daily.get("pivot_trades", daily.get("trades", 1)) > 0] if "pivot_trades" in daily.columns else net
    profitable = int((net > 0).sum())
    losing = int((net < 0).sum())
    flat = int((net == 0).sum())
    trading_days = len(daily)
    return {
        "label": label,
        "trading_days": trading_days,
        "active_days": int((daily.get("pivot_trades", 0) > 0).sum()) if "pivot_trades" in daily.columns else trading_days,
        "profitable_days": profitable,
        "losing_days": losing,
        "flat_days": flat,
        "profitable_day_pct": round(profitable / trading_days * 100.0, 1) if trading_days else 0.0,
        "avg_day_net": round(float(net.mean()), 2),
        "best_day": round(float(net.max()), 2),
        "worst_day": round(float(net.min()), 2),
    }


def combined_effectiveness_score(pivot_metrics: dict[str, Any], pe_metrics: dict[str, Any]) -> dict[str, Any]:
    """Simple blended score for quick reading (0-100, higher is better)."""
    pivot_score = 0.0
    if pivot_metrics["trades"]:
        pf = pivot_metrics["profit_factor"] or 0.0
        pivot_score = min(100.0, max(0.0, pivot_metrics["win_rate_pct"] * 0.5 + min(pf, 2.0) * 25.0))
        if pivot_metrics["net"] > 0:
            pivot_score = min(100.0, pivot_score + 15.0)
        else:
            pivot_score = max(0.0, pivot_score - 20.0)

    pe_score = 0.0
    if pe_metrics["trades"]:
        pe_score = pe_metrics["win_rate_pct"]
        if pe_metrics["net"] > 0:
            pe_score = min(100.0, pe_score + 20.0)

    overall = round(pivot_score * 0.75 + pe_score * 0.25, 1)
    verdict = "STRONG" if overall >= 55 and pivot_metrics["net"] > 0 else (
        "MIXED" if overall >= 40 else "WEAK"
    )
    return {
        "pivot_component_score": round(pivot_score, 1),
        "pe_component_score": round(pe_score, 1),
        "overall_score": overall,
        "verdict": verdict,
    }


def compute_exit_analytics(pivot: pd.DataFrame, rr: float | None = None) -> dict[str, Any]:
    """Exit reason mix and realized R-multiples (why win% may not match RR theory)."""
    empty: dict[str, Any] = {
        "trades": 0,
        "exit_reasons": {},
        "target_hits": 0,
        "stop_hits": 0,
        "flat_exits": 0,
        "other_exits": 0,
        "avg_gross_win_r": None,
        "avg_gross_loss_r": None,
        "avg_net_win_r": None,
        "avg_net_loss_r": None,
        "avg_cost_points": 0.0,
        "breakeven_win_rate_pct": None,
        "theoretical_breakeven_win_rate_pct": None,
        "avg_win_points": 0.0,
        "avg_loss_points": 0.0,
    }
    if pivot.empty or "risk_points" not in pivot.columns:
        return empty

    risk = pd.to_numeric(pivot["risk_points"], errors="coerce")
    net = pd.to_numeric(pivot["net_pnl_points"], errors="coerce")
    valid = risk > 0
    if not valid.any():
        return empty

    net = net[valid]
    risk = risk[valid]
    net_r = net / risk
    gross = pd.to_numeric(pivot.loc[valid, "gross_pnl_points"], errors="coerce") if "gross_pnl_points" in pivot.columns else None
    gross_r = gross / risk if gross is not None else None
    cost = (
        pd.to_numeric(pivot.loc[valid, "estimated_cost_points"], errors="coerce")
        if "estimated_cost_points" in pivot.columns else None
    )

    reasons = pivot.loc[valid, "exit_reason"].astype(str).fillna("unknown")
    exit_reasons = {str(k): int(v) for k, v in reasons.value_counts().items()}
    target_hits = int(reasons.str.contains("target", case=False, na=False).sum())
    stop_hits = int(reasons.str.contains("stop", case=False, na=False).sum())
    flat_exits = int(reasons.str.contains("flat|period_end", case=False, na=False).sum())
    other_exits = int(len(reasons) - target_hits - stop_hits - flat_exits)

    wins = net > 0
    losses = net < 0
    avg_win_pts = float(net[wins].mean()) if wins.any() else 0.0
    avg_loss_pts = float(net[losses].mean()) if losses.any() else 0.0
    avg_net_win_r = round(float(net_r[wins].mean()), 2) if wins.any() else None
    avg_net_loss_r = round(float(net_r[losses].mean()), 2) if losses.any() else None
    avg_gross_win_r = round(float(gross_r[wins].mean()), 2) if gross_r is not None and wins.any() else None
    avg_gross_loss_r = round(float(gross_r[losses].mean()), 2) if gross_r is not None and losses.any() else None
    avg_cost = round(float(cost.mean()), 1) if cost is not None and cost.notna().any() else 0.0

    breakeven = None
    if wins.any() and losses.any() and avg_win_pts > 0:
        breakeven = round(abs(avg_loss_pts) / (avg_win_pts + abs(avg_loss_pts)) * 100.0, 1)

    theoretical_be = round(100.0 / (1.0 + rr), 1) if rr is not None and rr > 0 else None

    return {
        "trades": int(len(net)),
        "exit_reasons": exit_reasons,
        "target_hits": target_hits,
        "stop_hits": stop_hits,
        "flat_exits": flat_exits,
        "other_exits": max(0, other_exits),
        "avg_gross_win_r": avg_gross_win_r,
        "avg_gross_loss_r": avg_gross_loss_r,
        "avg_net_win_r": avg_net_win_r,
        "avg_net_loss_r": avg_net_loss_r,
        "avg_cost_points": avg_cost,
        "breakeven_win_rate_pct": breakeven,
        "theoretical_breakeven_win_rate_pct": theoretical_be,
        "avg_win_points": round(avg_win_pts, 1),
        "avg_loss_points": round(avg_loss_pts, 1),
    }


def exit_analytics_markdown(analytics: dict[str, Any], rr: float | None = None) -> str:
    if not analytics.get("trades"):
        return "_No pivot trades to analyze._"
    rr_label = f"1:{rr:g}" if rr else "n/a"
    be = analytics["breakeven_win_rate_pct"]
    be_text = f"{be:.1f}%" if be is not None else "n/a"
    theo = analytics["theoretical_breakeven_win_rate_pct"]
    theo_text = f"{theo:.1f}%" if theo is not None else "n/a"
    lines = [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Target RR setting | {rr_label} |",
        f"| Exits at target | {analytics['target_hits']} |",
        f"| Exits at stop | {analytics['stop_hits']} |",
        f"| Exits at flat / EOD | {analytics['flat_exits']} |",
        f"| Avg gross win (R) | {analytics['avg_gross_win_r'] or 'n/a'} |",
        f"| Avg gross loss (R) | {analytics['avg_gross_loss_r'] or 'n/a'} |",
        f"| **Avg net win (R)** | **{analytics['avg_net_win_r'] or 'n/a'}** |",
        f"| **Avg net loss (R)** | **{analytics['avg_net_loss_r'] or 'n/a'}** |",
        f"| Avg win / loss (pts) | {analytics['avg_win_points']:+.1f} / {analytics['avg_loss_points']:+.1f} |",
        f"| Avg cost per trade (pts) | {analytics['avg_cost_points']:.1f} |",
        f"| Theoretical BE win % (no costs) | {theo_text} |",
        f"| **Real BE win % (after costs)** | **{be_text}** |",
    ]
    return "\n".join(lines)


def metrics_markdown_table(metrics: dict[str, Any]) -> str:
    pf = metrics["profit_factor"]
    pf_text = f"{pf:.2f}" if pf is not None else "n/a"
    return "\n".join([
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Trades | {metrics['trades']} |",
        f"| Wins / Losses / Flat | {metrics['wins']} / {metrics['losses']} / {metrics['breakeven']} |",
        f"| **Win rate** | **{metrics['win_rate_pct']:.1f}%** |",
        f"| Net P&L | {metrics['net']:+.2f} {metrics['unit']} |",
        f"| Profit factor | {pf_text} |",
        f"| Avg per trade | {metrics['avg_trade']:+.2f} |",
        f"| Avg win / avg loss | {metrics['avg_win']:+.2f} / {metrics['avg_loss']:+.2f} |",
        f"| Max drawdown | {metrics['max_drawdown']:.2f} |",
        f"| Effective (net>0 & win%≥35) | {'Yes' if metrics['effective'] else 'No'} |",
    ])
