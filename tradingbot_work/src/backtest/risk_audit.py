"""Production risk-control audit for NIFTY S2 research / paper trading."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from src.backtest.options_engine import DAILY_LOSS, OptionsResearchEngine
from src.backtest.strategy_comparator import compare_strategies
from src.risk.risk_manager import RiskManager


@dataclass
class RiskAuditReport:
    underlying: str
    strategy_id: str
    source_run_id: str
    analysis_run_id: str
    from_date: str
    to_date: str
    trading_mode: str
    live_trading_enabled: bool
    client_requirements: dict[str, Any]
    implementation_summary: dict[str, Any]
    quantity_calculation: dict[str, Any]
    per_trade_risk: dict[str, Any]
    daily_loss_limit: dict[str, Any]
    interpretation: dict[str, Any]
    losses_above_limit_explanation: list[str]
    meets_client_requirement: bool
    gaps_and_residual_risks: list[str]
    code_changes: list[str]
    replay_metrics: dict[str, Any] = field(default_factory=dict)
    replay_trade_audit: list[dict[str, Any]] = field(default_factory=list)
    replay_daily_audit: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_trade_risk(trade: dict, rm: RiskManager, account_equity: float) -> dict[str, Any]:
    from src.models.risk import AccountState

    entry = float(trade.get("entry_price", 0))
    stop = float(trade.get("stop_loss", 0))
    qty = int(trade.get("quantity", 0))
    risk_per_unit = max(0.0, entry - stop)
    planned_max_loss = risk_per_unit * qty
    actual_pnl = float(trade.get("pnl", 0))
    account = AccountState(equity=account_equity)
    sizing = rm.calculate_option_quantity(account, entry, stop, 65)
    return {
        "signal_time": trade.get("signal_time"),
        "entry_time": trade.get("entry_time"),
        "quantity": qty,
        "entry_premium": entry,
        "stop_premium": stop,
        "risk_per_unit": risk_per_unit,
        "planned_max_loss_at_stop": planned_max_loss,
        "actual_pnl": actual_pnl,
        "exceeded_planned_risk": actual_pnl < -planned_max_loss - 1e-6,
        "fresh_account_max_loss_cap": sizing.max_loss,
        "fresh_account_quantity": sizing.quantity,
    }


def build_implementation_summary(rm: RiskManager, research_config, strategy_config) -> dict[str, Any]:
    from src.models.risk import AccountState

    account = AccountState(equity=float(research_config.risk.initial_equity))
    return {
        "risk_manager": "src.risk.risk_manager.RiskManager",
        "daily_loss_limit_inr": rm.daily_loss_limit(account),
        "gap_buffer_percent": float(getattr(strategy_config.risk, "max_loss_gap_buffer_percent", 0)),
        "gap_buffer_multiplier": rm.gap_loss_buffer_multiplier(),
        "risk_per_trade_percent": float(research_config.risk.risk_per_trade_percent),
        "per_trade_budget_inr": account.equity * research_config.risk.risk_per_trade_percent / 100,
        "lot_rounding": "floor(raw_quantity / lot_size) * lot_size",
        "paper_trading_only": True,
    }


def analyze_replay(
    result: dict,
    *,
    rm: RiskManager,
    initial_equity: float,
    daily_limit_inr: float,
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    trades = list(result.get("trades", []))
    skipped = list(result.get("skipped", []))
    trade_rows = [_audit_trade_risk(t, rm, initial_equity) for t in trades]

    daily: dict[str, dict[str, Any]] = {}
    for trade in trades:
        day = str(trade.get("entry_time", ""))[:10]
        if not day:
            continue
        row = daily.setdefault(day, {"date": day, "trades": 0, "realized_pnl": 0.0, "worst_trade": 0.0})
        pnl = float(trade.get("pnl", 0))
        row["trades"] += 1
        row["realized_pnl"] += pnl
        row["worst_trade"] = min(row["worst_trade"], pnl)

    daily_rows = []
    for row in sorted(daily.values(), key=lambda r: r["date"]):
        row["breached_daily_cap"] = row["realized_pnl"] <= -daily_limit_inr - 1e-6
        row["within_daily_cap"] = not row["breached_daily_cap"]
        daily_rows.append(row)

    planned_breaches = [r for r in trade_rows if r["planned_max_loss_at_stop"] > daily_limit_inr + 1e-6]
    actual_single_breaches = [r for r in trade_rows if r["actual_pnl"] < -daily_limit_inr - 1e-6]
    gap_breaches = [r for r in trade_rows if r["exceeded_planned_risk"]]
    daily_breaches = [r for r in daily_rows if r["breached_daily_cap"]]

    metrics = {
        "resolved_trades": len(trades),
        "signals_skipped_daily_loss": sum(1 for s in skipped if s.get("reason") == DAILY_LOSS),
        "planned_max_loss_above_daily_cap_count": len(planned_breaches),
        "actual_single_trade_loss_above_daily_cap_count": len(actual_single_breaches),
        "trades_exceeding_planned_stop_risk_count": len(gap_breaches),
        "days_realized_loss_above_daily_cap": len(daily_breaches),
        "max_single_trade_loss": min((float(t.get("pnl", 0)) for t in trades), default=0.0),
        "max_daily_realized_loss": min((r["realized_pnl"] for r in daily_rows), default=0.0),
        "net_pnl": sum(float(t.get("pnl", 0)) for t in trades),
    }
    return metrics, trade_rows, daily_rows


def run_risk_audit(
    *,
    service,
    source_run_id: str,
    comparison_path: Path,
    underlying: str = "NIFTY",
    strategy_id: str = "S2",
) -> RiskAuditReport:
    from src.backtest.robustness import coverage_for, load_final_comparison, row_for
    from src.safety.live_gate import live_trading_enabled, trading_mode
    import uuid

    final = load_final_comparison(comparison_path)
    if final.get("run_id") != source_run_id:
        raise ValueError(f"Expected run {source_run_id}, found {final.get('run_id')}")

    row = row_for(final, underlying, strategy_id)
    from_date = date.fromisoformat(row["from_date"])
    to_date = date.fromisoformat(row["to_date"])
    analysis_run_id = uuid.uuid4().hex[:12]

    candles = service.read_underlying(underlying, from_date, to_date)
    candles["symbol"] = underlying.upper()
    contracts = service._contracts_from_store(underlying)
    premium = service._premium_history(underlying, from_date, to_date)

    engine = OptionsResearchEngine(
        service.research_config,
        service.strategy_config,
        contracts,
        premium,
        run_id=analysis_run_id,
        master_resolver=service._research_resolver,
    )
    rm = engine.risk_manager
    result = engine.run(candles, underlying, strategy_id, from_date, to_date).as_dict()

    daily_limit = float(service.research_config.risk.max_daily_loss_inr)
    replay_metrics, trade_audit, daily_audit = analyze_replay(
        result,
        rm=rm,
        initial_equity=float(service.research_config.risk.initial_equity),
        daily_limit_inr=daily_limit,
    )

    losses_above = [
        "Before the fix, per-trade risk used 0.5% of equity (~₹5,000 on ₹10L) while the ₹3,000 rule applied only to blocking subsequent entries after realized loss.",
        "Multiple losing trades could occur on the same day before cumulative realized loss crossed ₹3,000.",
        "Gap/slippage and trailing-stop exits can realize losses beyond the initial stop-distance risk (R > 1).",
        "The daily cap blocked new entries but did not resize positions to the remaining daily budget.",
    ]

    meets = replay_metrics["days_realized_loss_above_daily_cap"] == 0
    gaps = []
    if replay_metrics["trades_exceeding_planned_stop_risk_count"] > 0:
        gaps.append(
            f"{replay_metrics['trades_exceeding_planned_stop_risk_count']} trades still realized beyond "
            "planned stop risk due to intrabar gaps/trailing stops; gap buffer sizing reduces but cannot eliminate this."
        )

    report = RiskAuditReport(
        underlying=underlying,
        strategy_id=strategy_id,
        source_run_id=source_run_id,
        analysis_run_id=analysis_run_id,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        trading_mode=trading_mode(),
        live_trading_enabled=live_trading_enabled(),
        client_requirements={
            "underlying": "NIFTY index options",
            "timeframe": "5m",
            "trading_start": "09:15",
            "stop_new_entries": "14:30",
            "max_daily_loss_inr": daily_limit,
            "paper_trading_only": True,
            "live_trading_disabled": True,
        },
        implementation_summary=build_implementation_summary(rm, service.research_config, service.strategy_config),
        quantity_calculation={
            "formula": "min(per_trade_budget_inr, remaining_daily_budget_inr) / (risk_per_unit * gap_buffer)",
            "per_trade_budget_inr": "equity * risk_per_trade_percent / 100",
            "remaining_daily_budget_inr": "max(0, max_daily_loss_inr + daily_realized_pnl - open_position_max_loss)",
            "raw_quantity": "floor(effective_budget / effective_risk_per_unit)",
            "final_quantity": "floor(raw_quantity / lot_size) * lot_size",
            "code": "RiskManager.calculate_option_quantity",
        },
        per_trade_risk={
            "planned_max_loss": "(entry_premium - stop_premium) * quantity",
            "worst_case_check": "planned_max_loss * gap_buffer_multiplier",
            "validation": "validate_option_entry blocks when daily_pnl - open_risk - worst_case < -max_daily_loss_inr",
            "code": "RiskManager.validate_option_entry",
        },
        daily_loss_limit={
            "tracking": "realized P&L per IST trading day (account.daily_pnl / daily_pnl in backtest)",
            "open_position_risk": "open_position_max_loss subtracted from remaining budget before sizing",
            "trigger": "remaining_daily_loss_budget <= 0 OR validate_option_entry failure",
            "blocks_subsequent_entries": True,
            "resets_next_trading_day": True,
            "paper_controller": "RiskController.daily_loss_locked uses max_daily_loss_inr when configured",
            "research_engine": "OptionsResearchEngine skips signals with reason DAILY_LOSS_LIMIT_HIT",
        },
        interpretation={
            "client_requirement": "₹3,000 maximum daily loss (realized)",
            "previous_behavior": "A only partially — blocked entries after realized <= -3000, but per-trade risk and cumulative intraday loss were not capped",
            "current_behavior": "A with conservative entry sizing — per-trade planned stop risk capped to remaining daily INR budget; entries blocked once budget exhausted",
            "not_max_risk_per_trade_alone": "Per-trade risk is the lesser of per-trade percent budget and remaining daily INR budget",
        },
        losses_above_limit_explanation=losses_above,
        meets_client_requirement=meets,
        gaps_and_residual_risks=gaps,
        code_changes=[
            "config/strategy.yaml: max_daily_loss_inr: 3000, max_loss_gap_buffer_percent: 10",
            "src/config/loader.py: RiskConfig.max_daily_loss_inr, max_loss_gap_buffer_percent",
            "src/risk/risk_manager.py: INR daily budget, remaining budget sizing, gap buffer validation",
            "src/risk/controller.py: daily_loss_limit prefers max_daily_loss_inr",
            "src/backtest/options_engine.py: research risk config + remaining budget gate",
        ],
        replay_metrics=replay_metrics,
        replay_trade_audit=trade_audit[:20],
        replay_daily_audit=daily_audit,
    )
    return report


def write_risk_audit_reports(report: RiskAuditReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{report.underlying}_{report.strategy_id}_risk_audit.json"
    md_path = out_dir / f"{report.underlying}_{report.strategy_id}_risk_audit.md"
    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    lines = [
        f"# {report.underlying} {report.strategy_id} Production Risk-Control Audit",
        "",
        f"**Source run:** `{report.source_run_id}`  ",
        f"**Replay run:** `{report.analysis_run_id}`  ",
        f"**Mode:** {report.trading_mode} (live={report.live_trading_enabled})",
        "",
        "## Client requirements",
        "",
    ]
    for key, value in report.client_requirements.items():
        lines.append(f"- **{key}:** {value}")
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- **Requirement:** {report.interpretation['client_requirement']}",
        f"- **Previous behavior:** {report.interpretation['previous_behavior']}",
        f"- **Current behavior:** {report.interpretation['current_behavior']}",
        "",
        "## Why historical losses exceeded ₹3,000",
        "",
    ])
    for item in report.losses_above_limit_explanation:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Quantity calculation",
        "",
        f"- Formula: `{report.quantity_calculation['formula']}`",
        f"- Code: `{report.quantity_calculation['code']}`",
        "",
        "## Daily ₹3,000 loss limit",
        "",
    ])
    for key, value in report.daily_loss_limit.items():
        lines.append(f"- **{key}:** {value}")
    lines.extend([
        "",
        "## Paper replay (post-fix)",
        "",
        f"- Resolved trades: {report.replay_metrics.get('resolved_trades')}",
        f"- Net P&L: {report.replay_metrics.get('net_pnl'):,.2f}",
        f"- Max single-trade loss: {report.replay_metrics.get('max_single_trade_loss'):,.2f}",
        f"- Max daily realized loss: {report.replay_metrics.get('max_daily_realized_loss'):,.2f}",
        f"- Days above daily cap: {report.replay_metrics.get('days_realized_loss_above_daily_cap')}",
        f"- Planned max loss above cap (entries): {report.replay_metrics.get('planned_max_loss_above_daily_cap_count')}",
        f"- Signals skipped (daily loss): {report.replay_metrics.get('signals_skipped_daily_loss')}",
        "",
        f"## Meets client requirement: **{'YES' if report.meets_client_requirement else 'PARTIAL'}**",
        "",
        "## Code changes",
        "",
    ])
    for change in report.code_changes:
        lines.append(f"- {change}")
    if report.gaps_and_residual_risks:
        lines.extend(["", "## Residual risks", ""])
        for gap in report.gaps_and_residual_risks:
            lines.append(f"- {gap}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "md": md_path}
