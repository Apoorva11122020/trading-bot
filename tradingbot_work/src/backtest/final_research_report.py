"""Final combined strategy research report (NIFTY / BANKNIFTY, S1-S4)."""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

MIN_RESOLVED_TRADES = 30


@dataclass
class StrategyResearchRow:
    underlying: str
    strategy_id: str
    from_date: str
    to_date: str
    trading_days: int
    signals: int
    resolved_trades: int
    unresolved_signals: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    net_pnl: float
    max_drawdown: float
    average_r: float
    median_r: float
    average_trades_per_day: float
    max_consecutive_losses: int
    daily_loss_limit_breaches: int
    premium_coverage_percent: float
    resolved_trade_percent: float
    required_contracts: int
    contracts_downloaded: int
    contracts_unavailable: int
    status: str
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FinalResearchReport:
    run_id: str
    requested_from: str
    requested_to: str
    limitation: str
    availability: dict[str, Any]
    coverage: list[dict[str, Any]] = field(default_factory=list)
    rows: list[StrategyResearchRow] = field(default_factory=list)
    rankings: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "requested_from": self.requested_from,
            "requested_to": self.requested_to,
            "limitation": self.limitation,
            "availability": self.availability,
            "coverage": self.coverage,
            "rows": [r.as_dict() for r in self.rows],
            "rankings": self.rankings,
        }


def _status(resolved: int) -> str:
    return "VALID RESEARCH SAMPLE" if resolved >= MIN_RESOLVED_TRADES else "INSUFFICIENT SAMPLE"


def _pf_text(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.2f}"


def row_from_result(
    *,
    underlying: str,
    strategy_id: str,
    from_date: date,
    to_date: date,
    result: dict,
    coverage: dict,
    trading_days: int,
) -> StrategyResearchRow:
    metrics = result.get("metrics", {})
    trades = result.get("trades", [])
    resolved = int(metrics.get("resolved_trades", 0))
    signals = int(metrics.get("total_signals", 0))
    wins = [t for t in trades if float(t.get("pnl", 0)) > 0]
    losses = [t for t in trades if float(t.get("pnl", 0)) < 0]
    gross_profit = sum(float(t.get("pnl", 0)) for t in wins)
    gross_loss = abs(sum(float(t.get("pnl", 0)) for t in losses))
    r_values = [float(t.get("r_multiple", 0)) for t in trades]
    pf = metrics.get("profit_factor", 0)
    return StrategyResearchRow(
        underlying=underlying,
        strategy_id=strategy_id,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        trading_days=trading_days,
        signals=signals,
        resolved_trades=resolved,
        unresolved_signals=max(0, signals - resolved),
        win_rate=float(metrics.get("win_rate", 0)),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=pf if pf != float("inf") else 999999.0,
        net_pnl=float(metrics.get("total_pnl", 0)),
        max_drawdown=float(metrics.get("max_drawdown", 0)),
        average_r=float(metrics.get("average_r", 0)),
        median_r=statistics.median(r_values) if r_values else 0.0,
        average_trades_per_day=float(metrics.get("average_trades_per_day", 0)),
        max_consecutive_losses=int(metrics.get("max_consecutive_losses", 0)),
        daily_loss_limit_breaches=int(metrics.get("daily_loss_hits", 0)),
        premium_coverage_percent=float(coverage.get("coverage_percent", 0)),
        resolved_trade_percent=round(100.0 * resolved / signals, 2) if signals else 0.0,
        required_contracts=int(coverage.get("unique_required_contracts", 0)),
        contracts_downloaded=int(coverage.get("contracts_downloaded", 0)),
        contracts_unavailable=int(coverage.get("contracts_unavailable", 0)),
        status=_status(resolved),
    )


def rank_strategies(rows: list[StrategyResearchRow], underlying: str) -> list[str]:
    subset = [r for r in rows if r.underlying == underlying and r.status == "VALID RESEARCH SAMPLE"]
    ordered = sorted(
        subset,
        key=lambda r: (r.profit_factor, -r.max_drawdown, r.win_rate, r.average_r),
        reverse=True,
    )
    for index, row in enumerate(ordered, start=1):
        row.rank = index
    return [r.strategy_id for r in ordered]


def write_final_reports(report: FinalResearchReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_comparison.json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8",
    )
    if report.rows:
        with (out_dir / "final_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=report.rows[0].as_dict().keys())
            writer.writeheader()
            for row in report.rows:
                writer.writerow(row.as_dict())

    lines = [
        "# Strategy Research — Final Comparison",
        "",
        f"**Run ID:** {report.run_id}",
        f"**Requested range:** {report.requested_from} → {report.requested_to}",
        "",
        "## Data availability limitation",
        "",
        report.limitation,
        "",
        "## Availability by underlying",
        "",
    ]
    for name, info in report.availability.items():
        lines.append(f"### {name}")
        lines.append(f"- Index available: {info.get('index_available_from')} → {info.get('index_available_to')}")
        lines.append(f"- Option available: {info.get('option_available_from')} → {info.get('option_available_to')}")
        lines.append(f"- **Research window used:** {info.get('research_from')} → {info.get('research_to')}")
        lines.append("")

    lines.extend(["## Coverage summary", ""])
    for cov in report.coverage:
        lines.append(
            f"- **{cov['underlying']} / {cov['strategy_id']}**: signals={cov['total_signals']}, "
            f"required contracts={cov['unique_required_contracts']}, downloaded={cov['contracts_downloaded']}, "
            f"unavailable={cov['contracts_unavailable']}, premium coverage={cov['coverage_percent']}%"
        )

    lines.extend(["", "## Strategy results", ""])
    lines.append(
        "| Underlying | Strategy | Days | Signals | Resolved | Win% | PF | Net P&L | Max DD | Avg R | Status |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in report.rows:
        lines.append(
            f"| {row.underlying} | {row.strategy_id} | {row.trading_days} | {row.signals} | "
            f"{row.resolved_trades} | {row.win_rate*100:.1f}% | {_pf_text(row.profit_factor)} | "
            f"{row.net_pnl:,.2f} | {row.max_drawdown:,.2f} | {row.average_r:.2f} | {row.status} |"
        )

    lines.extend(["", "## Rankings (VALID RESEARCH SAMPLE only)", ""])
    for underlying, order in report.rankings.items():
        lines.append(f"- **{underlying}:** {', '.join(order) if order else 'No strategy reached 30 resolved trades'}")

    lines.extend([
        "",
        "_Unresolved signals are not counted as losses. Results with fewer than 30 resolved trades "
        "are marked INSUFFICIENT SAMPLE and must not be used to claim profitability._",
    ])
    (out_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary_table(rows: list[StrategyResearchRow]) -> None:
    print("\nUNDERLYING | STRATEGY | DAYS | SIGNALS | RESOLVED | WIN% | PF | NET P&L | MAX DD | AVG R | STATUS")
    print("-" * 105)
    for row in rows:
        pf = _pf_text(row.profit_factor)
        print(
            f"{row.underlying:<10} | {row.strategy_id:<8} | {row.trading_days:>4} | {row.signals:>7} | "
            f"{row.resolved_trades:>8} | {row.win_rate*100:>5.1f}% | {pf:>4} | {row.net_pnl:>10,.0f} | "
            f"{row.max_drawdown:>8,.0f} | {row.average_r:>5.2f} | {row.status}"
        )
