"""S2 performance breakdown by classified market regime (historical data only)."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest.regime_classifier import REGIME_CATALOG, build_regime_labels
from src.backtest.robustness import _consecutive_streak, _max_drawdown, _profit_factor

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class RegimeMetricsRow:
    regime_id: str
    regime_label: str
    trading_days: int = 0
    signals: int = 0
    resolved_trades: int = 0
    unresolved_signals: int = 0
    blocked_or_skipped_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    average_r: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    option_coverage_percent: float = 0.0
    signals_with_coverage: int = 0
    signals_missing_coverage: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeRobustnessReport:
    run_id: str
    source_run_id: str
    underlying: str
    strategy_id: str
    from_date: str
    to_date: str
    total_trading_days: int
    regimes: list[RegimeMetricsRow] = field(default_factory=list)
    day_labels_sample: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["regimes"] = [r.as_dict() for r in self.regimes]
        return data


def _ts(value: str | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert(IST) if ts.tzinfo else ts.tz_localize(IST)


def _day_of(value: str) -> date:
    return _ts(value).date()


def _signal_times(result: dict[str, Any]) -> list[str]:
    times: list[str] = []
    for row in result.get("signals", []):
        if isinstance(row, dict) and row.get("timestamp"):
            times.append(str(row["timestamp"]))
    if not times:
        total = int(result.get("metrics", {}).get("total_signals", 0))
        for trade in result.get("trades", []):
            times.append(str(trade.get("signal_time") or trade.get("entry_time")))
        # unresolved-only signals may not be materialized; use trades + skipped times
    for row in result.get("skipped", []):
        if row.get("time"):
            times.append(str(row["time"]))
    return times


def _coverage_by_signal(coverage: dict[str, Any]) -> dict[str, bool]:
    missing = {item.get("signal_time") for item in coverage.get("unavailable", []) if item.get("signal_time")}
    out: dict[str, bool] = {}
    for t in missing:
        out[str(t)] = False
    return out


def _metrics_for_subset(
    trades: list[dict],
    *,
    initial_equity: float,
) -> dict[str, Any]:
    wins = [t for t in trades if float(t.get("pnl", 0)) > 0]
    losses = [t for t in trades if float(t.get("pnl", 0)) < 0]
    gross_profit = sum(float(t.get("pnl", 0)) for t in wins)
    gross_loss = abs(sum(float(t.get("pnl", 0)) for t in losses))
    net_pnl = sum(float(t.get("pnl", 0)) for t in trades)
    decided = len(wins) + len(losses)
    pf = _profit_factor([float(t.get("pnl", 0)) for t in wins], [float(t.get("pnl", 0)) for t in losses])
    r_values = [float(t.get("r_multiple", 0)) for t in trades]
    return {
        "resolved_trades": len(trades),
        "win_rate": len(wins) / decided if decided else 0.0,
        "profit_factor": pf,
        "net_pnl": net_pnl,
        "max_drawdown": _max_drawdown(trades, initial_equity),
        "average_r": sum(r_values) / len(r_values) if r_values else 0.0,
        "max_consecutive_wins": _consecutive_streak(trades, True),
        "max_consecutive_losses": _consecutive_streak(trades, False),
    }


def analyze_regime_robustness(
    *,
    run_id: str,
    source_run_id: str,
    underlying: str,
    strategy_id: str,
    from_date: date,
    to_date: date,
    trading_days: int,
    underlying_candles: pd.DataFrame,
    result: dict[str, Any],
    coverage: dict[str, Any],
    s2_cfg: dict[str, Any],
    initial_equity: float = 1_000_000.0,
) -> RegimeRobustnessReport:
    labels = build_regime_labels(underlying_candles, s2_cfg)
    trades = list(result.get("trades", []))
    skipped = list(result.get("skipped", []))
    signal_rows = list(result.get("signals", []))

    def day_has_regime(d: date, regime_id: str) -> bool:
        return regime_id in labels.get(d, set())

    coverage_missing = _coverage_by_signal(coverage)
    rows: list[RegimeMetricsRow] = []

    for regime_id, regime_label in REGIME_CATALOG:
        regime_days = {d for d, tags in labels.items() if regime_id in tags}
        regime_trades = [t for t in trades if _day_of(t["entry_time"]) in regime_days]
        regime_skipped = [s for s in skipped if _day_of(s["time"]) in regime_days]

        regime_signals = 0
        covered = 0
        missing = 0
        for sig in signal_rows:
            if not isinstance(sig, dict) or not sig.get("timestamp"):
                continue
            d = _day_of(sig["timestamp"])
            if d not in regime_days:
                continue
            regime_signals += 1
            st = str(sig["timestamp"])
            if st in coverage_missing:
                missing += 1
            else:
                covered += 1

        if regime_signals == 0:
            # Approximate from trades + skipped on regime days when signal list sparse
            regime_signals = len({t.get("signal_time") or t.get("entry_time") for t in regime_trades})
            regime_signals += len(regime_skipped)
            for t in regime_trades:
                st = str(t.get("signal_time") or t.get("entry_time"))
                if st in coverage_missing:
                    missing += 1
                else:
                    covered += 1

        subset = _metrics_for_subset(regime_trades, initial_equity=initial_equity)
        resolved = subset["resolved_trades"]
        unresolved = max(0, regime_signals - resolved)

        cov_pct = 100.0 * covered / regime_signals if regime_signals else 0.0

        rows.append(
            RegimeMetricsRow(
                regime_id=regime_id,
                regime_label=regime_label,
                trading_days=len(regime_days),
                signals=regime_signals,
                resolved_trades=resolved,
                unresolved_signals=unresolved,
                blocked_or_skipped_signals=len(regime_skipped),
                win_rate=subset["win_rate"],
                profit_factor=subset["profit_factor"],
                net_pnl=subset["net_pnl"],
                max_drawdown=subset["max_drawdown"],
                average_r=subset["average_r"],
                max_consecutive_wins=subset["max_consecutive_wins"],
                max_consecutive_losses=subset["max_consecutive_losses"],
                option_coverage_percent=round(cov_pct, 2),
                signals_with_coverage=covered,
                signals_missing_coverage=missing,
            )
        )

    sample_days = sorted(labels.keys())[-5:]
    day_sample = [{"day": d.isoformat(), "regimes": sorted(labels[d])} for d in sample_days]

    return RegimeRobustnessReport(
        run_id=run_id,
        source_run_id=source_run_id,
        underlying=underlying,
        strategy_id=strategy_id,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        total_trading_days=trading_days,
        regimes=rows,
        day_labels_sample=day_sample,
    )


def write_regime_robustness_reports(report: RegimeRobustnessReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.underlying}_{report.strategy_id}_regime_robustness"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    fieldnames = list(RegimeMetricsRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report.regimes:
            writer.writerow(row.as_dict())

    md_lines = [
        f"# {report.underlying} {report.strategy_id} Market-Regime Robustness",
        "",
        f"**Source run:** `{report.source_run_id}`  ",
        f"**Analysis run:** `{report.run_id}`  ",
        f"**Period:** {report.from_date} to {report.to_date} ({report.total_trading_days} trading days)",
        "",
        "Regimes are tagged per session from real 5m OHLC. Rolling percentile thresholds use "
        "only prior sessions (no look-ahead). S2 logic and risk rules are unchanged.",
        "",
        "| Regime | Days | Signals | Trades | Win% | PF | Net P&L | Max DD | Avg R | Max W | Max L | Unresolved | Skipped | Cov% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.regimes:
        pf = "inf" if row.profit_factor == float("inf") else f"{row.profit_factor:.2f}"
        md_lines.append(
            f"| {row.regime_label} | {row.trading_days} | {row.signals} | {row.resolved_trades} | "
            f"{row.win_rate * 100:.1f} | {pf} | {row.net_pnl:,.0f} | {row.max_drawdown:,.0f} | "
            f"{row.average_r:.2f} | {row.max_consecutive_wins} | {row.max_consecutive_losses} | "
            f"{row.unresolved_signals} | {row.blocked_or_skipped_signals} | {row.option_coverage_percent:.1f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "md": md_path}
