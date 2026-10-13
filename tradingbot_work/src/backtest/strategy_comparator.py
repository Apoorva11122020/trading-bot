"""Compare research strategies on train / validation / out-of-sample splits."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from .options_engine import OptionsResearchEngine, OptionsResearchResult


@dataclass
class StrategyComparisonRow:
    underlying: str
    strategy_id: str
    split: str
    status: str
    profit_factor: float
    max_drawdown: float
    win_rate: float
    average_r: float
    total_pnl: float
    resolved_trades: int
    total_signals: int
    rank_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyComparison:
    rows: list[StrategyComparisonRow] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)
    disclaimer: str = "Ranking is based on configured metrics over historical simulation only."

    def as_dict(self) -> dict[str, Any]:
        return {"rows": [r.as_dict() for r in self.rows], "ranking": self.ranking, "disclaimer": self.disclaimer}


def date_splits(candles: pd.DataFrame, config) -> dict[str, tuple[date, date]]:
    days = sorted(set(pd.to_datetime(candles["timestamp"]).dt.date))
    if not days:
        return {}
    n = len(days)
    train_n = max(1, int(n * config.split.train))
    val_n = max(1, int(n * config.split.validation))
    train_end = days[train_n - 1]
    val_end = days[min(n - 1, train_n + val_n - 1)]
    return {
        "TRAIN": (days[0], train_end),
        "VALIDATION": (days[train_n], val_end) if train_n < n else (days[0], days[0]),
        "OOS": (days[min(train_n + val_n, n - 1)], days[-1]),
        "FULL": (days[0], days[-1]),
    }


def _rank_value(row: StrategyComparisonRow, metric: str) -> float:
    if metric == "profit_factor":
        return row.profit_factor
    if metric == "max_drawdown":
        return -row.max_drawdown
    if metric == "win_rate":
        return row.win_rate
    if metric == "average_r":
        return row.average_r
    if metric == "total_pnl":
        return row.total_pnl
    return 0.0


def rank_rows(rows: list[StrategyComparisonRow], metrics: Iterable[str]) -> list[StrategyComparisonRow]:
    if not rows:
        return rows
    scored = []
    for row in rows:
        score = sum(_rank_value(row, metric) for metric in metrics)
        scored.append(StrategyComparisonRow(**{**row.as_dict(), "rank_score": score}))
    return sorted(scored, key=lambda r: r.rank_score, reverse=True)


def compare_strategies(
    *,
    research_config,
    strategy_config,
    underlying: str,
    underlying_candles: pd.DataFrame,
    premium_history: dict[str, pd.DataFrame],
    contracts,
    strategy_ids: list[str],
    run_id: str,
    master_resolver=None,
    splits: dict[str, tuple[date, date]] | None = None,
) -> tuple[list[OptionsResearchResult], StrategyComparison]:
    splits = splits or date_splits(underlying_candles, research_config)
    results: list[OptionsResearchResult] = []
    rows: list[StrategyComparisonRow] = []
    for strategy_id in strategy_ids:
        for split_name, (start, end) in splits.items():
            if start > end:
                continue
            engine = OptionsResearchEngine(
                research_config, strategy_config, contracts, premium_history,
                run_id=run_id, split=split_name, master_resolver=master_resolver,
            )
            result = engine.run(underlying_candles, underlying, strategy_id, start, end)
            results.append(result)
            m = result.metrics
            rows.append(StrategyComparisonRow(
                underlying=underlying.upper(), strategy_id=strategy_id, split=split_name, status=result.status,
                profit_factor=m.profit_factor, max_drawdown=m.max_drawdown, win_rate=m.win_rate,
                average_r=m.average_r, total_pnl=m.total_pnl, resolved_trades=m.resolved_trades,
                total_signals=m.total_signals))
    comparison = StrategyComparison(rows=rank_rows(rows, research_config.ranking), ranking=list(research_config.ranking))
    return results, comparison
