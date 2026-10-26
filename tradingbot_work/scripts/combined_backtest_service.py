"""Shared combined backtest runner for CLI and UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.classic_pivot_historical_backtest import GUIDE_BASELINE_VERSION
from scripts.multi_strategy_runner import run_pe_backtest, run_strategy_pivot
from scripts.run_combined_last7_report import (
    DEFAULT_ATR_BUFFER,
    DEFAULT_RR,
    END,
    run_backtest,
    write_comparison_report,
    write_report_files,
    pivot_label_for,
)
from scripts.strategy_catalog import StrategySpec, strategy_by_id

IST = ZoneInfo("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BacktestRunResult:
    preset: str
    pivot_label: str
    report_start: date
    report_end: date
    symbols: list[str]
    output_dir: Path
    metrics: dict[str, Any]
    pivot_trades: int
    pe_signals: int
    rr: float = DEFAULT_RR
    atr_buffer: float = DEFAULT_ATR_BUFFER
    strategy_id: str | None = None
    strategy_label: str | None = None
    guide_filter: str = "none"
    files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        pm = self.metrics["pivot"]
        pe = self.metrics["pe"]
        daily = self.metrics["daily"]
        score = self.metrics["score"]
        return {
            "preset": self.preset,
            "pivot_label": self.pivot_label,
            "strategy_id": self.strategy_id,
            "strategy_label": self.strategy_label,
            "guide_filter": self.guide_filter,
            "report_start": self.report_start.isoformat(),
            "report_end": self.report_end.isoformat(),
            "symbols": self.symbols,
            "output_dir": str(self.output_dir),
            "files": self.files,
            "pivot": pm,
            "pe": pe,
            "daily": daily,
            "score": score,
            "exit_analytics": self.metrics.get("exit_analytics", {}),
            "pivot_trades": self.pivot_trades,
            "pe_signals": self.pe_signals,
            "rr": self.rr,
            "atr_buffer": self.atr_buffer,
        }


def output_dir_for_run(
    preset: str,
    report_start: date,
    report_end: date,
    *,
    ui: bool = False,
    guide_filter: str = "none",
    rr: float = DEFAULT_RR,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
    strategy_id: str | None = None,
) -> Path:
    if ui:
        stamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        base = strategy_id or preset
        filt = f"_{guide_filter}" if guide_filter and guide_filter != "none" and not strategy_id else ""
        rr_tag = f"_rr{rr:g}" if rr != DEFAULT_RR else ""
        atr_tag = f"_atr{atr_buffer:g}" if atr_buffer != DEFAULT_ATR_BUFFER else ""
        return ROOT / "reports" / "ui_runs" / f"{stamp}_{base}{filt}{rr_tag}{atr_tag}_{report_start}_{report_end}"
    if preset == "guide_baseline":
        return ROOT / "reports" / "combined_last7_guide_pe"
    if preset == "client_v1":
        return ROOT / "reports" / "combined_client_v1_pe"
    return ROOT / "reports" / f"combined_{preset}_pe"


def execute_combined_backtest(
    report_start: date,
    report_end: date,
    *,
    preset: str = "guide_baseline",
    symbols: list[str] | None = None,
    ui: bool = False,
    compare_client: bool = False,
    guide_filter: str = "none",
    rr: float = DEFAULT_RR,
    atr_buffer: float = DEFAULT_ATR_BUFFER,
) -> BacktestRunResult:
    symbols = symbols or ["NIFTY", "BANKNIFTY"]
    pivot_label = pivot_label_for(preset)
    out = output_dir_for_run(
        preset, report_start, report_end, ui=ui,
        guide_filter=guide_filter, rr=rr, atr_buffer=atr_buffer,
    )
    out.mkdir(parents=True, exist_ok=True)

    pivot, pe, days, pe_armed_days = run_backtest(
        preset, report_start, report_end, symbols, guide_filter, rr, atr_buffer,
    )
    if not pivot.empty:
        pivot.to_csv(out / "pivot_trades.csv", index=False)
    if not pe.empty:
        pe.to_csv(out / "pe_trades.csv", index=False)

    metrics = write_report_files(
        out,
        pivot_label=pivot_label,
        report_start=report_start,
        report_end=report_end,
        days=days,
        pivot=pivot,
        pe=pe,
        pe_armed_days=pe_armed_days,
        pivot_preset=preset,
        guide_filter=guide_filter,
    )

    files = sorted(p.name for p in out.iterdir() if p.is_file())

    if compare_client and preset == "guide_baseline":
        cmp_out = out / "comparison"
        cmp_out.mkdir(parents=True, exist_ok=True)
        from scripts.client_strategy_config import CLIENT_V1_VERSION

        client_pivot, client_pe, _, client_armed = run_backtest(
            CLIENT_V1_VERSION, report_start, report_end, symbols,
        )
        client_metrics = write_report_files(
            cmp_out / "client_v1",
            pivot_label=CLIENT_V1_VERSION,
            report_start=report_start,
            report_end=report_end,
            days=days,
            pivot=client_pivot,
            pe=client_pe,
            pe_armed_days=client_armed,
            pivot_preset=CLIENT_V1_VERSION,
        )
        write_comparison_report(cmp_out, report_start, report_end, metrics, client_metrics)

    return BacktestRunResult(
        preset=preset,
        pivot_label=pivot_label,
        report_start=report_start,
        report_end=report_end,
        symbols=symbols,
        output_dir=out,
        metrics=metrics,
        pivot_trades=int(metrics["pivot"]["trades"]),
        pe_signals=int(metrics["pe"]["trades"]),
        rr=rr,
        atr_buffer=atr_buffer,
        guide_filter=guide_filter,
        files=files,
    )


def _pivot_label_for_spec(spec: StrategySpec) -> str:
    if spec.alt_engine:
        return spec.id
    if spec.preset:
        return pivot_label_for(spec.preset)
    return spec.id


def execute_strategy_backtest(
    strategy_id: str,
    report_start: date,
    report_end: date,
    *,
    symbols: list[str] | None = None,
    ui: bool = False,
    rr: float | None = None,
    atr_buffer: float | None = None,
    guide_filter: str | None = None,
) -> BacktestRunResult:
    spec = strategy_by_id(strategy_id)
    syms = symbols or list(spec.symbols_default)
    rr_val = rr if rr is not None else spec.rr
    atr_val = atr_buffer if atr_buffer is not None else spec.atr_buffer
    filt = guide_filter if guide_filter is not None else spec.guide_filter
    preset = spec.preset or "guide_baseline"
    pivot_label = _pivot_label_for_spec(spec)

    out = output_dir_for_run(
        preset, report_start, report_end, ui=ui,
        guide_filter=filt, rr=rr_val, atr_buffer=atr_val, strategy_id=spec.id,
    )
    out.mkdir(parents=True, exist_ok=True)

    pivot = run_strategy_pivot(
        spec, report_start, report_end, syms,
        rr=rr_val, atr_buffer=atr_val, guide_filter=filt,
    )
    pe = pd.DataFrame()
    days: list[date] = []
    pe_armed_days: list[str] = []
    days = [d.date() for d in pd.date_range(report_start, report_end, freq="D") if d.weekday() < 5]
    if (spec.include_pe and "NIFTY" in syms) or spec.id == "pe_only":
        pe, days, pe_armed_days = run_pe_backtest(report_start, report_end)

    if not pivot.empty:
        pivot.to_csv(out / "pivot_trades.csv", index=False)
    if not pe.empty:
        pe.to_csv(out / "pe_trades.csv", index=False)

    metrics = write_report_files(
        out,
        pivot_label=pivot_label,
        report_start=report_start,
        report_end=report_end,
        days=days,
        pivot=pivot,
        pe=pe,
        pe_armed_days=pe_armed_days,
        pivot_preset=preset,
        guide_filter=filt,
        rr=rr_val,
        atr_buffer=atr_val,
    )
    files = sorted(p.name for p in out.iterdir() if p.is_file())
    return BacktestRunResult(
        preset=preset,
        pivot_label=pivot_label,
        report_start=report_start,
        report_end=report_end,
        symbols=syms,
        output_dir=out,
        metrics=metrics,
        pivot_trades=int(metrics["pivot"]["trades"]),
        pe_signals=int(metrics["pe"]["trades"]),
        rr=rr_val,
        atr_buffer=atr_val,
        strategy_id=spec.id,
        strategy_label=spec.label,
        guide_filter=filt,
        files=files,
    )


def load_run_tables(output_dir: Path) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    mapping = {
        "daily_summary": "daily_summary.csv",
        "pivot_trades": "pivot_trades.csv",
        "pe_trades": "pe_trades.csv",
        "monthly_pivot_summary": "monthly_pivot_summary.csv",
        "symbol_summary": "symbol_summary.csv",
        "effectiveness_metrics": "effectiveness_metrics.csv",
        "exit_analytics": "exit_analytics.csv",
    }
    for key, filename in mapping.items():
        path = output_dir / filename
        if path.is_file():
            tables[key] = pd.read_csv(path).fillna("").to_dict(orient="records")
    report_md = output_dir / "REPORT.md"
    if report_md.is_file():
        tables["report_markdown"] = [{"content": report_md.read_text(encoding="utf-8")}]
    return tables
