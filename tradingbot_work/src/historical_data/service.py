"""Orchestration for historical downloads and research backtests."""
from __future__ import annotations

import csv
import json
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.loader import load_strategy_config
from src.config.research_loader import load_research_config
from src.options.instrument_resolver import build_resolver
from src.safety.live_gate import live_trading_enabled, trading_mode

from .contracts import ContractCatalog, load_instrument_master_rows
from .downloader import HistoricalDownloader, MetadataStore
from .models import HistoricalDataError
from .availability import discover_research_range
from .option_coverage import CoverageReport, ensure_option_coverage, extract_required_contracts, research_signals
from .storage import ParquetCandleStore
from .upstox_client import UpstoxHistoricalClient, token_source

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_CONFIG = ROOT / "config" / "strategy_research.yaml"
STRATEGY_CONFIG = ROOT / "config" / "strategy.yaml"
RUN_PREFIX = "backtest_run:"
UNDERLYING_MISSING_HINT = (
    "If underlying candles are missing, download the required underlying data before running the backtest."
)
DOWNLOAD_LOCK = threading.Lock()
BACKTEST_LOCK = threading.Lock()
_download_thread: threading.Thread | None = None
_backtest_thread: threading.Thread | None = None


def _refuse_live() -> None:
    if live_trading_enabled() or trading_mode() != "PAPER":
        raise HistoricalDataError("LIVE_TRADING_REFUSED", "Historical research requires PAPER mode with live trading disabled")


def build_catalog(client: UpstoxHistoricalClient | None = None) -> ContractCatalog:
    client = client or UpstoxHistoricalClient()
    resolver = build_resolver(load_strategy_config(STRATEGY_CONFIG).options)
    return ContractCatalog(client, lambda: load_instrument_master_rows(resolver.source),
                           underlying_keys={"NIFTY": "NSE_INDEX|Nifty 50", "BANKNIFTY": "NSE_INDEX|Nifty Bank"})


class HistoricalDataService:
    def __init__(self, database=None, store_root: str | Path = "data/historical") -> None:
        self.database = database
        self.store = ParquetCandleStore(store_root)
        self.metadata = MetadataStore(database)
        self.client = UpstoxHistoricalClient()
        self.catalog = build_catalog(self.client)
        self.research_config = load_research_config(RESEARCH_CONFIG)
        self.strategy_config = load_strategy_config(STRATEGY_CONFIG)
        self._research_resolver = build_resolver(self.strategy_config.options)

    def _downloader(self) -> HistoricalDownloader:
        return HistoricalDownloader(
            self.client,
            self.store,
            self.catalog,
            self.metadata,
            interval=self.research_config.interval_minutes,
            strike_window=int(self.research_config.download.get("strike_window", 4)),
        )

    def underlying_key(self, underlying: str) -> str:
        return self.catalog.underlying_key(underlying)

    def read_underlying(self, underlying: str, from_date: date, to_date: date) -> pd.DataFrame:
        key = self.underlying_key(underlying)
        return self.store.read("underlying", underlying.upper(), key, start=from_date, end=to_date)

    def has_underlying_data(self, underlying: str, from_date: date, to_date: date) -> bool:
        return not self.read_underlying(underlying, from_date, to_date).empty

    def has_option_data(self, underlying: str, from_date: date, to_date: date) -> bool:
        return bool(self._premium_history(underlying, from_date, to_date))

    def resolver_factory(self, as_of: date):
        self._research_resolver.today = lambda: as_of
        return self._research_resolver

    def ensure_strategy_option_coverage(
        self,
        underlying: str,
        strategy_id: str,
        from_date: date,
        to_date: date,
    ) -> CoverageReport:
        """Download only the option contracts required by strategy signals."""
        _refuse_live()
        candles = self.read_underlying(underlying, from_date, to_date)
        if candles.empty:
            raise HistoricalDataError(
                "UNDERLYING_DATA_MISSING",
                f"No stored underlying candles for {underlying.upper()}. {UNDERLYING_MISSING_HINT}",
            )
        candles["symbol"] = underlying.upper()
        return ensure_option_coverage(
            store=self.store,
            downloader=self._downloader(),
            resolver_factory=self.resolver_factory,
            research_config=self.research_config,
            strategy_config=self.strategy_config,
            underlying=underlying,
            strategy_id=strategy_id,
            underlying_candles=candles,
            from_date=from_date,
            to_date=to_date,
        )

    def trace_required_contracts(
        self,
        underlying: str,
        strategy_id: str,
        from_date: date,
        to_date: date,
    ) -> tuple[list, list]:
        from src.session.session_manager import SessionManager

        candles = self.read_underlying(underlying, from_date, to_date)
        candles["symbol"] = underlying.upper()
        session = SessionManager(self.strategy_config.session)
        signals = research_signals(
            candles, underlying, strategy_id, self.research_config, from_date, to_date, session,
        )
        return extract_required_contracts(
            signals, self.resolver_factory, self.research_config, from_date, to_date,
        ), signals

    def verify_underlying_data(self, underlying: str, from_date: date, to_date: date) -> dict[str, Any]:
        """Confirm stored underlying candles exist, are in range and pass OHLC checks."""
        from .validation import validate_candles

        name = underlying.upper()
        key = self.underlying_key(name)
        paths = self.store.files("underlying", name, key)
        frame = self.read_underlying(name, from_date, to_date)
        if frame.empty:
            raise HistoricalDataError(
                "UNDERLYING_DATA_MISSING",
                f"No stored underlying candles for {name}. {UNDERLYING_MISSING_HINT}",
            )
        clean, report = validate_candles(frame, interval=self.research_config.interval_minutes)
        if clean.empty:
            raise HistoricalDataError(
                "UNDERLYING_DATA_INVALID",
                f"Stored underlying candles for {name} failed validation: {report.issues}",
            )
        ts = pd.to_datetime(clean["timestamp"])
        in_range = clean[(ts.dt.date >= from_date) & (ts.dt.date <= to_date)]
        if in_range.empty:
            raise HistoricalDataError(
                "UNDERLYING_DATA_INCOMPLETE",
                f"No stored candles for {name} between {from_date} and {to_date}",
            )
        first, last = ts.min(), ts.max()
        return {
            "underlying": name,
            "instrument_key": key,
            "candle_count": len(clean),
            "files": [str(p) for p in paths],
            "first": first.isoformat(),
            "last": last.isoformat(),
            "validation": report.as_dict(),
        }

    def download_underlying_sync(self, underlying: str, from_date: date, to_date: date) -> dict[str, Any]:
        """Download index underlying candles via Upstox V3 and store to Parquet (read-only)."""
        _refuse_live()
        downloader = self._downloader()
        job = downloader.new_job(
            [underlying.upper()],
            from_date,
            to_date,
            include_options=False,
            include_futures=False,
        )
        job = downloader.run(job)
        if job.status == "FAILED":
            detail = job.failures[-1] if job.failures else {"message": job.message}
            code = detail.get("code", "DOWNLOAD_FAILED")
            message = detail.get("message", job.message)
            raise HistoricalDataError(code, message)
        if job.chunks_failed:
            failure = job.failures[-1]
            raise HistoricalDataError(
                failure.get("code", "DOWNLOAD_FAILED"),
                failure.get("message", job.message),
            )
        return self.verify_underlying_data(underlying, from_date, to_date)

    def download_research_data_sync(
        self,
        underlying: str,
        from_date: date,
        to_date: date,
        *,
        include_options: bool = True,
    ) -> dict[str, Any]:
        """Ensure underlying (and optionally option) candles are stored for a backtest window."""
        name = underlying.upper()
        result: dict[str, Any] = {"underlying": name, "from_date": from_date.isoformat(),
                                  "to_date": to_date.isoformat()}
        if not self.has_underlying_data(name, from_date, to_date):
            result["underlying_download"] = self.download_underlying_sync(name, from_date, to_date)
        else:
            result["underlying_download"] = {"skipped": True, **self.verify_underlying_data(name, from_date, to_date)}
        if include_options:
            enabled = [
                sid for sid, cfg in self.research_config.strategies.items() if cfg.get("enabled", True)
            ]
            coverage_reports = []
            for sid in enabled:
                coverage_reports.append(
                    self.ensure_strategy_option_coverage(name, sid, from_date, to_date).as_dict()
                )
            result["options_coverage"] = coverage_reports
        return result

    def count_trading_days(self, underlying: str, from_date: date, to_date: date) -> int:
        frame = self.read_underlying(underlying, from_date, to_date)
        if frame.empty:
            return 0
        return len(set(pd.to_datetime(frame["timestamp"]).dt.date))

    def discover_availability(
        self,
        underlyings: list[str],
        *,
        requested_from: date | None = None,
        requested_to: date | None = None,
    ) -> dict[str, Any]:
        availability = {}
        for underlying in underlyings:
            name = underlying.upper()
            key = self.underlying_key(name)
            resolver = build_resolver(self.strategy_config.options)
            availability[name] = discover_research_range(
                self.client, name, key, resolver,
                requested_from=requested_from, requested_to=requested_to,
            ).as_dict()
        return availability

    def run_full_research(
        self,
        underlyings: list[str] | None = None,
        *,
        requested_from: date | None = None,
        requested_to: date | None = None,
        strategies: list[str] | None = None,
    ):
        """Download max available data, ensure option coverage, backtest, write final reports."""
        from src.backtest.final_research_report import (
            FinalResearchReport,
            print_summary_table,
            rank_strategies,
            row_from_result,
            write_final_reports,
        )
        from src.backtest.strategy_comparator import compare_strategies

        _refuse_live()
        names = [u.upper() for u in (underlyings or self.research_config.underlyings)]
        strategy_ids = strategies or [
            sid for sid, cfg in self.research_config.strategies.items() if cfg.get("enabled", True)
        ]
        availability = self.discover_availability(names, requested_from=requested_from, requested_to=requested_to)

        research_windows = []
        limitation = ""
        for name, info in availability.items():
            limitation = info.get("limitation", limitation)
            start = info.get("research_from")
            end = info.get("research_to")
            if start and end:
                research_windows.append((date.fromisoformat(start), date.fromisoformat(end)))
        if not research_windows:
            raise HistoricalDataError(
                "NO_RESEARCH_WINDOW",
                "No overlapping index and option historical window is available with the current Upstox V3 token.",
            )
        from_date = min(w[0] for w in research_windows)
        to_date = min(max(w[1] for w in research_windows), date.today())

        run_id = uuid.uuid4().hex[:12]
        coverage_reports: list[dict] = []
        all_results = []

        for name in names:
            print(f"Downloading {name} underlying {from_date} .. {to_date} ...")
            self.download_underlying_sync(name, from_date, to_date)
            candles = self.read_underlying(name, from_date, to_date)
            candles["symbol"] = name
            for sid in strategy_ids:
                print(f"  Option coverage {name} / {sid} ...")
                cov = self.ensure_strategy_option_coverage(name, sid, from_date, to_date)
                coverage_reports.append(cov.as_dict())
            contracts = self._contracts_from_store(name)
            premium = self._premium_history(name, from_date, to_date)
            trading_days = self.count_trading_days(name, from_date, to_date)
            results, _ = compare_strategies(
                research_config=self.research_config,
                strategy_config=self.strategy_config,
                underlying=name,
                underlying_candles=candles,
                premium_history=premium,
                contracts=contracts,
                strategy_ids=strategy_ids,
                run_id=run_id,
                master_resolver=self._research_resolver,
                splits={"FULL": (from_date, to_date)},
            )
            all_results.extend([r.as_dict() for r in results])

        rows = []
        for name in names:
            trading_days = self.count_trading_days(name, from_date, to_date)
            for sid in strategy_ids:
                result = next(
                    (r for r in all_results if r["underlying"] == name and r["strategy_id"] == sid and r["split"] == "FULL"),
                    None,
                )
                cov = next((c for c in coverage_reports if c["underlying"] == name and c["strategy_id"] == sid), {})
                if result:
                    rows.append(row_from_result(
                        underlying=name, strategy_id=sid, from_date=from_date, to_date=to_date,
                        result=result, coverage=cov, trading_days=trading_days,
                    ))

        report = FinalResearchReport(
            run_id=run_id,
            requested_from=(requested_from or date(2022, 1, 1)).isoformat(),
            requested_to=(requested_to or date.today()).isoformat(),
            limitation=limitation,
            availability=availability,
            coverage=coverage_reports,
            rows=rows,
            rankings={name: rank_strategies(rows, name) for name in names},
        )
        out_dir = ROOT / "reports" / "strategy_research"
        write_final_reports(report, out_dir)
        print_summary_table(rows)
        return report

    def run_backtest_sync(
        self,
        underlyings: list[str],
        from_date: date,
        to_date: date,
        strategies: list[str] | None = None,
        *,
        splits: dict | None = None,
    ) -> dict[str, Any]:
        """Run strategy research synchronously (for CLI scripts)."""
        from src.backtest.strategy_comparator import compare_strategies

        _refuse_live()
        run_id = uuid.uuid4().hex[:12]
        strategy_ids = strategies or [
            sid for sid, cfg in self.research_config.strategies.items() if cfg.get("enabled", True)
        ]
        all_results = []
        comparisons = {}
        for underlying in underlyings:
            name = underlying.upper()
            if not self.has_underlying_data(name, from_date, to_date):
                raise HistoricalDataError(
                    "UNDERLYING_DATA_MISSING",
                    f"No stored underlying candles for {name}. {UNDERLYING_MISSING_HINT}",
                )
            key = self.underlying_key(name)
            candles = self.read_underlying(name, from_date, to_date)
            candles["symbol"] = name
            contracts = self._contracts_from_store(name)
            premium = self._premium_history(name, from_date, to_date)
            results, comparison = compare_strategies(
                research_config=self.research_config,
                strategy_config=self.strategy_config,
                underlying=name,
                underlying_candles=candles,
                premium_history=premium,
                contracts=contracts,
                strategy_ids=strategy_ids,
                run_id=run_id,
                master_resolver=self._research_resolver,
                splits=splits,
            )
            all_results.extend(results)
            comparisons[name] = comparison.as_dict()
        payload = {
            "run_id": run_id,
            "status": "COMPLETED",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "underlyings": [u.upper() for u in underlyings],
            "strategies": strategy_ids,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "results": [r.as_dict() for r in all_results],
            "comparisons": comparisons,
        }
        self.metadata.set(RUN_PREFIX + run_id, payload)
        self._write_reports(run_id, all_results, comparisons)
        return payload

    def status(self) -> dict[str, Any]:
        jobs = HistoricalDownloader(self.client, self.store, self.catalog, self.metadata).jobs()
        latest = jobs[0] if jobs else None
        return {
            "token_configured": token_source() is not None,
            "data_source": self.research_config.data_source_label,
            "timeframe": self.research_config.timeframe,
            "session": self.research_config.session.model_dump(),
            "max_daily_loss_inr": self.research_config.risk.max_daily_loss_inr,
            "availability": self.catalog.availability,
            "latest_job": latest,
            "summary": self.store.summary(self.research_config.underlyings),
            "download_running": _download_thread is not None and _download_thread.is_alive(),
            "backtest_running": _backtest_thread is not None and _backtest_thread.is_alive(),
        }

    def summary(self) -> dict[str, Any]:
        return {"summary": self.store.summary(self.research_config.underlyings),
                "availability": self.catalog.availability}

    def start_download(self, underlyings: list[str], from_date: date, to_date: date, *,
                       include_options: bool = True, include_futures: bool = True) -> dict[str, Any]:
        global _download_thread
        _refuse_live()
        with DOWNLOAD_LOCK:
            if _download_thread and _download_thread.is_alive():
                raise HistoricalDataError("DOWNLOAD_RUNNING", "A historical download is already running")
            downloader = HistoricalDownloader(self.client, self.store, self.catalog, self.metadata,
                                              interval=self.research_config.interval_minutes,
                                              strike_window=int(self.research_config.download.get("strike_window", 4)))
            job = downloader.new_job(underlyings, from_date, to_date, include_options=include_options,
                                     include_futures=include_futures)

            def _run():
                downloader.run(job)

            _download_thread = threading.Thread(target=_run, name=f"historical-download-{job.job_id}", daemon=True)
            _download_thread.start()
            return job.as_dict()

    def _contracts_from_store(self, underlying: str) -> list:
        from src.options.models import OptionContract, OptionType
        paths = self.store.files("options", underlying)
        contracts = []
        for path in paths:
            frame = pd.read_parquet(path, columns=["instrument_key", "underlying", "expiry", "strike", "option_type"])
            if frame.empty:
                continue
            row = frame.iloc[0]
            contracts.append(OptionContract(
                instrument_key=str(row["instrument_key"]),
                trading_symbol=path.stem,
                underlying=str(row["underlying"]).upper(),
                expiry=date.fromisoformat(str(row["expiry"])[:10]),
                strike=float(row["strike"]),
                option_type=OptionType(str(row["option_type"]).upper()),
                lot_size=65 if underlying.upper() == "NIFTY" else 30,
            ))
        return contracts

    def _premium_history(self, underlying: str, start: date, end: date) -> dict[str, pd.DataFrame]:
        return self.store.read_options_by_key(underlying.upper(), start=start, end=end)

    def start_backtest(self, underlyings: list[str], from_date: date, to_date: date,
                       strategies: list[str] | None = None) -> dict[str, Any]:
        global _backtest_thread
        _refuse_live()
        run_id = uuid.uuid4().hex[:12]
        strategy_ids = strategies or [sid for sid, cfg in self.research_config.strategies.items()
                                      if cfg.get("enabled", True)]
        payload = {"run_id": run_id, "status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
                   "underlyings": [u.upper() for u in underlyings], "strategies": strategy_ids,
                   "from_date": from_date.isoformat(), "to_date": to_date.isoformat()}
        self.metadata.set(RUN_PREFIX + run_id, payload)

        def _run():
            from src.backtest.strategy_comparator import compare_strategies
            all_results = []
            comparisons = {}
            try:
                for underlying in underlyings:
                    key = self.catalog.underlying_key(underlying)
                    candles = self.store.read("underlying", underlying, key, start=from_date, end=to_date)
                    if candles.empty:
                        raise HistoricalDataError(
                            "UNDERLYING_DATA_MISSING",
                            f"No stored underlying candles for {underlying}. {UNDERLYING_MISSING_HINT}",
                        )
                    candles["symbol"] = underlying.upper()
                    contracts = self._contracts_from_store(underlying)
                    premium = self._premium_history(underlying, from_date, to_date)
                    results, comparison = compare_strategies(
                        research_config=self.research_config, strategy_config=self.strategy_config,
                        underlying=underlying, underlying_candles=candles, premium_history=premium,
                        contracts=contracts, strategy_ids=strategy_ids, run_id=run_id,
                        master_resolver=self._research_resolver,
                    )
                    all_results.extend(results)
                    comparisons[underlying] = comparison.as_dict()
                payload.update(status="COMPLETED", finished_at=datetime.now(timezone.utc).isoformat(),
                               results=[r.as_dict() for r in all_results], comparisons=comparisons)
                self._write_reports(run_id, all_results, comparisons)
            except Exception as exc:
                payload.update(status="FAILED", finished_at=datetime.now(timezone.utc).isoformat(),
                               message=f"{type(exc).__name__}: {exc}")
            self.metadata.set(RUN_PREFIX + run_id, payload)

        with BACKTEST_LOCK:
            if _backtest_thread and _backtest_thread.is_alive():
                raise HistoricalDataError("BACKTEST_RUNNING", "A research backtest is already running")
            _backtest_thread = threading.Thread(target=_run, name=f"options-backtest-{run_id}", daemon=True)
            _backtest_thread.start()
        return {"run_id": run_id, "status": "RUNNING"}

    def backtest_status(self, run_id: str) -> dict[str, Any]:
        data = self.metadata.get(RUN_PREFIX + run_id)
        if not data:
            raise HistoricalDataError("RUN_NOT_FOUND", f"Unknown backtest run {run_id}")
        return data

    def backtest_result(self, run_id: str) -> dict[str, Any]:
        data = self.backtest_status(run_id)
        if data.get("status") == "RUNNING":
            raise HistoricalDataError("RUN_IN_PROGRESS", "Backtest is still running")
        return data

    def backtest_signals(self, run_id: str) -> list[dict]:
        data = self.backtest_result(run_id)
        signals = []
        for result in data.get("results", []):
            signals.extend(result.get("signals", []))
        return signals

    def strategies(self) -> list[dict[str, Any]]:
        from src.backtest.research_strategies import strategy_catalog
        return strategy_catalog(self.research_config)

    def _write_reports(self, run_id: str, results, comparisons) -> None:
        out = ROOT / "reports" / "strategy_research" / run_id
        out.mkdir(parents=True, exist_ok=True)
        summary = {"run_id": run_id, "comparisons": comparisons,
                   "results": [r.as_dict() for r in results]}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        rows = []
        for result in results:
            m = result.metrics.as_dict()
            rows.append({"underlying": result.underlying, "strategy_id": result.strategy_id, "split": result.split,
                         "status": result.status, **m})
        if rows:
            with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        trades = []
        for result in results:
            for trade in result.trades:
                trades.append({**trade, "underlying": result.underlying, "strategy_id": result.strategy_id,
                               "split": result.split})
        if trades:
            with (out / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trades[0].keys()))
                writer.writeheader()
                writer.writerows(trades)
        signals = []
        for result in results:
            for signal in result.signals:
                signals.append({**signal, "underlying": result.underlying, "strategy_id": result.strategy_id,
                                "split": result.split})
        if signals:
            with (out / "signals.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(signals[0].keys()))
                writer.writeheader()
                writer.writerows(signals)
