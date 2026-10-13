"""Parquet storage for historical candles.

Layout (partitioned by underlying / year / month / expiry):
  {root}/options/{UNDERLYING}/year=YYYY/month=MM/expiry=YYYY-MM-DD/{key}.parquet
  {root}/futures/{UNDERLYING}/year=YYYY/month=MM/expiry=YYYY-MM-DD/{key}.parquet
  {root}/underlying/{UNDERLYING}/year=YYYY/month=MM/{key}.parquet

Candle data lives only here; the database stores job metadata.
"""
from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .models import IST, HistoricalDataError

KINDS = ("options", "futures", "underlying")
DEFAULT_ROOT = Path("data/historical")


def safe_key(instrument_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(instrument_key)).strip("_")


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise HistoricalDataError("PARQUET_UNAVAILABLE", "pyarrow is required for Parquet storage") from exc


class ParquetCandleStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()

    def _base(self, kind: str, underlying: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"Unknown candle kind {kind!r}")
        return self.root / kind / safe_key(str(underlying).upper())

    def _path(self, kind: str, underlying: str, year: int, month: int, instrument_key: str,
              expiry: date | None) -> Path:
        path = self._base(kind, underlying) / f"year={year:04d}" / f"month={month:02d}"
        if kind != "underlying":
            if expiry is None:
                raise ValueError(f"{kind} candles require an expiry partition")
            path = path / f"expiry={expiry.isoformat()}"
        return path / f"{safe_key(instrument_key)}.parquet"

    @staticmethod
    def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        if data["timestamp"].dt.tz is None:
            data["timestamp"] = data["timestamp"].dt.tz_localize(IST)
        data["timestamp"] = data["timestamp"].dt.tz_convert(IST)
        if "expiry" in data:
            data["expiry"] = data["expiry"].astype(str)
        return data

    def write(self, kind: str, underlying: str, instrument_key: str, frame: pd.DataFrame,
              *, expiry: date | None = None) -> int:
        """Merge ``frame`` into the monthly partitions; returns rows newly added."""
        if frame is None or frame.empty:
            return 0
        _require_parquet()
        data = self._normalise(frame)
        added = 0
        with self._lock:
            for (year, month), part in data.groupby([data["timestamp"].dt.year, data["timestamp"].dt.month]):
                path = self._path(kind, underlying, int(year), int(month), instrument_key, expiry)
                path.parent.mkdir(parents=True, exist_ok=True)
                before = 0
                if path.exists():
                    existing = self._normalise(pd.read_parquet(path))
                    before = len(existing)
                    part = pd.concat([existing, part], ignore_index=True)
                part = (part.drop_duplicates("timestamp", keep="last")
                        .sort_values("timestamp").reset_index(drop=True))
                tmp = path.with_suffix(".parquet.tmp")
                part.to_parquet(tmp, index=False)
                tmp.replace(path)
                added += len(part) - before
        return added

    def files(self, kind: str, underlying: str, instrument_key: str | None = None,
              expiry: date | None = None) -> list[Path]:
        base = self._base(kind, underlying)
        if not base.exists():
            return []
        pattern = "year=*/month=*/"
        if kind != "underlying":
            pattern += f"expiry={expiry.isoformat()}/" if expiry else "expiry=*/"
        pattern += f"{safe_key(instrument_key)}.parquet" if instrument_key else "*.parquet"
        return sorted(base.glob(pattern))

    def read(self, kind: str, underlying: str, instrument_key: str | None = None, *,
             expiry: date | None = None, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        paths = self.files(kind, underlying, instrument_key, expiry)
        if not paths:
            return pd.DataFrame()
        _require_parquet()
        frame = pd.concat([self._normalise(pd.read_parquet(p)) for p in paths], ignore_index=True)
        if start is not None:
            frame = frame[frame["timestamp"].dt.date >= start]
        if end is not None:
            frame = frame[frame["timestamp"].dt.date <= end]
        keys = ["instrument_key", "timestamp"] if "instrument_key" in frame else ["timestamp"]
        return frame.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)

    def has_month(self, kind: str, underlying: str, instrument_key: str, year: int, month: int,
                  expiry: date | None = None) -> bool:
        return self._path(kind, underlying, year, month, instrument_key, expiry).exists()

    def read_options_by_key(
        self,
        underlying: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """All stored option frames keyed by ``instrument_key`` (not filename stem)."""
        indexed: dict[str, pd.DataFrame] = {}
        for path in self.files("options", underlying):
            part = self._normalise(pd.read_parquet(path))
            if part.empty or "instrument_key" not in part:
                continue
            key = str(part.iloc[0]["instrument_key"])
            if start is not None:
                part = part[part["timestamp"].dt.date >= start]
            if end is not None:
                part = part[part["timestamp"].dt.date <= end]
            if part.empty:
                continue
            if key in indexed:
                part = pd.concat([indexed[key], part], ignore_index=True)
            indexed[key] = (
                part.drop_duplicates(["instrument_key", "timestamp"], keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        return indexed

    def summary(self, underlyings: Iterable[str] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for kind in KINDS:
            kind_root = self.root / kind
            if not kind_root.exists():
                continue
            names = [p.name for p in kind_root.iterdir() if p.is_dir()]
            for name in names:
                if underlyings and name not in {safe_key(u.upper()) for u in underlyings}:
                    continue
                paths = sorted((kind_root / name).rglob("*.parquet"))
                if not paths:
                    continue
                rows, first, last, keys, expiries = 0, None, None, set(), set()
                for path in paths:
                    ts = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
                    rows += len(ts)
                    if len(ts):
                        first = ts.min() if first is None else min(first, ts.min())
                        last = ts.max() if last is None else max(last, ts.max())
                    keys.add(path.stem)
                    for part in path.parts:
                        if part.startswith("expiry="):
                            expiries.add(part.split("=", 1)[1])
                result.setdefault(name, {})[kind] = {
                    "files": len(paths), "rows": rows, "instruments": len(keys), "expiries": len(expiries),
                    "first": first.isoformat() if first is not None else None,
                    "last": last.isoformat() if last is not None else None,
                }
        return result
