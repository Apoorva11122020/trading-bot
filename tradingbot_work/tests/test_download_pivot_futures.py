from __future__ import annotations

import pandas as pd
import pytest

from scripts.download_pivot_futures import load_spot_csv


def test_load_spot_csv_uses_utc_timestamp_and_ignores_zero_volume(tmp_path):
    path = tmp_path / "spot.csv"
    pd.DataFrame({
        "timestamp": [
            "2026-06-01T03:45:00Z",  # 09:15 IST
            "2026-06-01T03:50:00Z",  # 09:20 IST
            "2026-06-01T10:00:00Z",  # 15:30 IST, outside regular session
            "2026-06-02T03:45:00Z",  # outside requested date range
        ],
        "open": [100, 101, 102, 103],
        "high": [101, 102, 103, 104],
        "low": [99, 100, 101, 102],
        "close": [100.5, 101.5, 102.5, 103.5],
        "volume": [0, 0, 0, 0],
    }).to_csv(path, index=False)

    out = load_spot_csv(path, pd.Timestamp("2026-06-01").date(), pd.Timestamp("2026-06-01").date())

    assert len(out) == 2
    assert str(out["timestamp"].dt.tz) == "Asia/Kolkata"
    assert out["timestamp"].dt.strftime("%H:%M").tolist() == ["09:15", "09:20"]
    assert "volume" not in out.columns
    assert out["close"].tolist() == [100.5, 101.5]


def test_load_spot_csv_prefers_timestamp_ist_column(tmp_path):
    path = tmp_path / "spot_ist.csv"
    pd.DataFrame({
        "timestamp": ["2026-06-01T03:45:00Z"],
        "timestamp_ist": ["2026-06-01 09:15:00"],
        "open": [100], "high": [101], "low": [99], "close": [100.5],
    }).to_csv(path, index=False)

    out = load_spot_csv(path, pd.Timestamp("2026-06-01").date(), pd.Timestamp("2026-06-01").date())

    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-06-01 09:15:00", tz="Asia/Kolkata")


def test_load_spot_csv_rejects_missing_or_empty_data(tmp_path):
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"timestamp": ["2026-06-01T03:45:00Z"], "close": [100]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_spot_csv(missing, pd.Timestamp("2026-06-01").date(), pd.Timestamp("2026-06-01").date())

    empty = tmp_path / "empty.csv"
    pd.DataFrame({
        "timestamp": ["2026-06-01T10:00:00Z"],
        "open": [100], "high": [101], "low": [99], "close": [100.5],
    }).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="No regular-session spot candles"):
        load_spot_csv(empty, pd.Timestamp("2026-06-01").date(), pd.Timestamp("2026-06-01").date())
