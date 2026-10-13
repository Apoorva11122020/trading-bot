from __future__ import annotations

import pytest

from src.historical_data import upstox_client


@pytest.fixture(autouse=True)
def _isolate_historical_env_file(tmp_path, monkeypatch):
    """Keep the real project .env out of token resolution during tests."""
    monkeypatch.setattr(upstox_client, "ENV_FILE", tmp_path / "no-such.env")
