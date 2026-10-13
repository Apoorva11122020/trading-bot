#!/usr/bin/env python3
"""Explicit Upstox Sandbox smoke test.

This script is OFF by default. It places one sandbox order only when:
  UPSTOX_RUN_REAL_SANDBOX_E2E=true
  UPSTOX_SANDBOX_ACCESS_TOKEN is set
  UPSTOX_SANDBOX_INSTRUMENT is set

It never changes the application's LIVE gate and never targets the live host.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integration.upstox_e2e import UpstoxSandboxE2EHarness


def main() -> None:
    if os.getenv("UPSTOX_RUN_REAL_SANDBOX_E2E", "false").lower() != "true":
        print("SKIPPED: set UPSTOX_RUN_REAL_SANDBOX_E2E=true for an explicit sandbox smoke test")
        return

    instrument = os.getenv("UPSTOX_SANDBOX_INSTRUMENT", "").strip()
    if not instrument:
        raise SystemExit("UPSTOX_SANDBOX_INSTRUMENT is required")

    harness = UpstoxSandboxE2EHarness()
    order_id = harness.run_real_order_smoke(
        instrument_token=instrument,
        side=os.getenv("UPSTOX_SANDBOX_SIDE", "BUY"),
        quantity=int(os.getenv("UPSTOX_SANDBOX_QUANTITY", "1")),
        order_type=os.getenv("UPSTOX_SANDBOX_ORDER_TYPE", "MARKET"),
    )
    print(f"SANDBOX ORDER PLACED: {order_id}")


if __name__ == "__main__":
    main()
