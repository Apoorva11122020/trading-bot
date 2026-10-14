#!/usr/bin/env python3
"""Run/resume the 3x3 PAPER-only pivot futures parameter sweep and compare results."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
# New versioned output avoids silently reusing old summaries created before the
# fee-model audit. OOS rows are reported only for validation, never used to rank candidates.
SWEEP_ROOT = REPORTS / "pivot_sweep_cost_audit_20261014"
RR_VALUES = (1.0, 1.5, 2.0)
ATR_VALUES = (0.1, 0.2, 0.3)
SPLITS = ("DEVELOPMENT", "OOS")


def main() -> int:
    all_rows: list[pd.DataFrame] = []
    for rr in RR_VALUES:
        for atr_buffer in ATR_VALUES:
            name = f"pivot_sweep_rr{rr:g}_atr{atr_buffer:g}"
            out_dir = SWEEP_ROOT / name
            summary_path = out_dir / "summary.csv"
            if summary_path.is_file():
                print(f"[SKIP existing] RR={rr:g}, ATR={atr_buffer:g}: {summary_path.relative_to(ROOT)}", flush=True)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "classic_pivot_futures_backtest.py"),
                    "--cost-model", "upstox",
                    "--slippage-points", "1",
                    "--rr", str(rr),
                    "--atr-buffer", str(atr_buffer),
                    "--out-dir", str(out_dir),
                ]
                print(f"[RUN] RR={rr:g}, ATR={atr_buffer:g}", flush=True)
                log_path = out_dir / "run.log"
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
                if proc.returncode != 0:
                    print(f"FAILED: {name}. Last log lines:", flush=True)
                    try:
                        print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-35:]))
                    except OSError:
                        pass
                    return proc.returncode or 1
                if not summary_path.is_file():
                    print(f"FAILED: summary not produced for {name}", flush=True)
                    return 1

            df = pd.read_csv(summary_path)
            needed = {"symbol", "volume_filter", "split", "trades", "net_points", "profit_factor", "max_drawdown_points"}
            missing = needed - set(df.columns)
            if missing:
                print(f"FAILED: {summary_path} missing columns: {sorted(missing)}", flush=True)
                return 1
            df = df[(df["volume_filter"].astype(str).str.lower() == "true") & df["split"].isin(SPLITS)].copy()
            df.insert(0, "atr_buffer", atr_buffer)
            df.insert(0, "rr", rr)
            df.insert(0, "combination", name)
            all_rows.append(df)

    if not all_rows:
        print("No summary rows found.", flush=True)
        return 1

    results = pd.concat(all_rows, ignore_index=True)
    for col in ("trades", "net_points", "profit_factor", "max_drawdown_points"):
        results[col] = pd.to_numeric(results[col], errors="coerce")
    results = results.sort_values(["symbol", "split", "profit_factor"], ascending=[True, True, False])
    csv_path = SWEEP_ROOT / "pivot_sweep_comparison.csv"
    results.to_csv(csv_path, index=False)

    cols = ["combination", "symbol", "split", "trades", "net_points", "profit_factor", "max_drawdown_points"]
    dev = results[results["split"] == "DEVELOPMENT"].copy()
    dev = dev.sort_values(["symbol", "profit_factor"], ascending=[True, False])
    oos = results[results["split"] == "OOS"].copy().sort_values(["symbol", "combination"])
    print("\nDEVELOPMENT RANKING ONLY — volume filter ON, estimated Upstox costs, 1 point/side slippage")
    print(dev[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nOOS VALIDATION ONLY — DO NOT use these rows to select parameters")
    print(oos[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved comparison: {csv_path}")
    print("PAPER | RESEARCH ONLY | ORDERS: 0")
    print("Select candidates using DEVELOPMENT only; OOS remains validation and must not drive parameter tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
