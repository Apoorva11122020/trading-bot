from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(".").resolve()

JSONL = ROOT / "reports" / "realtime_signal_accuracy" / "session_20261007.jsonl"
CSV = ROOT / "reports" / "live_signal_accuracy" / "2026-10-07.csv"

print("=" * 90)
print("7-OCT-2026 S2 RECORDED-CANDLE RECONSTRUCTION AUDIT")
print("=" * 90)
print("PAPER ONLY / READ ONLY")
print()

# ---------------------------------------------------------------------
# 1. LOAD JSONL
# ---------------------------------------------------------------------
records = []

with JSONL.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append((line_no, json.loads(line)))
        except Exception as exc:
            print(f"BAD JSON LINE {line_no}: {exc}")

print(f"JSONL records: {len(records)}")

# ---------------------------------------------------------------------
# 2. EXTRACT CANDLES
# ---------------------------------------------------------------------
candles = []

for line_no, rec in records:
    if rec.get("type") != "CANDLE":
        continue

    c = rec.get("candle") or {}
    ts = c.get("timestamp")

    if not ts:
        continue

    candles.append({
        "line": line_no,
        "timestamp": pd.Timestamp(ts),
        "open": c.get("open"),
        "high": c.get("high"),
        "low": c.get("low"),
        "close": c.get("close"),
        "volume": c.get("volume"),
        "complete": rec.get("complete"),
        "decision": rec.get("decision"),
    })

candles_df = pd.DataFrame(candles)

if candles_df.empty:
    print("NO CANDLES FOUND")
    raise SystemExit(1)

candles_df = candles_df.sort_values("timestamp")

# ---------------------------------------------------------------------
# 3. DEDUPLICATE RESTARTED SESSION CANDLES
# ---------------------------------------------------------------------
before = len(candles_df)

dupes = candles_df[candles_df.duplicated("timestamp", keep=False)].copy()

unique_df = (
    candles_df
    .sort_values(["timestamp", "line"])
    .drop_duplicates("timestamp", keep="last")
    .sort_values("timestamp")
    .reset_index(drop=True)
)

after = len(unique_df)

print()
print("=" * 90)
print("RECORDED NIFTY CANDLES")
print("=" * 90)

print(f"Raw CANDLE records       : {before}")
print(f"Duplicate timestamp rows : {before - after}")
print(f"Unique 5m candles        : {after}")
print(f"First recorded candle    : {unique_df.iloc[0]['timestamp']}")
print(f"Last recorded candle     : {unique_df.iloc[-1]['timestamp']}")

if not dupes.empty:
    print()
    print("DUPLICATE TIMESTAMPS:")
    print(
        dupes[["timestamp", "line", "open", "high", "low", "close"]]
        .to_string(index=False)
    )

# ---------------------------------------------------------------------
# 4. EXPECTED SESSION COVERAGE
# ---------------------------------------------------------------------
expected = pd.date_range(
    "2026-10-07 09:15:00+05:30",
    "2026-10-07 15:25:00+05:30",
    freq="5min",
)

recorded_ts = set(unique_df["timestamp"])

missing = [ts for ts in expected if ts not in recorded_ts]

print()
print("=" * 90)
print("SESSION COVERAGE")
print("=" * 90)

print(f"Expected NSE 5m candles : {len(expected)}")
print(f"Recorded unique candles : {len(recorded_ts)}")
print(f"Missing candles         : {len(missing)}")

if missing:
    print()
    print("FIRST 30 MISSING CANDLES:")
    for ts in missing[:30]:
        print(" ", ts)

# ---------------------------------------------------------------------
# 5. OPENING RANGE COVERAGE
# ---------------------------------------------------------------------
opening_range = unique_df[
    (unique_df["timestamp"] >= pd.Timestamp("2026-10-07 09:15:00+05:30")) &
    (unique_df["timestamp"] <= pd.Timestamp("2026-10-07 09:30:00+05:30"))
].copy()

print()
print("=" * 90)
print("OPENING RANGE")
print("=" * 90)

print(f"Opening-range candles recorded: {len(opening_range)}")

if not opening_range.empty:
    print(opening_range[
        ["timestamp", "open", "high", "low", "close", "complete"]
    ].to_string(index=False))

    print()
    print("Opening Range HIGH:", opening_range["high"].max())
    print("Opening Range LOW :", opening_range["low"].min())
else:
    print("OPENING RANGE NOT RECORDED")
    print("Therefore the 09:35 S2 signal cannot be independently regenerated")
    print("from this JSONL alone.")

# ---------------------------------------------------------------------
# 6. LOAD ACTUAL 7-OCT SIGNAL REPORT
# ---------------------------------------------------------------------
print()
print("=" * 90)
print("ACTUAL 7-OCT SIGNAL REPORT")
print("=" * 90)

if not CSV.exists():
    print("CSV REPORT NOT FOUND:", CSV)
    raise SystemExit(1)

actual = pd.read_csv(CSV)

print("Columns:")
print(list(actual.columns))

print()
print("Rows:")
print(actual.to_string(index=False))

# ---------------------------------------------------------------------
# 7. NORMALIZE ACTUAL SIGNAL TIMES
# ---------------------------------------------------------------------
if "time" in actual.columns:
    actual["signal_time"] = pd.to_datetime(actual["time"])
elif "timestamp" in actual.columns:
    actual["signal_time"] = pd.to_datetime(actual["timestamp"])
else:
    actual["signal_time"] = pd.NaT

# ---------------------------------------------------------------------
# 8. CHECK WHETHER SIGNAL CANDLE EXISTS
# ---------------------------------------------------------------------
print()
print("=" * 90)
print("SIGNAL-CANDLE AVAILABILITY")
print("=" * 90)

for _, row in actual.iterrows():
    ts = row["signal_time"]

    if pd.isna(ts):
        continue

    ts = pd.Timestamp(ts)

    match = unique_df[unique_df["timestamp"] == ts]

    print()
    print(f"SIGNAL: {ts}")

    if match.empty:
        print("  UNDERLYING CANDLE: MISSING")
        print("  -> Cannot independently verify this signal from JSONL")
    else:
        c = match.iloc[0]
        print("  UNDERLYING CANDLE: FOUND")
        print(
            f"  O={c['open']} H={c['high']} "
            f"L={c['low']} C={c['close']}"
        )

    if "direction" in row:
        print("  Direction:", row["direction"])

    if "signal_price" in row:
        print("  Reported signal price:", row["signal_price"])

# ---------------------------------------------------------------------
# 9. IMPORTANT KNOWN SIGNALS
# ---------------------------------------------------------------------
known = [
    "2026-10-07 09:35:00+05:30",
    "2026-10-07 10:25:00+05:30",
    "2026-10-07 10:55:00+05:30",
    "2026-10-07 11:00:00+05:30",
    "2026-10-07 11:05:00+05:30",
    "2026-10-07 12:00:00+05:30",
    "2026-10-07 12:05:00+05:30",
]

print()
print("=" * 90)
print("KNOWN 7 SIGNAL TIMESTAMPS")
print("=" * 90)

for x in known:
    ts = pd.Timestamp(x)
    found = unique_df[unique_df["timestamp"] == ts]

    if found.empty:
        print(f"{ts} -> MISSING")
    else:
        c = found.iloc[0]
        print(
            f"{ts} -> FOUND "
            f"O={c['open']} H={c['high']} "
            f"L={c['low']} C={c['close']}"
        )

# ---------------------------------------------------------------------
# 10. LATER CANDLE PATH
# ---------------------------------------------------------------------
print()
print("=" * 90)
print("CANDLE PATH AROUND ACTUAL SIGNALS")
print("=" * 90)

for signal_time in [
    "2026-10-07 10:25:00+05:30",
    "2026-10-07 10:55:00+05:30",
    "2026-10-07 11:00:00+05:30",
    "2026-10-07 11:05:00+05:30",
    "2026-10-07 12:00:00+05:30",
    "2026-10-07 12:05:00+05:30",
]:
    ts = pd.Timestamp(signal_time)

    window = unique_df[
        (unique_df["timestamp"] >= ts - pd.Timedelta(minutes=10)) &
        (unique_df["timestamp"] <= ts + pd.Timedelta(minutes=15))
    ]

    print()
    print(f"--- {ts} ---")

    if window.empty:
        print("No recorded candles")
    else:
        print(
            window[
                ["timestamp", "open", "high", "low", "close", "complete", "decision"]
            ].to_string(index=False)
        )

# ---------------------------------------------------------------------
# 11. SAVE RECONSTRUCTED CANDLES
# ---------------------------------------------------------------------
out = ROOT / "reports" / "forensics"
out.mkdir(parents=True, exist_ok=True)

out_file = out / "s2_2026-10-07_reconstructed_underlying_candles.csv"

unique_df.to_csv(out_file, index=False)

print()
print("=" * 90)
print("FINAL RESULT")
print("=" * 90)

if len(opening_range) < 4:
    print("STATUS: PARTIAL RECONSTRUCTION")
    print()
    print("Reason:")
    print("The JSONL starts after the market open, so the complete")
    print("09:15-09:30 opening range is unavailable.")
    print()
    print("Therefore:")
    print("1. We CAN reconstruct the recorded 5m candle stream.")
    print("2. We CAN verify later signal candles.")
    print("3. We CANNOT independently regenerate the 09:35 signal.")
    print("4. We should NOT modify S2 based on this evidence.")
else:
    print("STATUS: OPENING RANGE AVAILABLE")
    print("A full S2 replay can be attempted.")

print()
print("Reconstructed candle file:")
print(out_file)
print()
print("NO STRATEGY FILE MODIFIED")
print("NO BACKTEST FILE MODIFIED")
print("NO OOS/Q3 DATA TOUCHED")
print("NO ORDERS PLACED")
