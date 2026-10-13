
import os
import calendar
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TOKEN = (
    os.getenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN")
    or os.getenv("ACCESS_TOKEN")
)
if not TOKEN:
    raise SystemExit(
        "No Upstox token found. Check UPSTOX_MARKET_DATA_ACCESS_TOKEN or ACCESS_TOKEN in .env"
    )

# Use official instrument keys from your current Upstox instrument master.
# Verify these keys before relying on downloaded data.
INSTRUMENTS = {
    "NIFTY_50": "NSE_INDEX|Nifty 50",
    "BANK_NIFTY": "NSE_INDEX|Nifty Bank",
}

START = date(2025, 10, 1)
END = date(2026, 10, 9)
OUT = Path("data/historical/upstox_pivot_5m")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}

def month_ranges(start, end):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        first = date(year, month, 1)
        last = date(year, month, calendar.monthrange(year, month)[1])
        yield max(first, start), min(last, end)
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1

def fetch_month(key, start, end):
    encoded_key = quote(key, safe="")
    url = (
        "https://api.upstox.com/v3/historical-candle/"
        f"{encoded_key}/minutes/5/{end.isoformat()}/{start.isoformat()}"
    )
    response = requests.get(url, headers=HEADERS, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code} for {key}, {start} to {end}: "
            f"{response.text[:300]}"
        )

    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"API error for {key}: {payload}")

    candles = payload.get("data", {}).get("candles", [])
    rows = []
    for c in candles:
        rows.append({
            "timestamp": c[0],
            "open": c[1],
            "high": c[2],
            "low": c[3],
            "close": c[4],
            "volume": c[5],
            "open_interest": c[6] if len(c) > 6 else None,
        })
    return rows

for name, key in INSTRUMENTS.items():
    all_rows = []
    print(f"\nDownloading {name} ({key})")

    for start, end in month_ranges(START, END):
        print(f"  {start} to {end} ...", end=" ", flush=True)
        rows = fetch_month(key, start, end)
        print(f"{len(rows)} candles")
        all_rows.extend(rows)

    if not all_rows:
        print(f"WARNING: no candles returned for {name}. Check instrument key/access.")
        continue

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df.drop_duplicates(subset=["timestamp"])
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    df["timestamp_ist"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    path = OUT / f"{name}_5m_2025-10-01_to_2026-10-09.csv"
    df.to_csv(path, index=False)

    print(f"Saved: {path}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['timestamp_ist'].min()} -> {df['timestamp_ist'].max()}")

print("\nDownload script finished. Verify row counts and date coverage before backtesting.")
