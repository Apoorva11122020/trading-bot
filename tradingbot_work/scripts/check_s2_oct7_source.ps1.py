from __future__ import annotations

import sqlite3
from pathlib import Path
from collections import Counter

DB = Path("data/trading.db")

SIGNAL_IDS = [
    "4a662f37f2ecff77",
    "8821338d7210a883",
    "bedc11b60ea97656",
    "9230d26a5042bb35",
    "ab1fedc311aca8db",
    "88fe3f1a6387375f",
    "4ad2c500dc242912",
]

print("=" * 100)
print("7-OCT-2026 S2 SOURCE/DATABASE RE-EVALUATION")
print("=" * 100)
print("DB:", DB.resolve())
print("Mode: READ ONLY / PAPER")
print()

if not DB.exists():
    raise SystemExit(f"Database not found: {DB}")

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

placeholders = ",".join("?" for _ in SIGNAL_IDS)

rows = con.execute(
    f"""
    SELECT
        signal_id,
        timestamp,
        direction,
        option_type,
        strike,
        expiry,
        instrument_key,
        trading_symbol,
        signal_price,
        entry_price,
        stop_loss,
        target,
        quantity,
        source,
        status,
        exit_price,
        exit_time,
        pnl,
        r_multiple,
        reason
    FROM live_signal_outcomes
    WHERE signal_id IN ({placeholders})
    ORDER BY timestamp
    """,
    SIGNAL_IDS,
).fetchall()

con.close()

print(f"FOUND SOURCE RECORDS: {len(rows)}")
print()

if not rows:
    raise SystemExit("No 7-Oct source records found.")

print("=" * 100)
print("ALL 7 SOURCE RECORDS")
print("=" * 100)

for r in rows:
    print(
        f"{r['timestamp']} | "
        f"{r['direction']} {r['option_type']} {r['strike']} | "
        f"{r['trading_symbol']} | "
        f"entry={r['entry_price']} SL={r['stop_loss']} TP={r['target']} | "
        f"status={r['status']} | "
        f"exit={r['exit_price']} | "
        f"PnL={r['pnl']} | "
        f"reason={r['reason']} | "
        f"source={r['source']}"
    )

print()
print("=" * 100)
print("SOURCE CHECK")
print("=" * 100)

sources = Counter(r["source"] for r in rows)
statuses = Counter(r["status"] for r in rows)

print("Sources:", dict(sources))
print("Statuses:", dict(statuses))

resolved = [r for r in rows if r["status"] in ("WIN", "LOSS", "AMBIGUOUS")]
wins = [r for r in rows if r["status"] == "WIN"]
losses = [r for r in rows if r["status"] == "LOSS"]
ambiguous = [r for r in rows if r["status"] == "AMBIGUOUS"]
pending = [r for r in rows if r["status"] == "PENDING"]

print()
print("=" * 100)
print("CURRENT SOURCE-RECORDED PERFORMANCE")
print("=" * 100)

print("Total signals :", len(rows))
print("Resolved      :", len(resolved))
print("Wins          :", len(wins))
print("Losses        :", len(losses))
print("Ambiguous     :", len(ambiguous))
print("Pending       :", len(pending))

if resolved:
    accuracy = len(wins) / len(resolved) * 100
    print(f"Win rate      : {accuracy:.2f}%")
else:
    print("Win rate      : N/A")

gross_profit = sum(float(r["pnl"] or 0) for r in wins)
gross_loss = abs(sum(float(r["pnl"] or 0) for r in losses))
net_pnl = sum(float(r["pnl"] or 0) for r in resolved)

print(f"Gross profit  : Rs {gross_profit:.2f}")
print(f"Gross loss    : Rs {gross_loss:.2f}")
print(f"Net P&L       : Rs {net_pnl:.2f}")

if gross_loss:
    print(f"Profit factor : {gross_profit / gross_loss:.4f}")
else:
    print("Profit factor : N/A")

if resolved:
    avg_r = sum(float(r["r_multiple"] or 0) for r in resolved) / len(resolved)
    print(f"Average R     : {avg_r:.4f}")

print()
print("=" * 100)
print("IMPORTANT")
print("=" * 100)
print("This script ONLY reads the existing INTRADAY_REPLAY source records.")
print("It does NOT call Upstox historical intraday candles.")
print("It does NOT modify S2.")
print("It does NOT modify backtest code.")
print("It does NOT touch OOS/Q3.")
print("It does NOT place orders.")
print()
