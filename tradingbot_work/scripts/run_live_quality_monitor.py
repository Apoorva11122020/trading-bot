from __future__ import annotations

import sqlite3
import time
import os
from datetime import datetime
from pathlib import Path

DB = Path("data/trading.db")
CANDIDATE = "LIVE_QUALITY_GATE_V1"
POLL_SECONDS = 2


def money(value):
    if value is None:
        return "-"
    return f"Rs {value:,.2f}"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def load():
    con = sqlite3.connect(DB, timeout=2)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """
        SELECT
            timestamp,
            direction,
            decision,
            status,
            pnl,
            r_multiple
        FROM shadow_strategy_signals
        WHERE candidate = ?
        ORDER BY timestamp
        """,
        (CANDIDATE,),
    ).fetchall()

    con.close()
    return rows


def render(rows):
    today = datetime.now().astimezone().date().isoformat()

    today_rows = [
        r for r in rows
        if str(r["timestamp"]).startswith(today)
    ]

    accepted = [
        r for r in today_rows
        if str(r["decision"]).upper() == "ACCEPTED"
    ]

    skipped = [
        r for r in today_rows
        if str(r["decision"]).upper() == "SKIPPED"
    ]

    resolved = [
        r for r in accepted
        if str(r["status"]).upper() in {"WIN", "LOSS", "AMBIGUOUS"}
    ]

    wins = [
        r for r in resolved
        if str(r["status"]).upper() == "WIN"
    ]

    losses = [
        r for r in resolved
        if str(r["status"]).upper() == "LOSS"
    ]

    pending = [
        r for r in accepted
        if str(r["status"]).upper() == "PENDING"
    ]

    pnl = sum(float(r["pnl"] or 0) for r in resolved)

    accuracy = (
        (len(wins) / len(resolved) * 100)
        if resolved else None
    )

    clear()

    print("=" * 100)
    print("LIVE QUALITY GATE V1 - REAL-TIME MONITOR")
    print("=" * 100)
    print("PAPER ONLY | READ ONLY | NO ORDERS | PRODUCTION S2 UNCHANGED")
    print(f"Database : {DB}")
    print(f"Updated  : {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("TODAY")
    print("-" * 100)
    print(f"Generated : {len(today_rows)}")
    print(f"Accepted  : {len(accepted)}")
    print(f"Skipped   : {len(skipped)}")
    print(f"Resolved  : {len(resolved)}")
    print(f"Wins      : {len(wins)}")
    print(f"Losses    : {len(losses)}")
    print(f"Pending   : {len(pending)}")

    if accuracy is None:
        print("Accuracy  : N/A")
    else:
        print(f"Accuracy  : {accuracy:.1f}%")

    print(f"P&L       : {money(pnl)}")
    print()

    print("LATEST SIGNALS")
    print("-" * 100)

    latest = today_rows[-15:]

    if not latest:
        print("No Quality Gate signals recorded yet.")
    else:
        print(
            f"{'TIME':<9} "
            f"{'DIR':<5} "
            f"{'DECISION':<10} "
            f"{'STATUS':<10} "
            f"{'P&L':>14} "
            f"{'R':>8}"
        )

        for r in latest:
            ts = str(r["timestamp"])

            if "T" in ts:
                ts = ts.split("T", 1)[1][:5]

            print(
                f"{ts:<9} "
                f"{str(r['direction']):<5} "
                f"{str(r['decision']):<10} "
                f"{str(r['status']):<10} "
                f"{money(r['pnl']):>14} "
                f"{str(r['r_multiple'] if r['r_multiple'] is not None else '-'):>8}"
            )

    print()
    print("=" * 100)
    print("Refreshing every 2 seconds... Press Ctrl+C to stop.")
    print("=" * 100)


def main():
    if not DB.exists():
        print(f"Database not found: {DB}")
        return

    try:
        while True:
            try:
                rows = load()
                render(rows)
            except sqlite3.Error as exc:
                clear()
                print("=" * 100)
                print("LIVE QUALITY GATE MONITOR")
                print("=" * 100)
                print(f"Database temporarily busy/unavailable: {exc}")
                print("Retrying...")
            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nMonitor stopped. No files or strategy were changed.")


if __name__ == "__main__":
    main()
