import sqlite3
from pathlib import Path

db = Path("data/trading.db")

print("=" * 100)
print("S2 7-OCT DATABASE FORENSICS")
print("=" * 100)

con = sqlite3.connect(db)

# Tables
print("\nTABLES:")
tables = con.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name
""").fetchall()

for (name,) in tables:
    print(" -", name)

# Schema
print("\n" + "=" * 100)
print("SCHEMA / SIGNAL-RELATED TABLES")
print("=" * 100)

for (name,) in tables:
    if any(x in name.lower() for x in ["signal", "trade", "paper", "order"]):
        print(f"\n--- TABLE: {name} ---")
        cols = con.execute(f'PRAGMA table_info("{name}")').fetchall()
        for c in cols:
            print(c)

# Search all tables for 2026-10-07
print("\n" + "=" * 100)
print("7-OCT RECORDS")
print("=" * 100)

for (name,) in tables:
    try:
        cols = con.execute(f'PRAGMA table_info("{name}")').fetchall()
        colnames = [c[1] for c in cols]

        date_cols = [
            c for c in colnames
            if any(x in c.lower() for x in [
                "time", "date", "created", "signal"
            ])
        ]

        if not date_cols:
            continue

        rows = []

        for col in date_cols:
            try:
                q = f'''
                    SELECT *
                    FROM "{name}"
                    WHERE CAST("{col}" AS TEXT) LIKE '2026-10-07%'
                       OR CAST("{col}" AS TEXT) LIKE '2026-10-07T%'
                '''
                rows.extend(con.execute(q).fetchall())
            except:
                pass

        if rows:
            print(f"\n--- {name}: {len(rows)} matching rows ---")

            for row in rows[:100]:
                print(row)

    except Exception as e:
        print("ERROR", name, e)

# Exact IDs
ids = [
    "4a662f37f2ecff77",
    "8821338d7210a883",
    "bedc11b60ea97656",
    "9230d26a5042bb35",
    "ab1fedc311aca8db",
    "88fe3f1a6387375f",
    "4ad2c500dc242912",
]

print("\n" + "=" * 100)
print("EXACT SIGNAL ID LOOKUP")
print("=" * 100)

for (name,) in tables:
    try:
        cols = con.execute(f'PRAGMA table_info("{name}")').fetchall()
        colnames = [c[1] for c in cols]

        id_cols = [
            c for c in colnames
            if "id" in c.lower()
        ]

        for col in id_cols:
            for sid in ids:
                try:
                    q = f'''
                        SELECT *
                        FROM "{name}"
                        WHERE CAST("{col}" AS TEXT) = ?
                    '''
                    rows = con.execute(q, (sid,)).fetchall()

                    if rows:
                        print(f"\nTABLE={name}")
                        print(f"MATCH_COLUMN={col}")

                        print("COLUMNS:")
                        print(colnames)

                        for row in rows:
                            print("ROW:")
                            print(row)
                except:
                    pass

    except:
        pass

con.close()

print("\n" + "=" * 100)
print("FORENSICS COMPLETE")
print("READ-ONLY")
print("NO STRATEGY CHANGE")
print("NO BACKTEST CHANGE")
print("NO OOS/Q3 ACCESS")
print("NO ORDERS")
print("=" * 100)
