import sqlite3

db = r"data\trading.db"

con = sqlite3.connect(db)

rows = con.execute("""
SELECT
    candidate,
    timestamp,
    direction,
    decision,
    skip_reason,
    status,
    pnl,
    r_multiple,
    market_regime,
    record_json
FROM shadow_strategy_signals
WHERE session_date = '2026-10-07'
ORDER BY timestamp
""").fetchall()

print("=" * 110)
print("7-OCT SHADOW STRATEGY SIGNALS")
print("=" * 110)
print("ROWS:", len(rows))
print()

for r in rows:
    print(r)

con.close()
