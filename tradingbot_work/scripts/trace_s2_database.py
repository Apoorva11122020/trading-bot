from pathlib import Path
import re

ROOT = Path(".").resolve()

print("=" * 100)
print("DATABASE IMPLEMENTATION TRACE")
print("=" * 100)

# 1. src/database tree
dbroot = ROOT / "src" / "database"

print("\nDATABASE DIRECTORY:")
if dbroot.exists():
    for p in dbroot.rglob("*"):
        if p.is_file():
            print(p.relative_to(ROOT))
else:
    print("src/database NOT FOUND")

# 2. Database imports/usages
print("\n" + "=" * 100)
print("DATABASE USAGE")
print("=" * 100)

patterns = [
    "Database(",
    "signal_id",
    "signals",
    "insert_signal",
    "create_signal",
    "get_signal",
    "load_signal",
    "save_signal",
    "sqlite",
    "sqlite3",
    "SQLAlchemy",
    "SELECT",
    "INSERT",
    "UPDATE",
    "DATABASE_URL",
]

for base in [ROOT / "src", ROOT / "scripts", ROOT / "dashboard"]:
    if not base.exists():
        continue

    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".py", ".sql"}:
            continue

        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except:
            continue

        hits = []

        for i, line in enumerate(lines):
            if any(x.lower() in line.lower() for x in patterns):
                hits.append(i)

        if not hits:
            continue

        print("\n---", p.relative_to(ROOT), "---")

        shown = set()

        for i in hits:
            start = max(0, i - 3)
            end = min(len(lines), i + 6)

            key = (start, end)

            if key in shown:
                continue

            shown.add(key)

            for n in range(start, end):
                print(f"{n+1:5}: {lines[n]}")

# 3. Find actual DB files
print("\n" + "=" * 100)
print("DATABASE FILES")
print("=" * 100)

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    if p.suffix.lower() in {
        ".db",
        ".sqlite",
        ".sqlite3"
    }:
        try:
            print(
                f"{p.relative_to(ROOT)} "
                f"size={p.stat().st_size}"
            )
        except:
            print(p.relative_to(ROOT))

# 4. Exact signal IDs
print("\n" + "=" * 100)
print("EXACT 7-OCT SIGNAL IDs")
print("=" * 100)

ids = [
    "4a662f37f2ecff77",
    "8821338d7210a883",
    "bedc11b60ea97656",
    "9230d26a5042bb35",
    "ab1fedc311aca8db",
    "88fe3f1a6387375f",
    "4ad2c500dc242912",
]

for base in [ROOT / "src", ROOT / "scripts", ROOT / "dashboard", ROOT / "reports"]:
    if not base.exists():
        continue

    for p in base.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {
            ".py", ".json", ".jsonl", ".csv", ".db",
            ".sqlite", ".sqlite3", ".sql", ".txt"
        }:
            continue

        try:
            data = p.read_text(encoding="utf-8", errors="replace")
        except:
            continue

        found = [x for x in ids if x in data]

        if found:
            print("\n", p.relative_to(ROOT))
            print("  FOUND:", ", ".join(found))

print("\n" + "=" * 100)
print("TRACE COMPLETE")
print("=" * 100)
print("NO STRATEGY MODIFICATION")
print("NO BACKTEST MODIFICATION")
print("NO OOS/Q3 TOUCH")
print("NO ORDERS")
