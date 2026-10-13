from pathlib import Path
import re

ROOT = Path(".").resolve()

print("=" * 100)
print("7-OCT S2 SIGNAL SOURCE / DATABASE TRACE")
print("=" * 100)
print()

# ------------------------------------------------------------------
# 1. Inspect live premium accuracy script
# ------------------------------------------------------------------
targets = [
    ROOT / "scripts" / "run_live_premium_signal_accuracy.py",
    ROOT / "scripts" / "run_live_signal_accuracy.py",
    ROOT / "src" / "database",
    ROOT / "src" / "paper_trading",
    ROOT / "src" / "backtest" / "current_day_s2.py",
]

patterns = [
    "Database",
    "signals",
    "point_in_time_signals",
    "signal_tracker",
    "register",
    "resolve",
    "intraday_candles",
    "candles",
    "Nifty 50",
    "2026-10-07",
    "session",
    "opening_range",
]

def inspect_file(path):
    print()
    print("=" * 100)
    print("FILE:", path.relative_to(ROOT))
    print("=" * 100)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print("READ ERROR:", exc)
        return

    lines = text.splitlines()

    hits = []

    for i, line in enumerate(lines):
        low = line.lower()
        if any(p.lower() in low for p in patterns):
            hits.append(i)

    # unique nearby ranges
    ranges = []
    for i in hits:
        start = max(0, i - 2)
        end = min(len(lines), i + 4)

        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    print(f"Lines: {len(lines)}")
    print(f"Matching regions: {len(ranges)}")

    for start, end in ranges[:80]:
        print()
        print(f"--- lines {start + 1}-{end} ---")
        for n in range(start, end):
            print(f"{n + 1:5}: {lines[n]}")

    if len(ranges) > 80:
        print(f"... {len(ranges) - 80} additional regions omitted ...")


# inspect important files
for target in targets:
    if target.is_file():
        inspect_file(target)


# ------------------------------------------------------------------
# 2. List database files
# ------------------------------------------------------------------
print()
print("=" * 100)
print("DATABASE FILES")
print("=" * 100)

db_paths = []

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    name = p.name.lower()

    if (
        name.endswith(".db")
        or name.endswith(".sqlite")
        or name.endswith(".sqlite3")
        or name.endswith(".jsonl")
        or "database" in str(p.parent).lower()
    ):
        db_paths.append(p)

for p in sorted(db_paths):
    try:
        size = p.stat().st_size
    except:
        size = -1

    print(f"{size:12}  {p.relative_to(ROOT)}")


# ------------------------------------------------------------------
# 3. Search project for actual 7-Oct signal IDs/timestamps
# ------------------------------------------------------------------
print()
print("=" * 100)
print("SEARCHING ACTUAL SIGNAL TIMESTAMPS")
print("=" * 100)

signal_times = [
    "09:35",
    "10:25",
    "10:55",
    "11:00",
    "11:05",
    "12:00",
    "12:05",
]

search_roots = [
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "reports",
    ROOT / "dashboard",
]

extensions = {
    ".py",
    ".json",
    ".jsonl",
    ".csv",
    ".sql",
    ".txt",
    ".yaml",
    ".yml",
}

found = {}

for base in search_roots:
    if not base.exists():
        continue

    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except:
            continue

        matches = []

        for t in signal_times:
            if t in text:
                matches.append(t)

        if matches:
            found[p] = sorted(set(matches))

for p, matches in sorted(found.items()):
    print()
    print(p.relative_to(ROOT))
    print("  matches:", ", ".join(matches))


# ------------------------------------------------------------------
# 4. Search exact signal IDs from the 7-Oct report
# ------------------------------------------------------------------
print()
print("=" * 100)
print("SEARCHING EXACT SIGNAL IDs")
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

for base in search_roots:
    if not base.exists():
        continue

    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except:
            continue

        matches = [x for x in ids if x in text]

        if matches:
            print()
            print(p.relative_to(ROOT))
            print("  IDs:", ", ".join(matches))


# ------------------------------------------------------------------
# 5. Search for database schema / signal tables
# ------------------------------------------------------------------
print()
print("=" * 100)
print("SEARCHING SIGNAL TABLE / DB SCHEMA REFERENCES")
print("=" * 100)

schema_patterns = [
    "CREATE TABLE",
    "signals",
    "paper_signals",
    "signal_id",
    "INSERT INTO",
    "SELECT",
    "sqlite",
    "sqlite3",
    "postgres",
    "SQLAlchemy",
]

for base in [ROOT / "src", ROOT / "scripts"]:
    if not base.exists():
        continue

    for p in base.rglob("*.py"):
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except:
            continue

        hits = []

        for i, line in enumerate(lines):
            low = line.lower()

            if any(x.lower() in low for x in schema_patterns):
                hits.append(i)

        if hits:
            print()
            print("---", p.relative_to(ROOT), "---")

            shown = set()

            for i in hits:
                start = max(0, i - 1)
                end = min(len(lines), i + 3)

                key = (start, end)

                if key in shown:
                    continue

                shown.add(key)

                for n in range(start, end):
                    print(f"{n + 1:5}: {lines[n]}")


# ------------------------------------------------------------------
# 6. Inspect report directory timestamps
# ------------------------------------------------------------------
print()
print("=" * 100)
print("7-OCT REPORT ARTIFACTS")
print("=" * 100)

report_dirs = [
    ROOT / "reports" / "live_signal_accuracy",
    ROOT / "reports" / "realtime_signal_accuracy",
    ROOT / "reports" / "strategy_research",
]

for d in report_dirs:
    if not d.exists():
        continue

    print()
    print("DIRECTORY:", d.relative_to(ROOT))

    for p in sorted(d.iterdir()):
        if "2026-10-07" in p.name or "20261007" in p.name:
            try:
                st = p.stat()
                print(
                    f"{p.name:45} "
                    f"size={st.st_size:8} "
                    f"modified={st.st_mtime}"
                )
            except:
                print(p.name)


print()
print("=" * 100)
print("TRACE COMPLETE")
print("=" * 100)
print()
print("NO STRATEGY FILE MODIFIED")
print("NO BACKTEST FILE MODIFIED")
print("NO OOS/Q3 DATA TOUCHED")
print("NO BROKER ORDER PLACED")
