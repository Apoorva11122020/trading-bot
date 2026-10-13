from pathlib import Path

ROOT = Path(".")
TERMS = [
    "INTRADAY_REPLAY",
    "live_signal_outcomes",
    "register_signal",
    "register",
    "point_in_time",
    "opening_range",
    "09:30",
    "09:35",
    "S2",
    "signal_tracker",
]

print("=" * 110)
print("S2 INTRADAY_REPLAY SOURCE TRACE")
print("=" * 110)

files = []

for p in ROOT.rglob("*.py"):
    s = str(p).lower()

    if any(x in s for x in [
        ".venv",
        "__pycache__",
        "site-packages",
    ]):
        continue

    files.append(p)

for term in TERMS:
    print("\n" + "=" * 110)
    print(f"TERM: {term}")
    print("=" * 110)

    found = 0

    for p in files:
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, 1):
            if term.lower() in line.lower():
                found += 1

                start = max(1, i - 4)
                end = min(len(lines), i + 5)

                print(f"\nFILE: {p}")
                print(f"MATCH LINE: {i}")

                for n in range(start, end + 1):
                    marker = ">>" if n == i else "  "
                    print(f"{marker} {n:5}: {lines[n-1]}")

                if found >= 30:
                    print("\n[STOPPED AFTER 30 MATCHES FOR THIS TERM]")
                    break

        if found >= 30:
            break

    print(f"\nTOTAL SHOWN: {found}")

print("\n" + "=" * 110)
print("TARGET FILES")
print("=" * 110)

for p in files:
    name = p.name.lower()

    if any(x in name for x in [
        "s2",
        "signal",
        "replay",
        "live",
        "forward",
    ]):
        print(p)

print("\n" + "=" * 110)
print("READ-ONLY TRACE COMPLETE")
print("NO FILE MODIFIED")
print("NO STRATEGY CHANGE")
print("NO BACKTEST CHANGE")
print("NO OOS/Q3 ACCESS")
print("NO ORDERS")
print("=" * 110)
