import json
from pathlib import Path

p = Path("reports/realtime_signal_accuracy/session_20261007.jsonl")

print("=" * 100)
print("SEARCHING 7-OCT SIGNAL RECORDS FOR FILTER FEATURES")
print("=" * 100)

found = 0

for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
    try:
        x = json.loads(line)
    except Exception:
        continue

    if x.get("type") != "SIGNAL":
        continue

    found += 1

    print("\nLINE:", line_no)
    print("KEYS:", list(x.keys()))

    text = json.dumps(x, indent=2)

    for key in [
        "ema_sep_atr",
        "rsi14",
        "ema20",
        "ema50",
        "atr14",
        "trend_regime",
        "market_regime",
        "features",
        "indicators"
    ]:
        if key in text.lower():
            print("FOUND:", key)

    print(text)

print("\n" + "=" * 100)
print("TOTAL SIGNAL RECORDS:", found)
print("=" * 100)
