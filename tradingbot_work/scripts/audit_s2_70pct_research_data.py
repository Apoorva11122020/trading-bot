import pandas as pd
from pathlib import Path

base = Path("reports/strategy_improvement")

files = [
    "s2_feature_diagnostic.csv",
    "s2_regime_diagnostic.csv",
    "s2_time_diagnostic.csv",
    "s2_pattern_diagnostic.csv",
    "s2_win_loss_diagnostic.csv",
]

print("=" * 110)
print("S2 70% ACCURACY RESEARCH — DATA AUDIT")
print("=" * 110)

for name in files:
    p = base / name

    print(f"\n--- {name} ---")

    if not p.exists():
        print("MISSING")
        continue

    df = pd.read_csv(p)

    print("ROWS:", len(df))
    print("COLUMNS:")
    print(", ".join(df.columns.astype(str)))

    print("\nSAMPLE:")
    print(df.head(5).to_string(index=False))

print("\n" + "=" * 110)
print("DONE — READ ONLY")
print("=" * 110)
