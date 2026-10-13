from pathlib import Path
import pandas as pd

folder = Path("data/historical/upstox_pivot_5m")

for file in sorted(folder.glob("*.csv")):
    df = pd.read_csv(file)
    print("\nFILE:", file.name)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Missing values:", df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
    print("First candle:", df["timestamp"].iloc[0])
    print("Last candle:", df["timestamp"].iloc[-1])

print("\nBACKTEST SCRIPTS:")
for p in Path(".").rglob("*.py"):
    if ".venv" not in str(p) and any(x in p.name.lower() for x in ["backtest", "replay", "pivot"]):
        print(p)
