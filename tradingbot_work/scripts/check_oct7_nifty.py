import pandas as pd

p = r"data\historical\underlying\NIFTY\year=2026\month=10\NSE_INDEX_Nifty_50.parquet"

df = pd.read_parquet(p)

print("ROWS:", len(df))
print("COLUMNS:", list(df.columns))
print("MIN:", df.iloc[:, 0].min())
print("MAX:", df.iloc[:, 0].max())

ts = pd.to_datetime(df.iloc[:, 0])
day = df[ts.dt.date.astype(str) == "2026-10-07"]

print("7-OCT ROWS:", len(day))
print(day.head(20).to_string(index=False))
