from datetime import date
from src.historical_data.upstox_client import UpstoxHistoricalClient

client = UpstoxHistoricalClient()

for day in [date(2026, 10, 1), date(2026, 10, 6), date(2026, 10, 7)]:
    print(f"\n=== {day} ===")
    try:
        candles = client.candles(
            "NSE_INDEX|Nifty 50",
            day,
            day,
            unit="minutes",
            interval=5,
        )

        print("COUNT:", len(candles))

        if candles:
            print("FIRST:", candles[0])
            print("LAST :", candles[-1])

    except Exception as exc:
        print("ERROR:", type(exc).__name__)
        print(exc)
