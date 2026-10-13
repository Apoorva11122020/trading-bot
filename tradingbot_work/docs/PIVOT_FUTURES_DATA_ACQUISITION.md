# Real futures data for the classic-pivot audit

The old `NIFTY_50_5m...csv` and `BANK_NIFTY_5m...csv` files are **spot index** candles. Their zero volume is expected for this data source and makes them unsuitable for checking Pine's volume filter. Do not replace zero volume with synthetic volume and do not treat index bars as futures fills.

The repository now includes `scripts/download_pivot_futures.py`. It downloads genuine exchange-listed futures candles through the existing read-only Upstox historical client, keeps each expiry in its own Parquet file, and runs the existing futures quality checks: real volume, OHLC validity, session coverage, causal front-month selection, spot/futures timestamp alignment, VWAP, rollover diagnostics, and token-leak scan. The input spot CSV is only a reference for session dates/prices; its volume column is deliberately ignored.

## Run from `tradingbot_work`

First confirm `.env` has a valid **historical/live-account** token in one of:
- `UPSTOX_HISTORICAL_ACCESS_TOKEN`
- `UPSTOX_MARKET_DATA_ACCESS_TOKEN`
- `ACCESS_TOKEN`

`UPSTOX_SANDBOX_ACCESS_TOKEN` is not accepted for historical market data. The expired-contract endpoints may require Upstox Plus access. The downloader only sends GET requests and blocks order placement.

Run the NIFTY and BANK NIFTY downloads separately:

```powershell
python scripts/download_pivot_futures.py --symbol NIFTY --spot-csv ..\data\NIFTY_50_5m_2025-10-01_to_2026-10-09.csv
python scripts/download_pivot_futures.py --symbol BANKNIFTY --spot-csv ..\data\BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv
```

The actual source CSVs end on **2026-10-08**, so the default end date is 2026-10-08 even though the filenames say Oct 9. If your files are in another location, supply their real paths with `--spot-csv`.

Output is stored separately so the two underlying datasets cannot overwrite each other:
- `data/historical/pivot_futures/nifty/futures/`
- `data/historical/pivot_futures/banknifty/futures/`

The futures validator recognizes the 2025-10-21 Muhurat session as 12 bars (13:45–14:45 IST), rather than falsely requiring 75 regular-session bars. To validate an already-downloaded dataset without network calls, pass the same symbol and CSV with `--validate`. Check `futures/data_quality.json`: only a `PASS` dataset is suitable for the next research run. If it reports `FUTURES_VOLUME_UNAVAILABLE`, missing contracts, or alignment failures, stop rather than backtesting with the index volume.

## Safety and interpretation

- PAPER mode remains mandatory; no orders are placed.
- This script only acquires and validates data. It does not claim the strategy is profitable or exact TradingView parity.
- Use genuine futures OHLCV to validate a volume filter; for actual derivatives P&L, a separate contract-level execution/backtest is still required.
- Do not tune against the existing June–October 2026 holdout and then report it as untouched OOS data.
