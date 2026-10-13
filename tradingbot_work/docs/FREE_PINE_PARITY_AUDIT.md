# Free Pine/Python candidate audit

This audit is designed to run without TradingView's paid chart-data export. It uses the supplied 5-minute OHLCV CSVs and the shared Pine-compatible indicator helpers. It is a Python-side calculation and **does not establish exact TradingView candle-by-candle parity**.

## Run

From `tradingbot_work`:

```bash
python scripts/free_pine_parity_audit.py \
  --nifty ../data/NIFTY_50_5m_2025-10-01_to_2026-10-09.csv \
  --bank-nifty ../data/BANK_NIFTY_5m_2025-10-01_to_2026-10-09.csv \
  --out-dir reports/free_pine_parity
```

The script writes candidate rows with timestamps, pivots, EMA20/EMA50, RSI14, ATR14, cross flags, candidate/valid flags, stop and target references, plus a data-quality summary. The volume filter is OFF by default because the supplied data has no meaningful volume. Pass `--volume-filter` to test the Pine default; with all-zero volume it should produce no qualifying signals.

The default session end is 15:00 IST for the Strategy appendix. Use `--session-end 15:15` to audit the alert indicator's default entry window.

## Results from the supplied files

| Check | NIFTY 50 | BANK NIFTY |
|---|---:|---:|
| Rows | 18,837 | 18,837 |
| First candle (IST) | 2025-10-01 09:15 | 2025-10-01 09:15 |
| Last candle (IST) | 2026-10-08 15:25 | 2026-10-08 15:25 |
| Sessions | 252 | 252 |
| Non-zero volume bars | 0 | 0 |
| Candidate longs, volume OFF | 193 | 188 |
| Candidate shorts, volume OFF | 206 | 226 |
| Candidate longs, volume ON | 0 | 0 |
| Candidate shorts, volume ON | 0 | 0 |

The CSV filenames say they run through 2026-10-09, but their last candle is 2026-10-08 at 15:25 IST. The session on 2025-10-21 has only 12 candles in each file; investigate this short session before treating the data as complete. Other regular sessions in this sample have 75 five-minute candles.

## Exact parity limitation

A true parity check needs TradingView's actual PP/R1/S1, EMA20, EMA50, RSI14, ATR14, crossover and final signal series for the same ticker, 5-minute timeframe, session window, and volume-filter setting. The Python-side candidates can be audited for consistency without paid export, but must not be presented as exact TradingView parity. Do not enable live orders based on this report; retain PAPER mode.
