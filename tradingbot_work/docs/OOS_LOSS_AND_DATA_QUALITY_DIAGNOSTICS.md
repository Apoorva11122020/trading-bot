# OOS Loss and Historical Data Quality Diagnostics

**Branch:** `audit/oos-backtest-safety`  
**Data window:** 2025-10-01 through 2026-10-08 (CSV filenames say through Oct 9)  
**Scope:** read-only investigation of the supplied NIFTY 50 and BANK NIFTY 5-minute CSVs. No live trading.

## Findings at a glance

1. Both CSVs contain 18,837 rows and 252 dates. **2025-10-21 contains 12 candles, timestamped 13:45 through 14:40 IST**, matching the NSE Diwali Muhurat session scheduled for 13:45–14:45 IST. This is an expected special one-hour session, not a data gap. See the [NSE F&O circular](https://www.smifs.com/files/downloads/638950074544632565_Muhurat%20Trading%20session%20on%20account%20of%20Diwali.pdf) or [Upstox schedule](https://upstox.com/announcements/special-trading/muhurat-trading-2025-tuesday-21-october-145-pm-245-pm/).
2. Both CSVs have zero volume and zero open interest on every row. They are not suitable for validating the Pine default volume filter or actual futures/options P&L.
3. The short-only OOS backtest is negative even with zero costs. The October 2025 Muhurat session is far before the June–October 2026 OOS window and does **not** explain the observed OOS losses.
4. The dominant loss mechanism is stop-outs, not intraday time exits. The entry-hour breakdown points to poor results in late-day shorts, but this is a diagnostic observation—not permission to optimize against the same OOS sample.

## Session interpretation

The 12-candle session on 2025-10-21 is a valid special market session. The NSE scheduled Muhurat trading from 13:45 to 14:45 IST, which yields 12 five-minute candles when timestamps label candle starts. It must not be treated as a truncated normal trading day. No historical CSV/parquet files were modified or re-downloaded.

The standard-session audit still reports sessions with a candle count other than 75; that is a data-quality signal to interpret against the exchange calendar, not an automatic reason to discard a session. A future data validator should maintain a calendar of special sessions instead of blindly requiring 75 bars every day.

## OOS result (zero commission, zero slippage)

| Instrument | Trades | Net points | Win rate | Profit factor | Max drawdown |
|---|---:|---:|---:|---:|---:|
| NIFTY 50 | 62 | -194.50 | 37.10% | 0.795 | 377.26 |
| BANK NIFTY | 68 | -756.32 | 33.82% | 0.776 | 1,206.35 |

These are approximate spot-index points, not contract-level futures/options returns.

### Exit reasons

| Instrument | Exit reason | Trades | Net points |
|---|---|---:|---:|
| NIFTY 50 | Stop | 37 | -925.89 |
| NIFTY 50 | Target | 22 | +736.40 |
| NIFTY 50 | Intraday flat | 3 | -5.00 |
| BANK NIFTY | Stop | 44 | -3,296.81 |
| BANK NIFTY | Target | 19 | +2,444.89 |
| BANK NIFTY | Intraday flat | 5 | +95.60 |

The simulated payoff is -1R at a stop and +1.5R at a target. With no costs, that payoff needs a win rate above 40% to break even before flat exits. Observed win rates are below that threshold for both instruments. Costs make the result worse.

### Entry-hour cohorts (IST; zero costs)

| Instrument | Entry hour | Trades | Net points | Win rate |
|---|---:|---:|---:|---:|
| NIFTY 50 | 10:00–10:55 | 9 | -184.22 | 22.22% |
| NIFTY 50 | 14:00–14:55 | 8 | -163.42 | 0.00% |
| BANK NIFTY | 10:00–10:55 | 14 | -261.69 | 35.71% |
| BANK NIFTY | 13:00–13:55 | 12 | -590.56 | 8.33% |
| BANK NIFTY | 14:00–14:55 | 13 | -430.72 | 30.77% |

These cohorts are small and were inspected after seeing the OOS results. They are hypotheses for a *new* future holdout, not evidence to hard-code a time filter. Do not tune on June–October and then call that same period out-of-sample.

### Monthly OOS net points (zero costs)

| Month 2026 | NIFTY 50 | BANK NIFTY |
|---|---:|---:|
| June | -84.06 | +187.54 |
| July | -116.00 | -235.83 |
| August | -88.48 | +23.86 |
| September | +134.86 | -685.79 |
| October through Oct 8 | -40.82 | -46.09 |

NIFTY's total loss is spread across June–August and early October, offset by a profitable September. BANK NIFTY's total loss is heavily concentrated in September. That instability argues against declaring a reliable edge from this sample.

## Configuration mismatch / deployment gate

- Pine defaults the volume filter ON; the supplied data has all-zero volume, so that setting yields no qualifying signals. The OOS report intentionally used volume OFF. This is a research variant, not default-settings parity.
- The OOS test is short-only, while the Pine strategy can take longs too.
- Costs, index-point slippage, intrabar stop/target ordering, and spot-versus-derivative fills are approximations.
- Exact TradingView parity remains unverified.

**Decision:** keep PAPER mode. Do not enable real orders. Next valid test should use clean volume-bearing instrument data, verify complete sessions/daily pivots, include realistic costs, and reserve a future untouched holdout before changing strategy filters.
