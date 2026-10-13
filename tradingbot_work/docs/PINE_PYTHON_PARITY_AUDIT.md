# Pine Script ↔ Python Signal Parity Audit

**Audit date:** 2026-10-13  
**Branch:** `audit/oos-backtest-safety`  
**Scope:** static formula comparison plus the supplied NIFTY 50 and BANK NIFTY 5-minute CSVs. No live trading or orders.

## Result

**Status: formula-level parity is plausible, but end-to-end TradingView parity is not yet verified.** The original Pine code is available in the project guide `Nifty_BankNifty_Pivot_Point_Algo_Guide.docx`. A true candle-by-candle reconciliation still requires exporting TradingView indicator values/signals for the exact same symbol, chart session, and timeframe.

## Historical CSV audit

Both supplied files contain 18,837 rows and 252 distinct dates, from 2025-10-01 09:15 IST through 2026-10-08 15:25 IST. Although the filenames say `to_2026-10-09`, the final candle in each file is **2026-10-08 15:25 IST**.

| Check | NIFTY 50 | BANK NIFTY |
|---|---:|---:|
| Rows | 18,837 | 18,837 |
| Distinct sessions | 252 | 252 |
| Volume values | 100% zero | 100% zero |
| Open-interest values | 100% zero | 100% zero |

These are spot-index-style data without meaningful volume. They cannot validate a volume-confirmed strategy, and they are not futures/options fills.

## Signal logic compared

The Pine reference defines:

- Classic pivots from the previous completed daily candle: PP = (H + L + C) / 3; R1 = 2 × PP − L; S1 = 2 × PP − H.
- EMA(20)/EMA(50) trend filters.
- RSI(14) thresholds: above 55 for long, below 45 for short.
- Crosses: `ta.crossover(close, pp) or ta.crossover(close, r1)`; `ta.crossunder(close, pp) or ta.crossunder(close, s1)`.
- Confirmed 5-minute bars and close-based entry reference.
- ATR(14) stop buffer 0.1 ATR and initial target 1.5R.

The Python research scripts use the same pivot formulas, EMA/RSI/ATR helper functions, previous-bar pivot levels for cross detection, and the same trend/RSI thresholds. The pivot calculation deliberately uses regular NSE-session candles only; this matches the intended spot-index session in these CSVs. Indicator calculations are continuous across the history to preserve warm-up.

## Important configuration differences

1. **Volume defaults differ by experiment.** The Pine indicator and Pine Strategy appendix default `useVolume=true`. Since CSV volume is always zero, this filter rejects all signals (zero-volume bars cannot satisfy `volume > volAvg × 1.2`). The OOS Python experiment intentionally runs with the volume filter OFF. That is a different configuration, not default-settings parity.
2. **The two Pine snippets use different session ends.** The alert indicator defaults to 15:15; the Strategy Tester appendix allows entries only before 15:00 and calls `strategy.close_all` during the 15:00 hour. The Python OOS script allows entries before 15:00 and uses its own test-window/intraday flat logic at/after 15:15. Do not compare trade lists as if these were identical settings.
3. **Session close is now aligned for the OOS research script.** The Python OOS script has been changed to flatten at/after 15:00 IST, matching the Strategy appendix's close-all trigger on the 15:00 bar. The alert-only indicator still has its own 15:15 signal-session default.
4. **Direction and position management differ.** The Python OOS script is SHORT_ONLY and permits one open position; the Pine Strategy appendix can take both long and short signals and can reverse positions. Its Strategy Tester trades cannot be expected to match the short-only report.
5. **Costs/fills differ.** The Pine Strategy example sets 0.03% commission and 2 ticks of slippage. The OOS script's default costs are zero unless passed explicitly. Also, the report is in approximate spot-index points, not broker-accurate futures/options P&L.
6. **Tick-size guard.** Pine validates risk against `syminfo.mintick`; the Python OOS script now has a configurable `--min-tick` (default 0.05) and rejects risk at or below that value. No short signal in the supplied sample had risk at or below 0.05 points.
7. **Stop/target sequencing.** Python conservatively counts a stop before a target when both are touched in the same 5-minute candle. Pine Strategy Tester intrabar fill assumptions may differ; exact trade parity requires matching TradingView strategy properties and broker emulator settings.

## Candidate signal counts (volume filter OFF)

These are raw rule-based signal counts, not executed trades or proof of profitability.

| Instrument | Pine indicator session ending 15:15: long | short | Pine Strategy session ending 15:00: long | short |
|---|---:|---:|---:|---:|
| NIFTY 50 | 204 | 220 | 193 | 206 |
| BANK NIFTY | 200 | 236 | 188 | 226 |

With the Pine volume filter ON, the supplied zero-volume CSVs produce **zero qualifying signals**. The counts above assume the volume filter is explicitly disabled and the chart is using these same candles.

## OOS results interpretation

The OOS script has now been corrected to flatten at 15:00 IST and apply the Pine `syminfo.mintick` risk guard (default 0.05). With zero commission and zero slippage, the corrected short-only OOS run (2026-06-01 through 2026-10-08) produced NIFTY 50: 62 trades, −194.50 points, 37.10% wins, PF 0.795; BANK NIFTY: 68 trades, −756.32 points, 33.82% wins, PF 0.776. With 2 points of slippage per side and zero commission, results were NIFTY 50: −442.50 points (PF 0.599); BANK NIFTY: −1,028.32 points (PF 0.710). These are approximate spot-index points, not contract P&L. Both remain negative before brokerage/taxes, so do not deploy live.

## Next validation gate

1. Export TradingView values/signals for the exact same symbol and 5-minute chart, using volume filter OFF and matching the session end.
2. Join by candle timestamp and compare PP/R1/S1, EMA20, EMA50, RSI14, ATR14, cross flags, and final signal flags; report mismatch counts and first mismatching timestamps.
3. Reconcile tick size, commission, slippage, session close, position reversal, and intrabar stop/target handling before comparing Strategy Tester trades.
4. Keep the bot in PAPER mode. No live order path should be enabled on the basis of this audit.
