# Classic Pivot Algo for TradingView

Frozen **v5 short-only** (default paper/backtest). Use the **corresponding futures chart** on
5 minutes for volume and VWAP parity.

## Files

- `classic_pivot_alerts.pine`: pivot alerts (symbol preset sets windows).
- `classic_pivot_strategy_backtest.pine`: Strategy Tester version; no broker orders.

## Frozen v5 short-only rules (Python paper default)

- **NIFTY:** short-only 11:00–13:00
- **BANKNIFTY:** short-only 13:00–15:00 (no morning longs)
- **Risk cap:** NIFTY 40 pts | BANKNIFTY 100 pts | BE @ 0.75R
- Compare presets: `python scripts/run_frozen_split_compare.py` (v3 / v4 / v5)

### NIFTY PE recovery add-on (NIFTY preset only)

1. Session low reaches **morning open − 200** points.
2. Spot then **crosses up** through **morning open − 70**.
3. Alert: **BUY PE, 2 lots, stop 20 points on the PE premium** (not spot).
4. Once per session. **Paper tracker** (`run_daily_pivot_paper.py`) auto-picks nearest-expiry ATM PE from the Upstox instrument master and logs `PE_ENTRY` with contract + premium SL. TradingView alert is spot-only; use Python paper output for the contract.

Set **Symbol Preset** to match the chart. Spot index charts may lack meaningful
volume; prefer the front-month futures contract.

## Alert setup

1. Open NIFTY or BANKNIFTY **futures** chart, 5 minutes.
2. Paste `classic_pivot_alerts.pine`, save, add to chart.
3. Set **Symbol Preset** to NIFTY or BANKNIFTY.
4. Create alert: **Any alert() function call**.

Alerts show the signal-candle close as the entry reference; execute on the next
bar open. Python paper tracking uses the same frozen logic.

## Backtest

Run `python scripts/run_frozen_pivot_backtest.py` for the futures research model
(Upstox fee estimate, 1 point/side slippage). TradingView Strategy Tester fills
differ slightly. Keep alerts observational until forward paper results pass the
20-session review gate.
