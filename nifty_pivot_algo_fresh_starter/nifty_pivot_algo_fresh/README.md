# Nifty & Bank Nifty Classic Pivot Algo — Fresh Starter Pack

## Included
- `tradingview/classic_pivot_alerts.pine`: confirmed 5-minute signals and dynamic alerts; no broker orders.
- `tradingview/classic_pivot_backtest.pine`: separate TradingView Strategy Tester simulation.

## Rules from client guide
- Test Nifty 50 and Bank Nifty separately.
- Use a consistent chart basis: spot index OR corresponding futures. Pivot levels differ.
- Classic pivots from the previous completed daily candle: PP=(H+L+C)/3; R1=2*PP-L; S1=2*PP-H; R2=PP+(H-L); S2=PP-(H-L).
- Bullish: confirmed candle close crosses above PP or R1; optional EMA20>EMA50, RSI(14)>55 and volume > 20-bar average × multiplier.
- Bearish: confirmed close crosses below PP or S1; optional EMA20<EMA50, RSI(14)<45 and same volume filter.
- Stop: signal candle low minus ATR(14) × buffer for bullish; signal candle high plus buffer for bearish.
- Initial target: 1.5R.
- Indicator default session: 09:20–15:15. Backtest entries default to 09:20–15:00 with force-flat at 15:15.

## Setup
1. Open a Nifty or Bank Nifty chart in TradingView and select 5 minutes.
2. For alerts, paste `classic_pivot_alerts.pine` in Pine Editor, save, and add to chart.
3. Create an alert with **Any alert() function call** to get dynamic entry, SL and target details.
4. For backtesting, open a separate Pine Editor tab, paste `classic_pivot_backtest.pine`, save, add to chart, and open Strategy Tester.
5. Repeat separately for the other index.

## Validation checklist
- Backtest at least 6–12 months of 5-minute data, including trending and sideways markets.
- Set realistic commission and slippage.
- Record net profit, profit factor, max drawdown, win rate, trade count, average trade and costs.
- Compare volume filter on/off. Spot-index volume may be unavailable or meaningless; test futures or compare with volume filter disabled.
- Validate on a later unseen period and manually inspect trade lists/signals.

## Limitations and safety
- This is starter code, not proof of profitability or investment advice.
- Verify point value, quantity units and lot/contract sizing for the exact chart symbol; risk-based sizing is an approximation.
- Signal close is a reference, not a guarantee of live execution price.
- No Upstox WebSocket, options premium tracker, Python service, or live broker order routing is included. Those require a separate, explicit spec.
- Do not use live execution until historical and out-of-sample testing and paper observation are complete.
