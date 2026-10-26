# Daily pivot paper tracker

This is an observational paper tracker for the NIFTY or BANKNIFTY spot index. It reads Upstox historical candles for warm-up and the read-only market-data WebSocket for today's candles. It has no broker order placement.

## Run on a trading day

Run between 9:00 AM and 3:15 PM IST on a trading day; starting after that window exits without recording a zero-trade result. From PowerShell:

```powershell
cd D:\trading-bot-shadow-comparison\tradingbot_work
python scripts\run_daily_pivot_paper.py --symbol NIFTY
```

For BANKNIFTY, use `--symbol BANKNIFTY`. Leave the terminal open during the session. It stops after 3:35 PM IST; press Ctrl+C to stop earlier. The project must have a valid Upstox historical-data token and market-data token configured using the existing `.env` setup. Do not paste tokens into chat.

## Output

- One event file per date and symbol under `reports/daily_pivot_paper/`.
- `daily_summary.csv` contains closed-trade count, wins, losses, net hypothetical points, live-bar coverage and data gaps.
- No signal means no entry row. An open position at shutdown is marked in the daily summary.
- After collecting sessions, run `python scripts\summarize_pivot_paper_forward.py` to see progress toward the 20-session review gate.

## Interpretation

The tracker uses **frozen v5 short-only** rules: **NIFTY** short 11:00–13:00 IST; **BANKNIFTY** short 13:00–15:00 only (no morning longs). One short trade per symbol per session. VWAP via session typical-price average (spot proxy), max risk NIFTY 40 / BANKNIFTY 100 points, breakeven at 0.75R, ATR buffer 0.1, and 1.5R target. Spot volume is unavailable live; use `run_frozen_pivot_backtest.py` for futures parity. Entry uses the first tick of the next 5-minute candle. Stop is counted first when both stop and target touch in one candle.

Costs default to 2 index points of slippage plus 0.03% per side. They are scenario estimates, not validated broker charges, and the output is in spot-index points—not futures or options rupee P&L. Pass `--slippage-points-per-side` and `--commission-percent-per-side` to change the assumptions. Keep this tracker paper-only; its results do not authorize live trading.
