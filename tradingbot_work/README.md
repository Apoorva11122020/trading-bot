# Automated Indian Trading Bot — Phase 1

Phase 1 implements the strategy/configuration foundation only.

## Safety

- PAPER mode by default.
- No broker integration.
- No order execution.
- No live trading capability in this phase.
- Strategy thresholds are configuration-driven.

## Baseline Strategy

5-minute NSE equity strategy using:

- EMA 20 / EMA 50
- VWAP
- RSI 14
- ATR 14
- Volume SMA 20
- Previous 20-candle breakout

Long requires all configured long conditions. Short requires all configured short conditions.

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## Validate

```bash
python run.py --validate-config
```

## Demo

```bash
python run.py --demo
```

## Tests

```bash
pytest -q
```

## Deferred

Broker abstraction, authentication, WebSocket market data, order management,
database persistence, dashboard, paper execution simulator, deployment and
live trading are intentionally deferred to later phases.


## Phase 2 — Broker Abstraction

Phase 2 adds a broker-independent `BrokerInterface` and an in-memory `PaperBroker`. The paper adapter never contacts an exchange or external broker. Real broker integration is intentionally deferred.

## Current broker status

The application is intentionally broker-neutral. `BROKER=PAPER` is the default. A real broker adapter is not enabled until the client selects a broker and provides the required API/authentication details. Unsupported broker names fail closed.

## Trade journal

Filled entry orders create an OPEN trade record. A matching filled exit closes the trade and stores realized P&L. This is separate from order records.

## Upstox V3 References
- Official Market Data Feed V3: https://upstox.com/developer/api-documentation/v3/get-market-data-feed/
- Official V3 authorization endpoint documentation: https://upstox.com/developer/api-documentation/get-market-data-feed-authorize-v3/
- Official Python SDK V3 WebSocket sample: https://github.com/upstox/upstox-python/tree/master/examples/websocket/market_data/v3
