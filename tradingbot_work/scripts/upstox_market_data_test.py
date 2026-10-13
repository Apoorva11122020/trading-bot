from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import websocket
from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD .ENV
# ============================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.market_data.upstox_v3 import (
    UpstoxV3MarketDataError,
    authorize_v3,
    decode_ticks,
    resolve_market_data_access_token,
    subscription_payload,
)


# ============================================================
# CONFIG
# ============================================================

INSTRUMENT = os.getenv(
    "UPSTOX_MARKET_DATA_INSTRUMENT",
    "NSE_INDEX|Nifty 50",
)

MODE = os.getenv(
    "UPSTOX_MARKET_DATA_MODE",
    "ltpc",
)

RUN_SECONDS = int(
    os.getenv(
        "UPSTOX_MARKET_DATA_SECONDS",
        "30",
    )
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("UPSTOX MARKET DATA V3 WEBSOCKET TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Token
    # --------------------------------------------------------

    try:
        token, token_source = resolve_market_data_access_token()
    except UpstoxV3MarketDataError as exc:
        print()
        print("ERROR:")
        print(str(exc))
        print()
        print(f"Checked .env file: {ENV_FILE}")
        print(
            "Set UPSTOX_MARKET_DATA_ACCESS_TOKEN (preferred) or ACCESS_TOKEN. "
            "UPSTOX_SANDBOX_ACCESS_TOKEN is not used for market data."
        )
        raise RuntimeError("Upstox market-data access token is not configured.") from exc

    print()
    print("Token loaded: YES")
    print(f"Token source: {token_source}")
    print("Token value is hidden for security.")

    print()
    print(
        "INSTRUMENT :",
        INSTRUMENT,
    )

    print(
        "MODE       :",
        MODE,
    )

    print(
        "DURATION   :",
        f"{RUN_SECONDS} seconds",
    )

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    print()
    print(
        "Requesting Upstox V3 "
        "WebSocket authorization..."
    )

    ws_url = authorize_v3(
        access_token=token,
        timeout=10.0,
    )

    print(
        "Authorization: OK"
    )

    print(
        "WebSocket URL received."
    )

    # --------------------------------------------------------
    # WebSocket
    # --------------------------------------------------------

    print()
    print(
        "Connecting to WebSocket..."
    )

    ws = websocket.create_connection(
        ws_url,
        timeout=10,
    )

    print(
        "WebSocket connection: OK"
    )

    # --------------------------------------------------------
    # Subscription
    # --------------------------------------------------------

    payload = subscription_payload(
        [INSTRUMENT],
        mode=MODE,
        guid="trading-bot-market-data-test",
    )

    print()
    print(
        "Sending subscription..."
    )

    # Upstox V3 requires binary subscription.
    ws.send(
        payload,
        opcode=websocket.ABNF.OPCODE_BINARY,
    )

    print(
        "Subscription sent: OK"
    )

    # --------------------------------------------------------
    # Receive
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("WAITING FOR MARKET DATA")
    print("-" * 70)

    start = time.time()

    tick_count = 0

    try:

        while (
            time.time() - start
            < RUN_SECONDS
        ):

            try:

                message = ws.recv()

            except websocket.WebSocketTimeoutException:

                print(
                    "No message received "
                    "during timeout..."
                )

                continue

            if message is None:
                continue

            # ------------------------------------------------
            # Text message
            # ------------------------------------------------

            if isinstance(message, str):

                print(
                    "Received text message:",
                    message[:300],
                )

                continue

            # ------------------------------------------------
            # Decode Protobuf
            # ------------------------------------------------

            try:

                ticks = decode_ticks(
                    message
                )

            except UpstoxV3MarketDataError as exc:

                print(
                    "Decode error:",
                    exc,
                )

                continue

            # ------------------------------------------------
            # Print ticks
            # ------------------------------------------------

            for tick in ticks:

                tick_count += 1

                print(
                    f"[TICK {tick_count:04d}] "
                    f"{tick.symbol} | "
                    f"LTP={tick.ltp:.2f} | "
                    f"LTQ={tick.last_quantity:.0f} | "
                    f"TIME={tick.timestamp.isoformat()}"
                )

    finally:

        print()
        print(
            "Closing WebSocket..."
        )

        try:
            ws.close()

        except Exception:
            pass

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MARKET DATA TEST COMPLETE")
    print("=" * 70)

    print(
        f"Ticks received: {tick_count}"
    )

    if tick_count > 0:

        print(
            "RESULT: MARKET DATA STREAM OK"
        )

    else:

        print(
            "RESULT: NO TICKS RECEIVED"
        )

        print(
            "Check instrument key, token, "
            "market status and WebSocket response."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Test interrupted by user."
        )

    except Exception as exc:

        print()
        print(
            "ERROR:"
        )

        print(exc)

        raise