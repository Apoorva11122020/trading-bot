import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv

from src.broker.upstox_sandbox import (
    UpstoxSandboxBroker,
    UpstoxSandboxError,
)


# ------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


def main() -> int:

    print("=" * 70)
    print("UPSTOX SANDBOX ORDER TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Check test flag
    # --------------------------------------------------------------

    run_test = os.getenv(
        "UPSTOX_RUN_SANDBOX_ORDER_TEST",
        "false",
    ).strip().lower()

    if run_test != "true":

        print()
        print(
            "Sandbox order test is disabled."
        )
        print()
        print(
            "Set this in .env:"
        )
        print(
            "UPSTOX_RUN_SANDBOX_ORDER_TEST=true"
        )

        return 0

    # --------------------------------------------------------------
    # Read configuration
    # --------------------------------------------------------------

    instrument = os.getenv(
        "UPSTOX_SANDBOX_INSTRUMENT",
        "NSE_EQ|INE669E01016",
    ).strip()

    quantity_raw = os.getenv(
        "UPSTOX_SANDBOX_QUANTITY",
        "1",
    ).strip()

    limit_price_raw = os.getenv(
        "UPSTOX_SANDBOX_LIMIT_PRICE",
        "1",
    ).strip()

    # --------------------------------------------------------------
    # Validate quantity
    # --------------------------------------------------------------

    try:

        quantity = int(quantity_raw)

    except ValueError:

        print()
        print(
            f"ERROR: Invalid quantity: {quantity_raw}"
        )

        return 1

    if quantity <= 0:

        print()
        print(
            "ERROR: Quantity must be greater than 0."
        )

        return 1

    # --------------------------------------------------------------
    # Validate price
    # --------------------------------------------------------------

    try:

        limit_price = float(limit_price_raw)

    except ValueError:

        print()
        print(
            f"ERROR: Invalid limit price: "
            f"{limit_price_raw}"
        )

        return 1

    if limit_price <= 0:

        print()
        print(
            "ERROR: Limit price must be greater than 0."
        )

        return 1

    # --------------------------------------------------------------
    # Create broker
    # --------------------------------------------------------------

    try:

        broker = UpstoxSandboxBroker()

    except UpstoxSandboxError as exc:

        print()
        print(
            "BROKER INITIALIZATION FAILED"
        )
        print(exc)

        return 1

    # --------------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------------

    print()
    print(
        f"SANDBOX HOST: {broker.config.base_url}"
    )

    print(
        f"INSTRUMENT: {instrument}"
    )

    print(
        f"QUANTITY: {quantity}"
    )

    print(
        f"LIMIT PRICE: {limit_price}"
    )

    print()
    print(
        "This test will:"
    )

    print(
        "1. Connect to Upstox Sandbox"
    )

    print(
        "2. Place ONE LIMIT BUY order"
    )

    print(
        "3. Read the returned order ID"
    )

    print(
        "4. Fetch order details"
    )

    print(
        "5. Cancel the order"
    )

    print()
    print(
        "No live-market order will be submitted."
    )

    # --------------------------------------------------------------
    # Connect
    # --------------------------------------------------------------

    try:

        print()
        print(
            "Connecting to Upstox Sandbox..."
        )

        broker.connect()

        print(
            "Sandbox connection: OK"
        )

    except Exception as exc:

        print()
        print(
            "SANDBOX CONNECTION FAILED"
        )

        print(exc)

        return 1

    # --------------------------------------------------------------
    # Place order
    # --------------------------------------------------------------

    order_id = None

    try:

        print()
        print(
            "Placing Sandbox LIMIT BUY..."
        )

        order_id = broker.place_order(
            symbol=instrument,
            side="BUY",
            quantity=quantity,
            order_type="LIMIT",
            price=limit_price,
        )

        print()
        print(
            "ORDER PLACED SUCCESSFULLY"
        )

        print(
            f"ORDER ID: {order_id}"
        )

    except Exception as exc:

        print()
        print(
            "SANDBOX ORDER PLACEMENT FAILED"
        )

        print(exc)

        broker.disconnect()

        return 1

    # --------------------------------------------------------------
    # Get order details
    # --------------------------------------------------------------

    try:

        print()
        print(
            "Fetching order details..."
        )

        details = broker.get_order_details(
            order_id
        )

        print(
            "ORDER DETAILS RECEIVED:"
        )

        print(details)

    except Exception as exc:

        print()
        print(
            "WARNING: Could not fetch order details."
        )

        print(exc)

    # --------------------------------------------------------------
    # Cancel order
    # --------------------------------------------------------------

    try:

        print()
        print(
            "Cancelling Sandbox order..."
        )

        cancelled = broker.cancel_order(
            order_id
        )

        if cancelled:

            print()
            print(
                "ORDER CANCELLED SUCCESSFULLY"
            )

        else:

            print()
            print(
                "ORDER CANCELLATION WAS NOT CONFIRMED"
            )

    except Exception as exc:

        print()
        print(
            "SANDBOX ORDER CANCELLATION FAILED"
        )

        print(exc)

        broker.disconnect()

        return 1

    # --------------------------------------------------------------
    # Disconnect
    # --------------------------------------------------------------

    broker.disconnect()

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("SANDBOX ORDER TEST COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Order ID : {order_id}"
    )

    print(
        "Side     : BUY"
    )

    print(
        "Type     : LIMIT"
    )

    print(
        f"Quantity : {quantity}"
    )

    print(
        f"Price    : {limit_price}"
    )

    print()
    print(
        "Sandbox order placement and cancellation flow completed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())