import os

import requests
from dotenv import load_dotenv


load_dotenv()


TOKEN = os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN")


if not TOKEN:
    print("ERROR: UPSTOX_SANDBOX_ACCESS_TOKEN not found in .env")
    raise SystemExit(1)


# Upstox Get Order Book API
# NOTE:
# Order Book API is V2, not V3.
URL = "https://api-sandbox.upstox.com/v2/order/retrieve-all"


headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


try:
    response = requests.get(
        URL,
        headers=headers,
        timeout=15,
    )

    print(f"HTTP status: {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        data = response.text

    if response.ok:
        print("UPSTOX SANDBOX CONNECTIVITY: OK")

        if isinstance(data, dict):

            orders = data.get("data", [])

            if isinstance(orders, list):
                print(
                    f"Order-book records returned: {len(orders)}"
                )
            else:
                print("Order-book response received.")

        elif isinstance(data, list):

            print(
                f"Order-book records returned: {len(data)}"
            )

        else:

            print("Order-book response received.")

        print(
            "No order was placed, modified, or cancelled."
        )

    else:

        print("UPSTOX SANDBOX CONNECTIVITY: FAILED")

        print("Response:")
        print(data)

        raise SystemExit(1)


except requests.RequestException as exc:

    print("UPSTOX SANDBOX CONNECTIVITY: FAILED")

    print(f"Network error: {exc}")

    raise SystemExit(1)